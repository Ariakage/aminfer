# ADR-0004: NPU → GPU KV sharing through IOSurface-backed Metal textures

- Status: Accepted
- Date: 2026-11-20
- Source: user requirement (init session, "new requirements" 1)

## Context

The ANE's on-chip SRAM requires the channel dimension and strides of tensors
to be 64-byte aligned, and its native output object is a two-dimensional,
strided IOSurface. Forcing ANE output into a flat one-dimensional `MTLBuffer`
layout tends to make CoreML insert an internal relayout copy. The KV cache
produced by the ANE during chunked prefill must reach the GPU attention kernels
with zero bytes copied and no relayout.

## Decision

1. ANE-produced K and V live in IOSurface-backed `CVPixelBuffer`s of format
   `kCVPixelFormatType_OneComponent16Half` — the only fp16 format CoreML's
   `MLMultiArray(pixelBuffer:shape:)` accepts. CoreML writes into them through
   prediction output backings.
2. `apple_bridge.m` maps each IOSurface to an `MTLTexture` with
   `newTextureWithDescriptor:iosurface:plane:`. The default view is
   RGBA16Float (width = row elements / 4); RG16Float and R16Float views are
   allowed.
3. `.metal` kernels read ANE-produced KV with `tex.read(uint2(x, y))`.
4. Tiles respect the texture dimension limit (16,384) and the 16 KiB rule:
   tiles are created with an explicit allocation size that is a multiple of
   16 KiB, base addresses are checked, and the reported IOSurface allocation
   size is never used as a Metal buffer length.

## Consequences

- GPU attention kernels read KV through textures. GPU-produced decode KV uses
  the same tile format (shader write), so one kernel family serves both.
- CoreVideo joins the frameworks the bridge may call
  ([ADR-0001](0001-language-and-toolchain-boundaries.md)).
- Tile granularity and row order are fixed by spike S1, not by this ADR
  ([../architecture/memory.md](../architecture/memory.md#kv-tiles)).

## Evidence

Measured on the baseline ([../architecture/hardware.md](../architecture/hardware.md),
full raw telemetry in [../benchmarks/results.md](../benchmarks/results.md)):

- M0 probe: a 128 × 512 `OneComponent16Half` pixel buffer had 256-byte rows
  (multiple of 64 B) and a 16 KiB-aligned base; default `IOSurfaceGetAllocSize`
  was 131,136 B, not a 16 KiB multiple. R16Float (128 × 512) and RGBA16Float
  (32 × 512) textures were both created on that surface under
  `MTL_DEBUG_LAYER=1`, and `MLMultiArray(pixelBuffer:shape:)` accepted the
  surface with shape `[4, 128, 128]`.
- **Spike S1 (closed 2026-11-20, M2)** on Qwen3-30B-A3B KV geometry
  (`q=32, kv=4, d=128, chunk=512`, 32 chunks = 16,384 context, 2 rotated banks =
  64 MiB K+V, ReleaseFast commit `517ba0da`, `swapouts_delta = 0`):
  - Passing an explicit 16 KiB-aligned `kIOSurfaceAllocSize` yields exact
    `bytes_per_row = 256 B`, `data_bytes = alloc_bytes = surface_alloc_size = 524,288 B`
    (32 pages of 16 KiB) with 16 KiB-aligned base address.
  - `mlock()` over IOSurface-mapped pages succeeded on macOS 27.2
    (`iosurface_mlock_supported = yes`); `KvTile` exposes an optional `mlock`
    flag alongside `MTLResidencySet` tracking.
  - Texture reads match or exceed aliasing `MTLBuffer` reads (zero texture-view
    penalty): token-major `RGBA16Float` `80.23 GB/s`, `R16Float` `91.75 GB/s`,
    aliasing `MTLBuffer` `82.14 GB/s`; channel-major `RGBA16Float` `79.63 GB/s`,
    `R16Float` `80.59 GB/s`, aliasing `MTLBuffer` `78.69 GB/s`.
  - Production FlashDecoding split-KV GQA (`G4`) over bindless token-major
    `RGBA16Float` `KvTile`s reached `42.72 GB/s` (`24.55 us`) at `seq_len=512`,
    `78.42 GB/s` (`106.98 us`) at `seq_len=4096`, and **`85.61 GB/s` (`391.96 us`,
    71.3% of the 120 GB/s bus)** at `seq_len=16384`.
  - **Settled R4 layout**: one 512-token chunk per layer, **token-major
    `RGBA16Float`** (`width = head_dim / 4`, `height = num_kv_heads * 512`),
    giving single-instruction `tex.write(half4)` decode KV append without
    read-modify-write, clean partial-chunk `seq_len` bounds, and full-bus GQA
    reads.

## Open items

- int8 KV tiles (`OneComponent8`) — undecided (status row R5).
  - External survey ([open-decisions-research.md](../architecture/open-decisions-research.md#r5-int8-kv-tiles-onecomponent8)):
    Apple's [`init(pixelBuffer:shape:)`](https://developer.apple.com/documentation/coreml/mlmultiarray/init(pixelbuffer:shape:))
    docs restrict fp16 IOSurface `MLMultiArray`s to `OneComponent16Half`, while
    KIVI ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) and `mlx-lm`
    `8-bit g=64` / `llama.cpp q8_0` rely on explicit per-block scales to handle
    channel/token KV outliers; recommended direction is to keep
    `OneComponent16Half` for ANE↔GPU sharing and use group-64 `q8` `MTLBuffer`
    compression GPU-side only if needed.
