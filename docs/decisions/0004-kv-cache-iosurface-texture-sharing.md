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

Measured on the baseline ([../architecture/hardware.md](../architecture/hardware.md)):

- A 128 × 512 `OneComponent16Half` pixel buffer has 256-byte rows (multiple of
  64 B) and a 16 KiB-aligned base; `IOSurfaceGetAllocSize` was 131,136 B, not a
  16 KiB multiple.
- R16Float (128 × 512) and RGBA16Float (32 × 512) textures were both created on
  that surface, also with `MTL_DEBUG_LAYER=1`.
- `MLMultiArray(pixelBuffer:shape:)` accepted the surface with shape
  `[4, 128, 128]`.

## Open items

- Spike S1: RGBA16Float versus R16Float texture reads versus an aliasing
  `MTLBuffer` view; token-major versus channel-major rows; whether the mapped
  range should also be `mlock`ed.
- int8 KV tiles (`OneComponent8`) — undecided (status row R5).
