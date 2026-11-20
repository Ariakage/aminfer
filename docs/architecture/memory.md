# Memory model

> Design reference; status lives in [../status.md](../status.md).

## Page invariant

- `APPLE_PAGE_SIZE = 16384`. It equals `std.heap.page_size_min` for
  `aarch64-macos` (checked at compile time) and `hw.pagesize` (checked at
  startup).
- Every buffer shared with Metal (`newBufferWithBytesNoCopy`) or through
  IOSurface has a 16 KiB-aligned start address and a total length that is a
  multiple of 16 KiB.
- Our code enforces the rule with runtime checks at every bridge boundary —
  kept in ReleaseFast, because they run at startup, not per round. Metal
  itself accepted a non-multiple NoCopy length in our probe, even with
  `MTL_DEBUG_LAYER=1` ([hardware.md](hardware.md)), so its validation cannot be
  relied on.
- Rationale and consequences:
  [ADR-0002](../decisions/0002-page-alignment-and-zero-copy.md).

## Wired regions

- Allocated once at startup with `mmap(MAP_ANON | MAP_PRIVATE)` (page-aligned
  by construction), lengths rounded up to 16 KiB, then pinned with `mlock`.
- A failed `mlock` is an error, never a silent fallback to pageable memory.
- Regions are carved into arenas: dense weights and hot experts, KV tiles,
  activations and scratch, cold-expert slots, draft-model buffers, suffix
  automaton tables. Nothing is allocated after startup
  ([runtime-pipeline.md](runtime-pipeline.md#hot-path-rules)).

## Two limits, one budget

| Limit | Governs | Baseline default |
| --- | --- | --- |
| `vm.user_wire_limit`, `vm.global_user_wire_limit` | `mlock` (CPU-side wiring) | 18.24 GiB |
| `iogpu.wired_limit_mb` → `recommendedMaxWorkingSetSize` | GPU-resident working set | 18,186 MiB |

The same bytes count against both limits. The wired budget
([ADR-0012](../decisions/0012-wired-memory-budget-headroom.md)) is computed at
startup as the minimum of the configured cap, physical memory minus the
headroom (default 3 GiB) and both limits; a model plan larger than the budget
is rejected with a clear error. Raising the limits is described in
[../guides/system-tuning.md](../guides/system-tuning.md).

## Residency

- **CPU**: `mlock` of the wired regions.
- **GPU**: one `MTLResidencySet` holding every long-lived buffer and texture,
  committed at startup and attached to the Metal 4 command queue. Metal 4 does
  not track residency implicitly, so nothing outside the set may be used by a
  kernel ([ADR-0011](../decisions/0011-metal4-zero-allocation-hot-path.md)).
- **IOSurface tiles**: created at startup and added to the residency set.
  Spike S1 verified that `mlock()` succeeds on IOSurface-mapped pages on macOS
  27.2 (`KvTileOptions.mlock` is exposed alongside `MTLResidencySet` tracking).

## Zero-copy paths

| Producer → consumer | Mechanism |
| --- | --- |
| CPU ↔ GPU | Wired region wrapped once with `newBufferWithBytesNoCopy` (shared storage) |
| ANE ↔ GPU (KV, activations) | IOSurface created as an fp16 `CVPixelBuffer`; CoreML sees it as `MLMultiArray(pixelBuffer:shape:)` through output backings; Metal sees it through `newTextureWithDescriptor:iosurface:plane:` ([ADR-0004](../decisions/0004-kv-cache-iosurface-texture-sharing.md)) |
| NVMe → GPU | `F_NOCACHE` read straight into a wired slot whose `MTLBuffer` already exists ([ADR-0006](../decisions/0006-amf-cold-experts-direct-io.md)) |
| GPU → CPU (tokens, logits, acceptance) | Shared-storage buffers in a wired region; completion through `MTLSharedEvent` |

No tensor is ever copied between engines. Copies inside one engine — for
example moving the accepted path's KV rows into committed tiles on the GPU —
are ordinary kernel work.

## KV tiles

Accepted direction ([ADR-0004](../decisions/0004-kv-cache-iosurface-texture-sharing.md)):
KV lives in IOSurface-backed fp16 tiles that the ANE writes and the GPU reads
as textures with `tex.read(uint2(x, y))`.

Settled tile layout (confirmed in Spike S1, raw measurements in
[../benchmarks/results.md](../benchmarks/results.md)):

- One tile per layer, per K or V, per 512-token chunk (`R4`). For
  Qwen3-30B-A3B (4 KV heads, head dim 128) a K tile is
  4 × 512 × 128 × 2 B = 512 KiB: an fp16 surface of width 128 and height 2048
  with 256-byte rows and exact `kIOSurfaceAllocSize = 524,288 B` (32 pages of
  16 KiB).
- Tiles are created with an explicit `kIOSurfaceAllocSize` multiple of 16 KiB
  and base-address / 64 B row-stride checks at startup (`src/bridge/apple_bridge.m`).
- **Token-major rows** (`row = kv_head * chunk_tokens + slot_in_chunk`,
  `col = dim / 4`) viewed as **`RGBA16Float`** (`width = head_dim / 4`, e.g. 32
  for `head_dim = 128`). Spike S1 showed texture reads match or beat aliasing
  `MTLBuffer` views (`80.23–91.75 GB/s` vs `82.14 GB/s`), and production
  FlashDecoding GQA over token-major `RGBA16Float` tiles reaches `85.61 GB/s`
  at 16K context while keeping decode KV append a single `tex.write(half4)`
  with no read-modify-write. `R16Float` views remain supported by `KvTile`.
- Decode-time KV produced by the GPU is written into the same tile format
  (`ami_qknorm_rope_kv_write`), so one attention kernel family reads both
  prefill and decode KV.
- Kernels reach arbitrarily many resident chunks (up to 32K+ context) through a
  bindless `KvTilePair` (`{k_tex_id, v_tex_id}`) buffer in `WiredArena` paired
  with `MTLResidencySet` registration (`makeTilesResident`).
- Still undecided: int8 tiles (`OneComponent8`, status row R5).
- Unquantized `BF16` models (confirmed for M2.5,
  [ADR-0016](../decisions/0016-native-fp16-bf16-and-multishard-import.md)) keep
  `KvTile` in `OneComponent16Half` / `RGBA16Float`: neither `MTLPixelFormat` nor
  CoreML's zero-copy `MLMultiArray(pixelBuffer:shape:)` offers a `BF16`
  IOSurface format, and post-QK-Norm / post-RoPE K/V vectors fit comfortably in
  IEEE `fp16` range.

## Budget estimates

Estimates for the primary target on the 24 GB baseline, derived from public
model configurations (Qwen3-30B-A3B: 30.5B parameters, 48 layers, 4 KV heads,
head dim 128, 128 experts, top-8, expert intermediate size 768). Not measured.

| Item | 3-bit weights | 4-bit weights |
| --- | --- | --- |
| Weights (MLX-style affine, group 64: about 3.5 / 4.5 bits per weight) | ≈ 12.4 GiB | ≈ 16.0 GiB |
| KV cache, 32K tokens, fp16 (96 KiB per token) | 3.0 GiB | 3.0 GiB |
| EAGLE-3 head (≈ 120M parameters, fp16) | ≈ 0.22 GiB | ≈ 0.22 GiB |
| ANE attention-block weights for the hybrid prefill (≈ 0.9B parameters, 4-bit palettized) | ≈ 0.42 GiB | ≈ 0.42 GiB |
| Activations and scratch | ≈ 0.5 GiB | ≈ 0.5 GiB |
| Suffix automaton tables (32K tokens) | < 4 MiB | < 4 MiB |
| **Total** | **≈ 16.5 GiB** | **≈ 20.1 GiB** |

- The 3-bit plan fits under the default limits with about 7.5 GiB left for
  macOS, so cold experts on NVMe are not needed for it.
- The 4-bit plan exceeds the defaults. It needs raised limits (still leaving
  ≥ 3 GiB of headroom) or moving the coldest experts to NVMe; a 4-bit expert
  of this model is 2,654,208 B, exactly 162 pages of 16 KiB.
- Standard unquantized `FP16` / `BF16` weights (2 B/parameter, confirmed for
  M2.5 in [ADR-0016](../decisions/0016-native-fp16-bf16-and-multishard-import.md))
  scale across hardware tiers without changing the memory model: the **24 GB M4
  baseline** wires up to the ~8B class (`Qwen3-0.6B` ≈ 1.2 GiB, `4B` ≈ 7.6 GiB,
  `8B` ≈ 15.3 GiB), whereas full-wired `Qwen3-14B` (`BF16`, ≈ 27.6 GiB) and
  `Qwen3-30B-A3B` (`BF16`, ≈ 56.9 GiB) target **36 GB / 64 GB / 128 GB+** Macs
  (and are rejected upfront on 24 GB M4 by startup budget validation rather than
  swapping).

## No swap

Hot data never pages: wired regions cannot, and nothing on the hot path lives
outside them. The benchmark harness (status row C9) records `vm_stat` swap-outs
and `vm.swapusage` before and after each run and fails the run if either grows.
