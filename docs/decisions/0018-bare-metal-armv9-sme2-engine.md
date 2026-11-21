# ADR-0018: Bare-metal ARMv9 SME2 pre-packed and ZT0-LUT engine (M4.5)

- Status: Accepted
- Date: 2026-11-21
- Source: user proposal, microarchitectural trade-off review, and formal confirmation

## Context

On Apple Silicon M1–M3, CPU matrix acceleration relied on undocumented Apple AMX
instructions reachable only indirectly through Apple's `Accelerate` framework
(`cblas_sgemm` / `BNNS`). Starting with **Apple M4 (`ARMv9.2-A`)**, Apple
publicly exposes standard **ARM Scalable Matrix Extension (SME and SME2)**
(`FEAT_SME`, `FEAT_SME2`, `FEAT_SME_F16F32`, `FEAT_SME_B16F32`,
`FEAT_SME_I16I64`) directly to user-space software:

- **Streaming Vector Length (`SVL`)**: `512 bits (64 B)` per `Z` register
  (`16 × f32`, `32 × bf16/f16`, or `64 × u8` across `Z0..Z31`).
- **`ZA` 2D matrix array**: `64 × 64 B = 4 KiB` architectural accumulator state,
  addressable simultaneously as **four `16 × 16` `FP32` tiles (`ZA0.S..ZA3.S`)**
  fed by 2-way widening outer-product instructions (`bfmopa` / `fmopa`,
  `512 FLOPs` per instruction, `2,048 FLOPs` per 4-tile quad).
- **`ZT0` dedicated lookup-table register**: `512 bits (64 B)` holding 32
  16-bit (`bf16` / `f16`) entries, enabling single-instruction multi-vector
  2-bit (`luti2`) and 4-bit (`luti4`) register-resident codebook expansion
  straight into streaming `Z` registers without L1 cache bank conflicts.
- **P-core cache and cluster topology**: 16 MiB shared L2 cache on the base M4
  P-core cluster (`hw.perflevel0.l2cachesize`), scaling to 2–4 independent
  P-core clusters (each with its own SME unit) on M4 Pro, Max, and Ultra.

Relying on system `Accelerate` routines leaves substantial performance on the
table for LLM inference shapes:

1. **Non-square LLM matrix shapes (`M = 1` decode, `M = 5..16` speculative tree
   verification, `M = 128..512` chunked prefill)** suffer from black-box
   heuristics, runtime transpose/packing overhead, under-filled outer-product
   tiles, and suboptimal multi-cluster thread dispatch.
2. **Quantized weights (`2-bit` / `3-bit` / `4-bit`)** cannot be dequantized
   inside `Accelerate` registers; pre-dequantizing to `fp16` / `fp32` in unified
   DRAM multiplies memory traffic during bandwidth-bound decode.
3. **Pure C `<arm_sme.h>` intrinsics have toolchain and codegen limitations in
   our stack**:
   - Zig `0.17.1` removed `@cImport`
     ([ADR-0001](0001-language-and-toolchain-boundaries.md)), and
     `b.addTranslateC` cannot translate `<arm_sme.h>` sizeless SVE types
     (`svbfloat16_t`, `svuint8x4_t`) or ACLE streaming-state attributes
     (`__arm_streaming`, `__arm_inout("za")`, `__arm_in("zt0")`).
   - Compiler-lowered `__arm_locally_streaming` intrinsics around unrolled
     `svluti4_lane_zt` + 4-tile `svbfmopa_za32_bf16_m` loops can spill `Z` or
     predicate registers and cannot guarantee bubble-free software pipelining
     mixed with streaming non-temporal loads (`ldnt1b`).

Because offline or load-time pre-packing aligns every weight panel to `SVL = 64 B`
boundaries, the innermost SME2 loop requires no complex control flow or scalar
tail cleanup—making a **90% pure Zig outer engine + ~10% ultra-thin `.s`
assembly microkernel** architecture both minimal (~250–350 lines of assembly)
and uncompromising on throughput.

## Decision

Insert **Milestone M4.5** immediately after M4 and before M5 (authoritative
capability status tracked in [../status.md](../status.md) under `C11`–`C13` and
`W15`, plus Spike `S7` in [../process.md](../process.md#spikes)), structured
around six microarchitectural pillars and five mandatory safety invariants:

### A. Microarchitectural design (90% pure Zig + ~10% `.s` microkernels)

1. **Clean split between `src/cpu/sme.zig` and `src/cpu/sme2_kernels.s` (`C11`).**
   - `src/cpu/sme.zig` (pure Zig) owns startup `sysctl` feature and cluster
     topology discovery, panel pre-packing, L2 cache `M/N/K` blocking, tail
     padding, and lock-free multi-cluster P-core work distribution.
   - `src/cpu/sme2_kernels.s` (~250–350 lines of commented ARMv9.2-A assembly,
     compiled natively via `mod.addAssemblyFile` in `build.zig`) implements only
     fixed-ABI (`callconv(.c)`) inner panel loops over pre-packed buffers.

2. **Three-way `M`-dimension microkernel specialization (`C12`).**
   Rather than forcing a single square `32 × 32` tile across all batch sizes,
   `src/cpu/sme.zig` dispatches at zero dynamic inner-loop cost across three
   geometry-matched `.s` kernels:
   - **`M = 1..4` (single-token decode, MoE router gates, one-shot cold-expert
     streaming)**: **SME2 multi-vector `ZT0-LUT + Dot` streaming bandwidth
     kernel**. Uses 4-vector contiguous burst loads
     (`ld1b {z16.b - z19.b}, pn8/z, [w_ptr]`, `256 B` per instruction) paired
     with multi-vector `luti4` / `luti2` table expansion from `ZT0` and
     streaming dot-product accumulation across `Z` registers—eliminating 15/16
     idle `ZA` rows and `mova` slice-extraction overhead at `M = 1`.
   - **`M = 5..16` (speculative tree verification `G5` and `D4` small-draft
     batch expansion)**: **`1 × 4` wide-`N` outer-product kernel (`16 × 64`
     output across `ZA0.S..ZA3.S`)**. Broadcasts/loads one `16`-row activation
     vector into `Z0` (`whilelt` predicated for `M < 16`) against four distinct
     `16`-column weight panels (`64` columns total), achieving **100% `ZA` tile
     occupancy** on default 16-node speculative trees (versus 50% waste on a
     `32 × 32` `2 × 2` tile).
   - **`M >= 32` (prompt prefill and large batches)**: **`2 × 2` balanced
     outer-product kernel (`32 × 32` output across `ZA0.S..ZA3.S`)**. Pairs two
     activation vectors (`Z0, Z1`) with two weight vectors (`Z2, Z3`) across
     four `bfmopa` / `fmopa` instructions (`2,048 FLOPs` per quad), doubling
     register operand reuse and halving L1 load-port pressure.

3. **Hardware-native K-interleaved `sme2_panel64` pre-packing with inline
   group metadata (`W15`).**
   - ARMv9 2-way widening `bfmopa` and `fmopa` consume pairs of adjacent `K`
     elements interleaved across 16 columns (`[16_cols, 2_k] = 64 B`).
   - Offline `aminfer-tools pack-amf` (opt-in) or startup `WiredArena` / slot
     loading packs SME2-assigned matrices into `[N/16, K/2, 16_cols, 2_k]`
     panels and places each `group_size = 64` block's `scale`/`bias` (or
     16-entry `bf16`/`f16` `ZT0` codebook for
     [ADR-0017](0017-optional-gsq-rco-hybrid-quantization.md) LUT schemes)
     inline immediately ahead of its packed weight bytes.
   - Inner loops execute **zero runtime `zip1`/`zip2`/`trn` permute
     instructions**, **zero secondary scale-pointer chasing**, and **zero extra
     TLB misses**.

4. **Macro-task `smstart` residency and ping-pong software pipelining (`C11`,
   `C12`).**
   - Worker threads enter Streaming SVE + `ZA` + `ZT0` (`smstart`) **once per
     macro-task** (spanning an entire projection or L2 column panel) and exit
     (`smstop`) once on completion, amortizing streaming mode-switch latency to
     `< 20 ns` per layer.
   - Inside the K-loop, `Z16..Z19` and `Z20..Z23` alternate in a two-stage
     ping-pong pipeline so next-step `ld1b`/`ldnt1b` loads and `luti4`/`luti2`
     expansions overlap with current-step `bfmopa`/`fmopa` outer products.

5. **Dual-locality cache steering (`ld1b` vs streaming non-temporal `ldnt1b`,
   `C12`).**
   - **Hot L2-resident weights** (MoE routers, small draft projections, shared
     experts) use normal `ld1b` streaming loads paired with `M/N/K` blocking fit
     inside the 16 MiB P-core L2 cache.
   - **One-shot NVMe cold experts** streamed into `ExpertSlotPool` (`C5`,
     [ADR-0006](0006-amf-cold-experts-direct-io.md)) use **streaming
     non-temporal loads (`ldnt1b`)**, bypassing L2 allocation so multi-megabyte
     cold expert blobs never evict hot activations, suffix-automaton tables, or
     routers from P-core L2.

6. **Fused in-register `ZA` readout epilogue and cluster-aware lock-free
   scheduling (`C11`, `C12`).**
   - As `mova` extracts `FP32` rows from `ZA0.S..ZA3.S`, the epilogue fuses
     group scale/bias application, optional residual addition (`+ residual`) or
     SwiGLU gating (`silu(gate) * up`), and narrowing (`f32 → bf16` or `f16`)
     directly in streaming SVE registers before writing once to unified memory.
   - At startup, `src/cpu/sme.zig` queries `sysctl` (`hw.perflevel0.physicalcpu`,
     `hw.perflevel0.cpusperl2`, `hw.perflevel0.l2cachesize`) to count physical
     P-core clusters, launches the Spike-`S7`-calibrated thread count per
     cluster under `QOS_CLASS_USER_INTERACTIVE`, and claims macro-tiles from a
     128 B cacheline-padded `std.atomic.Value(u32)` counter with zero locks and
     zero hot-path heap allocation.

### B. Mandatory architectural safeguards

1. **Pre-`smstart` AAPCS64 `d8–d15` preservation and register-clobber canary
   tests.**
   Executing `smstart` architecturally zeroes all NEON/SVE vector registers
   `v0–v31`, including AAPCS64 callee-saved `d8–d15`. Every `.s` entry point
   must use a shared assembly prologue/epilogue macro that saves `d8–d15` and
   `x19–x30` **before** `smstart`, restores them **after** `smstop`, and emits
   `.cfi_*` unwind directives for `lldb`. `zig build test` must include both:
   - a **register-clobber canary test** loading sentinel bit patterns into
     `d8–d15` and `x19–x28` across every `.s` entrypoint and asserting bit-exact
     preservation on return; and
   - randomized and boundary-shape numerical parity tests against pure Zig
     `fp32` scalar references in Debug, ReleaseSafe, and ReleaseFast.

2. **Single-owner weight layout invariant (no duplicate wired copies).**
   Every tensor or routed expert is stored in memory in **at most one layout**
   (`row_major` for Metal GPU execution OR `sme2_panel64` for CPU SME2
   execution)—never duplicated in both formats—preserving the startup wired
   memory budget ([ADR-0012](0012-wired-memory-budget-headroom.md)).

3. **Spike `S7` unified-bus contention and placement gate (`C13`).**
   CPU SME2 and Metal GPU share the 120 GB/s M4 unified memory fabric (cf.
   Spike `S4` in [ADR-0008](0008-benchmark-gated-draft-placement.md)).
   P-core L2-resident SME2 workloads (MoE router gates `< 2 MiB`, Top-K /
   categorical sampling, L2-cached draft/head ops) may overlap freely with GPU
   passes; DRAM-streaming SME2 co-execution alongside GPU verification is
   governed by **Spike `S7`** and enabled automatically only when measured
   end-to-end `tok/s` improves by `> 1.05×` with `swapouts_delta == 0`.

4. **M1–M3 backward compatibility without `SIGILL`.**
   The platform floor ([ADR-0011](0011-metal4-zero-allocation-hot-path.md))
   includes M1–M3 Apple Silicon Macs that lack `FEAT_SME2`. The
   `-march=armv9.2-a+sme2+sme-b16f32+sme-f16f32+sme-i16i64` flag is passed
   **only** to `src/cpu/sme2_kernels.s` (never to global Zig target flags), and
   `src/cpu/sme.zig` gates all `.s` calls behind startup
   `sysctlbyname("hw.optional.arm.FEAT_SME2")` (plus `FEAT_SME_B16F32` /
   `FEAT_SME_F16F32`) checks, falling back cleanly to pure Zig SIMD / Metal GPU
   execution when unavailable.

5. **100% binary backward compatibility with existing `.amf` v0 checkpoints
   (`W15`).**
   `TensorLayout` (`0 = row_major`, `1 = sme2_panel64`) is stored in the low
   byte of the existing zero-initialized `TensorEntry.reserved0` (`u16` at byte
   offset 58 of the 128-byte `TensorEntry` in `src/format/amf.zig`; high byte
   shared with `QuantParamKind` in
   [ADR-0017](0017-optional-gsq-rco-hybrid-quantization.md)). Existing `.amf` v0
   files already carry `reserved0 == 0` and remain valid without repacking.

## Consequences

- [ADR-0001](0001-language-and-toolchain-boundaries.md) gains one narrowly
  scoped assembly allowance (`src/cpu/*.s`) compiled through `build.zig` with
  zero third-party dependencies and zero C/Objective-C glue.
- M4+ P-cores evolve from scalar/NEON control processors into a high-throughput
  matrix + LUT engine capable of accelerating L2-resident routers, small draft
  models, prefill tiles, and `ldnt1b` cold-expert evaluation while leaving
  M1–M3 compatibility intact.
- Capability rows `C11` (SME2 runtime detection, ABI harness & cluster
  scheduler), `C12` (3-way `bfmopa`/`fmopa` & `ZT0` `luti2`/`luti4` + `ldnt1b`
  microkernels + fused epilogue), `C13` (Spike-`S7`-gated heterogeneous
  CPU-SME2 / GPU placement), and `W15` (`sme2_panel64` pre-packing) are added
  to [../status.md](../status.md) under Milestone **M4.5**.

## Evidence

- Apple M4 (`ARMv9.2-A`) architectural features (`FEAT_SME`, `FEAT_SME2`,
  `FEAT_SME_F16F32`, `FEAT_SME_B16F32`, `SVL = 512-bit`, 16 MiB P-cluster L2)
  on the baseline host ([../architecture/hardware.md](../architecture/hardware.md)).
- Existing `.amf` v0 `TensorEntry` (`src/format/amf.zig`) is a fixed 128-byte
  `extern struct` with zero-initialized `reserved0: u16` and `reserved1: u64`.
- End-to-end microkernel throughput, threads-per-cluster saturation, `otool`
  instruction-stream verification (zero spills), and concurrent GPU bus
  contention are measured in **Spike `S7`** during M4.5 and archived in
  [../benchmarks/results.md](../benchmarks/results.md).

## Open items

- **Spike `S7` (M4.5)**: measure standalone M4 SME2 throughput (`GB/s` and
  `TFLOP/s`) across `M = 1..4` (`ZT0-LUT + Dot`), `M = 5..16` (`1 × 4` `ZA`),
  and `M >= 32` (`2 × 2` `ZA`) vs `Accelerate` `cblas_sgemm`/`BNNS`, sweep 1/2/4
  threads on the M4 P-cluster, and calibrate the concurrent GPU + SME2 net
  speedup threshold (`C13`).
