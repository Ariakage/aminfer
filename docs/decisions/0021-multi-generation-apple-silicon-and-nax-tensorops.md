# ADR-0021: Single-binary multi-generation Apple Silicon adaptation (`Apple9` and `Apple10+` GPU `NAX`)

- Status: Accepted
- Date: 2026-11-22
- Source: user roadmap confirmation following post-M6.8 `Apple10+` / `NAX` technical review

## Context

With Milestones M0–M6.8 complete on the 24 GiB Apple M4 (`MTLGPUFamilyApple9`)
baseline ([../benchmarks/results.md](../benchmarks/results.md)), `aminfer`
already reserves `supports_apple10` and `PrefillMatmulBackend` (`G16`,
[ADR-0020](0020-wide-gpu-prefill-tail-pruning-qos-and-live-ram-guards.md)).
Toolchain inspection on the local build host (`MacOSX27.0.sdk`, Xcode 27.1,
`metalfe-32023.921.6`) and measured M4 roofline telemetry establish five facts
governing support for M4 alongside post-M4 (`Apple10+`, M5 and later) Apple
Silicon:

1. **Existing `<metal_simdgroup_matrix>` shaders run on `Apple10+` without
   targeting GPU Neural Accelerators (`NAX`).** `simdgroup_half8x8` and
   `simdgroup_bfloat8x8` remain binary- and source-compatible on
   `MTLGPUFamilyApple10` (`1010`) and `MTLGPUFamilyApple11` (`1011`) and scale
   automatically with wider memory buses, whereas `NAX` tensor hardware is
   addressed only through Metal 4 TensorOps
   (`<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`,
   `mpp::tensor_ops::matmul2d` / `cooperative_tensor`, or
   `MTL4MachineLearningCommandEncoder` with `MTLTensor`).
2. **Compute-bound prefill, high-batch verification, vision, and M7 `FP4` gain
   from `NAX`; `M = 1` decode remains bandwidth-bound.** On M4 (`120.0 GB/s`
   bus), `M = 1` `gemv_q4` already reaches `98.74 GB/s` (`82.3%` of bus) and
   `gemv_bf16` reaches `88.25 GB/s` (`0.088 TFLOP/s`), so single-request decode
   cannot be accelerated by higher matrix TFLOP/s and only needs SIMT occupancy
   scaled to wider buses. Conversely, wide GPU prefill (`M = 64..1024+`, `G15`),
   high-batch `C14` / speculative verification (`M = 16..64+`), `G14` ViT patch
   projections, and planned M7 `FP4` (`R7`, `G11`) are compute-bound on M4
   (`gemm_f16` `64 × 2048 × 2048` reaches `2.690 TFLOP/s` at `37.2%` bus;
   `gemm_bf16` reaches `2.509 TFLOP/s` at `34.7%` bus) and directly benefit from
   `NAX`.
3. **Group-64 `.amf` affine weights require staged or `K = 64` sub-block `NAX`
   feeding, whereas M7 `FP4` maps directly.** `MPPTensorOpsMatMul2d.h` natively
   accepts `half`, `bfloat`, `int8`/`uint8`, `int4b`/`uint4b`, `int2b`/`uint2b`,
   `metal_fp4_e2m1_format`, `metal_fp8_e4m3_format`, and `metal_fp8_e5m2_format`,
   and compiles cleanly to AIR under `build.zig`'s exact flags
   (`-std=metal4.0 -Wall -Werror -mmacosx-version-min=26.0`, backed by
   `applegpu-nt/tensor.metallib` and `applegpu25-nt/tensor.metallib` in the
   toolchain). Because `.amf` `q2_affine`, `q3_affine` (packed 3-bit), and
   `q4_affine` store per-64-element `(scale, bias)` pairs, they cannot bind as a
   single unscaled `uint4b` `MTLTensor`; `NAX` execution on `.amf` affine weights
   must either dequantize `K = 64` panels into threadgroup `half`/`bfloat` tiles
   ahead of `matmul2d` or accumulate per-64-element `cooperative_tensor` slices.
   M7's `FP4` (`E2M1` + `E4M3` scale, `R7`/`G11`) aligns directly with native
   `MPPTensorOpsMatMul2d` formats.
4. **One binary and one `.metallib` can serve `M1–M3`, `M4` (`Apple9`), and
   `After-M4` (`Apple10+`) concurrently.** On the CPU side, `build.zig` isolates
   ARMv9.2 SME2 assembly in `src/cpu/sme2_kernels.s` behind `SmeCaps.probe()`
   ([ADR-0018](0018-bare-metal-armv9-sme2-engine.md)). On the GPU side,
   `ami_library_create` (`src/bridge/apple_bridge.m`) parses embedded AIR bitcode
   without compiling unused kernels, and `ami_pipeline_create_specialized`
   invokes `MTL4Compiler` per requested entry point at startup. Co-packaging
   `*_wide` (`simdgroup_matrix`) and `*_nax` (`mpp::tensor_ops::matmul2d`)
   kernels in `aminfer.metallib` adds zero runtime overhead on `Apple9` while
   enabling `NAX` on `Apple10+` from the same artifact.
5. **Bus-bandwidth constants and heterogeneous gates are currently M4-specific.**
   `src/gpu/attention.zig` hardcodes `M4_BUS_BANDWIDTH_GBS = 120.0` for `% bus`
   reporting, and heterogeneous crossover choices (`A1` ANE wavefront vs `G15`
   GPU prefill, `D5` draft placement, `C13` SME2+GPU co-execution) depend on the
   host chip's measured GPU, ANE, and memory-bus throughput.

The user confirmed codifying this cross-generation architecture into **Milestone
M6.9** (`G17`, `C16`, Spike `S8` in [../status.md](../status.md) and
[../process.md](../process.md)) prior to M7.

## Decision

### 1. Single-binary, single-`.metallib` dual GPU TensorOps backend (`G17`, M6.9)

- Extend `apple_bridge` and `metal.DeviceInfo` to report the highest supported
  Apple GPU family (`gpu_apple_family: u8`, probing `MTLGPUFamilyApple9 = 1009`,
  `MTLGPUFamilyApple10 = 1010`, and `MTLGPUFamilyApple11 = 1011`) alongside CPU
  `SmeCaps`.
- Build `MetalPerformancePrimitives` (`<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`,
  `mpp::tensor_ops::matmul2d` with `execution_simdgroups<4>`) entry points into
  the single embedded `aminfer.metallib`:
  - `ami_dense_gemm_nax` (`f16` / `bf16` with `fp32` `cooperative_tensor`
    accumulation) for wide prefill (`G15`), batched decode/verification (`C14`,
    `G5`), and `G14` vision patch projections;
  - `ami_dequant_gemm_nax` staging group-64 `.amf` `q2_affine`, `q3_affine`, and
    `q4_affine` weights into threadgroup `half` tiles fed into `matmul2d` (`BM =
    64, BN = 64, BK = 64`);
  - `FP4` (`metal_fp4_e2m1_format` + `metal_fp8_e4m3_format`) kernel hooks ready
    for M7 (`R7`, `G11`).
- Instantiate `*_nax` `MTLComputePipelineState` objects lazily via `MTL4Compiler`
  only when `gpu_apple_family >= 10` (or when explicitly requested via
  `--prefill-matmul-backend auto|simdgroup|nax`, default `auto`), leaving `Apple9`
  startup and steady-state execution unchanged.
- Keep `M = 1` decode GEMV on SIMT across all generations, scaling threadgroup
  dispatch geometry to device GPU core counts rather than routing single-row
  vectors through `NAX`.

### 2. `G16` canary-guarded `NAX` validation and automatic fallback (`G16` + `G17`, M6.9)

- When `auto` selects `NAX` on an `Apple10+` device, run the `G16` startup
  numerical canary (`f16`, `bf16`, `q4_affine`, including signed and exponent-edge
  patterns) through the `*_nax` pipelines before serving traffic.
- If `*_nax` fails finiteness or `fp32` tolerance checks due to an offline
  `metalfe` or OS driver defect, emit an explicit diagnostic record and fall back
  automatically to the verified `simdgroup_matrix` (`*_wide`) pipelines in the
  same `.metallib`; opt-in `--strict-nax` turns fallback into a hard startup
  error.

### 3. Chip-parameterized bandwidth telemetry and runtime crossover calibration (`C16`, M6.9)

- Replace hardcoded `M4_BUS_BANDWIDTH_GBS = 120.0` in `% bus` reporting with a
  chip-profile lookup (`MTLDevice.name` + `sysctl` core/family facts, overridable
  via `--bus-bandwidth-gbs <f64>`) covering M1–M4 / Pro / Max / Ultra and M5+
  tiers, reported explicitly as `spec_bus_gbs`.
- Parameterize runtime crossover gates per host chip rather than assuming M4
  ratios:
  - Pure-GPU wide prefill (`G15`/`G17`) versus ANE hybrid wavefront prefill
    (`A1`, [ADR-0010](0010-hybrid-wavefront-prefill.md));
  - `D5` draft-head placement (`gpu` vs `ane` vs `cpu_sme2`,
    [ADR-0008](0008-benchmark-gated-draft-placement.md)) and `C13` SME2+GPU
    co-execution ([ADR-0018](0018-bare-metal-armv9-sme2-engine.md)).

### 4. Strict two-stage verification boundary (`Apple9` build/fallback vs `Spike S8` on `Apple10+`)

- **Verifiable on the M4 (`Apple9`) baseline in M6.9**:
  1. Offline `xcrun metal` AIR + `.metallib` compilation of `*_nax` alongside
     `*_wide` under `-std=metal4.0 -Wall -Werror -mmacosx-version-min=26.0`;
  2. Lazy `MTL4Compiler` pipeline selection (`Apple9` skips `*_nax` PSO creation
     in `auto` mode with zero M4 latency or memory regression) plus `-Dcpu=apple_m1`
     cross-generation binary ABI checks;
  3. Unit-tested `G16` canary failure injection verifying automatic
     `nax -> simdgroup` fallback and `--strict-nax` rejection;
  4. Chip-profile `spec_bus_gbs` resolution and runtime prefill/draft crossover
     micro-calibration.
- **Gated on `Spike S8` on physical `Apple10+` (M5+) hardware**:
  - Actual `NAX` hardware execution, numerical parity, TFLOP/s, and TTFT speedup
    measurements cannot run on M4 (`Apple9`) and must never be simulated or
    claimed from M4 runs. Until `Spike S8` executes on an `Apple10+` machine,
    `docs/benchmarks/results.md` and `docs/status.md` distinguish M4-verified
    host/AIR/fallback infrastructure from pending `Apple10+` hardware numbers.

## Consequences

- **Positive**:
  - One macOS `arm64` binary and one embedded `aminfer.metallib` cover `M1–M3`
    (NEON + `simdgroup_matrix`), `M4` (SME2 + `simdgroup_matrix`), and `M5+`
    (SME2 + `NAX` `MetalPerformancePrimitives`) with zero hot-path branches or
    steady-state allocations ([ADR-0011](0011-metal4-zero-allocation-hot-path.md),
    [ADR-0014](0014-compile-time-specialization.md)).
  - `G16` startup canary protects `Apple10+` deployments from offline `metalfe`
    `bf16` TensorOps regressions by falling back cleanly to `simdgroup_matrix`.
  - Bandwidth utilization telemetry (`% spec_bus_gbs`) and ANE/GPU/SME2
    placement gates remain accurate on Max/Ultra and post-M4 machines instead of
    reporting `> 100%` against M4's `120 GB/s` constant.
- **Trade-offs and boundaries**:
  - Including `*_nax` bitcode increases embedded `.metallib` size modestly while
    leaving `Apple9` PSO compile time unchanged due to per-function lazy
    compilation.
  - Group-64 `q2`/`q3`/`q4_affine` `NAX` kernels incur threadgroup staging cost
    relative to native `f16`/`bf16`/`fp4_e2m1`; `Spike S8` on real `Apple10+`
    silicon will gate whether staged group-64 `NAX` beats `simdgroup_matrix` per
    `M` bucket (`64`, `128`, `512+`).

## Evidence

- M4 (`Apple9`) GEMV vs GEMM bus/compute telemetry and M6.8 `G15`/`G16` baseline
  in [../benchmarks/results.md](../benchmarks/results.md).
- Local `MacOSX27.0.sdk` headers (`MTLDevice.h` `MTLGPUFamilyApple10 = 1010`,
  `MTLGPUFamilyApple11 = 1011`; `MPPTensorOpsMatMul2d.h`) and verified
  `metalfe-32023.921.6` AIR compilation of `mpp::tensor_ops::matmul2d`.

## Open items

- **Spike `S8`** ([../process.md](../process.md#spikes)): run `aminfer bench-m69`
  on physical `Apple10+` (M5+) hardware to record `NAX` `f16`/`bf16`, staged
  group-64 `q2`/`q3`/`q4_affine`, and `FP4` TFLOP/s, `G16` canary margins, and
  GPU-vs-ANE wavefront prefill crossover points in
  [../benchmarks/results.md](../benchmarks/results.md).
