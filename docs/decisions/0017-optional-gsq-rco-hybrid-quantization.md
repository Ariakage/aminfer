# ADR-0017: Optional offline GSQ + RCO per-expert / per-layer hybrid quantization

- Status: Accepted
- Date: 2026-11-21
- Source: user roadmap direction and confirmed design invariants

## Context

Through M1–M2.5, aminfer established its 16 KiB-aligned `.amf` v0 container
([ADR-0006](0006-amf-cold-experts-direct-io.md)), compile-time-specialized
3-bit and 4-bit MLX-affine group-64 Metal 4 kernels (`G1`, `G2`,
[ADR-0014](0014-compile-time-specialization.md)), native `FP16` / `BF16`
execution (`W12`, `W13`, `G12`,
[ADR-0016](0016-native-fp16-bf16-and-multishard-import.md)), deduplicated CSR
MoE execution (`G3`), and startup wired-budget enforcement
([ADR-0012](0012-wired-memory-budget-headroom.md)), alongside the accepted
expert-byte-budget draft-tree pruning design
([ADR-0009](0009-expert-byte-budget-tree-pruning.md)).

On 24 GB unified-memory Apple Silicon and on large Tier 1–3 MoE models
(`Qwen3-30B-A3B`, `Qwen3.8-Flash-Next`, `DeepSeek-V4.1-Flash-1017`;
[ADR-0015](0015-tiered-frontier-model-roadmap.md)), uniform-precision
quantization creates a sharp trade-off between output fidelity and byte
footprint:

- Uniform `q4_affine` (~16.0 GiB weights on `Qwen3-30B-A3B`) crowds the
  wired-memory budget on 24 GB M4, leaving less room for long-context KV tiles
  and speculative draft heads, while uniform `q3_affine` (~12.4 GiB) or
  aggressive uniform 2-bit quantization degrades sensitive attention
  projections, early/late layers, routers, shared experts, and high-sensitivity
  routed experts.
- In sparse MoE checkpoints, sensitivity and routing frequency are highly
  non-uniform across layers and experts: low-sensitivity or rarely routed cold
  experts can tolerate lower precision (`q3_affine` or `q2_affine`) with
  minimal perplexity loss when paired with offline residual error compensation,
  whereas sensitive projections and hot experts benefit from higher precision
  (`q4_affine`, `q6_affine`, `q8_affine`, `f16`, or `bf16`).

At the same time, two common mixed-precision approaches would violate aminfer's
core engineering invariants:

- **Intra-tensor / sub-matrix mixed-bit packing** (varying bit-widths across
  channels, rows, or groups inside a single weight matrix) breaks 128-bit
  coalesced memory transactions, eliminates deterministic row strides, and
  introduces simdgroup divergence on Apple Silicon GPUs.
- **Runtime sensitivity heuristics or Python-backed quantization passes** would
  violate the zero-Python runtime boundary
  ([ADR-0001](0001-language-and-toolchain-boundaries.md)), add branches to hot
  loops ([ADR-0014](0014-compile-time-specialization.md)), and jeopardize
  zero-allocation Metal 4 command encoding
  ([ADR-0011](0011-metal4-zero-allocation-hot-path.md)).

## Decision

Adopt an **optional offline GSQ + RCO hybrid quantization pipeline** (Confirmed
design targeting M4/M5 — capability IDs `W14` and `G13` in
[../status.md](../status.md), not yet implemented) governed by five binding
invariants:

1. **100% offline Python sensitivity analysis (GSQ) and error compensation (RCO)
   in `tools/` only ([ADR-0001](0001-language-and-toolchain-boundaries.md),
   `W14`).**
   - **GSQ (Group / Layer / Expert Sensitivity-Guided Quantization)** runs
     exclusively offline in Python under `tools/` during `.amf` export. Using
     calibration-set sensitivity metrics (combined with offline expert heat
     statistics when available), GSQ solves for a discrete per-tensor /
     per-expert precision assignment under a target byte budget.
   - **RCO (Residual Compensation Optimization)** runs exclusively offline in
     `tools/` during `.amf` export to reconstruct quantized weights and
     compensate quantization residual error directly into the packed integer
     codes and affine `half2(scale, bias)` parameters before the `.amf` file is
     written and checksummed.
   - Zero Python, calibration datasets, sensitivity scoring, or compensation
     math ever enters the Zig inference runtime (`src/`), Objective-C bridge
     (`src/bridge/`), or Metal shaders (`src/kernels/`).

2. **Immutable per-block `QuantScheme` tag in `.amf` metadata
   ([ADR-0006](0006-amf-cold-experts-direct-io.md),
   [ADR-0014](0014-compile-time-specialization.md), `W14`).**
   - Every exported tensor and routed-expert blob in `.amf` records an
     explicit, immutable `QuantScheme` in its existing 128-byte `TensorEntry`
     (`dtype`, `quant_bits`, `group_size`).
   - `.amf` v0 binary compatibility is preserved unchanged: 128-byte `Header`,
     128-byte `TensorEntry`, `FORMAT_VERSION = 1`, 16 KiB blob alignment, 64 B
     `params_offset` alignment, and CRC-32 IEEE validation remain untouched.
   - Supported `QuantScheme` variants combine already-implemented types with
     additive `DType` enum entries (authoritative implementation status lives
     in [../status.md](../status.md)):

   | `QuantScheme` | Bit-width | Group size | Packed bytes / group-64 | Effective bits / weight | Status (see [../status.md](../status.md)) |
   | --- | --- | --- | --- | --- | --- |
   | `q2_affine` | 2-bit | 64 (compile-time) | 16 B + 4 B `half2` | 2.50 | Confirmed (`W14` / `G13`) |
   | `q3_affine` | 3-bit | 64 (compile-time) | 24 B + 4 B `half2` | 3.50 | Implemented (`W1` / `G1`) |
   | `q4_affine` | 4-bit | 64 (compile-time) | 32 B + 4 B `half2` | 4.50 | Implemented (`W1` / `G1`) |
   | `q6_affine` | 6-bit | 64 (compile-time) | 48 B + 4 B `half2` | 6.50 | Confirmed (`W14` / `G13`) |
   | `q8_affine` | 8-bit | 64 (compile-time) | 64 B + 4 B `half2` | 8.50 | Confirmed (`W14` / `G13`) |
   | `f16` | 16-bit | 0 (unquantized) | — | 16.00 | Implemented (`W1` / `G12`) |
   | `bf16` | 16-bit | 0 (unquantized) | — | 16.00 | Implemented (`W12` / `G12`) |

   - All affine schemes reuse the MLX-compatible LSB-first packed group layout
     followed at `params_offset` by interleaved `half2(scale, bias)` (`w = q *
     scale + bias`) so GPU kernels read blobs directly without repacking.

3. **Strict per-tensor / per-layer-projection / per-MoE-expert granularity —
   intra-tensor mixing is forbidden (`G13`).**
   - Precision assignment is locked strictly to whole tensors / layer
     projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, dense FFN
     projections, `embed_tokens`, `lm_head`, router `mlp.gate`, `shared_expert`)
     or whole routed MoE experts (`gate_proj`, `up_proj`, `down_proj`). Within
     any single matrix `[N, K]`, every row and group uses one uniform
     `QuantScheme`.
   - **Intra-tensor / sub-matrix mixed-bit packing is explicitly forbidden**:
     mixing different bit-widths across channels, rows, or quantization groups
     inside one weight matrix is disallowed so every kernel keeps 128-bit
     coalesced memory loads, constant `row_bytes` strides, and zero
     warp/simdgroup divergence.
   - For MoE layers whose experts carry heterogeneous `QuantScheme` tags (for
     example, hot or high-sensitivity experts at `q4_affine` / `bf16` and cold
     or low-sensitivity experts at `q3_affine` / `q2_affine`):
     - Each routed expert `(l, e)` keeps its fused `gate_proj` / `up_proj` pair
       (and by default its full `gate_proj` / `up_proj` / `down_proj` triple)
       on one uniform `QuantScheme`.
     - CSR active-expert deduplication (`G3`) buckets active experts by
       `QuantScheme` into preallocated per-scheme CSR slices and launches one
       compile-time-specialized Metal 4 pipeline per non-empty bucket.
     - Expert-byte-budget tree pruning
       ([ADR-0009](0009-expert-byte-budget-tree-pruning.md)) uses each expert's
       exact static byte footprint `b_{l,e}` read from `.amf` `TensorEntry`
       metadata at startup together with the calibrated effective bandwidth
       `β_s` of its scheme `s = scheme(l, e)` (plus the cold-expert NVMe
       penalty when non-resident), generalizing the expert-union cost sum to
       `Σ_l Σ_{e ∈ U_l(T)} b_{l,e} / β_{scheme(l,e)}` without changing the
       greedy submodular selection rule.

4. **Zero dynamic heuristics and zero extra branches in the Zig runtime hot
   path ([ADR-0011](0011-metal4-zero-allocation-hot-path.md),
   [ADR-0014](0014-compile-time-specialization.md), `G13`).**
   - At startup load time, the Zig loader validates every block's `QuantScheme`,
     computes wired-arena residency (`Budget.validate`) and static per-expert
     byte tables `b_{l,e}`, compiles specialized Metal 4 compute pipelines
     (`MTL4SpecializedFunctionDescriptor` + `MTLFunctionConstantValues`) only
     for the `QuantScheme` variants present in the loaded `.amf`, and binds
     each tensor / expert slot directly to its specialized pipeline handle.
   - On the hot path, the Zig scheduler dispatches strictly by the static block
     `QuantScheme` tag (and pre-built scheme buckets in MoE layers): zero
     runtime sensitivity scoring, zero inner-loop shader conditionals, and zero
     dynamic heap allocation.

5. **Strictly optional pipeline (`--quant-profile uniform | gsq-rco`) with
   mandatory uniform-precision baseline comparison.**
   - `aminfer-tools pack-amf` keeps `--quant-profile uniform` as the default.
     `--quant-profile gsq-rco` is an opt-in offline export mode and never
     replaces or deprecates standard uniform `3-bit`, `4-bit`, `FP16`, or
     `BF16` `.amf` builds.
   - Every GSQ+RCO `.amf` model build must be evaluated side-by-side against
     its uniform-precision baseline on the same Apple Silicon target, recording
     perplexity / output fidelity, wired DRAM footprint (`swapouts_delta = 0`),
     effective `bytes/token`, and end-to-end `tok/s` in
     [../benchmarks/results.md](../benchmarks/results.md).

## Consequences

- **No runtime bloat or invariant regression**: uniform `.amf` files remain
  bit-for-bit compatible, hot-path execution stays zero-allocation and
  branch-free inside Metal kernels, and Python remains confined to `tools/`.
- **Bounded startup and MoE dispatch overhead**: startup compiles one extra
  Metal 4 pipeline state per active `QuantScheme`, and a heterogeneous MoE layer
  issues up to `K_schemes` grouped dispatches (typically 2–3) instead of 1.
  Mandatory baseline benchmarking verifies that byte-traffic savings outweigh
  extra dispatch overhead on real checkpoints.
- **Offline export cost**: `--quant-profile gsq-rco` requires a calibration pass
  and offline reconstruction time in Python (`tools/`), unlike streaming
  `--quant-profile uniform` conversion; results depend on the calibration
  corpus and must be tracked reproducibly alongside each benchmark entry.
- **Direct fit with M4 NVMe cold experts and Tier 2/3 MoEs**: shrinking cold
  routed experts from 4-bit/3-bit down to 3-bit/2-bit reduces both wired DRAM
  pressure and NVMe direct-I/O bytes per cold fetch
  ([../architecture/experts-and-storage.md](../architecture/experts-and-storage.md#optional-gsq-rco-quantization),
  [../architecture/model-roadmap.md](../architecture/model-roadmap.md)).

## Evidence

GSQ + RCO is an accepted design not yet implemented or benchmarked; the
following verified M1–M2.5 foundations confirm the architecture requires no
breaking format or runtime redesign:

- `.amf` v0 `TensorEntry` (`src/format/amf.zig`) already carries per-tensor
  `dtype`, `quant_bits`, `group_size`, `blob_bytes`, `data_bytes`,
  `params_offset`, and `params_bytes` within a fixed 128-byte record.
- M2.5 ([ADR-0016](0016-native-fp16-bf16-and-multishard-import.md),
  [../benchmarks/results.md](../benchmarks/results.md)) already verified
  per-tensor mixed-dtype execution (`bf16`/`f16` norms and routers alongside
  `q3_affine`/`q4_affine` projections) on real multi-shard `Qwen3-14B-4bit`
  and `Qwen3-30B-A3B-3bit` checkpoints with `swapouts_delta = 0`.
- Compile-time specialization (`src/quant/spec.zig`,
  `ami_pipeline_create_specialized`,
  [ADR-0014](0014-compile-time-specialization.md)) and CSR active-expert
  deduplication (`src/gpu/moe.zig`, `G3`) already isolate bit-width logic into
  startup-compiled pipelines and decoupled expert gather schedules.

## Open items

- Calibration dataset selection, reproducibility metadata (written as an
  offline companion manifest outside the runtime hot path), and exact GSQ
  sensitivity / RCO solver formulation in `tools/` (`W14`, M4/M5).
- Whether the initial M4/M5 `G13` kernel set needs all three new affine widths
  (`q2_affine`, `q6_affine`, `q8_affine`) or whether `{q2_affine, q3_affine,
  q4_affine, f16, bf16}` already captures the quality/size Pareto frontier on
  `Qwen3-30B-A3B` and `Qwen3.8-Flash-Next` (decided by side-by-side baseline
  benchmarks in [../benchmarks/results.md](../benchmarks/results.md); see also
  [open-decisions-research.md](../architecture/open-decisions-research.md#remaining-m3m5-open-tuning-items-adr-0017-d8-adr-0005)).
- Measured decode (`M = 1`) and tree-verification (`M = 4..16`) dispatch
  overhead of 2-bucket vs 3-bucket `QuantScheme` CSR MoE execution and
  calibration of per-scheme effective bandwidth `β_s` for
  [ADR-0009](0009-expert-byte-budget-tree-pruning.md).
