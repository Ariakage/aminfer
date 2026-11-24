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

Adopt an **optional offline GSQ + RCO hybrid quantization pipeline** (capability
IDs `W14` and `G13`; authoritative implementation status lives in
[../status.md](../status.md)) governed by five binding invariants:

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
   | `q2_affine` | 2-bit | 64 (compile-time) | 16 B + 4 B `half2` (or inline `ZT0` LUT) | 2.50 | See [../status.md](../status.md) (`W14` / `G13`) |
   | `q3_affine` | 3-bit | 64 (compile-time) | 24 B + 4 B `half2` | 3.50 | Implemented (`W1` / `G1`) |
   | `q4_affine` | 4-bit | 64 (compile-time) | 32 B + 4 B `half2` (or inline `ZT0` LUT) | 4.50 | Implemented (`W1` / `G1`) |
   | `q6_affine` | 6-bit | 64 (compile-time) | 48 B + 4 B `half2` | 6.50 | Implemented in M8.1/M8.15 (`W21` / `G23`, `--fine-rungs`; see [../status.md](../status.md)) |
   | `q8_affine` | 8-bit | 64 (compile-time) | 64 B + 4 B `half2` | 8.50 | Implemented in M8.1/M8.15 (`W21` / `G23`, `--fine-rungs`; see [../status.md](../status.md)) |
   | `f16` | 16-bit | 0 (unquantized) | — | 16.00 | Implemented (`W1` / `G12`) |
   | `bf16` | 16-bit | 0 (unquantized) | — | 16.00 | Implemented (`W12` / `G12`) |

   - **Confirmed initial M5 scheme tier (`2026-11-21`):** M5 `G13` locks the
     active scheme set to **`{q2_affine, q3_affine, q4_affine, bf16, f16}`**,
     bounding MoE CSR dispatch to `<= 3` quantized buckets per layer; `q6_affine`
     and `q8_affine` stay deferred unless side-by-side benchmarks in
     [../benchmarks/results.md](../benchmarks/results.md) show an accuracy gap.
   - **100% binary-compatible `TensorEntry.reserved0` layout & codebook tags
     ([ADR-0018](0018-bare-metal-armv9-sme2-engine.md)):** the zero-initialized
     `reserved0: u16` field (byte offset 58 of the fixed 128-byte `TensorEntry`
     in `src/format/amf.zig`) encodes:
     - low byte (`reserved0 & 0xFF`): single-owner `TensorLayout` (`0 =
       row_major` for Metal GPU, `1 = sme2_panel64` for M4.5 CPU SME2);
     - high byte (`reserved0 >> 8`): `QuantParamKind` (`0 =
       mlx_affine_scale_bias` `half2(scale, bias)`, `1 = zt0_lut16_bf16`
       16-entry `ZT0` codebook for `luti4`, `2 = zt0_lut4_bf16` 4-entry `ZT0`
       codebook for `luti2`).
     Existing `.amf` v0 checkpoints already store `reserved0 == 0` (`row_major` +
     `mlx_affine_scale_bias`) and remain 100% valid without repacking.
   - All GPU `row_major` affine schemes reuse the MLX-compatible LSB-first
     packed group layout followed at `params_offset` by interleaved
     `half2(scale, bias)` (`w = q * scale + bias`) so GPU kernels read blobs
     directly without repacking.

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

### M8.15 GSQ-RCO v2 fidelity hardening (`W8`, `W14`, `W17`, `W24`, `C18`, `C20`)

Post-M8.1 root-cause diagnosis on `qwen3-30b-a3b-gsq-rco.amf` (`2.700b`, built in
M6 from already-quantized `mlx-community/Qwen3-30B-A3B-3bit`) vs the uniform
`3-bit` baseline (`qwen3-30b-a3b-3bit.amf`, which decodes conceptual reasoning
prompts accurately with zero repetition loops on the same engine) identified
three compounding failure modes:

1. **Low-bit `3-bit -> 2.70b` double-quantization with extreme `q2_affine`
   expert share**: transcoding a `3-bit` source checkpoint to `2.700b` forced
   `5,178 / 6,144` routed experts (`84.3%` overall, `52.3%`–`92.2%` of routed
   experts per layer — up to `118 / 128` at layer `29`; `15,534` expert
   projection tensors) down to `q2_affine`,
   incurring `~30%` per-matrix relative weight RMSE (`rel_rmse = 0.299–0.305`
   despite valid `cos = 0.952–0.954` bit packing), while unweighted OLS RCO
   weighted all input channels equally instead of weighting by per-channel
   activation second moments $E[x_c^2]$.
2. **Shallow embedding-only `expert-heat` calibration (`embed_router_sample`)**:
   without `--calibration-jsonl`, `expert-heat` multiplied `48` static token
   embeddings directly by router gates, ignoring pre-layer Transformer
   activation transformations when selecting the `15.7%` (`966`) experts kept at
   `q3_affine`.
3. **Unpenalized chat and batch-serving decoding (`C20`, `C18`)**:
   `ChatEngineRunner` (`aminfer chat`) and `BatchServeRunner`
   (`seen_frequencies = null`) passed raw `temperature` into full-vocabulary
   softmax sampling without `top_k` / `top_p` / `min_p` filtering or
   repetition/presence/frequency penalties, allowing open `<think>` chains or
   degraded 2-bit templates to lock into verbatim repetition loops.

To prevent these failure modes across all Dense and MoE checkpoints (including
upcoming [ADR-0025](0025-staged-frontier-deep-optimization-inkling2-mimo-glm5-ling3.md)
frontier tier-down profiles), Milestone **M8.15** hardens the offline
quantization toolchain, router-trace pipeline, and runtime samplers with four
binding rules:

1. **Activation-weighted OLS RCO (`W14`)**: when `--imatrix` is supplied,
   `parse_gguf_imatrix` provides both tensor-level sensitivity multipliers and
   per-column channel weight vectors $w_c = \max(E[x_c^2], 10^{-12})$ (across
   both 2D projections and 3D stacked MoE expert tensors via shared `[cols]`
   broadcast or per-expert `[num_experts * cols]` channel slicing + per-expert
   `.counts`) to `quantize_rows_affine` and `quantize_group_rco`, minimizing
   $\sum_{i \in g} w_i (\hat{x}_i - x_i)^2$ (for group weights $x_i$,
   reconstructed $\hat{x}_i = b + q_i s$, and channel importance $w_i$) over
   each group-32 / group-64 / group-128 slice (with automatic fallback to
   min-max under the same weighted objective if alternating refinement does not
   improve error).
2. **Hard backbone role floor, `--max-q2-share`, architecture-aware auto
   `target_bits`, CLI low-bit double-quantization guard, and `--fine-rungs`
   (`W14`, `W17`)**:
   - `enforce_backbone_floor = True` forbids `DType.Q2_AFFINE` candidates on
     all `18` `_BACKBONE_PROTECTED_ROLES` (`_PROTECTED_HIGH_PRECISION_ROLES`:
     `router`, `controller`, `sconv`, `ihc`, `mtp_dflare`, `vision`,
     `embedding`, `lm_head`, `shared_expert`, `attn_qk`, `attn_v`, `attn_o`,
     `attn_vo`, `norm`, plus GGUF/hybrid `embed_head`, `attn_qkv_o`,
     `ssm_dt_a`, `ssm_proj`), and `--max-q2-share` caps
     `q2_numel / quantizable_numel` inside `solve_gsq_allocation` (override via
     `--allow-unsafe-q2`);
   - Omitting `--target-bits` resolves automatically to `4.15` for Dense models
     (`num_experts <= 1`, default `max_q2_share = 0.0` so `0` tensors drop to
     `q2_affine`) and `3.50` for MoE models (`num_experts > 1`, CLI default
     `max_q2_share = 0.55`, including `qwen4_exp` and HF 3D stacked
     `mlp.experts.*` checkpoints);
   - CLI `pack-amf` and `gsq-convert` reject `q2/q3/q4_affine -> q2_affine`
     low-bit double-quantization by default unless `--allow-retranscode-lowbit`
     is explicitly passed;
   - `--fine-rungs` enables intermediate M8.1 schemes (`(Q2/Q3/Q4_AFFINE, g32)`
     and `(Q5/Q6/Q8_AFFINE, g64)`) on float sources while capping `moe_expert`
     candidate schemes to `<= 5` per layer (`moe.NUM_SCHEME_BUCKETS = 5`
     defined in `src/gpu/moe.zig` and enforced in `src/model/qwen3.zig` and
     `src/model/graph.zig`, with `lut16_sme2` gated strictly to group-64
     `Q2/Q3/Q4_AFFINE` and per-layer uniform MoE `TensorLayout` keeping all
     experts in a layer `ROW_MAJOR` whenever any expert selects a non-SME2
     scheme).
3. **Zero-copy single-frame GPU router top-k trace export and shallow-heat
   warning (`W8`)**: `aminfer decode --dump-router-trace <path.jsonl>` carves a
   `num_layers * 64 B` buffer in `scratch_arena` and redirects each layer's
   `scratch_topk_experts_off` inside single-frame decode (`runStep`,
   `runStepWithLora`, `runStepHeterogeneous`, plus `@memcpy` in `runStepCold`)
   to emit real multi-layer JSONL router traces (`calibration_mode =
   "router_trace"`), while `aminfer-tools expert-heat` emits a stderr warning
   whenever `calibration_mode != "router_trace"`.
4. **Runtime sampling/penalty parity and `W24` loop quality gate (`C18`, `C20`,
   `W24`)**: `aminfer chat` (`C20`) wires `--top-k` (`40`), `--top-p` (`0.95`),
   `--min-p` (`0.05`), `--repetition-penalty` (`1.05` when `T > 0`; `1.0` when
   `T = 0` for bit-exact greedy parity), `--presence-penalty` (`0.0`), and
   `--frequency-penalty` (`0.0`) into `ChatEngineRunner` via per-turn
   `DefaultPenaltyTracker`; `BatchServeRunner` (`C18`) binds `pending_jobs`
   metadata by `request_id` and `slotPromptBuffer` pointer (isolating concurrent
   out-of-order prefill slots and orphaned tokenizations) and maintains per-slot
   `slot_penalties: []DefaultPenaltyTracker` across prompt seeding, GPU
   `sampleSync` (`seen_frequencies`), CPU fallback `sampleSlotConstrained`, and
   forced thinking-close tokens; and `aminfer quality-gate` (`W24`) evaluates a
   3rd conceptual reasoning prompt (`Riemann zeta`) and flags 4-gram or
   consecutive-token repetition loops (`fail_repetition_loop`).

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

Authoritative status lives in [../status.md](../status.md), and side-by-side M5,
M6, and M8.15 measurements (`aminfer bench-m5`, real `Qwen3-0.6B` GSQ+RCO vs
MinMax vs BF16, `Qwen3-30B-A3B` root-cause & `GSQ-RCO v2` hardening, and
3-prompt `aminfer quality-gate` with `fail_repetition_loop`) live in
[../benchmarks/results.md](../benchmarks/results.md#m5-serving-pipelining-c6-nvme-n-gram-and-adr-0017-gsqrco-hybrid-quantization),
[../benchmarks/results.md](../benchmarks/results.md#m6-frontier-hybrid-sparse-models-recurrent-depth-large-moe-gsqrco-and-off-gpu-sme2-draft-heads),
and
[../benchmarks/results.md](../benchmarks/results.md#m815-sampling-penalty-parity-gsq-rco-v2-fidelity-hardening-and-router-trace-loop-quality-gate).
The following verified foundations confirm the architecture requires no breaking
format or runtime redesign:

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

- Calibration dataset selection, reproducibility metadata (`--manifest`), and
  GSQ sensitivity / RCO solver formulation in `tools/` (`W14`, implemented in M5
  and hardened with `--imatrix` activation-weighted OLS RCO + `router_trace`
  calibration in M8.15).
- Initial M5 `G13` kernel set **locked on 2026-11-21** to
  `{q2_affine, q3_affine, q4_affine, bf16, f16}` (extended in M8.1 `W21`/`G23`
  and M8.15 `--fine-rungs` with `g32` and `q5_affine` / `q6_affine` /
  `q8_affine`; see [../status.md](../status.md) and
  [open-decisions-research.md](../architecture/open-decisions-research.md#remaining-m3m5-open-tuning-items-adr-0017-d8-adr-0005)).
- Measured decode (`M = 1`) and tree-verification (`M = 4..16`) dispatch
  overhead of 2-bucket vs 3-bucket `QuantScheme` CSR MoE execution and
  calibration of per-scheme effective bandwidth `β_s` for
  [ADR-0009](0009-expert-byte-budget-tree-pruning.md).
- Per-layer normalization of `_imatrix_tensor_multipliers` in
  `tools/src/aminfer_tools/gsq_rco.py` so Pre-LN activation variance growth
  across depth (`L1` peak `204.56` vs `L0` peak `0.269` in `imatrix_unsloth.dat`)
  does not clip shallow-layer `L0`–`L3` multipliers to `0.5` and over-allocate
  `q2_affine` to `L0` (`60.2%` vs `32.9%` model-wide mean).
- Coding-priority calibration and `--max-q2-share 0.0` export profile: expand
  `--dump-router-trace` calibration to a 50–100+ prompt multi-language coding
  suite and evaluate zero-`q2` MoE exports (pairing `>= q3_affine` experts with
  lossless NVMe cold-expert offload rather than `q2_affine`) to avoid long-tail
  specialist-expert degradation on niche programming languages.
