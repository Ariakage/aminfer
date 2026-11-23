# ADR-0023: Native Metal 4 sublayer fusion, MMA FlashAttention-2, GPU indirect MoE dispatch, and NEON/SME2 accelerators (M7.8)

- Status: Accepted
- Date: 2026-11-23
- Source: user roadmap confirmation (`M7.8`) and non-regression engineering review following `ADR-0022` (`M7.5`)

## Context

Inspection of the steady-state decode and prefill hot paths (`src/model/qwen3.zig`, `src/model/graph.zig`, `src/kernels/*.metal`, `src/server/http_sse.zig`, `src/model/tokenizer.zig`, and `src/cpu/sme2_kernels.s`) alongside M4 roofline measurements ([../benchmarks/results.md](../benchmarks/results.md)) separates operations that are already memory-bus-saturated from genuine remaining native execution bottlenecks:

1. **Where M4 is already near roofline:** Large single-token (`M = 1`) quantized GEMV already sustains up to `98.8 GB/s` (`82.3%` of M4's `120.0 GB/s` unified memory bus). Rewriting large `M = 1` weight reads in lower-level assembly cannot exceed physical DRAM bandwidth.
2. **Where native Metal 4, ARM64 NEON, and ARMv9.2 SME2 code still yields measurable gains:**
   - **Per-layer `MTLStageDispatch` barrier and scratch round-trip overhead (`G18`).** Each Transformer block currently encodes separate `o_proj -> barrier -> ami_residual_add_f16 -> barrier -> ami_rmsnorm` dispatches, three separate `q_proj` / `k_proj` / `v_proj` dispatches reading `x_norm` three times, and (on dense layers) separate `gate_proj` + `up_proj -> barrier -> ami_silu_mul_f16 -> barrier -> down_proj -> barrier -> ami_residual_add_f16` dispatches (`4–5` extra GPU dispatch barriers per layer, or `192–240` barriers per token across 48 layers).
   - **Scalar/vector inner loops in long-context prefill attention (`G19`).** `G15` wide-chunk prefill (`ADR-0020`) uses `simdgroup_matrix` (`Apple9`) / `NAX` (`Apple10+`, `ADR-0021`) for linear projections, while `ami_gqa_causal_prefill_flash_attn` (and QSA/MLA prefill inner loops) still compute `S = Q · K^T` and `O += exp(S) · V` via per-thread vector dot products rather than hardware `8×8` matrix multiply-accumulate tiles.
   - **Over-dispatched MoE buckets and host bucket staging (`G20`).** `G3` / `G7` / `G13` MoE gather paths either pad threadgroup grids up to conservative worst-case row bounds (`M * top_k` per quantization bucket, launching empty threadgroups that immediately exit) or synchronize bucket counts on the host.
   - **Large-vocabulary (`151K–248K`) constrained sampling and JSON/BPE text scanning (`G21`, `C23`).** Live frontier checkpoints reach `vocab_size = 151,936` (`Qwen3`), `129,280` (`DeepSeek-V4.1-Flash-1017`), and `248,077` (`Qwen3.8-Flash-Next`). Scanning `248K` float logits (`~1 MiB`) on CPU per step for `M7.5` penalties, `logit_bias`, `top_k`/`top_p`/`min_p`, and `C22` grammar bitmasks would add `0.4–1.5 ms/tok`, while byte-at-a-time JSON string unescaping and BPE pre-tokenization leave ARM64 128-bit SIMD lanes idle on large tool-call payloads and long prompts.
   - **`2-bit` (`q2_affine`) off-GPU drafting and multi-LoRA adapters on P-cores (`C24`).** `ADR-0018` (`src/cpu/sme2_kernels.s`) implemented `4-bit` `luti4` `ZT0` expansion, whereas `2-bit` `q2_affine` cold/draft projections (`D8`, `W17`) and `M7.5` L2-resident multi-LoRA adapters (`W19`) can pack twice as many indices per byte via ARMv9.2 SME2 `luti2` (`2-bit` table lookup) with tuned multi-row `ZA` software prefetching (`prfm`).
3. **Why naive low-level optimization can regress latency if unguarded:** Over-aggressive Metal kernel fusion can inflate per-thread register usage and halve SIMD-group occupancy; `8×8` MMA FlashAttention adds `threadgroup_barrier` and `7/8` tile-padding waste at `M = 1`; GPU indirect dispatch adds an indirect command-buffer fetch stall when `M = 1` row counts are already static; and unguarded P-core SME2 `smstart`/`smstop` transitions or DRAM fetches can contend with GPU memory bandwidth.

The user confirmed codifying all five native optimization tracks (`G18`–`G21`, `C23`–`C24`) into **Milestone M7.8** ([../status.md](../status.md), [../process.md](../process.md)), paired with explicit microarchitectural anti-regression gates.

## Decision

### 1. Metal 4 fine-grained sublayer fusion and barrier elimination (`G18`, M7.8)

- **Fused residual epilogues:** Add Metal function-constant-specialized (`has_residual_epilogue`) variants of `o_proj` and `down_proj` across `ami_dequant_gemv`, `ami_dense_gemv`, `ami_dequant_gemm_wide`, `ami_dense_gemm_wide`, and `ami_dense_gemm_nax` so projections compute `out[i] = dot(W_i, x) + residual[i]` in a single write pass without a standalone `ami_residual_add_f16` dispatch or barrier.
- **Fused `ResidualAdd + RMSNorm`:** Add `ami_fused_residual_rmsnorm` reading `x` and `delta`, writing the updated `x_residual` in place and emitting normalized `x_norm` directly from threadgroup registers (`1` dispatch and `0` intermediate barrier between sublayers).
- **Fused homogeneous `QKV` and dense `Gate + Up + SwiGLU`:** Add `ami_fused_qkv_gemv` / `ami_fused_qkv_gemm` (reading `x_norm` once across `q_proj`, `k_proj`, and `v_proj`) and `ami_dense_gate_up_swiglu` (computing `SiLU(x · W_gate) * (x · W_up)` in registers without writing `2 × intermediate_size` scratch activations to unified DRAM, matching MoE `G3` SwiGLU fusion).
- **Occupancy and heterogeneous-GSQ guard:** Restrict `ami_fused_qkv` and `ami_dense_gate_up_swiglu` strictly to layers where participating tensors share a homogeneous `DType` (`q.dtype == k.dtype == v.dtype`, `gate.dtype == up.dtype`). Layers quantized with mixed per-tensor GSQ schemes (`ADR-0017`, `G13`) keep the non-divergent unfused dispatch path. Every fused kernel must pass a build-time `xcrun metal` zero-stack-spill check and an `M4` `bench-m78` occupancy/latency gate before replacing the unfused path for a given `(M, N, K)` shape.

### 2. Hardware `simdgroup_matrix` (`Apple9`) and `NAX` (`Apple10+`) FlashAttention-2 prefill (`G19`, M7.8)

- Upgrade `ami_gqa_causal_prefill_flash_attn` (and the dense/sparse attention tiles inside `G9` QSA and `G11` MLA prefill) to stage `BQ = 32..64` query rows and `BK = 32..64` KV rows in 16 KiB-bounded `threadgroup` SRAM (padded to eliminate shared-memory bank conflicts) and execute both `S = Q · K^T` and `O += exp(S - m) · V` via `simdgroup_half8x8` (`simdgroup_multiply_accumulate` on `Apple9` M4) and `mpp::tensor_ops::matmul2d` hooks on `Apple10+` (`ADR-0021`).
- **Short-sequence crossover guard:** Never route `M = 1` decode (`G4`) or narrow draft trees (`G5`, `Q_len < 32`) through `G19`, avoiding `8×8` tile padding waste (`87.5%` idle MACs at `M = 1`) and `threadgroup_barrier` overhead on latency-bound decode steps. Gate `G19` activation strictly on `Q_len >= 32` and `KV_len >= 64` (calibrated per `ChipProfile` in `C16`).

### 3. GPU-driven indirect MoE threadgroup dispatch (`G20`, M7.8)

- Extend `apple_bridge.h` / `apple_bridge.m` (`ami_mtl4_compute_dispatch_indirect`), `src/bridge/metal.zig`, and `src/gpu/moe.zig` to expose Metal 4 indirect threadgroup dispatch (`MTLDispatchThreadgroupsIndirectArguments`: `[threadgroupsPerGridX, Y, Z]`).
- Have `ami_moe_route_topk` and `ami_moe_route_topk_bucketed` (`G13`) write exact per-scheme `[ceil(N / BN), active_bucket_pairs, 1]` indirect dispatch structs directly in device-visible unified memory, eliminating empty over-dispatched threadgroups and host readback on mixed-precision GSQ MoEs (`G7`, `G13`).
- **`M = 1` static-grid guard:** When `M == 1` and all routed experts in the layer share a single homogeneous `DType` (`num_scheme_buckets == 1`), `active_bucket_pairs` is statically known (`top_k`) with zero empty threadgroups; keep direct CPU-encoded `dispatchThreadgroups` on that fast path so `M = 1` decode pays zero indirect command-processor fetch latency. Activate `G20` indirect dispatch only when `M > 1` (`verifyTree`, `C14` continuous batching, `G15` wide prefill) or `num_scheme_buckets > 1`.

### 4. Fused GPU `248K`-vocab radix constrained sampler (`G21`) and ARM64 NEON text/JSON scanners (`C23`, M7.5–M7.8)

- **Fused GPU radix/bucket constrained sampler (`G21`):** Implement `src/kernels/sampling.metal` executing a two-pass GPU reduction/radix-bucket sampler directly over device `logits` (`vocab_size` up to `262,144`, covering `Qwen3` `151,936`, `DeepSeek-V4.1` `129,280`, and `Qwen3.8` `248,077`). A single kernel fuses:
  1. `C22` allowed-token bitmask (`uint32_t[vocab_words]`, `~31 KiB` for `248K` vocab),
  2. sparse `logit_bias` + per-slot `presence_penalty`, `frequency_penalty`, and `repetition_penalty`,
  3. temperature scaling + `min_p` / `top_k` / `top_p` radix-bucket threshold selection + slot-local `D11` Gumbel-max / multinomial sampling, writing only the sampled `u32` token id (and optional requested `top_logprobs` pairs) to shared host memory (`< 0.05 ms`, `0` full-vocab CPU scan).
- **Fast-path bypass:** Unconstrained greedy (`temperature == 0`, no penalties, no `logit_bias`, no `C22` grammar mask) retains the minimal single-reduction GPU `argmax` kernel with zero penalty/bitmask buffer reads. When `C22` grammar masking is active alongside speculative decoding (`D2`–`D4`, `D10`, `D12`), propagate the state-advanced bitmask across draft tree proposals as well as `G5` / `D9` verification so speculative acceptance length does not collapse on strict JSON schemas.
- **ARM64 NEON structural JSON and BPE pre-tokenizer scanning (`C23`):** Vectorize HTTP/JSON string boundary, quote/backslash escape, and whitespace scanning in `src/server/http_sse.zig` and ASCII / UTF-8 class-boundary scanning in `src/model/tokenizer.zig` using explicit 16-byte / 64-byte unrolled `@Vector(16, u8)` ARM64 NEON intrinsics (`cmeq`, bitmask reduction via `shrn`), falling back cleanly to scalar tail loops for inputs `< 16 B` (`0` heap allocation, safe under `-Dcpu=apple_m1`).

### 5. ARMv9.2 SME2 `luti2` (`2-bit`) assembly microkernels and `ZA` prefetch tuning (`C24`, M7.8)

- Extend `src/cpu/sme2_kernels.s` and `src/cpu/sme.zig` (`ADR-0018`) with native ARMv9.2 SME2 `luti2` (`2-bit` 4-entry `ZT0` table expansion across 4× packed weight streams per byte) for `q2_affine` projections (`W14`, `W17`) and tuned multi-row `ZA` software prefetch hints (`prfm pldl1keep` for L2-hot `Step-LoRA` / `W19` multi-LoRA adapters; `prfm pldl1strm` / `ldnt1b` for cold streaming slices).
- **Streaming-SVE transition, L2 residency, and bus-contention guards:**
  - Preserve all `ADR-0018` invariants: `-Dcpu=apple_m1` baseline isolation (compiled via dedicated `.s` object + runtime `FEAT_SME2` gate), `d8–d15` preservation before `smstart`, and `0` `[sp` stack spills verified by disassembly test.
  - Hoist `smstart` / `smstop` outside layer/adapter loops (never toggling Streaming SVE per row).
  - Cap L2-resident SME2 `Step-LoRA` / `W19` multi-LoRA working sets to `<= 4 MiB` (`25%` of M4's `16 MiB` P-core L2) and honor the `D5` / `C13` / `C16` unified-memory bandwidth gate so P-core SME2 execution never steals DRAM bandwidth or thermal headroom from a bus-saturated GPU pass.

## Consequences

- **Positive:** Eliminates `192–240` GPU `MTLStageDispatch` barriers and intermediate DRAM round-trips per 48-layer forward pass (`G18`); brings hardware matrix-multiply throughput to `2K–32K` FlashAttention prefill (`G19`); removes empty MoE bucket threadgroups on `M > 1` and heterogeneous GSQ checkpoints (`G20`); keeps `248K`-vocab structured-output and penalty sampling under `0.05 ms` without CPU logit scans while accelerating large JSON/prompt ingestion (`G21`, `C23`); and doubles packed index density on off-GPU `q2_affine` SME2 draft/adapter paths (`C24`).
- **Negative / trade-offs:** Increases `aminfer.metallib` binary size by `~40–80 KiB` and adds shape/scheme dispatch rules in `src/gpu/`, `src/model/`, and `src/cpu/`. Extending the startup `G16` numerical canary (`< 0.35 ms` warm) and shape-gated A/B benchmarks in `aminfer bench-m78` bounds complexity and prevents occupancy or crossover regressions on `Apple9` and `Apple10+`.

## Evidence

- Per-layer dispatch structure audited in `src/model/qwen3.zig` (`L1869–1965`) and `src/model/graph.zig`.
- Baseline M4 roofline (`98.8 GB/s` peak GEMV on `120.0 GB/s` bus), `G15` wide prefill, `G16` `.metallib` canary, `G17` `NAX` dual-backend `.metallib`, and `C11`–`C13` SME2 microkernels documented in [../benchmarks/results.md](../benchmarks/results.md), [ADR-0018](0018-bare-metal-armv9-sme2-engine.md), [ADR-0020](0020-wide-gpu-prefill-tail-pruning-qos-and-live-ram-guards.md), and [ADR-0021](0021-multi-generation-apple-silicon-and-nax-tensorops.md).
- Real frontier vocabulary sizes verified against Hugging Face `tokenizer.json` / `config.json` in commit `cd72273` (`Qwen/Qwen3.8-Flash-Next`: `248,077`; `deepseek-ai/DeepSeek-V4.1-Flash-1017`: `129,280`).

## Open items

- Calibrate exact `G18` fused `QKV` / dense `SwiGLU` occupancy thresholds (`xcrun metal` register count audit) and `G19` `BQ × BK` tile sizes (`32×32` vs `64×32`) on `M4` (`Apple9`) during M7.8 implementation (`aminfer bench-m78`), keeping `Apple10+` `NAX` FlashAttention hardware numbers gated on `Spike S8`.
