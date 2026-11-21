# ADR-0019: Composable recurrent depth (`LoopSchedule`), `SelfSpecRecurrent`, and post-M5 heterogeneous roadmap

- Status: Accepted
- Date: 2026-11-22
- Source: user roadmap confirmation following post-M5 `/btw` technical review

## Context

With Milestones M0–M5 (including M2.5 and M4.5) implemented on the 24 GiB Apple
M4 baseline, measured M4.5 and M5 data ([../benchmarks/results.md](../benchmarks/results.md))
establish three constraints for M6 (`Qwen3.8-Flash-Next`), M6.5, and M7
(`DeepSeek-V4.1-Flash-1017`):

1. **Off-GPU ARMv9 SME2 avoids GPU shader contention when the workload fits L2.**
   In M4.5 (`C13`), running an L2-resident `4 MiB` SME2 router alongside GPU
   `gemv_q3` reached `155.57 GB/s` aggregate throughput (`1.80×`) and cut
   `Qwen3-30B-A3B + EAGLE-3` tree router latency from `975.5 us` to `78.0 us`.
   In M5 (`D8`), moving `0.6B-GSQ-RCO` drafting off the GPU onto CPU SME2
   (`ldnt1b`, `0` draft Metal command buffers) avoided the `+6.0..+44.8 ms`
   `verify_ms` slowdown caused by sharing M4's 10-core GPU (`10.67 → 11.81–12.10
   tok/s`), although sequential same-GPU `0.6B-GSQ-RCO` (`--pearl off`,
   `13.55 tok/s`) remained faster end-to-end because GPU `q4`/`q3` drafting
   (`15.0 ms`) outran full-model CPU SME2 drafting (`29.4 ms` + `19.4 ms` sync).
   Off-GPU SME2 co-execution therefore pays off specifically on compact,
   L2-resident subgraphs (`2–16 MiB` routers, draft projections, and loop
   adapters) rather than full multi-hundred-megabyte models.
2. **Offline GSQ + RCO (`W14`/`G13`) and external `.ang` tables (`C6`) have
   clear operating regimes.** On `Qwen3-0.6B`, `4.35`-bit GSQ + RCO reduced
   wired RAM by `66.2%–69.3%`, lowered quantization MSE by `16.82%` versus
   min-max, and ran at `171.05–205.52 tok/s` (`2.80–3.17×` over BF16); 3-bucket
   `{q2, q3, q4}` MoE reduced weight bytes by `27.7%–27.8%` at `1.00–1.11×` the
   speed of 1-bucket uniform `q4`. On `C6`, wired LRU hits took `0.04 us` (and
   freshly written `4 MiB` synthetic files hit drive DRAM cache at
   `1.13–1.54 us`), whereas uncached `F_NOCACHE` lookups against the `128.02 MiB`
   `aminfer-repo.ang` file on the external `/Volumes/Akari` SSD averaged
   `396–1,208 us`, accelerating decode when multi-token seeds hit and adding
   overhead when misses dominate.
3. **Recurrent depth (Looped Transformers) requires tree batching and state
   isolation.**
   - **Native looped models** (such as *Huginn-3.5B* and *Ouro / LoopLM* with
     `Prelude → [Core Block] × R → Coda`) increase effective depth at test time
     without adding static wired weights.
   - **Frozen non-looped checkpoints** (`Qwen3`, `Qwen3.8-Flash-Next`,
     `DeepSeek-V4.1-Flash-1017`) cannot feed late hidden states back into
     earlier layers directly without corrupting RMSNorm scales, RoPE KV
     statistics, and MoE router distributions; supporting recurrent passes on
     them requires an offline **Gated Loop Adapter** (`Step-LoRA` +
     zero-initialized scalar gate `α_r`, `< 0.5%` parameters, with `α_0 = 0` at
     pass `r = 0` for bit-exact identity).
   - Under `M = 1` serial decode on a `120 GB/s` bus, executing a Core block
     larger than the `16 MiB` P-core L2 `R` times re-reads Core weights `R`
     times and reduces single-token throughput by roughly `1/R` unless paired
     with `M = 8..16` candidate-tree verification and intermediate `lm_head`
     elision.

The user also confirmed tracking the remaining post-M5 optimization items across
M6 and M6.5: large-MoE GSQ + RCO with `ZT0` non-uniform LUTs, off-GPU L2 SME2
draft-head placement, server continuous batching with shared prefix caches,
32K+ GPU `q8` retired KV compression, and zero-copy multimodal ingress.

## Decision

### 1. Composable `LoopSchedule` (`W9`) and `Gated Loop Adapter` (`W16`, M6)

- Add a **`LoopSchedule`** descriptor (`start_layer`, `end_layer`, `max_loops`,
  `inject_x_init`, `adapter_kind: none | sme2_step_lora`,
  `early_exit_gate: none | sigmoid_linear`) to the `W9` composable model graph
  ([ADR-0015](0015-tiered-frontier-model-roadmap.md)). Native `Prelude → Core ×
  R → Coda` checkpoints and adapter-extended `Qwen3` / `Qwen3.8` / `deepseek41`
  checkpoints both lower to `LoopSchedule`; `max_loops = 1` executes as a
  single pass with no extra branches on the hot path.
- Support optional `.amf` **Gated Loop Adapters (`W16`)**: step-conditioned
  rank-16/32 `Step-LoRA` weights plus scalar gate `α_r` packed in `sme2_panel64`
  (`2–12 MiB` total), kept in the `16 MiB` P-cluster L2 (`ld1b`) and evaluated
  on ARMv9 SME2 alongside GPU base projections. Pass `r = 0` enforces
  `α_0 = 0`, preserving exact output parity when `max_loops = 1`.
- Enforce **four hybrid-sparse isolations** on intermediate passes `r < R - 1`
  at the same token position(s):
  1. **Gated DeltaNet state isolation (`R6`, `G8`)**: leave committed `GDN`
     state `S_t` untouched during `r < R - 1`; update preallocated wired scratch
     `S_tmp` from `S_{t-1}` and commit `S_t` once on final pass `R - 1`.
  2. **Single-lookup external memory (`C10` `51B PLE` / `196B Engram`)**: issue
     `F_NOCACHE` NVMe row reads once at `r = 0` into a wired slot and reuse
     those rows across `r >= 1` (`0` extra SSD reads).
  3. **Sparse attention index reuse (`G9` `QSA` / `G11` `CSA2 Reuse`)**: compute
     micro-block top-k indices once at `r = 0` and reuse tile selections across
     `r >= 1` while KV length is unchanged.
  4. **MoE CSR deduplication and pinned loop routing (`G3`/`G7`, `C5`)**: batch
     `M = 8..16` candidate nodes across loops for CSR deduplication and restrict
     `r >= 1` routing to wired or already pinned `ExpertSlotPool` slots so loop
     steps do not block on synchronous cold-expert NVMe reads.

### 2. `SelfSpecRecurrent` batch verification (`D12`, M6)

- Add `DraftSource.self_recurrent` under `SpecEngine`:
  - Native looped models draft an `M = 8..16` tree with a `r = 1` shallow pass
    (`0` extra draft-model RAM); non-looped or adapter-extended models draft
    with `r = 0` or existing `EAGLE-3` / `0.6B` / `4B MTP` / `DSpark` sources.
  - Full `Core × R` verification runs once across the `M = 8..16` tree using GPU
    GEMM / SME2 `1 × 4 ZA` kernels, `G5` ancestor masks, and lossless `D9`
    acceptance, amortizing Core weight reads across `M` candidates.
- Passes `r < R - 1` skip full-vocabulary `lm_head` projection and sampling; an
  opt-in token-adaptive gate (`g = sigmoid(w · h)`) allows early exit at `r = 0`
  when approximate decoding is explicitly enabled.

### 3. M6 heterogeneous scaling and verification strategy (`W17`, `D13`)

- **Large-MoE GSQ + RCO and non-uniform `ZT0` LUTs (`W17`, M6)**:
  - Apply `tools/` offline GSQ + RCO (`W14`) combined with `W8` expert-heat
    profiles to `Qwen3-30B-A3B` and `Qwen3.8-Flash-Next` (`512` routed experts +
    `shared_expert`), keeping attention, routers, `shared_expert`, and hot
    experts at `q4_affine`/`bf16` while quantizing cold routed experts to
    `q3_affine`/`q2_affine`.
  - Implement `.amf` `QuantParamKind = 1` non-uniform 16-centroid `ZT0` LUT
    packing ([ADR-0017](0017-optional-gsq-rco-hybrid-quantization.md),
    [ADR-0018](0018-bare-metal-armv9-sme2-engine.md)) for ARMv9 SME2 `luti4` /
    `luti2` draft, router, and loop-adapter weights.
- **Off-GPU L2-resident SME2 draft heads (`D13`, M6)**:
  - Add `Placement.cpu_sme2` under `D5`
    ([ADR-0008](0008-benchmark-gated-draft-placement.md)) for compact L2-resident
    `EAGLE-3` and `Qwen3.8` `4B MTP` projection subgraphs, gated by measured
    round latency against `.gpu`.
- **Synthetic-first M6 validation with drop-in checkpoint import**:
  - Implement deterministic synthetic and scale-calibrated `qwen4_exp` +
    `LoopSchedule` generators in `tools/` to test all M6 kernels, alignment
    checks, and 24 GiB M4 wired budgets (`swapouts_delta == 0`) upfront, while
    keeping `pack-amf` compatible with Hugging Face / MLX safetensors when real
    checkpoints are provided.

### 4. M6.5 serving, long-context memory and multimodal milestone (`C14`, `R8`, `R5`, `G14`)

Schedule **Milestone M6.5** between M6 and M7 for four systems extensions:

1. **Iteration-level continuous batching (`C14`)**: batch `M > 1` active HTTP/SSE
   slots per GPU GEMM / CSR-MoE and SME2 `ZA` decode step in `aminfer serve`,
   and support switching domain `.ang` NVMe N-gram packs (`C6`).
2. **Shared prefix `KvTile` and `GDN` state cache (`R8`)**: share read-only
   prefix `KvTile`s across slots with copy-on-write tail tiles and
   snapshot-cloned `GDN` recurrent states.
3. **GPU-side group-64 `q8` retired KV tier (`R5`)**: keep active IOSurface
   `KvTile`s in `RGBA16Float` / `OneComponent16Half`
   ([ADR-0004](0004-kv-cache-iosurface-texture-sharing.md)) and compress retired
   32K+ KV chunks into GPU `q8` `MTLBuffer` blocks when context growth
   approaches the `18,186 MiB` wired limit on 24 GiB M4.
4. **Zero-copy multimodal vision/video ingress (`G14`)**: decode image/video
   frames via `VideoToolbox` into `16 KiB`-aligned `CVPixelBuffer` / `IOSurface`
   planes bound zero-copy as `MTLTexture` and CoreML `MLMultiArray` inputs into
   interleaved MRoPE (`[11, 11, 10]`).

## Consequences

- `W9` executes standard decoder-only models, hybrid linear/sparse MoEs
  (`Qwen3.8`), native looped models, adapter-extended looped models (`W16`), and
  M7 `causal_enc_dec` (`deepseek41`) under one static schedule without runtime
  Python or hot-path heap allocation.
- Capability rows `R8`, `W16`, `W17`, `G14`, `D12`, `D13`, and `C14` (plus
  updated `R5`, `R6`, `W9`, `G8`, `G9`, `C10`) are tracked in
  [../status.md](../status.md); Milestone **M6** and **M6.5** scope lives in
  [../process.md](../process.md).

## Evidence

- Post-M5 measurements in [../benchmarks/results.md](../benchmarks/results.md):
  `C13` `155.57 GB/s` L2-router + GPU co-execution (`1.80×`), `D8` off-GPU SME2
  vs same-GPU PEARL contention breakdown (`12.10 tok/s` SME2 PEARL vs `13.55
  tok/s` sequential GPU GSQ-RCO), `W14`/`G13` `4.35b` `0.6B` (`171.05–205.52
  tok/s`, `-16.82%` MSE) and 3-bucket MoE (`-27.8%` bytes), and `C6` `.ang`
  (`0.04 us` wired LRU vs `396–1,208 us` uncached external SSD).

## Open items

- Measure `SelfSpecRecurrent` (`D12`) break-even tree batch width `M` against
  loop count `R = 2..4` on M4 (`120 GB/s`) in M6 `bench-m6`.
- Benchmark `Placement.cpu_sme2` against `.gpu` (`D13`) on `EAGLE-3` and `4B MTP`
  projection subgraphs under `ADR-0008` median latency gating.
