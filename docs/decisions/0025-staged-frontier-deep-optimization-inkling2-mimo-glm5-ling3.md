# ADR-0025: Deep architecture-specific optimizations for `Inkling 2` (`Mini` & `Small`) and staged `M9.1`–`M9.3` frontier roadmap (`MiMo-V2.6`, `GLM-5`, `Ling-3.1`)

- Status: Accepted
- Date: 2026-11-24
- Source: user roadmap confirmation (`2026-11-24`) following live Hugging Face `config.json` and checkpoint byte-budget audits

## Context

Live Hugging Face `config.json`, safetensors index, MLX, and GGUF tree audits across four frontier open-weight ecosystems (`Thinking Machines Inkling 2`, `Xiaomi MiMo-V2.6`, `Z.ai GLM-5.2 / GLM-5.3 / GLM-5.3-Flash`, and `inclusionAI Ling-3.1-flash-1019`) established three structural facts:

1. **Generic runtimes leave `2×–4×` throughput and entire Apple Silicon memory tiers on the table for `Inkling 2` (`Mini` & `Small`):**
   - **`Inkling-2-Mini` (`inkling2_loop`, `~26.4B` physical params, `18.78 GiB` native)**: executes `24` physical layers (`Prelude 0..5`, `Core 6..17 × 3 passes`, `Coda 18..23` = `48` effective steps, `128` experts top-8, `intermediate_size = 1152` non-divisible by `256`, `23L` ViT `0.59 GiB`, `3L` `DFlare` `block_size = 8` `0.59 GiB`). Naive execution hits five bottlenecks on the `24 GB` M4 (`17.76 GiB` wired limit, [ADR-0012](0012-wired-memory-budget-headroom.md)): (a) `18.78 GiB + KV` crosses the `17.76 GiB` wired ceiling and triggers macOS swap collapse; (b) serial `M = 1` execution streams `Core 6..17` weights `3×` per token (`~3.6 GiB/tok`); (c) unfused per-layer micro-ops (`k = 4` `sconv`, dynamic `weight_proj`, `d_rel = 16` relative attention, `qk/rel_l2_clip`, `log_scaling`, `attn_sink_proj`, `gate_attn`, 6-layer `attn_res`, and `4`/`25`/`30` softcaps) emit `400+` tiny GPU dispatches/token; (d) linear `48`-step KV allocation ignores that `20/24` physical layers (`40/48` effective steps) use `sliding_window_size = 1024`; and (e) `loop_config.gate_rank = 64` controllers match `sme2_panel64` (`64`-wide panels) for L2-resident CPU `SME2` execution (`0` GPU bus contention).
   - **`Inkling-2-Small` (`inkling2`, `~597B` total params, `363.27 GiB` native)**: despite `48` layers × `384` experts (`31.85M`/expert = `587.1B` routed params), `n_shared_experts = 0` and `8` KV heads keep **always-active parameters (including `5L` `DFlare`) to just `9.69B` (`~5.08 GiB` at `Q4`)**, with `40/48` `1024`-token SWA layers and `12.23B` routed params/token — enabling interactive cold-NVMe hybrid execution on `24 GB` M4 internal NVMe (`~3.2 GB/s`), high-precision `GSQ-RCO` on `128 GB` Max, full wired residency (`~168 GiB`) on `256 GB` Studio, and `110–160+ tok/s` native residency on `512 GB` Ultra.
2. **Always-active vs per-token routed split separates `300B–600B` `Flash` MoEs from `750B–1T` flagships under `aminfer`'s cold-expert + block-diffusion stack:**
   - **`MiMo-V2.6-Flash` (`310.8B` total)**: only **`7.96B` always-active** (`4.17 GiB` at `Q4`) and **`9.46B` routed/tok** (`47` MoE layers, `256` experts top-8, `n_shared_experts = null`, `40/48` layers `sliding_window = 128` with `swa_head_dim = 192` / `swa_v_head_dim = 128`, `8` global `4`-KV-head layers, plus native `3`-layer MTP and official `DFlash` block-diffusion drafter), leaving **`~12.1 GiB` wired RAM** on `24 GB` M4 for hot experts (`16%–21%` of experts). Its official dense sibling **`MiMo-V2.6-Distill-Qwen-9B`** (`9.41B` `qwen3_5` hybrid `GDN + GQA + ViT`) fits `100%` wired (`~5.5 GiB` `4-bit`) on `24 GB` M4.
   - **`GLM-5.3-Flash` (`321.3B` total, multimodal `glm5_next`)**: **`16.92B` always-active** (reducible by stripping `24L` ViT via `W20` `--vision-sidecar`) and **`8.46B` routed/tok** (`42` MoE layers, `288` experts top-8 + `1` shared, `20`-step `hc_mult = 4` `mHC` Sinkhorn, `34` linear-attention layers + `11` `NoPE MLA` `qk_rope_head_dim = 0` layers with `index_kpool = 4`, `1L` MTP + community `DFlash2`).
   - **`Ling-3.1-flash-1019` (`572.1B` total, `bailing_hybrid`)**: **`24.51B` always-active** across `63` `no_kda_lora = true` `KDA` (`Kernel Delta Attention`) layers + `9` Gated `MLA` (`g_proj`) layers, paired with **`512` tiny `15.73M` experts** (`moe_intermediate_size = 1280`) giving only **`8.56B` routed/tok**; compressing the `63` dense KDA projections to `RCO-Q3` shrinks always-active wired RAM from `12.84 GiB` to `~9.2 GiB`, doubling hot-expert cache on `24 GB` M4.
   - **`GLM-5.2` & `GLM-5.3` (`753.3B` total, identical `glm_moe_dsa` graph)** and **`MiMo-V2.6-Pro` (`1,024.2B` total, `mimo_v2` `70L` `384E`)**: **`24.0B–28.6B` always-active** and **`20.8B–22.7B` routed/tok** cap lossless `24 GB` M4 NVMe decode at `~2.2–3.5 tok/s`, while `GSQ-RCO` (`NVFP4`/`Q4` hot + `Q2`/`IQ2` cold tail) plus native `index_topk_freq = 4` / `index_share_for_mtp_iteration = true` (`GLM-5.2/5.3`) and `60/70` `128`-token SWA + `DFlash` (`MiMo-V2.6-Pro`) compress both models from multi-node clusters into **full wired residency on a single `512 GB` Mac Studio** (`~280–350 GiB`).
3. **Storage link speed governs cold-expert viability ([ADR-0006](0006-amf-cold-experts-direct-io.md), Spike S3):** `24 GB` M4 cold-expert targets require internal Apple NVMe (`~3.2–3.5 GB/s` `F_NOCACHE`) or Thunderbolt 4/5 NVMe (`3.0–5.5 GB/s`), whereas USB `10 Gbps` external volumes (`0.638 GB/s`) bottleneck `> 0.4 GiB/tok` workloads below `1.5 tok/s`.

The user confirmed delivering deep, architecture-specific optimizations for **`Inkling 2` (`Mini` & `Small`)** in `M8.2` and completing deep optimizations for **`MiMo-V2.6`**, **`GLM-5` (`5.2`, `5.3`, `5.3-Flash`)**, and **`Ling-3.1-flash-1019`** across three ordered post-`M8.2` stages (`M9.1` → `M9.2` → `M9.3`).

## Decision

### 1. Deep `Inkling 2` (`Mini` & `Small`) co-design in `M8.2` (`W23`, `G25`, `D14`, `C26`)

1. **SIMD register-fused Metal 4 sublayer pipelines (`G25`, zero stack spill):**
   - Fuse `RMSNorm + k = 4 sconv` (static depthwise + global dynamic `weight_proj`);
   - Fuse `qk_l2_clip` (`22.627`) + `log_scaling` + `d_rel = 16` (`rel_extent = 1024`) low-rank relative inner product + `rel_l2_clip` (`8.0`) + `attn_sink_proj` + FlashAttention/GQA + `gate_attn` + attention softcap inside a single register-resident kernel pass (`0` intermediate global-memory writes);
   - Fuse MoE router `bias + route_scale` (`8.0`) + SwiGLU softcaps (`4`/`25`/`30`) + 6-layer `attn_res` normalized block residual queries into the MoE epilogue.
2. **L2-resident `SME2` `gate_rank = 64` loop controllers + `DFlare` batch loop amortization (`W23`, `D14`, `C25`):**
   - Pack `Inkling-2-Mini` `controllers.{attn,k,v,mlp}.{down,forget,gain}_weight` (`gate_rank = 64`) directly into `sme2_panel64` inside M4's `16 MiB` P-cluster L2 (`Placement.cpu_sme2`, `0` Metal command-buffer submissions);
   - Pair `3L` (`Mini`) and `5L` (`Small`) `DFlare` (`block_size = 8`, `markov_rank = 256`, `rnn_hidden` Markov head over multi-layer `target_layer_ids`) with `D7` expert-byte-budget pruning and `M = 8` batched `Core × 3` verification (`D12`/`G5`) so `Mini`'s 3-pass recurrent weights and `Small`'s routed experts are amortized across `τ ≈ 4.5–6.0` accepted tokens.
3. **Constant-memory `40/48` `HybridKvRing` SWA + `W20` sidecar & `GSQ-RCO` tiering (`G25`, `C26`):**
   - Bind `local_layer_ids` (`20/24` physical = `40/48` effective steps in `Mini`; `40/48` in `Small`) to fixed `1024`-token `HybridKvRing` slots (`~80 MiB` constant in `Mini`), allocating growing KV tiles (`+ R5/R7` `q8`/`FP4` retirement) strictly on the `4` physical (`8` effective) / `8` global `8`-KV-head layers to unlock `128K–256K+` context on `24 GB` M4;
   - Strip `0.59 GiB` `23L` ViT into `--vision-sidecar` (`W20`) for text/code sessions and spill only the coldest `~10%` (`~2.2 GiB`) routed experts via `C3`–`C5` `F_NOCACHE` (or `GSQ-RCO` `Q2_0` cold tail) so native `Inkling-2-Mini` + `DFlare` run with `swapouts_delta = 0` on `24 GB` M4, while `Inkling-2-Small` `GSQ-RCO` profiles target `24 GB` M4 internal NVMe (`Q4` `9.69B` always-active + `11.5 GiB` hot cache + `Q2_0`/`IQ1_S` cold NVMe), `128 GB` Max, full-wired `256 GB` Studio (`~168 GiB`), and native `512 GB` Ultra.

### 2. Staged `M9.1`–`M9.3` frontier deep-optimization roadmap (`W25`–`W27`, `G26`–`G28`, `D15`, `C27`)

- **`M9.1` — `MiMo-V2.6` family (`Flash` `309B`, `Pro` `1.02T`, `Distill-Qwen-9B` `9.4B`; `W25`, `G26`, `D15`):**
  - Implement `mimo_v2` `.amf` conversion (`W25`), `GSQ-RCO` profiles (`Flash` `~92 GiB` full-wired on `128 GB` Max; `Pro` `~350 GiB` full-wired on single `512 GB` Ultra), and `qwen3_5` dense hybrid (`GDN + GQA + ViT`) mapping for local `24 GB` M4 full-wired `MiMo-V2.6-Distill-Qwen-9B`;
  - Add Metal 4 kernels for asymmetric SWA head dimensions (`swa_head_dim = 192`, `swa_v_head_dim = 128`, `sliding_window = 128` + `add_swa_attention_sink_bias` across `40/48` `Flash` & `60/70` `Pro` layers, `G26`);
  - Extend `D14` with `D15`: chained `3`-layer native `MTP` (`num_nextn_predict_layers = 3`) and `8–16`-token `DFlash` block-diffusion drafting + `D7` cold-expert block pruning (`~7–11 tok/s` target on `24 GB` M4 internal NVMe for `Flash`).
- **`M9.2` — `GLM-5` family (`GLM-5.2` & `5.3` `753B` `glm_moe_dsa`, `GLM-5.3-Flash` `320B` `glm5_next`; `W26`, `G27`, `D15`):**
  - Implement `glm_moe_dsa` and `glm5_next` `.amf` converters (`W26`, `W20` `--vision-sidecar` `24L` ViT split saving `~1.5–3 GiB` wired RAM on `GLM-5.3-Flash`, and `GSQ-RCO` `~280 GiB` single-`512 GB` Ultra / `~94 GiB` `128 GB` Max profiles);
  - Add `G27` Metal 4 kernels: `index_topk_freq = 4` (`3/4` layers `shared`) + `index_share_for_mtp_iteration = true` zero-reindex `MLA`/`DSA` (`GLM-5.2/5.3`, reusing `G11` + `G25` `IndexCache`), single-kernel register-fused 20-iteration `hc_mult = 4` `mHC` Sinkhorn (`0` multi-pass launch overhead), `34`-layer `fused_qkvbfg_a_proj` + `k = 4` conv1d + `A_log`/`dt_bias` linear attention, and `11`-layer `NoPE MLA` (`qk_rope_head_dim = 0`, `index_kpool = 4` reusing `G25` `IndexPool`), paired with `1L` MTP and `DFlash2` (`D15`).
- **`M9.3` — `Ling-3.1-flash-1019` (`572B` `bailing_hybrid`) & cross-family dynamic MoE tiering (`W27`, `G28`, `C27`):**
  - Implement `bailing_hybrid` `.amf` conversion + `RCO-Q3` dense-KDA compression (`W27`, shrinking `24.51B` always-active weights from `12.84 GiB` to `~9.2 GiB` to double `24 GB` M4 hot-expert cache to `~7.1 GiB`, and fitting `~172 GiB` `GSQ-RCO` full-wired on `256 GB` Studio);
  - Add `G28` Metal 4 kernels for `63`-layer `KDA` (`Kernel Delta Attention`: `kda_lower_bound = -5.0`, `kda_safe_gate = true`, `linear_silu = true`, `short_conv_kernel_size = 4`, `no_kda_lora = true`) + `9`-layer Gated `MLA` (`g_proj`) + L2 `SME2`-offloaded `eh_proj` `1L` MTP;
  - Add `C27` across all cold-NVMe MoE models: (1) **lossless** session-local dynamic hot-expert working-set promotion in `WiredArena` (`C5` two-tier pinned global top-heat + mutable session LFU ring), and (2) **opt-in approximate** `--cold-expert-min-weight <float>` (default `0.0` = strictly lossless) skipping only NVMe-cold routed tail experts whose normalized weight falls below threshold while executing all wired hot experts unconditionally.

## Consequences

- **Positive:** Maximizes primitive reuse (`M8.1` dtypes/tokenizers/templates → `M8.2` `DFlare`/`IndexCache`/`IndexPool`/`HybridKvRing` → `M9.1` `DFlash`/asymmetric SWA → `M9.2` zero-reindex `DSA`/Sinkhorn/`NoPE MLA` → `M9.3` `KDA`/`C27`) with zero rework; unlocks `128K+` lossless `Inkling-2-Mini` and `MiMo-V2.6-Distill-9B` locally on `24 GB` M4; brings `300B–600B` `Flash`/`Small` MoEs into interactive `~4–11 tok/s` range on `24 GB` M4 internal NVMe; and drops `309B–1.02T` full-wired memory requirements by a full Mac hardware tier via `GSQ-RCO`.
- **Negative / trade-offs:** Cold-expert runs on `24 GB` M4 remain bounded by internal/TB4/TB5 NVMe throughput (`~20–35 s` cold prefill scan on first uncached long prompt) and are non-viable on slow `0.64 GB/s` USB disks; `--cold-expert-min-weight > 0` introduces a controlled quality trade-off and therefore stays strictly opt-in (`0.0` default) and gated by `W24` `KL`/code-math evaluation.

## Evidence

- Verified live Hugging Face `config.json`, safetensors parameter totals, and quantized shard sizes (`2026-11-24`):
  - `thinkingmachines/Inkling-2-Mini` (`24L`, `core 6..18 × 3`, `gate_rank = 64`, `20/24` SWA `1024`, `128E` `1152`, `3L` `DFlare` `block = 8`, `18.78 GiB`) & `thinkingmachines/Inkling-2-Small` (`48L`, `40/48` SWA `1024`, `384E` `2304`, `9.69B` always-active + `587.1B` routed, `5L` `DFlare` `block = 8`, `363.27 GiB` native).
  - `XiaomiMiMo/MiMo-V2.6-Flash-RL` (`310,756,322,688` params, `MXFP4` `155.8 GiB`, `Q2_K` `117.7 GiB`, `dflash` `2.7 GiB`, `mtp` `1.2 GiB`), `XiaomiMiMo/MiMo-V2.6-Pro-RL` (`1,024,216,603,392` params, `MXFP4` `515.9 GiB`), `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` (`9,409,813,744` params, `qwen3_5`).
  - `zai-org/GLM-5.2` & `GLM-5.3` (`753,329,940,480` params, `glm_moe_dsa`, `NVFP4` `432.3–432.9 GiB`, `UD-Q2_K_XL` `236.4 GiB`, `UD-IQ1_S` `201.8 GiB`) & `zai-org/GLM-5.3-Flash` (`321,323,031,390` params, `glm5_next`, `4-bit` `186.0–190.0 GiB`, `UD-Q2_K_XL` `101.3 GiB`, `UD-IQ1_S` `86.7 GiB`).
  - `inclusionAI/Ling-3.1-flash-1019` (`572,120,680,768` params, `bailing_hybrid`, `oQ4e` `307.4 GiB`, `oQ2e` `179.9 GiB`, `IQ1_S` `122.0 GiB`).
- Measured M4 hardware baselines in [../benchmarks/results.md](../benchmarks/results.md): `18,186 MiB` wired ceiling ([ADR-0012](0012-wired-memory-budget-headroom.md)), Spike S3 `F_NOCACHE` I/O, M6 `D12`/`D13`/`W17` `SelfSpecRecurrent` + `SME2` L2 placement + `GSQ-RCO`, and M8.1 `W21`–`W24`/`G23`–`G24`/`C25` (`5da4f9e`).

## Open items

- Benchmark fused vs unfused `Inkling 2` sublayer occupancy (`-Wframe-larger-than=1` zero-spill invariant) and `3L`/`5L` `DFlare` acceptance `τ` during `M8.2`.
- Verify exact `DFlash` / `DFlash2` tensor schemas across `ggml-org/MiMo-V2.6-Flash-RL-GGUF` and `GLM-5.3-Flash` against `D14` `arch = dflash` draft bundles before `M9.1`/`M9.2`.
- Calibrate `W24` `KL`/code-math curves for `RCO-Q3` dense `KDA` (`Ling-3.1`) and opt-in `--cold-expert-min-weight` (`C27`) on synthetic and tiny/distilled reference fixtures.
