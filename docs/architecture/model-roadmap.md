# Multi-model architecture and frontier roadmap

> Design reference for scaling aminfer beyond the M2 `Qwen3-30B-A3B` /
> `Qwen3-14B` baselines to `Qwen3.8-Flash-Next` and `DeepSeek-V4.1-Flash-1017`
> ([ADR-0015](../decisions/0015-tiered-frontier-model-roadmap.md)). Implemented
> vs planned status lives only in [../status.md](../status.md); milestone
> ordering lives in [../process.md](../process.md).

## Why the core engine generalizes cleanly

aminfer's substrate — 16 KiB wired arenas and startup budget enforcement,
multi-shard `MTLBuffer` wrapping above 13.3 GiB, `.amf` `F_NOCACHE` direct I/O,
Metal 4 multi-frame command batching, fused 3/4-bit affine dequantization,
deduplicated MoE scheduling, IOSurface KV tiles, lock-free rings and native Zig
BPE tokenization — does not assume a single homogeneous transformer block.
Supporting frontier hybrid-sparse models requires refactoring model graph
wiring in `src/model/` into composable operator axes while keeping the memory,
I/O and Metal 4 submission layers intact:

| Axis | Variants | Models covered |
| --- | --- | --- |
| `BackboneKind` | `decoder_only` · `causal_enc_dec` (20L encoder → projected decoder KV → 20L decoder) | Qwen3, Qwen3.8 · DeepSeek-V4.1-Flash-1017 |
| `AttentionKind` | `gqa_full` · `gated_deltanet` (recurrent state) · `qsa_sparse` (micro-block indexer + YaRN/MRoPE) · `mla_csa2` (`Full`/`Reindex`/`Reuse` + FP4 KV) | Qwen3 · Qwen3.8 (3:1 GDN:QSA) · DeepSeek-V4.1-Flash-1017 |
| `FfnKind` | `dense_swiglu` · `moe_swiglu` (up to ≥512 routed experts, top-k + optional `shared_expert`) | Qwen3-14B · Qwen3-30B-A3B, Qwen3.8, DeepSeek-V4.1-Flash-1017 |
| `ResidualKind` | `single_stream` · `gated_4branch` (GR) · `mhc_single_pass` (Mega-mHC) | Qwen3 · Qwen3.8 · DeepSeek-V4.1-Flash-1017 |
| `MemoryLookupKind` | `none` · `ple_ngram` (51B) · `engram` (196B) | Qwen3 · Qwen3.8 · DeepSeek-V4.1-Flash-1017 |
| `LoopSchedule` | `single_pass` (`max_loops = 1`) · `native_recurrent` (`Prelude → Core × R → Coda`) · `gated_loop_adapter` (L2-resident SME2 `Step-LoRA` + zero-init scalar gate `α_r`, [ADR-0019](../decisions/0019-recurrent-depth-and-post-m5-accelerators.md)) | Standard models · Huginn / LoopLM · Adapter-extended Qwen3, Qwen3.8, DeepSeek-V4.1-Flash-1017 |

## Tiered hardware and model matrix

3-bit weights below assume MLX-compatible affine group-64 (~3.5 bits/weight,
matching M1/M2 `.amf` packing).

| Tier | Representative models | Total / active params | 3-bit backbone + table size | Target Apple Silicon hardware |
| --- | --- | --- | --- | --- |
| **Tier 1** (M2–M5, M2.5 FP16/BF16, M9.1 `qwen3_5`) | `Qwen3-30B-A3B` (3-bit / BF16), `Qwen3-14B` (4-bit / BF16), `Qwen3-0.6B`–`8B` (BF16 / FP16), **`MiMo-V2.6-Distill-Qwen-9B`** (`9.41B` `qwen3_5` GDN+GQA+ViT); extensible to Llama 3/4, Mistral, Gemma, DeepSeek-V2/V3 | 0.6B–30.5B (3B active MoE; 0.6B–14B dense) | ~12.4 GiB (30B-3bit), ~7.5 GiB (14B-4bit), `4.29 GiB` + `873 MiB` ViT (`Distill-9B` `GSQ-RCO`), ~1.2–17.5 GiB (0.6B–9B BF16) | **24 GB M4 baseline** (30B-3bit, 14B-4bit, `Distill-9B` `4.00 bpw` + `27L` ViT sidecar verified at `0` swap, ≤8B BF16/FP16 fully wired) · **36–128 GB+ Macs** (14B/30B BF16/FP16 fully wired, [ADR-0016](../decisions/0016-native-fp16-bf16-and-multishard-import.md)) |
| **Tier 2** (M6) | **`Qwen3.8-Flash-Next`** (`qwen4_exp`) | 125B backbone (6B active/tok; `2.36B` routed/tok) + 51B PLE + 4B MTP | ~54.7 GiB `3.5b` / ~36.3 GiB `GSQ-RCO` + NVMe PLE | **128 GB+ Max / Ultra** (full backbone wired) · **24 GB M4** (`~11.2–11.6 GiB` hot experts wired + NVMe cold experts & PLE) |
| **Tier 2B** (M8.2) | **`Inkling-2-Mini`** (`inkling2_loop`, 24L × 3 core passes = 48 eff. steps, 128E top-8) | ~26.4B params (`~5.7B` eff. active; `NVFP4`/`MXFP8` + `0.59 GiB` ViT + `0.59 GiB` `3L` DFlare) | `18.78 GiB` native (`17.60 GiB` excl. ViT); `10.9 GiB` (`UD-IQ3_XXS`) / `16.4 GiB` (`UD-Q4_K_XL`) | **24 GB M4 baseline** (`W20` ViT sidecar stripped + `~10%` cold-expert `F_NOCACHE` spill or `GSQ-RCO`/`UD-Q4`, `40/48` `1024`-token SWA ring unlocking `128K–256K` context, `SME2` `gate_rank=64` controllers + `3L` `DFlare`) |
| **Tier 2C** (M8.2 & M9.1–M9.3) | **`MiMo-V2.6-Flash`** (`309B`, `mimo_v2` `39/48` SWA `192/128`), **`GLM-5.3-Flash`** (`320B`, `glm5_next`), **`Inkling-2-Small`** (`~597B`, `inkling2`), **`Ling-3.1-flash-1019`** (`572B`, `bailing_hybrid`) | `309B–597B` total (`7.96B–24.51B` always-active; `8.46B–12.23B` routed/tok) | `GSQ-RCO`: `~92–94 GiB` (`MiMo`/`GLM` Flash), `~155–172 GiB` (`Inkling-Small`/`Ling-3.1`); native `155.8–533 GiB` | **`128 GB` Max** (`MiMo`/`GLM` Flash `GSQ-RCO` 100% wired; `Inkling-Small` hybrid) · **`256 GB` Studio** (`MiMo`/`GLM` native 4-bit, `Inkling-Small` & `Ling-3.1` `GSQ-RCO` 100% wired) · **`24 GB` M4 internal/TB NVMe** (`3.4–12.1 GiB` wired hot pool + `Q2`/`IQ1_S` cold NVMe + `DFlash`/`DFlare`/`MTP` + `D7`, [ADR-0025](../decisions/0025-staged-frontier-deep-optimization-inkling2-mimo-glm5-ling3.md)) |
| **Tier 3** (M7 & M9.4 HF alignment) | **`DeepSeek-V4.1-Flash-1017`** (`deepseek41`) | 552B CED MoE (pre-release est. 8B/16B; actual published HF `hidden=5120`, `384E`, `top_k=6`, 20L Enc + 20L Dec -> `4.25B` Decoder routed/tok) + 196B Engram (~748B total) | ~241.5 GiB backbone + ~86 GiB Engram (>330 GiB on disk; actual HF `224.8 GiB` `q3_affine` est.) | **512 GB Ultra** (full backbone wired on 800 GB/s bus) · **Multi-Mac 128–256 GB Thunderbolt 5 RDMA cluster** (`thunderbolt_peer`) · **24 GB M4 internal/TB4 PCIe NVMe (`3.2–3.8 GB/s`)** becomes viable for cold-NVMe Decoder tiering (`~11.7–13.3 GiB` hot cache, `~2.5–3.0 tok/s` FP4 / `~6–9 tok/s` `Q2_K` + spec dedup) once `M9.4` (`W28`) real-HF alignment lands ([ADR-0015](../decisions/0015-tiered-frontier-model-roadmap.md)) |
| **Tier 3B** (M8.2 & M9.1–M9.2 / HW-gated) | **`Hy4-flash`** (`220B`), **`Hy4` / `Hy4-preview`** (`770B`), **`GLM-5.2` / `GLM-5.3`** (`753B`, `glm_moe_dsa`), **`MiMo-V2.6-Pro`** (`1.02T`, `mimo_v2`) | `220B–1,024B` total (`Hy4-flash` `11.3B` active; `753B–1.02T` flagships `24.0B–28.6B` always-active & `20.8B–49B` active/tok) | `Hy4-flash` `76.3–267 GiB`; `GLM-5.2/5.3` `~280 GiB` `GSQ-RCO` (`432 GiB` `NVFP4`); `MiMo-Pro` `~350 GiB` `GSQ-RCO` (`515.9 GiB` `MXFP4`) | `Hy4-flash` on **`128–256 GB`**; **`GLM-5.2/5.3` (`753B`) & `MiMo-V2.6-Pro` (`1.02T`) `GSQ-RCO` 100% wired on a single `512 GB` Ultra** (native `FP8`/`MXFP4` & `Hy4` `4-bit` on **multi-Mac `TB5 RDMA` cluster**; `hy4-preview-tiny-random` on **24 GB M4**) |

### Optional GSQ-RCO footprint reduction across Tiers 1–3

> **Optional Pipeline** ([ADR-0017](../decisions/0017-optional-gsq-rco-hybrid-quantization.md)
> and [ADR-0019](../decisions/0019-recurrent-depth-and-post-m5-accelerators.md);
> capabilities `W14`, `W17`, and `G13` in [../status.md](../status.md); M5
> baseline measurements in
> [../benchmarks/results.md](../benchmarks/results.md#m5-serving-pipelining-c6-nvme-n-gram-and-adr-0017-gsqrco-hybrid-quantization)).

The uniform-precision sizes in the matrix above (`3-bit`, `4-bit`, `FP16` /
`BF16`) remain the default export path and mandatory benchmark baseline. As an
opt-in offline pipeline in `tools/` (`--quant-profile gsq-rco`), GSQ + RCO can
assign static per-tensor / per-expert `QuantScheme` tags in `.amf` (strictly
whole-matrix / whole-expert, never intra-tensor mixed-bit) to reduce wired DRAM
and NVMe footprint across MoE tiers:

- **Tier 1 (`Qwen3-30B-A3B`)**: keep sensitive attention projections, routers,
  and hot routed experts at higher precision (`q4_affine` / `bf16`) while
  packing low-sensitivity or cold routed experts at `q3_affine` / `q2_affine`,
  expanding 24 GB M4 wired headroom for KV tiles and speculative draft heads
  (or improving fidelity within the same wired budget).
- **Tier 2 (`Qwen3.8-Flash-Next`, 512 routed experts + `shared_expert`)**:
  protect `shared_expert`, dense GDN/QSA projections, and hot routed experts
  while compressing the long tail of cold routed experts to `q3_affine` /
  `q2_affine`, shrinking both the 128 GB+ wired footprint and per-expert NVMe
  direct-I/O bytes (`b_{l,e}`) streamed on 24 GB M4.
- **Tier 3 (`DeepSeek-V4.1-Flash-1017`, 552B CED MoE)**: lower total backbone
  wired footprint on 512 GB Ultra nodes and reduce per-expert bytes transferred
  across Thunderbolt 5 RDMA clusters (24 GB M4 remains out of scope per
  [ADR-0015](../decisions/0015-tiered-frontier-model-roadmap.md)).

Every `--quant-profile gsq-rco` build must be benchmarked side-by-side against
its uniform-precision `.amf` baseline on perplexity / output fidelity, wired
DRAM footprint (`swapouts_delta = 0`), effective `bytes/token`, and `tok/s` in
[../benchmarks/results.md](../benchmarks/results.md).

## Tier 2: `Qwen3.8-Flash-Next` (`qwen4_exp`) design fit and required work

### Architectural alignment with aminfer

1. **51B N-gram Per-Layer Embedding (PLE) ↔ NVMe `F_NOCACHE` direct I/O**:
   PLE performs sparse bigram/trigram row lookups with negligible arithmetic.
   Leaving the 51B PLE shards on NVMe in 16 KiB-aligned `.amf` files and
   prefetching needed rows into small wired ring slots avoids consuming wired
   DRAM while bypassing the macOS page cache ([ADR-0006](../decisions/0006-amf-cold-experts-direct-io.md)).
2. **75% KV-cache reduction ↔ IOSurface `KvTile` pool**:
   Out of 48 layers, only **12 QSA layers** allocate growing `KvTile`s; the
   **36 Gated DeltaNet layers** hold fixed-size recurrent state tensors in
   `WiredArena`, allowing 256K+ context windows on 24 GB machines without KV
   exhaustion.
3. **512-choose-10 MoE + native 4B MTP ↔ expert-byte-budget tree pruning**:
   The 4B MTP module feeds M3's MTP draft path (`D3`), while
   [ADR-0009](../decisions/0009-expert-byte-budget-tree-pruning.md) prevents
   speculative branches from activating an unbounded union of the 512 experts.

### Five delta modules over M2

1. **MoE scaling (`512` experts + `shared_expert`)**: raise `MAX_EXPERTS` in
   `src/gpu/moe.zig` / `moe_ffn.metal` from `256` to `>= 512` (`MAX_TOP_K = 16`
   already covers top-10) and accumulate one unconditional `shared_expert`
   SwiGLU branch per MoE layer.
2. **Gated DeltaNet kernels**: add Metal 4 chunked-prefill and single-step
   recurrent state-update kernels (gated decay + delta rule) and 3:1
   per-layer heterogeneous dispatch.
3. **Qwen Sparse Attention (QSA) + YaRN / interleaved MRoPE**: add `yarn` RoPE
   scaling, interleaved `mrope_section: [11, 11, 10]`, and QSA micro-block
   indexer top-k tile/block selection before reading `KvTile`s.
4. **4-branch Gated Residual (GR)**: widen the inter-layer residual buffer to
   4 branches with fused elementwise read-gate and scalar write-gate kernels.
5. **Offline `qwen4_exp` converter & synthetic-first validation**: extend
   `tools/` to strip optional vision towers for text/code serving, export the
   125B backbone + 4B MTP head, emit sharded 16 KiB-aligned PLE `.amf` tables,
   and generate deterministic synthetic / scale-calibrated `qwen4_exp` bundles so
   every M6 operator and 24 GB M4 wired budget is verified prior to drop-in real
   checkpoint import; reuse `Tokenizer.fromHuggingFaceJson`.

### Composable recurrent depth (`LoopSchedule`) and `SelfSpecRecurrent` (`ADR-0019`)

Accepted in [ADR-0019](../decisions/0019-recurrent-depth-and-post-m5-accelerators.md)
(capabilities `W9`, `W16`, `D12`, `D13` in [../status.md](../status.md)):

- **Unified schedule for native and adapter-extended models**: native looped
  models (`Prelude → [Core] × R → Coda`) and frozen checkpoints augmented offline
  with a non-destructive **Gated Loop Adapter** (`Step-LoRA` + zero-initialized
  scalar gate `α_r` where `α_0 = 0` preserves bit-exact `r = 0` identity) both
  execute through `LoopSchedule`. Compact `2–12 MiB` `Step-LoRA` adapters reside
  in `sme2_panel64` inside M4's `16 MiB` P-cluster L2 (`ld1b`) and run on ARMv9
  SME2 without Metal GPU contention (`D13`).
- **Batch tree verification flips the `1/R` bandwidth penalty (`D12`)**:
  `SelfSpecRecurrent` uses a shallow `r = 1` (or `r = 0` / `EAGLE-3` / `4B MTP`)
  pass to draft an `M = 8..16` candidate tree and verifies all `M` nodes in one
  compute-dense `Core × R` pass under lossless `D9` acceptance, skipping
  full-vocabulary `lm_head` on intermediate iterations `r < R - 1` (with opt-in
  `g = sigmoid(w · h)` early exit for approximate mode).
- **Four mandatory hybrid-sparse isolations on `r < R - 1`**:
  1. **Gated DeltaNet (`R6`, `G8`)**: evaluate intermediate loop steps into wired
     scratch `S_tmp` and commit persistent `S_t` strictly once at `r = R - 1`.
  2. **External tables (`C10` `51B PLE` / `196B Engram`)**: read NVMe `F_NOCACHE`
     rows once at `r = 0` and hold them in a wired slot across `r >= 1`.
  3. **Sparse indexers (`G9` `QSA` / `G11` `CSA2 Reuse`)**: reuse `r = 0` block
     top-k indices across `r >= 1` when KV keys do not grow.
  4. **MoE routing (`G3`/`G7`, `C5`)**: deduplicate active experts across the
     `M = 8..16` batch and confine `r >= 1` routing to wired / pinned
     `ExpertSlotPool` slots (`0` extra NVMe stalls).

### M6.5 serving, long-context and multimodal bridge (`ADR-0019`)

Between M6 (`Qwen3.8-Flash-Next` text/MoE/PLE core) and M7 (`deepseek41`
cluster scale-out), Milestone **M6.5** lands four orthogonal systems multipliers
tracked in [../status.md](../status.md): iteration-level continuous batching +
domain `.ang` switching (`C14`), shared prefix `KvTile` / `GDN` state caching
(`R8`), GPU-side group-64 `q8` retired 32K+ KV compression (`R5`), and zero-copy
`VideoToolbox → IOSurface → MTLTexture / MLMultiArray` vision/video ingress into
interleaved MRoPE (`G14`).

## Tier 3: `DeepSeek-V4.1-Flash-1017` (`deepseek41`) shared blocks and unique additions

### Primitives shared directly with Tier 2

- **Sparse NVMe table lookup**: 196B Engram conditional memory reuses the same
  `F_NOCACHE` `.amf` sparse row-prefetch pipeline as Qwen3.8's 51B PLE.
- **High-expert-count MoE + shared experts**: reuses the 512+ expert router and
  `shared_expert` path.
- **Generalized multi-branch residual kernel**: Single-Pass mHC (Mega-mHC)
  shares the multi-stream residual abstraction introduced for 4-branch GR.
- **Two-stage sparse attention (indexer → gather)**: CSA2 shares the
  index-then-read execution pattern with QSA.

### Three `deepseek41`-specific subsystems

1. **Causal Encoder-Decoder (CED) two-phase scheduler**: prefill executes only
   the 20-layer causal encoder (8B active/token) and projects the decoder's
   global KV cache straight from the final encoder hidden states; autoregressive
   decode executes only the 20-layer decoder (16B active/token).
2. **MLA + CSA2 (`Full` / `Reindex` / `Reuse`) + FP4 KV + SWA bounded replay**:
   compressed latent KV storage at **890 B/token** (`E2M1` values + `E4M3`
   per-16-channel scales), cross-layer KV and top-k index sharing across static
   CSA2 modes, hierarchical decoder sparse indexing, and bounded window replay
   for sliding-window states.
3. **DSpark speculative decoding + Thunderbolt 5 RDMA expert parallelism**:
   plug DSpark semi-autoregressive drafting into `DraftSource`, and activate
   `ExpertSource.thunderbolt_peer` over macOS 26.2+ Thunderbolt 5 RDMA
   ([ADR-0003](../decisions/0003-expert-source-tagged-union.md)) to partition
   the 552B MoE experts across multi-Mac clusters when a single 512 GB node is
   not used.

## Quantized-format parity, `Inkling 2` and `Hy4` extensions (`ADR-0024`)

Sub-milestones `M8.1` and `M8.2` ([ADR-0024](../decisions/0024-quantized-format-parity-and-inkling2-hy4-adaptation.md),
capabilities `W21`–`W24`, `G23`–`G25`, `D14`, `C25`–`C26` in
[../status.md](../status.md)) extend both weight-format parity and the model
operator axes without sacrificing runtime alignment or conversion fidelity:

- **Native multi-format `.amf` + zero-loss offline `GGUF` repacking (`M8.1`: `W21`,
  `W22`, `W24`, `G23`, `G24`, `C25`):** Add native `.amf` dtypes and bit-exact Metal 4
  register-dequant (`+ SME2`) kernels for `fp8_e4m3_block` (`128×128` `F32`/`BF16`
  inverse scales), `fp8_e4m3_row`, `fp8_e4m3_tensor`, `mxfp8_g32` (`UE8M0`),
  `nvfp4_g16` (`E2M1` + `E4M3` + per-tensor/expert `F32`), MLX `q5`/`q6`/`q8_affine`
  with `group_size in {32, 64, 128}`, and standard upstream `GGUF` types (`Q*`,
  `K`-quants, `IQ*` codebooks in threadgroup SRAM, `TQ1_0`/`TQ2_0`, `MXFP4`,
  `NVFP4` id `40`, `Q2_0` id `42`, `F8_E4M3` id `43`), rejecting colliding or
  custom fork encodings (such as AngelSlim `type 43` `STQ1_0` at `1.312 bpw`,
  `VQ-GGUF`, and `EXL3`) via byte-length validation, pairing `o200k`/`ignore_merges`
  and `Hy4` tokenizers with an offline-compiled declarative `.amf` JSON
  `chat_template_spec` + zero-alloc Zig state-machine `ChatTemplate` renderer,
  and gating all new profiles through opt-in `--act-quant emulate` and `W24`
  `KL`/top-1 quality checks.
- **`Inkling 2` (`inkling2`, `inkling2_loop`) deep co-design (`ADR-0024`, `ADR-0025`)**:
  Reuses `LoopSchedule` (3-pass recurrent core `layers 6..17` in `Mini`) with five
  architecture-specific optimizations: (1) SIMD register-fused sublayer kernels
  collapsing `RMSNorm + k = 4 sconv`, `q/rel_l2_clip + log_scaling + d_rel = 16
  rel_attn + attn_sink_proj + gate_attn + softcap`, and `MoE bias + route_scale +
  softcaps + 6-layer attn_res` (`0` intermediate global-memory writes); (2) L2-resident
  ARMv9.2 `SME2` `sme2_panel64` execution for `Mini`'s `gate_rank = 64` loop
  controllers paired with `3L`/`5L` `DFlare` (`block_size = 8`) `M = 8` batch
  verification (`D14` + `D12`) to amortize 3-pass `Core` reads across `4.5–6.0`
  accepted tokens; (3) `40/48`-step constant-memory `1024`-token `HybridKvRing`
  SWA (`~80 MiB` in `Mini`, growing KV on only `4` physical / `8` effective global
  layers for `128K–256K+` M4 context); (4) `W20` `23L` ViT sidecar stripping +
  `~10%` cold-expert `F_NOCACHE` spill keeping native `18.78 GiB` `Mini` swap-free
  on `24 GB` M4; and (5) `GSQ-RCO` profiles exploiting `Small`'s tiny `9.69B`
  always-active footprint (`~5.1 GiB` `Q4`) across `24 GB` M4 internal NVMe,
  `128 GB` Max, full-wired `256 GB` Studio (`~168 GiB`), and `512 GB` Ultra.
- **`Hy4` family (`hy_v4`, `hy_v4_vl`, `hy_v4_flash`)**: Reuses `G7` `512`-expert
  MoE, `G10` `4`-stream iHC/mHC, `G11` MLA + FP4 KV, `C10` NVMe Engram
  streaming (`F8_E4M3` row-scaled or BF16), and `W20`/`G22` `40L`/`27L` AnyRes
  `ViT`, adding `IndexCache` period-4 sharing, `IndexPool` compressed indexers,
  `LatentMoE`, sigmoid routing with correction bias, `linear_gate`,
  `learnable_sink_param`, `swiglu_limit = 10`, `sandwich_norm`, and `5L` `DFly2`
  block-parallel drafting (`D14`), parity-validated locally on
  `yujiepan/hy4-preview-tiny-random` (`29.77 MiB` source / `38.8 MiB` `.amf`)
  alongside reference-aligned real `thinkingmachines/Inkling-2-Mini`
  (`3L DFlare` & `dspark-Q8` GGUF `mean_accept_len = 3.150–3.750`, `100%`
  bit-exact greedy parity, coherent English/math/Python output, `cold = 24/128`
  `15,489.84 MiB` `m4_24gb = yes` verified and `cold = 120/128` `4,842 MiB`
  desktop-headroom profile running at `0` swapouts) in
  [../benchmarks/results.md](../benchmarks/results.md#m82--inkling-2--hy4-fused-operators-dflaredfly2-block-drafting-sme2-loop-controllers-and-real-m4-execution).

## Staged `M9.1`–`M9.5` frontier deep-optimization, real-HF parity, and universal multi-LoRA roadmap (`ADR-0025`, `ADR-0015`, `ADR-0022`)

Following `M8.2`, `M8.3` (`C28`), `M9.5` (`W30`), `M9.6` (`C29`), `M9.7` (`W20`),
and **`M9.1` (`MiMo-V2.6`, `W25`/`G26`/`D15` — implemented and verified on M4
for `MiMo-V2.6-Distill-Qwen-9B` + `27L` ViT, `synth-mimo`/`bench-m91`, and
staged `Flash`/`Pro` metadata)**, stages `M9.2`–`M9.4`
([ADR-0025](../decisions/0025-staged-frontier-deep-optimization-inkling2-mimo-glm5-ling3.md),
[ADR-0015](../decisions/0015-tiered-frontier-model-roadmap.md),
[ADR-0022](../decisions/0022-native-c-python-sdk-and-full-protocol-parity.md);
capabilities `W25`–`W28`, `W30`, `G26`–`G28`, `D15`, `C27` in
[../status.md](../status.md)) extend model-specific co-design across
`GLM-5`, `Ling-3.1`, and real-HF `Qwen3.8-Flash-Next` /
`DeepSeek-V4.1-Flash-1017` checkpoints with zero primitive rework. Across the
four model-family stages (`M9.1`–`M9.4`),
a mandatory safetensors JSON header pre-flight gate (`W28`) validates
`config.json`, `model.safetensors.index.json`, and HTTP Range safetensors
headers in `tools/tests/test_hf_metadata_validation.py`
(`long_backbone_names == 0` plus exact tensor shapes and scale/broadcast ranks)
before converter implementation:

- **`M9.1` — `MiMo-V2.6` (`Flash` `309B`, `Pro` `1.02T`, `Distill-Qwen-9B` `9.4B`, Implemented on M4 for `Distill-9B` & `mimo_v2` synth/metadata):**
  Exploits `39/48` (`Flash`) and `60/70` (`Pro`) `128`-token `HybridKvRing` SWA
  layers with asymmetric head dimensions (`swa_head_dim = 192`, `swa_v_head_dim = 128`,
  `add_swa_attention_sink_bias`, un-interleaving `TP=4`/`8` `qkv_proj`), pairs
  `Flash`'s ultra-light `7.96B` always-active footprint (`~12.1 GiB` wired
  hot-expert budget on `24 GB` M4) with chained `3`-layer native `MTP` +
  `8–16`-token `DFlash` (`5L` non-causal within-block + `5`-tap `fc`)
  block-diffusion drafting (`D15` + `D7`), provides `GSQ-RCO` tier profiles for
  `Flash` (`128 GB` Max) and `Pro` (`512 GB` Ultra), and runs converted
  `MiMo-V2.6-Distill-Qwen-9B` (`qwen3_5` `4.00 bpw` `4.29 GiB` + `873 MiB` `27L`
  ViT sidecar) fully wired on `24 GB` M4 at `0` swap
  ([../benchmarks/results.md](../benchmarks/results.md#m91-mimo-v26-family-g26-asymmetric-192128-swa-d15-chained-3l-mtpdflash-and-real-distill-qwen-9b-parity)).
- **`M9.2` — `GLM-5` (`GLM-5.2` & `5.3` `753B` `glm_moe_dsa`, `GLM-5.3-Flash` `320B` `glm5_next`) — Implemented on M4 (`22a3e7c`, `2e9ac87`, `293fe71`, `4c9f91e`, `30795a6`):**
  Reuses `G11` + `G25` `IndexCache` for `GLM-5.2/5.3` (`index_topk_freq = 4`
  with `index_skip_topk_offset = 3`: `21/78` full vs `57/78` shared `DSA`,
  skipping `73.1%` of `DSA` re-indexing + `index_share_for_mtp_iteration = true`
  zero-reindex `1L` `MTP`, `GSQ-RCO` `~280 GiB` single-`512 GB` Ultra fit), and
  pairs `GLM-5.3-Flash` with `W20` `24L` `Glm5NextVision` sidecar stripping
  (`q_norm`/`k_norm`, global half-split 2D axial RoPE, channel-first
  `visual.downsample`, `post_projection_norm -> GELU`, `swiglu_limit = 10.0`
  merger), single-kernel register-fused 20-iteration `hc_mult = 4` `mHC`
  Sinkhorn, `34` `KDA` (`-5.0 * sigmoid(...)`, `q/k/v_conv1d`, `o_norm * sigmoid(z)`)
  layers + `11` `NoPE MLA` (`qk_rope_head_dim = 0`, `index_kpool = 4` + tail selection)
  layers + `288E` (`swiglu_limit = 10.0`), and `DFlash2` (`D15` `DFlashGroupedConv` +
  `CandidateSelector`, `B = 8..16`). Full `320B`/`753B` checkpoints remain gated
  on local weight storage and `128–512 GB` hardware; see
  [status](../status.md) and [benchmark results](../benchmarks/results.md).
- **`M9.3` — `Ling-3.1-flash-1019` (`572B` `bailing_hybrid`) & `C27` dynamic MoE tiering:**
  Adds `63`-layer `KDA` (`Kernel Delta Attention`, `kda_lower_bound = -5.0`,
  `kda_safe_gate`, `k = 4` short conv) + `9`-layer Gated `MLA` (`g_proj`) +
  `SME2`-offloaded `eh_proj` `1L` MTP, applies `RCO-Q3` to dense `KDA` projections
  to double `24 GB` M4 hot-expert cache (`3.4 GiB → ~7.1 GiB`, `~172 GiB` full-wired
  on `256 GB` Studio), and introduces `C27` across all cold-NVMe MoEs: lossless
  session-local hot-expert working-set promotion in `WiredArena` plus opt-in
  `--cold-expert-min-weight <float>` (default `0.0` lossless) cold-tail skipping.
- **`M9.4` — Real Hugging Face checkpoint schema & `24 GB` M4 cold-tier parity (`Qwen3.8-Flash-Next` & `DeepSeek-V4.1-Flash-1017`, `W28`):**
  Aligns `qwen4_exp` (`< 47 B` name compaction eliminating all `300` long
  `hyper_connection` names, full `hc_count = 4` mixer projections, dense PLE
  conditioning vs `shard_{0..127}` external `SPARSE_TABLE` split, 3D stacked MoE /
  QSA indexer / MTP keys) and `deepseek41` (`24gb-m4` encoder-cold/decoder-hot
  wired budget + `--allow-tier-override`, `[32, 32]` `ue8m0` `fp8`+`fp4` scales
  and `layers.N.{hc_*,attn.{wkv,wo_a,wo_b},ffn.*}` mapping, `graph.zig`
  `wo_a`/`wo_b`/`compressor`/`sqrtsoftplus` + cold-tiered / `GSQ-RCO` `3L DSpark`)
  with published Hugging Face shard headers.
- **`M9.5` — Universal `GraphModel` & `M82Runtime` runtime multi-LoRA and PEFT DoRA hot-loading (`W30`, `W19`, `G29`, `C28`; [ADR-0022](../decisions/0022-native-c-python-sdk-and-full-protocol-parity.md#11-universal-graphmodel-and-m82runtime-multi-lora-and-dora-hot-loading-w30-m95), [ADR-0026](../decisions/0026-m825-inkling2-single-frame-hot-path-and-full-wired-quant-parity.md)):**
  Extends `W19` `.amlora` and `LoraRegistry` from `Qwen3Model` to `GraphModel`
  and `M82Runtime` across `inkling2`/`inkling2_loop` (`wq_du`, `wk_dv`, `wv_dv`,
  `wo_ud`, and `gate_attn` shared and per-loop-pass projections, including
  `Inkling-2-Mini-Unbound` `peft-r16`/`peft-r64` and DoRA adapters), `hy_v4`,
  `mimo_v2`, `glm_moe_dsa`/`glm5_next`, `bailing_hybrid`, and `deepseek41`; adds
  a single-frame Metal 4 GPU low-rank delta path preserving `G29` `0` mid-pass
  `flushFrame()` alongside `C24` `SME2` `sme2_panel64` offload and an
  `R9`-bounded configurable wired pool cap for `r = 16..64` adapters; and adds
  offline PEFT `r <= 64` + DoRA (`lora_magnitude_vector`) `.amlora` import in
  `tools/src/aminfer_tools/lora.py` with unified `-L`/`--lora` and
  `"<base>:<adapter>"` routing across `aminfer chat`, `aminfer serve`, `C28`
  `/v1/models/{id}`, and `C17` SDK.
