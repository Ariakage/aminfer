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
| **Tier 1** (M2–M5, incl. M2.5 FP16/BF16) | `Qwen3-30B-A3B` (3-bit / BF16), `Qwen3-14B` (4-bit / BF16), `Qwen3-0.6B`–`8B` (BF16 / FP16); extensible to Llama 3/4, Mistral, Gemma, DeepSeek-V2/V3 | 0.6B–30.5B (3B active MoE; 0.6B–14B dense) | ~12.4 GiB (30B-3bit), ~7.5 GiB (14B-4bit), ~1.2–15.3 GiB (0.6B–8B BF16) | **24 GB M4 baseline** (30B-3bit, 14B-4bit, ≤8B BF16/FP16 fully wired) · **36–128 GB+ Macs** (14B/30B BF16/FP16 fully wired, [ADR-0016](../decisions/0016-native-fp16-bf16-and-multishard-import.md)) |
| **Tier 2** (M6) | **`Qwen3.8-Flash-Next`** (`qwen4_exp`) | 125B backbone (6B active/tok) + 51B PLE + 4B MTP | ~54.7 GiB backbone + NVMe-sharded PLE | **128 GB+ Max / Ultra** (full backbone wired) · **24 GB M4** (wired hot set + NVMe cold experts & PLE via ANE prediction + E-core async I/O) |
| **Tier 3** (M7 / Future) | **`DeepSeek-V4.1-Flash-1017`** (`deepseek41`) | 552B CED MoE (8B prefill / 16B decode) + 196B Engram (~748B total) | ~241.5 GiB backbone + ~86 GiB Engram (>330 GiB on disk; ~7.0 GiB/tok decode reads) | **512 GB Ultra** (full backbone wired on 800 GB/s bus) · **Multi-Mac 128–256 GB Thunderbolt 5 RDMA cluster** (`thunderbolt_peer`); **not viable on 24 GB M4** (< 0.7 tok/s NVMe bound) |

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
