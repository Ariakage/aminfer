# ADR-0015: Tiered frontier model roadmap (`Qwen3.8-Flash-Next` and `DeepSeek-V4.1-Flash-1017`)

- Status: Accepted
- Date: 2026-11-20 (empirical addendum 2026-11-25)
- Source: user roadmap direction (post-M2 session)

## Context

With M2 complete, aminfer's core substrate — 16 KiB wired arenas and budget
enforcement, `.amf` v0 direct-I/O containers, Metal 4 multi-frame command
batching, fused 3/4-bit dequantization, deduplicated MoE execution, IOSurface
KV texture tiles, lock-free rings and native BPE tokenization — is
model-agnostic, while `src/model/qwen3.zig` currently wires a standard
decoder-only GQA + dense/MoE graph (`Qwen3-30B-A3B` and `Qwen3-14B`).

Two frontier ultra-sparse hybrid architectures are now confirmed roadmap
targets:

1. **`Qwen3.8-Flash-Next`** (`qwen4_exp`): 125B MoE backbone (6B active per
   token, 512 routed experts top-10 + 1 shared expert) + 51B Per-Layer
   Embedding / N-gram lookup table (PLE) + 3:1 hybrid attention across 48
   layers (36 Gated DeltaNet recurrent linear-attention layers + 12 Qwen Sparse
   Attention layers with micro-block indexing and YaRN / interleaved MRoPE) +
   4-branch Gated Residual (GR) + 4B native MTP head.
2. **`DeepSeek-V4.1-Flash-1017`** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`,
   `deepseek41`): 552B MoE backbone structured as a 40-layer Causal
   Encoder-Decoder (20-layer causal encoder activating 8B/token in prefill and
   projecting global decoder KV from final encoder states + 20-layer decoder
   activating 16B/token in decode) + 196B Engram conditional memory lookup
   table (~748B total parameters) + Compressed Sparse Attention 2 (CSA2:
   `Full`, `Reindex`, `Reuse` modes with hierarchical sparse indexing) + FP4
   main KV cache (`E2M1` + `E4M3` scale, 890 B/token, up to 1M context) +
   Single-Pass mHC residual mixing + DSpark speculative decoding.

Both architectures converge on the same underlying systems primitives (sparse
external N-gram/Engram memory tables on NVMe, high-expert-count MoE with shared
experts, sparse block-indexed attention, and multi-branch gated residual
streams), while differing sharply in physical memory footprint (~54.7 GiB vs
~241.5 GiB 3-bit backbones).

## Decision

1. **Composable model graph abstractions.** Rather than forking monolithic
   per-model executors, `src/model/` will evolve after M4/M5 into composable
   operator traits configured by a model registry: `BackboneKind`
   (`decoder_only`, `causal_enc_dec`), `AttentionKind` (`gqa_full`,
   `gated_deltanet`, `qsa_sparse`, `mla_csa2`), `FfnKind` (`dense_swiglu`,
   `moe_swiglu` up to ≥512 experts with optional `shared_expert`),
   `ResidualKind` (`single_stream`, `gated_4branch`, `mhc_single_pass`), and
   `MemoryLookupKind` (`none`, `ple_ngram`, `engram`).
2. **Three hardware-honest model tiers** (details in
   [../architecture/model-roadmap.md](../architecture/model-roadmap.md)):
   - **Tier 1 (M2–M5, 24 GB M4 primary)**: standard decoder-only GQA dense and
     MoE models (`Qwen3-30B-A3B` 3-bit, `Qwen3-14B` 4-bit; extensible to Llama,
     Mistral, Gemma, and DeepSeek-V2/V3).
   - **Tier 2 (M6, after M4 cold-expert I/O and M3 MTP)**:
     **`Qwen3.8-Flash-Next` (`qwen4_exp`)** — full 54.7 GiB wired residency on
     128 GB+ Max / Ultra machines, and hybrid wired hot-expert + NVMe cold
     expert / NVMe 51B PLE execution on the 24 GB M4 baseline driven by ANE
     expert prediction and E-core `F_NOCACHE` async prefetch.
   - **Tier 3 (M7 / Future scale-up & scale-out)**:
     **`DeepSeek-V4.1-Flash-1017` (`deepseek41`)** — targeted explicitly at
     **512 GB Ultra single nodes** (241.5 GiB 3-bit backbone wired on an
     800 GB/s bus) and **multi-Mac 128–256 GB Thunderbolt 5 RDMA expert-parallel
     clusters** (`ExpertSource.thunderbolt_peer`,
     [ADR-0003](0003-expert-source-tagged-union.md)). Running the 552B + 196B
     checkpoint on a 24 GB M4 / 500 GB SSD (~7 GiB/token cold reads, < 0.7
     tok/s) is explicitly non-viable and out of scope for the 24 GB baseline.
3. **Preserve M3–M5 delivery order.** Current speculative decoding (M3), ANE
   wavefront prefill + NVMe cold-expert async I/O (M4), and kqueue SSE serving
   (M5) remain prerequisites on `Qwen3-30B-A3B` before Tier 2/3 operators land.

## Consequences

- Five shared operator families (512+ expert MoE + shared expert, sparse
  NVMe `F_NOCACHE` PLE/Engram lookup, Gated DeltaNet recurrent state, sparse
  attention block indexers, and multi-branch residual mixing) serve both
  frontier families without duplicating scheduler or storage machinery.
- `.amf` will gain a sparse-table tensor kind / sharded pack convention for 51B
  PLE and 196B Engram lookup tables without breaking v0 16 KiB page alignment.
- Capability status for every new operator and model tier is tracked in
  [../status.md](../status.md); milestones M6 and M7 live in
  [../process.md](../process.md).

## Evidence

- M2 baseline measurements ([../benchmarks/results.md](../benchmarks/results.md))
  confirm `F_NOCACHE` `.amf` loading, multi-shard `WiredArena` support past the
  `13,639 MiB` single-`MTLBuffer` limit, IOSurface `KvTile` GQA at `85.6 GB/s`,
  and deduplicated MoE gather saving up to `62.5%` expert weight traffic.

## Post-release empirical note (2026-11-25)

Live Hugging Face `config.json` and `model.safetensors.index.json` verification
(`cd72273` and side-branch audit `ff7b26fc`) refined both pre-release parameter
estimates above:

- **`Qwen/Qwen3.8-Flash-Next` (`qwen4_exp`)**: actual published geometry
  (`hidden_size = 2560`, `512` routed experts `top_k = 10`,
  `moe_intermediate_size = 640`) activates `2.36B` routed parameters per token
  (vs the pre-release `6B` estimate) and introduces `300` long
  `hyper_connection` tensor names exceeding `.amf`'s `47`-byte limit, full
  `hc_count = 4` mixer projections, separate dense PLE conditioning tensors vs
  `128` external N-gram shards, and 3D stacked MoE / QSA indexer / MTP keys.
- **`deepseek-ai/DeepSeek-V4.1-Flash-1017` (`deepseek41`)**: actual published
  geometry (`hidden_size = 5120`, `384` routed experts `top_k = 6` + `1` shared
  expert, `20` encoder + `20` decoder layers) activates only `4.25B` routed
  parameters per token during Decoder decode (vs the pre-release `hidden = 7168`,
  `512E`, `top_k = 16` -> `16B/tok` estimate). On a `24 GB` M4, holding only
  Decoder hot experts wired leaves `~11.7–13.3 GiB` for the hot-expert cache,
  making cold-NVMe Decoder tiering viable (`~2.5–3.0 tok/s` lossless FP4 or
  `~6–9 tok/s` `Q2_K` + speculative dedup) on `3.2–3.8 GB/s` internal or
  `40 Gb/s` TB4 PCIe NVMe (`--io-backend pread`) once real-HF alignment lands;
  full wired residency still targets `512 GB` Ultra or multi-Mac TB5 RDMA
  clusters.

Full real-HF schema, safetensors-header pre-flight gating, and `24 GB` M4
cold-tier parity for both models are scheduled as Milestone **`M9.4`** (`W28`),
tracked in [../status.md](../status.md) and
[../process.md](../process.md#milestones).

## Open items

- Resolved in M6/M7 (`C10`, `R7`, `G11`): PLE / Engram `.amf` shard layout,
  E-core prefetch batching, and FP4 (`E2M1` + `E4M3` scale) KV cache
  representation.
- Scheduled in `M9.4` (`W28`): `Qwen3.8-Flash-Next` (`4` items) and
  `DeepSeek-V4.1-Flash-1017` (`3` items) real Hugging Face safetensors-header
  and `24 GB` M4 cold-tier alignment before running full published HF shards.
