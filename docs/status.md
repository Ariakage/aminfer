# Capability status

The single source of truth for what aminfer can do today, what is decided but
not built, and what is planned. Other documents link here instead of restating
status ([ADR-0007](decisions/0007-docs-single-source-of-truth.md)). Progress
and milestones live in [process.md](process.md).

Last updated: 2026-11-22.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| **Implemented** | In the repository and covered by `zig build test` or a documented check. |
| **Confirmed** | Decided (accepted ADR or project charter); not implemented yet. |
| **Draft** | Sketched; needs a spike or user confirmation before it becomes Confirmed. |
| **Undecided** | Open question; the options are listed in the referenced document. |
| **Future** | Long-term plan; at most an interface is reserved. |

Milestones (M0–M7, including M2.5, M4.5 and M6.5) are defined in [process.md](process.md#milestones).

## Foundation

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| F1 | `build.zig` graph: Zig core + ObjC bridge (translate-c) + embedded `.metallib` + tests | Implemented | M0 | [ADR-0001](decisions/0001-language-and-toolchain-boundaries.md) |
| F2 | 16 KiB page-aligned wired regions (`mmap` + `mlock`) with boundary checks | Implemented | M0 | [ADR-0002](decisions/0002-page-alignment-and-zero-copy.md) |
| F3 | Platform probe: Metal device and Neural Engine facts (`aminfer probe`) | Implemented | M0 | [hardware.md](architecture/hardware.md) |
| F4 | `ExpertSource` tagged union (`thunderbolt_peer` reserved) | Implemented | M0 | [ADR-0003](decisions/0003-expert-source-tagged-union.md) |
| F5 | Zero-copy `MTLBuffer` over a wired region (`newBufferWithBytesNoCopy`) | Implemented | M0 | [ADR-0002](decisions/0002-page-alignment-and-zero-copy.md) |
| F6 | Metal 4 command path (reusable allocator and command buffer, residency set, shared-event completion) | Implemented | M0 smoke, M1 hot path | [ADR-0011](decisions/0011-metal4-zero-allocation-hot-path.md) |
| F7 | uv-managed offline tools scaffold | Implemented | M0 | [ADR-0001](decisions/0001-language-and-toolchain-boundaries.md) |
| F8 | `justfile` shortcuts (`sysctl`, `uv`, `powermetrics`) | Implemented | M0 | [system-tuning.md](guides/system-tuning.md) |
| F9 | Platform probe: sysctl facts (page size, memory, core counts, wire limits) | Implemented | M0 | [hardware.md](architecture/hardware.md) |

## Memory and residency

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| R1 | Wired budget computed at startup (default headroom ≥ 3 GiB; refuse plans over budget) | Implemented | M1 | [ADR-0012](decisions/0012-wired-memory-budget-headroom.md) |
| R2 | GPU residency through `MTLResidencySet` | Implemented | M1 | [memory.md](architecture/memory.md) |
| R3 | KV tiles: IOSurface (fp16 `OneComponent16Half`, 64 B row stride) viewed as `MTLTexture` | Implemented | M2 | [ADR-0004](decisions/0004-kv-cache-iosurface-texture-sharing.md) |
| R4 | KV tile granularity (one 512-token chunk per layer) and token-major `RGBA16Float` rows | Implemented | M2 (spike S1) | [memory.md](architecture/memory.md#kv-tiles), [results.md](benchmarks/results.md) |
| R5 | int8 KV tiles (`OneComponent8` declined for ANE↔GPU IOSurface; keep `OneComponent16Half` / `RGBA16Float`; GPU-side group-64 `q8` `MTLBuffer` tier for retired 32K+ chunks) | Confirmed | M6.5 | [ADR-0004](decisions/0004-kv-cache-iosurface-texture-sharing.md), [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md), [memory.md](architecture/memory.md) |
| R6 | Fixed-size recurrent state arenas (plus `S_tmp` loop-isolation scratch) in `WiredArena` for Gated DeltaNet layers (3:1 hybrid with `KvTile`s) | Confirmed | M6 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md), [model-roadmap.md](architecture/model-roadmap.md) |
| R7 | MLA + CSA2 (`Full` / `Reindex` / `Reuse`) cross-layer KV sharing, FP4 KV (`E2M1` + `E4M3` scale, 890 B/tok) and SWA bounded replay | Future | M7 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [model-roadmap.md](architecture/model-roadmap.md) |
| R8 | Shared prefix `KvTile` (copy-on-write tail tiles) and snapshot-cloned `GDN` recurrent-state cache across concurrent requests | Confirmed | M6.5 | [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md), [memory.md](architecture/memory.md), [runtime-pipeline.md](architecture/runtime-pipeline.md) |

## Model, weights and format

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| W1 | `.amf` format: 16 KiB-aligned header, directory and blobs | Implemented | M1 | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md), [experts-and-storage.md](architecture/experts-and-storage.md#amf-format) |
| W2 | `.amf` writer in `tools/` | Implemented | M1 | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md) |
| W3 | `.amf` reader and startup validation in Zig | Implemented | M1 | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md) |
| W4 | Import MLX community quantized safetensors (affine, group 64) as the conversion source | Implemented | M1 | [experts-and-storage.md](architecture/experts-and-storage.md#amf-format) |
| W5 | Native Zig tokenizer (Qwen BPE) | Implemented | M2 | [runtime-pipeline.md](architecture/runtime-pipeline.md) |
| W6 | Primary target Qwen3-30B-A3B (3-bit); secondary Qwen3-14B (4-bit, dense) | Implemented | M2 | [memory.md](architecture/memory.md#budget-estimates), [results.md](benchmarks/results.md) |
| W7 | CoreML `.mlpackage` export for ANE graphs (draft model, attention blocks, expert predictor) | Implemented | M4 | [ADR-0010](decisions/0010-hybrid-wavefront-prefill.md), [results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft) |
| W8 | Offline expert heat statistics | Implemented | M4 | [experts-and-storage.md](architecture/experts-and-storage.md), [results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft) |
| W9 | Composable model operator registry (`BackboneKind`, `AttentionKind`, `FfnKind`, `ResidualKind`, `MemoryLookupKind`, `LoopSchedule`) | Confirmed | M6 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md), [model-roadmap.md](architecture/model-roadmap.md) |
| W10 | Tier 2 target: **`Qwen3.8-Flash-Next`** (`qwen4_exp`, 125B/A6B + 51B PLE + 4B MTP) `.amf` converter (vision-tower strip, sharded PLE) and runtime | Confirmed | M6 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [model-roadmap.md](architecture/model-roadmap.md) |
| W11 | Tier 3 target: **`DeepSeek-V4.1-Flash-1017`** (`deepseek41`, 552B CED MoE + 196B Engram) `.amf` converter and Causal Encoder-Decoder scheduler (512 GB Ultra / TB5 cluster) | Future | M7 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [model-roadmap.md](architecture/model-roadmap.md) |
| W12 | Multi-shard Hugging Face / MLX safetensors (`model.safetensors.index.json`) streaming import and `.amf` `bf16` (`DType.bf16 = 4`) weight support | Implemented | M2.5 | [ADR-0016](decisions/0016-native-fp16-bf16-and-multishard-import.md), [experts-and-storage.md](architecture/experts-and-storage.md#amf-format), [results.md](benchmarks/results.md) |
| W13 | Standard unquantized and mixed-precision `FP16` / `BF16` model execution (`tie_word_embeddings`, per-tensor dtype dispatch, `RGBA16Float` `KvTile` boundary; verified on `Qwen3-0.6B` BF16, `Qwen3-14B-4bit`, and `Qwen3-30B-A3B-3bit`) | Implemented | M2.5 | [ADR-0016](decisions/0016-native-fp16-bf16-and-multishard-import.md), [memory.md](architecture/memory.md#kv-tiles), [results.md](benchmarks/results.md) |
| W14 | Optional offline Python GSQ + RCO sensitivity & residual-compensation `.amf` exporter with static per-tensor / per-expert `QuantScheme` tags (`q2`/`q3`/`q4` affine + `f16`/`bf16`; `q6`/`q8` deferred post-M5) | Implemented | M5 | [ADR-0017](decisions/0017-optional-gsq-rco-hybrid-quantization.md), [experts-and-storage.md](architecture/experts-and-storage.md), [results.md](benchmarks/results.md#m5-serving-pipelining-c6-nvme-n-gram-and-adr-0017-gsqrco-hybrid-quantization) |
| W15 | `sme2_panel64` K-interleaved panel pre-packer with inline group metadata and binary-compatible `TensorEntry.reserved0` single-owner layout tag | Implemented | M4.5 | [ADR-0018](decisions/0018-bare-metal-armv9-sme2-engine.md), [results.md](benchmarks/results.md#m45-bare-metal-armv9-sme2-engine-spike-s7-c13-bus-contention-and-d11-trng) |
| W16 | Native recurrent-depth (`Prelude`/`Core`/`Coda`) `.amf` importer and offline Gated Loop Adapter (`Step-LoRA` + zero-init scalar gate `α_r` in L2-resident `sme2_panel64`) | Confirmed | M6 | [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md), [model-roadmap.md](architecture/model-roadmap.md) |
| W17 | Large-MoE heat-guided GSQ + RCO scaling (`Qwen3-30B-A3B` & 512-expert `Qwen3.8`) and `.amf` `QuantParamKind = 1` non-uniform 16-centroid `ZT0` LUT packing | Confirmed | M6 | [ADR-0017](decisions/0017-optional-gsq-rco-hybrid-quantization.md), [ADR-0018](decisions/0018-bare-metal-armv9-sme2-engine.md), [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md) |

## GPU (L3)

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| G1 | Fused 3/4-bit dequantization GEMV / GEMM kernels | Implemented | M1 | [overview.md](architecture/overview.md) |
| G2 | Compile-time specialization (Zig `comptime`; Metal function constants on the GPU side) | Implemented | M1 | [ADR-0014](decisions/0014-compile-time-specialization.md) |
| G3 | MoE gather / grouped GEMM with experts deduplicated across verified tokens | Implemented | M2 | [ADR-0009](decisions/0009-expert-byte-budget-tree-pruning.md), [results.md](benchmarks/results.md) |
| G4 | Decode attention over KV tiles | Implemented | M2 | [ADR-0004](decisions/0004-kv-cache-iosurface-texture-sharing.md), [results.md](benchmarks/results.md) |
| G5 | Tree-attention verification with ancestor mask and accepted-path KV commit | Implemented | M3 | [speculative-decoding.md](architecture/speculative-decoding.md), [results.md](benchmarks/results.md) |
| G6 | GPU half of the hybrid prefill (FFN / MoE) | Implemented | M4 | [ADR-0010](decisions/0010-hybrid-wavefront-prefill.md), [results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft) |
| G7 | High-expert-count MoE (`>= 512` routed experts, top-10+) with unconditional `shared_expert` SwiGLU accumulation | Confirmed | M6 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [model-roadmap.md](architecture/model-roadmap.md) |
| G8 | Gated DeltaNet (GDN) chunked-prefill and recurrent decode Metal 4 kernels (with `S_tmp` loop-isolation commit) | Confirmed | M6 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md), [model-roadmap.md](architecture/model-roadmap.md) |
| G9 | Qwen Sparse Attention (QSA) micro-block indexer with YaRN, interleaved MRoPE (`[11, 11, 10]`) and cross-loop index reuse | Confirmed | M6 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md), [model-roadmap.md](architecture/model-roadmap.md) |
| G10 | Multi-branch residual stream kernels: 4-branch Gated Residual (GR) and Single-Pass mHC (Mega-mHC) | Confirmed (GR: M6) / Future (mHC: M7) | M6–M7 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [model-roadmap.md](architecture/model-roadmap.md) |
| G11 | MLA + CSA2 hierarchical sparse indexer and FP4 (`E2M1` + `E4M3` scale) KV encode/decode kernels | Future | M7 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [model-roadmap.md](architecture/model-roadmap.md) |
| G12 | Native `FP16` (`half4` / `simdgroup_half8x8`) and `BF16` (`bfloat4` / `simdgroup_bfloat8x8`) dense & MoE GEMV / GEMM / SwiGLU Metal 4 kernels with FP32 accumulation | Implemented | M2.5 | [ADR-0016](decisions/0016-native-fp16-bf16-and-multishard-import.md), [ADR-0014](decisions/0014-compile-time-specialization.md), [results.md](benchmarks/results.md) |
| G13 | Branch-free `comptime` / Metal function-constant per-block `QuantScheme` dispatch (`q2`/`q3`/`q4` affine + `f16`/`bf16`) and scheme-bucketed MoE gather (strict per-tensor / per-expert granularity; zero intra-tensor bit mixing) | Implemented | M5 | [ADR-0017](decisions/0017-optional-gsq-rco-hybrid-quantization.md), [ADR-0014](decisions/0014-compile-time-specialization.md), [results.md](benchmarks/results.md#m5-serving-pipelining-c6-nvme-n-gram-and-adr-0017-gsqrco-hybrid-quantization) |
| G14 | Zero-copy multimodal vision/video ingress (`VideoToolbox` `CVPixelBuffer` → `16 KiB` `IOSurface` → `MTLTexture` / `MLMultiArray` patch encoder into MRoPE) | Confirmed | M6.5 | [ADR-0004](decisions/0004-kv-cache-iosurface-texture-sharing.md), [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md) |

## Speculative decoding (L1 / L2)

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| D1 | L1 suffix automaton, preallocated, zero allocation after startup | Implemented | M3 | [speculative-decoding.md](architecture/speculative-decoding.md#l1-suffix-automaton), [results.md](benchmarks/results.md) |
| D2 | EAGLE-2 dynamic draft tree with EAGLE-3 and EAGLE-2 head formats | Implemented | M3 | [ADR-0013](decisions/0013-eagle-tree-and-head-formats.md), [results.md](benchmarks/results.md) |
| D3 | MTP heads | Implemented | M3 | [ADR-0013](decisions/0013-eagle-tree-and-head-formats.md) |
| D4 | Same-family small draft model (≈0.5B class, e.g. Qwen3-0.6B) for models without EAGLE / MTP | Implemented | M4 | [ADR-0013](decisions/0013-eagle-tree-and-head-formats.md), [results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft) |
| D5 | Benchmark-gated placement of EAGLE / MTP heads (ANE vs GPU) | Implemented | M3 | [ADR-0008](decisions/0008-benchmark-gated-draft-placement.md), [results.md](benchmarks/results.md) |
| D6 | Dual-source tree merging (L1 seed prefix ≥ 4 tokens → L2) | Implemented | M3 | [ADR-0005](decisions/0005-dual-source-tree-merging.md), [results.md](benchmarks/results.md) |
| D7 | Expert-byte-budget tree pruning (dense models: node budget) | Implemented | M3 | [ADR-0009](decisions/0009-expert-byte-budget-tree-pruning.md), [results.md](benchmarks/results.md) |
| D8 | PEARL-style pipelined drafting for the independent small draft (target-latency-gated for high-latency targets such as `Qwen3-14B`) | Implemented | M5 | [ADR-0008](decisions/0008-benchmark-gated-draft-placement.md), [results.md](benchmarks/results.md#m5-serving-pipelining-c6-nvme-n-gram-and-adr-0017-gsqrco-hybrid-quantization) |
| D9 | Lossless stochastic acceptance for temperature > 0 | Implemented | M3 | [speculative-decoding.md](architecture/speculative-decoding.md), [results.md](benchmarks/results.md) |
| D10 | DSpark semi-autoregressive speculative decoding adapter under `DraftSource` | Future | M7 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [model-roadmap.md](architecture/model-roadmap.md) |
| D11 | Two-tier per-request sampling RNG: Apple Silicon Secure Enclave TRNG (`getentropy`) initial 64-bit seed → slot-local `Xoshiro256` + `seed_used` echo | Implemented | M4.5 | [speculative-decoding.md](architecture/speculative-decoding.md#sampling-rng-and-hardware-entropy-seeding), [runtime-pipeline.md](architecture/runtime-pipeline.md#request-lifecycle), [results.md](benchmarks/results.md#m45-bare-metal-armv9-sme2-engine-spike-s7-c13-bus-contention-and-d11-trng) |
| D12 | `SelfSpecRecurrent` shallow-draft (`r = 1` / `r = 0`) → full-loop (`Core × R`, `M = 8..16`) batch tree verification, latent `lm_head` elision, and opt-in token-adaptive early exit | Confirmed | M6 | [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md), [speculative-decoding.md](architecture/speculative-decoding.md) |
| D13 | Off-GPU L2-resident ARMv9 SME2 placement (`Placement.cpu_sme2`) for `EAGLE-3` / `4B MTP` draft projection paths and `Step-LoRA` loop adapters | Confirmed | M6 | [ADR-0008](decisions/0008-benchmark-gated-draft-placement.md), [ADR-0018](decisions/0018-bare-metal-armv9-sme2-engine.md), [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md) |

## Neural Engine

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| A1 | Hybrid wavefront prefill: ANE attention blocks write K/V into IOSurface tiles | Implemented | M4 | [ADR-0010](decisions/0010-hybrid-wavefront-prefill.md), [results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft) |
| A2 | ANE cross-layer MoE expert prediction feeding prefetch and pruning | Implemented | M4 | [experts-and-storage.md](architecture/experts-and-storage.md#expert-prediction-and-prefetch), [results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft) |
| A3 | Static-shape strategy for growing KV length (`bucketed_full_ane`, power-of-two KV buckets `512..4096`) | Implemented | M4 (spike S6) | [ADR-0010](decisions/0010-hybrid-wavefront-prefill.md), [results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft) |

## CPU runtime, scheduling and storage

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| C1 | Lock-free SPSC / MPSC rings with preallocated slots | Implemented | M2 | [runtime-pipeline.md](architecture/runtime-pipeline.md#threading-model) |
| C2 | QoS-based core steering (no hard affinity on Apple Silicon) and I/O policy for prefetch threads | Implemented | M2 | [hardware.md](architecture/hardware.md#core-steering) |
| C3 | Cold experts: `F_NOCACHE` + `aio_read` + `kqueue` (`EVFILT_AIO`) into 16 KiB slots | Implemented | M4 | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md), [results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft) |
| C4 | E-core `pread` worker fallback when kernel AIO concurrency limits throughput | Implemented | M4 (spike S3) | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md), [results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft) |
| C5 | Hot / cold expert cache policy (`ExpertSlotPool` + LFU-decay `ExpertCache` with pass pinning and async prefetch) | Implemented | M4 | [experts-and-storage.md](architecture/experts-and-storage.md), [results.md](benchmarks/results.md#m4-neural-engine-hybrid-prefill-cold-expert-nvme-streaming-and-d4-small-draft) |
| C6 | N-gram tables on NVMe for L1 (`.ang` v1 16 KiB-aligned `F_NOCACHE` lookup + wired LRU cache + async prefetch) | Implemented | M5 | [experts-and-storage.md](architecture/experts-and-storage.md#n-gram-tables-on-nvme), [results.md](benchmarks/results.md#m5-serving-pipelining-c6-nvme-n-gram-and-adr-0017-gsqrco-hybrid-quantization) |
| C7 | kqueue SSE server (`EVFILT_READ`/`EVFILT_WRITE`/`EVFILT_USER`, preallocated connection slots, `SO_NOSIGPIPE`, zero steady-state heap allocation) | Implemented | M5 | [runtime-pipeline.md](architecture/runtime-pipeline.md), [results.md](benchmarks/results.md#m5-serving-pipelining-c6-nvme-n-gram-and-adr-0017-gsqrco-hybrid-quantization) |
| C8 | OpenAI-compatible stateless HTTP / SSE server API (`POST /v1/chat/completions` with `delta.reasoning_content` & `seed_used`, `GET /v1/models`, `GET /health`; stateless `/v1/responses` & `/v1/messages`) | Implemented | M5 | [runtime-pipeline.md](architecture/runtime-pipeline.md#request-lifecycle), [results.md](benchmarks/results.md#m5-serving-pipelining-c6-nvme-n-gram-and-adr-0017-gsqrco-hybrid-quantization) |
| C9 | Benchmark harness with `powermetrics` and swap checks (`aminfer bench`, `just bench`) | Implemented | M1 | [system-tuning.md](guides/system-tuning.md) |
| C10 | Sparse external lookup tables on NVMe (`F_NOCACHE` `.amf` row prefetch with `r = 0` single-lookup loop reuse for 51B PLE and 196B Engram) | Confirmed (PLE: M6) / Future (Engram: M7) | M6–M7 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md), [model-roadmap.md](architecture/model-roadmap.md) |
| C11 | Bare-metal SME2 `FEAT_SME2` runtime probe, pre-`smstart` `d8–d15`-safe `.s` harness & canary test, and P-cluster lock-free dispatcher (M1–M3 fallback without `SIGILL`) | Implemented | M4.5 | [ADR-0018](decisions/0018-bare-metal-armv9-sme2-engine.md), [ADR-0001](decisions/0001-language-and-toolchain-boundaries.md), [results.md](benchmarks/results.md#m45-bare-metal-armv9-sme2-engine-spike-s7-c13-bus-contention-and-d11-trng) |
| C12 | 3-way ARMv9.2 `.s` SME2 microkernels (`M=1..4` `ZT0-LUT + Dot`, `M=5..16` `1×4` `ZA`, `M>=32` `2×2` `ZA`), `ld1b` / `ldnt1b` cold-expert cache steering & fused `ZA` epilogue | Implemented | M4.5 | [ADR-0018](decisions/0018-bare-metal-armv9-sme2-engine.md), [ADR-0001](decisions/0001-language-and-toolchain-boundaries.md), [results.md](benchmarks/results.md#m45-bare-metal-armv9-sme2-engine-spike-s7-c13-bus-contention-and-d11-trng) |
| C13 | Spike `S7`-gated heterogeneous CPU SME2 + Metal GPU co-execution (`> 1.05×` net `tok/s` & `0` swap) | Implemented | M4.5 (spike S7) | [ADR-0018](decisions/0018-bare-metal-armv9-sme2-engine.md), [ADR-0008](decisions/0008-benchmark-gated-draft-placement.md), [results.md](benchmarks/results.md#m45-bare-metal-armv9-sme2-engine-spike-s7-c13-bus-contention-and-d11-trng) |
| C14 | Server iteration-level continuous batching (`M > 1` active request slots per GPU / SME2 decode pass) and hot-swappable domain `.ang` NVMe corpus packs | Confirmed | M6.5 | [ADR-0019](decisions/0019-recurrent-depth-and-post-m5-accelerators.md), [runtime-pipeline.md](architecture/runtime-pipeline.md) |

## Scale-out

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| X1 | Thunderbolt-peer expert source (TB4 TCP; TB5 RDMA on macOS 26.2+) | Future (interface reserved) | M7 | [ADR-0003](decisions/0003-expert-source-tagged-union.md) |
| X2 | Multi-Mac expert-parallel inference | Future | M7 | [experts-and-storage.md](architecture/experts-and-storage.md#thunderbolt-peer) |
| X3 | Distributed serving of 500B+ frontier MoEs (`DeepSeek-V4.1-Flash-1017`) across Thunderbolt 5 RDMA Mac clusters | Future | M7 | [ADR-0015](decisions/0015-tiered-frontier-model-roadmap.md), [model-roadmap.md](architecture/model-roadmap.md) |
