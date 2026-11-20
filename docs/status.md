# Capability status

The single source of truth for what aminfer can do today, what is decided but
not built, and what is planned. Other documents link here instead of restating
status ([ADR-0007](decisions/0007-docs-single-source-of-truth.md)). Progress
and milestones live in [process.md](process.md).

Last updated: 2026-11-20.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| **Implemented** | In the repository and covered by `zig build test` or a documented check. |
| **Confirmed** | Decided (accepted ADR or project charter); not implemented yet. |
| **Draft** | Sketched; needs a spike or user confirmation before it becomes Confirmed. |
| **Undecided** | Open question; the options are listed in the referenced document. |
| **Future** | Long-term plan; at most an interface is reserved. |

Milestones (M0–M5) are defined in [process.md](process.md#milestones).

## Foundation

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| F1 | `build.zig` graph: Zig core + ObjC bridge (translate-c) + embedded `.metallib` + tests | Implemented | M0 | [ADR-0001](decisions/0001-language-and-toolchain-boundaries.md) |
| F2 | 16 KiB page-aligned wired regions (`mmap` + `mlock`) with boundary checks | Implemented | M0 | [ADR-0002](decisions/0002-page-alignment-and-zero-copy.md) |
| F3 | Platform probe: Metal device and Neural Engine facts (`aminfer probe`) | Implemented | M0 | [hardware.md](architecture/hardware.md) |
| F4 | `ExpertSource` tagged union (`thunderbolt_peer` reserved) | Implemented | M0 | [ADR-0003](decisions/0003-expert-source-tagged-union.md) |
| F5 | Zero-copy `MTLBuffer` over a wired region (`newBufferWithBytesNoCopy`) | Implemented | M0 | [ADR-0002](decisions/0002-page-alignment-and-zero-copy.md) |
| F6 | Metal 4 command path (reusable allocator and command buffer, residency set, shared-event completion) | Implemented (M0 smoke; M1 hot path Confirmed) | M0 smoke, M1 hot path | [ADR-0011](decisions/0011-metal4-zero-allocation-hot-path.md) |
| F7 | uv-managed offline tools scaffold | Implemented | M0 | [ADR-0001](decisions/0001-language-and-toolchain-boundaries.md) |
| F8 | `justfile` shortcuts (`sysctl`, `uv`, `powermetrics`) | Confirmed | M0 | [system-tuning.md](guides/system-tuning.md) |
| F9 | Platform probe: sysctl facts (page size, memory, core counts, wire limits) | Implemented | M0 | [hardware.md](architecture/hardware.md) |

## Memory and residency

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| R1 | Wired budget computed at startup (default headroom ≥ 3 GiB; refuse plans over budget) | Confirmed | M1 | [ADR-0012](decisions/0012-wired-memory-budget-headroom.md) |
| R2 | GPU residency through `MTLResidencySet` | Confirmed | M1 | [memory.md](architecture/memory.md) |
| R3 | KV tiles: IOSurface (fp16 `OneComponent16Half`, 64 B row stride) viewed as `MTLTexture` | Confirmed | M2 | [ADR-0004](decisions/0004-kv-cache-iosurface-texture-sharing.md) |
| R4 | KV tile granularity (one 512-token chunk per layer) and token- vs channel-major rows | Draft | M2 (spike S1) | [memory.md](architecture/memory.md#kv-tiles) |
| R5 | int8 KV tiles (`OneComponent8`) | Undecided | — | [ADR-0004](decisions/0004-kv-cache-iosurface-texture-sharing.md) |

## Model, weights and format

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| W1 | `.amf` format: 16 KiB-aligned header, directory and blobs | Confirmed (layout Draft) | M1 | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md), [experts-and-storage.md](architecture/experts-and-storage.md#amf-format) |
| W2 | `.amf` writer in `tools/` | Confirmed | M1 | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md) |
| W3 | `.amf` reader and startup validation in Zig | Confirmed | M1 | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md) |
| W4 | Import MLX community quantized safetensors (affine, group 64) as the conversion source | Draft | M1 | [experts-and-storage.md](architecture/experts-and-storage.md#amf-format) |
| W5 | Native Zig tokenizer (Qwen BPE) | Draft | M2 | [runtime-pipeline.md](architecture/runtime-pipeline.md) |
| W6 | Primary target Qwen3-30B-A3B (3-bit); secondary Qwen3-14B (4-bit, dense) | Draft | M2 | [memory.md](architecture/memory.md#budget-estimates) |
| W7 | CoreML `.mlpackage` export for ANE graphs (draft model, attention blocks, expert predictor) | Confirmed | M4 | [ADR-0010](decisions/0010-hybrid-wavefront-prefill.md) |
| W8 | Offline expert heat statistics | Confirmed | M4 | [experts-and-storage.md](architecture/experts-and-storage.md) |

## GPU (L3)

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| G1 | Fused 3/4-bit dequantization GEMV / GEMM kernels | Confirmed | M1 | [overview.md](architecture/overview.md) |
| G2 | Compile-time specialization (Zig `comptime`; Metal function constants on the GPU side) | Confirmed (GPU mechanism Draft) | M1 | [ADR-0014](decisions/0014-compile-time-specialization.md) |
| G3 | MoE gather / grouped GEMM with experts deduplicated across verified tokens | Confirmed | M2 | [ADR-0009](decisions/0009-expert-byte-budget-tree-pruning.md) |
| G4 | Decode attention over KV tiles | Confirmed | M2 | [ADR-0004](decisions/0004-kv-cache-iosurface-texture-sharing.md) |
| G5 | Tree-attention verification with ancestor mask and accepted-path KV commit | Confirmed | M3 | [speculative-decoding.md](architecture/speculative-decoding.md) |
| G6 | GPU half of the hybrid prefill (FFN / MoE) | Confirmed | M4 | [ADR-0010](decisions/0010-hybrid-wavefront-prefill.md) |

## Speculative decoding (L1 / L2)

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| D1 | L1 suffix automaton, preallocated, zero allocation after startup | Confirmed | M3 | [speculative-decoding.md](architecture/speculative-decoding.md#l1-suffix-automaton) |
| D2 | EAGLE-2 dynamic draft tree with EAGLE-3 and EAGLE-2 head formats | Confirmed | M3 | [ADR-0013](decisions/0013-eagle-tree-and-head-formats.md) |
| D3 | MTP heads | Confirmed | M3 | [ADR-0013](decisions/0013-eagle-tree-and-head-formats.md) |
| D4 | Same-family small draft model (≈0.5B class, e.g. Qwen3-0.6B) for models without EAGLE / MTP | Confirmed | M4 | [ADR-0013](decisions/0013-eagle-tree-and-head-formats.md) |
| D5 | Benchmark-gated placement of EAGLE / MTP heads (ANE vs GPU) | Confirmed | M3 | [ADR-0008](decisions/0008-benchmark-gated-draft-placement.md) |
| D6 | Dual-source tree merging (L1 seed prefix ≥ 4 tokens → L2) | Confirmed | M3 | [ADR-0005](decisions/0005-dual-source-tree-merging.md) |
| D7 | Expert-byte-budget tree pruning (dense models: node budget) | Confirmed | M3 | [ADR-0009](decisions/0009-expert-byte-budget-tree-pruning.md) |
| D8 | PEARL-style pipelined drafting for the independent small draft | Confirmed (design Draft) | M5 | [ADR-0008](decisions/0008-benchmark-gated-draft-placement.md) |
| D9 | Lossless stochastic acceptance for temperature > 0 | Confirmed | M3 | [speculative-decoding.md](architecture/speculative-decoding.md) |

## Neural Engine

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| A1 | Hybrid wavefront prefill: ANE attention blocks write K/V into IOSurface tiles | Confirmed | M4 | [ADR-0010](decisions/0010-hybrid-wavefront-prefill.md) |
| A2 | ANE cross-layer MoE expert prediction feeding prefetch and pruning | Confirmed (predictor design Undecided) | M4 | [experts-and-storage.md](architecture/experts-and-storage.md#expert-prediction-and-prefetch) |
| A3 | Static-shape strategy for growing KV length (buckets or multifunction models) | Undecided | M4 (spike S6) | [ADR-0010](decisions/0010-hybrid-wavefront-prefill.md) |

## CPU runtime, scheduling and storage

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| C1 | Lock-free SPSC / MPSC rings with preallocated slots | Confirmed | M2 | [runtime-pipeline.md](architecture/runtime-pipeline.md#threading-model) |
| C2 | QoS-based core steering (no hard affinity on Apple Silicon) and I/O policy for prefetch threads | Draft | M2 | [hardware.md](architecture/hardware.md#core-steering) |
| C3 | Cold experts: `F_NOCACHE` + `aio_read` + `kqueue` (`EVFILT_AIO`) into 16 KiB slots | Confirmed | M4 | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md) |
| C4 | E-core `pread` worker fallback when kernel AIO concurrency limits throughput | Draft | M4 (spike S3) | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md) |
| C5 | Hot / cold expert cache policy | Draft | M4 | [experts-and-storage.md](architecture/experts-and-storage.md) |
| C6 | N-gram tables on NVMe for L1 | Draft | M5 | [experts-and-storage.md](architecture/experts-and-storage.md#n-gram-tables-on-nvme) |
| C7 | kqueue SSE server | Confirmed | M5 | [runtime-pipeline.md](architecture/runtime-pipeline.md) |
| C8 | Server API shape (for example OpenAI-compatible streaming) | Undecided | M5 | [runtime-pipeline.md](architecture/runtime-pipeline.md) |
| C9 | Benchmark harness with `powermetrics` and swap checks | Draft | M1 | [system-tuning.md](guides/system-tuning.md) |

## Scale-out

| ID | Capability | Status | Target | References |
| --- | --- | --- | --- | --- |
| X1 | Thunderbolt-peer expert source (TB4 TCP; TB5 RDMA on macOS 26.2+) | Future (interface reserved) | — | [ADR-0003](decisions/0003-expert-source-tagged-union.md) |
| X2 | Multi-Mac expert-parallel inference | Future | — | [experts-and-storage.md](architecture/experts-and-storage.md#thunderbolt-peer) |
