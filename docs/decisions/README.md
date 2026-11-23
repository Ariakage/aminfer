# Decision records

Architecture Decision Records (ADRs) for aminfer. Each ADR captures one
decision: its context, the decision itself, consequences, evidence and open
items. ADRs record decisions, not capability status — that lives in
[../status.md](../status.md).

## Status values

| Status | Meaning |
| --- | --- |
| Proposed | Under discussion; not binding. |
| Accepted | Binding until superseded. |
| Superseded | Replaced by a later ADR (linked). |
| Rejected | Considered and declined; kept for the record. |

## Index

| ADR | Title | Status | Date |
| --- | --- | --- | --- |
| [0001](0001-language-and-toolchain-boundaries.md) | Language and toolchain boundaries | Accepted | 2026-11-20 |
| [0002](0002-page-alignment-and-zero-copy.md) | 16 KiB page alignment, wired memory and zero-copy sharing | Accepted | 2026-11-20 |
| [0003](0003-expert-source-tagged-union.md) | `ExpertSource` tagged union and the reserved Thunderbolt peer | Accepted | 2026-11-20 |
| [0004](0004-kv-cache-iosurface-texture-sharing.md) | NPU → GPU KV sharing through IOSurface-backed Metal textures | Accepted | 2026-11-20 |
| [0005](0005-dual-source-tree-merging.md) | Dual-source tree merging of L1 and L2 drafts | Accepted | 2026-11-20 |
| [0006](0006-amf-cold-experts-direct-io.md) | `.amf` cold-expert format and `F_NOCACHE` direct I/O | Accepted | 2026-11-20 |
| [0007](0007-docs-single-source-of-truth.md) | Documentation taxonomy and single sources of truth | Accepted | 2026-11-20 |
| [0008](0008-benchmark-gated-draft-placement.md) | Benchmark-gated placement of L2 drafting | Accepted | 2026-11-20 |
| [0009](0009-expert-byte-budget-tree-pruning.md) | Expert-byte-budget pruning of draft trees | Accepted | 2026-11-20 |
| [0010](0010-hybrid-wavefront-prefill.md) | Hybrid ANE / GPU wavefront prefill | Accepted | 2026-11-20 |
| [0011](0011-metal4-zero-allocation-hot-path.md) | Metal 4 zero-allocation hot path and the macOS 26.0 floor | Accepted | 2026-11-20 |
| [0012](0012-wired-memory-budget-headroom.md) | Wired memory budget with headroom | Accepted | 2026-11-20 |
| [0013](0013-eagle-tree-and-head-formats.md) | EAGLE-2 draft trees with EAGLE-3 / EAGLE-2 heads, MTP and small drafts | Accepted | 2026-11-20 |
| [0014](0014-compile-time-specialization.md) | Compile-time specialization of hot code | Accepted | 2026-11-20 |
| [0015](0015-tiered-frontier-model-roadmap.md) | Tiered frontier model roadmap (`Qwen3.8-Flash-Next` and `DeepSeek-V4.1-Flash-1017`) | Accepted | 2026-11-20 |
| [0016](0016-native-fp16-bf16-and-multishard-import.md) | Native FP16 / BF16 execution and multi-shard safetensors import (M2.5) | Accepted | 2026-11-20 |
| [0017](0017-optional-gsq-rco-hybrid-quantization.md) | Optional offline GSQ + RCO per-expert / per-layer hybrid quantization | Accepted | 2026-11-21 |
| [0018](0018-bare-metal-armv9-sme2-engine.md) | Bare-metal ARMv9 SME2 pre-packed and ZT0-LUT engine (M4.5) | Accepted | 2026-11-21 |
| [0019](0019-recurrent-depth-and-post-m5-accelerators.md) | Composable recurrent depth (`LoopSchedule`), `SelfSpecRecurrent`, and post-M5 heterogeneous accelerators | Accepted | 2026-11-22 |
| [0020](0020-wide-gpu-prefill-tail-pruning-qos-and-live-ram-guards.md) | Configurable wide-chunk GPU prefill, lossless tail pruning, serving QoS, and live non-anonymous RAM admission | Accepted | 2026-11-22 |
| [0021](0021-multi-generation-apple-silicon-and-nax-tensorops.md) | Single-binary multi-generation Apple Silicon adaptation (`Apple9` and `Apple10+` GPU `NAX`) | Accepted | 2026-11-22 |
| [0022](0022-native-c-python-sdk-and-full-protocol-parity.md) | Native C/Python SDK, full HTTP/SSE protocol parity, and rootless terminal monitor (M7.5) | Accepted | 2026-11-23 |
| [0023](0023-native-kernel-barrier-fusion-mma-flashattn-and-simd-accelerators.md) | Native Metal 4 sublayer fusion, MMA FlashAttention-2, GPU indirect MoE dispatch, and NEON/SME2 accelerators (M7.8) | Accepted | 2026-11-23 |
| [0024](0024-quantized-format-parity-and-inkling2-hy4-adaptation.md) | Zero-loss quantized-format compatibility (`FP8` / `MXFP8` / `NVFP4` / `GGUF` / `MLX` / sub-4-bit) and `Inkling 2` + `Hy4` family adaptation (M8) | Accepted | 2026-11-23 |

## Template

```markdown
# ADR-NNNN: Title

- Status: Proposed | Accepted | Superseded by ADR-XXXX | Rejected
- Date: YYYY-MM-DD
- Source: who asked for or approved it, and where

## Context
## Decision
## Consequences
## Evidence
## Open items
```

Open items that need measurement become spikes in
[../process.md](../process.md#spikes).
