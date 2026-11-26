# aminfer

A bare-metal, ultra-low-latency LLM inference engine for Apple Silicon.

> What works today is listed in [docs/status.md](docs/status.md); milestones,
> verification history, and current focus live in
> [docs/process.md](docs/process.md); measured benchmarks and raw outputs live
> in [docs/benchmarks/results.md](docs/benchmarks/results.md).

aminfer targets M4 / M5-class Apple Silicon. The baseline is a 24 GB M4, and
the design scales to 128 GB+ Max / Ultra machines and, later, to Macs joined
over Thunderbolt. It runs CPU P-cores (including ARMv9 SME2), CPU E-cores, the
Neural Engine, and the Metal 4 GPU concurrently on one zero-copy memory pool:

| Engine | Role |
| --- | --- |
| CPU P-cores + SME2 | L1 zero-allocation suffix-automaton drafting, lock-free scheduling, L2-resident routers / adapters & `ZT0`-LUT / `ZA` microkernels |
| CPU E-cores | Cold-expert and N-gram prefetch from NVMe (`F_NOCACHE`, `aio_read` + `kqueue`) |
| Neural Engine | Attention half of chunked wavefront prefill, cross-layer MoE expert prediction, gated L2 drafting |
| GPU (Metal 4) | L3 fused `q2`/`q3`/`q4` dequantization & native FP16 / BF16 compute, CSR-dedup MoE gather and tree-attention verification |
| Storage | 16 KiB-aligned wired hot experts; NVMe cold experts (`.amf`), `.ang` N-gram tables, and sparse PLE / Engram tables |

Speculative decoding prioritizes EAGLE-2 draft trees (with EAGLE-3 or EAGLE-2
heads) and MTP heads, falls back to a same-family small draft model, and
prices MoE draft trees by the expert bytes they make the GPU read. Beyond the
initial `Qwen3-30B-A3B` / `Qwen3-14B` baselines, the tiered model roadmap
targets **`Qwen3.8-Flash-Next`** (`qwen4_exp`: Gated DeltaNet + QSA + 512-expert
MoE + 51B NVMe PLE) and **`DeepSeek-V4.1-Flash-1017`** (`deepseek41`: 552B
Causal Encoder-Decoder MoE + 196B Engram + CSA2/FP4 KV on 512 GB Ultra and
Thunderbolt 5 RDMA clusters); see
[docs/architecture/model-roadmap.md](docs/architecture/model-roadmap.md). The
full design starts at
[docs/architecture/overview.md](docs/architecture/overview.md).

## Technology

- **Engine core** — Zig 0.17.1 + standalone ARMv9.2 SME2 `.s` microkernels:
  scheduler, memory pools, lock-free rings, suffix automaton, async I/O, zero-alloc
  kqueue HTTP/HTTPS SSE server (OpenAI Chat Completions / FIM Completions /
  Stateful Responses / Anthropic Messages & Models, CORS, Base64 multimodal
  vision ingress, universal multi-LoRA & DoRA).
- **Hardware bridge** — pure Objective-C (ARC) behind the C header
  `apple_bridge.h`, used only to call Metal, CoreML, IOSurface, Network,
  Security and Foundation.
- **GPU kernels** — Metal Shading Language, compiled with `xcrun metal`.
- **Offline tools** — Python managed by uv, in `tools/` only. No Python or
  PyTorch at runtime.
- **Build** — `build.zig` is the single build hub.

Requirements: Apple Silicon, macOS 26.0 or later, Zig 0.17.1, Xcode with the
Metal Toolchain component. Setup and build commands:
[docs/guides/development-setup.md](docs/guides/development-setup.md). Raising
the wired-memory limits: [docs/guides/system-tuning.md](docs/guides/system-tuning.md).

## Documentation

| Path | Contents |
| --- | --- |
| [docs/architecture/](docs/architecture/overview.md) | Architecture design |
| [docs/benchmarks/](docs/benchmarks/results.md) | Benchmark methodology, metrics and raw telemetry |
| [docs/decisions/](docs/decisions/README.md) | Decision records (ADRs) |
| [docs/guides/](docs/guides/development-setup.md) | Build and environment guides |
| [docs/status.md](docs/status.md) | Implemented, unimplemented and planned capabilities |
| [docs/process.md](docs/process.md) | Milestones and progress |
| [AGENTS.md](AGENTS.md) | Conventions for coding agents |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
