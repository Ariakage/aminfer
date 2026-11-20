# aminfer

A bare-metal, ultra-low-latency LLM inference engine for Apple Silicon.

> Early development — aminfer does not run a model yet. What works today is
> listed in [docs/status.md](docs/status.md); progress is in
> [docs/process.md](docs/process.md).

aminfer targets M4 / M5-class Apple Silicon. The baseline is a 24 GB M4, and
the design scales to 128 GB+ Max / Ultra machines and, later, to Macs joined
over Thunderbolt. Its core idea is to break through the unified-memory wall by
running every engine of the SoC concurrently on one zero-copy memory pool:

| Engine | Role |
| --- | --- |
| CPU P-cores | L1 zero-allocation suffix-automaton drafting, lock-free scheduling |
| CPU E-cores | Cold-expert prefetch from NVMe (`F_NOCACHE`, `aio_read` + `kqueue`) |
| Neural Engine | L2 drafting, attention half of the chunked prefill, MoE expert prediction |
| GPU (Metal) | L3 fused 3/4-bit dequantization, MoE gather and tree-attention verification |
| Storage | 16 KiB-aligned wired hot experts; NVMe cold experts (`.amf`) and N-gram tables |

Speculative decoding prioritizes EAGLE-2 draft trees (with EAGLE-3 or EAGLE-2
heads) and MTP heads, falls back to a same-family small draft model, and
prices MoE draft trees by the expert bytes they make the GPU read. The full
design starts at
[docs/architecture/overview.md](docs/architecture/overview.md).

## Technology

- **Engine core** — Zig 0.17.1: scheduler, memory pools, lock-free rings,
  suffix automaton, async I/O, kqueue SSE server.
- **Hardware bridge** — pure Objective-C (ARC) behind the C header
  `apple_bridge.h`, used only to call Metal, CoreML, IOSurface and Foundation.
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
