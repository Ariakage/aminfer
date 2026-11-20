# Development setup

> The build graph lands during M0; until then the build commands below are the
> target workflow. See [../process.md](../process.md).

## Requirements

| Tool | Version | Notes |
| --- | --- | --- |
| macOS | 26.0 or later (developed on 27.2) | Metal 4 floor ([ADR-0011](../decisions/0011-metal4-zero-allocation-hot-path.md)) |
| Hardware | Apple Silicon (baseline: M4, 24 GB) | 16 KiB pages |
| Zig | 0.17.1 | `brew install zig`; `@cImport` no longer exists — the bridge header is translated by `build.zig` |
| Xcode | 27.1 tested | Needs the Metal Toolchain component, below |
| uv | 0.11 or later | Offline tools in `tools/` only |
| just | optional | Shortcuts only; `build.zig` stays the single build hub |

## Metal Toolchain

Recent Xcode releases ship the Metal compiler as a separate component. Without
it, `xcrun metal` fails with "missing Metal Toolchain".

```sh
xcodebuild -showComponent MetalToolchain      # Status: installed?
xcodebuild -downloadComponent MetalToolchain  # about 840 MB
xcrun -sdk macosx metal --version
```

## Build and test

```sh
zig build                              # Debug build
zig build test                         # full test suite (Debug)
zig build test -Doptimize=ReleaseSafe  # safety and alignment checks kept on
zig build -Doptimize=ReleaseFast       # benchmarks only
```

Policy ([../../AGENTS.md](../../AGENTS.md)): after every code change,
`zig build` must compile cleanly and `zig build test` must pass. Changes that
touch memory alignment or GPU kernels are verified in Debug and ReleaseSafe
before any ReleaseFast benchmark.

Tests create a Metal device and run kernels, so they need a real Apple Silicon
GPU; they do not run in headless CI without one.

## Offline tools

`tools/` is a uv project. Python never runs inside the engine.

```sh
cd tools
uv sync
uv run --help
```

## Repository index

Agents orient through the local ikat index; see the "ikat repository index"
section of [../../AGENTS.md](../../AGENTS.md).
