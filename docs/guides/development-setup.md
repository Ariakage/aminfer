# Development setup

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
zig build run -- probe                 # print this machine's platform facts
```

Build notes:

- The default target is the native CPU with a deployment floor of macOS 26.0
  (the binary records `minos 26.0`). Overriding the minimum OS version makes
  the target non-native, which disables Zig's own SDK detection, so `build.zig`
  resolves the SDK with `xcrun --sdk macosx --show-sdk-path`.
- Kernels in `src/kernels/` are compiled into one `.metallib` that is embedded
  in the binary; nothing needs to be installed next to it.

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
uv run aminfer-tools --help
uv run python -m unittest discover -s tests
```

## Benchmark & test path configuration

Neither the Zig engine nor `tools/` hardcodes local volume mounts. Real
checkpoint benchmarks, metadata dry-runs, and temporary scratch files resolve
through explicit CLI flags and environment variables (`src/platform/paths.zig`):

| Interface | Resolution Order | Notes |
| --- | --- | --- |
| `--models-dir <dir>` (`--models-dir=<dir>`) | CLI flag -> `AMINFER_MODELS_DIR` -> unset | Supported on `bench-m65`, `bench-m68`, `bench-m75`, `bench-m78`, `bench-m81`, and `bench-m82`; when unset or absent, benchmarks fall back to synthetic fixtures and opportunistic tests skip cleanly |
| `AMINFER_METADATA_DIR` | `AMINFER_METADATA_DIR` -> `<models_dir>/metadata` -> unset | Optional root for offline Hugging Face `config.json` / `tokenizer.json` / `model.safetensors.index.json` metadata tests in `src/model/tokenizer.zig` and `tools/tests/` |
| `TMPDIR` | `TMPDIR` (trailing `/` trimmed) -> `/tmp` | Shared zero-allocation temp-directory root for benchmark probes, cold-expert/N-gram temp stores, and unit tests |

```sh
AMINFER_MODELS_DIR=/path/to/models ./zig-out/bin/aminfer bench-m82
./zig-out/bin/aminfer bench-m81 --models-dir /path/to/models
AMINFER_MODELS_DIR=/path/to/models AMINFER_METADATA_DIR=/path/to/metadata uv run --directory tools python -m unittest discover -s tests
```

## Repository index

Agents orient through the local ikat index; see the "ikat repository index"
section of [../../AGENTS.md](../../AGENTS.md).
