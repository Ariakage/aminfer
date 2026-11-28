# Optional command shortcuts for aminfer.
# build.zig remains the single build hub (ADR-0001).

set shell := ["zsh", "-euo", "pipefail", "-c"]

# List available recipes.
default:
    @just --list

# Debug build.
build:
    zig build

# ReleaseFast build (benchmarks only; verify Debug / ReleaseSafe first).
build-fast:
    zig build -Doptimize=ReleaseFast

# Full test suite in Debug + ReleaseSafe (-j1), Python SDK tests, and tools tests.
test:
    zig build test --summary all -j1
    zig build test -Doptimize=ReleaseSafe --summary all -j1
    just python-sdk-test
    just tools-test

# Full test suite with safety and alignment assertions kept on.
test-safe:
    zig build test -Doptimize=ReleaseSafe --summary all -j1

# Verify Zig formatting across build.zig and src/.
fmt-check:
    zig fmt --check build.zig src

# Run all Zig, Python SDK, and offline tools test suites plus formatting check.
check:
    just fmt-check
    just test

# Print kernel, Metal, and Neural Engine facts of this machine.
probe:
    zig build run -- probe

# Run full M1 + M2 benchmark suite (ReleaseFast).
bench:
    zig build -Doptimize=ReleaseFast && ./zig-out/bin/aminfer bench

# Run M2 benchmarks only: Spike S1, G3 MoE, G4 GQA, and end-to-end .amf decode (ReleaseFast).
bench-m2:
    zig build -Doptimize=ReleaseFast && ./zig-out/bin/aminfer bench-m2

# Show current wired-memory limits and physical RAM.
sysctl-show:
    sysctl iogpu.wired_limit_mb vm.user_wire_limit vm.global_user_wire_limit hw.memsize

# Everyday raised limits on 24 GB baseline: 21 GiB (keeps 3 GiB for macOS).
sysctl-wire-21g:
    sudo sysctl -w iogpu.wired_limit_mb=21504
    sudo sysctl -w vm.global_user_wire_limit=22548578304
    sudo sysctl -w vm.user_wire_limit=22548578304

# Dedicated benchmark ceiling on 24 GB baseline: 22.5 GiB (leaves ~1.5 GiB; see docs/guides/system-tuning.md).
sysctl-wire-bench:
    sudo sysctl -w iogpu.wired_limit_mb=23040
    sudo sysctl -w vm.global_user_wire_limit=24159191040
    sudo sysctl -w vm.user_wire_limit=24159191040

# Reset wired limits to the 24 GB baseline defaults (or pass your machine's default bytes).
sysctl-wire-reset vm_bytes="19585050869":
    sudo sysctl -w iogpu.wired_limit_mb=0
    sudo sysctl -w vm.global_user_wire_limit={{vm_bytes}}
    sudo sysctl -w vm.user_wire_limit={{vm_bytes}}

# Sample CPU, GPU, and ANE power/frequency every second.
powermetrics interval_ms="1000":
    sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i {{interval_ms}}

# Inspect current swap usage and pageout/swapout counters.
swap-check:
    sysctl vm.swapusage
    vm_stat | grep -E "Pageouts|Swapins|Swapouts"

# Sync the offline Python toolchain environment under tools/.
tools-sync:
    cd tools && uv sync

# Run unit tests for the Python ctypes SDK (builds shared library first).
python-sdk-test: build
    cd python && uv run python -m unittest discover -s tests -v

# Run unit tests for the offline Python toolchain.
tools-test:
    cd tools && uv run python -m unittest discover -s tests -v

# Refresh the local ikat repository index to match HEAD.
ikat-sync:
    npx -y ikat-cli@0.2.0 sync
