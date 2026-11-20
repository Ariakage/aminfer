# ADR-0001: Language and toolchain boundaries

- Status: Accepted
- Date: 2026-11-20
- Source: project charter (user); toolchain facts verified in the init session

## Context

The engine needs bare-metal control over memory and threads, Apple frameworks
that are only reachable from Objective-C, GPU kernels in Metal Shading
Language, and an offline conversion ecosystem that lives in Python. Each
language must stay inside a clear boundary, or Python and Objective-C creep
into the hot path.

## Decision

1. **Zig 0.17.1** implements the engine core: scheduler, memory pools,
   lock-free rings, L1 suffix automaton, async I/O and the kqueue SSE server.
   `build.zig.zon` records `minimum_zig_version = "0.17.1"`.
2. **Pure Objective-C bridge** (`src/bridge/apple_bridge.m`, compiled with
   `-fobjc-arc`) is the only code that calls Apple frameworks: Metal, CoreML,
   IOSurface and Foundation, plus CoreVideo where CoreML's
   pixel-buffer-backed `MLMultiArray` requires it.
3. **One C ABI contract.** `src/bridge/apple_bridge.h` is plain C: opaque
   handles, fixed-width integers, POD structs and status codes — no
   Objective-C types. Zig 0.17.1 removed `@cImport`, so `build.zig` translates
   the header with `b.addTranslateC` and exposes it as a module.
4. **Zig wrappers** turn status codes into explicit error sets and null-check
   every returned pointer.
5. **Metal Shading Language** kernels in `src/kernels/` are compiled by
   `xcrun metal` (`-std=metal4.0 -mmacosx-version-min=26.0`) into one
   `.metallib`. Implementation choice for M0: `build.zig` embeds the metallib
   in the binary, and the bridge loads it with `newLibraryWithData:`.
6. **Python only offline**, in `tools/`, managed by uv: weight splitting into
   `.amf`, expert heat statistics, CoreML `.mlpackage` export. The runtime never
   embeds or calls Python or PyTorch.
7. **`build.zig` is the single build hub.** The optional `justfile` only wraps
   commands (`sysctl`, `uv`, `powermetrics`).
8. **No third-party C or Zig dependencies without user approval**; prefer Zig
   `std` and native macOS APIs.

## Consequences

- Builds require Xcode with the Metal Toolchain component and Zig 0.17.1.
- Every bridge function is callable from C, testable from Zig, and free of
  Objective-C exceptions crossing the boundary.
- An embedded metallib keeps tests and the CLI independent of install paths.

## Evidence

- `@cImport` under Zig 0.17.1: `error: invalid builtin function: '@cImport'`.
- A throwaway probe built an Objective-C file with
  `-fobjc-arc -Wall -Werror`, linked Metal, translated its header with
  `addTranslateC` and passed `zig build test`.
- `xcrun metal -std=metal4.0 -mmacosx-version-min=26.0` compiles after
  `xcodebuild -downloadComponent MetalToolchain`.

## Open items

- None.
