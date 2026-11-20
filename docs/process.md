# Process

The single record of project progress: milestones, spikes, the current task
list, the verification log and the session handoff snapshot. Capability status
lives in [status.md](status.md); decisions live in
[decisions/](decisions/README.md).

## Current focus

M0 — project initialization (started 2026-11-20).

## Milestones

Status-row IDs refer to [status.md](status.md).

| Milestone | Goal | Status rows | Exit criteria | State |
| --- | --- | --- | --- | --- |
| M0 | Project initialization | F1–F9 | `zig build` and `zig build test` pass in Debug and ReleaseSafe; docs baseline committed; local ikat index built | In progress |
| M1 | GPU and weight foundation | R1, R2, F6 (hot path), W1–W4, G1, G2, C9 | Wired budget enforced at startup; `.amf` round-trip (tools → Zig) with alignment validation; fused dequant GEMV / GEMM match a reference within tolerance; effective bandwidth measured per kernel | Not started |
| M2 | Non-speculative decode | W5, W6, C1, C2, R3, R4, G3, G4 | Greedy end-to-end decode of the primary target from `.amf`; baseline tokens/s and bytes per token recorded | Not started |
| M3 | Speculative decoding | D1–D3, D5–D7, D9, G5 | Lossless output versus M2; speedup over the M2 baseline, acceptance-length and experts-per-pass statistics recorded | Not started |
| M4 | Neural Engine and storage | A1–A3, G6, W7, W8, C3–C5, D4 | Hybrid prefill throughput versus GPU-only prefill; expert-prediction recall and prefetch lead time; cold-expert decode without swap | Not started |
| M5 | Serving and pipelining | C6–C8, D8 | Streaming server under load; pipelined-drafting gain measured | Not started |
| Future | Scale-out | X1, X2 | — | — |

## Spikes

Short measurements that must run before the related design moves from Draft to
Confirmed, or before an accepted design is tuned.

| ID | Question | Feeds | Milestone | State |
| --- | --- | --- | --- | --- |
| S1 | KV tile read path: R16Float vs RGBA16Float texture vs aliasing `MTLBuffer`; token- vs channel-major rows; decode-attention bandwidth | [ADR-0004](decisions/0004-kv-cache-iosurface-texture-sharing.md), R4 | M2 | Open |
| S2 | Metal 4 steady-state allocations and encode overhead per decode round (Instruments Allocations, signposts) | [ADR-0011](decisions/0011-metal4-zero-allocation-hot-path.md) | M1 | Open |
| S3 | Cold-expert reads: `aio_read` + `kqueue` vs E-core `pread` workers; queue depth 1–16; 2–3 MiB reads with `F_NOCACHE`; QoS and I/O-policy effects; p50 / p99 latency | [ADR-0006](decisions/0006-amf-cold-experts-direct-io.md), C4 | M4 | Open |
| S4 | ANE handoff latency (GPU → CPU → ANE → CPU → GPU) and bandwidth contention of concurrent ANE weight streaming on GPU GEMV | [ADR-0008](decisions/0008-benchmark-gated-draft-placement.md) | M3 | Open |
| S5 | Unique experts per layer versus draft-tree size for Qwen3-30B-A3B; effective bandwidth of deduplicated vs per-slot expert kernels | [ADR-0009](decisions/0009-expert-byte-budget-tree-pruning.md) | M3 | Open |
| S6 | ANE attention block: CoreML compilation, static shapes for a growing KV length, fp16 numerics, throughput vs GPU | [ADR-0010](decisions/0010-hybrid-wavefront-prefill.md), A3 | M4 | Open |

## M0 task list

- [x] Environment probe and design review with the user; decisions recorded
      as ADR-0001 to ADR-0014
- [x] Install the Metal Toolchain component (27A266a)
- [x] `AGENTS.md` (commit `1ad0da8`)
- [x] Dual MIT / Apache-2.0 license (commit `6d5053d`)
- [x] Documentation baseline: README, architecture, decisions, guides, status,
      process
- [x] `build.zig` graph: Zig core + ObjC bridge (translate-c) + embedded
      `.metallib` + tests, with `aminfer probe` for Metal and ANE facts (F1, F3)
- [x] 16 KiB wired regions and sysctl probe, test first (F2, F9)
- [x] `ExpertSource` tagged union, test first (F4)
- [x] Bridge: NoCopy buffer and Metal 4 smoke dispatch, test first (F5, F6)
- [ ] `tools/` uv scaffold (F7) and `justfile` (F8)
- [ ] Verify Debug / ReleaseSafe / ReleaseFast; build the local ikat index

## Verification log

Only checks that actually ran are listed.

| Date | Check | Result |
| --- | --- | --- |
| 2026-11-20 | Throwaway probe (not in repo): Zig 0.17.1 `addTranslateC` + ObjC `-fobjc-arc -Werror` + Metal link + `zig build test` | Pass |
| 2026-11-20 | `@cImport` under Zig 0.17.1 | `error: invalid builtin function: '@cImport'` |
| 2026-11-20 | `aio_read` with `SIGEV_KEVENT` on macOS 27.2 | Pass: `EVFILT_AIO` delivered, bytes in `kevent64.ext[1]`, data matches; `aio_return` afterwards fails with `EINVAL` |
| 2026-11-20 | IOSurface fp16 tile → R16Float and RGBA16Float textures, with `MTL_DEBUG_LAYER=1` | Pass |
| 2026-11-20 | `xcrun metal -std=metal4.0 -mmacosx-version-min=26.0` | Pass (after installing the Metal Toolchain) |
| 2026-11-20 | F1: `zig build`; `zig build test` (Debug, ReleaseSafe): 3/3 tests; `aminfer probe` matches the throwaway probe; binary `minos 26.0` | Pass |
| 2026-11-20 | F2, F9: `zig build test` (Debug, ReleaseSafe): 10/10 tests; `aminfer probe` sysctl output matches [hardware.md](architecture/hardware.md) | Pass |
| 2026-11-20 | F4: `zig build test` (Debug, ReleaseSafe): 16/16 tests | Pass |
| 2026-11-20 | F5, F6: `zig build`; `zig build test` (Debug, ReleaseSafe): 20/20 tests; direct test binary with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`: 20/20 tests | Pass |

The measured platform facts are collected in
[architecture/hardware.md](architecture/hardware.md).

## Decision log

Decisions are recorded as ADRs; the index is
[decisions/README.md](decisions/README.md). In the 2026-11-20 init session the
user accepted ADR-0008 (benchmark-gated draft placement), ADR-0009
(expert-byte-budget pruning), ADR-0010 (hybrid wavefront prefill), ADR-0011
(Metal 4 hot path, macOS 26.0 floor), ADR-0012 (wired budget headroom) and
ADR-0013 (EAGLE-2 tree with EAGLE-3 / EAGLE-2 heads). ADR-0001 to ADR-0007 and
ADR-0014 record the project charter and the four new requirements of the same
session.

## Session handoff snapshot

Update this before the context window runs out or when handing off.

- Last updated: 2026-11-20
- Commits so far: `1ad0da8` AGENTS.md, `6d5053d` license, `3a573a5` docs baseline,
  `0518ea8` build graph and probe, `06e8656` wired regions, `e6c9221` ExpertSource
- In flight: `tools/` uv scaffold and `justfile` (F7, F8)
- Awaiting the user: nothing
- Key evidence: [architecture/hardware.md](architecture/hardware.md)
