# AGENTS.md

Conventions for coding agents working in this repository.

## Project overview

**aminfer** is a bare-metal, ultra-low-latency LLM inference engine for Apple
Silicon. The baseline is a 24 GB M4; the design scales up to 128 GB+ Max/Ultra
machines and, later, Thunderbolt-bridged clusters. It runs CPU P-cores, CPU
E-cores, the Neural Engine (ANE) and the GPU concurrently over one unified
memory pool. See [README.md](README.md) and
[docs/architecture/overview.md](docs/architecture/overview.md).

- **Engine core**: Zig 0.17.1 — scheduler, memory pools, lock-free ring
  buffers, L1 suffix automaton, async I/O, kqueue SSE server.
- **Hardware bridge**: pure Objective-C (`.m` + `.h`, `-fobjc-arc`), only for
  Apple frameworks, exposed to Zig through the pure C header `apple_bridge.h`.
  Zig 0.17.1 removed `@cImport`; the header is translated by `b.addTranslateC`
  in `build.zig` ([ADR-0001](docs/decisions/0001-language-and-toolchain-boundaries.md)).
- **GPU kernels**: Metal Shading Language (`.metal`) compiled with `xcrun metal`
  into a `.metallib`.
- **Offline tools**: Python managed by `uv`, under `tools/` only.
- **Build**: `build.zig` is the single build hub; an optional `justfile` only
  wraps common commands.
- **Platform floor**: macOS 26.0+ on Apple Silicon
  ([ADR-0011](docs/decisions/0011-metal4-zero-allocation-hot-path.md)).
- **License**: MIT OR Apache-2.0.
- **Status and progress**: capability status lives only in
  [docs/status.md](docs/status.md); progress lives only in
  [docs/process.md](docs/process.md).

Repository layout (conventions — directories appear as milestones land; check
the tree or the ikat digest for what exists):

```
AGENTS.md            # this file — agent conventions
README.md            # project introduction
LICENSE-MIT          # dual license, part 1
LICENSE-APACHE       # dual license, part 2
build.zig            # single build hub: Zig core, ObjC bridge, metallib, tests
build.zig.zon        # package manifest (no third-party dependencies)
justfile             # optional shortcuts (sysctl, uv, powermetrics)
src/                 # Zig engine core (one responsibility per file)
src/bridge/          # apple_bridge.h (pure C ABI) + apple_bridge.m (ObjC, ARC)
src/kernels/         # Metal Shading Language kernels
tools/               # uv-managed Python offline toolchain (never at runtime)
docs/process.md      # the only progress record
docs/status.md       # the only implemented / unimplemented / planned record
docs/architecture/   # architecture design
docs/benchmarks/     # benchmark methodology, environment, metrics and raw records
docs/decisions/      # ADRs (technical decision records)
docs/guides/         # build and environment guides
```

Checks (all must pass before delivering a code change — see Development rules):

```
zig build                              # Debug build, must compile cleanly
zig build test                         # full test suite, Debug
zig build test -Doptimize=ReleaseSafe  # alignment / safety assertions kept on
```

## ikat repository index

This repository is indexed with the ikat skill. The index is local: `.ikat/`
is untracked and listed in `.git/info/exclude`. Below, `ikat` means an
installed `ikat` CLI, or `npx -y ikat-cli@0.2.0` when it is not installed
(the version is pinned; bump it deliberately).

- On entering the repo, read `.ikat/digest.md` first. If it is missing, run
  `ikat init --yes`. If the digest header's `index:` sha differs from
  `git rev-parse HEAD`, run `ikat sync`.
- Orient from the digest instead of re-scanning the tree; drill down with
  `ikat file <path>` and `ikat refs <path>`.
- After modifying files: `ikat sync`, then `ikat enrich --only-stale`, and
  rewrite entries only for files you actually re-read.
- After every commit, run `ikat sync` so the HEAD recorded by the local index
  matches the latest commit. This never modifies tracked files.

## Collaboration and scope

- Communicate with the user in Chinese by default; write documentation in
  English; use English for code identifiers.
- Release notes are in English unless explicitly requested otherwise.
  Repository release notes stay consistent with GitHub Releases on features,
  compatibility, and known limitations.
- Keep changes focused — no unrelated refactors, dependency swaps, or
  reformatting.
- Clearly distinguish implemented / confirmed / draft / undecided. Never write
  a suggestion as a fact.
- Sync user-confirmed new directions into the corresponding docs and decision
  records.
- When changing behavior, public interfaces, or configuration, update the
  related docs and examples in the same change.
- Do not fabricate run results, and do not claim checks passed that were not
  run.
- After finishing an independently verifiable stage of work, commit per the
  Git conventions below. Do not push or publish unless asked.
- When the context window nears capacity, first persist the current progress
  and key points to [docs/process.md](docs/process.md) — commits made,
  verifications in flight, decisions awaiting the user, key files and evidence
  locations — before continuing or handing off to a new session, so
  uncommitted on-site state is not lost when the window runs out.

## Git commits

- Create a focused commit after each independently verifiable stage (for
  example a DEV task, a milestone, or a standalone feature), once applicable
  checks pass and related docs are synced. If the user explicitly asks not to
  commit, follow that instead. If required checks fail or cannot run, state why
  and what remains unverified — do not mark the task done.
- One commit, one purpose: each commit carries a single, clear objective and
  logical theme. Do not mix unrelated features, fixes, refactors, or docs in
  one commit.
- Before committing, review the working tree and the staged content; commit
  only what belongs to this stage and keep unrelated work from the user and
  other collaborators intact.
- Commit messages follow
  [Conventional Commits 1.0.0](https://www.conventionalcommits.org/zh-hans/v1.0.0/):
  - Title format: `type(scope): description`; `scope` is optional.
  - `feat` for new features, `fix` for bug fixes; use `docs`, `test`,
    `refactor`, `build`, `chore`, etc. as appropriate.
  - Title and body descriptions are in English by default.
  - Mark breaking changes with `!` before the colon, or with a
    `BREAKING CHANGE:` footer.
  - Example: `docs(agents): record Conventional Commits usage in English`
- Commits and pull requests carry only the user's identity — no AI tool
  attribution, co-author trailers, "generated with" footers, badges, or links.
- After a commit, if the local `.ikat/` index is in use, run `ikat sync` per
  the "ikat repository index" convention so the index HEAD matches the latest
  commit (tracked files are not touched).

## Runtime boundaries

- Logs, errors, and examples must not leak secrets or tokens; never commit real
  credentials.

## Engineering invariants

Hard rules from the project charter. Rationale and evidence live in
[docs/architecture/](docs/architecture/) and [docs/decisions/](docs/decisions/).

- **No Python in the inference runtime.** Zero embedded Python or PyTorch
  runtime dependencies; Python exists only in `tools/` for offline work
  (weight splitting, expert heat statistics, CoreML `.mlpackage` export).
- **No dynamic heap allocation on the hot path.** Tensor buffers, KV cache
  tiles and ring-buffer slots are preallocated once at startup through explicit
  Zig allocators. GPU command encoding uses reusable Metal 4 command
  allocators and command buffers.
- **No implicit OS paging or swap.** Hot weights and active buffers are 16 KiB
  page-aligned and pinned with `mlock`, within the wired budget computed at
  startup ([ADR-0012](docs/decisions/0012-wired-memory-budget-headroom.md)).
- **16 KiB page alignment.** `APPLE_PAGE_SIZE = 16384`. Every buffer handed to
  Metal (`newBufferWithBytesNoCopy`) or shared through IOSurface has a start
  address and total length aligned to 16 KiB. Our code enforces this; Metal
  does not reliably reject violations.
- **Zero-copy bridging.** Never `memcpy` tensors between CPU, ANE
  (`MLMultiArray` / IOSurface) and Metal (`MTLBuffer` / `MTLTexture`); all three
  share the same physical pages.
- **Extensible expert storage.** `ExpertSource` stays a Zig tagged union
  (`local_wired_ram`, `local_nvme_ssd`, `thunderbolt_peer`) so Thunderbolt
  expert parallelism needs no scheduler or kernel refactor.
- **Compile-time specialization.** Use Zig `comptime` for quantization groups
  (3-bit / 4-bit), MoE top-k and speculative tree widths to generate
  branch-free specialized code.
- **No unapproved dependencies.** Do not add third-party C or Zig dependencies
  without user approval; prefer Zig `std` and native macOS APIs. If a
  third-party library brings a clear benefit, propose it first.

## Development rules

- **Error handling**: never swallow errors — `catch {}` is forbidden. Propagate
  Zig error sets explicitly, and null-check every pointer returned from C or
  Objective-C.
- **Verification before delivery**: after every code change, `zig build` must
  compile without errors and `zig build test` must pass in full. For changes
  touching memory alignment or GPU kernels, first verify the alignment
  assertions in Debug / ReleaseSafe, then switch to ReleaseFast for
  benchmarks.

## Documentation rules

- Categorize documents under `docs/` subdirectories; do not put everything in
  the `docs/` root:
  - `docs/architecture/` — architecture design;
  - `docs/benchmarks/` — benchmark methodology, environment, summary metrics
    and raw run records, converged in `docs/benchmarks/results.md`;
  - `docs/decisions/` — ADRs (technical decision records);
  - `docs/guides/` — build and environment guides.
- Project progress is recorded only in `docs/process.md`; do not scatter it
  elsewhere.
- Implemented capabilities, unimplemented capabilities and future plans are
  recorded only in `docs/status.md`; other documents link to it instead of
  restating status.
- Measured performance metrics and raw benchmark records converge in
  `docs/benchmarks/results.md`; other documents link to it instead of
  duplicating raw run outputs.

## Delivery notes

When reporting finished work, briefly state:

- what changed and why;
- which verifications ran and their results;
- what remains unresolved.

Reference code or docs with locatable paths.

<!-- ikat:index -->
This repo has an ikat index: read .ikat/digest.md for architecture context.
Run `ikat init` to regenerate it; remove .ikat/ and this block to opt out.
<!-- /ikat:index -->
