# ADR-0002: 16 KiB page alignment, wired memory and zero-copy sharing

- Status: Accepted
- Date: 2026-11-20
- Source: project charter (user)

## Context

Apple Silicon uses 16,384-byte pages. Metal's `newBufferWithBytesNoCopy`
requires page-aligned memory, IOSurface memory is page-granular, and the ANE
and GPU can share physical pages with the CPU. Any copy between engines, any
page fault and any swap-out shows up directly as latency.

## Decision

1. `APPLE_PAGE_SIZE = 16384`, asserted at compile time to equal
   `std.heap.page_size_min` for `aarch64-macos` and checked at startup against
   `hw.pagesize`.
2. Every buffer handed to Metal (`newBufferWithBytesNoCopy`) or shared through
   IOSurface has a 16 KiB-aligned start address and a total length that is a
   multiple of 16 KiB.
3. Hot weights and active buffers live in wired regions: anonymous `mmap`,
   lengths rounded up to 16 KiB, pinned with `mlock`. A failed `mlock` is an
   error, never a fallback to pageable memory.
4. Tensors are never copied between CPU, ANE (`MLMultiArray` / IOSurface) and
   Metal (`MTLBuffer` / `MTLTexture`); all three map the same pages.
5. Nothing is allocated on the hot path: tensor buffers, KV tiles and ring
   slots are preallocated at startup through explicit Zig allocators.
6. Alignment checks at bridge boundaries are runtime errors kept in every
   build mode, not only debug assertions.

## Consequences

- Up to 16 KiB of padding per region, and a startup cost for wiring.
- Wiring is bounded by the wired budget
  ([ADR-0012](0012-wired-memory-budget-headroom.md)).
- Code that needs per-round memory must take it from a preallocated arena.

## Evidence

- `hw.pagesize` = 16384 on the baseline.
- Metal accepted a NoCopy length that is not a 16 KiB multiple, even with
  `MTL_DEBUG_LAYER=1`, so the rule must be enforced by our code
  ([../architecture/hardware.md](../architecture/hardware.md)).

## Open items

- None.
