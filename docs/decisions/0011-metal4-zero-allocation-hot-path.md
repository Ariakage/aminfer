# ADR-0011: Metal 4 zero-allocation hot path and the macOS 26.0 floor

- Status: Accepted
- Date: 2026-11-20
- Source: proposal discussed and accepted by the user (init session)

## Context

Classic Metal creates a new `MTLCommandBuffer` (and encoders) for every
submission, which allocates inside the driver and the Objective-C runtime on
every decode round. That conflicts with the rule that the hot path performs no
dynamic allocation.

## Decision

1. GPU work is submitted through Metal 4: an `MTL4CommandQueue`, long-lived
   `MTL4CommandBuffer` objects reused with `beginCommandBufferWithAllocator:` /
   `endCommandBuffer`, and one `MTL4CommandAllocator` per frame in flight,
   `reset` once its work has completed.
2. Resources are bound through argument tables and made resident through an
   `MTLResidencySet` attached to the queue; Metal 4 does not track residency
   implicitly.
3. Completion is signalled through `MTLSharedEvent`; pipelines are built at
   startup only.
4. The deployment target is **macOS 26.0** on Apple Silicon.

## Consequences

- macOS releases before 26.0 are unsupported.
- Every resource used by a kernel must be in the residency set.
- Allocations inside Apple frameworks that we cannot control (for example
  CoreML prediction calls) are bounded and measured rather than assumed away
  (spike S2).

## Evidence

- SDK headers mark the Metal 4 command types `API_AVAILABLE(macos(26.0))`;
  `MTL4CommandAllocator` documents `reset` as reusing its internal heaps.
- The baseline exposes `newMTL4CommandQueue`, `newCommandAllocator` and
  `MTLResidencySet`.
- **Spike S2 (measured 2026-11-20 on Apple M4, macOS 27.2, ReleaseFast `aminfer bench`):**
  - Workload: 1,000 steady-state multi-dispatch rounds on a 2-frame `Compute`
    ring (32 fused-dequant dispatches/round with `insert_barrier_before` and
    uniform ring streaming, 32,000 dispatches total) after a 2,048-round
    driver-pool warmup.
  - CPU encode overhead: `7.69–7.90 us/round` (`240–247 ns/dispatch`); full
    pipelined round time `96.0–101.7 us/round`.
  - Steady-state `malloc_zone_statistics` deltas across the 1,000 rounds:
    `blocks_in_use = 0`, `size_in_use = 0 B`, `size_allocated = 0 B`.
  - Driver warmup observation: on a fresh `MTL4CommandQueue`, Apple's Metal 4
    driver lazily populates a bounded internal pool (~2,000 × 64 B ≈ 128 KiB)
    over the first ~2,000 commits, after which allocations flatten to zero in
    steady state.

## Open items

- None (Spike S2 closed on 2026-11-20; full M1 benchmark summary recorded in
  [../process.md](../process.md#m1-calibration-and-spike-s2-results)).
