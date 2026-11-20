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

## Open items

- Spike S2: steady-state allocations and encode overhead per decode round.
