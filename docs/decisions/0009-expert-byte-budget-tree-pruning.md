# ADR-0009: Expert-byte-budget pruning of draft trees

- Status: Accepted
- Date: 2026-11-20
- Source: proposal discussed and accepted by the user (init session)

## Context

The reference design verified fixed 16- or 32-node trees. For an MoE target,
each tree node routes to its own experts, and one verification pass must read
the union of all of them — so the pass costs bytes of unique experts, not
nodes. Large trees can make verification slower than plain decoding.

## Decision

1. Verification cost is modeled as fixed cost + per-node cost + the bytes of
   the unique experts per layer divided by the calibrated effective bandwidth
   of the expert kernel in use
   ([../architecture/speculative-decoding.md](../architecture/speculative-decoding.md#expert-byte-budget-pruning)).
2. The verified subtree is the ancestor-closed subtree with the best expected
   accepted tokens per predicted time, chosen greedily by marginal gain after
   each draft level; the same rule decides when to stop drafting.
3. Per-node expert sets come from expert prediction; cold experts carry an
   NVMe penalty.
4. Dense models fall back to a node budget. Compile-time caps on tree size
   remain.

## Consequences

- Expert prediction becomes part of the decode loop, not only of prefetch.
- Effective bandwidth must be calibrated per chip and per kernel variant.

## Evidence

- Speculation on MoE raised verification time 2–3× and could slow decoding by
  up to 1.5× (arXiv 2506.20675, Utility-Driven Speculative Decoding for MoE).
- Choosing the verified subtree by expert-union cost made Qwen3-30B-A3B decode
  1.97× and 1.72× faster than vanilla EAGLE-3 in an offloaded setting
  (arXiv 2610.15417, SpecFit).
- A per-layer expert budget gave 10–30% higher throughput than EAGLE-3
  (arXiv 2602.16052, MoE-Spec).
- On llama.cpp Metal and MLX-LM, per-slot expert kernels had higher effective
  bandwidth than deduplicating ones, so deduplication only paid off at 5–8
  tokens per pass or never (arXiv 2610.34440) — our kernels and cost model
  must account for this.

## Open items

- Spike S5: unique experts per layer versus tree size for Qwen3-30B-A3B, and
  the effective bandwidth of deduplicated versus per-slot kernels.
