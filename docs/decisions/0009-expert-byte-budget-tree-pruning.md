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
- **Spike S5 and real-checkpoint M3 validation closed (2026-11-21, Apple M4
  24 GiB)** — see
  [../benchmarks/results.md](../benchmarks/results.md#m3-speculative-decoding-and-spikes-s4--s5):
  - On `Qwen3-30B-A3B` geometry (`q3`, `E = 128`, `top_k = 8`, `H = 2048`,
    `I = 768`), our weight-stationary CSR-deduplicated MoE kernel crossed over
    the per-slot baseline at `M = 2` under correlated routing (`1.36×` at
    `M = 4`, `1.78×` / `36.78 GB/s` at `M = 8`, `1.87×` at `M = 16`, saving
    `65.0–91.0%` of weight reads) and at `M = 16` under uniform random routing
    (`1.08×` at `M = 16`, `1.35×` at `M = 32`).
  - On the real 48-layer `Qwen3-30B-A3B-3bit` checkpoint paired with
    `RedHatAI/Qwen3-30B-A3B-speculator.eagle3` (reasoning prompt, `T = 0`, 100%
    bit-for-bit lossless), verifying an unpruned `cap = 16` EAGLE-3 tree slowed
    decoding from `46.00` to `24.93 tok/s` (`0.54×`, `31.28` unique
    experts/layer). Enabling `D7` submodular expert-byte-budget pruning cut
    verified nodes from `13.44` to `2.93`, halved active experts to `15.83/layer`
    (`-49.4%`) in `0.390 ms/round`, cut DRAM bytes per accepted token by
    `-32.4%`, and restored throughput to `48.38 tok/s` (**`1.94×`** vs unpruned
    and **`1.05×`** vs baseline).

## Open items

- Spike S5: **closed in M3** (recorded in
  [../benchmarks/results.md](../benchmarks/results.md#m3-speculative-decoding-and-spikes-s4--s5)).
- Learned ANE cross-layer expert prediction (`A2`) and NVMe cold-expert
  prefetch penalties (`C3`–`C5`) remain scheduled for M4.
