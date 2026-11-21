# ADR-0008: Benchmark-gated placement of L2 drafting

- Status: Accepted
- Date: 2026-11-20
- Source: proposal discussed and accepted by the user (init session)

## Context

The reference architecture placed all L2 drafting on the ANE, overlapped with
GPU verification, on the premise that the ANE runs from on-chip SRAM and leaves
GPU bandwidth alone. Two facts contradict that premise:

- EAGLE and MTP heads draft from the target features of the tokens just
  accepted, so their drafting is serial with verification.
- The ANE streams weights from the same DRAM the GPU uses (on-chip SRAM is
  megabyte-scale), and an ANE round trip adds two GPU → CPU → ANE hand-offs per
  round.

## Decision

1. A `DraftSource` abstraction lets EAGLE and MTP heads run either on the ANE
   or on the GPU, in the same command buffer as verification.
2. The default placement is chosen per model and chip from measured round
   latency (draft + hand-offs + verification), and the measurement is recorded
   in `docs/process.md`.
3. The independent same-family small draft model, PEARL-style pipelined
   drafting and expert prediction are always placed on the ANE.
4. The scheduler budgets DRAM bytes across ANE and GPU traffic.

## Consequences

- Both placements of the heads must exist before the gate can choose.
- A benchmark harness and per-chip calibration are prerequisites of M3.

## Evidence

- On-chip SRAM: about 2 MB working set on M1 (arXiv 2606.22283); throughput
  cliff near 32 MB on M4 (maderix ANE benchmarks). See
  [../architecture/hardware.md](../architecture/hardware.md).
- **Spike S4 closed (M3, 2026-11-21, Apple M4 24 GiB)** — see
  [../benchmarks/results.md](../benchmarks/results.md#m3-speculative-decoding-and-spikes-s4--s5):
  - Zero-copy IOSurface `GPU → CPU → ANE → CPU → GPU` round-trip handoff
    measured `574.64 us` (`ane_tiny_predict_us = 92.46 us`, `uses_ane = yes`,
    `zero_copy_backing = yes`).
  - For a `2L × 2048×2048 fp16` draft workload, ANE took `366.93 us` (`941.57 us`
    with handoff) at `M = 1` and `363.20 us` (`937.84 us` with handoff) at
    `batch = 32`, versus `201.87 us` (`M = 1`) and `277.36 us` (`batch = 32`)
    for in-command-buffer GPU execution. The `D5` startup gate therefore selects
    `.gpu` on Apple M4.
  - Concurrent `40 MiB` ANE weight streaming (`50.00 GB/s`) reduced GPU
    `gemv_q3` bandwidth by `30.8%` (`70.91 → 49.09 GB/s`) and `gemv_q4` by
    `32.5%` (`77.92 → 52.57 GB/s`), confirming unified-DRAM contention that
    must be budgeted during M4/M5 concurrent ANE workloads.

## Open items

- Spike S4: **closed in M3** (recorded in
  [../benchmarks/results.md](../benchmarks/results.md#m3-speculative-decoding-and-spikes-s4--s5)).
- Pipelined-drafting design for the small draft model (status row D8; see
  *PEARL* [arXiv:2408.11850](https://arxiv.org/abs/2408.11850), *P-EAGLE*
  [arXiv:2602.01469](https://arxiv.org/abs/2602.01469), and Spike S4 contention
  analysis in [open-decisions-research.md](../architecture/open-decisions-research.md#remaining-m3m5-open-tuning-items-adr-0017-d8-adr-0005)).
