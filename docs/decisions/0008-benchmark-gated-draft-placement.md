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

## Open items

- Spike S4: hand-off latency and bandwidth contention.
- Pipelined-drafting design for the small draft model (status row D8).
