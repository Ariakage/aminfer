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
   drafting and expert prediction are placed according to measured round
   latency and unified-DRAM contention:
   - **Confirmed `D8` pipelined small-draft gate (2026-11-21):** Because Spike
     `S4` measured `574.64 us` ANE round-trip handoff and `30.8%–32.5%` unified
     DRAM bandwidth contention, PEARL-style ([arXiv:2408.11850](https://arxiv.org/abs/2408.11850))
     overlapped round `r + 1` drafting with an independent small draft model
     (`D4`, e.g. `Qwen3-0.6B`) is enabled selectively when target verification
     latency is long enough to amortize handoff and contention (such as dense
     `Qwen3-14B` at `~95 ms/step`), whereas fast MoE targets paired with
     in-command-buffer EAGLE-3 (`~19 ms/step`) stay un-pipelined on the GPU.
4. The scheduler budgets DRAM bytes across ANE, CPU SME2
   ([ADR-0018](0018-bare-metal-armv9-sme2-engine.md)), and GPU traffic.

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
- `D8` pipelined small-draft latency gate **confirmed on 2026-11-21** for M5
  implementation (measured end-to-end gain on `Qwen3-14B + Qwen3-0.6B` tracked
  in [../status.md](../status.md) and
  [open-decisions-research.md](../architecture/open-decisions-research.md#remaining-m3m5-open-tuning-items-adr-0017-d8-adr-0005)).
