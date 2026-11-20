# ADR-0014: Compile-time specialization of hot code

- Status: Accepted
- Date: 2026-11-20
- Source: project charter (user); the GPU-side mechanism is a draft

## Context

Quantization groups (3-bit, 4-bit), MoE top-k and speculative tree widths
vary by model and configuration. Branching on them inside hot loops costs
cycles on the CPU and divergence on the GPU.

## Decision

1. **Confirmed**: Zig `comptime` generates branch-free specialized host code
   for each supported combination of quantization bits, group size, MoE top-k
   and tree width; unsupported combinations are compile errors.
2. **Draft**: on the GPU the same configuration tables drive Metal function
   constants, so each kernel variant is specialized when its pipeline is
   created at startup, never on the hot path.

## Consequences

- Supported combinations are enumerated explicitly; adding one is a code
  change, not a runtime flag.
- Binary size grows with the number of variants.

## Open items

- Confirm the function-constant mechanism and the variant count when the first
  kernels land (M1).
