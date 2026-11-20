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
   and tree width (`src/quant/spec.zig`); unsupported combinations are compile
   errors.
2. **Confirmed**: on the GPU the same configuration tables drive Metal
   function constants through `MTL4SpecializedFunctionDescriptor` and
   `MTLFunctionConstantValues` (`ami_pipeline_create_specialized`), so each
   kernel variant is specialized when its pipeline is created at startup, never
   on the hot path.

## Consequences

- Supported combinations are enumerated explicitly; adding one is a code
  change, not a runtime flag.
- Startup compiles one pipeline state per active specialization; the binary
  embeds one shared `.metallib`.

## Open items

- None for M1 (3-bit and 4-bit group-64 variants); MoE top-k and tree-width
  variants join in M2–M3.
