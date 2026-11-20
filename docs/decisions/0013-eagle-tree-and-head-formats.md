# ADR-0013: EAGLE-2 draft trees with EAGLE-3 / EAGLE-2 heads, MTP and small drafts

- Status: Accepted
- Date: 2026-11-20
- Source: project charter ("prioritize EAGLE-2 / MTP"; "for models without
  EAGLE-2 / MTP, try a same-family 0.5B draft") and the user's clarification in
  the init session

## Context

EAGLE-2's contribution is the dynamic, confidence-driven draft tree; EAGLE-3
keeps that tree and changes how the head is trained and which target features
it reads. Published draft heads for Qwen3-30B-A3B (RedHatAI, NVIDIA, lmsys)
are EAGLE-3 heads.

## Decision

1. Draft trees are built with the EAGLE-2 dynamic expansion algorithm.
2. Head weights are supported in EAGLE-3 format (low-, mid- and high-level
   target features, reduced draft vocabulary) and EAGLE-2 format (top-layer
   feature).
3. Models that ship MTP modules use them as L2.
4. Models with neither EAGLE nor MTP heads use a same-family small draft model
   (≈ 0.5B class, same tokenizer; for example Qwen3-0.6B for Qwen3), placed on
   the ANE ([ADR-0008](0008-benchmark-gated-draft-placement.md)).
5. L1 runs in every configuration.

## Consequences

- The head loader supports two layouts and the feature taps they require.
- Tree construction is shared by every head type.

## Evidence

- Public EAGLE-3 heads for Qwen3-30B-A3B exist, for example
  `RedHatAI/Qwen3-30B-A3B-speculator.eagle3` (reported chain acceptance length
  3.44 on HumanEval with 5 draft tokens) and
  `nvidia/Qwen3-30B-A3B-Thinking-2507-Eagle3` (about 120M parameters).

## Open items

- How to choose between MTP and EAGLE heads when a model has both (benchmark).
