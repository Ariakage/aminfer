# ADR-0005: Dual-source tree merging of L1 and L2 drafts

- Status: Accepted
- Date: 2026-11-20
- Source: user requirement (init session, "new requirements" 2)

## Context

When L1 (CPU suffix automaton, under 0.05 ms) and L2 (EAGLE-2 / MTP heads or
the small draft model) produce candidates independently, the GPU's tree budget
is wasted on duplicate paths.

## Decision

1. An L1 match of at least 4 consecutive tokens is a high-confidence seed
   prefix. It is attached to the root as a chain, and L2 drafts onward from
   the seed's last node.
2. Shorter L1 candidates enter as low-priority branches. All branches share one
   prefix trie, so every path is stored and verified once, and the budget
   counts each node once.
3. Seeding an EAGLE head steps the head through the seed tokens with its own
   predicted features; seeding the small draft model is one batched forward.
4. Seed nodes are priced like any other node by the pruning cost model
   ([ADR-0009](0009-expert-byte-budget-tree-pruning.md)). For L1 seed/branch
   nodes and independent `D4` small-draft nodes that lack target hidden states,
   expert sets `E_l(v)` are predicted via decoupled token-embedding + target
   router gate projection (executable in a single batched GPU frame or L2-resident
   CPU SME2 pass per [ADR-0018](0018-bare-metal-armv9-sme2-engine.md)).
5. **Confirmed seed threshold policy (2026-11-21).** Keep `SEED_MIN_MATCH_LEN = 4`
   fixed through M4/M4.5 so short 1–3 token coincidences do not crowd `D7`
   byte/node budgets; revisit adaptive thresholds only after M5 NVMe N-gram
   tables (`C6`) land.

## Consequences

- L1 and L2 must agree on token ids (same tokenizer) and on the trie format.
- The expected ceiling of tokens accepted per verification pass rises from
  about 4–6 to about 8–14 (project charter target). It is measured in M3, not
  assumed.

## Open items

- Post-M5 ablation of adaptive seed thresholds over persistent NVMe N-gram
  tables (`C6`; see
  [../architecture/open-decisions-research.md](../architecture/open-decisions-research.md#remaining-m3m5-open-tuning-items-adr-0017-d8-adr-0005)).
