# Speculative decoding

> Design reference; status lives in [../status.md](../status.md).

## Layers

| Layer | Engine | Source | Output |
| --- | --- | --- | --- |
| L1 | CPU P-core | Suffix automaton over the prompt and generated tokens (optionally a persistent N-gram store) | Continuations with their match length |
| L2 | ANE or GPU ([ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md)) | EAGLE heads, MTP heads or a same-family small draft model ([ADR-0013](../decisions/0013-eagle-tree-and-head-formats.md)) | Draft tree with per-node draft probabilities |
| L3 | GPU | Target model with tree attention | Accepted path, bonus token, target features |

Verification is lossless. At temperature 0 a node is accepted when it equals
the target's argmax at its position. At temperature > 0 the tree is verified
with recursive rejection sampling over sibling candidates, which preserves the
target distribution.

## L1 suffix automaton

- Online construction (Blumer et al.) with amortized O(1) work per appended
  token.
- Capacity is fixed at startup: at most 2n − 1 states and 3n − 4 transitions
  for n tokens of context. States live in a preallocated arena; transitions in
  a preallocated open-addressing table keyed by (state, token), sized for a
  load factor of at most 0.5. Nothing is allocated after startup.
- Each state stores its length, suffix link and the end position of its first
  occurrence, so a match yields the tokens that followed that occurrence.
- The matching state is advanced as tokens are appended; a query returns the
  current match length and a continuation. A match of at least 4 tokens is a
  high-confidence seed ([ADR-0005](../decisions/0005-dual-source-tree-merging.md)).
- Latency target: under 0.05 ms per query (project charter; measured in M3 in
  [../benchmarks/results.md](../benchmarks/results.md#m3-speculative-decoding-and-spikes-s4--s5)).
- Larger-than-context reuse comes from N-gram tables on NVMe (status row C6).

## L2 draft sources

Per [ADR-0013](../decisions/0013-eagle-tree-and-head-formats.md):

- **EAGLE heads.** The tree is built with EAGLE-2's dynamic expansion: expand
  the top-k frontier nodes by cumulative draft confidence, then rerank all
  candidates and keep the best N. Two head formats are supported: EAGLE-3
  (fuses low-, mid- and high-level target features; reduced draft vocabulary
  with a token map) and EAGLE-2 (top-layer feature plus token embedding).
  Public EAGLE-3 heads exist for Qwen3-30B-A3B.
- **MTP heads.** Multi-token-prediction modules shipped with a model, chained
  beyond depth 1 like a small EAGLE head.
- **Same-family small draft.** A ≈ 0.5B-class model sharing the target's
  tokenizer (for example Qwen3-0.6B for Qwen3 targets) for models without
  EAGLE or MTP heads. It keeps its own KV cache, runs on the ANE, and does not
  need target features, which allows pipelined drafting.

EAGLE and MTP heads consume the target features of the accepted tokens, so
their drafting is serial with verification
([runtime-pipeline.md](runtime-pipeline.md#dependencies-and-overlap)).

## Dual-source tree merging

Accepted in [ADR-0005](../decisions/0005-dual-source-tree-merging.md):

- An L1 match of at least 4 consecutive tokens becomes a seed prefix: a chain
  attached to the root. L2 continues drafting from the seed's last node instead
  of the root; other L2 branches may still start at the root within the same
  budget.
- Shorter L1 candidates enter as ordinary low-priority branches. All branches
  share one prefix trie, so duplicate paths are stored and verified once.
- Seeding an EAGLE head runs the head through the seed tokens with its own
  predicted features, one step per seed token. Seeding the small draft model is
  a single batched forward over the seed.
- Target: lift the tokens accepted per verification pass on repetitive code
  (project charter; measured in M3 in
  [../benchmarks/results.md](../benchmarks/results.md#m3-speculative-decoding-and-spikes-s4--s5)).

## Expert-byte-budget pruning

Accepted in [ADR-0009](../decisions/0009-expert-byte-budget-tree-pruning.md).
For an MoE target, a verification pass reads the union of the experts its tree
activates, so the tree is priced in bytes, not nodes.

Cost of verifying tree T:

```
t(T) = t_fixed + |T| · c_tok + Σ_layers Σ_{e ∈ U_l(T)} b_e / β
U_l(T) = ∪_{v ∈ T} E_l(v)
```

- E_l(v): predicted experts of node v at layer l (ANE cross-layer prediction or,
  for L1 seed/branch and independent `D4` small-draft nodes lacking target
  hidden states, decoupled token-embedding + target router gate projection per
  [ADR-0005](../decisions/0005-dual-source-tree-merging.md)).
- b_e: bytes of expert e, plus an NVMe penalty when it is cold and not yet in a
  slot.
- β: effective bandwidth of the expert kernel variant used for |T| tokens,
  calibrated per chip; deduplicating and per-slot kernels have different β.
- c_tok, t_fixed: per-token compute and attention cost, fixed pass cost.

Value: V(T) = expected accepted tokens = Σ over nodes of the probability that
the path to the node is accepted (from draft confidences), plus the bonus
token.

Selection: choose the ancestor-closed subtree T with the best V(T) / t(T),
subject to |T| ≤ N_max (a compile-time cap). Expert coverage is monotone
submodular, so nodes are added greedily by marginal value per marginal cost —
each node pays only for experts it adds — while keeping ancestor closure. The
search runs after every draft level and also decides when to stop drafting.
For dense models U is empty and the rule reduces to a node budget.

## Verification on the GPU

- **Tree attention**: queries of all tree nodes attend to the committed KV
  tiles plus tree-local KV under an ancestor mask (node i sees node j only when
  j is i or an ancestor of i).
- **MoE**: all nodes are routed, node indices are bucketed per expert, each
  selected expert's weights are read once per pass, results are scattered back.
- **Commit**: the accepted path's KV rows are moved into the committed tiles by
  a GPU kernel; rejected rows are dropped.
- **Acceptance** runs in the same command buffer; accepted length, tokens and
  the features for EAGLE / MTP land in a shared buffer.
- Tree-width caps are compile-time variants
  ([ADR-0014](../decisions/0014-compile-time-specialization.md)).

## Sampling RNG and hardware entropy seeding

Confirmed on 2026-11-21 (capability row `D11` in [../status.md](../status.md)):
stochastic decoding (`temperature > 0`) separates hardware entropy acquisition at
request admission from deterministic, zero-syscall pseudo-random generation on
the per-token hot path:

- **Request admission (cold path, once per request, `< 0.3 us`).** CLI commands
  (`decode`, `spec-decode`) and M5 HTTP/SSE request payloads accept an optional
  seed (`seed: ?u64 = null`):
  - When `seed == null` and `temperature > 0`, the engine draws one 64-bit
    initial seed from macOS's **Apple Silicon Secure Enclave Processor (SEP)
    hardware TRNG-fed kernel CSPRNG** (`getentropy(&seed_bytes, 8)`, checking
    return codes explicitly with zero `catch {}`).
  - When `seed != null`, the caller-supplied 64-bit seed is used directly for
    bit-exact reproducibility.
  - Both CLI metrics and SSE response metadata echo `seed_used` so any
    stochastic completion can be replayed identically. Greedy runs (`T = 0`),
    `bench-*` suites, and unit tests always use explicit deterministic seeds.
- **Per-token / per-round hot path (`~1.2 ns/step`).** The 64-bit seed
  initializes a 32-byte `std.Random.Xoshiro256` (`DefaultPrng`) state
  preallocated inside the owning request slot. Recursive rejection sampling
  (`verifyStochastic`) and bonus-token categorical sampling advance only that
  slot-local state—incurring **zero syscalls, zero global locks, zero
  cross-request stream interference, and zero dynamic heap allocation**.

## Calibration and metrics

- Calibrated per chip at startup or offline: β per expert-kernel variant,
  c_tok, t_fixed, ANE hand-off latency, draft-head cost on ANE versus GPU.
- Recorded per run: acceptance-length histogram, unique experts per pass, bytes
  per accepted token, tokens per second, ANE and GPU utilization
  (`powermetrics`).

## Open questions

- Post-M5 ablation of adaptive L1 seed thresholds once persistent NVMe N-gram
  tables (`C6`) land (`SEED_MIN_MATCH_LEN = 4` stays fixed through M4/M4.5 per
  [ADR-0005](../decisions/0005-dual-source-tree-merging.md)).
- Optimal tree shape for multi-depth MTP heads on Tier 2 models (`Qwen3.8-Flash-Next`
  4B MTP in M6).
