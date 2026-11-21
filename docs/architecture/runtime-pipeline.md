# Runtime pipeline

> Design reference; status lives in [../status.md](../status.md).

## Request lifecycle

1. The kqueue-based server accepts a request and streams results back as
   Server-Sent Events. The external API shape (for example an
   OpenAI-compatible endpoint) is undecided (status row C8; comparison of
   `/v1/chat/completions`, `reasoning_content`, stateless `/v1/responses`, and
   `/v1/messages` in
   [open-decisions-research.md](open-decisions-research.md#c8-kqueue-http--sse-server-api-shape)).
2. A native Zig tokenizer turns text into token ids (status row W5, Draft).
   No Python runs at any point.
3. Prefill builds the KV cache: the hybrid wavefront prefill for prompts longer
   than 512 tokens, the GPU-only prefill otherwise.
4. The speculative decode loop produces tokens until a stop condition.
5. Detokenized text is emitted as SSE events while decoding continues.

## Prefill

### Hybrid wavefront prefill

Accepted in [ADR-0010](../decisions/0010-hybrid-wavefront-prefill.md). The
prompt is cut into 512-token chunks — a static CoreML graph shape. Each
transformer layer ℓ of each chunk c is split in two:

- **A(ℓ, c) on the ANE** — QKV projection, attention over the KV of chunks
  ≤ c, output projection. The chunk's K and V are written straight into the
  layer's IOSurface KV tiles
  ([ADR-0004](../decisions/0004-kv-cache-iosurface-texture-sharing.md)).
- **F(ℓ, c) on the GPU** — FFN or MoE (expert gather) plus the residual.

Dependencies: A(ℓ, c) needs F(ℓ − 1, c) and the K/V of chunk c − 1 at layer ℓ;
F(ℓ, c) needs A(ℓ, c).

Schedule: two chunks are in flight, the earlier one running one layer ahead.
While the GPU runs F(ℓ, c + 1), the ANE runs A(ℓ + 1, c); next the ANE runs
A(ℓ + 1, c + 1) while the GPU runs F(ℓ + 1, c), and so on. Every dependency is
already satisfied when a step starts, so both engines stay busy and the steady
state costs about max(t_A, t_F) per layer and chunk instead of t_A + t_F. When
the leading chunk leaves the last layer, the next chunk enters layer 0 behind
the trailing one.

Hand-offs between the engines go through IOSurface-backed activations (no
copies) and are synchronized with shared events; their latency is measured in
spike S4 ([../process.md](../process.md#spikes)).

The ANE holds only the attention-block weights (CoreML, palettized), not a
second copy of the model — see [memory.md](memory.md#budget-estimates).

### GPU-only prefill

Used for prompts of at most 512 tokens, as the fallback when the ANE path is
unavailable, and as the baseline the hybrid path must beat.

## Speculative decode loop

One round:

1. **L1 (P-core)** — query the suffix automaton with the latest tokens. A match
   of at least 4 tokens becomes a seed prefix
   ([ADR-0005](../decisions/0005-dual-source-tree-merging.md)).
2. **L2** — build a draft tree that continues from the accepted prefix plus the
   seed: an EAGLE-2 dynamic tree driven by an EAGLE-3 or EAGLE-2 head, MTP
   heads, or the same-family small draft model
   ([ADR-0013](../decisions/0013-eagle-tree-and-head-formats.md)). Placement
   follows [ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md).
3. **Expert prediction (ANE)** — predicted expert sets per tree node and
   layer.
4. **Pruning (P-core)** — keep the subtree with the best expected accepted
   tokens per predicted verification time
   ([ADR-0009](../decisions/0009-expert-byte-budget-tree-pruning.md)); queue
   the cold experts it needs.
5. **Prefetch (E-cores)** — cold experts are read from NVMe into 16 KiB slots
   ([ADR-0006](../decisions/0006-amf-cold-experts-direct-io.md)).
6. **L3 (GPU)** — one verification pass over the whole tree: tree attention,
   fused dequantization, MoE gather with experts deduplicated across nodes.
7. **Accept** — take the longest accepted path plus the bonus token, commit its
   KV rows, update the suffix automaton, and hand the target features to the
   EAGLE / MTP head.

### Dependencies and overlap

- **Strict dependency.** EAGLE and MTP heads draft round r + 1 from the target
  features of round r's accepted tokens, so with these heads drafting and
  verification are serial.
- **Overlappable work.** L1 queries and updates, expert prediction for the next
  tree, NVMe prefetch, SSE emission and detokenization run alongside GPU
  verification.
- **Pipelined drafting.** The independent small draft model does not need
  target features, so the ANE can draft round r + 1 from the most likely
  outcome of round r while the GPU verifies (PEARL-style); a wrong guess is
  discarded (status row D8).
- **Placement gate.** EAGLE / MTP heads on the GPU avoid extra
  synchronization; on the ANE they cost two GPU → CPU → ANE hand-offs per
  round. The default is chosen per model and chip from measured round latency
  ([ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md)).

## Threading model

| Thread | Count | QoS | Role |
| --- | --- | --- | --- |
| Scheduler | 1 | `USER_INTERACTIVE` | Round orchestration, tree merging and pruning, GPU submission |
| L1 | 1 | `USER_INTERACTIVE` | Suffix automaton queries and updates |
| ANE driver | 1 | `USER_INITIATED` | CoreML predictions (draft, attention blocks, expert prediction) |
| I/O | 1–N | `UTILITY` + `IOPOL_IMPORTANT` (Draft) | `aio_read` submission and `kqueue` completion; `pread` workers as fallback |
| Server | 1 | `USER_INITIATED` | kqueue HTTP / SSE |

Threads exchange work through preallocated lock-free SPSC / MPSC rings (status
row C1). The wake-up mechanism — `kqueue` `EVFILT_USER`, `MTLSharedEvent`
listeners or `os_sync_wait_on_address` — is chosen in M2. QoS only steers
placement ([hardware.md](hardware.md#core-steering)).

## Hot-path rules

The hot path is everything that runs once per decode round or per prefill
chunk.

- No heap allocation in our code: Zig allocators are used only during startup
  and teardown.
- GPU work is encoded with reusable Metal 4 command buffers whose command
  allocators are reset per frame in flight
  ([ADR-0011](../decisions/0011-metal4-zero-allocation-hot-path.md)). Apple
  framework internals (for example CoreML prediction calls) may still allocate;
  that residue is bounded by reusing feature providers and output backings and
  measured in spike S2.
- No page faults on wired memory, no blocking system calls on P-core threads
  other than waits on events, no shader compilation (every pipeline is built at
  startup).
