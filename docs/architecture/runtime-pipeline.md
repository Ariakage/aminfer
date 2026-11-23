# Runtime pipeline

> Design reference; status lives in [../status.md](../status.md).

## Request lifecycle

1. The kqueue-based server accepts a request and streams results back as
   Server-Sent Events (**confirmed `C8` API contract, 2026-11-21**, status
   tracked in [../status.md](../status.md)):
   - Core stateless endpoints: **`POST /v1/chat/completions`** (OpenAI-compatible
     SSE `chat.completion.chunk`, `<think>...</think>` separated into
     `choices[0].delta.reasoning_content` / `message.reasoning_content`, optional
     `"seed"` u64 field with `seed_used` echoed in response metadata),
     **`GET /v1/models`**, and **`GET /health`**.
   - Optional thin stateless protocol adapters for `POST /v1/responses` and
     `POST /v1/messages` share the same zero-allocation `kqueue` SSE token
     emitter without server-side session storage, maximizing L1 suffix-automaton
     prefix hits (`D1`; rationale in
     [open-decisions-research.md](open-decisions-research.md#c8-kqueue-http--sse-server-api-shape)).
   - Per [ADR-0022](../decisions/0022-native-c-python-sdk-and-full-protocol-parity.md)
     (status tracked in [../status.md](../status.md)), the planned M7.5
     extensions add full three-endpoint protocol parity (`C18`: streaming tool
     calling, bounded wired-slot `previous_response_id` chaining and
     `GET`/`DELETE`/`cancel` on `/v1/responses/{id}`, Anthropic `system`
     `cache_control` / `thinking` `budget_tokens` + `POST /v1/messages/count_tokens`,
     shared sampling, zero-alloc streaming stop holdback, constant-time
     `--api-key` auth, and `AF_UNIX` `--unix-socket`), a preallocated
     `json_schema`/GBNF token-bitmask engine compatible with `G5`/`D9`
     speculative tree verification (`C22`), native `/v1/embeddings` and
     `/v1/rerank` (`C21`), runtime multi-LoRA `"<base>:<adapter>"` routing over
     one shared wired base `.amf` (`W19`), opt-in `--disk-prefix-cache` NVMe
     KV/GDN persistence (`R10`), self-contained single-file `.amf` metadata
     with `aminfer ls`/`inspect` (`W18`), a plain scrolling `aminfer chat` CLI
     REPL + `stdin` pipe (`C20`), an outward plain-C shared library
     (`libaminfer.dylib` + `include/aminfer.h`) with a zero-dependency Python
     `ctypes` SDK (`C17`, preserving
     [ADR-0001](../decisions/0001-language-and-toolchain-boundaries.md)), and
     the rootless `aminfer monitor` terminal TUI (`C19`, default `trans-pride`
     palette + `btop` `.theme` support and `--server` live attach).
   - On request admission, when `temperature > 0` and `"seed"` is omitted, one
     64-bit seed is drawn from Apple Silicon's Secure Enclave TRNG-backed kernel
     CSPRNG (`getentropy`) to initialize that request slot's dedicated 32-byte
     `std.Random.Xoshiro256` PRNG (`D11`,
     [speculative-decoding.md](speculative-decoding.md#sampling-rng-and-hardware-entropy-seeding)).
2. A native Zig tokenizer turns text into token ids (status row W5).
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
unavailable, and as the baseline the hybrid path must beat. Per
[ADR-0020](../decisions/0020-wide-gpu-prefill-tail-pruning-qos-and-live-ram-guards.md)
(`G15`, M6.8), `--gpu-prefill-mode wide|tree64` (default `wide`) executes
`512–1024`-token chunks through dedicated Metal 4 FlashAttention prefill kernels
(`BQ = 32, BK = 32`, `15 KiB` threadgroup SRAM for `2x` occupancy and
`MTL_SHADER_VALIDATION=1` compliance, skipping upper-triangular KV blocks and
writing K/V directly into `KvTile`s) plus `BM = 64, BN = 64` prefill GEMM tiles, and prunes
unused final-layer (`L - 1`) `O_proj` / `MoE` / `lm_head` work on non-terminal
prompt positions (`prompt_tail`); `tree64` retains the legacy `<= 64`-row
`verifyTree` + `commitTreeKv` path for A/B comparison.

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
- **Pipelined drafting (`D8`, confirmed 2026-11-21).** The independent small
  draft model does not need target features, so round `r + 1` can be drafted
  from the most likely outcome of round `r` while the GPU verifies (PEARL-style;
  a wrong guess is discarded). Per
  [ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md), overlapped
  small-draft pipelining is enabled selectively when target verification latency
  is long enough to amortize `~575 us` ANE handoff and unified-DRAM contention
  (for example dense `Qwen3-14B` at `~95 ms/step`), leaving fast MoE targets
  with in-command-buffer EAGLE-3 (`~19 ms/step`) un-pipelined.
- **Placement gate.** EAGLE / MTP heads on the GPU avoid extra
  synchronization; on the ANE they cost two GPU → CPU → ANE hand-offs per
  round; compact L2-resident projection subgraphs and `Step-LoRA` adapters can
  also target P-core SME2 (`Placement.cpu_sme2`, `D13`,
  [ADR-0019](../decisions/0019-recurrent-depth-and-post-m5-accelerators.md)).
  The default is chosen per model and chip from measured round latency
  ([ADR-0008](../decisions/0008-benchmark-gated-draft-placement.md)).
- **Iteration-level continuous batching, QoS, and prefix sharing (`C14`, `C15`,
  `R8`, M6.5–M6.8).** M5 `aminfer serve` executes preallocated request slots
  without steady-state heap allocation; M6.5 batches `M > 1` active slots into
  shared GPU GEMM / CSR-MoE and SME2 `ZA` passes while sharing read-only prefix
  `KvTile`s and snapshot-cloned `GDN` recurrent states across concurrent
  requests ([ADR-0019](../decisions/0019-recurrent-depth-and-post-m5-accelerators.md)).
  M6.8 (`C15`,
  [ADR-0020](../decisions/0020-wide-gpu-prefill-tail-pruning-qos-and-live-ram-guards.md))
  adds default-on toggles for a wall-time `--decode-share` QoS ledger,
  `--adaptive-prefill-narrow` depth-scaled chunk narrowing (`256` at `8K+`,
  `128` at `16K+` while decode slots are active), and `--message-boundary-cache`
  evict-first ChatML pre-last-user `PrefixCache` checkpoints.

## Threading model

| Thread | Count | QoS | Role |
| --- | --- | --- | --- |
| Scheduler | 1 | `USER_INTERACTIVE` | Round orchestration, tree merging and pruning, GPU submission |
| SME2 workers (M4.5) | 1–N (per P-cluster) | `USER_INTERACTIVE` | Lock-free P-cluster-aware ARMv9 SME2 pre-packed & `ZT0`-LUT microkernels ([ADR-0018](../decisions/0018-bare-metal-armv9-sme2-engine.md)) |
| L1 | 1 | `USER_INTERACTIVE` | Suffix automaton queries and updates |
| ANE driver | 1 | `USER_INITIATED` | CoreML predictions (draft, attention blocks, expert prediction) |
| I/O | 1–N | `UTILITY` + `IOPOL_IMPORTANT` | `aio_read` submission and `kqueue` completion; `pread` workers as fallback |
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
