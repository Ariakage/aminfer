# ADR-0022: Native C/Python SDK, full HTTP/SSE protocol parity, and rootless terminal monitor (M7.5)

- Status: Accepted
- Date: 2026-11-23
- Source: user roadmap confirmation following M7 M4-loopback close-out and live HF frontier metadata verification

## Context

Milestones M0–M6.8 are complete, M6.9 (`Apple10+` `NAX` dual-backend `.metallib`)
is code-complete pending `Apple10+` hardware, and M7 (`DeepSeek-V4.1-Flash-1017`
CED, MLA/CSA2, Mega-mHC, Engram, DSpark, and loopback TCP cluster) is verified
on M4 pending 512 GB Ultra or multi-Mac Thunderbolt 5 RDMA hardware
([../process.md](../process.md)). Three developer-facing gaps now govern daily
adoption on single-node and clustered Apple Silicon hosts:

1. **In-process native library and Python SDK (`C17`).** Today `aminfer` is
   invoked only through the standalone CLI binary or HTTP server (`C7`, `C8`).
   Python applications, evaluation harnesses, and native Swift/C/C++ hosts
   cannot embed the engine directly without spawning a subprocess or copying
   logits over sockets. Exposing an outward plain-C ABI (`libaminfer.dylib` +
   `include/aminfer.h`) and a thin zero-dependency Python `ctypes` wrapper
   (`python/aminfer/`) preserves [ADR-0001](0001-language-and-toolchain-boundaries.md):
   no Python ever enters the inference runtime, while external Python callers
   gain direct zero-copy access to native prefill, decode, logits, and vision
   buffers.
2. **Full three-endpoint HTTP/SSE protocol parity (`C18`).** M5 (`C8`) delivered
   a lean stateless `kqueue` SSE server (`POST /v1/chat/completions`, `GET
   /v1/models`, `GET /health`, and minimal stateless `/v1/responses` and
   `/v1/messages` adapters). Modern agent clients and SDKs (OpenAI Agents SDK,
   Anthropic Messages SDK, tool-calling coding agents) additionally expect
   end-to-end tool calling across all three endpoints, stateful
   `previous_response_id` chaining and management routes on `/v1/responses`,
   full Anthropic headers/blocks/thinking/token-counting on `/v1/messages`, and
   shared sampling, stop-sequence holdback, structured JSON output, and cache
   usage telemetry.
3. **Rootless Apple Silicon and `aminfer` observability (`C19`).** Diagnosing
   P/E-core saturation, wired-memory headroom against
   [ADR-0012](0012-wired-memory-budget-headroom.md) / `R9`, GPU utilization,
   thermal throttling, NVMe cold-expert/PLE/Engram I/O, and live `aminfer serve`
   slot/KV/cluster state currently requires combining `aminfer probe`,
   `vm_stat`, `sysctl`, and root-only `powermetrics`. A zero-dependency
   `aminfer monitor` terminal TUI (plus `--once` / `--json` snapshot mode) gives
   immediate rootless visibility using native macOS Mach, IOKit, and Foundation
   APIs.

The user confirmed codifying `C17`, `C18`, `C19`, `W18`, `C20`, `C21`, `C22`,
`W19`, and `R10` into **Milestone M7.5** ([../status.md](../status.md),
[../process.md](../process.md)), explicitly excluding audio/speech ingress from
this milestone and keeping `aminfer chat` (`C20`) as a plain scrolling CLI REPL
rather than a full-screen TUI.

## Decision

### 1. Native C-ABI shared library and zero-dependency Python `ctypes` SDK (`C17`, M7.5)

- **Build boundary and runtime isolation ([ADR-0001](0001-language-and-toolchain-boundaries.md)):**
  - `build.zig` adds a shared-library target producing `libaminfer.dylib` (with
    the unified `aminfer.metallib` embedded) paired with a plain-C public header
    `include/aminfer.h` using opaque handles, explicit `ami_status` return codes,
    fixed-width integer types, and caller-supplied or library-scoped buffers.
  - Zero Python runtime or CPython headers are linked into `libaminfer.dylib`.
  - A pure-stdlib Python package under `python/aminfer/` binds `libaminfer.dylib`
    via `ctypes` with zero third-party runtime dependencies (NumPy support is
    optional via the standard Python buffer protocol).
- **Lifecycle, co-tenant memory safety, and cancellation:**
  - Engine teardown explicitly invokes `munlock` on every wired region and drains
    Metal 4 Objective-C allocations inside an `@autoreleasepool` block so repeated
    create/destroy cycles inside a host process leave zero pinned or leaked pages.
  - Startup enforces a configurable co-tenant wired-budget ceiling alongside the
    `R9` live non-anonymous RAM admission guard
    ([ADR-0012](0012-wired-memory-budget-headroom.md),
    [ADR-0020](0020-wide-gpu-prefill-tail-pruning-qos-and-live-ram-guards.md)).
  - Cooperative request cancellation uses a thread-safe `ami_cancel_token` atomic
    flag; `libaminfer.dylib` never installs or overrides process-wide POSIX signal
    handlers.
- **Tokenization, generation, zero-copy views, and vision ingress:**
  - Expose native Zig BPE (`W5`, Qwen and DeepSeek dialects) `tokenize`,
    `detokenize`, and `count_tokens` entry points.
  - Provide synchronous and streaming generation callbacks for text tokens,
    reasoning deltas, and parsed tool-call deltas.
  - Expose read-only zero-copy `logits` (`fp32` / `fp16`) and final hidden-state
    views over preallocated wired buffers, surfaced in Python as `memoryview`
    (and optional NumPy buffer-protocol views) while releasing the Python GIL
    (`ctypes.CDLL`) across native prefill and decode calls.
  - Expose direct zero-copy `CVPixelBufferRef` and `IOSurfaceRef` vision ingress
    into the `G14` multimodal pipeline without intermediate pixel copies.

### 2. Full HTTP/SSE protocol parity across all three endpoints (`C18`, M7.5)

Extend the zero-steady-state-allocation `kqueue` HTTP/SSE server (`C7`, `C8`,
`C14`, `C15`) so `POST /v1/chat/completions`, `POST /v1/responses`, and
`POST /v1/messages` share one canonical parser, sampler, tool-call state machine,
and usage-accounting layer:

1. **End-to-end tool calling across all three endpoints:**
   - Parse and apply `tools`, `tool_choice` (`auto`, `none`, `required`, named
     tool), and `parallel_tool_calls`.
   - Accept multi-turn tool histories across all three wire schemas: OpenAI Chat
     `role: "tool"`, OpenAI Responses `function_call_output`, and Anthropic
     `tool_use` / `tool_result` blocks including `is_error`.
   - Run a zero-allocation streaming `<tool_call>` parser that emits
     `delta.tool_calls` (Chat Completions),
     `response.function_call_arguments.delta` (Responses), and
     `input_json_delta` (Anthropic Messages), terminating with
     `finish_reason: "tool_calls"` or `stop_reason: "tool_use"`.
2. **Full OpenAI `/v1/responses` lifecycle and bounded stateful chaining:**
   - Emit the complete SSE event sequence from `response.created` through
     `response.completed` (including output item, content part, text/reasoning,
     and function-call argument events), honoring `reasoning.effort` and
     `text.format`.
   - Back `previous_response_id` chaining with a preallocated, fixed-capacity
     wired response slot table anchored to `R8`/`C15` `PrefixCache` checkpoints
     (evicting oldest entries without dynamic heap growth), and serve
     `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}`, and
     `POST /v1/responses/{id}/cancel`.
3. **Full Anthropic `/v1/messages` parity and native token counting:**
   - Validate and honor `x-api-key` and `anthropic-version` headers, string or
     content-block-array `system` prompts with `cache_control` prefix pinning,
     `thinking` (`budget_tokens`) with `signature_delta`, and the complete
     `message_start` through `message_stop` SSE lifecycle.
   - Add `POST /v1/messages/count_tokens` backed directly by the native Zig BPE
     tokenizer (`W5`) with zero Python execution.
4. **Shared sampling, stop-sequence holdback, structured outputs, security, IPC, and telemetry:**
   - Support `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
     `repetition_penalty`, `logit_bias`, `logprobs` / `top_logprobs`, and
     `max_tokens` / `max_completion_tokens` / `max_output_tokens`.
   - Implement a zero-allocation streaming `stop` / `stop_sequences` prefix
     holdback ring so partial stop-string prefixes are buffered until
     disambiguated and matched stop sequences are never leaked to SSE clients.
   - Delegate `json_object`, `json_schema` (`strict: true`), and strict
     OpenAI/Anthropic tool-call schemas to the preallocated P-core grammar
     bitmask engine (`C22`).
   - Add opt-in constant-time `--api-key` authentication validating both
     `Authorization: Bearer <key>` and `x-api-key` headers without timing leaks,
     plus a `kqueue` `AF_UNIX` `--unix-socket <path>` listener alongside TCP for
     low-overhead local IPC.
   - Emit `stream_options.include_usage`, OpenAI `prompt_tokens_details.cached_tokens`
     and `completion_tokens_details.reasoning_tokens`, and Anthropic
     `cache_read_input_tokens` from `PrefixCache` (`R8`/`C15`) accounting.

### 3. `aminfer monitor` rootless terminal TUI and JSON monitor (`C19`, M7.5)

- **Architecture and zero-allocation rendering:**
  - Implemented in pure Zig plus minimal C/ObjC bridge helpers (`0` third-party
    C or Zig dependencies; [ADR-0001](0001-language-and-toolchain-boundaries.md)).
  - Event loop driven by `kqueue` (`EVFILT_TIMER`, `EVFILT_READ` for keyboard
    input, and `EVFILT_SIGNAL` for `SIGWINCH` resize handling) over a raw
    `termios` alternate-screen terminal.
  - Preallocated double-buffered cell and ANSI TrueColor + Unicode Braille
    (`⠀..⣿`) sparkline canvas performing zero per-frame heap allocation; diffed
    writes flush in one syscall per tick.
  - Non-interactive `--once` and `--json` flags emit a single deterministic
    telemetry snapshot for shell pipelines and automated unit tests.
- **Default `trans-pride` TrueColor theme and `btop` `.theme` compatibility:**
  - The built-in default TrueColor palette is **`trans-pride`** ("soft pink,
    white and blue on deep navy"), and `--theme <path.theme>` loads any standard
    `btop` `.theme` file (`theme[key]="#RRGGBB"`) at startup with zero
    steady-state allocation.
  - Built-in `trans-pride` base chrome:

| Theme key | Hex | Role |
| --- | --- | --- |
| `main_bg` | `#11131D` | Canvas background |
| `main_fg` | `#F7F7FC` | Primary text |
| `title` | `#F5A9B8` | Box titles |
| `hi_fg` | `#5BCEFA` | Highlighted keys and active badges |
| `selected_bg` | `#F5A9B8` | Selected row background |
| `selected_fg` | `#11131D` | Selected row foreground |
| `inactive_fg` | `#9297AC` | Muted labels and inactive rows |
| `graph_text` | `#F7F7FC` | Graph annotations |
| `meter_bg` | `#343849` | Unfilled meter track |
| `proc_misc` | `#F5A9B8` | Process/slot secondary indicators |
| `cpu_box` | `#5BCEFA` | CPU & SME2 panel border |
| `mem_box` | `#F5A9B8` | Unified memory panel border |
| `net_box` | `#DAEAF4` | GPU & thermal / network panel border |
| `proc_box` | `#5BCEFA` | NVMe & serve/cluster panel border |
| `div_line` | `#777F94` | Internal panel dividers |

  - Built-in `trans-pride` 3-stop smooth RGB gradients (`start -> mid -> end`):

| Gradient key | Start | Mid | End |
| --- | --- | --- | --- |
| `temp` | `#5BCEFA` | `#F7F7FC` | `#F5A9B8` |
| `cpu` | `#5BCEFA` | `#C1E5F7` | `#F5A9B8` |
| `free` | `#5BCEFA` | `#A5DFF5` | `#E6F6FF` |
| `cached` | `#E9B7D0` | `#D0CCED` | `#A1DDF5` |
| `available` | `#88D5F4` | `#C5E9F9` | `#F1FAFF` |
| `used` | `#F5A9B8` | `#E8BFDD` | `#B9DFF3` |
| `download` | `#5BCEFA` | `#B6E6F8` | `#F7F7FC` |
| `upload` | `#F5A9B8` | `#F7C8D8` | `#F7F7FC` |
| `process` | `#5BCEFA` | `#D2E7F4` | `#F5A9B8` |

- **Four Apple Silicon and `aminfer` live panels:**
  1. **CPU & SME2:** per-core utilization bars and Braille history split across
     P-cores and E-cores (`processor_info` `PROCESSOR_CPU_LOAD_INFO`),
     `ChipProfile` (`C16`) badge, and host load/process summary.
  2. **Unified Memory, Wired Budget & Zero-Swap Pool:** live `HOST_VM_INFO64`
     breakdown (`wire` vs [ADR-0012](0012-wired-memory-budget-headroom.md)
     `user_wire_limit` / `iogpu.wired_limit_mb`, `R9` reclaimable
     `free + speculative + purgeable + external`, anonymous active/inactive,
     `compressor`, and `vm.swapusage` / `swapouts` delta indicator).
  3. **Rootless Metal GPU & Thermal State:** unprivileged `IOKit`
     `IOAccelerator` (`PerformanceStatistics`: `Device Utilization %`,
     `In use system memory`, `Alloc system memory`) paired with
     `NSProcessInfo.thermalState` (`Nominal` / `Fair` / `Serious` / `Critical`,
     zero `sudo` required).
  4. **NVMe Storage & Live `aminfer serve` / Cluster Telemetry:**
     `IOBlockStorageDriver` read/write `MB/s` and IOPS, plus optional
     `--server <host:port>` live attach polling extended `/health` (or
     `/v1/metrics`) for `C14` active prefill/decode slots, `tok/s`,
     `decode-share` QoS (`C15`), `PrefixCache` (`R8`/`C15`) hit rate, `R5`
     retired `q8` KV chunks, active `.ang` pack, and `X1`–`X3` Thunderbolt peer
     status.
  - Interactive keybindings: `q` / `Esc` to quit, `+` / `-` to adjust refresh
    interval, `1`–`4` to toggle panels, and `r` to reset cumulative deltas.

### 4. Self-contained single-file `.amf` metadata and `ls` / `inspect` CLI (`W18`, M7.5)

- Extend `.amf` v0 packing (`W1`–`W3`, [ADR-0006](0006-amf-cold-experts-direct-io.md))
  to embed `config.json` and `tokenizer.json` inside dedicated 16 KiB-aligned
  metadata sections within a single `.amf` file while maintaining full
  backward compatibility with sidecar files and explicit `--config` /
  `--tokenizer` CLI overrides.
- Add `aminfer ls` and `aminfer inspect <model.amf>` subcommands to list local
  `.amf` models and print architecture axes, quantization breakdown, embedded
  metadata CRC/presence, wired-memory estimates, and hardware-tier fit without
  loading weight tensors into wired RAM.

### 5. Simple streaming CLI REPL `aminfer chat` and `stdin` pipe mode (`C20`, M7.5)

- Add `aminfer chat` as a **plain scrolling terminal line-REPL** (explicitly
  NOT an alternate-screen full-screen TUI; full-screen terminal rendering
  belongs exclusively to `aminfer monitor` `C19`) paired with non-interactive
  `stdin` pipe mode.
- Features: inline streaming token output, optional dimmed `<think>` reasoning
  output, a compact 1-line post-turn summary (`tok/s`, TTFT, `PrefixCache`
  reuse), interactive slash commands `/clear`, `/retry`, and `/edit` (reusing
  `C15` ChatML message-boundary `PrefixCache` checkpoints), and `/image` /
  `/video` attachments routed through `G14` zero-copy vision ingress.

### 6. Native embeddings and reranking (`C21`, M7.5)

- Add `POST /v1/embeddings` and `POST /v1/rerank` to `aminfer serve` plus
  matching `embed()` and `rerank()` methods in the `C17` C/Python SDK.
- Reuse the `G15` / `G17` wide prefill path while eliding `lm_head`, supporting
  `last_token`, `mean`, and `cls` pooling, Matryoshka `dimensions` prefix
  slicing, L2 normalization, `float` and `base64` wire encodings, and
  cross-encoder rerank logit scoring with zero steady-state heap allocation.

### 7. Preallocated grammar, JSON Schema, and GBNF token-bitmask engine (`C22`, M7.5)

- Compile `json_object`, `json_schema` (`strict: true` structured outputs and
  OpenAI/Anthropic strict tool schemas in `C18`), and GBNF grammars at request
  admission into a preallocated pushdown state machine and `vocab_size`
  allowed-token bitmask.
- Evaluate the bitmask on P-cores with zero per-token heap allocation and
  integrate mask advancement with `G5` / `D9` speculative tree verification so
  drafted branches remain strictly grammar-valid and distribution-preserving.

### 8. Runtime multi-LoRA hot-loading and per-request adapter routing (`W19`, M7.5)

- Generalize `W16` `sme2_panel64` `Step-LoRA` adapters to standard Hugging Face
  PEFT LoRA checkpoints (`.amlora` / safetensors offline import and runtime hot
  loading within a `<= 4 MiB` wired L2 pool on `Qwen3Model` / `BatchRuntime`).
- Route `"model": "<base>:<adapter>"` per request across concurrent `C14`
  continuous-batching slots over a single shared wired base `.amf`, eliminating
  base-weight duplication across adapters. Extension to non-`Qwen3` `GraphModel`
  and `M82Runtime` architectures and DoRA checkpoints is specified in
  [section 11](#11-universal-graphmodel-and-m82runtime-multi-lora-and-dora-hot-loading-w30-m95)
  (`W30`, M9.5).

### 9. Opt-in NVMe persistent prefix KV and GDN cache (`R10`, M7.5)

- Add opt-in `--disk-prefix-cache <dir>` enabling asynchronous 16 KiB-aligned
  `F_NOCACHE` disk spill and reload of compressed `R5` `q8` / `R7` `FP4` KV
  tiles and `GDN` recurrent state snapshots.
- Key every persisted prefix artifact by model `.amf` CRC, prefix hash, and
  full token-slice verification so cold-restart recovery never collides across
  models, quantization profiles, or diverged prefixes.

### 10. Model detail discovery and native TLS 1.2/1.3 HTTPS serving (`C28`, M8.3)

- **Startup `ModelDetailMeta` and `LoraAdapterMeta` discovery (`GET /v1/models` and `GET /v1/models/{id}`):**
  - At startup, `aminfer serve` populates immutable `ModelDetailMeta` from `AmfInspection` and `ArchAxesSummary` (`architecture`, `max_model_len`, `vocab_size`, `effective_bits_per_weight`, `wired_mib`, `cold_bytes`, and `ModelCapabilities` flags including `vision`, `reasoning`, `moe`, `lora`, `embeddings`, and `rerank`) alongside `LoraAdapterMeta` entries for every mounted `-L`/`--lora` `<base>:<adapter>` adapter (`parent`, `rank`, `lora_alpha`, `scale`, `wired_bytes`).
  - `GET /v1/models` formats the base model and all mounted adapter records into a preallocated `16 KiB` buffer (`0` steady-state heap allocations), and `GET /v1/models/{id}` performs zero-allocation URL percent-decoding (`%2F`, `%3A`) so slash- and colon-delimited IDs resolve directly, returning `405` on non-`GET` methods, `401` on missing/invalid `--api-key`, and `404 model_not_found` on unknown IDs.
- **Zero-dependency native `Network.framework` + `Security.framework` TLS 1.2/1.3 listener:**
  - `aminfer serve` accepts `--tls-cert`, `--tls-key`, `--tls-key-passphrase`, and `--tls-passphrase-env` without adding any third-party TLS library ([ADR-0001](0001-language-and-toolchain-boundaries.md)).
  - `src/bridge/apple_bridge.{h,m}` and `src/bridge/tls.zig` parse PEM (single certificate or fullchain paired with PKCS#8, PKCS#1, or SEC1 keys, including encrypted PBES2, PBES1, PKCS#12-PBE, and legacy PEM) or `.p12` bundles exclusively in memory via `Security.framework` (`kSecImportToMemoryOnly`), verifying certificate/private-key public-key match with zero persistent Keychain side effects. Passphrase and key buffers are wiped immediately after `Server.init` via `memset_s` / `secureZero`, and secrets are never logged.
  - A `Network.framework` `nw_listener` configured for TLS 1.2 and TLS 1.3 hands each accepted client to a pure-C refcounted `socketpair(AF_UNIX, SOCK_STREAM)` pump (`ami_tls_pump`) on a `DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM` serial queue and signals the `kqueue` event loop through a non-blocking `notify_fd` (`UDATA_LISTEN`).
  - `http_sse.zig` reuses the existing preallocated connection slots, HTTP/1.1 `keep-alive`, SSE streaming across `/v1/chat/completions`, `/v1/responses`, and `/v1/messages`, `503` saturation responses drained cleanly with TLS `close_notify`, constant-time `--api-key` auth, and concurrent plaintext `AF_UNIX` `--unix-socket` serving with `0` hot-path heap allocations.

### 11. Universal `GraphModel` and `M82Runtime` multi-LoRA and DoRA hot-loading (`W30`, M9.5)

- **Extending `W19` `.amlora` and `LoraRegistry` beyond `Qwen3Model`:** Extend
  runtime multi-LoRA hot-loading (`-L`/`--lora <name>=<path.amlora>` and
  `"<base>:<adapter>"` routing across `aminfer chat`, `aminfer serve`, `C28`
  `GET /v1/models/{id}`, and the `C17` C/Python SDK) from `Qwen3Model` /
  `BatchRuntime` to `GraphModel` and `M82Runtime`, covering `inkling2` /
  `inkling2_loop` (`wq_du`, `wk_dv`, `wv_dv`, `wo_ud`, and `gate_attn` shared
  and per-loop-pass projections, including `Inkling-2-Mini-Unbound`
  `peft-r16`/`peft-r64` and DoRA adapters), `hy_v4`, `mimo_v2`,
  `glm_moe_dsa`/`glm5_next`, `bailing_hybrid`, and `deepseek41`.
- **Zero-flush GPU low-rank fusion and `SME2` coexistence:** Add a single-frame
  Metal 4 GPU low-rank delta path (`B(A x)` plus optional DoRA row-wise
  `lora_magnitude_vector` rescaling) that preserves `G29` `0` mid-pass
  `flushFrame()`
  ([ADR-0026](0026-m825-inkling2-single-frame-hot-path-and-full-wired-quant-parity.md))
  alongside `C24` `SME2` `sme2_panel64` offload for small L2-resident adapters,
  raising an optional configurable wired pool cap inside the `R9` wired budget
  for `r = 16..64` adapters without unpinning base weights.
- **Offline PEFT `r <= 64` and DoRA `.amlora` import:** Extend
  `tools/src/aminfer_tools/lora.py` to import Hugging Face PEFT `r <= 64` and
  DoRA (`lora_magnitude_vector`) adapters into 16 KiB-aligned `.amlora`
  containers while preserving zero-overhead fast-path bypass when
  `active_adapters == 0`.

### 12. Full local OpenAI and Anthropic HTTP/SSE endpoint and multimodal parity (`C29`, M9.6)

Scheduled as capability `C29` (`Status: Confirmed`, `Target: M9.6`, and
user-prioritizable ahead of `M9.1`–`M9.5` because it is pure serving and
multimodal ingress with no dependency on frontier model-family converters), this
extension closes the remaining local-inference OpenAI/ChatGPT and Anthropic
HTTP/SSE protocol gaps while preserving `0` hot-path heap allocations:

1. **Anthropic Models API parity (`GET /v1/models` and `GET /v1/models/{model_id}`):**
   Preserve every OpenAI and `aminfer` `C28` metadata field (`object`,
   `created`, `owned_by`, `root`, `parent`, `architecture`, `max_model_len`,
   `vocab_size`, `effective_bits_per_weight`, `wired_mib`, `cold_bytes`,
   `capabilities`) while adding Anthropic Models API fields (`type: "model"`,
   `display_name`, RFC 3339 `created_at`, `max_input_tokens`, `max_tokens`) and
   list pagination envelope fields (`first_id`, `last_id`, `has_more`) with
   `limit`, `before_id`, and `after_id` query parameters, formatted strictly
   inside preallocated `16 KiB` static buffers.
2. **Prefix-free route aliases (without `/v1`):**
   Support `GET /models`, `GET /models/{id}`, `POST /messages`,
   `POST /messages/count_tokens`, `POST /chat/completions`,
   `POST /completions`, `POST /responses`, `POST /responses/input_tokens`,
   `GET|DELETE /responses/{id}`, `POST /responses/{id}/cancel`,
   `GET /responses/{id}/input_items`, `POST /embeddings`, and `POST /rerank`
   under the same constant-time `--api-key` authentication guard as `/v1/*`
   (`/health` stays unauthenticated).
3. **OpenAI Responses input token counting (`POST /v1/responses/input_tokens` and `/responses/input_tokens`):**
   Mirror `POST /v1/messages/count_tokens` using the native Zig BPE tokenizer
   (`W5`) and declarative `ChatTemplate` renderer (`W24`) for zero-inference
   token counting over `input`, `instructions`, `tools`, and
   `previous_response_id`.
4. **OpenAI Stateful Responses input-items listing (`GET /v1/responses/{response_id}/input_items` and `/responses/{response_id}/input_items`):**
   Expose stored and paginated input items from `StatefulResponseSlot` alongside
   `GET`, `DELETE`, and `cancel` without introducing dynamic per-request heap
   allocations.
5. **OpenAI Legacy and FIM Text Completions (`POST /v1/completions` and `/completions`):**
   Provide a zero-allocation raw-prompt completion endpoint supporting both
   streaming SSE and JSON responses, string or token-ID array `prompt`, `suffix`
   Fill-in-the-Middle generation, and shared sampler, penalties, `logit_bias`,
   `logprobs`, `stop`, and `seed` parameters.
6. **HTTP `OPTIONS` and CORS preflight headers:**
   Handle `OPTIONS` requests across routes and emit configurable
   `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods`, and
   `Access-Control-Allow-Headers` (`Authorization`, `x-api-key`,
   `anthropic-version`, `Content-Type`) headers for direct browser WebUI
   connectivity over both plaintext HTTP and native TLS 1.2/1.3 HTTPS (`C28`).
7. **HTTP multimodal vision media pixel ingestion (`image_url` / `input_image` / Anthropic `image` / `video_url`):**
   Upgrade `http_sse.zig` from template-placeholder-only parsing to bounded
   preallocated base64 (`data:image/...;base64,...`) image and video-frame
   decoding wired into `G14` `ImageIO`/`VideoToolbox`/`CVPixelBuffer` zero-copy
   IOSurface ingress and the `W20` vision encoder across
   `POST /v1/chat/completions`, `POST /v1/responses`, and `POST /v1/messages`.
8. **Explicit non-goals (excluded by design):**
   Cloud-hosted storage, batch-job orchestration, and non-LLM endpoints — Audio
   (`/v1/audio/*`), Images (`/v1/images/*`), Moderations (`/v1/moderations`),
   Files (`/v1/files*`), Batches (`/v1/messages/batches*`),
   Assistants/Threads/VectorStores, and Fine-tuning — are explicitly excluded
   from `aminfer`'s bare-metal local LLM/VLM serving scope. None of them fit a
   zero-allocation, zero-database, single-model wired-RAM inference server, and
   omitting them keeps `src/server/` lean, auditable, and strictly bounded in
   memory.

### Scope boundary and zero-overhead non-regression guardrails

- Cloud-hosted and non-LLM API families — Audio (`/v1/audio/*`), Images
  (`/v1/images/*`), Moderations (`/v1/moderations`), Files (`/v1/files*`),
  Batches (`/v1/messages/batches*`), Assistants/Threads/VectorStores, and
  Fine-tuning — are explicitly excluded from Milestones M7.5, M8.3, and M9.6.
- **Strict zero-overhead fast paths when optional M7.5 / M8.3 / M9.5 / M9.6 features are inactive:**
  1. `aminfer monitor` (`C19`) and `aminfer chat` (`C20`) are standalone CLI subcommands; `aminfer serve` and `decode` run zero `IOReport` polling when `monitor` is not running, and attaching `monitor --server` polls `/health` + rootless `IOReport` on `QOS_CLASS_UTILITY` E-cores at `>= 500 ms` intervals (`< 0.1%` E-core CPU, `0 B` GPU/wired bandwidth).
  2. Single-file `.amf` metadata (`W18`), `--api-key`, `--unix-socket` (`C18`), `ModelDetailMeta`/`LoraAdapterMeta` discovery, opt-in native TLS 1.2/1.3 HTTPS (`C28`), and `C29` HTTP protocol extensions (Anthropic Models pagination, prefix-free route aliases, `OPTIONS`/CORS headers, `/v1/responses/input_tokens`, `/v1/responses/{id}/input_items`, and `/v1/completions` FIM) execute only at startup or at the HTTP/TLS connection boundary using static preallocated buffers (`0` hot-path heap allocations, `0` instructions added to text-only per-token decode, and `0` TLS listener overhead when `--tls-cert` is omitted); `C29` base64 vision decoding (`G14`/`W20`) is bypassed completely on text-only requests; `libaminfer.dylib` (`C17`) is a separate build target that leaves the static `aminfer` CLI binary unchanged.
  3. Unconstrained requests (`response_format` unset, penalties `0`, `logit_bias` empty, `active_adapters == 0`) bypass `C22` grammar state machines and `W19`/`W30` LoRA adapter dispatches outside the layer loop. When `C22` / penalties are active on large-vocabulary models (`151K–248K` tokens), `C22` pairs with the fused GPU radix/bucket sampler (`G21` in [ADR-0023](0023-native-kernel-barrier-fusion-mma-flashattn-and-simd-accelerators.md)) and propagates masks into speculative draft proposals so neither CPU logit scans nor draft rejections degrade throughput.
  4. `W19` L2-resident SME2 LoRA adapters cap their P-core L2 footprint at `<= 4 MiB` (`25%` of M4's `16 MiB` P-core L2) inside the `R9` wired budget (never unpinning base weights), while `W30` adds an explicit configurable wired pool cap inside `R9` for `r = 16..64` `GraphModel` / `M82Runtime` GPU/SME2 and DoRA adapters without unpinning base weights. Stateful `/v1/responses` slots obey `R9` `evict_first` governance, and opt-in `--disk-prefix-cache` (`R10`, off by default) schedules `F_NOCACHE` writes on background utility I/O queues that yield strictly to `C3`–`C5` decode-critical cold-expert `pread`s.

## Consequences

- **Positive:** Python and native hosts can embed `aminfer` in-process with zero
  weight/logit copies and zero Python inside the engine (`C17`); standard
  OpenAI, Anthropic, browser WebUI, FIM, multimodal VLM, tool-calling,
  embedding, reranking, and multi-LoRA clients work directly against
  `aminfer serve` over TCP, native TLS 1.2/1.3 HTTPS, or `AF_UNIX` with
  `--api-key` auth, `/v1/models/{id}` discovery, CORS preflight, prefix-free
  aliases, and strict grammar masks (`C18`, `C21`, `C22`, `C28`, `C29`, `W19`);
  single-file `.amf` bundles (`W18`), scrolling `aminfer chat` (`C20`), NVMe
  persistent prefix recovery (`R10`), and rootless `aminfer monitor` (`C19`)
  cover daily local workflows without third-party runtime dependencies.
- **Negative / trade-offs:** Maintaining a stable C ABI (`include/aminfer.h`),
  stateful `/v1/responses` slot retention, grammar-constrained decoding,
  multi-LoRA slot routing, persistent KV disk validation, and three-endpoint
  SSE formatters adds surface area in `src/runtime/`, `src/server/`,
  `src/format/`, and `src/bridge/`. Preallocated fixed-slot tables keep memory
  bounded and zero-alloc on the hot path, while capping concurrent stateful
  responses, active LoRA adapters, and grammar stack depth to startup limits.

## Evidence

- Invariant baselines: [ADR-0001](0001-language-and-toolchain-boundaries.md),
  [ADR-0011](0011-metal4-zero-allocation-hot-path.md),
  [ADR-0012](0012-wired-memory-budget-headroom.md),
  [ADR-0020](0020-wide-gpu-prefill-tail-pruning-qos-and-live-ram-guards.md),
  [ADR-0021](0021-multi-generation-apple-silicon-and-nax-tensorops.md).
- Existing `kqueue` server (`C7`, `C8`, `C14`, `C15`), live `HOST_VM_INFO64`
  probe (`R9`), `ChipProfile` (`C16`), and native Zig Qwen/DeepSeek BPE
  tokenizer (`W5`) verified on the M4 baseline in
  [../benchmarks/results.md](../benchmarks/results.md).
- Reference palette specification: user-supplied `trans-pride.theme` (`btop`
  format, soft pink, white and blue on deep navy).

## Open items

- Finalize exact `include/aminfer.h` struct versioning (`struct_size` +
  `abi_version`) and bounded default capacity for the `/v1/responses`
  `previous_response_id` wired slot table during M7.5 implementation.
- Measure `json_schema` logit-mask latency on `M = 1..8` decode slots to confirm
  zero regression on steady-state token latency and zero hot-path heap
  allocation.
