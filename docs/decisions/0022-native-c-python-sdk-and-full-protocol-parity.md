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

The user confirmed codifying `C17`, `C18`, and `C19` into **Milestone M7.5**
([../status.md](../status.md), [../process.md](../process.md)).

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
4. **Shared sampling, stop-sequence holdback, structured outputs, and telemetry:**
   - Support `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
     `repetition_penalty`, `logit_bias`, `logprobs` / `top_logprobs`, and
     `max_tokens` / `max_completion_tokens` / `max_output_tokens`.
   - Implement a zero-allocation streaming `stop` / `stop_sequences` prefix
     holdback ring so partial stop-string prefixes are buffered until
     disambiguated and matched stop sequences are never leaked to SSE clients.
   - Support `json_object` and `json_schema` constrained structured output
     decoding via preallocated finite-state / grammar logit masks.
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

## Consequences

- **Positive:** Python and native hosts can embed `aminfer` in-process with zero
  weight/logit copies and zero Python inside the engine; standard OpenAI,
  Anthropic, and tool-calling agents work directly against `aminfer serve`; and
  operators gain a rootless, zero-dependency live Apple Silicon TUI with full
  `btop` theme compatibility and scriptable `--json` output.
- **Negative / trade-offs:** Maintaining a stable C ABI (`include/aminfer.h`),
  stateful `/v1/responses` slot retention, grammar-constrained JSON decoding,
  and three-endpoint SSE formatters adds surface area in `src/runtime/`,
  `src/server/`, and `src/bridge/`. Preallocated fixed-slot tables keep memory
  bounded and zero-alloc on the hot path, while capping concurrent stateful
  responses and grammar state depth to configured startup limits.

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
