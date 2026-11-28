# Deep code review — aminfer @ `d4541f5` / `bd88a3a` (2026-11-28)

This report records a two-pass whole-repository review of aminfer against source
tree `d4541f5258c1167820b32959d2055e0b6debbcfd` (Pass 1 recorded in
`bd88a3a8`, Pass 2 expanding deep line-level coverage across the remaining
subsystems on 2026-11-28, `main`). It covers correctness, memory safety, ABI
contracts, concurrency, security of the network and file-ingress surfaces, the
engineering invariants in [AGENTS.md](../../AGENTS.md), test coverage and
maintainability. It is a point-in-time review: status of capabilities still
lives only in [docs/status.md](../status.md), and nothing here changes it.

Every finding carries one of these labels:

| Label | Meaning |
| --- | --- |
| **Reproduced** | Triggered in a standalone reproduction; output quoted in Appendix A. |
| **Confirmed (read)** | The exact code path was read end to end; not executed. |
| **Observation** | A risk or design gap without a concrete failing input today. |

Recommendations are proposals for the maintainer, not decisions. Pass 1 finding
identifiers (`F1`–`F5`, `L1`–`L5`, `T1`, `A1`, `N1`–`N4`) are preserved intact;
Pass 2 findings continue with `F6`–`F25`, `L6`–`L9` and `N5`–`N7`.

## 1. Scope, method and limits

Environment: Apple M4 (10 cores, 24 GB), macOS 27.2 (`26B123`), Zig 0.17.1,
macOS SDK 27.0.

Size of the reviewed tree (tracked files):

| Area | Lines |
| --- | --- |
| Zig engine core (`src/**/*.zig`, 592 `test` blocks) | 295,250 |
| Metal kernels (`src/kernels/*.metal`) | 35,465 |
| Objective-C bridge (`apple_bridge.m` + `.h`) | 8,621 |
| SME2 assembly (`src/cpu/sme2_kernels.s`) | 3,996 |
| Python offline tools (`tools/src`) / tool tests | 43,171 / 14,764 |
| Python SDK + tests (`python/`) | 3,167 |

Method:
- **Pass 1** swept repository-wide invariants (`catch {}`, `catch unreachable`,
  request-path heap allocation, 16 KiB alignment, `mlock`, NoCopy wrapping,
  wired-budget checks, unsafe Python calls, hardcoded paths) and read the core
  memory, `.amf` header, HTTP framing, cluster framing, Metal 4 bridge, TLS, and
  SDK lifecycle paths in depth.
- **Pass 2** audited the previously unreviewed or lightly sampled slices in
  parallel and verified every reported defect directly in the source (plus
  standalone Zig and Python reproductions for seven representative issues):
  GPU shaders and Zig wrappers, ANE/CoreML and ImageIO/VideoToolbox bridges,
  ARMv9.2 SME2 assembly and loop controllers, async I/O / expert / PLE / N-gram
  storage and `PrefixCache`, cluster coordinator state machines, tokenizers,
  chat templates, LoRA/DoRA, speculative decoding and constrained sampling, HTTP
  route/keep-alive/SSE/embed/rerank lifecycles, C/Python ABI parity, CLI/TUI,
  and the Python offline converters.

| Depth | Files / areas |
| --- | --- |
| Read in depth (Pass 1 + Pass 2) | `build.zig`; `justfile`; `src/root.zig`; `src/mem/{page,arena,budget,prefix_cache}.zig`; `src/expert/source.zig`; `src/runtime/{ring,batch_scheduler}.zig`; `src/format/amf.zig`; `src/server/http_sse.zig` (framing, chunked decoder, auth, CORS, keep-alive slot lifecycle, SSE drain, stateful `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/v1/ngram-packs`); `src/cluster/{transport,expert_parallel}.zig`; `src/bridge/{apple_bridge.h,apple_bridge.m,metal.zig,tls.zig,vision.zig}` (Metal 4 ring, NoCopy, IOSurface, CoreML ANE, ImageIO, VideoToolbox, TLS); `src/ane/{predictor,wavefront}.zig`; `src/cpu/{sme2_kernels.s,sme.zig,inkling2_ctrl.zig}`; `src/storage/{aio,cache,ple}.zig`; `src/spec/{sam,tree,accept,grammar,sampling,draft,engine,ngram_nvme}.zig`; `src/model/{tokenizer,chat_template,lora,embed_rerank}.zig`; `include/aminfer.h`; `src/sdk/c_api.zig`; `python/aminfer/{_bindings,engine}.py`; `src/cli/chat_repl.zig`; `src/monitor/tui.zig`; `tools/src/aminfer_tools/{amf,gguf,chat_template,lora,coreml,hy4}.py` |
| Sampled / targeted audit | `src/model/{qwen3,graph,m82_runtime}.zig` (tensor binding, CPU SME2 prefill/draft, `verifyTree`/`commitTreeKv`, multimodal prefill, batch slot teardown, `config.json` parsing); `src/spec/{dspark,dflare_dfly}.zig`; `src/gpu/{dequant,attention,moe,hybrid,inkling2_hy4,quant_formats,lora_gpu,vision,sampler}.zig` and all 12 `src/kernels/*.metal` files (uniform struct ABI parity across 195 structs, 64-bit stride indexing, `threadgroup_barrier` uniformity, workspace staging); `src/main.zig` (budgeting, sidecar resolution, `ServeRunner` / `BatchServeRunner` / `ServeEmbedRerankCtx`); `src/platform/*.zig`; `tools/src/aminfer_tools/{cli,gsq_rco,deepseek41,glm5,mimo_v2,ling3,inkling2,qwen4_exp,ngram,heat,synth}.py` |
| Residual limits | Mathematical/numerical equivalence of every architecture-specific attention/MoE/SSM formula (`G25`–`G28`, `D14`–`D15`) against upstream PyTorch references was not re-derived from scratch; `src/bench/*` was checked for test-graph wiring and invariant sweeps rather than benchmark methodology. |

## 2. Verification performed

| Check | Result |
| --- | --- |
| `zig build` (Debug, Pass 1) | Exit 0 (Zig build cache). |
| `zig build test --summary all` (Debug & ReleaseSafe, Pass 1) | `10/10 steps succeeded` (`cached` replay; model-gated HF parity tests skipped when `AMINFER_MODELS_DIR` is unset — see T1). |
| `zig fmt --check build.zig src` (Pass 2) | Exit 0 (`0` formatting diffs). |
| `uv run --directory python python -m unittest discover -s tests` (Pass 2) | Executed live: `Ran 9 tests in 1.171s — OK`. |
| Pass 1 standalone reproductions | `F1` `decodeChunkedInPlace` overflow in Debug / ReleaseSafe / ReleaseFast (`A.1`); `L4` `std.fmt.parseInt` leniency (`A.2`). |
| Pass 2 standalone reproductions | `F12` `VocabIndex.fillAllowedTokenBitmask` trailing-word leak (`A.3`); `F17` C ABI uninitialized `out_info.struct_size` rejection (`A.4`); `F18` safetensors `index.json` path traversal and negative `data_offsets` header leak (`A.5`); `F20` `repetition_penalty` `+Inf -> NaN` logprob (`A.6`); `F24` `JsonScan.readKeyOrSkip` `> 64 B` key failure (`A.6`) and `chat_template.py` UTF-8 mojibake (`A.5`). |
| Live server / multi-Mac cluster | Not started; network, ImageIO/VideoToolbox, AIO, and cluster findings were verified by reading exact execution paths end to end. |

## 3. Summary

| ID | Severity | Area | Finding | Label |
| --- | --- | --- | --- | --- |
| F1 | **Critical** | HTTP server | Integer overflow in chunked body decoding: one unauthenticated request aborts the server (Debug/ReleaseSafe) or writes past the receive buffer (ReleaseFast) | Reproduced |
| F6 | **High** | HTTP server | Keep-alive `ConnectionSlot` reuse while inference worker is still active (`stop`-sequence early completion + `.finished` store order): data race on slot buffers/RNG/grammar and slot ABA across clients | Confirmed (read) |
| F7 | **High** | HTTP server | Lock-order inversion deadlock: `/v1/embeddings` and `/v1/rerank` lock `model_mutex` on the `kqueue` thread while the worker holds `model_mutex` spinning on a full `token_ring` | Confirmed (read) |
| F8 | **High** | Vision bridge | ImageIO decompression bomb (`kCGImageSourceShouldCacheImmediately: @YES` before `8192` dimension check: ~40 KB PNG -> 17.2 GiB raster) and unbounded multi-frame `IOSurface` RAM | Confirmed (read) |
| F9 | **High** | Vision bridge | Synchronous VideoToolbox GOP decode amplification (up to `250,000` frames on aliased `stco` chunks in ~2 MB MP4) blocks worker thread | Confirmed (read) |
| F10 | **High** | Prefix cache | `PrefixCache` can evict its own unpinned `parent_id` during child insertion, creating a self-parented radix-tree cycle (`parent_id == id`) that asserts in safe builds and loops/aliases free+evict lists in ReleaseFast | Confirmed (read) |
| F11 | **High** | CPU / SME2 | `packF32AsQ4Panel64NoHeap` writes a `576 B/group` layout into a `640 B/group` `sme2_panel64` allocation, corrupting gates (`NaN`/garbage) and reading uninitialized `WiredArena` tail memory | Confirmed (read) |
| F12 | **High** | Tokenizer / Grammar | `Tokenizer.vocabSize()` vs `model.vocab_size` mismatch leaves trailing grammar-bitmask words uninitialized (`tok < model`), writes past `grammar_mask` on the heap (`tok > model`), and skips prompt `tid < model.vocab_size` checks | Reproduced |
| F13 | **High** | Tokenizer / ChatTemplate | Flat prompt rendering + unconditional `added_tokens` matching lets untrusted `user`/`tool`/rerank text inject `<\|im_start\|>` / `<\|im_end\|>` / dialect role tokens | Confirmed (read) |
| F14 | **High** | Model / Speculative | Missing bounds/shift guards: unguarded `u6` shift at `ni = 63` in Hy4 64-node linear chains, unchecked `tid`/`start_pos` in `Qwen3Model` SME2 prefill, and unchecked `accepted_nodes[i]` in `GraphModel`/`M82Runtime.commitTreeKv` | Confirmed (read) |
| F15 | **High** | Async I/O / GPU teardown | Timeout / partial-submit lifetime hazards: `KqueueAioEngine.deinit` & `readSync`, `ExpertCache.startSlotLoad` rollback, `DomainNgramRegistry.swapDomainPack` live-fd `errdefer` close, and `Compute.wait` timeout unmapping in-flight `NoCopy` / `IOSurface` buffers | Confirmed (read) |
| F16 | **High** | Cluster | `ExpertParallelCoordinator` splices new frames into unsent `tx_buf` bytes after timeout, cascades `SequenceMismatch` on late responses, livelocks `kevent64` at 100% CPU on non-participating peer EOF, and ignores `PeerWorkerConfig.timeout_ms` | Confirmed (read) |
| F17 | **High** | C / Python SDK | Reading uninitialized `struct_size` on `*out_*` structs (`AMI_SDK_ERR_ABI_MISMATCH`), dangling writable `memoryview` in `VisionInput.lock_plane()`, and non-optional inner pointers / `usize` sum overflows in C entry points | Reproduced |
| F18 | **High** | Offline tools & CLI | Path traversal in `amf.resolve_checkpoint` (`model.safetensors.index.json`) and `main.zig` sidecar resolution, negative `data_offsets` leaking safetensors header bytes into `.amf`, unbounded `hy4.py` header read, and unbounded nested GGUF arrays | Reproduced |
| F2 | Medium | HTTP server | Defaults `cors_allow_origin = "*"`, no API key and no `Host` validation let any web page drive the local server and read its responses | Confirmed (read) |
| F3 | Medium | C / Python SDK | No per-engine concurrency guard; Python `stream()` runs native generation on a worker thread while `close()` can destroy the engine underneath it | Confirmed (read) |
| F4 | Medium | Runtime robustness | `catch unreachable` on prefix-cache and slot-pool invariants in request teardown turns a refcount bug into undefined behavior in ReleaseFast | Confirmed (read) |
| F5 | Medium | Memory policy | No process-wide wired-memory ledger: components check their own allocation against the full host budget, so the sum can exceed ADR-0012 | Confirmed (read) |
| F19 | Medium | Storage / Cache formats | Untrusted `.ang` `NgramSlot.cont_len > 8` OOB read, `PleStore` ignoring `entry.group_size != 64`, dropped batch completions wedging `.inflight` cache slots, and `.amkvc` header `u64` addition overflow | Confirmed (read) |
| F20 | Medium | Speculative / Sampling | `verifyConstrainedGpuBatch` mutates `grammar_state` and emits grammar-rejected token as `bonus_token`; `repetition_penalty` `+Inf` overflow yields `NaN` logprobs; Hy4 `k_fine` cache branch contamination near `max_seq_len` | Reproduced |
| F21 | Medium | GPU LoRA & Vision | Active LoRA wide-prefill / batching overflows Metal 4's `4,096` uniform-slot ring (`error.OutOfMemory`) and `row_slots_per_frame`; multimodal prefill performs per-request heap/`mmap`/`mlock` allocations and `prefillVisionQwen3` uses hardcoded pseudo-random weights | Confirmed (read) |
| F22 | Medium | LoRA & SME2 runtime | LoRA hot-unload/reload ABA slot aliasing on active batch requests, `flexible_dims` slice/DoRA calibration gaps, and `SmeDispatcher.evalLinearBatchHoisted` pre-packing activations across dependent ops | Confirmed (read) |
| F23 | Medium | ANE / CoreML bridge | `AneExpertPredictor.deinit` spin-deadlock and single-tile async data race, `WavefrontPrefill` un-drained error return, and non-atomic `CFGetRetainCount` ownership heuristic in `apple_bridge.m` | Confirmed (read) |
| F24 | Medium | Templates & Responses | Depth-blind `previous_response_id` pre-scan allows nested JSON fields to inject cached transcripts; `JsonScan.readKeyOrSkip` fails on valid `> 64 B` keys; `chat_template.py` `unicode_escape` corrupts UTF-8 delimiters into Latin-1 mojibake | Reproduced |
| F25 | Medium | CLI & TUI | `chat_repl.zig` and `tui.zig` leave terminal raw `termios` / alternate screen broken on signals; `tui.zig` writes unsanitized ESC/OSC bytes from `/health` to the terminal; `chat_repl.zig` silently truncates piped `stdin` at 64 KiB | Confirmed (read) |
| L1 | Low | Metal 4 bridge | Frame ring allows overlapping dependent frames with only intra-encoder barriers (latent; current callers serialize) | Observation |
| L2 | Low | TLS bridge | Exported TLS client accepts any certificate (test/bench use only today) | Confirmed (read) |
| L3 | Low | HTTP server | Slowloris exposure (32 slots, 30 s/60 s timeouts, auth after full body) and no warning for non-loopback bind without `--api-key` | Confirmed (read) |
| L4 | Low | HTTP server | Lenient number parsing (`+5`, `1_0`, `-0`) and `Transfer-Encoding` substring match | Reproduced |
| L5 | Low | Cluster | Peer protocol has CRC32 integrity but no peer authentication | Observation |
| L6 | Low | Config & `.amf` I/O | Narrowing `@intCast` panics on `config.json` `bits`/`group_size`, `NaN` bypassing `<= 0.0` norm/RoPE checks, and unclamped `std.c.pread` failing with `EINVAL` on single blobs `>= 2 GiB` | Confirmed (read) |
| L7 | Low | GPU sampler & MSL | `GpuSamplerWorkspace` mutates single-buffered shared `row_cfgs` on CPU during encoding; latent MSL alignment / barrier / modulo hazards if dispatched outside current Zig guards | Observation |
| L8 | Low | HTTP / Rerank / SDK | `/v1/responses/{id}/cancel` closes client socket with 0 bytes, `skipString` accepts lone surrogates, `Engine.embed([])` raises instead of returning `[]`, and `NaN` rerank logits violate sort ordering | Confirmed (read) |
| L9 | Low | Runtime edge cases | Unchecked `u32` `base_pos + count` before `max_seq_len`, layer-0-only `.active_tile` check in `PrefixCache.matchAndPin`, and re-entrant `BatchScheduler.cancelSlot` double-finish | Confirmed (read) |
| T1 | — | Tests | Headline real-HF parity tests skip by default; `zig build test` / `just test` do not run Python `tools/tests` or `python/tests`; no fuzzing of untrusted parsers; cached runs read as fresh | Observation |
| A1 | — | Maintainability | Very large files (`graph.zig` 31k lines, `http_sse.zig` 17.6k), duplicated runtime/ring logic, `@errorName` string matching in `c_api.zig` | Observation |
| N1–N7 | Nit | Various | See section 6 | — |

## 4. Detailed findings

### F1 — Critical: integer overflow in `decodeChunkedInPlace`

**Location.** [`src/server/http_sse.zig:5636`](../../src/server/http_sse.zig)
(`decodeChunkedInPlace`), overflow at line 5660, unbounded copy at line 5680;
called from `tryDispatchHttpRequest` at line 8064.

**Defect.** The first pass validates each chunk with

```zig
const chunk_len = std.fmt.parseInt(usize, size_line, 16) catch return error.BadRequest;
...
if (body_buf.len - scan < chunk_len + 2) return .incomplete;   // line 5660
if (body_buf[scan + chunk_len] != '\r' or body_buf[scan + chunk_len + 1] != '\n') ...
scan += chunk_len + 2;
```

`chunk_len` is attacker-controlled and may be any 64-bit value, so
`chunk_len + 2` overflows for `fffffffffffffffe` / `ffffffffffffffff`.

**Failure scenario.** Request body `fffffffffffffffe\r\n0\r\n\r\n` with
`Transfer-Encoding: chunked`:

- Debug and ReleaseSafe: integer-overflow panic. The panic aborts the whole
  process — every connection and the loaded model go down with it.
- ReleaseFast (the mode the launch guide prescribes for daily serving):
  `chunk_len + 2` wraps to `0`, the CRLF check then reads `body_buf[scan-2..scan]`
  — the CRLF of the size line itself — and passes; `scan` does not advance. The
  second pass executes
  `copyForwards(body_buf[0 .. 2^64-2], body_buf[18 .. 16])` without bounds
  checks, writing past the receive slot until the process faults.

Body framing runs before route dispatch and before the `--api-key` gate
(line 8128), so the request needs no credentials. The native TLS listener
proxies decrypted bytes into the same loop, so HTTPS serving is affected too.

**Evidence.** Appendix A.1: Debug → `panic: integer overflow` at the
`chunk_len + 2` expression; ReleaseSafe → abort (exit 134, integer-overflow
panic); ReleaseFast → `SIGSEGV` (exit 139).

**Recommendation.** Use overflow-free comparisons in pass 1 and bound the chunk
length by the slot capacity, then add regression tests for `fffffffffffffffe`,
`ffffffffffffffff`, a chunk length larger than `rx_buf.len`, and a
non-terminated huge chunk:

```zig
const remaining = body_buf.len - scan;
if (chunk_len > remaining or remaining - chunk_len < 2) return .incomplete;
```

At the call site, reject any `chunk_len` (or running decoded total) larger than
`slot.rx_buf.len` with `413` instead of waiting for more data. Pass 2 is then
safe because it only replays lengths already bounded by pass 1.

### F2 — Medium: permissive network defaults (CORS `*`, no key, no `Host` check)

**Location.** `ServerConfig` in
[`src/server/http_sse.zig`](../../src/server/http_sse.zig): `api_key = null`
(line 699), `cors_allow_origin = "*"` (line 725); headers written by
`appendCorsResponseHeaders` (line 8321); the auth gate (line 8128) is skipped
when no key is configured; `parseHttpHeaders` never inspects `Host`.

**Failure scenario.** With the defaults (`127.0.0.1:8080`, no `--api-key`), any
web page the user opens can `fetch("http://127.0.0.1:8080/v1/chat/completions",
…)`. The preflight allows `Content-Type` and `Authorization`, and every response
carries `Access-Control-Allow-Origin: *`, so the page can read generated text,
consume GPU time, enumerate `/v1/models`, and change global state through
`POST /v1/ngram-packs/select`. Even with CORS disabled, the missing `Host` check
leaves DNS rebinding open while no key is set.

**Recommendation.** Make CORS opt-in (default `null`), or reflect only loopback
origins. When `api_key` is unset, accept only `Host` values that name the bound
address, `localhost`, `127.0.0.1` or `[::1]`. Treat binding a non-loopback
address without `--api-key` as an error unless the operator passes an explicit
override flag (see also L3).

### F3 — Medium: SDK engine has no concurrency guard; `stream()` + `close()` is a use-after-free

**Location.** `SdkEngine` in [`src/sdk/c_api.zig:1044`](../../src/sdk/c_api.zig)
(no mutex or busy flag; `ami_sdk_engine_destroy` at line 2102 frees
unconditionally); [`include/aminfer.h`](../../include/aminfer.h) states no
threading contract; [`python/aminfer/engine.py`](../../python/aminfer/engine.py)
`Engine.stream` (line 1405) and `Engine.close` (line 977).

**Defect.** `stream()` runs `self.generate(...)` on a daemon worker thread.
`ctypes` releases the GIL during the native call, so the caller thread keeps
full use of the same `Engine`. `close()` (also reached through `__exit__`)
calls `ami_sdk_engine_destroy` without stopping or joining an active stream.

**Failure scenario.** `gen = eng.stream(prompt); next(gen); eng.close()` — or
leaving a `with Engine(...)` block while holding a live generator — destroys the
native engine (wired arena, Metal buffers, KV tiles) while the worker thread is
still inside `ami_sdk_generate_*`: use-after-free in native code. Two threads
calling `generate()` on one engine race on the KV cache and scratch buffers.
`logits_view()` / `hidden_states_view()` return `memoryview`s over wired memory
that dangle after `close()`.

**Recommendation.** In C, guard each engine with a mutex or an atomic busy flag
(return a new `AMI_SDK_ERR_BUSY`) and refuse `destroy` while busy. In Python,
hold a `threading.Lock` per `Engine`, have `close()` signal and join any active
stream worker, and invalidate or own the tensor views. Document "one call at a
time per engine" in `aminfer.h`.

### F4 — Medium: `catch unreachable` on runtime invariants

**Location.** About 30 non-test sites outside string formatting, mainly in
request-slot teardown, for example:

- [`src/model/qwen3.zig:11439`](../../src/model/qwen3.zig) and
  [`src/model/graph.zig:20352`](../../src/model/graph.zig) —
  `resetSlotState`: `pc.unpin(eid) catch unreachable`. `PrefixCache.unpin`
  ([`src/mem/prefix_cache.zig:801`](../../src/mem/prefix_cache.zig)) returns
  `error.InvalidEntry` / `error.UnbalancedUnpin`.
- `qwen3.zig` 11271, 11326, 11344, 11457 (`rpool.releaseSlot`), 11505 and
  11533 (`releaseMatch`), 11563 (`releaseTailPin`), 11624 (`unpin`).
- `graph.zig` 20175–20658: eleven snapshot-pool `releaseSlot(...) catch
  unreachable`, plus `entryHandle` / `entryInfo` at 20425 / 20439.
- [`src/cluster/expert_parallel.zig:173`](../../src/cluster/expert_parallel.zig).

(The 361 hits in `graph.zig` are dominated by `bufPrint` into fixed buffers with
constant formats; those are benign and excluded here.)

**Failure scenario.** A pin/refcount imbalance on a cancel, evict or error path
(the paths least covered by tests) panics the server in safe builds and is
undefined behavior in ReleaseFast — the optimizer may assume the error branch
never happens. AGENTS.md forbids swallowing errors; `catch unreachable` on a
fallible runtime call is the strongest form of that.

**Recommendation.** Return errors from teardown where possible; otherwise use
`std.debug.assert` for the invariant plus an explicit recovery path (poison the
slot, reset the cache, log) that also runs in ReleaseFast. Add a CI search that
rejects new `catch unreachable` outside `bufPrint`/`utf8Encode` with constant
inputs.

### F5 — Medium: no process-wide wired-memory ledger

**Location.** [`src/mem/budget.zig`](../../src/mem/budget.zig) offers
`checkPlan` against a static limit; callers compute a fresh full-host budget and
check only their own bytes. There are 51 non-test
`Budget.fromHost(...)` / `defaultWiredBudget()` call sites (19 of them in
`src/bench/`), for example
[`src/spec/dspark.zig:3024`](../../src/spec/dspark.zig) (draft weights +
scratch + cold pool), `src/spec/dflare_dfly.zig` (1146, 6445, 6953, helper at
8420), `src/spec/ngram_nvme.zig` (384, 925), `src/cpu/sme.zig` (3402, 4087),
`src/cpu/inkling2_ctrl.zig:358`. In `src/main.zig` only the helper at line 441
subtracts `already_wired_bytes`, and the model + LoRA pool joint check
(lines 7694, 8288, 9820, 9965) runs only when `-W` sets `budget_override`.

**Failure scenario.** A 12 GiB main model, a 2 GiB DSpark draft, a prefix cache
and a LoRA pool each pass against an ~17.7 GiB limit while their sum exceeds it.
The remaining backstops are `mlock` failing at `vm.user_wire_limit` (which sits
above the ADR-0012 3 GiB headroom policy) and nothing at all for Metal's
`recommendedMaxWorkingSetSize`. ADR-0012's "refuse plans over budget" therefore
holds per component, not per process.

**Recommendation.** Add a process-wide atomic ledger in `mem/budget.zig`,
charged by `WiredRegion.init` and credited by `deinit`, and make `checkPlan`
compare against `limit - charged`. Thread one budget handle through the loaders
instead of recomputing host facts, and add a test where model + draft together
exceed the budget and must fail.

### F6 — High: keep-alive `ConnectionSlot` reuse while inference worker is still active (data race + slot ABA)

**Location.** [`src/server/http_sse.zig`](../../src/server/http_sse.zig):
`drainSlotTokens` (`10643–10650`), `finishHttpExchange` (`7888–7906`),
`admitInferenceRequest` (`10389`, `10448–10533`), `runWorkerLoop` (`7408–7416`),
`runBatchedWorkerLoop` (`7481–7488`).

**Defect.** Two race windows allow a `ConnectionSlot` to be recycled for a new
HTTP request while an inference worker still owns pointers into that slot:

1. **Early `stop`-sequence completion on non-streaming keep-alive (`stream: false`).**
   `admitInferenceRequest` (`10448`) disables `slot.keep_alive` only when
   `parsed.stream == true`. When `drainSlotTokens` (`10643`) matches a `stop`
   sequence on a non-streaming HTTP/1.1 keep-alive request, it sets
   `slot.cancelled.store(true, .release)` and immediately calls
   `finishSlotResponse(slot, .stop_sequence)` → `finishHttpExchange(slot)`
   (`7888`). Unlike `closeSlotSocketAndRelease` (`7923`), `finishHttpExchange`
   never inspects `slot.worker_active`: it immediately calls
   `slot.resetForNextKeepAliveRequest(...)` (`7902`) and dispatches any
   pipelined/next request (`7904`).
2. **`.finished` event pushed before `worker_active = false`.** Both
   `runWorkerLoop` (`7409` vs `7415`) and `runBatchedWorkerLoop` (`7482` vs
   `7486`) push `.finished` into `slot.token_ring` and wake `kqueue` *before*
   executing `slot.worker_active.store(false, .release)`.

**Failure scenario.** Two pipelined or back-to-back `stream: false` requests on
one keep-alive connection (particularly when Request #1 stops on a `stop`
string): Request #2 overwrites `slot.prompt_buf`, `slot.rng`, `slot.stop_config`,
`slot.media_buf`, `slot.logit_bias_entries`, and
`self.grammar_programs[slot.grammar_slot_index]` while Worker #1 is actively
reading/mutating them; Request #2 resets `slot.cancelled.store(false, .release)`
(`10389`, un-cancelling Worker #1); and when Worker #1 finally exits, it writes
`slot.worker_active = false` while Job #2 is still running — allowing
`closeSlotSocketAndRelease` + `findFreeSlot` to assign the slot to a third TCP
client while Worker #2 is still mutating its state.

**Recommendation.** Never transition a keep-alive slot back to
`.reading_request` while `slot.worker_active.load(.acquire)` is `true` (hold it
in a `.waiting_keepalive_worker` state drained in `drainAllSlots`), and store
`worker_active = false` (or a generation-tagged worker state) *before* waking
the network thread on `.finished` — or track an atomic active-worker counter
rather than a boolean so two jobs on one slot index can never overlap.

### F7 — High: lock-order inversion deadlock between `kqueue` network thread (`/v1/embeddings`, `/v1/rerank`) and worker thread

**Location.** [`src/server/http_sse.zig:6378–6390`](../../src/server/http_sse.zig)
(`TokenStream.pushEvent`), `9288` (`handleEmbeddings`), `9367` (`handleRerank`);
[`src/main.zig:6265`](../../src/main.zig) (`ServeRunner.generate`), `7325` and
`7373` (`ServeEmbedRerankCtx`).

**Defect.** `handleEmbeddings` and `handleRerank` execute synchronously on the
single `kqueue` event-loop thread and call `pthread_mutex_lock(self.model_mutex)`.
Meanwhile, single-stream serving (`ServeRunner.generate`, used for all
`GraphModel` architectures, speculative decoding, multimodal vision, and
`--no-continuous`) holds `model_mutex` across the entire generation and busy-waits
inside `TokenStream.pushEvent` (`http_sse.zig:6380–6388`) whenever the 256-slot
`slot.token_ring` fills, waiting for the `kqueue` thread to drain `token_ring`
or set `slot.cancelled` on client `EV_EOF`.

**Failure scenario.** One streaming request stalls TCP writes long enough to
fill 256 `token_ring` entries (holding `model_mutex`), and any client sends
`POST /v1/embeddings` or `POST /v1/rerank`. The `kqueue` thread blocks forever
on `model_mutex`, and the worker spins forever waiting for `kqueue` to drain
`token_ring` — a permanent server-wide deadlock.

*(Related `ServeEmbedRerankCtx` issues at `main.zig:7345–7348, 7392–7400`:
`std.fmt.bufPrint(...) catch doc` silently drops `req.query` and the ChatML
wrapper when a document overflows `rerank_prompt_buf`; fallback truncation
`text[0..@min(text.len, self.token_ids_buf.len)]` slices raw UTF-8 mid-codepoint
instead of returning `413`; and `rerankCallback` tokenizes untrusted documents
with special tokens enabled — see F13.)*

**Recommendation.** Dispatch `/v1/embeddings` and `/v1/rerank` asynchronously
through `job_ring` onto the worker thread instead of locking `model_mutex` on
the `kqueue` loop, and return `413` when a rerank/embedding input exceeds
scratch capacity.

### F8 — High: ImageIO decompression bomb (`kCGImageSourceShouldCacheImmediately: @YES` before dimension check) and unbounded multi-frame `IOSurface` allocation

**Location.** [`src/bridge/apple_bridge.m:2799–2812`](../../src/bridge/apple_bridge.m)
(`ami_image_decode_bytes_iosurface`), `5249–5268`
(`ami_video_decode_bytes_iosurface`), `2724–2731` (`ami_image_load_iosurface`);
called from [`src/main.zig:6163, 6176`](../../src/main.zig).

**Defect.** Both byte-decoding entry points call `CGImageSourceCreateImageAtIndex`
with `kCGImageSourceShouldCacheImmediately: @YES` *before* inspecting image
header dimensions via `CGImageSourceCopyPropertiesAtIndex`.
`ShouldCacheImmediately: @YES` forces ImageIO to eagerly allocate and decompress
the full RGBA raster in heap memory inside `CGImageSourceCreateImageAtIndex`
before `vision_frame_from_cg_image` (`apple_bridge.m:2611`) ever checks
`width, height <= AMI_VISION_MAX_BYTES_IMAGE_DIM` (`8192`). In addition,
`ami_video_decode_bytes_iosurface` allows up to `64` frames (`8` per attachment
in `main.zig:6175` across multiple media items) at `8192 x 8192` (`256 MiB`
BGRA8 each) with no aggregate byte or pixel cap, and `ShouldCacheImmediately`
doubles peak memory even on valid `8192 x 8192` frames (`256 MiB` `CGImage`
raster + `256 MiB` `IOSurface`).

**Failure scenario.** A ~40 KB valid solid-color PNG/GIF/WebP with header
dimensions `65535 x 65535` sent to `/v1/chat/completions` triggers a `17.2 GiB`
eager raster allocation inside ImageIO before the `8192` guard runs, OOM-killing
a 24 GB M4 whose model weights are `mlock`ed.

**Recommendation.** Query `CGImageSourceCopyPropertiesAtIndex` first to reject
`width > 8192`, `height > 8192`, or aggregate request pixel budgets before
creating a `CGImageRef`; drop `kCGImageSourceShouldCacheImmediately: @YES` so
`CGContextDrawImage` decodes directly into the locked `IOSurface`; and enforce a
per-request total vision byte cap.

### F9 — High: synchronous VideoToolbox GOP decode amplification (up to 250,000 frames) on ISO-BMFF attachments

**Location.** [`src/bridge/apple_bridge.m:4740–4822`](../../src/bridge/apple_bridge.m)
(`decode_isobmff_bytes`) and `4030–4077, 4130–4190`
(`decode_compressed_samples_iosurface`).

**Defect.** `decode_isobmff_bytes` checks `uniform_size * sample_count <= len`
only when `stsz` `uniform_size != 0` (`4749`). When `uniform_size == 0`, each
`stco` chunk resets `off = stco[c]` (`4805`) without checking for overlapping
chunk offsets or cumulative compressed bytes, allowing up to `250,000` sample
descriptors in a ~2 MB MP4 to alias the same small P-slice in `mdat`. If `stss`
marks only sample `1` as a sync frame, `decode_compressed_samples_iosurface`
(`4064–4076`) walks backward from the last target frame to `sync_start = 0` and
marks `need_decode[0..249999] = true`, invoking `VTDecompressionSessionDecodeFrame`
`250,000` times synchronously on the inference worker thread. (Separately,
`decode_compressed_samples_iosurface` checks `<= 8192` only on the container's
initial `avcC`/`hvcC` descriptor at `4031`, whereas
`vision_frame_init_from_mtl_device` at `1970` allows up to `16384` if an
in-band SPS switches dimensions.)

**Recommendation.** Bound total `need_decode` frames per request (e.g. `<= 2048`
frames across all GOPs), enforce non-overlapping/bounded cumulative `stsz` byte
spans, and thread `AMI_VISION_MAX_BYTES_IMAGE_DIM` into
`vision_frame_init_from_mtl_device`.

### F10 — High: `PrefixCache` evicts its own unpinned `parent_id` during child insertion, creating a self-parented radix-tree cycle

**Location.** [`src/mem/prefix_cache.zig:441–510`](../../src/mem/prefix_cache.zig)
(`insertFullBlock`), `536–616` (`insertTailBlock`), `632–648`
(`insertBoundaryCheckpoint`), `908–953` (`evictLru`).

**Defect.** `insertFullBlock` (`441–450`) and `insertTailBlock` (`536–545`)
validate `entries[parent_id].in_use` *before* calling
`allocateEntrySlot(opts.evict_if_full, ...)` (`464`, `567`), without temporarily
pinning `parent_id` or excluding it from `evictLru`. `InsertOptions.pin_on_insert`
defaults to `false` (`line 111`), and `insertBoundaryCheckpoint` inserts with
`evict_first = true` (`evict_first_tail` is evicted ahead of `evict_tail` at
`line 909`).

**Failure scenario.** When `PrefixCache` is full and `parent_id` is the next LRU
victim (`pin_count == 0`, `child_count == 0` — e.g., an `evict_first` boundary
checkpoint or consecutive unpinned block insertions when all other slots are
pinned), `evictLru` evicts `parent_id` and `allocateEntrySlot` returns
`id == parent_id`. `insert*Block` then sets `entries[id].parent_id = id`, calls
`linkChild(id, id)` (`first_child = id`, `child_count = 1`), and pushes `id`
onto `evict_head` (`508` / `610`) with `in_evict_list = true` and
`child_count == 1`. The next eviction hits
`std.debug.assert(e.in_use and e.pin_count == 0 and e.child_count == 0)` (`918`)
in Debug/ReleaseSafe; in ReleaseFast, `id` ends up simultaneously on `free_head`
and `evict_list`, and sibling walks (`next_sibling == id`) spin forever.

**Recommendation.** Pin `parent_id` across `allocateEntrySlot` (unpinning once
`linkChild` increments `child_count`) so `parent_id` and all of its ancestors
are un-evictable while inserting a child.

### F11 — High: broken `sme2_panel64` Q4 binary layout (`576 B` vs `640 B`/group) and uninitialized `WiredArena` over-read in `packF32AsQ4Panel64NoHeap`

**Location.** [`src/cpu/inkling2_ctrl.zig:1047–1053, 1085–1143`](../../src/cpu/inkling2_ctrl.zig)
vs [`src/cpu/sme.zig:840–862`](../../src/cpu/sme.zig) and
[`src/cpu/sme2_kernels.s:1051–1073`](../../src/cpu/sme2_kernels.s).

**Defect.** Canonical `.sme2_panel64` `q4_affine` uses `640 B` per group per
16-column panel: `0..64` duplicated `[scale, scale]`, `64..128` duplicated
`[bias, bias]` (`QUANT_HEADER_BYTES = 128`), and `128..640` `512 B` packed 4-bit
codes. `allocAndPackWeight` allocates `sme.packedBlobBytes(.q4_affine, n, k)`
(`640 B/group`) from unzeroed `WiredArena` memory, but `packF32AsQ4Panel64NoHeap`
(`1094`) hardcodes `grp_bytes = 64 + 32 * 16` (`576 B/group`), writes
interleaved `[scale, bias]` into `0..64` (`1126–1127`), and writes codes at
`64..576` (`1130`). (Existing tests at `inkling2_ctrl.zig:1485` passed
`.q4_affine` pre-quantized inputs, missing the `.dense_f32` -> `.q4_affine`
branch.)

**Failure scenario.** Initializing `Inkling2LoopController` from `.dense_f32`
weights with `.q4_affine` branch config causes SME2 assembly and scalar fallback
to reinterpret packed Q4 nibbles at `64..128` as IEEE-754 `f16` biases
(`NaN`/`Inf`), misalign every subsequent group by `64 B`, and read
`64 * panels * groups` uninitialized `WiredArena` bytes at the end of the blob.

**Recommendation.** Write the canonical `128 B` duplicated scale/bias header
(`grp_bytes = 640`) matching `sme.packQuantAffinePanel64`, and add a parity test
covering `.dense_f32` with `.q4_affine`.

### F12 — High: `Tokenizer.vocabSize()` vs `model.vocab_size` mismatch causes uninitialized bitmask reads, heap OOB writes, and unchecked OOB token IDs

**Location.** [`src/model/tokenizer.zig:2505`](../../src/model/tokenizer.zig),
[`src/spec/grammar.zig:563, 631, 775–784`](../../src/spec/grammar.zig),
[`src/spec/accept.zig:482–495`](../../src/spec/accept.zig),
[`src/spec/engine.zig:2358–2366`](../../src/spec/engine.zig),
[`src/main.zig:6588, 6794, 6895`](../../src/main.zig).

**Defect & failure scenarios.**
1. **`tok.vocabSize() < model.vocab_size` (standard on padded Hugging Face
   checkpoints, e.g. Qwen3 `151,669` tokenizer entries vs `151,936` padded
   `lm_head`).** `VocabIndex.initFromTokenizer` (`grammar.zig:563`) sets
   `vi.vocab_size = 151,669` (`words = 4,740`), while `SpecEngine.fillBitmaskForState`
   (`engine.zig:2361`) and `GrammarTreeAdapter.fillBitmask` (`accept.zig:485`)
   pass `out_mask` of length `bitmaskWordCount(151,936) = 4,748`.
   `fillAllowedTokenBitmask` (`grammar.zig:783`) zeros only `out_mask[0..4740]`,
   leaving trailing words `4740..4748` uninitialized (or holding stale `1` bits
   from `allowAllTokens`), so constrained decoding can sample padded OOV token
   IDs `151,680..151,935` that then fail `tok.decode` with `error.UnknownTokenId`.
2. **`tok.vocabSize() > model.vocab_size` (e.g., `--tokenizer` override or extra
   added tokens).** `out_mask.len` (`bitmaskWordCount(model.vocab_size)`) is
   smaller than `words` (`bitmaskWordCount(vi.vocab_size)`): `grammar.zig:776`
   panics in safe builds, and `@memset(out_mask[0..words], 0)` (`783`) writes
   past `grammar_mask` on the heap in ReleaseFast (`accept.zig:492` and
   `engine.zig:2365` likewise slice `slices[0..v]` unchecked).
3. **Sparse `added_tokens` and unchecked prompt token IDs.** `Tokenizer.vocabSize()`
   returns `self.token_count` rather than `max_id + 1`, so sparse IDs `>= token_count`
   are skipped by `VocabIndex.initGeneric`; meanwhile `BatchServeRunner.tokenizeWithBoundaryCallback`
   (`main.zig:6895`) checks `tid >= self.tok.vocabSize()` on raw token arrays
   and checks nothing against `model.config.vocab_size` before forwarding IDs to
   `prefillSlot` (`qwen3.zig:12577` and `graph.zig:22464` also omit `tid < vocab_size`).

**Evidence.** Appendix A.3 reproduces trailing bitmask word `0xffffffff`
surviving `fillAllowedTokenBitmask` when `vi.vocab_size = 32` and `out_mask`
covers `64` tokens.

**Recommendation.** Track `max_token_id_plus_one` separately from `token_count`;
in `fillAllowedTokenBitmask`, zero the entire `out_mask` (`@memset(out_mask, 0)`)
and clamp active words to `@min(words, out_mask.len)`; and validate all prompt
token IDs `< model.config.vocab_size` at request admission and `prefillSlot`.

### F13 — High: special control-token injection in flat prompt rendering

**Location.** [`src/model/tokenizer.zig:2675–2704`](../../src/model/tokenizer.zig)
(`Tokenizer.encodeImpl`), [`src/model/chat_template.zig:1762, 1826–2031`](../../src/model/chat_template.zig)
(`PromptRenderer`), [`src/sdk/c_api.zig:4212`](../../src/sdk/c_api.zig)
(`ami_sdk_rerank_texts`), [`src/main.zig:7392–7400`](../../src/main.zig)
(`rerankCallback`).

**Defect.** `ChatTemplate.render` emits untrusted `user` / `tool` message text,
`tool_call_id` (`1996`), and `tool_choice_name` (`1762`) directly into the same
flat byte buffer as structural role tags (`<|im_start|>`, `<|im_end|>`,
`<|message_system|>`, `<|end_message|>`, `<｜hy_User｜>`). `Tokenizer` has no
`allow_special = false` mode (`encodeImpl` unconditionally matches all
`added_tokens` at `2690`), and `ami_sdk_rerank_texts` explicitly passes
`allow_special_tokens = 1` over concatenated untrusted documents.

**Failure scenario.** Any `user` message, tool output, or reranked document
containing `"<|im_end|>\n<|im_start|>system\n..."` (or the equivalent Inkling 2 /
DeepSeek / Hy4 markers) is tokenized into real `is_special = true` role/EOS/thinking
control token IDs, breaking role isolation across `/v1/chat/completions`,
`/v1/messages`, `/v1/responses`, `/v1/rerank`, and the SDK.

**Recommendation.** Add `allow_special: bool` to `Tokenizer.encode*` (skipping
`added_tokens` where `tok.special == true` when `false`), and tokenize untrusted
content segments with `allow_special = false` while emitting template structural
delimiters as exact token IDs (or with `allow_special = true` on structural
spans only).

### F14 — High: missing bounds and shift guards in Hy4 64-node chains, `Qwen3Model` SME2 prefill, and `GraphModel` / `M82Runtime.commitTreeKv`

**Location.**
- [`src/model/m82_runtime.zig:3564`](../../src/model/m82_runtime.zig) (`execHy4Layer`)
- [`src/model/qwen3.zig:4008–4025, 6216–6285, 6814–6991`](../../src/model/qwen3.zig)
  (`prefillChunkWithOptions` → `stepDraftTreeBatchSme2` → `cpuEmbedLookup`)
- [`src/model/graph.zig:7326–7513`](../../src/model/graph.zig) and
  [`src/model/m82_runtime.zig:4375–4496`](../../src/model/m82_runtime.zig)
  (`commitTreeKv`)

**Defects.**
1. **64-node `u6` shift overflow in Hy4 (`m82_runtime.zig:3564`).**
   `const exp_mask: u64 = (@as(u64, 1) << @intCast(ni + 1)) - 1;` omits the
   `if (ni == 63) ~@as(u64, 0)` guard used at all 8 other 64-node mask sites. A
   full `m = 64` (`MAX_TREE_NODES`) linear speculative chain (e.g., 63-token L1
   SAM / N-gram hit) reaches `ni = 63` and evaluates `@intCast(64)` to `u6`
   (panic in safe builds; UB in ReleaseFast).
2. **Unchecked `tid` and `start_pos` in `Qwen3Model` pure-CPU SME2 prefill
   (`qwen3.zig:4013`).** When `has_pure_cpu_sme2_dense` is active,
   `prefillChunkWithOptions` calls `stepDraftTreeBatchSme2` directly without
   checking `tokens[i] < vocab_size` or `start_pos + m <= max_seq_len` (and
   without retiring KV chunks or updating `self.seq_len`). Out-of-range `tid`
   reaches `dequantRowCpuRef(...) catch unreachable` (`6234`, UB in ReleaseFast)
   or OOB slice indexing; `start_pos + m > max_seq_len` indexes `kvTileAt` OOB
   in ReleaseFast.
3. **Unchecked `accepted_nodes[i]` in `commitTreeKv` (`graph.zig:7326`,
   `m82_runtime.zig:4375`).** Unlike `Qwen3Model.commitTreeKv` (`qwen3.zig:2967`),
   neither function checks `nid < self.max_tree_nodes` (or `< last_verified_nodes`),
   allowing host slice OOB reads/panics (`mla_csa2` at `graph.zig:7413`, Hy4 at
   `m82_runtime.zig:4468`) and GPU scratch OOB reads.

**Recommendation.** Guard `ni == 63` with `~@as(u64, 0)`, enforce `tid < vocab_size`
and `start_pos + m <= max_seq_len` at the top of `prefillChunkWithOptions` and
`stepDraftTreeBatchSme2`, and validate `nid < self.last_verified_nodes` in both
`commitTreeKv` implementations.

### F15 — High: async I/O, fd-swap, and GPU timeout teardown use-after-free / use-after-unmap hazards

**Location.**
- [`src/storage/aio.zig:338–351, 434–448, 878–892`](../../src/storage/aio.zig)
  (`KqueueAioEngine.deinit` / `readSync`, `EcorePreadPool.readSync`)
- [`src/storage/cache.zig:1416–1425, 1478–1483`](../../src/storage/cache.zig)
  (`ExpertCache.startSlotLoad` / `waitSlotDoneBestEffort`)
- [`src/spec/ngram_nvme.zig:1205–1234`](../../src/spec/ngram_nvme.zig)
  (`DomainNgramRegistry.swapDomainPack`)
- [`src/model/graph.zig:7761–7824, 8079–8279`](../../src/model/graph.zig),
  [`src/main.zig:6154`](../../src/main.zig),
  [`src/sdk/c_api.zig:4678–4685`](../../src/sdk/c_api.zig),
  [`src/bridge/apple_bridge.m:787–810`](../../src/bridge/apple_bridge.m)

**Defects.**
1. **Kernel AIO & `readSync` lifetime (`aio.zig`).** `KqueueAioEngine.deinit`
   breaks after a single 5 s wait (`wait(&scratch, 5_000) catch break`) and
   immediately `allocator.free(self.slots)` without calling `aio_cancel` or
   blocking until `inflight == 0`, while callers immediately `munmap` the
   destination `WiredArena` still referenced by XNU `aiocb`. Both `readSync`
   implementations also return `error.Timeout` while DMA / worker `pread` into
   caller `req.dst` remains active, and reap `comp[0]` without matching
   `user_data` (returning stale timed-out completions on subsequent calls).
2. **`ExpertCache` partial-submit rollback (`cache.zig:1416–1425`).** If blob 0
   submits and blob 1/2 fails `submit`, `waitSlotDoneBestEffort` breaks on
   timeout/error, yet lines `1422–1423` still reset `slot.state = .empty` and
   `slot.pending_blobs = 0`, allowing immediate reallocation of the wired slot
   while blob 0's `aio_read`/`pread` is still writing into `gateSlice(slot_idx)`.
3. **Live `fd` use-after-close in `swapDomainPack` (`ngram_nvme.zig:1205–1229`).**
   `errdefer opened.file.close()` (`1206`) remains armed after `opened.file` is
   copied into `pack.banks[new_bank]` and published via `pack.active_bank.store(new_bank)`
   (`1220`). If `reapPendingForDrain()` (`1227`) returns an I/O error while
   draining `old_bank`, `errdefer` closes `new_bank.file.fd` while lock-free
   readers are actively `pread`ing from it (risking OS fd-reuse corruption and
   leaking `old_bank.file`).
4. **GPU `Compute.wait` timeout teardown (`graph.zig`, `main.zig`, `c_api.zig`,
   `apple_bridge.m`).** On `compute.wait` timeout, `defer` blocks immediately
   call `removeResident`, `munmap` `NoCopy` `WiredArena`s, and close `IOSurface`
   `VisionFrame`s while the timed-out Metal 4 command buffer is still executing
   on the GPU (`ami_compute_destroy` at `apple_bridge.m:788` likewise ignores a
   false return from `waitUntilSignaledValue:max_sig timeoutMS:5000`).

**Recommendation.** Cancel and drain in-flight kernel `aiocb` / worker `pread`
operations before freeing slots or `munmap`ping wired arenas; quarantine cache
slots whose drain failed instead of marking them `.empty`; disarm `errdefer`
immediately once `new_bank` is published; and wait/poison GPU command queues
before unmapping `NoCopy` or `IOSurface` memory on timeout.

### F16 — High: `ExpertParallelCoordinator` mid-frame TCP stream splice, late-response cascade, and 100% CPU `kevent64` livelock

**Location.** [`src/cluster/expert_parallel.zig:563, 752, 1281–1287, 1324–1362, 1425–1542, 1604–1622`](../../src/cluster/expert_parallel.zig);
[`src/cluster/transport.zig:1014–1035`](../../src/cluster/transport.zig).

**Defects.**
1. **Mid-frame TX splice.** `FrameChannel.commitFrameChecked` resets
   `tx_len = total` and `tx_sent = 0` without checking `!self.hasPendingTx()`.
   If `dispatchLayerExperts` times out while peer A has a partially sent frame,
   peer A remains `connected = true` and the next `commitFrame` splices a new
   `AMI7` header into the middle of the half-sent TCP stream.
2. **Late-response `SequenceMismatch` cascade (`1515`).** A delayed
   `eval_experts_resp` (`seq_id = N`) arriving after a transient timeout stays
   queued in the socket buffer and causes step `N + 1` to fail with
   `error.SequenceMismatch` without draining `seq_id < expected` or reconnecting.
3. **100% CPU `kevent64` livelock (`1281–1287, 1357, 1489, 1617`).** Persistent
   level-triggered `EVFILT_READ` (without `EV_CLEAR`) is registered on one
   shared `kq_fd` for all peers, while `dispatchLayerExperts` ignores events on
   peers with `!peer_pending_recv[p_idx]` (`1489`) and `pingPeer` /
   `drivePeerTxToCompletion` ignore returned events altogether. An EOF or unread
   byte on any non-participating peer makes `kqueueWait` return in `0 ns` in a
   100% CPU spin until deadline.
4. **Dead `PeerWorkerConfig.timeout_ms` (`563, 752`).** `PeerWorkerServer.run`
   passes `null` timeout to `kqueueWait`, never reading `config.timeout_ms`, so
   `max_clients` (`4`) idle sockets wedge all worker slots indefinitely.

**Recommendation.** Disconnect and reset any peer channel whose call exits with
pending TX or timed-out RX; drain or close peers whose `EVFILT_READ` fires
outside an active exchange; and enforce `PeerWorkerConfig.timeout_ms` in
`PeerWorkerServer.run`.

### F17 — High: C ABI uninitialized `*out_*` `struct_size` check, writable `VisionInput.lock_plane()` UAF, and C pointer/length hazards

**Location.** [`src/sdk/c_api.zig:1399, 2121, 2396, 2667, 2782, 3366, 3652, 3701, 4069, 4126, 4150, 4190, 4532, 4957, 4999`](../../src/sdk/c_api.zig);
[`include/aminfer.h:348–355`](../../include/aminfer.h);
[`python/aminfer/engine.py:847–876`](../../python/aminfer/engine.py).

**Defects.**
1. **Reading uninitialized `struct_size` on pure output structs (`c_api.zig`).**
   `aminfer.h` provides `*_init()` helpers only for input parameter structs and
   documents no zero-init requirement on `out_*` results (`ami_sdk_host_vm_info`,
   `ami_sdk_stats`, `ami_sdk_lora_info`, `ami_sdk_spec_stats`,
   `ami_sdk_gen_result`, `ami_sdk_tensor_view`, `ami_sdk_vision_info`). All 9 C
   functions read `out.struct_size` before overwriting `out.*`, triggering C UB
   and failing with `AMI_SDK_ERR_ABI_MISMATCH (-2)` whenever uninitialized stack
   bytes in `0..4` fall in `1 .. sizeof(T) - 1` (reproduced in Appendix A.4).
2. **Writable use-after-free in Python `VisionInput.lock_plane()` (`engine.py:847–876`).**
   `lock_plane(read_only=False)` constructs a writable `_PYBUF_WRITE`
   `memoryview` via `PyMemoryView_FromMemory`, yields it, and unlocks the
   `CVPixelBuffer` in `finally:` *without* calling `mv.release()`. Retaining
   `mv` past `with vis.lock_plane() as mv:` and `vis.close()` allows pure Python
   code (`mv[0, 0] = 1`) to write directly into unmapped/reused memory.
3. **Non-optional inner pointers and `usize` overflows (`c_api.zig`).**
   `ami_sdk_prefill_vision` (`4957`), `ami_sdk_rerank_tokens` (`4069`), and
   `ami_sdk_rerank_texts` (`4150`) declare non-optional inner pointers
   (`?[*]const *const SdkVisionInput`, `?[*]const [*]const u8`), so a `NULL`
   element (e.g. `(NULL, 0)` empty string) causes immediate null-dereference UB;
   unchecked `usize` additions at `4190`, `4126`, and `2782` overflow on crafted
   `size_t` lengths (writing past `rerank_buf` / `prompt_ids_scratch` in
   ReleaseFast).

**Recommendation.** Either remove the `out.struct_size` pre-read on pure output
structs or document/provide `*_init` for them; call `mv.release()` inside
`lock_plane()`'s `finally:` block; change inner pointer types to optional
(`?*const`, `?[*]const`) with explicit null checks; and use `std.math.add` on C
`size_t` inputs.

### F18 — High: path traversal and negative `data_offsets` / unbounded headers in offline checkpoint readers and sidecar resolution

**Location.** [`tools/src/aminfer_tools/amf.py:1172–1228, 1323–1334, 1361–1375`](../../tools/src/aminfer_tools/amf.py),
[`lora.py:333`](../../tools/src/aminfer_tools/lora.py),
[`hy4.py:636–638`](../../tools/src/aminfer_tools/hy4.py),
[`gguf.py:396–408`](../../tools/src/aminfer_tools/gguf.py);
[`src/main.zig:1213–1230`](../../src/main.zig) (`resolveRelativeModelFileAlloc`).

**Defects.**
1. **Path traversal via shard / sidecar filenames.** `amf.resolve_checkpoint`
   (`amf.py:1330`, shared by all offline converters) evaluates
   `shard_path = parent / shard` on `weight_map` entries from
   `model.safetensors.index.json` without rejecting `..`, absolute paths
   (`parent / "/abs/file"` returns `Path("/abs/file")` in `pathlib`), or
   escaping symlinks. On the Zig runtime side, `resolveRelativeModelFileAlloc`
   (`main.zig:1219–1230`) likewise honors absolute paths and `../` segments in
   embedded `__meta.config_json` fields (`dspark.amf_file`, `mtp.amf_file`,
   `ple.shard_files`, `engram_files`).
2. **Negative `data_offsets` and negative `shape` dims (`amf.py:1361–1375`).**
   `_read_shard_header_from_reader` checks `end < start`,
   `end - start != numel * elem`, and `data_base + end > file_size`, never
   `start < 0` or `any(d < 0 for d in shape)`. A crafted `.safetensors` with
   `data_offsets: [-16, 0]` passes validation and sets `offset = data_base - 16`,
   packing raw JSON header bytes into `.amf` / `.amlora` weight blobs.
3. **Unbounded header / nested-array reads.** `hy4.py:636–638` executes
   `fh.read(struct.unpack("<Q", fh.read(8))[0])` with no `256 MiB` or `file_size`
   cap; `gguf.py:396–408` caps a single array at `10^7` items without bounding
   nested `_GGUF_TYPE_ARRAY` depth or cumulative element count; and
   `HttpRangeStreamReader` (`amf.py:1172`) accepts `file://` URLs in `urlopen`.

**Evidence.** Appendix A.5 reproduces both `resolve_checkpoint` traversal outside
the checkpoint directory and `_read_shard_header_from_reader` leaking JSON
header bytes via `data_offsets: [-16, 0]`.

**Recommendation.** Resolve candidate shard/sidecar paths and require
`resolved.is_relative_to(base_dir.resolve())` (rejecting symlinks/FIFOs); enforce
`start >= 0` and `all(d >= 0 for d in shape)`; apply `amf.py`'s `256 MiB` /
`file_size` header check in `hy4.py`; bound GGUF array nesting depth and total
elements; and restrict `HttpRangeStreamReader` to `http`/`https`.

### F19 — Medium: on-disk cache and sparse-table validation gaps (`.ang`, `PleStore`, `.amkvc`, batch completion loops)

**Location.**
- [`src/spec/ngram_nvme.zig:542–639, 696–697, 1710–1711`](../../src/spec/ngram_nvme.zig)
- [`src/storage/ple.zig:202–204, 834–947, 1254–1256`](../../src/storage/ple.zig)
- [`src/mem/prefix_cache.zig:1866–1875`](../../src/mem/prefix_cache.zig)

**Defects.**
1. **Untrusted `.ang` `NgramSlot.cont_len > 8` out-of-bounds read
   (`ngram_nvme.zig:696–697, 1710–1711`).** `slot.cont_tokens` is `[8]u32`, while
   `slot.cont_len` is an unvalidated `u8` read from disk (`NvmeNgramTable.open`
   does not run full-file `verifyFile`). Calling `query(..., out_cont)` with
   `out_cont.len > 8` executes `@memcpy(out_cont[0..n], slot.cont_tokens[0..n])`
   with `n > 8` (panic in safe builds; OOB read across `WiredArena` / page
   boundary in ReleaseFast; `cont_tokens[i]` is also not checked `< vocab_size`).
2. **`PleStore.open` ignores `entry.group_size` (`ple.zig:202–204, 1254–1256`).**
   `PleStore` hardcodes `group_size = 64` for `q2`/`q3`/`q4_affine` sparse
   tables even though `amf.parse` (`src/format/amf.zig:900`) accepts `32` and
   `128`. On a `group_size = 128` `.amf`, `row_param_bytes` is 2x too large and
   `decodeRowF32` reads past `entry.blob_bytes`.
3. **Dropped batch completions permanently wedge `.inflight` slots
   (`ngram_nvme.zig:542–639`, `ple.zig:834–947`).** `for (comps[0..got]) |c| try self.applyCompletion(c)`
   returns early on `comps[0]`'s error after `poll`/`wait` has already popped
   `comps[1..got]` and decremented `inflight` to `0`, leaving slots `1..got`
   stuck in `.inflight` forever (`DomainNgramRegistry.reapBatch` at `1636`
   already applies all completions before returning `first_err`).
4. **Unchecked `u64` sums in `DiskPrefixCache.reload` (`prefix_cache.zig:1868–1872`).**
   `hdr.tokens_offset + hdr.tokens_blob_bytes`, `hdr.kv_offset + hdr.kv_blob_bytes`,
   and `APPLE_PAGE_SIZE + ... + hdr.gdn_blob_bytes` use raw `+` on untrusted
   `.amkvc` header fields (16 KiB-aligned values such as `0x8000_0000_0000_0000`
   pass `page.alignUp` and then overflow `u64`).

**Recommendation.** Clamp/reject `slot.cont_len > MAX_CONT_TOKENS` and validate
`cont_tokens[i] < vocab_size`; require `entry.group_size == 64` (or dispatch per
group size) in `PleStore.open`; apply every reaped completion before returning
the first error; and use `std.math.add` in `DiskPrefixCache.reload`.

### F20 — Medium: constrained GPU speculative fallback emits rejected token, `repetition_penalty` `+Inf -> NaN` logprobs, and Hy4 `k_fine` contamination

**Location.**
- [`src/spec/engine.zig:2892, 2932–2965`](../../src/spec/engine.zig) (`verifyConstrainedGpuBatch`)
- [`src/spec/sampling.zig:103, 248–280, 300–331`](../../src/spec/sampling.zig) &
  [`src/spec/accept.zig:749–820`](../../src/spec/accept.zig)
- [`src/model/m82_runtime.zig:3821–3826, 4479–4490`](../../src/model/m82_runtime.zig)

**Defects.**
1. **`verifyConstrainedGpuBatch` emits a grammar-rejected token (`engine.zig:2933–2965`).**
   When `advanceGrammarOnState(&self.grammar_state, t)` returns `false`, it has
   already mutated `self.grammar_state` in place to `stack_count = 0`, and `break`
   falls through to lines `2957–2965`, which re-reads the same node `u` and
   returns the rejected token `t` as `bonus_token` (contrast `accept.zig:720–723`,
   which snapshots/restores and returns `error.EmptyAllowedSet`).
2. **Post-bias / post-penalty `+Inf` overflow produces `NaN` (`sampling.zig:250, 263`).**
   `SamplingParams.validate()` accepts arbitrarily small `repetition_penalty > 0`
   (e.g. `1e-38`) and large `logit_bias`, while `prepareConstrainedLogits` checks
   `isFinite` only before applying biases and dividing positive logits by
   `repetition_penalty`. Overflow to `+Inf` sets `max_val = +Inf`, turning
   `(l - max_f64)` into `(+Inf - +Inf) = NaN` in `out.logprob` (reproduced in
   Appendix A.6) and causing spurious `error.EmptyAllowedSet` in
   `accept.verifyConstrained`.
3. **Hy4 sparse indexer `k_fine` cache branch contamination (`m82_runtime.zig:3821–3826`).**
   On branching trees, `execHy4Layer` stashes uncommitted node keys at
   `k_fine_dst[base_pos + ni]`, dropping `base_pos + ni >= max_seq_len`. When
   `base_pos` is within `m` tokens of `max_seq_len`, `commitTreeKv` (`4485`)
   skips copying accepted nodes with `base_pos + nid >= max_seq_len`, leaving
   `k_fine_cache[dst_pos]` overwritten with a rejected branch's key vector.

**Recommendation.** Snapshot/restore `self.grammar_state` and return
`error.EmptyAllowedSet` when root node `u == 0` fails grammar advance (or stop
before `u` when `draft_accepted > 0`); clamp post-penalty/post-bias logits to
`<= std.math.floatMax(f32)` (and bound `repetition_penalty >= 1e-4`); and stage
Hy4 tree `k_fine` rows in a dedicated `[max_tree_nodes, idx_d]` scratch slice.

### F21 — Medium: GPU LoRA wide-prefill / batching ring exhaustion, multimodal hot-path allocations, and synthetic SDK vision weights

**Location.** [`src/bridge/apple_bridge.h:25`](../../src/bridge/apple_bridge.h)
(`AMI_UNIFORM_SLOTS_PER_FRAME = 4096`),
[`src/gpu/lora_gpu.zig:380–451, 550`](../../src/gpu/lora_gpu.zig),
[`src/model/graph.zig:3230, 5069–5140, 7725–7771, 7878, 8089–8102, 8325–8327`](../../src/model/graph.zig),
[`src/sdk/c_api.zig:4656–4676, 4811–4834`](../../src/sdk/c_api.zig).

**Defects.**
1. **Metal 4 uniform-ring (`4,096`) and `row_slots_per_frame` (`16,384`) exhaustion
   under GPU LoRA.** `GraphModel` initializes `lora_ws.max_rows = 64`
   (`graph.zig:3230`) while `prefillWideChunk` encodes an entire `512`–`1024`-token
   chunk across all layers in a single `Frame`. At `m = 512` on a 36-layer
   7-target LoRA (`36 * 7 * 8 * 2 = 4,032` LoRA dispatches + ~`360` base
   dispatches > `4,096`), `ami_compute_encode_dispatch_ex` (`apple_bridge.m:1045`)
   fails with `AMI_ERR_OUT_OF_MEMORY`. Similarly, `encodeBatchProjection`
   (`lora_gpu.zig:550`) re-stages `batch_lora_slots` on every projection of
   every layer (`40 * 7 * 64 = 17,920 > 16,384`), failing with
   `error.StagingExhausted`.
2. **Per-request heap / `mmap`+`mlock` allocations on multimodal serving path.**
   `GraphModel.prefillMultimodal` (`graph.zig:7742–7771, 7878, 8089–8102,
   8325–8327`) allocates `mrope_planar`, `all_pos3d`, and `full_toks` via
   `self.allocator` and creates/destroys `WiredArena`s + Metal residency per
   request, violating AGENTS.md's preallocation rule (and uses unchecked `u32`
   addition on `prefix_u32 + total_vis_tokens + suffix_u32` at `7727`).
3. **Hardcoded pseudo-random vision projector in C SDK (`c_api.zig:4668–4674`).**
   `prefillVisionQwen3` synthesizes pseudo-random `proj_f16` weights from
   `0x9e3779b97f4a7c15` on every call instead of loading trained vision weights,
   ignores `params.cancel_token` during vision token encoding (`4763`), and
   passes `retired_pool = null` at `4828` even when `m.num_retired_chunks > 0`.

**Recommendation.** Size `lora_ws.max_rows` to `wide_prefill_max_tokens`,
deduplicate `stageRowSlots` once per frame, and flush or expand uniform slots
when approaching `AMI_UNIFORM_SLOTS_PER_FRAME`; preallocate multimodal scratch
buffers at model load; and either wire real `.amf` vision weights into
`prefillVisionQwen3` or return `AMI_SDK_ERR_UNSUPPORTED` when none are loaded.

### F22 — Medium: LoRA hot-swap ABA slot aliasing, `flexible_dims` bounds gaps, and SME2 batch activation pre-packing

**Location.** [`src/model/lora.zig:668, 1185–1205, 1308–1316, 1411–1425, 1608–1615, 1712–1715, 1791`](../../src/model/lora.zig),
[`src/model/graph.zig:3495`](../../src/model/graph.zig),
[`src/cpu/sme.zig:2899–3140, 3245–3248`](../../src/cpu/sme.zig).

**Defects.**
1. **LoRA slot ABA aliasing.** Active batch slots hold raw `u32` `LoraRegistry`
   indices with no pin/refcount or generation tag: unloading and loading an
   adapter reuses `slot_id` underneath an in-flight decoding request, switching
   its weights mid-response.
2. **`flexible_dims` & alias validation (`lora.zig`).** With `flexible_dims = true`
   (`GraphModel`), `GraphModel.calibrateLoraDora` (`graph.zig:3495`) silently
   skips mismatched `(n, k)` pairs (leaving `dora_row_scale_f32 = 1.0`), CPU
   fallback paths (`lora.zig:1310, 1419`) slice `x_f16`/`y_f16` without bounds
   checks, duplicate `(layer, target)` aliases (`1791`) overwrite prior entries
   and strand wired pool pages, and `applySingleRowF16` (`668`) rejects
   quantized `b_weight` whenever `n > 8192`.
3. **`SmeDispatcher.evalLinearBatchHoisted` (`sme.zig:2899–3140`).** In `.sme2`
   mode, all ops' activations are packed before `ami_sme2_exec_batch` runs
   `ops[0]`, whereas `.scalar_fallback` executes sequentially — any caller
   chaining `ops[1]`'s input to `ops[0]`'s output silently computes stale values
   on SME2. (`SmeLinearWeight.fromAmfEntry` at `3246` also permits invalid
   `q4_affine` + `.lut4_codebook` combinations.)

**Recommendation.** Pin active `LoraRegistry` slots per batch request (with a
generation counter), validate `pair.n == lin.n and pair.k == lin.k` during
`GraphModel` registration, and either reject/flush aliased input-output buffers
in `evalLinearBatchHoisted` or document independent-ops-only semantics.

### F23 — Medium: `AneExpertPredictor.deinit` deadlock, single-tile async data race, `WavefrontPrefill` error drain gap, and `CFGetRetainCount` ownership heuristic

**Location.** [`src/ane/predictor.zig:162–176, 539–555, 933–955, 1050–1073`](../../src/ane/predictor.zig);
[`src/ane/wavefront.zig:441–454`](../../src/ane/wavefront.zig);
[`src/bridge/apple_bridge.m:1632–1668, 2360–2376, 2422–2445`](../../src/bridge/apple_bridge.m).

**Defects.**
1. **`AneExpertPredictor.deinit` deadlock & tile race (`predictor.zig`).**
   `asyncWorkerMain` busy-spins at `while (!self.resp_ring.tryPush(resp))` when
   `resp_ring` (`7` usable slots) is full, while `deinit` spins on
   `req_ring.tryPush(stop_req)` and calls `t.join()` without draining
   `resp_ring` — deadlocking teardown if `>= 7` predictions were queued without
   polling. Moreover, `submitAsync` allows 7 queued requests over a single
   shared `x_tile` and per-block `logits_tile`, racing if `> 1` request is in
   flight.
2. **Un-drained ANE worker on error (`wavefront.zig:441–454`).** After `pushCmd`
   starts `ane_step_fn(ctx, nxt)`, lines `447` and `449` execute `try vmstat.nowNs()`
   before `self.waitDone()`, violating the documented drain-on-error invariant.
3. **`CFGetRetainCount` ownership heuristic (`apple_bridge.m:1633–1642, 2367, 2423–2431`).**
   Inspecting `CFGetRetainCount(pb)` before and after
   `[[MLMultiArray alloc] initWithPixelBuffer:shape:]` is non-atomic under
   concurrent CoreVideo/Metal retain activity, leaks `pb` on `calloc == NULL`
   (`1645`, `2434`), and `ami_ane_tensor_destroy` (`1659`) never calls
   `CVPixelBufferRelease` for any compensating retain.

**Recommendation.** Add an atomic `shutdown` flag checked inside the `resp_ring`
push loop and double-buffer (or restrict `submitAsync` to `1` in-flight step)
`x_tile`/`logits_tile`; `errdefer _ = self.waitDone() catch {};` right after
`pushCmd`; and unconditionally `CVPixelBufferRetain` in `ami_ane_tensor` init
paired with `CVPixelBufferRelease` after `CFRelease(ml_array)` in
`ami_ane_tensor_destroy`.

### F24 — Medium: depth-blind `previous_response_id` pre-scan, `JsonScan.readKeyOrSkip` `> 64 B` key failure, and `chat_template.py` UTF-8 mojibake

**Location.**
- [`src/server/http_sse.zig:2048–2086, 9169, 10147`](../../src/server/http_sse.zig)
- [`src/model/chat_template.zig:559–566, 1098, 1125, 1235`](../../src/model/chat_template.zig)
- [`tools/src/aminfer_tools/chat_template.py:49`](../../tools/src/aminfer_tools/chat_template.py)

**Defects.**
1. **Depth-blind `previous_response_id` lookup (`http_sse.zig:2048`).**
   `admitInferenceRequest` (`10147`) and `handleResponsesInputTokens` (`9169`)
   locate `"previous_response_id"` using `findJsonFieldValueStart`, a flat quote
   scanner that ignores `{`/`[` nesting depth. When top-level
   `previous_response_id` is omitted, any nested `"previous_response_id": "resp_N"`
   inside `metadata`, `tools` schemas, or structured `input` items injects
   `resp_N`'s cached `chatml_buf` from `response_store` into the prompt (a
   cross-request transcript leak in shared-server deployments).
2. **`JsonScan.readKeyOrSkip` halts mid-string on keys `> 32`/`64` bytes
   (`chat_template.zig:559–566`).** On `error.BufferTooSmall`, `readKeyOrSkip`
   returns `null` without advancing `self.pos` to the closing `"`, so
   `expectByte(':')` immediately fails with `error.InvalidJson` on any valid
   `chat_template_spec` or OpenAI/Anthropic `tools` schema containing an unknown
   key longer than `32`/`64` bytes (reproduced in Appendix A.6).
3. **UTF-8 -> Latin-1 mojibake in offline Jinja compilation (`chat_template.py:49`).**
   `m.group(3).encode("utf-8").decode("unicode_escape")` decodes UTF-8 bytes
   `>= 0x80` as Latin-1 whenever a `{% set %}` string literal contains any
   backslash escape (`\n`, `\'`), turning Hy4's fullwidth bar `｜` (`U+FF5C`)
   into `ï½\x9c` (reproduced in Appendix A.5).

**Recommendation.** Restrict `findJsonFieldValueStart` to depth `1` (and unescape
keys); rewind and call `skipValue(0)` on `error.BufferTooSmall` in
`readKeyOrSkip`; and replace `unicode_escape` in `chat_template.py` with
`ast.literal_eval` or an explicit escape-sequence regex over `str`.

### F25 — Medium: CLI/TUI signal `termios` corruption, terminal escape injection from `/health`, and silent 64 KiB `stdin` truncation

**Location.** [`src/cli/chat_repl.zig:1525–1565`](../../src/cli/chat_repl.zig);
[`src/monitor/tui.zig:421–469, 685–711, 862–925, 1955–1977`](../../src/monitor/tui.zig).

**Defects.**
1. **Unrestored terminal state on signals.** `chat_repl.zig:1552–1564` disables
   `ICANON | ECHO | IEXTEN` while keeping `ISIG = true` and installs no signal
   handlers, so `Ctrl+C` (`SIGINT`), `Ctrl+Z`, or `SIGTERM` exits via `SIG_DFL`
   without running `defer tcsetattr(...)`, leaving the user's shell without echo.
   `tui.zig:1956` traps only `SIGWINCH` via `EVFILT_SIGNAL`, leaving raw mode,
   alternate screen (`\x1b[?1049h`), and hidden cursor active on `SIGTERM`/`SIGHUP`.
2. **Terminal ANSI/OSC escape injection (`tui.zig:439–469, 909`).**
   `pollServerHealth` extracts `"model"`, `"active_ngram_pack"`, and
   `"cluster_mode"` without stripping `0x1B` ESC or C0/C1 control characters and
   writes them directly to the user's terminal (`findJsonValueAfterKey` at `421`
   also scans inside JSON string values).
3. **Silent 64 KiB pipe truncation (`chat_repl.zig:1525–1539`).** Non-interactive
   `cat large.txt | aminfer chat` stops reading at `DEFAULT_LINE_BUF_BYTES`
   (64 KiB) without checking for EOF or returning `error.PromptTooLarge`,
   silently truncating piped prompts (potentially mid-UTF-8 codepoint).

**Recommendation.** Register `SIGINT`/`SIGTERM`/`SIGHUP` restoration in both
`chat_repl.zig` and `tui.zig`; filter `< 0x20`, `0x7F`, and `U+0080..U+009F`
before rendering `/health` strings in `tui.zig`; and detect non-EOF after
`line_buf.len` in `chat_repl.zig` to report `error.PromptTooLarge`.

### L1 — Low: Metal 4 frame ring permits overlapping dependent frames

**Location.** [`src/bridge/apple_bridge.m`](../../src/bridge/apple_bridge.m)
`ami_compute_begin_frame` (938), `ami_compute_encode_dispatch_ex` (barrier at
1056), `ami_compute_commit_frame` (1174); `AMI_MAX_FRAMES_IN_FLIGHT = 4`
([`apple_bridge.h:23`](../../src/bridge/apple_bridge.h)); default 2 frames
([`src/bridge/metal.zig:645`](../../src/bridge/metal.zig)).

**Observation.** `begin_frame` waits only when the *same* slot is still in
flight, and the only barrier the bridge emits is
`barrierAfterEncoderStages:` (intra-encoder). Metal 4 does not track hazards
between command buffers, so a frame that reads what the previous, still-running
frame writes would race. Every model path sampled waits on the previous signal
before `beginFrame` (e.g. `qwen3.zig` 5723/5860 overlap only CPU work), so no
live race was found; the API simply does not prevent one.

**Recommendation.** Either emit a queue-level barrier
(`barrierAfterQueueStages:`) on the first dispatch of each frame, or assert in
`Frame` that the previous frame has signaled, and document the contract in
`apple_bridge.h`.

### L2 — Low: TLS client accepts any server certificate

**Location.** `ami_tls_client_connect_socketpair`
([`apple_bridge.m:7667`](../../src/bridge/apple_bridge.m)); its verify block
(line 7691) calls `complete(true)` unconditionally; exported as
`tls.clientConnectSocketpair` in `src/bridge/tls.zig`.

**Observation.** Only benches and tests use it, against loopback. Nothing in
the name warns a future caller (monitor, cluster) that it performs no
verification.

**Recommendation.** Rename to `…InsecureLoopback` (or restrict it to test
builds) and add a verifying variant before any production use.

### L3 — Low: serve hardening (slowloris, unauthenticated public bind)

**Location.** `DEFAULT_MAX_CONNS = 32`
([`http_sse.zig:47`](../../src/server/http_sse.zig)),
`http_read_timeout_ms = 30_000`, `keepalive_idle_timeout_ms = 60_000`;
`100 Continue` is sent before authentication (line 8057, return value
discarded); `src/main.zig` prints no warning when `-H` is non-loopback and
`--api-key` is absent.

**Failure scenario.** 32 slow or idle sockets hold every slot for 30–60 s and
lock out all clients without credentials, because the key is checked only after
the full request has been buffered.

**Recommendation.** A short header-phase deadline, a per-peer connection cap,
early `401` when a key is configured and the headers carry none, and the
bind-address policy from F2.

### L4 — Low: lenient HTTP number parsing

**Location.** `parseHttpHeaders` (`Content-Length` at
[`http_sse.zig:5566`](../../src/server/http_sse.zig), `Transfer-Encoding` at
5572) and the chunk-size parser.

**Evidence.** `std.fmt.parseInt` accepts a leading `+` and `_` digit
separators: `+5 → 5`, `1_0 → 10`, `-0 → 0` (base 10); `+a → 10`,
`a_b → 171` (base 16) — Appendix A.2. `Transfer-Encoding: gzip, chunked` is
treated as plain chunked.

**Impact.** Request-smuggling desync only matters behind a proxy that parses
differently; direct local use is unaffected.

**Recommendation.** Parse `1*DIGIT` / `1*HEXDIG` manually; accept only
`Transfer-Encoding: chunked` exactly.

### L5 — Low: cluster peer protocol has no authentication

**Location.** [`src/cluster/transport.zig`](../../src/cluster/transport.zig).

**Observation.** Framing is solid (header CRC, payload CRC, `payload_len`
bounded by the receive buffer, exact payload-length validation for
`eval_experts` requests and responses). There is no shared secret or MAC, so any
host that reaches the listener can submit expert evaluations or impersonate a
peer and return wrong activations.

**Recommendation.** Document the trusted-link assumption in the cluster ADR, and
add a PSK + HMAC handshake before multi-Mac deployments.

### L6 — Low: untrusted `config.json` integer casts / `NaN` checks and `.amf` `pread` `>= 2 GiB` limit

**Location.** [`src/model/graph.zig:825–985, 1650–1651`](../../src/model/graph.zig),
[`src/model/qwen3.zig:225`](../../src/model/qwen3.zig),
[`src/format/amf.zig:709–721, 758–770`](../../src/format/amf.zig).

**Observation.** `GraphConfig.fromJson` (`1650–1651`) narrows `u32` JSON
`quantization.bits` / `group_size` via `@intCast` without range checks (`"bits":
256` panics); `GraphConfig.validate` omits `rms_norm_eps` / `rope_theta` checks
and `qwen3.zig:225` (`<= 0.0`) lets `NaN` pass (`NaN <= 0.0` is `false`).
Separately, macOS `pread(2)` fails with `EINVAL` whenever `nbyte > INT_MAX`, so
`amf.preadExact` (`709`) fails on any single wired blob `>= 2 GiB` (clamp each
syscall chunk to `0x7fff_c000`).

### L7 — Low: `GpuSamplerWorkspace` single-buffered CPU writes and latent shader alignment/barrier guards

**Location.** [`src/gpu/sampler.zig:464–550, 1012–1068`](../../src/gpu/sampler.zig);
[`src/kernels/nax_tensorops.metal:380–407`](../../src/kernels/nax_tensorops.metal);
[`src/kernels/quant_formats.metal:2805–2834`](../../src/kernels/quant_formats.metal);
[`src/kernels/dflare_gpu.metal:800`](../../src/kernels/dflare_gpu.metal).

**Observation.** Unlike `GpuLoraWorkspace`, `GpuSamplerWorkspace` writes
`ws.row_cfgs` directly on the CPU into single-buffered shared memory during
`encodeSampleBatch` (a second sample dispatch in one uncommitted frame or an
overlapping frame would overwrite Dispatch 1's parameters). In MSL, three
currently wrapper-guarded paths have latent hazards if called with unaligned
shapes: `nax_dequant_stage_tile` casts unaligned row offsets to `device uint*` /
`ushort*` when `K % 64 != 0`; `ami_act_quant_emulate_f16` (`QF_ACT_FP8_G128`)
executes `if (col >= p.k) return;` before `simd_max` and `threadgroup_barrier`
when `K % 128 != 0`; and `dflare_ctx_kv_norm_rope_store` computes `% p.max_ctx`
without guarding `p.max_ctx > 0`.

### L8 — Low: HTTP cancel response, lone JSON surrogates, `Engine.embed([])`, and `NaN` rerank sort

**Location.** [`src/server/http_sse.zig:1141, 2252, 2340, 9727–9738, 10607–10610`](../../src/server/http_sse.zig);
[`python/aminfer/engine.py:1574–1578`](../../python/aminfer/engine.py);
[`src/model/embed_rerank.zig:106, 156, 339–374`](../../src/model/embed_rerank.zig).

**Observation.** `POST /v1/responses/{id}/cancel` sets `slot.cancelled = true`,
causing `drainSlotTokens` (`10609`) to close the original requester's socket
without an HTTP body (`stream: false`) or terminal `response.cancelled` SSE
event; `JsonReader.skipString` (`2340`) accepts lone UTF-16 surrogates (`\uD800`)
and `writeJsonEscaped` (`1141`) echoes unvalidated UTF-8 bytes; Python
`engine.embed([])` raises `InvalidArgumentError` instead of returning `[]`; and
`scoreCrossEncoderLogitsF16` does not guard `(-Inf) - (-Inf) = NaN` before
`std.mem.sort` in `sortRerankScoresDesc`.

### L9 — Low: `u32` position addition, hybrid layer-0 CoW check, and re-entrant `BatchScheduler.cancelSlot`

**Location.** [`src/model/qwen3.zig:2964, 3993, 4009`](../../src/model/qwen3.zig);
[`src/spec/draft.zig:1785, 4069`](../../src/spec/draft.zig);
[`src/mem/prefix_cache.zig:704`](../../src/mem/prefix_cache.zig);
[`src/runtime/batch_scheduler.zig:433, 666, 908, 930`](../../src/runtime/batch_scheduler.zig).

**Observation.** `base_pos + count` is computed in `u32` before comparing to
`max_seq_len`; `PrefixCache.matchAndPin` (`704`) checks `.active_tile` only on
layer `0` when selecting a CoW tail source (missing per-layer retirement via
`retireEntryLayerToQ8`); and if a `BatchScheduler` `onPrefillDoneFn`/`onTokenFn`
callback both calls `cancelSlot` and returns `false`, `finishSlotInternal` runs
twice.

### T1 — Test strategy gaps

- 8 tests skip when `AMINFER_MODELS_DIR` or their model files are absent
  (`src/model/graph.zig` 3, `src/spec/engine.zig` 2, `src/spec/dspark.zig` 1,
  `src/spec/draft.zig` 1, `src/storage/ple.zig` 1; 19 skip points in total).
  They carry the M9.4 real-HF parity claims, so `zig build test` passing does
  not exercise them.
- Neither `zig build test` nor `just test` runs `python/tests/` (the 9 C/Python
  SDK tests) or `tools/tests/` (the 20 offline converter test modules; `just
  tools-test` is separate), and `src/root.zig` omits direct `test` references to
  `src/bench/{m81,m82,m825}.zig` (relying on transitive `exe` / `runner.zig`
  analysis).
- Existing unit tests missed `F10` (tested pinned parents only), `F11` (tested
  pre-quantized `.q4_affine` rather than `.dense_f32 -> .q4_affine`), and `F17`
  (Python bindings always pre-fill `struct_size`, unlike idiomatic C callers).
- No fuzz or property tests exist for untrusted-input parsers: HTTP header/body
  framing (`F1`), JSON readers (`F24`), `.amf` / `.amlora` / `.ang` / `.amkvc`
  headers (`F18`, `F19`), `tokenizer.json`, chat templates, ImageIO / ISO-BMFF
  media (`F8`, `F9`), and cluster frames.
- Zig caches passing `run test` steps; delivery notes should state whether a
  result was freshly executed or replayed from cache.

### A1 — Maintainability

- File sizes conflict with AGENTS.md's "one responsibility per file":
  `src/model/graph.zig` 30,979 lines, `src/server/http_sse.zig` 17,649,
  `src/bench/runner.zig` 16,903, `src/model/qwen3.zig` 16,614,
  `src/gpu/hybrid.zig` 16,270, `src/spec/dflare_dfly.zig` 12,081,
  `src/main.zig` 11,791.
- Duplicated runtime logic: `Qwen3BatchRuntime` and `GraphBatchRuntime` carry
  near-identical slot teardown (`resetSlotState`, `qwen3.zig` ~11434 vs
  `graph.zig` ~20346), so fixes such as F4 and F12 must land twice. A third copy
  of the Vyukov bounded queue lives in `src/storage/aio.zig:540` (`WorkRing`,
  MPMC) next to `src/runtime/ring.zig`.
- The SDK maps internal errors to status codes by comparing `@errorName`
  strings (`src/sdk/c_api.zig` around lines 1300–1352), which silently drifts
  when an error is renamed. An exhaustive `switch` over the error set would
  fail to compile instead.

## 5. What is solid

- **Memory primitives** (`mem/page.zig`, `arena.zig`, `budget.zig`) are small,
  explicit and tested; an `mlock` failure is an error, never a silent fallback
  to pageable memory.
- **ARMv9.2 SME2 assembly** (`src/cpu/sme2_kernels.s`, all 3,996 lines and 66
  entrypoints): `SME2_PROLOGUE` saves `x19–x30` and `d8–d15` on a 16-byte-aligned
  stack frame *before* `smstart` (preventing Streaming SVE register zeroing from
  clobbering caller Neon callee-saved state), performs zero `[sp]` accesses
  while in streaming mode, executes `smstop` before restoring `d8–d15` on every
  return and `cbz` early-exit path, and matches all 66 Zig `extern fn`
  declarations and `KernelArgs` / `BatchKernelTask` layouts bit-for-bit.
- **Cross-language ABI layout parity**: field-by-field verification across all
  195 uniform/descriptor structs shared between `src/kernels/*.metal` and
  `src/gpu/*.zig`, and all 14 C/Python structs shared across
  `include/aminfer.h`, `src/sdk/c_api.zig`, `python/aminfer/_bindings.py`, and
  `apple_bridge.m`, found zero size, alignment, padding, or field-order
  mismatches. Active caller-reachable `threadgroup_barrier` sites across all 11
  compute shaders are threadgroup-uniform and promote device offsets to 64-bit
  `ulong`.
- **`.amf` parsing** validates header CRC, directory bounds and CRC, per-dtype
  byte math with checked arithmetic, blob alignment and non-overlap; the
  `qwen3` and `GraphModel` loaders check tensor shapes against the config
  (`validateLinear2D`, `bindLinear`).
- **Lock-free rings & Dekker wakeups** (`runtime/ring.zig`, `cpu/sme.zig`): SPSC
  and Vyukov MPSC use correct acquire/release pairs with 128-byte cache-line
  padding, and `SmeDispatcher` uses `.seq_cst` store-buffering between `epoch`
  and `sleeping_workers`.
- **Speculative tree & suffix automaton structures**: `SuffixAutomaton` (`2N+1`
  states / `3N+4` transitions at `<= 0.5` load factor), `DraftTree` (`<= 64`
  nodes), EAGLE-3 `d2t < vocab_size` load-time checks, PEARL `errdefer
  waitIdle()`, and `SpecEngine`'s `avail_seq - 1` depth clamp (preventing KV
  overflow on full-tree + bonus acceptance) are well-bounded.
- **HTTP steady-state allocation & media URL policy**: steady-state text serving
  allocates only in `Server.init`, `MAX_JSON_DEPTH = 64` bounds recursion, API
  keys are compared in constant time, `/v1/ngram-packs/select` switches only
  pre-registered packs (`[A-Za-z0-9._-]`), and `http://`/`https://`/`file://`
  media URLs are rejected in favor of inline `data:...;base64,` payloads.
- **ObjC Metal/IOSurface core & Python stdlib discipline**: out-parameters are
  nulled first, NoCopy wrapping re-checks 16 KiB address/length on both Zig and
  ObjC sides, `mach_host_self()` send rights are deallocated in `vmstat.zig`,
  `zig fmt --check` is clean, and Python offline tools use zero `pickle`,
  `eval`, or `shell=True`.

## 6. Nits

- **N1** — `WiredRegion.deinit` discards the `munlock` result
  ([`src/mem/page.zig`](../../src/mem/page.zig)); `WiredArena.initFromAlignedSlice`
  and `offsetOf` ([`src/mem/arena.zig`](../../src/mem/arena.zig)) guard
  alignment only with `std.debug.assert`, which ReleaseFast removes — weaker
  than "enforced in every build mode".
- **N2** — `.gitignore` does not cover `python/*.egg-info/` (currently
  untracked in `git status`); `python/uv.lock` is untracked while
  `tools/uv.lock` is tracked — pick one policy.
- **N3** — `GrammarRuntime` drops pushdown stacks beyond
  `MAX_ACTIVE_STACKS = 16` without signaling
  ([`src/spec/grammar.zig:375`](../../src/spec/grammar.zig)); an ambiguous
  grammar can then mask valid tokens. Surface a metric or an error.
- **N4** — `frame.commit() catch return false` / `compute.wait(...) catch
  return false` in the prefix CoW path (`qwen3.zig` ~11428) drops the error
  cause; propagate or log it.
- **N5** — `is_isobmff_container`
  ([`src/bridge/apple_bridge.m:4650, 5217`](../../src/bridge/apple_bridge.m))
  treats any `ftyp` payload as video without falling back to ImageIO, so single
  HEIC/AVIF images passed to `decodeVideoBytesInto` fail with
  `error.VisionFailed` instead of using the documented single-image fallback.
- **N6** — `ami_compute_make_vision_frames_resident`
  ([`apple_bridge.m:2473`](../../src/bridge/apple_bridge.m)) calls
  `[user_res commit]` without `[user_res requestResidency]`; `DequantPipelines`
  ([`src/gpu/dequant.zig:1129`](../../src/gpu/dequant.zig)) allocates
  `NaxRuntimeState` via `std.heap.c_allocator` instead of an explicit allocator.
- **N7** — `AneStreamWorker.run`
  ([`src/gpu/attention.zig:4102`](../../src/gpu/attention.zig), Spike S4 bench
  helper) uses a multi-line bare `catch { ... }` without capturing the error
  value (it sets `self.failed = true` and returns, so the failure is recorded,
  not silently ignored).

## 7. Suggested order of work

1. **HTTP ingress & server availability (`F1`, `F6`, `F7`, `F8`, `F9`, `F2`, `L3`)**:
   fix the `decodeChunkedInPlace` overflow (`F1`), gate keep-alive slot reuse on
   `!worker_active` and fix `.finished` store order (`F6`), move `/v1/embeddings`
   and `/v1/rerank` off the `kqueue` thread (`F7`), check ImageIO dimensions
   before decoding and cap VideoToolbox GOP decode counts (`F8`, `F9`), and
   tighten CORS / `Host` / bind-address defaults (`F2`, `L3`).
2. **Core cache, tokenizer/template, and model execution safety (`F10`, `F11`, `F12`, `F13`, `F14`, `F20`)**:
   pin `parent_id` across `PrefixCache` eviction (`F10`), fix
   `packF32AsQ4Panel64NoHeap`'s 640 B layout (`F11`), reconcile tokenizer vs
   model `vocab_size` and bitmask zeroing (`F12`), add `allow_special = false`
   for untrusted message/rerank text (`F13`), add the missing Hy4 `ni == 63` /
   SME2 prefill / `commitTreeKv` bounds checks (`F14`), and fix
   `verifyConstrainedGpuBatch` + post-penalty `+Inf` clamping (`F20`).
3. **Async I/O, GPU timeout teardown, and cluster resilience (`F15`, `F16`, `F19`, `F23`)**:
   drain/cancel in-flight AIO/pread/GPU work before unmapping wired or
   `IOSurface` memory (`F15`), harden `ExpertParallelCoordinator` TX/RX/kqueue
   handling (`F16`), validate `.ang` `cont_len`, `PleStore` `group_size`, and
   `.amkvc` header math (`F19`), and fix `AneExpertPredictor` / CoreML
   `CVPixelBuffer` ownership (`F23`).
4. **SDK concurrency/ABI, offline tools, GPU LoRA/vision, and CLI/TUI (`F3`, `F17`, `F18`, `F21`, `F22`, `F24`, `F25`)**:
   add per-engine concurrency guards (`F3`), fix `*out_*` `struct_size`,
   `lock_plane()` `mv.release()`, and nullable inner C pointers (`F17`), enforce
   path-containment and non-negative `data_offsets` in converters (`F18`), size
   GPU LoRA workspaces for wide prefill and preallocate multimodal scratch
   (`F21`, `F22`), fix `readKeyOrSkip`, depth-1 `previous_response_id`, and
   `chat_template.py` unescaping (`F24`), and harden CLI/TUI signal/escape
   handling (`F25`).
5. **Process-wide invariants, fuzzing, and modularization (`F4`, `F5`, `T1`, `A1`)**:
   replace teardown `catch unreachable` (`F4`), introduce a process-wide wired
   ledger (`F5`), wire Python tests and parser fuzz harnesses into CI (`T1`),
   and split oversized files along existing module seams (`A1`).

## Appendix A — reproductions

### A.1 F1 chunked overflow

A verbatim copy of `decodeChunkedInPlace` (lines 5636–5684) with
`neonFindCrlf` / `neonFindCrlfCrlf` replaced by `std.mem.indexOf`, fed the body
`"fffffffffffffffe\r\n0\r\n\r\n"` inside a 4096-byte buffer:

```text
Debug:        thread … panic: integer overflow
              chunked_repro.zig:40:45 in decodeChunkedInPlace
                  if (body_buf.len - scan < chunk_len + 2) return .incomplete;
ReleaseSafe:  exit 134 (panic: integer overflow)
ReleaseFast:  exit 139 (SIGSEGV)
```

The equivalent wire request (not sent to a live server in this review):

```text
POST /v1/chat/completions HTTP/1.1\r\n
Host: 127.0.0.1\r\n
Transfer-Encoding: chunked\r\n
\r\n
fffffffffffffffe\r\n
0\r\n
\r\n
```

### A.2 `std.fmt.parseInt` leniency (Zig 0.17.1)

```text
base10     +5 -> 5
base10    1_0 -> 10
base10     -0 -> 0
base10   0x10 -> InvalidCharacter
base16     +a -> 10
base16    a_b -> 171
```

### A.3 F12 `VocabIndex.fillAllowedTokenBitmask` trailing bitmask words left unzeroed

Standalone `zig test` against `src/spec/grammar.zig` (`VocabIndex` with `32`
tokens = `1` `u32` word, `out_mask` sized for a padded `64`-token model = `2`
words):

```text
[Repro S1] after fillAllowedTokenBitmask(tok=32, model=64): word[0]=0x00000001, word[1]=0xffffffff (unzeroed!)
All 1 tests passed.
```

### A.4 F17 C ABI uninitialized `out_info.struct_size` rejection

Calling `ami_sdk_host_vm_snapshot` in `zig-out/lib/libaminfer.dylib` with
non-zero uninitialized stack bytes (`struct_size = 1 < sizeof(ami_sdk_host_vm_info)`):

```text
=== 4. C ABI uninitialized out_info.struct_size -> AMI_SDK_ERR_ABI_MISMATCH ===
ami_sdk_host_vm_snapshot with uninit struct_size=1 -> rc=-2 ('HostVmInfo struct_size mismatch')
```

### A.5 F18 & F24 Python offline toolchain (`amf.py` & `chat_template.py`)

Executed with `uv run --directory tools python` against `aminfer_tools.amf` and
`aminfer_tools.chat_template`:

```text
=== 1. amf.resolve_checkpoint path traversal ===
traversed shard: outside_secret.txt escaped parent: True
=== 2. amf._read_shard_header_from_reader negative data_offsets ===
accepted negative start: offset=64 (< data_base=80), leaked_header_bytes=b'ets": [-16, 0]}}'
=== 3. chat_template._resolve_jinja_string_vars UTF-8 mojibake ===
decoded user_hdr: '<\xef\xbd\x9chy_User\xef\xbd\x9c>\n' expected: '<\uff5chy_User\uff5c>\n'
```

### A.6 F20 `repetition_penalty` `NaN` logprob & F24 `JsonScan.readKeyOrSkip` `> 64 B` key failure

Standalone `zig test` against `src/model/chat_template.zig` and
`src/spec/sampling.zig`:

- `ChatTemplate.fromSpecJson` fed valid JSON
  `{"schema_version":1,"dialect":"chatml","aaaa...[70 'a's]...":123}` returns
  `error.InvalidJson` (`expectError(error.InvalidJson, ...)` passes).
- `sampling.sampleToken(f32, &.{4.0, 1.0, 0.5, -1.0}, .{ .temperature = 1.0,
  .repetition_penalty = 1e-38, .logprobs = true }, ...)` passes
  `params.validate()` and produces:

```text
[Repro S8] valid params produced NaN logprob: isNan=true
```

## Appendix B — invariant sweeps

| Sweep | Result |
| --- | --- |
| Single-line `catch {}` / `catch \|_\| {}` in `src/` | 0 hits; 1 multi-line bare `catch { ... }` in bench helper `src/gpu/attention.zig:4102` (N7, records `self.failed = true`) |
| `catch unreachable` (all, incl. tests) | 833 hits; non-test hits outside `bufPrint`/`utf8Encode` listed in F4 and `qwen3.zig:6234` (F14) |
| Heap / `WiredArena` allocation on request paths | Steady-state text serving allocates only in `Server.init` (`http_sse.zig:6610–6679`); multimodal prefill in `graph.zig:7742–8327` and `c_api.zig:4656–4676` still allocates per request (F21) |
| NoCopy buffer creation | One site (`ami_buffer_wrap`), guarded by 16 KiB checks in Zig and ObjC |
| Shader / C / Python ABI struct layouts | 195 GPU/Metal structs + 14 C/Python SDK structs verified matching in size, alignment and field offsets |
| SME2 streaming-mode register & stack hygiene | 66/66 entrypoints + batch runner save `x19–x30` & `d8–d15` before `smstart`, zero `[sp]` inside streaming mode, `smstop` on all exits |
| Hardcoded `/Volumes/` paths in `src/`, `tools/`, `python/` | 0 hits |
| `pickle` / `eval` / `exec` / `shell=True` / `os.system` in Python | 0 hits |
| Independent full-host budget computations (non-test) | 51, of which 19 in `src/bench/` (F5) |
