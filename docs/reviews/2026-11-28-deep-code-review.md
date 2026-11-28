# Deep code review — aminfer @ `d4541f5` (2026-11-28)

This report records a whole-repository review of aminfer at commit
`d4541f5258c1167820b32959d2055e0b6debbcfd` (2026-11-28, `main`). It covers
correctness, memory safety, security of the network and file-ingress surfaces,
the engineering invariants in [AGENTS.md](../../AGENTS.md), test coverage and
maintainability. It is a point-in-time review: status of capabilities still
lives only in [docs/status.md](../status.md), and nothing here changes it.

Every finding carries one of these labels:

| Label | Meaning |
| --- | --- |
| **Reproduced** | Triggered in a standalone reproduction; output quoted below. |
| **Confirmed (read)** | The exact code path was read end to end; not executed. |
| **Observation** | A risk or design gap without a concrete failing input today. |

Recommendations are proposals for the maintainer, not decisions.

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

Method: a single reviewer, without delegated agents. First, repository-wide
searches for violations of the engineering invariants and development rules
(`catch {}`, `catch unreachable`, heap allocation on request paths, 16 KiB
alignment checks, `mlock` handling, NoCopy buffer creation, wired-budget
enforcement, unsafe Python calls, hardcoded paths). Second, line-level reading
of the surfaces where a defect has the largest blast radius:

| Depth | Files / areas |
| --- | --- |
| Read in depth | `build.zig`; `src/root.zig`; `src/mem/{page,arena,budget}.zig`; `src/expert/source.zig`; `src/runtime/ring.zig`; `src/format/amf.zig` (parse, open, header/entry validation); `src/server/http_sse.zig` (config, header parsing, body framing, chunked decoding, auth gate, CORS, JSON reader, base64 media decoder, init allocations); `src/cluster/transport.zig` (framing, payload views, receive state machine); `src/bridge/apple_bridge.m` (handles, NoCopy wrap, Metal 4 frame ring, dispatch, TLS identity and client); `src/sdk/c_api.zig` (error mapping, tokenizer locking, engine struct, destroy); `python/aminfer/engine.py` (`close`, `stream`, tensor views) |
| Sampled | `src/model/qwen3.zig` and `src/model/graph.zig` (tensor binding and shape validation, frame commit/wait sites, batch-runtime slot teardown); `src/spec/{dspark,grammar,engine}.zig`; `src/storage/aio.zig` (work ring); `src/main.zig` (budget planning); `src/kernels/dequant_gemm.metal`; `tools/src/aminfer_tools/coreml.py` |
| Not reviewed | Numerical correctness of the model graphs and of most Metal kernels; `src/gpu/{hybrid,attention,moe,vision,inkling2_hy4,quant_formats}.zig` beyond searches; SME2 assembly; ANE wavefront/predictor; tokenizer and chat-template parsers; `src/bench/*`; the per-model Python converters |

Because of the tree size, absence of a finding outside the "read in depth" row
is not evidence of correctness.

## 2. Verification performed

| Check | Result |
| --- | --- |
| `zig build` (Debug) | Exit 0. Served from the Zig build cache. |
| `zig build test --summary all` (Debug) | `10/10 steps succeeded`. Every `run test` step reported `cached`: Zig replayed a previous passing result for identical inputs; the tests were not re-executed in this session. |
| `zig build test -Doptimize=ReleaseSafe --summary all` | Exit 0, same caching caveat. |
| Model-gated tests | 8 tests (19 skip points) return `error.SkipZigTest` when `AMINFER_MODELS_DIR` / their model files are absent; it was not set, so the real-HF parity tests did not run (see T1). |
| F1 reproduction | Standalone copy of `decodeChunkedInPlace` built in Debug, ReleaseSafe and ReleaseFast (Appendix A). |
| `std.fmt.parseInt` leniency (L4) | Standalone check (Appendix A). |
| Live server | Not started (no model loaded); F1 was not sent to a running `aminfer serve`. |

## 3. Summary

| ID | Severity | Area | Finding | Label |
| --- | --- | --- | --- | --- |
| F1 | **Critical** | HTTP server | Integer overflow in chunked body decoding: one unauthenticated request aborts the server (Debug/ReleaseSafe) or writes past the receive buffer (ReleaseFast) | Reproduced |
| F2 | Medium | HTTP server | Defaults `cors_allow_origin = "*"`, no API key and no `Host` validation let any web page drive the local server and read its responses | Confirmed (read) |
| F3 | Medium | C / Python SDK | No per-engine concurrency guard; Python `stream()` runs native generation on a worker thread while `close()` can destroy the engine underneath it | Confirmed (read) |
| F4 | Medium | Runtime robustness | `catch unreachable` on prefix-cache and slot-pool invariants in request teardown turns a refcount bug into undefined behavior in ReleaseFast | Confirmed (read) |
| F5 | Medium | Memory policy | No process-wide wired-memory ledger: components check their own allocation against the full host budget, so the sum can exceed ADR-0012 | Confirmed (read) |
| L1 | Low | Metal 4 bridge | Frame ring allows overlapping dependent frames with only intra-encoder barriers (latent; current callers serialize) | Observation |
| L2 | Low | TLS bridge | Exported TLS client accepts any certificate (test/bench use only today) | Confirmed (read) |
| L3 | Low | HTTP server | Slowloris exposure (32 slots, 30 s/60 s timeouts, auth after full body) and no warning for non-loopback bind without `--api-key` | Confirmed (read) |
| L4 | Low | HTTP server | Lenient number parsing (`+5`, `1_0`, `-0`) and `Transfer-Encoding` substring match | Reproduced |
| L5 | Low | Cluster | Peer protocol has CRC32 integrity but no peer authentication | Observation |
| T1 | — | Tests | Headline real-HF parity tests skip by default; no fuzzing of untrusted-input parsers; cached runs read as fresh | Observation |
| A1 | — | Maintainability | Very large files, duplicated runtime code, error mapping by error-name strings | Observation |
| N1–N4 | Nit | Various | See section 6 | — |

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

**Evidence.** Appendix A: Debug → `panic: integer overflow` at the
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
safe because it only replays lengths already bounded by pass 1. A
`std.testing.fuzz` harness over `parseHttpHeaders` + `decodeChunkedInPlace`
would have found this class of bug (see T1).

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
`a_b → 171` (base 16) — Appendix A. `Transfer-Encoding: gzip, chunked` is
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

### T1 — Test strategy gaps

- 8 tests skip when `AMINFER_MODELS_DIR` or their model files are absent
  (`src/model/graph.zig` 3, `src/spec/engine.zig` 2, `src/spec/dspark.zig` 1,
  `src/spec/draft.zig` 1, `src/storage/ple.zig` 1; 19 skip points in total).
  They carry the M9.4 real-HF parity claims, so `zig build test` passing does
  not exercise them. Recommendation: a documented "with models" test
  invocation and a visible skip count in delivery notes.
- No fuzz or property tests for the parsers that face untrusted input: HTTP
  header/body framing, JSON request reader, `.amf` directory, `tokenizer.json`,
  chat templates, grammar compilation, cluster frames. F1 is the kind of bug
  these would catch.
- Zig caches passing `run test` steps; a rerun with unchanged inputs prints
  `cached`. Delivery notes should say whether a result was re-executed or
  replayed.

### A1 — Maintainability

- File sizes conflict with AGENTS.md's "one responsibility per file":
  `src/model/graph.zig` 30,979 lines, `src/server/http_sse.zig` 17,649,
  `src/bench/runner.zig` 16,903, `src/model/qwen3.zig` 16,614,
  `src/gpu/hybrid.zig` 16,270, `src/spec/dflare_dfly.zig` 12,081,
  `src/main.zig` 11,791.
- Duplicated runtime logic: `Qwen3BatchRuntime` and `GraphBatchRuntime` carry
  near-identical slot teardown (`resetSlotState`, `qwen3.zig` ~11434 vs
  `graph.zig` ~20346), so fixes such as F4 must land twice. A third copy of the
  Vyukov bounded queue lives in `src/storage/aio.zig:540` (`WorkRing`, MPMC)
  next to `src/runtime/ring.zig`.
- The SDK maps internal errors to status codes by comparing `@errorName`
  strings (`src/sdk/c_api.zig` around lines 1300–1352), which silently drifts
  when an error is renamed. An exhaustive `switch` over the error set would
  fail to compile instead.

## 5. What is solid

- **Memory primitives** (`mem/page.zig`, `arena.zig`, `budget.zig`) are small,
  explicit and tested; an `mlock` failure is an error, never a silent fallback
  to pageable memory.
- **`.amf` parsing** validates header CRC, directory bounds and CRC, per-dtype
  byte math with checked arithmetic, blob alignment and non-overlap; the
  `qwen3` and `GraphModel` loaders check tensor shapes against the config
  (`validateLinear2D`, `bindLinear`).
- **Lock-free rings** (`runtime/ring.zig`): SPSC and Vyukov MPSC use correct
  acquire/release pairs, 128-byte padding is asserted, wrap-around is tested;
  the server's job ring has exactly one producer and one consumer thread.
- **HTTP request path**: every allocation happens in `Server.init`; the JSON
  reader is depth-limited (`MAX_JSON_DEPTH = 64`); the streaming base64 media
  decoder is bounds-checked; API keys are compared in constant time; grammar
  compilation runs within fixed capacities.
- **ObjC bridge**: out-parameters are nulled first, every handle is
  NULL-checked, NoCopy wrapping re-checks 16 KiB address/length on the ObjC
  side, `__bridge_retained`/`CFRelease` are paired, and ObjC work sits inside
  `@autoreleasepool`.
- **Metal kernels** (sampled): 64-bit offset math (`ulong(row) * row_bytes`),
  per-row guards, and threadgroup-uniform early returns before barriers.
- **Offline tools**: standard library only; no `pickle`, `eval` or shell
  invocation (`coreml.py` calls `xcrun` with an argument list).

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

## 7. Suggested order of work

1. F1 fix with regression tests (smallest change, largest risk).
2. F2 defaults and the bind-address policy (with L3).
3. F3 SDK concurrency guard and Python `close()` semantics.
4. F4 invariant handling plus the CI search.
5. F5 process-wide wired ledger.
6. T1: fuzz harnesses for untrusted-input parsers and a documented
   "with models" test run.
7. A1: split the largest files along existing seams (HTTP framing vs routing
   vs protocol encoders; per-architecture graph builders), starting with the
   files touched by items 1–5.

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

## Appendix B — invariant sweeps

| Sweep | Result |
| --- | --- |
| `catch {}` / `catch \|_\| {}` in `src/` | 0 hits |
| `catch unreachable` (all, incl. tests) | 833 hits; non-test hits outside `bufPrint`/`utf8Encode` listed in F4 |
| Heap allocation in `http_sse.zig` outside tests | Only `Server.init` (lines 6610–6679) |
| NoCopy buffer creation | One site (`ami_buffer_wrap`), guarded by 16 KiB checks in Zig and ObjC |
| Hardcoded `/Volumes/` paths in `src/`, `tools/`, `python/` | 0 hits |
| `pickle` / `eval` / `exec` / `shell=True` / `os.system` in Python | 0 hits |
| Independent full-host budget computations (non-test) | 51, of which 19 in `src/bench/` (F5) |
