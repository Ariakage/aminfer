//! Bare-metal zero-allocation macOS `kqueue` HTTP/1.1 + SSE inference server
//! (capabilities C7 & C8, `docs/architecture/runtime-pipeline.md`).
//!
//! Architecture:
//! - Pure Zig `std` + Darwin `std.c.kqueue` / `kevent64` (`EVFILT_READ`,
//!   `EVFILT_WRITE`, `EVFILT_USER` cross-thread lock-free wakeup, `O_NONBLOCK`,
//!   `SO_REUSEADDR`, `TCP_NODELAY`, and Darwin `SO_NOSIGPIPE`).
//! - All connection slots, HTTP receive buffers, formatted ChatML prompt buffers,
//!   outbound TX buffers, non-streaming response accumulators, per-slot `SpscRing`
//!   token rings, server-to-worker `SpscRing` job queue, and embedded 32-byte
//!   `RequestRng` (`D11` Xoshiro256) states are preallocated once in `Server.init`.
//! - Steady-state request parsing, `D11` hardware TRNG seeding (`getentropy`),
//!   UTF-8 streaming assembly, `<think>...</think>` splitting, JSON escaping, and
//!   SSE / JSON response emission perform zero dynamic heap allocations and zero
//!   swallowed errors.
//! - Stateless API contract:
//!   - `GET /health` (includes `active_batch_size`, `mean_batch_size`, `prefix_hit_tokens`)
//!   - `GET /v1/models`
//!   - `GET /v1/ngram-packs` (loaded `.ang` NVMe domain packs & hit stats)
//!   - `POST /v1/ngram-packs/select` (atomic runtime default domain switch)
//!   - `POST /v1/chat/completions` (OpenAI streaming SSE & JSON, `reasoning_content`, `seed_used`, optional `ngram_domain`)
//!   - `POST /v1/responses` (stateless OpenAI Responses adapter)
//!   - `POST /v1/messages` (stateless Anthropic Messages adapter)

const std = @import("std");
const ring = @import("../runtime/ring.zig");
const rng_mod = @import("../runtime/rng.zig");
const batch_sched = @import("../runtime/batch_scheduler.zig");
const ngram_nvme = @import("../spec/ngram_nvme.zig");
const grammar_mod = @import("../spec/grammar.zig");
const sampling_mod = @import("../spec/sampling.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const qos = @import("../platform/qos.zig");

/// Apple Silicon page size for backing slab alignment.
pub const APPLE_PAGE_SIZE: usize = 16384;
/// Maximum byte length of a per-request `"ngram_domain"` identifier.
pub const MAX_NGRAM_DOMAIN_BYTES: usize = ngram_nvme.MAX_DOMAIN_ID_BYTES;

pub const DEFAULT_MAX_CONNS: usize = 32;
pub const DEFAULT_MAX_ACTIVE_REQUESTS: usize = 16;
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 64 * 1024;
pub const DEFAULT_MAX_PROMPT_BYTES: usize = 64 * 1024;
pub const DEFAULT_TX_BUF_BYTES: usize = 64 * 1024;
pub const DEFAULT_NONSTREAM_REASONING_BYTES: usize = 16 * 1024;
pub const DEFAULT_NONSTREAM_CONTENT_BYTES: usize = 32 * 1024;
pub const DEFAULT_TOKEN_RING_CAP: usize = 256;
pub const DEFAULT_JOB_RING_CAP: usize = 64;
pub const DEFAULT_MAX_TOKENS: u32 = 256;

/// Maximum UTF-8 bytes carried in one lock-free `TokenEvent`.
pub const MAX_CHUNK_TEXT_BYTES: usize = 128;
pub const MAX_MODEL_ID_BYTES: usize = 128;
pub const MAX_JSON_DEPTH: usize = 64;
pub const MAX_API_KEY_BYTES: usize = 256;
pub const MAX_UNIX_SOCKET_PATH_BYTES: usize = 103;
pub const MAX_ANTHROPIC_VERSION_BYTES: usize = 32;
pub const MAX_RESPONSE_ID_BYTES: usize = 48;
pub const MAX_TOOL_NAME_BYTES: usize = 64;
pub const MAX_TOOL_CALL_ID_BYTES: usize = 48;
pub const MAX_TOOLS_PER_REQUEST: usize = 16;
pub const MAX_STORED_TOOL_CALLS: usize = 8;
pub const DEFAULT_STATEFUL_RESPONSE_SLOTS: usize = 32;
pub const DEFAULT_STATEFUL_RESPONSE_PROMPT_BYTES: usize = 16 * 1024;
pub const DEFAULT_STATEFUL_RESPONSE_CONTENT_BYTES: usize = 8 * 1024;
pub const DEFAULT_STATEFUL_RESPONSE_REASONING_BYTES: usize = 4 * 1024;
pub const MAX_EMBEDDING_INPUTS: usize = 32;
pub const MAX_RERANK_DOCS: usize = 64;
pub const MAX_EMBEDDING_DIM: usize = 8192;

/// Minimum contiguous TX headroom required before popping another streaming
/// `TokenEvent` from a slot's SPSC ring (prevents truncation under TCP backpressure).
const STREAM_TX_HEADROOM_BYTES: usize = 2048;

const UDATA_LISTEN: u64 = std.math.maxInt(u64);
const UDATA_WAKEUP: u64 = std.math.maxInt(u64) - 1;
const UDATA_UNIX_LISTEN: u64 = std.math.maxInt(u64) - 2;
const USER_IDENT_WAKEUP: u64 = 1;
const WORKER_IDENT_WAKEUP: u64 = 2;

const STATIC_503_BODY: []const u8 =
    "{\"error\":{\"message\":\"server busy: all preallocated request slots are in use\",\"type\":\"server_error\",\"code\":\"server_overloaded\"}}";

const STATIC_503_RESPONSE: []const u8 = std.fmt.comptimePrint(
    "HTTP/1.1 503 Service Unavailable\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{s}",
    .{ STATIC_503_BODY.len, STATIC_503_BODY },
);

const STATIC_100_CONTINUE: []const u8 = "HTTP/1.1 100 Continue\r\n\r\n";

pub const Error = error{
    InvalidConfig,
    InvalidIpv4,
    OutOfMemory,
    SocketFailed,
    SetSockOptFailed,
    FcntlFailed,
    BindFailed,
    ListenFailed,
    GetSockNameFailed,
    KqueueFailed,
    KeventFailed,
    ThreadSpawnFailed,
    AlreadyStarted,
    RingInitFailed,
};

pub const ParseError = error{
    InvalidJson,
    MissingRequiredField,
    InvalidFieldValue,
    UnsupportedFeature,
    BufferTooSmall,
    DepthExceeded,
};

pub const Endpoint = enum(u8) {
    chat_completions = 0,
    responses = 1,
    messages = 2,
};

pub const EffectiveSeedSource = enum(u8) {
    greedy = 0,
    explicit = 1,
    hardware_trng = 2,

    pub fn name(self: EffectiveSeedSource) []const u8 {
        return switch (self) {
            .greedy => "greedy",
            .explicit => "explicit",
            .hardware_trng => "hardware_trng",
        };
    }
};

pub const FinishReason = enum(u8) {
    stop = 0,
    length = 1,
    cancelled = 2,
    err = 3,
    tool_calls = 4,
    stop_sequence = 5,

    pub fn openAiName(self: FinishReason) []const u8 {
        return switch (self) {
            .stop, .stop_sequence => "stop",
            .length => "length",
            .cancelled => "stop",
            .err => "stop",
            .tool_calls => "tool_calls",
        };
    }

    pub fn anthropicName(self: FinishReason) []const u8 {
        return switch (self) {
            .stop => "end_turn",
            .length => "max_tokens",
            .cancelled, .stop_sequence => "stop_sequence",
            .err => "end_turn",
            .tool_calls => "tool_use",
        };
    }
};

/// `W19` model selector split (`"<base>:<adapter>"`).
pub const ModelTag = struct {
    base: []const u8,
    adapter: ?[]const u8 = null,
};

/// Splits `"<base>:<adapter>"` (`W19` multi-LoRA selector) with zero allocation.
/// Empty base or trailing colon without adapter falls back cleanly to `adapter = null`.
pub fn parseModelTag(model_id: []const u8) ModelTag {
    if (std.mem.lastIndexOfScalar(u8, model_id, ':')) |colon| {
        if (colon > 0 and colon + 1 < model_id.len) {
            return .{
                .base = model_id[0..colon],
                .adapter = model_id[colon + 1 ..],
            };
        }
    }
    return .{ .base = model_id, .adapter = null };
}

/// Constant-time API key comparison (`C18`, zero branch timing leaks on key bytes).
pub fn constantTimeEql(a: []const u8, b: []const u8) bool {
    const max_len = @max(a.len, b.len);
    var diff: usize = a.len ^ b.len;
    var i: usize = 0;
    while (i < max_len) : (i += 1) {
        const ba: u8 = if (i < a.len) a[i] else 0;
        const bb: u8 = if (i < b.len) b[i] else 0;
        diff |= @as(usize, ba ^ bb);
    }
    return diff == 0;
}

pub const ToolChoiceMode = enum(u8) {
    auto = 0,
    none = 1,
    required = 2,
    named = 3,
};

pub const ReasoningEffort = enum(u8) {
    none = 0,
    low = 1,
    medium = 2,
    high = 3,

    pub fn name(self: ReasoningEffort) []const u8 {
        return switch (self) {
            .none => "none",
            .low => "low",
            .medium => "medium",
            .high => "high",
        };
    }
};

pub const ResponseFormatKind = enum(u8) {
    text = 0,
    json_object = 1,
    json_schema = 2,
    gbnf = 3,
};

pub const ParsedToolSpec = struct {
    name_buf: [MAX_TOOL_NAME_BYTES]u8 = @splat(0),
    name_len: u8 = 0,
    strict: bool = false,
    /// Raw JSON slice of `parameters` (OpenAI) or `input_schema` (Anthropic) inside the request body.
    parameters_json: []const u8 = "",

    pub fn nameSlice(self: *const ParsedToolSpec) []const u8 {
        return self.name_buf[0..self.name_len];
    }
};

pub const EmbedEncodingFormat = enum(u8) {
    float = 0,
    base64 = 1,
};

pub const EmbedRequest = struct {
    model: []const u8,
    model_tag: ModelTag,
    inputs: [MAX_EMBEDDING_INPUTS][]const u8 = undefined,
    input_count: usize = 0,
    dimensions: ?u32 = null,
    encoding_format: EmbedEncodingFormat = .float,
};

pub const EmbedResult = struct {
    /// Flattened `input_count * dim` f32 embeddings written into `out_vectors`.
    dim: usize,
    prompt_tokens: u32 = 0,
};

pub const EmbeddingHandler = struct {
    ctx: *anyopaque,
    embedFn: *const fn (
        ctx: *anyopaque,
        req: *const EmbedRequest,
        out_vectors: []f32,
    ) anyerror!EmbedResult,
};

pub const RerankRequest = struct {
    model: []const u8,
    model_tag: ModelTag,
    query: []const u8,
    documents: [MAX_RERANK_DOCS][]const u8 = undefined,
    document_count: usize = 0,
    top_n: ?u32 = null,
    return_documents: bool = false,
};

pub const RerankItem = struct {
    index: u32,
    relevance_score: f32,
};

pub const RerankResult = struct {
    count: usize,
    prompt_tokens: u32 = 0,
};

pub const RerankHandler = struct {
    ctx: *anyopaque,
    rerankFn: *const fn (
        ctx: *anyopaque,
        req: *const RerankRequest,
        out_items: []RerankItem,
    ) anyerror!RerankResult,
};

/// Extended live telemetry fields exposed on `GET /health` for `aminfer monitor --server`.
pub const HealthTelemetry = struct {
    active_prefill_slots: u32 = 0,
    active_decode_slots: u32 = 0,
    tok_per_s: f64 = 0.0,
    decode_share: f32 = 0.0,
    prefix_cache_hits: u64 = 0,
    retired_q8_chunks: u64 = 0,
    ang_pack: []const u8 = "",
    tb_peers: u32 = 0,
};

pub const TokenEventKind = enum(u8) {
    prefill_done = 0,
    token_text = 1,
    finished = 2,
    failed = 3,
};

pub const TokenEvent = struct {
    generation: u64 = 0,
    kind: TokenEventKind = .token_text,
    finish_reason: FinishReason = .stop,
    token_count: u32 = 0,
    cached_tokens: u32 = 0,
    text_len: u16 = 0,
    text: [MAX_CHUNK_TEXT_BYTES]u8 = undefined,

    pub fn textSlice(self: *const TokenEvent) []const u8 {
        return self.text[0..self.text_len];
    }
};

pub const ServerConfig = struct {
    host: [4]u8 = .{ 127, 0, 0, 1 },
    /// `0` binds an ephemeral loopback port; read back via `server.boundPort()`.
    port: u16 = 8080,
    model_id: []const u8 = "aminfer-local",
    wired_mib: u64 = 0,
    max_conns: usize = DEFAULT_MAX_CONNS,
    /// Maximum concurrent in-flight inference requests before returning `503`.
    /// Must be `>= 1` and `<= max_conns`.
    max_active_requests: usize = DEFAULT_MAX_ACTIVE_REQUESTS,
    max_request_bytes: usize = DEFAULT_MAX_REQUEST_BYTES,
    max_prompt_bytes: usize = DEFAULT_MAX_PROMPT_BYTES,
    tx_buf_bytes: usize = DEFAULT_TX_BUF_BYTES,
    nonstream_reasoning_bytes: usize = DEFAULT_NONSTREAM_REASONING_BYTES,
    nonstream_content_bytes: usize = DEFAULT_NONSTREAM_CONTENT_BYTES,
    /// Power-of-two capacity of each connection's lock-free token SPSC ring.
    token_ring_cap: usize = DEFAULT_TOKEN_RING_CAP,
    /// Power-of-two capacity of the server->worker lock-free job SPSC ring.
    job_ring_cap: usize = DEFAULT_JOB_RING_CAP,
    backlog: u32 = 128,
    /// Steer the kqueue network thread to `QOS_CLASS_USER_INITIATED` (`runtime-pipeline.md`).
    apply_qos: bool = true,
    /// Maximum concurrent decode batch size in `BatchScheduler` (`0` = `min(max_active_requests, MAX_BATCH_SLOTS)`).
    max_batch_size: usize = 0,
    /// Chunked prefill token budget per continuous-batching iteration (`C14`).
    max_prefill_chunk_tokens: u32 = batch_sched.DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
    /// Preallocated `u32` prompt token capacity per continuous-batching slot.
    max_prompt_tokens: usize = batch_sched.DEFAULT_MAX_PROMPT_TOKENS,
    /// Enable ChatML pre-last-user message-boundary `PrefixCache` checkpoints (`C15`, ADR-0020).
    message_boundary_cache: bool = true,
    /// Target decode wall-time share (`0.0..=1.0`, `0.0` = off) under prefill/decode contention (`C15`, ADR-0020).
    decode_share: f32 = batch_sched.DEFAULT_DECODE_SHARE,
    /// Narrow background prefill chunks (`8K -> 256`, `16K -> 128`) while decode slots are active (`C15`, ADR-0020).
    adaptive_prefill_narrow: bool = true,
    /// Optional hot-swappable multi-pack `.ang` NVMe domain registry exposed on `/v1/ngram-packs*`.
    ngram_registry: ?*ngram_nvme.DomainNgramRegistry = null,
    /// Opt-in constant-time API key (`C18`); when non-null and non-empty, requires matching
    /// `Authorization: Bearer <key>` or `x-api-key: <key>` on `/v1/*` routes.
    api_key: ?[]const u8 = null,
    /// Optional `AF_UNIX` local IPC socket path (`--unix-socket <path>`, `C18`) served
    /// concurrently on the same `kqueue` loop alongside TCP.
    unix_socket_path: ?[]const u8 = null,
    /// Fixed preallocated capacity of stateful `/v1/responses` slot table (`evict_first` LRU).
    stateful_response_slots: usize = DEFAULT_STATEFUL_RESPONSE_SLOTS,
    /// Optional native Zig BPE `Tokenizer` (`W5`) backing `POST /v1/messages/count_tokens`.
    tokenizer: ?*const tokenizer_mod.Tokenizer = null,
    /// Optional `C21` embeddings backend callback for `POST /v1/embeddings`.
    embedding_handler: ?EmbeddingHandler = null,
    /// Optional `C21` rerank backend callback for `POST /v1/rerank`.
    rerank_handler: ?RerankHandler = null,
    /// Require `anthropic-version` header on `/v1/messages*` when set to `true` (default `false`
    /// keeps legacy clients working while still validating non-empty header format when provided).
    require_anthropic_version: bool = false,
};

/// Parses a dotted-quad IPv4 address string (e.g. `"127.0.0.1"`, `"0.0.0.0"`).
pub fn parseIpv4(text: []const u8) Error![4]u8 {
    var out: [4]u8 = undefined;
    var octet_idx: usize = 0;
    var it = std.mem.splitScalar(u8, text, '.');
    while (it.next()) |part| {
        if (octet_idx >= 4 or part.len == 0) return error.InvalidIpv4;
        const val = std.fmt.parseInt(u8, part, 10) catch return error.InvalidIpv4;
        out[octet_idx] = val;
        octet_idx += 1;
    }
    if (octet_idx != 4) return error.InvalidIpv4;
    return out;
}

pub const ServerStats = struct {
    active_conns: usize,
    active_requests: usize,
    total_requests: u64,
    rejected_503_count: u64,
    cancelled_count: u64,
    wired_mib: u64,
    active_batch_size: u32 = 0,
    max_observed_batch_size: u32 = 0,
    total_batch_steps: u64 = 0,
    total_slot_tokens: u64 = 0,
    mean_batch_size: f64 = 0.0,
    prefix_hit_tokens: u64 = 0,
    prefix_hit_requests: u64 = 0,
    qos_decode_steps: u64 = 0,
    qos_prefill_steps: u64 = 0,
    narrowed_prefill_chunks: u64 = 0,
    message_boundary_checkpoints: u64 = 0,
};

/// Admitted inference job handed from the `kqueue` server thread to the inference
/// worker through the preallocated lock-free `SpscRing(InferenceJob)`.
pub const InferenceJob = struct {
    slot_index: u16,
    generation: u64,
    request_id: u64,
    endpoint: Endpoint,
    /// Preformatted Qwen3 ChatML prompt stored in the slot's preallocated buffer.
    /// Valid for the entire duration of `EngineHandler.generateFn` or until the
    /// slot finishes in `runBatchedWorkerLoop`.
    prompt: []const u8,
    /// Exact byte offset in `prompt` immediately before the final ChatML `user`
    /// turn (`<|im_start|>user\n`) on multi-turn requests (`>= 2` messages where
    /// the final turn is `user`, `C15` ADR-0020); `null` otherwise.
    last_user_msg_byte_offset: ?usize = null,
    /// Dedicated 32-byte `RequestRng` (`D11`) owned by this slot for this request.
    rng: *rng_mod.RequestRng,
    seed_used: u64,
    seed_source: EffectiveSeedSource,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
    ignore_eos: bool,
    enable_thinking: bool,
    /// Optional per-request `.ang` domain pack ID (`""` selects active default pack).
    ngram_domain: []const u8 = "",
    /// `W19` parsed `"<base>:<adapter>"` model selector (`adapter == null` when base-only).
    model_base: []const u8 = "",
    lora_adapter: ?[]const u8 = null,
    /// Shared `C18` sampling controls (`top_p`, `top_k`, `min_p`, penalties, sparse `logit_bias`, `logprobs`).
    sampling: sampling_mod.SamplingParams = .{},
    /// Preallocated `C18` `stop` / `stop_sequences` configuration.
    stop_config: *const sampling_mod.StopConfig = &EMPTY_STOP_CONFIG,
    /// Optional compiled `C22` grammar program (`null` = strict zero-overhead bypass).
    grammar_program: ?*const grammar_mod.GrammarProgram = null,
    response_format: ResponseFormatKind = .text,
    tool_choice: ToolChoiceMode = .auto,
    parallel_tool_calls: bool = true,
    has_tools: bool = false,
    reasoning_effort: ?ReasoningEffort = null,
    thinking_budget_tokens: ?u32 = null,
    cache_control_pin: bool = false,
    stream_include_usage: bool = false,
};

const EMPTY_STOP_CONFIG: sampling_mod.StopConfig = .{};

/// Decoupled single-slot inference callback interface implemented by real `Qwen3Model` /
/// `SpecEngine` runners or deterministic mock generators in tests.
pub const EngineHandler = struct {
    ctx: *anyopaque,
    generateFn: *const fn (ctx: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void,
};

pub const InferenceBackend = EngineHandler;

/// Zero-allocation tokenization result carrying both the total prompt token
/// count and optional exact `message_boundary_token` mapped from
/// `InferenceJob.last_user_msg_byte_offset` (`C15`, ADR-0020).
pub const TokenizeResult = struct {
    prompt_len: usize,
    message_boundary_token: ?u32 = null,
};

/// Iteration-level continuous-batching inference handler (`C14` / `C15`) bridging
/// HTTP ChatML requests and per-slot SSE `TokenStream`s with `batch_sched.BatchScheduler`.
pub const BatchEngineHandler = struct {
    ctx: *anyopaque,
    /// Tokenizes `job.prompt` directly into the slot's preallocated `out_tokens` buffer,
    /// returning the number of prompt tokens written (`>= 1`).
    tokenizeFn: *const fn (
        ctx: *anyopaque,
        job: *const InferenceJob,
        out_tokens: []u32,
    ) batch_sched.BackendError!usize,
    /// Optional boundary-aware tokenizer (`C15`, ADR-0020): tokenizes `job.prompt`
    /// into `out_tokens` and maps `job.last_user_msg_byte_offset` to an exact
    /// token index (`message_boundary_token`) with zero heap allocation.
    /// When `null`, `runBatchedWorkerLoop` invokes `tokenizeFn` with `message_boundary_token = null`.
    tokenizeWithBoundaryFn: ?*const fn (
        ctx: *anyopaque,
        job: *const InferenceJob,
        out_tokens: []u32,
    ) batch_sched.BackendError!TokenizeResult = null,
    /// Detokenizes `token_id` into raw UTF-8 bytes in `out_bytes` (`0` heap allocs).
    detokenizeFn: *const fn (
        ctx: *anyopaque,
        token_id: u32,
        out_bytes: []u8,
    ) []const u8,
    /// Continuous-batching backend (`lookupPrefix`, `prefillSlot`, `stepDecodeBatch`, `releaseSlot`).
    backend: batch_sched.BatchBackend,
    /// Optional default EOS/stop token IDs for this model.
    eos_token_ids: [batch_sched.MAX_STOP_TOKENS]u32 = @splat(0),
    eos_token_count: u8 = 0,
};

pub const BatchInferenceBackend = BatchEngineHandler;

// ---------------------------------------------------------------------------
// C23: ARM64 NEON structural JSON & HTTP vector scanners (ADR-0023)
//   - Explicit 16-byte / 64-byte unrolled `@Vector(16, u8)` NEON fast paths
//   - `shrn`-equivalent nibble-mask reduction (compatible with `-Dcpu=apple_m1`)
//   - Clean `< 16 B` scalar tail fallback, zero dynamic heap allocation
// ---------------------------------------------------------------------------

const NeonVec16 = @Vector(16, u8);
const NeonVec8u16 = @Vector(8, u16);
const NeonVec8u8 = @Vector(8, u8);
const NeonShift4: @Vector(8, u4) = @splat(4);

inline fn neonBoolToMask(cond: @Vector(16, bool)) NeonVec16 {
    return @select(u8, cond, @as(NeonVec16, @splat(0xFF)), @as(NeonVec16, @splat(0)));
}

/// Compresses a 16-byte `0x00`/`0xFF` lane mask into a 64-bit nibble bitmask
/// (`4` bits per lane, lowering to ARM64 NEON `shrn v.8b, v.8h, #4`).
/// Compatible with `-Dcpu=apple_m1` through `apple_m5` without SVE.
inline fn neonMaskToU64(mask: NeonVec16) u64 {
    const pairs: NeonVec8u16 = @bitCast(mask);
    const nibbles: NeonVec8u8 = @truncate(pairs >> NeonShift4);
    return @bitCast(nibbles);
}

inline fn neonJsonSpecialMask(v: NeonVec16) NeonVec16 {
    return neonBoolToMask(v == @as(NeonVec16, @splat('"'))) |
        neonBoolToMask(v == @as(NeonVec16, @splat('\\'))) |
        neonBoolToMask(v < @as(NeonVec16, @splat(0x20)));
}

inline fn neonQuoteOrBackslashMask(v: NeonVec16) NeonVec16 {
    return neonBoolToMask(v == @as(NeonVec16, @splat('"'))) |
        neonBoolToMask(v == @as(NeonVec16, @splat('\\')));
}

inline fn neonJsonNonWsMask(v: NeonVec16) NeonVec16 {
    const is_ws = neonBoolToMask(v == @as(NeonVec16, @splat(' '))) |
        neonBoolToMask(v == @as(NeonVec16, @splat('\t'))) |
        neonBoolToMask(v == @as(NeonVec16, @splat('\r'))) |
        neonBoolToMask(v == @as(NeonVec16, @splat('\n')));
    return ~is_ws;
}

inline fn isJsonWsByte(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\r' or b == '\n';
}

/// Returns the byte offset of the first JSON string boundary or escape/control
/// byte (`"`, `\`, or `< 0x20`) in `bytes`, or `bytes.len` if none exists.
/// Uses 64-byte and 16-byte unrolled ARM64 NEON vectors with `< 16 B` scalar tail.
pub fn neonFindJsonStringSpecial(bytes: []const u8) usize {
    var pos: usize = 0;
    const len = bytes.len;

    while (pos + 64 <= len) : (pos += 64) {
        const m0 = neonJsonSpecialMask(bytes[pos..][0..16].*);
        const m1 = neonJsonSpecialMask(bytes[pos + 16 ..][0..16].*);
        const m2 = neonJsonSpecialMask(bytes[pos + 32 ..][0..16].*);
        const m3 = neonJsonSpecialMask(bytes[pos + 48 ..][0..16].*);
        const any = (m0 | m1) | (m2 | m3);
        if (@reduce(.Max, any) != 0) {
            const b0 = neonMaskToU64(m0);
            if (b0 != 0) return pos + (@as(usize, @ctz(b0)) >> 2);
            const b1 = neonMaskToU64(m1);
            if (b1 != 0) return pos + 16 + (@as(usize, @ctz(b1)) >> 2);
            const b2 = neonMaskToU64(m2);
            if (b2 != 0) return pos + 32 + (@as(usize, @ctz(b2)) >> 2);
            const b3 = neonMaskToU64(m3);
            return pos + 48 + (@as(usize, @ctz(b3)) >> 2);
        }
    }

    while (pos + 16 <= len) : (pos += 16) {
        const m = neonJsonSpecialMask(bytes[pos..][0..16].*);
        const u = neonMaskToU64(m);
        if (u != 0) return pos + (@as(usize, @ctz(u)) >> 2);
    }

    while (pos < len) : (pos += 1) {
        const b = bytes[pos];
        if (b == '"' or b == '\\' or b < 0x20) return pos;
    }
    return len;
}

/// Returns the byte offset of the first `"` or `\` in `bytes`, or `bytes.len`.
pub fn neonFindQuoteOrBackslash(bytes: []const u8) usize {
    var pos: usize = 0;
    const len = bytes.len;

    while (pos + 64 <= len) : (pos += 64) {
        const m0 = neonQuoteOrBackslashMask(bytes[pos..][0..16].*);
        const m1 = neonQuoteOrBackslashMask(bytes[pos + 16 ..][0..16].*);
        const m2 = neonQuoteOrBackslashMask(bytes[pos + 32 ..][0..16].*);
        const m3 = neonQuoteOrBackslashMask(bytes[pos + 48 ..][0..16].*);
        const any = (m0 | m1) | (m2 | m3);
        if (@reduce(.Max, any) != 0) {
            const b0 = neonMaskToU64(m0);
            if (b0 != 0) return pos + (@as(usize, @ctz(b0)) >> 2);
            const b1 = neonMaskToU64(m1);
            if (b1 != 0) return pos + 16 + (@as(usize, @ctz(b1)) >> 2);
            const b2 = neonMaskToU64(m2);
            if (b2 != 0) return pos + 32 + (@as(usize, @ctz(b2)) >> 2);
            const b3 = neonMaskToU64(m3);
            return pos + 48 + (@as(usize, @ctz(b3)) >> 2);
        }
    }

    while (pos + 16 <= len) : (pos += 16) {
        const m = neonQuoteOrBackslashMask(bytes[pos..][0..16].*);
        const u = neonMaskToU64(m);
        if (u != 0) return pos + (@as(usize, @ctz(u)) >> 2);
    }

    while (pos < len) : (pos += 1) {
        const b = bytes[pos];
        if (b == '"' or b == '\\') return pos;
    }
    return len;
}

/// Returns the byte offset of the first occurrence of `needle` in `bytes`, or `bytes.len`.
pub fn neonFindByte(bytes: []const u8, needle: u8) usize {
    var pos: usize = 0;
    const len = bytes.len;
    const target: NeonVec16 = @splat(needle);

    while (pos + 64 <= len) : (pos += 64) {
        const m0 = neonBoolToMask(@as(NeonVec16, bytes[pos..][0..16].*) == target);
        const m1 = neonBoolToMask(@as(NeonVec16, bytes[pos + 16 ..][0..16].*) == target);
        const m2 = neonBoolToMask(@as(NeonVec16, bytes[pos + 32 ..][0..16].*) == target);
        const m3 = neonBoolToMask(@as(NeonVec16, bytes[pos + 48 ..][0..16].*) == target);
        const any = (m0 | m1) | (m2 | m3);
        if (@reduce(.Max, any) != 0) {
            const b0 = neonMaskToU64(m0);
            if (b0 != 0) return pos + (@as(usize, @ctz(b0)) >> 2);
            const b1 = neonMaskToU64(m1);
            if (b1 != 0) return pos + 16 + (@as(usize, @ctz(b1)) >> 2);
            const b2 = neonMaskToU64(m2);
            if (b2 != 0) return pos + 32 + (@as(usize, @ctz(b2)) >> 2);
            const b3 = neonMaskToU64(m3);
            return pos + 48 + (@as(usize, @ctz(b3)) >> 2);
        }
    }

    while (pos + 16 <= len) : (pos += 16) {
        const m = neonBoolToMask(@as(NeonVec16, bytes[pos..][0..16].*) == target);
        const u = neonMaskToU64(m);
        if (u != 0) return pos + (@as(usize, @ctz(u)) >> 2);
    }

    while (pos < len) : (pos += 1) {
        if (bytes[pos] == needle) return pos;
    }
    return len;
}

/// Returns the byte offset of the first non-RFC-8259 whitespace byte
/// (not `' '`, `'\t'`, `'\r'`, or `'\n'`) in `bytes`, or `bytes.len`.
pub fn neonSkipJsonWhitespace(bytes: []const u8) usize {
    const len = bytes.len;
    if (len == 0 or !isJsonWsByte(bytes[0])) return 0;

    var pos: usize = 0;
    while (pos + 64 <= len) : (pos += 64) {
        const nw0 = neonJsonNonWsMask(bytes[pos..][0..16].*);
        const nw1 = neonJsonNonWsMask(bytes[pos + 16 ..][0..16].*);
        const nw2 = neonJsonNonWsMask(bytes[pos + 32 ..][0..16].*);
        const nw3 = neonJsonNonWsMask(bytes[pos + 48 ..][0..16].*);
        const any = (nw0 | nw1) | (nw2 | nw3);
        if (@reduce(.Max, any) != 0) {
            const b0 = neonMaskToU64(nw0);
            if (b0 != 0) return pos + (@as(usize, @ctz(b0)) >> 2);
            const b1 = neonMaskToU64(nw1);
            if (b1 != 0) return pos + 16 + (@as(usize, @ctz(b1)) >> 2);
            const b2 = neonMaskToU64(nw2);
            if (b2 != 0) return pos + 32 + (@as(usize, @ctz(b2)) >> 2);
            const b3 = neonMaskToU64(nw3);
            return pos + 48 + (@as(usize, @ctz(b3)) >> 2);
        }
    }

    while (pos + 16 <= len) : (pos += 16) {
        const nw = neonJsonNonWsMask(bytes[pos..][0..16].*);
        const u = neonMaskToU64(nw);
        if (u != 0) return pos + (@as(usize, @ctz(u)) >> 2);
    }

    while (pos < len and isJsonWsByte(bytes[pos])) : (pos += 1) {}
    return pos;
}

/// NEON-accelerated HTTP `\r\n` boundary scanner.
pub fn neonFindCrlf(bytes: []const u8) ?usize {
    if (bytes.len < 2) return null;
    var pos: usize = 0;
    const limit = bytes.len - 1;
    while (pos < limit) {
        const rel = neonFindByte(bytes[pos..limit], '\r');
        const idx = pos + rel;
        if (idx >= limit) return null;
        if (bytes[idx + 1] == '\n') return idx;
        pos = idx + 1;
    }
    return null;
}

/// NEON-accelerated HTTP header terminator (`\r\n\r\n`) scanner.
pub fn neonFindCrlfCrlf(bytes: []const u8) ?usize {
    if (bytes.len < 4) return null;
    var pos: usize = 0;
    const limit = bytes.len - 3;
    while (pos < limit) {
        const rel = neonFindByte(bytes[pos..limit], '\r');
        const idx = pos + rel;
        if (idx >= limit) return null;
        if (bytes[idx + 1] == '\n' and bytes[idx + 2] == '\r' and bytes[idx + 3] == '\n') {
            return idx;
        }
        pos = idx + 1;
    }
    return null;
}

// ---------------------------------------------------------------------------
// Zero-allocation fixed-slice writer & JSON escaping
// ---------------------------------------------------------------------------

pub const FixedWriter = struct {
    buf: []u8,
    pos: usize = 0,

    pub fn init(buf: []u8) FixedWriter {
        return .{ .buf = buf, .pos = 0 };
    }

    pub fn remaining(self: *const FixedWriter) usize {
        return self.buf.len - self.pos;
    }

    pub fn written(self: *const FixedWriter) []const u8 {
        return self.buf[0..self.pos];
    }

    pub fn writeByte(self: *FixedWriter, b: u8) error{BufferTooSmall}!void {
        if (self.pos >= self.buf.len) return error.BufferTooSmall;
        self.buf[self.pos] = b;
        self.pos += 1;
    }

    pub fn writeAll(self: *FixedWriter, bytes: []const u8) error{BufferTooSmall}!void {
        if (bytes.len > self.buf.len - self.pos) return error.BufferTooSmall;
        @memcpy(self.buf[self.pos .. self.pos + bytes.len], bytes);
        self.pos += bytes.len;
    }

    pub fn print(self: *FixedWriter, comptime fmt: []const u8, args: anytype) error{BufferTooSmall}!void {
        const s = std.fmt.bufPrint(self.buf[self.pos..], fmt, args) catch return error.BufferTooSmall;
        self.pos += s.len;
    }

    /// Writes `bytes` escaped for inside a JSON string literal (without outer quotes).
    /// Vectorized with 16-byte / 64-byte ARM64 NEON scanning (`C23`, ADR-0023).
    pub fn writeJsonEscaped(self: *FixedWriter, bytes: []const u8) error{BufferTooSmall}!void {
        const hex = "0123456789abcdef";
        var run_start: usize = 0;
        while (run_start < bytes.len) {
            const span = neonFindJsonStringSpecial(bytes[run_start..]);
            if (span > 0) {
                try self.writeAll(bytes[run_start .. run_start + span]);
            }
            const i = run_start + span;
            if (i >= bytes.len) break;
            const b = bytes[i];
            switch (b) {
                '"' => try self.writeAll("\\\""),
                '\\' => try self.writeAll("\\\\"),
                0x08 => try self.writeAll("\\b"),
                0x0c => try self.writeAll("\\f"),
                '\n' => try self.writeAll("\\n"),
                '\r' => try self.writeAll("\\r"),
                '\t' => try self.writeAll("\\t"),
                else => {
                    const esc = [6]u8{ '\\', 'u', '0', '0', hex[b >> 4], hex[b & 0x0f] };
                    try self.writeAll(&esc);
                },
            }
            run_start = i + 1;
        }
    }
};

// ---------------------------------------------------------------------------
// Zero-allocation incremental UTF-8 stream assembler (byte-level BPE safe)
// ---------------------------------------------------------------------------

/// Assembles arbitrary byte slices from byte-level BPE tokens into valid UTF-8
/// chunks without heap allocation. Buffers incomplete 1..3-byte trailing UTF-8
/// sequences until completion and replaces malformed bytes with U+FFFD.
pub const Utf8StreamAssembler = struct {
    carry: [4]u8 = undefined,
    carry_len: u8 = 0,

    pub fn reset(self: *Utf8StreamAssembler) void {
        self.carry_len = 0;
    }

    /// Consumes `input` bytes and invokes `emit_fn(ctx, valid_utf8_slice)` with
    /// one or more valid UTF-8 slices. Returns `false` immediately if `emit_fn`
    /// returns `false` (e.g. cancelled stream).
    pub fn push(
        self: *Utf8StreamAssembler,
        input: []const u8,
        ctx: anytype,
        comptime emit_fn: fn (@TypeOf(ctx), []const u8) bool,
    ) bool {
        var pos: usize = 0;

        // First resolve any carried incomplete leading sequence from previous calls.
        if (self.carry_len > 0) {
            const expected_len = utf8ExpectedLen(self.carry[0]);
            while (self.carry_len < expected_len and pos < input.len) {
                const b = input[pos];
                if ((b & 0xC0) != 0x80) {
                    // Truncated sequence interrupted by a non-continuation byte.
                    self.carry_len = 0;
                    if (!emit_fn(ctx, "\xEF\xBF\xBD")) return false;
                    break;
                }
                self.carry[self.carry_len] = b;
                self.carry_len += 1;
                pos += 1;
            }
            if (self.carry_len == expected_len and expected_len > 0) {
                const seq = self.carry[0..expected_len];
                self.carry_len = 0;
                if (isValidUtf8Sequence(seq)) {
                    if (!emit_fn(ctx, seq)) return false;
                } else {
                    if (!emit_fn(ctx, "\xEF\xBF\xBD")) return false;
                }
            }
            if (pos >= input.len) return true;
        }

        // Fast-path scan of the remaining bytes in `input[pos..]`.
        var valid_start: usize = pos;
        while (pos < input.len) {
            const b = input[pos];
            if (b < 0x80) {
                pos += 1;
                continue;
            }
            const seq_len = utf8ExpectedLen(b);
            if (seq_len == 0) {
                // Invalid leading byte.
                if (pos > valid_start) {
                    if (!emit_fn(ctx, input[valid_start..pos])) return false;
                }
                if (!emit_fn(ctx, "\xEF\xBF\xBD")) return false;
                pos += 1;
                valid_start = pos;
                continue;
            }
            if (pos + seq_len > input.len) {
                // Potential incomplete sequence at tail: verify all available trailing bytes
                // are continuation bytes; if so, carry them over.
                const rem = input[pos..];
                var all_cont = true;
                for (rem[1..]) |cb| {
                    if ((cb & 0xC0) != 0x80) {
                        all_cont = false;
                        break;
                    }
                }
                if (pos > valid_start) {
                    if (!emit_fn(ctx, input[valid_start..pos])) return false;
                }
                if (all_cont) {
                    @memcpy(self.carry[0..rem.len], rem);
                    self.carry_len = @intCast(rem.len);
                    return true;
                }
                if (!emit_fn(ctx, "\xEF\xBF\xBD")) return false;
                pos += 1;
                valid_start = pos;
                continue;
            }
            if (isValidUtf8Sequence(input[pos .. pos + seq_len])) {
                pos += seq_len;
            } else {
                if (pos > valid_start) {
                    if (!emit_fn(ctx, input[valid_start..pos])) return false;
                }
                if (!emit_fn(ctx, "\xEF\xBF\xBD")) return false;
                pos += 1;
                valid_start = pos;
            }
        }

        if (pos > valid_start) {
            if (!emit_fn(ctx, input[valid_start..pos])) return false;
        }
        return true;
    }

    /// Flushes any trailing incomplete sequence as U+FFFD at end-of-stream.
    pub fn flush(
        self: *Utf8StreamAssembler,
        ctx: anytype,
        comptime emit_fn: fn (@TypeOf(ctx), []const u8) bool,
    ) bool {
        if (self.carry_len == 0) return true;
        self.carry_len = 0;
        return emit_fn(ctx, "\xEF\xBF\xBD");
    }

    fn utf8ExpectedLen(first: u8) u8 {
        if (first < 0x80) return 1;
        if (first >= 0xC2 and first <= 0xDF) return 2;
        if (first >= 0xE0 and first <= 0xEF) return 3;
        if (first >= 0xF0 and first <= 0xF4) return 4;
        return 0;
    }

    fn isValidUtf8Sequence(seq: []const u8) bool {
        _ = std.unicode.utf8Decode(seq) catch return false;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Zero-allocation streaming `<think>...</think>` state machine
// ---------------------------------------------------------------------------

pub const StreamChannel = enum(u8) {
    content = 0,
    reasoning = 1,
};

/// Splits streaming text into `.reasoning` (inside `<think>...</think>`) and
/// `.content` (outside) with zero heap allocation, handling tags arbitrarily
/// fragmented across chunk/token boundaries.
pub const ThinkSplitter = struct {
    const OPEN_TAG = "<think>";
    const CLOSE_TAG = "</think>";

    in_think: bool = false,
    pending: [CLOSE_TAG.len]u8 = undefined,
    pending_len: u8 = 0,

    pub fn reset(self: *ThinkSplitter, start_in_think: bool) void {
        self.in_think = start_in_think;
        self.pending_len = 0;
    }

    /// Feeds `chunk` through the splitter, invoking `emit_fn(ctx, channel, slice)`
    /// for each non-empty classified span. Propagates any error from `emit_fn`.
    pub fn feed(
        self: *ThinkSplitter,
        chunk: []const u8,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        var i: usize = 0;
        while (i < chunk.len) {
            const tag: []const u8 = if (self.in_think) CLOSE_TAG else OPEN_TAG;
            const channel: StreamChannel = if (self.in_think) .reasoning else .content;

            if (self.pending_len == 0) {
                // Fast path: emit everything up to the next '<' in one slice.
                if (std.mem.indexOfScalarPos(u8, chunk, i, '<')) |lt_pos| {
                    if (lt_pos > i) {
                        try emit_fn(ctx, channel, chunk[i..lt_pos]);
                    }
                    self.pending[0] = '<';
                    self.pending_len = 1;
                    i = lt_pos + 1;
                } else {
                    try emit_fn(ctx, channel, chunk[i..]);
                    return;
                }
            } else {
                const b = chunk[i];
                if (b == tag[self.pending_len]) {
                    self.pending[self.pending_len] = b;
                    self.pending_len += 1;
                    i += 1;
                    if (self.pending_len == tag.len) {
                        self.pending_len = 0;
                        self.in_think = !self.in_think;
                    }
                } else {
                    // Since '<' appears only at index 0 of both `<think>` and `</think>`,
                    // no proper suffix of `pending[1..pending_len]` can start a tag.
                    const flushed = self.pending[0..self.pending_len];
                    self.pending_len = 0;
                    try emit_fn(ctx, channel, flushed);
                    // Re-process `chunk[i]` in the next loop iteration (it may itself be '<').
                }
            }
        }
    }

    /// Flushes any trailing partial tag bytes at end-of-stream.
    pub fn flush(
        self: *ThinkSplitter,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        if (self.pending_len == 0) return;
        const channel: StreamChannel = if (self.in_think) .reasoning else .content;
        const flushed = self.pending[0..self.pending_len];
        self.pending_len = 0;
        try emit_fn(ctx, channel, flushed);
    }
};

// ---------------------------------------------------------------------------
// Zero-allocation streaming `<tool_call>` parser (`C18`)
// ---------------------------------------------------------------------------

pub const ToolCallStreamEvent = union(enum) {
    text: []const u8,
    tool_start: struct {
        index: u16,
        id: []const u8,
        name: []const u8,
    },
    tool_args_delta: struct {
        index: u16,
        id: []const u8,
        delta: []const u8,
    },
    tool_end: struct {
        index: u16,
        id: []const u8,
        name: []const u8,
    },
};

/// Zero-allocation streaming `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
/// parser that handles arbitrary token/byte fragmentation across `<tool_call>` tags,
/// extracts tool `name` and streams `arguments` JSON fragments incrementally as
/// `tool_start`, `tool_args_delta`, and `tool_end` events.
pub const ToolCallParser = struct {
    const OPEN_TAG = "<tool_call>";
    const CLOSE_TAG = "</tool_call>";

    const Mode = enum(u8) {
        outside = 0,
        inside_preamble = 1,
        streaming_args = 2,
        inside_tail = 3,
    };

    mode: Mode = .outside,
    pending: [CLOSE_TAG.len]u8 = undefined,
    pending_len: u8 = 0,
    preamble_buf: [512]u8 = undefined,
    preamble_len: u16 = 0,
    name_buf: [MAX_TOOL_NAME_BYTES]u8 = undefined,
    name_len: u8 = 0,
    id_buf: [MAX_TOOL_CALL_ID_BYTES]u8 = undefined,
    id_len: u8 = 0,
    request_id: u64 = 0,
    call_count: u16 = 0,
    active_index: u16 = 0,
    args_is_string: bool = false,
    args_brace_depth: i32 = 0,
    args_in_string: bool = false,
    args_escape: bool = false,
    args_started: bool = false,
    emitted_any_args: bool = false,

    pub fn reset(self: *ToolCallParser, request_id: u64) void {
        self.mode = .outside;
        self.pending_len = 0;
        self.preamble_len = 0;
        self.name_len = 0;
        self.id_len = 0;
        self.request_id = request_id;
        self.call_count = 0;
        self.active_index = 0;
        self.args_is_string = false;
        self.args_brace_depth = 0;
        self.args_in_string = false;
        self.args_escape = false;
        self.args_started = false;
        self.emitted_any_args = false;
    }

    pub fn hasToolCalls(self: *const ToolCallParser) bool {
        return self.call_count > 0;
    }

    fn formatCallId(self: *ToolCallParser, idx: u16) []const u8 {
        var w = FixedWriter.init(&self.id_buf);
        w.print("call_{d}_{d}", .{ self.request_id, idx }) catch {
            self.id_len = 0;
            return "";
        };
        self.id_len = @intCast(w.pos);
        return self.id_buf[0..self.id_len];
    }

    fn idSlice(self: *const ToolCallParser) []const u8 {
        return self.id_buf[0..self.id_len];
    }

    fn nameSlice(self: *const ToolCallParser) []const u8 {
        return self.name_buf[0..self.name_len];
    }

    pub fn feed(
        self: *ToolCallParser,
        chunk: []const u8,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        var i: usize = 0;
        while (i < chunk.len) {
            if (self.mode == .outside) {
                if (self.pending_len == 0) {
                    if (std.mem.indexOfScalarPos(u8, chunk, i, '<')) |lt| {
                        if (lt > i) {
                            try emit_fn(ctx, ToolCallStreamEvent{ .text = chunk[i..lt] });
                        }
                        self.pending[0] = '<';
                        self.pending_len = 1;
                        i = lt + 1;
                    } else {
                        try emit_fn(ctx, ToolCallStreamEvent{ .text = chunk[i..] });
                        return;
                    }
                } else {
                    const b = chunk[i];
                    if (b == OPEN_TAG[self.pending_len]) {
                        self.pending[self.pending_len] = b;
                        self.pending_len += 1;
                        i += 1;
                        if (self.pending_len == OPEN_TAG.len) {
                            self.pending_len = 0;
                            self.mode = .inside_preamble;
                            self.preamble_len = 0;
                            self.name_len = 0;
                            self.args_is_string = false;
                            self.args_brace_depth = 0;
                            self.args_in_string = false;
                            self.args_escape = false;
                            self.args_started = false;
                            self.emitted_any_args = false;
                        }
                    } else {
                        const flushed = self.pending[0..self.pending_len];
                        self.pending_len = 0;
                        try emit_fn(ctx, ToolCallStreamEvent{ .text = flushed });
                    }
                }
                continue;
            }

            // Inside `<tool_call> ... </tool_call>`: scan for `</tool_call>` tag across chunks.
            const b = chunk[i];
            i += 1;
            if (self.pending_len == 0) {
                if (b == '<' and !self.args_in_string) {
                    self.pending[0] = '<';
                    self.pending_len = 1;
                } else {
                    try self.consumeInsideByte(b, ctx, emit_fn);
                }
            } else {
                if (b == CLOSE_TAG[self.pending_len]) {
                    self.pending[self.pending_len] = b;
                    self.pending_len += 1;
                    if (self.pending_len == CLOSE_TAG.len) {
                        self.pending_len = 0;
                        try self.finishActiveCall(ctx, emit_fn);
                        self.mode = .outside;
                    }
                } else {
                    // Flush false prefix bytes into current inside mode, then re-handle `b`!
                    var tmp: [CLOSE_TAG.len]u8 = undefined;
                    const n = self.pending_len;
                    @memcpy(tmp[0..n], self.pending[0..n]);
                    self.pending_len = 0;
                    for (tmp[0..n]) |pb| {
                        try self.consumeInsideByte(pb, ctx, emit_fn);
                    }
                    if (b == '<' and !self.args_in_string) {
                        self.pending[0] = '<';
                        self.pending_len = 1;
                    } else {
                        try self.consumeInsideByte(b, ctx, emit_fn);
                    }
                }
            }
        }
    }

    fn consumeInsideByte(
        self: *ToolCallParser,
        b: u8,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        switch (self.mode) {
            .outside => unreachable,
            .inside_tail => {},
            .inside_preamble => {
                if (self.preamble_len < self.preamble_buf.len) {
                    self.preamble_buf[self.preamble_len] = b;
                    self.preamble_len += 1;
                }
                try self.tryStartStreamingArgs(ctx, emit_fn);
            },
            .streaming_args => {
                try self.consumeArgByte(b, ctx, emit_fn);
            },
        }
    }

    fn tryStartStreamingArgs(
        self: *ToolCallParser,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        const buf = self.preamble_buf[0..self.preamble_len];
        // Locate `"name"` value first (if not yet captured).
        if (self.name_len == 0) {
            if (findJsonStringField(buf, "name", &self.name_buf)) |n_slice| {
                self.name_len = @intCast(n_slice.len);
            }
        }
        if (self.name_len == 0) return;

        // Locate `"arguments"` or `"input"` key followed by `:` and first value token byte!
        const val_pos = findJsonFieldValueStart(buf, "arguments") orelse
            findJsonFieldValueStart(buf, "input") orelse return;
        if (val_pos >= buf.len) return;

        const first_b = buf[val_pos];
        if (first_b == ' ' or first_b == '\t' or first_b == '\r' or first_b == '\n') return;

        self.active_index = self.call_count;
        self.call_count += 1;
        const call_id = self.formatCallId(self.active_index);

        try emit_fn(ctx, ToolCallStreamEvent{ .tool_start = .{
            .index = self.active_index,
            .id = call_id,
            .name = self.nameSlice(),
        } });

        self.mode = .streaming_args;
        self.args_started = true;
        self.args_is_string = (first_b == '"');
        self.args_brace_depth = 0;
        self.args_in_string = false;
        self.args_escape = false;

        // Replay bytes from `val_pos` onward (`val_pos .. buf.len`) through `consumeArgByte`!
        var replay_copy: [512]u8 = undefined;
        const replay_len = buf.len - val_pos;
        @memcpy(replay_copy[0..replay_len], buf[val_pos..]);
        self.preamble_len = 0;
        for (replay_copy[0..replay_len]) |rb| {
            if (self.mode != .streaming_args) break;
            try self.consumeArgByte(rb, ctx, emit_fn);
        }
    }

    fn consumeArgByte(
        self: *ToolCallParser,
        b: u8,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        if (self.args_is_string) {
            // Arguments provided as an escaped JSON string: `"{\"a\":1}"` -> emit unescaped inner bytes.
            if (!self.args_in_string) {
                if (b == '"') self.args_in_string = true;
                return;
            }
            if (self.args_escape) {
                self.args_escape = false;
                const unesc: u8 = switch (b) {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    else => b,
                };
                const one = [1]u8{unesc};
                self.emitted_any_args = true;
                try emit_fn(ctx, ToolCallStreamEvent{ .tool_args_delta = .{
                    .index = self.active_index,
                    .id = self.idSlice(),
                    .delta = &one,
                } });
                return;
            }
            if (b == '\\') {
                self.args_escape = true;
                return;
            }
            if (b == '"') {
                self.args_in_string = false;
                self.mode = .inside_tail;
                return;
            }
            const one = [1]u8{b};
            self.emitted_any_args = true;
            try emit_fn(ctx, ToolCallStreamEvent{ .tool_args_delta = .{
                .index = self.active_index,
                .id = self.idSlice(),
                .delta = &one,
            } });
            return;
        }

        // Standard JSON object/array/primitive arguments: track nested `{}`/`[]` depth and strings.
        if (self.args_in_string) {
            if (self.args_escape) {
                self.args_escape = false;
            } else if (b == '\\') {
                self.args_escape = true;
            } else if (b == '"') {
                self.args_in_string = false;
            }
        } else {
            if (b == '"') {
                self.args_in_string = true;
            } else if (b == '{' or b == '[') {
                self.args_brace_depth += 1;
            } else if (b == '}' or b == ']') {
                self.args_brace_depth -= 1;
            }
        }

        const one = [1]u8{b};
        self.emitted_any_args = true;
        try emit_fn(ctx, ToolCallStreamEvent{ .tool_args_delta = .{
            .index = self.active_index,
            .id = self.idSlice(),
            .delta = &one,
        } });

        if (!self.args_in_string and self.args_brace_depth == 0) {
            self.mode = .inside_tail;
        }
    }

    fn finishActiveCall(
        self: *ToolCallParser,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        if (self.mode == .inside_preamble) {
            // Arguments omitted (`{"name": "ping"}`): still emit a valid `{}` tool call if name found!
            if (self.name_len == 0) {
                if (findJsonStringField(self.preamble_buf[0..self.preamble_len], "name", &self.name_buf)) |ns| {
                    self.name_len = @intCast(ns.len);
                }
            }
            if (self.name_len > 0) {
                self.active_index = self.call_count;
                self.call_count += 1;
                const call_id = self.formatCallId(self.active_index);
                try emit_fn(ctx, ToolCallStreamEvent{ .tool_start = .{
                    .index = self.active_index,
                    .id = call_id,
                    .name = self.nameSlice(),
                } });
                try emit_fn(ctx, ToolCallStreamEvent{ .tool_args_delta = .{
                    .index = self.active_index,
                    .id = call_id,
                    .delta = "{}",
                } });
                try emit_fn(ctx, ToolCallStreamEvent{ .tool_end = .{
                    .index = self.active_index,
                    .id = call_id,
                    .name = self.nameSlice(),
                } });
            }
            self.preamble_len = 0;
            return;
        }

        if (self.args_started) {
            if (!self.emitted_any_args) {
                try emit_fn(ctx, ToolCallStreamEvent{ .tool_args_delta = .{
                    .index = self.active_index,
                    .id = self.idSlice(),
                    .delta = "{}",
                } });
            }
            try emit_fn(ctx, ToolCallStreamEvent{ .tool_end = .{
                .index = self.active_index,
                .id = self.idSlice(),
                .name = self.nameSlice(),
            } });
        }
        self.args_started = false;
        self.preamble_len = 0;
    }

    pub fn flush(
        self: *ToolCallParser,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        if (self.mode == .outside) {
            if (self.pending_len > 0) {
                const fl = self.pending[0..self.pending_len];
                self.pending_len = 0;
                try emit_fn(ctx, ToolCallStreamEvent{ .text = fl });
            }
            return;
        }
        // Unclosed `<tool_call>` at EOF: close gracefully if active.
        self.pending_len = 0;
        try self.finishActiveCall(ctx, emit_fn);
        self.mode = .outside;
    }
};

fn findJsonFieldValueStart(buf: []const u8, field_name: []const u8) ?usize {
    var i: usize = 0;
    while (i < buf.len) {
        const rel_quote = neonFindByte(buf[i..], '"');
        i += rel_quote;
        if (i >= buf.len) return null;
        const key_start = i + 1;
        i += 1;
        while (i < buf.len) {
            i += neonFindQuoteOrBackslash(buf[i..]);
            if (i >= buf.len) break;
            if (buf[i] == '\\') {
                i += 2;
                continue;
            }
            // buf[i] == '"'
            break;
        }
        if (i >= buf.len) return null;
        const key_slice = buf[key_start..i];
        i += 1; // skip closing '"'
        var j = i + neonSkipJsonWhitespace(buf[i..]);
        if (j < buf.len and buf[j] == ':') {
            j += 1;
            j += neonSkipJsonWhitespace(buf[j..]);
            if (std.mem.eql(u8, key_slice, field_name)) {
                return if (j < buf.len) j else null;
            }
        }
    }
    return null;
}

fn findJsonStringField(buf: []const u8, field_name: []const u8, dst: []u8) ?[]const u8 {
    const start = findJsonFieldValueStart(buf, field_name) orelse return null;
    if (start >= buf.len or buf[start] != '"') return null;
    var r = JsonReader{ .src = buf[start..] };
    return r.readSmallString(dst) catch null;
}

// ---------------------------------------------------------------------------
// Zero-allocation RFC 8259 JSON parser & ChatML request decoder
// ---------------------------------------------------------------------------

pub const ParsedRequest = struct {
    endpoint: Endpoint,
    model_len: u16,
    prompt_len: usize,
    /// Exact byte offset in `prompt_buf` right before the final `<|im_start|>user\n`
    /// turn on multi-turn chat requests (`>= 2` messages ending with a `user` turn).
    last_user_msg_byte_offset: ?usize = null,
    max_tokens: u32,
    temperature: f32,
    seed: ?u64,
    stream: bool,
    ignore_eos: bool,
    enable_thinking: bool,
    ngram_domain_buf: [MAX_NGRAM_DOMAIN_BYTES]u8 = @splat(0),
    ngram_domain_len: u8 = 0,

    // C18 extended sampling, stop sequences, tool calling, structured outputs & protocol parity:
    top_p: f32 = 1.0,
    top_k: u32 = 0,
    min_p: f32 = 0.0,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    repetition_penalty: f32 = 1.0,
    logit_bias_entries: [sampling_mod.MAX_LOGIT_BIAS_ENTRIES]sampling_mod.LogitBiasEntry = undefined,
    logit_bias_len: u8 = 0,
    logprobs: bool = false,
    top_logprobs: u8 = 0,
    stop_config: sampling_mod.StopConfig = .{},

    stream_include_usage: bool = false,
    cache_control_pin: bool = false,
    thinking_budget_tokens: ?u32 = null,
    reasoning_effort: ?ReasoningEffort = null,
    store_response: bool = true,

    previous_response_id_buf: [MAX_RESPONSE_ID_BYTES]u8 = @splat(0),
    previous_response_id_len: u8 = 0,

    tools: [MAX_TOOLS_PER_REQUEST]ParsedToolSpec = undefined,
    tool_count: u8 = 0,
    tool_choice: ToolChoiceMode = .auto,
    tool_choice_name_buf: [MAX_TOOL_NAME_BYTES]u8 = @splat(0),
    tool_choice_name_len: u8 = 0,
    parallel_tool_calls: bool = true,

    response_format: ResponseFormatKind = .text,
    response_format_strict: bool = false,
    /// Slice referencing the caller's request body for `json_schema` / `gbnf` compilation.
    response_format_schema_json: []const u8 = "",

    pub fn ngramDomainSlice(self: *const ParsedRequest) []const u8 {
        return self.ngram_domain_buf[0..self.ngram_domain_len];
    }

    pub fn previousResponseIdSlice(self: *const ParsedRequest) []const u8 {
        return self.previous_response_id_buf[0..self.previous_response_id_len];
    }

    pub fn toolChoiceNameSlice(self: *const ParsedRequest) []const u8 {
        return self.tool_choice_name_buf[0..self.tool_choice_name_len];
    }

    pub fn logitBiasSlice(self: *const ParsedRequest) []const sampling_mod.LogitBiasEntry {
        return self.logit_bias_entries[0..self.logit_bias_len];
    }

    pub fn toolsSlice(self: *const ParsedRequest) []const ParsedToolSpec {
        return self.tools[0..self.tool_count];
    }

    pub fn samplingParams(self: *const ParsedRequest) sampling_mod.SamplingParams {
        return .{
            .temperature = self.temperature,
            .top_p = self.top_p,
            .top_k = self.top_k,
            .min_p = self.min_p,
            .presence_penalty = self.presence_penalty,
            .frequency_penalty = self.frequency_penalty,
            .repetition_penalty = self.repetition_penalty,
            .logit_bias = self.logitBiasSlice(),
            .logprobs = self.logprobs,
            .top_logprobs = self.top_logprobs,
        };
    }
};

const JsonReader = struct {
    src: []const u8,
    pos: usize = 0,

    fn skipWs(self: *JsonReader) void {
        self.pos += neonSkipJsonWhitespace(self.src[self.pos..]);
    }

    fn peek(self: *JsonReader) ?u8 {
        self.skipWs();
        if (self.pos >= self.src.len) return null;
        return self.src[self.pos];
    }

    fn expectByte(self: *JsonReader, expected: u8) ParseError!void {
        self.skipWs();
        if (self.pos >= self.src.len or self.src[self.pos] != expected) {
            return error.InvalidJson;
        }
        self.pos += 1;
    }

    fn consumeKeyword(self: *JsonReader, kw: []const u8) bool {
        self.skipWs();
        if (self.src.len - self.pos < kw.len) return false;
        if (!std.mem.eql(u8, self.src[self.pos .. self.pos + kw.len], kw)) return false;
        self.pos += kw.len;
        return true;
    }

    fn readHex4(self: *JsonReader) ParseError!u16 {
        if (self.src.len - self.pos < 4) return error.InvalidJson;
        var val: u16 = 0;
        for (self.src[self.pos .. self.pos + 4]) |ch| {
            const digit: u16 = switch (ch) {
                '0'...'9' => ch - '0',
                'a'...'f' => ch - 'a' + 10,
                'A'...'F' => ch - 'A' + 10,
                else => return error.InvalidJson,
            };
            val = (val << 4) | digit;
        }
        self.pos += 4;
        return val;
    }

    /// Reads a JSON string and writes its unescaped UTF-8 bytes into `writer`.
    /// Vectorized with 16-byte / 64-byte ARM64 NEON string-boundary scanning (`C23`).
    fn readStringInto(self: *JsonReader, writer: *FixedWriter) ParseError!void {
        try self.expectByte('"');
        while (self.pos < self.src.len) {
            const span = neonFindJsonStringSpecial(self.src[self.pos..]);
            if (span > 0) {
                try writer.writeAll(self.src[self.pos .. self.pos + span]);
                self.pos += span;
            }
            if (self.pos >= self.src.len) return error.InvalidJson;
            const b = self.src[self.pos];
            if (b == '"') {
                self.pos += 1;
                return;
            }
            if (b < 0x20) return error.InvalidJson;
            // b == '\\'
            self.pos += 1;
            if (self.pos >= self.src.len) return error.InvalidJson;
            const esc = self.src[self.pos];
            self.pos += 1;
            switch (esc) {
                '"' => try writer.writeByte('"'),
                '\\' => try writer.writeByte('\\'),
                '/' => try writer.writeByte('/'),
                'b' => try writer.writeByte(0x08),
                'f' => try writer.writeByte(0x0c),
                'n' => try writer.writeByte('\n'),
                'r' => try writer.writeByte('\r'),
                't' => try writer.writeByte('\t'),
                'u' => {
                    const hi = try self.readHex4();
                    var cp: u21 = hi;
                    if (hi >= 0xD800 and hi <= 0xDBFF) {
                        if (self.src.len - self.pos < 6 or self.src[self.pos] != '\\' or self.src[self.pos + 1] != 'u') {
                            return error.InvalidJson;
                        }
                        self.pos += 2;
                        const lo = try self.readHex4();
                        if (lo < 0xDC00 or lo > 0xDFFF) return error.InvalidJson;
                        cp = 0x10000 + (@as(u21, hi - 0xD800) << 10) + @as(u21, lo - 0xDC00);
                    } else if (hi >= 0xDC00 and hi <= 0xDFFF) {
                        return error.InvalidJson;
                    }
                    var utf8_tmp: [4]u8 = undefined;
                    const n = std.unicode.utf8Encode(cp, &utf8_tmp) catch return error.InvalidJson;
                    try writer.writeAll(utf8_tmp[0..n]);
                },
                else => return error.InvalidJson,
            }
        }
        return error.InvalidJson;
    }

    fn readSmallString(self: *JsonReader, dst: []u8) ParseError![]const u8 {
        var w = FixedWriter.init(dst);
        try self.readStringInto(&w);
        return w.written();
    }

    /// Reads a JSON object key into `dst`; if the key is longer than `dst.len`,
    /// finishes validating/skipping the string and returns `null` (unknown key).
    fn readKeyOrSkip(self: *JsonReader, dst: []u8) ParseError!?[]const u8 {
        const saved = self.pos;
        var w = FixedWriter.init(dst);
        self.readStringInto(&w) catch |err| switch (err) {
            error.BufferTooSmall => {
                self.pos = saved;
                try self.skipString();
                return null;
            },
            else => return err,
        };
        return w.written();
    }

    fn skipString(self: *JsonReader) ParseError!void {
        try self.expectByte('"');
        while (self.pos < self.src.len) {
            self.pos += neonFindJsonStringSpecial(self.src[self.pos..]);
            if (self.pos >= self.src.len) return error.InvalidJson;
            const b = self.src[self.pos];
            if (b == '"') {
                self.pos += 1;
                return;
            }
            if (b < 0x20) return error.InvalidJson;
            // b == '\\'
            self.pos += 1;
            if (self.pos >= self.src.len) return error.InvalidJson;
            const esc = self.src[self.pos];
            self.pos += 1;
            switch (esc) {
                '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => {},
                'u' => {
                    _ = try self.readHex4();
                },
                else => return error.InvalidJson,
            }
        }
        return error.InvalidJson;
    }

    fn readNumberSlice(self: *JsonReader) ParseError![]const u8 {
        self.skipWs();
        const start = self.pos;
        if (self.pos >= self.src.len) return error.InvalidJson;
        if (self.src[self.pos] == '-') self.pos += 1;
        if (self.pos >= self.src.len) return error.InvalidJson;
        if (self.src[self.pos] == '0') {
            self.pos += 1;
        } else if (self.src[self.pos] >= '1' and self.src[self.pos] <= '9') {
            while (self.pos < self.src.len and self.src[self.pos] >= '0' and self.src[self.pos] <= '9') {
                self.pos += 1;
            }
        } else {
            return error.InvalidJson;
        }
        if (self.pos < self.src.len and self.src[self.pos] == '.') {
            self.pos += 1;
            const frac_start = self.pos;
            while (self.pos < self.src.len and self.src[self.pos] >= '0' and self.src[self.pos] <= '9') {
                self.pos += 1;
            }
            if (self.pos == frac_start) return error.InvalidJson;
        }
        if (self.pos < self.src.len and (self.src[self.pos] == 'e' or self.src[self.pos] == 'E')) {
            self.pos += 1;
            if (self.pos < self.src.len and (self.src[self.pos] == '+' or self.src[self.pos] == '-')) {
                self.pos += 1;
            }
            const exp_start = self.pos;
            while (self.pos < self.src.len and self.src[self.pos] >= '0' and self.src[self.pos] <= '9') {
                self.pos += 1;
            }
            if (self.pos == exp_start) return error.InvalidJson;
        }
        return self.src[start..self.pos];
    }

    fn readBool(self: *JsonReader) ParseError!bool {
        if (self.consumeKeyword("true")) return true;
        if (self.consumeKeyword("false")) return false;
        return error.InvalidFieldValue;
    }

    fn readU32(self: *JsonReader) ParseError!u32 {
        const s = try self.readNumberSlice();
        return std.fmt.parseInt(u32, s, 10) catch return error.InvalidFieldValue;
    }

    fn readOptionalU64(self: *JsonReader) ParseError!?u64 {
        if (self.consumeKeyword("null")) return null;
        const s = try self.readNumberSlice();
        return std.fmt.parseInt(u64, s, 10) catch return error.InvalidFieldValue;
    }

    fn readF32(self: *JsonReader) ParseError!f32 {
        const s = try self.readNumberSlice();
        const v = std.fmt.parseFloat(f32, s) catch return error.InvalidFieldValue;
        if (!std.math.isFinite(v) or v < 0.0) return error.InvalidFieldValue;
        return v;
    }

    fn readSignedF32(self: *JsonReader) ParseError!f32 {
        const s = try self.readNumberSlice();
        const v = std.fmt.parseFloat(f32, s) catch return error.InvalidFieldValue;
        if (!std.math.isFinite(v)) return error.InvalidFieldValue;
        return v;
    }

    /// Skips any valid RFC 8259 JSON value and returns its exact source slice.
    fn captureValueSlice(self: *JsonReader, depth: usize) ParseError![]const u8 {
        self.skipWs();
        const start = self.pos;
        try self.skipValue(depth);
        return self.src[start..self.pos];
    }

    fn skipValue(self: *JsonReader, depth: usize) ParseError!void {
        if (depth >= MAX_JSON_DEPTH) return error.DepthExceeded;
        const ch = self.peek() orelse return error.InvalidJson;
        switch (ch) {
            '"' => try self.skipString(),
            '{' => {
                self.pos += 1;
                if (self.peek() == '}') {
                    self.pos += 1;
                    return;
                }
                while (true) {
                    try self.skipString();
                    try self.expectByte(':');
                    try self.skipValue(depth + 1);
                    const next_ch = self.peek() orelse return error.InvalidJson;
                    if (next_ch == ',') {
                        self.pos += 1;
                    } else if (next_ch == '}') {
                        self.pos += 1;
                        return;
                    } else {
                        return error.InvalidJson;
                    }
                }
            },
            '[' => {
                self.pos += 1;
                if (self.peek() == ']') {
                    self.pos += 1;
                    return;
                }
                while (true) {
                    try self.skipValue(depth + 1);
                    const next_ch = self.peek() orelse return error.InvalidJson;
                    if (next_ch == ',') {
                        self.pos += 1;
                    } else if (next_ch == ']') {
                        self.pos += 1;
                        return;
                    } else {
                        return error.InvalidJson;
                    }
                }
            },
            't' => if (!self.consumeKeyword("true")) return error.InvalidJson,
            'f' => if (!self.consumeKeyword("false")) return error.InvalidJson,
            'n' => if (!self.consumeKeyword("null")) return error.InvalidJson,
            '-', '0'...'9' => _ = try self.readNumberSlice(),
            else => return error.InvalidJson,
        }
    }
};

/// Appends either a JSON string, `null`, or an array of content blocks
/// (OpenAI `text`/`input_text`/`output_text`, Anthropic `text`, `tool_use`,
/// `tool_result` with `is_error`, and `cache_control` detection) into `dst`.
fn appendContentValueEx(reader: *JsonReader, dst: *FixedWriter, out_cache_control: ?*bool) ParseError!void {
    const ch = reader.peek() orelse return error.InvalidJson;
    switch (ch) {
        '"' => try reader.readStringInto(dst),
        'n' => {
            if (!reader.consumeKeyword("null")) return error.InvalidJson;
        },
        '[' => {
            reader.pos += 1;
            if (reader.peek() == ']') {
                reader.pos += 1;
                return;
            }
            while (true) {
                const elem_ch = reader.peek() orelse return error.InvalidJson;
                if (elem_ch == '"') {
                    try reader.readStringInto(dst);
                } else if (elem_ch == '{') {
                    reader.pos += 1;
                    var block_type_buf: [32]u8 = undefined;
                    var block_type: []const u8 = "";
                    var text_slice: ?[]const u8 = null;
                    var name_buf: [MAX_TOOL_NAME_BYTES]u8 = undefined;
                    var name_slice: []const u8 = "";
                    var id_buf: [MAX_TOOL_CALL_ID_BYTES]u8 = undefined;
                    var id_slice: []const u8 = "";
                    var input_slice: ?[]const u8 = null;
                    var content_slice: ?[]const u8 = null;
                    var is_error_flag: bool = false;

                    if (reader.peek() != '}') {
                        while (true) {
                            var key_buf: [64]u8 = undefined;
                            const maybe_key = try reader.readKeyOrSkip(&key_buf);
                            try reader.expectByte(':');
                            if (maybe_key) |key| {
                                if (std.mem.eql(u8, key, "type")) {
                                    if (reader.peek() == '"') {
                                        block_type = reader.readSmallString(&block_type_buf) catch |err| switch (err) {
                                            error.BufferTooSmall => "",
                                            else => return err,
                                        };
                                    } else {
                                        try reader.skipValue(2);
                                    }
                                } else if (std.mem.eql(u8, key, "text")) {
                                    text_slice = try reader.captureValueSlice(2);
                                } else if (std.mem.eql(u8, key, "name")) {
                                    if (reader.peek() == '"') {
                                        name_slice = try reader.readSmallString(&name_buf);
                                    } else {
                                        try reader.skipValue(2);
                                    }
                                } else if (std.mem.eql(u8, key, "id") or std.mem.eql(u8, key, "tool_use_id")) {
                                    if (reader.peek() == '"') {
                                        id_slice = reader.readSmallString(&id_buf) catch |err| switch (err) {
                                            error.BufferTooSmall => "",
                                            else => return err,
                                        };
                                    } else {
                                        try reader.skipValue(2);
                                    }
                                } else if (std.mem.eql(u8, key, "input")) {
                                    input_slice = try reader.captureValueSlice(2);
                                } else if (std.mem.eql(u8, key, "content")) {
                                    content_slice = try reader.captureValueSlice(2);
                                } else if (std.mem.eql(u8, key, "is_error")) {
                                    if (!reader.consumeKeyword("null")) {
                                        is_error_flag = try reader.readBool();
                                    }
                                } else if (std.mem.eql(u8, key, "cache_control")) {
                                    if (!reader.consumeKeyword("null")) {
                                        try reader.skipValue(2);
                                        if (out_cache_control) |cc| cc.* = true;
                                    }
                                } else {
                                    try reader.skipValue(2);
                                }
                            } else {
                                try reader.skipValue(2);
                            }
                            const sep = reader.peek() orelse return error.InvalidJson;
                            if (sep == ',') {
                                reader.pos += 1;
                            } else if (sep == '}') {
                                reader.pos += 1;
                                break;
                            } else {
                                return error.InvalidJson;
                            }
                        }
                    } else {
                        reader.pos += 1;
                    }

                    if (std.mem.eql(u8, block_type, "tool_use")) {
                        try dst.writeAll("<tool_call>\n{\"name\":\"");
                        try dst.writeJsonEscaped(name_slice);
                        try dst.writeAll("\",\"arguments\":");
                        try dst.writeAll(input_slice orelse "{}");
                        try dst.writeAll("}\n</tool_call>\n");
                    } else if (std.mem.eql(u8, block_type, "tool_result")) {
                        try dst.writeAll("<tool_response>\n");
                        if (id_slice.len > 0) {
                            try dst.writeAll("[tool_use_id=");
                            try dst.writeAll(id_slice);
                            if (is_error_flag) try dst.writeAll(" is_error=true");
                            try dst.writeAll("]\n");
                        } else if (is_error_flag) {
                            try dst.writeAll("[is_error=true]\n");
                        }
                        if (content_slice) |cs| {
                            var sub = JsonReader{ .src = cs };
                            try appendContentValueEx(&sub, dst, out_cache_control);
                        }
                        try dst.writeAll("\n</tool_response>\n");
                    } else if (text_slice) |ts| {
                        var sub = JsonReader{ .src = ts };
                        if (sub.peek() == '"') {
                            try sub.readStringInto(dst);
                        }
                    }
                } else {
                    return error.InvalidFieldValue;
                }

                const sep = reader.peek() orelse return error.InvalidJson;
                if (sep == ',') {
                    reader.pos += 1;
                } else if (sep == ']') {
                    reader.pos += 1;
                    return;
                } else {
                    return error.InvalidJson;
                }
            }
        },
        else => return error.InvalidFieldValue,
    }
}

fn appendContentValue(reader: *JsonReader, dst: *FixedWriter) ParseError!void {
    return appendContentValueEx(reader, dst, null);
}

fn appendChatMlMessageEx(dst: *FixedWriter, role: []const u8, content_json: []const u8, out_cache_control: ?*bool) ParseError!void {
    const mapped_role = if (std.mem.eql(u8, role, "developer")) "system" else role;
    try dst.writeAll("<|im_start|>");
    try dst.writeAll(mapped_role);
    try dst.writeByte('\n');
    var sub = JsonReader{ .src = content_json };
    try appendContentValueEx(&sub, dst, out_cache_control);
    try dst.writeAll("<|im_end|>\n");
}

fn appendChatMlMessage(dst: *FixedWriter, role: []const u8, content_json: []const u8) ParseError!void {
    return appendChatMlMessageEx(dst, role, content_json, null);
}

fn appendOpenAiToolCallsArray(tool_calls_json: []const u8, dst: *FixedWriter) ParseError!void {
    var r = JsonReader{ .src = tool_calls_json };
    try r.expectByte('[');
    if (r.peek() == ']') return;
    while (true) {
        try r.expectByte('{');
        var fn_name_buf: [MAX_TOOL_NAME_BYTES]u8 = undefined;
        var fn_name: []const u8 = "";
        var fn_args_slice: ?[]const u8 = null;

        if (r.peek() != '}') {
            while (true) {
                var kbuf: [64]u8 = undefined;
                const mk = try r.readKeyOrSkip(&kbuf);
                try r.expectByte(':');
                if (mk != null and std.mem.eql(u8, mk.?, "function") and r.peek() == '{') {
                    r.pos += 1;
                    if (r.peek() != '}') {
                        while (true) {
                            var fk_buf: [64]u8 = undefined;
                            const mfk = try r.readKeyOrSkip(&fk_buf);
                            try r.expectByte(':');
                            if (mfk != null and std.mem.eql(u8, mfk.?, "name")) {
                                fn_name = try r.readSmallString(&fn_name_buf);
                            } else if (mfk != null and std.mem.eql(u8, mfk.?, "arguments")) {
                                fn_args_slice = try r.captureValueSlice(4);
                            } else {
                                try r.skipValue(4);
                            }
                            const fsep = r.peek() orelse return error.InvalidJson;
                            if (fsep == ',') {
                                r.pos += 1;
                            } else if (fsep == '}') {
                                r.pos += 1;
                                break;
                            } else return error.InvalidJson;
                        }
                    } else r.pos += 1;
                } else {
                    try r.skipValue(3);
                }
                const sep = r.peek() orelse return error.InvalidJson;
                if (sep == ',') {
                    r.pos += 1;
                } else if (sep == '}') {
                    r.pos += 1;
                    break;
                } else return error.InvalidJson;
            }
        } else r.pos += 1;

        if (fn_name.len > 0) {
            try dst.writeAll("<tool_call>\n{\"name\":\"");
            try dst.writeJsonEscaped(fn_name);
            try dst.writeAll("\",\"arguments\":");
            if (fn_args_slice) |as| {
                var ar = JsonReader{ .src = as };
                if (ar.peek() == '"') {
                    // OpenAI stores `arguments` as a JSON-encoded string -> write unescaped raw JSON.
                    try ar.readStringInto(dst);
                } else {
                    try dst.writeAll(as);
                }
            } else {
                try dst.writeAll("{}");
            }
            try dst.writeAll("}\n</tool_call>\n");
        }

        const asep = r.peek() orelse return error.InvalidJson;
        if (asep == ',') {
            r.pos += 1;
        } else if (asep == ']') {
            r.pos += 1;
            return;
        } else return error.InvalidJson;
    }
}

const MessagesAppendResult = struct {
    count: usize = 0,
    last_msg_offset: usize = 0,
    last_msg_is_user: bool = false,
};

fn appendMessagesArrayEx(messages_json: []const u8, dst: *FixedWriter, out_cache_control: ?*bool) ParseError!MessagesAppendResult {
    var r = JsonReader{ .src = messages_json };
    try r.expectByte('[');
    if (r.peek() == ']') {
        r.pos += 1;
        return .{};
    }
    var result: MessagesAppendResult = .{};
    while (true) {
        try r.expectByte('{');
        var role_buf: [32]u8 = undefined;
        var role_slice: ?[]const u8 = null;
        var item_type_buf: [32]u8 = undefined;
        var item_type: []const u8 = "";
        var content_slice: ?[]const u8 = null;
        var tool_calls_slice: ?[]const u8 = null;
        var tool_call_id_buf: [MAX_TOOL_CALL_ID_BYTES]u8 = undefined;
        var tool_call_id: []const u8 = "";
        var fn_name_buf: [MAX_TOOL_NAME_BYTES]u8 = undefined;
        var fn_name: []const u8 = "";
        var fn_args_slice: ?[]const u8 = null;
        var output_slice: ?[]const u8 = null;

        if (r.peek() != '}') {
            while (true) {
                var key_buf: [64]u8 = undefined;
                const maybe_key = try r.readKeyOrSkip(&key_buf);
                try r.expectByte(':');
                if (maybe_key) |key| {
                    if (std.mem.eql(u8, key, "role")) {
                        role_slice = try r.readSmallString(&role_buf);
                    } else if (std.mem.eql(u8, key, "type")) {
                        if (r.peek() == '"') {
                            item_type = r.readSmallString(&item_type_buf) catch |err| switch (err) {
                                error.BufferTooSmall => "",
                                else => return err,
                            };
                        } else {
                            try r.skipValue(2);
                        }
                    } else if (std.mem.eql(u8, key, "content")) {
                        content_slice = try r.captureValueSlice(2);
                    } else if (std.mem.eql(u8, key, "tool_calls")) {
                        if (!r.consumeKeyword("null")) {
                            tool_calls_slice = try r.captureValueSlice(2);
                        }
                    } else if (std.mem.eql(u8, key, "tool_call_id") or std.mem.eql(u8, key, "call_id")) {
                        if (r.peek() == '"') {
                            tool_call_id = r.readSmallString(&tool_call_id_buf) catch |err| switch (err) {
                                error.BufferTooSmall => "",
                                else => return err,
                            };
                        } else {
                            try r.skipValue(2);
                        }
                    } else if (std.mem.eql(u8, key, "name")) {
                        if (r.peek() == '"') {
                            fn_name = r.readSmallString(&fn_name_buf) catch |err| switch (err) {
                                error.BufferTooSmall => "",
                                else => return err,
                            };
                        } else {
                            try r.skipValue(2);
                        }
                    } else if (std.mem.eql(u8, key, "arguments")) {
                        fn_args_slice = try r.captureValueSlice(2);
                    } else if (std.mem.eql(u8, key, "output")) {
                        output_slice = try r.captureValueSlice(2);
                    } else {
                        try r.skipValue(2);
                    }
                } else {
                    try r.skipValue(2);
                }
                const sep = r.peek() orelse return error.InvalidJson;
                if (sep == ',') {
                    r.pos += 1;
                } else if (sep == '}') {
                    r.pos += 1;
                    break;
                } else {
                    return error.InvalidJson;
                }
            }
        } else {
            r.pos += 1;
        }

        // OpenAI Responses `function_call_output` item (no `role` field!).
        if (std.mem.eql(u8, item_type, "function_call_output")) {
            result.last_msg_offset = dst.pos;
            result.last_msg_is_user = true;
            try dst.writeAll("<|im_start|>user\n<tool_response>\n");
            if (tool_call_id.len > 0) {
                try dst.writeAll("[call_id=");
                try dst.writeAll(tool_call_id);
                try dst.writeAll("]\n");
            }
            if (output_slice) |os| {
                var sub = JsonReader{ .src = os };
                try appendContentValueEx(&sub, dst, out_cache_control);
            }
            try dst.writeAll("\n</tool_response><|im_end|>\n");
            result.count += 1;
        } else if (std.mem.eql(u8, item_type, "function_call")) {
            // OpenAI Responses assistant `function_call` item in multi-turn history.
            result.last_msg_offset = dst.pos;
            result.last_msg_is_user = false;
            try dst.writeAll("<|im_start|>assistant\n<tool_call>\n{\"name\":\"");
            try dst.writeJsonEscaped(fn_name);
            try dst.writeAll("\",\"arguments\":");
            if (fn_args_slice) |as| {
                var ar = JsonReader{ .src = as };
                if (ar.peek() == '"') {
                    try ar.readStringInto(dst);
                } else {
                    try dst.writeAll(as);
                }
            } else {
                try dst.writeAll("{}");
            }
            try dst.writeAll("}\n</tool_call><|im_end|>\n");
            result.count += 1;
        } else {
            const role = role_slice orelse return error.MissingRequiredField;
            if (role.len == 0) return error.InvalidFieldValue;
            const content = content_slice orelse "\"\"";
            result.last_msg_offset = dst.pos;

            if (std.mem.eql(u8, role, "tool")) {
                // OpenAI Chat Completions `role: "tool"` history turn.
                result.last_msg_is_user = true;
                try dst.writeAll("<|im_start|>user\n<tool_response>\n");
                if (tool_call_id.len > 0) {
                    try dst.writeAll("[tool_call_id=");
                    try dst.writeAll(tool_call_id);
                    try dst.writeAll("]\n");
                }
                var sub = JsonReader{ .src = content };
                try appendContentValueEx(&sub, dst, out_cache_control);
                try dst.writeAll("\n</tool_response><|im_end|>\n");
            } else if (tool_calls_slice) |tcs| {
                // OpenAI Chat Completions assistant turn carrying `tool_calls` array.
                result.last_msg_is_user = false;
                try dst.writeAll("<|im_start|>assistant\n");
                var sub = JsonReader{ .src = content };
                try appendContentValueEx(&sub, dst, out_cache_control);
                try appendOpenAiToolCallsArray(tcs, dst);
                try dst.writeAll("<|im_end|>\n");
            } else {
                result.last_msg_is_user = std.mem.eql(u8, role, "user");
                try appendChatMlMessageEx(dst, role, content, out_cache_control);
            }
            result.count += 1;
        }

        const sep = r.peek() orelse return error.InvalidJson;
        if (sep == ',') {
            r.pos += 1;
        } else if (sep == ']') {
            r.pos += 1;
            return result;
        } else {
            return error.InvalidJson;
        }
    }
}

fn appendMessagesArray(messages_json: []const u8, dst: *FixedWriter) ParseError!MessagesAppendResult {
    return appendMessagesArrayEx(messages_json, dst, null);
}

/// Parses OpenAI Chat (`type:"function", function:{name,parameters,strict}`),
/// OpenAI Responses (`type:"function", name, parameters, strict`), or Anthropic
/// (`name, input_schema, strict`) tool definitions into `out_tools` without allocation.
fn parseToolsArray(
    tools_json: []const u8,
    out_tools: *[MAX_TOOLS_PER_REQUEST]ParsedToolSpec,
) ParseError!u8 {
    var r = JsonReader{ .src = tools_json };
    try r.expectByte('[');
    if (r.peek() == ']') {
        r.pos += 1;
        return 0;
    }
    var count: u8 = 0;
    while (true) {
        try r.expectByte('{');
        var spec = ParsedToolSpec{};
        if (r.peek() != '}') {
            while (true) {
                var kbuf: [64]u8 = undefined;
                const mk = try r.readKeyOrSkip(&kbuf);
                try r.expectByte(':');
                if (mk) |key| {
                    if (std.mem.eql(u8, key, "function") and r.peek() == '{') {
                        r.pos += 1;
                        if (r.peek() != '}') {
                            while (true) {
                                var fk_buf: [64]u8 = undefined;
                                const mfk = try r.readKeyOrSkip(&fk_buf);
                                try r.expectByte(':');
                                if (mfk) |fkey| {
                                    if (std.mem.eql(u8, fkey, "name")) {
                                        const nm = r.readSmallString(&spec.name_buf) catch |err| switch (err) {
                                            error.BufferTooSmall => return error.InvalidFieldValue,
                                            else => return err,
                                        };
                                        spec.name_len = @intCast(nm.len);
                                    } else if (std.mem.eql(u8, fkey, "parameters")) {
                                        spec.parameters_json = try r.captureValueSlice(3);
                                    } else if (std.mem.eql(u8, fkey, "strict")) {
                                        spec.strict = try r.readBool();
                                    } else {
                                        try r.skipValue(3);
                                    }
                                } else {
                                    try r.skipValue(3);
                                }
                                const fsep = r.peek() orelse return error.InvalidJson;
                                if (fsep == ',') {
                                    r.pos += 1;
                                } else if (fsep == '}') {
                                    r.pos += 1;
                                    break;
                                } else return error.InvalidJson;
                            }
                        } else r.pos += 1;
                    } else if (std.mem.eql(u8, key, "name")) {
                        const nm = r.readSmallString(&spec.name_buf) catch |err| switch (err) {
                            error.BufferTooSmall => return error.InvalidFieldValue,
                            else => return err,
                        };
                        spec.name_len = @intCast(nm.len);
                    } else if (std.mem.eql(u8, key, "parameters") or std.mem.eql(u8, key, "input_schema")) {
                        spec.parameters_json = try r.captureValueSlice(2);
                    } else if (std.mem.eql(u8, key, "strict")) {
                        spec.strict = try r.readBool();
                    } else {
                        try r.skipValue(2);
                    }
                } else {
                    try r.skipValue(2);
                }
                const sep = r.peek() orelse return error.InvalidJson;
                if (sep == ',') {
                    r.pos += 1;
                } else if (sep == '}') {
                    r.pos += 1;
                    break;
                } else return error.InvalidJson;
            }
        } else r.pos += 1;

        if (spec.name_len == 0) return error.InvalidFieldValue;
        if (count >= MAX_TOOLS_PER_REQUEST) return error.BufferTooSmall;
        out_tools[count] = spec;
        count += 1;

        const sep = r.peek() orelse return error.InvalidJson;
        if (sep == ',') {
            r.pos += 1;
        } else if (sep == ']') {
            r.pos += 1;
            return count;
        } else return error.InvalidJson;
    }
}

/// Appends Qwen3 `<tools>...</tools>` declaration block into the ChatML system prompt.
fn appendToolDeclarationsPrompt(
    tools_json: []const u8,
    tool_choice: ToolChoiceMode,
    tool_choice_name: []const u8,
    parallel_tool_calls: bool,
    dst: *FixedWriter,
) ParseError!void {
    try dst.writeAll("<|im_start|>system\n# Tools\nYou may call one or more functions to assist with the user query.\n");
    switch (tool_choice) {
        .auto => {},
        .none => try dst.writeAll("Do NOT call any tools.\n"),
        .required => try dst.writeAll("You MUST call at least one tool.\n"),
        .named => {
            try dst.writeAll("You MUST call the tool named \"");
            try dst.writeAll(tool_choice_name);
            try dst.writeAll("\".\n");
        },
    }
    if (!parallel_tool_calls) {
        try dst.writeAll("Call at most one tool per turn.\n");
    }
    try dst.writeAll("Return each tool call inside <tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>.\n<tools>\n");
    try dst.writeAll(tools_json);
    try dst.writeAll("\n</tools><|im_end|>\n");
}

fn parseStopValue(r: *JsonReader, out_cfg: *sampling_mod.StopConfig) ParseError!void {
    if (r.consumeKeyword("null")) return;
    const ch = r.peek() orelse return error.InvalidJson;
    if (ch == '"') {
        var sbuf: [sampling_mod.MAX_STOP_BYTES]u8 = undefined;
        const s = r.readSmallString(&sbuf) catch |err| switch (err) {
            error.BufferTooSmall => return error.InvalidFieldValue,
            else => return err,
        };
        out_cfg.add(s) catch return error.InvalidFieldValue;
        return;
    }
    if (ch == '[') {
        r.pos += 1;
        if (r.peek() == ']') {
            r.pos += 1;
            return;
        }
        while (true) {
            var sbuf: [sampling_mod.MAX_STOP_BYTES]u8 = undefined;
            const s = r.readSmallString(&sbuf) catch |err| switch (err) {
                error.BufferTooSmall => return error.InvalidFieldValue,
                else => return err,
            };
            out_cfg.add(s) catch return error.InvalidFieldValue;
            const sep = r.peek() orelse return error.InvalidJson;
            if (sep == ',') {
                r.pos += 1;
            } else if (sep == ']') {
                r.pos += 1;
                return;
            } else return error.InvalidJson;
        }
    }
    return error.InvalidFieldValue;
}

fn parseResponseFormatObject(
    r: *JsonReader,
    out_kind: *ResponseFormatKind,
    out_strict: *bool,
    out_schema_json: *[]const u8,
) ParseError!void {
    try r.expectByte('{');
    if (r.peek() == '}') {
        r.pos += 1;
        return;
    }
    while (true) {
        var kbuf: [64]u8 = undefined;
        const mk = try r.readKeyOrSkip(&kbuf);
        try r.expectByte(':');
        if (mk) |key| {
            if (std.mem.eql(u8, key, "type")) {
                var tbuf: [32]u8 = undefined;
                const t = try r.readSmallString(&tbuf);
                if (std.mem.eql(u8, t, "text")) {
                    out_kind.* = .text;
                } else if (std.mem.eql(u8, t, "json_object")) {
                    out_kind.* = .json_object;
                } else if (std.mem.eql(u8, t, "json_schema")) {
                    out_kind.* = .json_schema;
                } else if (std.mem.eql(u8, t, "gbnf")) {
                    out_kind.* = .gbnf;
                } else return error.InvalidFieldValue;
            } else if (std.mem.eql(u8, key, "json_schema") and r.peek() == '{') {
                r.pos += 1;
                if (r.peek() != '}') {
                    while (true) {
                        var sk_buf: [64]u8 = undefined;
                        const msk = try r.readKeyOrSkip(&sk_buf);
                        try r.expectByte(':');
                        if (msk != null and std.mem.eql(u8, msk.?, "schema")) {
                            out_schema_json.* = try r.captureValueSlice(3);
                        } else if (msk != null and std.mem.eql(u8, msk.?, "strict")) {
                            out_strict.* = try r.readBool();
                        } else {
                            try r.skipValue(3);
                        }
                        const ssep = r.peek() orelse return error.InvalidJson;
                        if (ssep == ',') {
                            r.pos += 1;
                        } else if (ssep == '}') {
                            r.pos += 1;
                            break;
                        } else return error.InvalidJson;
                    }
                } else r.pos += 1;
            } else if (std.mem.eql(u8, key, "schema")) {
                out_schema_json.* = try r.captureValueSlice(2);
            } else if (std.mem.eql(u8, key, "strict")) {
                out_strict.* = try r.readBool();
            } else if (std.mem.eql(u8, key, "grammar")) {
                out_schema_json.* = try r.captureValueSlice(2);
            } else {
                try r.skipValue(2);
            }
        } else {
            try r.skipValue(2);
        }
        const sep = r.peek() orelse return error.InvalidJson;
        if (sep == ',') {
            r.pos += 1;
        } else if (sep == '}') {
            r.pos += 1;
            return;
        } else return error.InvalidJson;
    }
}

/// Zero-allocation parser for `/v1/chat/completions`, `/v1/responses`, and
/// `/v1/messages`. Translates all three protocols into Qwen3 ChatML directly
/// inside `prompt_buf` regardless of JSON key ordering.
pub fn parseInferenceRequest(
    endpoint: Endpoint,
    body: []const u8,
    default_model: []const u8,
    prompt_buf: []u8,
    model_buf: []u8,
) ParseError!ParsedRequest {
    return parseInferenceRequestWithPriorPrompt(endpoint, body, default_model, null, prompt_buf, model_buf);
}

/// Extended zero-allocation parser supporting optional `prior_chatml` prefix
/// for stateful `/v1/responses` (`previous_response_id`) chaining (`C18`).
pub fn parseInferenceRequestWithPriorPrompt(
    endpoint: Endpoint,
    body: []const u8,
    default_model: []const u8,
    prior_chatml: ?[]const u8,
    prompt_buf: []u8,
    model_buf: []u8,
) ParseError!ParsedRequest {
    var r = JsonReader{ .src = body };
    try r.expectByte('{');

    var model_len: usize = 0;
    var max_tokens: u32 = DEFAULT_MAX_TOKENS;
    var has_max_completion_tokens = false;
    var temperature: f32 = 0.0;
    var seed: ?u64 = null;
    var stream: bool = false;
    var ignore_eos: bool = false;
    var enable_thinking: bool = true;
    var ngram_domain_buf: [MAX_NGRAM_DOMAIN_BYTES]u8 = @splat(0);
    var ngram_domain_len: u8 = 0;

    var top_p: f32 = 1.0;
    var top_k: u32 = 0;
    var min_p: f32 = 0.0;
    var presence_penalty: f32 = 0.0;
    var frequency_penalty: f32 = 0.0;
    var repetition_penalty: f32 = 1.0;
    var logit_bias_entries: [sampling_mod.MAX_LOGIT_BIAS_ENTRIES]sampling_mod.LogitBiasEntry = undefined;
    var logit_bias_len: u8 = 0;
    var logprobs: bool = false;
    var top_logprobs: u8 = 0;
    var stop_config = sampling_mod.StopConfig.init();
    var stream_include_usage: bool = false;
    var cache_control_pin: bool = false;
    var thinking_budget_tokens: ?u32 = null;
    var reasoning_effort: ?ReasoningEffort = null;
    var store_response: bool = true;

    var prev_resp_id_buf: [MAX_RESPONSE_ID_BYTES]u8 = @splat(0);
    var prev_resp_id_len: u8 = 0;

    var tools: [MAX_TOOLS_PER_REQUEST]ParsedToolSpec = undefined;
    var tool_count: u8 = 0;
    var tools_raw_slice: ?[]const u8 = null;
    var tool_choice: ToolChoiceMode = .auto;
    var tool_choice_name_buf: [MAX_TOOL_NAME_BYTES]u8 = @splat(0);
    var tool_choice_name_len: u8 = 0;
    var parallel_tool_calls: bool = true;

    var response_format: ResponseFormatKind = .text;
    var response_format_strict: bool = false;
    var response_format_schema_json: []const u8 = "";

    var system_slice: ?[]const u8 = null;
    var messages_slice: ?[]const u8 = null;
    var input_slice: ?[]const u8 = null;

    if (r.peek() != '}') {
        while (true) {
            var key_buf: [64]u8 = undefined;
            const maybe_key = try r.readKeyOrSkip(&key_buf);
            try r.expectByte(':');

            if (maybe_key == null) {
                try r.skipValue(1);
            } else if (std.mem.eql(u8, maybe_key.?, "model")) {
                if (r.consumeKeyword("null")) {
                    model_len = 0;
                } else {
                    const m = try r.readSmallString(model_buf);
                    model_len = m.len;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "messages")) {
                messages_slice = try r.captureValueSlice(1);
            } else if (std.mem.eql(u8, maybe_key.?, "input")) {
                input_slice = try r.captureValueSlice(1);
            } else if (std.mem.eql(u8, maybe_key.?, "system") or std.mem.eql(u8, maybe_key.?, "instructions")) {
                if (!r.consumeKeyword("null")) {
                    system_slice = try r.captureValueSlice(1);
                }
            } else if (std.mem.eql(u8, maybe_key.?, "max_completion_tokens")) {
                if (!r.consumeKeyword("null")) {
                    max_tokens = try r.readU32();
                    if (max_tokens == 0) return error.InvalidFieldValue;
                    has_max_completion_tokens = true;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "max_output_tokens") or std.mem.eql(u8, maybe_key.?, "max_tokens")) {
                if (!r.consumeKeyword("null")) {
                    const val = try r.readU32();
                    if (val == 0) return error.InvalidFieldValue;
                    if (!has_max_completion_tokens) {
                        max_tokens = val;
                    }
                }
            } else if (std.mem.eql(u8, maybe_key.?, "temperature")) {
                if (!r.consumeKeyword("null")) {
                    temperature = try r.readF32();
                }
            } else if (std.mem.eql(u8, maybe_key.?, "top_p")) {
                if (!r.consumeKeyword("null")) {
                    top_p = try r.readF32();
                    if (!(top_p > 0.0) or top_p > 1.0) return error.InvalidFieldValue;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "top_k")) {
                if (!r.consumeKeyword("null")) {
                    top_k = try r.readU32();
                }
            } else if (std.mem.eql(u8, maybe_key.?, "min_p")) {
                if (!r.consumeKeyword("null")) {
                    min_p = try r.readF32();
                    if (min_p > 1.0) return error.InvalidFieldValue;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "presence_penalty")) {
                if (!r.consumeKeyword("null")) {
                    presence_penalty = try r.readSignedF32();
                    if (presence_penalty < -2.0 or presence_penalty > 2.0) return error.InvalidFieldValue;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "frequency_penalty")) {
                if (!r.consumeKeyword("null")) {
                    frequency_penalty = try r.readSignedF32();
                    if (frequency_penalty < -2.0 or frequency_penalty > 2.0) return error.InvalidFieldValue;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "repetition_penalty")) {
                if (!r.consumeKeyword("null")) {
                    repetition_penalty = try r.readF32();
                    if (!(repetition_penalty > 0.0)) return error.InvalidFieldValue;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "logit_bias")) {
                if (!r.consumeKeyword("null")) {
                    try r.expectByte('{');
                    if (r.peek() != '}') {
                        while (true) {
                            var tid_buf: [24]u8 = undefined;
                            const tid_str = r.readSmallString(&tid_buf) catch |err| switch (err) {
                                error.BufferTooSmall => return error.InvalidFieldValue,
                                else => return err,
                            };
                            try r.expectByte(':');
                            const tid = std.fmt.parseInt(u32, tid_str, 10) catch return error.InvalidFieldValue;
                            const bias = try r.readSignedF32();
                            if (bias < -100.0 or bias > 100.0) return error.InvalidFieldValue;
                            if (logit_bias_len >= sampling_mod.MAX_LOGIT_BIAS_ENTRIES) return error.BufferTooSmall;
                            logit_bias_entries[logit_bias_len] = .{ .token_id = tid, .bias = bias };
                            logit_bias_len += 1;

                            const sep = r.peek() orelse return error.InvalidJson;
                            if (sep == ',') {
                                r.pos += 1;
                            } else if (sep == '}') {
                                r.pos += 1;
                                break;
                            } else return error.InvalidJson;
                        }
                    } else r.pos += 1;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "logprobs")) {
                if (!r.consumeKeyword("null")) {
                    logprobs = try r.readBool();
                }
            } else if (std.mem.eql(u8, maybe_key.?, "top_logprobs")) {
                if (!r.consumeKeyword("null")) {
                    const v = try r.readU32();
                    if (v > sampling_mod.MAX_TOP_LOGPROBS) return error.InvalidFieldValue;
                    top_logprobs = @intCast(v);
                    if (top_logprobs > 0) logprobs = true;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "stop") or std.mem.eql(u8, maybe_key.?, "stop_sequences")) {
                try parseStopValue(&r, &stop_config);
            } else if (std.mem.eql(u8, maybe_key.?, "stream_options")) {
                if (!r.consumeKeyword("null")) {
                    try r.expectByte('{');
                    if (r.peek() != '}') {
                        while (true) {
                            var sk_buf: [64]u8 = undefined;
                            const msk = try r.readKeyOrSkip(&sk_buf);
                            try r.expectByte(':');
                            if (msk != null and std.mem.eql(u8, msk.?, "include_usage")) {
                                stream_include_usage = try r.readBool();
                            } else {
                                try r.skipValue(2);
                            }
                            const sep = r.peek() orelse return error.InvalidJson;
                            if (sep == ',') {
                                r.pos += 1;
                            } else if (sep == '}') {
                                r.pos += 1;
                                break;
                            } else return error.InvalidJson;
                        }
                    } else r.pos += 1;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "tools")) {
                if (!r.consumeKeyword("null")) {
                    const t_slice = try r.captureValueSlice(1);
                    tool_count = try parseToolsArray(t_slice, &tools);
                    if (tool_count > 0) tools_raw_slice = t_slice;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "tool_choice")) {
                if (!r.consumeKeyword("null")) {
                    const tc_ch = r.peek() orelse return error.InvalidJson;
                    if (tc_ch == '"') {
                        var tcbuf: [32]u8 = undefined;
                        const mode_s = try r.readSmallString(&tcbuf);
                        if (std.mem.eql(u8, mode_s, "auto")) {
                            tool_choice = .auto;
                        } else if (std.mem.eql(u8, mode_s, "none")) {
                            tool_choice = .none;
                        } else if (std.mem.eql(u8, mode_s, "required") or std.mem.eql(u8, mode_s, "any")) {
                            tool_choice = .required;
                        } else return error.InvalidFieldValue;
                    } else if (tc_ch == '{') {
                        r.pos += 1;
                        if (r.peek() != '}') {
                            while (true) {
                                var tk_buf: [64]u8 = undefined;
                                const mtk = try r.readKeyOrSkip(&tk_buf);
                                try r.expectByte(':');
                                if (mtk) |tkey| {
                                    if (std.mem.eql(u8, tkey, "type")) {
                                        var tv_buf: [32]u8 = undefined;
                                        const tv = try r.readSmallString(&tv_buf);
                                        if (std.mem.eql(u8, tv, "auto")) {
                                            tool_choice = .auto;
                                        } else if (std.mem.eql(u8, tv, "none")) {
                                            tool_choice = .none;
                                        } else if (std.mem.eql(u8, tv, "any") or std.mem.eql(u8, tv, "required")) {
                                            tool_choice = .required;
                                        } else if (std.mem.eql(u8, tv, "tool") or std.mem.eql(u8, tv, "function")) {
                                            tool_choice = .named;
                                        } else return error.InvalidFieldValue;
                                    } else if (std.mem.eql(u8, tkey, "name")) {
                                        const nm = r.readSmallString(&tool_choice_name_buf) catch |err| switch (err) {
                                            error.BufferTooSmall => return error.InvalidFieldValue,
                                            else => return err,
                                        };
                                        tool_choice_name_len = @intCast(nm.len);
                                        tool_choice = .named;
                                    } else if (std.mem.eql(u8, tkey, "function") and r.peek() == '{') {
                                        r.pos += 1;
                                        if (r.peek() != '}') {
                                            while (true) {
                                                var fk_buf: [64]u8 = undefined;
                                                const mfk = try r.readKeyOrSkip(&fk_buf);
                                                try r.expectByte(':');
                                                if (mfk != null and std.mem.eql(u8, mfk.?, "name")) {
                                                    const nm = r.readSmallString(&tool_choice_name_buf) catch |err| switch (err) {
                                                        error.BufferTooSmall => return error.InvalidFieldValue,
                                                        else => return err,
                                                    };
                                                    tool_choice_name_len = @intCast(nm.len);
                                                    tool_choice = .named;
                                                } else {
                                                    try r.skipValue(3);
                                                }
                                                const fsep = r.peek() orelse return error.InvalidJson;
                                                if (fsep == ',') {
                                                    r.pos += 1;
                                                } else if (fsep == '}') {
                                                    r.pos += 1;
                                                    break;
                                                } else return error.InvalidJson;
                                            }
                                        } else r.pos += 1;
                                    } else if (std.mem.eql(u8, tkey, "disable_parallel_tool_use")) {
                                        parallel_tool_calls = !(try r.readBool());
                                    } else {
                                        try r.skipValue(2);
                                    }
                                } else {
                                    try r.skipValue(2);
                                }
                                const sep = r.peek() orelse return error.InvalidJson;
                                if (sep == ',') {
                                    r.pos += 1;
                                } else if (sep == '}') {
                                    r.pos += 1;
                                    break;
                                } else return error.InvalidJson;
                            }
                        } else r.pos += 1;
                    } else return error.InvalidFieldValue;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "parallel_tool_calls")) {
                if (!r.consumeKeyword("null")) {
                    parallel_tool_calls = try r.readBool();
                }
            } else if (std.mem.eql(u8, maybe_key.?, "response_format")) {
                if (!r.consumeKeyword("null")) {
                    try parseResponseFormatObject(&r, &response_format, &response_format_strict, &response_format_schema_json);
                }
            } else if (std.mem.eql(u8, maybe_key.?, "text")) {
                // OpenAI `/v1/responses` `text.format` structured output configuration.
                if (!r.consumeKeyword("null")) {
                    try r.expectByte('{');
                    if (r.peek() != '}') {
                        while (true) {
                            var tk_buf: [64]u8 = undefined;
                            const mtk = try r.readKeyOrSkip(&tk_buf);
                            try r.expectByte(':');
                            if (mtk != null and std.mem.eql(u8, mtk.?, "format")) {
                                if (!r.consumeKeyword("null")) {
                                    try parseResponseFormatObject(&r, &response_format, &response_format_strict, &response_format_schema_json);
                                }
                            } else {
                                try r.skipValue(2);
                            }
                            const sep = r.peek() orelse return error.InvalidJson;
                            if (sep == ',') {
                                r.pos += 1;
                            } else if (sep == '}') {
                                r.pos += 1;
                                break;
                            } else return error.InvalidJson;
                        }
                    } else r.pos += 1;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "reasoning")) {
                // OpenAI `/v1/responses` `reasoning.effort` (`none`|`low`|`medium`|`high`).
                if (!r.consumeKeyword("null")) {
                    try r.expectByte('{');
                    if (r.peek() != '}') {
                        while (true) {
                            var rk_buf: [64]u8 = undefined;
                            const mrk = try r.readKeyOrSkip(&rk_buf);
                            try r.expectByte(':');
                            if (mrk != null and std.mem.eql(u8, mrk.?, "effort")) {
                                var ebuf: [16]u8 = undefined;
                                const ef = try r.readSmallString(&ebuf);
                                if (std.mem.eql(u8, ef, "none")) {
                                    reasoning_effort = .none;
                                    enable_thinking = false;
                                } else if (std.mem.eql(u8, ef, "low")) {
                                    reasoning_effort = .low;
                                    enable_thinking = true;
                                } else if (std.mem.eql(u8, ef, "medium")) {
                                    reasoning_effort = .medium;
                                    enable_thinking = true;
                                } else if (std.mem.eql(u8, ef, "high")) {
                                    reasoning_effort = .high;
                                    enable_thinking = true;
                                } else return error.InvalidFieldValue;
                            } else {
                                try r.skipValue(2);
                            }
                            const sep = r.peek() orelse return error.InvalidJson;
                            if (sep == ',') {
                                r.pos += 1;
                            } else if (sep == '}') {
                                r.pos += 1;
                                break;
                            } else return error.InvalidJson;
                        }
                    } else r.pos += 1;
                }
            } else if (std.mem.eql(u8, maybe_key.?, "store")) {
                if (!r.consumeKeyword("null")) {
                    store_response = try r.readBool();
                }
            } else if (std.mem.eql(u8, maybe_key.?, "seed")) {
                seed = try r.readOptionalU64();
            } else if (std.mem.eql(u8, maybe_key.?, "stream")) {
                stream = try r.readBool();
            } else if (std.mem.eql(u8, maybe_key.?, "ignore_eos")) {
                ignore_eos = try r.readBool();
            } else if (std.mem.eql(u8, maybe_key.?, "enable_thinking")) {
                enable_thinking = try r.readBool();
            } else if (std.mem.eql(u8, maybe_key.?, "chat_template_kwargs")) {
                if (r.peek() == '{') {
                    r.pos += 1;
                    if (r.peek() != '}') {
                        while (true) {
                            var sub_key_buf: [64]u8 = undefined;
                            const maybe_sub = try r.readKeyOrSkip(&sub_key_buf);
                            try r.expectByte(':');
                            if (maybe_sub != null and std.mem.eql(u8, maybe_sub.?, "enable_thinking")) {
                                enable_thinking = try r.readBool();
                            } else {
                                try r.skipValue(2);
                            }
                            const sep = r.peek() orelse return error.InvalidJson;
                            if (sep == ',') {
                                r.pos += 1;
                            } else if (sep == '}') {
                                r.pos += 1;
                                break;
                            } else {
                                return error.InvalidJson;
                            }
                        }
                    } else {
                        r.pos += 1;
                    }
                } else {
                    try r.skipValue(1);
                }
            } else if (std.mem.eql(u8, maybe_key.?, "thinking")) {
                if (r.peek() == '{') {
                    r.pos += 1;
                    if (r.peek() != '}') {
                        while (true) {
                            var sub_key_buf: [64]u8 = undefined;
                            const maybe_sub = try r.readKeyOrSkip(&sub_key_buf);
                            try r.expectByte(':');
                            if (maybe_sub != null and std.mem.eql(u8, maybe_sub.?, "type")) {
                                var type_buf: [32]u8 = undefined;
                                const t = try r.readSmallString(&type_buf);
                                if (std.mem.eql(u8, t, "disabled")) {
                                    enable_thinking = false;
                                } else if (std.mem.eql(u8, t, "enabled")) {
                                    enable_thinking = true;
                                } else {
                                    return error.InvalidFieldValue;
                                }
                            } else if (maybe_sub != null and std.mem.eql(u8, maybe_sub.?, "budget_tokens")) {
                                if (!r.consumeKeyword("null")) {
                                    const bt = try r.readU32();
                                    if (bt == 0) return error.InvalidFieldValue;
                                    thinking_budget_tokens = bt;
                                }
                            } else {
                                try r.skipValue(2);
                            }
                            const sep = r.peek() orelse return error.InvalidJson;
                            if (sep == ',') {
                                r.pos += 1;
                            } else if (sep == '}') {
                                r.pos += 1;
                                break;
                            } else {
                                return error.InvalidJson;
                            }
                        }
                    } else {
                        r.pos += 1;
                    }
                } else {
                    try r.skipValue(1);
                }
            } else if (std.mem.eql(u8, maybe_key.?, "previous_response_id")) {
                if (r.consumeKeyword("null")) {
                    prev_resp_id_len = 0;
                } else {
                    const pid = r.readSmallString(&prev_resp_id_buf) catch |err| switch (err) {
                        error.BufferTooSmall => return error.InvalidFieldValue,
                        else => return err,
                    };
                    if (pid.len == 0) return error.InvalidFieldValue;
                    prev_resp_id_len = @intCast(pid.len);
                }
            } else if (std.mem.eql(u8, maybe_key.?, "ngram_domain")) {
                if (r.consumeKeyword("null")) {
                    ngram_domain_len = 0;
                } else {
                    const dom = r.readSmallString(&ngram_domain_buf) catch |err| switch (err) {
                        error.BufferTooSmall => return error.InvalidFieldValue,
                        else => return err,
                    };
                    if (!ngram_nvme.isValidDomainId(dom)) return error.InvalidFieldValue;
                    ngram_domain_len = @intCast(dom.len);
                }
            } else {
                try r.skipValue(1);
            }

            const sep = r.peek() orelse return error.InvalidJson;
            if (sep == ',') {
                r.pos += 1;
            } else if (sep == '}') {
                r.pos += 1;
                break;
            } else {
                return error.InvalidJson;
            }
        }
    } else {
        r.pos += 1;
    }

    // Reject trailing non-whitespace bytes after the root JSON object.
    if (r.peek() != null) return error.InvalidJson;

    if (model_len == 0) {
        const copy_len = @min(default_model.len, model_buf.len);
        @memcpy(model_buf[0..copy_len], default_model[0..copy_len]);
        model_len = copy_len;
    }

    var pw = FixedWriter.init(prompt_buf);
    var total_messages: usize = 0;
    var last_msg_offset: usize = 0;
    var last_msg_is_user: bool = false;

    // Prepend prior stateful `/v1/responses` ChatML conversation transcript when chaining.
    if (prior_chatml) |prior| {
        if (prior.len > 0) {
            try pw.writeAll(prior);
            total_messages += 2;
        }
    }

    // Emit optional top-level system/instructions turn first.
    if (system_slice) |sys_json| {
        last_msg_offset = pw.pos;
        last_msg_is_user = false;
        try appendChatMlMessageEx(&pw, "system", sys_json, &cache_control_pin);
        total_messages += 1;
    }

    // Inject tool schemas into Qwen3 system prompt when tools are supplied and not disabled (`tool_choice != none`).
    if (tools_raw_slice) |ts_raw| {
        if (tool_choice != .none) {
            last_msg_offset = pw.pos;
            last_msg_is_user = false;
            try appendToolDeclarationsPrompt(
                ts_raw,
                tool_choice,
                tool_choice_name_buf[0..tool_choice_name_len],
                parallel_tool_calls,
                &pw,
            );
            total_messages += 1;
        }
    }

    switch (endpoint) {
        .chat_completions, .messages => {
            const msgs = messages_slice orelse return error.MissingRequiredField;
            const res = try appendMessagesArrayEx(msgs, &pw, &cache_control_pin);
            if (res.count == 0) return error.MissingRequiredField;
            total_messages += res.count;
            last_msg_offset = res.last_msg_offset;
            last_msg_is_user = res.last_msg_is_user;
        },
        .responses => {
            const inp = input_slice orelse return error.MissingRequiredField;
            var ir = JsonReader{ .src = inp };
            const ch = ir.peek() orelse return error.InvalidJson;
            if (ch == '"') {
                last_msg_offset = pw.pos;
                last_msg_is_user = true;
                try appendChatMlMessageEx(&pw, "user", inp, &cache_control_pin);
                total_messages += 1;
            } else if (ch == '[') {
                const res = try appendMessagesArrayEx(inp, &pw, &cache_control_pin);
                if (res.count == 0) return error.MissingRequiredField;
                total_messages += res.count;
                last_msg_offset = res.last_msg_offset;
                last_msg_is_user = res.last_msg_is_user;
            } else {
                return error.InvalidFieldValue;
            }
        },
    }

    // Qwen3 ChatML assistant generation prompt (disabling thinking pre-fills an empty think block).
    if (enable_thinking) {
        try pw.writeAll("<|im_start|>assistant\n");
    } else {
        try pw.writeAll("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    const last_user_msg_byte_offset: ?usize = if (total_messages >= 2 and last_msg_is_user and last_msg_offset > 0)
        last_msg_offset
    else
        null;

    return .{
        .endpoint = endpoint,
        .model_len = @intCast(model_len),
        .prompt_len = pw.pos,
        .last_user_msg_byte_offset = last_user_msg_byte_offset,
        .max_tokens = max_tokens,
        .temperature = temperature,
        .seed = seed,
        .stream = stream,
        .ignore_eos = ignore_eos,
        .enable_thinking = enable_thinking,
        .ngram_domain_buf = ngram_domain_buf,
        .ngram_domain_len = ngram_domain_len,
        .top_p = top_p,
        .top_k = top_k,
        .min_p = min_p,
        .presence_penalty = presence_penalty,
        .frequency_penalty = frequency_penalty,
        .repetition_penalty = repetition_penalty,
        .logit_bias_entries = logit_bias_entries,
        .logit_bias_len = logit_bias_len,
        .logprobs = logprobs,
        .top_logprobs = top_logprobs,
        .stop_config = stop_config,
        .stream_include_usage = stream_include_usage,
        .cache_control_pin = cache_control_pin,
        .thinking_budget_tokens = thinking_budget_tokens,
        .reasoning_effort = reasoning_effort,
        .store_response = store_response,
        .previous_response_id_buf = prev_resp_id_buf,
        .previous_response_id_len = prev_resp_id_len,
        .tools = tools,
        .tool_count = tool_count,
        .tool_choice = tool_choice,
        .tool_choice_name_buf = tool_choice_name_buf,
        .tool_choice_name_len = tool_choice_name_len,
        .parallel_tool_calls = parallel_tool_calls,
        .response_format = response_format,
        .response_format_strict = response_format_strict,
        .response_format_schema_json = response_format_schema_json,
    };
}

/// Zero-allocation parser for `POST /v1/embeddings` (`C21` + `W19`).
pub fn parseEmbeddingsRequest(
    body: []const u8,
    default_model: []const u8,
    model_buf: []u8,
    text_scratch: []u8,
) ParseError!EmbedRequest {
    var r = JsonReader{ .src = body };
    try r.expectByte('{');

    var model_len: usize = 0;
    var dimensions: ?u32 = null;
    var encoding_format: EmbedEncodingFormat = .float;
    var inputs: [MAX_EMBEDDING_INPUTS][]const u8 = undefined;
    var input_count: usize = 0;
    var tw = FixedWriter.init(text_scratch);

    if (r.peek() != '}') {
        while (true) {
            var kbuf: [64]u8 = undefined;
            const mk = try r.readKeyOrSkip(&kbuf);
            try r.expectByte(':');
            if (mk) |key| {
                if (std.mem.eql(u8, key, "model")) {
                    if (!r.consumeKeyword("null")) {
                        const m = try r.readSmallString(model_buf);
                        model_len = m.len;
                    }
                } else if (std.mem.eql(u8, key, "dimensions")) {
                    if (!r.consumeKeyword("null")) {
                        const d = try r.readU32();
                        if (d == 0 or d > MAX_EMBEDDING_DIM) return error.InvalidFieldValue;
                        dimensions = d;
                    }
                } else if (std.mem.eql(u8, key, "encoding_format")) {
                    if (!r.consumeKeyword("null")) {
                        var ebuf: [16]u8 = undefined;
                        const ef = try r.readSmallString(&ebuf);
                        if (std.mem.eql(u8, ef, "float")) {
                            encoding_format = .float;
                        } else if (std.mem.eql(u8, ef, "base64")) {
                            encoding_format = .base64;
                        } else return error.InvalidFieldValue;
                    }
                } else if (std.mem.eql(u8, key, "input")) {
                    const ch = r.peek() orelse return error.InvalidJson;
                    if (ch == '"') {
                        const start = tw.pos;
                        try r.readStringInto(&tw);
                        inputs[0] = text_scratch[start..tw.pos];
                        input_count = 1;
                    } else if (ch == '[') {
                        r.pos += 1;
                        if (r.peek() == ']') return error.MissingRequiredField;
                        while (true) {
                            if (input_count >= MAX_EMBEDDING_INPUTS) return error.BufferTooSmall;
                            const start = tw.pos;
                            try r.readStringInto(&tw);
                            inputs[input_count] = text_scratch[start..tw.pos];
                            input_count += 1;
                            const asep = r.peek() orelse return error.InvalidJson;
                            if (asep == ',') {
                                r.pos += 1;
                            } else if (asep == ']') {
                                r.pos += 1;
                                break;
                            } else return error.InvalidJson;
                        }
                    } else return error.InvalidFieldValue;
                } else {
                    try r.skipValue(1);
                }
            } else {
                try r.skipValue(1);
            }
            const sep = r.peek() orelse return error.InvalidJson;
            if (sep == ',') {
                r.pos += 1;
            } else if (sep == '}') {
                r.pos += 1;
                break;
            } else return error.InvalidJson;
        }
    } else r.pos += 1;

    if (r.peek() != null) return error.InvalidJson;
    if (input_count == 0) return error.MissingRequiredField;

    if (model_len == 0) {
        const n = @min(default_model.len, model_buf.len);
        @memcpy(model_buf[0..n], default_model[0..n]);
        model_len = n;
    }
    const model_slice = model_buf[0..model_len];
    return .{
        .model = model_slice,
        .model_tag = parseModelTag(model_slice),
        .inputs = inputs,
        .input_count = input_count,
        .dimensions = dimensions,
        .encoding_format = encoding_format,
    };
}

/// Zero-allocation parser for `POST /v1/rerank` (`C21` + `W19`).
pub fn parseRerankRequest(
    body: []const u8,
    default_model: []const u8,
    model_buf: []u8,
    text_scratch: []u8,
) ParseError!RerankRequest {
    var r = JsonReader{ .src = body };
    try r.expectByte('{');

    var model_len: usize = 0;
    var query_slice: ?[]const u8 = null;
    var docs: [MAX_RERANK_DOCS][]const u8 = undefined;
    var doc_count: usize = 0;
    var top_n: ?u32 = null;
    var return_documents: bool = false;
    var tw = FixedWriter.init(text_scratch);

    if (r.peek() != '}') {
        while (true) {
            var kbuf: [64]u8 = undefined;
            const mk = try r.readKeyOrSkip(&kbuf);
            try r.expectByte(':');
            if (mk) |key| {
                if (std.mem.eql(u8, key, "model")) {
                    if (!r.consumeKeyword("null")) {
                        const m = try r.readSmallString(model_buf);
                        model_len = m.len;
                    }
                } else if (std.mem.eql(u8, key, "query")) {
                    const start = tw.pos;
                    try r.readStringInto(&tw);
                    query_slice = text_scratch[start..tw.pos];
                } else if (std.mem.eql(u8, key, "top_n")) {
                    if (!r.consumeKeyword("null")) {
                        const tn = try r.readU32();
                        if (tn == 0) return error.InvalidFieldValue;
                        top_n = tn;
                    }
                } else if (std.mem.eql(u8, key, "return_documents")) {
                    if (!r.consumeKeyword("null")) {
                        return_documents = try r.readBool();
                    }
                } else if (std.mem.eql(u8, key, "documents")) {
                    try r.expectByte('[');
                    if (r.peek() == ']') return error.MissingRequiredField;
                    while (true) {
                        if (doc_count >= MAX_RERANK_DOCS) return error.BufferTooSmall;
                        const dch = r.peek() orelse return error.InvalidJson;
                        const start = tw.pos;
                        if (dch == '"') {
                            try r.readStringInto(&tw);
                        } else if (dch == '{') {
                            r.pos += 1;
                            if (r.peek() != '}') {
                                while (true) {
                                    var dk_buf: [64]u8 = undefined;
                                    const mdk = try r.readKeyOrSkip(&dk_buf);
                                    try r.expectByte(':');
                                    if (mdk != null and std.mem.eql(u8, mdk.?, "text")) {
                                        try r.readStringInto(&tw);
                                    } else {
                                        try r.skipValue(3);
                                    }
                                    const dsep = r.peek() orelse return error.InvalidJson;
                                    if (dsep == ',') {
                                        r.pos += 1;
                                    } else if (dsep == '}') {
                                        r.pos += 1;
                                        break;
                                    } else return error.InvalidJson;
                                }
                            } else r.pos += 1;
                        } else return error.InvalidFieldValue;
                        docs[doc_count] = text_scratch[start..tw.pos];
                        doc_count += 1;
                        const asep = r.peek() orelse return error.InvalidJson;
                        if (asep == ',') {
                            r.pos += 1;
                        } else if (asep == ']') {
                            r.pos += 1;
                            break;
                        } else return error.InvalidJson;
                    }
                } else {
                    try r.skipValue(1);
                }
            } else {
                try r.skipValue(1);
            }
            const sep = r.peek() orelse return error.InvalidJson;
            if (sep == ',') {
                r.pos += 1;
            } else if (sep == '}') {
                r.pos += 1;
                break;
            } else return error.InvalidJson;
        }
    } else r.pos += 1;

    if (r.peek() != null) return error.InvalidJson;
    const query = query_slice orelse return error.MissingRequiredField;
    if (doc_count == 0) return error.MissingRequiredField;

    if (model_len == 0) {
        const n = @min(default_model.len, model_buf.len);
        @memcpy(model_buf[0..n], default_model[0..n]);
        model_len = n;
    }
    const model_slice = model_buf[0..model_len];
    return .{
        .model = model_slice,
        .model_tag = parseModelTag(model_slice),
        .query = query,
        .documents = docs,
        .document_count = doc_count,
        .top_n = top_n,
        .return_documents = return_documents,
    };
}

/// Zero-allocation JSON response formatter for `POST /v1/embeddings` (`float` & `base64` IEEE-754 LE).
pub fn formatEmbeddingsResponse(
    req: *const EmbedRequest,
    vectors: []const f32,
    res: EmbedResult,
    out_buf: []u8,
) error{BufferTooSmall}![]const u8 {
    var w = FixedWriter.init(out_buf);
    try w.writeAll("{\"object\":\"list\",\"data\":[");
    for (0..req.input_count) |idx| {
        if (idx > 0) try w.writeByte(',');
        try w.print("{{\"object\":\"embedding\",\"index\":{d},\"embedding\":", .{idx});
        const vec = vectors[idx * res.dim .. (idx + 1) * res.dim];
        switch (req.encoding_format) {
            .float => {
                try w.writeByte('[');
                for (vec, 0..) |val, j| {
                    if (j > 0) try w.writeByte(',');
                    try w.print("{d:.6}", .{val});
                }
                try w.writeByte(']');
            },
            .base64 => {
                try w.writeByte('"');
                const raw_bytes: []const u8 = std.mem.sliceAsBytes(vec);
                const encoded_len = std.base64.standard.Encoder.calcSize(raw_bytes.len);
                if (w.remaining() < encoded_len + 1) return error.BufferTooSmall;
                _ = std.base64.standard.Encoder.encode(w.buf[w.pos .. w.pos + encoded_len], raw_bytes);
                w.pos += encoded_len;
                try w.writeByte('"');
            },
        }
        try w.writeByte('}');
    }
    try w.writeAll("],\"model\":\"");
    try w.writeJsonEscaped(req.model);
    try w.print("\",\"usage\":{{\"prompt_tokens\":{d},\"total_tokens\":{d}}}}}", .{
        res.prompt_tokens,
        res.prompt_tokens,
    });
    return w.written();
}

/// Zero-allocation JSON response formatter for `POST /v1/rerank`.
pub fn formatRerankResponse(
    req: *const RerankRequest,
    items: []const RerankItem,
    prompt_tokens: u32,
    out_buf: []u8,
) error{BufferTooSmall}![]const u8 {
    const limit = if (req.top_n) |tn| @min(items.len, @as(usize, tn)) else items.len;
    var w = FixedWriter.init(out_buf);
    try w.writeAll("{\"model\":\"");
    try w.writeJsonEscaped(req.model);
    try w.writeAll("\",\"results\":[");
    for (items[0..limit], 0..) |it, i| {
        if (i > 0) try w.writeByte(',');
        try w.print("{{\"index\":{d},\"relevance_score\":{d:.6}", .{ it.index, it.relevance_score });
        if (req.return_documents and it.index < req.document_count) {
            try w.writeAll(",\"document\":{\"text\":\"");
            try w.writeJsonEscaped(req.documents[it.index]);
            try w.writeAll("\"}");
        }
        try w.writeByte('}');
    }
    try w.print("],\"usage\":{{\"total_tokens\":{d}}}}}", .{prompt_tokens});
    return w.written();
}

/// Zero-allocation parser for `POST /v1/ngram-packs/select` (`{"domain":"<id>"}`).
pub fn parseSelectNgramDomainRequest(
    body: []const u8,
    out_buf: *[MAX_NGRAM_DOMAIN_BYTES]u8,
) ParseError![]const u8 {
    var r = JsonReader{ .src = body };
    try r.expectByte('{');
    var domain_len: usize = 0;
    var saw_domain = false;

    if (r.peek() != '}') {
        while (true) {
            var key_buf: [64]u8 = undefined;
            const maybe_key = try r.readKeyOrSkip(&key_buf);
            try r.expectByte(':');
            if (maybe_key != null and std.mem.eql(u8, maybe_key.?, "domain")) {
                const dom = r.readSmallString(out_buf) catch |err| switch (err) {
                    error.BufferTooSmall => return error.InvalidFieldValue,
                    else => return err,
                };
                if (!ngram_nvme.isValidDomainId(dom)) return error.InvalidFieldValue;
                domain_len = dom.len;
                saw_domain = true;
            } else {
                try r.skipValue(1);
            }

            const sep = r.peek() orelse return error.InvalidJson;
            if (sep == ',') {
                r.pos += 1;
            } else if (sep == '}') {
                r.pos += 1;
                break;
            } else {
                return error.InvalidJson;
            }
        }
    } else {
        r.pos += 1;
    }

    if (r.peek() != null) return error.InvalidJson;
    if (!saw_domain) return error.MissingRequiredField;
    return out_buf[0..domain_len];
}

// ---------------------------------------------------------------------------
// Zero-allocation HTTP/1.1 request parser
// ---------------------------------------------------------------------------

const HttpMethod = enum(u8) {
    get,
    post,
    delete,
    other,
};

const HttpHeadersResult = struct {
    method: HttpMethod,
    path: []const u8,
    header_end: usize,
    content_length: ?usize,
    chunked: bool,
    expect_continue: bool,
    authorization_bearer: ?[]const u8 = null,
    x_api_key: ?[]const u8 = null,
    anthropic_version: ?[]const u8 = null,
};

const HttpParseError = error{
    Incomplete,
    BadRequest,
    PayloadTooLarge,
};

fn parseHttpHeaders(buf: []const u8) HttpParseError!HttpHeadersResult {
    const sep_idx = neonFindCrlfCrlf(buf) orelse return error.Incomplete;
    const header_end = sep_idx + 4;
    const header_block = buf[0..sep_idx];

    var lines = std.mem.splitSequence(u8, header_block, "\r\n");
    const req_line = lines.next() orelse return error.BadRequest;

    var parts = std.mem.splitScalar(u8, req_line, ' ');
    const method_str = parts.next() orelse return error.BadRequest;
    const target_str = parts.next() orelse return error.BadRequest;
    const version_str = parts.next() orelse return error.BadRequest;
    if (parts.next() != null) return error.BadRequest;
    if (!std.mem.startsWith(u8, version_str, "HTTP/1.")) return error.BadRequest;
    if (target_str.len == 0 or target_str[0] != '/') return error.BadRequest;

    const method: HttpMethod = if (std.mem.eql(u8, method_str, "GET"))
        .get
    else if (std.mem.eql(u8, method_str, "POST"))
        .post
    else if (std.mem.eql(u8, method_str, "DELETE"))
        .delete
    else
        .other;

    const path = if (std.mem.indexOfScalar(u8, target_str, '?')) |q| target_str[0..q] else target_str;

    var content_length: ?usize = null;
    var chunked = false;
    var expect_continue = false;
    var authorization_bearer: ?[]const u8 = null;
    var x_api_key: ?[]const u8 = null;
    var anthropic_version: ?[]const u8 = null;

    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.BadRequest;
        const name = std.mem.trim(u8, line[0..colon], " \t");
        const val = std.mem.trim(u8, line[colon + 1 ..], " \t");

        if (std.ascii.eqlIgnoreCase(name, "content-length")) {
            const parsed = std.fmt.parseInt(usize, val, 10) catch return error.BadRequest;
            if (content_length) |prev| {
                if (prev != parsed) return error.BadRequest;
            }
            content_length = parsed;
        } else if (std.ascii.eqlIgnoreCase(name, "transfer-encoding")) {
            if (asciiContainsIgnoreCase(val, "chunked")) {
                chunked = true;
            } else {
                return error.BadRequest;
            }
        } else if (std.ascii.eqlIgnoreCase(name, "expect")) {
            if (std.ascii.eqlIgnoreCase(val, "100-continue")) {
                expect_continue = true;
            }
        } else if (std.ascii.eqlIgnoreCase(name, "authorization")) {
            if (val.len >= 7 and std.ascii.eqlIgnoreCase(val[0..7], "bearer ")) {
                authorization_bearer = std.mem.trim(u8, val[7..], " \t");
            }
        } else if (std.ascii.eqlIgnoreCase(name, "x-api-key")) {
            x_api_key = val;
        } else if (std.ascii.eqlIgnoreCase(name, "anthropic-version")) {
            anthropic_version = val;
        }
    }

    // RFC 9112: reject requests carrying both Content-Length and Transfer-Encoding.
    if (chunked and content_length != null) return error.BadRequest;

    return .{
        .method = method,
        .path = path,
        .header_end = header_end,
        .content_length = content_length,
        .chunked = chunked,
        .expect_continue = expect_continue,
        .authorization_bearer = authorization_bearer,
        .x_api_key = x_api_key,
        .anthropic_version = anthropic_version,
    };
}

fn asciiContainsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

const ChunkedDecodeStatus = union(enum) {
    incomplete,
    complete: usize,
};

/// Validates and decodes an HTTP/1.1 `Transfer-Encoding: chunked` body in place
/// once the terminating `0\r\n...\r\n` chunk has fully arrived.
fn decodeChunkedInPlace(body_buf: []u8) error{BadRequest}!ChunkedDecodeStatus {
    // First verify the entire chunked payload has arrived before mutating in place.
    var scan: usize = 0;
    while (true) {
        const line_rel = neonFindCrlf(body_buf[scan..]) orelse return .incomplete;
        const size_line_full = body_buf[scan .. scan + line_rel];
        const size_line = if (std.mem.indexOfScalar(u8, size_line_full, ';')) |semi|
            std.mem.trim(u8, size_line_full[0..semi], " \t")
        else
            std.mem.trim(u8, size_line_full, " \t");
        if (size_line.len == 0) return error.BadRequest;
        const chunk_len = std.fmt.parseInt(usize, size_line, 16) catch return error.BadRequest;
        scan += line_rel + 2;
        if (chunk_len == 0) {
            if (body_buf.len - scan < 2) return .incomplete;
            if (body_buf[scan] == '\r' and body_buf[scan + 1] == '\n') {
                break;
            }
            const trailer_end = neonFindCrlfCrlf(body_buf[scan..]) orelse return .incomplete;
            _ = trailer_end;
            break;
        }
        if (body_buf.len - scan < chunk_len + 2) return .incomplete;
        if (body_buf[scan + chunk_len] != '\r' or body_buf[scan + chunk_len + 1] != '\n') {
            return error.BadRequest;
        }
        scan += chunk_len + 2;
    }

    // Complete payload verified: compact decoded bytes in place (`out_pos <= read_pos` always).
    var read_pos: usize = 0;
    var out_pos: usize = 0;
    while (true) {
        const line_rel = neonFindCrlf(body_buf[read_pos..]) orelse unreachable;
        const size_line_full = body_buf[read_pos .. read_pos + line_rel];
        const size_line = if (std.mem.indexOfScalar(u8, size_line_full, ';')) |semi|
            std.mem.trim(u8, size_line_full[0..semi], " \t")
        else
            std.mem.trim(u8, size_line_full, " \t");
        const chunk_len = std.fmt.parseInt(usize, size_line, 16) catch unreachable;
        read_pos += line_rel + 2;
        if (chunk_len == 0) break;
        std.mem.copyForwards(u8, body_buf[out_pos .. out_pos + chunk_len], body_buf[read_pos .. read_pos + chunk_len]);
        out_pos += chunk_len;
        read_pos += chunk_len + 2;
    }
    return .{ .complete = out_pos };
}

// ---------------------------------------------------------------------------
// Darwin time, socket & kqueue helpers
// ---------------------------------------------------------------------------

fn monotonicNowNs() u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.UPTIME_RAW, &ts) != 0) return 0;
    if (ts.sec < 0 or ts.nsec < 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn unixTimestampSec() u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts) != 0) return 0;
    if (ts.sec < 0) return 0;
    return @intCast(ts.sec);
}

fn nsToTimespec(timeout_ns: u64) std.c.timespec {
    const secs = timeout_ns / std.time.ns_per_s;
    const rem_ns = timeout_ns % std.time.ns_per_s;
    const max_sec: u64 = @intCast(std.math.maxInt(isize));
    return .{
        .sec = @intCast(@min(secs, max_sec)),
        .nsec = @intCast(rem_ns),
    };
}

fn sleepNs(ns: u64) void {
    const req = nsToTimespec(ns);
    _ = std.c.nanosleep(&req, null);
}

fn setCloexec(fd: std.c.fd_t) Error!void {
    if (std.c.fcntl(fd, std.c.F.SETFD, @as(c_int, std.c.FD_CLOEXEC)) != 0) {
        return error.FcntlFailed;
    }
}

fn setNonBlocking(fd: std.c.fd_t) Error!void {
    const raw = std.c.fcntl(fd, std.c.F.GETFL, @as(c_int, 0));
    if (raw < 0) return error.FcntlFailed;
    var flags: std.c.O = @bitCast(@as(u32, @intCast(raw)));
    flags.NONBLOCK = true;
    const updated: c_int = @intCast(@as(u32, @bitCast(flags)));
    if (std.c.fcntl(fd, std.c.F.SETFL, updated) != 0) {
        return error.FcntlFailed;
    }
}

fn setSockOptInt(fd: std.c.fd_t, level: i32, optname: u32, value: c_int) Error!void {
    var v = value;
    if (std.c.setsockopt(fd, level, optname, &v, @sizeOf(c_int)) != 0) {
        return error.SetSockOptFailed;
    }
}

fn registerUserWakeup(kq_fd: std.c.fd_t, ident: u64, udata: u64) Error!void {
    const change = [1]std.c.kevent64_s{.{
        .ident = ident,
        .filter = std.c.EVFILT.USER,
        .flags = std.c.EV.ADD | std.c.EV.CLEAR,
        .fflags = 0,
        .data = 0,
        .udata = udata,
        .ext = .{ 0, 0 },
    }};
    var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
    const rc = std.c.kevent64(kq_fd, &change, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null);
    if (rc < 0) return error.KeventFailed;
}

fn triggerUserWakeup(kq_fd: std.c.fd_t, ident: u64, udata: u64) void {
    const change = [1]std.c.kevent64_s{.{
        .ident = ident,
        .filter = std.c.EVFILT.USER,
        .flags = 0,
        .fflags = std.c.NOTE.TRIGGER,
        .data = 0,
        .udata = udata,
        .ext = .{ 0, 0 },
    }};
    var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
    while (true) {
        const rc = std.c.kevent64(kq_fd, &change, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null);
        if (rc < 0 and std.c.errno(rc) == .INTR) continue;
        return;
    }
}

// ---------------------------------------------------------------------------
// Preallocated ConnectionSlot & TokenStream
// ---------------------------------------------------------------------------

const SlotState = enum(u8) {
    free = 0,
    reading_request = 1,
    streaming_active = 2,
    draining_response = 3,
    waiting_worker_cancel = 4,
};

const AnthropicBlockState = enum(u8) {
    none = 0,
    thinking = 1,
    text = 2,
    tool_use = 3,
};

const StoredToolCall = struct {
    id_buf: [MAX_TOOL_CALL_ID_BYTES]u8 = @splat(0),
    id_len: u8 = 0,
    name_buf: [MAX_TOOL_NAME_BYTES]u8 = @splat(0),
    name_len: u8 = 0,
    args_buf: [1024]u8 = undefined,
    args_len: u16 = 0,

    fn idSlice(self: *const StoredToolCall) []const u8 {
        return self.id_buf[0..self.id_len];
    }

    fn nameSlice(self: *const StoredToolCall) []const u8 {
        return self.name_buf[0..self.name_len];
    }

    fn argsSlice(self: *const StoredToolCall) []const u8 {
        return self.args_buf[0..self.args_len];
    }
};

const ResponseRecordStatus = enum(u8) {
    in_progress = 0,
    completed = 1,
    cancelled = 2,
    failed = 3,

    fn name(self: ResponseRecordStatus) []const u8 {
        return switch (self) {
            .in_progress => "in_progress",
            .completed => "completed",
            .cancelled => "cancelled",
            .failed => "failed",
        };
    }
};

/// Preallocated fixed-slot stateful `/v1/responses` record (`evict_first` LRU,
/// zero steady-state heap allocation, `C18`).
const StatefulResponseSlot = struct {
    occupied: bool = false,
    lru_tick: u64 = 0,
    request_id: u64 = 0,
    created_unix: u64 = 0,
    status: ResponseRecordStatus = .completed,
    seed_used: u64 = 0,
    seed_source: EffectiveSeedSource = .greedy,
    prompt_tokens: u32 = 0,
    cached_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    reasoning_tokens: u32 = 0,
    /// Optional owning `ConnectionSlot.index` while `status == .in_progress` (for `POST /v1/responses/{id}/cancel`).
    active_conn_slot: ?u16 = null,
    active_generation: u64 = 0,
    id_buf: [MAX_RESPONSE_ID_BYTES]u8 = @splat(0),
    id_len: u8 = 0,
    model_buf: [MAX_MODEL_ID_BYTES]u8 = @splat(0),
    model_len: u16 = 0,
    previous_id_buf: [MAX_RESPONSE_ID_BYTES]u8 = @splat(0),
    previous_id_len: u8 = 0,
    /// Slices carved once at `Server.init` from `response_store_slab` (16 KiB aligned).
    chatml_buf: []u8 = &.{},
    chatml_len: usize = 0,
    reasoning_buf: []u8 = &.{},
    reasoning_len: usize = 0,
    content_buf: []u8 = &.{},
    content_len: usize = 0,
    tool_calls: [MAX_STORED_TOOL_CALLS]StoredToolCall = undefined,
    tool_call_count: u8 = 0,

    fn idSlice(self: *const StatefulResponseSlot) []const u8 {
        return self.id_buf[0..self.id_len];
    }

    fn modelSlice(self: *const StatefulResponseSlot) []const u8 {
        return self.model_buf[0..self.model_len];
    }

    fn previousIdSlice(self: *const StatefulResponseSlot) []const u8 {
        return self.previous_id_buf[0..self.previous_id_len];
    }
};

const ConnectionSlot = struct {
    index: u16,
    fd: std.c.fd_t,
    state: SlotState,
    write_armed: bool,
    sent_continue: bool,
    /// Server-thread-only flag guaranteeing `active_requests` is decremented
    /// at most once per admitted request across all completion/abort races.
    request_counted: bool,

    // Preallocated slices carved once from `Server.arena_slab`.
    rx_buf: []u8,
    rx_len: usize,
    prompt_buf: []u8,
    prompt_len: usize,
    tx_buf: []u8,
    tx_read_pos: usize,
    tx_write_pos: usize,
    nonstream_reasoning_buf: []u8,
    nonstream_reasoning_len: usize,
    nonstream_content_buf: []u8,
    nonstream_content_len: usize,

    // Lock-free worker->server token ring backed by `Server.token_ring_storage`.
    token_ring: ring.SpscRing(TokenEvent),

    // Cross-thread cancellation & lifecycle handoff flags.
    cancelled: std.atomic.Value(bool) align(ring.CACHE_LINE_BYTES),
    worker_active: std.atomic.Value(bool) align(ring.CACHE_LINE_BYTES),

    // Per-request state & embedded D11 RequestRng (32B Xoshiro256).
    generation: u64,
    request_id: u64,
    endpoint: Endpoint,
    stream: bool,
    model_buf: [MAX_MODEL_ID_BYTES]u8,
    model_len: u16,
    ngram_domain_buf: [MAX_NGRAM_DOMAIN_BYTES]u8,
    ngram_domain_len: u8,
    rng: rng_mod.RequestRng,
    seed_used: u64,
    seed_source: EffectiveSeedSource,

    // C18 extended per-request sampling, stop holdback, grammar, tool calling & stateful response tracking:
    logit_bias_entries: [sampling_mod.MAX_LOGIT_BIAS_ENTRIES]sampling_mod.LogitBiasEntry,
    logit_bias_len: u8,
    stop_config: sampling_mod.StopConfig,
    stop_holdback: sampling_mod.StopHoldback,
    has_grammar: bool,
    grammar_slot_index: u16,
    stream_include_usage: bool,
    store_response: bool,
    previous_response_id_buf: [MAX_RESPONSE_ID_BYTES]u8,
    previous_response_id_len: u8,
    stateful_resp_slot_idx: ?u16,

    // Timing, accounting, and streaming protocol state.
    admit_ns: u64,
    first_token_ns: u64,
    created_unix: u64,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    reasoning_tokens: u32,
    sent_first_chunk: bool,
    preamble_sent: bool,
    responses_seq_num: u32,
    responses_output_index: u32,
    responses_reasoning_opened: bool,
    responses_reasoning_closed: bool,
    responses_reasoning_output_idx: u32,
    responses_text_opened: bool,
    responses_text_closed: bool,
    responses_text_output_idx: u32,
    responses_active_call_output_idx: u32,
    anthropic_block: AnthropicBlockState,
    anthropic_block_idx: u32,
    think_splitter: ThinkSplitter,
    tool_parser: ToolCallParser,
    stored_tool_calls: [MAX_STORED_TOOL_CALLS]StoredToolCall,
    stored_tool_call_count: u8,

    fn modelSlice(self: *const ConnectionSlot) []const u8 {
        return self.model_buf[0..self.model_len];
    }

    fn ngramDomainSlice(self: *const ConnectionSlot) []const u8 {
        return self.ngram_domain_buf[0..self.ngram_domain_len];
    }

    fn previousResponseIdSlice(self: *const ConnectionSlot) []const u8 {
        return self.previous_response_id_buf[0..self.previous_response_id_len];
    }

    fn pendingTxLen(self: *const ConnectionSlot) usize {
        return self.tx_write_pos - self.tx_read_pos;
    }

    fn txHeadroom(self: *const ConnectionSlot) usize {
        return self.tx_buf.len - (self.tx_write_pos - self.tx_read_pos);
    }

    fn compactTx(self: *ConnectionSlot) void {
        const pending = self.tx_write_pos - self.tx_read_pos;
        if (pending == 0) {
            self.tx_read_pos = 0;
            self.tx_write_pos = 0;
        } else if (self.tx_read_pos > 0) {
            std.mem.copyForwards(u8, self.tx_buf[0..pending], self.tx_buf[self.tx_read_pos..self.tx_write_pos]);
            self.tx_read_pos = 0;
            self.tx_write_pos = pending;
        }
    }

    fn appendTx(self: *ConnectionSlot, data: []const u8) error{BufferTooSmall}!void {
        if (data.len == 0) return;
        if (self.tx_buf.len - self.tx_write_pos < data.len) {
            self.compactTx();
        }
        if (self.tx_buf.len - self.tx_write_pos < data.len) {
            return error.BufferTooSmall;
        }
        @memcpy(self.tx_buf[self.tx_write_pos .. self.tx_write_pos + data.len], data);
        self.tx_write_pos += data.len;
    }

    fn resetForConnection(self: *ConnectionSlot, fd: std.c.fd_t) void {
        self.fd = fd;
        self.state = .reading_request;
        self.write_armed = false;
        self.sent_continue = false;
        self.request_counted = false;
        self.rx_len = 0;
        self.prompt_len = 0;
        self.tx_read_pos = 0;
        self.tx_write_pos = 0;
        self.nonstream_reasoning_len = 0;
        self.nonstream_content_len = 0;
        self.ngram_domain_len = 0;
        self.logit_bias_len = 0;
        self.stop_config.reset();
        self.stop_holdback.reset();
        self.has_grammar = false;
        self.stream_include_usage = false;
        self.store_response = true;
        self.previous_response_id_len = 0;
        self.stateful_resp_slot_idx = null;
        self.cancelled.store(false, .release);
        self.admit_ns = 0;
        self.first_token_ns = 0;
        self.created_unix = 0;
        self.prompt_tokens = 0;
        self.cached_tokens = 0;
        self.completion_tokens = 0;
        self.reasoning_tokens = 0;
        self.sent_first_chunk = false;
        self.preamble_sent = false;
        self.responses_seq_num = 0;
        self.responses_output_index = 0;
        self.responses_reasoning_opened = false;
        self.responses_reasoning_closed = false;
        self.responses_reasoning_output_idx = 0;
        self.responses_text_opened = false;
        self.responses_text_closed = false;
        self.responses_text_output_idx = 0;
        self.responses_active_call_output_idx = 0;
        self.anthropic_block = .none;
        self.anthropic_block_idx = 0;
        self.think_splitter.reset(false);
        self.tool_parser.reset(0);
        self.stored_tool_call_count = 0;
        // Drain any stale token events from a previous generation.
        while (self.token_ring.tryPop() != null) {}
    }
};

/// Handle passed to `EngineHandler.generateFn` for pushing detokenized bytes
/// onto the slot's preallocated lock-free SPSC token ring and waking `kqueue`
/// via `EVFILT_USER`.
pub const TokenStream = struct {
    server: *Server,
    slot: *ConnectionSlot,
    generation: u64,
    utf8: Utf8StreamAssembler = .{},
    prompt_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    finished: bool = false,

    /// Non-blocking cancellation check (`true` if the client disconnected or
    /// server shutdown was requested).
    pub fn isCancelled(self: *const TokenStream) bool {
        return self.slot.cancelled.load(.acquire) or self.server.stop_requested.load(.acquire);
    }

    /// Reports the exact prompt token count right after prefill completes.
    pub fn setPromptTokens(self: *TokenStream, count: u32) bool {
        return self.setPromptTokensWithCached(count, 0);
    }

    /// Reports both prompt token count and `PrefixCache` cached token count (`C18`).
    pub fn setPromptTokensWithCached(self: *TokenStream, count: u32, cached_tokens: u32) bool {
        self.prompt_tokens = count;
        if (self.isCancelled()) return false;
        const ev = TokenEvent{
            .generation = self.generation,
            .kind = .prefill_done,
            .token_count = count,
            .cached_tokens = cached_tokens,
            .text_len = 0,
        };
        return self.pushEvent(ev);
    }

    /// Pushes raw detokenized bytes (e.g. from `Tokenizer.tokenBytes`) representing
    /// `token_delta` newly generated tokens. Safely buffers split multi-byte UTF-8
    /// sequences across token boundaries. Returns `false` if cancelled.
    pub fn emitBytes(self: *TokenStream, bytes: []const u8, token_delta: u32) bool {
        if (self.isCancelled()) return false;
        self.completion_tokens +|= token_delta;

        const EmitCtx = struct {
            stream: *TokenStream,
            pending_tokens: u32,
            emitted_any: bool = false,

            fn onUtf8(ctx: *@This(), valid_utf8: []const u8) bool {
                var off: usize = 0;
                while (off < valid_utf8.len) {
                    const take = utf8ChunkLen(valid_utf8[off..], MAX_CHUNK_TEXT_BYTES);
                    const tok_cnt = ctx.pending_tokens;
                    ctx.pending_tokens = 0;
                    var ev = TokenEvent{
                        .generation = ctx.stream.generation,
                        .kind = .token_text,
                        .token_count = tok_cnt,
                        .text_len = @intCast(take),
                    };
                    @memcpy(ev.text[0..take], valid_utf8[off .. off + take]);
                    if (!ctx.stream.pushEvent(ev)) return false;
                    ctx.emitted_any = true;
                    off += take;
                }
                return true;
            }
        };

        var ctx = EmitCtx{
            .stream = self,
            .pending_tokens = token_delta,
        };
        if (!self.utf8.push(bytes, &ctx, EmitCtx.onUtf8)) return false;

        // If `bytes` only buffered incomplete UTF-8 carry bytes (or was empty) while
        // advancing `token_delta`, still record the token count accurately.
        if (!ctx.emitted_any and ctx.pending_tokens > 0) {
            const ev = TokenEvent{
                .generation = self.generation,
                .kind = .token_text,
                .token_count = ctx.pending_tokens,
                .text_len = 0,
            };
            if (!self.pushEvent(ev)) return false;
        }
        return true;
    }

    /// Convenience wrapper for emitting a single token's raw bytes.
    pub fn emitTokenBytes(self: *TokenStream, bytes: []const u8) bool {
        return self.emitBytes(bytes, 1);
    }

    /// Convenience wrapper for emitting already-valid UTF-8 text.
    pub fn emitText(self: *TokenStream, text: []const u8, token_delta: u32) bool {
        return self.emitBytes(text, token_delta);
    }

    /// Flushes any trailing incomplete UTF-8 sequence and emits the terminal
    /// `.finished` event.
    pub fn finish(self: *TokenStream, reason: FinishReason, prompt_tokens: u32) void {
        if (self.finished) return;
        self.finished = true;
        if (prompt_tokens > 0) self.prompt_tokens = prompt_tokens;
        if (self.isCancelled()) return;

        const FlushCtx = struct {
            stream: *TokenStream,
            fn onUtf8(ctx: *@This(), valid_utf8: []const u8) bool {
                var ev = TokenEvent{
                    .generation = ctx.stream.generation,
                    .kind = .token_text,
                    .token_count = 0,
                    .text_len = @intCast(@min(valid_utf8.len, MAX_CHUNK_TEXT_BYTES)),
                };
                @memcpy(ev.text[0..ev.text_len], valid_utf8[0..ev.text_len]);
                return ctx.stream.pushEvent(ev);
            }
        };
        var fctx = FlushCtx{ .stream = self };
        _ = self.utf8.flush(&fctx, FlushCtx.onUtf8);

        const ev = TokenEvent{
            .generation = self.generation,
            .kind = .finished,
            .finish_reason = reason,
            .token_count = self.prompt_tokens,
            .text_len = 0,
        };
        _ = self.pushEvent(ev);
    }

    /// Signals an unrecoverable inference error for this request.
    pub fn fail(self: *TokenStream, message: []const u8) void {
        if (self.finished) return;
        self.finished = true;
        if (self.isCancelled()) return;
        const n: usize = @min(message.len, MAX_CHUNK_TEXT_BYTES);
        var ev = TokenEvent{
            .generation = self.generation,
            .kind = .failed,
            .finish_reason = .err,
            .token_count = self.prompt_tokens,
            .text_len = @intCast(n),
        };
        if (n > 0) @memcpy(ev.text[0..n], message[0..n]);
        _ = self.pushEvent(ev);
    }

    fn pushEvent(self: *TokenStream, ev: TokenEvent) bool {
        while (true) {
            if (self.isCancelled()) return false;
            if (self.slot.token_ring.tryPush(ev)) {
                self.server.wakeNetworkLoop();
                return true;
            }
            // Ring momentarily full under slow client TCP backpressure: wake kqueue
            // to drain and yield briefly until space opens or client cancels.
            self.server.wakeNetworkLoop();
            sleepNs(50_000); // 50 us
        }
    }
};

fn utf8ChunkLen(slice: []const u8, max_len: usize) usize {
    if (slice.len <= max_len) return slice.len;
    var cut = max_len;
    // Do not split in the middle of a multi-byte UTF-8 continuation sequence.
    while (cut > 0 and (slice[cut] & 0xC0) == 0x80) {
        cut -= 1;
    }
    return if (cut == 0) max_len else cut;
}

// ---------------------------------------------------------------------------
// Zero-allocation `kqueue` HTTP/1.1 + SSE Server
// ---------------------------------------------------------------------------

pub const Server = struct {
    config: ServerConfig,
    model_id_buf: [MAX_MODEL_ID_BYTES]u8,
    model_id_len: u16,
    api_key_buf: [MAX_API_KEY_BYTES]u8,
    api_key_len: u16,
    unix_socket_path_buf: [MAX_UNIX_SOCKET_PATH_BYTES + 1]u8,
    unix_socket_path_len: u16,

    kq_fd: std.c.fd_t,
    worker_kq_fd: std.c.fd_t,
    listen_fd: std.c.fd_t,
    unix_listen_fd: std.c.fd_t,
    bound_port: u16,

    // Preallocated startup-only backing allocations (freed once in `deinit`).
    arena_slab: []align(APPLE_PAGE_SIZE) u8,
    slots: []ConnectionSlot,
    grammar_programs: []grammar_mod.GrammarProgram,
    response_store_slab: []align(APPLE_PAGE_SIZE) u8,
    response_store: []StatefulResponseSlot,
    response_lru_clock: u64,
    token_ring_storage: []TokenEvent,
    job_ring_storage: []InferenceJob,
    events: []std.c.kevent64_s,
    /// Preallocated scratch for synchronous `/v1/messages/count_tokens`, `/v1/embeddings`, `/v1/rerank`.
    aux_text_scratch: []u8,
    aux_embed_vectors: []f32,
    aux_token_ids: []u32,
    aux_encode_scratch: tokenizer_mod.EncodeScratch,

    // Lock-free server->worker inference job queue (`C1`).
    job_ring: ring.SpscRing(InferenceJob),

    // Preallocated C14 iteration-level continuous batch scheduler & slot stream state.
    batch_scheduler: batch_sched.BatchScheduler,
    batch_slot_conns: [batch_sched.MAX_BATCH_SLOTS]?u16,
    batch_slot_streams: [batch_sched.MAX_BATCH_SLOTS]TokenStream,

    // Optional hot-swappable multi-pack `.ang` NVMe domain registry (`0` = none).
    ngram_registry_ptr: std.atomic.Value(usize),
    // Optional native Zig `Tokenizer` (`W5`) for `POST /v1/messages/count_tokens`.
    tokenizer_ptr: std.atomic.Value(usize),
    // Optional `C21` embedding and rerank backend callbacks.
    embedding_handler: ?EmbeddingHandler,
    rerank_handler: ?RerankHandler,

    // Shared atomic counters & wakeup coalescing flags.
    stop_requested: std.atomic.Value(bool) align(ring.CACHE_LINE_BYTES),
    wakeup_pending: std.atomic.Value(bool) align(ring.CACHE_LINE_BYTES),
    worker_wakeup_pending: std.atomic.Value(bool) align(ring.CACHE_LINE_BYTES),
    wired_mib: std.atomic.Value(u64),
    active_conns: std.atomic.Value(usize),
    active_requests: std.atomic.Value(usize),
    total_requests: std.atomic.Value(u64),
    rejected_503_count: std.atomic.Value(u64),
    cancelled_count: std.atomic.Value(u64),
    active_batch_size: std.atomic.Value(u32),
    active_prefill_slots: std.atomic.Value(u32),
    active_decode_slots: std.atomic.Value(u32),
    max_observed_batch_size: std.atomic.Value(u32),
    total_batch_steps: std.atomic.Value(u64),
    total_slot_tokens: std.atomic.Value(u64),
    prefix_hit_tokens: std.atomic.Value(u64),
    prefix_hit_requests: std.atomic.Value(u64),
    qos_decode_steps: std.atomic.Value(u64),
    qos_prefill_steps: std.atomic.Value(u64),
    narrowed_prefill_chunks: std.atomic.Value(u64),
    message_boundary_checkpoints: std.atomic.Value(u64),
    last_tok_per_s_x100: std.atomic.Value(u64),
    retired_q8_chunks: std.atomic.Value(u64),
    tb_peers: std.atomic.Value(u32),
    next_request_id: u64,

    // Optional managed threads (when started via `start` / `startNetworkThread`).
    net_thread: ?std.Thread,
    worker_thread: ?std.Thread,

    /// Preallocates all connection slots, 16 KiB-aligned request/TX/prompt slabs,
    /// grammar programs, stateful response slots, lock-free rings, TCP listen socket
    /// (`SO_REUSEADDR`, `SO_NOSIGPIPE`, `O_NONBLOCK`), optional `AF_UNIX` socket,
    /// and `kqueue` descriptors once at startup.
    pub fn init(allocator: std.mem.Allocator, config: ServerConfig) Error!*Server {
        if (config.max_conns == 0 or config.max_conns > std.math.maxInt(u16)) return error.InvalidConfig;
        if (config.max_active_requests == 0 or config.max_active_requests > config.max_conns) return error.InvalidConfig;
        if (config.max_request_bytes < 512 or config.max_prompt_bytes < 256) return error.InvalidConfig;
        if (config.tx_buf_bytes < STREAM_TX_HEADROOM_BYTES * 2) return error.InvalidConfig;
        if (config.nonstream_reasoning_bytes < 256 or config.nonstream_content_bytes < 256) return error.InvalidConfig;
        if (config.model_id.len == 0 or config.model_id.len > MAX_MODEL_ID_BYTES) return error.InvalidConfig;
        if (config.api_key) |ak| {
            if (ak.len > MAX_API_KEY_BYTES) return error.InvalidConfig;
        }
        if (config.unix_socket_path) |up| {
            if (up.len == 0 or up.len > MAX_UNIX_SOCKET_PATH_BYTES) return error.InvalidConfig;
        }
        if (!std.math.isPowerOfTwo(config.token_ring_cap) or !std.math.isPowerOfTwo(config.job_ring_cap)) {
            return error.InvalidConfig;
        }
        if (config.job_ring_cap < config.max_active_requests) return error.InvalidConfig;

        const eff_batch_size: usize = if (config.max_batch_size == 0)
            @min(config.max_active_requests, batch_sched.MAX_BATCH_SLOTS)
        else
            config.max_batch_size;
        if (eff_batch_size == 0 or eff_batch_size > batch_sched.MAX_BATCH_SLOTS or eff_batch_size > config.max_active_requests) {
            return error.InvalidConfig;
        }
        if (config.max_prefill_chunk_tokens == 0 or config.max_prompt_tokens == 0) {
            return error.InvalidConfig;
        }
        if (std.math.isNan(config.decode_share) or config.decode_share < 0.0 or config.decode_share > 1.0) {
            return error.InvalidConfig;
        }

        const self = allocator.create(Server) catch return error.OutOfMemory;
        errdefer allocator.destroy(self);

        const per_slot_bytes = blk: {
            var sum: usize = 0;
            sum = std.math.add(usize, sum, config.max_request_bytes) catch return error.InvalidConfig;
            sum = std.math.add(usize, sum, config.max_prompt_bytes) catch return error.InvalidConfig;
            sum = std.math.add(usize, sum, config.tx_buf_bytes) catch return error.InvalidConfig;
            sum = std.math.add(usize, sum, config.nonstream_reasoning_bytes) catch return error.InvalidConfig;
            sum = std.math.add(usize, sum, config.nonstream_content_bytes) catch return error.InvalidConfig;
            break :blk sum;
        };
        const raw_slab_bytes = std.math.mul(usize, per_slot_bytes, config.max_conns) catch return error.InvalidConfig;
        const aligned_slab_bytes = std.mem.alignForward(usize, raw_slab_bytes, APPLE_PAGE_SIZE);

        const arena_slab = allocator.alignedAlloc(u8, .fromByteUnits(APPLE_PAGE_SIZE), aligned_slab_bytes) catch {
            return error.OutOfMemory;
        };
        errdefer allocator.free(arena_slab);

        const slots = allocator.alloc(ConnectionSlot, config.max_conns) catch return error.OutOfMemory;
        errdefer allocator.free(slots);

        const grammar_programs = allocator.alloc(grammar_mod.GrammarProgram, config.max_conns) catch return error.OutOfMemory;
        errdefer allocator.free(grammar_programs);

        // Preallocated fixed-slot stateful `/v1/responses` table (`evict_first` LRU, 16 KiB aligned).
        const resp_slot_count = config.stateful_response_slots;
        const per_resp_bytes = DEFAULT_STATEFUL_RESPONSE_PROMPT_BYTES +
            DEFAULT_STATEFUL_RESPONSE_REASONING_BYTES +
            DEFAULT_STATEFUL_RESPONSE_CONTENT_BYTES;
        const raw_resp_bytes = std.math.mul(usize, per_resp_bytes, resp_slot_count) catch return error.InvalidConfig;
        const aligned_resp_bytes = if (raw_resp_bytes == 0)
            APPLE_PAGE_SIZE
        else
            std.mem.alignForward(usize, raw_resp_bytes, APPLE_PAGE_SIZE);
        const response_store_slab = allocator.alignedAlloc(u8, .fromByteUnits(APPLE_PAGE_SIZE), aligned_resp_bytes) catch {
            return error.OutOfMemory;
        };
        errdefer allocator.free(response_store_slab);

        const response_store = allocator.alloc(StatefulResponseSlot, resp_slot_count) catch return error.OutOfMemory;
        errdefer allocator.free(response_store);
        {
            var r_off: usize = 0;
            for (response_store) |*rs| {
                const c_buf = response_store_slab[r_off .. r_off + DEFAULT_STATEFUL_RESPONSE_PROMPT_BYTES];
                r_off += DEFAULT_STATEFUL_RESPONSE_PROMPT_BYTES;
                const r_buf = response_store_slab[r_off .. r_off + DEFAULT_STATEFUL_RESPONSE_REASONING_BYTES];
                r_off += DEFAULT_STATEFUL_RESPONSE_REASONING_BYTES;
                const o_buf = response_store_slab[r_off .. r_off + DEFAULT_STATEFUL_RESPONSE_CONTENT_BYTES];
                r_off += DEFAULT_STATEFUL_RESPONSE_CONTENT_BYTES;
                rs.* = .{
                    .chatml_buf = c_buf,
                    .reasoning_buf = r_buf,
                    .content_buf = o_buf,
                };
            }
        }

        const aux_text_scratch = allocator.alloc(u8, config.max_request_bytes) catch return error.OutOfMemory;
        errdefer allocator.free(aux_text_scratch);
        const aux_embed_vectors = allocator.alloc(f32, MAX_EMBEDDING_INPUTS * MAX_EMBEDDING_DIM) catch return error.OutOfMemory;
        errdefer allocator.free(aux_embed_vectors);
        const aux_token_ids = allocator.alloc(u32, @max(config.max_prompt_tokens, 4096)) catch return error.OutOfMemory;
        errdefer allocator.free(aux_token_ids);
        var aux_encode_scratch = tokenizer_mod.EncodeScratch.init(allocator, 4096) catch return error.OutOfMemory;
        errdefer aux_encode_scratch.deinit(allocator);

        const total_token_slots = std.math.mul(usize, config.max_conns, config.token_ring_cap) catch return error.InvalidConfig;
        const token_ring_storage = allocator.alloc(TokenEvent, total_token_slots) catch return error.OutOfMemory;
        errdefer allocator.free(token_ring_storage);

        const job_ring_storage = allocator.alloc(InferenceJob, config.job_ring_cap) catch return error.OutOfMemory;
        errdefer allocator.free(job_ring_storage);

        const event_cap = config.max_conns * 2 + 6;
        const events = allocator.alloc(std.c.kevent64_s, event_cap) catch return error.OutOfMemory;
        errdefer allocator.free(events);

        var batch_scheduler = batch_sched.BatchScheduler.init(allocator, .{
            .max_batch_size = eff_batch_size,
            .max_prefill_chunk_tokens = config.max_prefill_chunk_tokens,
            .max_prompt_tokens = config.max_prompt_tokens,
            .decode_share = config.decode_share,
            .adaptive_prefill_narrow = config.adaptive_prefill_narrow,
            .message_boundary_cache = config.message_boundary_cache,
        }) catch |err| return switch (err) {
            error.InvalidConfig => error.InvalidConfig,
            error.OutOfMemory => error.OutOfMemory,
        };
        errdefer batch_scheduler.deinit(allocator);

        const job_ring = ring.SpscRing(InferenceJob).init(job_ring_storage) catch return error.RingInitFailed;

        // Carve per-slot slices from the single 16 KiB-aligned backing slab.
        var slab_off: usize = 0;
        for (slots, 0..) |*slot, i| {
            const rx = arena_slab[slab_off .. slab_off + config.max_request_bytes];
            slab_off += config.max_request_bytes;
            const prompt = arena_slab[slab_off .. slab_off + config.max_prompt_bytes];
            slab_off += config.max_prompt_bytes;
            const tx = arena_slab[slab_off .. slab_off + config.tx_buf_bytes];
            slab_off += config.tx_buf_bytes;
            const ns_reason = arena_slab[slab_off .. slab_off + config.nonstream_reasoning_bytes];
            slab_off += config.nonstream_reasoning_bytes;
            const ns_content = arena_slab[slab_off .. slab_off + config.nonstream_content_bytes];
            slab_off += config.nonstream_content_bytes;

            const tok_start = i * config.token_ring_cap;
            const tok_ring = ring.SpscRing(TokenEvent).init(
                token_ring_storage[tok_start .. tok_start + config.token_ring_cap],
            ) catch return error.RingInitFailed;

            grammar_programs[i].reset();

            slot.* = .{
                .index = @intCast(i),
                .fd = -1,
                .state = .free,
                .write_armed = false,
                .sent_continue = false,
                .request_counted = false,
                .rx_buf = rx,
                .rx_len = 0,
                .prompt_buf = prompt,
                .prompt_len = 0,
                .tx_buf = tx,
                .tx_read_pos = 0,
                .tx_write_pos = 0,
                .nonstream_reasoning_buf = ns_reason,
                .nonstream_reasoning_len = 0,
                .nonstream_content_buf = ns_content,
                .nonstream_content_len = 0,
                .token_ring = tok_ring,
                .cancelled = .init(false),
                .worker_active = .init(false),
                .generation = 0,
                .request_id = 0,
                .endpoint = .chat_completions,
                .stream = false,
                .model_buf = undefined,
                .model_len = 0,
                .ngram_domain_buf = @splat(0),
                .ngram_domain_len = 0,
                .rng = rng_mod.RequestRng.initDeterministic(0),
                .seed_used = 0,
                .seed_source = .greedy,
                .logit_bias_entries = undefined,
                .logit_bias_len = 0,
                .stop_config = .{},
                .stop_holdback = .{},
                .has_grammar = false,
                .grammar_slot_index = @intCast(i),
                .stream_include_usage = false,
                .store_response = true,
                .previous_response_id_buf = @splat(0),
                .previous_response_id_len = 0,
                .stateful_resp_slot_idx = null,
                .admit_ns = 0,
                .first_token_ns = 0,
                .created_unix = 0,
                .prompt_tokens = 0,
                .cached_tokens = 0,
                .completion_tokens = 0,
                .reasoning_tokens = 0,
                .sent_first_chunk = false,
                .preamble_sent = false,
                .responses_seq_num = 0,
                .responses_output_index = 0,
                .responses_reasoning_opened = false,
                .responses_reasoning_closed = false,
                .responses_reasoning_output_idx = 0,
                .responses_text_opened = false,
                .responses_text_closed = false,
                .responses_text_output_idx = 0,
                .responses_active_call_output_idx = 0,
                .anthropic_block = .none,
                .anthropic_block_idx = 0,
                .think_splitter = .{},
                .tool_parser = .{},
                .stored_tool_calls = undefined,
                .stored_tool_call_count = 0,
            };
        }

        // Open network & worker kqueues and register EVFILT_USER wakeups.
        const kq_fd = std.c.kqueue();
        if (kq_fd < 0) return error.KqueueFailed;
        errdefer _ = std.c.close(kq_fd);
        try setCloexec(kq_fd);
        try registerUserWakeup(kq_fd, USER_IDENT_WAKEUP, UDATA_WAKEUP);

        const worker_kq_fd = std.c.kqueue();
        if (worker_kq_fd < 0) return error.KqueueFailed;
        errdefer _ = std.c.close(worker_kq_fd);
        try setCloexec(worker_kq_fd);
        try registerUserWakeup(worker_kq_fd, WORKER_IDENT_WAKEUP, WORKER_IDENT_WAKEUP);

        // Open, configure, bind, and listen on the non-blocking TCP socket.
        const listen_fd = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
        if (listen_fd < 0) return error.SocketFailed;
        errdefer _ = std.c.close(listen_fd);

        try setCloexec(listen_fd);
        try setNonBlocking(listen_fd);
        try setSockOptInt(listen_fd, std.c.SOL.SOCKET, std.c.SO.REUSEADDR, 1);
        try setSockOptInt(listen_fd, std.c.SOL.SOCKET, std.c.SO.NOSIGPIPE, 1);

        var addr = std.c.sockaddr.in{
            .port = std.mem.nativeToBig(u16, config.port),
            .addr = @bitCast(config.host),
        };
        if (std.c.bind(listen_fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) != 0) {
            return error.BindFailed;
        }
        if (std.c.listen(listen_fd, config.backlog) != 0) {
            return error.ListenFailed;
        }

        var bound_addr: std.c.sockaddr.in = std.mem.zeroes(std.c.sockaddr.in);
        var bound_len: std.c.socklen_t = @sizeOf(std.c.sockaddr.in);
        if (std.c.getsockname(listen_fd, @ptrCast(&bound_addr), &bound_len) != 0) {
            return error.GetSockNameFailed;
        }
        const bound_port = std.mem.bigToNative(u16, bound_addr.port);

        // Register TCP listening socket on `kq_fd`.
        const listen_ev = [1]std.c.kevent64_s{.{
            .ident = @intCast(listen_fd),
            .filter = std.c.EVFILT.READ,
            .flags = std.c.EV.ADD | std.c.EV.ENABLE,
            .fflags = 0,
            .data = 0,
            .udata = UDATA_LISTEN,
            .ext = .{ 0, 0 },
        }};
        var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
        if (std.c.kevent64(kq_fd, &listen_ev, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null) < 0) {
            return error.KeventFailed;
        }

        // Optional `AF_UNIX` local IPC socket (`--unix-socket <path>`, `C18`) on the same kqueue!
        var unix_listen_fd: std.c.fd_t = -1;
        var unix_path_buf: [MAX_UNIX_SOCKET_PATH_BYTES + 1]u8 = @splat(0);
        var unix_path_len: u16 = 0;
        errdefer {
            if (unix_listen_fd >= 0) _ = std.c.close(unix_listen_fd);
            if (unix_path_len > 0) _ = std.c.unlink(@ptrCast(&unix_path_buf));
        }

        if (config.unix_socket_path) |upath| {
            @memcpy(unix_path_buf[0..upath.len], upath);
            unix_path_buf[upath.len] = 0;
            unix_path_len = @intCast(upath.len);
            // Best-effort remove stale socket node from previous run before binding.
            _ = std.c.unlink(@ptrCast(&unix_path_buf));

            const ufd = std.c.socket(std.c.AF.UNIX, std.c.SOCK.STREAM, 0);
            if (ufd < 0) return error.SocketFailed;
            unix_listen_fd = ufd;

            try setCloexec(ufd);
            try setNonBlocking(ufd);
            try setSockOptInt(ufd, std.c.SOL.SOCKET, std.c.SO.NOSIGPIPE, 1);

            var uaddr = std.c.sockaddr.un{
                .path = @splat(0),
            };
            @memcpy(uaddr.path[0..upath.len], upath);
            if (std.c.bind(ufd, @ptrCast(&uaddr), @sizeOf(std.c.sockaddr.un)) != 0) {
                return error.BindFailed;
            }
            if (std.c.listen(ufd, config.backlog) != 0) {
                return error.ListenFailed;
            }
            const uev = [1]std.c.kevent64_s{.{
                .ident = @intCast(ufd),
                .filter = std.c.EVFILT.READ,
                .flags = std.c.EV.ADD | std.c.EV.ENABLE,
                .fflags = 0,
                .data = 0,
                .udata = UDATA_UNIX_LISTEN,
                .ext = .{ 0, 0 },
            }};
            if (std.c.kevent64(kq_fd, &uev, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null) < 0) {
                return error.KeventFailed;
            }
        }

        var api_key_buf: [MAX_API_KEY_BYTES]u8 = @splat(0);
        var api_key_len: u16 = 0;
        if (config.api_key) |ak| {
            if (ak.len > 0) {
                @memcpy(api_key_buf[0..ak.len], ak);
                api_key_len = @intCast(ak.len);
            }
        }

        self.* = .{
            .config = config,
            .model_id_buf = undefined,
            .model_id_len = @intCast(config.model_id.len),
            .api_key_buf = api_key_buf,
            .api_key_len = api_key_len,
            .unix_socket_path_buf = unix_path_buf,
            .unix_socket_path_len = unix_path_len,
            .kq_fd = kq_fd,
            .worker_kq_fd = worker_kq_fd,
            .listen_fd = listen_fd,
            .unix_listen_fd = unix_listen_fd,
            .bound_port = bound_port,
            .arena_slab = arena_slab,
            .slots = slots,
            .grammar_programs = grammar_programs,
            .response_store_slab = response_store_slab,
            .response_store = response_store,
            .response_lru_clock = 1,
            .token_ring_storage = token_ring_storage,
            .job_ring_storage = job_ring_storage,
            .events = events,
            .aux_text_scratch = aux_text_scratch,
            .aux_embed_vectors = aux_embed_vectors,
            .aux_token_ids = aux_token_ids,
            .aux_encode_scratch = aux_encode_scratch,
            .job_ring = job_ring,
            .batch_scheduler = batch_scheduler,
            .batch_slot_conns = @splat(null),
            .batch_slot_streams = undefined,
            .ngram_registry_ptr = .init(if (config.ngram_registry) |r| @intFromPtr(r) else 0),
            .tokenizer_ptr = .init(if (config.tokenizer) |t| @intFromPtr(t) else 0),
            .embedding_handler = config.embedding_handler,
            .rerank_handler = config.rerank_handler,
            .stop_requested = .init(false),
            .wakeup_pending = .init(false),
            .worker_wakeup_pending = .init(false),
            .wired_mib = .init(config.wired_mib),
            .active_conns = .init(0),
            .active_requests = .init(0),
            .total_requests = .init(0),
            .rejected_503_count = .init(0),
            .cancelled_count = .init(0),
            .active_batch_size = .init(0),
            .active_prefill_slots = .init(0),
            .active_decode_slots = .init(0),
            .max_observed_batch_size = .init(0),
            .total_batch_steps = .init(0),
            .total_slot_tokens = .init(0),
            .prefix_hit_tokens = .init(0),
            .prefix_hit_requests = .init(0),
            .qos_decode_steps = .init(0),
            .qos_prefill_steps = .init(0),
            .narrowed_prefill_chunks = .init(0),
            .message_boundary_checkpoints = .init(0),
            .last_tok_per_s_x100 = .init(0),
            .retired_q8_chunks = .init(0),
            .tb_peers = .init(0),
            .next_request_id = 1,
            .net_thread = null,
            .worker_thread = null,
        };
        @memcpy(self.model_id_buf[0..config.model_id.len], config.model_id);
        return self;
    }

    pub fn deinit(self: *Server, allocator: std.mem.Allocator) void {
        self.stop();

        for (self.slots) |*slot| {
            if (slot.fd >= 0) {
                _ = std.c.close(slot.fd);
                slot.fd = -1;
            }
            slot.state = .free;
        }
        if (self.listen_fd >= 0) {
            _ = std.c.close(self.listen_fd);
            self.listen_fd = -1;
        }
        if (self.unix_listen_fd >= 0) {
            _ = std.c.close(self.unix_listen_fd);
            self.unix_listen_fd = -1;
        }
        if (self.unix_socket_path_len > 0) {
            _ = std.c.unlink(@ptrCast(&self.unix_socket_path_buf));
            self.unix_socket_path_len = 0;
        }
        if (self.kq_fd >= 0) {
            _ = std.c.close(self.kq_fd);
            self.kq_fd = -1;
        }
        if (self.worker_kq_fd >= 0) {
            _ = std.c.close(self.worker_kq_fd);
            self.worker_kq_fd = -1;
        }

        self.batch_scheduler.deinit(allocator);
        self.aux_encode_scratch.deinit(allocator);
        allocator.free(self.aux_token_ids);
        allocator.free(self.aux_embed_vectors);
        allocator.free(self.aux_text_scratch);
        allocator.free(self.response_store);
        allocator.free(self.response_store_slab);
        allocator.free(self.grammar_programs);
        allocator.free(self.events);
        allocator.free(self.job_ring_storage);
        allocator.free(self.token_ring_storage);
        allocator.free(self.slots);
        allocator.free(self.arena_slab);
        allocator.destroy(self);
    }

    pub fn boundPort(self: *const Server) u16 {
        return self.bound_port;
    }

    pub fn unixSocketPath(self: *const Server) ?[]const u8 {
        if (self.unix_socket_path_len == 0) return null;
        return self.unix_socket_path_buf[0..self.unix_socket_path_len];
    }

    pub fn modelId(self: *const Server) []const u8 {
        return self.model_id_buf[0..self.model_id_len];
    }

    pub fn setWiredMib(self: *Server, mib: u64) void {
        self.wired_mib.store(mib, .release);
    }

    pub fn setRetiredQ8Chunks(self: *Server, chunks: u64) void {
        self.retired_q8_chunks.store(chunks, .release);
    }

    pub fn setTbPeers(self: *Server, peers: u32) void {
        self.tb_peers.store(peers, .release);
    }

    pub fn setTokenizer(self: *Server, tok: ?*const tokenizer_mod.Tokenizer) void {
        self.tokenizer_ptr.store(if (tok) |t| @intFromPtr(t) else 0, .release);
    }

    pub fn tokenizer(self: *const Server) ?*const tokenizer_mod.Tokenizer {
        const ptr_val = self.tokenizer_ptr.load(.acquire);
        if (ptr_val == 0) return null;
        return @ptrFromInt(ptr_val);
    }

    pub fn setEmbeddingHandler(self: *Server, h: ?EmbeddingHandler) void {
        self.embedding_handler = h;
    }

    pub fn setRerankHandler(self: *Server, h: ?RerankHandler) void {
        self.rerank_handler = h;
    }

    pub fn setNgramRegistry(self: *Server, registry: ?*ngram_nvme.DomainNgramRegistry) void {
        self.ngram_registry_ptr.store(if (registry) |r| @intFromPtr(r) else 0, .release);
    }

    pub fn ngramRegistry(self: *const Server) ?*ngram_nvme.DomainNgramRegistry {
        const ptr_val = self.ngram_registry_ptr.load(.acquire);
        if (ptr_val == 0) return null;
        return @ptrFromInt(ptr_val);
    }

    pub fn stats(self: *const Server) ServerStats {
        const steps = self.total_batch_steps.load(.acquire);
        const slot_tokens = self.total_slot_tokens.load(.acquire);
        const mean_bs: f64 = if (steps == 0)
            0.0
        else
            @as(f64, @floatFromInt(slot_tokens)) / @as(f64, @floatFromInt(steps));
        return .{
            .active_conns = self.active_conns.load(.acquire),
            .active_requests = self.active_requests.load(.acquire),
            .total_requests = self.total_requests.load(.acquire),
            .rejected_503_count = self.rejected_503_count.load(.acquire),
            .cancelled_count = self.cancelled_count.load(.acquire),
            .wired_mib = self.wired_mib.load(.acquire),
            .active_batch_size = self.active_batch_size.load(.acquire),
            .max_observed_batch_size = self.max_observed_batch_size.load(.acquire),
            .total_batch_steps = steps,
            .total_slot_tokens = slot_tokens,
            .mean_batch_size = mean_bs,
            .prefix_hit_tokens = self.prefix_hit_tokens.load(.acquire),
            .prefix_hit_requests = self.prefix_hit_requests.load(.acquire),
            .qos_decode_steps = self.qos_decode_steps.load(.acquire),
            .qos_prefill_steps = self.qos_prefill_steps.load(.acquire),
            .narrowed_prefill_chunks = self.narrowed_prefill_chunks.load(.acquire),
            .message_boundary_checkpoints = self.message_boundary_checkpoints.load(.acquire),
        };
    }

    /// Signals graceful shutdown to both the `kqueue` loop and worker loop,
    /// cancelling all in-flight inference jobs immediately.
    pub fn requestStop(self: *Server) void {
        self.stop_requested.store(true, .seq_cst);
        for (self.slots) |*slot| {
            slot.cancelled.store(true, .release);
        }
        self.wakeNetworkLoop();
        self.wakeWorkerLoop();
    }

    /// Spawns ONLY the `kqueue` network thread (`QOS_CLASS_USER_INITIATED`).
    /// Callers who want to run `runWorkerLoop(handler)` on their own scheduler
    /// thread (e.g. main thread owning Metal/ANE context) use this entrypoint.
    pub fn startNetworkThread(self: *Server) Error!void {
        if (self.net_thread != null) return error.AlreadyStarted;
        self.stop_requested.store(false, .release);
        const NetRunner = struct {
            fn entry(s: *Server) void {
                if (s.config.apply_qos) {
                    qos.applyRole(.server) catch |err| {
                        std.debug.assert(err == error.PermissionDenied or err == error.InvalidArgument or err == error.Unexpected);
                    };
                }
                s.serveLoop() catch |err| {
                    std.debug.assert(err == error.KeventFailed);
                };
            }
        };
        self.net_thread = std.Thread.spawn(.{}, NetRunner.entry, .{self}) catch {
            return error.ThreadSpawnFailed;
        };
    }

    /// Spawns both the `kqueue` network thread (`QOS_CLASS_USER_INITIATED`) and
    /// an inference worker thread executing single-slot `handler`.
    pub fn start(self: *Server, handler: EngineHandler) Error!void {
        if (self.net_thread != null or self.worker_thread != null) return error.AlreadyStarted;
        try self.startNetworkThread();
        errdefer {
            self.requestStop();
            if (self.net_thread) |t| {
                t.join();
                self.net_thread = null;
            }
        }

        const WorkerRunner = struct {
            fn entry(s: *Server, h: EngineHandler) void {
                s.runWorkerLoop(h);
            }
        };
        self.worker_thread = std.Thread.spawn(.{}, WorkerRunner.entry, .{ self, handler }) catch {
            return error.ThreadSpawnFailed;
        };
    }

    /// Spawns both the `kqueue` network thread (`QOS_CLASS_USER_INITIATED`) and
    /// a `C14` iteration-level continuous-batching worker thread executing up to
    /// `max_batch_size` concurrent requests simultaneously via `BatchScheduler`.
    pub fn startBatched(self: *Server, handler: BatchEngineHandler) Error!void {
        if (self.net_thread != null or self.worker_thread != null) return error.AlreadyStarted;
        try self.startNetworkThread();
        errdefer {
            self.requestStop();
            if (self.net_thread) |t| {
                t.join();
                self.net_thread = null;
            }
        }

        const BatchWorkerRunner = struct {
            fn entry(s: *Server, h: BatchEngineHandler) void {
                s.runBatchedWorkerLoop(h);
            }
        };
        self.worker_thread = std.Thread.spawn(.{}, BatchWorkerRunner.entry, .{ self, handler }) catch {
            return error.ThreadSpawnFailed;
        };
    }

    /// Requests shutdown and joins any managed worker & network threads.
    pub fn stop(self: *Server) void {
        if (self.net_thread == null and self.worker_thread == null) {
            self.requestStop();
            return;
        }
        self.requestStop();
        if (self.worker_thread) |t| {
            t.join();
            self.worker_thread = null;
        }
        if (self.net_thread) |t| {
            t.join();
            self.net_thread = null;
        }
    }

    fn publishBatchTelemetry(self: *Server) void {
        const tel = self.batch_scheduler.telemetry();
        self.active_batch_size.store(tel.active_batch_size, .release);
        self.active_prefill_slots.store(@intCast(self.batch_scheduler.prefillReadySlotCount()), .release);
        self.active_decode_slots.store(@intCast(self.batch_scheduler.decodingSlotCount()), .release);
        self.max_observed_batch_size.store(tel.max_observed_batch_size, .release);
        self.total_batch_steps.store(tel.total_batch_steps, .release);
        self.total_slot_tokens.store(tel.total_slot_tokens, .release);
        self.prefix_hit_tokens.store(tel.prefix_hit_tokens, .release);
        self.prefix_hit_requests.store(tel.prefix_hit_requests, .release);
        self.qos_decode_steps.store(tel.qos_decode_steps, .release);
        self.qos_prefill_steps.store(tel.qos_prefill_steps, .release);
        self.narrowed_prefill_chunks.store(tel.narrowed_prefill_chunks, .release);
        self.message_boundary_checkpoints.store(tel.message_boundary_checkpoints, .release);
    }

    /// Single-slot inference worker loop (backward compatible with M5 `EngineHandler`):
    /// blocks with 0% idle CPU in `worker_kq_fd` (`EVFILT_USER`), drains admitted
    /// `InferenceJob`s from the lock-free `job_ring`, and executes `handler.generateFn`
    /// with zero heap allocation.
    pub fn runWorkerLoop(self: *Server, handler: EngineHandler) void {
        var empty_changes: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
        var ev_out: [4]std.c.kevent64_s = undefined;

        while (true) {
            // Clear wakeup latch before polling `job_ring` (seq_cst barrier guarantees
            // no lost wakeup between `job_ring.tryPush` and `worker_wakeup_pending`).
            self.worker_wakeup_pending.store(false, .seq_cst);

            while (self.job_ring.tryPop()) |job| {
                const slot = &self.slots[job.slot_index];
                if (!self.stop_requested.load(.acquire) and !slot.cancelled.load(.acquire)) {
                    self.active_batch_size.store(1, .release);
                    var stream = TokenStream{
                        .server = self,
                        .slot = slot,
                        .generation = job.generation,
                    };
                    handler.generateFn(handler.ctx, &job, &stream);
                    if (!stream.finished and !stream.isCancelled()) {
                        stream.finish(.stop, stream.prompt_tokens);
                    }
                    self.active_batch_size.store(0, .release);
                }
                // Release worker ownership of the slot and wake kqueue so cancelled
                // slots recycle immediately.
                slot.worker_active.store(false, .release);
                self.wakeNetworkLoop();
            }

            if (self.stop_requested.load(.acquire)) return;

            var timeout = nsToTimespec(50 * std.time.ns_per_ms);
            const rc = std.c.kevent64(
                self.worker_kq_fd,
                &empty_changes,
                0,
                &ev_out,
                @intCast(ev_out.len),
                std.c.KEVENT.FLAG.NONE,
                &timeout,
            );
            if (rc < 0 and std.c.errno(rc) != .INTR) return;
        }
    }

    /// `C14` iteration-level continuous-batching worker loop:
    /// drains admitted `InferenceJob`s into preallocated `BatchScheduler` slots,
    /// executes fair chunked prefill (`max_prefill_chunk_tokens` + `R8` prefix skip)
    /// and batched `M = B` decode steps advancing all `B` active requests together
    /// per iteration, interleaving SSE chunks to all open client sockets via `kqueue`
    /// `EVFILT_USER` + `EVFILT_WRITE` with zero steady-state heap allocations.
    pub fn runBatchedWorkerLoop(self: *Server, handler: BatchEngineHandler) void {
        var empty_changes: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
        var ev_out: [4]std.c.kevent64_s = undefined;

        for (0..batch_sched.MAX_BATCH_SLOTS) |i| {
            self.batch_slot_conns[i] = null;
        }

        const BatchObserverCtx = struct {
            server: *Server,
            handler: *const BatchEngineHandler,

            fn onPrefillDone(ctx_ptr: *anyopaque, slot_id: u16, _: u64, prompt_tokens: u32, prefix_hit_tokens: u32) bool {
                const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
                return ctx.server.batch_slot_streams[slot_id].setPromptTokensWithCached(prompt_tokens, prefix_hit_tokens);
            }

            fn onToken(ctx_ptr: *anyopaque, slot_id: u16, _: u64, token_id: u32, _: u32) bool {
                const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
                var tok_buf: [MAX_CHUNK_TEXT_BYTES]u8 = undefined;
                const bytes = ctx.handler.detokenizeFn(ctx.handler.ctx, token_id, &tok_buf);
                return ctx.server.batch_slot_streams[slot_id].emitTokenBytes(bytes);
            }

            fn onFinish(
                ctx_ptr: *anyopaque,
                slot_id: u16,
                _: u64,
                reason: batch_sched.FinishReason,
                prompt_tokens: u32,
                _: u32,
            ) void {
                const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
                const stream = &ctx.server.batch_slot_streams[slot_id];
                const mapped_reason: FinishReason = switch (reason) {
                    .stop, .none => .stop,
                    .length => .length,
                    .cancelled => .cancelled,
                    .err => .err,
                };
                if (mapped_reason != .cancelled and !stream.finished and !stream.isCancelled()) {
                    stream.finish(mapped_reason, prompt_tokens);
                }
                if (ctx.server.batch_slot_conns[slot_id]) |conn_idx| {
                    ctx.server.batch_slot_conns[slot_id] = null;
                    ctx.server.slots[conn_idx].worker_active.store(false, .release);
                    ctx.server.wakeNetworkLoop();
                }
            }

            fn onError(ctx_ptr: *anyopaque, slot_id: u16, _: u64, _: batch_sched.BackendError) void {
                const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
                const stream = &ctx.server.batch_slot_streams[slot_id];
                if (!stream.finished and !stream.isCancelled()) {
                    stream.fail("batched inference step failed");
                }
                if (ctx.server.batch_slot_conns[slot_id]) |conn_idx| {
                    ctx.server.batch_slot_conns[slot_id] = null;
                    ctx.server.slots[conn_idx].worker_active.store(false, .release);
                    ctx.server.wakeNetworkLoop();
                }
            }
        };

        var obs_ctx = BatchObserverCtx{
            .server = self,
            .handler = &handler,
        };
        const observer = batch_sched.StepObserver{
            .ctx = &obs_ctx,
            .onPrefillDoneFn = BatchObserverCtx.onPrefillDone,
            .onTokenFn = BatchObserverCtx.onToken,
            .onFinishFn = BatchObserverCtx.onFinish,
            .onErrorFn = BatchObserverCtx.onError,
        };

        const max_b = self.batch_scheduler.maxBatchSize();

        while (true) {
            self.worker_wakeup_pending.store(false, .seq_cst);

            // 1. Cancel any active batch slots whose TCP client disconnected or on shutdown.
            const stopping = self.stop_requested.load(.acquire);
            for (0..max_b) |i| {
                const conn_idx = self.batch_slot_conns[i] orelse continue;
                const conn = &self.slots[conn_idx];
                if (stopping or conn.cancelled.load(.acquire)) {
                    _ = self.batch_scheduler.cancelSlot(@intCast(i), handler.backend, observer);
                    if (self.batch_slot_conns[i] != null) {
                        self.batch_slot_conns[i] = null;
                        conn.worker_active.store(false, .release);
                        self.wakeNetworkLoop();
                    }
                }
            }

            // 2. Drain newly admitted jobs into all available preallocated batch slots.
            while (self.batch_scheduler.findFreeSlot()) |free_bsid| {
                const job = self.job_ring.tryPop() orelse break;
                const conn = &self.slots[job.slot_index];
                if (self.stop_requested.load(.acquire) or conn.cancelled.load(.acquire)) {
                    conn.worker_active.store(false, .release);
                    self.wakeNetworkLoop();
                    continue;
                }

                self.batch_slot_streams[free_bsid] = .{
                    .server = self,
                    .slot = conn,
                    .generation = job.generation,
                };

                const prompt_buf = self.batch_scheduler.slotPromptBuffer(free_bsid);
                const tok_res: TokenizeResult = if (handler.tokenizeWithBoundaryFn) |f|
                    f(handler.ctx, &job, prompt_buf) catch {
                        self.batch_slot_streams[free_bsid].fail("prompt tokenization failed");
                        conn.worker_active.store(false, .release);
                        self.wakeNetworkLoop();
                        continue;
                    }
                else blk: {
                    const prompt_len = handler.tokenizeFn(handler.ctx, &job, prompt_buf) catch {
                        self.batch_slot_streams[free_bsid].fail("prompt tokenization failed");
                        conn.worker_active.store(false, .release);
                        self.wakeNetworkLoop();
                        continue;
                    };
                    break :blk .{ .prompt_len = prompt_len, .message_boundary_token = null };
                };

                const msg_boundary_tok: ?u32 = if (self.config.message_boundary_cache) blk: {
                    if (tok_res.message_boundary_token) |b| {
                        if (b > 0 and b < tok_res.prompt_len) break :blk b;
                    }
                    break :blk null;
                } else null;

                self.batch_scheduler.enqueueInPlace(free_bsid, tok_res.prompt_len, .{
                    .request_id = job.request_id,
                    .user_tag = job.slot_index,
                    .max_tokens = job.max_tokens,
                    .temperature = job.temperature,
                    .external_rng = job.rng,
                    .ignore_eos = job.ignore_eos,
                    .eos_token_ids = handler.eos_token_ids[0..handler.eos_token_count],
                    .ngram_domain = job.ngram_domain,
                    .message_boundary_token = msg_boundary_tok,
                }) catch {
                    self.batch_slot_streams[free_bsid].fail("failed to enqueue batch slot");
                    conn.worker_active.store(false, .release);
                    self.wakeNetworkLoop();
                    continue;
                };

                self.batch_slot_conns[free_bsid] = job.slot_index;
            }

            // 3. On shutdown, release any remaining queued jobs cleanly before returning.
            if (self.stop_requested.load(.acquire)) {
                while (self.job_ring.tryPop()) |job| {
                    self.slots[job.slot_index].worker_active.store(false, .release);
                    self.wakeNetworkLoop();
                }
                self.publishBatchTelemetry();
                return;
            }

            // 4. Advance all active slots together in ONE iteration-level batch step!
            if (self.batch_scheduler.activeSlotCount() > 0) {
                self.publishBatchTelemetry();
                _ = self.batch_scheduler.step(handler.backend, observer);
                self.publishBatchTelemetry();
                continue;
            }

            // 5. Idle: publish 0 active batch size and wait with 0% CPU in `worker_kq_fd`.
            self.publishBatchTelemetry();
            if (self.stop_requested.load(.acquire)) return;

            var timeout = nsToTimespec(50 * std.time.ns_per_ms);
            const rc = std.c.kevent64(
                self.worker_kq_fd,
                &empty_changes,
                0,
                &ev_out,
                @intCast(ev_out.len),
                std.c.KEVENT.FLAG.NONE,
                &timeout,
            );
            if (rc < 0 and std.c.errno(rc) != .INTR) return;
        }
    }

    /// Runs the `kqueue` event loop until `requestStop()` is called.
    pub fn serveLoop(self: *Server) Error!void {
        while (!self.stop_requested.load(.acquire)) {
            _ = try self.runOnce(50 * std.time.ns_per_ms);
        }
        // Final drain pass to reclaim any newly cancelled slots.
        self.drainAllSlots();
    }

    /// Executes one `kevent64` pump (up to `timeout_ns`; `null` blocks until
    /// an event arrives). Returns the number of kernel events processed.
    pub fn runOnce(self: *Server, timeout_ns: ?u64) Error!usize {
        // Clear coalesced wakeup flag before draining all token rings so any
        // concurrent worker push either becomes visible now or triggers EVFILT_USER.
        self.wakeup_pending.store(false, .seq_cst);
        self.drainAllSlots();

        var empty_changes: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
        var ts: std.c.timespec = undefined;
        const ts_ptr: ?*const std.c.timespec = if (timeout_ns) |ns| blk: {
            ts = nsToTimespec(ns);
            break :blk &ts;
        } else null;

        const nev = std.c.kevent64(
            self.kq_fd,
            &empty_changes,
            0,
            self.events.ptr,
            @intCast(self.events.len),
            std.c.KEVENT.FLAG.NONE,
            ts_ptr,
        );
        if (nev < 0) {
            if (std.c.errno(nev) == .INTR) return 0;
            return error.KeventFailed;
        }

        const count: usize = @intCast(nev);
        var need_ring_drain = false;
        for (self.events[0..count]) |ev| {
            if (ev.udata == UDATA_LISTEN) {
                self.acceptPendingConnectionsOn(self.listen_fd, true);
            } else if (ev.udata == UDATA_UNIX_LISTEN) {
                self.acceptPendingConnectionsOn(self.unix_listen_fd, false);
            } else if (ev.udata == UDATA_WAKEUP) {
                need_ring_drain = true;
            } else if (ev.udata < self.slots.len) {
                const slot = &self.slots[@intCast(ev.udata)];
                if (slot.state == .free or slot.fd < 0) continue;
                if (@as(u64, @intCast(slot.fd)) != ev.ident) continue;

                if (ev.filter == std.c.EVFILT.READ) {
                    self.handleReadable(slot, ev.flags);
                } else if (ev.filter == std.c.EVFILT.WRITE) {
                    self.handleWritable(slot, ev.flags);
                }
            }
        }

        if (need_ring_drain) {
            self.wakeup_pending.store(false, .seq_cst);
            self.drainAllSlots();
        }
        return count;
    }

    fn wakeNetworkLoop(self: *Server) void {
        if (!self.wakeup_pending.swap(true, .seq_cst)) {
            if (self.kq_fd >= 0) {
                triggerUserWakeup(self.kq_fd, USER_IDENT_WAKEUP, UDATA_WAKEUP);
            }
        }
    }

    fn wakeWorkerLoop(self: *Server) void {
        if (!self.worker_wakeup_pending.swap(true, .seq_cst)) {
            if (self.worker_kq_fd >= 0) {
                triggerUserWakeup(self.worker_kq_fd, WORKER_IDENT_WAKEUP, WORKER_IDENT_WAKEUP);
            }
        }
    }

    fn findFreeSlot(self: *Server) ?*ConnectionSlot {
        for (self.slots) |*slot| {
            if (slot.state == .free and !slot.worker_active.load(.acquire)) {
                return slot;
            }
        }
        return null;
    }

    fn acceptPendingConnectionsOn(self: *Server, listen_fd: std.c.fd_t, is_tcp: bool) void {
        if (listen_fd < 0) return;
        while (true) {
            const client_fd = std.c.accept(listen_fd, null, null);
            if (client_fd < 0) {
                const err = std.c.errno(client_fd);
                if (err == .INTR) continue;
                return;
            }

            // Configure non-blocking + SO_NOSIGPIPE (+ TCP_NODELAY on IPv4 TCP sockets) immediately.
            setCloexec(client_fd) catch {
                _ = std.c.close(client_fd);
                continue;
            };
            setNonBlocking(client_fd) catch {
                _ = std.c.close(client_fd);
                continue;
            };
            setSockOptInt(client_fd, std.c.SOL.SOCKET, std.c.SO.NOSIGPIPE, 1) catch {
                _ = std.c.close(client_fd);
                continue;
            };
            if (is_tcp) {
                setSockOptInt(client_fd, std.c.IPPROTO.TCP, std.c.TCP.NODELAY, 1) catch {
                    _ = std.c.close(client_fd);
                    continue;
                };
            }

            const slot = self.findFreeSlot() orelse {
                // Connection pool completely saturated: best-effort zero-alloc 503 + graceful half-close.
                _ = self.rejected_503_count.fetchAdd(1, .monotonic);
                _ = std.c.send(client_fd, STATIC_503_RESPONSE.ptr, STATIC_503_RESPONSE.len, 0);
                _ = std.c.shutdown(client_fd, std.c.SHUT.WR);
                var drain_sink: [512]u8 = undefined;
                _ = std.c.recv(client_fd, &drain_sink, drain_sink.len, 0);
                _ = std.c.close(client_fd);
                continue;
            };

            slot.resetForConnection(client_fd);
            _ = self.active_conns.fetchAdd(1, .acq_rel);

            const reg = [1]std.c.kevent64_s{.{
                .ident = @intCast(client_fd),
                .filter = std.c.EVFILT.READ,
                .flags = std.c.EV.ADD | std.c.EV.ENABLE,
                .fflags = 0,
                .data = 0,
                .udata = slot.index,
                .ext = .{ 0, 0 },
            }};
            var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
            const rc = std.c.kevent64(self.kq_fd, &reg, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null);
            if (rc < 0) {
                self.closeSlotSocketAndRelease(slot, false);
                continue;
            }
        }
    }

    fn setWriteInterest(self: *Server, slot: *ConnectionSlot, enable: bool) void {
        if (slot.fd < 0 or slot.write_armed == enable) return;
        const flags: u16 = if (enable)
            (std.c.EV.ADD | std.c.EV.ENABLE)
        else
            std.c.EV.DELETE;
        const change = [1]std.c.kevent64_s{.{
            .ident = @intCast(slot.fd),
            .filter = std.c.EVFILT.WRITE,
            .flags = flags,
            .fflags = 0,
            .data = 0,
            .udata = slot.index,
            .ext = .{ 0, 0 },
        }};
        var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
        _ = std.c.kevent64(self.kq_fd, &change, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null);
        slot.write_armed = enable;
    }

    fn releaseRequestCount(self: *Server, slot: *ConnectionSlot) void {
        if (slot.request_counted) {
            slot.request_counted = false;
            _ = self.active_requests.fetchSub(1, .acq_rel);
        }
    }

    fn closeSlotSocketAndRelease(self: *Server, slot: *ConnectionSlot, was_client_abort: bool) void {
        if (slot.fd >= 0) {
            _ = std.c.close(slot.fd);
            slot.fd = -1;
            slot.write_armed = false;
            _ = self.active_conns.fetchSub(1, .acq_rel);
        }

        const had_active_job = (slot.state == .streaming_active or slot.state == .waiting_worker_cancel);
        if (was_client_abort and had_active_job) {
            if (!slot.cancelled.swap(true, .acq_rel)) {
                _ = self.cancelled_count.fetchAdd(1, .monotonic);
            }
        }

        if (slot.worker_active.load(.acquire)) {
            slot.cancelled.store(true, .release);
            slot.state = .waiting_worker_cancel;
            self.wakeWorkerLoop();
            return;
        }

        self.releaseRequestCount(slot);
        while (slot.token_ring.tryPop() != null) {}
        slot.state = .free;
    }

    fn handleReadable(self: *Server, slot: *ConnectionSlot, ev_flags: u16) void {
        switch (slot.state) {
            .reading_request => {
                var eof_seen = (ev_flags & std.c.EV.EOF) != 0;
                while (slot.rx_len < slot.rx_buf.len) {
                    const avail = slot.rx_buf[slot.rx_len..];
                    const n = std.c.recv(slot.fd, avail.ptr, avail.len, 0);
                    if (n > 0) {
                        slot.rx_len += @intCast(n);
                    } else if (n == 0) {
                        eof_seen = true;
                        break;
                    } else {
                        const err = std.c.errno(n);
                        if (err == .INTR) continue;
                        if (err == .AGAIN) break;
                        self.closeSlotSocketAndRelease(slot, true);
                        return;
                    }
                }

                self.tryDispatchHttpRequest(slot, eof_seen);
            },
            .streaming_active => {
                // Detect mid-stream client disconnect immediately via EV_EOF or recv == 0.
                var discard: [64]u8 = undefined;
                const n = std.c.recv(slot.fd, &discard, discard.len, 0);
                if (n == 0 or (ev_flags & (std.c.EV.EOF | std.c.EV.ERROR)) != 0) {
                    self.closeSlotSocketAndRelease(slot, true);
                    return;
                }
                if (n < 0) {
                    const err = std.c.errno(n);
                    if (err != .AGAIN and err != .INTR) {
                        self.closeSlotSocketAndRelease(slot, true);
                    }
                }
            },
            .draining_response => {
                if ((ev_flags & std.c.EV.ERROR) != 0) {
                    self.closeSlotSocketAndRelease(slot, true);
                }
            },
            .free, .waiting_worker_cancel => {},
        }
    }

    fn handleWritable(self: *Server, slot: *ConnectionSlot, ev_flags: u16) void {
        if ((ev_flags & std.c.EV.ERROR) != 0) {
            self.closeSlotSocketAndRelease(slot, true);
            return;
        }
        if (!self.flushTx(slot)) return;
        if (slot.state == .streaming_active) {
            self.drainSlotTokens(slot);
        } else if (slot.state == .draining_response and slot.pendingTxLen() == 0) {
            self.closeSlotSocketAndRelease(slot, false);
        }
    }

    /// Flushes queued `tx_buf` bytes to `slot.fd` using non-blocking `send`.
    /// Returns `false` if the socket broke and the slot was closed/cancelled.
    fn flushTx(self: *Server, slot: *ConnectionSlot) bool {
        while (slot.pendingTxLen() > 0) {
            const slice = slot.tx_buf[slot.tx_read_pos..slot.tx_write_pos];
            const n = std.c.send(slot.fd, slice.ptr, slice.len, 0);
            if (n > 0) {
                slot.tx_read_pos += @intCast(n);
                if (slot.tx_read_pos == slot.tx_write_pos) {
                    slot.tx_read_pos = 0;
                    slot.tx_write_pos = 0;
                }
            } else if (n < 0) {
                const err = std.c.errno(n);
                if (err == .INTR) continue;
                if (err == .AGAIN) {
                    self.setWriteInterest(slot, true);
                    return true;
                }
                // EPIPE / ECONNRESET / etc. — abrupt client disconnect.
                self.closeSlotSocketAndRelease(slot, true);
                return false;
            } else {
                break;
            }
        }
        self.setWriteInterest(slot, false);
        return true;
    }

    fn tryDispatchHttpRequest(self: *Server, slot: *ConnectionSlot, eof_seen: bool) void {
        const hdr = parseHttpHeaders(slot.rx_buf[0..slot.rx_len]) catch |err| switch (err) {
            error.Incomplete => {
                if (slot.rx_len >= slot.rx_buf.len) {
                    self.sendErrorResponse(slot, 413, "Payload Too Large", null, "request headers exceed preallocated slot buffer", "invalid_request_error", "payload_too_large");
                } else if (eof_seen) {
                    self.closeSlotSocketAndRelease(slot, true);
                }
                return;
            },
            error.BadRequest => {
                self.sendErrorResponse(slot, 400, "Bad Request", null, "malformed HTTP/1.1 request", "invalid_request_error", "bad_request");
                return;
            },
            error.PayloadTooLarge => {
                self.sendErrorResponse(slot, 413, "Payload Too Large", null, "request exceeds preallocated slot buffer", "invalid_request_error", "payload_too_large");
                return;
            },
        };

        // Support curl's `Expect: 100-continue` immediately so large POST bodies never stall.
        if (hdr.expect_continue and !slot.sent_continue) {
            slot.sent_continue = true;
            _ = std.c.send(slot.fd, STATIC_100_CONTINUE.ptr, STATIC_100_CONTINUE.len, 0);
        }

        // Wait until the full request body (if any) has arrived.
        var body: []const u8 = "";
        if (hdr.chunked) {
            const decoded = decodeChunkedInPlace(slot.rx_buf[hdr.header_end..slot.rx_len]) catch {
                self.sendErrorResponse(slot, 400, "Bad Request", null, "malformed chunked request body", "invalid_request_error", "bad_request");
                return;
            };
            switch (decoded) {
                .incomplete => {
                    if (slot.rx_len >= slot.rx_buf.len) {
                        self.sendErrorResponse(slot, 413, "Payload Too Large", null, "chunked body exceeds preallocated slot buffer", "invalid_request_error", "payload_too_large");
                    } else if (eof_seen) {
                        self.sendErrorResponse(slot, 400, "Bad Request", null, "truncated chunked request body", "invalid_request_error", "bad_request");
                    }
                    return;
                },
                .complete => |n| {
                    body = slot.rx_buf[hdr.header_end .. hdr.header_end + n];
                },
            }
        } else if (hdr.content_length) |cl| {
            const total_needed = std.math.add(usize, hdr.header_end, cl) catch {
                self.sendErrorResponse(slot, 413, "Payload Too Large", null, "content-length overflow", "invalid_request_error", "payload_too_large");
                return;
            };
            if (total_needed > slot.rx_buf.len) {
                self.sendErrorResponse(slot, 413, "Payload Too Large", null, "request body exceeds preallocated slot buffer", "invalid_request_error", "payload_too_large");
                return;
            }
            if (slot.rx_len < total_needed) {
                if (eof_seen) {
                    self.sendErrorResponse(slot, 400, "Bad Request", null, "truncated HTTP request body", "invalid_request_error", "bad_request");
                }
                return;
            }
            body = slot.rx_buf[hdr.header_end..total_needed];
        }

        // Route dispatch.
        if (std.mem.eql(u8, hdr.path, "/health")) {
            if (hdr.method != .get) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "GET", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleHealth(slot);
            return;
        }

        // Constant-time `--api-key` validation (`C18`) across all `/v1/*` routes:
        // accepts either `Authorization: Bearer <key>` or `x-api-key: <key>`.
        if (self.api_key_len > 0) {
            const expected = self.api_key_buf[0..self.api_key_len];
            const auth_ok = blk: {
                if (hdr.authorization_bearer) |b| {
                    if (constantTimeEql(b, expected)) break :blk true;
                }
                if (hdr.x_api_key) |k| {
                    if (constantTimeEql(k, expected)) break :blk true;
                }
                break :blk false;
            };
            if (!auth_ok) {
                self.sendErrorResponse(
                    slot,
                    401,
                    "Unauthorized",
                    null,
                    "invalid or missing API key",
                    "authentication_error",
                    "invalid_api_key",
                );
                return;
            }
        }

        if (std.mem.eql(u8, hdr.path, "/v1/models")) {
            if (hdr.method != .get) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "GET", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleModels(slot);
            return;
        }
        if (std.mem.eql(u8, hdr.path, "/v1/ngram-packs")) {
            if (hdr.method != .get) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "GET", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleNgramPacks(slot);
            return;
        }
        if (std.mem.eql(u8, hdr.path, "/v1/ngram-packs/select")) {
            if (hdr.method != .post) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "POST", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleNgramPacksSelect(slot, body);
            return;
        }
        if (std.mem.eql(u8, hdr.path, "/v1/embeddings")) {
            if (hdr.method != .post) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "POST", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleEmbeddings(slot, body);
            return;
        }
        if (std.mem.eql(u8, hdr.path, "/v1/rerank")) {
            if (hdr.method != .post) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "POST", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleRerank(slot, body);
            return;
        }
        if (std.mem.eql(u8, hdr.path, "/v1/messages/count_tokens")) {
            if (hdr.method != .post) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "POST", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            if (!self.validateAnthropicHeaders(slot, &hdr)) return;
            self.handleMessagesCountTokens(slot, body);
            return;
        }

        // Stateful `/v1/responses/{id}` and `/v1/responses/{id}/cancel` routes (`C18`).
        if (std.mem.startsWith(u8, hdr.path, "/v1/responses/")) {
            const rest = hdr.path["/v1/responses/".len..];
            if (std.mem.endsWith(u8, rest, "/cancel")) {
                const resp_id = rest[0 .. rest.len - "/cancel".len];
                if (resp_id.len == 0 or std.mem.indexOfScalar(u8, resp_id, '/') != null) {
                    self.sendErrorResponse(slot, 404, "Not Found", null, "endpoint not found", "invalid_request_error", "not_found");
                    return;
                }
                if (hdr.method != .post) {
                    self.sendErrorResponse(slot, 405, "Method Not Allowed", "POST", "method not allowed", "invalid_request_error", "method_not_allowed");
                    return;
                }
                self.handleResponseCancel(slot, resp_id);
                return;
            }
            if (rest.len > 0 and std.mem.indexOfScalar(u8, rest, '/') == null) {
                switch (hdr.method) {
                    .get => self.handleResponseGet(slot, rest),
                    .delete => self.handleResponseDelete(slot, rest),
                    else => self.sendErrorResponse(slot, 405, "Method Not Allowed", "GET, DELETE", "method not allowed", "invalid_request_error", "method_not_allowed"),
                }
                return;
            }
        }

        const endpoint: Endpoint = if (std.mem.eql(u8, hdr.path, "/v1/chat/completions"))
            .chat_completions
        else if (std.mem.eql(u8, hdr.path, "/v1/responses"))
            .responses
        else if (std.mem.eql(u8, hdr.path, "/v1/messages"))
            .messages
        else {
            self.sendErrorResponse(slot, 404, "Not Found", null, "endpoint not found", "invalid_request_error", "not_found");
            return;
        };

        if (hdr.method != .post) {
            self.sendErrorResponse(slot, 405, "Method Not Allowed", "POST", "method not allowed", "invalid_request_error", "method_not_allowed");
            return;
        }
        if (endpoint == .messages and !self.validateAnthropicHeaders(slot, &hdr)) return;

        self.admitInferenceRequest(slot, endpoint, body);
    }

    fn validateAnthropicHeaders(self: *Server, slot: *ConnectionSlot, hdr: *const HttpHeadersResult) bool {
        if (hdr.anthropic_version) |ver| {
            if (ver.len == 0 or ver.len > MAX_ANTHROPIC_VERSION_BYTES) {
                self.sendErrorResponse(
                    slot,
                    400,
                    "Bad Request",
                    null,
                    "invalid anthropic-version header",
                    "invalid_request_error",
                    "invalid_anthropic_version",
                );
                return false;
            }
        } else if (self.config.require_anthropic_version) {
            self.sendErrorResponse(
                slot,
                400,
                "Bad Request",
                null,
                "missing required anthropic-version header",
                "invalid_request_error",
                "missing_anthropic_version",
            );
            return false;
        }
        return true;
    }

    fn sendJsonResponse(
        self: *Server,
        slot: *ConnectionSlot,
        status_code: u16,
        reason: []const u8,
        allow_header: ?[]const u8,
        seed_used: ?u64,
        seed_source: ?EffectiveSeedSource,
        json_body: []const u8,
    ) void {
        var hdr_buf: [512]u8 = undefined;
        var w = FixedWriter.init(&hdr_buf);
        w.print("HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n", .{
            status_code,
            reason,
            json_body.len,
        }) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        if (allow_header) |allow| {
            w.print("Allow: {s}\r\n", .{allow}) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
        }
        if (seed_used) |su| {
            w.print("X-Aminfer-Seed-Used: {d}\r\n", .{su}) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
        }
        if (seed_source) |ss| {
            w.print("X-Aminfer-Seed-Source: {s}\r\n", .{ss.name()}) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
        }
        w.writeAll("\r\n") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };

        slot.appendTx(w.written()) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        slot.appendTx(json_body) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        slot.state = .draining_response;
        if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
            self.closeSlotSocketAndRelease(slot, false);
        }
    }

    fn sendErrorResponse(
        self: *Server,
        slot: *ConnectionSlot,
        status_code: u16,
        reason: []const u8,
        allow_header: ?[]const u8,
        message: []const u8,
        err_type: []const u8,
        err_code: []const u8,
    ) void {
        var body_buf: [512]u8 = undefined;
        var w = FixedWriter.init(&body_buf);
        w.writeAll("{\"error\":{\"message\":\"") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        w.writeJsonEscaped(message) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        w.print("\",\"type\":\"{s}\",\"code\":\"{s}\"}}}}", .{ err_type, err_code }) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        self.sendJsonResponse(slot, status_code, reason, allow_header, null, null, w.written());
    }

    fn handleHealth(self: *Server, slot: *ConnectionSlot) void {
        var body_buf: [1024]u8 = undefined;
        var w = FixedWriter.init(&body_buf);
        const steps = self.total_batch_steps.load(.acquire);
        const slot_tokens = self.total_slot_tokens.load(.acquire);
        const mean_bs: f64 = if (steps == 0)
            0.0
        else
            @as(f64, @floatFromInt(slot_tokens)) / @as(f64, @floatFromInt(steps));
        const tok_per_s: f64 = @as(f64, @floatFromInt(self.last_tok_per_s_x100.load(.acquire))) / 100.0;

        var ang_buf: [MAX_NGRAM_DOMAIN_BYTES]u8 = undefined;
        const ang_pack: []const u8 = if (self.ngramRegistry()) |reg| reg.defaultDomainId(&ang_buf) else "";

        w.writeAll("{\"status\":\"ok\",\"model\":\"") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        w.writeJsonEscaped(self.modelId()) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        w.print(
            "\",\"wired_mib\":{d},\"active_requests\":{d},\"active_batch_size\":{d},\"mean_batch_size\":{d:.2},\"prefix_hit_tokens\":{d}," ++
                "\"active_prefill_slots\":{d},\"active_decode_slots\":{d},\"tok_per_s\":{d:.2},\"decode_share\":{d:.2}," ++
                "\"prefix_cache_hits\":{d},\"retired_q8_chunks\":{d},\"ang_pack\":",
            .{
                self.wired_mib.load(.acquire),
                self.active_requests.load(.acquire),
                self.active_batch_size.load(.acquire),
                mean_bs,
                self.prefix_hit_tokens.load(.acquire),
                self.active_prefill_slots.load(.acquire),
                self.active_decode_slots.load(.acquire),
                tok_per_s,
                self.config.decode_share,
                self.prefix_hit_requests.load(.acquire),
                self.retired_q8_chunks.load(.acquire),
            },
        ) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        if (ang_pack.len == 0) {
            w.writeAll("null") catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
        } else {
            w.writeByte('"') catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            w.writeJsonEscaped(ang_pack) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            w.writeByte('"') catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
        }
        w.print(",\"tb_peers\":{d}}}", .{self.tb_peers.load(.acquire)}) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        self.sendJsonResponse(slot, 200, "OK", null, null, null, w.written());
    }

    fn handleNgramPacks(self: *Server, slot: *ConnectionSlot) void {
        var body_buf: [4096]u8 = undefined;
        var w = FixedWriter.init(&body_buf);
        const reg = self.ngramRegistry() orelse {
            w.writeAll("{\"object\":\"list\",\"default_domain\":null,\"pack_count\":0,\"cache_pages\":0,\"wired_bytes\":0,\"data\":[]}") catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            self.sendJsonResponse(slot, 200, "OK", null, null, null, w.written());
            return;
        };

        var infos: [ngram_nvme.MAX_DOMAIN_PACKS]ngram_nvme.DomainPackInfo = undefined;
        const count = reg.snapshotPacks(&infos);
        var def_buf: [MAX_NGRAM_DOMAIN_BYTES]u8 = undefined;
        const def_dom = reg.defaultDomainId(&def_buf);

        w.writeAll("{\"object\":\"list\",\"default_domain\":") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        if (def_dom.len == 0) {
            w.writeAll("null") catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
        } else {
            w.writeByte('"') catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            w.writeJsonEscaped(def_dom) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            w.writeByte('"') catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
        }
        w.print(",\"pack_count\":{d},\"cache_pages\":{d},\"wired_bytes\":{d},\"data\":[", .{
            count,
            reg.cachePageCount(),
            reg.totalWiredBytes(),
        }) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };

        for (infos[0..count], 0..) |*info, idx| {
            if (idx > 0) {
                w.writeByte(',') catch {
                    self.closeSlotSocketAndRelease(slot, false);
                    return;
                };
            }
            w.writeAll("{\"domain\":\"") catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            w.writeJsonEscaped(info.domainIdSlice()) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            w.writeAll("\",\"corpus_id\":\"") catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            w.writeJsonEscaped(info.corpusIdSlice()) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            w.print(
                "\",\"is_default\":{s},\"entries\":{d},\"buckets\":{d},\"min_ngram\":{d},\"max_ngram\":{d},\"generation\":{d},\"swaps\":{d},\"stats\":{{\"queries\":{d},\"hits\":{d},\"seed_hits\":{d},\"branch_hits\":{d},\"cache_hits\":{d},\"prefetch_submitted\":{d},\"prefetch_hits\":{d},\"nvme_reads\":{d},\"hit_rate\":{d:.4}}}}}",
                .{
                    if (info.is_default) "true" else "false",
                    info.total_entries,
                    info.num_buckets,
                    info.min_ngram,
                    info.max_ngram,
                    info.generation,
                    info.swaps,
                    info.stats.queries,
                    info.stats.hits,
                    info.stats.seed_hits,
                    info.stats.branch_hits,
                    info.stats.cache_hits,
                    info.stats.prefetch_submitted,
                    info.stats.prefetch_hits,
                    info.stats.nvme_reads,
                    info.stats.hitRate(),
                },
            ) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
        }

        w.writeAll("]}") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        self.sendJsonResponse(slot, 200, "OK", null, null, null, w.written());
    }

    fn handleNgramPacksSelect(self: *Server, slot: *ConnectionSlot, body: []const u8) void {
        var dom_buf: [MAX_NGRAM_DOMAIN_BYTES]u8 = undefined;
        const domain = parseSelectNgramDomainRequest(body, &dom_buf) catch |err| {
            const msg = switch (err) {
                error.InvalidJson => "invalid JSON request body",
                error.MissingRequiredField => "missing required field 'domain'",
                error.InvalidFieldValue => "invalid 'domain' value",
                error.BufferTooSmall => "domain identifier too long",
                error.DepthExceeded => "JSON nesting depth exceeded",
                error.UnsupportedFeature => "unsupported request feature",
            };
            self.sendErrorResponse(slot, 400, "Bad Request", null, msg, "invalid_request_error", "bad_request");
            return;
        };

        const reg = self.ngramRegistry() orelse {
            self.sendErrorResponse(slot, 404, "Not Found", null, "no ngram pack registry configured", "invalid_request_error", "ngram_registry_unavailable");
            return;
        };

        reg.selectDefaultDomain(domain) catch |err| switch (err) {
            error.InvalidDomainId => {
                self.sendErrorResponse(slot, 400, "Bad Request", null, "invalid domain identifier", "invalid_request_error", "invalid_domain");
                return;
            },
            error.DomainNotFound => {
                self.sendErrorResponse(slot, 404, "Not Found", null, "domain pack not found", "invalid_request_error", "domain_not_found");
                return;
            },
        };

        var resp_buf: [256]u8 = undefined;
        var w = FixedWriter.init(&resp_buf);
        w.writeAll("{\"status\":\"ok\",\"default_domain\":\"") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        w.writeJsonEscaped(domain) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        w.writeAll("\"}") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        self.sendJsonResponse(slot, 200, "OK", null, null, null, w.written());
    }

    fn handleModels(self: *Server, slot: *ConnectionSlot) void {
        var body_buf: [512]u8 = undefined;
        var w = FixedWriter.init(&body_buf);
        w.writeAll("{\"object\":\"list\",\"data\":[{\"id\":\"") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        w.writeJsonEscaped(self.modelId()) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        w.print("\",\"object\":\"model\",\"created\":{d},\"owned_by\":\"aminfer\"}}]}}", .{
            unixTimestampSec(),
        }) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        self.sendJsonResponse(slot, 200, "OK", null, null, null, w.written());
    }

    // -----------------------------------------------------------------------
    // C18 / C21 synchronous handlers: count_tokens, embeddings, rerank,
    // stateful /v1/responses/{id} GET / DELETE / cancel
    // -----------------------------------------------------------------------

    fn handleMessagesCountTokens(self: *Server, slot: *ConnectionSlot, body: []const u8) void {
        const parsed = parseInferenceRequest(
            .messages,
            body,
            self.modelId(),
            slot.prompt_buf,
            &slot.model_buf,
        ) catch |err| {
            const msg = switch (err) {
                error.InvalidJson => "invalid JSON request body",
                error.MissingRequiredField => "missing required field in request body",
                error.InvalidFieldValue => "invalid parameter value in request body",
                error.UnsupportedFeature => "unsupported request feature",
                error.BufferTooSmall => "prompt exceeds preallocated slot buffer",
                error.DepthExceeded => "JSON nesting depth exceeded",
            };
            const code: u16 = if (err == error.BufferTooSmall) 413 else 400;
            const reason = if (err == error.BufferTooSmall) "Payload Too Large" else "Bad Request";
            self.sendErrorResponse(slot, code, reason, null, msg, "invalid_request_error", "bad_request");
            return;
        };

        const tok = self.tokenizer() orelse {
            self.sendErrorResponse(
                slot,
                501,
                "Not Implemented",
                null,
                "native tokenizer not attached for /v1/messages/count_tokens",
                "api_error",
                "tokenizer_unavailable",
            );
            return;
        };

        const prompt_bytes = slot.prompt_buf[0..parsed.prompt_len];
        const ids = tok.encodeWithScratch(&self.aux_encode_scratch, prompt_bytes, self.aux_token_ids) catch {
            self.sendErrorResponse(
                slot,
                413,
                "Payload Too Large",
                null,
                "prompt exceeds preallocated token counting capacity",
                "invalid_request_error",
                "token_limit_exceeded",
            );
            return;
        };

        var resp_buf: [128]u8 = undefined;
        var w = FixedWriter.init(&resp_buf);
        w.print("{{\"input_tokens\":{d}}}", .{ids.len}) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        self.sendJsonResponse(slot, 200, "OK", null, null, null, w.written());
    }

    fn handleEmbeddings(self: *Server, slot: *ConnectionSlot, body: []const u8) void {
        const req = parseEmbeddingsRequest(
            body,
            self.modelId(),
            &slot.model_buf,
            self.aux_text_scratch,
        ) catch |err| {
            const msg = switch (err) {
                error.InvalidJson => "invalid JSON request body",
                error.MissingRequiredField => "missing required field 'input'",
                error.InvalidFieldValue => "invalid field value in embeddings request",
                error.BufferTooSmall => "embeddings request exceeds preallocated scratch capacity",
                error.DepthExceeded => "JSON nesting depth exceeded",
                error.UnsupportedFeature => "unsupported embeddings feature",
            };
            const code: u16 = if (err == error.BufferTooSmall) 413 else 400;
            const reason = if (err == error.BufferTooSmall) "Payload Too Large" else "Bad Request";
            self.sendErrorResponse(slot, code, reason, null, msg, "invalid_request_error", "bad_request");
            return;
        };

        const handler = self.embedding_handler orelse {
            self.sendErrorResponse(
                slot,
                501,
                "Not Implemented",
                null,
                "embeddings backend not configured",
                "api_error",
                "embeddings_unavailable",
            );
            return;
        };

        const res = handler.embedFn(handler.ctx, &req, self.aux_embed_vectors) catch {
            self.sendErrorResponse(
                slot,
                500,
                "Internal Server Error",
                null,
                "embedding computation failed",
                "server_error",
                "embedding_failed",
            );
            return;
        };
        if (res.dim == 0 or req.input_count * res.dim > self.aux_embed_vectors.len) {
            self.sendErrorResponse(
                slot,
                500,
                "Internal Server Error",
                null,
                "invalid embedding dimension returned by backend",
                "server_error",
                "embedding_failed",
            );
            return;
        }

        const HEADER_RESERVE: usize = 256;
        if (slot.tx_buf.len <= HEADER_RESERVE) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        const json_body = formatEmbeddingsResponse(
            &req,
            self.aux_embed_vectors[0 .. req.input_count * res.dim],
            res,
            slot.tx_buf[HEADER_RESERVE..],
        ) catch {
            self.sendErrorResponse(slot, 500, "Internal Server Error", null, "embeddings response exceeds tx buffer", "server_error", "tx_overflow");
            return;
        };

        var hdr_buf: [HEADER_RESERVE]u8 = undefined;
        var hw = FixedWriter.init(&hdr_buf);
        hw.print(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n",
            .{json_body.len},
        ) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        const hdr_start = HEADER_RESERVE - hw.pos;
        @memcpy(slot.tx_buf[hdr_start..HEADER_RESERVE], hw.written());
        slot.tx_read_pos = hdr_start;
        slot.tx_write_pos = HEADER_RESERVE + json_body.len;
        slot.state = .draining_response;
        if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
            self.closeSlotSocketAndRelease(slot, false);
        }
    }

    fn handleRerank(self: *Server, slot: *ConnectionSlot, body: []const u8) void {
        const req = parseRerankRequest(
            body,
            self.modelId(),
            &slot.model_buf,
            self.aux_text_scratch,
        ) catch |err| {
            const msg = switch (err) {
                error.InvalidJson => "invalid JSON request body",
                error.MissingRequiredField => "missing required field 'query' or 'documents'",
                error.InvalidFieldValue => "invalid field value in rerank request",
                error.BufferTooSmall => "rerank request exceeds preallocated scratch capacity",
                error.DepthExceeded => "JSON nesting depth exceeded",
                error.UnsupportedFeature => "unsupported rerank feature",
            };
            const code: u16 = if (err == error.BufferTooSmall) 413 else 400;
            const reason = if (err == error.BufferTooSmall) "Payload Too Large" else "Bad Request";
            self.sendErrorResponse(slot, code, reason, null, msg, "invalid_request_error", "bad_request");
            return;
        };

        const handler = self.rerank_handler orelse {
            self.sendErrorResponse(
                slot,
                501,
                "Not Implemented",
                null,
                "rerank backend not configured",
                "api_error",
                "rerank_unavailable",
            );
            return;
        };

        var items_buf: [MAX_RERANK_DOCS]RerankItem = undefined;
        const res = handler.rerankFn(handler.ctx, &req, &items_buf) catch {
            self.sendErrorResponse(
                slot,
                500,
                "Internal Server Error",
                null,
                "rerank computation failed",
                "server_error",
                "rerank_failed",
            );
            return;
        };
        const n = @min(res.count, MAX_RERANK_DOCS);

        const HEADER_RESERVE: usize = 256;
        if (slot.tx_buf.len <= HEADER_RESERVE) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        const json_body = formatRerankResponse(
            &req,
            items_buf[0..n],
            res.prompt_tokens,
            slot.tx_buf[HEADER_RESERVE..],
        ) catch {
            self.sendErrorResponse(slot, 500, "Internal Server Error", null, "rerank response exceeds tx buffer", "server_error", "tx_overflow");
            return;
        };

        var hdr_buf: [HEADER_RESERVE]u8 = undefined;
        var hw = FixedWriter.init(&hdr_buf);
        hw.print(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n",
            .{json_body.len},
        ) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        const hdr_start = HEADER_RESERVE - hw.pos;
        @memcpy(slot.tx_buf[hdr_start..HEADER_RESERVE], hw.written());
        slot.tx_read_pos = hdr_start;
        slot.tx_write_pos = HEADER_RESERVE + json_body.len;
        slot.state = .draining_response;
        if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
            self.closeSlotSocketAndRelease(slot, false);
        }
    }

    fn findStatefulResponseById(self: *Server, resp_id: []const u8) ?*StatefulResponseSlot {
        for (self.response_store) |*rec| {
            if (rec.occupied and std.mem.eql(u8, rec.idSlice(), resp_id)) {
                rec.lru_tick = self.response_lru_clock;
                self.response_lru_clock +%= 1;
                return rec;
            }
        }
        return null;
    }

    /// Allocates or LRU-evicts (`evict_first`) a preallocated fixed slot for a `/v1/responses` record.
    fn allocStatefulResponseSlot(self: *Server) ?u16 {
        if (self.response_store.len == 0) return null;
        var victim_idx: ?usize = null;
        var oldest_tick: u64 = std.math.maxInt(u64);
        for (self.response_store, 0..) |*rec, idx| {
            if (!rec.occupied) {
                return @intCast(idx);
            }
            // Prefer evicting completed/cancelled/failed slots before any in-progress slot.
            if (rec.status != .in_progress and rec.lru_tick < oldest_tick) {
                oldest_tick = rec.lru_tick;
                victim_idx = idx;
            }
        }
        if (victim_idx) |vi| return @intCast(vi);
        // All slots in-progress: evict global oldest.
        var any_idx: usize = 0;
        var any_tick: u64 = self.response_store[0].lru_tick;
        for (self.response_store[1..], 1..) |*rec, idx| {
            if (rec.lru_tick < any_tick) {
                any_tick = rec.lru_tick;
                any_idx = idx;
            }
        }
        return @intCast(any_idx);
    }

    fn writeResponseObjectJson(bw: *FixedWriter, rec: *const StatefulResponseSlot) error{BufferTooSmall}!void {
        try bw.writeAll("{\"id\":\"");
        try bw.writeJsonEscaped(rec.idSlice());
        try bw.print("\",\"object\":\"response\",\"created_at\":{d},\"status\":\"{s}\",\"model\":\"", .{
            rec.created_unix,
            rec.status.name(),
        });
        try bw.writeJsonEscaped(rec.modelSlice());
        try bw.print("\",\"seed_used\":{d},\"seed_source\":\"{s}\"", .{
            rec.seed_used,
            rec.seed_source.name(),
        });
        if (rec.previous_id_len > 0) {
            try bw.writeAll(",\"previous_response_id\":\"");
            try bw.writeJsonEscaped(rec.previousIdSlice());
            try bw.writeByte('"');
        }
        try bw.writeAll(",\"output\":[");
        var need_comma = false;
        const reasoning = rec.reasoning_buf[0..rec.reasoning_len];
        const content = rec.content_buf[0..rec.content_len];

        if (reasoning.len > 0) {
            try bw.print(
                "{{\"id\":\"rs_{d}\",\"type\":\"reasoning\",\"summary\":[{{\"type\":\"summary_text\",\"text\":\"",
                .{rec.request_id},
            );
            try bw.writeJsonEscaped(reasoning);
            try bw.writeAll("\"}]}");
            need_comma = true;
        }
        if (content.len > 0 or rec.tool_call_count == 0) {
            if (need_comma) try bw.writeByte(',');
            try bw.print(
                "{{\"id\":\"msg_{d}\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"{s}\",\"content\":[{{\"type\":\"output_text\",\"text\":\"",
                .{ rec.request_id, rec.status.name() },
            );
            try bw.writeJsonEscaped(content);
            try bw.writeAll("\"}]}");
            need_comma = true;
        }
        for (rec.tool_calls[0..rec.tool_call_count], 0..) |*tc, idx| {
            if (need_comma) try bw.writeByte(',');
            try bw.print("{{\"id\":\"fc_{d}_{d}\",\"type\":\"function_call\",\"status\":\"completed\",\"call_id\":\"", .{
                rec.request_id,
                idx,
            });
            try bw.writeJsonEscaped(tc.idSlice());
            try bw.writeAll("\",\"name\":\"");
            try bw.writeJsonEscaped(tc.nameSlice());
            try bw.writeAll("\",\"arguments\":\"");
            try bw.writeJsonEscaped(tc.argsSlice());
            try bw.writeAll("\"}");
            need_comma = true;
        }

        const total_tokens = rec.prompt_tokens +| rec.completion_tokens;
        try bw.writeAll("],\"output_text\":\"");
        try bw.writeJsonEscaped(content);
        try bw.print(
            "\",\"usage\":{{\"input_tokens\":{d},\"input_tokens_details\":{{\"cached_tokens\":{d}}},\"output_tokens\":{d},\"output_tokens_details\":{{\"reasoning_tokens\":{d}}},\"total_tokens\":{d}}}}}",
            .{
                rec.prompt_tokens,
                rec.cached_tokens,
                rec.completion_tokens,
                rec.reasoning_tokens,
                total_tokens,
            },
        );
    }

    fn handleResponseGet(self: *Server, slot: *ConnectionSlot, resp_id: []const u8) void {
        const rec = self.findStatefulResponseById(resp_id) orelse {
            self.sendErrorResponse(slot, 404, "Not Found", null, "response not found", "invalid_request_error", "response_not_found");
            return;
        };
        const HEADER_RESERVE: usize = 256;
        if (slot.tx_buf.len <= HEADER_RESERVE) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        var bw = FixedWriter.init(slot.tx_buf[HEADER_RESERVE..]);
        writeResponseObjectJson(&bw, rec) catch {
            self.sendErrorResponse(slot, 500, "Internal Server Error", null, "response exceeds tx buffer", "server_error", "tx_overflow");
            return;
        };
        var hdr_buf: [HEADER_RESERVE]u8 = undefined;
        var hw = FixedWriter.init(&hdr_buf);
        hw.print(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n",
            .{bw.pos},
        ) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        const hdr_start = HEADER_RESERVE - hw.pos;
        @memcpy(slot.tx_buf[hdr_start..HEADER_RESERVE], hw.written());
        slot.tx_read_pos = hdr_start;
        slot.tx_write_pos = HEADER_RESERVE + bw.pos;
        slot.state = .draining_response;
        if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
            self.closeSlotSocketAndRelease(slot, false);
        }
    }

    fn handleResponseDelete(self: *Server, slot: *ConnectionSlot, resp_id: []const u8) void {
        const rec = self.findStatefulResponseById(resp_id) orelse {
            self.sendErrorResponse(slot, 404, "Not Found", null, "response not found", "invalid_request_error", "response_not_found");
            return;
        };
        if (rec.status == .in_progress) {
            if (rec.active_conn_slot) |cidx| {
                if (cidx < self.slots.len and self.slots[cidx].generation == rec.active_generation) {
                    self.slots[cidx].cancelled.store(true, .release);
                    self.wakeWorkerLoop();
                }
            }
        }
        var resp_buf: [256]u8 = undefined;
        var w = FixedWriter.init(&resp_buf);
        w.writeAll("{\"id\":\"") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        w.writeJsonEscaped(rec.idSlice()) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        w.writeAll("\",\"object\":\"response.deleted\",\"deleted\":true}") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        rec.occupied = false;
        rec.active_conn_slot = null;
        self.sendJsonResponse(slot, 200, "OK", null, null, null, w.written());
    }

    fn handleResponseCancel(self: *Server, slot: *ConnectionSlot, resp_id: []const u8) void {
        const rec = self.findStatefulResponseById(resp_id) orelse {
            self.sendErrorResponse(slot, 404, "Not Found", null, "response not found", "invalid_request_error", "response_not_found");
            return;
        };
        if (rec.status == .in_progress) {
            rec.status = .cancelled;
            if (rec.active_conn_slot) |cidx| {
                if (cidx < self.slots.len and self.slots[cidx].generation == rec.active_generation) {
                    if (!self.slots[cidx].cancelled.swap(true, .acq_rel)) {
                        _ = self.cancelled_count.fetchAdd(1, .monotonic);
                    }
                    self.wakeWorkerLoop();
                }
            }
            rec.active_conn_slot = null;
        }
        const HEADER_RESERVE: usize = 256;
        if (slot.tx_buf.len <= HEADER_RESERVE) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        var bw = FixedWriter.init(slot.tx_buf[HEADER_RESERVE..]);
        writeResponseObjectJson(&bw, rec) catch {
            self.sendErrorResponse(slot, 500, "Internal Server Error", null, "response exceeds tx buffer", "server_error", "tx_overflow");
            return;
        };
        var hdr_buf: [HEADER_RESERVE]u8 = undefined;
        var hw = FixedWriter.init(&hdr_buf);
        hw.print(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n",
            .{bw.pos},
        ) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        const hdr_start = HEADER_RESERVE - hw.pos;
        @memcpy(slot.tx_buf[hdr_start..HEADER_RESERVE], hw.written());
        slot.tx_read_pos = hdr_start;
        slot.tx_write_pos = HEADER_RESERVE + bw.pos;
        slot.state = .draining_response;
        if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
            self.closeSlotSocketAndRelease(slot, false);
        }
    }

    fn finalizeStatefulResponseRecord(self: *Server, slot: *ConnectionSlot, status: ResponseRecordStatus) void {
        const r_idx = slot.stateful_resp_slot_idx orelse return;
        if (r_idx >= self.response_store.len) return;
        const rec = &self.response_store[r_idx];
        if (!rec.occupied or rec.request_id != slot.request_id) return;

        rec.status = status;
        rec.active_conn_slot = null;
        rec.prompt_tokens = slot.prompt_tokens;
        rec.cached_tokens = slot.cached_tokens;
        rec.completion_tokens = slot.completion_tokens;
        rec.reasoning_tokens = slot.reasoning_tokens;
        rec.lru_tick = self.response_lru_clock;
        self.response_lru_clock +%= 1;

        const r_len = @min(slot.nonstream_reasoning_len, rec.reasoning_buf.len);
        if (r_len > 0) @memcpy(rec.reasoning_buf[0..r_len], slot.nonstream_reasoning_buf[0..r_len]);
        rec.reasoning_len = r_len;

        const c_len = @min(slot.nonstream_content_len, rec.content_buf.len);
        if (c_len > 0) @memcpy(rec.content_buf[0..c_len], slot.nonstream_content_buf[0..c_len]);
        rec.content_len = c_len;

        rec.tool_call_count = slot.stored_tool_call_count;
        for (0..slot.stored_tool_call_count) |i| {
            rec.tool_calls[i] = slot.stored_tool_calls[i];
        }

        // Persist full completed ChatML transcript (`prompt` + assistant output + `<|im_end|>\n`)
        // so follow-up requests referencing `previous_response_id` chain losslessly!
        var cw = FixedWriter.init(rec.chatml_buf);
        const prompt_bytes = slot.prompt_buf[0..slot.prompt_len];
        const copy_p = @min(prompt_bytes.len, cw.remaining());
        if (copy_p > 0) {
            @memcpy(cw.buf[cw.pos .. cw.pos + copy_p], prompt_bytes[0..copy_p]);
            cw.pos += copy_p;
        }
        if (c_len > 0) {
            const copy_c = @min(c_len, cw.remaining());
            if (copy_c > 0) {
                @memcpy(cw.buf[cw.pos .. cw.pos + copy_c], rec.content_buf[0..copy_c]);
                cw.pos += copy_c;
            }
        }
        for (rec.tool_calls[0..rec.tool_call_count]) |*tc| {
            cw.writeAll("<tool_call>\n{\"name\":\"") catch break;
            cw.writeJsonEscaped(tc.nameSlice()) catch break;
            cw.writeAll("\",\"arguments\":") catch break;
            cw.writeAll(if (tc.args_len > 0) tc.argsSlice() else "{}") catch break;
            cw.writeAll("}\n</tool_call>\n") catch break;
        }
        const end_tok = "<|im_end|>\n";
        if (cw.remaining() >= end_tok.len) {
            @memcpy(cw.buf[cw.pos .. cw.pos + end_tok.len], end_tok);
            cw.pos += end_tok.len;
        }
        rec.chatml_len = cw.pos;
    }

    fn admitInferenceRequest(self: *Server, slot: *ConnectionSlot, endpoint: Endpoint, body: []const u8) void {
        // Resolve optional `previous_response_id` on `/v1/responses` before building ChatML prompt
        // so the full chained transcript is prepended directly into `slot.prompt_buf` (`0` heap alloc).
        var prior_chatml: ?[]const u8 = null;
        if (endpoint == .responses) {
            var prev_probe_buf: [MAX_RESPONSE_ID_BYTES]u8 = undefined;
            if (findJsonStringField(body, "previous_response_id", &prev_probe_buf)) |prev_id| {
                if (prev_id.len > 0) {
                    const prev_rec = self.findStatefulResponseById(prev_id) orelse {
                        self.sendErrorResponse(
                            slot,
                            404,
                            "Not Found",
                            null,
                            "previous_response_id not found or evicted",
                            "invalid_request_error",
                            "response_not_found",
                        );
                        return;
                    };
                    prior_chatml = prev_rec.chatml_buf[0..prev_rec.chatml_len];
                }
            }
        }

        const parsed = parseInferenceRequestWithPriorPrompt(
            endpoint,
            body,
            self.modelId(),
            prior_chatml,
            slot.prompt_buf,
            &slot.model_buf,
        ) catch |err| {
            const msg = switch (err) {
                error.InvalidJson => "invalid JSON request body",
                error.MissingRequiredField => "missing required field in request body",
                error.InvalidFieldValue => "invalid parameter value in request body",
                error.UnsupportedFeature => "unsupported request feature",
                error.BufferTooSmall => "prompt exceeds preallocated slot buffer",
                error.DepthExceeded => "JSON nesting depth exceeded",
            };
            const code: u16 = if (err == error.BufferTooSmall) 413 else 400;
            const reason = if (err == error.BufferTooSmall) "Payload Too Large" else "Bad Request";
            self.sendErrorResponse(slot, code, reason, null, msg, "invalid_request_error", "bad_request");
            return;
        };

        // Compile `C22` preallocated grammar program at request admission when structured outputs
        // (`json_object`, `json_schema`, `gbnf`) or forced strict tool schemas are requested.
        // Strictly bypassed (`grammar_program = null`) when unconstrained!
        slot.has_grammar = false;
        const gprog = &self.grammar_programs[slot.grammar_slot_index];
        switch (parsed.response_format) {
            .text => {
                // Forced tool call (`required` or `named`) with tools -> compile strict `<tool_call>` grammar!
                if (parsed.tool_count > 0 and (parsed.tool_choice == .required or parsed.tool_choice == .named)) {
                    var specs: [MAX_TOOLS_PER_REQUEST]grammar_mod.ToolSchemaSpec = undefined;
                    var spec_count: usize = 0;
                    const chosen_name = parsed.toolChoiceNameSlice();
                    for (parsed.toolsSlice()) |*t| {
                        if (parsed.tool_choice == .named and !std.mem.eql(u8, t.nameSlice(), chosen_name)) {
                            continue;
                        }
                        specs[spec_count] = .{
                            .name = t.nameSlice(),
                            .parameters_json = if (t.parameters_json.len > 0) t.parameters_json else "{\"type\":\"object\"}",
                        };
                        spec_count += 1;
                    }
                    if (parsed.tool_choice == .named and spec_count == 0) {
                        self.sendErrorResponse(slot, 400, "Bad Request", null, "tool_choice references unknown tool", "invalid_request_error", "invalid_tool_choice");
                        return;
                    }
                    if (spec_count > 0) {
                        grammar_mod.compileStrictToolCalls(
                            gprog,
                            specs[0..spec_count],
                            parsed.parallel_tool_calls,
                            parsed.enable_thinking,
                        ) catch {
                            self.sendErrorResponse(slot, 400, "Bad Request", null, "failed to compile tool JSON schema", "invalid_request_error", "invalid_schema");
                            return;
                        };
                        slot.has_grammar = true;
                    }
                }
            },
            .json_object => {
                grammar_mod.compileJsonObject(gprog) catch {
                    self.sendErrorResponse(slot, 400, "Bad Request", null, "failed to compile json_object grammar", "invalid_request_error", "invalid_schema");
                    return;
                };
                slot.has_grammar = true;
            },
            .json_schema => {
                grammar_mod.compileJsonSchema(gprog, parsed.response_format_schema_json) catch {
                    self.sendErrorResponse(slot, 400, "Bad Request", null, "invalid json_schema in response_format/text.format", "invalid_request_error", "invalid_schema");
                    return;
                };
                slot.has_grammar = true;
            },
            .gbnf => {
                // Unescape JSON string literal for GBNF grammar into `aux_text_scratch` if quoted.
                var gbnf_text = parsed.response_format_schema_json;
                if (gbnf_text.len >= 2 and gbnf_text[0] == '"') {
                    var gr = JsonReader{ .src = gbnf_text };
                    var gw = FixedWriter.init(self.aux_text_scratch);
                    gr.readStringInto(&gw) catch {
                        self.sendErrorResponse(slot, 400, "Bad Request", null, "invalid GBNF grammar string", "invalid_request_error", "invalid_grammar");
                        return;
                    };
                    gbnf_text = gw.written();
                }
                grammar_mod.compileGbnf(gprog, gbnf_text) catch {
                    self.sendErrorResponse(slot, 400, "Bad Request", null, "invalid GBNF grammar syntax", "invalid_request_error", "invalid_grammar");
                    return;
                };
                slot.has_grammar = true;
            },
        }

        // Validate requested `"ngram_domain"` against attached DomainNgramRegistry when present.
        if (parsed.ngram_domain_len > 0) {
            if (self.ngramRegistry()) |reg| {
                if (!reg.hasDomain(parsed.ngramDomainSlice())) {
                    self.sendErrorResponse(
                        slot,
                        400,
                        "Bad Request",
                        null,
                        "unknown ngram_domain",
                        "invalid_request_error",
                        "domain_not_found",
                    );
                    return;
                }
            }
        }

        // Graceful 503 backpressure when all active inference slots or job ring entries are busy.
        const current_active = self.active_requests.load(.acquire);
        if (current_active >= self.config.max_active_requests or self.job_ring.isFull()) {
            _ = self.rejected_503_count.fetchAdd(1, .monotonic);
            self.sendErrorResponse(
                slot,
                503,
                "Service Unavailable",
                null,
                "server busy: all preallocated request slots are in use",
                "server_error",
                "server_overloaded",
            );
            return;
        }

        // D11 two-tier per-request RNG initialization.
        if (parsed.seed) |explicit_seed| {
            slot.rng = rng_mod.RequestRng.initDeterministic(explicit_seed);
            slot.seed_used = explicit_seed;
            slot.seed_source = .explicit;
        } else if (parsed.temperature > 0.0) {
            slot.rng = rng_mod.RequestRng.init(null) catch {
                self.sendErrorResponse(slot, 503, "Service Unavailable", null, "hardware entropy unavailable", "server_error", "entropy_unavailable");
                return;
            };
            slot.seed_used = slot.rng.seed_used;
            slot.seed_source = .hardware_trng;
        } else {
            slot.rng = rng_mod.RequestRng.initDeterministic(0);
            slot.seed_used = 0;
            slot.seed_source = .greedy;
        }

        slot.generation +%= 1;
        slot.request_id = self.next_request_id;
        self.next_request_id +%= 1;
        slot.endpoint = endpoint;
        slot.stream = parsed.stream;
        slot.model_len = parsed.model_len;
        slot.prompt_len = parsed.prompt_len;
        slot.ngram_domain_buf = parsed.ngram_domain_buf;
        slot.ngram_domain_len = parsed.ngram_domain_len;
        slot.logit_bias_entries = parsed.logit_bias_entries;
        slot.logit_bias_len = parsed.logit_bias_len;
        slot.stop_config = parsed.stop_config;
        slot.stop_holdback.reset();
        slot.stream_include_usage = parsed.stream_include_usage;
        slot.store_response = parsed.store_response;
        slot.previous_response_id_buf = parsed.previous_response_id_buf;
        slot.previous_response_id_len = parsed.previous_response_id_len;
        slot.stateful_resp_slot_idx = null;
        slot.admit_ns = monotonicNowNs();
        slot.first_token_ns = 0;
        slot.created_unix = unixTimestampSec();
        slot.prompt_tokens = 0;
        slot.cached_tokens = 0;
        slot.completion_tokens = 0;
        slot.reasoning_tokens = 0;
        slot.sent_first_chunk = false;
        slot.preamble_sent = false;
        slot.responses_seq_num = 0;
        slot.responses_output_index = 0;
        slot.responses_reasoning_opened = false;
        slot.responses_reasoning_closed = false;
        slot.responses_reasoning_output_idx = 0;
        slot.responses_text_opened = false;
        slot.responses_text_closed = false;
        slot.responses_text_output_idx = 0;
        slot.responses_active_call_output_idx = 0;
        slot.anthropic_block = .none;
        slot.anthropic_block_idx = 0;
        slot.nonstream_reasoning_len = 0;
        slot.nonstream_content_len = 0;
        slot.think_splitter.reset(false);
        slot.tool_parser.reset(slot.request_id);
        slot.stored_tool_call_count = 0;
        slot.cancelled.store(false, .release);
        while (slot.token_ring.tryPop() != null) {}

        // Reserve stateful `/v1/responses` slot (`in_progress`) so GET & POST /cancel work live!
        if (endpoint == .responses and parsed.store_response) {
            if (self.allocStatefulResponseSlot()) |r_idx| {
                const rec = &self.response_store[r_idx];
                rec.occupied = true;
                rec.lru_tick = self.response_lru_clock;
                self.response_lru_clock +%= 1;
                rec.request_id = slot.request_id;
                rec.created_unix = slot.created_unix;
                rec.status = .in_progress;
                rec.seed_used = slot.seed_used;
                rec.seed_source = slot.seed_source;
                rec.prompt_tokens = 0;
                rec.cached_tokens = 0;
                rec.completion_tokens = 0;
                rec.reasoning_tokens = 0;
                rec.active_conn_slot = slot.index;
                rec.active_generation = slot.generation;
                var idw = FixedWriter.init(&rec.id_buf);
                idw.print("resp_{d}", .{slot.request_id}) catch {
                    rec.occupied = false;
                };
                rec.id_len = @intCast(idw.pos);
                rec.model_len = slot.model_len;
                @memcpy(rec.model_buf[0..slot.model_len], slot.model_buf[0..slot.model_len]);
                rec.previous_id_len = slot.previous_response_id_len;
                if (rec.previous_id_len > 0) {
                    @memcpy(rec.previous_id_buf[0..rec.previous_id_len], slot.previous_response_id_buf[0..rec.previous_id_len]);
                }
                rec.chatml_len = 0;
                rec.reasoning_len = 0;
                rec.content_len = 0;
                rec.tool_call_count = 0;
                slot.stateful_resp_slot_idx = r_idx;
            }
        }

        // For SSE streams, send HTTP 200 headers (with X-Aminfer-Seed-Used) immediately.
        if (parsed.stream) {
            var hdr_buf: [512]u8 = undefined;
            var w = FixedWriter.init(&hdr_buf);
            w.print(
                "HTTP/1.1 200 OK\r\n" ++
                    "Content-Type: text/event-stream\r\n" ++
                    "Cache-Control: no-cache\r\n" ++
                    "Connection: close\r\n" ++
                    "X-Aminfer-Seed-Used: {d}\r\n" ++
                    "X-Aminfer-Seed-Source: {s}\r\n" ++
                    "\r\n",
                .{ slot.seed_used, slot.seed_source.name() },
            ) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            slot.appendTx(w.written()) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
        }

        slot.worker_active.store(true, .release);
        slot.state = .streaming_active;
        slot.request_counted = true;
        _ = self.active_requests.fetchAdd(1, .acq_rel);
        _ = self.total_requests.fetchAdd(1, .monotonic);

        const model_tag = parseModelTag(slot.modelSlice());
        var samp_params = parsed.samplingParams();
        samp_params.logit_bias = slot.logit_bias_entries[0..slot.logit_bias_len];

        const job = InferenceJob{
            .slot_index = slot.index,
            .generation = slot.generation,
            .request_id = slot.request_id,
            .endpoint = endpoint,
            .prompt = slot.prompt_buf[0..parsed.prompt_len],
            .last_user_msg_byte_offset = parsed.last_user_msg_byte_offset,
            .rng = &slot.rng,
            .seed_used = slot.seed_used,
            .seed_source = slot.seed_source,
            .max_tokens = parsed.max_tokens,
            .temperature = parsed.temperature,
            .stream = parsed.stream,
            .ignore_eos = parsed.ignore_eos,
            .enable_thinking = parsed.enable_thinking,
            .ngram_domain = slot.ngramDomainSlice(),
            .model_base = model_tag.base,
            .lora_adapter = model_tag.adapter,
            .sampling = samp_params,
            .stop_config = &slot.stop_config,
            .grammar_program = if (slot.has_grammar) gprog else null,
            .response_format = parsed.response_format,
            .tool_choice = parsed.tool_choice,
            .parallel_tool_calls = parsed.parallel_tool_calls,
            .has_tools = parsed.tool_count > 0,
            .reasoning_effort = parsed.reasoning_effort,
            .thinking_budget_tokens = parsed.thinking_budget_tokens,
            .cache_control_pin = parsed.cache_control_pin,
            .stream_include_usage = parsed.stream_include_usage,
        };

        if (!self.job_ring.tryPush(job)) {
            slot.worker_active.store(false, .release);
            self.releaseRequestCount(slot);
            _ = self.rejected_503_count.fetchAdd(1, .monotonic);
            slot.tx_read_pos = 0;
            slot.tx_write_pos = 0;
            self.sendErrorResponse(
                slot,
                503,
                "Service Unavailable",
                null,
                "server busy: inference job queue full",
                "server_error",
                "server_overloaded",
            );
            return;
        }

        self.wakeWorkerLoop();
        if (parsed.stream) {
            _ = self.flushTx(slot);
        }
    }

    fn drainAllSlots(self: *Server) void {
        for (self.slots) |*slot| {
            switch (slot.state) {
                .streaming_active => self.drainSlotTokens(slot),
                .waiting_worker_cancel => {
                    if (!slot.worker_active.load(.acquire)) {
                        self.closeSlotSocketAndRelease(slot, false);
                    }
                },
                .draining_response => {
                    if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
                        self.closeSlotSocketAndRelease(slot, false);
                    }
                },
                .free, .reading_request => {},
            }
        }
    }

    fn drainSlotTokens(self: *Server, slot: *ConnectionSlot) void {
        while (slot.state == .streaming_active) {
            // Preserve TX headroom so slow TCP clients apply clean backpressure
            // back to `token_ring` without ever truncating a formatted SSE frame.
            if (slot.stream and slot.txHeadroom() < STREAM_TX_HEADROOM_BYTES) {
                if (!self.flushTx(slot)) return;
                if (slot.txHeadroom() < STREAM_TX_HEADROOM_BYTES) {
                    self.setWriteInterest(slot, true);
                    return;
                }
            }

            const ev = slot.token_ring.tryPop() orelse {
                // If worker exited unexpectedly without pushing `.finished`, finish cleanly.
                if (!slot.worker_active.load(.acquire) and slot.token_ring.isEmpty()) {
                    if (slot.cancelled.load(.acquire)) {
                        self.closeSlotSocketAndRelease(slot, false);
                    } else {
                        self.finishSlotResponse(slot, .stop);
                    }
                }
                break;
            };

            if (ev.generation != slot.generation) continue;

            switch (ev.kind) {
                .prefill_done => {
                    slot.prompt_tokens = ev.token_count;
                    slot.cached_tokens = ev.cached_tokens;
                    if (slot.stream) {
                        self.ensureStreamPreamble(slot);
                    }
                },
                .token_text => {
                    if (slot.first_token_ns == 0) {
                        slot.first_token_ns = monotonicNowNs();
                    }
                    slot.completion_tokens +|= ev.token_count;
                    if (slot.think_splitter.in_think) {
                        slot.reasoning_tokens +|= ev.token_count;
                    }
                    if (slot.stream) {
                        self.ensureStreamPreamble(slot);
                    }
                    const text = ev.textSlice();
                    if (text.len > 0) {
                        self.feedSlotText(slot, text);
                        if (slot.stop_holdback.stopped) {
                            // Matched a stop sequence: stop worker immediately and emit final frame
                            // without ever leaking the stop string!
                            slot.cancelled.store(true, .release);
                            self.wakeWorkerLoop();
                            self.finishSlotResponse(slot, .stop_sequence);
                            return;
                        }
                    }
                },
                .finished => {
                    if (ev.token_count > 0) slot.prompt_tokens = ev.token_count;
                    if (slot.first_token_ns == 0) slot.first_token_ns = monotonicNowNs();
                    const final_reason: FinishReason = if (slot.stop_holdback.stopped)
                        .stop_sequence
                    else
                        ev.finish_reason;
                    self.finishSlotResponse(slot, final_reason);
                    return;
                },
                .failed => {
                    if (ev.token_count > 0) slot.prompt_tokens = ev.token_count;
                    self.finalizeStatefulResponseRecord(slot, .failed);
                    self.failSlotResponse(slot, ev.textSlice());
                    return;
                },
            }

            if (slot.stream and slot.pendingTxLen() > 0) {
                if (!self.flushTx(slot)) return;
            }
        }
    }

    fn feedSlotText(self: *Server, slot: *ConnectionSlot, text: []const u8) void {
        const HoldCtx = struct {
            server: *Server,
            slot: *ConnectionSlot,
            fn onSafe(ctx: *@This(), safe_bytes: []const u8) error{BufferTooSmall}!void {
                if (safe_bytes.len == 0) return;
                const ThinkCtx = struct {
                    server: *Server,
                    slot: *ConnectionSlot,
                    fn onSpan(tctx: *@This(), channel: StreamChannel, span: []const u8) error{BufferTooSmall}!void {
                        if (span.len == 0) return;
                        try tctx.server.dispatchChannelSpan(tctx.slot, channel, span);
                    }
                };
                var tctx = ThinkCtx{ .server = ctx.server, .slot = ctx.slot };
                try ctx.slot.think_splitter.feed(safe_bytes, &tctx, ThinkCtx.onSpan);
            }
        };
        var hctx = HoldCtx{ .server = self, .slot = slot };
        _ = slot.stop_holdback.feed(&slot.stop_config, text, &hctx, HoldCtx.onSafe) catch {
            self.closeSlotSocketAndRelease(slot, true);
        };
    }

    fn dispatchChannelSpan(
        self: *Server,
        slot: *ConnectionSlot,
        channel: StreamChannel,
        span: []const u8,
    ) error{BufferTooSmall}!void {
        if (channel == .reasoning) {
            self.appendNonStreamSpan(slot, .reasoning, span);
            if (slot.stream) {
                try self.appendStreamDelta(slot, .reasoning, span);
            }
            return;
        }
        // `.content` channel flows through the zero-alloc streaming `<tool_call>` parser!
        const ToolCtx = struct {
            server: *Server,
            slot: *ConnectionSlot,
            fn onToolEvent(ctx: *@This(), tev: ToolCallStreamEvent) error{BufferTooSmall}!void {
                try ctx.server.handleToolCallEvent(ctx.slot, tev);
            }
        };
        var tctx = ToolCtx{ .server = self, .slot = slot };
        try slot.tool_parser.feed(span, &tctx, ToolCtx.onToolEvent);
    }

    fn handleToolCallEvent(
        self: *Server,
        slot: *ConnectionSlot,
        tev: ToolCallStreamEvent,
    ) error{BufferTooSmall}!void {
        switch (tev) {
            .text => |txt| {
                if (txt.len == 0) return;
                self.appendNonStreamSpan(slot, .content, txt);
                if (slot.stream) {
                    try self.appendStreamDelta(slot, .content, txt);
                }
            },
            .tool_start => |st| {
                if (slot.stored_tool_call_count < MAX_STORED_TOOL_CALLS) {
                    const dst = &slot.stored_tool_calls[slot.stored_tool_call_count];
                    const il = @min(st.id.len, dst.id_buf.len);
                    @memcpy(dst.id_buf[0..il], st.id[0..il]);
                    dst.id_len = @intCast(il);
                    const nl = @min(st.name.len, dst.name_buf.len);
                    @memcpy(dst.name_buf[0..nl], st.name[0..nl]);
                    dst.name_len = @intCast(nl);
                    dst.args_len = 0;
                    slot.stored_tool_call_count += 1;
                }
                if (slot.stream) {
                    try self.appendToolStartStream(slot, st.index, st.id, st.name);
                }
            },
            .tool_args_delta => |ad| {
                if (ad.index < slot.stored_tool_call_count) {
                    const dst = &slot.stored_tool_calls[ad.index];
                    const avail = dst.args_buf.len - dst.args_len;
                    const take = @min(avail, ad.delta.len);
                    if (take > 0) {
                        @memcpy(dst.args_buf[dst.args_len .. dst.args_len + take], ad.delta[0..take]);
                        dst.args_len += @intCast(take);
                    }
                }
                if (slot.stream) {
                    try self.appendToolArgsDeltaStream(slot, ad.index, ad.id, ad.delta);
                }
            },
            .tool_end => |en| {
                if (slot.stream) {
                    try self.appendToolEndStream(slot, en.index, en.id, en.name);
                }
            },
        }
    }

    fn appendToolStartStream(
        self: *Server,
        slot: *ConnectionSlot,
        index: u16,
        id: []const u8,
        name: []const u8,
    ) error{BufferTooSmall}!void {
        var buf: [STREAM_TX_HEADROOM_BYTES]u8 = undefined;
        var w = FixedWriter.init(&buf);
        switch (slot.endpoint) {
            .chat_completions => {
                try w.print(
                    "data: {{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"",
                    .{ slot.request_id, slot.created_unix },
                );
                try w.writeJsonEscaped(slot.modelSlice());
                try w.print(
                    "\",\"seed_used\":{d},\"seed_source\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{",
                    .{ slot.seed_used, slot.seed_source.name() },
                );
                if (!slot.sent_first_chunk) {
                    try w.writeAll("\"role\":\"assistant\",");
                    slot.sent_first_chunk = true;
                }
                try w.print("\"tool_calls\":[{{\"index\":{d},\"id\":\"", .{index});
                try w.writeJsonEscaped(id);
                try w.writeAll("\",\"type\":\"function\",\"function\":{\"name\":\"");
                try w.writeJsonEscaped(name);
                try w.writeAll("\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
                try slot.appendTx(w.written());
            },
            .responses => {
                try self.closeResponsesOpenItems(slot, &w);
                const out_idx = slot.responses_output_index;
                slot.responses_output_index += 1;
                slot.responses_active_call_output_idx = out_idx;
                const seq = slot.responses_seq_num;
                slot.responses_seq_num += 1;
                try w.print(
                    "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"sequence_number\":{d},\"output_index\":{d},\"item\":{{\"id\":\"fc_{d}_{d}\",\"type\":\"function_call\",\"status\":\"in_progress\",\"call_id\":\"",
                    .{ seq, out_idx, slot.request_id, index },
                );
                try w.writeJsonEscaped(id);
                try w.writeAll("\",\"name\":\"");
                try w.writeJsonEscaped(name);
                try w.writeAll("\",\"arguments\":\"\"}}\n\n");
                try slot.appendTx(w.written());
            },
            .messages => {
                if (slot.anthropic_block != .none) {
                    try self.writeAnthropicBlockStop(slot, &w);
                    slot.anthropic_block_idx += 1;
                }
                slot.anthropic_block = .tool_use;
                try w.print(
                    "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":{d},\"content_block\":{{\"type\":\"tool_use\",\"id\":\"",
                    .{slot.anthropic_block_idx},
                );
                try w.writeJsonEscaped(id);
                try w.writeAll("\",\"name\":\"");
                try w.writeJsonEscaped(name);
                try w.writeAll("\",\"input\":{}}}\n\n");
                try slot.appendTx(w.written());
            },
        }
    }

    fn appendToolArgsDeltaStream(
        _: *Server,
        slot: *ConnectionSlot,
        index: u16,
        id: []const u8,
        delta: []const u8,
    ) error{BufferTooSmall}!void {
        if (delta.len == 0) return;
        _ = id;
        var buf: [STREAM_TX_HEADROOM_BYTES]u8 = undefined;
        var w = FixedWriter.init(&buf);
        switch (slot.endpoint) {
            .chat_completions => {
                try w.print(
                    "data: {{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"",
                    .{ slot.request_id, slot.created_unix },
                );
                try w.writeJsonEscaped(slot.modelSlice());
                try w.print(
                    "\",\"seed_used\":{d},\"seed_source\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":{d},\"function\":{{\"arguments\":\"",
                    .{ slot.seed_used, slot.seed_source.name(), index },
                );
                try w.writeJsonEscaped(delta);
                try w.writeAll("\"}}]},\"finish_reason\":null}]}\n\n");
                try slot.appendTx(w.written());
            },
            .responses => {
                const seq = slot.responses_seq_num;
                slot.responses_seq_num += 1;
                try w.print(
                    "event: response.function_call_arguments.delta\ndata: {{\"type\":\"response.function_call_arguments.delta\",\"sequence_number\":{d},\"item_id\":\"fc_{d}_{d}\",\"output_index\":{d},\"delta\":\"",
                    .{ seq, slot.request_id, index, slot.responses_active_call_output_idx },
                );
                try w.writeJsonEscaped(delta);
                try w.writeAll("\"}\n\n");
                try slot.appendTx(w.written());
            },
            .messages => {
                try w.print(
                    "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{d},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":\"",
                    .{slot.anthropic_block_idx},
                );
                try w.writeJsonEscaped(delta);
                try w.writeAll("\"}}\n\n");
                try slot.appendTx(w.written());
            },
        }
    }

    fn appendToolEndStream(
        _: *Server,
        slot: *ConnectionSlot,
        index: u16,
        id: []const u8,
        name: []const u8,
    ) error{BufferTooSmall}!void {
        var buf: [STREAM_TX_HEADROOM_BYTES]u8 = undefined;
        var w = FixedWriter.init(&buf);
        switch (slot.endpoint) {
            .chat_completions => return,
            .responses => {
                const args = if (index < slot.stored_tool_call_count)
                    slot.stored_tool_calls[index].argsSlice()
                else
                    "{}";
                const seq1 = slot.responses_seq_num;
                slot.responses_seq_num += 1;
                try w.print(
                    "event: response.function_call_arguments.done\ndata: {{\"type\":\"response.function_call_arguments.done\",\"sequence_number\":{d},\"item_id\":\"fc_{d}_{d}\",\"output_index\":{d},\"arguments\":\"",
                    .{ seq1, slot.request_id, index, slot.responses_active_call_output_idx },
                );
                try w.writeJsonEscaped(args);
                try w.writeAll("\"}\n\n");

                const seq2 = slot.responses_seq_num;
                slot.responses_seq_num += 1;
                try w.print(
                    "event: response.output_item.done\ndata: {{\"type\":\"response.output_item.done\",\"sequence_number\":{d},\"output_index\":{d},\"item\":{{\"id\":\"fc_{d}_{d}\",\"type\":\"function_call\",\"status\":\"completed\",\"call_id\":\"",
                    .{ seq2, slot.responses_active_call_output_idx, slot.request_id, index },
                );
                try w.writeJsonEscaped(id);
                try w.writeAll("\",\"name\":\"");
                try w.writeJsonEscaped(name);
                try w.writeAll("\",\"arguments\":\"");
                try w.writeJsonEscaped(args);
                try w.writeAll("\"}}\n\n");
                try slot.appendTx(w.written());
            },
            .messages => {
                if (slot.anthropic_block == .tool_use) {
                    try w.print(
                        "event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{d}}}\n\n",
                        .{slot.anthropic_block_idx},
                    );
                    slot.anthropic_block_idx += 1;
                    slot.anthropic_block = .none;
                    try slot.appendTx(w.written());
                }
            },
        }
    }

    fn writeAnthropicBlockStop(_: *Server, slot: *ConnectionSlot, w: *FixedWriter) error{BufferTooSmall}!void {
        if (slot.anthropic_block == .thinking) {
            // Emit Anthropic `signature_delta` right before closing a `thinking` block (`C18`)!
            try w.print(
                "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{d},\"delta\":{{\"type\":\"signature_delta\",\"signature\":\"aminfer_sig_{d}_{d}\"}}}}\n\n",
                .{ slot.anthropic_block_idx, slot.request_id, slot.anthropic_block_idx },
            );
        }
        try w.print(
            "event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{d}}}\n\n",
            .{slot.anthropic_block_idx},
        );
    }

    fn closeResponsesOpenItems(_: *Server, slot: *ConnectionSlot, w: *FixedWriter) error{BufferTooSmall}!void {
        if (slot.responses_reasoning_opened and !slot.responses_reasoning_closed) {
            slot.responses_reasoning_closed = true;
            const reasoning = slot.nonstream_reasoning_buf[0..slot.nonstream_reasoning_len];
            const s1 = slot.responses_seq_num;
            slot.responses_seq_num += 1;
            try w.print(
                "event: response.reasoning_summary_text.done\ndata: {{\"type\":\"response.reasoning_summary_text.done\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"summary_index\":0,\"text\":\"",
                .{ s1, slot.request_id, slot.responses_reasoning_output_idx },
            );
            try w.writeJsonEscaped(reasoning);
            try w.writeAll("\"}\n\n");

            const s2 = slot.responses_seq_num;
            slot.responses_seq_num += 1;
            try w.print(
                "event: response.output_item.done\ndata: {{\"type\":\"response.output_item.done\",\"sequence_number\":{d},\"output_index\":{d},\"item\":{{\"id\":\"rs_{d}\",\"type\":\"reasoning\",\"status\":\"completed\",\"summary\":[{{\"type\":\"summary_text\",\"text\":\"",
                .{ s2, slot.responses_reasoning_output_idx, slot.request_id },
            );
            try w.writeJsonEscaped(reasoning);
            try w.writeAll("\"}]}}\n\n");
        }
        if (slot.responses_text_opened and !slot.responses_text_closed) {
            slot.responses_text_closed = true;
            const content = slot.nonstream_content_buf[0..slot.nonstream_content_len];
            const s1 = slot.responses_seq_num;
            slot.responses_seq_num += 1;
            try w.print(
                "event: response.output_text.done\ndata: {{\"type\":\"response.output_text.done\",\"sequence_number\":{d},\"item_id\":\"msg_{d}\",\"output_index\":{d},\"content_index\":0,\"text\":\"",
                .{ s1, slot.request_id, slot.responses_text_output_idx },
            );
            try w.writeJsonEscaped(content);
            try w.writeAll("\"}\n\n");

            const s2 = slot.responses_seq_num;
            slot.responses_seq_num += 1;
            try w.print(
                "event: response.content_part.done\ndata: {{\"type\":\"response.content_part.done\",\"sequence_number\":{d},\"item_id\":\"msg_{d}\",\"output_index\":{d},\"content_index\":0,\"part\":{{\"type\":\"output_text\",\"text\":\"",
                .{ s2, slot.request_id, slot.responses_text_output_idx },
            );
            try w.writeJsonEscaped(content);
            try w.writeAll("\"}}\n\n");

            const s3 = slot.responses_seq_num;
            slot.responses_seq_num += 1;
            try w.print(
                "event: response.output_item.done\ndata: {{\"type\":\"response.output_item.done\",\"sequence_number\":{d},\"output_index\":{d},\"item\":{{\"id\":\"msg_{d}\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{{\"type\":\"output_text\",\"text\":\"",
                .{ s3, slot.responses_text_output_idx, slot.request_id },
            );
            try w.writeJsonEscaped(content);
            try w.writeAll("\"}]}}\n\n");
        }
    }

    fn appendNonStreamSpan(_: *Server, slot: *ConnectionSlot, channel: StreamChannel, span: []const u8) void {
        switch (channel) {
            .reasoning => {
                const avail = slot.nonstream_reasoning_buf.len - slot.nonstream_reasoning_len;
                const n = @min(avail, span.len);
                if (n > 0) {
                    @memcpy(slot.nonstream_reasoning_buf[slot.nonstream_reasoning_len .. slot.nonstream_reasoning_len + n], span[0..n]);
                    slot.nonstream_reasoning_len += n;
                }
            },
            .content => {
                const avail = slot.nonstream_content_buf.len - slot.nonstream_content_len;
                const n = @min(avail, span.len);
                if (n > 0) {
                    @memcpy(slot.nonstream_content_buf[slot.nonstream_content_len .. slot.nonstream_content_len + n], span[0..n]);
                    slot.nonstream_content_len += n;
                }
            },
        }
    }

    fn ensureStreamPreamble(self: *Server, slot: *ConnectionSlot) void {
        if (slot.preamble_sent) return;
        slot.preamble_sent = true;

        var buf: [1536]u8 = undefined;
        var w = FixedWriter.init(&buf);
        switch (slot.endpoint) {
            .chat_completions => return,
            .responses => {
                const s0 = slot.responses_seq_num;
                slot.responses_seq_num += 1;
                w.print(
                    "event: response.created\ndata: {{\"type\":\"response.created\",\"sequence_number\":{d},\"response\":{{\"id\":\"resp_{d}\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"",
                    .{ s0, slot.request_id },
                ) catch return;
                w.writeJsonEscaped(slot.modelSlice()) catch return;
                w.print("\",\"seed_used\":{d},\"seed_source\":\"{s}\"}}}}\n\n", .{
                    slot.seed_used,
                    slot.seed_source.name(),
                }) catch return;

                const s1 = slot.responses_seq_num;
                slot.responses_seq_num += 1;
                w.print(
                    "event: response.in_progress\ndata: {{\"type\":\"response.in_progress\",\"sequence_number\":{d},\"response\":{{\"id\":\"resp_{d}\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"",
                    .{ s1, slot.request_id },
                ) catch return;
                w.writeJsonEscaped(slot.modelSlice()) catch return;
                w.writeAll("\"}}\n\n") catch return;
            },
            .messages => {
                w.print(
                    "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_{d}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"",
                    .{slot.request_id},
                ) catch return;
                w.writeJsonEscaped(slot.modelSlice()) catch return;
                w.print(
                    "\",\"seed_used\":{d},\"seed_source\":\"{s}\",\"content\":[],\"stop_reason\":null,\"usage\":{{\"input_tokens\":{d},\"cache_read_input_tokens\":{d},\"output_tokens\":0}}}}}}\n\n",
                    .{ slot.seed_used, slot.seed_source.name(), slot.prompt_tokens, slot.cached_tokens },
                ) catch return;
            },
        }
        slot.appendTx(w.written()) catch {
            self.closeSlotSocketAndRelease(slot, true);
        };
    }

    fn appendStreamDelta(
        self: *Server,
        slot: *ConnectionSlot,
        channel: StreamChannel,
        span: []const u8,
    ) error{BufferTooSmall}!void {
        var buf: [STREAM_TX_HEADROOM_BYTES]u8 = undefined;
        var w = FixedWriter.init(&buf);

        switch (slot.endpoint) {
            .chat_completions => {
                try w.print(
                    "data: {{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"",
                    .{ slot.request_id, slot.created_unix },
                );
                try w.writeJsonEscaped(slot.modelSlice());
                try w.print(
                    "\",\"seed_used\":{d},\"seed_source\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{",
                    .{ slot.seed_used, slot.seed_source.name() },
                );
                if (!slot.sent_first_chunk) {
                    try w.writeAll("\"role\":\"assistant\",");
                    slot.sent_first_chunk = true;
                }
                switch (channel) {
                    .reasoning => try w.writeAll("\"reasoning_content\":\""),
                    .content => try w.writeAll("\"content\":\""),
                }
                try w.writeJsonEscaped(span);
                try w.writeAll("\"},\"finish_reason\":null}]}\n\n");
                try slot.appendTx(w.written());
            },
            .responses => {
                switch (channel) {
                    .reasoning => {
                        if (!slot.responses_reasoning_opened) {
                            slot.responses_reasoning_opened = true;
                            slot.responses_reasoning_output_idx = slot.responses_output_index;
                            slot.responses_output_index += 1;
                            const s0 = slot.responses_seq_num;
                            slot.responses_seq_num += 1;
                            try w.print(
                                "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"sequence_number\":{d},\"output_index\":{d},\"item\":{{\"id\":\"rs_{d}\",\"type\":\"reasoning\",\"status\":\"in_progress\",\"summary\":[]}}}}\n\n",
                                .{ s0, slot.responses_reasoning_output_idx, slot.request_id },
                            );
                        }
                        const sd = slot.responses_seq_num;
                        slot.responses_seq_num += 1;
                        try w.print(
                            "event: response.reasoning_summary_text.delta\ndata: {{\"type\":\"response.reasoning_summary_text.delta\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"summary_index\":0,\"delta\":\"",
                            .{ sd, slot.request_id, slot.responses_reasoning_output_idx },
                        );
                        try w.writeJsonEscaped(span);
                        try w.writeAll("\"}\n\n");
                    },
                    .content => {
                        if (slot.responses_reasoning_opened and !slot.responses_reasoning_closed) {
                            try self.closeResponsesOpenItems(slot, &w);
                        }
                        if (!slot.responses_text_opened) {
                            slot.responses_text_opened = true;
                            slot.responses_text_output_idx = slot.responses_output_index;
                            slot.responses_output_index += 1;
                            const s0 = slot.responses_seq_num;
                            slot.responses_seq_num += 1;
                            try w.print(
                                "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"sequence_number\":{d},\"output_index\":{d},\"item\":{{\"id\":\"msg_{d}\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}}}\n\n",
                                .{ s0, slot.responses_text_output_idx, slot.request_id },
                            );
                            const s1 = slot.responses_seq_num;
                            slot.responses_seq_num += 1;
                            try w.print(
                                "event: response.content_part.added\ndata: {{\"type\":\"response.content_part.added\",\"sequence_number\":{d},\"item_id\":\"msg_{d}\",\"output_index\":{d},\"content_index\":0,\"part\":{{\"type\":\"output_text\",\"text\":\"\"}}}}\n\n",
                                .{ s1, slot.request_id, slot.responses_text_output_idx },
                            );
                        }
                        const sd = slot.responses_seq_num;
                        slot.responses_seq_num += 1;
                        try w.print(
                            "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"sequence_number\":{d},\"item_id\":\"msg_{d}\",\"output_index\":{d},\"content_index\":0,\"delta\":\"",
                            .{ sd, slot.request_id, slot.responses_text_output_idx },
                        );
                        try w.writeJsonEscaped(span);
                        try w.writeAll("\"}\n\n");
                    },
                }
                try slot.appendTx(w.written());
            },
            .messages => {
                const desired: AnthropicBlockState = switch (channel) {
                    .reasoning => .thinking,
                    .content => .text,
                };
                if (slot.anthropic_block != desired) {
                    if (slot.anthropic_block != .none) {
                        try self.writeAnthropicBlockStop(slot, &w);
                        slot.anthropic_block_idx += 1;
                    }
                    slot.anthropic_block = desired;
                    switch (desired) {
                        .thinking => try w.print(
                            "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":{d},\"content_block\":{{\"type\":\"thinking\",\"thinking\":\"\"}}}}\n\n",
                            .{slot.anthropic_block_idx},
                        ),
                        .text => try w.print(
                            "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":{d},\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n",
                            .{slot.anthropic_block_idx},
                        ),
                        .none, .tool_use => unreachable,
                    }
                }
                switch (desired) {
                    .thinking => {
                        try w.print(
                            "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{d},\"delta\":{{\"type\":\"thinking_delta\",\"thinking\":\"",
                            .{slot.anthropic_block_idx},
                        );
                        try w.writeJsonEscaped(span);
                        try w.writeAll("\"}}\n\n");
                    },
                    .text => {
                        try w.print(
                            "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{d},\"delta\":{{\"type\":\"text_delta\",\"text\":\"",
                            .{slot.anthropic_block_idx},
                        );
                        try w.writeJsonEscaped(span);
                        try w.writeAll("\"}}\n\n");
                    },
                    .none, .tool_use => unreachable,
                }
                try slot.appendTx(w.written());
            },
        }
    }

    const Timings = struct {
        ttft_ms: f64,
        tpot_ms: f64,
        tok_per_s: f64,
    };

    fn computeTimings(slot: *const ConnectionSlot, end_ns: u64) Timings {
        const first_ns = if (slot.first_token_ns > 0) slot.first_token_ns else end_ns;
        const ttft_ns = first_ns -| slot.admit_ns;
        const ttft_ms = @as(f64, @floatFromInt(ttft_ns)) / 1_000_000.0;

        if (slot.completion_tokens > 1) {
            const decode_ns = end_ns -| first_ns;
            if (decode_ns > 0) {
                const denom = @as(f64, @floatFromInt(slot.completion_tokens - 1));
                const decode_s = @as(f64, @floatFromInt(decode_ns)) / 1_000_000_000.0;
                return .{
                    .ttft_ms = ttft_ms,
                    .tpot_ms = (decode_s * 1000.0) / denom,
                    .tok_per_s = denom / decode_s,
                };
            }
        }
        if (slot.completion_tokens >= 1) {
            const total_ns = end_ns -| slot.admit_ns;
            if (total_ns > 0) {
                const total_s = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
                const cnt = @as(f64, @floatFromInt(slot.completion_tokens));
                return .{
                    .ttft_ms = ttft_ms,
                    .tpot_ms = (total_s * 1000.0) / cnt,
                    .tok_per_s = cnt / total_s,
                };
            }
        }
        return .{ .ttft_ms = ttft_ms, .tpot_ms = 0.0, .tok_per_s = 0.0 };
    }

    fn finishSlotResponse(self: *Server, slot: *ConnectionSlot, raw_reason: FinishReason) void {
        // 1. Flush any un-matched partial stop-sequence prefix bytes (only when NOT stopped by a match!).
        {
            const HoldFlushCtx = struct {
                server: *Server,
                slot: *ConnectionSlot,
                fn onSafe(ctx: *@This(), safe_bytes: []const u8) error{BufferTooSmall}!void {
                    if (safe_bytes.len == 0) return;
                    const ThinkCtx = struct {
                        server: *Server,
                        slot: *ConnectionSlot,
                        fn onSpan(tctx: *@This(), channel: StreamChannel, span: []const u8) error{BufferTooSmall}!void {
                            if (span.len == 0) return;
                            try tctx.server.dispatchChannelSpan(tctx.slot, channel, span);
                        }
                    };
                    var tctx = ThinkCtx{ .server = ctx.server, .slot = ctx.slot };
                    try ctx.slot.think_splitter.feed(safe_bytes, &tctx, ThinkCtx.onSpan);
                }
            };
            var hfctx = HoldFlushCtx{ .server = self, .slot = slot };
            slot.stop_holdback.flush(&hfctx, HoldFlushCtx.onSafe) catch {
                self.closeSlotSocketAndRelease(slot, true);
                return;
            };
        }

        // 2. Flush any buffered partial `<think>` / `</think>` bytes.
        {
            const FlushCtx = struct {
                server: *Server,
                slot: *ConnectionSlot,
                fn onSpan(ctx: *@This(), channel: StreamChannel, span: []const u8) error{BufferTooSmall}!void {
                    if (span.len == 0) return;
                    try ctx.server.dispatchChannelSpan(ctx.slot, channel, span);
                }
            };
            var fctx = FlushCtx{ .server = self, .slot = slot };
            slot.think_splitter.flush(&fctx, FlushCtx.onSpan) catch {
                self.closeSlotSocketAndRelease(slot, true);
                return;
            };
        }

        // 3. Flush any buffered partial `<tool_call>` bytes.
        {
            const ToolFlushCtx = struct {
                server: *Server,
                slot: *ConnectionSlot,
                fn onToolEvent(ctx: *@This(), tev: ToolCallStreamEvent) error{BufferTooSmall}!void {
                    try ctx.server.handleToolCallEvent(ctx.slot, tev);
                }
            };
            var tfctx = ToolFlushCtx{ .server = self, .slot = slot };
            slot.tool_parser.flush(&tfctx, ToolFlushCtx.onToolEvent) catch {
                self.closeSlotSocketAndRelease(slot, true);
                return;
            };
        }

        // Upgrade finish_reason to `.tool_calls` (`"tool_calls"` / `"tool_use"`) when tool calls were emitted!
        const reason: FinishReason = if (slot.tool_parser.hasToolCalls() and (raw_reason == .stop or raw_reason == .stop_sequence))
            .tool_calls
        else if (slot.stop_holdback.stopped)
            .stop_sequence
        else
            raw_reason;

        const end_ns = monotonicNowNs();
        const timings = computeTimings(slot, end_ns);
        if (timings.tok_per_s > 0.0 and std.math.isFinite(timings.tok_per_s)) {
            self.last_tok_per_s_x100.store(@intFromFloat(@min(timings.tok_per_s * 100.0, 1e12)), .release);
        }
        const total_tokens: u32 = slot.prompt_tokens +| slot.completion_tokens;
        const rec_status: ResponseRecordStatus = if (reason == .cancelled) .cancelled else .completed;
        self.finalizeStatefulResponseRecord(slot, rec_status);

        if (slot.stream) {
            self.ensureStreamPreamble(slot);
            var buf: [4096]u8 = undefined;
            var w = FixedWriter.init(&buf);
            switch (slot.endpoint) {
                .chat_completions => {
                    w.print(
                        "data: {{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"",
                        .{ slot.request_id, slot.created_unix },
                    ) catch {
                        self.closeSlotSocketAndRelease(slot, true);
                        return;
                    };
                    w.writeJsonEscaped(slot.modelSlice()) catch {
                        self.closeSlotSocketAndRelease(slot, true);
                        return;
                    };
                    w.print(
                        "\",\"seed_used\":{d},\"seed_source\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{s}\"}}]," ++
                            "\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}},\"completion_tokens_details\":{{\"reasoning_tokens\":{d}}}}}," ++
                            "\"aminfer_timings\":{{\"ttft_ms\":{d:.3},\"tpot_ms\":{d:.3},\"tok_per_s\":{d:.2}}}}}\n\n",
                        .{
                            slot.seed_used,
                            slot.seed_source.name(),
                            reason.openAiName(),
                            slot.prompt_tokens,
                            slot.completion_tokens,
                            total_tokens,
                            slot.cached_tokens,
                            slot.reasoning_tokens,
                            timings.ttft_ms,
                            timings.tpot_ms,
                            timings.tok_per_s,
                        },
                    ) catch {
                        self.closeSlotSocketAndRelease(slot, true);
                        return;
                    };
                    if (slot.stream_include_usage) {
                        // OpenAI `stream_options.include_usage`: emit dedicated final usage chunk (`choices: []`).
                        w.print(
                            "data: {{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"",
                            .{ slot.request_id, slot.created_unix },
                        ) catch {
                            self.closeSlotSocketAndRelease(slot, true);
                            return;
                        };
                        w.writeJsonEscaped(slot.modelSlice()) catch {
                            self.closeSlotSocketAndRelease(slot, true);
                            return;
                        };
                        w.print(
                            "\",\"choices\":[],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}},\"completion_tokens_details\":{{\"reasoning_tokens\":{d}}}}}}}\n\n",
                            .{
                                slot.prompt_tokens,
                                slot.completion_tokens,
                                total_tokens,
                                slot.cached_tokens,
                                slot.reasoning_tokens,
                            },
                        ) catch {
                            self.closeSlotSocketAndRelease(slot, true);
                            return;
                        };
                    }
                    w.writeAll("data: [DONE]\n\n") catch {
                        self.closeSlotSocketAndRelease(slot, true);
                        return;
                    };
                },
                .responses => {
                    self.closeResponsesOpenItems(slot, &w) catch {
                        self.closeSlotSocketAndRelease(slot, true);
                        return;
                    };
                    const sc = slot.responses_seq_num;
                    slot.responses_seq_num += 1;
                    w.print(
                        "event: response.completed\ndata: {{\"type\":\"response.completed\",\"sequence_number\":{d},\"response\":{{\"id\":\"resp_{d}\",\"object\":\"response\",\"status\":\"{s}\",\"model\":\"",
                        .{ sc, slot.request_id, rec_status.name() },
                    ) catch {
                        self.closeSlotSocketAndRelease(slot, true);
                        return;
                    };
                    w.writeJsonEscaped(slot.modelSlice()) catch {
                        self.closeSlotSocketAndRelease(slot, true);
                        return;
                    };
                    w.print(
                        "\",\"seed_used\":{d},\"seed_source\":\"{s}\"," ++
                            "\"usage\":{{\"input_tokens\":{d},\"input_tokens_details\":{{\"cached_tokens\":{d}}},\"output_tokens\":{d},\"output_tokens_details\":{{\"reasoning_tokens\":{d}}},\"total_tokens\":{d}}}," ++
                            "\"aminfer_timings\":{{\"ttft_ms\":{d:.3},\"tpot_ms\":{d:.3},\"tok_per_s\":{d:.2}}}}}}}\n\n",
                        .{
                            slot.seed_used,
                            slot.seed_source.name(),
                            slot.prompt_tokens,
                            slot.cached_tokens,
                            slot.completion_tokens,
                            slot.reasoning_tokens,
                            total_tokens,
                            timings.ttft_ms,
                            timings.tpot_ms,
                            timings.tok_per_s,
                        },
                    ) catch {
                        self.closeSlotSocketAndRelease(slot, true);
                        return;
                    };
                },
                .messages => {
                    if (slot.anthropic_block != .none) {
                        self.writeAnthropicBlockStop(slot, &w) catch {
                            self.closeSlotSocketAndRelease(slot, true);
                            return;
                        };
                        slot.anthropic_block = .none;
                    }
                    w.print(
                        "event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"{s}\",\"stop_sequence\":",
                        .{reason.anthropicName()},
                    ) catch {
                        self.closeSlotSocketAndRelease(slot, true);
                        return;
                    };
                    if (slot.stop_holdback.matchedSequence(&slot.stop_config)) |mseq| {
                        w.writeByte('"') catch {
                            self.closeSlotSocketAndRelease(slot, true);
                            return;
                        };
                        w.writeJsonEscaped(mseq) catch {
                            self.closeSlotSocketAndRelease(slot, true);
                            return;
                        };
                        w.writeByte('"') catch {
                            self.closeSlotSocketAndRelease(slot, true);
                            return;
                        };
                    } else {
                        w.writeAll("null") catch {
                            self.closeSlotSocketAndRelease(slot, true);
                            return;
                        };
                    }
                    w.print(
                        "}},\"usage\":{{\"input_tokens\":{d},\"cache_read_input_tokens\":{d},\"output_tokens\":{d}}},\"aminfer_timings\":{{\"ttft_ms\":{d:.3},\"tpot_ms\":{d:.3},\"tok_per_s\":{d:.2}}}}}\n\n" ++
                            "event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
                        .{
                            slot.prompt_tokens,
                            slot.cached_tokens,
                            slot.completion_tokens,
                            timings.ttft_ms,
                            timings.tpot_ms,
                            timings.tok_per_s,
                        },
                    ) catch {
                        self.closeSlotSocketAndRelease(slot, true);
                        return;
                    };
                },
            }

            slot.appendTx(w.written()) catch {
                self.closeSlotSocketAndRelease(slot, true);
                return;
            };

            // Request accounting transitions to `.draining_response` once worker finishes.
            self.releaseRequestCount(slot);
            slot.state = .draining_response;
            if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
                self.closeSlotSocketAndRelease(slot, false);
            }
            return;
        }

        // Non-streaming JSON response: format directly into `slot.tx_buf` with a
        // reserved header prefix so zero extra scratch buffers are required.
        self.releaseRequestCount(slot);
        self.writeNonStreamJsonResponse(slot, reason, timings, total_tokens);
    }

    fn writeNonStreamJsonResponse(
        self: *Server,
        slot: *ConnectionSlot,
        reason: FinishReason,
        timings: Timings,
        total_tokens: u32,
    ) void {
        const HEADER_RESERVE: usize = 512;
        if (slot.tx_buf.len <= HEADER_RESERVE) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        var bw = FixedWriter.init(slot.tx_buf[HEADER_RESERVE..]);
        const reasoning = slot.nonstream_reasoning_buf[0..slot.nonstream_reasoning_len];
        const content = slot.nonstream_content_buf[0..slot.nonstream_content_len];
        const has_tools = slot.stored_tool_call_count > 0;

        const ok: bool = blk: {
            switch (slot.endpoint) {
                .chat_completions => {
                    bw.print(
                        "{{\"id\":\"chatcmpl-{d}\",\"object\":\"chat.completion\",\"created\":{d},\"model\":\"",
                        .{ slot.request_id, slot.created_unix },
                    ) catch break :blk false;
                    bw.writeJsonEscaped(slot.modelSlice()) catch break :blk false;
                    bw.print(
                        "\",\"seed_used\":{d},\"seed_source\":\"{s}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":",
                        .{ slot.seed_used, slot.seed_source.name() },
                    ) catch break :blk false;
                    if (has_tools and content.len == 0) {
                        bw.writeAll("null") catch break :blk false;
                    } else {
                        bw.writeByte('"') catch break :blk false;
                        bw.writeJsonEscaped(content) catch break :blk false;
                        bw.writeByte('"') catch break :blk false;
                    }
                    if (reasoning.len > 0) {
                        bw.writeAll(",\"reasoning_content\":\"") catch break :blk false;
                        bw.writeJsonEscaped(reasoning) catch break :blk false;
                        bw.writeByte('"') catch break :blk false;
                    } else {
                        bw.writeAll(",\"reasoning_content\":null") catch break :blk false;
                    }
                    if (has_tools) {
                        bw.writeAll(",\"tool_calls\":[") catch break :blk false;
                        for (slot.stored_tool_calls[0..slot.stored_tool_call_count], 0..) |*tc, idx| {
                            if (idx > 0) bw.writeByte(',') catch break :blk false;
                            bw.writeAll("{\"id\":\"") catch break :blk false;
                            bw.writeJsonEscaped(tc.idSlice()) catch break :blk false;
                            bw.writeAll("\",\"type\":\"function\",\"function\":{\"name\":\"") catch break :blk false;
                            bw.writeJsonEscaped(tc.nameSlice()) catch break :blk false;
                            bw.writeAll("\",\"arguments\":\"") catch break :blk false;
                            bw.writeJsonEscaped(if (tc.args_len > 0) tc.argsSlice() else "{}") catch break :blk false;
                            bw.writeAll("\"}}") catch break :blk false;
                        }
                        bw.writeByte(']') catch break :blk false;
                    }
                    bw.print(
                        "}},\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}},\"completion_tokens_details\":{{\"reasoning_tokens\":{d}}}}}," ++
                            "\"aminfer_timings\":{{\"ttft_ms\":{d:.3},\"tpot_ms\":{d:.3},\"tok_per_s\":{d:.2}}}}}",
                        .{
                            reason.openAiName(),
                            slot.prompt_tokens,
                            slot.completion_tokens,
                            total_tokens,
                            slot.cached_tokens,
                            slot.reasoning_tokens,
                            timings.ttft_ms,
                            timings.tpot_ms,
                            timings.tok_per_s,
                        },
                    ) catch break :blk false;
                },
                .responses => {
                    bw.print(
                        "{{\"id\":\"resp_{d}\",\"object\":\"response\",\"created_at\":{d},\"status\":\"completed\",\"model\":\"",
                        .{ slot.request_id, slot.created_unix },
                    ) catch break :blk false;
                    bw.writeJsonEscaped(slot.modelSlice()) catch break :blk false;
                    bw.print("\",\"seed_used\":{d},\"seed_source\":\"{s}\"", .{
                        slot.seed_used,
                        slot.seed_source.name(),
                    }) catch break :blk false;
                    if (slot.previous_response_id_len > 0) {
                        bw.writeAll(",\"previous_response_id\":\"") catch break :blk false;
                        bw.writeJsonEscaped(slot.previousResponseIdSlice()) catch break :blk false;
                        bw.writeByte('"') catch break :blk false;
                    }
                    bw.writeAll(",\"output\":[") catch break :blk false;
                    var need_comma = false;
                    if (reasoning.len > 0) {
                        bw.print(
                            "{{\"id\":\"rs_{d}\",\"type\":\"reasoning\",\"summary\":[{{\"type\":\"summary_text\",\"text\":\"",
                            .{slot.request_id},
                        ) catch break :blk false;
                        bw.writeJsonEscaped(reasoning) catch break :blk false;
                        bw.writeAll("\"}]}") catch break :blk false;
                        need_comma = true;
                    }
                    if (content.len > 0 or !has_tools) {
                        if (need_comma) bw.writeByte(',') catch break :blk false;
                        bw.print(
                            "{{\"id\":\"msg_{d}\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{{\"type\":\"output_text\",\"text\":\"",
                            .{slot.request_id},
                        ) catch break :blk false;
                        bw.writeJsonEscaped(content) catch break :blk false;
                        bw.writeAll("\"}]}") catch break :blk false;
                        need_comma = true;
                    }
                    for (slot.stored_tool_calls[0..slot.stored_tool_call_count], 0..) |*tc, idx| {
                        if (need_comma) bw.writeByte(',') catch break :blk false;
                        bw.print("{{\"id\":\"fc_{d}_{d}\",\"type\":\"function_call\",\"status\":\"completed\",\"call_id\":\"", .{
                            slot.request_id,
                            idx,
                        }) catch break :blk false;
                        bw.writeJsonEscaped(tc.idSlice()) catch break :blk false;
                        bw.writeAll("\",\"name\":\"") catch break :blk false;
                        bw.writeJsonEscaped(tc.nameSlice()) catch break :blk false;
                        bw.writeAll("\",\"arguments\":\"") catch break :blk false;
                        bw.writeJsonEscaped(if (tc.args_len > 0) tc.argsSlice() else "{}") catch break :blk false;
                        bw.writeAll("\"}") catch break :blk false;
                        need_comma = true;
                    }
                    bw.writeAll("],\"output_text\":\"") catch break :blk false;
                    bw.writeJsonEscaped(content) catch break :blk false;
                    bw.print(
                        "\",\"usage\":{{\"input_tokens\":{d},\"input_tokens_details\":{{\"cached_tokens\":{d}}},\"output_tokens\":{d},\"output_tokens_details\":{{\"reasoning_tokens\":{d}}},\"total_tokens\":{d}}}," ++
                            "\"aminfer_timings\":{{\"ttft_ms\":{d:.3},\"tpot_ms\":{d:.3},\"tok_per_s\":{d:.2}}}}}",
                        .{
                            slot.prompt_tokens,
                            slot.cached_tokens,
                            slot.completion_tokens,
                            slot.reasoning_tokens,
                            total_tokens,
                            timings.ttft_ms,
                            timings.tpot_ms,
                            timings.tok_per_s,
                        },
                    ) catch break :blk false;
                },
                .messages => {
                    bw.print(
                        "{{\"id\":\"msg_{d}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"",
                        .{slot.request_id},
                    ) catch break :blk false;
                    bw.writeJsonEscaped(slot.modelSlice()) catch break :blk false;
                    bw.print("\",\"seed_used\":{d},\"seed_source\":\"{s}\",\"content\":[", .{
                        slot.seed_used,
                        slot.seed_source.name(),
                    }) catch break :blk false;
                    var need_comma = false;
                    if (reasoning.len > 0) {
                        bw.writeAll("{\"type\":\"thinking\",\"thinking\":\"") catch break :blk false;
                        bw.writeJsonEscaped(reasoning) catch break :blk false;
                        bw.writeAll("\"}") catch break :blk false;
                        need_comma = true;
                    }
                    if (content.len > 0 or !has_tools) {
                        if (need_comma) bw.writeByte(',') catch break :blk false;
                        bw.writeAll("{\"type\":\"text\",\"text\":\"") catch break :blk false;
                        bw.writeJsonEscaped(content) catch break :blk false;
                        bw.writeAll("\"}") catch break :blk false;
                        need_comma = true;
                    }
                    for (slot.stored_tool_calls[0..slot.stored_tool_call_count]) |*tc| {
                        if (need_comma) bw.writeByte(',') catch break :blk false;
                        bw.writeAll("{\"type\":\"tool_use\",\"id\":\"") catch break :blk false;
                        bw.writeJsonEscaped(tc.idSlice()) catch break :blk false;
                        bw.writeAll("\",\"name\":\"") catch break :blk false;
                        bw.writeJsonEscaped(tc.nameSlice()) catch break :blk false;
                        bw.writeAll("\",\"input\":") catch break :blk false;
                        bw.writeAll(if (tc.args_len > 0) tc.argsSlice() else "{}") catch break :blk false;
                        bw.writeByte('}') catch break :blk false;
                        need_comma = true;
                    }
                    bw.print(
                        "],\"stop_reason\":\"{s}\",\"stop_sequence\":",
                        .{reason.anthropicName()},
                    ) catch break :blk false;
                    if (slot.stop_holdback.matchedSequence(&slot.stop_config)) |mseq| {
                        bw.writeByte('"') catch break :blk false;
                        bw.writeJsonEscaped(mseq) catch break :blk false;
                        bw.writeByte('"') catch break :blk false;
                    } else {
                        bw.writeAll("null") catch break :blk false;
                    }
                    bw.print(
                        ",\"usage\":{{\"input_tokens\":{d},\"cache_read_input_tokens\":{d},\"output_tokens\":{d}}}," ++
                            "\"aminfer_timings\":{{\"ttft_ms\":{d:.3},\"tpot_ms\":{d:.3},\"tok_per_s\":{d:.2}}}}}",
                        .{
                            slot.prompt_tokens,
                            slot.cached_tokens,
                            slot.completion_tokens,
                            timings.ttft_ms,
                            timings.tpot_ms,
                            timings.tok_per_s,
                        },
                    ) catch break :blk false;
                },
            }
            break :blk true;
        };

        if (!ok) {
            self.sendErrorResponse(slot, 500, "Internal Server Error", null, "response exceeded preallocated tx buffer", "server_error", "tx_overflow");
            return;
        }

        const body_len = bw.pos;
        var hdr_buf: [HEADER_RESERVE]u8 = undefined;
        var hw = FixedWriter.init(&hdr_buf);
        hw.print(
            "HTTP/1.1 200 OK\r\n" ++
                "Content-Type: application/json\r\n" ++
                "Content-Length: {d}\r\n" ++
                "Connection: close\r\n" ++
                "X-Aminfer-Seed-Used: {d}\r\n" ++
                "X-Aminfer-Seed-Source: {s}\r\n" ++
                "\r\n",
            .{ body_len, slot.seed_used, slot.seed_source.name() },
        ) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };

        const hdr_start = HEADER_RESERVE - hw.pos;
        @memcpy(slot.tx_buf[hdr_start..HEADER_RESERVE], hw.written());
        slot.tx_read_pos = hdr_start;
        slot.tx_write_pos = HEADER_RESERVE + body_len;
        slot.state = .draining_response;
        if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
            self.closeSlotSocketAndRelease(slot, false);
        }
    }

    fn failSlotResponse(self: *Server, slot: *ConnectionSlot, message: []const u8) void {
        const msg = if (message.len > 0) message else "inference engine error";
        if (!slot.stream) {
            self.releaseRequestCount(slot);
            self.sendErrorResponse(slot, 500, "Internal Server Error", null, msg, "server_error", "inference_error");
            return;
        }
        // SSE headers already sent: emit terminal error event and close cleanly.
        var buf: [512]u8 = undefined;
        var w = FixedWriter.init(&buf);
        w.writeAll("event: error\ndata: {\"error\":{\"message\":\"") catch {
            self.closeSlotSocketAndRelease(slot, true);
            return;
        };
        w.writeJsonEscaped(msg) catch {
            self.closeSlotSocketAndRelease(slot, true);
            return;
        };
        w.writeAll("\",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n") catch {
            self.closeSlotSocketAndRelease(slot, true);
            return;
        };
        slot.appendTx(w.written()) catch {
            self.closeSlotSocketAndRelease(slot, true);
            return;
        };
        self.releaseRequestCount(slot);
        slot.state = .draining_response;
        if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
            self.closeSlotSocketAndRelease(slot, false);
        }
    }
};

// ===========================================================================
// Comprehensive Unit, Loopback, Concurrent Load & Zero-Allocation Tests
// ===========================================================================

const MallocStatistics = extern struct {
    blocks_in_use: c_uint,
    size_in_use: usize,
    max_size_in_use: usize,
    size_allocated: usize,
};

extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *MallocStatistics) void;

const TestClientError = error{
    SocketFailed,
    ConnectFailed,
    SendFailed,
    RecvFailed,
    BufferTooSmall,
};

/// Zero-heap-allocation blocking loopback TCP helper for unit/load tests.
fn testConnectLoopback(port: u16) TestClientError!std.c.fd_t {
    const fd = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    errdefer _ = std.c.close(fd);

    var one: c_int = 1;
    _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.NOSIGPIPE, &one, @sizeOf(c_int));

    var addr = std.c.sockaddr.in{
        .port = std.mem.nativeToBig(u16, port),
        .addr = @bitCast([4]u8{ 127, 0, 0, 1 }),
    };
    if (std.c.connect(fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) != 0) {
        return error.ConnectFailed;
    }
    return fd;
}

fn testSendAll(fd: std.c.fd_t, data: []const u8) TestClientError!void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.send(fd, data.ptr + sent, data.len - sent, 0);
        if (n > 0) {
            sent += @intCast(n);
        } else if (n < 0) {
            if (std.c.errno(n) == .INTR) continue;
            return error.SendFailed;
        } else {
            return error.SendFailed;
        }
    }
}

fn testRecvAll(fd: std.c.fd_t, out_buf: []u8) TestClientError![]const u8 {
    var total: usize = 0;
    while (true) {
        if (total >= out_buf.len) return error.BufferTooSmall;
        const n = std.c.recv(fd, out_buf.ptr + total, out_buf.len - total, 0);
        if (n > 0) {
            total += @intCast(n);
        } else if (n == 0) {
            return out_buf[0..total];
        } else {
            if (std.c.errno(n) == .INTR) continue;
            return error.RecvFailed;
        }
    }
}

fn testHttpExchange(port: u16, raw_request: []const u8, resp_buf: []u8) TestClientError![]const u8 {
    const fd = try testConnectLoopback(port);
    defer _ = std.c.close(fd);
    try testSendAll(fd, raw_request);
    return testRecvAll(fd, resp_buf);
}

fn testHttpPostJson(port: u16, path: []const u8, json_body: []const u8, req_buf: []u8, resp_buf: []u8) TestClientError![]const u8 {
    var w = FixedWriter.init(req_buf);
    w.print(
        "POST {s} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
        .{ path, json_body.len, json_body },
    ) catch return error.BufferTooSmall;
    return testHttpExchange(port, w.written(), resp_buf);
}

fn extractHeaderValue(resp: []const u8, header_name: []const u8) ?[]const u8 {
    const sep = std.mem.indexOf(u8, resp, "\r\n\r\n") orelse return null;
    var lines = std.mem.splitSequence(u8, resp[0..sep], "\r\n");
    _ = lines.next();
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = std.mem.trim(u8, line[0..colon], " \t");
        if (std.ascii.eqlIgnoreCase(name, header_name)) {
            return std.mem.trim(u8, line[colon + 1 ..], " \t");
        }
    }
    return null;
}

fn extractHttpBody(resp: []const u8) ?[]const u8 {
    const sep = std.mem.indexOf(u8, resp, "\r\n\r\n") orelse return null;
    return resp[sep + 4 ..];
}

fn expectValidJson(body: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, body, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value == .object);
}

fn expectValidSseJsonFrames(sse: []const u8) !usize {
    var count: usize = 0;
    var lines = std.mem.splitScalar(u8, sse, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const payload = line["data: ".len..];
        if (std.mem.eql(u8, payload, "[DONE]")) continue;
        try expectValidJson(payload);
        count += 1;
    }
    try std.testing.expect(count > 0);
    return count;
}

test "ThinkSplitter separates fragmented <think>...</think> tags and handles false prefixes" {
    const Accum = struct {
        reasoning: [256]u8 = undefined,
        r_len: usize = 0,
        content: [256]u8 = undefined,
        c_len: usize = 0,

        fn emit(self: *@This(), ch: StreamChannel, s: []const u8) error{BufferTooSmall}!void {
            switch (ch) {
                .reasoning => {
                    if (self.r_len + s.len > self.reasoning.len) return error.BufferTooSmall;
                    @memcpy(self.reasoning[self.r_len .. self.r_len + s.len], s);
                    self.r_len += s.len;
                },
                .content => {
                    if (self.c_len + s.len > self.content.len) return error.BufferTooSmall;
                    @memcpy(self.content[self.c_len .. self.c_len + s.len], s);
                    self.c_len += s.len;
                },
            }
        }
    };

    // Case 1: Tags split across arbitrary 1..3 byte chunks, plus false prefixes & `<<think>`.
    {
        var splitter = ThinkSplitter{};
        var acc = Accum{};
        const chunks = [_][]const u8{
            "<thi",  "ng ",     "<",    "<t", "hin", "k>",
            "a < b", " and ",   "</th", "x ", "<",   "</th",
            "ink>",  "Answer:", " <th",
        };
        for (chunks) |c| {
            try splitter.feed(c, &acc, Accum.emit);
        }
        try splitter.flush(&acc, Accum.emit);

        try std.testing.expectEqualStrings("a < b and </thx <", acc.reasoning[0..acc.r_len]);
        try std.testing.expectEqualStrings("<thing <Answer: <th", acc.content[0..acc.c_len]);
    }

    // Case 2: Byte-by-byte feeding produces identical output.
    {
        const full = "<think>step 1\nstep 2</think>Final 42!";
        var splitter = ThinkSplitter{};
        var acc = Accum{};
        for (full) |b| {
            const single = [1]u8{b};
            try splitter.feed(&single, &acc, Accum.emit);
        }
        try splitter.flush(&acc, Accum.emit);
        try std.testing.expectEqualStrings("step 1\nstep 2", acc.reasoning[0..acc.r_len]);
        try std.testing.expectEqualStrings("Final 42!", acc.content[0..acc.c_len]);
    }
}

test "Utf8StreamAssembler reassembles split multi-byte CJK and emoji bytes losslessly" {
    const Sink = struct {
        buf: [128]u8 = undefined,
        len: usize = 0,
        calls: usize = 0,

        fn onChunk(self: *@This(), s: []const u8) bool {
            // Every emitted chunk must be valid UTF-8 on its own.
            if (!std.unicode.utf8ValidateSlice(s)) return false;
            @memcpy(self.buf[self.len .. self.len + s.len], s);
            self.len += s.len;
            self.calls += 1;
            return true;
        }
    };

    // "你好🚀!" = E4 BD A0 | E5 A5 BD | F0 9F 9A 80 | 21
    const src = "你好🚀!";
    var asm_u8 = Utf8StreamAssembler{};
    var sink = Sink{};

    // Feed 1 byte at a time so every multi-byte codepoint is split across calls.
    for (src) |b| {
        const one = [1]u8{b};
        try std.testing.expect(asm_u8.push(&one, &sink, Sink.onChunk));
    }
    try std.testing.expect(asm_u8.flush(&sink, Sink.onChunk));
    try std.testing.expectEqualStrings(src, sink.buf[0..sink.len]);

    // Truncated trailing UTF-8 sequence flushes cleanly as U+FFFD.
    var sink2 = Sink{};
    asm_u8.reset();
    try std.testing.expect(asm_u8.push(&[_]u8{ 'A', 0xE4, 0xBD }, &sink2, Sink.onChunk));
    try std.testing.expectEqualStrings("A", sink2.buf[0..sink2.len]);
    try std.testing.expect(asm_u8.flush(&sink2, Sink.onChunk));
    try std.testing.expectEqualStrings("A\xEF\xBF\xBD", sink2.buf[0..sink2.len]);
}

test "parseInferenceRequest formats ChatML across chat/completions, responses, and messages" {
    var prompt_buf: [4096]u8 = undefined;
    var model_buf: [MAX_MODEL_ID_BYTES]u8 = undefined;

    // 1. /v1/chat/completions with enable_thinking: false & array content & explicit seed.
    {
        const body =
            \\{"messages":[{"role":"developer","content":"Be brief."},{"role":"user","content":[{"type":"text","text":"Hi "},{"type":"text","text":"\u4f60\u597d"}]}],"enable_thinking":false,"max_completion_tokens":64,"temperature":0.6,"seed":42,"stream":true}
        ;
        const req = try parseInferenceRequest(.chat_completions, body, "qwen3-0.6b", &prompt_buf, &model_buf);
        try std.testing.expectEqualStrings("qwen3-0.6b", model_buf[0..req.model_len]);
        try std.testing.expectEqual(@as(u32, 64), req.max_tokens);
        try std.testing.expectEqual(@as(?u64, 42), req.seed);
        try std.testing.expect(req.stream);
        try std.testing.expect(!req.enable_thinking);
        try std.testing.expectEqualStrings(
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi 你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt_buf[0..req.prompt_len],
        );
    }

    // 2. /v1/responses with instructions appearing AFTER string input.
    {
        const body =
            \\{"model":"qwen3-14b","input":"What is 2+2?","instructions":"Think step by step.","max_output_tokens":32,"stream":true}
        ;
        const req = try parseInferenceRequest(.responses, body, "default", &prompt_buf, &model_buf);
        try std.testing.expectEqualStrings("qwen3-14b", model_buf[0..req.model_len]);
        try std.testing.expectEqual(@as(u32, 32), req.max_tokens);
        try std.testing.expectEqualStrings(
            "<|im_start|>system\nThink step by step.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
            prompt_buf[0..req.prompt_len],
        );
    }

    // 3. /v1/messages with system blocks AFTER messages & thinking.type = "disabled".
    {
        const body =
            \\{"messages":[{"role":"user","content":"Hello"}],"system":[{"type":"text","text":"Sys prompt"}],"thinking":{"type":"disabled"},"max_tokens":16}
        ;
        const req = try parseInferenceRequest(.messages, body, "qwen3-30b", &prompt_buf, &model_buf);
        try std.testing.expect(!req.enable_thinking);
        try std.testing.expectEqualStrings(
            "<|im_start|>system\nSys prompt<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt_buf[0..req.prompt_len],
        );
    }

    // 4. Parse stateful previous_response_id and reject malformed/invalid values.
    {
        const stateful_req = try parseInferenceRequest(.responses, "{\"input\":\"hi\",\"previous_response_id\":\"resp_1\"}", "m", &prompt_buf, &model_buf);
        try std.testing.expectEqualStrings("resp_1", stateful_req.previousResponseIdSlice());
    }
    try std.testing.expectError(
        error.MissingRequiredField,
        parseInferenceRequest(.chat_completions, "{\"messages\":[]}", "m", &prompt_buf, &model_buf),
    );
    try std.testing.expectError(
        error.InvalidFieldValue,
        parseInferenceRequest(.chat_completions, "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"temperature\":-0.5}", "m", &prompt_buf, &model_buf),
    );
    try std.testing.expectError(
        error.InvalidJson,
        parseInferenceRequest(.chat_completions, "{not json", "m", &prompt_buf, &model_buf),
    );
}

test "loopback GET /health, GET /v1/models, error statuses (400/404/405), and chunked POST" {
    const Mock = struct {
        fn generate(_: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(5);
            _ = stream.emitTokenBytes("ok");
            _ = job;
            stream.finish(.stop, 5);
        }
    };
    var dummy_ctx: u8 = 0;

    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-test-0.6b",
        .wired_mib = 1234,
        .max_conns = 8,
        .max_active_requests = 4,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &dummy_ctx, .generateFn = Mock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    try std.testing.expect(port > 0);
    var resp_buf: [8192]u8 = undefined;

    // GET /health (includes extended C18 monitor metrics)
    {
        const resp = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const body = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(body);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"status\":\"ok\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"model\":\"qwen3-test-0.6b\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"wired_mib\":1234") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"active_prefill_slots\":0") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"active_decode_slots\":0") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"tok_per_s\":") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"decode_share\":0.50") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"prefix_cache_hits\":0") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"retired_q8_chunks\":0") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"ang_pack\":null") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"tb_peers\":0") != null);
    }

    // Dynamic wired_mib update reflected live in GET /health
    srv.setWiredMib(2048);
    {
        const resp = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        const body = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(body);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"wired_mib\":2048") != null);
    }

    // GET /v1/models
    {
        const resp = try testHttpExchange(port, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const body = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(body);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"object\":\"list\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"id\":\"qwen3-test-0.6b\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"owned_by\":\"aminfer\"") != null);
    }

    // 404 Not Found
    {
        const resp = try testHttpExchange(port, "GET /v1/does-not-exist HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 404 Not Found\r\n"));
        const body = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(body);
    }

    // 405 Method Not Allowed
    {
        const resp = try testHttpExchange(port, "POST /health HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 405 Method Not Allowed\r\n"));
        try std.testing.expectEqualStrings("GET", extractHeaderValue(resp, "Allow") orelse "");
        try expectValidJson(extractHttpBody(resp) orelse return error.TestUnexpectedResult);

        const resp2 = try testHttpExchange(port, "GET /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp2, "HTTP/1.1 405 Method Not Allowed\r\n"));
        try std.testing.expectEqualStrings("POST", extractHeaderValue(resp2, "Allow") orelse "");
        try expectValidJson(extractHttpBody(resp2) orelse return error.TestUnexpectedResult);
    }

    // 400 Bad Request
    {
        var req_buf: [1024]u8 = undefined;
        const resp = try testHttpPostJson(port, "/v1/chat/completions", "{bad json", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 400 Bad Request\r\n"));
        try expectValidJson(extractHttpBody(resp) orelse return error.TestUnexpectedResult);
    }

    // Transfer-Encoding: chunked POST support
    {
        const chunked_req =
            "POST /v1/chat/completions HTTP/1.1\r\n" ++
            "Host: localhost\r\n" ++
            "Transfer-Encoding: chunked\r\n" ++
            "\r\n" ++
            "1c\r\n{\"messages\":[{\"role\":\"user\",\r\n" ++
            "16\r\n\"content\":\"chunked\"}]}\r\n" ++
            "0\r\n\r\n";
        const resp = try testHttpExchange(port, chunked_req, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const body = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(body);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"content\":\"ok\"") != null);
    }
}

test "POST /v1/chat/completions streaming & non-streaming <think> split and D11 TRNG seed_used + replay" {
    const MockEngine = struct {
        last_prompt: [512]u8 = undefined,
        last_prompt_len: std.atomic.Value(usize) = .init(0),

        fn generate(ctx_raw: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_raw));
            const copy_len = @min(job.prompt.len, self.last_prompt.len);
            @memcpy(self.last_prompt[0..copy_len], job.prompt[0..copy_len]);
            self.last_prompt_len.store(copy_len, .release);

            _ = stream.setPromptTokens(12);

            if (job.temperature > 0.0) {
                // Deterministic function of the slot's D11 Xoshiro256 PRNG stream so
                // explicit `seed` replays bit-exact tokens while unseeded TRNG varies.
                const vocab = [_][]const u8{ "alpha", "beta", "gamma", "delta", "omega", "zeta", "sigma", "kappa" };
                _ = stream.emitTokenBytes("<thi");
                _ = stream.emitTokenBytes("nk>roll ");
                var i: u32 = 0;
                while (i < 4) : (i += 1) {
                    const pick = job.rng.nextU64() % vocab.len;
                    if (!stream.emitTokenBytes(vocab[pick])) return;
                    if (i < 3) {
                        if (!stream.emitBytes(" ", 0)) return;
                    }
                }
                _ = stream.emitTokenBytes("</th");
                _ = stream.emitTokenBytes("ink>pick=");
                var num_buf: [32]u8 = undefined;
                const draw = job.rng.nextU64() % 1_000_000;
                const s = std.fmt.bufPrint(&num_buf, "{d}", .{draw}) catch unreachable;
                _ = stream.emitTokenBytes(s);
                stream.finish(.stop, 12);
                return;
            }

            // Greedy mode: split `<think>`, `</think>`, quotes, newlines, and multi-byte UTF-8.
            const pieces = [_][]const u8{
                "<t",
                "hink>",
                "step \"1\"\n",
                "step 2",
                "</t",
                "hink>",
                "Ans: ",
                "\xE4\xBD", // First 2 bytes of '你'
                "\xA0\xE5\xA5\xBD", // Last byte of '你' + '好'
            };
            for (pieces) |p| {
                if (!stream.emitTokenBytes(p)) return;
            }
            stream.finish(.stop, 12);
        }
    };

    var mock = MockEngine{};
    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-0.6b",
        .max_conns = 8,
        .max_active_requests = 4,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &mock, .generateFn = MockEngine.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [2048]u8 = undefined;
    var resp_buf: [16384]u8 = undefined;

    // 1. Greedy stream: true verifying ChatML prompt, fragmented <think> split, UTF-8, and timings.
    {
        const body =
            \\{"messages":[{"role":"system","content":"You are aminfer."},{"role":"user","content":"Hi"}],"stream":true,"temperature":0.0}
        ;
        const resp = try testHttpPostJson(port, "/v1/chat/completions", body, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualStrings("text/event-stream", extractHeaderValue(resp, "Content-Type") orelse "");
        try std.testing.expectEqualStrings("0", extractHeaderValue(resp, "X-Aminfer-Seed-Used") orelse "");
        try std.testing.expectEqualStrings("greedy", extractHeaderValue(resp, "X-Aminfer-Seed-Source") orelse "");

        const plen = mock.last_prompt_len.load(.acquire);
        try std.testing.expectEqualStrings(
            "<|im_start|>system\nYou are aminfer.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
            mock.last_prompt[0..plen],
        );

        const sse = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        _ = try expectValidSseJsonFrames(sse);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"seed_used\":0,\"seed_source\":\"greedy\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"reasoning_content\":\"step \\\"1\\\"\\n\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"reasoning_content\":\"step 2\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"content\":\"Ans: \"") != null);
        // Split 3-byte UTF-8 `'你'` completed on the next token, followed by `'好'`,
        // each emitted as valid UTF-8 with zero broken surrogate/byte sequences.
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"content\":\"你\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"content\":\"好\"") != null);
        try std.testing.expect(std.unicode.utf8ValidateSlice(sse));
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"finish_reason\":\"stop\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"prompt_tokens\":12,\"completion_tokens\":9,\"total_tokens\":21") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"aminfer_timings\":{\"ttft_ms\":") != null);
        try std.testing.expect(std.mem.endsWith(u8, sse, "data: [DONE]\n\n"));
    }

    // 2. Greedy stream: false verifying message.reasoning_content & message.content.
    {
        const body =
            \\{"messages":[{"role":"user","content":"Hi"}],"stream":false,"temperature":0.0}
        ;
        const resp = try testHttpPostJson(port, "/v1/chat/completions", body, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const json = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(json);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"object\":\"chat.completion\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"content\":\"Ans: 你好\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"reasoning_content\":\"step \\\"1\\\"\\nstep 2\"") != null);
    }

    // 3. D11 unseeded hardware TRNG diversity across 4 requests + bit-exact explicit replay!
    var hw_seeds: [4]u64 = undefined;
    var first_hw_content: [256]u8 = undefined;
    var first_hw_content_len: usize = 0;

    for (0..4) |i| {
        const body =
            \\{"messages":[{"role":"user","content":"roll"}],"stream":false,"temperature":0.8}
        ;
        const resp = try testHttpPostJson(port, "/v1/chat/completions", body, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualStrings("hardware_trng", extractHeaderValue(resp, "X-Aminfer-Seed-Source") orelse "");
        const seed_hdr = extractHeaderValue(resp, "X-Aminfer-Seed-Used") orelse return error.TestUnexpectedResult;
        const seed_val = try std.fmt.parseInt(u64, seed_hdr, 10);
        hw_seeds[i] = seed_val;
        for (hw_seeds[0..i]) |prev| {
            try std.testing.expect(prev != seed_val);
        }

        const json = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(json);
        var expected_seed_json: [96]u8 = undefined;
        const needle = try std.fmt.bufPrint(&expected_seed_json, "\"seed_used\":{d},\"seed_source\":\"hardware_trng\"", .{seed_val});
        try std.testing.expect(std.mem.indexOf(u8, json, needle) != null);

        if (i == 0) {
            const marker = "\"message\":";
            const start = std.mem.indexOf(u8, json, marker) orelse return error.TestUnexpectedResult;
            const end = std.mem.indexOfPos(u8, json, start, "},\"finish_reason\"") orelse return error.TestUnexpectedResult;
            const msg_slice = json[start..end];
            @memcpy(first_hw_content[0..msg_slice.len], msg_slice);
            first_hw_content_len = msg_slice.len;
        }
    }

    // Replay hw_seeds[0] explicitly and assert bit-exact identical reasoning_content + content!
    {
        var replay_body_buf: [256]u8 = undefined;
        const replay_body = try std.fmt.bufPrint(
            &replay_body_buf,
            "{{\"messages\":[{{\"role\":\"user\",\"content\":\"roll\"}}],\"stream\":false,\"temperature\":0.8,\"seed\":{d}}}",
            .{hw_seeds[0]},
        );
        const resp = try testHttpPostJson(port, "/v1/chat/completions", replay_body, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualStrings("explicit", extractHeaderValue(resp, "X-Aminfer-Seed-Source") orelse "");
        const json = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(json);
        try std.testing.expect(std.mem.indexOf(u8, json, first_hw_content[0..first_hw_content_len]) != null);
        var exp_buf: [96]u8 = undefined;
        const exp_seed = try std.fmt.bufPrint(&exp_buf, "\"seed_used\":{d},\"seed_source\":\"explicit\"", .{hw_seeds[0]});
        try std.testing.expect(std.mem.indexOf(u8, json, exp_seed) != null);
    }
}

test "POST /v1/responses and POST /v1/messages stateless streaming & non-streaming adapters" {
    const Mock = struct {
        fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(7);
            const parts = [_][]const u8{ "<think>", "reason A", "</think>", "Final B" };
            for (parts) |p| {
                if (!stream.emitTokenBytes(p)) return;
            }
            stream.finish(.stop, 7);
        }
    };
    var dummy_ctx: u8 = 0;

    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-adapter",
        .max_conns = 8,
        .max_active_requests = 4,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &dummy_ctx, .generateFn = Mock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [2048]u8 = undefined;
    var resp_buf: [16384]u8 = undefined;

    // 1. /v1/responses stream: true
    {
        const body =
            \\{"input":"Solve X","instructions":"Be concise","stream":true,"seed":77,"temperature":0.5}
        ;
        const resp = try testHttpPostJson(port, "/v1/responses", body, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualStrings("77", extractHeaderValue(resp, "X-Aminfer-Seed-Used") orelse "");
        const sse = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        _ = try expectValidSseJsonFrames(sse);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.created\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.reasoning_summary_text.delta\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"delta\":\"reason A\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.output_text.delta\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"delta\":\"Final B\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.completed\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"seed_used\":77,\"seed_source\":\"explicit\"") != null);
    }

    // 2. /v1/responses stream: false
    {
        const body =
            \\{"input":[{"role":"user","content":"Solve X"}],"stream":false}
        ;
        const resp = try testHttpPostJson(port, "/v1/responses", body, &req_buf, &resp_buf);
        const json = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(json);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"object\":\"response\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"reasoning\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"text\":\"reason A\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"output_text\":\"Final B\"") != null);
    }

    // 3. /v1/messages stream: true
    {
        const body =
            \\{"system":"You are helpful","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"stream":true,"seed":99,"temperature":0.7}
        ;
        const resp = try testHttpPostJson(port, "/v1/messages", body, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualStrings("99", extractHeaderValue(resp, "X-Aminfer-Seed-Used") orelse "");
        const sse = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        _ = try expectValidSseJsonFrames(sse);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: message_start\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"input_tokens\":7") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"type\":\"thinking_delta\",\"thinking\":\"reason A\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"type\":\"text_delta\",\"text\":\"Final B\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: message_delta\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"stop_reason\":\"end_turn\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: message_stop\n") != null);
    }

    // 4. /v1/messages stream: false
    {
        const body =
            \\{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"stream":false}
        ;
        const resp = try testHttpPostJson(port, "/v1/messages", body, &req_buf, &resp_buf);
        const json = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(json);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"message\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, json, "{\"type\":\"thinking\",\"thinking\":\"reason A\"}") != null);
        try std.testing.expect(std.mem.indexOf(u8, json, "{\"type\":\"text\",\"text\":\"Final B\"}") != null);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"stop_reason\":\"end_turn\"") != null);
    }
}

test "16 simultaneous concurrent loopback streaming clients complete losslessly" {
    const Mock = struct {
        fn generate(_: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(4);
            _ = stream.emitTokenBytes("<think>t</think>");
            var buf: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "seed={d}", .{job.seed_used}) catch unreachable;
            _ = stream.emitTokenBytes(s);
            stream.finish(.stop, 4);
        }
    };
    var dummy_ctx: u8 = 0;

    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-load",
        .max_conns = 32,
        .max_active_requests = 24,
        .job_ring_cap = 64,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &dummy_ctx, .generateFn = Mock.generate });
    defer srv.stop();

    const num_clients: usize = 16;
    const ClientOutcome = struct {
        ok: bool = false,
    };

    const ClientTask = struct {
        fn run(port: u16, id: usize, start_gate: *std.atomic.Value(bool), out: *ClientOutcome) void {
            while (!start_gate.load(.acquire)) {
                std.atomic.spinLoopHint();
            }
            const seed: u64 = 1000 + id;
            var body_buf: [256]u8 = undefined;
            var req_buf: [1024]u8 = undefined;
            var resp_buf: [8192]u8 = undefined;

            const endpoint_mod = id % 3;
            const path: []const u8 = switch (endpoint_mod) {
                0 => "/v1/chat/completions",
                1 => "/v1/responses",
                else => "/v1/messages",
            };
            const body = switch (endpoint_mod) {
                1 => std.fmt.bufPrint(
                    &body_buf,
                    "{{\"input\":\"c{d}\",\"stream\":true,\"temperature\":0.7,\"seed\":{d}}}",
                    .{ id, seed },
                ) catch return,
                else => std.fmt.bufPrint(
                    &body_buf,
                    "{{\"messages\":[{{\"role\":\"user\",\"content\":\"c{d}\"}}],\"stream\":true,\"temperature\":0.7,\"seed\":{d}}}",
                    .{ id, seed },
                ) catch return,
            };

            const resp = testHttpPostJson(port, path, body, &req_buf, &resp_buf) catch return;
            if (!std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n")) return;
            var expected_buf: [32]u8 = undefined;
            const expected_tok = std.fmt.bufPrint(&expected_buf, "seed={d}", .{seed}) catch return;
            if (std.mem.indexOf(u8, resp, expected_tok) == null) return;
            out.ok = true;
        }
    };

    var start_gate = std.atomic.Value(bool).init(false);
    var outcomes: [num_clients]ClientOutcome = @splat(.{});
    var threads: [num_clients]std.Thread = undefined;
    var spawned: usize = 0;
    defer for (threads[0..spawned]) |t| t.join();

    for (0..num_clients) |i| {
        threads[i] = try std.Thread.spawn(.{}, ClientTask.run, .{ srv.boundPort(), i, &start_gate, &outcomes[i] });
        spawned += 1;
    }
    start_gate.store(true, .release);

    for (threads[0..spawned]) |t| t.join();
    spawned = 0;

    for (outcomes) |o| {
        try std.testing.expect(o.ok);
    }
    try std.testing.expectEqual(@as(u64, num_clients), srv.stats().total_requests);
}

test "mid-stream client disconnect cancels worker immediately without SIGPIPE and recycles slot" {
    const SlowMock = struct {
        tokens_emitted: std.atomic.Value(u32) = .init(0),
        saw_cancel: std.atomic.Value(bool) = .init(false),
        started: std.atomic.Value(bool) = .init(false),

        fn generate(ctx_raw: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_raw));
            if (job.max_tokens == 2) {
                // Fast post-cancel health check request.
                _ = stream.setPromptTokens(2);
                _ = stream.emitTokenBytes("recovered");
                stream.finish(.stop, 2);
                return;
            }
            self.started.store(true, .release);
            _ = stream.setPromptTokens(5);
            var i: u32 = 0;
            while (i < 10_000) : (i += 1) {
                if (stream.isCancelled()) {
                    self.saw_cancel.store(true, .release);
                    return;
                }
                if (!stream.emitTokenBytes("tok ")) {
                    self.saw_cancel.store(true, .release);
                    return;
                }
                _ = self.tokens_emitted.fetchAdd(1, .release);
                sleepNs(2 * std.time.ns_per_ms);
            }
            stream.finish(.length, 5);
        }
    };

    var mock = SlowMock{};
    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-cancel",
        .max_conns = 4,
        .max_active_requests = 1,
        .job_ring_cap = 4,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &mock, .generateFn = SlowMock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    // Connect, start streaming, read initial bytes, then abruptly close socket!
    {
        const fd = try testConnectLoopback(port);
        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"stream forever\"}],\"stream\":true,\"max_tokens\":10000}";
        var req_buf: [512]u8 = undefined;
        var w = FixedWriter.init(&req_buf);
        try w.print(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {d}\r\n\r\n{s}",
            .{ body.len, body },
        );
        try testSendAll(fd, w.written());

        var first_buf: [512]u8 = undefined;
        const n = std.c.recv(fd, &first_buf, first_buf.len, 0);
        try std.testing.expect(n > 0);
        // Abrupt disconnect mid-stream!
        _ = std.c.close(fd);
    }

    // Wait up to 1 second for the worker to observe cancellation and slot to recycle.
    const deadline = monotonicNowNs() + 1 * std.time.ns_per_s;
    while (monotonicNowNs() < deadline) {
        if (mock.saw_cancel.load(.acquire) and srv.stats().active_requests == 0) break;
        sleepNs(2 * std.time.ns_per_ms);
    }

    try std.testing.expect(mock.saw_cancel.load(.acquire));
    try std.testing.expect(mock.tokens_emitted.load(.acquire) < 500);
    try std.testing.expectEqual(@as(usize, 0), srv.stats().active_requests);

    // Verify the single slot recycled cleanly and serves a new request normally.
    var req_buf: [512]u8 = undefined;
    var resp_buf: [4096]u8 = undefined;
    const resp = try testHttpPostJson(
        port,
        "/v1/chat/completions",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"stream\":false,\"max_tokens\":2}",
        &req_buf,
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(std.mem.indexOf(u8, resp, "\"content\":\"recovered\"") != null);
}

test "503 Service Unavailable backpressure when all preallocated active request slots are full" {
    const GateMock = struct {
        in_flight: std.atomic.Value(bool) = .init(false),
        release: std.atomic.Value(bool) = .init(false),

        fn generate(ctx_raw: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_raw));
            self.in_flight.store(true, .release);
            while (!self.release.load(.acquire) and !stream.isCancelled()) {
                sleepNs(1 * std.time.ns_per_ms);
            }
            _ = stream.setPromptTokens(3);
            _ = stream.emitTokenBytes("done");
            stream.finish(.stop, 3);
        }
    };

    var mock = GateMock{};
    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-503",
        .max_conns = 6,
        .max_active_requests = 1,
        .job_ring_cap = 4,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &mock, .generateFn = GateMock.generate });
    defer srv.stop();

    const port = srv.boundPort();

    // Occupy the single active inference slot in a background client thread.
    const Holder = struct {
        ok: bool = false,
        fn run(p: u16, out: *@This()) void {
            var req_buf: [512]u8 = undefined;
            var resp_buf: [4096]u8 = undefined;
            const resp = testHttpPostJson(
                p,
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hold\"}],\"stream\":true}",
                &req_buf,
                &resp_buf,
            ) catch return;
            out.ok = std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n") and
                std.mem.indexOf(u8, resp, "\"content\":\"done\"") != null;
        }
    };
    var holder = Holder{};
    var holder_thread: ?std.Thread = try std.Thread.spawn(.{}, Holder.run, .{ port, &holder });
    defer {
        mock.release.store(true, .release);
        if (holder_thread) |t| t.join();
    }

    const deadline = monotonicNowNs() + 1 * std.time.ns_per_s;
    while (!mock.in_flight.load(.acquire) and monotonicNowNs() < deadline) {
        sleepNs(1 * std.time.ns_per_ms);
    }
    try std.testing.expect(mock.in_flight.load(.acquire));

    // Second concurrent inference request must receive 503 Service Unavailable immediately.
    var req_buf: [512]u8 = undefined;
    var resp_buf: [4096]u8 = undefined;
    {
        const resp = try testHttpPostJson(
            port,
            "/v1/chat/completions",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"overflow\"}],\"stream\":true}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 503 Service Unavailable\r\n"));
        try expectValidJson(extractHttpBody(resp) orelse return error.TestUnexpectedResult);
        try std.testing.expect(std.mem.indexOf(u8, resp, "server_overloaded") != null);
        try std.testing.expectEqual(@as(u64, 1), srv.stats().rejected_503_count);
    }

    // Meanwhile GET /health still succeeds with active_requests == 1!
    {
        const resp = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, resp, "\"active_requests\":1") != null);
    }

    // Release the holder and confirm it completes cleanly.
    mock.release.store(true, .release);
    if (holder_thread) |t| {
        t.join();
        holder_thread = null;
    }
    try std.testing.expect(holder.ok);
}

test "steady-state HTTP/SSE serving performs zero dynamic heap allocations (malloc_zone_statistics)" {
    const Mock = struct {
        fn generate(_: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(6);
            _ = stream.emitTokenBytes("<think>ok</think>");
            _ = job.rng.nextU64();
            _ = stream.emitTokenBytes("zero-alloc!");
            stream.finish(.stop, 6);
        }
    };
    var dummy_ctx: u8 = 0;

    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-zero-alloc",
        .max_conns = 8,
        .max_active_requests = 4,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &dummy_ctx, .generateFn = Mock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [1024]u8 = undefined;
    var resp_buf: [8192]u8 = undefined;

    // Warm up every endpoint, syscall, and dyld lazy symbol once before snapshotting.
    _ = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
    _ = try testHttpExchange(port, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
    _ = try testHttpPostJson(port, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"w\"}],\"stream\":true,\"temperature\":0.7}", &req_buf, &resp_buf);
    _ = try testHttpPostJson(port, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"w\"}],\"stream\":false,\"seed\":1}", &req_buf, &resp_buf);
    _ = try testHttpPostJson(port, "/v1/responses", "{\"input\":\"w\",\"stream\":true}", &req_buf, &resp_buf);
    _ = try testHttpPostJson(port, "/v1/messages", "{\"messages\":[{\"role\":\"user\",\"content\":\"w\"}],\"stream\":true}", &req_buf, &resp_buf);

    // Wait until all warmup connections have fully closed on the server thread.
    while (srv.stats().active_conns > 0 or srv.stats().active_requests > 0) {
        sleepNs(500_000);
    }

    var m_warm: MallocStatistics = std.mem.zeroes(MallocStatistics);
    malloc_zone_statistics(null, &m_warm);

    var m_before: MallocStatistics = std.mem.zeroes(MallocStatistics);
    malloc_zone_statistics(null, &m_before);

    // Steady-state loop across all 5 endpoints (24 full TCP HTTP/SSE exchanges).
    for (0..4) |_| {
        const r1 = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r1, "HTTP/1.1 200 OK\r\n"));

        const r2 = try testHttpExchange(port, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r2, "HTTP/1.1 200 OK\r\n"));

        const r3 = try testHttpPostJson(
            port,
            "/v1/chat/completions",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true,\"temperature\":0.7}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.indexOf(u8, r3, "data: [DONE]\n\n") != null);

        const r4 = try testHttpPostJson(
            port,
            "/v1/chat/completions",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false,\"temperature\":0.7,\"seed\":12345}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.indexOf(u8, r4, "\"content\":\"zero-alloc!\"") != null);

        const r5 = try testHttpPostJson(
            port,
            "/v1/responses",
            "{\"input\":\"hello\",\"stream\":true}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.indexOf(u8, r5, "event: response.completed\n") != null);

        const r6 = try testHttpPostJson(
            port,
            "/v1/messages",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.indexOf(u8, r6, "event: message_stop\n") != null);
    }

    while (srv.stats().active_conns > 0 or srv.stats().active_requests > 0) {
        sleepNs(500_000);
    }

    var m_after: MallocStatistics = std.mem.zeroes(MallocStatistics);
    malloc_zone_statistics(null, &m_after);

    try std.testing.expect(m_after.blocks_in_use <= m_before.blocks_in_use);
    try std.testing.expect(m_after.size_in_use <= m_before.size_in_use);
}

test "parseIpv4, QoS USER_INITIATED startup, finish_reason length, and connection-table 503 overflow" {
    try std.testing.expectEqual([4]u8{ 127, 0, 0, 1 }, try parseIpv4("127.0.0.1"));
    try std.testing.expectEqual([4]u8{ 0, 0, 0, 0 }, try parseIpv4("0.0.0.0"));
    try std.testing.expectError(error.InvalidIpv4, parseIpv4("127.0.0"));
    try std.testing.expectError(error.InvalidIpv4, parseIpv4("127.0.0.256"));
    try std.testing.expectError(error.InvalidIpv4, parseIpv4("localhost"));

    const LengthMock = struct {
        fn generate(_: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(3);
            var i: u32 = 0;
            while (i < job.max_tokens) : (i += 1) {
                if (!stream.emitTokenBytes("x")) return;
            }
            stream.finish(.length, 3);
        }
    };
    var dummy_ctx: u8 = 0;

    // Exercise `apply_qos = true` (QOS_CLASS_USER_INITIATED) and `max_conns = 2` overflow 503.
    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-qos",
        .max_conns = 2,
        .max_active_requests = 2,
        .job_ring_cap = 4,
        .apply_qos = true,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &dummy_ctx, .generateFn = LengthMock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [512]u8 = undefined;
    var resp_buf: [4096]u8 = undefined;

    // Verify finish_reason: "length" on /v1/chat/completions and stop_reason: "max_tokens" on /v1/messages.
    {
        const r1 = try testHttpPostJson(
            port,
            "/v1/chat/completions",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3,\"enable_thinking\":false,\"stream\":true}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.indexOf(u8, r1, "\"finish_reason\":\"length\"") != null);

        const r2 = try testHttpPostJson(
            port,
            "/v1/messages",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"stream\":true}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.indexOf(u8, r2, "\"stop_reason\":\"max_tokens\"") != null);
    }

    // Hold both connection slots open simultaneously, then verify a 3rd TCP connection
    // immediately receives the static zero-alloc 503 response.
    const idle1 = try testConnectLoopback(port);
    defer _ = std.c.close(idle1);
    const idle2 = try testConnectLoopback(port);
    defer _ = std.c.close(idle2);

    const deadline = monotonicNowNs() + 1 * std.time.ns_per_s;
    while (srv.stats().active_conns < 2 and monotonicNowNs() < deadline) {
        sleepNs(1 * std.time.ns_per_ms);
    }
    try std.testing.expectEqual(@as(usize, 2), srv.stats().active_conns);

    const overflow_fd = try testConnectLoopback(port);
    defer _ = std.c.close(overflow_fd);
    const overflow_resp = try testRecvAll(overflow_fd, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, overflow_resp, "HTTP/1.1 503 Service Unavailable\r\n"));
    try std.testing.expect(std.mem.indexOf(u8, overflow_resp, "server_overloaded") != null);
    try std.testing.expect(srv.stats().rejected_503_count >= 1);
}

fn testTmpAngPathServer(buf: *[256]u8, name: []const u8) ![:0]const u8 {
    const dir_slice: []const u8 = if (std.c.getenv("TMPDIR")) |raw|
        std.mem.sliceTo(raw, 0)
    else
        "/Volumes/Akari/.tmp";
    const trimmed = std.mem.trimEnd(u8, dir_slice, "/");
    const s = try std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/{s}", .{ trimmed, name });
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

/// Extracts concatenated `delta.content` strings from an OpenAI SSE response into `out`.
fn extractOpenAiSseContent(sse: []const u8, out: []u8) []const u8 {
    var pos: usize = 0;
    var lines = std.mem.splitScalar(u8, sse, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const payload = line[6..];
        if (std.mem.eql(u8, payload, "[DONE]")) continue;
        const needle = "\"content\":\"";
        const idx = std.mem.indexOf(u8, payload, needle) orelse continue;
        const start = idx + needle.len;
        const end = std.mem.indexOfScalarPos(u8, payload, start, '"') orelse continue;
        const chunk = payload[start..end];
        if (pos + chunk.len <= out.len) {
            @memcpy(out[pos .. pos + chunk.len], chunk);
            pos += chunk.len;
        }
    }
    return out[0..pos];
}

test "C14 8 concurrent batched SSE clients: M>=4..8, interleaved delivery, p50/p99 TTFT << serial, D11 seed parity, disconnect cleanup, and zero steady-state malloc" {
    const gpa = std.testing.allocator;

    const BatchMock = struct {
        step_delay_ns: u64 = 2 * std.time.ns_per_ms,
        prefix_hit_len: u32 = 8,
        released_count: std.atomic.Value(u32) = .init(0),
        unpinned_count: std.atomic.Value(u32) = .init(0),
        pinned_handles: [batch_sched.MAX_BATCH_SLOTS]u64 = @splat(0),
        next_handle: u64 = 1,

        fn tokenize(_: *anyopaque, job: *const InferenceJob, out_tokens: []u32) batch_sched.BackendError!usize {
            const n = @min(@max(job.prompt.len / 4, 12), out_tokens.len);
            for (0..n) |i| {
                out_tokens[i] = if (i == 0 and std.mem.indexOf(u8, job.prompt, "SHARED_SYS") != null)
                    7777
                else
                    @intCast((job.prompt[i % job.prompt.len]) + 1);
            }
            return n;
        }

        fn detokenize(_: *anyopaque, tok: u32, out_bytes: []u8) []const u8 {
            const alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
            const c = alphabet[tok % alphabet.len];
            out_bytes[0] = c;
            return out_bytes[0..1];
        }

        fn lookupPrefix(ctx_ptr: *anyopaque, slot_id: u16, prompt_tokens: []const u32) batch_sched.PrefixLookupResult {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            if (prompt_tokens.len > self.prefix_hit_len and prompt_tokens[0] == 7777) {
                const h = self.next_handle;
                self.next_handle += 1;
                self.pinned_handles[slot_id] = h;
                return .{ .matched_tokens = self.prefix_hit_len, .prefix_handle = h };
            }
            return .{};
        }

        fn sampleToken(rng_ptr: *rng_mod.RequestRng, temp: f32, pos: u32, prev: u32) u32 {
            if (temp > 0.0) {
                return @intCast((rng_ptr.nextU64() ^ (@as(u64, pos) *% 131) ^ prev) % 36);
            }
            return (pos * 7 + prev) % 36;
        }

        fn prefillSlot(ctx_ptr: *anyopaque, chunk: *const batch_sched.PrefillChunkView) batch_sched.BackendError!batch_sched.PrefillResult {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            if (!chunk.is_final_chunk) return .{};
            if (self.step_delay_ns > 0) sleepNs(self.step_delay_ns / 4);
            const tok = sampleToken(chunk.rng, chunk.temperature, chunk.total_prompt_len, chunk.tokens[chunk.tokens.len - 1]);
            return .{ .first_token = tok };
        }

        fn stepDecodeBatch(ctx_ptr: *anyopaque, view: *const batch_sched.BatchStepView, out: []u32) batch_sched.BackendError!void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            if (self.step_delay_ns > 0) sleepNs(self.step_delay_ns);
            for (0..view.batchSize()) |b| {
                out[b] = sampleToken(view.rngs[b], view.temperatures[b], view.seq_positions[b], view.input_tokens[b]);
            }
        }

        fn releaseSlot(ctx_ptr: *anyopaque, slot_id: u16, prefix_handle: u64) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            _ = self.released_count.fetchAdd(1, .acq_rel);
            if (prefix_handle != 0) {
                std.debug.assert(self.pinned_handles[slot_id] == prefix_handle);
                self.pinned_handles[slot_id] = 0;
                _ = self.unpinned_count.fetchAdd(1, .acq_rel);
            }
        }

        fn batchHandler(self: *@This()) BatchEngineHandler {
            return .{
                .ctx = self,
                .tokenizeFn = tokenize,
                .detokenizeFn = detokenize,
                .backend = .{
                    .ctx = self,
                    .lookupPrefixFn = lookupPrefix,
                    .prefillSlotFn = prefillSlot,
                    .stepDecodeBatchFn = stepDecodeBatch,
                    .releaseSlotFn = releaseSlot,
                },
            };
        }

        // Matching single-slot serial generator with identical per-token delay for TTFT comparison.
        fn serialGenerate(ctx_ptr: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            _ = stream.setPromptTokens(12);
            if (self.step_delay_ns > 0) sleepNs(self.step_delay_ns / 4);
            for (0..job.max_tokens) |step_i| {
                if (stream.isCancelled()) return;
                if (step_i > 0 and self.step_delay_ns > 0) sleepNs(self.step_delay_ns);
                var b: [1]u8 = .{'a'};
                if (!stream.emitTokenBytes(&b)) return;
            }
            stream.finish(.length, 12);
        }
    };

    const ClientJob = struct {
        port: u16,
        path: []const u8,
        body: []const u8,
        resp_buf: [16384]u8 = undefined,
        resp_len: usize = 0,
        ttft_ns: u64 = 0,
        ok: bool = false,

        fn run(c: *@This()) void {
            var req_buf: [1024]u8 = undefined;
            const req = std.fmt.bufPrint(
                &req_buf,
                "POST {s} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
                .{ c.path, c.body.len, c.body },
            ) catch return;
            const t0 = monotonicNowNs();
            const fd = testConnectLoopback(c.port) catch return;
            defer _ = std.c.close(fd);
            testSendAll(fd, req) catch return;

            var total: usize = 0;
            var saw_data = false;
            while (total < c.resp_buf.len) {
                const n = std.c.recv(fd, &c.resp_buf[total], c.resp_buf.len - total, 0);
                if (n > 0) {
                    total += @intCast(n);
                    if (!saw_data and std.mem.indexOf(u8, c.resp_buf[0..total], "data: ") != null) {
                        c.ttft_ns = monotonicNowNs() -| t0;
                        saw_data = true;
                    }
                } else if (n == 0) {
                    break;
                } else {
                    if (std.c.errno(n) == .INTR) continue;
                    return;
                }
            }
            c.resp_len = total;
            c.ok = saw_data and total > 0;
        }
    };

    // 1. Measure 8-client serial queueing TTFT baseline (`start(EngineHandler)`).
    var serial_mock = BatchMock{ .step_delay_ns = 2 * std.time.ns_per_ms };
    const serial_srv = try Server.init(gpa, .{
        .port = 0,
        .model_id = "c14-serial",
        .max_conns = 16,
        .max_active_requests = 16,
        .apply_qos = false,
    });
    defer serial_srv.deinit(gpa);
    try serial_srv.start(.{ .ctx = &serial_mock, .generateFn = BatchMock.serialGenerate });
    defer serial_srv.stop();

    var serial_clients: [8]ClientJob = undefined;
    var serial_threads: [8]std.Thread = undefined;
    const serial_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true,\"max_tokens\":12,\"temperature\":0.0}";
    for (&serial_clients, 0..) |*c, i| {
        c.* = .{ .port = serial_srv.boundPort(), .path = "/v1/chat/completions", .body = serial_body };
        serial_threads[i] = try std.Thread.spawn(.{}, ClientJob.run, .{c});
    }
    for (&serial_threads) |t| t.join();
    var serial_ttfts: [8]u64 = undefined;
    for (serial_clients, 0..) |c, i| {
        try std.testing.expect(c.ok);
        serial_ttfts[i] = c.ttft_ns;
    }
    std.mem.sort(u64, &serial_ttfts, {}, std.sort.asc(u64));
    const serial_p50 = serial_ttfts[4];
    const serial_p99 = serial_ttfts[7];

    // 2. Now run the continuous-batching server (`startBatched`)!
    var batch_mock = BatchMock{ .step_delay_ns = 2 * std.time.ns_per_ms };
    const srv = try Server.init(gpa, .{
        .port = 0,
        .model_id = "c14-batched",
        .max_conns = 16,
        .max_active_requests = 16,
        .max_batch_size = 8,
        .max_prefill_chunk_tokens = 512,
        .max_prompt_tokens = 256,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try srv.startBatched(batch_mock.batchHandler());
    defer srv.stop();

    const port = srv.boundPort();

    // 2a. Solo reference run with explicit D11 seed (`B = 1`).
    var solo_req_buf: [1024]u8 = undefined;
    var solo_resp_buf: [16384]u8 = undefined;
    const seeded_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"SHARED_SYS deterministic replay\"}],\"stream\":true,\"max_tokens\":12,\"temperature\":0.8,\"seed\":123456789012345}";
    const solo_resp = try testHttpPostJson(port, "/v1/chat/completions", seeded_body, &solo_req_buf, &solo_resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, solo_resp, "HTTP/1.1 200 OK\r\n"));
    _ = try expectValidSseJsonFrames(extractHttpBody(solo_resp) orelse return error.TestUnexpectedResult);
    var solo_text_buf: [128]u8 = undefined;
    const solo_text = extractOpenAiSseContent(extractHttpBody(solo_resp).?, &solo_text_buf);
    try std.testing.expectEqual(@as(usize, 12), solo_text.len);

    // 2b. Launch 8 simultaneous SSE clients (including /v1/chat/completions, /v1/responses,
    //     /v1/messages, shared prefix hits, and two identical explicit-seed requests!).
    const body_trng = "{\"messages\":[{\"role\":\"user\",\"content\":\"SHARED_SYS concurrent batch\"}],\"stream\":true,\"max_tokens\":12,\"temperature\":0.8}";
    const body_resp_ep = "{\"input\":\"SHARED_SYS responses stream\",\"stream\":true,\"max_tokens\":12,\"temperature\":0.7}";
    const body_msg_ep = "{\"messages\":[{\"role\":\"user\",\"content\":\"SHARED_SYS messages stream\"}],\"stream\":true,\"max_tokens\":12,\"temperature\":0.7}";

    var clients: [8]ClientJob = undefined;
    var threads: [8]std.Thread = undefined;
    for (&clients, 0..) |*c, i| {
        if (i == 0 or i == 4) {
            c.* = .{ .port = port, .path = "/v1/chat/completions", .body = seeded_body };
        } else if (i == 5) {
            c.* = .{ .port = port, .path = "/v1/responses", .body = body_resp_ep };
        } else if (i == 6) {
            c.* = .{ .port = port, .path = "/v1/messages", .body = body_msg_ep };
        } else {
            c.* = .{ .port = port, .path = "/v1/chat/completions", .body = body_trng };
        }
        threads[i] = try std.Thread.spawn(.{}, ClientJob.run, .{c});
    }
    for (&threads) |t| t.join();

    var batched_ttfts: [8]u64 = undefined;
    for (&clients, 0..) |*c, i| {
        try std.testing.expect(c.ok);
        const resp = c.resp_buf[0..c.resp_len];
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const body = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        const frames = try expectValidSseJsonFrames(body);
        try std.testing.expect(frames >= 12);
        batched_ttfts[i] = c.ttft_ns;
    }

    // Verify D11 explicit seed bit-exactness under concurrent B=8 batching!
    var c0_buf: [128]u8 = undefined;
    var c4_buf: [128]u8 = undefined;
    const c0_text = extractOpenAiSseContent(extractHttpBody(clients[0].resp_buf[0..clients[0].resp_len]).?, &c0_buf);
    const c4_text = extractOpenAiSseContent(extractHttpBody(clients[4].resp_buf[0..clients[4].resp_len]).?, &c4_buf);
    try std.testing.expectEqualStrings(solo_text, c0_text);
    try std.testing.expectEqualStrings(solo_text, c4_text);

    // Verify continuous batching actually batched M >= 4..8 and beat serial p50/p99 TTFT!
    std.mem.sort(u64, &batched_ttfts, {}, std.sort.asc(u64));
    const batched_p50 = batched_ttfts[4];
    const batched_p99 = batched_ttfts[7];
    try std.testing.expect(batched_p50 * 2 < serial_p50);
    try std.testing.expect(batched_p99 * 3 < serial_p99);

    const st_after_8 = srv.stats();
    try std.testing.expect(st_after_8.max_observed_batch_size >= 4);
    try std.testing.expect(st_after_8.mean_batch_size > 2.0);
    try std.testing.expect(st_after_8.prefix_hit_tokens >= 8 * 8);
    try std.testing.expect(st_after_8.prefix_hit_requests >= 8);

    // Verify GET /health reflects batch telemetry and parses as valid JSON!
    {
        var h_buf: [2048]u8 = undefined;
        const h_resp = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", &h_buf);
        try std.testing.expect(std.mem.startsWith(u8, h_resp, "HTTP/1.1 200 OK\r\n"));
        const h_body = extractHttpBody(h_resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(h_body);
        try std.testing.expect(std.mem.indexOf(u8, h_body, "\"active_batch_size\":0") != null);
        try std.testing.expect(std.mem.indexOf(u8, h_body, "\"mean_batch_size\":") != null);
        try std.testing.expect(std.mem.indexOf(u8, h_body, "\"prefix_hit_tokens\":") != null);
    }

    // 2c. Also test non-streaming (`stream: false`) across all 3 endpoints under `startBatched`!
    {
        const r_chat = try testHttpPostJson(port, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"SHARED_SYS ns\"}],\"stream\":false,\"max_tokens\":6}", &solo_req_buf, &solo_resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_chat, "HTTP/1.1 200 OK\r\n"));
        try expectValidJson(extractHttpBody(r_chat) orelse return error.TestUnexpectedResult);

        const r_resp = try testHttpPostJson(port, "/v1/responses", "{\"input\":\"SHARED_SYS ns\",\"stream\":false,\"max_tokens\":6}", &solo_req_buf, &solo_resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_resp, "HTTP/1.1 200 OK\r\n"));
        try expectValidJson(extractHttpBody(r_resp) orelse return error.TestUnexpectedResult);

        const r_msg = try testHttpPostJson(port, "/v1/messages", "{\"messages\":[{\"role\":\"user\",\"content\":\"SHARED_SYS ns\"}],\"stream\":false,\"max_tokens\":6}", &solo_req_buf, &solo_resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_msg, "HTTP/1.1 200 OK\r\n"));
        try expectValidJson(extractHttpBody(r_msg) orelse return error.TestUnexpectedResult);
    }

    // 2d. Mid-stream client disconnect during concurrent batching:
    //     open 4 clients, abort client 0 after its first SSE bytes; verify immediate
    //     releaseSlot + prefix unpin while clients 1..3 complete losslessly.
    {
        const before_unpins = batch_mock.unpinned_count.load(.acquire);
        const before_cancels = srv.stats().cancelled_count;
        const long_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"SHARED_SYS mid-cancel\"}],\"stream\":true,\"max_tokens\":24,\"temperature\":0.7}";
        var req_str_buf: [1024]u8 = undefined;
        const req_str = try std.fmt.bufPrint(
            &req_str_buf,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ long_body.len, long_body },
        );

        var fds: [4]std.c.fd_t = .{ -1, -1, -1, -1 };
        defer for (&fds) |*fd| {
            if (fd.* >= 0) {
                _ = std.c.close(fd.*);
                fd.* = -1;
            }
        };
        for (&fds) |*fd| {
            fd.* = try testConnectLoopback(port);
            try testSendAll(fd.*, req_str);
        }

        // Read first bytes on client 0, then abort it immediately mid-stream!
        var first_buf: [512]u8 = undefined;
        const first_n = std.c.recv(fds[0], &first_buf, first_buf.len, 0);
        try std.testing.expect(first_n > 0);
        _ = std.c.close(fds[0]);
        fds[0] = -1;

        // Remaining 3 concurrent clients MUST complete losslessly!
        for (fds[1..4]) |fd| {
            const resp = try testRecvAll(fd, &solo_resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
            try std.testing.expect(std.mem.indexOf(u8, resp, "data: [DONE]\n\n") != null);
        }

        const deadline = monotonicNowNs() + 2 * std.time.ns_per_s;
        while ((srv.stats().active_requests > 0 or srv.stats().active_conns > 0) and monotonicNowNs() < deadline) {
            sleepNs(1 * std.time.ns_per_ms);
        }
        try std.testing.expectEqual(@as(usize, 0), srv.stats().active_requests);
        try std.testing.expectEqual(@as(usize, 0), srv.stats().active_conns);
        try std.testing.expect(srv.stats().cancelled_count >= before_cancels + 1);
        try std.testing.expect(batch_mock.unpinned_count.load(.acquire) >= before_unpins + 4);
    }

    // 2e. Zero steady-state `malloc_zone_statistics` delta under 8-client simultaneous batched SSE load!
    //     Drive 8 non-blocking loopback sockets simultaneously from this thread (0 pthread spawns).
    {
        batch_mock.step_delay_ns = 1 * std.time.ns_per_ms;
        var req_bytes_buf: [1024]u8 = undefined;
        const req_bytes = try std.fmt.bufPrint(
            &req_bytes_buf,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ body_trng.len, body_trng },
        );

        // Wait until all prior slots are completely idle.
        while (srv.stats().active_conns > 0 or srv.stats().active_requests > 0) {
            sleepNs(500_000);
        }

        var m_warm: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &m_warm);
        var m_before: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &m_before);

        var sock_fds: [8]std.c.fd_t = .{ -1, -1, -1, -1, -1, -1, -1, -1 };
        var done_flags: [8]bool = @splat(false);
        var recv_totals: [8]usize = @splat(0);
        var scratch: [2048]u8 = undefined;

        for (&sock_fds) |*fd| {
            fd.* = try testConnectLoopback(port);
            try testSendAll(fd.*, req_bytes);
            try setNonBlocking(fd.*);
        }

        var remaining: usize = 8;
        const deadline = monotonicNowNs() + 5 * std.time.ns_per_s;
        while (remaining > 0 and monotonicNowNs() < deadline) {
            var progressed = false;
            for (&sock_fds, 0..) |*fd, i| {
                if (done_flags[i]) continue;
                while (true) {
                    const n = std.c.recv(fd.*, &scratch, scratch.len, 0);
                    if (n > 0) {
                        recv_totals[i] += @intCast(n);
                        progressed = true;
                    } else if (n == 0) {
                        _ = std.c.close(fd.*);
                        fd.* = -1;
                        done_flags[i] = true;
                        remaining -= 1;
                        progressed = true;
                        break;
                    } else {
                        const err = std.c.errno(n);
                        if (err == .INTR) continue;
                        if (err == .AGAIN) break;
                        return error.TestUnexpectedResult;
                    }
                }
            }
            if (!progressed) sleepNs(200_000);
        }
        try std.testing.expectEqual(@as(usize, 0), remaining);
        for (recv_totals) |n| try std.testing.expect(n > 100);

        while (srv.stats().active_conns > 0 or srv.stats().active_requests > 0) {
            sleepNs(500_000);
        }

        var m_after: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &m_after);
        try std.testing.expect(m_after.blocks_in_use <= m_before.blocks_in_use);
        try std.testing.expect(m_after.size_in_use <= m_before.size_in_use);
    }
}

test "C14 DomainNgramRegistry HTTP endpoints (GET /v1/ngram-packs, POST /v1/ngram-packs/select), per-request ngram_domain routing, and live hot-swap" {
    const gpa = std.testing.allocator;

    var p_repo1_buf: [256]u8 = undefined;
    var p_repo2_buf: [256]u8 = undefined;
    var p_docs_buf: [256]u8 = undefined;
    const p_repo1 = try testTmpAngPathServer(&p_repo1_buf, "aminfer-test-http-repo-v1.ang");
    const p_repo2 = try testTmpAngPathServer(&p_repo2_buf, "aminfer-test-http-repo-v2.ang");
    const p_docs = try testTmpAngPathServer(&p_docs_buf, "aminfer-test-http-docs.ang");
    defer _ = std.c.unlink(p_repo1.ptr);
    defer _ = std.c.unlink(p_repo2.ptr);
    defer _ = std.c.unlink(p_docs.ptr);

    const ctx_4 = [_]u32{ 10, 20, 30, 40 };
    try ngram_nvme.writeTableFile(gpa, p_repo1, &[_]ngram_nvme.BuildEntry{
        .{ .ctx = &ctx_4, .cont = &[_]u32{ 'R', '1' }, .hit_count = 50, .step_prob = 0.92 },
    }, .{ .num_buckets = 4, .corpus_id = "repo-v1", .include_table_crc = true });
    try ngram_nvme.writeTableFile(gpa, p_repo2, &[_]ngram_nvme.BuildEntry{
        .{ .ctx = &ctx_4, .cont = &[_]u32{ 'R', '2' }, .hit_count = 99, .step_prob = 0.96 },
    }, .{ .num_buckets = 4, .corpus_id = "repo-v2", .include_table_crc = true });
    try ngram_nvme.writeTableFile(gpa, p_docs, &[_]ngram_nvme.BuildEntry{
        .{ .ctx = &ctx_4, .cont = &[_]u32{ 'D', '1' }, .hit_count = 40, .step_prob = 0.88 },
    }, .{ .num_buckets = 4, .corpus_id = "docs-v1", .include_table_crc = true });

    var reg = try ngram_nvme.DomainNgramRegistry.init(gpa, .{
        .cache_pages = 8,
        .max_inflight = 2,
        .io_backend = .kqueue_aio,
        .verify_table_crc = true,
    });
    defer reg.deinit(gpa);
    _ = try reg.registerPack("repo", p_repo1, true);
    _ = try reg.registerPack("docs", p_docs, true);

    const NgramBatchBackend = struct {
        reg: *ngram_nvme.DomainNgramRegistry,

        fn tokenize(_: *anyopaque, _: *const InferenceJob, out: []u32) batch_sched.BackendError!usize {
            out[0] = 10;
            out[1] = 20;
            out[2] = 30;
            out[3] = 40;
            return 4;
        }

        fn detokenize(_: *anyopaque, tok: u32, out: []u8) []const u8 {
            out[0] = @intCast(tok & 0x7F);
            return out[0..1];
        }

        fn prefill(ctx_ptr: *anyopaque, chunk: *const batch_sched.PrefillChunkView) batch_sched.BackendError!batch_sched.PrefillResult {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            var cont: [8]u32 = undefined;
            const dom: ?[]const u8 = if (chunk.ngram_domain.len > 0) chunk.ngram_domain else null;
            const res = self.reg.queryDomain(dom, chunk.tokens, &cont) catch return error.PrefillFailed;
            return .{ .first_token = if (res.draft_len >= 1) cont[0] else '?' };
        }

        fn decode(ctx_ptr: *anyopaque, view: *const batch_sched.BatchStepView, out: []u32) batch_sched.BackendError!void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            const fixed_ctx = [_]u32{ 10, 20, 30, 40 };
            for (0..view.batchSize()) |b| {
                var cont: [8]u32 = undefined;
                const dom: ?[]const u8 = if (view.ngram_domains[b].len > 0) view.ngram_domains[b] else null;
                const res = self.reg.queryDomain(dom, &fixed_ctx, &cont) catch return error.DecodeFailed;
                out[b] = if (res.draft_len >= 2) cont[1] else '!';
            }
        }

        fn release(_: *anyopaque, _: u16, _: u64) void {}
    };

    var nb = NgramBatchBackend{ .reg = &reg };
    const srv = try Server.init(gpa, .{
        .port = 0,
        .model_id = "c14-ngram-srv",
        .max_conns = 8,
        .max_active_requests = 8,
        .max_batch_size = 8,
        .ngram_registry = &reg,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try srv.startBatched(.{
        .ctx = &nb,
        .tokenizeFn = NgramBatchBackend.tokenize,
        .detokenizeFn = NgramBatchBackend.detokenize,
        .backend = .{
            .ctx = &nb,
            .prefillSlotFn = NgramBatchBackend.prefill,
            .stepDecodeBatchFn = NgramBatchBackend.decode,
            .releaseSlotFn = NgramBatchBackend.release,
        },
    });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [1024]u8 = undefined;
    var resp_buf: [8192]u8 = undefined;

    // 1. GET /v1/ngram-packs lists both loaded packs with active default "repo".
    {
        const resp = try testHttpExchange(port, "GET /v1/ngram-packs HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const body = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(body);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"default_domain\":\"repo\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"pack_count\":2") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"domain\":\"repo\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"domain\":\"docs\"") != null);
    }

    // 2. Per-request `"ngram_domain"` routing: default -> "R1", explicit "docs" -> "D1", unknown -> 400 JSON.
    {
        const r_def = try testHttpPostJson(port, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],\"stream\":false,\"max_tokens\":2}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_def, "HTTP/1.1 200 OK\r\n"));
        const b_def = extractHttpBody(r_def) orelse return error.TestUnexpectedResult;
        try expectValidJson(b_def);
        try std.testing.expect(std.mem.indexOf(u8, b_def, "\"content\":\"R1\"") != null);

        const r_docs = try testHttpPostJson(port, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],\"stream\":false,\"max_tokens\":2,\"ngram_domain\":\"docs\"}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_docs, "HTTP/1.1 200 OK\r\n"));
        const b_docs = extractHttpBody(r_docs) orelse return error.TestUnexpectedResult;
        try expectValidJson(b_docs);
        try std.testing.expect(std.mem.indexOf(u8, b_docs, "\"content\":\"D1\"") != null);

        const r_bad = try testHttpPostJson(port, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],\"stream\":false,\"max_tokens\":2,\"ngram_domain\":\"nope\"}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_bad, "HTTP/1.1 400 Bad Request\r\n"));
        try expectValidJson(extractHttpBody(r_bad) orelse return error.TestUnexpectedResult);
    }

    // 3. POST /v1/ngram-packs/select switches active default live to "docs" (+ validates error paths).
    {
        const r_sel = try testHttpPostJson(port, "/v1/ngram-packs/select", "{\"domain\":\"docs\"}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_sel, "HTTP/1.1 200 OK\r\n"));
        const b_sel = extractHttpBody(r_sel) orelse return error.TestUnexpectedResult;
        try expectValidJson(b_sel);
        try std.testing.expect(std.mem.indexOf(u8, b_sel, "\"default_domain\":\"docs\"") != null);

        // Request omitting `"ngram_domain"` now resolves to "docs" -> "D1"!
        const r_now_docs = try testHttpPostJson(port, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],\"stream\":false,\"max_tokens\":2}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_now_docs, "HTTP/1.1 200 OK\r\n"));
        const b_now_docs = extractHttpBody(r_now_docs) orelse return error.TestUnexpectedResult;
        try expectValidJson(b_now_docs);
        try std.testing.expect(std.mem.indexOf(u8, b_now_docs, "\"content\":\"D1\"") != null);

        // Error paths all return valid JSON!
        const r_405a = try testHttpPostJson(port, "/v1/ngram-packs", "{}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_405a, "HTTP/1.1 405 Method Not Allowed\r\n"));
        try expectValidJson(extractHttpBody(r_405a) orelse return error.TestUnexpectedResult);

        const r_405b = try testHttpExchange(port, "GET /v1/ngram-packs/select HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_405b, "HTTP/1.1 405 Method Not Allowed\r\n"));
        try expectValidJson(extractHttpBody(r_405b) orelse return error.TestUnexpectedResult);

        const r_404 = try testHttpPostJson(port, "/v1/ngram-packs/select", "{\"domain\":\"unknown\"}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_404, "HTTP/1.1 404 Not Found\r\n"));
        try expectValidJson(extractHttpBody(r_404) orelse return error.TestUnexpectedResult);

        const r_400 = try testHttpPostJson(port, "/v1/ngram-packs/select", "{\"wrong\":1}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_400, "HTTP/1.1 400 Bad Request\r\n"));
        try expectValidJson(extractHttpBody(r_400) orelse return error.TestUnexpectedResult);
    }

    // 4. Live hot-swap `swapDomainPack("repo", p_repo2)` reflected immediately on explicit "repo" & GET /v1/ngram-packs!
    {
        _ = try reg.swapDomainPack("repo", p_repo2, true);
        const r_v2 = try testHttpPostJson(port, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],\"stream\":false,\"max_tokens\":2,\"ngram_domain\":\"repo\"}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_v2, "HTTP/1.1 200 OK\r\n"));
        const b_v2 = extractHttpBody(r_v2) orelse return error.TestUnexpectedResult;
        try expectValidJson(b_v2);
        try std.testing.expect(std.mem.indexOf(u8, b_v2, "\"content\":\"R2\"") != null);

        const r_list = try testHttpExchange(port, "GET /v1/ngram-packs HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        const b_list = extractHttpBody(r_list) orelse return error.TestUnexpectedResult;
        try expectValidJson(b_list);
        try std.testing.expect(std.mem.indexOf(u8, b_list, "\"corpus_id\":\"repo-v2\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, b_list, "\"swaps\":1") != null);
        try std.testing.expect(std.mem.indexOf(u8, b_list, "\"default_domain\":\"docs\"") != null);
    }

    // 5. Verify zero steady-state malloc allocations on GET /v1/ngram-packs and POST /v1/ngram-packs/select!
    {
        while (srv.stats().active_conns > 0 or srv.stats().active_requests > 0) {
            sleepNs(500_000);
        }
        var m_warm: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &m_warm);
        var m_before: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &m_before);

        for (0..4) |k| {
            const sel_json = if ((k & 1) == 0) "{\"domain\":\"repo\"}" else "{\"domain\":\"docs\"}";
            const rs = try testHttpPostJson(port, "/v1/ngram-packs/select", sel_json, &req_buf, &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, rs, "HTTP/1.1 200 OK\r\n"));
            const rl = try testHttpExchange(port, "GET /v1/ngram-packs HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, rl, "HTTP/1.1 200 OK\r\n"));
        }

        while (srv.stats().active_conns > 0 or srv.stats().active_requests > 0) {
            sleepNs(500_000);
        }
        var m_after: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &m_after);
        try std.testing.expect(m_after.blocks_in_use <= m_before.blocks_in_use);
        try std.testing.expect(m_after.size_in_use <= m_before.size_in_use);
    }
}

test "C15 parseInferenceRequest extracts exact last_user_msg_byte_offset across chat/completions, responses, and messages" {
    var prompt_buf: [2048]u8 = undefined;
    var model_buf: [64]u8 = undefined;

    // 1. /v1/chat/completions: 4-turn conversation ending in `user` -> offset right before final `<|im_start|>user\n`.
    {
        const body =
            \\{"messages":[
            \\  {"role":"system","content":"You are helpful."},
            \\  {"role":"user","content":"First question"},
            \\  {"role":"assistant","content":"First answer"},
            \\  {"role":"user","content":"Edited second question"}
            \\]}
        ;
        const p = try parseInferenceRequest(.chat_completions, body, "m", &prompt_buf, &model_buf);
        const prompt = prompt_buf[0..p.prompt_len];
        const off = p.last_user_msg_byte_offset orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings(
            "<|im_start|>user\nEdited second question<|im_end|>\n<|im_start|>assistant\n",
            prompt[off..],
        );
        try std.testing.expectEqualStrings(
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nFirst question<|im_end|>\n<|im_start|>assistant\nFirst answer<|im_end|>\n",
            prompt[0..off],
        );
    }

    // 2. /v1/chat/completions: single user turn (< 2 messages) -> null.
    {
        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Only turn\"}]}";
        const p = try parseInferenceRequest(.chat_completions, body, "m", &prompt_buf, &model_buf);
        try std.testing.expectEqual(@as(?usize, null), p.last_user_msg_byte_offset);
    }

    // 3. /v1/chat/completions: final turn is `assistant` (not `user`) -> null.
    {
        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"Hello\"}]}";
        const p = try parseInferenceRequest(.chat_completions, body, "m", &prompt_buf, &model_buf);
        try std.testing.expectEqual(@as(?usize, null), p.last_user_msg_byte_offset);
    }

    // 4. /v1/responses: top-level `instructions` + string `input` (>= 2 turns ending in user).
    {
        const body = "{\"instructions\":\"Be concise.\",\"input\":\"Compute 2+2\"}";
        const p = try parseInferenceRequest(.responses, body, "m", &prompt_buf, &model_buf);
        const prompt = prompt_buf[0..p.prompt_len];
        const off = p.last_user_msg_byte_offset orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings("<|im_start|>system\nBe concise.<|im_end|>\n", prompt[0..off]);
        try std.testing.expect(std.mem.startsWith(u8, prompt[off..], "<|im_start|>user\nCompute 2+2<|im_end|>\n"));
    }

    // 5. /v1/responses: multi-turn `input` array without instructions.
    {
        const body = "{\"input\":[{\"role\":\"user\",\"content\":\"Turn 1\"},{\"role\":\"assistant\",\"content\":\"Ack\"},{\"role\":\"user\",\"content\":\"Turn 2\"}]}";
        const p = try parseInferenceRequest(.responses, body, "m", &prompt_buf, &model_buf);
        const prompt = prompt_buf[0..p.prompt_len];
        const off = p.last_user_msg_byte_offset orelse return error.TestUnexpectedResult;
        try std.testing.expect(std.mem.startsWith(u8, prompt[off..], "<|im_start|>user\nTurn 2<|im_end|>\n"));
    }

    // 6. /v1/responses: bare single string `input` without `instructions` -> null.
    {
        const body = "{\"input\":\"Single turn\"}";
        const p = try parseInferenceRequest(.responses, body, "m", &prompt_buf, &model_buf);
        try std.testing.expectEqual(@as(?usize, null), p.last_user_msg_byte_offset);
    }

    // 7. /v1/messages: top-level `system` + `messages` ending in user, plus multi-turn messages.
    {
        const body = "{\"system\":\"System policy\",\"messages\":[{\"role\":\"user\",\"content\":\"Question A\"}]}";
        const p = try parseInferenceRequest(.messages, body, "m", &prompt_buf, &model_buf);
        const prompt = prompt_buf[0..p.prompt_len];
        const off = p.last_user_msg_byte_offset orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings("<|im_start|>system\nSystem policy<|im_end|>\n", prompt[0..off]);
        try std.testing.expect(std.mem.startsWith(u8, prompt[off..], "<|im_start|>user\nQuestion A<|im_end|>\n"));
    }
    {
        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Q1\"},{\"role\":\"assistant\",\"content\":\"A1\"},{\"role\":\"user\",\"content\":\"Q2\"}]}";
        const p = try parseInferenceRequest(.messages, body, "m", &prompt_buf, &model_buf);
        const prompt = prompt_buf[0..p.prompt_len];
        const off = p.last_user_msg_byte_offset orelse return error.TestUnexpectedResult;
        try std.testing.expect(std.mem.startsWith(u8, prompt[off..], "<|im_start|>user\nQ2<|im_end|>\n"));
    }
    {
        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Only user\"}]}";
        const p = try parseInferenceRequest(.messages, body, "m", &prompt_buf, &model_buf);
        try std.testing.expectEqual(@as(?usize, null), p.last_user_msg_byte_offset);
    }
}

test "C15 Server forwards last_user_msg_byte_offset through BatchEngineHandler to BatchScheduler checkpoint split" {
    const gpa = std.testing.allocator;

    const BoundaryServerMock = struct {
        saw_byte_offset: std.atomic.Value(usize) = .init(0),
        saw_boundary_checkpoint: std.atomic.Value(u32) = .init(0),
        checkpoint_end_pos: std.atomic.Value(u32) = .init(0),

        fn tokenize(_: *anyopaque, job: *const InferenceJob, out_tokens: []u32) batch_sched.BackendError!usize {
            const n = @min(job.prompt.len, out_tokens.len);
            for (0..n) |i| out_tokens[i] = @as(u32, job.prompt[i]) + 1;
            return n;
        }

        // Byte-level 1:1 tokenizer mapping `job.last_user_msg_byte_offset` directly to exact token index.
        fn tokenizeWithBoundary(
            ctx_raw: *anyopaque,
            job: *const InferenceJob,
            out_tokens: []u32,
        ) batch_sched.BackendError!TokenizeResult {
            const self: *@This() = @ptrCast(@alignCast(ctx_raw));
            const n = @min(job.prompt.len, out_tokens.len);
            for (0..n) |i| out_tokens[i] = @as(u32, job.prompt[i]) + 1;
            var boundary: ?u32 = null;
            if (job.last_user_msg_byte_offset) |off| {
                self.saw_byte_offset.store(off, .release);
                if (off > 0 and off < n) boundary = @intCast(off);
            }
            return .{
                .prompt_len = n,
                .message_boundary_token = boundary,
            };
        }

        fn detokenize(_: *anyopaque, _: u32, out_bytes: []u8) []const u8 {
            out_bytes[0] = 'X';
            return out_bytes[0..1];
        }

        fn prefill(ctx_raw: *anyopaque, chunk: *const batch_sched.PrefillChunkView) batch_sched.BackendError!batch_sched.PrefillResult {
            const self: *@This() = @ptrCast(@alignCast(ctx_raw));
            if (chunk.is_message_boundary_checkpoint) {
                _ = self.saw_boundary_checkpoint.fetchAdd(1, .acq_rel);
                self.checkpoint_end_pos.store(chunk.start_pos + @as(u32, @intCast(chunk.tokens.len)), .release);
            }
            return .{ .first_token = if (chunk.is_final_chunk) 10 else 0 };
        }

        fn decode(_: *anyopaque, view: *const batch_sched.BatchStepView, out: []u32) batch_sched.BackendError!void {
            for (0..view.batchSize()) |b| out[b] = 11;
        }
        fn release(_: *anyopaque, _: u16, _: u64) void {}

        fn handler(self: *@This()) BatchEngineHandler {
            return .{
                .ctx = self,
                .tokenizeFn = tokenize,
                .tokenizeWithBoundaryFn = tokenizeWithBoundary,
                .detokenizeFn = detokenize,
                .backend = .{
                    .ctx = self,
                    .prefillSlotFn = prefill,
                    .stepDecodeBatchFn = decode,
                    .releaseSlotFn = release,
                },
            };
        }
    };

    var mock = BoundaryServerMock{};
    const srv = try Server.init(gpa, .{
        .port = 0,
        .model_id = "c15-boundary",
        .max_conns = 4,
        .max_active_requests = 4,
        .max_batch_size = 4,
        .max_prefill_chunk_tokens = 512,
        .max_prompt_tokens = 1024,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try srv.startBatched(mock.handler());
    defer srv.stop();

    var req_buf: [1024]u8 = undefined;
    var resp_buf: [4096]u8 = undefined;
    const body = "{\"messages\":[{\"role\":\"system\",\"content\":\"System rules\"},{\"role\":\"user\",\"content\":\"Hello C15\"}],\"stream\":false,\"max_tokens\":2}";
    const resp = try testHttpPostJson(srv.boundPort(), "/v1/chat/completions", body, &req_buf, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));

    const expected_off: usize = "<|im_start|>system\nSystem rules<|im_end|>\n".len;
    try std.testing.expectEqual(expected_off, mock.saw_byte_offset.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), mock.saw_boundary_checkpoint.load(.acquire));
    try std.testing.expectEqual(@as(u32, @intCast(expected_off)), mock.checkpoint_end_pos.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), srv.stats().message_boundary_checkpoints);

    // Turning `message_boundary_cache = false` must suppress boundary splitting.
    var mock_off = BoundaryServerMock{};
    const srv_off = try Server.init(gpa, .{
        .port = 0,
        .model_id = "c15-boundary-off",
        .max_conns = 4,
        .max_active_requests = 4,
        .max_batch_size = 4,
        .max_prefill_chunk_tokens = 512,
        .max_prompt_tokens = 1024,
        .message_boundary_cache = false,
        .apply_qos = false,
    });
    defer srv_off.deinit(gpa);
    try srv_off.startBatched(mock_off.handler());
    defer srv_off.stop();

    const resp_off = try testHttpPostJson(srv_off.boundPort(), "/v1/chat/completions", body, &req_buf, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, resp_off, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expectEqual(@as(u32, 0), mock_off.saw_boundary_checkpoint.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), srv_off.stats().message_boundary_checkpoints);

    // Invalid `decode_share` must fail `Server.init`.
    try std.testing.expectError(error.InvalidConfig, Server.init(gpa, .{ .port = 0, .decode_share = -0.25 }));
    try std.testing.expectError(error.InvalidConfig, Server.init(gpa, .{ .port = 0, .decode_share = 1.50 }));
}

fn testConnectUnix(path: []const u8) TestClientError!std.c.fd_t {
    if (path.len == 0 or path.len > MAX_UNIX_SOCKET_PATH_BYTES) return error.ConnectFailed;
    const fd = std.c.socket(std.c.AF.UNIX, std.c.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    errdefer _ = std.c.close(fd);
    var addr = std.c.sockaddr.un{
        .path = @splat(0),
    };
    @memcpy(addr.path[0..path.len], path);
    if (std.c.connect(fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.un)) != 0) {
        return error.ConnectFailed;
    }
    return fd;
}

fn testUnixExchange(path: []const u8, raw_request: []const u8, resp_buf: []u8) TestClientError![]const u8 {
    const fd = try testConnectUnix(path);
    defer _ = std.c.close(fd);
    try testSendAll(fd, raw_request);
    return testRecvAll(fd, resp_buf);
}

test "C18 end-to-end tool calling SSE & non-streaming across /v1/chat/completions, /v1/responses, and /v1/messages + multi-turn history" {
    const ToolMock = struct {
        saw_grammar: std.atomic.Value(u32) = .init(0),
        saw_lora_match: std.atomic.Value(u32) = .init(0),
        saw_tool_history: std.atomic.Value(u32) = .init(0),
        saw_cache_pin: std.atomic.Value(u32) = .init(0),

        fn generate(ctx_ptr: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            if (job.grammar_program != null) {
                _ = self.saw_grammar.fetchAdd(1, .acq_rel);
            }
            if (job.lora_adapter) |ad| {
                if (std.mem.eql(u8, job.model_base, "qwen3-8b") and std.mem.eql(u8, ad, "coder")) {
                    _ = self.saw_lora_match.fetchAdd(1, .acq_rel);
                }
            }
            if (std.mem.indexOf(u8, job.prompt, "<tool_response>") != null) {
                _ = self.saw_tool_history.fetchAdd(1, .acq_rel);
            }
            if (job.cache_control_pin) {
                _ = self.saw_cache_pin.fetchAdd(1, .acq_rel);
            }

            _ = stream.setPromptTokensWithCached(20, 12);
            // Fragment `<think>` and `<tool_call>` arbitrarily across tiny byte chunks!
            const chunks = [_][]const u8{
                "<thi",      "nk>Plan ",     "weather</th",   "ink>",
                "<tool_",    "call>\n{\"na", "me\":\"get_",   "weather\",\"arg",
                "uments\":", "{\"city\":",   "\"Kyoto\"}}\n", "</tool_call>",
            };
            for (chunks) |ch| {
                if (!stream.emitTokenBytes(ch)) return;
            }
            stream.finish(.stop, 20);
        }
    };

    var mock = ToolMock{};
    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-8b",
        .max_conns = 8,
        .max_active_requests = 4,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &mock, .generateFn = ToolMock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [4096]u8 = undefined;
    var resp_buf: [24 * 1024]u8 = undefined;

    // 1. OpenAI `/v1/chat/completions` streaming tool calls + W19 `"qwen3-8b:coder"` + `stream_options.include_usage` + `role:"tool"` history!
    {
        const body =
            \\{"model":"qwen3-8b:coder","tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false},"strict":true}}],"tool_choice":{"type":"function","function":{"name":"get_weather"}},"parallel_tool_calls":false,"stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Weather?"},{"role":"assistant","content":null,"tool_calls":[{"id":"c0","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Osaka\"}"}}]},{"role":"tool","tool_call_id":"c0","content":"22C"},{"role":"user","content":"Now Kyoto"}]}
        ;
        const resp = try testHttpPostJson(port, "/v1/chat/completions", body, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const sse = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        _ = try expectValidSseJsonFrames(sse);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"tool_calls\":[{\"index\":0") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"name\":\"get_weather\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"finish_reason\":\"tool_calls\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"choices\":[],\"usage\":{") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"cached_tokens\":12") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"reasoning_tokens\":") != null);
    }

    // 2. OpenAI `/v1/responses` full SSE lifecycle (`response.created` .. `response.function_call_arguments.delta` .. `response.completed`) + `function_call_output`!
    {
        const body =
            \\{"model":"qwen3-8b","tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"tool_choice":"required","reasoning":{"effort":"high"},"input":[{"role":"user","content":"Hi"},{"type":"function_call_output","call_id":"c1","output":"Sunny"}],"stream":true}
        ;
        const resp = try testHttpPostJson(port, "/v1/responses", body, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const sse = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        _ = try expectValidSseJsonFrames(sse);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.created\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.in_progress\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.output_item.added\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.function_call_arguments.delta\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.function_call_arguments.done\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.output_item.done\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.completed\n") != null);
    }

    // 3. Anthropic `/v1/messages` streaming `signature_delta`, `tool_use`, `input_json_delta`, `stop_reason: "tool_use"`, `cache_control`, `cache_read_input_tokens`, and `tool_result` + `is_error`!
    {
        const body =
            \\{"system":[{"type":"text","text":"Be helpful","cache_control":{"type":"ephemeral"}}],"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"tool_choice":{"type":"any"},"thinking":{"type":"enabled","budget_tokens":1024},"max_tokens":64,"stream":true,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","is_error":true,"content":"timeout"}]}]}
        ;
        var raw_req_w = FixedWriter.init(&req_buf);
        try raw_req_w.print(
            "POST /v1/messages HTTP/1.1\r\nHost: 127.0.0.1\r\nanthropic-version: 2023-06-01\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ body.len, body },
        );
        const resp = try testHttpExchange(port, raw_req_w.written(), &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const sse = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        _ = try expectValidSseJsonFrames(sse);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"cache_read_input_tokens\":12") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"type\":\"signature_delta\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"type\":\"tool_use\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"type\":\"input_json_delta\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"stop_reason\":\"tool_use\"") != null);
    }

    try std.testing.expectEqual(@as(u32, 3), mock.saw_grammar.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), mock.saw_lora_match.load(.acquire));
    try std.testing.expectEqual(@as(u32, 3), mock.saw_tool_history.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), mock.saw_cache_pin.load(.acquire));
}

test "C18 stateful /v1/responses chaining, GET, DELETE, POST /cancel, and bounded evict_first LRU" {
    const StatefulMock = struct {
        saw_chained_prompt: std.atomic.Value(u32) = .init(0),
        hold_open: std.atomic.Value(bool) = .init(false),
        entered_hold: std.atomic.Value(bool) = .init(false),

        fn generate(ctx_ptr: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            if (std.mem.indexOf(u8, job.prompt, "Turn1Reply") != null) {
                _ = self.saw_chained_prompt.fetchAdd(1, .acq_rel);
            }
            _ = stream.setPromptTokens(8);
            if (self.hold_open.load(.acquire)) {
                self.entered_hold.store(true, .release);
                while (!stream.isCancelled()) {
                    sleepNs(1_000_000);
                }
                return;
            }
            if (std.mem.indexOf(u8, job.prompt, "Turn1") != null and std.mem.indexOf(u8, job.prompt, "Turn2") == null) {
                _ = stream.emitTokenBytes("Turn1Reply");
            } else {
                _ = stream.emitTokenBytes("NextReply");
            }
            stream.finish(.stop, 8);
        }
    };

    var mock = StatefulMock{};
    // Fixed table of 2 slots to verify bounded `evict_first` LRU without heap growth!
    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-stateful",
        .max_conns = 8,
        .max_active_requests = 4,
        .stateful_response_slots = 2,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &mock, .generateFn = StatefulMock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [2048]u8 = undefined;
    var resp_buf: [16384]u8 = undefined;

    // 1. Create `resp_1`
    const r1 = try testHttpPostJson(port, "/v1/responses", "{\"input\":\"Turn1\",\"stream\":false}", &req_buf, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r1, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(std.mem.indexOf(u8, r1, "\"id\":\"resp_1\"") != null);

    // 2. GET `/v1/responses/resp_1`
    {
        const g1 = try testHttpExchange(port, "GET /v1/responses/resp_1 HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, g1, "HTTP/1.1 200 OK\r\n"));
        const body = extractHttpBody(g1) orelse return error.TestUnexpectedResult;
        try expectValidJson(body);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"output_text\":\"Turn1Reply\"") != null);
    }

    // 3. Chain `resp_2` referencing `previous_response_id: "resp_1"` -> verify prior transcript injected!
    {
        const r2 = try testHttpPostJson(port, "/v1/responses", "{\"input\":\"Turn2\",\"previous_response_id\":\"resp_1\",\"stream\":false}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r2, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, r2, "\"previous_response_id\":\"resp_1\"") != null);
        try std.testing.expectEqual(@as(u32, 1), mock.saw_chained_prompt.load(.acquire));
    }

    // 4. Create `resp_3` -> evicts oldest LRU (`resp_1`), keeping `resp_2` & `resp_3`!
    {
        const r3 = try testHttpPostJson(port, "/v1/responses", "{\"input\":\"Turn3\",\"stream\":false}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r3, "HTTP/1.1 200 OK\r\n"));
        const g1_gone = try testHttpExchange(port, "GET /v1/responses/resp_1 HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, g1_gone, "HTTP/1.1 404 Not Found\r\n"));
        const g2_alive = try testHttpExchange(port, "GET /v1/responses/resp_2 HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, g2_alive, "HTTP/1.1 200 OK\r\n"));
    }

    // 5. DELETE `/v1/responses/resp_2` -> returns `deleted: true`, subsequent GET returns 404.
    {
        const d2 = try testHttpExchange(port, "DELETE /v1/responses/resp_2 HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, d2, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, d2, "\"deleted\":true") != null);
        const g2_gone = try testHttpExchange(port, "GET /v1/responses/resp_2 HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, g2_gone, "HTTP/1.1 404 Not Found\r\n"));
    }

    // 6. Live `POST /v1/responses/resp_4/cancel` cancels an in-flight streaming response!
    mock.hold_open.store(true, .release);
    const holder_fd = try testConnectLoopback(port);
    defer _ = std.c.close(holder_fd);
    const slow_body = "{\"input\":\"SlowTurn\",\"stream\":true}";
    var sw = FixedWriter.init(&req_buf);
    try sw.print(
        "POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
        .{ slow_body.len, slow_body },
    );
    try testSendAll(holder_fd, sw.written());
    var waited: usize = 0;
    while (!mock.entered_hold.load(.acquire) and waited < 200) : (waited += 1) {
        sleepNs(2_000_000);
    }
    try std.testing.expect(mock.entered_hold.load(.acquire));

    const c4 = try testHttpExchange(port, "POST /v1/responses/resp_4/cancel HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n", &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, c4, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(std.mem.indexOf(u8, c4, "\"status\":\"cancelled\"") != null);
}

test "C18 /v1/messages/count_tokens (native W5 Tokenizer), C21 /v1/embeddings & /v1/rerank, StopHoldback SSE, --api-key auth, and AF_UNIX --unix-socket" {
    // 1. Minimal valid Hugging Face tokenizer JSON (`W5`) for native zero-Python count_tokens!
    const mini_tok_json =
        \\{"model":{"type":"BPE","vocab":{"<|im_start|>":0,"<|im_end|>":1,"a":2,"b":3,"c":4,"d":5,"e":6,"f":7,"g":8,"h":9,"i":10,"j":11,"k":12,"l":13,"m":14,"n":15,"o":16,"p":17,"q":18,"r":19,"s":20,"t":21,"u":22,"v":23,"w":24,"x":25,"y":26,"z":27," ":28,"\n":29,"Ġ":30,"Ċ":31,"he":32,"ll":33,"lo":34,"hell":35,"hello":36},"merges":["h e","l l","l o","he ll","hell o"]},"added_tokens":[{"id":0,"content":"<|im_start|>","special":true},{"id":1,"content":"<|im_end|>","special":true}]}
    ;
    var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(std.testing.allocator, mini_tok_json);
    defer tok.deinit(std.testing.allocator);

    const Handlers = struct {
        fn generate(_: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(4);
            // Emit text containing stop sequence `STOP_HERE` split across chunks + trailing text that must NEVER leak!
            const parts = [_][]const u8{ "alpha ", "beta ", "STO", "P_HE", "RE_LEAK_FORBIDDEN" };
            for (parts) |p| {
                if (!stream.emitTokenBytes(p)) return;
            }
            _ = job;
            stream.finish(.stop, 4);
        }

        fn embed(_: *anyopaque, req: *const EmbedRequest, out_vectors: []f32) anyerror!EmbedResult {
            const dim: usize = if (req.dimensions) |d| @intCast(d) else 4;
            for (0..req.input_count) |i| {
                for (0..dim) |j| {
                    out_vectors[i * dim + j] = @as(f32, @floatFromInt(i + 1)) * 0.25 + @as(f32, @floatFromInt(j)) * 0.1;
                }
            }
            return .{ .dim = dim, .prompt_tokens = @intCast(req.input_count * 3) };
        }

        fn rerank(_: *anyopaque, req: *const RerankRequest, out_items: []RerankItem) anyerror!RerankResult {
            // Rank documents in reverse index order for deterministic check
            for (0..req.document_count) |i| {
                const doc_idx: u32 = @intCast(req.document_count - 1 - i);
                out_items[i] = .{
                    .index = doc_idx,
                    .relevance_score = 1.0 - @as(f32, @floatFromInt(i)) * 0.25,
                };
            }
            return .{ .count = req.document_count, .prompt_tokens = 11 };
        }
    };

    var uds_path_buf: [128]u8 = undefined;
    var uw = FixedWriter.init(&uds_path_buf);
    try uw.print("/tmp/aminfer_m75_test_{d}.sock", .{std.c.getpid()});
    const uds_path = uw.written();

    var dummy: u8 = 0;
    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-full",
        .api_key = "sk-aminfer-secret-42",
        .unix_socket_path = uds_path,
        .tokenizer = &tok,
        .embedding_handler = .{ .ctx = &dummy, .embedFn = Handlers.embed },
        .rerank_handler = .{ .ctx = &dummy, .rerankFn = Handlers.rerank },
        .max_conns = 8,
        .max_active_requests = 4,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &dummy, .generateFn = Handlers.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [2048]u8 = undefined;
    var resp_buf: [16384]u8 = undefined;

    // 1. `/health` works unauthenticated, while `/v1/models` rejects missing/wrong key with 401!
    {
        const h = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, h, "HTTP/1.1 200 OK\r\n"));

        const unauth = try testHttpExchange(port, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, unauth, "HTTP/1.1 401 Unauthorized\r\n"));

        const wrong = try testHttpExchange(port, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-wrong\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, wrong, "HTTP/1.1 401 Unauthorized\r\n"));

        const ok_bearer = try testHttpExchange(port, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-aminfer-secret-42\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, ok_bearer, "HTTP/1.1 200 OK\r\n"));

        const ok_xkey = try testHttpExchange(port, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nx-api-key: sk-aminfer-secret-42\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, ok_xkey, "HTTP/1.1 200 OK\r\n"));
    }

    // 2. Live `AF_UNIX` (`--unix-socket`) IPC request on the same kqueue loop!
    {
        const u_resp = try testUnixExchange(uds_path, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-aminfer-secret-42\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, u_resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, u_resp, "\"id\":\"qwen3-full\"") != null);
    }

    // 3. `POST /v1/messages/count_tokens` backed directly by native Zig `Tokenizer` (`W5`)!
    {
        const cbody = "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}";
        var rw = FixedWriter.init(&req_buf);
        try rw.print(
            "POST /v1/messages/count_tokens HTTP/1.1\r\nHost: localhost\r\nx-api-key: sk-aminfer-secret-42\r\nanthropic-version: 2023-06-01\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ cbody.len, cbody },
        );
        const resp = try testHttpExchange(port, rw.written(), &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const body = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(body);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"input_tokens\":") != null);
        // Must count > 0 tokens natively!
        try std.testing.expect(std.mem.indexOf(u8, body, "\"input_tokens\":0") == null);
    }

    // 4. Streaming `stop` sequence holdback (`STOP_HERE`) never leaks matched sequence or trailing bytes!
    {
        const sbody = "{\"messages\":[{\"role\":\"user\",\"content\":\"go\"}],\"stop\":[\"STOP_HERE\"],\"stream\":true}";
        var rw = FixedWriter.init(&req_buf);
        try rw.print(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-aminfer-secret-42\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ sbody.len, sbody },
        );
        const resp = try testHttpExchange(port, rw.written(), &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const sse = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        _ = try expectValidSseJsonFrames(sse);
        var joined_buf: [256]u8 = undefined;
        const joined = extractOpenAiSseContent(sse, &joined_buf);
        try std.testing.expectEqualStrings("alpha beta ", joined);
        try std.testing.expect(std.mem.indexOf(u8, sse, "STOP_HERE") == null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "LEAK_FORBIDDEN") == null);
    }

    // 5. `POST /v1/embeddings` (`float` & `base64`, `dimensions`, W19 model tag) + `POST /v1/rerank` (`top_n`, `return_documents`)!
    {
        const ebody = "{\"model\":\"qwen3-embed:lora1\",\"input\":[\"doc a\",\"doc b\"],\"dimensions\":2,\"encoding_format\":\"float\"}";
        var rw = FixedWriter.init(&req_buf);
        try rw.print(
            "POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-aminfer-secret-42\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ ebody.len, ebody },
        );
        const eresp = try testHttpExchange(port, rw.written(), &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, eresp, "HTTP/1.1 200 OK\r\n"));
        const ej = extractHttpBody(eresp) orelse return error.TestUnexpectedResult;
        try expectValidJson(ej);
        try std.testing.expect(std.mem.indexOf(u8, ej, "\"object\":\"embedding\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, ej, "\"model\":\"qwen3-embed:lora1\"") != null);

        // Base64 encoding format check
        const ebody_b64 = "{\"input\":\"vec\",\"dimensions\":2,\"encoding_format\":\"base64\"}";
        var rw2 = FixedWriter.init(&req_buf);
        try rw2.print(
            "POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-aminfer-secret-42\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ ebody_b64.len, ebody_b64 },
        );
        const eresp_b64 = try testUnixExchange(uds_path, rw2.written(), &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, eresp_b64, "HTTP/1.1 200 OK\r\n"));
        const ej_b64 = extractHttpBody(eresp_b64) orelse return error.TestUnexpectedResult;
        try expectValidJson(ej_b64);
        try std.testing.expect(std.mem.indexOf(u8, ej_b64, "\"embedding\":\"") != null);

        // Rerank check
        const rbody = "{\"model\":\"qwen3-rerank\",\"query\":\"apple silicon\",\"documents\":[\"d0\",\"d1\",\"d2\"],\"top_n\":2,\"return_documents\":true}";
        var rw3 = FixedWriter.init(&req_buf);
        try rw3.print(
            "POST /v1/rerank HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-aminfer-secret-42\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ rbody.len, rbody },
        );
        const rresp = try testHttpExchange(port, rw3.written(), &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, rresp, "HTTP/1.1 200 OK\r\n"));
        const rj = extractHttpBody(rresp) orelse return error.TestUnexpectedResult;
        try expectValidJson(rj);
        try std.testing.expect(std.mem.indexOf(u8, rj, "\"index\":2,\"relevance_score\":1.000000,\"document\":{\"text\":\"d2\"}") != null);
        try std.testing.expect(std.mem.indexOf(u8, rj, "\"index\":0") == null); // trimmed by top_n=2!
    }
}

fn scalarFindJsonStringSpecialRef(bytes: []const u8) usize {
    for (bytes, 0..) |b, i| {
        if (b == '"' or b == '\\' or b < 0x20) return i;
    }
    return bytes.len;
}

fn scalarFindQuoteOrBackslashRef(bytes: []const u8) usize {
    for (bytes, 0..) |b, i| {
        if (b == '"' or b == '\\') return i;
    }
    return bytes.len;
}

fn scalarSkipJsonWhitespaceRef(bytes: []const u8) usize {
    for (bytes, 0..) |b, i| {
        if (b != ' ' and b != '\t' and b != '\r' and b != '\n') return i;
    }
    return bytes.len;
}

fn scalarWriteJsonEscapedRef(writer: *FixedWriter, bytes: []const u8) error{BufferTooSmall}!void {
    const hex = "0123456789abcdef";
    for (bytes) |b| {
        switch (b) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            0x08 => try writer.writeAll("\\b"),
            0x0c => try writer.writeAll("\\f"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0x00...0x07, 0x0b, 0x0e...0x1f => {
                const esc = [6]u8{ '\\', 'u', '0', '0', hex[b >> 4], hex[b & 0x0f] };
                try writer.writeAll(&esc);
            },
            else => try writer.writeByte(b),
        }
    }
}

test "C23 ARM64 NEON JSON/HTTP scanners match scalar bit-for-bit across 0..129 offsets, escapes, UTF-8, and huge tool-call payloads" {
    // 1. Exhaustive 0..129 length & match-offset sweep (covers <16 B tail, 16..63 B single-vec, 64..129 B 4x unrolled).
    var buf: [160]u8 = undefined;
    for (0..130) |len| {
        // Fill with mix of printable ASCII and high-bit UTF-8 bytes (0x80..0xFE) to ensure unsigned `< 0x20` compare.
        for (0..len) |i| {
            buf[i] = switch (i % 5) {
                0 => 'a',
                1 => 0xE4, // UTF-8 CJK lead byte (>= 0x80, must NOT trigger < 0x20!)
                2 => 0xB8, // UTF-8 continuation
                3 => 0xAD,
                else => 'Z',
            };
        }
        // Test unaligned sub-slice starts too.
        const base = buf[0..len];
        try std.testing.expectEqual(scalarFindJsonStringSpecialRef(base), neonFindJsonStringSpecial(base));
        try std.testing.expectEqual(scalarFindQuoteOrBackslashRef(base), neonFindQuoteOrBackslash(base));
        try std.testing.expectEqual(len, neonFindByte(base, '"'));

        // Place each special byte category at every position `0..len` (sampled densely around SIMD boundaries).
        if (len > 0) {
            const specials = [_]u8{ '"', '\\', 0x00, 0x08, 0x09, 0x0A, 0x0D, 0x1F };
            for (0..len) |pos| {
                // Sweep all positions for len <= 72, and boundary-rich positions for len > 72.
                if (len > 72 and pos >= 18 and pos + 18 < len and (pos < 60 or pos > 70)) continue;
                const saved = buf[pos];
                for (specials) |sp| {
                    buf[pos] = sp;
                    const s = buf[0..len];
                    try std.testing.expectEqual(scalarFindJsonStringSpecialRef(s), neonFindJsonStringSpecial(s));
                    try std.testing.expectEqual(scalarFindQuoteOrBackslashRef(s), neonFindQuoteOrBackslash(s));
                    try std.testing.expectEqual(
                        std.mem.indexOfScalar(u8, s, sp) orelse len,
                        neonFindByte(s, sp),
                    );
                }
                buf[pos] = saved;
            }
        }
    }

    // 2. Whitespace scanner parity across 0..129 lengths and all 4 RFC 8259 whitespace bytes + non-ws bytes (0x00, 0x0B, 0x0C, 0x80, 'x').
    const ws_chars = [_]u8{ ' ', '\t', '\r', '\n' };
    const non_ws_chars = [_]u8{ 'x', '{', '"', 0x00, 0x0B, 0x0C, 0xA0, 0xFF };
    for (0..130) |len| {
        for (0..len) |i| buf[i] = ws_chars[i % ws_chars.len];
        try std.testing.expectEqual(scalarSkipJsonWhitespaceRef(buf[0..len]), neonSkipJsonWhitespace(buf[0..len]));
        for (0..len) |pos| {
            if (len > 72 and pos >= 18 and pos + 18 < len and (pos < 60 or pos > 70)) continue;
            const saved = buf[pos];
            for (non_ws_chars) |nw| {
                buf[pos] = nw;
                try std.testing.expectEqual(scalarSkipJsonWhitespaceRef(buf[0..len]), neonSkipJsonWhitespace(buf[0..len]));
            }
            buf[pos] = saved;
        }
    }

    // 3. HTTP CRLF & CRLFCRLF scanner parity across false `\r` triggers and 0..129 offsets.
    for (0..130) |len| {
        @memset(buf[0..len], 'H');
        if (len >= 5) buf[len / 3] = '\r'; // stray CR without LF
        try std.testing.expectEqual(std.mem.indexOf(u8, buf[0..len], "\r\n"), neonFindCrlf(buf[0..len]));
        try std.testing.expectEqual(std.mem.indexOf(u8, buf[0..len], "\r\n\r\n"), neonFindCrlfCrlf(buf[0..len]));
        if (len >= 4) {
            const p = len - 4;
            @memcpy(buf[p .. p + 4], "\r\n\r\n");
            try std.testing.expectEqual(std.mem.indexOf(u8, buf[0..len], "\r\n"), neonFindCrlf(buf[0..len]));
            try std.testing.expectEqual(std.mem.indexOf(u8, buf[0..len], "\r\n\r\n"), neonFindCrlfCrlf(buf[0..len]));
        }
    }

    // 4. Escaped quotes, UTF-8 CJK + emoji, control bytes, and round-trip parity (`writeJsonEscaped` -> `JsonReader.readStringInto` / `skipString`).
    const rich_payload =
        "Prefix 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef " ++
        "escaped \"quote\" and \\backslash\\ and \\\"nested\\\" " ++
        "中文日本語한국어 🚀🔥🧠 " ++
        "ctrl:\x00\x07\x08\x09\x0a\x0c\x0d\x1f " ++
        "suffix_tail_longer_than_64_bytes_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    var neon_esc_buf: [2048]u8 = undefined;
    var scalar_esc_buf: [2048]u8 = undefined;
    var wn = FixedWriter.init(&neon_esc_buf);
    var ws = FixedWriter.init(&scalar_esc_buf);
    try wn.writeByte('"');
    try wn.writeJsonEscaped(rich_payload);
    try wn.writeByte('"');
    try ws.writeByte('"');
    try scalarWriteJsonEscapedRef(&ws, rich_payload);
    try ws.writeByte('"');
    try std.testing.expectEqualStrings(ws.written(), wn.written());

    var unesc_buf: [2048]u8 = undefined;
    var uw = FixedWriter.init(&unesc_buf);
    var jr = JsonReader{ .src = wn.written() };
    try jr.readStringInto(&uw);
    try std.testing.expectEqualStrings(rich_payload, uw.written());
    try std.testing.expectEqual(wn.written().len, jr.pos);

    var jr_skip = JsonReader{ .src = wn.written() };
    try jr_skip.skipString();
    try std.testing.expectEqual(wn.written().len, jr_skip.pos);

    // Raw unescaped control byte inside JSON string must still be rejected at > 64 B offset.
    var bad_json_str: [128]u8 = @splat('a');
    bad_json_str[0] = '"';
    bad_json_str[95] = 0x1F;
    bad_json_str[127] = '"';
    var bad_out: [128]u8 = undefined;
    var bad_w = FixedWriter.init(&bad_out);
    var bad_r = JsonReader{ .src = &bad_json_str };
    try std.testing.expectError(error.InvalidJson, bad_r.readStringInto(&bad_w));
    var bad_r2 = JsonReader{ .src = &bad_json_str };
    try std.testing.expectError(error.InvalidJson, bad_r2.skipString());

    // 5. Huge (> 16 KiB) indented tool-call JSON payload parsed end-to-end with zero heap allocation.
    const gpa = std.testing.allocator;
    const huge_json = try gpa.alloc(u8, 32 * 1024);
    defer gpa.free(huge_json);
    var hw = FixedWriter.init(huge_json);
    for (0..20) |_| try hw.writeAll("    \t\r\n"); // > 128 B leading whitespace
    try hw.writeAll("{\"model\":\"qwen3.8\",\"messages\":[{\"role\":\"user\",\"content\":\"");
    for (0..180) |_| {
        try hw.writeJsonEscaped("def compute(x: \"str\") -> str:\n\treturn \"结果🚀:\\\"ok\\\"\" # 0123456789abcdef\n");
    }
    try hw.writeAll("\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"run_sql\",\"description\":\"");
    for (0..80) |_| {
        try hw.writeAll("Execute analytical query over unified memory pool with zero copies; ");
    }
    try hw.writeAll("\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}}}}}],\"stream\":false}");
    try std.testing.expect(hw.written().len > 16 * 1024);

    // `findJsonFieldValueStart` must locate fields accurately even after huge escaped strings.
    const stream_pos = findJsonFieldValueStart(hw.written(), "stream") orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.startsWith(u8, hw.written()[stream_pos..], "false"));

    const prompt_buf = try gpa.alloc(u8, 64 * 1024);
    defer gpa.free(prompt_buf);
    var model_buf: [MAX_MODEL_ID_BYTES]u8 = undefined;
    const parsed = try parseInferenceRequest(.chat_completions, hw.written(), "default", prompt_buf, &model_buf);
    try std.testing.expectEqualStrings("qwen3.8", model_buf[0..parsed.model_len]);
    try std.testing.expectEqual(@as(u8, 1), parsed.tool_count);
    try std.testing.expectEqualStrings("run_sql", parsed.toolsSlice()[0].nameSlice());
    try std.testing.expect(parsed.prompt_len > 10 * 1024);
}
