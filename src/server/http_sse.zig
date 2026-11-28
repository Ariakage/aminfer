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
const paths = @import("../platform/paths.zig");
const grammar_mod = @import("../spec/grammar.zig");
const sampling_mod = @import("../spec/sampling.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const qos = @import("../platform/qos.zig");
const chat_template = @import("../model/chat_template.zig");
const amf = @import("../format/amf.zig");
const lora_mod = @import("../model/lora.zig");
const tls_mod = @import("../bridge/tls.zig");
const HmacSha256 = std.crypto.auth.hmac.sha2.HmacSha256;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// Apple Silicon page size for backing slab alignment.
pub const APPLE_PAGE_SIZE: usize = 16384;
/// Maximum byte length of a per-request `"ngram_domain"` identifier.
pub const MAX_NGRAM_DOMAIN_BYTES: usize = ngram_nvme.MAX_DOMAIN_ID_BYTES;

pub const DEFAULT_MAX_CONNS: usize = 32;
pub const DEFAULT_MAX_ACTIVE_REQUESTS: usize = 16;
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 64 * 1024;
pub const DEFAULT_MAX_PROMPT_BYTES: usize = 64 * 1024;
pub const DEFAULT_TX_BUF_BYTES: usize = 256 * 1024;
pub const DEFAULT_NONSTREAM_REASONING_BYTES: usize = 64 * 1024;
pub const DEFAULT_NONSTREAM_CONTENT_BYTES: usize = 128 * 1024;
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 256 * 1024;
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
pub const DEFAULT_STATEFUL_RESPONSE_SLOTS: usize = 64;
pub const DEFAULT_STATEFUL_RESPONSE_PROMPT_BYTES: usize = 64 * 1024;
pub const DEFAULT_STATEFUL_RESPONSE_CONTENT_BYTES: usize = 64 * 1024;
pub const DEFAULT_STATEFUL_RESPONSE_REASONING_BYTES: usize = 32 * 1024;
pub const DEFAULT_STATEFUL_RESPONSE_INPUT_ITEMS_BYTES: usize = 16 * 1024;
pub const MAX_STATEFUL_INPUT_ITEMS: usize = 16;
pub const MAX_DISPLAY_NAME_BYTES: usize = 96;
pub const MAX_CORS_ORIGIN_BYTES: usize = 256;
pub const MAX_FIM_TOKEN_BYTES: usize = 64;
pub const MAX_COMPLETION_TOKEN_IDS: usize = 1024;
pub const MAX_MEDIA_ITEMS_PER_REQUEST: usize = 4;
pub const DEFAULT_MAX_MEDIA_BYTES: usize = 4 * 1024 * 1024;
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
} || tls_mod.TlsError;

pub const ParseError = error{
    InvalidJson,
    MissingRequiredField,
    InvalidFieldValue,
    UnsupportedFeature,
    BufferTooSmall,
    DepthExceeded,
    RemoteMediaNotSupported,
    InvalidMediaBase64,
    TooManyMediaItems,
    MediaTooLarge,
};

pub const Endpoint = enum(u8) {
    chat_completions = 0,
    responses = 1,
    messages = 2,
    completions = 3,
};

pub const JobKind = enum(u8) {
    generate = 0,
    embeddings = 1,
    rerank = 2,
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

pub const ReasoningEffort = chat_template.ReasoningEffort;

pub const ReasoningSummaryMode = enum(u8) {
    auto = 0,
    detailed = 1,
    concise = 2,
    none = 3,

    pub fn parseString(raw: []const u8) error{InvalidFieldValue}!ReasoningSummaryMode {
        const s = std.mem.trim(u8, raw, " \t\r\n");
        if (std.mem.eql(u8, s, "auto")) return .auto;
        if (std.mem.eql(u8, s, "detailed")) return .detailed;
        if (std.mem.eql(u8, s, "concise")) return .concise;
        if (std.mem.eql(u8, s, "none")) return .none;
        return error.InvalidFieldValue;
    }

    pub fn name(self: ReasoningSummaryMode) []const u8 {
        return switch (self) {
            .auto => "auto",
            .detailed => "detailed",
            .concise => "concise",
            .none => "none",
        };
    }
};

fn isSentenceTerm(b: u8) bool {
    return b == '.' or b == '!' or b == '?' or b == '\n';
}

/// Zero-alloc O(head + tail) CPU sentence extractor for `reasoning.summary = "concise"`.
pub fn extractConciseReasoningSummary(raw: []const u8, out: []u8) []const u8 {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0 or out.len == 0) return "";
    if (trimmed.len <= 160 and trimmed.len <= out.len) {
        @memcpy(out[0..trimmed.len], trimmed);
        return out[0..trimmed.len];
    }

    // Head: up to end of first sentence, capped at 128 bytes.
    const head_limit = @min(trimmed.len, 128);
    var head_end: usize = head_limit;
    for (0..head_limit) |idx| {
        if (isSentenceTerm(trimmed[idx])) {
            if (idx > 0) {
                head_end = idx + 1;
                while (head_end < head_limit and isSentenceTerm(trimmed[head_end])) : (head_end += 1) {}
                break;
            }
        }
    }
    const head = std.mem.trimEnd(u8, trimmed[0..head_end], " \t\r\n");

    // Tail: up to last sentence, capped at 128 bytes.
    const tail_floor = if (trimmed.len > 128) trimmed.len - 128 else 0;
    var scan = trimmed.len;
    while (scan > tail_floor and isSentenceTerm(trimmed[scan - 1])) : (scan -= 1) {}
    var tail_start: usize = tail_floor;
    var j = scan;
    while (j > tail_floor) {
        j -= 1;
        if (isSentenceTerm(trimmed[j])) {
            tail_start = j + 1;
            break;
        }
    }
    const tail = std.mem.trim(u8, trimmed[tail_start..], " \t\r\n");

    if (head_end >= tail_start or tail.len == 0) {
        const take = @min(trimmed.len, out.len);
        @memcpy(out[0..take], trimmed[0..take]);
        return out[0..take];
    }

    var w = FixedWriter.init(out);
    w.writeAll(head) catch {
        const take = @min(head.len, out.len);
        @memcpy(out[0..take], head[0..take]);
        return out[0..take];
    };
    w.writeAll(" ... ") catch return w.written();
    w.writeAll(tail) catch {
        const rem = w.remaining();
        if (rem > 0) {
            const take = @min(tail.len, rem);
            @memcpy(out[w.pos .. w.pos + take], tail[0..take]);
            w.pos += take;
        }
        return w.written();
    };
    return w.written();
}

/// Initializes an HMAC-SHA256 state bound to `model_id` (`u32` LE length prefix
/// + `model_id` bytes) so Anthropic thinking signatures cannot replay across
/// models sharing `--api-key` (`ADR-0024` L69).
pub fn initThinkingHmac(key: *const [32]u8, model_id: []const u8) HmacSha256 {
    var mac = HmacSha256.init(key);
    var len_le: [4]u8 = undefined;
    std.mem.writeInt(u32, &len_le, @intCast(model_id.len), .little);
    mac.update(&len_le);
    mac.update(model_id);
    return mac;
}

pub fn computeThinkingSignature(
    key: *const [32]u8,
    model_id: []const u8,
    thinking: []const u8,
    out_hex: *[64]u8,
) []const u8 {
    var mac = initThinkingHmac(key, model_id);
    mac.update(thinking);
    var tag: [32]u8 = undefined;
    mac.final(&tag);
    out_hex.* = std.fmt.bytesToHex(tag, .lower);
    return out_hex[0..];
}

pub fn verifyThinkingSignature(
    key: *const [32]u8,
    model_id: []const u8,
    thinking: []const u8,
    sig_hex: []const u8,
) bool {
    if (sig_hex.len != 64) return false;
    var provided: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&provided, sig_hex) catch return false;
    var mac = initThinkingHmac(key, model_id);
    mac.update(thinking);
    var expected: [32]u8 = undefined;
    mac.final(&expected);
    return std.crypto.timing_safe.eql([32]u8, expected, provided);
}

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
    description_json: []const u8 = "",

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
    timestamp_ns: u64 = 0,
    text_len: u16 = 0,
    text: [MAX_CHUNK_TEXT_BYTES]u8 = undefined,

    pub fn textSlice(self: *const TokenEvent) []const u8 {
        return self.text[0..self.text_len];
    }
};

pub const ModelCapabilities = struct {
    chat: bool = true,
    responses: bool = true,
    messages: bool = true,
    tool_calling: bool = true,
    structured_outputs: bool = true,
    reasoning: bool = false,
    vision: bool = false,
    embeddings: bool = false,
    rerank: bool = false,
    lora: bool = false,
    speculative_decoding: bool = false,
    moe: bool = false,
};

pub const ModelDetailMeta = struct {
    architecture: []const u8 = "unknown",
    display_name: []const u8 = "",
    max_model_len: u32 = 0,
    max_output_tokens: u32 = 0,
    vocab_size: u32 = 0,
    effective_bits_per_weight: f64 = 0.0,
    wired_mib: u64 = 0,
    cold_bytes: u64 = 0,
    capabilities: ModelCapabilities = .{},

    pub fn fromInspection(
        insp: *const amf.AmfInspection,
        max_seq_len: u32,
        caps_override: ?ModelCapabilities,
    ) ModelDetailMeta {
        const arch = insp.archSlice();
        var caps = caps_override orelse ModelCapabilities{};
        if (insp.arch.has_moe) caps.moe = true;
        if (insp.arch.has_vision) caps.vision = true;
        return .{
            .architecture = if (arch.len > 0) arch else "unknown",
            .max_model_len = if (max_seq_len > 0) max_seq_len else insp.arch.max_position_embeddings,
            .vocab_size = insp.arch.vocab_size,
            .effective_bits_per_weight = insp.effective_bits_per_weight,
            .wired_mib = insp.estimated_total_wired_bytes >> 20,
            .cold_bytes = insp.cold_weight_bytes + insp.sparse_table_bytes,
            .capabilities = caps,
        };
    }
};

pub const LoraAdapterMeta = struct {
    name: []const u8,
    display_name: []const u8 = "",
    rank: u32,
    lora_alpha: f32,
    scale: f32,
    wired_bytes: u64 = 0,

    pub fn populateFromRegistry(
        reg: *const lora_mod.LoraRegistry,
        out: *[lora_mod.MAX_ADAPTERS]LoraAdapterMeta,
    ) usize {
        var count: usize = 0;
        for (&reg.slots) |*slot| {
            if (!slot.in_use) continue;
            out[count] = .{
                .name = slot.nameSlice(),
                .rank = slot.rank,
                .lora_alpha = slot.lora_alpha,
                .scale = slot.scale,
                .wired_bytes = slot.wired_bytes,
            };
            count += 1;
        }
        return count;
    }
};

pub const MediaKind = enum(u8) {
    image = 0,
    video = 1,
};

pub const MediaAttachment = struct {
    kind: MediaKind,
    mime_buf: [32]u8 = @splat(0),
    mime_len: u8 = 0,
    data: []const u8,

    pub fn mimeSlice(self: *const MediaAttachment) []const u8 {
        return self.mime_buf[0..self.mime_len];
    }
};

pub const FimTokens = struct {
    prefix: []const u8 = "<|fim_prefix|>",
    suffix: []const u8 = "<|fim_suffix|>",
    middle: []const u8 = "<|fim_middle|>",
};

/// Formats a Unix epoch timestamp (seconds) as RFC 3339 UTC (`YYYY-MM-DDTHH:MM:SSZ`)
/// using Howard Hinnant's civil-from-days integer math with zero heap allocations.
pub fn formatRfc3339Utc(unix_sec: i64, out: *[32]u8) []const u8 {
    const clamped: i64 = if (unix_sec < 0) 0 else unix_sec;
    const days: i64 = @divFloor(clamped, 86400);
    const rem_sec: u64 = @intCast(@mod(clamped, 86400));
    const hour: u64 = rem_sec / 3600;
    const minute: u64 = (rem_sec % 3600) / 60;
    const second: u64 = rem_sec % 60;

    const z: i64 = days + 719468;
    const era: i64 = @divFloor(z, 146097);
    const doe: u64 = @intCast(z - era * 146097);
    const yoe: u64 = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const y: i64 = @as(i64, @intCast(yoe)) + era * 400;
    const doy: u64 = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const mp: u64 = (5 * doy + 2) / 153;
    const day: u64 = doy - (153 * mp + 2) / 5 + 1;
    const month: u64 = if (mp < 10) mp + 3 else mp - 9;
    const year_i: i64 = y + @as(i64, if (month <= 2) 1 else 0);
    const year: u64 = @intCast(@max(@as(i64, 0), @min(year_i, 9999)));

    return std.fmt.bufPrint(out, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        year,
        month,
        day,
        hour,
        minute,
        second,
    }) catch "1970-01-01T00:00:00Z";
}

fn hexNibble(c: u8) ?u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => null,
    };
}

/// Decodes `%XX` URL percent-encoded bytes into `out` without heap allocation.
/// Literal `'+'` stays `'+'`; `%00`, truncated `%`, and non-hex escapes return
/// `error.InvalidPercentEncoding`.
pub fn percentDecodeUrlPath(
    raw: []const u8,
    out: []u8,
) error{ InvalidPercentEncoding, BufferTooSmall }![]const u8 {
    var out_len: usize = 0;
    var overflow = false;
    var i: usize = 0;
    while (i < raw.len) {
        const ch = raw[i];
        const decoded: u8 = if (ch == '%') blk: {
            if (i + 2 >= raw.len) return error.InvalidPercentEncoding;
            const hi = hexNibble(raw[i + 1]) orelse return error.InvalidPercentEncoding;
            const lo = hexNibble(raw[i + 2]) orelse return error.InvalidPercentEncoding;
            const val: u8 = (hi << 4) | lo;
            if (val == 0) return error.InvalidPercentEncoding;
            i += 3;
            break :blk val;
        } else blk: {
            if (ch == 0) return error.InvalidPercentEncoding;
            i += 1;
            break :blk ch;
        };

        if (out_len < out.len) {
            out[out_len] = decoded;
            out_len += 1;
        } else {
            overflow = true;
        }
    }
    if (overflow) return error.BufferTooSmall;
    return out[0..out_len];
}

pub const ServerTlsConfig = struct {
    cert_bytes: []const u8,
    key_bytes: ?[]const u8 = null,
    passphrase: ?[]const u8 = null,
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
    chat_template: chat_template.ChatTemplate = chat_template.ChatTemplate.preset(.chatml),
    max_response_bytes: ?usize = null,
    http_read_timeout_ms: u32 = 30_000,
    /// Shorter deadline (ms) for receiving HTTP request line + headers (`\r\n\r\n`, `L3`).
    header_read_timeout_ms: u32 = 5_000,
    keepalive_idle_timeout_ms: u32 = 60_000,
    /// Maximum concurrent connections allowed from a single non-loopback IPv4 peer (`0` = unlimited, `L3`).
    max_conns_per_peer: usize = 8,
    /// Explicit override permitting unauthenticated non-loopback TCP binds (`F2`).
    allow_unauthenticated_public_bind: bool = false,
    /// Explicit override bypassing TCP `Host` allowlist validation when `api_key == null` (`F2`).
    allow_any_host: bool = false,
    /// Optional native Zig BPE `Tokenizer` (`W5`) backing `POST /v1/messages/count_tokens`.
    tokenizer: ?*const tokenizer_mod.Tokenizer = null,
    /// Optional `C21` embeddings backend callback for `POST /v1/embeddings`.
    embedding_handler: ?EmbeddingHandler = null,
    /// Optional `C21` rerank backend callback for `POST /v1/rerank`.
    rerank_handler: ?RerankHandler = null,
    /// Require `anthropic-version` header on `/v1/messages*` when set to `true` (default `false`
    /// keeps legacy clients working while still validating non-empty header format when provided).
    require_anthropic_version: bool = false,
    /// Optional model metadata reported on `GET /v1/models` and `GET /v1/models/{id}` (`C28`).
    model_detail: ?ModelDetailMeta = null,
    /// Optional mounted LoRA adapter metadata reported alongside the base model (`C28`).
    lora_adapters: []const LoraAdapterMeta = &.{},
    /// Optional native macOS `Network.framework` TLS 1.2/1.3 listener configuration (`C18`).
    tls: ?ServerTlsConfig = null,
    /// Optional CORS `Access-Control-Allow-Origin` header value (`null` or `""` disables CORS, `F2`/`C29`).
    cors_allow_origin: ?[]const u8 = null,
    /// Configurable Fill-in-the-Middle delimiters for `POST /v1/completions` when `suffix` is non-empty (`C29`).
    fim_tokens: FimTokens = .{},
    /// Per-connection preallocated decoded media buffer capacity carved from `arena_slab` at `Server.init` (`C29`).
    max_media_bytes: usize = DEFAULT_MAX_MEDIA_BYTES,
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
    kind: JobKind = .generate,
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
    template_dialect: chat_template.Dialect = .chatml,
    starts_in_think: bool = false,
    stream_include_usage: bool = false,
    /// Optional pre-tokenized completion prompt token IDs (`POST /v1/completions` with `[u32, ...]`, `C29`).
    prompt_token_ids: ?[]const u32 = null,
    /// Role-isolated `(len, allow_special)` spans covering `prompt` (`F13`).
    prompt_spans: ?[]const chat_template.PromptSpan = null,
    /// Preallocated decoded multimodal attachments pointing into `slot.media_buf` (`C29`).
    media_items: [MAX_MEDIA_ITEMS_PER_REQUEST]MediaAttachment = undefined,
    media_count: u8 = 0,

    pub fn mediaSlice(self: *const InferenceJob) []const MediaAttachment {
        return self.media_items[0..self.media_count];
    }
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

    fn writeSanitizedUtf8(self: *FixedWriter, chunk: []const u8) error{BufferTooSmall}!void {
        if (std.unicode.utf8ValidateSlice(chunk)) {
            try self.writeAll(chunk);
            return;
        }
        var pos: usize = 0;
        while (pos < chunk.len) {
            if (chunk[pos] < 0x80) {
                const ascii_start = pos;
                while (pos < chunk.len and chunk[pos] < 0x80) : (pos += 1) {}
                try self.writeAll(chunk[ascii_start..pos]);
                continue;
            }
            const seq_len = Utf8StreamAssembler.utf8ExpectedLen(chunk[pos]);
            if (seq_len > 0 and pos + seq_len <= chunk.len and Utf8StreamAssembler.isValidUtf8Sequence(chunk[pos .. pos + seq_len])) {
                try self.writeAll(chunk[pos .. pos + seq_len]);
                pos += seq_len;
            } else {
                try self.writeAll("\xEF\xBF\xBD");
                pos += 1;
            }
        }
    }

    /// Writes `bytes` escaped for inside a JSON string literal (without outer quotes).
    /// Vectorized with 16-byte / 64-byte ARM64 NEON scanning (`C23`, ADR-0023),
    /// replacing any malformed UTF-8 sequences with U+FFFD (`L8`).
    pub fn writeJsonEscaped(self: *FixedWriter, bytes: []const u8) error{BufferTooSmall}!void {
        const hex = "0123456789abcdef";
        var run_start: usize = 0;
        while (run_start < bytes.len) {
            const span = neonFindJsonStringSpecial(bytes[run_start..]);
            if (span > 0) {
                try self.writeSanitizedUtf8(bytes[run_start .. run_start + span]);
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

pub const StreamChannel = chat_template.StreamChannel;
pub const ThinkSplitter = chat_template.ThinkSplitter;

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
    const OPEN_INKLING = "<|content_invoke_tool_json|>";
    const CLOSE_INKLING = "<|end_message|>";
    const OPEN_HY4 = "<tool_call:opensource>";
    const CLOSE_HY4 = "</tool_call:opensource>";
    const WRAP_HY4_OPEN = "<tool_calls:opensource>";
    const WRAP_HY4_CLOSE = "</tool_calls:opensource>";
    const INKLING_MODEL_TAG = "<|message_model|>";
    const INKLING_TEXT_TAG = "<|content_text|>";
    const INKLING_EOS_TAG = "<|content_model_end_sampling|>";
    const MAX_TAG_LEN: usize = 32;

    const ActiveFormat = enum(u8) {
        qwen = 0,
        inkling2 = 1,
        hy4 = 2,
    };

    const Mode = enum(u8) {
        outside = 0,
        inside_preamble = 1,
        streaming_args = 2,
        inside_tail = 3,
        inside_hy4_xml = 4,
        inkling_header = 5,
    };

    dialect: chat_template.Dialect = .chatml,
    active_format: ActiveFormat = .qwen,
    mode: Mode = .outside,
    pending: [MAX_TAG_LEN]u8 = undefined,
    pending_len: u8 = 0,
    header_buf: [MAX_TOOL_NAME_BYTES]u8 = undefined,
    header_len: u8 = 0,
    has_header_name: bool = false,
    preamble_buf: [512]u8 = undefined,
    preamble_len: u16 = 0,
    hy4_buf: [2048]u8 = undefined,
    hy4_len: u16 = 0,
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
        self.resetForDialect(request_id, .chatml);
    }

    pub fn resetForDialect(self: *ToolCallParser, request_id: u64, dialect: chat_template.Dialect) void {
        self.dialect = dialect;
        self.active_format = .qwen;
        self.mode = .outside;
        self.pending_len = 0;
        self.header_len = 0;
        self.has_header_name = false;
        self.preamble_len = 0;
        self.hy4_len = 0;
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

    inline fn activeCloseTag(self: *const ToolCallParser) []const u8 {
        return switch (self.active_format) {
            .qwen => CLOSE_TAG,
            .inkling2 => CLOSE_INKLING,
            .hy4 => CLOSE_HY4,
        };
    }

    fn matchOutsideTag(self: *const ToolCallParser, cand: []const u8) enum {
        none,
        prefix,
        open_qwen,
        open_inkling,
        open_hy4,
        inkling_model,
        strip_tag,
    } {
        if (std.mem.eql(u8, cand, OPEN_TAG)) return .open_qwen;
        if (std.mem.eql(u8, cand, OPEN_INKLING)) return .open_inkling;
        if (std.mem.eql(u8, cand, OPEN_HY4)) return .open_hy4;

        const strip_hy4 = (self.dialect == .hy4);
        const strip_ink = (self.dialect == .inkling2);
        if (strip_hy4 and (std.mem.eql(u8, cand, WRAP_HY4_OPEN) or std.mem.eql(u8, cand, WRAP_HY4_CLOSE))) return .strip_tag;
        if (strip_ink and std.mem.eql(u8, cand, INKLING_MODEL_TAG)) return .inkling_model;
        if (strip_ink and (std.mem.eql(u8, cand, INKLING_TEXT_TAG) or std.mem.eql(u8, cand, INKLING_EOS_TAG) or std.mem.eql(u8, cand, CLOSE_INKLING))) return .strip_tag;

        if ((cand.len < OPEN_TAG.len and std.mem.startsWith(u8, OPEN_TAG, cand)) or
            (cand.len < OPEN_INKLING.len and std.mem.startsWith(u8, OPEN_INKLING, cand)) or
            (cand.len < OPEN_HY4.len and std.mem.startsWith(u8, OPEN_HY4, cand)) or
            (strip_hy4 and ((cand.len < WRAP_HY4_OPEN.len and std.mem.startsWith(u8, WRAP_HY4_OPEN, cand)) or
                (cand.len < WRAP_HY4_CLOSE.len and std.mem.startsWith(u8, WRAP_HY4_CLOSE, cand)))) or
            (strip_ink and ((cand.len < INKLING_MODEL_TAG.len and std.mem.startsWith(u8, INKLING_MODEL_TAG, cand)) or
                (cand.len < INKLING_TEXT_TAG.len and std.mem.startsWith(u8, INKLING_TEXT_TAG, cand)) or
                (cand.len < INKLING_EOS_TAG.len and std.mem.startsWith(u8, INKLING_EOS_TAG, cand)) or
                (cand.len < CLOSE_INKLING.len and std.mem.startsWith(u8, CLOSE_INKLING, cand)))))
        {
            return .prefix;
        }
        return .none;
    }

    fn beginOpenToolCall(self: *ToolCallParser, fmt: ActiveFormat) void {
        self.pending_len = 0;
        self.active_format = fmt;
        self.mode = if (fmt == .hy4) .inside_hy4_xml else .inside_preamble;
        self.preamble_len = 0;
        self.hy4_len = 0;
        if (fmt == .inkling2 and self.header_len > 0) {
            const trimmed = std.mem.trim(u8, self.header_buf[0..self.header_len], " \t\r\n");
            if (trimmed.len > 0) {
                const nlen = @min(trimmed.len, self.name_buf.len);
                @memcpy(self.name_buf[0..nlen], trimmed[0..nlen]);
                self.name_len = @intCast(nlen);
                self.has_header_name = true;
            } else {
                self.name_len = 0;
                self.has_header_name = false;
            }
        } else {
            self.name_len = 0;
            self.has_header_name = false;
        }
        self.header_len = 0;
        self.args_is_string = false;
        self.args_brace_depth = 0;
        self.args_in_string = false;
        self.args_escape = false;
        self.args_started = false;
        self.emitted_any_args = false;
    }

    pub fn feed(
        self: *ToolCallParser,
        chunk: []const u8,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        var i: usize = 0;
        while (i < chunk.len) {
            if (self.mode == .inkling_header and self.pending_len == 0) {
                if (std.mem.indexOfScalarPos(u8, chunk, i, '<')) |lt| {
                    const seg = chunk[i..lt];
                    if (self.header_len + seg.len <= self.header_buf.len) {
                        @memcpy(self.header_buf[self.header_len..][0..seg.len], seg);
                        self.header_len += @intCast(seg.len);
                        self.pending[0] = '<';
                        self.pending_len = 1;
                        i = lt + 1;
                    } else {
                        const prev_h = self.header_buf[0..self.header_len];
                        self.header_len = 0;
                        self.mode = .outside;
                        if (prev_h.len > 0) try emit_fn(ctx, ToolCallStreamEvent{ .text = prev_h });
                        if (seg.len > 0) try emit_fn(ctx, ToolCallStreamEvent{ .text = seg });
                        self.pending[0] = '<';
                        self.pending_len = 1;
                        i = lt + 1;
                    }
                } else {
                    const seg = chunk[i..];
                    if (self.header_len + seg.len <= self.header_buf.len) {
                        @memcpy(self.header_buf[self.header_len..][0..seg.len], seg);
                        self.header_len += @intCast(seg.len);
                        return;
                    }
                    const prev_h = self.header_buf[0..self.header_len];
                    self.header_len = 0;
                    self.mode = .outside;
                    if (prev_h.len > 0) try emit_fn(ctx, ToolCallStreamEvent{ .text = prev_h });
                    if (seg.len > 0) try emit_fn(ctx, ToolCallStreamEvent{ .text = seg });
                    return;
                }
                continue;
            }
            if (self.mode == .outside or self.mode == .inkling_header) {
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
                    if (self.pending_len < MAX_TAG_LEN) {
                        self.pending[self.pending_len] = b;
                        const cand = self.pending[0 .. self.pending_len + 1];
                        switch (self.matchOutsideTag(cand)) {
                            .prefix => {
                                self.pending_len += 1;
                                i += 1;
                            },
                            .open_qwen => {
                                i += 1;
                                self.beginOpenToolCall(.qwen);
                            },
                            .open_inkling => {
                                i += 1;
                                self.beginOpenToolCall(.inkling2);
                            },
                            .open_hy4 => {
                                i += 1;
                                self.beginOpenToolCall(.hy4);
                            },
                            .inkling_model => {
                                self.pending_len = 0;
                                self.header_len = 0;
                                self.mode = .inkling_header;
                                i += 1;
                            },
                            .strip_tag => {
                                self.pending_len = 0;
                                self.header_len = 0;
                                self.mode = .outside;
                                i += 1;
                            },
                            .none => {
                                if (self.mode == .inkling_header) {
                                    const prev_h = self.header_buf[0..self.header_len];
                                    self.header_len = 0;
                                    self.mode = .outside;
                                    if (prev_h.len > 0) try emit_fn(ctx, ToolCallStreamEvent{ .text = prev_h });
                                }
                                const flushed = self.pending[0..self.pending_len];
                                self.pending_len = 0;
                                try emit_fn(ctx, ToolCallStreamEvent{ .text = flushed });
                            },
                        }
                    } else {
                        if (self.mode == .inkling_header) {
                            const prev_h = self.header_buf[0..self.header_len];
                            self.header_len = 0;
                            self.mode = .outside;
                            if (prev_h.len > 0) try emit_fn(ctx, ToolCallStreamEvent{ .text = prev_h });
                        }
                        const flushed = self.pending[0..self.pending_len];
                        self.pending_len = 0;
                        try emit_fn(ctx, ToolCallStreamEvent{ .text = flushed });
                    }
                }
                continue;
            }

            // Inside active tool call: scan for `activeCloseTag()` across chunks.
            const b = chunk[i];
            i += 1;
            const ctag = self.activeCloseTag();
            if (self.pending_len == 0) {
                if (b == '<' and !self.args_in_string) {
                    self.pending[0] = '<';
                    self.pending_len = 1;
                } else {
                    try self.consumeInsideByte(b, ctx, emit_fn);
                }
            } else {
                if (self.pending_len < ctag.len and b == ctag[self.pending_len]) {
                    self.pending[self.pending_len] = b;
                    self.pending_len += 1;
                    if (self.pending_len == ctag.len) {
                        self.pending_len = 0;
                        try self.finishActiveCall(ctx, emit_fn);
                        self.mode = .outside;
                    }
                } else {
                    // Flush false prefix bytes into current inside mode, then re-handle `b`!
                    var tmp: [MAX_TAG_LEN]u8 = undefined;
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
            .outside, .inkling_header => unreachable,
            .inside_tail => {},
            .inside_hy4_xml => {
                if (self.hy4_len < self.hy4_buf.len) {
                    self.hy4_buf[self.hy4_len] = b;
                    self.hy4_len += 1;
                }
            },
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

        // Locate `"arguments"`, `"args"` (Inkling 2), or `"input"` key followed by `:` and first value byte!
        var val_pos_opt = findJsonFieldValueStart(buf, "arguments") orelse
            findJsonFieldValueStart(buf, "args") orelse
            findJsonFieldValueStart(buf, "input");
        if (val_pos_opt == null and self.active_format == .inkling2 and self.has_header_name) {
            // Inkling 2 recipient-header form: `<|message_model|>{tool_name}<|content_invoke_tool_json|>{"city":"Tokyo"}<|end_message|>`
            // When the first JSON key is NOT `"name"`/`"args"`/`"arguments"`/`"input"`, stream the raw JSON object directly!
            const trimmed_start = std.mem.indexOfScalar(u8, buf, '{');
            if (trimmed_start) |brace_pos| {
                const rest = buf[brace_pos + 1 ..];
                if (std.mem.indexOfScalar(u8, rest, ':') != null or std.mem.indexOfScalar(u8, rest, '}') != null) {
                    const has_wrapper_key = (findJsonFieldValueStart(buf, "name") != null) or
                        (std.mem.indexOf(u8, rest, "\"args\"") != null) or
                        (std.mem.indexOf(u8, rest, "\"arguments\"") != null) or
                        (std.mem.indexOf(u8, rest, "\"input\"") != null);
                    if (!has_wrapper_key) {
                        val_pos_opt = brace_pos;
                    }
                }
            }
        }
        const val_pos = val_pos_opt orelse return;
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

    fn emitHy4XmlToolCall(
        self: *ToolCallParser,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        const xml = self.hy4_buf[0..self.hy4_len];
        const first_key_pos = std.mem.indexOf(u8, xml, "<arg_key:opensource>") orelse xml.len;
        const fn_name = std.mem.trim(u8, xml[0..first_key_pos], " \t\r\n");
        if (fn_name.len == 0) {
            self.hy4_len = 0;
            return;
        }
        const nlen = @min(fn_name.len, self.name_buf.len);
        @memcpy(self.name_buf[0..nlen], fn_name[0..nlen]);
        self.name_len = @intCast(nlen);

        self.active_index = self.call_count;
        self.call_count += 1;
        const call_id = self.formatCallId(self.active_index);

        try emit_fn(ctx, ToolCallStreamEvent{ .tool_start = .{
            .index = self.active_index,
            .id = call_id,
            .name = self.nameSlice(),
        } });

        var jbuf: [2048]u8 = undefined;
        const args_json = formatHy4ArgsJson(xml, first_key_pos, &jbuf) catch |err| switch (err) {
            error.BufferTooSmall => "{}",
        };

        try emit_fn(ctx, ToolCallStreamEvent{ .tool_args_delta = .{
            .index = self.active_index,
            .id = call_id,
            .delta = args_json,
        } });
        try emit_fn(ctx, ToolCallStreamEvent{ .tool_end = .{
            .index = self.active_index,
            .id = call_id,
            .name = self.nameSlice(),
        } });
        self.hy4_len = 0;
    }

    fn formatHy4ArgsJson(xml: []const u8, first_key_pos: usize, out: []u8) error{BufferTooSmall}![]const u8 {
        const KEY_OPEN = "<arg_key:opensource>";
        const KEY_CLOSE = "</arg_key:opensource>";
        const VAL_OPEN = "<arg_value:opensource>";
        const VAL_CLOSE = "</arg_value:opensource>";

        var jw = FixedWriter.init(out);
        try jw.writeByte('{');
        var pos: usize = first_key_pos;
        var first_arg = true;
        while (pos < xml.len) {
            const ko = std.mem.indexOfPos(u8, xml, pos, KEY_OPEN) orelse break;
            const k_start = ko + KEY_OPEN.len;
            const kc = std.mem.indexOfPos(u8, xml, k_start, KEY_CLOSE) orelse break;
            const k_raw = std.mem.trim(u8, xml[k_start..kc], " \t\r\n");
            const vo = std.mem.indexOfPos(u8, xml, kc + KEY_CLOSE.len, VAL_OPEN) orelse break;
            const v_start = vo + VAL_OPEN.len;
            const vc = std.mem.indexOfPos(u8, xml, v_start, VAL_CLOSE) orelse break;
            const v_raw = xml[v_start..vc];
            const v_trim = std.mem.trim(u8, v_raw, " \t\r\n");
            pos = vc + VAL_CLOSE.len;

            if (!first_arg) try jw.writeByte(',');
            first_arg = false;
            try jw.writeByte('"');
            try jw.writeJsonEscaped(k_raw);
            try jw.writeAll("\":");
            if (isRawJsonLiteral(v_trim)) {
                try jw.writeAll(v_trim);
            } else {
                try jw.writeByte('"');
                try jw.writeJsonEscaped(v_raw);
                try jw.writeByte('"');
            }
        }
        try jw.writeByte('}');
        return jw.written();
    }

    fn isRawJsonLiteral(s: []const u8) bool {
        if (s.len == 0) return false;
        if (std.mem.eql(u8, s, "true") or std.mem.eql(u8, s, "false") or std.mem.eql(u8, s, "null")) return true;
        if ((s[0] == '{' and s[s.len - 1] == '}') or (s[0] == '[' and s[s.len - 1] == ']')) return true;
        if (s[0] == '-' or (s[0] >= '0' and s[0] <= '9')) {
            _ = std.fmt.parseFloat(f64, s) catch return false;
            return true;
        }
        return false;
    }

    fn finishActiveCall(
        self: *ToolCallParser,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        if (self.mode == .inside_hy4_xml) {
            try self.emitHy4XmlToolCall(ctx, emit_fn);
            return;
        }
        if (self.mode == .inside_preamble) {
            const raw_pre = std.mem.trim(u8, self.preamble_buf[0..self.preamble_len], " \t\r\n");
            // Arguments omitted (`{"name": "ping"}`) or raw Inkling 2 header-named args (`{"city":"Tokyo"}`):
            if (self.name_len == 0) {
                if (findJsonStringField(raw_pre, "name", &self.name_buf)) |ns| {
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
                const raw_args: []const u8 = if (self.active_format == .inkling2 and self.has_header_name and raw_pre.len > 0 and raw_pre[0] == '{')
                    raw_pre
                else
                    "{}";
                try emit_fn(ctx, ToolCallStreamEvent{ .tool_args_delta = .{
                    .index = self.active_index,
                    .id = call_id,
                    .delta = raw_args,
                } });
                try emit_fn(ctx, ToolCallStreamEvent{ .tool_end = .{
                    .index = self.active_index,
                    .id = call_id,
                    .name = self.nameSlice(),
                } });
            }
            self.preamble_len = 0;
            self.has_header_name = false;
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
        self.has_header_name = false;
        self.preamble_len = 0;
    }

    pub fn flush(
        self: *ToolCallParser,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        if (self.mode == .inkling_header) {
            if (self.header_len > 0) {
                const fl_h = self.header_buf[0..self.header_len];
                self.header_len = 0;
                try emit_fn(ctx, ToolCallStreamEvent{ .text = fl_h });
            }
            if (self.pending_len > 0) {
                const fl = self.pending[0..self.pending_len];
                self.pending_len = 0;
                try emit_fn(ctx, ToolCallStreamEvent{ .text = fl });
            }
            self.mode = .outside;
            return;
        }
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

/// Depth-1 top-level JSON object string field pre-scan (`F24`).
/// Unescapes object keys, skips keys `> 64 B`, and skips nested values without
/// descending into `input`/`messages`/`tools`/`metadata` objects or string literals.
fn findTopLevelJsonStringField(buf: []const u8, field_name: []const u8, dst: []u8) ?[]const u8 {
    var r = JsonReader{ .src = buf };
    r.expectByte('{') catch return null;
    if (r.peek() == '}') return null;
    while (true) {
        var key_buf: [64]u8 = undefined;
        const maybe_key = r.readKeyOrSkip(&key_buf) catch return null;
        r.expectByte(':') catch return null;
        if (maybe_key) |k| {
            if (std.mem.eql(u8, k, field_name)) {
                if (r.consumeKeyword("null")) return null;
                if (r.peek() != '"') return null;
                return r.readSmallString(dst) catch null;
            }
        }
        r.skipValue(1) catch return null;
        const sep = r.peek() orelse return null;
        if (sep == ',') {
            r.pos += 1;
        } else {
            return null;
        }
    }
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
    preserved_thinking: ?bool = null,
    reasoning_summary: ReasoningSummaryMode = .auto,
    starts_in_think: bool = false,
    template_dialect: chat_template.Dialect = .chatml,
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

    /// Optional legacy integer `logprobs` (`0..5`) on `POST /v1/completions` (`C29`).
    legacy_logprobs: ?u8 = null,
    /// Pre-tokenized prompt IDs on `POST /v1/completions` when `prompt` is `[u32, ...]` (`C29`).
    prompt_token_ids_buf: [MAX_COMPLETION_TOKEN_IDS]u32 = undefined,
    prompt_token_ids_len: u16 = 0,
    /// Role-isolated `(len, allow_special)` spans for `prompt_buf[0..prompt_len]` (`F13`).
    prompt_spans_buf: [chat_template.MAX_PROMPT_SPANS]chat_template.PromptSpan = undefined,
    prompt_span_count: u16 = 0,
    /// Decoded multimodal attachments extracted into `media_buf` (`C29`).
    media_items: [MAX_MEDIA_ITEMS_PER_REQUEST]MediaAttachment = undefined,
    media_count: u8 = 0,

    pub fn promptSpansSlice(self: *const ParsedRequest) []const chat_template.PromptSpan {
        return self.prompt_spans_buf[0..self.prompt_span_count];
    }

    pub fn promptTokenIdsSlice(self: *const ParsedRequest) []const u32 {
        return self.prompt_token_ids_buf[0..self.prompt_token_ids_len];
    }

    pub fn mediaSlice(self: *const ParsedRequest) []const MediaAttachment {
        return self.media_items[0..self.media_count];
    }

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

    fn readUnicodeEscape(self: *JsonReader) ParseError!u21 {
        const hi = try self.readHex4();
        if (hi >= 0xD800 and hi <= 0xDBFF) {
            if (self.src.len - self.pos < 6 or self.src[self.pos] != '\\' or self.src[self.pos + 1] != 'u') {
                return error.InvalidJson;
            }
            self.pos += 2;
            const lo = try self.readHex4();
            if (lo < 0xDC00 or lo > 0xDFFF) return error.InvalidJson;
            return 0x10000 + (@as(u21, hi - 0xD800) << 10) + @as(u21, lo - 0xDC00);
        } else if (hi >= 0xDC00 and hi <= 0xDFFF) {
            return error.InvalidJson;
        }
        return hi;
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
                    const cp = try self.readUnicodeEscape();
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
                    _ = try self.readUnicodeEscape();
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

/// Fuzz/property-test entrypoint exercising `JsonReader` over untrusted JSON bytes.
pub fn fuzzJsonReader(src: []const u8) ParseError!void {
    var r = JsonReader{ .src = src };
    try r.skipValue(0);
}

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
                                    } else if (std.mem.eql(u8, fkey, "description")) {
                                        spec.description_json = try r.captureValueSlice(3);
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
                    } else if (std.mem.eql(u8, key, "description")) {
                        spec.description_json = try r.captureValueSlice(2);
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

fn mapTemplateError(err: chat_template.TemplateError) ParseError {
    return switch (err) {
        error.BufferTooSmall => error.BufferTooSmall,
        error.InvalidJson => error.InvalidJson,
        error.DepthExceeded,
        error.InvalidReasoningEffort,
        error.InvalidSpec,
        error.MissingToolName,
        error.UnsupportedContent,
        error.UnsupportedDialect,
        => error.InvalidFieldValue,
    };
}

fn unescapeQuotedJson(
    raw_quoted: []const u8,
    scratch: []u8,
    scratch_pos: *usize,
) ParseError![]const u8 {
    if (raw_quoted.len < 2 or raw_quoted[0] != '"' or raw_quoted[raw_quoted.len - 1] != '"') {
        return error.InvalidFieldValue;
    }
    const inner = raw_quoted[1 .. raw_quoted.len - 1];
    if (std.mem.indexOfScalar(u8, inner, '\\') == null) {
        return inner;
    }
    var jr = JsonReader{ .src = raw_quoted };
    var fw = FixedWriter.init(scratch[scratch_pos.*..]);
    try jr.readStringInto(&fw);
    const slice = fw.written();
    scratch_pos.* += fw.pos;
    return slice;
}

fn readReasoningEffortValue(r: *JsonReader) ParseError!ReasoningEffort {
    const ch = r.peek() orelse return error.InvalidJson;
    if (ch == '"') {
        var ebuf: [24]u8 = undefined;
        const s = try r.readSmallString(&ebuf);
        return ReasoningEffort.parseString(s) catch return error.InvalidFieldValue;
    }
    if (ch == '-' or (ch >= '0' and ch <= '9')) {
        const s = try r.readNumberSlice();
        const v = std.fmt.parseFloat(f64, s) catch return error.InvalidFieldValue;
        return ReasoningEffort.fromFloat(v) catch return error.InvalidFieldValue;
    }
    return error.InvalidFieldValue;
}

const MAX_CONTENT_PARTS_PER_MSG: usize = 32;
const MAX_TOOL_CALLS_PER_MSG: usize = 16;

const PendingToolResultMsg = struct {
    tool_use_id: []const u8,
    content: []const u8,
};

fn readJsonStringValueToScratch(
    reader: *JsonReader,
    scratch: []u8,
    scratch_pos: *usize,
) ParseError![]const u8 {
    const raw = try reader.captureValueSlice(4);
    if (raw.len < 2 or raw[0] != '"') return error.InvalidFieldValue;
    return unescapeQuotedJson(raw, scratch, scratch_pos);
}

fn appendToolResultContentToWriter(
    reader: *JsonReader,
    fw: *FixedWriter,
    out_cache_control: *bool,
) ParseError!void {
    const ch = reader.peek() orelse return error.InvalidJson;
    if (ch == '"') {
        try reader.readStringInto(fw);
    } else if (ch == '[') {
        reader.pos += 1;
        if (reader.peek() != ']') {
            while (true) {
                try reader.expectByte('{');
                var text_raw: ?[]const u8 = null;
                if (reader.peek() != '}') {
                    while (true) {
                        var kb: [64]u8 = undefined;
                        const mk = try reader.readKeyOrSkip(&kb);
                        try reader.expectByte(':');
                        if (mk != null and std.mem.eql(u8, mk.?, "text")) {
                            text_raw = try reader.captureValueSlice(5);
                        } else if (mk != null and std.mem.eql(u8, mk.?, "cache_control")) {
                            try reader.skipValue(5);
                            out_cache_control.* = true;
                        } else {
                            try reader.skipValue(5);
                        }
                        const s = reader.peek() orelse return error.InvalidJson;
                        if (s == ',') {
                            reader.pos += 1;
                        } else if (s == '}') {
                            reader.pos += 1;
                            break;
                        } else return error.InvalidJson;
                    }
                } else reader.pos += 1;
                if (text_raw) |tr| {
                    var sub = JsonReader{ .src = tr };
                    try sub.readStringInto(fw);
                }
                const s2 = reader.peek() orelse return error.InvalidJson;
                if (s2 == ',') {
                    reader.pos += 1;
                } else if (s2 == ']') {
                    reader.pos += 1;
                    break;
                } else return error.InvalidJson;
            }
        } else reader.pos += 1;
    } else if (reader.consumeKeyword("null")) {
        // empty
    } else {
        return error.InvalidFieldValue;
    }
}

fn parseOpenAiToolCallsToViews(
    tool_calls_json: []const u8,
    calls_buf: *[MAX_TOOL_CALLS_PER_MSG]chat_template.ToolCallView,
    calls_len: *usize,
    scratch: []u8,
    scratch_pos: *usize,
) ParseError!void {
    var r = JsonReader{ .src = tool_calls_json };
    try r.expectByte('[');
    if (r.peek() == ']') {
        r.pos += 1;
        return;
    }
    while (true) {
        try r.expectByte('{');
        var call_id: []const u8 = "";
        var fn_name: []const u8 = "";
        var fn_args: []const u8 = "{}";
        if (r.peek() != '}') {
            while (true) {
                var kb: [64]u8 = undefined;
                const mk = try r.readKeyOrSkip(&kb);
                try r.expectByte(':');
                if (mk) |key| {
                    if (std.mem.eql(u8, key, "id")) {
                        if (!r.consumeKeyword("null")) {
                            call_id = try readJsonStringValueToScratch(&r, scratch, scratch_pos);
                        }
                    } else if (std.mem.eql(u8, key, "name")) {
                        if (!r.consumeKeyword("null")) {
                            fn_name = try readJsonStringValueToScratch(&r, scratch, scratch_pos);
                        }
                    } else if (std.mem.eql(u8, key, "arguments")) {
                        if (r.peek() == '"') {
                            fn_args = try readJsonStringValueToScratch(&r, scratch, scratch_pos);
                        } else if (!r.consumeKeyword("null")) {
                            fn_args = try r.captureValueSlice(3);
                        }
                    } else if (std.mem.eql(u8, key, "function") and r.peek() == '{') {
                        r.pos += 1;
                        if (r.peek() != '}') {
                            while (true) {
                                var fkb: [64]u8 = undefined;
                                const mfk = try r.readKeyOrSkip(&fkb);
                                try r.expectByte(':');
                                if (mfk) |fkey| {
                                    if (std.mem.eql(u8, fkey, "name")) {
                                        fn_name = try readJsonStringValueToScratch(&r, scratch, scratch_pos);
                                    } else if (std.mem.eql(u8, fkey, "arguments")) {
                                        if (r.peek() == '"') {
                                            fn_args = try readJsonStringValueToScratch(&r, scratch, scratch_pos);
                                        } else {
                                            fn_args = try r.captureValueSlice(4);
                                        }
                                    } else {
                                        try r.skipValue(4);
                                    }
                                } else {
                                    try r.skipValue(4);
                                }
                                const fs = r.peek() orelse return error.InvalidJson;
                                if (fs == ',') {
                                    r.pos += 1;
                                } else if (fs == '}') {
                                    r.pos += 1;
                                    break;
                                } else return error.InvalidJson;
                            }
                        } else r.pos += 1;
                    } else {
                        try r.skipValue(3);
                    }
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

        if (calls_len.* >= calls_buf.len) return error.BufferTooSmall;
        calls_buf[calls_len.*] = .{
            .id = call_id,
            .name = fn_name,
            .arguments_json = fn_args,
        };
        calls_len.* += 1;

        const sep2 = r.peek() orelse return error.InvalidJson;
        if (sep2 == ',') {
            r.pos += 1;
        } else if (sep2 == ']') {
            r.pos += 1;
            return;
        } else return error.InvalidJson;
    }
}

const MediaContext = struct {
    buf: ?[]u8,
    pos: usize = 0,
    items: *[MAX_MEDIA_ITEMS_PER_REQUEST]MediaAttachment,
    count: *u8,
};

const JsonStringIter = struct {
    src: []const u8,
    pos: usize = 1,
    done: bool = false,
    pending: [4]u8 = undefined,
    pending_idx: u8 = 0,
    pending_len: u8 = 0,

    fn init(raw_quoted: []const u8) ParseError!JsonStringIter {
        if (raw_quoted.len < 2 or raw_quoted[0] != '"') return error.InvalidFieldValue;
        return .{ .src = raw_quoted };
    }

    fn readHex4(self: *JsonStringIter) ParseError!u16 {
        if (self.pos + 4 > self.src.len) return error.InvalidJson;
        var val: u16 = 0;
        for (0..4) |_| {
            const c = self.src[self.pos];
            self.pos += 1;
            const nib = hexNibble(c) orelse return error.InvalidJson;
            val = (val << 4) | @as(u16, nib);
        }
        return val;
    }

    fn nextByte(self: *JsonStringIter) ParseError!?u8 {
        if (self.pending_idx < self.pending_len) {
            const b = self.pending[self.pending_idx];
            self.pending_idx += 1;
            return b;
        }
        if (self.done) return null;
        if (self.pos >= self.src.len) return error.InvalidJson;
        const b = self.src[self.pos];
        self.pos += 1;
        if (b == '"') {
            self.done = true;
            return null;
        }
        if (b < 0x20) return error.InvalidJson;
        if (b != '\\') return b;
        if (self.pos >= self.src.len) return error.InvalidJson;
        const esc = self.src[self.pos];
        self.pos += 1;
        switch (esc) {
            '"' => return '"',
            '\\' => return '\\',
            '/' => return '/',
            'b' => return 0x08,
            'f' => return 0x0c,
            'n' => return '\n',
            'r' => return '\r',
            't' => return '\t',
            'u' => {
                const hi = try self.readHex4();
                var cp: u21 = hi;
                if (hi >= 0xD800 and hi <= 0xDBFF) {
                    if (self.pos + 2 > self.src.len or self.src[self.pos] != '\\' or self.src[self.pos + 1] != 'u') {
                        return error.InvalidJson;
                    }
                    self.pos += 2;
                    const lo = try self.readHex4();
                    if (lo < 0xDC00 or lo > 0xDFFF) return error.InvalidJson;
                    cp = 0x10000 + (@as(u21, hi - 0xD800) << 10) + @as(u21, lo - 0xDC00);
                } else if (hi >= 0xDC00 and hi <= 0xDFFF) {
                    return error.InvalidJson;
                }
                const n = std.unicode.utf8Encode(cp, &self.pending) catch return error.InvalidJson;
                self.pending_len = @intCast(n);
                self.pending_idx = 1;
                return self.pending[0];
            },
            else => return error.InvalidJson,
        }
    }
};

inline fn decodeBase64Char(c: u8) ?u6 {
    return switch (c) {
        'A'...'Z' => @intCast(c - 'A'),
        'a'...'z' => @intCast(c - 'a' + 26),
        '0'...'9' => @intCast(c - '0' + 52),
        '+', '-' => 62,
        '/', '_' => 63,
        else => null,
    };
}

fn decodeBase64MediaStream(
    prefix_tail: []const u8,
    it: *JsonStringIter,
    media_ctx: *MediaContext,
) ParseError!usize {
    const start_pos = media_ctx.pos;
    var q: [4]u6 = undefined;
    var q_len: usize = 0;
    var eq_count: usize = 0;
    var saw_padding = false;
    var decoded_total: usize = 0;
    var tail_idx: usize = 0;

    const emitByte = struct {
        inline fn call(ctx: *MediaContext, total: *usize, byte: u8) ParseError!void {
            if (ctx.buf) |mbuf| {
                if (ctx.pos >= mbuf.len) return error.MediaTooLarge;
                mbuf[ctx.pos] = byte;
                ctx.pos += 1;
            }
            total.* += 1;
        }
    }.call;

    while (true) {
        const ch: u8 = if (tail_idx < prefix_tail.len) blk: {
            const b = prefix_tail[tail_idx];
            tail_idx += 1;
            break :blk b;
        } else (try it.nextByte()) orelse break;

        if (ch == ' ' or ch == '\t' or ch == '\r' or ch == '\n') continue;
        if (ch == '=') {
            if (saw_padding or q_len < 2 or q_len + eq_count >= 4) return error.InvalidMediaBase64;
            eq_count += 1;
            if (q_len + eq_count == 4) {
                if (eq_count == 2) {
                    try emitByte(media_ctx, &decoded_total, (@as(u8, q[0]) << 2) | (@as(u8, q[1]) >> 4));
                } else {
                    try emitByte(media_ctx, &decoded_total, (@as(u8, q[0]) << 2) | (@as(u8, q[1]) >> 4));
                    try emitByte(media_ctx, &decoded_total, ((@as(u8, q[1]) & 0x0F) << 4) | (@as(u8, q[2]) >> 2));
                }
                q_len = 0;
                eq_count = 0;
                saw_padding = true;
            }
            continue;
        }
        if (saw_padding or eq_count > 0) return error.InvalidMediaBase64;
        const val = decodeBase64Char(ch) orelse return error.InvalidMediaBase64;
        q[q_len] = val;
        q_len += 1;
        if (q_len == 4) {
            try emitByte(media_ctx, &decoded_total, (@as(u8, q[0]) << 2) | (@as(u8, q[1]) >> 4));
            try emitByte(media_ctx, &decoded_total, ((@as(u8, q[1]) & 0x0F) << 4) | (@as(u8, q[2]) >> 2));
            try emitByte(media_ctx, &decoded_total, ((@as(u8, q[2]) & 0x03) << 6) | @as(u8, q[3]));
            q_len = 0;
        }
    }

    if (eq_count > 0 or q_len == 1) return error.InvalidMediaBase64;
    if (q_len == 2) {
        try emitByte(media_ctx, &decoded_total, (@as(u8, q[0]) << 2) | (@as(u8, q[1]) >> 4));
    } else if (q_len == 3) {
        try emitByte(media_ctx, &decoded_total, (@as(u8, q[0]) << 2) | (@as(u8, q[1]) >> 4));
        try emitByte(media_ctx, &decoded_total, ((@as(u8, q[1]) & 0x0F) << 4) | (@as(u8, q[2]) >> 2));
    }
    if (decoded_total == 0) return error.InvalidMediaBase64;
    return start_pos;
}

fn parseMediaUrlField(r: *JsonReader) ParseError!?[]const u8 {
    const pch = r.peek() orelse return error.InvalidJson;
    if (pch == '"') {
        return try r.captureValueSlice(4);
    }
    if (pch == '{') {
        r.pos += 1;
        var url_raw: ?[]const u8 = null;
        if (r.peek() != '}') {
            while (true) {
                var kb: [64]u8 = undefined;
                const mk = try r.readKeyOrSkip(&kb);
                try r.expectByte(':');
                if (mk != null and std.mem.eql(u8, mk.?, "url")) {
                    if (!r.consumeKeyword("null")) {
                        url_raw = try r.captureValueSlice(5);
                    }
                } else {
                    try r.skipValue(5);
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
        return url_raw;
    }
    return error.InvalidFieldValue;
}

fn parseAnthropicMediaSource(
    r: *JsonReader,
    src_type_buf: *[24]u8,
    src_type: *[]const u8,
    src_media_type_buf: *[32]u8,
    src_media_type: *[]const u8,
    src_data_raw: *?[]const u8,
    media_url_raw: *?[]const u8,
) ParseError!void {
    try r.expectByte('{');
    if (r.peek() == '}') {
        r.pos += 1;
        return;
    }
    while (true) {
        var kb: [64]u8 = undefined;
        const mk = try r.readKeyOrSkip(&kb);
        try r.expectByte(':');
        if (mk) |key| {
            if (std.mem.eql(u8, key, "type")) {
                src_type.* = r.readSmallString(src_type_buf) catch |err| switch (err) {
                    error.BufferTooSmall => return error.InvalidFieldValue,
                    else => return err,
                };
            } else if (std.mem.eql(u8, key, "media_type")) {
                src_media_type.* = r.readSmallString(src_media_type_buf) catch |err| switch (err) {
                    error.BufferTooSmall => return error.InvalidFieldValue,
                    else => return err,
                };
            } else if (std.mem.eql(u8, key, "data")) {
                if (!r.consumeKeyword("null")) {
                    src_data_raw.* = try r.captureValueSlice(5);
                }
            } else if (std.mem.eql(u8, key, "url")) {
                if (!r.consumeKeyword("null")) {
                    media_url_raw.* = try r.captureValueSlice(5);
                }
            } else {
                try r.skipValue(5);
            }
        } else {
            try r.skipValue(5);
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

fn extractAndDecodeMediaBlock(
    kind: MediaKind,
    media_url_raw: ?[]const u8,
    src_type: []const u8,
    src_media_type: []const u8,
    src_data_raw: ?[]const u8,
    media_ctx: *MediaContext,
) ParseError!void {
    if (media_ctx.count.* >= MAX_MEDIA_ITEMS_PER_REQUEST) return error.TooManyMediaItems;

    var mime_buf: [32]u8 = @splat(0);
    var mime_len: u8 = 0;
    var start_pos: usize = media_ctx.pos;

    if (src_data_raw != null or std.mem.eql(u8, src_type, "base64")) {
        if (src_type.len > 0 and !std.mem.eql(u8, src_type, "base64")) {
            return error.InvalidFieldValue;
        }
        const data_raw = src_data_raw orelse return error.InvalidMediaBase64;
        if (src_media_type.len > mime_buf.len) return error.InvalidFieldValue;
        if (src_media_type.len > 0) {
            @memcpy(mime_buf[0..src_media_type.len], src_media_type);
            mime_len = @intCast(src_media_type.len);
        }
        var it = try JsonStringIter.init(data_raw);
        start_pos = try decodeBase64MediaStream(&.{}, &it, media_ctx);
    } else if (media_url_raw) |url_raw| {
        if (src_type.len > 0 and !std.mem.eql(u8, src_type, "url")) {
            return error.InvalidFieldValue;
        }
        var it = try JsonStringIter.init(url_raw);
        var hdr_buf: [128]u8 = undefined;
        var hdr_len: usize = 0;
        var comma_idx: ?usize = null;
        while (hdr_len < hdr_buf.len) {
            const b = (try it.nextByte()) orelse break;
            hdr_buf[hdr_len] = b;
            if (b == ',') {
                comma_idx = hdr_len;
                hdr_len += 1;
                break;
            }
            hdr_len += 1;
            if (hdr_len == 7 and std.ascii.eqlIgnoreCase(hdr_buf[0..7], "http://")) {
                return error.RemoteMediaNotSupported;
            }
            if (hdr_len == 8 and std.ascii.eqlIgnoreCase(hdr_buf[0..8], "https://")) {
                return error.RemoteMediaNotSupported;
            }
        }
        const prefix = hdr_buf[0..hdr_len];
        if (std.ascii.startsWithIgnoreCase(prefix, "http://") or std.ascii.startsWithIgnoreCase(prefix, "https://")) {
            return error.RemoteMediaNotSupported;
        }
        if (!std.ascii.startsWithIgnoreCase(prefix, "data:")) {
            return error.InvalidFieldValue;
        }
        const cidx = comma_idx orelse return error.InvalidMediaBase64;
        const meta = hdr_buf[5..cidx];
        var parts = std.mem.splitScalar(u8, meta, ';');
        const raw_mime = parts.next() orelse "";
        const eff_mime = if (raw_mime.len > 0) raw_mime else "application/octet-stream";
        if (eff_mime.len > mime_buf.len) return error.InvalidFieldValue;
        @memcpy(mime_buf[0..eff_mime.len], eff_mime);
        mime_len = @intCast(eff_mime.len);

        var has_base64 = false;
        while (parts.next()) |seg| {
            if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, seg, " \t"), "base64")) {
                has_base64 = true;
            }
        }
        if (!has_base64) return error.InvalidMediaBase64;
        start_pos = try decodeBase64MediaStream(hdr_buf[cidx + 1 .. hdr_len], &it, media_ctx);
    } else {
        return error.InvalidFieldValue;
    }

    const data_slice: []const u8 = if (media_ctx.buf) |mbuf| mbuf[start_pos..media_ctx.pos] else &.{};
    media_ctx.items[media_ctx.count.*] = .{
        .kind = kind,
        .mime_buf = mime_buf,
        .mime_len = mime_len,
        .data = data_slice,
    };
    media_ctx.count.* += 1;
}

fn parseMessageContentForTemplate(
    dialect: chat_template.Dialect,
    content_json: []const u8,
    parts_buf: *[MAX_CONTENT_PARTS_PER_MSG]chat_template.ContentPart,
    parts_len: *usize,
    calls_buf: *[MAX_TOOL_CALLS_PER_MSG]chat_template.ToolCallView,
    calls_len: *usize,
    tool_results_buf: *[MAX_TOOL_CALLS_PER_MSG]PendingToolResultMsg,
    tool_results_len: *usize,
    out_reasoning: *[]const u8,
    scratch: []u8,
    scratch_pos: *usize,
    signature_key: ?*const [32]u8,
    signature_model_id: []const u8,
    out_cache_control: *bool,
    media_ctx: *MediaContext,
) ParseError!void {
    var r = JsonReader{ .src = content_json };
    const ch = r.peek() orelse return error.InvalidJson;
    if (ch == '"') {
        const s = try unescapeQuotedJson(content_json[r.pos..], scratch, scratch_pos);
        if (s.len > 0) {
            if (parts_len.* >= parts_buf.len) return error.BufferTooSmall;
            parts_buf[parts_len.*] = .{ .text = s };
            parts_len.* += 1;
        }
        return;
    }
    if (r.consumeKeyword("null")) return;
    if (ch != '[') return error.InvalidFieldValue;

    r.pos += 1;
    if (r.peek() == ']') {
        r.pos += 1;
        return;
    }
    while (true) {
        try r.expectByte('{');
        var btype_buf: [32]u8 = undefined;
        var btype: []const u8 = "";
        var text_raw: ?[]const u8 = null;
        var thinking_raw: ?[]const u8 = null;
        var sig_raw: ?[]const u8 = null;
        var tu_id: []const u8 = "";
        var tu_name: []const u8 = "";
        var tu_input: []const u8 = "{}";
        var tr_id: []const u8 = "";
        var tr_is_error = false;
        var tr_content_raw: ?[]const u8 = null;
        var media_url_raw: ?[]const u8 = null;
        var src_type_buf: [24]u8 = undefined;
        var src_type: []const u8 = "";
        var src_media_type_buf: [32]u8 = undefined;
        var src_media_type: []const u8 = "";
        var src_data_raw: ?[]const u8 = null;

        if (r.peek() != '}') {
            while (true) {
                var kb: [64]u8 = undefined;
                const mk = try r.readKeyOrSkip(&kb);
                try r.expectByte(':');
                if (mk) |key| {
                    if (std.mem.eql(u8, key, "type")) {
                        btype = try r.readSmallString(&btype_buf);
                    } else if (std.mem.eql(u8, key, "text")) {
                        text_raw = try r.captureValueSlice(3);
                    } else if (std.mem.eql(u8, key, "thinking")) {
                        thinking_raw = try r.captureValueSlice(3);
                    } else if (std.mem.eql(u8, key, "signature")) {
                        if (!r.consumeKeyword("null")) {
                            sig_raw = try r.captureValueSlice(3);
                        }
                    } else if (std.mem.eql(u8, key, "id")) {
                        tu_id = try readJsonStringValueToScratch(&r, scratch, scratch_pos);
                    } else if (std.mem.eql(u8, key, "name")) {
                        tu_name = try readJsonStringValueToScratch(&r, scratch, scratch_pos);
                    } else if (std.mem.eql(u8, key, "input")) {
                        tu_input = try r.captureValueSlice(3);
                    } else if (std.mem.eql(u8, key, "tool_use_id")) {
                        tr_id = try readJsonStringValueToScratch(&r, scratch, scratch_pos);
                    } else if (std.mem.eql(u8, key, "is_error")) {
                        if (!r.consumeKeyword("null")) tr_is_error = try r.readBool();
                    } else if (std.mem.eql(u8, key, "content")) {
                        tr_content_raw = try r.captureValueSlice(3);
                    } else if (std.mem.eql(u8, key, "image_url") or std.mem.eql(u8, key, "video_url")) {
                        if (!r.consumeKeyword("null")) {
                            media_url_raw = try parseMediaUrlField(&r);
                        }
                    } else if (std.mem.eql(u8, key, "url")) {
                        if (!r.consumeKeyword("null")) {
                            media_url_raw = try r.captureValueSlice(3);
                        }
                    } else if (std.mem.eql(u8, key, "source")) {
                        if (!r.consumeKeyword("null")) {
                            try parseAnthropicMediaSource(
                                &r,
                                &src_type_buf,
                                &src_type,
                                &src_media_type_buf,
                                &src_media_type,
                                &src_data_raw,
                                &media_url_raw,
                            );
                        }
                    } else if (std.mem.eql(u8, key, "cache_control")) {
                        try readerSkipCacheControl(&r, out_cache_control);
                    } else {
                        try r.skipValue(3);
                    }
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

        if (btype.len == 0 or std.mem.eql(u8, btype, "text") or std.mem.eql(u8, btype, "input_text") or std.mem.eql(u8, btype, "output_text")) {
            if (text_raw) |tr| {
                const txt = try unescapeQuotedJson(tr, scratch, scratch_pos);
                if (txt.len > 0) {
                    if (parts_len.* >= parts_buf.len) return error.BufferTooSmall;
                    parts_buf[parts_len.*] = .{ .text = txt };
                    parts_len.* += 1;
                }
            }
        } else if (std.mem.eql(u8, btype, "image_url") or std.mem.eql(u8, btype, "image") or std.mem.eql(u8, btype, "input_image")) {
            try extractAndDecodeMediaBlock(.image, media_url_raw, src_type, src_media_type, src_data_raw, media_ctx);
            if (parts_len.* >= parts_buf.len) return error.BufferTooSmall;
            parts_buf[parts_len.*] = .image;
            parts_len.* += 1;
        } else if (std.mem.eql(u8, btype, "video_url") or std.mem.eql(u8, btype, "video") or std.mem.eql(u8, btype, "input_video")) {
            try extractAndDecodeMediaBlock(.video, media_url_raw, src_type, src_media_type, src_data_raw, media_ctx);
            if (parts_len.* >= parts_buf.len) return error.BufferTooSmall;
            parts_buf[parts_len.*] = .video;
            parts_len.* += 1;
        } else if (std.mem.eql(u8, btype, "thinking")) {
            const th_raw = thinking_raw orelse return error.InvalidFieldValue;
            const th_txt = try unescapeQuotedJson(th_raw, scratch, scratch_pos);
            if (signature_key) |key| {
                const sr = sig_raw orelse return error.InvalidFieldValue;
                const sig_txt = try unescapeQuotedJson(sr, scratch, scratch_pos);
                if (!verifyThinkingSignature(key, signature_model_id, th_txt, sig_txt)) {
                    return error.InvalidFieldValue;
                }
            }
            if (out_reasoning.len == 0) {
                out_reasoning.* = th_txt;
            } else {
                // Append adjacent thinking blocks cleanly in scratch.
                const prev = out_reasoning.*;
                var fw = FixedWriter.init(scratch[scratch_pos.*..]);
                try fw.writeAll(prev);
                try fw.writeAll(th_txt);
                out_reasoning.* = fw.written();
                scratch_pos.* += fw.pos;
            }
        } else if (std.mem.eql(u8, btype, "tool_use")) {
            if (calls_len.* >= calls_buf.len) return error.BufferTooSmall;
            calls_buf[calls_len.*] = .{
                .id = tu_id,
                .name = tu_name,
                .arguments_json = tu_input,
            };
            calls_len.* += 1;
        } else if (std.mem.eql(u8, btype, "tool_result")) {
            var fw = FixedWriter.init(scratch[scratch_pos.*..]);
            if (dialect == .chatml or dialect == .deepseek or dialect == .custom) {
                try fw.writeAll("<tool_response>\n");
                if (tr_id.len > 0) {
                    try fw.writeAll("[tool_use_id=");
                    try fw.writeAll(tr_id);
                    if (tr_is_error) try fw.writeAll(" is_error=true");
                    try fw.writeAll("]\n");
                } else if (tr_is_error) {
                    try fw.writeAll("[is_error=true]\n");
                }
                if (tr_content_raw) |cr| {
                    var sub = JsonReader{ .src = cr };
                    try appendToolResultContentToWriter(&sub, &fw, out_cache_control);
                }
                try fw.writeAll("\n</tool_response>\n");
                const txt = fw.written();
                scratch_pos.* += fw.pos;
                if (parts_len.* >= parts_buf.len) return error.BufferTooSmall;
                parts_buf[parts_len.*] = .{ .text = txt };
                parts_len.* += 1;
            } else {
                if (tr_is_error) try fw.writeAll("[is_error=true]\n");
                if (tr_content_raw) |cr| {
                    var sub = JsonReader{ .src = cr };
                    try appendToolResultContentToWriter(&sub, &fw, out_cache_control);
                }
                const txt = fw.written();
                scratch_pos.* += fw.pos;
                if (tool_results_len.* >= tool_results_buf.len) return error.BufferTooSmall;
                tool_results_buf[tool_results_len.*] = .{
                    .tool_use_id = tr_id,
                    .content = txt,
                };
                tool_results_len.* += 1;
            }
        } else {
            return error.InvalidFieldValue;
        }

        const sep2 = r.peek() orelse return error.InvalidJson;
        if (sep2 == ',') {
            r.pos += 1;
        } else if (sep2 == ']') {
            r.pos += 1;
            return;
        } else return error.InvalidJson;
    }
}

fn readerSkipCacheControl(r: *JsonReader, out_cache_control: *bool) ParseError!void {
    out_cache_control.* = true;
    try r.skipValue(4);
}

fn scanLastUserMsgIndex(
    has_leading_sys_header: bool,
    array_json: []const u8,
) ParseError!struct { count: usize, last_user_idx: ?usize, has_first_system: bool } {
    var r = JsonReader{ .src = array_json };
    try r.expectByte('[');
    if (r.peek() == ']') {
        r.pos += 1;
        return .{ .count = 0, .last_user_idx = null, .has_first_system = has_leading_sys_header };
    }
    var count: usize = 0;
    var last_user_idx: ?usize = null;
    var has_first_system = has_leading_sys_header;
    const base_idx: usize = if (has_leading_sys_header) 1 else 0;
    while (true) {
        try r.expectByte('{');
        var role_buf: [32]u8 = undefined;
        var role: []const u8 = "";
        var itype_buf: [32]u8 = undefined;
        var itype: []const u8 = "";
        if (r.peek() != '}') {
            while (true) {
                var kb: [64]u8 = undefined;
                const mk = try r.readKeyOrSkip(&kb);
                try r.expectByte(':');
                if (mk != null and std.mem.eql(u8, mk.?, "role")) {
                    role = try r.readSmallString(&role_buf);
                } else if (mk != null and std.mem.eql(u8, mk.?, "type")) {
                    itype = try r.readSmallString(&itype_buf);
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

        if (count == 0 and !has_leading_sys_header and (std.mem.eql(u8, role, "system") or std.mem.eql(u8, role, "developer"))) {
            has_first_system = true;
        }
        if (std.mem.eql(u8, role, "user") or std.mem.eql(u8, itype, "function_call_output")) {
            last_user_idx = base_idx + count;
        }
        count += 1;
        const sep2 = r.peek() orelse return error.InvalidJson;
        if (sep2 == ',') {
            r.pos += 1;
        } else if (sep2 == ']') {
            r.pos += 1;
            return .{ .count = count, .last_user_idx = last_user_idx, .has_first_system = has_first_system };
        } else return error.InvalidJson;
    }
}

fn renderProtocolToTemplate(
    endpoint: Endpoint,
    tpl: *const chat_template.ChatTemplate,
    prompt_buf: []u8,
    prior_prompt: ?[]const u8,
    prior_spans: ?[]const chat_template.PromptSpan,
    system_slice: ?[]const u8,
    messages_slice: ?[]const u8,
    input_slice: ?[]const u8,
    tools: []const ParsedToolSpec,
    tools_raw_slice: ?[]const u8,
    tool_choice: ToolChoiceMode,
    tool_choice_name: []const u8,
    parallel_tool_calls: bool,
    reasoning_effort: ?ReasoningEffort,
    enable_thinking: bool,
    preserved_thinking: ?bool,
    unescape_scratch: ?[]u8,
    signature_key: ?*const [32]u8,
    signature_model_id: []const u8,
    out_cache_control: *bool,
    media_ctx: *MediaContext,
    out_spans: []chat_template.PromptSpan,
) ParseError!struct { prompt_len: usize, span_count: u16, last_user_msg_byte_offset: ?usize, starts_in_think: bool } {
    var stack_scratch: [32 * 1024]u8 = undefined;
    const scratch: []u8 = unescape_scratch orelse &stack_scratch;
    var scratch_pos: usize = 0;

    var tool_views: [MAX_TOOLS_PER_REQUEST]chat_template.ToolDeclView = undefined;
    for (tools, 0..) |t, ti| {
        const desc: []const u8 = if (t.description_json.len >= 2 and t.description_json[0] == '"')
            try unescapeQuotedJson(t.description_json, scratch, &scratch_pos)
        else
            "";
        tool_views[ti] = .{
            .tool_type = "function",
            .name = t.nameSlice(),
            .description = desc,
            .parameters_json = t.parameters_json,
        };
    }
    const tool_scratch_base = scratch_pos;

    var input_is_string = false;
    var has_leading_sys = (system_slice != null);
    var last_user_idx: ?usize = null;
    switch (endpoint) {
        .chat_completions, .messages => {
            const msgs = messages_slice orelse return error.MissingRequiredField;
            const scan = try scanLastUserMsgIndex(has_leading_sys, msgs);
            if (scan.count == 0) return error.MissingRequiredField;
            has_leading_sys = scan.has_first_system;
            last_user_idx = scan.last_user_idx;
        },
        .responses => {
            const inp = input_slice orelse return error.MissingRequiredField;
            var ir = JsonReader{ .src = inp };
            const ch = ir.peek() orelse return error.InvalidJson;
            if (ch == '"') {
                input_is_string = true;
                last_user_idx = if (has_leading_sys) 1 else 0;
            } else if (ch == '[') {
                const scan = try scanLastUserMsgIndex(has_leading_sys, inp);
                if (scan.count == 0) return error.MissingRequiredField;
                has_leading_sys = scan.has_first_system;
                last_user_idx = scan.last_user_idx;
            } else {
                return error.InvalidFieldValue;
            }
        },
        .completions => unreachable,
    }

    const mapped_tool_choice: chat_template.ToolChoiceKind = switch (tool_choice) {
        .auto => .auto,
        .none => .none,
        .required => .required,
        .named => .named,
    };

    var fw = chat_template.FixedWriter.init(prompt_buf);
    var span_rec = chat_template.PromptSpanRecorder.init(out_spans);
    var renderer = chat_template.PromptRenderer.initWithSink(
        tpl,
        &fw,
        .{
            .reasoning_effort = reasoning_effort,
            .enable_thinking = enable_thinking,
            .preserved_thinking = preserved_thinking,
            .tools = tool_views[0..tools.len],
            .raw_tools_json = if (tpl.dialect == .chatml or tpl.dialect == .deepseek or tpl.dialect == .custom) tools_raw_slice else null,
            .tool_choice = mapped_tool_choice,
            .tool_choice_name = tool_choice_name,
            .parallel_tool_calls = parallel_tool_calls,
            .prior_prompt = prior_prompt,
            .prior_spans = prior_spans,
        },
        has_leading_sys,
        last_user_idx,
        span_rec.sink(),
    );
    renderer.begin() catch |err| return mapTemplateError(err);

    var msg_idx: usize = 0;
    var last_msg_is_user = false;
    var total_messages: usize = if (prior_prompt) |p| if (p.len > 0) @as(usize, 2) else 0 else 0;

    if (system_slice) |sys_json| {
        var sys_parts: [MAX_CONTENT_PARTS_PER_MSG]chat_template.ContentPart = undefined;
        var sys_parts_len: usize = 0;
        var dummy_calls: [MAX_TOOL_CALLS_PER_MSG]chat_template.ToolCallView = undefined;
        var dummy_calls_len: usize = 0;
        var dummy_trs: [MAX_TOOL_CALLS_PER_MSG]PendingToolResultMsg = undefined;
        var dummy_trs_len: usize = 0;
        var dummy_reason: []const u8 = "";
        scratch_pos = tool_scratch_base;
        try parseMessageContentForTemplate(
            tpl.dialect,
            sys_json,
            &sys_parts,
            &sys_parts_len,
            &dummy_calls,
            &dummy_calls_len,
            &dummy_trs,
            &dummy_trs_len,
            &dummy_reason,
            scratch,
            &scratch_pos,
            signature_key,
            signature_model_id,
            out_cache_control,
            media_ctx,
        );
        renderer.pushMessage(.{
            .role = .system,
            .parts = sys_parts[0..sys_parts_len],
        }, msg_idx) catch |err| return mapTemplateError(err);
        msg_idx += 1;
        total_messages += 1;
    }

    if (tools_raw_slice != null and tool_choice != .none) {
        total_messages += 1;
    }

    if (endpoint == .responses and input_is_string) {
        scratch_pos = tool_scratch_base;
        const user_txt = try unescapeQuotedJson(input_slice.?, scratch, &scratch_pos);
        renderer.pushMessage(.{
            .role = .user,
            .content = user_txt,
        }, msg_idx) catch |err| return mapTemplateError(err);
        last_msg_is_user = true;
        total_messages += 1;
    } else {
        const arr_json = if (endpoint == .responses) input_slice.? else messages_slice.?;
        var r = JsonReader{ .src = arr_json };
        try r.expectByte('[');
        if (r.peek() != ']') {
            while (true) {
                try r.expectByte('{');
                scratch_pos = tool_scratch_base;
                var role_buf: [32]u8 = undefined;
                var role_str: []const u8 = "";
                var itype_buf: [32]u8 = undefined;
                var itype: []const u8 = "";
                var msg_name: []const u8 = "";
                var call_id: []const u8 = "";
                var fc_name: []const u8 = "";
                var fc_args: []const u8 = "{}";
                var content_raw: ?[]const u8 = null;
                var output_raw: ?[]const u8 = null;
                var tool_calls_raw: ?[]const u8 = null;
                var reasoning_txt: []const u8 = "";

                if (r.peek() != '}') {
                    while (true) {
                        var kb: [64]u8 = undefined;
                        const mk = try r.readKeyOrSkip(&kb);
                        try r.expectByte(':');
                        if (mk) |key| {
                            if (std.mem.eql(u8, key, "role")) {
                                role_str = try r.readSmallString(&role_buf);
                            } else if (std.mem.eql(u8, key, "type")) {
                                itype = try r.readSmallString(&itype_buf);
                            } else if (std.mem.eql(u8, key, "name")) {
                                if (!r.consumeKeyword("null")) {
                                    const nr = try r.captureValueSlice(2);
                                    msg_name = try unescapeQuotedJson(nr, scratch, &scratch_pos);
                                    fc_name = msg_name;
                                }
                            } else if (std.mem.eql(u8, key, "call_id") or std.mem.eql(u8, key, "tool_call_id")) {
                                if (!r.consumeKeyword("null")) {
                                    const cr = try r.captureValueSlice(2);
                                    call_id = try unescapeQuotedJson(cr, scratch, &scratch_pos);
                                }
                            } else if (std.mem.eql(u8, key, "arguments")) {
                                if (r.peek() == '"') {
                                    const ar = try r.captureValueSlice(2);
                                    fc_args = try unescapeQuotedJson(ar, scratch, &scratch_pos);
                                } else if (!r.consumeKeyword("null")) {
                                    fc_args = try r.captureValueSlice(2);
                                }
                            } else if (std.mem.eql(u8, key, "reasoning_content")) {
                                if (!r.consumeKeyword("null")) {
                                    const rr = try r.captureValueSlice(2);
                                    reasoning_txt = try unescapeQuotedJson(rr, scratch, &scratch_pos);
                                }
                            } else if (std.mem.eql(u8, key, "content")) {
                                content_raw = try r.captureValueSlice(2);
                            } else if (std.mem.eql(u8, key, "output")) {
                                output_raw = try r.captureValueSlice(2);
                            } else if (std.mem.eql(u8, key, "tool_calls")) {
                                if (!r.consumeKeyword("null")) {
                                    tool_calls_raw = try r.captureValueSlice(2);
                                }
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

                if (std.mem.eql(u8, itype, "function_call_output")) {
                    const out_txt = if (output_raw) |oraw| blk: {
                        var sub = JsonReader{ .src = oraw };
                        var sub_fw = FixedWriter.init(scratch[scratch_pos..]);
                        if (sub.peek() == '"') {
                            try sub.readStringInto(&sub_fw);
                        } else {
                            try sub_fw.writeAll(oraw);
                        }
                        const slice = sub_fw.written();
                        scratch_pos += sub_fw.pos;
                        break :blk slice;
                    } else "";
                    renderer.pushMessage(.{
                        .role = .tool,
                        .tool_call_id = call_id,
                        .name = msg_name,
                        .content = out_txt,
                    }, msg_idx) catch |err| return mapTemplateError(err);
                    last_msg_is_user = true;
                } else if (std.mem.eql(u8, itype, "function_call")) {
                    const single_tc = [_]chat_template.ToolCallView{.{
                        .id = call_id,
                        .name = fc_name,
                        .arguments_json = fc_args,
                    }};
                    renderer.pushMessage(.{
                        .role = .assistant,
                        .tool_calls = &single_tc,
                    }, msg_idx) catch |err| return mapTemplateError(err);
                    last_msg_is_user = false;
                } else {
                    if (role_str.len == 0) return error.MissingRequiredField;
                    const role = chat_template.Role.parse(role_str) orelse return error.InvalidFieldValue;

                    var parts_buf: [MAX_CONTENT_PARTS_PER_MSG]chat_template.ContentPart = undefined;
                    var parts_len: usize = 0;
                    var calls_buf: [MAX_TOOL_CALLS_PER_MSG]chat_template.ToolCallView = undefined;
                    var calls_len: usize = 0;
                    var trs_buf: [MAX_TOOL_CALLS_PER_MSG]PendingToolResultMsg = undefined;
                    var trs_len: usize = 0;

                    if (content_raw) |craw| {
                        try parseMessageContentForTemplate(
                            tpl.dialect,
                            craw,
                            &parts_buf,
                            &parts_len,
                            &calls_buf,
                            &calls_len,
                            &trs_buf,
                            &trs_len,
                            &reasoning_txt,
                            scratch,
                            &scratch_pos,
                            signature_key,
                            signature_model_id,
                            out_cache_control,
                            media_ctx,
                        );
                    }
                    if (tool_calls_raw) |tc_raw| {
                        try parseOpenAiToolCallsToViews(tc_raw, &calls_buf, &calls_len, scratch, &scratch_pos);
                    }

                    if (trs_len > 0) {
                        if (parts_len > 0) {
                            renderer.pushMessage(.{
                                .role = role,
                                .name = msg_name,
                                .parts = parts_buf[0..parts_len],
                            }, msg_idx) catch |err| return mapTemplateError(err);
                        }
                        for (trs_buf[0..trs_len]) |tr_item| {
                            renderer.pushMessage(.{
                                .role = .tool,
                                .tool_call_id = tr_item.tool_use_id,
                                .content = tr_item.content,
                            }, msg_idx) catch |err| return mapTemplateError(err);
                        }
                        last_msg_is_user = true;
                    } else {
                        renderer.pushMessage(.{
                            .role = role,
                            .name = msg_name,
                            .tool_call_id = call_id,
                            .reasoning_content = reasoning_txt,
                            .parts = parts_buf[0..parts_len],
                            .tool_calls = calls_buf[0..calls_len],
                        }, msg_idx) catch |err| return mapTemplateError(err);
                        last_msg_is_user = (role == .user or role == .tool);
                    }
                }
                msg_idx += 1;
                total_messages += 1;

                const sep = r.peek() orelse return error.InvalidJson;
                if (sep == ',') {
                    r.pos += 1;
                } else if (sep == ']') {
                    r.pos += 1;
                    break;
                } else return error.InvalidJson;
            }
        }
    }

    renderer.finish() catch |err| return mapTemplateError(err);
    const boundary: ?usize = if (total_messages >= 2 and last_msg_is_user)
        renderer.last_user_msg_byte_offset
    else
        null;
    return .{
        .prompt_len = fw.pos,
        .span_count = @intCast(span_rec.count),
        .last_user_msg_byte_offset = boundary,
        .starts_in_think = renderer.starts_in_think,
    };
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
/// `/v1/messages`. Translates all three protocols via the declarative
/// `ChatTemplate` (`ChatML`, `DeepSeek`, `Inkling 2`, `Hy4`, or `custom`).
pub fn parseInferenceRequest(
    endpoint: Endpoint,
    body: []const u8,
    default_model: []const u8,
    prompt_buf: []u8,
    model_buf: []u8,
) ParseError!ParsedRequest {
    const default_tpl = chat_template.ChatTemplate.preset(.chatml);
    return parseInferenceRequestWithTemplate(endpoint, body, default_model, &default_tpl, null, null, null, prompt_buf, model_buf);
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
    const default_tpl = chat_template.ChatTemplate.preset(.chatml);
    return parseInferenceRequestWithTemplate(endpoint, body, default_model, &default_tpl, prior_chatml, null, null, prompt_buf, model_buf);
}

pub fn parseInferenceRequestWithTemplate(
    endpoint: Endpoint,
    body: []const u8,
    default_model: []const u8,
    tpl: *const chat_template.ChatTemplate,
    prior_chatml: ?[]const u8,
    unescape_scratch: ?[]u8,
    signature_key: ?*const [32]u8,
    prompt_buf: []u8,
    model_buf: []u8,
) ParseError!ParsedRequest {
    return parseInferenceRequestFull(
        endpoint,
        body,
        default_model,
        tpl,
        prior_chatml,
        unescape_scratch,
        signature_key,
        .{},
        null,
        null,
        prompt_buf,
        model_buf,
    );
}

fn renderCompletionsPrompt(
    prompt_slice: ?[]const u8,
    suffix_slice: ?[]const u8,
    fim_tokens: FimTokens,
    tok: ?*const tokenizer_mod.Tokenizer,
    unescape_scratch: ?[]u8,
    prompt_buf: []u8,
    out_token_ids: *[MAX_COMPLETION_TOKEN_IDS]u32,
    out_token_ids_len: *u16,
) ParseError!usize {
    const p_raw = prompt_slice orelse return error.MissingRequiredField;
    var stack_scratch: [16 * 1024]u8 = undefined;
    const scratch: []u8 = unescape_scratch orelse &stack_scratch;
    var scratch_pos: usize = 0;

    var suffix_txt: []const u8 = "";
    if (suffix_slice) |sr| {
        var sr_r = JsonReader{ .src = sr };
        if (sr_r.peek() != '"') return error.InvalidFieldValue;
        suffix_txt = try unescapeQuotedJson(sr[sr_r.pos..], scratch, &scratch_pos);
    }

    var pr = JsonReader{ .src = p_raw };
    const pch = pr.peek() orelse return error.InvalidJson;
    if (pch == '"') {
        const p_txt = try unescapeQuotedJson(p_raw[pr.pos..], scratch, &scratch_pos);
        var fw = FixedWriter.init(prompt_buf);
        if (suffix_txt.len > 0) {
            try fw.writeAll(fim_tokens.prefix);
            try fw.writeAll(p_txt);
            try fw.writeAll(fim_tokens.suffix);
            try fw.writeAll(suffix_txt);
            try fw.writeAll(fim_tokens.middle);
        } else {
            try fw.writeAll(p_txt);
        }
        return fw.pos;
    }

    if (pch == '[') {
        pr.pos += 1;
        const first_ch = pr.peek() orelse return error.InvalidJson;
        if (first_ch == ']') return error.InvalidFieldValue;
        if (first_ch == '"') {
            const elem_raw = try pr.captureValueSlice(2);
            const sep = pr.peek() orelse return error.InvalidJson;
            if (sep == ',') return error.UnsupportedFeature;
            if (sep != ']') return error.InvalidJson;
            pr.pos += 1;
            const p_txt = try unescapeQuotedJson(elem_raw, scratch, &scratch_pos);
            var fw = FixedWriter.init(prompt_buf);
            if (suffix_txt.len > 0) {
                try fw.writeAll(fim_tokens.prefix);
                try fw.writeAll(p_txt);
                try fw.writeAll(fim_tokens.suffix);
                try fw.writeAll(suffix_txt);
                try fw.writeAll(fim_tokens.middle);
            } else {
                try fw.writeAll(p_txt);
            }
            return fw.pos;
        }

        var count: usize = 0;
        while (true) {
            const tid = try pr.readU32();
            if (count >= MAX_COMPLETION_TOKEN_IDS) return error.BufferTooSmall;
            out_token_ids[count] = tid;
            count += 1;
            const sep = pr.peek() orelse return error.InvalidJson;
            if (sep == ',') {
                pr.pos += 1;
            } else if (sep == ']') {
                pr.pos += 1;
                break;
            } else return error.InvalidJson;
        }
        out_token_ids_len.* = @intCast(count);

        if (tok) |t| {
            const dec_dst = if (suffix_txt.len > 0) scratch[scratch_pos..] else prompt_buf;
            const decoded = t.decode(out_token_ids[0..count], dec_dst) catch |err| switch (err) {
                error.BufferTooSmall => return error.BufferTooSmall,
                else => return error.InvalidFieldValue,
            };
            if (suffix_txt.len > 0) {
                var fw = FixedWriter.init(prompt_buf);
                try fw.writeAll(fim_tokens.prefix);
                try fw.writeAll(decoded);
                try fw.writeAll(fim_tokens.suffix);
                try fw.writeAll(suffix_txt);
                try fw.writeAll(fim_tokens.middle);
                return fw.pos;
            }
            return decoded.len;
        }

        if (suffix_txt.len > 0) {
            var fw = FixedWriter.init(prompt_buf);
            try fw.writeAll(fim_tokens.prefix);
            try fw.writeAll(fim_tokens.suffix);
            try fw.writeAll(suffix_txt);
            try fw.writeAll(fim_tokens.middle);
            return fw.pos;
        }
        return 0;
    }

    return error.InvalidFieldValue;
}

pub fn parseInferenceRequestFull(
    endpoint: Endpoint,
    body: []const u8,
    default_model: []const u8,
    tpl: *const chat_template.ChatTemplate,
    prior_chatml: ?[]const u8,
    unescape_scratch: ?[]u8,
    signature_key: ?*const [32]u8,
    fim_tokens: FimTokens,
    tok: ?*const tokenizer_mod.Tokenizer,
    media_buf: ?[]u8,
    prompt_buf: []u8,
    model_buf: []u8,
) ParseError!ParsedRequest {
    return parseInferenceRequestFullWithPriorSpans(
        endpoint,
        body,
        default_model,
        tpl,
        prior_chatml,
        null,
        unescape_scratch,
        signature_key,
        fim_tokens,
        tok,
        media_buf,
        prompt_buf,
        model_buf,
    );
}

pub fn parseInferenceRequestFullWithPriorSpans(
    endpoint: Endpoint,
    body: []const u8,
    default_model: []const u8,
    tpl: *const chat_template.ChatTemplate,
    prior_chatml: ?[]const u8,
    prior_spans: ?[]const chat_template.PromptSpan,
    unescape_scratch: ?[]u8,
    signature_key: ?*const [32]u8,
    fim_tokens: FimTokens,
    tok: ?*const tokenizer_mod.Tokenizer,
    media_buf: ?[]u8,
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
    var top_enable_thinking: ?bool = null;
    var kw_enable_thinking: ?bool = null;
    var thinking_type_enabled: ?bool = null;
    var kw_preserved_thinking: ?bool = null;
    var top_reasoning_effort: ?ReasoningEffort = null;
    var kw_reasoning_effort: ?ReasoningEffort = null;
    var nested_reasoning_effort: ?ReasoningEffort = null;
    var reasoning_summary: ReasoningSummaryMode = .auto;
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
    var legacy_logprobs: ?u8 = null;
    var top_logprobs: u8 = 0;
    var stop_config = sampling_mod.StopConfig.init();
    var stream_include_usage: bool = false;
    var cache_control_pin: bool = false;
    var thinking_budget_tokens: ?u32 = null;
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
    var prompt_slice: ?[]const u8 = null;
    var suffix_slice: ?[]const u8 = null;

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
            } else if (std.mem.eql(u8, maybe_key.?, "prompt")) {
                if (!r.consumeKeyword("null")) {
                    prompt_slice = try r.captureValueSlice(1);
                }
            } else if (std.mem.eql(u8, maybe_key.?, "suffix")) {
                if (!r.consumeKeyword("null")) {
                    suffix_slice = try r.captureValueSlice(1);
                }
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
                    if (endpoint == .completions) {
                        const pch = r.peek() orelse return error.InvalidJson;
                        if (pch == 't' or pch == 'f') {
                            logprobs = try r.readBool();
                            legacy_logprobs = if (logprobs) 1 else null;
                        } else {
                            const lv = try r.readU32();
                            if (lv > 5) return error.InvalidFieldValue;
                            legacy_logprobs = @intCast(lv);
                            logprobs = true;
                            top_logprobs = @intCast(lv);
                        }
                    } else {
                        logprobs = try r.readBool();
                    }
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
                // OpenAI `/v1/responses` `reasoning.effort` + `reasoning.summary`.
                if (!r.consumeKeyword("null")) {
                    try r.expectByte('{');
                    if (r.peek() != '}') {
                        while (true) {
                            var rk_buf: [64]u8 = undefined;
                            const mrk = try r.readKeyOrSkip(&rk_buf);
                            try r.expectByte(':');
                            if (mrk != null and std.mem.eql(u8, mrk.?, "effort")) {
                                if (!r.consumeKeyword("null")) {
                                    nested_reasoning_effort = try readReasoningEffortValue(&r);
                                }
                            } else if (mrk != null and std.mem.eql(u8, mrk.?, "summary")) {
                                if (r.consumeKeyword("null")) {
                                    reasoning_summary = .none;
                                } else {
                                    var sbuf: [24]u8 = undefined;
                                    const sm = try r.readSmallString(&sbuf);
                                    reasoning_summary = try ReasoningSummaryMode.parseString(sm);
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
            } else if (std.mem.eql(u8, maybe_key.?, "reasoning_effort")) {
                if (!r.consumeKeyword("null")) {
                    top_reasoning_effort = try readReasoningEffortValue(&r);
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
                top_enable_thinking = try r.readBool();
            } else if (std.mem.eql(u8, maybe_key.?, "chat_template_kwargs")) {
                if (r.peek() == '{') {
                    r.pos += 1;
                    if (r.peek() != '}') {
                        while (true) {
                            var sub_key_buf: [64]u8 = undefined;
                            const maybe_sub = try r.readKeyOrSkip(&sub_key_buf);
                            try r.expectByte(':');
                            if (maybe_sub != null and std.mem.eql(u8, maybe_sub.?, "enable_thinking")) {
                                kw_enable_thinking = try r.readBool();
                            } else if (maybe_sub != null and std.mem.eql(u8, maybe_sub.?, "reasoning_effort")) {
                                if (!r.consumeKeyword("null")) {
                                    kw_reasoning_effort = try readReasoningEffortValue(&r);
                                }
                            } else if (maybe_sub != null and std.mem.eql(u8, maybe_sub.?, "preserved_thinking")) {
                                kw_preserved_thinking = try r.readBool();
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
                                    thinking_type_enabled = false;
                                } else if (std.mem.eql(u8, t, "enabled")) {
                                    thinking_type_enabled = true;
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

    if (top_enable_thinking != null and kw_enable_thinking != null and top_enable_thinking.? != kw_enable_thinking.?) {
        return error.InvalidFieldValue;
    }
    const en_bool: ?bool = kw_enable_thinking orelse top_enable_thinking;
    if (en_bool != null and thinking_type_enabled != null and en_bool.? != thinking_type_enabled.?) {
        return error.InvalidFieldValue;
    }
    const resolved_enable_opt: ?bool = en_bool orelse thinking_type_enabled;

    const efforts = [_]?ReasoningEffort{ nested_reasoning_effort, top_reasoning_effort, kw_reasoning_effort };
    var first_effort: ?ReasoningEffort = null;
    for (efforts) |me| {
        if (me) |eff| {
            if (first_effort) |prev| {
                if (!prev.eqlNormalized(eff)) return error.InvalidFieldValue;
            } else {
                first_effort = eff;
            }
        }
    }
    const resolved_effort_opt: ?ReasoningEffort = kw_reasoning_effort orelse top_reasoning_effort orelse nested_reasoning_effort;

    if (resolved_enable_opt != null and resolved_effort_opt != null) {
        if (resolved_enable_opt.? != resolved_effort_opt.?.isThinkingEnabled()) {
            return error.InvalidFieldValue;
        }
    }
    const enable_thinking: bool = if (endpoint == .completions)
        false
    else if (resolved_enable_opt) |b|
        b
    else if (resolved_effort_opt) |e|
        e.isThinkingEnabled()
    else
        true;
    const resolved_effort: ?ReasoningEffort = if (resolved_effort_opt) |e|
        e
    else if (!enable_thinking)
        .none
    else
        null;
    const resolved_budget_tokens: ?u32 = if (enable_thinking) thinking_budget_tokens else null;

    var prompt_token_ids_buf: [MAX_COMPLETION_TOKEN_IDS]u32 = undefined;
    var prompt_token_ids_len: u16 = 0;
    var prompt_spans_buf: [chat_template.MAX_PROMPT_SPANS]chat_template.PromptSpan = undefined;
    var prompt_span_count: u16 = 0;
    var media_items: [MAX_MEDIA_ITEMS_PER_REQUEST]MediaAttachment = undefined;
    var media_count: u8 = 0;

    const rendered_prompt_len: usize, const rendered_boundary: ?usize, const starts_in_think: bool = if (endpoint == .completions) blk: {
        const plen = try renderCompletionsPrompt(
            prompt_slice,
            suffix_slice,
            fim_tokens,
            tok,
            unescape_scratch,
            prompt_buf,
            &prompt_token_ids_buf,
            &prompt_token_ids_len,
        );
        break :blk .{ plen, null, false };
    } else blk: {
        var media_ctx = MediaContext{
            .buf = media_buf,
            .items = &media_items,
            .count = &media_count,
        };
        const rendered = try renderProtocolToTemplate(
            endpoint,
            tpl,
            prompt_buf,
            prior_chatml,
            prior_spans,
            system_slice,
            messages_slice,
            input_slice,
            tools[0..tool_count],
            tools_raw_slice,
            tool_choice,
            tool_choice_name_buf[0..tool_choice_name_len],
            parallel_tool_calls,
            resolved_effort,
            enable_thinking,
            kw_preserved_thinking,
            unescape_scratch,
            signature_key,
            model_buf[0..model_len],
            &cache_control_pin,
            &media_ctx,
            &prompt_spans_buf,
        );
        prompt_span_count = rendered.span_count;
        break :blk .{ rendered.prompt_len, rendered.last_user_msg_byte_offset, rendered.starts_in_think };
    };

    return .{
        .endpoint = endpoint,
        .model_len = @intCast(model_len),
        .prompt_len = rendered_prompt_len,
        .last_user_msg_byte_offset = rendered_boundary,
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
        .thinking_budget_tokens = resolved_budget_tokens,
        .reasoning_effort = resolved_effort,
        .preserved_thinking = kw_preserved_thinking,
        .reasoning_summary = reasoning_summary,
        .starts_in_think = starts_in_think,
        .template_dialect = tpl.dialect,
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
        .legacy_logprobs = legacy_logprobs,
        .prompt_token_ids_buf = prompt_token_ids_buf,
        .prompt_token_ids_len = prompt_token_ids_len,
        .prompt_spans_buf = prompt_spans_buf,
        .prompt_span_count = prompt_span_count,
        .media_items = media_items,
        .media_count = media_count,
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
    options,
    other,
};

const HttpHeadersResult = struct {
    method: HttpMethod,
    path: []const u8,
    query: []const u8 = "",
    header_end: usize,
    content_length: ?usize,
    chunked: bool,
    expect_continue: bool,
    host: ?[]const u8 = null,
    authorization_bearer: ?[]const u8 = null,
    x_api_key: ?[]const u8 = null,
    anthropic_version: ?[]const u8 = null,
    connection_close: bool = false,
};

const HttpParseError = error{
    Incomplete,
    BadRequest,
    PayloadTooLarge,
};

const StrictParseError = error{
    InvalidSyntax,
    Overflow,
};

/// Parses strictly `1*DIGIT` decimal integers (rejects `+`, `-`, `_`, whitespace, empty; `L4`).
fn parseStrictDecimalUsize(s: []const u8) StrictParseError!usize {
    if (s.len == 0) return error.InvalidSyntax;
    var val: usize = 0;
    for (s) |c| {
        if (c < '0' or c > '9') return error.InvalidSyntax;
        val = std.math.mul(usize, val, 10) catch return error.Overflow;
        val = std.math.add(usize, val, c - '0') catch return error.Overflow;
    }
    return val;
}

/// Parses strictly `1*HEXDIG` hexadecimal integers (rejects `+`, `-`, `_`, `0x`, empty; `L4`).
fn parseStrictHexUsize(s: []const u8) StrictParseError!usize {
    if (s.len == 0) return error.InvalidSyntax;
    var val: usize = 0;
    for (s) |c| {
        const digit: usize = switch (c) {
            '0'...'9' => c - '0',
            'a'...'f' => c - 'a' + 10,
            'A'...'F' => c - 'A' + 10,
            else => return error.InvalidSyntax,
        };
        val = std.math.mul(usize, val, 16) catch return error.Overflow;
        val = std.math.add(usize, val, digit) catch return error.Overflow;
    }
    return val;
}

fn isLoopbackIpv4(host: [4]u8) bool {
    return host[0] == 127;
}

const HostValidation = enum { ok, bad_request, forbidden };

fn validateHostHeader(raw_host: []const u8, bound_host: [4]u8) HostValidation {
    if (raw_host.len == 0) return .bad_request;
    for (raw_host) |c| {
        if (c <= ' ' or c == 0x7F or c == '@' or c == '/' or c == '?' or c == '#') {
            return .bad_request;
        }
    }
    var host_part: []const u8 = raw_host;
    var port_part: ?[]const u8 = null;
    if (raw_host[0] == '[') {
        const close_idx = std.mem.indexOfScalar(u8, raw_host, ']') orelse return .bad_request;
        host_part = raw_host[0 .. close_idx + 1];
        const rest = raw_host[close_idx + 1 ..];
        if (rest.len > 0) {
            if (rest[0] != ':') return .bad_request;
            port_part = rest[1..];
        }
    } else if (std.mem.indexOfScalar(u8, raw_host, ':')) |colon| {
        host_part = raw_host[0..colon];
        port_part = raw_host[colon + 1 ..];
        if (std.mem.indexOfScalar(u8, port_part.?, ':') != null) return .bad_request;
    }
    if (host_part.len == 0) return .bad_request;
    if (port_part) |p| {
        const port_num = parseStrictDecimalUsize(p) catch return .bad_request;
        if (port_num == 0 or port_num > 65535) return .bad_request;
    }
    if (std.ascii.eqlIgnoreCase(host_part, "localhost")) return .ok;
    if (std.mem.eql(u8, host_part, "127.0.0.1")) return .ok;
    if (std.mem.eql(u8, host_part, "[::1]")) return .ok;
    if (std.mem.eql(u8, host_part, "0.0.0.0")) return .ok;
    var bound_buf: [16]u8 = undefined;
    const bound_str = std.fmt.bufPrint(&bound_buf, "{d}.{d}.{d}.{d}", .{
        bound_host[0],
        bound_host[1],
        bound_host[2],
        bound_host[3],
    }) catch return .forbidden;
    if (std.mem.eql(u8, host_part, bound_str)) return .ok;
    return .forbidden;
}

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
    const default_close = std.mem.eql(u8, version_str, "HTTP/1.0");
    if (target_str.len == 0 or target_str[0] != '/') return error.BadRequest;

    const method: HttpMethod = if (std.mem.eql(u8, method_str, "GET"))
        .get
    else if (std.mem.eql(u8, method_str, "POST"))
        .post
    else if (std.mem.eql(u8, method_str, "DELETE"))
        .delete
    else if (std.mem.eql(u8, method_str, "OPTIONS"))
        .options
    else
        .other;

    const q_idx = std.mem.indexOfScalar(u8, target_str, '?');
    const path = if (q_idx) |q| target_str[0..q] else target_str;
    const query = if (q_idx) |q| target_str[q + 1 ..] else "";

    var content_length: ?usize = null;
    var chunked = false;
    var transfer_encoding_seen = false;
    var expect_continue = false;
    var host: ?[]const u8 = null;
    var authorization_bearer: ?[]const u8 = null;
    var x_api_key: ?[]const u8 = null;
    var anthropic_version: ?[]const u8 = null;
    var connection_close: bool = default_close;

    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.BadRequest;
        const name = std.mem.trim(u8, line[0..colon], " \t");
        const val = std.mem.trim(u8, line[colon + 1 ..], " \t");

        if (std.ascii.eqlIgnoreCase(name, "content-length")) {
            const parsed = parseStrictDecimalUsize(val) catch |err| switch (err) {
                error.InvalidSyntax => return error.BadRequest,
                error.Overflow => return error.PayloadTooLarge,
            };
            if (content_length) |prev| {
                if (prev != parsed) return error.BadRequest;
            }
            content_length = parsed;
        } else if (std.ascii.eqlIgnoreCase(name, "transfer-encoding")) {
            if (transfer_encoding_seen) return error.BadRequest;
            transfer_encoding_seen = true;
            if (std.ascii.eqlIgnoreCase(val, "chunked")) {
                chunked = true;
            } else {
                return error.BadRequest;
            }
        } else if (std.ascii.eqlIgnoreCase(name, "host")) {
            if (host != null) return error.BadRequest;
            host = val;
        } else if (std.ascii.eqlIgnoreCase(name, "connection")) {
            if (asciiContainsIgnoreCase(val, "close")) {
                connection_close = true;
            } else if (asciiContainsIgnoreCase(val, "keep-alive")) {
                connection_close = false;
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
        .query = query,
        .header_end = header_end,
        .content_length = content_length,
        .chunked = chunked,
        .expect_continue = expect_continue,
        .host = host,
        .authorization_bearer = authorization_bearer,
        .x_api_key = x_api_key,
        .anthropic_version = anthropic_version,
        .connection_close = connection_close,
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
    complete: struct {
        decoded_len: usize,
        consumed_len: usize,
    },
};

/// Validates and decodes an HTTP/1.1 `Transfer-Encoding: chunked` body in place
/// once the terminating `0\r\n...\r\n` chunk has fully arrived.
/// Strictly overflow-free and immediately rejects any chunk or cumulative decoded
/// size exceeding `max_body_bytes` with `error.PayloadTooLarge` (`F1`, `L4`).
fn decodeChunkedInPlace(body_buf: []u8, max_body_bytes: usize) error{ BadRequest, PayloadTooLarge }!ChunkedDecodeStatus {
    // First verify the entire chunked payload has arrived before mutating in place.
    var scan: usize = 0;
    var decoded_total: usize = 0;
    var consumed_len: usize = 0;
    while (true) {
        const line_rel = neonFindCrlf(body_buf[scan..]) orelse return .incomplete;
        const size_line_full = body_buf[scan .. scan + line_rel];
        const size_line = if (std.mem.indexOfScalar(u8, size_line_full, ';')) |semi|
            std.mem.trim(u8, size_line_full[0..semi], " \t")
        else
            std.mem.trim(u8, size_line_full, " \t");
        if (size_line.len == 0) return error.BadRequest;
        const chunk_len = parseStrictHexUsize(size_line) catch |err| switch (err) {
            error.InvalidSyntax => return error.BadRequest,
            error.Overflow => return error.PayloadTooLarge,
        };
        scan += line_rel + 2;
        if (chunk_len == 0) {
            if (body_buf.len - scan < 2) return .incomplete;
            if (body_buf[scan] == '\r' and body_buf[scan + 1] == '\n') {
                consumed_len = scan + 2;
                break;
            }
            const trailer_end = neonFindCrlfCrlf(body_buf[scan..]) orelse return .incomplete;
            consumed_len = scan + trailer_end + 4;
            break;
        }
        const next_total = std.math.add(usize, decoded_total, chunk_len) catch return error.PayloadTooLarge;
        if (chunk_len > max_body_bytes or next_total > max_body_bytes) {
            return error.PayloadTooLarge;
        }
        decoded_total = next_total;

        const remaining = body_buf.len - scan;
        if (chunk_len > remaining or remaining - chunk_len < 2) return .incomplete;
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
        const chunk_len = parseStrictHexUsize(size_line) catch unreachable;
        read_pos += line_rel + 2;
        if (chunk_len == 0) break;
        std.mem.copyForwards(u8, body_buf[out_pos .. out_pos + chunk_len], body_buf[read_pos .. read_pos + chunk_len]);
        out_pos += chunk_len;
        read_pos += chunk_len + 2;
    }
    return .{ .complete = .{ .decoded_len = out_pos, .consumed_len = consumed_len } };
}

/// Fuzz/property-test entrypoint exercising HTTP request-line & header parsing.
pub fn fuzzParseHttpHeaders(buf: []const u8) HttpParseError!void {
    _ = try parseHttpHeaders(buf);
}

/// Fuzz/property-test entrypoint exercising `decodeChunkedInPlace` framing.
pub fn fuzzDecodeChunkedInPlace(body_buf: []u8, max_body_bytes: usize) error{ BadRequest, PayloadTooLarge }!void {
    _ = try decodeChunkedInPlace(body_buf, max_body_bytes);
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
    draining_worker = 5,
};

const AuxWorkerError = enum(u8) {
    none = 0,
    unavailable = 1,
    payload_too_large = 2,
    embed_failed = 3,
    embed_invalid_dim = 4,
    rerank_failed = 5,
    tx_overflow = 6,
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
    incomplete = 4,

    fn name(self: ResponseRecordStatus) []const u8 {
        return switch (self) {
            .in_progress => "in_progress",
            .completed => "completed",
            .cancelled => "cancelled",
            .failed => "failed",
            .incomplete => "incomplete",
        };
    }
};

const IncompleteReason = enum(u8) {
    none = 0,
    max_output_tokens = 1,
    max_response_bytes = 2,

    fn name(self: IncompleteReason) []const u8 {
        return switch (self) {
            .none => "",
            .max_output_tokens => "max_output_tokens",
            .max_response_bytes => "max_response_bytes",
        };
    }
};

const InputItemRole = enum(u8) {
    user = 0,
    system = 1,
    developer = 2,
    assistant = 3,
    tool = 4,

    fn name(self: InputItemRole) []const u8 {
        return switch (self) {
            .user => "user",
            .system => "system",
            .developer => "developer",
            .assistant => "assistant",
            .tool => "tool",
        };
    }
};

const StatefulInputItem = struct {
    origin_request_id: u64 = 0,
    origin_item_idx: u16 = 0,
    offset: u32 = 0,
    len: u32 = 0,
    role: InputItemRole = .user,
};

/// Preallocated fixed-slot stateful `/v1/responses` record (`evict_first` LRU,
/// zero steady-state heap allocation, `C18`).
const StatefulResponseSlot = struct {
    occupied: bool = false,
    lru_tick: u64 = 0,
    request_id: u64 = 0,
    created_unix: u64 = 0,
    status: ResponseRecordStatus = .completed,
    incomplete_reason: IncompleteReason = .none,
    reasoning_summary: ReasoningSummaryMode = .auto,
    transcript_corrupt: bool = false,
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
    chatml_spans: [chat_template.MAX_PROMPT_SPANS]chat_template.PromptSpan = undefined,
    chatml_span_count: u16 = 0,
    reasoning_buf: []u8 = &.{},
    reasoning_len: usize = 0,
    content_buf: []u8 = &.{},
    content_len: usize = 0,
    input_items_buf: []u8 = &.{},
    input_items_len: usize = 0,
    input_items: [MAX_STATEFUL_INPUT_ITEMS]StatefulInputItem = undefined,
    input_item_count: u8 = 0,
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

    fn chatmlSpansSlice(self: *const StatefulResponseSlot) []const chat_template.PromptSpan {
        return self.chatml_spans[0..self.chatml_span_count];
    }
};

const ConnectionSlot = struct {
    index: u16,
    fd: std.c.fd_t,
    state: SlotState,
    read_armed: bool,
    write_armed: bool,
    headers_received: bool,
    is_network_tcp: bool,
    peer_ipv4: ?[4]u8,
    sent_continue: bool,
    /// Server-thread-only flag guaranteeing `active_requests` is decremented
    /// at most once per admitted request across all completion/abort races.
    request_counted: bool,
    api_cancelled: bool,
    aux_kind: JobKind,
    embed_req: EmbedRequest,
    rerank_req: RerankRequest,
    aux_response_len: usize,
    aux_err: AuxWorkerError,

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
    media_buf: []u8,
    prompt_token_ids_buf: [MAX_COMPLETION_TOKEN_IDS]u32,
    prompt_token_ids_len: u16,
    prompt_spans_buf: [chat_template.MAX_PROMPT_SPANS]chat_template.PromptSpan,
    prompt_span_count: u16,
    legacy_logprobs: ?u8,

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
    keep_alive: bool,
    has_completed_request: bool,
    rx_consumed_len: usize,
    read_start_ns: u64,
    last_activity_ns: u64,
    admit_ns: u64,
    prefill_done_ns: u64,
    first_generated_token_ns: u64,
    first_emitted_ns: u64,
    first_token_ns: u64,
    created_unix: u64,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    reasoning_tokens: u32,
    nonstream_reasoning_truncated: bool,
    nonstream_content_truncated: bool,
    tool_calls_truncated: bool,
    thinking_budget_tokens: ?u32,
    thinking_budget_exhausted: bool,
    reasoning_summary: ReasoningSummaryMode,
    template_dialect: chat_template.Dialect,
    thinking_hmac: HmacSha256,
    anthropic_block_had_delta: bool,
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

    fn resetForConnection(self: *ConnectionSlot, fd: std.c.fd_t, is_network_tcp: bool, peer_ipv4: ?[4]u8) void {
        self.fd = fd;
        self.state = .reading_request;
        self.read_armed = false;
        self.write_armed = false;
        self.headers_received = false;
        self.is_network_tcp = is_network_tcp;
        self.peer_ipv4 = peer_ipv4;
        self.sent_continue = false;
        self.request_counted = false;
        self.keep_alive = false;
        self.has_completed_request = false;
        self.rx_len = 0;
        self.rx_consumed_len = 0;
        const now = monotonicNowNs();
        self.read_start_ns = now;
        self.last_activity_ns = now;
        self.resetPerRequestState();
        // Drain any stale token events from a previous generation.
        while (self.token_ring.tryPop() != null) {}
    }

    fn resetForNextKeepAliveRequest(self: *ConnectionSlot, rem_rx_len: usize, now_ns: u64) void {
        self.state = .reading_request;
        self.write_armed = false;
        self.headers_received = false;
        self.sent_continue = false;
        self.request_counted = false;
        self.has_completed_request = true;
        self.rx_len = rem_rx_len;
        self.rx_consumed_len = 0;
        self.read_start_ns = if (rem_rx_len > 0) now_ns else 0;
        self.last_activity_ns = now_ns;
        self.resetPerRequestState();
        while (self.token_ring.tryPop() != null) {}
    }

    fn resetPerRequestState(self: *ConnectionSlot) void {
        self.api_cancelled = false;
        self.aux_kind = .generate;
        self.aux_response_len = 0;
        self.aux_err = .none;
        self.prompt_len = 0;
        self.prompt_token_ids_len = 0;
        self.prompt_span_count = 0;
        self.legacy_logprobs = null;
        self.tx_read_pos = 0;
        self.tx_write_pos = 0;
        self.nonstream_reasoning_len = 0;
        self.nonstream_content_len = 0;
        self.nonstream_reasoning_truncated = false;
        self.nonstream_content_truncated = false;
        self.tool_calls_truncated = false;
        self.thinking_budget_tokens = null;
        self.thinking_budget_exhausted = false;
        self.reasoning_summary = .auto;
        self.template_dialect = .chatml;
        self.anthropic_block_had_delta = false;
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
        self.prefill_done_ns = 0;
        self.first_generated_token_ns = 0;
        self.first_emitted_ns = 0;
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
    defer_terminal: bool = false,
    deferred_terminal: ?TokenEvent = null,
    thinking_budget_tokens: ?u32 = null,
    worker_think_splitter: ThinkSplitter = .{},
    worker_splitter_initialized: bool = false,
    reasoning_tokens_seen: u32 = 0,
    thinking_budget_reached: std.atomic.Value(bool) = .init(false),

    pub fn setThinkingBudget(
        self: *TokenStream,
        budget: ?u32,
        dialect: chat_template.Dialect,
        starts_in_think: bool,
    ) void {
        self.thinking_budget_tokens = budget;
        self.worker_think_splitter.resetForDialect(dialect, starts_in_think);
        self.worker_splitter_initialized = true;
        self.reasoning_tokens_seen = 0;
        self.thinking_budget_reached.store(false, .release);
    }

    pub fn thinkingBudgetExceeded(self: *const TokenStream) bool {
        return self.thinking_budget_reached.load(.acquire);
    }

    pub fn thinkingBudgetRemaining(self: *const TokenStream) ?u32 {
        const budget = self.thinking_budget_tokens orelse self.slot.thinking_budget_tokens orelse return null;
        return budget -| self.reasoning_tokens_seen;
    }

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
            .timestamp_ns = monotonicNowNs(),
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
        const now_ns = monotonicNowNs();
        if (!self.worker_splitter_initialized) {
            self.worker_think_splitter.resetForDialect(self.slot.template_dialect, self.slot.think_splitter.in_think);
            self.worker_splitter_initialized = true;
            if (self.thinking_budget_tokens == null) {
                self.thinking_budget_tokens = self.slot.thinking_budget_tokens;
            }
        }

        const EmitCtx = struct {
            stream: *TokenStream,
            pending_tokens: u32,
            ts_ns: u64,
            emitted_any: bool = false,
            had_reasoning: bool = false,

            fn onSplitSpan(ctx: *@This(), ch: StreamChannel, span: []const u8) error{}!void {
                if (ch == .reasoning and span.len > 0) {
                    ctx.had_reasoning = true;
                }
            }

            fn onUtf8(ctx: *@This(), valid_utf8: []const u8) bool {
                ctx.stream.worker_think_splitter.feed(valid_utf8, ctx, onSplitSpan) catch unreachable;
                var off: usize = 0;
                while (off < valid_utf8.len) {
                    const take = utf8ChunkLen(valid_utf8[off..], MAX_CHUNK_TEXT_BYTES);
                    const tok_cnt = ctx.pending_tokens;
                    ctx.pending_tokens = 0;
                    var ev = TokenEvent{
                        .generation = ctx.stream.generation,
                        .kind = .token_text,
                        .token_count = tok_cnt,
                        .timestamp_ns = ctx.ts_ns,
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
            .ts_ns = now_ns,
        };
        if (!self.utf8.push(bytes, &ctx, EmitCtx.onUtf8)) return false;

        if (ctx.had_reasoning and token_delta > 0) {
            self.reasoning_tokens_seen +|= token_delta;
            if (self.thinking_budget_tokens orelse self.slot.thinking_budget_tokens) |budget| {
                if (self.reasoning_tokens_seen >= budget) {
                    self.thinking_budget_reached.store(true, .release);
                }
            }
        }

        // If `bytes` only buffered incomplete UTF-8 carry bytes (or was empty) while
        // advancing `token_delta`, still record the token count accurately.
        if (!ctx.emitted_any and ctx.pending_tokens > 0) {
            const ev = TokenEvent{
                .generation = self.generation,
                .kind = .token_text,
                .token_count = ctx.pending_tokens,
                .timestamp_ns = now_ns,
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

    /// Emits a zero-token-count synthetic dialect control delimiter (e.g. closing
    /// `</think>` injected into the model's KV state after `thinking_budget_tokens`
    /// cutoff) so the network-side `ThinkSplitter` transitions cleanly to `.content`
    /// without inflating `completion_tokens` or skewing `TTFT`/`TPOT`.
    pub fn emitControlBytes(self: *TokenStream, bytes: []const u8) bool {
        return self.emitBytes(bytes, 0);
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
        if (self.defer_terminal) {
            self.deferred_terminal = ev;
        } else {
            _ = self.pushEvent(ev);
        }
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
        if (self.defer_terminal) {
            self.deferred_terminal = ev;
        } else {
            _ = self.pushEvent(ev);
        }
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

const StoredLoraAdapter = struct {
    name_buf: [lora_mod.MAX_ADAPTER_NAME_LEN]u8 = @splat(0),
    name_len: u8 = 0,
    display_name_buf: [MAX_DISPLAY_NAME_BYTES]u8 = @splat(0),
    display_name_len: u8 = 0,
    rank: u32 = 0,
    lora_alpha: f32 = 0.0,
    scale: f32 = 0.0,
    wired_bytes: u64 = 0,

    fn nameSlice(self: *const StoredLoraAdapter) []const u8 {
        return self.name_buf[0..self.name_len];
    }

    fn displayNameSlice(self: *const StoredLoraAdapter) []const u8 {
        return self.display_name_buf[0..self.display_name_len];
    }
};

const StoredFimTokens = struct {
    prefix_buf: [MAX_FIM_TOKEN_BYTES]u8 = @splat(0),
    prefix_len: u8 = 0,
    suffix_buf: [MAX_FIM_TOKEN_BYTES]u8 = @splat(0),
    suffix_len: u8 = 0,
    middle_buf: [MAX_FIM_TOKEN_BYTES]u8 = @splat(0),
    middle_len: u8 = 0,

    fn asFimTokens(self: *const StoredFimTokens) FimTokens {
        return .{
            .prefix = self.prefix_buf[0..self.prefix_len],
            .suffix = self.suffix_buf[0..self.suffix_len],
            .middle = self.middle_buf[0..self.middle_len],
        };
    }
};

pub const Server = struct {
    config: ServerConfig,
    model_id_buf: [MAX_MODEL_ID_BYTES]u8,
    model_id_len: u16,
    created_unix_sec: i64,
    model_arch_buf: [64]u8,
    model_arch_len: u8,
    model_display_name_buf: [MAX_DISPLAY_NAME_BYTES]u8,
    model_display_name_len: u8,
    model_detail: ModelDetailMeta,
    lora_adapters_storage: [lora_mod.MAX_ADAPTERS]StoredLoraAdapter,
    lora_adapter_count: u8,
    api_key_buf: [MAX_API_KEY_BYTES]u8,
    api_key_len: u16,
    cors_allow_origin_buf: [MAX_CORS_ORIGIN_BYTES]u8,
    cors_allow_origin_len: u16,
    fim_tokens: StoredFimTokens,
    signature_key: [32]u8,
    unix_socket_path_buf: [MAX_UNIX_SOCKET_PATH_BYTES + 1]u8,
    unix_socket_path_len: u16,

    kq_fd: std.c.fd_t,
    worker_kq_fd: std.c.fd_t,
    listen_fd: std.c.fd_t,
    tls_listener: ?tls_mod.TlsListener,
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
        if (config.lora_adapters.len > lora_mod.MAX_ADAPTERS) return error.InvalidConfig;
        for (config.lora_adapters, 0..) |adp, i| {
            if (adp.name.len == 0 or adp.name.len > lora_mod.MAX_ADAPTER_NAME_LEN) return error.InvalidConfig;
            if (adp.display_name.len > MAX_DISPLAY_NAME_BYTES) return error.InvalidConfig;
            if (adp.rank == 0) return error.InvalidConfig;
            if (!std.math.isFinite(adp.lora_alpha) or !std.math.isFinite(adp.scale)) return error.InvalidConfig;
            if (config.model_id.len + 1 + adp.name.len > MAX_MODEL_ID_BYTES) return error.InvalidConfig;
            for (config.lora_adapters[0..i]) |prev| {
                if (std.mem.eql(u8, prev.name, adp.name)) return error.InvalidConfig;
            }
        }
        var init_meta = config.model_detail orelse ModelDetailMeta{};
        if (init_meta.display_name.len > MAX_DISPLAY_NAME_BYTES) return error.InvalidConfig;
        if (!std.math.isFinite(init_meta.effective_bits_per_weight)) return error.InvalidConfig;
        if (config.cors_allow_origin) |origin| {
            if (origin.len > MAX_CORS_ORIGIN_BYTES) return error.InvalidConfig;
            if (std.mem.indexOfAny(u8, origin, "\r\n") != null) return error.InvalidConfig;
        }
        if (config.fim_tokens.prefix.len > MAX_FIM_TOKEN_BYTES or
            config.fim_tokens.suffix.len > MAX_FIM_TOKEN_BYTES or
            config.fim_tokens.middle.len > MAX_FIM_TOKEN_BYTES)
        {
            return error.InvalidConfig;
        }
        if (config.api_key) |ak| {
            if (ak.len > MAX_API_KEY_BYTES) return error.InvalidConfig;
        }
        const has_api_key = if (config.api_key) |ak| ak.len > 0 else false;
        if (!isLoopbackIpv4(config.host) and !has_api_key and !config.allow_unauthenticated_public_bind) {
            return error.InvalidConfig;
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
            sum = std.math.add(usize, sum, config.max_media_bytes) catch return error.InvalidConfig;
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
        const resp_slot_count = if (config.stateful_response_slots == DEFAULT_STATEFUL_RESPONSE_SLOTS and config.max_conns > config.stateful_response_slots)
            config.max_conns
        else
            config.stateful_response_slots;
        const per_resp_bytes = DEFAULT_STATEFUL_RESPONSE_PROMPT_BYTES +
            DEFAULT_STATEFUL_RESPONSE_REASONING_BYTES +
            DEFAULT_STATEFUL_RESPONSE_CONTENT_BYTES +
            DEFAULT_STATEFUL_RESPONSE_INPUT_ITEMS_BYTES;
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
                const ii_buf = response_store_slab[r_off .. r_off + DEFAULT_STATEFUL_RESPONSE_INPUT_ITEMS_BYTES];
                r_off += DEFAULT_STATEFUL_RESPONSE_INPUT_ITEMS_BYTES;
                rs.* = .{
                    .chatml_buf = c_buf,
                    .reasoning_buf = r_buf,
                    .content_buf = o_buf,
                    .input_items_buf = ii_buf,
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
            const media = arena_slab[slab_off .. slab_off + config.max_media_bytes];
            slab_off += config.max_media_bytes;

            const tok_start = i * config.token_ring_cap;
            const tok_ring = ring.SpscRing(TokenEvent).init(
                token_ring_storage[tok_start .. tok_start + config.token_ring_cap],
            ) catch return error.RingInitFailed;

            grammar_programs[i].reset();

            slot.* = .{
                .index = @intCast(i),
                .fd = -1,
                .state = .free,
                .read_armed = false,
                .write_armed = false,
                .headers_received = false,
                .is_network_tcp = false,
                .peer_ipv4 = null,
                .sent_continue = false,
                .request_counted = false,
                .api_cancelled = false,
                .aux_kind = .generate,
                .embed_req = .{ .model = "", .model_tag = .{ .base = "" } },
                .rerank_req = .{ .model = "", .model_tag = .{ .base = "" }, .query = "" },
                .aux_response_len = 0,
                .aux_err = .none,
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
                .media_buf = media,
                .prompt_token_ids_buf = undefined,
                .prompt_token_ids_len = 0,
                .prompt_spans_buf = undefined,
                .prompt_span_count = 0,
                .legacy_logprobs = null,
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
                .keep_alive = false,
                .has_completed_request = false,
                .rx_consumed_len = 0,
                .read_start_ns = 0,
                .last_activity_ns = 0,
                .admit_ns = 0,
                .prefill_done_ns = 0,
                .first_generated_token_ns = 0,
                .first_emitted_ns = 0,
                .first_token_ns = 0,
                .created_unix = 0,
                .prompt_tokens = 0,
                .cached_tokens = 0,
                .completion_tokens = 0,
                .reasoning_tokens = 0,
                .nonstream_reasoning_truncated = false,
                .nonstream_content_truncated = false,
                .tool_calls_truncated = false,
                .thinking_budget_tokens = null,
                .thinking_budget_exhausted = false,
                .reasoning_summary = .auto,
                .template_dialect = .chatml,
                .thinking_hmac = HmacSha256.init(&@as([32]u8, @splat(0))),
                .anthropic_block_had_delta = false,
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

        // Open either the native TLS 1.2/1.3 listener (`Network.framework`) or the
        // plain non-blocking TCP socket and register it on `kq_fd` under `UDATA_LISTEN`.
        var listen_fd: std.c.fd_t = -1;
        var tls_listener: ?tls_mod.TlsListener = null;
        var bound_port: u16 = 0;
        errdefer {
            if (listen_fd >= 0) _ = std.c.close(listen_fd);
            if (tls_listener) |tl| tl.deinit();
        }
        var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};

        if (config.tls) |tcfg| {
            var tls_lis = try tls_mod.TlsListener.init(.{
                .host_ipv4 = config.host,
                .port = config.port,
                .backlog = @intCast(@min(config.backlog, std.math.maxInt(u16))),
                .cert_bytes = tcfg.cert_bytes,
                .key_bytes = tcfg.key_bytes,
                .passphrase = tcfg.passphrase,
            });
            errdefer tls_lis.deinit();

            const notify_fd = tls_lis.notifyFd();
            if (notify_fd < 0) return error.KeventFailed;
            const listen_ev = [1]std.c.kevent64_s{.{
                .ident = @intCast(notify_fd),
                .filter = std.c.EVFILT.READ,
                .flags = std.c.EV.ADD | std.c.EV.ENABLE,
                .fflags = 0,
                .data = 0,
                .udata = UDATA_LISTEN,
                .ext = .{ 0, 0 },
            }};
            if (std.c.kevent64(kq_fd, &listen_ev, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null) < 0) {
                return error.KeventFailed;
            }
            bound_port = tls_lis.boundPort();
            tls_listener = tls_lis;
        } else {
            const tcp_fd = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
            if (tcp_fd < 0) return error.SocketFailed;
            listen_fd = tcp_fd;

            try setCloexec(tcp_fd);
            try setNonBlocking(tcp_fd);
            try setSockOptInt(tcp_fd, std.c.SOL.SOCKET, std.c.SO.REUSEADDR, 1);
            try setSockOptInt(tcp_fd, std.c.SOL.SOCKET, std.c.SO.NOSIGPIPE, 1);

            var addr = std.c.sockaddr.in{
                .port = std.mem.nativeToBig(u16, config.port),
                .addr = @bitCast(config.host),
            };
            if (std.c.bind(tcp_fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) != 0) {
                return error.BindFailed;
            }
            if (std.c.listen(tcp_fd, config.backlog) != 0) {
                return error.ListenFailed;
            }

            var bound_addr: std.c.sockaddr.in = std.mem.zeroes(std.c.sockaddr.in);
            var bound_len: std.c.socklen_t = @sizeOf(std.c.sockaddr.in);
            if (std.c.getsockname(tcp_fd, @ptrCast(&bound_addr), &bound_len) != 0) {
                return error.GetSockNameFailed;
            }
            bound_port = std.mem.bigToNative(u16, bound_addr.port);

            // Register TCP listening socket on `kq_fd`.
            const listen_ev = [1]std.c.kevent64_s{.{
                .ident = @intCast(tcp_fd),
                .filter = std.c.EVFILT.READ,
                .flags = std.c.EV.ADD | std.c.EV.ENABLE,
                .fflags = 0,
                .data = 0,
                .udata = UDATA_LISTEN,
                .ext = .{ 0, 0 },
            }};
            if (std.c.kevent64(kq_fd, &listen_ev, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null) < 0) {
                return error.KeventFailed;
            }
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
        var signature_key: [32]u8 = undefined;
        if (config.api_key) |ak| {
            if (ak.len > 0) {
                @memcpy(api_key_buf[0..ak.len], ak);
                api_key_len = @intCast(ak.len);
                var sh = Sha256.init(.{});
                sh.update("aminfer-anthropic-thinking-hmac-v1:");
                sh.update(ak);
                sh.final(&signature_key);
            } else {
                var r = rng_mod.RequestRng.init(null) catch rng_mod.RequestRng.initDeterministic(monotonicNowNs() ^ 0x5A17_9A23);
                r.random().bytes(&signature_key);
            }
        } else {
            var r = rng_mod.RequestRng.init(null) catch rng_mod.RequestRng.initDeterministic(monotonicNowNs() ^ 0x5A17_9A23);
            r.random().bytes(&signature_key);
        }

        const raw_arch = if (init_meta.architecture.len > 0) init_meta.architecture else "unknown";
        var model_arch_buf: [64]u8 = @splat(0);
        const model_arch_len: u8 = @intCast(@min(raw_arch.len, model_arch_buf.len));
        @memcpy(model_arch_buf[0..model_arch_len], raw_arch[0..model_arch_len]);

        var model_display_name_buf: [MAX_DISPLAY_NAME_BYTES]u8 = @splat(0);
        const model_display_name_len: u8 = @intCast(init_meta.display_name.len);
        if (model_display_name_len > 0) {
            @memcpy(model_display_name_buf[0..model_display_name_len], init_meta.display_name);
        }

        var cors_allow_origin_buf: [MAX_CORS_ORIGIN_BYTES]u8 = @splat(0);
        var cors_allow_origin_len: u16 = 0;
        if (config.cors_allow_origin) |origin| {
            if (origin.len > 0) {
                @memcpy(cors_allow_origin_buf[0..origin.len], origin);
                cors_allow_origin_len = @intCast(origin.len);
            }
        }

        var fim_tokens_stored: StoredFimTokens = .{};
        if (config.fim_tokens.prefix.len > 0) {
            @memcpy(fim_tokens_stored.prefix_buf[0..config.fim_tokens.prefix.len], config.fim_tokens.prefix);
            fim_tokens_stored.prefix_len = @intCast(config.fim_tokens.prefix.len);
        }
        if (config.fim_tokens.suffix.len > 0) {
            @memcpy(fim_tokens_stored.suffix_buf[0..config.fim_tokens.suffix.len], config.fim_tokens.suffix);
            fim_tokens_stored.suffix_len = @intCast(config.fim_tokens.suffix.len);
        }
        if (config.fim_tokens.middle.len > 0) {
            @memcpy(fim_tokens_stored.middle_buf[0..config.fim_tokens.middle.len], config.fim_tokens.middle);
            fim_tokens_stored.middle_len = @intCast(config.fim_tokens.middle.len);
        }

        init_meta.architecture = &.{};
        init_meta.display_name = &.{};
        if (init_meta.max_model_len == 0) {
            init_meta.max_model_len = @intCast(@min(config.max_prompt_tokens, std.math.maxInt(u32)));
        }
        if (init_meta.vocab_size == 0) {
            if (config.tokenizer) |tok| {
                init_meta.vocab_size = @intCast(@min(tok.vocabSize(), std.math.maxInt(u32)));
            }
        }
        init_meta.capabilities.embeddings = init_meta.capabilities.embeddings or (config.embedding_handler != null);
        init_meta.capabilities.rerank = init_meta.capabilities.rerank or (config.rerank_handler != null);
        init_meta.capabilities.lora = init_meta.capabilities.lora or (config.lora_adapters.len > 0);

        var lora_adapters_storage: [lora_mod.MAX_ADAPTERS]StoredLoraAdapter = @splat(.{});
        for (config.lora_adapters, 0..) |adp, i| {
            @memcpy(lora_adapters_storage[i].name_buf[0..adp.name.len], adp.name);
            lora_adapters_storage[i].name_len = @intCast(adp.name.len);
            if (adp.display_name.len > 0) {
                @memcpy(lora_adapters_storage[i].display_name_buf[0..adp.display_name.len], adp.display_name);
                lora_adapters_storage[i].display_name_len = @intCast(adp.display_name.len);
            }
            lora_adapters_storage[i].rank = adp.rank;
            lora_adapters_storage[i].lora_alpha = adp.lora_alpha;
            lora_adapters_storage[i].scale = adp.scale;
            lora_adapters_storage[i].wired_bytes = adp.wired_bytes;
        }
        const created_unix_sec: i64 = @intCast(@min(unixTimestampSec(), @as(u64, std.math.maxInt(i64))));

        var stored_config = config;
        stored_config.model_detail = null;
        stored_config.lora_adapters = &.{};
        stored_config.tls = null;
        stored_config.cors_allow_origin = null;
        stored_config.fim_tokens = .{};

        self.* = .{
            .config = stored_config,
            .model_id_buf = undefined,
            .model_id_len = @intCast(config.model_id.len),
            .created_unix_sec = created_unix_sec,
            .model_arch_buf = model_arch_buf,
            .model_arch_len = model_arch_len,
            .model_display_name_buf = model_display_name_buf,
            .model_display_name_len = model_display_name_len,
            .model_detail = init_meta,
            .lora_adapters_storage = lora_adapters_storage,
            .lora_adapter_count = @intCast(config.lora_adapters.len),
            .api_key_buf = api_key_buf,
            .api_key_len = api_key_len,
            .cors_allow_origin_buf = cors_allow_origin_buf,
            .cors_allow_origin_len = cors_allow_origin_len,
            .fim_tokens = fim_tokens_stored,
            .signature_key = signature_key,
            .unix_socket_path_buf = unix_path_buf,
            .unix_socket_path_len = unix_path_len,
            .kq_fd = kq_fd,
            .worker_kq_fd = worker_kq_fd,
            .listen_fd = listen_fd,
            .tls_listener = tls_listener,
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
        if (self.tls_listener) |*tl| {
            tl.deinit();
            self.tls_listener = null;
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

    pub fn isTlsEnabled(self: *const Server) bool {
        return self.tls_listener != null;
    }

    pub fn unixSocketPath(self: *const Server) ?[]const u8 {
        if (self.unix_socket_path_len == 0) return null;
        return self.unix_socket_path_buf[0..self.unix_socket_path_len];
    }

    pub fn modelId(self: *const Server) []const u8 {
        return self.model_id_buf[0..self.model_id_len];
    }

    pub fn modelArchSlice(self: *const Server) []const u8 {
        return self.model_arch_buf[0..self.model_arch_len];
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

    /// Lock-free query used by continuous-batching workers (`BatchServeRunner`)
    /// to check whether `slot_id` has exceeded its `thinking_budget_tokens` and
    /// requires injecting the dialect thinking-close delimiter into KV state.
    pub fn isBatchSlotThinkingBudgetExceeded(self: *const Server, slot_id: u16) bool {
        if (slot_id >= batch_sched.MAX_BATCH_SLOTS) return false;
        return self.batch_slot_streams[slot_id].thinkingBudgetExceeded();
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

    fn pushSlotControlEvent(self: *Server, slot: *ConnectionSlot, ev: TokenEvent) bool {
        while (true) {
            if (slot.cancelled.load(.acquire) or self.stop_requested.load(.acquire)) return false;
            if (slot.token_ring.tryPush(ev)) {
                self.wakeNetworkLoop();
                return true;
            }
            self.wakeNetworkLoop();
            sleepNs(50_000);
        }
    }

    fn executeAuxWorkerJob(self: *Server, job: *const InferenceJob) void {
        const slot = &self.slots[job.slot_index];
        const HEADER_RESERVE: usize = 768;
        if (!self.stop_requested.load(.acquire) and !slot.cancelled.load(.acquire)) {
            slot.aux_response_len = 0;
            slot.aux_err = .none;
            switch (job.kind) {
                .embeddings => {
                    if (self.embedding_handler) |handler| {
                        if (handler.embedFn(handler.ctx, &slot.embed_req, self.aux_embed_vectors)) |res| {
                            if (res.dim == 0 or slot.embed_req.input_count * res.dim > self.aux_embed_vectors.len) {
                                slot.aux_err = .embed_invalid_dim;
                            } else if (slot.tx_buf.len <= HEADER_RESERVE) {
                                slot.aux_err = .tx_overflow;
                            } else if (formatEmbeddingsResponse(
                                &slot.embed_req,
                                self.aux_embed_vectors[0 .. slot.embed_req.input_count * res.dim],
                                res,
                                slot.tx_buf[HEADER_RESERVE..],
                            )) |json_body| {
                                slot.aux_response_len = json_body.len;
                            } else |_| {
                                slot.aux_err = .tx_overflow;
                            }
                        } else |err| {
                            slot.aux_err = if (err == error.BufferTooSmall) .payload_too_large else .embed_failed;
                        }
                    } else {
                        slot.aux_err = .unavailable;
                    }
                },
                .rerank => {
                    if (self.rerank_handler) |handler| {
                        var items_buf: [MAX_RERANK_DOCS]RerankItem = undefined;
                        if (handler.rerankFn(handler.ctx, &slot.rerank_req, &items_buf)) |res| {
                            const n = @min(res.count, MAX_RERANK_DOCS);
                            if (slot.tx_buf.len <= HEADER_RESERVE) {
                                slot.aux_err = .tx_overflow;
                            } else if (formatRerankResponse(
                                &slot.rerank_req,
                                items_buf[0..n],
                                res.prompt_tokens,
                                slot.tx_buf[HEADER_RESERVE..],
                            )) |json_body| {
                                slot.aux_response_len = json_body.len;
                            } else |_| {
                                slot.aux_err = .tx_overflow;
                            }
                        } else |err| {
                            slot.aux_err = if (err == error.BufferTooSmall) .payload_too_large else .rerank_failed;
                        }
                    } else {
                        slot.aux_err = .unavailable;
                    }
                },
                .generate => {},
            }
            const ev = TokenEvent{
                .generation = job.generation,
                .kind = if (slot.aux_err == .none) .finished else .failed,
                .token_count = @intCast(@min(slot.aux_response_len, std.math.maxInt(u32))),
            };
            _ = self.pushSlotControlEvent(slot, ev);
        }
        slot.worker_active.store(false, .release);
        self.wakeNetworkLoop();
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
                if (job.kind != .generate) {
                    self.executeAuxWorkerJob(&job);
                    continue;
                }
                const slot = &self.slots[job.slot_index];
                if (!self.stop_requested.load(.acquire) and !slot.cancelled.load(.acquire)) {
                    self.active_batch_size.store(1, .release);
                    var stream = TokenStream{
                        .server = self,
                        .slot = slot,
                        .generation = job.generation,
                        .defer_terminal = true,
                    };
                    stream.setThinkingBudget(job.thinking_budget_tokens, job.template_dialect, job.starts_in_think);
                    handler.generateFn(handler.ctx, &job, &stream);
                    if (!stream.finished and !stream.isCancelled()) {
                        stream.finish(.stop, stream.prompt_tokens);
                    }
                    self.active_batch_size.store(0, .release);
                    if (stream.deferred_terminal) |term_ev| {
                        if (!stream.isCancelled()) {
                            _ = stream.pushEvent(term_ev);
                        }
                    }
                }
                // Release worker ownership of the slot and wake kqueue so cancelled
                // or keep-alive slots recycle immediately.
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

    fn enqueueBatchGenerateJob(
        self: *Server,
        handler: *const BatchEngineHandler,
        free_bsid: u16,
        job: *const InferenceJob,
    ) void {
        const conn = &self.slots[job.slot_index];
        if (self.stop_requested.load(.acquire) or conn.cancelled.load(.acquire)) {
            conn.worker_active.store(false, .release);
            self.wakeNetworkLoop();
            return;
        }

        self.batch_slot_streams[free_bsid] = .{
            .server = self,
            .slot = conn,
            .generation = job.generation,
        };
        self.batch_slot_streams[free_bsid].setThinkingBudget(job.thinking_budget_tokens, job.template_dialect, job.starts_in_think);

        const prompt_buf = self.batch_scheduler.slotPromptBuffer(free_bsid);
        const tok_res: TokenizeResult = if (handler.tokenizeWithBoundaryFn) |f|
            f(handler.ctx, job, prompt_buf) catch {
                self.batch_slot_streams[free_bsid].fail("prompt tokenization failed");
                conn.worker_active.store(false, .release);
                self.wakeNetworkLoop();
                return;
            }
        else blk: {
            const prompt_len = handler.tokenizeFn(handler.ctx, job, prompt_buf) catch {
                self.batch_slot_streams[free_bsid].fail("prompt tokenization failed");
                conn.worker_active.store(false, .release);
                self.wakeNetworkLoop();
                return;
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
            return;
        };

        self.batch_slot_conns[free_bsid] = job.slot_index;
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
        var pending_gen_job: ?InferenceJob = null;

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
                if (ctx.server.batch_slot_conns[slot_id]) |conn_idx| {
                    ctx.server.batch_slot_conns[slot_id] = null;
                    if (mapped_reason != .cancelled and !stream.finished and !stream.isCancelled()) {
                        stream.finish(mapped_reason, prompt_tokens);
                    }
                    ctx.server.slots[conn_idx].worker_active.store(false, .release);
                    ctx.server.wakeNetworkLoop();
                }
            }

            fn onError(ctx_ptr: *anyopaque, slot_id: u16, _: u64, _: batch_sched.BackendError) void {
                const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
                const stream = &ctx.server.batch_slot_streams[slot_id];
                if (ctx.server.batch_slot_conns[slot_id]) |conn_idx| {
                    ctx.server.batch_slot_conns[slot_id] = null;
                    if (!stream.finished and !stream.isCancelled()) {
                        stream.fail("batched inference step failed");
                    }
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
            if (pending_gen_job) |pj| {
                const pconn = &self.slots[pj.slot_index];
                if (stopping or pconn.cancelled.load(.acquire)) {
                    pconn.worker_active.store(false, .release);
                    self.wakeNetworkLoop();
                    pending_gen_job = null;
                }
            }

            // 2. Drain newly admitted jobs into all available preallocated batch slots
            //    and execute any `/v1/embeddings` / `/v1/rerank` jobs on the worker thread.
            if (pending_gen_job) |pj| {
                if (self.batch_scheduler.findFreeSlot()) |free_bsid| {
                    pending_gen_job = null;
                    self.enqueueBatchGenerateJob(&handler, free_bsid, &pj);
                }
            }
            while (pending_gen_job == null) {
                const job = self.job_ring.tryPop() orelse break;
                if (job.kind != .generate) {
                    self.executeAuxWorkerJob(&job);
                    continue;
                }
                if (self.batch_scheduler.findFreeSlot()) |free_bsid| {
                    self.enqueueBatchGenerateJob(&handler, free_bsid, &job);
                } else {
                    pending_gen_job = job;
                    break;
                }
            }

            // 3. On shutdown, release any remaining queued jobs cleanly before returning.
            if (self.stop_requested.load(.acquire)) {
                if (pending_gen_job) |pj| {
                    self.slots[pj.slot_index].worker_active.store(false, .release);
                    self.wakeNetworkLoop();
                    pending_gen_job = null;
                }
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

        const now_before = monotonicNowNs();
        self.sweepConnectionTimeouts(now_before);

        var eff_timeout_ns = timeout_ns;
        if (self.nextConnectionDeadlineNs(now_before)) |rem_ns| {
            eff_timeout_ns = if (eff_timeout_ns) |cur| @min(cur, rem_ns) else rem_ns;
        }

        var empty_changes: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
        var ts: std.c.timespec = undefined;
        const ts_ptr: ?*const std.c.timespec = if (eff_timeout_ns) |ns| blk: {
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
                if (self.tls_listener) |*tl| {
                    self.acceptPendingTlsConnections(tl);
                } else {
                    self.acceptPendingConnectionsOn(self.listen_fd, true);
                }
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
        self.sweepConnectionTimeouts(monotonicNowNs());
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

    fn countActiveConnsFromPeer(self: *const Server, peer_ipv4: [4]u8) usize {
        var count: usize = 0;
        for (self.slots) |*slot| {
            if (slot.state != .free) {
                if (slot.peer_ipv4) |p| {
                    if (std.mem.eql(u8, &p, &peer_ipv4)) count += 1;
                }
            }
        }
        return count;
    }

    fn shouldRejectPeerLimit(self: *const Server, peer_ipv4: [4]u8) bool {
        if (isLoopbackIpv4(peer_ipv4)) return false;
        if (self.config.max_conns_per_peer == 0) return false;
        return self.countActiveConnsFromPeer(peer_ipv4) >= self.config.max_conns_per_peer;
    }

    fn rejectClientWith503(self: *Server, client_fd: std.c.fd_t) void {
        _ = self.rejected_503_count.fetchAdd(1, .monotonic);
        _ = std.c.send(client_fd, STATIC_503_RESPONSE.ptr, STATIC_503_RESPONSE.len, 0);
        _ = std.c.shutdown(client_fd, std.c.SHUT.WR);
        var drain_sink: [512]u8 = undefined;
        _ = std.c.recv(client_fd, &drain_sink, drain_sink.len, 0);
        _ = std.c.close(client_fd);
    }

    fn admitAcceptedClientFd(self: *Server, client_fd: std.c.fd_t, is_network_tcp: bool, peer_ipv4: ?[4]u8) void {
        if (peer_ipv4) |p| {
            if (self.shouldRejectPeerLimit(p)) {
                self.rejectClientWith503(client_fd);
                return;
            }
        }
        const slot = self.findFreeSlot() orelse {
            // Connection pool completely saturated: best-effort zero-alloc 503 + graceful half-close.
            self.rejectClientWith503(client_fd);
            return;
        };

        slot.resetForConnection(client_fd, is_network_tcp, peer_ipv4);
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
            return;
        }
        slot.read_armed = true;
    }

    fn acceptPendingTlsConnections(self: *Server, tl: *tls_mod.TlsListener) void {
        while (true) {
            const client_fd = (tl.popAcceptedFd() catch return) orelse return;
            // Accepted TLS bridge fd is already an AF_UNIX SOCK_STREAM socketpair with
            // FD_CLOEXEC, O_NONBLOCK, and SO_NOSIGPIPE configured by `apple_bridge.m`;
            // do NOT apply TCP_NODELAY to it. TLS clients arrive over TCP so Host validation applies.
            self.admitAcceptedClientFd(client_fd, true, null);
        }
    }

    fn acceptPendingConnectionsOn(self: *Server, listen_fd: std.c.fd_t, is_tcp: bool) void {
        if (listen_fd < 0) return;
        while (true) {
            var peer_addr: std.c.sockaddr.in = std.mem.zeroes(std.c.sockaddr.in);
            var peer_len: std.c.socklen_t = @sizeOf(std.c.sockaddr.in);
            const client_fd = if (is_tcp)
                std.c.accept(listen_fd, @ptrCast(&peer_addr), &peer_len)
            else
                std.c.accept(listen_fd, null, null);
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

            const peer_ipv4: ?[4]u8 = if (is_tcp and peer_len >= @sizeOf(std.c.sockaddr.in))
                @bitCast(peer_addr.addr)
            else
                null;
            self.admitAcceptedClientFd(client_fd, is_tcp, peer_ipv4);
        }
    }

    fn setReadInterest(self: *Server, slot: *ConnectionSlot, enable: bool) void {
        if (slot.fd < 0 or slot.read_armed == enable) return;
        const flags: u16 = if (enable) std.c.EV.ENABLE else std.c.EV.DISABLE;
        const change = [1]std.c.kevent64_s{.{
            .ident = @intCast(slot.fd),
            .filter = std.c.EVFILT.READ,
            .flags = flags,
            .fflags = 0,
            .data = 0,
            .udata = slot.index,
            .ext = .{ 0, 0 },
        }};
        var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
        _ = std.c.kevent64(self.kq_fd, &change, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null);
        slot.read_armed = enable;
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

    fn effectiveReadTimeoutMs(self: *const Server, slot: *const ConnectionSlot) u32 {
        const body_ms = self.config.http_read_timeout_ms;
        if (body_ms == 0) return 0;
        if (slot.headers_received) return body_ms;
        const hdr_ms = self.config.header_read_timeout_ms;
        return if (hdr_ms > 0) @min(hdr_ms, body_ms) else body_ms;
    }

    fn nextConnectionDeadlineNs(self: *const Server, now_ns: u64) ?u64 {
        const idle_ms = self.config.keepalive_idle_timeout_ms;
        var min_rem: ?u64 = null;
        for (self.slots) |*slot| {
            if (slot.state != .reading_request) continue;
            const read_ms = self.effectiveReadTimeoutMs(slot);
            const deadline_ns: ?u64 = if (slot.read_start_ns > 0 and read_ms > 0)
                slot.read_start_ns + @as(u64, read_ms) * std.time.ns_per_ms
            else if (slot.read_start_ns == 0 and slot.has_completed_request and idle_ms > 0)
                slot.last_activity_ns + @as(u64, idle_ms) * std.time.ns_per_ms
            else
                null;
            if (deadline_ns) |dl| {
                const rem = dl -| now_ns;
                min_rem = if (min_rem) |prev| @min(prev, rem) else rem;
            }
        }
        return min_rem;
    }

    fn sweepConnectionTimeouts(self: *Server, now_ns: u64) void {
        const idle_ms = self.config.keepalive_idle_timeout_ms;
        for (self.slots) |*slot| {
            if (slot.state != .reading_request) continue;
            const read_ms = self.effectiveReadTimeoutMs(slot);
            if (slot.read_start_ns > 0 and read_ms > 0) {
                if ((now_ns -| slot.read_start_ns) >= @as(u64, read_ms) * std.time.ns_per_ms) {
                    slot.keep_alive = false;
                    self.sendErrorResponse(
                        slot,
                        408,
                        "Request Timeout",
                        null,
                        "request read timed out",
                        "invalid_request_error",
                        "request_timeout",
                    );
                }
            } else if (slot.read_start_ns == 0 and slot.has_completed_request and idle_ms > 0) {
                if ((now_ns -| slot.last_activity_ns) >= @as(u64, idle_ms) * std.time.ns_per_ms) {
                    self.closeSlotSocketAndRelease(slot, false);
                }
            }
        }
    }

    fn finishHttpExchange(self: *Server, slot: *ConnectionSlot) void {
        if (!slot.keep_alive or slot.fd < 0) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        if (slot.worker_active.load(.acquire)) {
            // Worker is still unwinding its turn (e.g. early `stop`-sequence completion):
            // park keep-alive recycling until worker clears `worker_active` (`F6`).
            self.setWriteInterest(slot, false);
            self.setReadInterest(slot, false);
            slot.state = .draining_worker;
            return;
        }
        self.completeKeepAliveRecycle(slot);
    }

    fn completeKeepAliveRecycle(self: *Server, slot: *ConnectionSlot) void {
        if (!slot.keep_alive or slot.fd < 0) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        if (slot.worker_active.load(.acquire)) {
            self.setWriteInterest(slot, false);
            self.setReadInterest(slot, false);
            slot.state = .draining_worker;
            return;
        }
        const rem = slot.rx_len -| slot.rx_consumed_len;
        if (rem > 0 and slot.rx_consumed_len > 0) {
            std.mem.copyForwards(
                u8,
                slot.rx_buf[0..rem],
                slot.rx_buf[slot.rx_consumed_len..slot.rx_len],
            );
        }
        self.setWriteInterest(slot, false);
        self.setReadInterest(slot, true);
        slot.resetForNextKeepAliveRequest(rem, monotonicNowNs());
        if (rem > 0) {
            self.tryDispatchHttpRequest(slot, false);
        }
    }

    fn closeSlotSocketAndRelease(self: *Server, slot: *ConnectionSlot, was_client_abort: bool) void {
        if (slot.fd >= 0) {
            _ = std.c.close(slot.fd);
            slot.fd = -1;
            slot.read_armed = false;
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
        slot.peer_ipv4 = null;
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
                        const now = monotonicNowNs();
                        if (slot.read_start_ns == 0) slot.read_start_ns = now;
                        slot.last_activity_ns = now;
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
            .free, .waiting_worker_cancel, .draining_worker => {},
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
            self.finishHttpExchange(slot);
        } else if (slot.state == .reading_request and slot.pendingTxLen() == 0 and slot.rx_len > 0) {
            self.tryDispatchHttpRequest(slot, false);
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

    fn checkApiKeyHeaders(self: *const Server, hdr: *const HttpHeadersResult) bool {
        if (self.api_key_len == 0) return true;
        const expected = self.api_key_buf[0..self.api_key_len];
        if (hdr.authorization_bearer) |b| {
            if (constantTimeEql(b, expected)) return true;
        }
        if (hdr.x_api_key) |k| {
            if (constantTimeEql(k, expected)) return true;
        }
        return false;
    }

    fn tryDispatchHttpRequest(self: *Server, slot: *ConnectionSlot, eof_seen: bool) void {
        const hdr = parseHttpHeaders(slot.rx_buf[0..slot.rx_len]) catch |err| switch (err) {
            error.Incomplete => {
                if (slot.rx_len >= slot.rx_buf.len) {
                    slot.keep_alive = false;
                    self.sendErrorResponse(slot, 413, "Payload Too Large", null, "request headers exceed preallocated slot buffer", "invalid_request_error", "payload_too_large");
                } else if (eof_seen) {
                    const was_abort = (slot.rx_len > 0) or !slot.has_completed_request;
                    self.closeSlotSocketAndRelease(slot, was_abort);
                }
                return;
            },
            error.BadRequest => {
                slot.keep_alive = false;
                self.sendErrorResponse(slot, 400, "Bad Request", null, "malformed HTTP/1.1 request", "invalid_request_error", "bad_request");
                return;
            },
            error.PayloadTooLarge => {
                slot.keep_alive = false;
                self.sendErrorResponse(slot, 413, "Payload Too Large", null, "request exceeds preallocated slot buffer", "invalid_request_error", "payload_too_large");
                return;
            },
        };

        slot.headers_received = true;
        slot.keep_alive = !hdr.connection_close;

        // Validate `Host` header against DNS rebinding on TCP sockets when `api_key` is unset (`F2`).
        if (self.api_key_len == 0 and !self.config.allow_any_host and slot.is_network_tcp) {
            const raw_host = hdr.host orelse {
                slot.keep_alive = false;
                self.sendErrorResponse(slot, 400, "Bad Request", null, "missing Host header", "invalid_request_error", "bad_request");
                return;
            };
            switch (validateHostHeader(raw_host, self.config.host)) {
                .ok => {},
                .bad_request => {
                    slot.keep_alive = false;
                    self.sendErrorResponse(slot, 400, "Bad Request", null, "invalid Host header", "invalid_request_error", "bad_request");
                    return;
                },
                .forbidden => {
                    slot.keep_alive = false;
                    self.sendErrorResponse(slot, 403, "Forbidden", null, "Host header rejected", "invalid_request_error", "forbidden_host");
                    return;
                },
            }
        }

        const route_path = hdr.path;
        const norm_path = normalizeApiPath(route_path);
        const is_preflight = (hdr.method == .options and isKnownRoute(route_path, norm_path));
        const is_health = std.mem.eql(u8, route_path, "/health");

        // Early `--api-key` validation (`L3`) as soon as `\r\n\r\n` arrives, BEFORE `100 Continue`
        // and BEFORE buffering multi-MB request bodies.
        if (self.api_key_len > 0 and !is_preflight and !is_health) {
            if (!self.checkApiKeyHeaders(&hdr)) {
                slot.keep_alive = false;
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

        // Support curl's `Expect: 100-continue` immediately so large POST bodies never stall (`L3` checked send).
        if (hdr.expect_continue and !slot.sent_continue) {
            slot.sent_continue = true;
            var sent: usize = 0;
            while (sent < STATIC_100_CONTINUE.len) {
                const n = std.c.send(slot.fd, STATIC_100_CONTINUE[sent..].ptr, STATIC_100_CONTINUE.len - sent, 0);
                if (n > 0) {
                    sent += @intCast(n);
                } else if (n < 0) {
                    const err = std.c.errno(n);
                    if (err == .INTR) continue;
                    if (err == .AGAIN) {
                        slot.appendTx(STATIC_100_CONTINUE[sent..]) catch {
                            self.closeSlotSocketAndRelease(slot, true);
                            return;
                        };
                        self.setWriteInterest(slot, true);
                        break;
                    }
                    self.closeSlotSocketAndRelease(slot, true);
                    return;
                } else {
                    self.closeSlotSocketAndRelease(slot, true);
                    return;
                }
            }
        }

        // Wait until the full request body (if any) has arrived.
        var body: []const u8 = "";
        slot.rx_consumed_len = hdr.header_end;
        if (hdr.chunked) {
            const max_body_bytes = slot.rx_buf.len - hdr.header_end;
            const decoded = decodeChunkedInPlace(slot.rx_buf[hdr.header_end..slot.rx_len], max_body_bytes) catch |err| switch (err) {
                error.BadRequest => {
                    slot.keep_alive = false;
                    self.sendErrorResponse(slot, 400, "Bad Request", null, "malformed chunked request body", "invalid_request_error", "bad_request");
                    return;
                },
                error.PayloadTooLarge => {
                    slot.keep_alive = false;
                    self.sendErrorResponse(slot, 413, "Payload Too Large", null, "chunked body exceeds preallocated slot buffer", "invalid_request_error", "payload_too_large");
                    return;
                },
            };
            switch (decoded) {
                .incomplete => {
                    if (slot.rx_len >= slot.rx_buf.len) {
                        slot.keep_alive = false;
                        self.sendErrorResponse(slot, 413, "Payload Too Large", null, "chunked body exceeds preallocated slot buffer", "invalid_request_error", "payload_too_large");
                    } else if (eof_seen) {
                        slot.keep_alive = false;
                        self.sendErrorResponse(slot, 400, "Bad Request", null, "truncated chunked request body", "invalid_request_error", "bad_request");
                    }
                    return;
                },
                .complete => |cd| {
                    body = slot.rx_buf[hdr.header_end .. hdr.header_end + cd.decoded_len];
                    slot.rx_consumed_len = hdr.header_end + cd.consumed_len;
                },
            }
        } else if (hdr.content_length) |cl| {
            const total_needed = std.math.add(usize, hdr.header_end, cl) catch {
                slot.keep_alive = false;
                self.sendErrorResponse(slot, 413, "Payload Too Large", null, "content-length overflow", "invalid_request_error", "payload_too_large");
                return;
            };
            if (total_needed > slot.rx_buf.len) {
                slot.keep_alive = false;
                self.sendErrorResponse(slot, 413, "Payload Too Large", null, "request body exceeds preallocated slot buffer", "invalid_request_error", "payload_too_large");
                return;
            }
            if (slot.rx_len < total_needed) {
                if (eof_seen) {
                    slot.keep_alive = false;
                    self.sendErrorResponse(slot, 400, "Bad Request", null, "truncated HTTP request body", "invalid_request_error", "bad_request");
                }
                return;
            }
            body = slot.rx_buf[hdr.header_end..total_needed];
            slot.rx_consumed_len = total_needed;
        }

        if (slot.pendingTxLen() > 0) return;

        // Handle CORS preflight (`OPTIONS`) BEFORE `--api-key` authentication on known routes (`C29`).
        if (is_preflight) {
            self.handleOptionsPreflight(slot);
            return;
        }

        // Route dispatch.
        if (is_health) {
            if (hdr.method != .get) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "GET", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleHealth(slot);
            return;
        }

        // Constant-time `--api-key` defense-in-depth check.
        if (!self.checkApiKeyHeaders(&hdr)) {
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

        if (std.mem.eql(u8, norm_path, "/models")) {
            if (hdr.method != .get) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "GET", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleModels(slot, hdr.query);
            return;
        }
        if (std.mem.startsWith(u8, norm_path, "/models/")) {
            const raw_id = norm_path["/models/".len..];
            if (raw_id.len == 0) {
                self.sendErrorResponse(slot, 404, "Not Found", null, "model not found", "invalid_request_error", "model_not_found");
                return;
            }
            if (hdr.method != .get) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "GET", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleModelById(slot, raw_id);
            return;
        }
        if (std.mem.eql(u8, norm_path, "/ngram-packs")) {
            if (hdr.method != .get) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "GET", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleNgramPacks(slot);
            return;
        }
        if (std.mem.eql(u8, norm_path, "/ngram-packs/select")) {
            if (hdr.method != .post) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "POST", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleNgramPacksSelect(slot, body);
            return;
        }
        if (std.mem.eql(u8, norm_path, "/embeddings")) {
            if (hdr.method != .post) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "POST", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleEmbeddings(slot, body);
            return;
        }
        if (std.mem.eql(u8, norm_path, "/rerank")) {
            if (hdr.method != .post) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "POST", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleRerank(slot, body);
            return;
        }
        if (std.mem.eql(u8, norm_path, "/messages/count_tokens")) {
            if (hdr.method != .post) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "POST", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            if (!self.validateAnthropicHeaders(slot, &hdr)) return;
            self.handleMessagesCountTokens(slot, body);
            return;
        }
        if (std.mem.eql(u8, norm_path, "/responses/input_tokens")) {
            if (hdr.method != .post) {
                self.sendErrorResponse(slot, 405, "Method Not Allowed", "POST", "method not allowed", "invalid_request_error", "method_not_allowed");
                return;
            }
            self.handleResponsesInputTokens(slot, body);
            return;
        }

        // Stateful `/v1/responses/{id}`, `/v1/responses/{id}/cancel`, and `/v1/responses/{id}/input_items` (`C18`/`C29`).
        if (std.mem.startsWith(u8, norm_path, "/responses/")) {
            const rest = norm_path["/responses/".len..];
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
            if (std.mem.endsWith(u8, rest, "/input_items")) {
                const resp_id = rest[0 .. rest.len - "/input_items".len];
                if (resp_id.len == 0 or std.mem.indexOfScalar(u8, resp_id, '/') != null) {
                    self.sendErrorResponse(slot, 404, "Not Found", null, "endpoint not found", "invalid_request_error", "not_found");
                    return;
                }
                if (hdr.method != .get) {
                    self.sendErrorResponse(slot, 405, "Method Not Allowed", "GET", "method not allowed", "invalid_request_error", "method_not_allowed");
                    return;
                }
                self.handleResponseInputItems(slot, resp_id, hdr.query);
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

        const endpoint: Endpoint = if (std.mem.eql(u8, norm_path, "/chat/completions"))
            .chat_completions
        else if (std.mem.eql(u8, norm_path, "/completions"))
            .completions
        else if (std.mem.eql(u8, norm_path, "/responses"))
            .responses
        else if (std.mem.eql(u8, norm_path, "/messages"))
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

    fn normalizeApiPath(path: []const u8) []const u8 {
        if (std.mem.startsWith(u8, path, "/v1/")) {
            return path["/v1".len..];
        }
        return path;
    }

    fn isKnownRoute(raw_path: []const u8, norm_path: []const u8) bool {
        if (std.mem.eql(u8, raw_path, "/health")) return true;
        if (std.mem.eql(u8, norm_path, "/models")) return true;
        if (std.mem.startsWith(u8, norm_path, "/models/") and norm_path.len > "/models/".len) return true;
        if (std.mem.eql(u8, norm_path, "/ngram-packs") or std.mem.eql(u8, norm_path, "/ngram-packs/select")) return true;
        if (std.mem.eql(u8, norm_path, "/embeddings") or std.mem.eql(u8, norm_path, "/rerank")) return true;
        if (std.mem.eql(u8, norm_path, "/messages") or std.mem.eql(u8, norm_path, "/messages/count_tokens")) return true;
        if (std.mem.eql(u8, norm_path, "/chat/completions") or std.mem.eql(u8, norm_path, "/completions")) return true;
        if (std.mem.eql(u8, norm_path, "/responses") or std.mem.eql(u8, norm_path, "/responses/input_tokens")) return true;
        if (std.mem.startsWith(u8, norm_path, "/responses/")) {
            const rest = norm_path["/responses/".len..];
            if (std.mem.endsWith(u8, rest, "/cancel")) {
                const rid = rest[0 .. rest.len - "/cancel".len];
                return rid.len > 0 and std.mem.indexOfScalar(u8, rid, '/') == null;
            }
            if (std.mem.endsWith(u8, rest, "/input_items")) {
                const rid = rest[0 .. rest.len - "/input_items".len];
                return rid.len > 0 and std.mem.indexOfScalar(u8, rid, '/') == null;
            }
            return rest.len > 0 and std.mem.indexOfScalar(u8, rest, '/') == null;
        }
        return false;
    }

    fn corsOriginSlice(self: *const Server) ?[]const u8 {
        if (self.cors_allow_origin_len == 0) return null;
        return self.cors_allow_origin_buf[0..self.cors_allow_origin_len];
    }

    fn appendCorsResponseHeaders(self: *const Server, w: *FixedWriter, is_preflight: bool) error{BufferTooSmall}!void {
        const origin = self.corsOriginSlice() orelse return;
        try w.writeAll("Access-Control-Allow-Origin: ");
        try w.writeAll(origin);
        try w.writeAll("\r\n");
        if (is_preflight) {
            try w.writeAll(
                "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n" ++
                    "Access-Control-Allow-Headers: Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta, OpenAI-Beta\r\n" ++
                    "Access-Control-Max-Age: 86400\r\n",
            );
        } else {
            try w.writeAll("Access-Control-Expose-Headers: x-request-id, Content-Type\r\n");
        }
    }

    fn handleOptionsPreflight(self: *Server, slot: *ConnectionSlot) void {
        var hdr_buf: [1024]u8 = undefined;
        var w = FixedWriter.init(&hdr_buf);
        w.print("HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: {s}\r\n", .{
            if (slot.keep_alive) "keep-alive" else "close",
        }) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        self.appendCorsResponseHeaders(&w, true) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        w.writeAll("\r\n") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        slot.appendTx(w.written()) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        slot.state = .draining_response;
        if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
            self.finishHttpExchange(slot);
        }
    }

    fn writeHttpJsonOkHeadersAndSend(
        self: *Server,
        slot: *ConnectionSlot,
        body_len: usize,
        header_reserve: usize,
        seed_used: ?u64,
        seed_source: ?EffectiveSeedSource,
    ) void {
        var hdr_buf: [1024]u8 = undefined;
        var hw = FixedWriter.init(&hdr_buf);
        hw.print(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: {s}\r\n",
            .{ body_len, if (slot.keep_alive) "keep-alive" else "close" },
        ) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        if (seed_used) |su| {
            hw.print("X-Aminfer-Seed-Used: {d}\r\n", .{su}) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
        }
        if (seed_source) |ss| {
            hw.print("X-Aminfer-Seed-Source: {s}\r\n", .{ss.name()}) catch {
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
        }
        self.appendCorsResponseHeaders(&hw, false) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        hw.writeAll("\r\n") catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        if (hw.pos > header_reserve) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        const hdr_start = header_reserve - hw.pos;
        @memcpy(slot.tx_buf[hdr_start..header_reserve], hw.written());
        slot.tx_read_pos = hdr_start;
        slot.tx_write_pos = header_reserve + body_len;
        slot.state = .draining_response;
        if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
            self.finishHttpExchange(slot);
        }
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
        var hdr_buf: [1024]u8 = undefined;
        var w = FixedWriter.init(&hdr_buf);
        w.print("HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: {s}\r\n", .{
            status_code,
            reason,
            json_body.len,
            if (slot.keep_alive) "keep-alive" else "close",
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
        self.appendCorsResponseHeaders(&w, false) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
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
            self.finishHttpExchange(slot);
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
                error.RemoteMediaNotSupported, error.InvalidMediaBase64, error.TooManyMediaItems, error.MediaTooLarge => "invalid 'domain' value",
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

    inline fn jsonBoolStr(b: bool) []const u8 {
        return if (b) "true" else "false";
    }

    inline fn sanitizeFiniteF64(v: f64) f64 {
        if (!std.math.isFinite(v) or v == 0.0) return 0.0;
        return v;
    }

    inline fn sanitizeFiniteF32(v: f32) f32 {
        if (!std.math.isFinite(v) or v == 0.0) return 0.0;
        return v;
    }

    fn writeModelCapabilitiesJson(w: *FixedWriter, caps: ModelCapabilities) error{BufferTooSmall}!void {
        try w.print(
            "{{\"chat\":{s},\"responses\":{s},\"messages\":{s},\"tool_calling\":{s},\"structured_outputs\":{s},\"reasoning\":{s},\"vision\":{s},\"embeddings\":{s},\"rerank\":{s},\"lora\":{s},\"speculative_decoding\":{s},\"moe\":{s}}}",
            .{
                jsonBoolStr(caps.chat),
                jsonBoolStr(caps.responses),
                jsonBoolStr(caps.messages),
                jsonBoolStr(caps.tool_calling),
                jsonBoolStr(caps.structured_outputs),
                jsonBoolStr(caps.reasoning),
                jsonBoolStr(caps.vision),
                jsonBoolStr(caps.embeddings),
                jsonBoolStr(caps.rerank),
                jsonBoolStr(caps.lora),
                jsonBoolStr(caps.speculative_decoding),
                jsonBoolStr(caps.moe),
            },
        );
    }

    fn writeBaseModelJson(self: *const Server, w: *FixedWriter) error{BufferTooSmall}!void {
        const base_id = self.modelId();
        const arch = self.modelArchSlice();
        const meta = self.model_detail;
        const live_wired = self.wired_mib.load(.monotonic);
        const effective_wired_mib = if (live_wired > 0) live_wired else meta.wired_mib;
        const safe_bpw = sanitizeFiniteF64(meta.effective_bits_per_weight);
        var ts_buf: [32]u8 = undefined;
        const created_at = formatRfc3339Utc(self.created_unix_sec, &ts_buf);
        const display_name = if (self.model_display_name_len > 0)
            self.model_display_name_buf[0..self.model_display_name_len]
        else
            base_id;
        const max_output: u32 = if (meta.max_output_tokens > 0) meta.max_output_tokens else meta.max_model_len;

        try w.writeAll("{\"id\":\"");
        try w.writeJsonEscaped(base_id);
        try w.writeAll("\",\"object\":\"model\",\"type\":\"model\",\"display_name\":\"");
        try w.writeJsonEscaped(display_name);
        try w.print("\",\"created\":{d},\"created_at\":\"{s}\",\"owned_by\":\"aminfer\",\"root\":\"", .{
            self.created_unix_sec,
            created_at,
        });
        try w.writeJsonEscaped(base_id);
        try w.writeAll("\",\"parent\":null,\"architecture\":\"");
        try w.writeJsonEscaped(arch);
        try w.print(
            "\",\"max_model_len\":{d},\"max_input_tokens\":{d},\"max_tokens\":{d},\"vocab_size\":{d},\"effective_bits_per_weight\":{d:.6},\"wired_mib\":{d},\"cold_bytes\":{d},\"capabilities\":",
            .{
                meta.max_model_len,
                meta.max_model_len,
                max_output,
                meta.vocab_size,
                safe_bpw,
                effective_wired_mib,
                meta.cold_bytes,
            },
        );
        try writeModelCapabilitiesJson(w, meta.capabilities);
        try w.writeByte('}');
    }

    fn writeLoraModelJson(
        self: *const Server,
        w: *FixedWriter,
        adp: *const StoredLoraAdapter,
    ) error{BufferTooSmall}!void {
        const base_id = self.modelId();
        const arch = self.modelArchSlice();
        const meta = self.model_detail;
        const safe_alpha = sanitizeFiniteF32(adp.lora_alpha);
        const safe_scale = sanitizeFiniteF32(adp.scale);
        var ts_buf: [32]u8 = undefined;
        const created_at = formatRfc3339Utc(self.created_unix_sec, &ts_buf);
        const max_output: u32 = if (meta.max_output_tokens > 0) meta.max_output_tokens else meta.max_model_len;

        try w.writeAll("{\"id\":\"");
        try w.writeJsonEscaped(base_id);
        try w.writeByte(':');
        try w.writeJsonEscaped(adp.nameSlice());
        try w.writeAll("\",\"object\":\"model\",\"type\":\"model\",\"display_name\":\"");
        if (adp.display_name_len > 0) {
            try w.writeJsonEscaped(adp.displayNameSlice());
        } else {
            try w.writeJsonEscaped(base_id);
            try w.writeByte(':');
            try w.writeJsonEscaped(adp.nameSlice());
        }
        try w.print("\",\"created\":{d},\"created_at\":\"{s}\",\"owned_by\":\"aminfer\",\"root\":\"", .{
            self.created_unix_sec,
            created_at,
        });
        try w.writeJsonEscaped(base_id);
        try w.writeAll("\",\"parent\":\"");
        try w.writeJsonEscaped(base_id);
        try w.writeAll("\",\"architecture\":\"");
        try w.writeJsonEscaped(arch);
        try w.print(
            "\",\"max_model_len\":{d},\"max_input_tokens\":{d},\"max_tokens\":{d},\"vocab_size\":{d},\"rank\":{d},\"lora_alpha\":{d:.6},\"scale\":{d:.6},\"wired_bytes\":{d},\"capabilities\":",
            .{
                meta.max_model_len,
                meta.max_model_len,
                max_output,
                meta.vocab_size,
                adp.rank,
                safe_alpha,
                safe_scale,
                adp.wired_bytes,
            },
        );
        try writeModelCapabilitiesJson(w, meta.capabilities);
        try w.writeByte('}');
    }

    fn modelItemIdAt(self: *const Server, idx: usize, buf: *[MAX_MODEL_ID_BYTES]u8) []const u8 {
        if (idx == 0) return self.modelId();
        const adp = &self.lora_adapters_storage[idx - 1];
        return std.fmt.bufPrint(buf, "{s}:{s}", .{ self.modelId(), adp.nameSlice() }) catch self.modelId();
    }

    fn writeModelItemAt(self: *const Server, w: *FixedWriter, idx: usize) error{BufferTooSmall}!void {
        if (idx == 0) {
            try self.writeBaseModelJson(w);
        } else {
            try self.writeLoraModelJson(w, &self.lora_adapters_storage[idx - 1]);
        }
    }

    fn writeModelsListJson(
        self: *const Server,
        w: *FixedWriter,
        limit: usize,
        after_id: ?[]const u8,
        before_id: ?[]const u8,
    ) error{BufferTooSmall}!void {
        const total: usize = 1 + @as(usize, self.lora_adapter_count);
        var id_tmp: [MAX_MODEL_ID_BYTES]u8 = undefined;

        var range_start: usize = 0;
        var range_end: usize = total;
        if (after_id) |aid| {
            var matched: ?usize = null;
            for (0..total) |i| {
                if (std.mem.eql(u8, self.modelItemIdAt(i, &id_tmp), aid)) {
                    matched = i;
                    break;
                }
            }
            range_start = if (matched) |m| m + 1 else total;
        }
        if (before_id) |bid| {
            var matched: ?usize = null;
            for (0..total) |i| {
                if (std.mem.eql(u8, self.modelItemIdAt(i, &id_tmp), bid)) {
                    matched = i;
                    break;
                }
            }
            range_end = if (matched) |m| @max(range_start, m) else range_start;
        }

        const slice_start: usize, const slice_end: usize, const has_more: bool = if (before_id != null and after_id == null) blk: {
            const se = range_end;
            const ss = if (range_end - range_start > limit) range_end - limit else range_start;
            break :blk .{ ss, se, ss > range_start };
        } else blk: {
            const ss = range_start;
            const se = @min(range_end, range_start + limit);
            break :blk .{ ss, se, se < range_end };
        };

        try w.writeAll("{\"object\":\"list\",\"data\":[");
        var idx = slice_start;
        while (idx < slice_end) : (idx += 1) {
            if (idx > slice_start) try w.writeByte(',');
            try self.writeModelItemAt(w, idx);
        }
        try w.writeAll("],\"first_id\":");
        if (slice_start < slice_end) {
            try w.writeByte('"');
            try w.writeJsonEscaped(self.modelItemIdAt(slice_start, &id_tmp));
            try w.writeByte('"');
        } else {
            try w.writeAll("null");
        }
        try w.writeAll(",\"last_id\":");
        if (slice_start < slice_end) {
            try w.writeByte('"');
            try w.writeJsonEscaped(self.modelItemIdAt(slice_end - 1, &id_tmp));
            try w.writeByte('"');
        } else {
            try w.writeAll("null");
        }
        try w.print(",\"has_more\":{s}}}", .{jsonBoolStr(has_more)});
    }

    fn sendModelJsonOrError(self: *Server, slot: *ConnectionSlot, json_body: []const u8) void {
        if (json_body.len + 768 > slot.tx_buf.len) {
            self.sendErrorResponse(slot, 500, "Internal Server Error", null, "model response exceeds slot tx buffer", "server_error", "response_too_large");
            return;
        }
        self.sendJsonResponse(slot, 200, "OK", null, null, null, json_body);
    }

    fn handleModels(self: *Server, slot: *ConnectionSlot, query: []const u8) void {
        var limit: usize = 20;
        var after_buf: [MAX_MODEL_ID_BYTES]u8 = undefined;
        var after_id: ?[]const u8 = null;
        var before_buf: [MAX_MODEL_ID_BYTES]u8 = undefined;
        var before_id: ?[]const u8 = null;

        if (query.len > 0) {
            var pairs = std.mem.splitScalar(u8, query, '&');
            while (pairs.next()) |pair| {
                if (pair.len == 0) continue;
                const eq = std.mem.indexOfScalar(u8, pair, '=');
                const k = if (eq) |e| pair[0..e] else pair;
                const v = if (eq) |e| pair[e + 1 ..] else "";
                if (std.mem.eql(u8, k, "limit")) {
                    const parsed = std.fmt.parseInt(i64, v, 10) catch {
                        self.sendErrorResponse(slot, 400, "Bad Request", null, "invalid limit query parameter", "invalid_request_error", "invalid_limit");
                        return;
                    };
                    if (parsed <= 0 or parsed > 1000) {
                        self.sendErrorResponse(slot, 400, "Bad Request", null, "limit must be between 1 and 1000", "invalid_request_error", "invalid_limit");
                        return;
                    }
                    limit = @intCast(parsed);
                } else if (std.mem.eql(u8, k, "after_id")) {
                    if (v.len > 0) {
                        after_id = percentDecodeUrlPath(v, &after_buf) catch |err| switch (err) {
                            error.InvalidPercentEncoding => {
                                self.sendErrorResponse(slot, 400, "Bad Request", null, "invalid percent-encoded after_id", "invalid_request_error", "invalid_cursor");
                                return;
                            },
                            error.BufferTooSmall => "__nonmatching__",
                        };
                    }
                } else if (std.mem.eql(u8, k, "before_id")) {
                    if (v.len > 0) {
                        before_id = percentDecodeUrlPath(v, &before_buf) catch |err| switch (err) {
                            error.InvalidPercentEncoding => {
                                self.sendErrorResponse(slot, 400, "Bad Request", null, "invalid percent-encoded before_id", "invalid_request_error", "invalid_cursor");
                                return;
                            },
                            error.BufferTooSmall => "__nonmatching__",
                        };
                    }
                }
            }
        }

        const HEADER_RESERVE: usize = 768;
        if (slot.tx_buf.len <= HEADER_RESERVE) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        var bw = FixedWriter.init(slot.tx_buf[HEADER_RESERVE..]);
        self.writeModelsListJson(&bw, limit, after_id, before_id) catch {
            self.sendErrorResponse(slot, 500, "Internal Server Error", null, "models list exceeds response buffer", "server_error", "response_too_large");
            return;
        };
        self.writeHttpJsonOkHeadersAndSend(slot, bw.pos, HEADER_RESERVE, null, null);
    }

    fn handleModelById(self: *Server, slot: *ConnectionSlot, raw_id: []const u8) void {
        var decoded_buf: [MAX_MODEL_ID_BYTES]u8 = undefined;
        const decoded_id = percentDecodeUrlPath(raw_id, &decoded_buf) catch |err| switch (err) {
            error.InvalidPercentEncoding => {
                self.sendErrorResponse(slot, 400, "Bad Request", null, "invalid percent-encoded model identifier", "invalid_request_error", "invalid_model_id");
                return;
            },
            error.BufferTooSmall => {
                self.sendErrorResponse(slot, 404, "Not Found", null, "model not found", "invalid_request_error", "model_not_found");
                return;
            },
        };

        const base_id = self.modelId();
        var body_buf: [4096]u8 = undefined;
        var w = FixedWriter.init(&body_buf);

        if (std.mem.eql(u8, decoded_id, base_id)) {
            self.writeBaseModelJson(&w) catch {
                self.sendErrorResponse(slot, 500, "Internal Server Error", null, "model detail exceeds response buffer", "server_error", "response_too_large");
                return;
            };
        } else if (decoded_id.len > base_id.len + 1 and
            std.mem.startsWith(u8, decoded_id, base_id) and
            decoded_id[base_id.len] == ':')
        {
            const adp_name = decoded_id[base_id.len + 1 ..];
            const matched: ?*const StoredLoraAdapter = for (self.lora_adapters_storage[0..self.lora_adapter_count]) |*adp| {
                if (std.mem.eql(u8, adp.nameSlice(), adp_name)) break adp;
            } else null;
            const adp = matched orelse {
                self.sendErrorResponse(slot, 404, "Not Found", null, "model not found", "invalid_request_error", "model_not_found");
                return;
            };
            self.writeLoraModelJson(&w, adp) catch {
                self.sendErrorResponse(slot, 500, "Internal Server Error", null, "model detail exceeds response buffer", "server_error", "response_too_large");
                return;
            };
        } else {
            self.sendErrorResponse(slot, 404, "Not Found", null, "model not found", "invalid_request_error", "model_not_found");
            return;
        }

        self.sendModelJsonOrError(slot, w.written());
    }

    // -----------------------------------------------------------------------
    // C18 / C21 synchronous handlers: count_tokens, embeddings, rerank,
    // stateful /v1/responses/{id} GET / DELETE / cancel
    // -----------------------------------------------------------------------

    fn sendParseErrorResponse(self: *Server, slot: *ConnectionSlot, err: ParseError) void {
        const msg = switch (err) {
            error.InvalidJson => "invalid JSON request body",
            error.MissingRequiredField => "missing required field in request body",
            error.InvalidFieldValue => "invalid parameter value in request body",
            error.UnsupportedFeature => "unsupported request feature",
            error.BufferTooSmall => "prompt exceeds preallocated slot buffer",
            error.DepthExceeded => "JSON nesting depth exceeded",
            error.RemoteMediaNotSupported => "remote media URLs are not supported; provide base64 data",
            error.InvalidMediaBase64 => "invalid base64 media data",
            error.TooManyMediaItems => "too many media items in request",
            error.MediaTooLarge => "decoded media exceeds preallocated max_media_bytes",
        };
        const err_code = switch (err) {
            error.BufferTooSmall, error.MediaTooLarge => "payload_too_large",
            error.RemoteMediaNotSupported => "unsupported_media_url",
            error.InvalidMediaBase64 => "invalid_media",
            error.TooManyMediaItems => "too_many_media_items",
            else => "bad_request",
        };
        const code: u16 = if (err == error.BufferTooSmall or err == error.MediaTooLarge) 413 else 400;
        const reason = if (code == 413) "Payload Too Large" else "Bad Request";
        self.sendErrorResponse(slot, code, reason, null, msg, "invalid_request_error", err_code);
    }

    fn handleMessagesCountTokens(self: *Server, slot: *ConnectionSlot, body: []const u8) void {
        const parsed = parseInferenceRequestFull(
            .messages,
            body,
            self.modelId(),
            &self.config.chat_template,
            null,
            slot.nonstream_content_buf,
            &self.signature_key,
            self.fim_tokens.asFimTokens(),
            self.tokenizer(),
            slot.media_buf,
            slot.prompt_buf,
            &slot.model_buf,
        ) catch |err| {
            self.sendParseErrorResponse(slot, err);
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
        const spans_opt: ?[]const chat_template.PromptSpan = if (parsed.prompt_span_count > 0)
            parsed.promptSpansSlice()
        else
            null;
        const ids = chat_template.encodePromptSpansWithScratch(tok, &self.aux_encode_scratch, prompt_bytes, spans_opt, self.aux_token_ids) catch {
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

    fn handleResponsesInputTokens(self: *Server, slot: *ConnectionSlot, body: []const u8) void {
        var prior_chatml: ?[]const u8 = null;
        var prior_spans: ?[]const chat_template.PromptSpan = null;
        var prev_probe_buf: [MAX_RESPONSE_ID_BYTES]u8 = undefined;
        if (findTopLevelJsonStringField(body, "previous_response_id", &prev_probe_buf)) |prev_id| {
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
                if (prev_rec.transcript_corrupt or prev_rec.status != .completed or prev_rec.chatml_len == 0) {
                    self.sendErrorResponse(
                        slot,
                        409,
                        "Conflict",
                        null,
                        "previous_response_id transcript is incomplete or truncated",
                        "invalid_request_error",
                        "incomplete_previous_response",
                    );
                    return;
                }
                prior_chatml = prev_rec.chatml_buf[0..prev_rec.chatml_len];
                if (prev_rec.chatml_span_count > 0) {
                    prior_spans = prev_rec.chatmlSpansSlice();
                }
            }
        }

        const parsed = parseInferenceRequestFullWithPriorSpans(
            .responses,
            body,
            self.modelId(),
            &self.config.chat_template,
            prior_chatml,
            prior_spans,
            slot.nonstream_content_buf,
            &self.signature_key,
            self.fim_tokens.asFimTokens(),
            self.tokenizer(),
            slot.media_buf,
            slot.prompt_buf,
            &slot.model_buf,
        ) catch |err| {
            self.sendParseErrorResponse(slot, err);
            return;
        };

        const tok = self.tokenizer() orelse {
            self.sendErrorResponse(
                slot,
                501,
                "Not Implemented",
                null,
                "native tokenizer not attached for /v1/responses/input_tokens",
                "api_error",
                "tokenizer_unavailable",
            );
            return;
        };

        const prompt_bytes = slot.prompt_buf[0..parsed.prompt_len];
        const r_spans_opt: ?[]const chat_template.PromptSpan = if (parsed.prompt_span_count > 0)
            parsed.promptSpansSlice()
        else
            null;
        const ids = chat_template.encodePromptSpansWithScratch(tok, &self.aux_encode_scratch, prompt_bytes, r_spans_opt, self.aux_token_ids) catch {
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
        w.print("{{\"object\":\"response.input_tokens\",\"input_tokens\":{d}}}", .{ids.len}) catch {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        };
        self.sendJsonResponse(slot, 200, "OK", null, null, null, w.written());
    }

    fn admitAuxWorkerRequest(
        self: *Server,
        slot: *ConnectionSlot,
        kind: JobKind,
        embed_req_opt: ?EmbedRequest,
        rerank_req_opt: ?RerankRequest,
    ) void {
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

        slot.generation +%= 1;
        slot.request_id = self.next_request_id;
        self.next_request_id +%= 1;
        slot.aux_kind = kind;
        slot.aux_response_len = 0;
        slot.aux_err = .none;
        slot.stream = false;
        if (embed_req_opt) |er| slot.embed_req = er;
        if (rerank_req_opt) |rr| slot.rerank_req = rr;
        slot.cancelled.store(false, .release);
        while (slot.token_ring.tryPop() != null) {}

        slot.worker_active.store(true, .release);
        slot.state = .streaming_active;
        slot.request_counted = true;
        _ = self.active_requests.fetchAdd(1, .acq_rel);
        _ = self.total_requests.fetchAdd(1, .monotonic);

        const model_tag = if (kind == .embeddings) slot.embed_req.model_tag else slot.rerank_req.model_tag;
        const job = InferenceJob{
            .slot_index = slot.index,
            .generation = slot.generation,
            .request_id = slot.request_id,
            .kind = kind,
            .endpoint = .completions,
            .prompt = "",
            .rng = &slot.rng,
            .seed_used = 0,
            .seed_source = .greedy,
            .max_tokens = 0,
            .temperature = 0.0,
            .stream = false,
            .ignore_eos = false,
            .enable_thinking = false,
            .model_base = model_tag.base,
            .lora_adapter = model_tag.adapter,
        };

        if (!self.job_ring.tryPush(job)) {
            slot.worker_active.store(false, .release);
            self.releaseRequestCount(slot);
            _ = self.rejected_503_count.fetchAdd(1, .monotonic);
            slot.keep_alive = false;
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
    }

    fn sendAuxErrorResponse(self: *Server, slot: *ConnectionSlot, err_kind: AuxWorkerError) void {
        switch (err_kind) {
            .none => self.closeSlotSocketAndRelease(slot, false),
            .unavailable => self.sendErrorResponse(
                slot,
                501,
                "Not Implemented",
                null,
                if (slot.aux_kind == .embeddings) "embeddings backend not configured" else "rerank backend not configured",
                "api_error",
                if (slot.aux_kind == .embeddings) "embeddings_unavailable" else "rerank_unavailable",
            ),
            .payload_too_large => self.sendErrorResponse(
                slot,
                413,
                "Payload Too Large",
                null,
                "request input exceeds preallocated backend scratch capacity",
                "invalid_request_error",
                "payload_too_large",
            ),
            .embed_failed => self.sendErrorResponse(
                slot,
                500,
                "Internal Server Error",
                null,
                "embedding computation failed",
                "server_error",
                "embedding_failed",
            ),
            .embed_invalid_dim => self.sendErrorResponse(
                slot,
                500,
                "Internal Server Error",
                null,
                "invalid embedding dimension returned by backend",
                "server_error",
                "embedding_failed",
            ),
            .rerank_failed => self.sendErrorResponse(
                slot,
                500,
                "Internal Server Error",
                null,
                "rerank computation failed",
                "server_error",
                "rerank_failed",
            ),
            .tx_overflow => self.sendErrorResponse(
                slot,
                500,
                "Internal Server Error",
                null,
                if (slot.aux_kind == .embeddings) "embeddings response exceeds tx buffer" else "rerank response exceeds tx buffer",
                "server_error",
                "tx_overflow",
            ),
        }
    }

    fn handleEmbeddings(self: *Server, slot: *ConnectionSlot, body: []const u8) void {
        const scratch_buf = if (slot.nonstream_content_buf.len >= slot.prompt_buf.len)
            slot.nonstream_content_buf
        else
            slot.prompt_buf;
        const req = parseEmbeddingsRequest(
            body,
            self.modelId(),
            &slot.model_buf,
            scratch_buf,
        ) catch |err| {
            const msg = switch (err) {
                error.InvalidJson => "invalid JSON request body",
                error.MissingRequiredField => "missing required field 'input'",
                error.InvalidFieldValue => "invalid field value in embeddings request",
                error.BufferTooSmall => "embeddings request exceeds preallocated scratch capacity",
                error.DepthExceeded => "JSON nesting depth exceeded",
                error.UnsupportedFeature => "unsupported embeddings feature",
                error.RemoteMediaNotSupported, error.InvalidMediaBase64, error.TooManyMediaItems, error.MediaTooLarge => "invalid embeddings media field",
            };
            const code: u16 = if (err == error.BufferTooSmall or err == error.MediaTooLarge) 413 else 400;
            const reason = if (code == 413) "Payload Too Large" else "Bad Request";
            self.sendErrorResponse(slot, code, reason, null, msg, "invalid_request_error", "bad_request");
            return;
        };

        if (self.embedding_handler == null) {
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
        }

        // Offload `embedFn` execution onto the worker thread via `job_ring` (`F7`)
        // so `kqueue` never acquires `model_mutex`.
        self.admitAuxWorkerRequest(slot, .embeddings, req, null);
    }

    fn handleRerank(self: *Server, slot: *ConnectionSlot, body: []const u8) void {
        const scratch_buf = if (slot.nonstream_content_buf.len >= slot.prompt_buf.len)
            slot.nonstream_content_buf
        else
            slot.prompt_buf;
        const req = parseRerankRequest(
            body,
            self.modelId(),
            &slot.model_buf,
            scratch_buf,
        ) catch |err| {
            const msg = switch (err) {
                error.InvalidJson => "invalid JSON request body",
                error.MissingRequiredField => "missing required field 'query' or 'documents'",
                error.InvalidFieldValue => "invalid field value in rerank request",
                error.BufferTooSmall => "rerank request exceeds preallocated scratch capacity",
                error.DepthExceeded => "JSON nesting depth exceeded",
                error.UnsupportedFeature => "unsupported rerank feature",
                error.RemoteMediaNotSupported, error.InvalidMediaBase64, error.TooManyMediaItems, error.MediaTooLarge => "invalid rerank media field",
            };
            const code: u16 = if (err == error.BufferTooSmall or err == error.MediaTooLarge) 413 else 400;
            const reason = if (code == 413) "Payload Too Large" else "Bad Request";
            self.sendErrorResponse(slot, code, reason, null, msg, "invalid_request_error", "bad_request");
            return;
        };

        if (self.rerank_handler == null) {
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
        }

        // Offload `rerankFn` execution onto the worker thread via `job_ring` (`F7`)
        // so `kqueue` never acquires `model_mutex`.
        self.admitAuxWorkerRequest(slot, .rerank, null, req);
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
    /// Never evicts active `.in_progress` slots (`C18`/`W24`).
    fn allocStatefulResponseSlot(self: *Server) ?u16 {
        if (self.response_store.len == 0) return null;
        var victim_idx: ?usize = null;
        var oldest_tick: u64 = std.math.maxInt(u64);
        for (self.response_store, 0..) |*rec, idx| {
            if (!rec.occupied) {
                return @intCast(idx);
            }
            // Only completed/cancelled/failed/incomplete slots are eligible for eviction.
            if (rec.status != .in_progress and rec.lru_tick < oldest_tick) {
                oldest_tick = rec.lru_tick;
                victim_idx = idx;
            }
        }
        return if (victim_idx) |vi| @intCast(vi) else null;
    }

    fn writeResponsesOutputArrayJson(
        bw: *FixedWriter,
        request_id: u64,
        item_status: []const u8,
        reasoning: []const u8,
        content: []const u8,
        summary_mode: ReasoningSummaryMode,
        has_tools: bool,
        tool_calls: []const StoredToolCall,
    ) error{BufferTooSmall}!void {
        try bw.writeByte('[');
        var need_comma = false;
        if (reasoning.len > 0) {
            try bw.print("{{\"id\":\"rs_{d}\",\"type\":\"reasoning\",\"status\":\"completed\",", .{request_id});
            try bw.writeAll("\"content\":[{\"type\":\"reasoning_text\",\"text\":\"");
            try bw.writeJsonEscaped(reasoning);
            try bw.writeAll("\"}],\"summary\":[");
            switch (summary_mode) {
                .none => {},
                .auto, .detailed => {
                    try bw.writeAll("{\"type\":\"summary_text\",\"text\":\"");
                    try bw.writeJsonEscaped(reasoning);
                    try bw.writeAll("\"}");
                },
                .concise => {
                    var cbuf: [320]u8 = undefined;
                    const concise = extractConciseReasoningSummary(reasoning, &cbuf);
                    if (concise.len > 0) {
                        try bw.writeAll("{\"type\":\"summary_text\",\"text\":\"");
                        try bw.writeJsonEscaped(concise);
                        try bw.writeAll("\"}");
                    }
                },
            }
            try bw.writeAll("]}");
            need_comma = true;
        }
        if (content.len > 0 or !has_tools) {
            if (need_comma) try bw.writeByte(',');
            try bw.print(
                "{{\"id\":\"msg_{d}\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"{s}\",\"content\":[{{\"type\":\"output_text\",\"text\":\"",
                .{ request_id, item_status },
            );
            try bw.writeJsonEscaped(content);
            try bw.writeAll("\"}]}");
            need_comma = true;
        }
        for (tool_calls, 0..) |*tc, idx| {
            if (need_comma) try bw.writeByte(',');
            try bw.print("{{\"id\":\"fc_{d}_{d}\",\"type\":\"function_call\",\"status\":\"completed\",\"call_id\":\"", .{
                request_id,
                idx,
            });
            try bw.writeJsonEscaped(tc.idSlice());
            try bw.writeAll("\",\"name\":\"");
            try bw.writeJsonEscaped(tc.nameSlice());
            try bw.writeAll("\",\"arguments\":\"");
            try bw.writeJsonEscaped(if (tc.args_len > 0) tc.argsSlice() else "{}");
            try bw.writeAll("\"}");
            need_comma = true;
        }
        try bw.writeByte(']');
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
        if (rec.status == .incomplete and rec.incomplete_reason != .none) {
            try bw.print(",\"incomplete_details\":{{\"reason\":\"{s}\"}}", .{rec.incomplete_reason.name()});
        }
        if (rec.previous_id_len > 0) {
            try bw.writeAll(",\"previous_response_id\":\"");
            try bw.writeJsonEscaped(rec.previousIdSlice());
            try bw.writeByte('"');
        }
        const reasoning = rec.reasoning_buf[0..rec.reasoning_len];
        const content = rec.content_buf[0..rec.content_len];
        try bw.writeAll(",\"output\":");
        try writeResponsesOutputArrayJson(
            bw,
            rec.request_id,
            rec.status.name(),
            reasoning,
            content,
            rec.reasoning_summary,
            rec.tool_call_count > 0,
            rec.tool_calls[0..rec.tool_call_count],
        );

        const total_tokens = rec.prompt_tokens +| rec.completion_tokens;
        try bw.writeAll(",\"output_text\":\"");
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
        const HEADER_RESERVE: usize = 768;
        if (slot.tx_buf.len <= HEADER_RESERVE) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        var bw = FixedWriter.init(slot.tx_buf[HEADER_RESERVE..]);
        writeResponseObjectJson(&bw, rec) catch {
            self.sendErrorResponse(slot, 500, "Internal Server Error", null, "response exceeds tx buffer", "server_error", "tx_overflow");
            return;
        };
        self.writeHttpJsonOkHeadersAndSend(slot, bw.pos, HEADER_RESERVE, null, null);
    }

    const InputItemsOrder = enum { asc, desc };

    fn formatInputItemId(item: *const StatefulInputItem, buf: *[48]u8) []const u8 {
        return std.fmt.bufPrint(buf, "msg_{d}_{d}", .{ item.origin_request_id, item.origin_item_idx }) catch "msg_0_0";
    }

    fn handleResponseInputItems(
        self: *Server,
        slot: *ConnectionSlot,
        resp_id: []const u8,
        query: []const u8,
    ) void {
        var limit: usize = 20;
        var order: InputItemsOrder = .desc;
        var after_buf: [64]u8 = undefined;
        var after_id: ?[]const u8 = null;

        if (query.len > 0) {
            var pairs = std.mem.splitScalar(u8, query, '&');
            while (pairs.next()) |pair| {
                if (pair.len == 0) continue;
                const eq = std.mem.indexOfScalar(u8, pair, '=');
                const k = if (eq) |e| pair[0..e] else pair;
                const v = if (eq) |e| pair[e + 1 ..] else "";
                if (std.mem.eql(u8, k, "limit")) {
                    const parsed = std.fmt.parseInt(i64, v, 10) catch {
                        self.sendErrorResponse(slot, 400, "Bad Request", null, "invalid limit query parameter", "invalid_request_error", "invalid_limit");
                        return;
                    };
                    if (parsed <= 0 or parsed > 100) {
                        self.sendErrorResponse(slot, 400, "Bad Request", null, "limit must be between 1 and 100", "invalid_request_error", "invalid_limit");
                        return;
                    }
                    limit = @intCast(parsed);
                } else if (std.mem.eql(u8, k, "order")) {
                    if (std.mem.eql(u8, v, "asc")) {
                        order = .asc;
                    } else if (std.mem.eql(u8, v, "desc")) {
                        order = .desc;
                    } else {
                        self.sendErrorResponse(slot, 400, "Bad Request", null, "order must be 'asc' or 'desc'", "invalid_request_error", "invalid_order");
                        return;
                    }
                } else if (std.mem.eql(u8, k, "after")) {
                    if (v.len > 0) {
                        after_id = percentDecodeUrlPath(v, &after_buf) catch |err| switch (err) {
                            error.InvalidPercentEncoding => {
                                self.sendErrorResponse(slot, 400, "Bad Request", null, "invalid percent-encoded after cursor", "invalid_request_error", "invalid_cursor");
                                return;
                            },
                            error.BufferTooSmall => "__nonmatching__",
                        };
                    }
                }
            }
        }

        const rec = self.findStatefulResponseById(resp_id) orelse {
            self.sendErrorResponse(slot, 404, "Not Found", null, "response not found", "invalid_request_error", "response_not_found");
            return;
        };

        const total: usize = rec.input_item_count;
        var id_tmp: [48]u8 = undefined;
        var start_pos: usize = 0;
        if (after_id) |aid| {
            var matched: ?usize = null;
            for (0..total) |pos| {
                const idx = if (order == .asc) pos else (total - 1 - pos);
                if (std.mem.eql(u8, formatInputItemId(&rec.input_items[idx], &id_tmp), aid)) {
                    matched = pos;
                    break;
                }
            }
            start_pos = if (matched) |m| m + 1 else total;
        }
        const end_pos = @min(total, start_pos + limit);
        const has_more = end_pos < total;

        const HEADER_RESERVE: usize = 768;
        if (slot.tx_buf.len <= HEADER_RESERVE) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        var bw = FixedWriter.init(slot.tx_buf[HEADER_RESERVE..]);
        const write_ok: bool = blk: {
            bw.writeAll("{\"object\":\"list\",\"data\":[") catch break :blk false;
            var pos = start_pos;
            while (pos < end_pos) : (pos += 1) {
                if (pos > start_pos) bw.writeByte(',') catch break :blk false;
                const idx = if (order == .asc) pos else (total - 1 - pos);
                const item = &rec.input_items[idx];
                const item_id = formatInputItemId(item, &id_tmp);
                const off: usize = item.offset;
                const len: usize = item.len;
                const text = if (off + len <= rec.input_items_buf.len)
                    rec.input_items_buf[off .. off + len]
                else
                    "";
                bw.print(
                    "{{\"id\":\"{s}\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"{s}\",\"content\":[{{\"type\":\"input_text\",\"text\":\"",
                    .{ item_id, item.role.name() },
                ) catch break :blk false;
                bw.writeJsonEscaped(text) catch break :blk false;
                bw.writeAll("\"}]}") catch break :blk false;
            }
            bw.writeAll("],\"first_id\":") catch break :blk false;
            if (start_pos < end_pos) {
                const f_idx = if (order == .asc) start_pos else (total - 1 - start_pos);
                bw.print("\"{s}\"", .{formatInputItemId(&rec.input_items[f_idx], &id_tmp)}) catch break :blk false;
            } else {
                bw.writeAll("null") catch break :blk false;
            }
            bw.writeAll(",\"last_id\":") catch break :blk false;
            if (start_pos < end_pos) {
                const l_idx = if (order == .asc) (end_pos - 1) else (total - end_pos);
                bw.print("\"{s}\"", .{formatInputItemId(&rec.input_items[l_idx], &id_tmp)}) catch break :blk false;
            } else {
                bw.writeAll("null") catch break :blk false;
            }
            bw.print(",\"has_more\":{s}}}", .{jsonBoolStr(has_more)}) catch break :blk false;
            break :blk true;
        };
        if (!write_ok) {
            self.sendErrorResponse(slot, 500, "Internal Server Error", null, "input items exceed tx buffer", "server_error", "tx_overflow");
            return;
        }
        self.writeHttpJsonOkHeadersAndSend(slot, bw.pos, HEADER_RESERVE, null, null);
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
                    self.slots[cidx].api_cancelled = true;
                    if (!self.slots[cidx].cancelled.swap(true, .acq_rel)) {
                        _ = self.cancelled_count.fetchAdd(1, .monotonic);
                    }
                    self.wakeWorkerLoop();
                    self.wakeNetworkLoop();
                }
            }
            rec.active_conn_slot = null;
        }
        const HEADER_RESERVE: usize = 768;
        if (slot.tx_buf.len <= HEADER_RESERVE) {
            self.closeSlotSocketAndRelease(slot, false);
            return;
        }
        var bw = FixedWriter.init(slot.tx_buf[HEADER_RESERVE..]);
        writeResponseObjectJson(&bw, rec) catch {
            self.sendErrorResponse(slot, 500, "Internal Server Error", null, "response exceeds tx buffer", "server_error", "tx_overflow");
            return;
        };
        self.writeHttpJsonOkHeadersAndSend(slot, bw.pos, HEADER_RESERVE, null, null);
    }

    fn appendTranscriptSlice(
        cw: *FixedWriter,
        span_rec: *chat_template.PromptSpanRecorder,
        bytes: []const u8,
        allow_special: bool,
    ) error{BufferTooSmall}!void {
        if (bytes.len == 0) return;
        try cw.writeAll(bytes);
        span_rec.record(bytes.len, allow_special) catch return error.BufferTooSmall;
    }

    fn appendTranscriptJsonEscaped(
        cw: *FixedWriter,
        span_rec: *chat_template.PromptSpanRecorder,
        bytes: []const u8,
    ) error{BufferTooSmall}!void {
        if (bytes.len == 0) return;
        const begin_pos = cw.pos;
        try cw.writeJsonEscaped(bytes);
        span_rec.record(cw.pos - begin_pos, false) catch return error.BufferTooSmall;
    }

    fn appendCompletedTranscriptTurn(
        tpl: *const chat_template.ChatTemplate,
        rec: *const StatefulResponseSlot,
        prompt_bytes: []const u8,
        prompt_spans: []const chat_template.PromptSpan,
        cw: *FixedWriter,
        span_rec: *chat_template.PromptSpanRecorder,
    ) error{BufferTooSmall}!void {
        try cw.writeAll(prompt_bytes);
        if (prompt_spans.len > 0) {
            for (prompt_spans) |sp| {
                span_rec.record(sp.len, sp.allow_special) catch return error.BufferTooSmall;
            }
        } else {
            span_rec.record(prompt_bytes.len, true) catch return error.BufferTooSmall;
        }
        const reasoning = rec.reasoning_buf[0..rec.reasoning_len];
        const content = rec.content_buf[0..rec.content_len];
        switch (tpl.dialect) {
            .inkling2 => {
                if (reasoning.len > 0) {
                    try appendTranscriptSlice(cw, span_rec, tpl.role_tokens.header_assistant.slice(), true);
                    try appendTranscriptSlice(cw, span_rec, tpl.thinking.open_tag.slice(), true);
                    try appendTranscriptSlice(cw, span_rec, reasoning, false);
                    try appendTranscriptSlice(cw, span_rec, tpl.role_tokens.end_message_tag.slice(), true);
                }
                if (content.len > 0) {
                    try appendTranscriptSlice(cw, span_rec, tpl.role_tokens.header_assistant.slice(), true);
                    try appendTranscriptSlice(cw, span_rec, tpl.role_tokens.content_text_tag.slice(), true);
                    try appendTranscriptSlice(cw, span_rec, content, false);
                    try appendTranscriptSlice(cw, span_rec, tpl.role_tokens.end_message_tag.slice(), true);
                }
                for (rec.tool_calls[0..rec.tool_call_count]) |*tc| {
                    try appendTranscriptSlice(cw, span_rec, tpl.role_tokens.header_assistant.slice(), true);
                    try appendTranscriptSlice(cw, span_rec, tc.nameSlice(), false);
                    try appendTranscriptSlice(cw, span_rec, tpl.tool_calling.call_open.slice(), true);
                    try appendTranscriptSlice(cw, span_rec, "{\"name\":\"", true);
                    try appendTranscriptJsonEscaped(cw, span_rec, tc.nameSlice());
                    try appendTranscriptSlice(cw, span_rec, "\",\"args\":", true);
                    try appendTranscriptSlice(cw, span_rec, if (tc.args_len > 0) tc.argsSlice() else "{}", false);
                    try appendTranscriptSlice(cw, span_rec, "}", true);
                    try appendTranscriptSlice(cw, span_rec, tpl.tool_calling.call_close.slice(), true);
                }
                try appendTranscriptSlice(cw, span_rec, tpl.eos_token.slice(), true);
            },
            .hy4 => {
                const open_tag = tpl.thinking.open_tag.slice();
                if (std.mem.endsWith(u8, prompt_bytes, open_tag)) {
                    try appendTranscriptSlice(cw, span_rec, reasoning, false);
                    try appendTranscriptSlice(cw, span_rec, tpl.thinking.close_tag.slice(), true);
                } else if (reasoning.len > 0) {
                    try appendTranscriptSlice(cw, span_rec, open_tag, true);
                    try appendTranscriptSlice(cw, span_rec, reasoning, false);
                    try appendTranscriptSlice(cw, span_rec, tpl.thinking.close_tag.slice(), true);
                }
                if (content.len > 0) try appendTranscriptSlice(cw, span_rec, content, false);
                if (rec.tool_call_count > 0) {
                    try appendTranscriptSlice(cw, span_rec, tpl.tool_calling.call_open.slice(), true);
                    for (rec.tool_calls[0..rec.tool_call_count]) |*tc| {
                        try appendTranscriptSlice(cw, span_rec, "<tool_call:opensource>", true);
                        try appendTranscriptSlice(cw, span_rec, tc.nameSlice(), false);
                        try appendTranscriptSlice(cw, span_rec, "<arg_key:opensource>args</arg_key:opensource><arg_value:opensource>", true);
                        try appendTranscriptSlice(cw, span_rec, if (tc.args_len > 0) tc.argsSlice() else "{}", false);
                        try appendTranscriptSlice(cw, span_rec, "</arg_value:opensource></tool_call:opensource>", true);
                    }
                    try appendTranscriptSlice(cw, span_rec, tpl.tool_calling.call_close.slice(), true);
                }
                try appendTranscriptSlice(cw, span_rec, tpl.role_tokens.hy4_end.slice(), true);
            },
            .chatml, .deepseek, .custom => {
                if (content.len > 0) try appendTranscriptSlice(cw, span_rec, content, false);
                for (rec.tool_calls[0..rec.tool_call_count]) |*tc| {
                    try appendTranscriptSlice(cw, span_rec, tpl.tool_calling.call_open.slice(), true);
                    try appendTranscriptSlice(cw, span_rec, "{\"name\":\"", true);
                    try appendTranscriptJsonEscaped(cw, span_rec, tc.nameSlice());
                    try appendTranscriptSlice(cw, span_rec, "\",\"arguments\":", true);
                    try appendTranscriptSlice(cw, span_rec, if (tc.args_len > 0) tc.argsSlice() else "{}", false);
                    try appendTranscriptSlice(cw, span_rec, "}", true);
                    try appendTranscriptSlice(cw, span_rec, tpl.tool_calling.call_close.slice(), true);
                }
                try appendTranscriptSlice(cw, span_rec, tpl.assistantEndDelimiter(), true);
            },
        }
    }

    fn finalizeStatefulResponseRecord(self: *Server, slot: *ConnectionSlot, status: ResponseRecordStatus, inc_reason: IncompleteReason) void {
        const r_idx = slot.stateful_resp_slot_idx orelse return;
        if (r_idx >= self.response_store.len) return;
        const rec = &self.response_store[r_idx];
        if (!rec.occupied or rec.request_id != slot.request_id) return;

        const was_truncated = slot.nonstream_reasoning_truncated or
            slot.nonstream_content_truncated or
            slot.tool_calls_truncated or
            (slot.nonstream_reasoning_len > rec.reasoning_buf.len) or
            (slot.nonstream_content_len > rec.content_buf.len);

        var final_status = status;
        var final_inc_reason = inc_reason;
        if (was_truncated and final_status == .completed) {
            final_status = .incomplete;
            if (final_inc_reason == .none) final_inc_reason = .max_response_bytes;
        }

        rec.status = final_status;
        rec.incomplete_reason = final_inc_reason;
        rec.reasoning_summary = slot.reasoning_summary;
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

        // Refuse to persist truncated or incomplete transcripts into `rec.chatml_buf` (`W24`/`C18`).
        if (was_truncated or final_status != .completed) {
            rec.transcript_corrupt = true;
            rec.chatml_len = 0;
            return;
        }

        var cw = FixedWriter.init(rec.chatml_buf);
        var c_spans = chat_template.PromptSpanRecorder.init(&rec.chatml_spans);
        const slot_spans = slot.prompt_spans_buf[0..slot.prompt_span_count];
        appendCompletedTranscriptTurn(&self.config.chat_template, rec, slot.prompt_buf[0..slot.prompt_len], slot_spans, &cw, &c_spans) catch {
            rec.status = .incomplete;
            rec.incomplete_reason = .max_response_bytes;
            rec.transcript_corrupt = true;
            rec.chatml_len = 0;
            rec.chatml_span_count = 0;
            return;
        };
        rec.transcript_corrupt = false;
        rec.chatml_len = cw.pos;
        rec.chatml_span_count = @intCast(c_spans.count);
    }

    const TempInputItem = struct {
        role: InputItemRole,
        text: []const u8,
    };

    fn appendContentTextToScratch(content_raw: []const u8, scratch: []u8, scratch_pos: *usize) []const u8 {
        var r = JsonReader{ .src = content_raw };
        const ch = r.peek() orelse return "";
        const text_start = scratch_pos.*;
        if (ch == '"') {
            var fw = FixedWriter.init(scratch[text_start..]);
            r.readStringInto(&fw) catch return "";
            scratch_pos.* += fw.pos;
            return fw.written();
        }
        if (ch != '[') return "";
        r.pos += 1;
        if (r.peek() == ']') return "";
        var fw = FixedWriter.init(scratch[text_start..]);
        while (true) {
            r.expectByte('{') catch break;
            var btype_buf: [32]u8 = undefined;
            var btype: []const u8 = "";
            var text_raw: ?[]const u8 = null;
            if (r.peek() != '}') {
                while (true) {
                    var kb: [64]u8 = undefined;
                    const mk = r.readKeyOrSkip(&kb) catch break;
                    r.expectByte(':') catch break;
                    if (mk) |key| {
                        if (std.mem.eql(u8, key, "type")) {
                            btype = r.readSmallString(&btype_buf) catch "";
                        } else if (std.mem.eql(u8, key, "text")) {
                            text_raw = r.captureValueSlice(4) catch null;
                        } else {
                            r.skipValue(4) catch break;
                        }
                    } else {
                        r.skipValue(4) catch break;
                    }
                    const sep = r.peek() orelse break;
                    if (sep == ',') {
                        r.pos += 1;
                    } else if (sep == '}') {
                        r.pos += 1;
                        break;
                    } else break;
                }
            } else r.pos += 1;
            if (btype.len == 0 or std.mem.eql(u8, btype, "text") or std.mem.eql(u8, btype, "input_text") or std.mem.eql(u8, btype, "output_text")) {
                if (text_raw) |tr| {
                    var sr = JsonReader{ .src = tr };
                    if (sr.peek() == '"') {
                        sr.readStringInto(&fw) catch break;
                    }
                }
            }
            const sep2 = r.peek() orelse break;
            if (sep2 == ',') {
                r.pos += 1;
            } else break;
        }
        scratch_pos.* += fw.pos;
        return fw.written();
    }

    fn collectResponsesInputItemsFromBody(
        body: []const u8,
        scratch: []u8,
        out_items: *[MAX_STATEFUL_INPUT_ITEMS]TempInputItem,
    ) usize {
        var r = JsonReader{ .src = body };
        r.expectByte('{') catch return 0;
        var instr_raw: ?[]const u8 = null;
        var input_raw: ?[]const u8 = null;
        if (r.peek() != '}') {
            while (true) {
                var kb: [64]u8 = undefined;
                const mk = r.readKeyOrSkip(&kb) catch return 0;
                r.expectByte(':') catch return 0;
                if (mk) |key| {
                    if (std.mem.eql(u8, key, "instructions") or std.mem.eql(u8, key, "system")) {
                        if (!r.consumeKeyword("null")) {
                            instr_raw = r.captureValueSlice(2) catch null;
                        }
                    } else if (std.mem.eql(u8, key, "input")) {
                        input_raw = r.captureValueSlice(2) catch null;
                    } else {
                        r.skipValue(2) catch return 0;
                    }
                } else {
                    r.skipValue(2) catch return 0;
                }
                const sep = r.peek() orelse return 0;
                if (sep == ',') {
                    r.pos += 1;
                } else if (sep == '}') {
                    break;
                } else return 0;
            }
        }

        var count: usize = 0;
        var scratch_pos: usize = 0;
        if (instr_raw) |ir| {
            const txt = appendContentTextToScratch(ir, scratch, &scratch_pos);
            if (txt.len > 0 and count < MAX_STATEFUL_INPUT_ITEMS) {
                out_items[count] = .{ .role = .system, .text = txt };
                count += 1;
            }
        }
        const inp = input_raw orelse return count;
        var ir = JsonReader{ .src = inp };
        const ch = ir.peek() orelse return count;
        if (ch == '"') {
            if (count < MAX_STATEFUL_INPUT_ITEMS) {
                const txt = appendContentTextToScratch(inp[ir.pos..], scratch, &scratch_pos);
                out_items[count] = .{ .role = .user, .text = txt };
                count += 1;
            }
            return count;
        }
        if (ch != '[') return count;
        ir.pos += 1;
        if (ir.peek() == ']') return count;
        while (count < MAX_STATEFUL_INPUT_ITEMS) {
            ir.expectByte('{') catch break;
            var role_buf: [32]u8 = undefined;
            var role_str: []const u8 = "";
            var itype_buf: [32]u8 = undefined;
            var itype: []const u8 = "";
            var content_raw: ?[]const u8 = null;
            var output_raw: ?[]const u8 = null;
            if (ir.peek() != '}') {
                while (true) {
                    var kb: [64]u8 = undefined;
                    const mk = ir.readKeyOrSkip(&kb) catch break;
                    ir.expectByte(':') catch break;
                    if (mk) |key| {
                        if (std.mem.eql(u8, key, "role")) {
                            role_str = ir.readSmallString(&role_buf) catch "";
                        } else if (std.mem.eql(u8, key, "type")) {
                            itype = ir.readSmallString(&itype_buf) catch "";
                        } else if (std.mem.eql(u8, key, "content")) {
                            content_raw = ir.captureValueSlice(3) catch null;
                        } else if (std.mem.eql(u8, key, "output")) {
                            output_raw = ir.captureValueSlice(3) catch null;
                        } else {
                            ir.skipValue(3) catch break;
                        }
                    } else {
                        ir.skipValue(3) catch break;
                    }
                    const sep = ir.peek() orelse break;
                    if (sep == ',') {
                        ir.pos += 1;
                    } else if (sep == '}') {
                        ir.pos += 1;
                        break;
                    } else break;
                }
            } else ir.pos += 1;

            if (std.mem.eql(u8, itype, "function_call_output")) {
                if (output_raw) |oraw| {
                    const txt = appendContentTextToScratch(oraw, scratch, &scratch_pos);
                    out_items[count] = .{ .role = .tool, .text = txt };
                    count += 1;
                }
            } else if (!std.mem.eql(u8, itype, "function_call")) {
                const mapped_role: InputItemRole = if (std.mem.eql(u8, role_str, "system"))
                    .system
                else if (std.mem.eql(u8, role_str, "developer"))
                    .developer
                else if (std.mem.eql(u8, role_str, "assistant"))
                    .assistant
                else if (std.mem.eql(u8, role_str, "tool"))
                    .tool
                else
                    .user;
                const txt = if (content_raw) |cr| appendContentTextToScratch(cr, scratch, &scratch_pos) else "";
                out_items[count] = .{ .role = mapped_role, .text = txt };
                count += 1;
            }
            const sep2 = ir.peek() orelse break;
            if (sep2 == ',') {
                ir.pos += 1;
            } else break;
        }
        return count;
    }

    fn populateStatefulResponseInputItems(
        rec: *StatefulResponseSlot,
        prev_rec_opt: ?*const StatefulResponseSlot,
        new_items: []const TempInputItem,
    ) void {
        rec.input_item_count = 0;
        rec.input_items_len = 0;

        const prev_count: usize = if (prev_rec_opt) |pr| if (pr != rec) pr.input_item_count else 0 else 0;
        const cap_new = @min(new_items.len, MAX_STATEFUL_INPUT_ITEMS);
        const keep_prev = @min(prev_count, MAX_STATEFUL_INPUT_ITEMS - cap_new);
        const skip_prev = prev_count - keep_prev;

        var buf_off: usize = 0;
        if (prev_rec_opt) |pr| {
            if (pr != rec) {
                var pi: usize = skip_prev;
                while (pi < prev_count) : (pi += 1) {
                    const src_item = &pr.input_items[pi];
                    const avail = rec.input_items_buf.len -| buf_off;
                    const take = @min(avail, @as(usize, src_item.len));
                    const s_off: usize = src_item.offset;
                    if (take > 0 and s_off + take <= pr.input_items_buf.len) {
                        @memcpy(rec.input_items_buf[buf_off .. buf_off + take], pr.input_items_buf[s_off .. s_off + take]);
                    }
                    rec.input_items[rec.input_item_count] = .{
                        .origin_request_id = src_item.origin_request_id,
                        .origin_item_idx = src_item.origin_item_idx,
                        .offset = @intCast(buf_off),
                        .len = @intCast(take),
                        .role = src_item.role,
                    };
                    rec.input_item_count += 1;
                    buf_off += take;
                }
            }
        }

        for (new_items[0..cap_new], 0..) |item, idx| {
            const avail = rec.input_items_buf.len -| buf_off;
            const take = @min(avail, item.text.len);
            if (take > 0) {
                @memcpy(rec.input_items_buf[buf_off .. buf_off + take], item.text[0..take]);
            }
            rec.input_items[rec.input_item_count] = .{
                .origin_request_id = rec.request_id,
                .origin_item_idx = @intCast(idx),
                .offset = @intCast(buf_off),
                .len = @intCast(take),
                .role = item.role,
            };
            rec.input_item_count += 1;
            buf_off += take;
        }
        rec.input_items_len = buf_off;
    }

    fn admitInferenceRequest(self: *Server, slot: *ConnectionSlot, endpoint: Endpoint, body: []const u8) void {
        // Resolve optional `previous_response_id` on `/v1/responses` before building ChatML prompt
        // so the full chained transcript is prepended directly into `slot.prompt_buf` (`0` heap alloc).
        var prior_chatml: ?[]const u8 = null;
        var prior_spans: ?[]const chat_template.PromptSpan = null;
        var prev_rec_ptr: ?*const StatefulResponseSlot = null;
        if (endpoint == .responses) {
            var prev_probe_buf: [MAX_RESPONSE_ID_BYTES]u8 = undefined;
            if (findTopLevelJsonStringField(body, "previous_response_id", &prev_probe_buf)) |prev_id| {
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
                    if (prev_rec.transcript_corrupt or prev_rec.status != .completed or prev_rec.chatml_len == 0) {
                        self.sendErrorResponse(
                            slot,
                            409,
                            "Conflict",
                            null,
                            "previous_response_id transcript is incomplete or truncated",
                            "invalid_request_error",
                            "incomplete_previous_response",
                        );
                        return;
                    }
                    prior_chatml = prev_rec.chatml_buf[0..prev_rec.chatml_len];
                    if (prev_rec.chatml_span_count > 0) {
                        prior_spans = prev_rec.chatmlSpansSlice();
                    }
                    prev_rec_ptr = prev_rec;
                }
            }
        }

        const parsed = parseInferenceRequestFullWithPriorSpans(
            endpoint,
            body,
            self.modelId(),
            &self.config.chat_template,
            prior_chatml,
            prior_spans,
            slot.nonstream_content_buf,
            &self.signature_key,
            self.fim_tokens.asFimTokens(),
            self.tokenizer(),
            slot.media_buf,
            slot.prompt_buf,
            &slot.model_buf,
        ) catch |err| {
            self.sendParseErrorResponse(slot, err);
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
        slot.prompt_token_ids_len = parsed.prompt_token_ids_len;
        if (parsed.prompt_token_ids_len > 0) {
            @memcpy(
                slot.prompt_token_ids_buf[0..parsed.prompt_token_ids_len],
                parsed.prompt_token_ids_buf[0..parsed.prompt_token_ids_len],
            );
        }
        slot.prompt_span_count = parsed.prompt_span_count;
        if (parsed.prompt_span_count > 0) {
            @memcpy(
                slot.prompt_spans_buf[0..parsed.prompt_span_count],
                parsed.prompt_spans_buf[0..parsed.prompt_span_count],
            );
        }
        slot.legacy_logprobs = parsed.legacy_logprobs;
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
        slot.prefill_done_ns = 0;
        slot.first_generated_token_ns = 0;
        slot.first_emitted_ns = 0;
        slot.first_token_ns = 0;
        slot.created_unix = unixTimestampSec();
        slot.prompt_tokens = 0;
        slot.cached_tokens = 0;
        slot.completion_tokens = 0;
        slot.reasoning_tokens = 0;
        slot.nonstream_reasoning_truncated = false;
        slot.nonstream_content_truncated = false;
        slot.tool_calls_truncated = false;
        slot.thinking_budget_tokens = parsed.thinking_budget_tokens;
        slot.thinking_budget_exhausted = false;
        slot.reasoning_summary = parsed.reasoning_summary;
        slot.template_dialect = parsed.template_dialect;
        slot.thinking_hmac = initThinkingHmac(&self.signature_key, slot.modelSlice());
        slot.anthropic_block_had_delta = false;
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
        slot.think_splitter.resetForDialect(parsed.template_dialect, parsed.starts_in_think);
        // Downstream `ToolCallParser` consumes Inkling 2 `<|message_model|>` recipient headers
        // and `<|end_message|>` closers on the `.content` channel and strips envelope tags itself.
        slot.think_splitter.strip_envelope_tags = false;
        slot.tool_parser.resetForDialect(slot.request_id, parsed.template_dialect);
        slot.stored_tool_call_count = 0;
        slot.cancelled.store(false, .release);
        while (slot.token_ring.tryPop() != null) {}

        // Reserve stateful `/v1/responses` slot (`in_progress`) so GET, input_items & POST /cancel work live!
        // Never evict active `.in_progress` slots (`503 server_overloaded` when all slots active).
        if (endpoint == .responses and parsed.store_response) {
            var new_items: [MAX_STATEFUL_INPUT_ITEMS]TempInputItem = undefined;
            const new_item_count = collectResponsesInputItemsFromBody(body, slot.nonstream_content_buf, &new_items);

            const r_idx = self.allocStatefulResponseSlot() orelse {
                _ = self.rejected_503_count.fetchAdd(1, .monotonic);
                self.sendErrorResponse(
                    slot,
                    503,
                    "Service Unavailable",
                    null,
                    "server busy: all stateful response slots in progress",
                    "server_error",
                    "server_overloaded",
                );
                return;
            };
            const rec = &self.response_store[r_idx];
            rec.occupied = true;
            rec.lru_tick = self.response_lru_clock;
            self.response_lru_clock +%= 1;
            rec.request_id = slot.request_id;
            rec.created_unix = slot.created_unix;
            rec.status = .in_progress;
            rec.incomplete_reason = .none;
            rec.reasoning_summary = parsed.reasoning_summary;
            rec.transcript_corrupt = false;
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
            populateStatefulResponseInputItems(rec, prev_rec_ptr, new_items[0..new_item_count]);
            slot.stateful_resp_slot_idx = r_idx;
        }

        if (parsed.stream) {
            slot.keep_alive = false;
        }

        // For SSE streams, send HTTP 200 headers (with X-Aminfer-Seed-Used + CORS) immediately.
        if (parsed.stream) {
            var hdr_buf: [1024]u8 = undefined;
            var w = FixedWriter.init(&hdr_buf);
            w.print(
                "HTTP/1.1 200 OK\r\n" ++
                    "Content-Type: text/event-stream\r\n" ++
                    "Cache-Control: no-cache\r\n" ++
                    "Connection: close\r\n" ++
                    "X-Aminfer-Seed-Used: {d}\r\n" ++
                    "X-Aminfer-Seed-Source: {s}\r\n",
                .{ slot.seed_used, slot.seed_source.name() },
            ) catch {
                self.rollbackStatefulResponseSlot(slot);
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            self.appendCorsResponseHeaders(&w, false) catch {
                self.rollbackStatefulResponseSlot(slot);
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            w.writeAll("\r\n") catch {
                self.rollbackStatefulResponseSlot(slot);
                self.closeSlotSocketAndRelease(slot, false);
                return;
            };
            slot.appendTx(w.written()) catch {
                self.rollbackStatefulResponseSlot(slot);
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
            .template_dialect = parsed.template_dialect,
            .starts_in_think = parsed.starts_in_think,
            .stream_include_usage = parsed.stream_include_usage,
            .prompt_token_ids = if (slot.prompt_token_ids_len > 0)
                slot.prompt_token_ids_buf[0..slot.prompt_token_ids_len]
            else
                null,
            .prompt_spans = if (slot.prompt_span_count > 0)
                slot.prompt_spans_buf[0..slot.prompt_span_count]
            else
                null,
            .media_items = parsed.media_items,
            .media_count = parsed.media_count,
        };

        if (!self.job_ring.tryPush(job)) {
            self.rollbackStatefulResponseSlot(slot);
            slot.worker_active.store(false, .release);
            self.releaseRequestCount(slot);
            _ = self.rejected_503_count.fetchAdd(1, .monotonic);
            slot.tx_read_pos = 0;
            slot.tx_write_pos = 0;
            slot.keep_alive = false;
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

    fn rollbackStatefulResponseSlot(self: *Server, slot: *ConnectionSlot) void {
        if (slot.stateful_resp_slot_idx) |r_idx| {
            if (r_idx < self.response_store.len) {
                const rec = &self.response_store[r_idx];
                rec.occupied = false;
                rec.active_conn_slot = null;
                rec.id_len = 0;
                rec.chatml_len = 0;
            }
            slot.stateful_resp_slot_idx = null;
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
                        self.finishHttpExchange(slot);
                    }
                },
                .draining_worker => {
                    if (!slot.worker_active.load(.acquire)) {
                        self.completeKeepAliveRecycle(slot);
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
                // If worker exited unexpectedly (or stopped after API cancel) without pushing `.finished`.
                if (!slot.worker_active.load(.acquire) and slot.token_ring.isEmpty()) {
                    if (slot.cancelled.load(.acquire)) {
                        if (slot.api_cancelled and slot.fd >= 0) {
                            self.finishSlotResponse(slot, .cancelled);
                        } else {
                            self.closeSlotSocketAndRelease(slot, false);
                        }
                    } else if (slot.aux_kind != .generate) {
                        self.releaseRequestCount(slot);
                        self.sendAuxErrorResponse(slot, if (slot.aux_err != .none) slot.aux_err else .embed_failed);
                    } else {
                        self.finishSlotResponse(slot, .stop);
                    }
                }
                break;
            };

            if (ev.generation != slot.generation) continue;

            if (slot.aux_kind != .generate) {
                switch (ev.kind) {
                    .finished => {
                        self.releaseRequestCount(slot);
                        const HEADER_RESERVE: usize = 768;
                        if (slot.tx_buf.len <= HEADER_RESERVE) {
                            self.closeSlotSocketAndRelease(slot, false);
                            return;
                        }
                        if (slot.aux_err != .none) {
                            self.sendAuxErrorResponse(slot, slot.aux_err);
                            return;
                        }
                        self.writeHttpJsonOkHeadersAndSend(slot, slot.aux_response_len, HEADER_RESERVE, null, null);
                        return;
                    },
                    .failed => {
                        self.releaseRequestCount(slot);
                        self.sendAuxErrorResponse(slot, if (slot.aux_err != .none) slot.aux_err else .embed_failed);
                        return;
                    },
                    .prefill_done, .token_text => continue,
                }
            }

            switch (ev.kind) {
                .prefill_done => {
                    slot.prompt_tokens = ev.token_count;
                    slot.cached_tokens = ev.cached_tokens;
                    if (slot.prefill_done_ns == 0) {
                        slot.prefill_done_ns = if (ev.timestamp_ns != 0) ev.timestamp_ns else monotonicNowNs();
                    }
                    if (slot.stream) {
                        self.ensureStreamPreamble(slot);
                    }
                },
                .token_text => {
                    const text = ev.textSlice();
                    if (ev.token_count > 0) {
                        slot.completion_tokens +|= ev.token_count;
                        if (text.len > 0 and slot.first_generated_token_ns == 0) {
                            slot.first_generated_token_ns = if (ev.timestamp_ns != 0) ev.timestamp_ns else monotonicNowNs();
                        }
                    }
                    if (slot.stream) {
                        self.ensureStreamPreamble(slot);
                    }
                    if (text.len > 0) {
                        self.feedSlotText(slot, text, ev.token_count);
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
                    const final_reason: FinishReason = if (slot.api_cancelled)
                        .cancelled
                    else if (slot.stop_holdback.stopped)
                        .stop_sequence
                    else
                        ev.finish_reason;
                    self.finishSlotResponse(slot, final_reason);
                    return;
                },
                .failed => {
                    if (ev.token_count > 0) slot.prompt_tokens = ev.token_count;
                    self.finalizeStatefulResponseRecord(slot, .failed, .none);
                    self.failSlotResponse(slot, ev.textSlice());
                    return;
                },
            }

            if (slot.stream and slot.pendingTxLen() > 0) {
                if (!self.flushTx(slot)) return;
            }
        }
    }

    fn feedSlotText(self: *Server, slot: *ConnectionSlot, text: []const u8, token_count: u32) void {
        if (slot.endpoint == .completions) {
            const RawHoldCtx = struct {
                server: *Server,
                slot: *ConnectionSlot,
                fn onSafe(ctx: *@This(), safe_bytes: []const u8) error{BufferTooSmall}!void {
                    if (safe_bytes.len == 0) return;
                    if (ctx.slot.first_emitted_ns == 0) {
                        const now = monotonicNowNs();
                        ctx.slot.first_emitted_ns = now;
                        if (ctx.slot.first_token_ns == 0) ctx.slot.first_token_ns = now;
                    }
                    ctx.server.appendNonStreamSpan(ctx.slot, .content, safe_bytes);
                    if (ctx.slot.stream) {
                        try ctx.server.appendStreamDelta(ctx.slot, .content, safe_bytes);
                    }
                }
            };
            var rctx = RawHoldCtx{ .server = self, .slot = slot };
            _ = slot.stop_holdback.feed(&slot.stop_config, text, &rctx, RawHoldCtx.onSafe) catch {
                self.closeSlotSocketAndRelease(slot, true);
                return;
            };
            return;
        }
        const ThinkCtx = struct {
            server: *Server,
            slot: *ConnectionSlot,
            reasoning_bytes: usize = 0,
            content_bytes: usize = 0,

            fn onSpan(tctx: *@This(), channel: StreamChannel, span: []const u8) error{BufferTooSmall}!void {
                if (span.len == 0) return;
                if (channel == .reasoning) {
                    if (tctx.slot.thinking_budget_exhausted) return;
                    tctx.reasoning_bytes += span.len;
                    try tctx.server.dispatchChannelSpan(tctx.slot, .reasoning, span);
                } else {
                    tctx.content_bytes += span.len;
                    try tctx.server.dispatchChannelSpan(tctx.slot, .content, span);
                }
            }
        };
        const HoldCtx = struct {
            tctx: *ThinkCtx,
            fn onSafe(ctx: *@This(), safe_bytes: []const u8) error{BufferTooSmall}!void {
                if (safe_bytes.len == 0) return;
                try ctx.tctx.slot.think_splitter.feed(safe_bytes, ctx.tctx, ThinkCtx.onSpan);
            }
        };
        var tctx = ThinkCtx{ .server = self, .slot = slot };
        var hctx = HoldCtx{ .tctx = &tctx };
        _ = slot.stop_holdback.feed(&slot.stop_config, text, &hctx, HoldCtx.onSafe) catch {
            self.closeSlotSocketAndRelease(slot, true);
            return;
        };

        // Post-`ThinkSplitter` channel-attributed `reasoning_tokens` (`W24`/`C18`).
        if (tctx.reasoning_bytes > 0 and token_count > 0) {
            if (tctx.content_bytes == 0 or token_count == 1) {
                slot.reasoning_tokens +|= token_count;
            } else {
                const total_b = tctx.reasoning_bytes + tctx.content_bytes;
                const numer = @as(u64, token_count) * tctx.reasoning_bytes + total_b / 2;
                const r_toks: u32 = @max(@as(u32, 1), @min(token_count, @as(u32, @intCast(numer / total_b))));
                slot.reasoning_tokens +|= r_toks;
            }
        }

        if (!slot.thinking_budget_exhausted) {
            if (slot.thinking_budget_tokens) |budget| {
                if (slot.reasoning_tokens >= budget and slot.think_splitter.in_think) {
                    slot.thinking_budget_exhausted = true;
                    self.forceCloseThinkingAfterBudget(slot) catch {
                        self.closeSlotSocketAndRelease(slot, true);
                    };
                }
            }
        }
    }

    fn forceCloseThinkingAfterBudget(self: *Server, slot: *ConnectionSlot) error{BufferTooSmall}!void {
        if (!slot.stream) return;
        switch (slot.endpoint) {
            .messages => {
                if (slot.anthropic_block == .thinking) {
                    var buf: [512]u8 = undefined;
                    var w = FixedWriter.init(&buf);
                    try self.writeAnthropicBlockStop(slot, &w);
                    slot.anthropic_block_idx += 1;
                    slot.anthropic_block = .none;
                    try slot.appendTx(w.written());
                }
            },
            .responses => {
                if (slot.responses_reasoning_opened and !slot.responses_reasoning_closed) {
                    try self.flushAndCloseResponsesOpenItems(slot);
                }
            },
            .chat_completions, .completions => {},
        }
    }

    fn dispatchChannelSpan(
        self: *Server,
        slot: *ConnectionSlot,
        channel: StreamChannel,
        span: []const u8,
    ) error{BufferTooSmall}!void {
        if (channel == .reasoning) {
            if (span.len > 0 and slot.first_emitted_ns == 0) {
                const now = monotonicNowNs();
                slot.first_emitted_ns = now;
                if (slot.first_token_ns == 0) slot.first_token_ns = now;
            }
            slot.thinking_hmac.update(span);
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
                if (slot.first_emitted_ns == 0) {
                    const now = monotonicNowNs();
                    slot.first_emitted_ns = now;
                    if (slot.first_token_ns == 0) slot.first_token_ns = now;
                }
                self.appendNonStreamSpan(slot, .content, txt);
                if (slot.stream) {
                    try self.appendStreamDelta(slot, .content, txt);
                }
            },
            .tool_start => |st| {
                if (slot.first_emitted_ns == 0) {
                    const now = monotonicNowNs();
                    slot.first_emitted_ns = now;
                    if (slot.first_token_ns == 0) slot.first_token_ns = now;
                }
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
                } else {
                    slot.tool_calls_truncated = true;
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
                    if (take < ad.delta.len) slot.tool_calls_truncated = true;
                } else {
                    slot.tool_calls_truncated = true;
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
                try self.flushAndCloseResponsesOpenItems(slot);
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
            .completions => return,
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
            .completions => return,
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
            .chat_completions, .completions => return,
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

    fn writeAnthropicBlockStop(self: *Server, slot: *ConnectionSlot, w: *FixedWriter) error{BufferTooSmall}!void {
        if (slot.anthropic_block == .thinking) {
            // Emit cryptographic HMAC-SHA256 `signature_delta` before closing a `thinking` block (`C18`/`W24`).
            var mac = slot.thinking_hmac;
            var tag: [32]u8 = undefined;
            mac.final(&tag);
            const sig_hex = std.fmt.bytesToHex(tag, .lower);
            try w.print(
                "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{d},\"delta\":{{\"type\":\"signature_delta\",\"signature\":\"{s}\"}}}}\n\n",
                .{ slot.anthropic_block_idx, sig_hex[0..] },
            );
            slot.thinking_hmac = initThinkingHmac(&self.signature_key, slot.modelSlice());
        }
        try w.print(
            "event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{d}}}\n\n",
            .{slot.anthropic_block_idx},
        );
    }

    fn flushAndCloseResponsesOpenItems(self: *Server, slot: *ConnectionSlot) error{BufferTooSmall}!void {
        if ((!slot.responses_reasoning_opened or slot.responses_reasoning_closed) and
            (!slot.responses_text_opened or slot.responses_text_closed))
        {
            return;
        }
        _ = self.flushTx(slot);
        slot.compactTx();
        var fw = FixedWriter.init(slot.tx_buf[slot.tx_write_pos..]);
        try self.closeResponsesOpenItems(slot, &fw);
        slot.tx_write_pos += fw.pos;
    }

    fn closeResponsesOpenItems(_: *Server, slot: *ConnectionSlot, w: *FixedWriter) error{BufferTooSmall}!void {
        if (slot.responses_reasoning_opened and !slot.responses_reasoning_closed) {
            slot.responses_reasoning_closed = true;
            const reasoning = slot.nonstream_reasoning_buf[0..slot.nonstream_reasoning_len];

            const s_done = slot.responses_seq_num;
            slot.responses_seq_num += 1;
            try w.print(
                "event: response.reasoning_text.done\ndata: {{\"type\":\"response.reasoning_text.done\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"content_index\":0,\"text\":\"",
                .{ s_done, slot.request_id, slot.responses_reasoning_output_idx },
            );
            try w.writeJsonEscaped(reasoning);
            try w.writeAll("\"}\n\n");

            const s_part = slot.responses_seq_num;
            slot.responses_seq_num += 1;
            try w.print(
                "event: response.content_part.done\ndata: {{\"type\":\"response.content_part.done\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"content_index\":0,\"part\":{{\"type\":\"reasoning_text\",\"text\":\"",
                .{ s_part, slot.request_id, slot.responses_reasoning_output_idx },
            );
            try w.writeJsonEscaped(reasoning);
            try w.writeAll("\"}}\n\n");

            var concise_buf: [320]u8 = undefined;
            const summary_text: []const u8 = switch (slot.reasoning_summary) {
                .none => "",
                .auto, .detailed => reasoning,
                .concise => extractConciseReasoningSummary(reasoning, &concise_buf),
            };
            if (slot.reasoning_summary == .concise and summary_text.len > 0) {
                const s_pa = slot.responses_seq_num;
                slot.responses_seq_num += 1;
                try w.print(
                    "event: response.reasoning_summary_part.added\ndata: {{\"type\":\"response.reasoning_summary_part.added\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"summary_index\":0,\"part\":{{\"type\":\"summary_text\",\"text\":\"\"}}}}\n\n",
                    .{ s_pa, slot.request_id, slot.responses_reasoning_output_idx },
                );
                const s_dl = slot.responses_seq_num;
                slot.responses_seq_num += 1;
                try w.print(
                    "event: response.reasoning_summary_text.delta\ndata: {{\"type\":\"response.reasoning_summary_text.delta\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"summary_index\":0,\"delta\":\"",
                    .{ s_dl, slot.request_id, slot.responses_reasoning_output_idx },
                );
                try w.writeJsonEscaped(summary_text);
                try w.writeAll("\"}\n\n");
            }
            if (slot.reasoning_summary != .none) {
                const s1 = slot.responses_seq_num;
                slot.responses_seq_num += 1;
                try w.print(
                    "event: response.reasoning_summary_text.done\ndata: {{\"type\":\"response.reasoning_summary_text.done\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"summary_index\":0,\"text\":\"",
                    .{ s1, slot.request_id, slot.responses_reasoning_output_idx },
                );
                try w.writeJsonEscaped(summary_text);
                try w.writeAll("\"}\n\n");

                const s_spd = slot.responses_seq_num;
                slot.responses_seq_num += 1;
                try w.print(
                    "event: response.reasoning_summary_part.done\ndata: {{\"type\":\"response.reasoning_summary_part.done\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"summary_index\":0,\"part\":{{\"type\":\"summary_text\",\"text\":\"",
                    .{ s_spd, slot.request_id, slot.responses_reasoning_output_idx },
                );
                try w.writeJsonEscaped(summary_text);
                try w.writeAll("\"}}\n\n");
            }

            const s2 = slot.responses_seq_num;
            slot.responses_seq_num += 1;
            try w.print(
                "event: response.output_item.done\ndata: {{\"type\":\"response.output_item.done\",\"sequence_number\":{d},\"output_index\":{d},\"item\":{{\"id\":\"rs_{d}\",\"type\":\"reasoning\",\"status\":\"completed\",\"content\":[{{\"type\":\"reasoning_text\",\"text\":\"",
                .{ s2, slot.responses_reasoning_output_idx, slot.request_id },
            );
            try w.writeJsonEscaped(reasoning);
            try w.writeAll("\"}],\"summary\":[");
            if (slot.reasoning_summary != .none and (slot.reasoning_summary != .concise or summary_text.len > 0)) {
                try w.writeAll("{\"type\":\"summary_text\",\"text\":\"");
                try w.writeJsonEscaped(summary_text);
                try w.writeAll("\"}");
            }
            try w.writeAll("]}}\n\n");
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

    fn appendNonStreamSpan(self: *Server, slot: *ConnectionSlot, channel: StreamChannel, span: []const u8) void {
        const max_total: usize = self.config.max_response_bytes orelse std.math.maxInt(usize);
        const total_used = slot.nonstream_reasoning_len + slot.nonstream_content_len;
        const total_avail = max_total -| total_used;
        switch (channel) {
            .reasoning => {
                const buf_avail = slot.nonstream_reasoning_buf.len - slot.nonstream_reasoning_len;
                const avail = @min(buf_avail, total_avail);
                const n = @min(avail, span.len);
                if (n > 0) {
                    @memcpy(slot.nonstream_reasoning_buf[slot.nonstream_reasoning_len .. slot.nonstream_reasoning_len + n], span[0..n]);
                    slot.nonstream_reasoning_len += n;
                }
                if (n < span.len) slot.nonstream_reasoning_truncated = true;
            },
            .content => {
                const buf_avail = slot.nonstream_content_buf.len - slot.nonstream_content_len;
                const avail = @min(buf_avail, total_avail);
                const n = @min(avail, span.len);
                if (n > 0) {
                    @memcpy(slot.nonstream_content_buf[slot.nonstream_content_len .. slot.nonstream_content_len + n], span[0..n]);
                    slot.nonstream_content_len += n;
                }
                if (n < span.len) slot.nonstream_content_truncated = true;
            },
        }
    }

    fn ensureStreamPreamble(self: *Server, slot: *ConnectionSlot) void {
        if (slot.preamble_sent) return;
        slot.preamble_sent = true;

        var buf: [1536]u8 = undefined;
        var w = FixedWriter.init(&buf);
        switch (slot.endpoint) {
            .chat_completions, .completions => return,
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
            .completions => {
                try w.print(
                    "data: {{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":\"",
                    .{ slot.request_id, slot.created_unix },
                );
                try w.writeJsonEscaped(slot.modelSlice());
                try w.print(
                    "\",\"seed_used\":{d},\"seed_source\":\"{s}\",\"choices\":[{{\"text\":\"",
                    .{ slot.seed_used, slot.seed_source.name() },
                );
                try w.writeJsonEscaped(span);
                try w.writeAll("\",\"index\":0,\"logprobs\":null,\"finish_reason\":null}]}\n\n");
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
                                "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"sequence_number\":{d},\"output_index\":{d},\"item\":{{\"id\":\"rs_{d}\",\"type\":\"reasoning\",\"status\":\"in_progress\",\"content\":[],\"summary\":[]}}}}\n\n",
                                .{ s0, slot.responses_reasoning_output_idx, slot.request_id },
                            );
                            const s1 = slot.responses_seq_num;
                            slot.responses_seq_num += 1;
                            try w.print(
                                "event: response.content_part.added\ndata: {{\"type\":\"response.content_part.added\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"content_index\":0,\"part\":{{\"type\":\"reasoning_text\",\"text\":\"\"}}}}\n\n",
                                .{ s1, slot.request_id, slot.responses_reasoning_output_idx },
                            );
                            if (slot.reasoning_summary == .auto or slot.reasoning_summary == .detailed) {
                                const s2 = slot.responses_seq_num;
                                slot.responses_seq_num += 1;
                                try w.print(
                                    "event: response.reasoning_summary_part.added\ndata: {{\"type\":\"response.reasoning_summary_part.added\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"summary_index\":0,\"part\":{{\"type\":\"summary_text\",\"text\":\"\"}}}}\n\n",
                                    .{ s2, slot.request_id, slot.responses_reasoning_output_idx },
                                );
                            }
                        }
                        const srt = slot.responses_seq_num;
                        slot.responses_seq_num += 1;
                        try w.print(
                            "event: response.reasoning_text.delta\ndata: {{\"type\":\"response.reasoning_text.delta\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"content_index\":0,\"delta\":\"",
                            .{ srt, slot.request_id, slot.responses_reasoning_output_idx },
                        );
                        try w.writeJsonEscaped(span);
                        try w.writeAll("\"}\n\n");

                        if (slot.reasoning_summary == .auto or slot.reasoning_summary == .detailed) {
                            const sd = slot.responses_seq_num;
                            slot.responses_seq_num += 1;
                            try w.print(
                                "event: response.reasoning_summary_text.delta\ndata: {{\"type\":\"response.reasoning_summary_text.delta\",\"sequence_number\":{d},\"item_id\":\"rs_{d}\",\"output_index\":{d},\"summary_index\":0,\"delta\":\"",
                                .{ sd, slot.request_id, slot.responses_reasoning_output_idx },
                            );
                            try w.writeJsonEscaped(span);
                            try w.writeAll("\"}\n\n");
                        }
                    },
                    .content => {
                        if (slot.responses_reasoning_opened and !slot.responses_reasoning_closed) {
                            try self.flushAndCloseResponsesOpenItems(slot);
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
        const ttft_ms: f64 = if (slot.first_emitted_ns > 0)
            @as(f64, @floatFromInt(slot.first_emitted_ns -| slot.admit_ns)) / 1_000_000.0
        else
            0.0;

        if (slot.completion_tokens > 1) {
            const gen_anchor: u64 = if (slot.first_generated_token_ns > 0)
                slot.first_generated_token_ns
            else if (slot.prefill_done_ns > 0)
                slot.prefill_done_ns
            else
                0;
            if (gen_anchor > 0) {
                const decode_ns = end_ns -| gen_anchor;
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
        }
        if (slot.completion_tokens >= 1) {
            const decode_anchor = if (slot.prefill_done_ns > 0 and slot.prefill_done_ns <= end_ns)
                slot.prefill_done_ns
            else if (slot.first_generated_token_ns > 0 and slot.first_generated_token_ns <= end_ns)
                slot.first_generated_token_ns
            else
                slot.admit_ns;
            const total_ns = end_ns -| decode_anchor;
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
        if (slot.endpoint == .completions) {
            const RawHoldFlushCtx = struct {
                server: *Server,
                slot: *ConnectionSlot,
                fn onSafe(ctx: *@This(), safe_bytes: []const u8) error{BufferTooSmall}!void {
                    if (safe_bytes.len == 0) return;
                    if (ctx.slot.first_emitted_ns == 0) {
                        const now = monotonicNowNs();
                        ctx.slot.first_emitted_ns = now;
                        if (ctx.slot.first_token_ns == 0) ctx.slot.first_token_ns = now;
                    }
                    ctx.server.appendNonStreamSpan(ctx.slot, .content, safe_bytes);
                    if (ctx.slot.stream) {
                        try ctx.server.appendStreamDelta(ctx.slot, .content, safe_bytes);
                    }
                }
            };
            var rhfctx = RawHoldFlushCtx{ .server = self, .slot = slot };
            slot.stop_holdback.flush(&rhfctx, RawHoldFlushCtx.onSafe) catch {
                self.closeSlotSocketAndRelease(slot, true);
                return;
            };
        } else {
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
                                if (channel == .reasoning and tctx.slot.thinking_budget_exhausted) return;
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
                        if (channel == .reasoning and ctx.slot.thinking_budget_exhausted) return;
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
        }

        // Upgrade finish_reason to `.tool_calls` (`"tool_calls"` / `"tool_use"`) when tool calls were emitted!
        const reason: FinishReason = if (slot.endpoint != .completions and slot.tool_parser.hasToolCalls() and (raw_reason == .stop or raw_reason == .stop_sequence))
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
        const was_truncated = slot.nonstream_reasoning_truncated or slot.nonstream_content_truncated or slot.tool_calls_truncated;
        const inc_reason: IncompleteReason = if (was_truncated)
            .max_response_bytes
        else if (reason == .length)
            .max_output_tokens
        else
            .none;
        const rec_status: ResponseRecordStatus = if (reason == .cancelled)
            .cancelled
        else if (was_truncated or reason == .length)
            .incomplete
        else
            .completed;
        self.finalizeStatefulResponseRecord(slot, rec_status, inc_reason);

        if (slot.stream) {
            self.ensureStreamPreamble(slot);
            if (slot.endpoint == .responses) {
                self.flushAndCloseResponsesOpenItems(slot) catch {
                    self.closeSlotSocketAndRelease(slot, true);
                    return;
                };
                _ = self.flushTx(slot);
                slot.compactTx();
                var w = FixedWriter.init(slot.tx_buf[slot.tx_write_pos..]);
                const sc = slot.responses_seq_num;
                slot.responses_seq_num += 1;
                const reasoning = slot.nonstream_reasoning_buf[0..slot.nonstream_reasoning_len];
                const content = slot.nonstream_content_buf[0..slot.nonstream_content_len];
                const write_ok: bool = blk: {
                    const term_event_name = if (rec_status == .cancelled) "response.cancelled" else "response.completed";
                    w.print(
                        "event: {s}\ndata: {{\"type\":\"{s}\",\"sequence_number\":{d},\"response\":{{\"id\":\"resp_{d}\",\"object\":\"response\",\"created_at\":{d},\"status\":\"{s}\",\"model\":\"",
                        .{ term_event_name, term_event_name, sc, slot.request_id, slot.created_unix, rec_status.name() },
                    ) catch break :blk false;
                    w.writeJsonEscaped(slot.modelSlice()) catch break :blk false;
                    w.print("\",\"seed_used\":{d},\"seed_source\":\"{s}\"", .{
                        slot.seed_used,
                        slot.seed_source.name(),
                    }) catch break :blk false;
                    if (rec_status == .incomplete and inc_reason != .none) {
                        w.print(",\"incomplete_details\":{{\"reason\":\"{s}\"}}", .{inc_reason.name()}) catch break :blk false;
                    }
                    if (slot.previous_response_id_len > 0) {
                        w.writeAll(",\"previous_response_id\":\"") catch break :blk false;
                        w.writeJsonEscaped(slot.previousResponseIdSlice()) catch break :blk false;
                        w.writeByte('"') catch break :blk false;
                    } else {
                        w.writeAll(",\"previous_response_id\":null") catch break :blk false;
                    }
                    w.writeAll(",\"output\":") catch break :blk false;
                    writeResponsesOutputArrayJson(
                        &w,
                        slot.request_id,
                        rec_status.name(),
                        reasoning,
                        content,
                        slot.reasoning_summary,
                        slot.stored_tool_call_count > 0,
                        slot.stored_tool_calls[0..slot.stored_tool_call_count],
                    ) catch break :blk false;
                    w.writeAll(",\"output_text\":\"") catch break :blk false;
                    w.writeJsonEscaped(content) catch break :blk false;
                    w.print(
                        "\",\"usage\":{{\"input_tokens\":{d},\"input_tokens_details\":{{\"cached_tokens\":{d}}},\"output_tokens\":{d},\"output_tokens_details\":{{\"reasoning_tokens\":{d}}},\"total_tokens\":{d}}}," ++
                            "\"aminfer_timings\":{{\"ttft_ms\":{d:.3},\"tpot_ms\":{d:.3},\"tok_per_s\":{d:.2}}}}}}}\n\n",
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
                    break :blk true;
                };
                if (!write_ok) {
                    self.closeSlotSocketAndRelease(slot, true);
                    return;
                }
                slot.tx_write_pos += w.pos;
                self.releaseRequestCount(slot);
                slot.state = .draining_response;
                if (self.flushTx(slot) and slot.pendingTxLen() == 0) {
                    self.closeSlotSocketAndRelease(slot, false);
                }
                return;
            }

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
                .completions => {
                    w.print(
                        "data: {{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":\"",
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
                        "\",\"seed_used\":{d},\"seed_source\":\"{s}\",\"choices\":[{{\"text\":\"\",\"index\":0,\"logprobs\":null,\"finish_reason\":\"{s}\"}}]," ++
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
                        w.print(
                            "data: {{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":\"",
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
                .responses => unreachable,
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
        self.writeNonStreamJsonResponse(slot, reason, rec_status, inc_reason, timings, total_tokens);
    }

    fn writeNonStreamJsonResponse(
        self: *Server,
        slot: *ConnectionSlot,
        reason: FinishReason,
        rec_status: ResponseRecordStatus,
        inc_reason: IncompleteReason,
        timings: Timings,
        total_tokens: u32,
    ) void {
        const HEADER_RESERVE: usize = 768;
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
                .completions => {
                    bw.print(
                        "{{\"id\":\"cmpl-{d}\",\"object\":\"text_completion\",\"created\":{d},\"model\":\"",
                        .{ slot.request_id, slot.created_unix },
                    ) catch break :blk false;
                    bw.writeJsonEscaped(slot.modelSlice()) catch break :blk false;
                    bw.print(
                        "\",\"seed_used\":{d},\"seed_source\":\"{s}\",\"choices\":[{{\"text\":\"",
                        .{ slot.seed_used, slot.seed_source.name() },
                    ) catch break :blk false;
                    bw.writeJsonEscaped(content) catch break :blk false;
                    bw.print(
                        "\",\"index\":0,\"logprobs\":null,\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}},\"completion_tokens_details\":{{\"reasoning_tokens\":{d}}}}}," ++
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
                        "{{\"id\":\"resp_{d}\",\"object\":\"response\",\"created_at\":{d},\"status\":\"{s}\",\"model\":\"",
                        .{ slot.request_id, slot.created_unix, rec_status.name() },
                    ) catch break :blk false;
                    bw.writeJsonEscaped(slot.modelSlice()) catch break :blk false;
                    bw.print("\",\"seed_used\":{d},\"seed_source\":\"{s}\"", .{
                        slot.seed_used,
                        slot.seed_source.name(),
                    }) catch break :blk false;
                    if (rec_status == .incomplete and inc_reason != .none) {
                        bw.print(",\"incomplete_details\":{{\"reason\":\"{s}\"}}", .{inc_reason.name()}) catch break :blk false;
                    }
                    if (slot.previous_response_id_len > 0) {
                        bw.writeAll(",\"previous_response_id\":\"") catch break :blk false;
                        bw.writeJsonEscaped(slot.previousResponseIdSlice()) catch break :blk false;
                        bw.writeByte('"') catch break :blk false;
                    }
                    bw.writeAll(",\"output\":") catch break :blk false;
                    writeResponsesOutputArrayJson(
                        &bw,
                        slot.request_id,
                        rec_status.name(),
                        reasoning,
                        content,
                        slot.reasoning_summary,
                        has_tools,
                        slot.stored_tool_calls[0..slot.stored_tool_call_count],
                    ) catch break :blk false;
                    bw.writeAll(",\"output_text\":\"") catch break :blk false;
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
                        var sig_hex: [64]u8 = undefined;
                        const sig = computeThinkingSignature(&self.signature_key, slot.modelSlice(), reasoning, &sig_hex);
                        bw.writeAll("{\"type\":\"thinking\",\"thinking\":\"") catch break :blk false;
                        bw.writeJsonEscaped(reasoning) catch break :blk false;
                        bw.writeAll("\",\"signature\":\"") catch break :blk false;
                        bw.writeAll(sig) catch break :blk false;
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
        self.writeHttpJsonOkHeadersAndSend(slot, body_len, HEADER_RESERVE, slot.seed_used, slot.seed_source);
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
            self.finishHttpExchange(slot);
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

fn testRecvOneHttpResponse(fd: std.c.fd_t, out_buf: []u8) TestClientError![]const u8 {
    var total: usize = 0;
    var expected_end: ?usize = null;
    while (true) {
        if (expected_end) |ee| {
            if (total >= ee) return out_buf[0..ee];
        }
        if (total >= out_buf.len) return error.BufferTooSmall;
        const n = std.c.recv(fd, out_buf.ptr + total, out_buf.len - total, 0);
        if (n > 0) {
            total += @intCast(n);
            if (expected_end == null) {
                if (std.mem.indexOf(u8, out_buf[0..total], "\r\n\r\n")) |sep| {
                    const hdr_end = sep + 4;
                    if (extractHeaderValue(out_buf[0..hdr_end], "Content-Length")) |cl_str| {
                        const cl = std.fmt.parseInt(usize, cl_str, 10) catch return error.RecvFailed;
                        expected_end = hdr_end + cl;
                        if (expected_end.? > out_buf.len) return error.BufferTooSmall;
                    }
                }
            }
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
    return testRecvOneHttpResponse(fd, resp_buf);
}

fn testHttpPostJson(port: u16, path: []const u8, json_body: []const u8, req_buf: []u8, resp_buf: []u8) TestClientError![]const u8 {
    var w = FixedWriter.init(req_buf);
    w.print(
        "POST {s} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
        .{ path, json_body.len, json_body },
    ) catch return error.BufferTooSmall;
    return testHttpExchange(port, w.written(), resp_buf);
}

fn testHttpPostJsonWithHeaders(
    port: u16,
    path: []const u8,
    extra_headers: []const u8,
    json_body: []const u8,
    req_buf: []u8,
    resp_buf: []u8,
) TestClientError![]const u8 {
    var w = FixedWriter.init(req_buf);
    w.print(
        "POST {s} HTTP/1.1\r\nHost: 127.0.0.1\r\n{s}Content-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
        .{ path, extra_headers, json_body.len, json_body },
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
        try std.testing.expect(std.mem.indexOf(u8, json, "{\"type\":\"thinking\",\"thinking\":\"reason A\",\"signature\":\"") != null);
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
    return paths.tmpPathZ(buf, name);
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
    return testRecvOneHttpResponse(fd, resp_buf);
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

test "W24 (a)+(b) reasoning_effort, chat_template_kwargs, contradiction 400 rejection, and Inkling 2 / Hy4 tool calling" {
    var prompt_buf: [8192]u8 = undefined;
    var model_buf: [MAX_MODEL_ID_BYTES]u8 = undefined;
    const ink_tpl = chat_template.ChatTemplate.preset(.inkling2);
    const hy4_tpl = chat_template.ChatTemplate.preset(.hy4);

    // 1. `/v1/chat/completions` top-level `reasoning_effort` + float + `chat_template_kwargs` on Inkling 2 & Hy4.
    {
        const body =
            \\{"messages":[{"role":"user","content":"Solve"}],"reasoning_effort":"xhigh"}
        ;
        const req = try parseInferenceRequestWithTemplate(.chat_completions, body, "ink", &ink_tpl, null, null, null, &prompt_buf, &model_buf);
        try std.testing.expect(req.enable_thinking);
        try std.testing.expectEqualStrings(
            "<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|><|message_user|><|content_text|>Solve<|end_message|>",
            prompt_buf[0..req.prompt_len],
        );
    }
    {
        const body =
            \\{"messages":[{"role":"user","content":"Solve"}],"chat_template_kwargs":{"reasoning_effort":0.5,"preserved_thinking":true,"enable_thinking":true}}
        ;
        const req = try parseInferenceRequestWithTemplate(.chat_completions, body, "ink", &ink_tpl, null, null, null, &prompt_buf, &model_buf);
        try std.testing.expect(req.enable_thinking);
        try std.testing.expectEqual(@as(?bool, true), req.preserved_thinking);
        try std.testing.expect(std.mem.indexOf(u8, prompt_buf[0..req.prompt_len], "Thinking effort level: 0.5") != null);
    }
    {
        const body =
            \\{"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"medium"}
        ;
        const req = try parseInferenceRequestWithTemplate(.chat_completions, body, "hy4", &hy4_tpl, null, null, null, &prompt_buf, &model_buf);
        try std.testing.expect(req.enable_thinking);
        try std.testing.expect(req.starts_in_think);
        try std.testing.expect(std.mem.indexOf(u8, prompt_buf[0..req.prompt_len], "<｜reasoning_mode:opensource｜>reasoning_effort:high") != null);
        try std.testing.expect(std.mem.endsWith(u8, prompt_buf[0..req.prompt_len], "<think:opensource>"));
    }
    {
        const body =
            \\{"messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"reasoning_effort":"no_think","enable_thinking":false}}
        ;
        const req = try parseInferenceRequestWithTemplate(.chat_completions, body, "hy4", &hy4_tpl, null, null, null, &prompt_buf, &model_buf);
        try std.testing.expect(!req.enable_thinking);
        try std.testing.expect(!req.starts_in_think);
        try std.testing.expect(std.mem.indexOf(u8, prompt_buf[0..req.prompt_len], "reasoning_effort:no_think") != null);
    }

    // 2. Contradictory thinking parameters MUST fail with `error.InvalidFieldValue`.
    const contradictions = [_][]const u8{
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"enable_thinking\":false,\"reasoning_effort\":\"high\"}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"enable_thinking\":true,\"reasoning_effort\":\"none\"}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"reasoning_effort\":\"low\",\"chat_template_kwargs\":{\"reasoning_effort\":\"high\"}}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"enable_thinking\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"thinking\":{\"type\":\"disabled\"},\"reasoning_effort\":\"medium\"}",
    };
    for (contradictions) |bad_body| {
        try std.testing.expectError(
            error.InvalidFieldValue,
            parseInferenceRequest(.chat_completions, bad_body, "m", &prompt_buf, &model_buf),
        );
    }

    // 3. HTTP loopback test: contradiction returns 400 + Inkling 2 & Hy4 tool calling SSE works end-to-end.
    const InklingMock = struct {
        fn generate(_: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(8);
            if (std.mem.indexOf(u8, job.prompt, "Thinking effort level: 0.7") == null) {
                stream.fail("missing inkling effort");
                return;
            }
            const chunks = [_][]const u8{
                "<|message_model|><|content_thin",
                "king|>plan<|end_mes",
                "sage|><|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Tokyo\"}}<|end_message|><|content_model_end_sampling|>",
            };
            for (chunks) |c| {
                if (!stream.emitTokenBytes(c)) return;
            }
            stream.finish(.stop, 8);
        }
    };
    var dummy: u8 = 0;
    const srv_ink = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "inkling-2",
        .chat_template = ink_tpl,
        .apply_qos = false,
    });
    defer srv_ink.deinit(std.testing.allocator);
    try srv_ink.start(.{ .ctx = &dummy, .generateFn = InklingMock.generate });
    defer srv_ink.stop();

    var req_buf: [2048]u8 = undefined;
    var resp_buf: [16 * 1024]u8 = undefined;

    const bad_resp = try testHttpPostJson(
        srv_ink.boundPort(),
        "/v1/chat/completions",
        contradictions[0],
        &req_buf,
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, bad_resp, "HTTP/1.1 400 Bad Request\r\n"));
    try std.testing.expect(std.mem.indexOf(u8, bad_resp, "invalid_request_error") != null);

    const ink_resp = try testHttpPostJson(
        srv_ink.boundPort(),
        "/v1/chat/completions",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\"}}}],\"reasoning_effort\":\"medium\",\"stream\":false}",
        &req_buf,
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, ink_resp, "HTTP/1.1 200 OK\r\n"));
    const ink_json = extractHttpBody(ink_resp) orelse return error.TestUnexpectedResult;
    try expectValidJson(ink_json);
    try std.testing.expect(std.mem.indexOf(u8, ink_json, "\"reasoning_content\":\"plan\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, ink_json, "\"content\":null") != null);
    try std.testing.expect(std.mem.indexOf(u8, ink_json, "<|message_model|>") == null);
    try std.testing.expect(std.mem.indexOf(u8, ink_json, "\"name\":\"get_weather\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, ink_json, "Tokyo") != null);
    try std.testing.expect(std.mem.indexOf(u8, ink_json, "\"finish_reason\":\"tool_calls\"") != null);

    // 4. Hy4 loopback with starts_in_think == true + XML tool call!
    const Hy4Mock = struct {
        fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(6);
            const chunks = [_][]const u8{
                "hy_plan</think:open",
                "source><tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Kyoto</arg_value:opensource><arg_key:opensource>days</arg_key:opensource><arg_value:opensource>3</arg_value:opensource></tool_call:opensource></tool_calls:opensource>",
            };
            for (chunks) |c| {
                if (!stream.emitTokenBytes(c)) return;
            }
            stream.finish(.stop, 6);
        }
    };
    const srv_hy4 = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "hy4-flash",
        .chat_template = hy4_tpl,
        .apply_qos = false,
    });
    defer srv_hy4.deinit(std.testing.allocator);
    try srv_hy4.start(.{ .ctx = &dummy, .generateFn = Hy4Mock.generate });
    defer srv_hy4.stop();

    const hy_resp = try testHttpPostJson(
        srv_hy4.boundPort(),
        "/v1/chat/completions",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Kyoto?\"}],\"reasoning_effort\":\"low\",\"stream\":false}",
        &req_buf,
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, hy_resp, "HTTP/1.1 200 OK\r\n"));
    const hy_json = extractHttpBody(hy_resp) orelse return error.TestUnexpectedResult;
    try expectValidJson(hy_json);
    try std.testing.expect(std.mem.indexOf(u8, hy_json, "\"reasoning_content\":\"hy_plan\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, hy_json, "\"name\":\"get_weather\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, hy_json, "Kyoto") != null);
    try std.testing.expect(std.mem.indexOf(u8, hy_json, "\"finish_reason\":\"tool_calls\"") != null);
}

test "C18 (c)+(e) active thinking_budget_tokens enforcement and post-ThinkSplitter reasoning_tokens counting" {
    const BudgetMock = struct {
        mode: enum { ignore_budget, check_budget, split_edge } = .ignore_budget,
        saw_worker_cutoff: std.atomic.Value(bool) = .init(false),

        fn generate(ctx: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            _ = stream.setPromptTokens(5);
            switch (self.mode) {
                .ignore_budget => {
                    // Worker ignores budget and emits 5 reasoning tokens + close + answer.
                    const toks = [_][]const u8{ "<think>", "t1", "t2", "t3", "t4", "t5", "</think>", "Done" };
                    for (toks) |t| {
                        if (!stream.emitTokenBytes(t)) return;
                    }
                },
                .check_budget => {
                    const toks = [_][]const u8{ "<think>", "r1", "r2", "r3", "r4" };
                    for (toks) |t| {
                        if (!stream.emitTokenBytes(t)) return;
                        if (stream.thinkingBudgetExceeded()) {
                            self.saw_worker_cutoff.store(true, .release);
                            _ = stream.emitTokenBytes("</think>");
                            break;
                        }
                    }
                    _ = stream.emitTokenBytes("Answer");
                },
                .split_edge => {
                    // Test (e): opening `<think>a` batch must count, closing `</think>body` must NOT overcount!
                    const toks = [_][]const u8{ "<think>a", "b", "</think>body" };
                    for (toks) |t| {
                        if (!stream.emitTokenBytes(t)) return;
                    }
                },
            }
            stream.finish(.stop, 5);
        }
    };

    var mock = BudgetMock{ .mode = .ignore_budget };
    const srv = try Server.init(std.testing.allocator, .{ .port = 0, .model_id = "qwen3", .apply_qos = false });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &mock, .generateFn = BudgetMock.generate });
    defer srv.stop();

    var req_buf: [2048]u8 = undefined;
    var resp_buf: [16 * 1024]u8 = undefined;

    // 1. Server-enforced budget cutoff (`budget_tokens: 3`): drops `t4`, `t5`, keeps `Done`, `reasoning_tokens == 3`!
    const r1 = try testHttpPostJson(
        srv.boundPort(),
        "/v1/messages",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":3},\"max_tokens\":64,\"stream\":false}",
        &req_buf,
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, r1, "HTTP/1.1 200 OK\r\n"));
    const j1 = extractHttpBody(r1) orelse return error.TestUnexpectedResult;
    try expectValidJson(j1);
    try std.testing.expect(std.mem.indexOf(u8, j1, "\"thinking\":\"t1t2t3\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, j1, "t4") == null);
    try std.testing.expect(std.mem.indexOf(u8, j1, "\"text\":\"Done\"") != null);

    // 2. Worker-cooperative budget cutoff via `stream.thinkingBudgetExceeded()`.
    mock.mode = .check_budget;
    const r2 = try testHttpPostJson(
        srv.boundPort(),
        "/v1/messages",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":3},\"max_tokens\":64,\"stream\":true}",
        &req_buf,
        &resp_buf,
    );
    const sse2 = extractHttpBody(r2) orelse return error.TestUnexpectedResult;
    _ = try expectValidSseJsonFrames(sse2);
    try std.testing.expect(mock.saw_worker_cutoff.load(.acquire));
    try std.testing.expect(std.mem.indexOf(u8, sse2, "r4") == null);
    try std.testing.expect(std.mem.indexOf(u8, sse2, "\"text\":\"Answer\"") != null);

    // 3. (e) Opening `<think>a` + closing `</think>body` -> exact `completion_tokens=3`, `reasoning_tokens=2`!
    mock.mode = .split_edge;
    const r3 = try testHttpPostJson(
        srv.boundPort(),
        "/v1/chat/completions",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}",
        &req_buf,
        &resp_buf,
    );
    const j3 = extractHttpBody(r3) orelse return error.TestUnexpectedResult;
    try expectValidJson(j3);
    try std.testing.expect(std.mem.indexOf(u8, j3, "\"completion_tokens\":3") != null);
    try std.testing.expect(std.mem.indexOf(u8, j3, "\"reasoning_tokens\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, j3, "\"reasoning_content\":\"ab\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, j3, "\"content\":\"body\"") != null);
}

test "C18 (d)+(f)+(g) reasoning_text, summary modes, full response.completed output, and HMAC-SHA256 thinking signature" {
    const RichMock = struct {
        fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(10);
            const reason =
                "First sentence establishes the problem framing carefully. " ++
                "Middle step explores intermediate algebraic transformations in considerable detail so total length exceeds concise threshold. " ++
                "Final sentence concludes the exact numerical result cleanly.";
            _ = stream.emitTokenBytes("<think>");
            _ = stream.emitTokenBytes(reason);
            _ = stream.emitTokenBytes("</think>Result=42");
            stream.finish(.stop, 10);
        }
    };
    var dummy: u8 = 0;
    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-sig",
        .api_key = "test-secret-key",
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &dummy, .generateFn = RichMock.generate });
    defer srv.stop();

    var req_buf: [4096]u8 = undefined;
    var resp_buf: [24 * 1024]u8 = undefined;
    const AuthHelper = struct {
        fn post(port: u16, path: []const u8, body: []const u8, rbuf: []u8, obuf: []u8) ![]const u8 {
            var w = FixedWriter.init(rbuf);
            try w.print(
                "POST {s} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test-secret-key\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
                .{ path, body.len, body },
            );
            return testHttpExchange(port, w.written(), obuf);
        }
    };

    // 1. (d)+(f): `/v1/responses` with `summary: "concise"` streaming -> reasoning_text + head..tail summary + full response.completed!
    {
        const r = try AuthHelper.post(
            srv.boundPort(),
            "/v1/responses",
            "{\"input\":\"hi\",\"reasoning\":{\"effort\":\"high\",\"summary\":\"concise\"},\"stream\":true}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r, "HTTP/1.1 200 OK\r\n"));
        const sse = extractHttpBody(r) orelse return error.TestUnexpectedResult;
        _ = try expectValidSseJsonFrames(sse);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.reasoning_text.delta\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "event: response.reasoning_text.done\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "First sentence establishes the problem framing carefully. ... Final sentence concludes the exact numerical result cleanly.") != null);
        // `response.completed.response` includes `created_at`, `previous_response_id`, and `output` array!
        const comp_pos = std.mem.indexOf(u8, sse, "event: response.completed\n") orelse return error.TestUnexpectedResult;
        const comp_tail = sse[comp_pos..];
        try std.testing.expect(std.mem.indexOf(u8, comp_tail, "\"created_at\":") != null);
        try std.testing.expect(std.mem.indexOf(u8, comp_tail, "\"previous_response_id\":null") != null);
        try std.testing.expect(std.mem.indexOf(u8, comp_tail, "\"output\":[") != null);
        try std.testing.expect(std.mem.indexOf(u8, comp_tail, "\"type\":\"reasoning_text\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, comp_tail, "\"output_text\":\"Result=42\"") != null);
    }

    // 2. (d) `/v1/responses` with `summary: "none"` -> `content: [reasoning_text]`, `summary: []`.
    {
        const r = try AuthHelper.post(
            srv.boundPort(),
            "/v1/responses",
            "{\"input\":\"hi\",\"reasoning\":{\"effort\":\"high\",\"summary\":\"none\"},\"stream\":false}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r, "HTTP/1.1 200 OK\r\n"));
        const json = extractHttpBody(r) orelse return error.TestUnexpectedResult;
        try expectValidJson(json);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"reasoning_text\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"summary\":[]") != null);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"summary_text\"") == null);
    }

    // 3. (g) Anthropic `/v1/messages`: extract HMAC-SHA256 signature from response, verify turn-2 echo passes & tampered signature returns 400!
    {
        const r_stream = try AuthHelper.post(
            srv.boundPort(),
            "/v1/messages",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":32,\"stream\":true}",
            &req_buf,
            &resp_buf,
        );
        const sse = extractHttpBody(r_stream) orelse return error.TestUnexpectedResult;
        _ = try expectValidSseJsonFrames(sse);

        const r_non = try AuthHelper.post(
            srv.boundPort(),
            "/v1/messages",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":32,\"stream\":false}",
            &req_buf,
            &resp_buf,
        );
        const json = extractHttpBody(r_non) orelse return error.TestUnexpectedResult;
        try expectValidJson(json);

        var sig_buf: [128]u8 = undefined;
        const sig = findJsonStringField(json, "signature", &sig_buf) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(@as(usize, 64), sig.len);
        // Streaming signature_delta emitted the exact same HMAC-SHA256 digest!
        try std.testing.expect(std.mem.indexOf(u8, sse, sig) != null);

        const reason =
            "First sentence establishes the problem framing carefully. " ++
            "Middle step explores intermediate algebraic transformations in considerable detail so total length exceeds concise threshold. " ++
            "Final sentence concludes the exact numerical result cleanly.";

        // Valid multi-turn echo with genuine HMAC signature -> 200 OK!
        var valid_body: [2048]u8 = undefined;
        var vw = FixedWriter.init(&valid_body);
        try vw.print(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}},{{\"role\":\"assistant\",\"content\":[{{\"type\":\"thinking\",\"thinking\":\"{s}\",\"signature\":\"{s}\"}},{{\"type\":\"text\",\"text\":\"Result=42\"}}]}},{{\"role\":\"user\",\"content\":\"next\"}}],\"max_tokens\":32,\"stream\":false}}",
            .{ reason, sig },
        );
        var hdr_req: [3072]u8 = undefined;
        var hw = FixedWriter.init(&hdr_req);
        try hw.print(
            "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test-secret-key\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ vw.pos, vw.written() },
        );
        const ok_turn2 = try testHttpExchange(srv.boundPort(), hw.written(), &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, ok_turn2, "HTTP/1.1 200 OK\r\n"));

        // Tampered signature (flip last hex char) -> 400 Bad Request!
        var bad_sig: [64]u8 = undefined;
        @memcpy(&bad_sig, sig[0..64]);
        bad_sig[63] = if (bad_sig[63] == '0') '1' else '0';
        var bad_body: [2048]u8 = undefined;
        var bw2 = FixedWriter.init(&bad_body);
        try bw2.print(
            "{{\"messages\":[{{\"role\":\"assistant\",\"content\":[{{\"type\":\"thinking\",\"thinking\":\"{s}\",\"signature\":\"{s}\"}}]}},{{\"role\":\"user\",\"content\":\"next\"}}],\"max_tokens\":32}}",
            .{ reason, bad_sig[0..] },
        );
        var bad_hdr: [3072]u8 = undefined;
        var bhw = FixedWriter.init(&bad_hdr);
        try bhw.print(
            "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test-secret-key\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ bw2.pos, bw2.written() },
        );
        const bad_turn2 = try testHttpExchange(srv.boundPort(), bhw.written(), &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, bad_turn2, "HTTP/1.1 400 Bad Request\r\n"));
    }
}

test "C18 (h)+(i)+(j) truncation incomplete status + refuse corrupt chain, non-evictable in_progress slots 503, and tryPush rollback" {
    // 1. (h) `max_response_bytes` cap sets `status: "incomplete"`, `incomplete_details`, and refuses corrupt chaining.
    {
        const LongMock = struct {
            fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
                _ = stream.setPromptTokens(4);
                _ = stream.emitTokenBytes("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
                stream.finish(.stop, 4);
            }
        };
        var dummy: u8 = 0;
        const srv = try Server.init(std.testing.allocator, .{
            .port = 0,
            .model_id = "qwen3-trunc",
            .max_response_bytes = 24,
            .apply_qos = false,
        });
        defer srv.deinit(std.testing.allocator);
        try srv.start(.{ .ctx = &dummy, .generateFn = LongMock.generate });
        defer srv.stop();

        var req_buf: [1024]u8 = undefined;
        var resp_buf: [8192]u8 = undefined;
        const r = try testHttpPostJson(
            srv.boundPort(),
            "/v1/responses",
            "{\"input\":\"long\",\"stream\":false}",
            &req_buf,
            &resp_buf,
        );
        const json = extractHttpBody(r) orelse return error.TestUnexpectedResult;
        try expectValidJson(json);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"status\":\"incomplete\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, json, "\"reason\":\"max_response_bytes\"") != null);

        var id_buf: [64]u8 = undefined;
        const rid = findJsonStringField(json, "id", &id_buf) orelse return error.TestUnexpectedResult;
        var chain_body: [256]u8 = undefined;
        var cw = FixedWriter.init(&chain_body);
        try cw.print("{{\"input\":\"follow up\",\"previous_response_id\":\"{s}\",\"stream\":false}}", .{rid});
        const r_chain = try testHttpPostJson(srv.boundPort(), "/v1/responses", cw.written(), &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_chain, "HTTP/1.1 409 Conflict\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, r_chain, "incomplete_previous_response") != null);
    }

    // 2. (i)+(j): Active `.in_progress` stateful slots never evicted (`503 server_overloaded`) + `tryPush` rollback!
    {
        const GateMock = struct {
            entered: std.atomic.Value(u32) = .init(0),
            release: std.atomic.Value(bool) = .init(false),
            fn generate(ctx: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
                const self: *@This() = @ptrCast(@alignCast(ctx));
                _ = self.entered.fetchAdd(1, .acq_rel);
                _ = stream.setPromptTokens(2);
                while (!self.release.load(.acquire) and !stream.isCancelled()) {
                    sleepNs(500_000);
                }
                _ = stream.emitTokenBytes("ok");
                stream.finish(.stop, 2);
            }
        };
        var mock = GateMock{};
        // (i) 2 stateful slots, 6 conns, 4 active requests -> 2 active `/v1/responses` streams saturate store!
        const srv = try Server.init(std.testing.allocator, .{
            .port = 0,
            .model_id = "qwen3-slot",
            .max_conns = 6,
            .max_active_requests = 4,
            .stateful_response_slots = 2,
            .job_ring_cap = 8,
            .apply_qos = false,
        });
        defer srv.deinit(std.testing.allocator);
        try srv.start(.{ .ctx = &mock, .generateFn = GateMock.generate });
        defer srv.stop();

        const port = srv.boundPort();
        const stream_body = "{\"input\":\"hold\",\"stream\":true}";
        var sreq: [512]u8 = undefined;
        var sw = FixedWriter.init(&sreq);
        try sw.print("POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n{s}", .{ stream_body.len, stream_body });

        const fd1 = try testConnectLoopback(port);
        defer _ = std.c.close(fd1);
        try testSendAll(fd1, sw.written());
        const fd2 = try testConnectLoopback(port);
        defer _ = std.c.close(fd2);
        try testSendAll(fd2, sw.written());

        // Wait until first request is held in worker AND second has at least been admitted (store holds 2 `.in_progress`).
        while (mock.entered.load(.acquire) < 1 or srv.stats().active_requests < 2) {
            sleepNs(500_000);
        }

        // 3rd `/v1/responses` MUST get `503 Service Unavailable` (`server_overloaded`), NOT evict `.in_progress` slots!
        var req_buf: [512]u8 = undefined;
        var resp_buf: [4096]u8 = undefined;
        const r3 = try testHttpPostJson(port, "/v1/responses", "{\"input\":\"overflow\",\"stream\":false}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r3, "HTTP/1.1 503 Service Unavailable\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, r3, "server_overloaded") != null);

        // Both active slots remain intact!
        try std.testing.expectEqual(ResponseRecordStatus.in_progress, srv.response_store[0].status);
        try std.testing.expectEqual(ResponseRecordStatus.in_progress, srv.response_store[1].status);

        mock.release.store(true, .release);
        var tmp1: [4096]u8 = undefined;
        var tmp2: [4096]u8 = undefined;
        _ = try testRecvAll(fd1, &tmp1);
        _ = try testRecvAll(fd2, &tmp2);
    }

    // (j) Separate check: `max_active_requests = 2, job_ring_cap = 2` -> worker holds 1st, 2nd queued, 3rd returns 503 with zero leaked slot; plus direct `rollbackStatefulResponseSlot` verification!
    {
        const HoldMock = struct {
            entered: std.atomic.Value(bool) = .init(false),
            release: std.atomic.Value(bool) = .init(false),
            fn generate(ctx: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
                const self: *@This() = @ptrCast(@alignCast(ctx));
                self.entered.store(true, .release);
                _ = stream.setPromptTokens(1);
                while (!self.release.load(.acquire) and !stream.isCancelled()) {
                    sleepNs(500_000);
                }
                _ = stream.emitTokenBytes("done");
                stream.finish(.stop, 1);
            }
        };
        var hmock = HoldMock{};
        const srv = try Server.init(std.testing.allocator, .{
            .port = 0,
            .model_id = "qwen3-rollback",
            .max_conns = 6,
            .max_active_requests = 2,
            .stateful_response_slots = 4,
            .job_ring_cap = 2,
            .apply_qos = false,
        });
        defer srv.deinit(std.testing.allocator);
        try srv.start(.{ .ctx = &hmock, .generateFn = HoldMock.generate });
        defer srv.stop();

        const port = srv.boundPort();
        const b1 = "{\"input\":\"first\",\"stream\":false}";
        var rb1: [512]u8 = undefined;
        var w1 = FixedWriter.init(&rb1);
        try w1.print("POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ b1.len, b1 });
        const c1 = try testConnectLoopback(port);
        defer _ = std.c.close(c1);
        try testSendAll(c1, w1.written());
        while (!hmock.entered.load(.acquire)) sleepNs(500_000);

        // 2nd request queued while worker is held -> active_requests == 2.
        const c2 = try testConnectLoopback(port);
        defer _ = std.c.close(c2);
        try testSendAll(c2, w1.written());
        while (srv.stats().active_requests < 2) sleepNs(500_000);

        // 3rd request MUST hit 503 AND leave only 2 response_store slots occupied!
        var req_buf: [512]u8 = undefined;
        var resp_buf: [4096]u8 = undefined;
        const r3 = try testHttpPostJson(port, "/v1/responses", b1, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r3, "HTTP/1.1 503 Service Unavailable\r\n"));

        var occupied_in_progress: usize = 0;
        for (srv.response_store) |*rec| {
            if (rec.occupied) occupied_in_progress += 1;
        }
        try std.testing.expectEqual(@as(usize, 2), occupied_in_progress);

        hmock.release.store(true, .release);
        var o1: [4096]u8 = undefined;
        var o2: [4096]u8 = undefined;
        _ = try testRecvAll(c1, &o1);
        _ = try testRecvAll(c2, &o2);

        // Direct `rollbackStatefulResponseSlot` check matching `job_ring.tryPush` failure cleanup.
        const rs_idx: usize = 3;
        srv.response_store[rs_idx].occupied = true;
        srv.response_store[rs_idx].active_conn_slot = 0;
        srv.response_store[rs_idx].id_len = 5;
        srv.response_store[rs_idx].chatml_len = 5;
        srv.slots[0].stateful_resp_slot_idx = rs_idx;
        srv.rollbackStatefulResponseSlot(&srv.slots[0]);
        try std.testing.expect(!srv.response_store[rs_idx].occupied);
        try std.testing.expectEqual(@as(?u16, null), srv.response_store[rs_idx].active_conn_slot);
        try std.testing.expectEqual(@as(u8, 0), srv.response_store[rs_idx].id_len);
        try std.testing.expectEqual(@as(usize, 0), srv.response_store[rs_idx].chatml_len);
        try std.testing.expectEqual(@as(?u16, null), srv.slots[0].stateful_resp_slot_idx);
    }
}

test "C18 (k)+(l) http_read_timeout_ms 408 across headers/Content-Length/chunked, HTTP/1.1 keep-alive reuse, and separated timings" {
    const TimingMock = struct {
        fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            sleepNs(3_000_000); // 3 ms simulated prefill
            _ = stream.setPromptTokens(6);
            sleepNs(2_000_000); // 2 ms to first generated token
            _ = stream.emitTokenBytes("tok");
            sleepNs(2_000_000); // 2 ms decode
            stream.finish(.stop, 6);
        }
    };
    var dummy: u8 = 0;
    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .model_id = "qwen3-ka",
        .http_read_timeout_ms = 60,
        .keepalive_idle_timeout_ms = 120,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &dummy, .generateFn = TimingMock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var resp_buf: [8192]u8 = undefined;

    // 1. (k) Slow incomplete headers -> `408 Request Timeout`.
    {
        const fd = try testConnectLoopback(port);
        defer _ = std.c.close(fd);
        try testSendAll(fd, "GET /health HTTP/1.1\r\nHost: localhost\r\n");
        const r = try testRecvAll(fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r, "HTTP/1.1 408 Request Timeout\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, r, "request_timeout") != null);
    }

    // 2. (k) Incomplete `Content-Length` body -> `408 Request Timeout`.
    {
        const fd = try testConnectLoopback(port);
        defer _ = std.c.close(fd);
        try testSendAll(fd, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 50\r\n\r\n{\"a\":");
        const r = try testRecvAll(fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r, "HTTP/1.1 408 Request Timeout\r\n"));
    }

    // 3. (k) Incomplete `Transfer-Encoding: chunked` body -> `408 Request Timeout`.
    {
        const fd = try testConnectLoopback(port);
        defer _ = std.c.close(fd);
        try testSendAll(fd, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n");
        const r = try testRecvAll(fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r, "HTTP/1.1 408 Request Timeout\r\n"));
    }

    // 4. (l) HTTP/1.1 `keep-alive` reuse across multiple sequential non-streaming requests + `Connection: close` + accurate timings!
    {
        const fd = try testConnectLoopback(port);
        defer _ = std.c.close(fd);

        // Request 1 on socket (default HTTP/1.1 keep-alive)
        try testSendAll(fd, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
        const r1 = try testRecvOneHttpResponse(fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r1, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualStrings("keep-alive", extractHeaderValue(r1, "Connection") orelse "");

        // Request 2 on SAME socket: single-token inference verifying separated timings (`ttft_ms` >= 4ms, `tpot_ms` > 0ms)!
        const post_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}";
        var req2: [512]u8 = undefined;
        var w2 = FixedWriter.init(&req2);
        try w2.print("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n{s}", .{ post_body.len, post_body });
        try testSendAll(fd, w2.written());
        const r2 = try testRecvOneHttpResponse(fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r2, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualStrings("keep-alive", extractHeaderValue(r2, "Connection") orelse "");
        const j2 = extractHttpBody(r2) orelse return error.TestUnexpectedResult;
        try expectValidJson(j2);
        try std.testing.expect(std.mem.indexOf(u8, j2, "\"content\":\"tok\"") != null);
        // Single-token tpot_ms uses prefill_done_ns instead of 0.0!
        try std.testing.expect(std.mem.indexOf(u8, j2, "\"tpot_ms\":0.000") == null);

        // Request 3 + 4 pipelined in one send, with `Connection: close` on Request 4!
        try testSendAll(
            fd,
            "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n" ++
                "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        );
        // Read both pipelined responses back cleanly:
        const combined = try testRecvAll(fd, &resp_buf);
        try std.testing.expect(std.mem.indexOf(u8, combined, "\"object\":\"list\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, combined, "\"status\":\"ok\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, combined, "Connection: close\r\n") != null);
    }

    // 5. (l) Idle keep-alive socket reaped after `keepalive_idle_timeout_ms` without 408 or client-abort inflation.
    {
        const before_cancels = srv.stats().cancelled_count;
        const fd = try testConnectLoopback(port);
        defer _ = std.c.close(fd);
        try testSendAll(fd, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
        const r1 = try testRecvOneHttpResponse(fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r1, "HTTP/1.1 200 OK\r\n"));

        // Wait past `keepalive_idle_timeout_ms` (120 ms): server closes TCP cleanly with EOF (`recv == 0`), NOT 408!
        const eof_tail = try testRecvAll(fd, &resp_buf);
        try std.testing.expectEqual(@as(usize, 0), eof_tail.len);
        try std.testing.expectEqual(before_cancels, srv.stats().cancelled_count);
    }
}

test "C18 M8.1 hardening: Inkling2 recipient-header tool call, model_id-bound HMAC, and zero-output TTFT" {
    const gpa = std.testing.allocator;

    // 1. Model-bound HMAC-SHA256 prevents cross-model signature replay sharing `--api-key`.
    var key: [32]u8 = @splat(0x42);
    var sig_a: [64]u8 = undefined;
    const s_a = computeThinkingSignature(&key, "model-a", "chain of thought", &sig_a);
    try std.testing.expect(verifyThinkingSignature(&key, "model-a", "chain of thought", s_a));
    try std.testing.expect(!verifyThinkingSignature(&key, "model-b", "chain of thought", s_a));
    try std.testing.expect(!verifyThinkingSignature(&key, "model-a", "tampered", s_a));

    // HTTP-level cross-model replay rejection (shared api_key, different model_id -> 400).
    const EchoMock = struct {
        fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(4);
            _ = stream.emitTokenBytes("<think>secret thought</think>answer");
            stream.finish(.stop, 4);
        }
    };
    var dummy: u8 = 0;
    const srv_a = try Server.init(gpa, .{ .port = 0, .model_id = "model-a", .api_key = "shared-key", .apply_qos = false });
    defer srv_a.deinit(gpa);
    try srv_a.start(.{ .ctx = &dummy, .generateFn = EchoMock.generate });
    defer srv_a.stop();

    const srv_b = try Server.init(gpa, .{ .port = 0, .model_id = "model-b", .api_key = "shared-key", .apply_qos = false });
    defer srv_b.deinit(gpa);
    try srv_b.start(.{ .ctx = &dummy, .generateFn = EchoMock.generate });
    defer srv_b.stop();

    var req_buf: [4096]u8 = undefined;
    var resp_buf: [16 * 1024]u8 = undefined;

    // Turn 1 on model-a produces a valid model-a signature.
    const r_a = try testHttpPostJsonWithHeaders(
        srv_a.boundPort(),
        "/v1/messages",
        "x-api-key: shared-key\r\n",
        "{\"model\":\"model-a\",\"max_tokens\":32,\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":16},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        &req_buf,
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, r_a, "HTTP/1.1 200 OK\r\n"));
    const body_a = extractHttpBody(r_a) orelse return error.TestUnexpectedResult;
    const sig_marker = "\"signature\":\"";
    const sig_pos = (std.mem.indexOf(u8, body_a, sig_marker) orelse return error.TestUnexpectedResult) + sig_marker.len;
    const sig_from_a = body_a[sig_pos .. sig_pos + 64];
    var saved_sig: [64]u8 = undefined;
    @memcpy(&saved_sig, sig_from_a);

    var replay_body_buf: [1024]u8 = undefined;
    const replay_a_json = try std.fmt.bufPrint(
        &replay_body_buf,
        "{{\"model\":\"model-a\",\"max_tokens\":16,\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}},{{\"role\":\"assistant\",\"content\":[{{\"type\":\"thinking\",\"thinking\":\"secret thought\",\"signature\":\"{s}\"}},{{\"type\":\"text\",\"text\":\"answer\"}}]}},{{\"role\":\"user\",\"content\":\"next\"}}]}}",
        .{saved_sig[0..]},
    );
    const ok_replay = try testHttpPostJsonWithHeaders(
        srv_a.boundPort(),
        "/v1/messages",
        "x-api-key: shared-key\r\n",
        replay_a_json,
        &req_buf,
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, ok_replay, "HTTP/1.1 200 OK\r\n"));

    const replay_b_json = try std.fmt.bufPrint(
        &replay_body_buf,
        "{{\"model\":\"model-b\",\"max_tokens\":16,\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}},{{\"role\":\"assistant\",\"content\":[{{\"type\":\"thinking\",\"thinking\":\"secret thought\",\"signature\":\"{s}\"}},{{\"type\":\"text\",\"text\":\"answer\"}}]}},{{\"role\":\"user\",\"content\":\"next\"}}]}}",
        .{saved_sig[0..]},
    );
    const bad_replay = try testHttpPostJsonWithHeaders(
        srv_b.boundPort(),
        "/v1/messages",
        "x-api-key: shared-key\r\n",
        replay_b_json,
        &req_buf,
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, bad_replay, "HTTP/1.1 400 Bad Request\r\n"));

    // 2. Inkling 2 `<|message_model|>{tool_name}<|content_invoke_tool_json|>{args}` leaves `content:null`!
    const InkHeaderMock = struct {
        fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(6);
            const parts = [_][]const u8{
                "<|message_model|><|content_thinking|>check<|end_message|>",
                "<|message_model|>get_wea",
                "ther<|content_invoke_tool_json|>{\"city\":\"Osaka\"}<|end_message|><|content_model_end_sampling|>",
            };
            for (parts) |p| _ = stream.emitTokenBytes(p);
            stream.finish(.stop, 6);
        }
    };
    const srv_ink = try Server.init(gpa, .{
        .port = 0,
        .model_id = "inkling-2",
        .chat_template = chat_template.ChatTemplate.preset(.inkling2),
        .apply_qos = false,
    });
    defer srv_ink.deinit(gpa);
    try srv_ink.start(.{ .ctx = &dummy, .generateFn = InkHeaderMock.generate });
    defer srv_ink.stop();

    const r_ink = try testHttpPostJson(
        srv_ink.boundPort(),
        "/v1/chat/completions",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Osaka?\"}],\"stream\":false}",
        &req_buf,
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, r_ink, "HTTP/1.1 200 OK\r\n"));
    const j_ink = extractHttpBody(r_ink) orelse return error.TestUnexpectedResult;
    try expectValidJson(j_ink);
    try std.testing.expect(std.mem.indexOf(u8, j_ink, "\"content\":null") != null);
    try std.testing.expect(std.mem.indexOf(u8, j_ink, "\"reasoning_content\":\"check\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, j_ink, "\"name\":\"get_weather\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, j_ink, "Osaka") != null);

    // 3. `computeTimings` & live TTFT regression: immediate EOS, empty chunk, and stop-buffered zero visible output.
    const TimingMock = struct {
        fn generate(_: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(5);
            if (std.mem.indexOf(u8, job.prompt, "IMMEDIATE_EOS") != null) {
                // Sleep briefly so `end_ns - admit_ns > 5 ms`: TTFT must still be 0.000!
                var ts = std.c.timespec{ .sec = 0, .nsec = 10 * 1_000_000 };
                _ = std.c.nanosleep(&ts, null);
                stream.finish(.stop, 5);
                return;
            }
            if (std.mem.indexOf(u8, job.prompt, "STOP_ZERO_VIS") != null) {
                var ts = std.c.timespec{ .sec = 0, .nsec = 10 * 1_000_000 };
                _ = std.c.nanosleep(&ts, null);
                _ = stream.emitTokenBytes("HALT");
                stream.finish(.stop, 5);
                return;
            }
            // Empty chunk followed by 25ms pause then visible token: TTFT must reflect visible token (>= 20ms)!
            _ = stream.emitBytes("", 1);
            var ts = std.c.timespec{ .sec = 0, .nsec = 25 * 1_000_000 };
            _ = std.c.nanosleep(&ts, null);
            _ = stream.emitTokenBytes("hello");
            stream.finish(.stop, 5);
        }
    };
    const srv_t = try Server.init(gpa, .{ .port = 0, .model_id = "timing-test", .apply_qos = false });
    defer srv_t.deinit(gpa);
    try srv_t.start(.{ .ctx = &dummy, .generateFn = TimingMock.generate });
    defer srv_t.stop();

    // 3a. Immediate EOS (0 generated tokens): ttft_ms == 0.000, tpot_ms == 0.000
    const r_eos = try testHttpPostJson(
        srv_t.boundPort(),
        "/v1/chat/completions",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"IMMEDIATE_EOS\"}],\"stream\":false}",
        &req_buf,
        &resp_buf,
    );
    const j_eos = extractHttpBody(r_eos) orelse return error.TestUnexpectedResult;
    try expectValidJson(j_eos);
    try std.testing.expect(std.mem.indexOf(u8, j_eos, "\"ttft_ms\":0.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, j_eos, "\"tpot_ms\":0.000") != null);

    // 3b. Stop-sequence buffered zero visible output: ttft_ms == 0.000
    const r_stop = try testHttpPostJson(
        srv_t.boundPort(),
        "/v1/chat/completions",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"STOP_ZERO_VIS\"}],\"stop\":[\"HALT\"],\"stream\":false}",
        &req_buf,
        &resp_buf,
    );
    const j_stop = extractHttpBody(r_stop) orelse return error.TestUnexpectedResult;
    try expectValidJson(j_stop);
    try std.testing.expect(std.mem.indexOf(u8, j_stop, "\"ttft_ms\":0.000") != null);

    // 3c. Empty chunk before 25ms-delayed visible text: ttft_ms >= 20.0ms (NOT skewed by empty chunk!)
    const r_emp = try testHttpPostJson(
        srv_t.boundPort(),
        "/v1/chat/completions",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"EMPTY_THEN_TEXT\"}],\"stream\":false}",
        &req_buf,
        &resp_buf,
    );
    const j_emp = extractHttpBody(r_emp) orelse return error.TestUnexpectedResult;
    try expectValidJson(j_emp);
    const ttft_tag = "\"ttft_ms\":";
    const tpos = (std.mem.indexOf(u8, j_emp, ttft_tag) orelse return error.TestUnexpectedResult) + ttft_tag.len;
    const tend = std.mem.indexOfAnyPos(u8, j_emp, tpos, ",}") orelse return error.TestUnexpectedResult;
    const measured_ttft = try std.fmt.parseFloat(f64, j_emp[tpos..tend]);
    try std.testing.expect(measured_ttft >= 20.0);

    // 3d. Direct `computeTimings` unit check: `max_tokens == 0` / zero emitted bytes yields `ttft_ms == 0.0`.
    var zero_slot: ConnectionSlot = undefined;
    zero_slot.admit_ns = 1_000_000;
    zero_slot.prefill_done_ns = 12_000_000;
    zero_slot.first_emitted_ns = 0;
    zero_slot.first_generated_token_ns = 0;
    zero_slot.completion_tokens = 0;
    const zt = Server.computeTimings(&zero_slot, 25_000_000);
    try std.testing.expectEqual(@as(f64, 0.0), zt.ttft_ms);
    try std.testing.expectEqual(@as(f64, 0.0), zt.tpot_ms);
    try std.testing.expectEqual(@as(f64, 0.0), zt.tok_per_s);
}

test "C28 GET /v1/models (>4096 B) and GET /v1/models/{id} with LoRA adapters, percent-decoding, auth, and zero hot-path allocs" {
    const gpa = std.testing.allocator;
    const Mock = struct {
        fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(2);
            _ = stream.emitTokenBytes("ok");
            stream.finish(.stop, 2);
        }

        fn embed(_: *anyopaque, _: *const EmbedRequest, _: []f32) anyerror!EmbedResult {
            return .{ .dim = 4, .prompt_tokens = 1 };
        }

        fn rerank(_: *anyopaque, _: *const RerankRequest, _: []RerankItem) anyerror!RerankResult {
            return .{ .count = 0, .prompt_tokens = 1 };
        }
    };
    var dummy: u8 = 0;

    // 1. Unit tests for `percentDecodeUrlPath` and `ModelDetailMeta.fromInspection` and `LoraAdapterMeta.populateFromRegistry`.
    {
        var out: [64]u8 = undefined;
        try std.testing.expectEqualStrings("unsloth/inkling-2-mini", try percentDecodeUrlPath("unsloth%2Finkling-2-mini", &out));
        try std.testing.expectEqualStrings("unsloth/inkling-2-mini:sql-r8", try percentDecodeUrlPath("unsloth%2finkling-2-mini%3asql-r8", &out));
        try std.testing.expectEqualStrings("a b+c+d", try percentDecodeUrlPath("a%20b%2Bc+d", &out));
        try std.testing.expectError(error.InvalidPercentEncoding, percentDecodeUrlPath("bad%ZZid", &out));
        try std.testing.expectError(error.InvalidPercentEncoding, percentDecodeUrlPath("bad%00id", &out));
        try std.testing.expectError(error.InvalidPercentEncoding, percentDecodeUrlPath("trunc%2", &out));
        try std.testing.expectError(error.InvalidPercentEncoding, percentDecodeUrlPath("trunc%", &out));
        var tiny: [4]u8 = undefined;
        try std.testing.expectError(error.BufferTooSmall, percentDecodeUrlPath("abcde", &tiny));
        try std.testing.expectError(error.InvalidPercentEncoding, percentDecodeUrlPath("abcd%ZZ", &tiny));

        var insp = std.mem.zeroInit(amf.AmfInspection, .{
            .path = @as([]u8, &.{}),
            .cold_weight_bytes = 1234,
            .sparse_table_bytes = 566,
            .estimated_total_wired_bytes = 6400 * (1024 * 1024),
            .effective_bits_per_weight = 4.375,
        });
        insp.arch.setArch("qwen3_vl_moe");
        insp.arch.vocab_size = 151936;
        insp.arch.max_position_embeddings = 131072;
        insp.arch.has_moe = true;
        insp.arch.has_vision = true;

        const m_auto = ModelDetailMeta.fromInspection(&insp, 0, .{ .reasoning = true });
        try std.testing.expectEqualStrings("qwen3_vl_moe", m_auto.architecture);
        try std.testing.expectEqual(@as(u32, 131072), m_auto.max_model_len);
        try std.testing.expectEqual(@as(u32, 151936), m_auto.vocab_size);
        try std.testing.expectApproxEqAbs(@as(f64, 4.375), m_auto.effective_bits_per_weight, 1e-6);
        try std.testing.expectEqual(@as(u64, 6400), m_auto.wired_mib);
        try std.testing.expectEqual(@as(u64, 1800), m_auto.cold_bytes);
        try std.testing.expect(m_auto.capabilities.reasoning);
        try std.testing.expect(m_auto.capabilities.moe);
        try std.testing.expect(m_auto.capabilities.vision);

        const m_override = ModelDetailMeta.fromInspection(&insp, 32768, null);
        try std.testing.expectEqual(@as(u32, 32768), m_override.max_model_len);

        insp.arch.arch_len = 0;
        const m_unk = ModelDetailMeta.fromInspection(&insp, 0, null);
        try std.testing.expectEqualStrings("unknown", m_unk.architecture);

        var reg = try lora_mod.LoraRegistry.init(.{
            .pool_bytes = APPLE_PAGE_SIZE,
            .base_shape = .{ .num_layers = 1, .hidden_size = 64, .q_dim = 64, .kv_dim = 64, .intermediate_size = 128 },
        });
        defer reg.deinit(gpa);
        reg.slots[0].in_use = true;
        @memcpy(reg.slots[0].name[0..6], "sql-r8");
        reg.slots[0].name_len = 6;
        reg.slots[0].rank = 8;
        reg.slots[0].lora_alpha = 16.0;
        reg.slots[0].scale = 2.0;
        reg.slots[0].wired_bytes = 65536;
        reg.slots[3].in_use = true;
        @memcpy(reg.slots[3].name[0..7], "code-r4");
        reg.slots[3].name_len = 7;
        reg.slots[3].rank = 4;
        reg.slots[3].lora_alpha = 8.0;
        reg.slots[3].scale = 2.0;
        reg.slots[3].wired_bytes = 32768;

        var reg_out: [lora_mod.MAX_ADAPTERS]LoraAdapterMeta = undefined;
        const pop_n = LoraAdapterMeta.populateFromRegistry(&reg, &reg_out);
        try std.testing.expectEqual(@as(usize, 2), pop_n);
        try std.testing.expectEqualStrings("sql-r8", reg_out[0].name);
        try std.testing.expectEqual(@as(u32, 8), reg_out[0].rank);
        try std.testing.expectEqual(@as(u64, 65536), reg_out[0].wired_bytes);
        try std.testing.expectEqualStrings("code-r4", reg_out[1].name);
        try std.testing.expectEqual(@as(u32, 4), reg_out[1].rank);
    }

    // 2. Server.init validation checks for model_detail & lora_adapters.
    {
        const dup_adps = [_]LoraAdapterMeta{
            .{ .name = "dup", .rank = 8, .lora_alpha = 16.0, .scale = 2.0 },
            .{ .name = "dup", .rank = 16, .lora_alpha = 32.0, .scale = 2.0 },
        };
        try std.testing.expectError(error.InvalidConfig, Server.init(gpa, .{
            .port = 0,
            .model_id = "m",
            .lora_adapters = &dup_adps,
            .apply_qos = false,
        }));

        const zero_rank = [_]LoraAdapterMeta{
            .{ .name = "bad", .rank = 0, .lora_alpha = 16.0, .scale = 2.0 },
        };
        try std.testing.expectError(error.InvalidConfig, Server.init(gpa, .{
            .port = 0,
            .model_id = "m",
            .lora_adapters = &zero_rank,
            .apply_qos = false,
        }));

        const nan_scale = [_]LoraAdapterMeta{
            .{ .name = "nan", .rank = 8, .lora_alpha = 16.0, .scale = std.math.nan(f32) },
        };
        try std.testing.expectError(error.InvalidConfig, Server.init(gpa, .{
            .port = 0,
            .model_id = "m",
            .lora_adapters = &nan_scale,
            .apply_qos = false,
        }));

        const long_model: [100]u8 = @splat('a');
        const long_adp: [29]u8 = @splat('b');
        const too_long_adp = [_]LoraAdapterMeta{
            .{ .name = &long_adp, .rank = 8, .lora_alpha = 16.0, .scale = 2.0 },
        };
        try std.testing.expectError(error.InvalidConfig, Server.init(gpa, .{
            .port = 0,
            .model_id = &long_model,
            .lora_adapters = &too_long_adp,
            .apply_qos = false,
        }));

        var too_many: [lora_mod.MAX_ADAPTERS + 1]LoraAdapterMeta = undefined;
        for (&too_many, 0..) |*slot, idx| {
            slot.* = .{ .name = if (idx == 0) "a" else "b", .rank = 8, .lora_alpha = 16.0, .scale = 2.0 };
        }
        try std.testing.expectError(error.InvalidConfig, Server.init(gpa, .{
            .port = 0,
            .model_id = "m",
            .lora_adapters = &too_many,
            .apply_qos = false,
        }));
    }

    // 3. Full HTTP integration: 8 mounted LoRA adapters (> 4096 B response), URL percent-decoding,
    //    dangling-slice safety, live wired_mib, 405/404/400/401, and zero hot-path mallocs!
    var arch_tmp: [20]u8 = undefined;
    @memcpy(&arch_tmp, "qwen3_next_moe_coder");
    var adp_names_tmp: [8][32]u8 = undefined;
    var adps_tmp: [8]LoraAdapterMeta = undefined;
    const adp_suffixes = [_][]const u8{
        "sql-r8",
        "code-python-r16",
        "rust-systems-r8",
        "zig-metal-r16",
        "finance-qa-r32",
        "medical-en-r16",
        "legal-zh-r8",
        "math-proof-r16",
    };
    for (adp_suffixes, 0..) |nm, idx| {
        const len = nm.len;
        @memcpy(adp_names_tmp[idx][0..len], nm);
        adps_tmp[idx] = .{
            .name = adp_names_tmp[idx][0..len],
            .rank = @intCast(8 * ((idx % 3) + 1)),
            .lora_alpha = @as(f32, @floatFromInt(16 * ((idx % 3) + 1))),
            .scale = 2.0 + @as(f32, @floatFromInt(idx)) * 0.25,
            .wired_bytes = @as(u64, 65536) * (idx + 1),
        };
    }

    const srv = try Server.init(gpa, .{
        .port = 0,
        .model_id = "unsloth/inkling-2-mini",
        .wired_mib = 0,
        .api_key = "sk-c28-test",
        .embedding_handler = .{ .ctx = &dummy, .embedFn = Mock.embed },
        .rerank_handler = .{ .ctx = &dummy, .rerankFn = Mock.rerank },
        .model_detail = .{
            .architecture = &arch_tmp,
            .max_model_len = 65536,
            .vocab_size = 151936,
            .effective_bits_per_weight = 4.25,
            .wired_mib = 4096,
            .cold_bytes = 987654321,
            .capabilities = .{
                .reasoning = true,
                .vision = true,
                .speculative_decoding = true,
                .moe = true,
            },
        },
        .lora_adapters = &adps_tmp,
        .max_conns = 8,
        .max_active_requests = 4,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);

    // Poison caller stack buffers to prove `Server.init` owns copies with zero dangling slices!
    @memset(&arch_tmp, 'X');
    for (&adp_names_tmp) |*buf| @memset(buf, 'Z');

    try srv.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var resp_buf: [32768]u8 = undefined;

    const ModelCapsJson = struct {
        chat: bool,
        responses: bool,
        messages: bool,
        tool_calling: bool,
        structured_outputs: bool,
        reasoning: bool,
        vision: bool,
        embeddings: bool,
        rerank: bool,
        lora: bool,
        speculative_decoding: bool,
        moe: bool,
    };
    const ModelEntryJson = struct {
        id: []const u8,
        object: []const u8,
        type: []const u8,
        display_name: []const u8,
        created: i64,
        created_at: []const u8,
        owned_by: []const u8,
        root: []const u8,
        parent: ?[]const u8 = null,
        architecture: []const u8,
        max_model_len: u32,
        max_input_tokens: u32,
        max_tokens: u32,
        vocab_size: u32,
        effective_bits_per_weight: ?f64 = null,
        wired_mib: ?u64 = null,
        cold_bytes: ?u64 = null,
        rank: ?u32 = null,
        lora_alpha: ?f64 = null,
        scale: ?f64 = null,
        wired_bytes: ?u64 = null,
        capabilities: ModelCapsJson,
    };
    const ModelsListJson = struct {
        object: []const u8,
        data: []const ModelEntryJson,
        first_id: ?[]const u8 = null,
        last_id: ?[]const u8 = null,
        has_more: bool,
    };

    // 3a. GET /v1/models (> 4096 B payload parsed via std.json.parseFromSlice).
    {
        const resp = try testHttpExchange(
            port,
            "GET /v1/models?page=1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, resp, "HTTP/1.1 200 OK\r\n"));
        const body = extractHttpBody(resp) orelse return error.TestUnexpectedResult;
        try std.testing.expect(body.len > 4096);

        const parsed = try std.json.parseFromSlice(ModelsListJson, gpa, body, .{});
        defer parsed.deinit();
        try std.testing.expectEqualStrings("list", parsed.value.object);
        try std.testing.expectEqual(@as(usize, 9), parsed.value.data.len);

        const base = parsed.value.data[0];
        try std.testing.expectEqualStrings("unsloth/inkling-2-mini", base.id);
        try std.testing.expectEqualStrings("model", base.object);
        try std.testing.expectEqualStrings("aminfer", base.owned_by);
        try std.testing.expectEqualStrings("unsloth/inkling-2-mini", base.root);
        try std.testing.expectEqual(@as(?[]const u8, null), base.parent);
        try std.testing.expectEqualStrings("qwen3_next_moe_coder", base.architecture);
        try std.testing.expectEqual(@as(u32, 65536), base.max_model_len);
        try std.testing.expectEqual(@as(u32, 151936), base.vocab_size);
        try std.testing.expectApproxEqAbs(@as(f64, 4.25), base.effective_bits_per_weight.?, 1e-6);
        try std.testing.expectEqual(@as(u64, 4096), base.wired_mib.?);
        try std.testing.expectEqual(@as(u64, 987654321), base.cold_bytes.?);
        try std.testing.expect(base.capabilities.chat);
        try std.testing.expect(base.capabilities.responses);
        try std.testing.expect(base.capabilities.messages);
        try std.testing.expect(base.capabilities.tool_calling);
        try std.testing.expect(base.capabilities.structured_outputs);
        try std.testing.expect(base.capabilities.reasoning);
        try std.testing.expect(base.capabilities.vision);
        try std.testing.expect(base.capabilities.embeddings);
        try std.testing.expect(base.capabilities.rerank);
        try std.testing.expect(base.capabilities.lora);
        try std.testing.expect(base.capabilities.speculative_decoding);
        try std.testing.expect(base.capabilities.moe);

        for (adp_suffixes, 0..) |nm, idx| {
            const item = parsed.value.data[idx + 1];
            var exp_id_buf: [128]u8 = undefined;
            const exp_id = try std.fmt.bufPrint(&exp_id_buf, "unsloth/inkling-2-mini:{s}", .{nm});
            try std.testing.expectEqualStrings(exp_id, item.id);
            try std.testing.expectEqualStrings("model", item.object);
            try std.testing.expectEqualStrings("aminfer", item.owned_by);
            try std.testing.expectEqualStrings("unsloth/inkling-2-mini", item.root);
            try std.testing.expectEqualStrings("unsloth/inkling-2-mini", item.parent.?);
            try std.testing.expectEqualStrings("qwen3_next_moe_coder", item.architecture);
            try std.testing.expectEqual(@as(u32, 65536), item.max_model_len);
            try std.testing.expectEqual(@as(u32, 151936), item.vocab_size);
            try std.testing.expectEqual(adps_tmp[idx].rank, item.rank.?);
            try std.testing.expectApproxEqAbs(@as(f64, adps_tmp[idx].lora_alpha), item.lora_alpha.?, 1e-5);
            try std.testing.expectApproxEqAbs(@as(f64, adps_tmp[idx].scale), item.scale.?, 1e-5);
            try std.testing.expectEqual(adps_tmp[idx].wired_bytes, item.wired_bytes.?);
            try std.testing.expect(item.capabilities.lora);
        }
    }

    // Live wired_mib update overrides meta.wired_mib when > 0.
    srv.setWiredMib(8192);

    // 3b. GET /v1/models/{id} with uppercase & lowercase %2F / %3A.
    {
        const r_base_upper = try testHttpExchange(
            port,
            "GET /v1/models/unsloth%2Finkling-2-mini HTTP/1.1\r\nHost: localhost\r\nx-api-key: sk-c28-test\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_base_upper, "HTTP/1.1 200 OK\r\n"));
        const p_base = try std.json.parseFromSlice(ModelEntryJson, gpa, extractHttpBody(r_base_upper).?, .{});
        defer p_base.deinit();
        try std.testing.expectEqualStrings("unsloth/inkling-2-mini", p_base.value.id);
        try std.testing.expectEqual(@as(u64, 8192), p_base.value.wired_mib.?);
        try std.testing.expectEqual(@as(?[]const u8, null), p_base.value.parent);

        const r_lora_upper = try testHttpExchange(
            port,
            "GET /v1/models/unsloth%2Finkling-2-mini%3Asql-r8 HTTP/1.1\r\nHost: localhost\r\nx-api-key: sk-c28-test\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_lora_upper, "HTTP/1.1 200 OK\r\n"));
        const p_lora_u = try std.json.parseFromSlice(ModelEntryJson, gpa, extractHttpBody(r_lora_upper).?, .{});
        defer p_lora_u.deinit();
        try std.testing.expectEqualStrings("unsloth/inkling-2-mini:sql-r8", p_lora_u.value.id);
        try std.testing.expectEqualStrings("unsloth/inkling-2-mini", p_lora_u.value.parent.?);
        try std.testing.expectEqual(@as(u32, 8), p_lora_u.value.rank.?);

        const r_lora_lower = try testHttpExchange(
            port,
            "GET /v1/models/unsloth%2finkling-2-mini%3asql-r8?detail=full HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_lora_lower, "HTTP/1.1 200 OK\r\n"));
        const p_lora_l = try std.json.parseFromSlice(ModelEntryJson, gpa, extractHttpBody(r_lora_lower).?, .{});
        defer p_lora_l.deinit();
        try std.testing.expectEqualStrings("unsloth/inkling-2-mini:sql-r8", p_lora_l.value.id);
    }

    // 3c. 405 Method Not Allowed with Allow: GET (checked BEFORE lookup).
    {
        const r_post_id = try testHttpExchange(
            port,
            "POST /v1/models/unsloth%2Finkling-2-mini HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\nContent-Length: 0\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_post_id, "HTTP/1.1 405 Method Not Allowed\r\n"));
        try std.testing.expectEqualStrings("GET", extractHeaderValue(r_post_id, "Allow") orelse "");
        try std.testing.expect(std.mem.indexOf(u8, extractHttpBody(r_post_id).?, "\"code\":\"method_not_allowed\"") != null);

        const r_del = try testHttpExchange(
            port,
            "DELETE /v1/models/foo HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_del, "HTTP/1.1 405 Method Not Allowed\r\n"));
        try std.testing.expectEqualStrings("GET", extractHeaderValue(r_del, "Allow") orelse "");
    }

    // 3d. 404 Not Found with code="model_not_found".
    {
        const r_miss = try testHttpExchange(
            port,
            "GET /v1/models/nonexistent HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_miss, "HTTP/1.1 404 Not Found\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, extractHttpBody(r_miss).?, "\"code\":\"model_not_found\"") != null);

        const r_miss_adp = try testHttpExchange(
            port,
            "GET /v1/models/unsloth%2Finkling-2-mini%3Aunknown HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_miss_adp, "HTTP/1.1 404 Not Found\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, extractHttpBody(r_miss_adp).?, "\"code\":\"model_not_found\"") != null);

        const r_empty = try testHttpExchange(
            port,
            "GET /v1/models/ HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_empty, "HTTP/1.1 404 Not Found\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, extractHttpBody(r_empty).?, "\"code\":\"model_not_found\"") != null);

        const over_x: [140]u8 = @splat('x');
        var over_req_buf: [256]u8 = undefined;
        const over_req = try std.fmt.bufPrint(
            &over_req_buf,
            "GET /v1/models/{s} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n",
            .{over_x[0..]},
        );
        const r_over_buf = try testHttpExchange(port, over_req, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_over_buf, "HTTP/1.1 404 Not Found\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, extractHttpBody(r_over_buf).?, "\"code\":\"model_not_found\"") != null);
    }

    // 3e. 400 Bad Request for malformed percent-encoding (%ZZ / %00) & 401 Unauthorized without API key.
    {
        const r_zz = try testHttpExchange(
            port,
            "GET /v1/models/bad%ZZid HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_zz, "HTTP/1.1 400 Bad Request\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, extractHttpBody(r_zz).?, "\"code\":\"invalid_model_id\"") != null);

        const r_nul = try testHttpExchange(
            port,
            "GET /v1/models/bad%00id HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_nul, "HTTP/1.1 400 Bad Request\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, extractHttpBody(r_nul).?, "\"code\":\"invalid_model_id\"") != null);

        const r_401 = try testHttpExchange(
            port,
            "GET /v1/models/unsloth%2Finkling-2-mini HTTP/1.1\r\nHost: localhost\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_401, "HTTP/1.1 401 Unauthorized\r\n"));
    }

    // 3f. Zero hot-path heap allocations across GET /v1/models and GET /v1/models/{id}.
    {
        _ = try testHttpExchange(port, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n", &resp_buf);
        _ = try testHttpExchange(port, "GET /v1/models/unsloth%2Finkling-2-mini HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n", &resp_buf);
        _ = try testHttpExchange(port, "GET /v1/models/unsloth%2Finkling-2-mini%3Asql-r8 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n", &resp_buf);
        while (srv.stats().active_conns > 0 or srv.stats().active_requests > 0) {
            sleepNs(500_000);
        }

        var m_before: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &m_before);

        for (0..8) |_| {
            const rl = try testHttpExchange(port, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n", &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, rl, "HTTP/1.1 200 OK\r\n"));
            const rb = try testHttpExchange(port, "GET /v1/models/unsloth%2Finkling-2-mini HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n", &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, rb, "HTTP/1.1 200 OK\r\n"));
            const ra = try testHttpExchange(port, "GET /v1/models/unsloth%2Finkling-2-mini%3Asql-r8 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c28-test\r\n\r\n", &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, ra, "HTTP/1.1 200 OK\r\n"));
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

fn testRunCmdChecked(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !void {
    const res = try std.process.run(allocator, io, .{
        .argv = argv,
        .stdout_limit = .limited(64 * 1024),
        .stderr_limit = .limited(64 * 1024),
    });
    defer allocator.free(res.stdout);
    defer allocator.free(res.stderr);
    switch (res.term) {
        .exited => |code| if (code != 0) return error.SubprocessFailed,
        else => return error.SubprocessFailed,
    }
}

fn testNbSendAll(fd: std.c.fd_t, data: []const u8) TestClientError!void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.send(fd, data.ptr + sent, data.len - sent, 0);
        if (n > 0) {
            sent += @intCast(n);
            continue;
        }
        if (n < 0) {
            const err = std.c.errno(n);
            if (err == .INTR) continue;
            if (err == .AGAIN) {
                var pfd = [1]std.c.pollfd{.{ .fd = fd, .events = std.c.POLL.OUT, .revents = 0 }};
                if (std.c.poll(&pfd, 1, 5000) <= 0) return error.SendFailed;
                continue;
            }
        }
        return error.SendFailed;
    }
}

fn testNbRecvAll(fd: std.c.fd_t, out_buf: []u8) TestClientError![]const u8 {
    var total: usize = 0;
    while (true) {
        if (total >= out_buf.len) return error.BufferTooSmall;
        const n = std.c.recv(fd, out_buf.ptr + total, out_buf.len - total, 0);
        if (n > 0) {
            total += @intCast(n);
            continue;
        }
        if (n == 0) return out_buf[0..total];
        const err = std.c.errno(n);
        if (err == .INTR) continue;
        if (err == .AGAIN) {
            var pfd = [1]std.c.pollfd{.{ .fd = fd, .events = std.c.POLL.IN, .revents = 0 }};
            if (std.c.poll(&pfd, 1, 5000) <= 0) return error.RecvFailed;
            continue;
        }
        return error.RecvFailed;
    }
}

fn testNbRecvOneHttpResponse(fd: std.c.fd_t, out_buf: []u8) TestClientError![]const u8 {
    var total: usize = 0;
    var expected_end: ?usize = null;
    while (true) {
        if (expected_end) |ee| {
            if (total >= ee) return out_buf[0..ee];
        }
        if (total >= out_buf.len) return error.BufferTooSmall;
        const n = std.c.recv(fd, out_buf.ptr + total, out_buf.len - total, 0);
        if (n > 0) {
            total += @intCast(n);
            if (expected_end == null) {
                if (std.mem.indexOf(u8, out_buf[0..total], "\r\n\r\n")) |sep| {
                    const hdr_end = sep + 4;
                    if (extractHeaderValue(out_buf[0..hdr_end], "Content-Length")) |cl_str| {
                        const cl = std.fmt.parseInt(usize, cl_str, 10) catch return error.RecvFailed;
                        expected_end = hdr_end + cl;
                        if (expected_end.? > out_buf.len) return error.BufferTooSmall;
                    }
                }
            }
            continue;
        }
        if (n == 0) return out_buf[0..total];
        const err = std.c.errno(n);
        if (err == .INTR) continue;
        if (err == .AGAIN) {
            var pfd = [1]std.c.pollfd{.{ .fd = fd, .events = std.c.POLL.IN, .revents = 0 }};
            if (std.c.poll(&pfd, 1, 5000) <= 0) return error.RecvFailed;
            continue;
        }
        return error.RecvFailed;
    }
}

fn testTlsHttpExchange(
    port: u16,
    min_ver: u16,
    max_ver: u16,
    raw_request: []const u8,
    resp_buf: []u8,
) !struct { resp: []const u8, tls_version: u16 } {
    const cc = try tls_mod.clientConnectSocketpair(.{ 127, 0, 0, 1 }, port, min_ver, max_ver);
    defer _ = std.c.close(cc.fd);
    try testNbSendAll(cc.fd, raw_request);
    const resp = try testNbRecvOneHttpResponse(cc.fd, resp_buf);
    return .{ .resp = resp, .tls_version = cc.tls_version };
}

fn testTlsHttpSsePost(
    port: u16,
    min_ver: u16,
    max_ver: u16,
    path: []const u8,
    extra_headers: []const u8,
    json_body: []const u8,
    req_buf: []u8,
    resp_buf: []u8,
) !struct { resp: []const u8, tls_version: u16 } {
    var w = FixedWriter.init(req_buf);
    w.print(
        "POST {s} HTTP/1.1\r\nHost: 127.0.0.1\r\n{s}Content-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
        .{ path, extra_headers, json_body.len, json_body },
    ) catch return error.BufferTooSmall;
    const cc = try tls_mod.clientConnectSocketpair(.{ 127, 0, 0, 1 }, port, min_ver, max_ver);
    defer _ = std.c.close(cc.fd);
    try testNbSendAll(cc.fd, w.written());
    const resp = try testNbRecvAll(cc.fd, resp_buf);
    return .{ .resp = resp, .tls_version = cc.tls_version };
}

test "M8.3 HTTPS Server: TLS 1.3/1.2, encrypted PEM & .p12, C28 /v1/models/{id}, SSE, keep-alive, AF_UNIX, 503 saturation, and zero hot-path mallocs" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var threaded_io: std.Io.Threaded = .init(gpa, .{});
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);

    const ec_key = try std.fmt.allocPrint(gpa, "{s}/ec.key.pem", .{dir});
    defer gpa.free(ec_key);
    const ec_cert = try std.fmt.allocPrint(gpa, "{s}/ec.cert.pem", .{dir});
    defer gpa.free(ec_cert);
    const rsa_key = try std.fmt.allocPrint(gpa, "{s}/rsa.key.pem", .{dir});
    defer gpa.free(rsa_key);
    const rsa_cert = try std.fmt.allocPrint(gpa, "{s}/rsa.cert.pem", .{dir});
    defer gpa.free(rsa_cert);
    const rsa_enc_key = try std.fmt.allocPrint(gpa, "{s}/rsa.enc.key.pem", .{dir});
    defer gpa.free(rsa_enc_key);
    const ec_p12 = try std.fmt.allocPrint(gpa, "{s}/ec.p12", .{dir});
    defer gpa.free(ec_p12);
    const uds_path = try std.fmt.allocPrint(gpa, "/tmp/aminfer_m83_tls_{d}.sock", .{std.c.getpid()});
    defer gpa.free(uds_path);

    try testRunCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", ec_key,
    });
    try testRunCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "req", "-x509", "-new", "-key", ec_key, "-subj", "/CN=127.0.0.1", "-days", "1", "-out", ec_cert,
    });
    try testRunCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "req",           "-x509", "-newkey", "rsa:2048", "-nodes",
        "-subj",            "/CN=127.0.0.1", "-days", "1",       "-keyout",  rsa_key,
        "-out",             rsa_cert,
    });
    try testRunCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "pkcs8", "-topk8", "-v2", "aes256", "-in", rsa_key, "-passout", "pass:m83-p8-pass", "-out", rsa_enc_key,
    });
    try testRunCmdChecked(gpa, io, &.{
        "/usr/bin/openssl", "pkcs12", "-export", "-in", ec_cert, "-inkey", ec_key, "-passout", "pass:m83-p12-pass", "-out", ec_p12,
    });

    const Mock = struct {
        fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(6);
            if (!stream.emitText("<think>plan</think>Hello TLS!", 4)) return;
            stream.finish(.stop, 6);
        }
    };
    var dummy: u8 = 0;
    var req_buf: [4096]u8 = undefined;
    var resp_buf: [32768]u8 = undefined;

    // 1. Encrypted PKCS#8 PEM & .p12 error propagation + successful TLS 1.3 handshake via Server.
    {
        var enc_files = try tls_mod.loadFiles(gpa, rsa_cert, rsa_enc_key);
        defer enc_files.deinit(gpa);

        try std.testing.expectError(error.TlsPassphraseRequired, Server.init(gpa, .{
            .port = 0,
            .tls = .{ .cert_bytes = enc_files.cert_bytes, .key_bytes = enc_files.key_bytes, .passphrase = null },
            .apply_qos = false,
        }));
        try std.testing.expectError(error.TlsIdentityInvalid, Server.init(gpa, .{
            .port = 0,
            .tls = .{ .cert_bytes = enc_files.cert_bytes, .key_bytes = enc_files.key_bytes, .passphrase = "wrong-pass" },
            .apply_qos = false,
        }));

        const srv_p8 = try Server.init(gpa, .{
            .port = 0,
            .max_conns = 4,
            .max_active_requests = 2,
            .tls = .{ .cert_bytes = enc_files.cert_bytes, .key_bytes = enc_files.key_bytes, .passphrase = "m83-p8-pass" },
            .apply_qos = false,
        });
        defer srv_p8.deinit(gpa);
        try std.testing.expect(srv_p8.isTlsEnabled());
        try srv_p8.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
        defer srv_p8.stop();

        const ex = try testTlsHttpExchange(
            srv_p8.boundPort(),
            tls_mod.TLS_VERSION_1_3,
            tls_mod.TLS_VERSION_1_3,
            "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expectEqual(tls_mod.TLS_VERSION_1_3, ex.tls_version);
        try std.testing.expect(std.mem.startsWith(u8, ex.resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, ex.resp, "\"status\":\"ok\"") != null);
    }

    {
        var p12_files = try tls_mod.loadFiles(gpa, ec_p12, null);
        defer p12_files.deinit(gpa);

        try std.testing.expectError(error.TlsPassphraseRequired, Server.init(gpa, .{
            .port = 0,
            .tls = .{ .cert_bytes = p12_files.cert_bytes, .key_bytes = null, .passphrase = null },
            .apply_qos = false,
        }));
        try std.testing.expectError(error.TlsIdentityInvalid, Server.init(gpa, .{
            .port = 0,
            .tls = .{ .cert_bytes = p12_files.cert_bytes, .key_bytes = null, .passphrase = "bad-p12" },
            .apply_qos = false,
        }));

        const srv_p12 = try Server.init(gpa, .{
            .port = 0,
            .max_conns = 4,
            .max_active_requests = 2,
            .tls = .{ .cert_bytes = p12_files.cert_bytes, .key_bytes = null, .passphrase = "m83-p12-pass" },
            .apply_qos = false,
        });
        defer srv_p12.deinit(gpa);
        try std.testing.expect(srv_p12.isTlsEnabled());
        try srv_p12.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
        defer srv_p12.stop();

        const ex = try testTlsHttpExchange(
            srv_p12.boundPort(),
            tls_mod.TLS_VERSION_1_3,
            tls_mod.TLS_VERSION_1_3,
            "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expectEqual(tls_mod.TLS_VERSION_1_3, ex.tls_version);
        try std.testing.expect(std.mem.startsWith(u8, ex.resp, "HTTP/1.1 200 OK\r\n"));
    }

    // 2. Full EC P-256 HTTPS Server + api_key + AF_UNIX + model_detail + 8 LoRA adapters.
    var ec_files = try tls_mod.loadFiles(gpa, ec_cert, ec_key);
    defer ec_files.deinit(gpa);

    const adp_suffixes = [_][]const u8{
        "sql-r8",         "code-python-r16", "rust-systems-r8", "zig-metal-r16",
        "finance-qa-r32", "medical-en-r16",  "legal-zh-r8",     "math-proof-r16",
    };
    var adps: [8]LoraAdapterMeta = undefined;
    for (adp_suffixes, 0..) |nm, idx| {
        adps[idx] = .{
            .name = nm,
            .rank = 8,
            .lora_alpha = 16.0,
            .scale = 2.0,
            .wired_bytes = @as(u64, 65536) * (idx + 1),
        };
    }

    const srv = try Server.init(gpa, .{
        .port = 0,
        .model_id = "unsloth/inkling-2-mini",
        .wired_mib = 4096,
        .api_key = "sk-m83-https",
        .unix_socket_path = uds_path,
        .model_detail = .{
            .architecture = "inkling2_moe",
            .max_model_len = 65536,
            .vocab_size = 151936,
            .effective_bits_per_weight = 4.25,
            .wired_mib = 4096,
            .cold_bytes = 0,
            .capabilities = .{ .reasoning = true, .moe = true, .lora = true },
        },
        .lora_adapters = &adps,
        .tls = .{ .cert_bytes = ec_files.cert_bytes, .key_bytes = ec_files.key_bytes },
        .max_conns = 8,
        .max_active_requests = 4,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try std.testing.expect(srv.isTlsEnabled());
    try std.testing.expectEqualStrings(uds_path, srv.unixSocketPath().?);
    try srv.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
    defer srv.stop();

    const port = srv.boundPort();

    // 2a. TLS 1.3 (0x0304) and forced TLS 1.2 (0x0303) route matrix.
    for ([_]u16{ tls_mod.TLS_VERSION_1_3, tls_mod.TLS_VERSION_1_2 }) |ver| {
        const r_health = try testTlsHttpExchange(
            port,
            ver,
            ver,
            "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expectEqual(ver, r_health.tls_version);
        try std.testing.expect(std.mem.startsWith(u8, r_health.resp, "HTTP/1.1 200 OK\r\n"));

        const r_models = try testTlsHttpExchange(
            port,
            ver,
            ver,
            "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-m83-https\r\nConnection: close\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expectEqual(ver, r_models.tls_version);
        try std.testing.expect(std.mem.startsWith(u8, r_models.resp, "HTTP/1.1 200 OK\r\n"));
        const body_models = extractHttpBody(r_models.resp) orelse return error.TestUnexpectedResult;
        try std.testing.expect(body_models.len > 4096);
        try expectValidJson(body_models);

        const r_base = try testTlsHttpExchange(
            port,
            ver,
            ver,
            "GET /v1/models/unsloth%2Finkling-2-mini HTTP/1.1\r\nHost: 127.0.0.1\r\nx-api-key: sk-m83-https\r\nConnection: close\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_base.resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, r_base.resp, "\"id\":\"unsloth/inkling-2-mini\"") != null);

        const r_adp = try testTlsHttpExchange(
            port,
            ver,
            ver,
            "GET /v1/models/unsloth%2Finkling-2-mini%3Asql-r8 HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-m83-https\r\nConnection: close\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_adp.resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, r_adp.resp, "\"id\":\"unsloth/inkling-2-mini:sql-r8\"") != null);

        const r_405 = try testTlsHttpExchange(
            port,
            ver,
            ver,
            "POST /v1/models/unsloth%2Finkling-2-mini HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-m83-https\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_405.resp, "HTTP/1.1 405 Method Not Allowed\r\n"));

        const r_404 = try testTlsHttpExchange(
            port,
            ver,
            ver,
            "GET /v1/models/unsloth%2Finkling-2-mini%3Anope HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-m83-https\r\nConnection: close\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_404.resp, "HTTP/1.1 404 Not Found\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, r_404.resp, "\"code\":\"model_not_found\"") != null);

        const r_401_missing = try testTlsHttpExchange(
            port,
            ver,
            ver,
            "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_401_missing.resp, "HTTP/1.1 401 Unauthorized\r\n"));

        const r_401_wrong = try testTlsHttpExchange(
            port,
            ver,
            ver,
            "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-wrong\r\nConnection: close\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_401_wrong.resp, "HTTP/1.1 401 Unauthorized\r\n"));
    }

    // 2b. Streaming SSE over HTTPS: /v1/chat/completions, /v1/responses, /v1/messages.
    {
        const sse_chat = try testTlsHttpSsePost(
            port,
            tls_mod.TLS_VERSION_1_3,
            tls_mod.TLS_VERSION_1_3,
            "/v1/chat/completions",
            "Authorization: Bearer sk-m83-https\r\n",
            "{\"model\":\"unsloth/inkling-2-mini\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, sse_chat.resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, sse_chat.resp, "text/event-stream") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse_chat.resp, "data: [DONE]") != null);
        _ = try expectValidSseJsonFrames(extractHttpBody(sse_chat.resp).?);

        const sse_resp = try testTlsHttpSsePost(
            port,
            tls_mod.TLS_VERSION_1_3,
            tls_mod.TLS_VERSION_1_3,
            "/v1/responses",
            "Authorization: Bearer sk-m83-https\r\n",
            "{\"model\":\"unsloth/inkling-2-mini\",\"stream\":true,\"input\":\"hi\"}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, sse_resp.resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, sse_resp.resp, "event: response.completed") != null);
        _ = try expectValidSseJsonFrames(extractHttpBody(sse_resp.resp).?);

        const sse_msg = try testTlsHttpSsePost(
            port,
            tls_mod.TLS_VERSION_1_3,
            tls_mod.TLS_VERSION_1_3,
            "/v1/messages",
            "x-api-key: sk-m83-https\r\nanthropic-version: 2023-06-01\r\n",
            "{\"model\":\"unsloth/inkling-2-mini\",\"max_tokens\":32,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, sse_msg.resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, sse_msg.resp, "event: message_stop") != null);
        _ = try expectValidSseJsonFrames(extractHttpBody(sse_msg.resp).?);
    }

    // 2c. Keep-alive multi-turn reuse over one TLS 1.3 connection + concurrent AF_UNIX plain HTTP
    //     + zero steady-state hot-path heap growth check.
    {
        const ka = try tls_mod.clientConnectSocketpair(.{ 127, 0, 0, 1 }, port, tls_mod.TLS_VERSION_1_3, tls_mod.TLS_VERSION_1_3);
        defer _ = std.c.close(ka.fd);
        try std.testing.expectEqual(tls_mod.TLS_VERSION_1_3, ka.tls_version);

        try testNbSendAll(ka.fd, "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n");
        const ka1 = try testNbRecvOneHttpResponse(ka.fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, ka1, "HTTP/1.1 200 OK\r\n"));

        // Concurrent plain HTTP over AF_UNIX while the TLS keep-alive slot is open.
        {
            const ufd = try testConnectUnix(uds_path);
            defer _ = std.c.close(ufd);
            try testSendAll(ufd, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-m83-https\r\nConnection: close\r\n\r\n");
            const uresp = try testRecvOneHttpResponse(ufd, &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, uresp, "HTTP/1.1 200 OK\r\n"));
            try std.testing.expect(extractHttpBody(uresp).?.len > 4096);
        }

        // Reuse the SAME TLS keep-alive fd for /v1/models/{id} and POST /v1/chat/completions.
        try testNbSendAll(ka.fd, "GET /v1/models/unsloth%2Finkling-2-mini%3Asql-r8 HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-m83-https\r\nConnection: keep-alive\r\n\r\n");
        const ka2 = try testNbRecvOneHttpResponse(ka.fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, ka2, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, ka2, "\"id\":\"unsloth/inkling-2-mini:sql-r8\"") != null);

        const chat_body = "{\"model\":\"unsloth/inkling-2-mini\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}";
        var w = FixedWriter.init(&req_buf);
        try w.print(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-m83-https\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n{s}",
            .{ chat_body.len, chat_body },
        );
        try testNbSendAll(ka.fd, w.written());
        const ka3 = try testNbRecvOneHttpResponse(ka.fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, ka3, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, ka3, "Hello TLS!") != null);

        // Wait for earlier connections to drain, warm up steady-state TLS keep-alive path,
        // then verify 0 net hot-path heap growth (up to 3 full passes to absorb any one-time
        // macOS GCD workqueue thread creation; a real per-request leak fails every pass).
        while (srv.stats().active_conns > 1 or srv.stats().active_requests > 0) {
            sleepNs(500_000);
        }
        const ka_req = "GET /v1/models/unsloth%2Finkling-2-mini HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-m83-https\r\nConnection: keep-alive\r\n\r\n";
        for (0..12) |_| {
            try testNbSendAll(ka.fd, ka_req);
            const rw = try testNbRecvOneHttpResponse(ka.fd, &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, rw, "HTTP/1.1 200 OK\r\n"));
        }
        sleepNs(20 * std.time.ns_per_ms);

        var zero_growth_verified = false;
        var pass_idx: usize = 0;
        while (pass_idx < 3) : (pass_idx += 1) {
            var m_before: MallocStatistics = std.mem.zeroes(MallocStatistics);
            malloc_zone_statistics(null, &m_before);
            for (0..12) |_| {
                try testNbSendAll(ka.fd, ka_req);
                const r = try testNbRecvOneHttpResponse(ka.fd, &resp_buf);
                try std.testing.expect(std.mem.startsWith(u8, r, "HTTP/1.1 200 OK\r\n"));
            }

            var m_after: MallocStatistics = std.mem.zeroes(MallocStatistics);
            var settle_iter: usize = 0;
            while (settle_iter < 20) : (settle_iter += 1) {
                sleepNs(10 * std.time.ns_per_ms);
                malloc_zone_statistics(null, &m_after);
                if (m_after.blocks_in_use <= m_before.blocks_in_use) {
                    zero_growth_verified = true;
                    break;
                }
            }
            if (zero_growth_verified) break;
        }
        try std.testing.expect(zero_growth_verified);
    }

    // 3. 503 pool saturation over HTTPS delivers the complete STATIC_503_RESPONSE without truncation.
    {
        const srv_sat = try Server.init(gpa, .{
            .port = 0,
            .max_conns = 1,
            .max_active_requests = 1,
            .tls = .{ .cert_bytes = ec_files.cert_bytes, .key_bytes = ec_files.key_bytes },
            .apply_qos = false,
        });
        defer srv_sat.deinit(gpa);
        try srv_sat.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
        defer srv_sat.stop();

        const sat_port = srv_sat.boundPort();
        const holder = try tls_mod.clientConnectSocketpair(.{ 127, 0, 0, 1 }, sat_port, tls_mod.TLS_VERSION_1_3, tls_mod.TLS_VERSION_1_3);
        defer _ = std.c.close(holder.fd);
        try testNbSendAll(holder.fd, "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n");
        const h_resp = try testNbRecvOneHttpResponse(holder.fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, h_resp, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqual(@as(usize, 1), srv_sat.stats().active_conns);

        const overflow = try tls_mod.clientConnectSocketpair(.{ 127, 0, 0, 1 }, sat_port, tls_mod.TLS_VERSION_1_3, tls_mod.TLS_VERSION_1_3);
        defer _ = std.c.close(overflow.fd);
        try testNbSendAll(overflow.fd, "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
        const r503 = try testNbRecvAll(overflow.fd, &resp_buf);
        try std.testing.expectEqualStrings(STATIC_503_RESPONSE, r503);
        try std.testing.expect(srv_sat.stats().rejected_503_count >= 1);

        // Existing keep-alive holder connection remains healthy after rejecting the 2nd peer.
        try testNbSendAll(holder.fd, "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n");
        const h_resp2 = try testNbRecvOneHttpResponse(holder.fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, h_resp2, "HTTP/1.1 200 OK\r\n"));
    }
}

test "C29 Anthropic /v1/models parity, RFC3339 civil-from-days, pagination cursors, and prefix-free route aliases with --api-key" {
    // 1. Verify integer-math civil-from-days UTC RFC 3339 formatter (zero heap allocs).
    {
        var ts_buf: [32]u8 = undefined;
        try std.testing.expectEqualStrings("1970-01-01T00:00:00Z", formatRfc3339Utc(0, &ts_buf));
        try std.testing.expectEqualStrings("2024-02-29T12:00:00Z", formatRfc3339Utc(1709208000, &ts_buf));
        try std.testing.expectEqualStrings("2026-11-26T14:00:00Z", formatRfc3339Utc(1795701600, &ts_buf));
    }

    const gpa = std.testing.allocator;
    const mini_tok_json =
        \\{"model":{"type":"BPE","vocab":{"<|im_start|>":0,"<|im_end|>":1,"a":2,"b":3,"c":4,"d":5,"e":6,"f":7,"g":8,"h":9,"i":10,"j":11,"k":12,"l":13,"m":14,"n":15,"o":16,"p":17,"q":18,"r":19,"s":20,"t":21,"u":22,"v":23,"w":24,"x":25,"y":26,"z":27," ":28,"\n":29,"hello":30},"merges":[]},"added_tokens":[{"id":0,"content":"<|im_start|>","special":true},{"id":1,"content":"<|im_end|>","special":true}]}
    ;
    var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(gpa, mini_tok_json);
    defer tok.deinit(gpa);

    const Handlers = struct {
        fn generate(_: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(3);
            if (!stream.emitText("ok:", 1)) return;
            if (!stream.emitText(job.model_base, 1)) return;
            stream.finish(.stop, 3);
        }

        fn embed(_: *anyopaque, req: *const EmbedRequest, out_vectors: []f32) anyerror!EmbedResult {
            for (0..req.input_count) |i| {
                out_vectors[i * 2] = 1.0;
                out_vectors[i * 2 + 1] = 0.0;
            }
            return .{ .dim = 2, .prompt_tokens = 2 };
        }

        fn rerank(_: *anyopaque, req: *const RerankRequest, out_items: []RerankItem) anyerror!RerankResult {
            for (0..req.document_count) |i| {
                out_items[i] = .{ .index = @intCast(i), .relevance_score = 0.9 };
            }
            return .{ .count = req.document_count, .prompt_tokens = 2 };
        }
    };

    const adps = [_]LoraAdapterMeta{
        .{ .name = "coder", .display_name = "Coder Adapter", .rank = 8, .lora_alpha = 16.0, .scale = 2.0, .wired_bytes = 1024 },
        .{ .name = "sql", .rank = 16, .lora_alpha = 32.0, .scale = 2.0, .wired_bytes = 2048 },
        .{ .name = "math", .display_name = "Math Adapter", .rank = 4, .lora_alpha = 8.0, .scale = 2.0, .wired_bytes = 512 },
    };

    var dummy: u8 = 0;
    const srv = try Server.init(gpa, .{
        .port = 0,
        .model_id = "org/base-model",
        .api_key = "sk-c29-auth",
        .tokenizer = &tok,
        .embedding_handler = .{ .ctx = &dummy, .embedFn = Handlers.embed },
        .rerank_handler = .{ .ctx = &dummy, .rerankFn = Handlers.rerank },
        .model_detail = .{
            .architecture = "qwen3",
            .display_name = "Base Display",
            .max_model_len = 32768,
            .max_output_tokens = 8192,
            .vocab_size = 6,
            .effective_bits_per_weight = 4.5,
        },
        .lora_adapters = &adps,
        .max_conns = 6,
        .max_active_requests = 4,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try srv.start(.{ .ctx = &dummy, .generateFn = Handlers.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [2048]u8 = undefined;
    var resp_buf: [32768]u8 = undefined;

    const ModelItem = struct {
        id: []const u8,
        object: []const u8,
        type: []const u8,
        display_name: []const u8,
        created: i64,
        created_at: []const u8,
        owned_by: []const u8,
        root: []const u8,
        parent: ?[]const u8 = null,
        architecture: []const u8,
        max_model_len: u32,
        max_input_tokens: u32,
        max_tokens: u32,
        vocab_size: u32,
        effective_bits_per_weight: ?f64 = null,
        wired_mib: ?u64 = null,
        cold_bytes: ?u64 = null,
        rank: ?u32 = null,
        lora_alpha: ?f64 = null,
        scale: ?f64 = null,
        wired_bytes: ?u64 = null,
        capabilities: std.json.Value,
    };
    const ModelsList = struct {
        object: []const u8,
        data: []const ModelItem,
        first_id: ?[]const u8 = null,
        last_id: ?[]const u8 = null,
        has_more: bool,
    };

    // 2. Verify Anthropic fields + OpenAI/C28 fields + pagination on both /v1/models and /models.
    {
        const r_p1 = try testHttpExchange(
            port,
            "GET /models?limit=2 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c29-auth\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_p1, "HTTP/1.1 200 OK\r\n"));
        const p1 = try std.json.parseFromSlice(ModelsList, gpa, extractHttpBody(r_p1).?, .{});
        defer p1.deinit();
        try std.testing.expectEqual(@as(usize, 2), p1.value.data.len);
        try std.testing.expectEqualStrings("org/base-model", p1.value.first_id.?);
        try std.testing.expectEqualStrings("org/base-model:coder", p1.value.last_id.?);
        try std.testing.expect(p1.value.has_more);
        try std.testing.expectEqualStrings("model", p1.value.data[0].type);
        try std.testing.expectEqualStrings("Base Display", p1.value.data[0].display_name);
        try std.testing.expectEqual(@as(u32, 32768), p1.value.data[0].max_input_tokens);
        try std.testing.expectEqual(@as(u32, 8192), p1.value.data[0].max_tokens);
        try std.testing.expectEqual(@as(usize, 20), p1.value.data[0].created_at.len);
        try std.testing.expectEqualStrings("Coder Adapter", p1.value.data[1].display_name);

        // Next page with percent-encoded after_id.
        const r_p2 = try testHttpExchange(
            port,
            "GET /v1/models?after_id=org%2Fbase-model%3Acoder&limit=2 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c29-auth\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_p2, "HTTP/1.1 200 OK\r\n"));
        const p2 = try std.json.parseFromSlice(ModelsList, gpa, extractHttpBody(r_p2).?, .{});
        defer p2.deinit();
        try std.testing.expectEqual(@as(usize, 2), p2.value.data.len);
        try std.testing.expectEqualStrings("org/base-model:sql", p2.value.data[0].id);
        // Empty LoRA display_name falls back to the item's full id.
        try std.testing.expectEqualStrings("org/base-model:sql", p2.value.data[0].display_name);
        try std.testing.expectEqualStrings("org/base-model:math", p2.value.data[1].id);
        try std.testing.expect(!p2.value.has_more);

        // Past-end cursor produces empty list with null first_id/last_id.
        const r_empty = try testHttpExchange(
            port,
            "GET /models?after_id=org%2Fbase-model%3Amath HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c29-auth\r\n\r\n",
            &resp_buf,
        );
        const pe = try std.json.parseFromSlice(ModelsList, gpa, extractHttpBody(r_empty).?, .{});
        defer pe.deinit();
        try std.testing.expectEqual(@as(usize, 0), pe.value.data.len);
        try std.testing.expectEqual(@as(?[]const u8, null), pe.value.first_id);
        try std.testing.expectEqual(@as(?[]const u8, null), pe.value.last_id);
        try std.testing.expect(!pe.value.has_more);

        // before_id cursor returns preceding items.
        const r_bef = try testHttpExchange(
            port,
            "GET /v1/models?before_id=org%2Fbase-model%3Amath&limit=2 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c29-auth\r\n\r\n",
            &resp_buf,
        );
        const pb = try std.json.parseFromSlice(ModelsList, gpa, extractHttpBody(r_bef).?, .{});
        defer pb.deinit();
        try std.testing.expectEqual(@as(usize, 2), pb.value.data.len);
        try std.testing.expectEqualStrings("org/base-model:coder", pb.value.data[0].id);
        try std.testing.expectEqualStrings("org/base-model:sql", pb.value.data[1].id);
        try std.testing.expect(pb.value.has_more);

        // Invalid limit and cursor percent-encoding -> 400 invalid_request_error.
        const bad_qs = [_][]const u8{
            "GET /models?limit=0 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c29-auth\r\n\r\n",
            "GET /models?limit=abc HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c29-auth\r\n\r\n",
            "GET /models?limit=1001 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c29-auth\r\n\r\n",
            "GET /models?after_id=bad%ZZ HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c29-auth\r\n\r\n",
        };
        for (bad_qs) |qreq| {
            const br = try testHttpExchange(port, qreq, &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, br, "HTTP/1.1 400 Bad Request\r\n"));
            try std.testing.expect(std.mem.indexOf(u8, br, "\"invalid_request_error\"") != null);
        }

        // Prefix-free detail route GET /models/{id}.
        const r_det = try testHttpExchange(
            port,
            "GET /models/org%2Fbase-model%3Acoder HTTP/1.1\r\nHost: localhost\r\nx-api-key: sk-c29-auth\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_det, "HTTP/1.1 200 OK\r\n"));
        const pd = try std.json.parseFromSlice(ModelItem, gpa, extractHttpBody(r_det).?, .{});
        defer pd.deinit();
        try std.testing.expectEqualStrings("model", pd.value.type);
        try std.testing.expectEqualStrings("Coder Adapter", pd.value.display_name);
    }

    // 3. Prefix-free routes enforce --api-key identically and serve identical responses; /health remains open.
    {
        const r_health = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_health, "HTTP/1.1 200 OK\r\n"));

        const unauth_reqs = [_][]const u8{
            "GET /models HTTP/1.1\r\nHost: localhost\r\n\r\n",
            "POST /chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}",
            "POST /completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}",
            "POST /messages HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}",
            "POST /responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}",
            "POST /responses/input_tokens HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}",
            "GET /responses/resp_1/input_items HTTP/1.1\r\nHost: localhost\r\n\r\n",
        };
        for (unauth_reqs) |ureq| {
            const ur = try testHttpExchange(port, ureq, &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, ur, "HTTP/1.1 401 Unauthorized\r\n"));
        }

        const r_unk = try testHttpExchange(
            port,
            "GET /unknown-route HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-c29-auth\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_unk, "HTTP/1.1 404 Not Found\r\n"));

        const auth_hdr = "Authorization: Bearer sk-c29-auth\r\n";
        const r_chat = try testHttpPostJsonWithHeaders(port, "/chat/completions", auth_hdr, "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}]}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_chat, "HTTP/1.1 200 OK\r\n"));

        const r_msg = try testHttpPostJsonWithHeaders(port, "/messages", auth_hdr, "{\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"a\"}]}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_msg, "HTTP/1.1 200 OK\r\n"));

        const r_mc = try testHttpPostJsonWithHeaders(port, "/messages/count_tokens", auth_hdr, "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_mc, "HTTP/1.1 200 OK\r\n"));

        const r_emb = try testHttpPostJsonWithHeaders(port, "/embeddings", auth_hdr, "{\"input\":\"a\"}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_emb, "HTTP/1.1 200 OK\r\n"));

        const r_rer = try testHttpPostJsonWithHeaders(port, "/rerank", auth_hdr, "{\"query\":\"a\",\"documents\":[\"b\"]}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_rer, "HTTP/1.1 200 OK\r\n"));
    }
}

test "C29 /v1/responses/input_tokens and /v1/responses/{id}/input_items across previous_response_id chains" {
    const gpa = std.testing.allocator;
    const mini_tok_json =
        \\{"model":{"type":"BPE","vocab":{"<|im_start|>":0,"<|im_end|>":1,"a":2,"b":3,"c":4,"d":5,"e":6,"f":7,"g":8,"h":9,"i":10,"j":11,"k":12,"l":13,"m":14,"n":15,"o":16,"p":17,"q":18,"r":19,"s":20,"t":21,"u":22,"v":23,"w":24,"x":25,"y":26,"z":27," ":28,"\n":29,"<":30,">":31,"/":32,"_":33,".":34,"!":35},"merges":[]},"added_tokens":[{"id":0,"content":"<|im_start|>","special":true},{"id":1,"content":"<|im_end|>","special":true}]}
    ;
    var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(gpa, mini_tok_json);
    defer tok.deinit(gpa);

    const Mock = struct {
        fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(4);
            if (!stream.emitText("done", 1)) return;
            stream.finish(.stop, 4);
        }
    };
    var dummy: u8 = 0;

    // Without tokenizer -> 501 Not Implemented on /v1/responses/input_tokens.
    {
        const srv_notok = try Server.init(gpa, .{
            .port = 0,
            .max_conns = 4,
            .max_active_requests = 2,
            .max_media_bytes = 4096,
            .apply_qos = false,
        });
        defer srv_notok.deinit(gpa);
        try srv_notok.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
        defer srv_notok.stop();

        var req_buf: [2048]u8 = undefined;
        var resp_buf: [8192]u8 = undefined;
        const r_501 = try testHttpPostJson(srv_notok.boundPort(), "/v1/responses/input_tokens", "{\"input\":\"hello\"}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_501, "HTTP/1.1 501 Not Implemented\r\n"));
    }

    const srv = try Server.init(gpa, .{
        .port = 0,
        .model_id = "qwen3-resp",
        .tokenizer = &tok,
        .max_conns = 6,
        .max_active_requests = 4,
        .max_media_bytes = 4096,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try srv.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [4096]u8 = undefined;
    var resp_buf: [16384]u8 = undefined;

    // 405 on GET /v1/responses/input_tokens.
    {
        const r_405 = try testHttpExchange(port, "GET /v1/responses/input_tokens HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_405, "HTTP/1.1 405 Method Not Allowed\r\n"));
        try std.testing.expectEqualStrings("POST", extractHeaderValue(r_405, "Allow") orelse "");
    }

    // 404 on unknown previous_response_id.
    {
        const r_404 = try testHttpPostJson(port, "/responses/input_tokens", "{\"input\":\"hi\",\"previous_response_id\":\"resp_missing\"}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_404, "HTTP/1.1 404 Not Found\r\n"));
    }

    // Turn 1 stateful response with instructions + developer + user input items.
    const r1 = try testHttpPostJson(
        port,
        "/responses",
        "{\"instructions\":\"be brief\",\"input\":[{\"role\":\"developer\",\"content\":\"use json\"},{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"first turn\"}]}]}",
        &req_buf,
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, r1, "HTTP/1.1 200 OK\r\n"));
    const RespObj = struct { id: []const u8 };
    const p_r1 = try std.json.parseFromSlice(RespObj, gpa, extractHttpBody(r1).?, .{ .ignore_unknown_fields = true });
    defer p_r1.deinit();
    var r1_id_buf: [MAX_RESPONSE_ID_BYTES]u8 = undefined;
    @memcpy(r1_id_buf[0..p_r1.value.id.len], p_r1.value.id);
    const r1_id = r1_id_buf[0..p_r1.value.id.len];

    // Count tokens for turn 1 alone vs turn 2 chained to resp_1.
    const TokCountObj = struct { object: []const u8, input_tokens: u32 };
    const r_cnt1 = try testHttpPostJson(port, "/v1/responses/input_tokens", "{\"input\":\"follow up\"}", &req_buf, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r_cnt1, "HTTP/1.1 200 OK\r\n"));
    const pc1 = try std.json.parseFromSlice(TokCountObj, gpa, extractHttpBody(r_cnt1).?, .{});
    defer pc1.deinit();
    try std.testing.expectEqualStrings("response.input_tokens", pc1.value.object);
    try std.testing.expect(pc1.value.input_tokens > 0);

    var chain_body_buf: [512]u8 = undefined;
    const chain_body = try std.fmt.bufPrint(
        &chain_body_buf,
        "{{\"previous_response_id\":\"{s}\",\"input\":\"follow up\"}}",
        .{r1_id},
    );
    const r_cnt2 = try testHttpPostJson(port, "/responses/input_tokens", chain_body, &req_buf, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r_cnt2, "HTTP/1.1 200 OK\r\n"));
    const pc2 = try std.json.parseFromSlice(TokCountObj, gpa, extractHttpBody(r_cnt2).?, .{});
    defer pc2.deinit();
    try std.testing.expect(pc2.value.input_tokens > pc1.value.input_tokens);

    // Turn 2 stateful response chained to turn 1.
    const r2 = try testHttpPostJson(port, "/v1/responses", chain_body, &req_buf, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r2, "HTTP/1.1 200 OK\r\n"));
    const p_r2 = try std.json.parseFromSlice(RespObj, gpa, extractHttpBody(r2).?, .{ .ignore_unknown_fields = true });
    defer p_r2.deinit();
    var r2_id_buf: [MAX_RESPONSE_ID_BYTES]u8 = undefined;
    @memcpy(r2_id_buf[0..p_r2.value.id.len], p_r2.value.id);
    const r2_id = r2_id_buf[0..p_r2.value.id.len];

    // input_items validation: 405 on POST, 404 on missing id, 400 on bad order/limit.
    {
        var path_buf: [128]u8 = undefined;
        const p_post = try std.fmt.bufPrint(&path_buf, "/v1/responses/{s}/input_items", .{r2_id});
        const r_405 = try testHttpPostJson(port, p_post, "{}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_405, "HTTP/1.1 405 Method Not Allowed\r\n"));
        try std.testing.expectEqualStrings("GET", extractHeaderValue(r_405, "Allow") orelse "");

        const r_404 = try testHttpExchange(port, "GET /v1/responses/resp_unknown/input_items HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_404, "HTTP/1.1 404 Not Found\r\n"));

        var bad_req_buf: [256]u8 = undefined;
        const bad_ord = try std.fmt.bufPrint(&bad_req_buf, "GET /responses/{s}/input_items?order=bad HTTP/1.1\r\nHost: localhost\r\n\r\n", .{r2_id});
        const r_bad_ord = try testHttpExchange(port, bad_ord, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_bad_ord, "HTTP/1.1 400 Bad Request\r\n"));
    }

    // Verify input_items preserves all 4 items across chain (system, developer, first turn, follow up), asc/desc + limit/after!
    const InputContent = struct { type: []const u8, text: []const u8 };
    const InputItem = struct {
        id: []const u8,
        type: []const u8,
        status: []const u8,
        role: []const u8,
        content: []const InputContent,
    };
    const InputItemsList = struct {
        object: []const u8,
        data: []const InputItem,
        first_id: ?[]const u8 = null,
        last_id: ?[]const u8 = null,
        has_more: bool,
    };

    var get_buf: [256]u8 = undefined;
    const req_asc = try std.fmt.bufPrint(&get_buf, "GET /v1/responses/{s}/input_items?order=asc HTTP/1.1\r\nHost: localhost\r\n\r\n", .{r2_id});
    const r_asc = try testHttpExchange(port, req_asc, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r_asc, "HTTP/1.1 200 OK\r\n"));
    const p_asc = try std.json.parseFromSlice(InputItemsList, gpa, extractHttpBody(r_asc).?, .{});
    defer p_asc.deinit();
    try std.testing.expectEqualStrings("list", p_asc.value.object);
    try std.testing.expectEqual(@as(usize, 4), p_asc.value.data.len);
    try std.testing.expectEqualStrings("system", p_asc.value.data[0].role);
    try std.testing.expectEqualStrings("be brief", p_asc.value.data[0].content[0].text);
    try std.testing.expectEqualStrings("developer", p_asc.value.data[1].role);
    try std.testing.expectEqualStrings("use json", p_asc.value.data[1].content[0].text);
    try std.testing.expectEqualStrings("user", p_asc.value.data[2].role);
    try std.testing.expectEqualStrings("first turn", p_asc.value.data[2].content[0].text);
    try std.testing.expectEqualStrings("user", p_asc.value.data[3].role);
    try std.testing.expectEqualStrings("follow up", p_asc.value.data[3].content[0].text);
    try std.testing.expect(!p_asc.value.has_more);

    // Default desc order + limit=2 + after cursor pagination.
    const req_desc = try std.fmt.bufPrint(&get_buf, "GET /responses/{s}/input_items?limit=2 HTTP/1.1\r\nHost: localhost\r\n\r\n", .{r2_id});
    const r_desc = try testHttpExchange(port, req_desc, &resp_buf);
    const p_desc = try std.json.parseFromSlice(InputItemsList, gpa, extractHttpBody(r_desc).?, .{});
    defer p_desc.deinit();
    try std.testing.expectEqual(@as(usize, 2), p_desc.value.data.len);
    try std.testing.expectEqualStrings("follow up", p_desc.value.data[0].content[0].text);
    try std.testing.expectEqualStrings("first turn", p_desc.value.data[1].content[0].text);
    try std.testing.expect(p_desc.value.has_more);
    var last_item_id_buf: [48]u8 = undefined;
    @memcpy(last_item_id_buf[0..p_desc.value.last_id.?.len], p_desc.value.last_id.?);
    const last_item_id = last_item_id_buf[0..p_desc.value.last_id.?.len];

    const req_p2 = try std.fmt.bufPrint(&get_buf, "GET /v1/responses/{s}/input_items?order=desc&limit=2&after={s} HTTP/1.1\r\nHost: localhost\r\n\r\n", .{ r2_id, last_item_id });
    const r_p2 = try testHttpExchange(port, req_p2, &resp_buf);
    const p_p2 = try std.json.parseFromSlice(InputItemsList, gpa, extractHttpBody(r_p2).?, .{});
    defer p_p2.deinit();
    try std.testing.expectEqual(@as(usize, 2), p_p2.value.data.len);
    try std.testing.expectEqualStrings("use json", p_p2.value.data[0].content[0].text);
    try std.testing.expectEqualStrings("be brief", p_p2.value.data[1].content[0].text);
    try std.testing.expect(!p_p2.value.has_more);
}

test "C29 /v1/completions streaming SSE, non-streaming JSON, FIM suffix, token-ID array prompt, and batch_slot_streams" {
    const gpa = std.testing.allocator;
    const mini_tok_json =
        \\{"model":{"type":"BPE","vocab":{"<|im_start|>":0,"<|im_end|>":1,"a":2,"b":3,"c":4,"d":5},"merges":[]},"added_tokens":[{"id":0,"content":"<|im_start|>","special":true},{"id":1,"content":"<|im_end|>","special":true}]}
    ;
    var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(gpa, mini_tok_json);
    defer tok.deinit(gpa);

    const MockState = struct {
        last_prompt_buf: [2048]u8 = undefined,
        last_prompt_len: usize = 0,
        last_ids_buf: [64]u32 = undefined,
        last_ids_len: usize = 0,

        fn generate(ctx_ptr: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            self.last_prompt_len = @min(job.prompt.len, self.last_prompt_buf.len);
            @memcpy(self.last_prompt_buf[0..self.last_prompt_len], job.prompt[0..self.last_prompt_len]);
            if (job.prompt_token_ids) |ids| {
                self.last_ids_len = @min(ids.len, self.last_ids_buf.len);
                @memcpy(self.last_ids_buf[0..self.last_ids_len], ids[0..self.last_ids_len]);
            } else {
                self.last_ids_len = 0;
            }
            _ = stream.setPromptTokens(5);
            if (!stream.emitText("<think>keep</think>", 1)) return;
            if (!stream.emitText("world!", 1)) return;
            stream.finish(.stop, 5);
        }
    };

    // 1. Single-slot Server with custom FimTokens + Tokenizer.
    var mock = MockState{};
    const srv = try Server.init(gpa, .{
        .port = 0,
        .model_id = "qwen3-coder",
        .tokenizer = &tok,
        .fim_tokens = .{
            .prefix = "<PRE>",
            .suffix = "<SUF>",
            .middle = "<MID>",
        },
        .max_conns = 6,
        .max_active_requests = 4,
        .max_media_bytes = 4096,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try srv.start(.{ .ctx = &mock, .generateFn = MockState.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [4096]u8 = undefined;
    var resp_buf: [16384]u8 = undefined;

    // 405 on GET /v1/completions.
    {
        const r_405 = try testHttpExchange(port, "GET /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_405, "HTTP/1.1 405 Method Not Allowed\r\n"));
        try std.testing.expectEqualStrings("POST", extractHeaderValue(r_405, "Allow") orelse "");
    }

    // Non-streaming raw prompt (no ChatML wrapping, and literal <think> preserved in text!).
    {
        const r = try testHttpPostJson(
            port,
            "/v1/completions",
            "{\"model\":\"qwen3-coder\",\"prompt\":\"def hi():\",\"logprobs\":3,\"max_tokens\":16,\"stream\":false}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualStrings("def hi():", mock.last_prompt_buf[0..mock.last_prompt_len]);
        try std.testing.expect(std.mem.indexOf(u8, mock.last_prompt_buf[0..mock.last_prompt_len], "<|im_start|>") == null);
        const body = extractHttpBody(r).?;
        try expectValidJson(body);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"object\":\"text_completion\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"id\":\"cmpl-") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"text\":\"<think>keep</think>world!\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, "\"finish_reason\":\"stop\"") != null);
    }

    // Invalid logprobs > 5 -> 400.
    {
        const r_bad = try testHttpPostJson(port, "/completions", "{\"prompt\":\"hi\",\"logprobs\":6}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_bad, "HTTP/1.1 400 Bad Request\r\n"));
    }

    // FIM suffix wrapping + streaming SSE with stream_options.include_usage.
    {
        const r_sse = try testHttpPostJson(
            port,
            "/completions",
            "{\"prompt\":\"fn add(a: i32, \",\"suffix\":\") -> i32 { a + b }\",\"stream\":true,\"stream_options\":{\"include_usage\":true}}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_sse, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualStrings("<PRE>fn add(a: i32, <SUF>) -> i32 { a + b }<MID>", mock.last_prompt_buf[0..mock.last_prompt_len]);
        const sse = extractHttpBody(r_sse).?;
        _ = try expectValidSseJsonFrames(sse);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"object\":\"text_completion\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "\"choices\":[],\"usage\":") != null);
        try std.testing.expect(std.mem.indexOf(u8, sse, "data: [DONE]\n\n") != null);
    }

    // Token-ID array prompt [2, 3, 4] decoded via Tokenizer AND forwarded on InferenceJob.prompt_token_ids.
    {
        const r_ids = try testHttpPostJson(
            port,
            "/v1/completions",
            "{\"prompt\":[2,3,4],\"stream\":false}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_ids, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualSlices(u32, &.{ 2, 3, 4 }, mock.last_ids_buf[0..mock.last_ids_len]);
        try std.testing.expectEqualStrings("abc", mock.last_prompt_buf[0..mock.last_prompt_len]);
    }

    // 2. Continuous-batching `batch_slot_streams` path for .completions (default FimTokens + token_ids without tokenizer)!
    {
        const BatchMock = struct {
            saw_fim: bool = false,
            saw_ids: bool = false,

            fn tokenize(ctx_ptr: *anyopaque, job: *const InferenceJob, out_tokens: []u32) batch_sched.BackendError!usize {
                const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
                if (job.prompt_token_ids) |ids| {
                    self.saw_ids = std.mem.eql(u32, ids, &.{ 7, 8, 9 });
                    @memcpy(out_tokens[0..ids.len], ids);
                    return ids.len;
                }
                if (std.mem.eql(u8, job.prompt, "<|fim_prefix|>x<|fim_suffix|>z<|fim_middle|>")) {
                    self.saw_fim = true;
                }
                out_tokens[0] = 1;
                out_tokens[1] = 2;
                return 2;
            }
            fn detokenize(_: *anyopaque, tok_id: u32, out_bytes: []u8) []const u8 {
                if (tok_id == 10) return out_bytes[0..0];
                @memcpy(out_bytes[0..1], "y");
                return out_bytes[0..1];
            }
            fn lookupPrefix(_: *anyopaque, _: u16, _: []const u32) batch_sched.PrefixLookupResult {
                return .{};
            }
            fn prefillSlot(_: *anyopaque, chunk: *const batch_sched.PrefillChunkView) batch_sched.BackendError!batch_sched.PrefillResult {
                if (!chunk.is_final_chunk) return .{};
                return .{ .first_token = 11 };
            }
            fn stepDecodeBatch(_: *anyopaque, view: *const batch_sched.BatchStepView, out: []u32) batch_sched.BackendError!void {
                for (0..view.batchSize()) |i| {
                    out[i] = 10;
                }
            }
            fn releaseSlot(_: *anyopaque, _: u16, _: u64) void {}
        };
        var bmock = BatchMock{};
        const srv_b = try Server.init(gpa, .{
            .port = 0,
            .max_conns = 4,
            .max_active_requests = 4,
            .max_media_bytes = 4096,
            .apply_qos = false,
        });
        defer srv_b.deinit(gpa);
        try srv_b.startBatched(.{
            .ctx = &bmock,
            .tokenizeFn = BatchMock.tokenize,
            .detokenizeFn = BatchMock.detokenize,
            .backend = .{
                .ctx = &bmock,
                .lookupPrefixFn = BatchMock.lookupPrefix,
                .prefillSlotFn = BatchMock.prefillSlot,
                .stepDecodeBatchFn = BatchMock.stepDecodeBatch,
                .releaseSlotFn = BatchMock.releaseSlot,
            },
            .eos_token_ids = .{ 10, 0, 0, 0 },
            .eos_token_count = 1,
        });
        defer srv_b.stop();

        const r_b1 = try testHttpPostJson(srv_b.boundPort(), "/v1/completions", "{\"prompt\":\"x\",\"suffix\":\"z\",\"stream\":true}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_b1, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(bmock.saw_fim);
        try std.testing.expect(std.mem.indexOf(u8, r_b1, "\"object\":\"text_completion\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, r_b1, "\"text\":\"y\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, r_b1, "data: [DONE]\n\n") != null);

        const r_b2 = try testHttpPostJson(srv_b.boundPort(), "/completions", "{\"prompt\":[7,8,9],\"stream\":false}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_b2, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(bmock.saw_ids);
    }
}

test "C29 unauthenticated OPTIONS CORS preflight, CORS headers across JSON/SSE/HTTPS, base64 multimodal media ingress, and zero hot-path mallocs" {
    const gpa = std.testing.allocator;
    const mini_tok_json =
        \\{"model":{"type":"BPE","vocab":{"<|im_start|>":0,"<|im_end|>":1,"a":2,"b":3,"c":4,"d":5,"e":6,"f":7,"g":8,"h":9,"i":10,"j":11,"k":12,"l":13,"m":14,"n":15,"o":16,"p":17,"q":18,"r":19,"s":20,"t":21,"u":22,"v":23,"w":24,"x":25,"y":26,"z":27," ":28,"\n":29},"merges":[]},"added_tokens":[{"id":0,"content":"<|im_start|>","special":true},{"id":1,"content":"<|im_end|>","special":true}]}
    ;
    var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(gpa, mini_tok_json);
    defer tok.deinit(gpa);

    const MediaMock = struct {
        count: u8 = 0,
        kinds: [MAX_MEDIA_ITEMS_PER_REQUEST]MediaKind = undefined,
        mimes: [MAX_MEDIA_ITEMS_PER_REQUEST][32]u8 = undefined,
        mime_lens: [MAX_MEDIA_ITEMS_PER_REQUEST]u8 = @splat(0),
        data_buf: [MAX_MEDIA_ITEMS_PER_REQUEST][64]u8 = undefined,
        data_lens: [MAX_MEDIA_ITEMS_PER_REQUEST]usize = @splat(0),
        saw_image_tag: bool = false,
        saw_video_tag: bool = false,

        fn generate(ctx_ptr: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            const items = job.mediaSlice();
            self.count = @intCast(items.len);
            for (items, 0..) |m, idx| {
                self.kinds[idx] = m.kind;
                const ms = m.mimeSlice();
                self.mime_lens[idx] = @intCast(ms.len);
                @memcpy(self.mimes[idx][0..ms.len], ms);
                const take = @min(m.data.len, self.data_buf[idx].len);
                self.data_lens[idx] = take;
                @memcpy(self.data_buf[idx][0..take], m.data[0..take]);
            }
            self.saw_image_tag = std.mem.indexOf(u8, job.prompt, "<|image_pad|>") != null;
            self.saw_video_tag = std.mem.indexOf(u8, job.prompt, "<|video_pad|>") != null;
            _ = stream.setPromptTokens(4);
            if (!stream.emitText("mm-ok", 1)) return;
            stream.finish(.stop, 4);
        }
    };

    var mm = MediaMock{};
    const srv = try Server.init(gpa, .{
        .port = 0,
        .model_id = "qwen3-vl",
        .api_key = "sk-cors-test",
        .cors_allow_origin = "https://ui.local",
        .tokenizer = &tok,
        .max_conns = 6,
        .max_active_requests = 4,
        .max_media_bytes = 64,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try srv.start(.{ .ctx = &mm, .generateFn = MediaMock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [8192]u8 = undefined;
    var resp_buf: [16384]u8 = undefined;

    // 1. Unauthenticated OPTIONS preflight on known routes -> 204 No Content with full CORS headers.
    {
        const r_opt = try testHttpExchange(port, "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_opt, "HTTP/1.1 204 No Content\r\n"));
        try std.testing.expectEqualStrings("0", extractHeaderValue(r_opt, "Content-Length") orelse "");
        try std.testing.expectEqualStrings("https://ui.local", extractHeaderValue(r_opt, "Access-Control-Allow-Origin") orelse "");
        try std.testing.expectEqualStrings("GET, POST, DELETE, OPTIONS", extractHeaderValue(r_opt, "Access-Control-Allow-Methods") orelse "");
        try std.testing.expect(std.mem.indexOf(u8, extractHeaderValue(r_opt, "Access-Control-Allow-Headers") orelse "", "anthropic-version") != null);
        try std.testing.expectEqualStrings("86400", extractHeaderValue(r_opt, "Access-Control-Max-Age") orelse "");

        const r_opt_pf = try testHttpExchange(port, "OPTIONS /completions HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_opt_pf, "HTTP/1.1 204 No Content\r\n"));
    }

    // 2. CORS headers injected on 401 error, 200 JSON, and SSE 200 OK headers.
    {
        const r_401 = try testHttpExchange(port, "GET /models HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_401, "HTTP/1.1 401 Unauthorized\r\n"));
        try std.testing.expectEqualStrings("https://ui.local", extractHeaderValue(r_401, "Access-Control-Allow-Origin") orelse "");
        try std.testing.expectEqualStrings("x-request-id, Content-Type", extractHeaderValue(r_401, "Access-Control-Expose-Headers") orelse "");

        const r_200 = try testHttpExchange(port, "GET /models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-cors-test\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_200, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualStrings("https://ui.local", extractHeaderValue(r_200, "Access-Control-Allow-Origin") orelse "");

        const r_sse = try testHttpPostJsonWithHeaders(
            port,
            "/v1/completions",
            "Authorization: Bearer sk-cors-test\r\n",
            "{\"prompt\":\"a\",\"stream\":true}",
            &req_buf,
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_sse, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqualStrings("https://ui.local", extractHeaderValue(r_sse, "Access-Control-Allow-Origin") orelse "");
    }

    // 3. Base64 media extraction across Chat Completions (image_url + video_url), Responses (input_image), Messages (source.base64).
    const auth_hdr = "Authorization: Bearer sk-cors-test\r\n";
    {
        // Standard base64 "PNG!" -> UExH is "PLG", "iVBORw0K" = \x89PNG\r\n ; use exact bytes "PNG_OK" = "UE5HX09L", "MP4_OK" = "TVA0X09L" (URL-safe + whitespace!).
        const chat_mm =
            \\{"messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,UE5H\nX09L"}},{"type":"video_url","video_url":{"url":"data:video/mp4;base64,TVA0X09L"}}]}]}
        ;
        const r_cmm = try testHttpPostJsonWithHeaders(port, "/v1/chat/completions", auth_hdr, chat_mm, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_cmm, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqual(@as(u8, 2), mm.count);
        try std.testing.expectEqual(MediaKind.image, mm.kinds[0]);
        try std.testing.expectEqualStrings("image/png", mm.mimes[0][0..mm.mime_lens[0]]);
        try std.testing.expectEqualStrings("PNG_OK", mm.data_buf[0][0..mm.data_lens[0]]);
        try std.testing.expectEqual(MediaKind.video, mm.kinds[1]);
        try std.testing.expectEqualStrings("video/mp4", mm.mimes[1][0..mm.mime_lens[1]]);
        try std.testing.expectEqualStrings("MP4_OK", mm.data_buf[1][0..mm.data_lens[1]]);
        try std.testing.expect(mm.saw_image_tag);
        try std.testing.expect(mm.saw_video_tag);

        // OpenAI /v1/responses input_image with URL-safe base64 ("-_8=" -> 0xfb 0xff).
        const resp_mm =
            \\{"input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/webp;base64,-_8"}]}]}
        ;
        const r_rmm = try testHttpPostJsonWithHeaders(port, "/v1/responses", auth_hdr, resp_mm, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_rmm, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqual(@as(u8, 1), mm.count);
        try std.testing.expectEqual(MediaKind.image, mm.kinds[0]);
        try std.testing.expectEqualStrings("image/webp", mm.mimes[0][0..mm.mime_lens[0]]);
        try std.testing.expectEqualSlices(u8, &.{ 0xfb, 0xff }, mm.data_buf[0][0..mm.data_lens[0]]);

        // Anthropic /v1/messages source.type=base64.
        const msg_mm =
            \\{"max_tokens":8,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"SlBFRw=="}},{"type":"text","text":"hi"}]}]}
        ;
        const r_mmm = try testHttpPostJsonWithHeaders(port, "/v1/messages", auth_hdr, msg_mm, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_mmm, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqual(@as(u8, 1), mm.count);
        try std.testing.expectEqualStrings("image/jpeg", mm.mimes[0][0..mm.mime_lens[0]]);
        try std.testing.expectEqualStrings("JPEG", mm.data_buf[0][0..mm.data_lens[0]]);
    }

    // 4. Reject remote URLs (400), invalid base64 (400), >4 items (400), and >max_media_bytes (413).
    {
        const remote_body =
            \\{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}
        ;
        const r_rem = try testHttpPostJsonWithHeaders(port, "/v1/chat/completions", auth_hdr, remote_body, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_rem, "HTTP/1.1 400 Bad Request\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, r_rem, "remote media URLs are not supported; provide base64 data") != null);

        const bad_b64_body =
            \\{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,@@@!"}}]}]}
        ;
        const r_b64 = try testHttpPostJsonWithHeaders(port, "/v1/chat/completions", auth_hdr, bad_b64_body, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_b64, "HTTP/1.1 400 Bad Request\r\n"));

        const five_items =
            \\{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,QQ=="}},{"type":"image_url","image_url":{"url":"data:image/png;base64,QQ=="}},{"type":"image_url","image_url":{"url":"data:image/png;base64,QQ=="}},{"type":"image_url","image_url":{"url":"data:image/png;base64,QQ=="}},{"type":"image_url","image_url":{"url":"data:image/png;base64,QQ=="}}]}]}
        ;
        const r_5 = try testHttpPostJsonWithHeaders(port, "/v1/chat/completions", auth_hdr, five_items, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_5, "HTTP/1.1 400 Bad Request\r\n"));

        // 72 decoded bytes (`QUFB...` 96 chars) > max_media_bytes (64) -> 413 Payload Too Large!
        const big_b64 = "QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFB";
        var big_body_buf: [512]u8 = undefined;
        const big_body = try std.fmt.bufPrint(
            &big_body_buf,
            "{{\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"data:image/png;base64,{s}\"}}}}]}}]}}",
            .{big_b64},
        );
        const r_413 = try testHttpPostJsonWithHeaders(port, "/v1/chat/completions", auth_hdr, big_body, &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_413, "HTTP/1.1 413 Payload Too Large\r\n"));
    }

    // 5. Zero hot-path heap allocations (`malloc_zone_statistics`) across all new C29 routes!
    {
        const warm_chat_mm =
            \\{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,UE5HX09L"}}]}]}
        ;
        const r_resp_seed = try testHttpPostJsonWithHeaders(port, "/responses", auth_hdr, "{\"input\":\"seed item\"}", &req_buf, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_resp_seed, "HTTP/1.1 200 OK\r\n"));
        const RespId = struct { id: []const u8 };
        const rid = try std.json.parseFromSlice(RespId, gpa, extractHttpBody(r_resp_seed).?, .{ .ignore_unknown_fields = true });
        defer rid.deinit();
        var rid_buf: [MAX_RESPONSE_ID_BYTES]u8 = undefined;
        @memcpy(rid_buf[0..rid.value.id.len], rid.value.id);
        const rid_slice = rid_buf[0..rid.value.id.len];

        var items_req_buf: [256]u8 = undefined;
        const items_req = try std.fmt.bufPrint(&items_req_buf, "GET /responses/{s}/input_items HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-cors-test\r\n\r\n", .{rid_slice});

        // Warm up every C29 route first.
        _ = try testHttpExchange(port, "OPTIONS /completions HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        _ = try testHttpExchange(port, "GET /models?limit=1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-cors-test\r\n\r\n", &resp_buf);
        _ = try testHttpPostJsonWithHeaders(port, "/responses/input_tokens", auth_hdr, "{\"input\":\"abc\"}", &req_buf, &resp_buf);
        _ = try testHttpExchange(port, items_req, &resp_buf);
        _ = try testHttpPostJsonWithHeaders(port, "/completions", auth_hdr, "{\"prompt\":\"a\",\"suffix\":\"b\",\"stream\":false}", &req_buf, &resp_buf);
        _ = try testHttpPostJsonWithHeaders(port, "/v1/chat/completions", auth_hdr, warm_chat_mm, &req_buf, &resp_buf);

        while (srv.stats().active_conns > 0 or srv.stats().active_requests > 0) {
            sleepNs(500_000);
        }

        var m_before: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &m_before);

        for (0..4) |_| {
            const ro = try testHttpExchange(port, "OPTIONS /completions HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, ro, "HTTP/1.1 204 No Content\r\n"));
            const rm = try testHttpExchange(port, "GET /models?limit=1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-cors-test\r\n\r\n", &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, rm, "HTTP/1.1 200 OK\r\n"));
            const rit = try testHttpPostJsonWithHeaders(port, "/responses/input_tokens", auth_hdr, "{\"input\":\"abc\"}", &req_buf, &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, rit, "HTTP/1.1 200 OK\r\n"));
            const rii = try testHttpExchange(port, items_req, &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, rii, "HTTP/1.1 200 OK\r\n"));
            const rcm = try testHttpPostJsonWithHeaders(port, "/completions", auth_hdr, "{\"prompt\":\"a\",\"suffix\":\"b\",\"stream\":false}", &req_buf, &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, rcm, "HTTP/1.1 200 OK\r\n"));
            const rmm = try testHttpPostJsonWithHeaders(port, "/v1/chat/completions", auth_hdr, warm_chat_mm, &req_buf, &resp_buf);
            try std.testing.expect(std.mem.startsWith(u8, rmm, "HTTP/1.1 200 OK\r\n"));
        }

        while (srv.stats().active_conns > 0 or srv.stats().active_requests > 0) {
            sleepNs(500_000);
        }

        var m_after: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &m_after);
        try std.testing.expect(m_after.blocks_in_use <= m_before.blocks_in_use);
        try std.testing.expect(m_after.size_in_use <= m_before.size_in_use);
    }

    // 6. CORS disabled (`cors_allow_origin: null`) + HTTPS TLS 1.3 CORS verification.
    {
        const srv_nocors = try Server.init(gpa, .{
            .port = 0,
            .cors_allow_origin = null,
            .max_conns = 2,
            .max_active_requests = 1,
            .max_media_bytes = 1024,
            .apply_qos = false,
        });
        defer srv_nocors.deinit(gpa);
        try srv_nocors.start(.{ .ctx = &mm, .generateFn = MediaMock.generate });
        defer srv_nocors.stop();

        const r_nc = try testHttpExchange(srv_nocors.boundPort(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_nc, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expectEqual(@as(?[]const u8, null), extractHeaderValue(r_nc, "Access-Control-Allow-Origin"));

        // HTTPS TLS 1.3 OPTIONS preflight + CORS header check.
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var threaded_io: std.Io.Threaded = .init(gpa, .{});
        defer threaded_io.deinit();
        const io = threaded_io.io();
        const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        defer gpa.free(dir);
        const ec_key = try std.fmt.allocPrint(gpa, "{s}/c29.key.pem", .{dir});
        defer gpa.free(ec_key);
        const ec_cert = try std.fmt.allocPrint(gpa, "{s}/c29.cert.pem", .{dir});
        defer gpa.free(ec_cert);
        try testRunCmdChecked(gpa, io, &.{
            "/usr/bin/openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", ec_key,
        });
        try testRunCmdChecked(gpa, io, &.{
            "/usr/bin/openssl", "req", "-x509", "-new", "-key", ec_key, "-subj", "/CN=127.0.0.1", "-days", "1", "-out", ec_cert,
        });
        const cert_bytes = try std.Io.Dir.cwd().readFileAlloc(io, ec_cert, gpa, .limited(64 * 1024));
        defer gpa.free(cert_bytes);
        const key_bytes = try std.Io.Dir.cwd().readFileAlloc(io, ec_key, gpa, .limited(64 * 1024));
        defer gpa.free(key_bytes);

        const srv_tls = try Server.init(gpa, .{
            .port = 0,
            .api_key = "sk-tls-c29",
            .cors_allow_origin = "*",
            .tls = .{ .cert_bytes = cert_bytes, .key_bytes = key_bytes },
            .max_conns = 4,
            .max_active_requests = 2,
            .max_media_bytes = 1024,
            .apply_qos = false,
        });
        defer srv_tls.deinit(gpa);
        try srv_tls.start(.{ .ctx = &mm, .generateFn = MediaMock.generate });
        defer srv_tls.stop();

        const r_tls_opt = try testTlsHttpExchange(
            srv_tls.boundPort(),
            tls_mod.TLS_VERSION_1_3,
            tls_mod.TLS_VERSION_1_3,
            "OPTIONS /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
            &resp_buf,
        );
        try std.testing.expect(std.mem.startsWith(u8, r_tls_opt.resp, "HTTP/1.1 204 No Content\r\n"));
        try std.testing.expectEqualStrings("*", extractHeaderValue(r_tls_opt.resp, "Access-Control-Allow-Origin") orelse "");
    }
}

test "F1 & L4 overflow-free decodeChunkedInPlace, 413 oversized chunks, and strict HTTP number/TE parsing" {
    // 1. Appendix A.1 standalone `decodeChunkedInPlace` overflow and 413 checks.
    {
        var b1: [4096]u8 = undefined;
        const s1 = "fffffffffffffffe\r\n0\r\n\r\n";
        @memcpy(b1[0..s1.len], s1);
        try std.testing.expectError(error.PayloadTooLarge, decodeChunkedInPlace(b1[0..s1.len], b1.len));

        var b2: [4096]u8 = undefined;
        const s2 = "ffffffffffffffff\r\n0\r\n\r\n";
        @memcpy(b2[0..s2.len], s2);
        try std.testing.expectError(error.PayloadTooLarge, decodeChunkedInPlace(b2[0..s2.len], b2.len));

        var b3: [4096]u8 = undefined;
        const s3 = "2000\r\nabcd"; // chunk_len (8192) > rx_buf.len (4096)
        @memcpy(b3[0..s3.len], s3);
        try std.testing.expectError(error.PayloadTooLarge, decodeChunkedInPlace(b3[0..s3.len], b3.len));

        var b4: [4096]u8 = undefined;
        const s4 = "100000\r\npartial_unterminated";
        @memcpy(b4[0..s4.len], s4);
        try std.testing.expectError(error.PayloadTooLarge, decodeChunkedInPlace(b4[0..s4.len], b4.len));

        // Cumulative chunk size exceeding max_body_bytes (64): 40 + 40 = 80 > 64 -> 413!
        var b5: [128]u8 = undefined;
        const s5 = "28\r\n0123456789012345678901234567890123456789\r\n28\r\n";
        @memcpy(b5[0..s5.len], s5);
        try std.testing.expectError(error.PayloadTooLarge, decodeChunkedInPlace(b5[0..s5.len], 64));

        // Incomplete chunk within bounds returns `.incomplete` rather than erroring.
        var b6: [128]u8 = undefined;
        const s6 = "10\r\nhello";
        @memcpy(b6[0..s6.len], s6);
        const st6 = try decodeChunkedInPlace(b6[0..s6.len], b6.len);
        try std.testing.expect(st6 == .incomplete);
    }

    // 2. Appendix A.2 strict decimal/hex parsers and Transfer-Encoding exact match.
    {
        try std.testing.expectError(error.InvalidSyntax, parseStrictDecimalUsize(""));
        try std.testing.expectError(error.InvalidSyntax, parseStrictDecimalUsize("+5"));
        try std.testing.expectError(error.InvalidSyntax, parseStrictDecimalUsize("-0"));
        try std.testing.expectError(error.InvalidSyntax, parseStrictDecimalUsize("1_0"));
        try std.testing.expectError(error.InvalidSyntax, parseStrictDecimalUsize("0x10"));
        try std.testing.expectEqual(@as(usize, 42), try parseStrictDecimalUsize("42"));

        try std.testing.expectError(error.InvalidSyntax, parseStrictHexUsize(""));
        try std.testing.expectError(error.InvalidSyntax, parseStrictHexUsize("+a"));
        try std.testing.expectError(error.InvalidSyntax, parseStrictHexUsize("-1"));
        try std.testing.expectError(error.InvalidSyntax, parseStrictHexUsize("a_b"));
        try std.testing.expectError(error.InvalidSyntax, parseStrictHexUsize("0x10"));
        try std.testing.expectEqual(@as(usize, 0x1aF), try parseStrictHexUsize("1aF"));

        try std.testing.expectError(
            error.BadRequest,
            parseHttpHeaders("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: gzip, chunked\r\n\r\n"),
        );
        try std.testing.expectError(
            error.BadRequest,
            parseHttpHeaders("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: +5\r\n\r\nhello"),
        );
        try std.testing.expectError(
            error.BadRequest,
            parseHttpHeaders("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1_0\r\n\r\n0123456789"),
        );
        const ok_te = try parseHttpHeaders("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: Chunked\r\n\r\n");
        try std.testing.expect(ok_te.chunked);
    }

    // 3. Live server wire checks for F1 (413) & L4 (400).
    const Mock = struct {
        fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(1);
            _ = stream.emitTokenBytes("ok");
            stream.finish(.stop, 1);
        }
    };
    var dummy: u8 = 0;
    const srv = try Server.init(std.testing.allocator, .{
        .port = 0,
        .max_request_bytes = 2048,
        .max_conns = 4,
        .max_active_requests = 2,
        .apply_qos = false,
    });
    defer srv.deinit(std.testing.allocator);
    try srv.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var resp_buf: [4096]u8 = undefined;

    const r_overflow_fe = try testHttpExchange(
        port,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\nfffffffffffffffe\r\n0\r\n\r\n",
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, r_overflow_fe, "HTTP/1.1 413 Payload Too Large\r\n"));

    const r_overflow_ff = try testHttpExchange(
        port,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\nffffffffffffffff\r\n0\r\n\r\n",
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, r_overflow_ff, "HTTP/1.1 413 Payload Too Large\r\n"));

    const r_huge_chunk = try testHttpExchange(
        port,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n1000\r\npartial",
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, r_huge_chunk, "HTTP/1.1 413 Payload Too Large\r\n"));

    const r_bad_te = try testHttpExchange(
        port,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: gzip, chunked\r\n\r\n0\r\n\r\n",
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, r_bad_te, "HTTP/1.1 400 Bad Request\r\n"));

    const r_bad_chunk_hex = try testHttpExchange(
        port,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n+5\r\nhello\r\n0\r\n\r\n",
        &resp_buf,
    );
    try std.testing.expect(std.mem.startsWith(u8, r_bad_chunk_hex, "HTTP/1.1 400 Bad Request\r\n"));
}

test "F2 & L3 default CORS null, public-bind guard, Host DNS-rebinding rejection, early auth before 100-continue, header timeout, and peer cap" {
    const gpa = std.testing.allocator;
    const Mock = struct {
        fn generate(_: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            _ = stream.setPromptTokens(2);
            _ = stream.emitTokenBytes("ok");
            stream.finish(.stop, 2);
        }
    };
    var dummy: u8 = 0;

    // 1. Default `ServerConfig` disables CORS (`cors_allow_origin == null`) and rejects non-loopback unauthenticated bind.
    const def_cfg = ServerConfig{};
    try std.testing.expect(def_cfg.cors_allow_origin == null);
    try std.testing.expectError(error.InvalidConfig, Server.init(gpa, .{
        .host = .{ 0, 0, 0, 0 },
        .port = 0,
        .apply_qos = false,
    }));
    try std.testing.expectError(error.InvalidConfig, Server.init(gpa, .{
        .host = .{ 192, 168, 1, 50 },
        .port = 0,
        .apply_qos = false,
    }));

    // Explicit override `allow_unauthenticated_public_bind = true` permits bind.
    {
        const pub_srv = try Server.init(gpa, .{
            .host = .{ 0, 0, 0, 0 },
            .port = 0,
            .allow_unauthenticated_public_bind = true,
            .max_conns = 2,
            .max_active_requests = 1,
            .apply_qos = false,
        });
        pub_srv.deinit(gpa);
    }

    // 2. Host validation & default-off CORS on TCP + unrestricted Host on AF_UNIX + header_read_timeout_ms.
    var uds_buf: [128]u8 = undefined;
    var uw = FixedWriter.init(&uds_buf);
    try uw.print("/tmp/aminfer_f2_l3_{d}.sock", .{std.c.getpid()});
    const uds_path = uw.written();

    const srv = try Server.init(gpa, .{
        .port = 0,
        .unix_socket_path = uds_path,
        .header_read_timeout_ms = 50,
        .http_read_timeout_ms = 5_000,
        .max_conns = 6,
        .max_active_requests = 2,
        .max_conns_per_peer = 2,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try srv.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var resp_buf: [4096]u8 = undefined;

    // Default CORS is disabled.
    const r_def = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: localhost:8080\r\n\r\n", &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r_def, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expectEqual(@as(?[]const u8, null), extractHeaderValue(r_def, "Access-Control-Allow-Origin"));

    // Loopback Host variants allowed (`localhost`, `127.0.0.1`, `[::1]`, `0.0.0.0`).
    for ([_][]const u8{ "localhost", "127.0.0.1:8080", "[::1]:8080", "0.0.0.0" }) |h| {
        var req: [256]u8 = undefined;
        var rw = FixedWriter.init(&req);
        try rw.print("GET /health HTTP/1.1\r\nHost: {s}\r\n\r\n", .{h});
        const rh = try testHttpExchange(port, rw.written(), &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, rh, "HTTP/1.1 200 OK\r\n"));
    }

    // Foreign Host rejected with 403 against DNS rebinding; malformed/missing Host rejected with 400.
    const r_evil = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: evil.com\r\n\r\n", &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r_evil, "HTTP/1.1 403 Forbidden\r\n"));
    const r_sub = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: localhost.evil.com:8080\r\n\r\n", &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r_sub, "HTTP/1.1 403 Forbidden\r\n"));
    const r_nohost = try testHttpExchange(port, "GET /health HTTP/1.1\r\n\r\n", &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r_nohost, "HTTP/1.1 400 Bad Request\r\n"));

    // AF_UNIX IPC allows arbitrary Host header even without api_key.
    const r_unix = try testUnixExchange(uds_path, "GET /health HTTP/1.1\r\nHost: local-ipc\r\n\r\n", &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r_unix, "HTTP/1.1 200 OK\r\n"));

    // Short header_read_timeout_ms (50 ms) fires 408 promptly even though http_read_timeout_ms = 5000 ms!
    {
        const t0 = monotonicNowNs();
        const fd = try testConnectLoopback(port);
        defer _ = std.c.close(fd);
        try testSendAll(fd, "GET /health HTTP/1.1\r\nHost: localhost\r\n");
        const r_to = try testRecvAll(fd, &resp_buf);
        const elapsed_ms = (monotonicNowNs() -| t0) / std.time.ns_per_ms;
        try std.testing.expect(std.mem.startsWith(u8, r_to, "HTTP/1.1 408 Request Timeout\r\n"));
        try std.testing.expect(elapsed_ms < 1000);
    }

    // Per-peer cap ignores loopback (127.0.0.1) and caps non-loopback IPv4 peers at `max_conns_per_peer` (2).
    {
        srv.slots[0].state = .reading_request;
        srv.slots[0].peer_ipv4 = .{ 10, 20, 30, 40 };
        srv.slots[1].state = .reading_request;
        srv.slots[1].peer_ipv4 = .{ 10, 20, 30, 40 };
        defer {
            srv.slots[0].state = .free;
            srv.slots[0].peer_ipv4 = null;
            srv.slots[1].state = .free;
            srv.slots[1].peer_ipv4 = null;
        }
        try std.testing.expect(!srv.shouldRejectPeerLimit(.{ 127, 0, 0, 1 }));
        try std.testing.expect(srv.shouldRejectPeerLimit(.{ 10, 20, 30, 40 }));
        try std.testing.expect(!srv.shouldRejectPeerLimit(.{ 10, 20, 30, 41 }));
    }

    // 3. `allow_any_host = true` bypasses Host check, and early `--api-key` auth rejects BEFORE `100 Continue` & body buffering.
    const auth_srv = try Server.init(gpa, .{
        .port = 0,
        .api_key = "sk-early-auth",
        .allow_any_host = true,
        .max_conns = 4,
        .max_active_requests = 2,
        .apply_qos = false,
    });
    defer auth_srv.deinit(gpa);
    try auth_srv.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
    defer auth_srv.stop();

    {
        // Send ONLY headers with Expect: 100-continue + huge Content-Length (32768) and NO body and NO API key!
        // Server must reject immediately with 401 without waiting for body and without emitting 100 Continue.
        const fd = try testConnectLoopback(auth_srv.boundPort());
        defer _ = std.c.close(fd);
        try testSendAll(
            fd,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: evil.example\r\nExpect: 100-continue\r\nContent-Length: 32768\r\n\r\n",
        );
        const r_401 = try testRecvAll(fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_401, "HTTP/1.1 401 Unauthorized\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, r_401, "100 Continue") == null);
    }
    {
        // With valid API key + Expect: 100-continue, server sends 100 Continue then accepts the body and replies 200 OK.
        const fd = try testConnectLoopback(auth_srv.boundPort());
        defer _ = std.c.close(fd);
        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}";
        var hbuf: [256]u8 = undefined;
        var hw = FixedWriter.init(&hbuf);
        try hw.print(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: custom.local\r\nAuthorization: Bearer sk-early-auth\r\nExpect: 100-continue\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n",
            .{body.len},
        );
        try testSendAll(fd, hw.written());
        var cont_buf: [64]u8 = undefined;
        var got: usize = 0;
        while (got < STATIC_100_CONTINUE.len) {
            const n = std.c.recv(fd, cont_buf[got..].ptr, cont_buf.len - got, 0);
            try std.testing.expect(n > 0);
            got += @intCast(n);
        }
        try std.testing.expectEqualStrings(STATIC_100_CONTINUE, cont_buf[0..STATIC_100_CONTINUE.len]);
        try testSendAll(fd, body);
        const r_ok = try testRecvAll(fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, r_ok, "HTTP/1.1 200 OK\r\n"));
    }
}

test "F6 keep-alive slot waits for worker_active via .draining_worker on early stop-sequence completion without slot clobber" {
    const gpa = std.testing.allocator;
    const SlowTailMock = struct {
        req_count: std.atomic.Value(u32) = .init(0),
        req1_still_reading_prompt: std.atomic.Value(bool) = .init(false),
        saw_clobbered_prompt: std.atomic.Value(bool) = .init(false),

        fn generate(ctx_ptr: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            const n = self.req_count.fetchAdd(1, .acq_rel) + 1;
            if (n == 1) {
                const had_first = std.mem.indexOf(u8, job.prompt, "FIRST_REQ") != null;
                _ = stream.setPromptTokens(4);
                _ = stream.emitTokenBytes("prefix STOP_NOW tail");
                // Deliberately hold worker_active == true AFTER stop sequence triggers early completion on kqueue,
                // then re-read job.prompt & job.rng to verify Request #2 did NOT clobber the slot!
                self.req1_still_reading_prompt.store(true, .release);
                sleepNs(25 * std.time.ns_per_ms);
                const still_first = std.mem.indexOf(u8, job.prompt, "FIRST_REQ") != null;
                const has_second = std.mem.indexOf(u8, job.prompt, "SECOND_REQ") != null;
                if (!had_first or !still_first or has_second) {
                    self.saw_clobbered_prompt.store(true, .release);
                }
                _ = job.rng.nextU64();
                stream.finish(.stop, 4);
            } else {
                _ = stream.setPromptTokens(4);
                _ = stream.emitTokenBytes("second_ok");
                stream.finish(.stop, 4);
            }
        }
    };

    var mock = SlowTailMock{};
    const srv = try Server.init(gpa, .{
        .port = 0,
        .max_conns = 2,
        .max_active_requests = 1,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try srv.start(.{ .ctx = &mock, .generateFn = SlowTailMock.generate });
    defer srv.stop();

    const fd = try testConnectLoopback(srv.boundPort());
    defer _ = std.c.close(fd);

    const b1 = "{\"messages\":[{\"role\":\"user\",\"content\":\"FIRST_REQ\"}],\"stop\":[\"STOP_NOW\"],\"stream\":false}";
    const b2 = "{\"messages\":[{\"role\":\"user\",\"content\":\"SECOND_REQ\"}],\"stream\":false}";
    var req_buf: [1024]u8 = undefined;
    var w = FixedWriter.init(&req_buf);
    try w.print(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n{s}" ++
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
        .{ b1.len, b1, b2.len, b2 },
    );
    try testSendAll(fd, w.written());

    var resp_buf: [16384]u8 = undefined;
    const r1 = try testRecvOneHttpResponse(fd, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r1, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(std.mem.indexOf(u8, r1, "\"content\":\"prefix \"") != null);
    try std.testing.expect(std.mem.indexOf(u8, r1, "STOP_NOW") == null);

    const r2 = try testRecvOneHttpResponse(fd, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r2, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(std.mem.indexOf(u8, r2, "\"content\":\"second_ok\"") != null);
    try std.testing.expect(!mock.saw_clobbered_prompt.load(.acquire));
    try std.testing.expectEqual(@as(u32, 2), mock.req_count.load(.acquire));
}

test "F7 /v1/embeddings and /v1/rerank offload onto worker thread without kqueue model_mutex deadlock" {
    const gpa = std.testing.allocator;
    const SharedModelMock = struct {
        model_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
        entered_gen: std.atomic.Value(bool) = .init(false),
        release_gen: std.atomic.Value(bool) = .init(false),
        embed_on_worker: std.atomic.Value(bool) = .init(false),
        rerank_on_worker: std.atomic.Value(bool) = .init(false),
        worker_tid: std.atomic.Value(usize) = .init(0),

        fn generate(ctx_ptr: *anyopaque, _: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            self.worker_tid.store(@intFromPtr(std.c.pthread_self()), .release);
            _ = std.c.pthread_mutex_lock(&self.model_mutex);
            defer _ = std.c.pthread_mutex_unlock(&self.model_mutex);

            _ = stream.setPromptTokens(2);
            self.entered_gen.store(true, .release);
            // Hold model_mutex until cancelled by client disconnect or release flag.
            while (!self.release_gen.load(.acquire) and !stream.isCancelled()) {
                sleepNs(1_000_000);
            }
            if (stream.isCancelled()) return;
            _ = stream.emitTokenBytes("done");
            stream.finish(.stop, 2);
        }

        fn embed(ctx_ptr: *anyopaque, req: *const EmbedRequest, out_vectors: []f32) anyerror!EmbedResult {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            _ = std.c.pthread_mutex_lock(&self.model_mutex);
            defer _ = std.c.pthread_mutex_unlock(&self.model_mutex);
            const tid: usize = @intFromPtr(std.c.pthread_self());
            if (tid == self.worker_tid.load(.acquire)) {
                self.embed_on_worker.store(true, .release);
            }
            const dim: usize = if (req.dimensions) |d| @intCast(d) else 2;
            for (0..req.input_count) |i| {
                for (0..dim) |j| out_vectors[i * dim + j] = 0.5;
            }
            return .{ .dim = dim, .prompt_tokens = 3 };
        }

        fn rerank(ctx_ptr: *anyopaque, req: *const RerankRequest, out_items: []RerankItem) anyerror!RerankResult {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            _ = std.c.pthread_mutex_lock(&self.model_mutex);
            defer _ = std.c.pthread_mutex_unlock(&self.model_mutex);
            const tid: usize = @intFromPtr(std.c.pthread_self());
            if (tid == self.worker_tid.load(.acquire)) {
                self.rerank_on_worker.store(true, .release);
            }
            for (0..req.document_count) |i| {
                out_items[i] = .{ .index = @intCast(i), .relevance_score = 0.9 - @as(f32, @floatFromInt(i)) * 0.1 };
            }
            return .{ .count = req.document_count, .prompt_tokens = 4 };
        }
    };

    var mock = SharedModelMock{};
    const srv = try Server.init(gpa, .{
        .port = 0,
        .max_conns = 6,
        .max_active_requests = 2,
        .embedding_handler = .{ .ctx = &mock, .embedFn = SharedModelMock.embed },
        .rerank_handler = .{ .ctx = &mock, .rerankFn = SharedModelMock.rerank },
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try srv.start(.{ .ctx = &mock, .generateFn = SharedModelMock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [1024]u8 = undefined;
    var resp_buf: [8192]u8 = undefined;

    // 1. Start streaming request holding `model_mutex` on the worker thread.
    const s_fd = try testConnectLoopback(port);
    errdefer _ = std.c.close(s_fd);
    const s_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hold\"}],\"stream\":true}";
    var sw = FixedWriter.init(&req_buf);
    try sw.print("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ s_body.len, s_body });
    try testSendAll(s_fd, sw.written());

    var waited: usize = 0;
    while (!mock.entered_gen.load(.acquire) and waited < 400) : (waited += 1) {
        sleepNs(1_000_000);
    }
    try std.testing.expect(mock.entered_gen.load(.acquire));

    // 2. While worker holds `model_mutex`, open `/v1/embeddings` request on a second socket (non-blocking send).
    const e_fd = try testConnectLoopback(port);
    defer _ = std.c.close(e_fd);
    const e_body = "{\"input\":\"vec\",\"dimensions\":2}";
    var ew = FixedWriter.init(&req_buf);
    try ew.print("POST /v1/embeddings HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ e_body.len, e_body });
    try testSendAll(e_fd, ew.written());

    // 3. Even with `/v1/embeddings` queued and `model_mutex` locked by worker, kqueue responds to `GET /health` immediately!
    const h_resp = try testHttpExchange(port, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n", &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, h_resp, "HTTP/1.1 200 OK\r\n"));

    // 4. Disconnect stalled streaming client (`EV_EOF` handled by unblocked kqueue!), releasing worker & `model_mutex` -> `/v1/embeddings` completes!
    _ = std.c.close(s_fd);
    const e_resp = try testRecvAll(e_fd, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, e_resp, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(mock.embed_on_worker.load(.acquire));

    // 5. `/v1/rerank` also runs on the worker thread.
    const r_resp = try testHttpPostJson(port, "/v1/rerank", "{\"query\":\"q\",\"documents\":[\"d0\",\"d1\"]}", &req_buf, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r_resp, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(mock.rerank_on_worker.load(.acquire));
}

test "F24 & L8 depth-1 previous_response_id scan, >64B keys, lone JSON surrogates, U+FFFD escaping, and /responses cancel body/SSE" {
    const gpa = std.testing.allocator;

    // 1. L8 `JsonReader.skipString` rejects lone surrogates, accepts valid pair, and `readKeyOrSkip` skips >64 B keys.
    {
        var r_lone_hi = JsonReader{ .src = "\"\\uD800\"" };
        try std.testing.expectError(error.InvalidJson, r_lone_hi.skipString());

        var r_lone_lo = JsonReader{ .src = "\"\\uDC00\"" };
        try std.testing.expectError(error.InvalidJson, r_lone_lo.skipString());

        var r_bad_pair = JsonReader{ .src = "\"\\uD800\\u0041\"" };
        try std.testing.expectError(error.InvalidJson, r_bad_pair.skipString());

        var r_good_pair = JsonReader{ .src = "\"\\uD83D\\uDE80\"" };
        try r_good_pair.skipString();

        // > 64 B unknown key skipped cleanly without `error.InvalidJson` (Appendix A.6 parity).
        const long_key_json = "{\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\":123,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
        var pbuf: [1024]u8 = undefined;
        var mbuf: [MAX_MODEL_ID_BYTES]u8 = undefined;
        const parsed = try parseInferenceRequest(.chat_completions, long_key_json, "m", &pbuf, &mbuf);
        try std.testing.expect(parsed.prompt_len > 0);
    }

    // 2. L8 `writeJsonEscaped` replaces invalid UTF-8 bytes with U+FFFD (`\xEF\xBF\xBD`).
    {
        var out: [128]u8 = undefined;
        var w = FixedWriter.init(&out);
        try w.writeJsonEscaped("ok-\x80-\xE4\xBD-end");
        try std.testing.expect(std.unicode.utf8ValidateSlice(w.written()));
        try std.testing.expectEqualStrings("ok-\xEF\xBF\xBD-\xEF\xBF\xBD\xEF\xBF\xBD-end", w.written());
    }

    // 3. F24.1 `findTopLevelJsonStringField` ignores nested `"previous_response_id"` while matching top-level (even escaped or after >64 B key).
    {
        var dst: [MAX_RESPONSE_ID_BYTES]u8 = undefined;
        const nested_only =
            "{\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\":{\"previous_response_id\":\"resp_1\"}," ++
            "\"metadata\":{\"previous_response_id\":\"resp_1\"}," ++
            "\"input\":[{\"role\":\"user\",\"content\":\"{\\\"previous_response_id\\\":\\\"resp_1\\\"}\"}]}";
        try std.testing.expectEqual(@as(?[]const u8, null), findTopLevelJsonStringField(nested_only, "previous_response_id", &dst));

        const top_level_after_nested =
            "{\"metadata\":{\"previous_response_id\":\"resp_wrong\"},\"previous\\u005fresponse_id\":\"resp_right\"}";
        try std.testing.expectEqualStrings("resp_right", findTopLevelJsonStringField(top_level_after_nested, "previous_response_id", &dst).?);
    }

    // 4. Live server: F24.1 isolation + L8 `POST /v1/responses/{id}/cancel` sends valid cancelled JSON (`stream: false`) and `event: response.cancelled` (`stream: true`).
    const CancelMock = struct {
        hold_open: std.atomic.Value(bool) = .init(false),
        entered_hold: std.atomic.Value(bool) = .init(false),
        saw_secret: std.atomic.Value(bool) = .init(false),

        fn generate(ctx_ptr: *anyopaque, job: *const InferenceJob, stream: *TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            if (std.mem.indexOf(u8, job.prompt, "TOP_SECRET_TURN1") != null) {
                self.saw_secret.store(true, .release);
            }
            _ = stream.setPromptTokens(4);
            if (self.hold_open.load(.acquire)) {
                self.entered_hold.store(true, .release);
                while (!stream.isCancelled()) {
                    sleepNs(1_000_000);
                }
                return;
            }
            _ = stream.emitTokenBytes("TOP_SECRET_TURN1");
            stream.finish(.stop, 4);
        }
    };

    var mock = CancelMock{};
    const srv = try Server.init(gpa, .{
        .port = 0,
        .max_conns = 6,
        .max_active_requests = 2,
        .apply_qos = false,
    });
    defer srv.deinit(gpa);
    try srv.start(.{ .ctx = &mock, .generateFn = CancelMock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [2048]u8 = undefined;
    var resp_buf: [16384]u8 = undefined;

    // Seed resp_1 with TOP_SECRET_TURN1.
    const r1 = try testHttpPostJson(port, "/v1/responses", "{\"input\":\"turn1\",\"stream\":false}", &req_buf, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r1, "HTTP/1.1 200 OK\r\n"));

    // Send nested `"previous_response_id":"resp_1"` inside metadata/input WITHOUT top-level previous_response_id -> must NOT leak resp_1 transcript!
    const nested_req =
        \\{"metadata":{"previous_response_id":"resp_1"},"input":[{"role":"user","content":"{\"previous_response_id\":\"resp_1\"} hi"}],"stream":false}
    ;
    const r2 = try testHttpPostJson(port, "/v1/responses", nested_req, &req_buf, &resp_buf);
    try std.testing.expect(std.mem.startsWith(u8, r2, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(!mock.saw_secret.load(.acquire));

    // Non-streaming `/v1/responses` cancelled via `POST /v1/responses/resp_3/cancel` -> original requester receives valid JSON `"status":"cancelled"`.
    mock.hold_open.store(true, .release);
    {
        mock.entered_hold.store(false, .release);
        const orig_fd = try testConnectLoopback(port);
        defer _ = std.c.close(orig_fd);
        const body = "{\"input\":\"wait\",\"stream\":false}";
        var w = FixedWriter.init(&req_buf);
        try w.print("POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ body.len, body });
        try testSendAll(orig_fd, w.written());

        var waited: usize = 0;
        while (!mock.entered_hold.load(.acquire) and waited < 400) : (waited += 1) sleepNs(1_000_000);
        try std.testing.expect(mock.entered_hold.load(.acquire));

        const c_resp = try testHttpExchange(port, "POST /v1/responses/resp_3/cancel HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, c_resp, "HTTP/1.1 200 OK\r\n"));

        const orig_resp = try testRecvAll(orig_fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, orig_resp, "HTTP/1.1 200 OK\r\n"));
        const orig_json = extractHttpBody(orig_resp) orelse return error.TestUnexpectedResult;
        try expectValidJson(orig_json);
        try std.testing.expect(std.mem.indexOf(u8, orig_json, "\"status\":\"cancelled\"") != null);
    }

    // Streaming `/v1/responses` cancelled via `POST /v1/responses/resp_4/cancel` -> original requester receives `event: response.cancelled`.
    {
        mock.entered_hold.store(false, .release);
        const orig_fd = try testConnectLoopback(port);
        defer _ = std.c.close(orig_fd);
        const body = "{\"input\":\"wait-sse\",\"stream\":true}";
        var w = FixedWriter.init(&req_buf);
        try w.print("POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ body.len, body });
        try testSendAll(orig_fd, w.written());

        var waited: usize = 0;
        while (!mock.entered_hold.load(.acquire) and waited < 400) : (waited += 1) sleepNs(1_000_000);
        try std.testing.expect(mock.entered_hold.load(.acquire));

        const c_resp = try testHttpExchange(port, "POST /v1/responses/resp_4/cancel HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n", &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, c_resp, "HTTP/1.1 200 OK\r\n"));

        const orig_sse = try testRecvAll(orig_fd, &resp_buf);
        try std.testing.expect(std.mem.startsWith(u8, orig_sse, "HTTP/1.1 200 OK\r\n"));
        try std.testing.expect(std.mem.indexOf(u8, orig_sse, "event: response.cancelled\n") != null);
        try std.testing.expect(std.mem.indexOf(u8, orig_sse, "\"status\":\"cancelled\"") != null);
    }
}
