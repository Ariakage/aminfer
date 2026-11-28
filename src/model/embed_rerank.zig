//! Native model-side embeddings and cross-encoder reranking primitives (`C21`, `ADR-0022`).
//!
//! Designed for zero steady-state heap allocation on the hot path:
//! - Pooling (`last_token`, `mean` via streaming `MeanPoolAccumulator`, `cls`)
//! - Matryoshka `dimensions` prefix slicing
//! - L2 unit normalization (NaN-safe on zero vectors)
//! - Wire encoding helpers (`float` JSON array & OpenAI `base64` IEEE-754 f32)
//! - Cross-encoder rerank logit scoring (`yes`/`no` softmax probability, logit
//!   diff, or single-token logit) and deterministic descending ranking.

const std = @import("std");

pub const PoolingMethod = enum(u8) {
    last_token = 0,
    mean = 1,
    cls = 2,

    pub fn label(self: PoolingMethod) []const u8 {
        return switch (self) {
            .last_token => "last_token",
            .mean => "mean",
            .cls => "cls",
        };
    }

    pub fn parse(raw: []const u8) error{InvalidOption}!PoolingMethod {
        if (std.mem.eql(u8, raw, "last_token") or std.mem.eql(u8, raw, "last")) return .last_token;
        if (std.mem.eql(u8, raw, "mean") or std.mem.eql(u8, raw, "avg")) return .mean;
        if (std.mem.eql(u8, raw, "cls") or std.mem.eql(u8, raw, "first")) return .cls;
        return error.InvalidOption;
    }
};

pub const EncodingFormat = enum(u8) {
    float = 0,
    base64 = 1,

    pub fn label(self: EncodingFormat) []const u8 {
        return switch (self) {
            .float => "float",
            .base64 => "base64",
        };
    }

    pub fn parse(raw: []const u8) error{InvalidOption}!EncodingFormat {
        if (std.mem.eql(u8, raw, "float")) return .float;
        if (std.mem.eql(u8, raw, "base64")) return .base64;
        return error.InvalidOption;
    }
};

pub const EmbedOptions = struct {
    pooling: PoolingMethod = .last_token,
    /// Optional Matryoshka prefix dimension (`1 <= d <= hidden_size`); `null` uses full `hidden_size`.
    dimensions: ?u32 = null,
    /// L2-normalize the output vector after pooling and Matryoshka prefix slicing.
    normalize: bool = true,
    /// Reset KV cache before embedding the input sequence.
    reset_kv_before: bool = true,
};

pub const EmbedError = error{
    EmptyInput,
    InvalidDimensions,
    BufferTooSmall,
    InvalidOption,
    InvalidBase64,
    TokenOutOfRange,
};

pub fn effectiveDimensions(hidden_size: u32, dimensions: ?u32) EmbedError!u32 {
    if (hidden_size == 0) return error.InvalidDimensions;
    const dim = dimensions orelse hidden_size;
    if (dim == 0 or dim > hidden_size) return error.InvalidDimensions;
    return dim;
}

/// Normalizes `vec` to unit L2 norm in-place and returns the pre-normalization
/// L2 magnitude (`0.0` leaves `vec` zeroed without NaN).
pub fn l2NormalizeInPlace(vec: []f32) f32 {
    var sum_sq: f64 = 0.0;
    for (vec) |v| {
        const vf: f64 = v;
        sum_sq += vf * vf;
    }
    const norm: f64 = @sqrt(sum_sq);
    if (!(norm > 0.0) or !std.math.isFinite(norm)) return 0.0;
    const inv: f32 = @floatCast(1.0 / norm);
    for (vec) |*v| v.* *= inv;
    return @floatCast(norm);
}

/// Zero-allocation pooling, Matryoshka truncation, and optional L2 normalization
/// over `[seq_len, hidden_size]` `f16` final-RMSNorm hidden states.
pub fn poolHiddenF16(
    hidden_rows: []const f16,
    seq_len: usize,
    hidden_size: usize,
    opts: EmbedOptions,
    out: []f32,
) EmbedError![]f32 {
    if (seq_len == 0) return error.EmptyInput;
    if (hidden_size == 0 or hidden_size > std.math.maxInt(u32)) return error.InvalidDimensions;
    const eff_dim: usize = try effectiveDimensions(@intCast(hidden_size), opts.dimensions);
    if (out.len < eff_dim) return error.BufferTooSmall;
    if (hidden_rows.len < seq_len * hidden_size) return error.BufferTooSmall;

    const dst = out[0..eff_dim];
    switch (opts.pooling) {
        .cls => {
            const row0 = hidden_rows[0..eff_dim];
            for (0..eff_dim) |d| dst[d] = @floatCast(row0[d]);
        },
        .last_token => {
            const base = (seq_len - 1) * hidden_size;
            const row_last = hidden_rows[base .. base + eff_dim];
            for (0..eff_dim) |d| dst[d] = @floatCast(row_last[d]);
        },
        .mean => {
            @memset(dst, 0.0);
            for (0..seq_len) |t| {
                const row = hidden_rows[t * hidden_size ..][0..eff_dim];
                for (0..eff_dim) |d| dst[d] += @floatCast(row[d]);
            }
            const inv: f32 = 1.0 / @as(f32, @floatFromInt(seq_len));
            for (dst) |*v| v.* *= inv;
        },
    }
    if (opts.normalize) {
        _ = l2NormalizeInPlace(dst);
    }
    return dst;
}

/// Zero-allocation streaming chunk accumulator for multi-chunk `G15` wide prefill
/// `mean` pooling across arbitrary sequence lengths (`> wide_prefill_chunk_tokens`).
pub const MeanPoolAccumulator = struct {
    out: []f32,
    eff_dim: usize,
    total_tokens: usize,

    pub fn init(out_buf: []f32, eff_dim: usize) EmbedError!MeanPoolAccumulator {
        if (eff_dim == 0) return error.InvalidDimensions;
        if (out_buf.len < eff_dim) return error.BufferTooSmall;
        const dst = out_buf[0..eff_dim];
        @memset(dst, 0.0);
        return .{
            .out = dst,
            .eff_dim = eff_dim,
            .total_tokens = 0,
        };
    }

    pub fn addChunkF16(self: *MeanPoolAccumulator, chunk_rows: []const f16, num_rows: usize, hidden_size: usize) EmbedError!void {
        if (hidden_size < self.eff_dim) return error.InvalidDimensions;
        if (chunk_rows.len < num_rows * hidden_size) return error.BufferTooSmall;
        for (0..num_rows) |r| {
            const row = chunk_rows[r * hidden_size ..][0..self.eff_dim];
            for (0..self.eff_dim) |d| {
                self.out[d] += @floatCast(row[d]);
            }
        }
        self.total_tokens += num_rows;
    }

    pub fn finish(self: *MeanPoolAccumulator, normalize: bool) EmbedError![]f32 {
        if (self.total_tokens == 0) return error.EmptyInput;
        const inv: f32 = 1.0 / @as(f32, @floatFromInt(self.total_tokens));
        for (self.out) |*v| v.* *= inv;
        if (normalize) {
            _ = l2NormalizeInPlace(self.out);
        }
        return self.out;
    }
};

pub fn cosineSimilarity(a: []const f32, b: []const f32) EmbedError!f32 {
    if (a.len == 0 or a.len != b.len) return error.InvalidDimensions;
    var dot: f64 = 0.0;
    var na: f64 = 0.0;
    var nb: f64 = 0.0;
    for (a, b) |va, vb| {
        const fa: f64 = va;
        const fb: f64 = vb;
        dot += fa * fb;
        na += fa * fa;
        nb += fb * fb;
    }
    const denom = @sqrt(na * nb);
    if (!(denom > 0.0) or !std.math.isFinite(denom)) return 0.0;
    return @floatCast(dot / denom);
}

// ===========================================================================
// Wire Encoding Helpers (`float` JSON array & `base64` IEEE-754 LE f32)
// ===========================================================================

pub fn base64EncodedLen(num_floats: usize) usize {
    return std.base64.standard.Encoder.calcSize(num_floats * @sizeOf(f32));
}

/// Encodes `vec` as standard Base64 over little-endian IEEE-754 `f32` bytes
/// (OpenAI `POST /v1/embeddings` `encoding_format: "base64"`) with ZERO heap
/// allocation, streaming in 3-float (12-byte -> 16-char) chunks.
pub fn encodeBase64F32(vec: []const f32, dst: []u8) EmbedError![]const u8 {
    const needed = base64EncodedLen(vec.len);
    if (dst.len < needed) return error.BufferTooSmall;
    if (vec.len == 0) return dst[0..0];

    var raw_chunk: [12]u8 = undefined; // 3 floats = 12 bytes (multiple of 3 -> no mid-stream padding)
    var in_pos: usize = 0;
    var out_pos: usize = 0;
    while (in_pos + 3 <= vec.len) : (in_pos += 3) {
        for (0..3) |i| {
            std.mem.writeInt(u32, raw_chunk[i * 4 ..][0..4], @bitCast(vec[in_pos + i]), .little);
        }
        const encoded = std.base64.standard.Encoder.encode(dst[out_pos .. out_pos + 16], &raw_chunk);
        out_pos += encoded.len;
    }
    const rem = vec.len - in_pos;
    if (rem > 0) {
        for (0..rem) |i| {
            std.mem.writeInt(u32, raw_chunk[i * 4 ..][0..4], @bitCast(vec[in_pos + i]), .little);
        }
        const rem_bytes = raw_chunk[0 .. rem * 4];
        const rem_out_len = std.base64.standard.Encoder.calcSize(rem_bytes.len);
        const encoded = std.base64.standard.Encoder.encode(dst[out_pos .. out_pos + rem_out_len], rem_bytes);
        out_pos += encoded.len;
    }
    return dst[0..out_pos];
}

/// Decodes OpenAI `base64` IEEE-754 little-endian `f32` embedding into `dst` (`0` heap allocs).
pub fn decodeBase64F32(b64: []const u8, dst: []f32) EmbedError![]f32 {
    const raw_len = std.base64.standard.Decoder.calcSizeForSlice(b64) catch return error.InvalidBase64;
    if ((raw_len % @sizeOf(f32)) != 0) return error.InvalidBase64;
    const num_floats = raw_len / @sizeOf(f32);
    if (dst.len < num_floats) return error.BufferTooSmall;
    if (num_floats == 0) return dst[0..0];

    var raw_chunk: [12]u8 = undefined;
    var in_pos: usize = 0;
    var out_idx: usize = 0;
    while (in_pos + 16 <= b64.len) : (in_pos += 16) {
        std.base64.standard.Decoder.decode(&raw_chunk, b64[in_pos .. in_pos + 16]) catch return error.InvalidBase64;
        for (0..3) |i| {
            const bits = std.mem.readInt(u32, raw_chunk[i * 4 ..][0..4], .little);
            dst[out_idx] = @bitCast(bits);
            out_idx += 1;
        }
    }
    if (in_pos < b64.len) {
        const tail = b64[in_pos..];
        const tail_raw_len = std.base64.standard.Decoder.calcSizeForSlice(tail) catch return error.InvalidBase64;
        if (tail_raw_len > raw_chunk.len or (tail_raw_len % 4) != 0) return error.InvalidBase64;
        std.base64.standard.Decoder.decode(raw_chunk[0..tail_raw_len], tail) catch return error.InvalidBase64;
        const tail_floats = tail_raw_len / 4;
        for (0..tail_floats) |i| {
            const bits = std.mem.readInt(u32, raw_chunk[i * 4 ..][0..4], .little);
            dst[out_idx] = @bitCast(bits);
            out_idx += 1;
        }
    }
    return dst[0..out_idx];
}

/// Formats `vec` as a compact JSON number array (`[0.123456,-0.5]`) into `dst` with ZERO heap allocs.
pub fn formatFloatArrayJson(vec: []const f32, dst: []u8) EmbedError![]const u8 {
    var pos: usize = 0;
    if (dst.len < 2) return error.BufferTooSmall;
    dst[pos] = '[';
    pos += 1;
    for (vec, 0..) |v, i| {
        if (i > 0) {
            if (pos >= dst.len) return error.BufferTooSmall;
            dst[pos] = ',';
            pos += 1;
        }
        const safe_v: f32 = if (std.math.isFinite(v)) v else 0.0;
        const written = std.fmt.bufPrint(dst[pos..], "{d:.6}", .{safe_v}) catch return error.BufferTooSmall;
        pos += written.len;
    }
    if (pos >= dst.len) return error.BufferTooSmall;
    dst[pos] = ']';
    pos += 1;
    return dst[0..pos];
}

// ===========================================================================
// Cross-Encoder & Bi-Encoder Reranking (`C21`)
// ===========================================================================

pub const RerankScoringMode = enum(u8) {
    /// Binary cross-encoder `sigmoid((logit[pos] - logit[neg]) / T)` (e.g., Qwen3-Reranker yes/no).
    yes_no_sigmoid = 0,
    /// Raw scaled logit margin `(logit[pos] - logit[neg]) / T`.
    yes_no_logit_diff = 1,
    /// Single target token logit `logit[pos] / T`.
    single_token_logit = 2,
    /// Single target token probability `sigmoid(logit[pos] / T)`.
    single_token_sigmoid = 3,
};

pub const RerankConfig = struct {
    mode: RerankScoringMode = .yes_no_sigmoid,
    positive_token_id: u32 = 0,
    negative_token_id: u32 = 0,
    temperature: f32 = 1.0,
    top_n: ?u32 = null,
};

pub const RerankScore = struct {
    index: u32,
    score: f32,
    positive_logit: f32 = 0.0,
    negative_logit: f32 = 0.0,
};

fn stableSigmoid(x: f64) f32 {
    if (std.math.isNan(x)) return 0.0;
    if (x >= 0.0) {
        const z = @exp(-x);
        return @floatCast(1.0 / (1.0 + z));
    } else {
        const z = @exp(x);
        return @floatCast(z / (1.0 + z));
    }
}

fn safeLogitDiff(pos_logit: f32, neg_logit: f32) f64 {
    if (std.math.isNan(pos_logit) or std.math.isNan(neg_logit)) return std.math.nan(f64);
    // Equal infinities ((-Inf) - (-Inf) or (+Inf) - (+Inf)) have zero margin rather than NaN.
    if (!std.math.isFinite(pos_logit) and !std.math.isFinite(neg_logit) and pos_logit == neg_logit) {
        return 0.0;
    }
    return @as(f64, pos_logit) - @as(f64, neg_logit);
}

/// Computes a cross-encoder relevance score from raw `f16` final-token logits (`0` heap allocs).
pub fn scoreCrossEncoderLogitsF16(
    logits: []const f16,
    doc_index: u32,
    cfg: RerankConfig,
) EmbedError!RerankScore {
    if (logits.len == 0) return error.EmptyInput;
    if (!(cfg.temperature > 0.0) or !std.math.isFinite(cfg.temperature)) return error.InvalidOption;
    if (cfg.positive_token_id >= logits.len) return error.TokenOutOfRange;

    const pos_logit: f32 = @floatCast(logits[cfg.positive_token_id]);
    var neg_logit: f32 = 0.0;
    const inv_t: f64 = 1.0 / @as(f64, cfg.temperature);

    const raw_score: f32 = switch (cfg.mode) {
        .yes_no_sigmoid => blk: {
            if (cfg.negative_token_id >= logits.len) return error.TokenOutOfRange;
            neg_logit = @floatCast(logits[cfg.negative_token_id]);
            break :blk stableSigmoid(safeLogitDiff(pos_logit, neg_logit) * inv_t);
        },
        .yes_no_logit_diff => blk: {
            if (cfg.negative_token_id >= logits.len) return error.TokenOutOfRange;
            neg_logit = @floatCast(logits[cfg.negative_token_id]);
            break :blk @floatCast(safeLogitDiff(pos_logit, neg_logit) * inv_t);
        },
        .single_token_logit => @floatCast(@as(f64, pos_logit) * inv_t),
        .single_token_sigmoid => stableSigmoid(@as(f64, pos_logit) * inv_t),
    };
    const score: f32 = if (std.math.isNan(raw_score)) 0.0 else raw_score;

    return .{
        .index = doc_index,
        .score = score,
        .positive_logit = pos_logit,
        .negative_logit = neg_logit,
    };
}

fn rerankScoreDesc(_: void, a: RerankScore, b: RerankScore) bool {
    const a_nan = std.math.isNan(a.score);
    const b_nan = std.math.isNan(b.score);
    if (a_nan != b_nan) return !a_nan;
    if (a_nan) return a.index < b.index;
    if (a.score != b.score) return a.score > b.score;
    return a.index < b.index;
}

/// Deterministic in-place descending relevance sort (ties broken by ascending `index`).
pub fn sortRerankScoresDesc(scores: []RerankScore) void {
    std.mem.sort(RerankScore, scores, {}, rerankScoreDesc);
}

/// Bi-encoder cosine reranking helper (`0` heap allocs).
pub fn rerankBiEncoderEmbeddings(
    query_emb: []const f32,
    doc_embs: []const []const f32,
    top_n: ?u32,
    out_scores: []RerankScore,
) EmbedError![]RerankScore {
    if (out_scores.len < doc_embs.len) return error.BufferTooSmall;
    for (doc_embs, 0..) |doc, i| {
        const sim = try cosineSimilarity(query_emb, doc);
        out_scores[i] = .{ .index = @intCast(i), .score = sim };
    }
    const filled = out_scores[0..doc_embs.len];
    sortRerankScoresDesc(filled);
    if (top_n) |k| {
        const take: usize = @min(@as(usize, k), filled.len);
        return filled[0..take];
    }
    return filled;
}

// ===========================================================================
// Unit Tests
// ===========================================================================

test "C21 pooling (last_token, mean, cls), Matryoshka truncation, L2 norm, and base64/float encoding" {
    // 3 tokens x 4 hidden dims
    const rows = [_]f16{
        @floatCast(3.0), @floatCast(4.0), @floatCast(0.0), @floatCast(0.0), // tok 0 (cls)
        @floatCast(0.0), @floatCast(6.0), @floatCast(8.0), @floatCast(0.0), // tok 1
        @floatCast(0.0), @floatCast(0.0), @floatCast(5.0), @floatCast(12.0), // tok 2 (last)
    };
    var out: [4]f32 = undefined;

    // 1. cls + L2 norm: [3, 4, 0, 0] -> [0.6, 0.8, 0, 0]
    const cls_v = try poolHiddenF16(&rows, 3, 4, .{ .pooling = .cls, .normalize = true }, &out);
    try std.testing.expectEqual(@as(usize, 4), cls_v.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), cls_v[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), cls_v[1], 1e-4);

    // 2. last_token + Matryoshka dimensions=3 without norm -> [0, 0, 5], then with norm -> [0, 0, 1]
    const last_raw = try poolHiddenF16(&rows, 3, 4, .{ .pooling = .last_token, .dimensions = 3, .normalize = false }, &out);
    try std.testing.expectEqual(@as(usize, 3), last_raw.len);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), last_raw[2], 1e-4);
    const last_norm = try poolHiddenF16(&rows, 3, 4, .{ .pooling = .last_token, .dimensions = 3, .normalize = true }, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), last_norm[2], 1e-4);

    // 3. mean pooling across chunks via MeanPoolAccumulator matches one-shot poolHiddenF16.
    var mean_ref: [4]f32 = undefined;
    _ = try poolHiddenF16(&rows, 3, 4, .{ .pooling = .mean, .dimensions = 4, .normalize = true }, &mean_ref);
    var acc_buf: [4]f32 = undefined;
    var acc = try MeanPoolAccumulator.init(&acc_buf, 4);
    try acc.addChunkF16(rows[0..8], 2, 4);
    try acc.addChunkF16(rows[8..12], 1, 4);
    const mean_stream = try acc.finish(true);
    for (0..4) |d| {
        try std.testing.expectApproxEqAbs(mean_ref[d], mean_stream[d], 1e-5);
    }

    // 4. Base64 & JSON wire encoding round-trip (non-multiple-of-3 float count = 4).
    var b64_buf: [64]u8 = undefined;
    const b64 = try encodeBase64F32(mean_stream, &b64_buf);
    var decoded: [4]f32 = undefined;
    const dec_slice = try decodeBase64F32(b64, &decoded);
    try std.testing.expectEqual(@as(usize, 4), dec_slice.len);
    for (0..4) |d| {
        try std.testing.expectEqual(mean_stream[d], dec_slice[d]);
    }

    var json_buf: [128]u8 = undefined;
    const json_arr = try formatFloatArrayJson(&.{ 0.5, -1.25 }, &json_buf);
    try std.testing.expectEqualStrings("[0.500000,-1.250000]", json_arr);
}

test "C21 cross-encoder rerank logit scoring and deterministic top_n sorting" {
    const logits_doc0 = [_]f16{ @floatCast(-1.0), @floatCast(2.0), @floatCast(0.5) };
    const logits_doc1 = [_]f16{ @floatCast(0.0), @floatCast(-2.0), @floatCast(3.0) };
    const logits_doc2 = [_]f16{ @floatCast(0.0), @floatCast(5.0), @floatCast(1.0) };
    const cfg: RerankConfig = .{
        .mode = .yes_no_sigmoid,
        .positive_token_id = 1,
        .negative_token_id = 2,
    };

    var scores = [_]RerankScore{
        try scoreCrossEncoderLogitsF16(&logits_doc0, 0, cfg),
        try scoreCrossEncoderLogitsF16(&logits_doc1, 1, cfg),
        try scoreCrossEncoderLogitsF16(&logits_doc2, 2, cfg),
    };
    sortRerankScoresDesc(&scores);

    // doc2 (diff=+4.0) > doc0 (diff=+1.5) > doc1 (diff=-5.0)
    try std.testing.expectEqual(@as(u32, 2), scores[0].index);
    try std.testing.expectEqual(@as(u32, 0), scores[1].index);
    try std.testing.expectEqual(@as(u32, 1), scores[2].index);
    try std.testing.expect(scores[0].score > 0.98);
    try std.testing.expect(scores[1].score > 0.80 and scores[1].score < 0.85);
    try std.testing.expect(scores[2].score < 0.01);
}

test "L8 cross-encoder rerank guards (-Inf) - (-Inf) / NaN and sorts NaN deterministically last" {
    const neg_inf: f16 = -std.math.inf(f16);
    const pos_inf: f16 = std.math.inf(f16);
    const nan_f16: f16 = std.math.nan(f16);

    const logits_neg_inf = [_]f16{ neg_inf, neg_inf };
    const logits_pos_inf = [_]f16{ pos_inf, pos_inf };
    const logits_nan = [_]f16{ nan_f16, @floatCast(1.0) };

    const s_neg_sig = try scoreCrossEncoderLogitsF16(&logits_neg_inf, 0, .{
        .mode = .yes_no_sigmoid,
        .positive_token_id = 0,
        .negative_token_id = 1,
    });
    try std.testing.expect(!std.math.isNan(s_neg_sig.score));
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), s_neg_sig.score, 1e-6);

    const s_neg_diff = try scoreCrossEncoderLogitsF16(&logits_neg_inf, 1, .{
        .mode = .yes_no_logit_diff,
        .positive_token_id = 0,
        .negative_token_id = 1,
    });
    try std.testing.expect(!std.math.isNan(s_neg_diff.score));
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), s_neg_diff.score, 1e-6);

    const s_pos_sig = try scoreCrossEncoderLogitsF16(&logits_pos_inf, 2, .{
        .mode = .yes_no_sigmoid,
        .positive_token_id = 0,
        .negative_token_id = 1,
    });
    try std.testing.expect(!std.math.isNan(s_pos_sig.score));
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), s_pos_sig.score, 1e-6);

    const s_nan_sig = try scoreCrossEncoderLogitsF16(&logits_nan, 3, .{
        .mode = .yes_no_sigmoid,
        .positive_token_id = 0,
        .negative_token_id = 1,
    });
    try std.testing.expect(!std.math.isNan(s_nan_sig.score));
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), s_nan_sig.score, 1e-6);

    const s_nan_single = try scoreCrossEncoderLogitsF16(&logits_nan, 4, .{
        .mode = .single_token_logit,
        .positive_token_id = 0,
    });
    try std.testing.expect(!std.math.isNan(s_nan_single.score));
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), s_nan_single.score, 1e-6);

    var mixed = [_]RerankScore{
        .{ .index = 5, .score = std.math.nan(f32) },
        .{ .index = 1, .score = 0.25 },
        .{ .index = 3, .score = std.math.nan(f32) },
        .{ .index = 0, .score = 0.90 },
        .{ .index = 4, .score = -std.math.inf(f32) },
        .{ .index = 2, .score = 0.25 },
    };
    sortRerankScoresDesc(&mixed);
    try std.testing.expectEqual(@as(u32, 0), mixed[0].index);
    try std.testing.expectEqual(@as(u32, 1), mixed[1].index);
    try std.testing.expectEqual(@as(u32, 2), mixed[2].index);
    try std.testing.expectEqual(@as(u32, 4), mixed[3].index);
    try std.testing.expectEqual(@as(u32, 3), mixed[4].index);
    try std.testing.expectEqual(@as(u32, 5), mixed[5].index);
    try std.testing.expect(std.math.isNan(mixed[4].score));
    try std.testing.expect(std.math.isNan(mixed[5].score));
}
