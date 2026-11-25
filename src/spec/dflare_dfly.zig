//! D14 Block-Parallel Speculative Drafters: `DFlare` (`Inkling 2`, `3L`/`5L`,
//! `block_size = 8`, `markov_rank = 256`, standalone `arch = dflash`) and
//! `DFly2` (`Hy4`, `5L`, `block_size = 7`, `dfly_target_kv_layers`,
//! `hidden_correction`, `mtp_fusion`, `confidence_head` early-stop).
//!
//! Both proposers preallocate all activation, attention, Markov state, logit,
//! log-prob, confidence, and L1 SAM buffers once inside 16 KiB-aligned
//! `WiredArena` workspaces (`DFlareWorkspace` / `DFly2Workspace`), ensuring:
//! - Strictly zero dynamic heap allocations on the speculative draft hot path.
//! - Direct `DraftTree` (`src/spec/tree.zig`, `G5` `u64` ancestor masks) output
//!   with token log-probs, `D7` expert-byte-budget pruning (`ADR-0009`),
//!   `M = 8` batched `Core x 3` loop verification accounting, and lossless
//!   `D9` greedy (`T = 0`) and stochastic (`T > 0`) verification (`accept.zig`).

const std = @import("std");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const quant = @import("../quant/spec.zig");
const quant_formats = @import("../gpu/quant_formats.zig");
const amf = @import("../format/amf.zig");
const sam_mod = @import("sam.zig");
const tree_mod = @import("tree.zig");
const grammar_mod = @import("grammar.zig");
const accept_mod = @import("accept.zig");

/// Default DFlare parallel block length (`Inkling 2` Mini & Small).
pub const DFLARE_BLOCK_SIZE: u8 = 8;
/// Default DFlare low-rank `rnn_hidden` Markov head bottleneck dimension.
pub const DFLARE_MARKOV_RANK: u32 = 256;
/// Default DFly2 parallel block length (`Hy4` 5L).
pub const DFLY2_BLOCK_SIZE: u8 = 7;
/// Default DFly2 layer count (`Hy4`).
pub const DFLY2_DEFAULT_LAYERS: u8 = 5;

pub const MAX_BLOCK_SIZE: u8 = 16;
pub const MAX_LAYERS: usize = 8;
pub const MAX_TARGET_TAPS: usize = 8;
pub const MAX_TOP_K_PER_POS: u8 = 8;
pub const MAX_TREE_NODES: u8 = @intCast(tree_mod.MAX_SUPPORTED_NODES);
pub const MAX_L1_DRAFT_LEN: usize = 32;
pub const INKLING2_CORE_PASSES: u8 = 3;

const CACHELINE_ALIGN: usize = 64;

pub const CandidateTree = tree_mod.DefaultTree;

pub const Error = arena_mod.WiredArena.InitError ||
    arena_mod.WiredArena.AllocError ||
    sysinfo.Error ||
    tree_mod.PruneError ||
    accept_mod.VerifyError ||
    error{
        InvalidConfig,
        InvalidWeights,
        DimensionMismatch,
        WorkspaceMismatch,
        CallbackFailed,
        InvalidLogits,
        Overflow,
    };

/// Unified speculative draft source descriptor (`D14`, `ADR-0024`).
pub const DraftSourceKind = enum(u8) {
    l1_sam = 0,
    dflare_3l = 1,
    dflare_5l = 2,
    dflash_standalone = 3,
    dfly2_5l = 4,
};

pub const DraftSource = union(DraftSourceKind) {
    l1_sam: *const sam_mod.SuffixAutomaton,
    dflare_3l: *DFlareProposer,
    dflare_5l: *DFlareProposer,
    dflash_standalone: *DFlareProposer,
    dfly2_5l: *DFly2Proposer,
};

/// DFlare architecture family variant (`Inkling-2-Mini 3L`, `Small 5L`, `dflash`).
pub const DFlareVariant = enum(u8) {
    inkling2_mini_3l = 0,
    inkling2_small_5l = 1,
    dflash_standalone = 2,
};

/// Callback allowing GPU target `lm_head` (or tied Metal GEMM) to score `[batch, hidden]`
/// hidden states into `[batch, vocab_size]` FP32 logits without heap copies.
pub const LogitCallback = struct {
    ctx: ?*anyopaque = null,
    eval_fn: *const fn (
        ctx: ?*anyopaque,
        hiddens: []const f32,
        batch_rows: u8,
        hidden_size: u32,
        vocab_size: u32,
        out_logits: []f32,
    ) Error!void,
};

/// Output vocabulary projection interface (`lm_head`).
pub const LmHeadHook = union(enum) {
    dense_f32: []const f32,
    dense_f16: []const f16,
    dense_bf16: []const u16,
    callback: LogitCallback,
};

/// Zero-copy token-embedding lookup callback (shares target `embed_tokens` with `0` extra RAM).
pub const TokEmbedCallback = struct {
    ctx: ?*anyopaque = null,
    read_fn: *const fn (ctx: ?*anyopaque, token_id: u32, out_f32: []f32) Error!void,
};

pub const TokEmbedHook = union(enum) {
    dense_f32: []const f32,
    callback: TokEmbedCallback,
};

/// Zero-allocation historical target-tap reader registered by `GraphModel` so
/// `DFlareProposer` can maintain an incremental multi-token context KV cache
/// across the full prompt + accepted-token history without changing `SpecEngine`.
pub const ContextHistoryFn = *const fn (
    ctx: *const anyopaque,
    out_len: *u32,
    out_epoch: *u64,
    pos: u32,
    tap_idx: usize,
    dst_f32: []f32,
) bool;

const ContextHistorySlot = struct {
    ctx: ?*const anyopaque = null,
    read_fn: ?ContextHistoryFn = null,
};

var ctx_hist_lock: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);
var ctx_hist_slots: [16]ContextHistorySlot = @splat(.{});

fn acquireCtxHistLock() void {
    while (ctx_hist_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
        std.atomic.spinLoopHint();
    }
}

fn releaseCtxHistLock() void {
    ctx_hist_lock.store(false, .release);
}

pub fn registerContextHistoryProvider(ctx: *const anyopaque, read_fn: ContextHistoryFn) void {
    acquireCtxHistLock();
    defer releaseCtxHistLock();
    for (&ctx_hist_slots) |*s| {
        if (s.ctx == ctx) {
            s.read_fn = read_fn;
            return;
        }
    }
    for (&ctx_hist_slots) |*s| {
        if (s.ctx == null) {
            s.ctx = ctx;
            s.read_fn = read_fn;
            return;
        }
    }
    ctx_hist_slots[0] = .{ .ctx = ctx, .read_fn = read_fn };
}

pub fn unregisterContextHistoryProvider(ctx: *const anyopaque) void {
    acquireCtxHistLock();
    defer releaseCtxHistLock();
    for (&ctx_hist_slots) |*s| {
        if (s.ctx == ctx) {
            s.ctx = null;
            s.read_fn = null;
        }
    }
}

fn lookupContextHistoryProvider(ctx: ?*anyopaque) ?ContextHistoryFn {
    const key: *const anyopaque = ctx orelse return null;
    acquireCtxHistLock();
    defer releaseCtxHistLock();
    for (ctx_hist_slots) |s| {
        if (s.ctx == key) return s.read_fn;
    }
    return null;
}

pub const DraftKindDetected = enum {
    dflare,
    dfly2,
};

/// Inspects a draft config JSON (`dflare_config.json`, `dfly2_config.json`, or GGUF `dflash`)
/// and identifies whether it describes a `DFlare` or `DFly2` bundle.
pub fn detectDraftKindFromJson(json_bytes: []const u8) ?DraftKindDetected {
    if (std.mem.indexOf(u8, json_bytes, "\"dfly2\"") != null or
        std.mem.indexOf(u8, json_bytes, "\"dfly_mtp_fusion\"") != null or
        std.mem.indexOf(u8, json_bytes, "\"dfly_block_size\"") != null)
    {
        return .dfly2;
    }
    if (std.mem.indexOf(u8, json_bytes, "\"dflare\"") != null or
        std.mem.indexOf(u8, json_bytes, "\"dflash\"") != null or
        std.mem.indexOf(u8, json_bytes, "\"DFlashForCausalLM\"") != null or
        std.mem.indexOf(u8, json_bytes, "\"markov_head_type\"") != null)
    {
        return .dflare;
    }
    return null;
}

/// Optional MLA attention weight pack for `Hy4` `DFly2` draft layers.
pub const MlaBlockWeights = struct {
    q_rank: u32,
    kv_rank: u32,
    qk_nope: u32,
    qk_rope: u32,
    v_dim: u32,
    q_a_proj: []const f32, // [q_rank, H]
    q_a_norm: ?[]const f32 = null, // [q_rank]
    q_b_proj: []const f32, // [n_q * (qk_nope + qk_rope), q_rank]
    kv_a_proj: []const f32, // [kv_rank + qk_rope, H]
    kv_a_norm: ?[]const f32 = null, // [kv_rank]
    kv_b_proj: []const f32, // [n_q * (qk_nope + v_dim), kv_rank]
    o_proj: []const f32, // [H, n_q * v_dim]
    linear_gate: ?[]const f32 = null, // [n_q * v_dim, H]
    sink_f32: ?[]const f32 = null, // [n_q]
    post_proj_norm: ?[]const f32 = null, // [H]
};

/// Shared lightweight block-parallel transformer layer weights used by both
/// `DFlareProposer` (`3L`/`5L`) and `DFly2Proposer` (`5L`).
pub const TransformerBlockWeights = struct {
    in_norm_weight: ?[]const f32 = null,
    q_proj: []const f32 = &.{}, // [q_dim, hidden_size]
    k_proj: []const f32 = &.{}, // [kv_dim, hidden_size]
    v_proj: []const f32 = &.{}, // [kv_dim, hidden_size]
    o_proj: []const f32 = &.{}, // [hidden_size, q_dim]
    /// Optional compact BF16 projections (halves RAM & doubles BW on large real checkpoints).
    q_proj_bf16: ?[]const u16 = null,
    k_proj_bf16: ?[]const u16 = null,
    v_proj_bf16: ?[]const u16 = null,
    o_proj_bf16: ?[]const u16 = null,
    /// Optional dedicated context KV projections (`layers.{l}.self_attn.{k,v}_proj_ctx`).
    k_proj_ctx: ?[]const f32 = null,
    v_proj_ctx: ?[]const f32 = null,
    k_proj_ctx_bf16: ?[]const u16 = null,
    v_proj_ctx_bf16: ?[]const u16 = null,
    k_norm_ctx_weight: ?[]const f32 = null,
    ctx_norm_weight: ?[]const f32 = null,
    q_norm_weight: ?[]const f32 = null, // [head_dim]
    k_norm_weight: ?[]const f32 = null, // [head_dim]
    post_norm_weight: ?[]const f32 = null,
    mlp_gate_proj: []const f32 = &.{}, // [intermediate_size, hidden_size]
    mlp_up_proj: []const f32 = &.{}, // [intermediate_size, hidden_size]
    mlp_down_proj: []const f32 = &.{}, // [hidden_size, intermediate_size]
    mlp_gate_bf16: ?[]const u16 = null,
    mlp_up_bf16: ?[]const u16 = null,
    mlp_down_bf16: ?[]const u16 = null,
    /// Optional Hy4 MLA attention weights (`DFly2` layers).
    mla: ?MlaBlockWeights = null,

    pub fn validate(
        self: TransformerBlockWeights,
        hidden_size: usize,
        intermediate_size: usize,
        q_dim: usize,
        kv_dim: usize,
        head_dim: usize,
    ) Error!void {
        const qh = std.math.mul(usize, q_dim, hidden_size) catch return error.Overflow;
        const kvh = std.math.mul(usize, kv_dim, hidden_size) catch return error.Overflow;
        const hq = std.math.mul(usize, hidden_size, q_dim) catch return error.Overflow;
        const ih = std.math.mul(usize, intermediate_size, hidden_size) catch return error.Overflow;
        const hi = std.math.mul(usize, hidden_size, intermediate_size) catch return error.Overflow;

        if (self.in_norm_weight) |w| {
            if (w.len != hidden_size) return error.InvalidWeights;
        }
        if (self.mla == null) {
            const q_ok = (self.q_proj.len == qh) or (if (self.q_proj_bf16) |b| b.len == qh else false);
            const k_ok = (self.k_proj.len == kvh) or (if (self.k_proj_bf16) |b| b.len == kvh else false);
            const v_ok = (self.v_proj.len == kvh) or (if (self.v_proj_bf16) |b| b.len == kvh else false);
            const o_ok = (self.o_proj.len == hq) or (if (self.o_proj_bf16) |b| b.len == hq else false);
            if (!q_ok or !k_ok or !v_ok or !o_ok) {
                return error.InvalidWeights;
            }
            if (self.q_norm_weight) |w| {
                if (w.len != head_dim) return error.InvalidWeights;
            }
            if (self.k_norm_weight) |w| {
                if (w.len != head_dim) return error.InvalidWeights;
            }
        }
        if (self.post_norm_weight) |w| {
            if (w.len != hidden_size) return error.InvalidWeights;
        }
        const g_ok = (self.mlp_gate_proj.len == ih) or (if (self.mlp_gate_bf16) |b| b.len == ih else false);
        const u_ok = (self.mlp_up_proj.len == ih) or (if (self.mlp_up_bf16) |b| b.len == ih else false);
        const d_ok = (self.mlp_down_proj.len == hi) or (if (self.mlp_down_bf16) |b| b.len == hi else false);
        if (!g_ok or !u_ok or !d_ok) {
            return error.InvalidWeights;
        }
    }
};

// ============================================================================
// DFlare (`Inkling-2-Mini 3L`, `Inkling-2-Small 5L`, `arch = dflash`)
// ============================================================================

pub const DFlareConfig = struct {
    variant: DFlareVariant = .inkling2_mini_3l,
    num_layers: u8 = 3,
    block_size: u8 = DFLARE_BLOCK_SIZE,
    top_k_per_pos: u8 = 1,
    max_tree_nodes: u8 = 32,
    hidden_size: u32,
    intermediate_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    markov_rank: u32 = DFLARE_MARKOV_RANK,
    num_target_taps: u8 = 3,
    target_layer_ids: [MAX_TARGET_TAPS]u16 = .{ 2, 11, 20, 0, 0, 0, 0, 0 },
    rms_eps: f32 = 1e-6,
    rope_theta: f32 = 0.0,
    unpadded_vocab_size: u32 = 0,
    final_logit_softcapping: f32 = 0.0,
    temperature: f32 = 1.0,
    l1_seed_step_prob: f32 = 0.90,
    l1_branch_step_prob: f32 = 0.55,
    l1_max_draft_len: u8 = 8,

    pub fn miniDefault(
        hidden_size: u32,
        intermediate_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        vocab_size: u32,
    ) DFlareConfig {
        return .{
            .variant = .inkling2_mini_3l,
            .num_layers = 3,
            .block_size = DFLARE_BLOCK_SIZE,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .num_attention_heads = num_heads,
            .num_key_value_heads = num_kv_heads,
            .head_dim = head_dim,
            .vocab_size = vocab_size,
            .markov_rank = DFLARE_MARKOV_RANK,
            .num_target_taps = 3,
        };
    }

    pub fn smallDefault(
        hidden_size: u32,
        intermediate_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        vocab_size: u32,
    ) DFlareConfig {
        return .{
            .variant = .inkling2_small_5l,
            .num_layers = 5,
            .block_size = DFLARE_BLOCK_SIZE,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .num_attention_heads = num_heads,
            .num_key_value_heads = num_kv_heads,
            .head_dim = head_dim,
            .vocab_size = vocab_size,
            .markov_rank = DFLARE_MARKOV_RANK,
            .num_target_taps = 4,
            .target_layer_ids = .{ 4, 16, 28, 40, 0, 0, 0, 0 },
        };
    }

    pub inline fn qDim(self: DFlareConfig) u32 {
        return self.num_attention_heads * self.head_dim;
    }

    pub inline fn kvDim(self: DFlareConfig) u32 {
        return self.num_key_value_heads * self.head_dim;
    }

    pub fn validate(self: DFlareConfig) Error!void {
        if (self.num_layers == 0 or self.num_layers > MAX_LAYERS) return error.InvalidConfig;
        if (self.block_size == 0 or self.block_size > MAX_BLOCK_SIZE) return error.InvalidConfig;
        if (self.top_k_per_pos == 0 or self.top_k_per_pos > MAX_TOP_K_PER_POS) return error.InvalidConfig;
        if (self.max_tree_nodes < 2 or self.max_tree_nodes > MAX_TREE_NODES) return error.InvalidConfig;
        if (self.hidden_size == 0 or self.intermediate_size == 0 or self.vocab_size == 0) return error.InvalidConfig;
        if (self.num_attention_heads == 0 or self.num_key_value_heads == 0 or self.head_dim == 0) return error.InvalidConfig;
        if ((self.num_attention_heads % self.num_key_value_heads) != 0) return error.InvalidConfig;
        if (self.top_k_per_pos > self.vocab_size) return error.InvalidConfig;
        if (self.markov_rank == 0) return error.InvalidConfig;
        if (self.num_target_taps == 0 or self.num_target_taps > MAX_TARGET_TAPS) return error.InvalidConfig;
        if (!std.math.isFinite(self.rms_eps) or self.rms_eps <= 0.0) return error.InvalidConfig;
        if (!std.math.isFinite(self.temperature) or self.temperature <= 0.0) return error.InvalidConfig;
        if (self.l1_max_draft_len > MAX_L1_DRAFT_LEN) return error.InvalidConfig;
    }

    /// Parses `dflare_config.json` or GGUF `arch=dflash` config JSON with target fallbacks.
    pub fn fromJson(
        allocator: std.mem.Allocator,
        json_bytes: []const u8,
        fallback_hidden: u32,
        fallback_vocab: u32,
    ) (Error || error{OutOfMemory})!DFlareConfig {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch return error.InvalidConfig;
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidConfig;
        const obj = parsed.value.object;

        const num_layers: u8 = @intCast(jsonU32Opt(obj, "num_hidden_layers") orelse 3);
        const hidden_size: u32 = jsonU32Opt(obj, "hidden_size") orelse fallback_hidden;
        const inter_size: u32 = jsonU32Opt(obj, "intermediate_size") orelse (hidden_size * 4);
        const num_heads: u32 = jsonU32Opt(obj, "num_attention_heads") orelse 8;
        const num_kv_heads: u32 = jsonU32Opt(obj, "num_key_value_heads") orelse num_heads;
        const head_dim: u32 = jsonU32Opt(obj, "head_dim") orelse (hidden_size / @max(num_heads, 1));
        const vocab_size: u32 = if (fallback_vocab > 0) fallback_vocab else (jsonU32Opt(obj, "vocab_size") orelse 0);
        const block_size: u8 = @intCast(jsonU32Opt(obj, "block_size") orelse DFLARE_BLOCK_SIZE);
        const markov_rank: u32 = jsonU32Opt(obj, "markov_rank") orelse DFLARE_MARKOV_RANK;
        const rms_eps: f32 = jsonF32Opt(obj, "rms_norm_eps") orelse 1e-6;
        const rope_theta: f32 = jsonF32Opt(obj, "rope_theta") orelse 0.0;
        const unpadded_vocab: u32 = jsonU32Opt(obj, "unpadded_vocab_size") orelse 0;
        const final_softcap: f32 = jsonF32Opt(obj, "final_logit_softcapping") orelse 0.0;

        var tap_ids: [MAX_TARGET_TAPS]u16 = .{ 5, 23, 41, 48, 0, 0, 0, 0 };
        var num_taps: u8 = 4;
        if (obj.get("target_layer_ids")) |v| {
            if (v == .array and v.array.items.len > 0 and v.array.items.len <= MAX_TARGET_TAPS) {
                num_taps = @intCast(v.array.items.len);
                for (v.array.items, 0..) |item, i| {
                    if (item == .integer and item.integer >= 0) {
                        tap_ids[i] = @intCast(item.integer);
                    }
                }
            }
        }
        const variant: DFlareVariant = if (obj.get("draft_kind")) |dk| blk: {
            if (dk == .string and std.mem.eql(u8, dk.string, "dflash")) break :blk .dflash_standalone;
            break :blk if (num_layers >= 5) .inkling2_small_5l else .inkling2_mini_3l;
        } else if (num_layers >= 5) .inkling2_small_5l else .inkling2_mini_3l;

        const cfg: DFlareConfig = .{
            .variant = variant,
            .num_layers = num_layers,
            .block_size = block_size,
            .hidden_size = hidden_size,
            .intermediate_size = inter_size,
            .num_attention_heads = num_heads,
            .num_key_value_heads = num_kv_heads,
            .head_dim = head_dim,
            .vocab_size = vocab_size,
            .markov_rank = markov_rank,
            .num_target_taps = num_taps,
            .target_layer_ids = tap_ids,
            .rms_eps = rms_eps,
            .rope_theta = rope_theta,
            .unpadded_vocab_size = unpadded_vocab,
            .final_logit_softcapping = final_softcap,
        };
        try cfg.validate();
        return cfg;
    }
};

pub const DFlareWeights = struct {
    /// Multi-layer target tap fusion `[hidden_size, num_target_taps * hidden_size]`.
    /// Optional when `num_target_taps == 1` or `fuse_scalar_weights != null`.
    target_fusion_proj: ?[]const f32 = null,
    /// Optional per-layer scalar tap mixing weights `[rows, num_target_taps]`
    /// (real `Inkling-2-Mini` `fuse_weight [4, 4]` & `dflash-Q8` `ctx_mix`).
    fuse_scalar_weights: ?[]const f32 = null,
    target_fusion_norm: ?[]const f32 = null,
    /// Mask embedding: `[block_size, hidden_size]`, `[(block_size - 1), hidden_size]`, or shared `[hidden_size]`.
    mask_embed: []const f32,
    /// Token embeddings `[vocab_size, hidden_size]` for sequential Markov conditioning (or empty when `tok_embed_hook` is set).
    tok_embed: []const f32 = &.{},
    tok_embed_hook: ?TokEmbedHook = null,
    token_embd_norm: ?[]const f32 = null,
    /// `3L` / `5L` block-parallel transformer layers.
    layers: []const TransformerBlockWeights,
    final_norm: ?[]const f32 = null,
    /// Low-rank Markov head `markov_w1`: `[markov_rank, 2*H]`, `[markov_rank, H]`, or empty when `markov_w1_vocab_bf16` is set.
    markov_w1: []const f32 = &.{},
    /// Optional token-ID low-rank embedding table `[vocab_rows, markov_rank]` BF16 (real `Inkling-2-Mini`).
    markov_w1_vocab_bf16: ?[]const u16 = null,
    markov_w1_vocab_rows: u32 = 0,
    /// Optional 3-gate GRU joint projection `[3 * markov_rank, 2*H + 2*markov_rank]` (`768 x 5120`).
    markov_gru_proj_bf16: ?[]const u16 = null,
    markov_gru_proj_f32: ?[]const f32 = null,
    markov_gru_bias: ?[]const f32 = null,
    markov_ctx_norm: ?[]const f32 = null,
    /// Optional recurrent transition `[markov_rank, markov_rank]` on `rnn_hidden`.
    markov_rnn_proj: ?[]const f32 = null,
    markov_b1: ?[]const f32 = null,
    /// Output bottleneck expansion `markov_w2`: `[hidden_size, markov_rank]` (or empty when `markov_w2_vocab_bf16` is set).
    markov_w2: []const f32 = &.{},
    markov_hidden_bias: ?[]const f32 = null,
    /// Optional direct low-rank vocab logits head `[vocab_rows, markov_rank]` BF16 (`dflash-Q8` GGUF).
    markov_w2_vocab_bf16: ?[]const u16 = null,
    markov_w2_vocab_bias: ?[]const f32 = null,
    markov_w2_vocab_rows: u32 = 0,
    markov_norm: ?[]const f32 = null,
    lm_head: LmHeadHook,

    pub fn validate(self: DFlareWeights, cfg: DFlareConfig) Error!void {
        const h: usize = cfg.hidden_size;
        const inter: usize = cfg.intermediate_size;
        const v: usize = cfg.vocab_size;
        const mr: usize = cfg.markov_rank;
        const taps: usize = cfg.num_target_taps;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();

        const thh = std.math.mul(usize, taps, std.math.mul(usize, h, h) catch return error.Overflow) catch return error.Overflow;
        if (self.target_fusion_proj) |fp| {
            if (fp.len != thh) return error.InvalidWeights;
        } else if (self.fuse_scalar_weights) |fsw| {
            if (fsw.len < taps) return error.InvalidWeights;
        } else if (taps != 1) {
            return error.InvalidWeights;
        }
        if (self.target_fusion_norm) |fnw| {
            if (fnw.len != h) return error.InvalidWeights;
        }
        const mask_ok = self.mask_embed.len == h or
            self.mask_embed.len == (std.math.mul(usize, cfg.block_size, h) catch return error.Overflow) or
            (cfg.block_size > 1 and self.mask_embed.len == (std.math.mul(usize, cfg.block_size - 1, h) catch return error.Overflow));
        if (!mask_ok) return error.InvalidWeights;

        const vh = std.math.mul(usize, v, h) catch return error.Overflow;
        if (self.tok_embed_hook == null and self.tok_embed.len != vh) return error.InvalidWeights;
        if (self.layers.len != cfg.num_layers) return error.InvalidWeights;
        for (self.layers) |layer| {
            try layer.validate(h, inter, q_dim, kv_dim, cfg.head_dim);
        }
        if (self.final_norm) |fnw| {
            if (fnw.len != h) return error.InvalidWeights;
        }

        const has_gru = (self.markov_gru_proj_bf16 != null or self.markov_gru_proj_f32 != null);
        const mr_h = std.math.mul(usize, mr, h) catch return error.Overflow;
        if (!has_gru) {
            const mr_2h = std.math.mul(usize, mr_h, 2) catch return error.Overflow;
            if (self.markov_w1.len != mr_h and self.markov_w1.len != mr_2h) {
                return error.InvalidWeights;
            }
            if (self.markov_rnn_proj) |rp| {
                const mr2 = std.math.mul(usize, mr, mr) catch return error.Overflow;
                if (rp.len != mr2) return error.InvalidWeights;
            }
            if (self.markov_b1) |b1| {
                if (b1.len != mr) return error.InvalidWeights;
            }
        } else {
            const joint_in = 2 * h + 2 * mr;
            const req_gru = 3 * mr * joint_in;
            if (self.markov_gru_proj_bf16) |gb| {
                if (gb.len != req_gru) return error.InvalidWeights;
            } else if (self.markov_gru_proj_f32) |gf| {
                if (gf.len != req_gru) return error.InvalidWeights;
            }
            if (self.markov_w1_vocab_bf16 == null or self.markov_w1_vocab_rows == 0) return error.InvalidWeights;
        }
        if (self.markov_w2_vocab_bf16 == null and self.markov_w2.len != mr_h) return error.InvalidWeights;
        if (self.markov_norm) |mn| {
            if (mn.len != h) return error.InvalidWeights;
        }
        if (self.markov_w2_vocab_bf16 == null) {
            try validateLmHeadHook(self.lm_head, vh);
        }
    }
};

/// Preallocated 16 KiB-aligned `WiredArena` workspace for `DFlareProposer`.
pub const DFlareWorkspace = struct {
    arena: arena_mod.WiredArena,
    config: DFlareConfig,
    max_ctx_len: u32,
    taps_flat: []f32,
    ctx_hidden: []f32,
    ctx_markov: []f32,
    tok_embed_tmp: []f32,
    slot_hiddens: []f32,
    norm_buf: []f32,
    q_buf: []f32,
    k_buf: []f32,
    v_buf: []f32,
    ctx_k_cache: []f16,
    ctx_v_cache: []f16,
    attn_out: []f32,
    attn_scores: []f32,
    mlp_gate: []f32,
    mlp_up: []f32,
    mlp_down: []f32,
    node_rnn_hidden: []f32,
    markov_u: []f32,
    markov_out: []f32,
    markov_concat: []f32,
    head_hidden: []f32,
    base_logits_buf: []f32,
    logits_buf: []f32,
    topk_tokens: []u32,
    topk_probs: []f32,
    node_log_probs: []f32,
    l1_draft_buf: []u32,

    pub fn requiredBytes(cfg: DFlareConfig) Error!usize {
        try cfg.validate();
        const h: usize = cfg.hidden_size;
        const inter: usize = cfg.intermediate_size;
        const w: usize = cfg.block_size;
        const v: usize = cfg.vocab_size;
        const mr: usize = cfg.markov_rank;
        const taps: usize = cfg.num_target_taps;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();
        const max_nodes: usize = cfg.max_tree_nodes;
        const max_ctx: usize = if (cfg.hidden_size >= 1024) 1024 else 64;
        const ctx_kv_elems: usize = @as(usize, cfg.num_layers) * max_ctx * kv_dim;

        const chunks = [_]usize{
            try alignUp64((taps * h) * @sizeOf(f32)), // taps_flat
            try alignUp64(h * @sizeOf(f32)), // ctx_hidden
            try alignUp64(h * @sizeOf(f32)), // ctx_markov
            try alignUp64(h * @sizeOf(f32)), // tok_embed_tmp
            try alignUp64((w * h) * @sizeOf(f32)), // slot_hiddens
            try alignUp64((w * h) * @sizeOf(f32)), // norm_buf
            try alignUp64((w * q_dim) * @sizeOf(f32)), // q_buf
            try alignUp64(((w + 1) * kv_dim) * @sizeOf(f32)), // k_buf (ctx + block)
            try alignUp64(((w + 1) * kv_dim) * @sizeOf(f32)), // v_buf
            try alignUp64(ctx_kv_elems * @sizeOf(f16)), // ctx_k_cache
            try alignUp64(ctx_kv_elems * @sizeOf(f16)), // ctx_v_cache
            try alignUp64((w * q_dim) * @sizeOf(f32)), // attn_out
            try alignUp64((max_ctx + w) * @sizeOf(f32)), // attn_scores
            try alignUp64((w * inter) * @sizeOf(f32)), // mlp_gate
            try alignUp64((w * inter) * @sizeOf(f32)), // mlp_up
            try alignUp64((w * h) * @sizeOf(f32)), // mlp_down
            try alignUp64((max_nodes * mr) * @sizeOf(f32)), // node_rnn_hidden
            try alignUp64((3 * mr) * @sizeOf(f32)), // markov_u (3-gate GRU or 1x rank)
            try alignUp64(mr * @sizeOf(f32)), // markov_out
            try alignUp64((2 * h + 2 * mr) * @sizeOf(f32)), // markov_concat
            try alignUp64(h * @sizeOf(f32)), // head_hidden
            try alignUp64((w * v) * @sizeOf(f32)), // base_logits_buf
            try alignUp64(v * @sizeOf(f32)), // logits_buf
            try alignUp64(cfg.top_k_per_pos * @sizeOf(u32)), // topk_tokens
            try alignUp64(cfg.top_k_per_pos * @sizeOf(f32)), // topk_probs
            try alignUp64(max_nodes * @sizeOf(f32)), // node_log_probs
            try alignUp64(MAX_L1_DRAFT_LEN * @sizeOf(u32)), // l1_draft_buf
        };
        var total: usize = 0;
        for (chunks) |c| total = std.math.add(usize, total, c) catch return error.Overflow;
        return page.alignUp(total) catch return error.Overflow;
    }

    pub fn init(cfg: DFlareConfig, budget_override: ?budget_mod.Budget) Error!DFlareWorkspace {
        const cap_bytes = try requiredBytes(cfg);
        const wired_budget = budget_override orelse try defaultWiredBudget();
        try wired_budget.checkPlan(cap_bytes);

        var mem_arena = try arena_mod.WiredArena.init(cap_bytes, wired_budget);
        errdefer mem_arena.deinit();

        const h: usize = cfg.hidden_size;
        const inter: usize = cfg.intermediate_size;
        const w: usize = cfg.block_size;
        const v: usize = cfg.vocab_size;
        const mr: usize = cfg.markov_rank;
        const taps: usize = cfg.num_target_taps;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();
        const max_nodes: usize = cfg.max_tree_nodes;
        const max_ctx: usize = if (cfg.hidden_size >= 1024) 1024 else 64;
        const ctx_kv_elems: usize = @as(usize, cfg.num_layers) * max_ctx * kv_dim;

        const taps_flat = try allocArenaSlice(f32, &mem_arena, taps * h);
        const ctx_hidden = try allocArenaSlice(f32, &mem_arena, h);
        const ctx_markov = try allocArenaSlice(f32, &mem_arena, h);
        const tok_embed_tmp = try allocArenaSlice(f32, &mem_arena, h);
        const slot_hiddens = try allocArenaSlice(f32, &mem_arena, w * h);
        const norm_buf = try allocArenaSlice(f32, &mem_arena, w * h);
        const q_buf = try allocArenaSlice(f32, &mem_arena, w * q_dim);
        const k_buf = try allocArenaSlice(f32, &mem_arena, (w + 1) * kv_dim);
        const v_buf = try allocArenaSlice(f32, &mem_arena, (w + 1) * kv_dim);
        const ctx_k_cache = try allocArenaSlice(f16, &mem_arena, ctx_kv_elems);
        const ctx_v_cache = try allocArenaSlice(f16, &mem_arena, ctx_kv_elems);
        const attn_out = try allocArenaSlice(f32, &mem_arena, w * q_dim);
        const attn_scores = try allocArenaSlice(f32, &mem_arena, max_ctx + w);
        const mlp_gate = try allocArenaSlice(f32, &mem_arena, w * inter);
        const mlp_up = try allocArenaSlice(f32, &mem_arena, w * inter);
        const mlp_down = try allocArenaSlice(f32, &mem_arena, w * h);
        const node_rnn_hidden = try allocArenaSlice(f32, &mem_arena, max_nodes * mr);
        const markov_u = try allocArenaSlice(f32, &mem_arena, 3 * mr);
        const markov_out = try allocArenaSlice(f32, &mem_arena, mr);
        const markov_concat = try allocArenaSlice(f32, &mem_arena, 2 * h + 2 * mr);
        const head_hidden = try allocArenaSlice(f32, &mem_arena, h);
        const base_logits_buf = try allocArenaSlice(f32, &mem_arena, w * v);
        const logits_buf = try allocArenaSlice(f32, &mem_arena, v);
        const topk_tokens = try allocArenaSlice(u32, &mem_arena, cfg.top_k_per_pos);
        const topk_probs = try allocArenaSlice(f32, &mem_arena, cfg.top_k_per_pos);
        const node_log_probs = try allocArenaSlice(f32, &mem_arena, max_nodes);
        const l1_draft_buf = try allocArenaSlice(u32, &mem_arena, MAX_L1_DRAFT_LEN);

        @memset(node_log_probs, 0.0);
        @memset(node_rnn_hidden, 0.0);
        @memset(ctx_k_cache, 0.0);
        @memset(ctx_v_cache, 0.0);

        return .{
            .arena = mem_arena,
            .config = cfg,
            .max_ctx_len = @intCast(max_ctx),
            .taps_flat = taps_flat,
            .ctx_hidden = ctx_hidden,
            .ctx_markov = ctx_markov,
            .tok_embed_tmp = tok_embed_tmp,
            .slot_hiddens = slot_hiddens,
            .norm_buf = norm_buf,
            .q_buf = q_buf,
            .k_buf = k_buf,
            .v_buf = v_buf,
            .ctx_k_cache = ctx_k_cache,
            .ctx_v_cache = ctx_v_cache,
            .attn_out = attn_out,
            .attn_scores = attn_scores,
            .mlp_gate = mlp_gate,
            .mlp_up = mlp_up,
            .mlp_down = mlp_down,
            .node_rnn_hidden = node_rnn_hidden,
            .markov_u = markov_u,
            .markov_out = markov_out,
            .markov_concat = markov_concat,
            .head_hidden = head_hidden,
            .base_logits_buf = base_logits_buf,
            .logits_buf = logits_buf,
            .topk_tokens = topk_tokens,
            .topk_probs = topk_probs,
            .node_log_probs = node_log_probs,
            .l1_draft_buf = l1_draft_buf,
        };
    }

    pub fn deinit(self: *DFlareWorkspace) void {
        self.taps_flat = &.{};
        self.ctx_hidden = &.{};
        self.ctx_markov = &.{};
        self.tok_embed_tmp = &.{};
        self.slot_hiddens = &.{};
        self.norm_buf = &.{};
        self.q_buf = &.{};
        self.k_buf = &.{};
        self.v_buf = &.{};
        self.ctx_k_cache = &.{};
        self.ctx_v_cache = &.{};
        self.attn_out = &.{};
        self.attn_scores = &.{};
        self.mlp_gate = &.{};
        self.mlp_up = &.{};
        self.mlp_down = &.{};
        self.node_rnn_hidden = &.{};
        self.markov_u = &.{};
        self.markov_out = &.{};
        self.markov_concat = &.{};
        self.head_hidden = &.{};
        self.base_logits_buf = &.{};
        self.logits_buf = &.{};
        self.topk_tokens = &.{};
        self.topk_probs = &.{};
        self.node_log_probs = &.{};
        self.l1_draft_buf = &.{};
        self.arena.deinit();
    }

    pub fn totalWiredBytes(self: *const DFlareWorkspace) u64 {
        return self.arena.bytes().len;
    }
};

pub const DFlareProposeResult = struct {
    tree_nodes: u16,
    max_depth: u16,
    l1_match_len: u32,
    l1_inserted: u16,
    l1_is_seed: bool,
    backbone_runs: u64,
    markov_steps: u64,
    expected_tokens: f32,
};

pub const Core3BatchVerifyResult = struct {
    batch_size_m: u16,
    core_passes: u8,
    accept: accept_mod.AcceptResult,
};

/// D14 `DFlare` block-parallel speculative proposer (`Inkling 2` `3L`/`5L`,
/// `block_size = 8`, `markov_rank = 256` sequential `rnn_hidden` Markov head).
pub const DFlareProposer = struct {
    config: DFlareConfig,
    weights: DFlareWeights,
    workspace: *DFlareWorkspace,
    backbone_runs: u64 = 0,
    markov_steps: u64 = 0,
    ctx_cache_len: u32 = 0,
    ctx_cache_epoch: u64 = 0,
    base_logits_ready: bool = false,

    inline fn isRealGru(self: *const DFlareProposer) bool {
        return self.weights.markov_gru_proj_bf16 != null or self.weights.markov_gru_proj_f32 != null;
    }

    pub fn init(
        config: DFlareConfig,
        weights: DFlareWeights,
        workspace: *DFlareWorkspace,
    ) Error!DFlareProposer {
        try config.validate();
        try weights.validate(config);
        if (workspace.config.block_size != config.block_size or
            workspace.config.hidden_size != config.hidden_size or
            workspace.config.intermediate_size != config.intermediate_size or
            workspace.config.num_attention_heads != config.num_attention_heads or
            workspace.config.num_key_value_heads != config.num_key_value_heads or
            workspace.config.head_dim != config.head_dim or
            workspace.config.vocab_size != config.vocab_size or
            workspace.config.markov_rank != config.markov_rank or
            workspace.config.top_k_per_pos != config.top_k_per_pos or
            workspace.config.max_tree_nodes < config.max_tree_nodes)
        {
            return error.WorkspaceMismatch;
        }
        return .{
            .config = config,
            .weights = weights,
            .workspace = workspace,
        };
    }

    /// Runs the `3L` / `5L` block-parallel backbone ONCE across `0..block_size-1`
    /// conditioned on fused `target_layer_ids` hidden taps (`target_hiddens`).
    /// Results stay in `self.workspace.slot_hiddens` for sequential Markov decoding.
    pub fn runBlockBackboneOnce(
        self: *DFlareProposer,
        target_hiddens: []const []const f32,
    ) Error!void {
        return self.runBlockBackboneOnceForRoot(0, target_hiddens);
    }

    fn projectCtxKvOnePos(
        self: *DFlareProposer,
        layer: TransformerBlockWeights,
        li: usize,
        taps_flat: []const f32,
        pos: u32,
        dst_k: []f32,
        dst_v: []f32,
    ) void {
        const taps: usize = self.config.num_target_taps;
        const h: usize = self.config.hidden_size;
        const kv_dim: usize = self.config.kvDim();
        if (self.weights.fuse_scalar_weights) |fsw| {
            const row_idx = if ((li + 1) * taps <= fsw.len) li else 0;
            mixScalarTaps(fsw[row_idx * taps ..][0..taps], taps_flat[0 .. taps * h], taps, h, self.workspace.ctx_hidden[0..h]);
            if (layer.ctx_norm_weight) |cnw| {
                applyRmsNormInPlace(self.workspace.ctx_hidden[0..h], cnw, self.config.rms_eps);
            }
        } else if (layer.ctx_norm_weight) |cnw| {
            @memcpy(self.workspace.norm_buf[0..h], self.workspace.ctx_hidden[0..h]);
            applyRmsNormInPlace(self.workspace.norm_buf[0..h], cnw, self.config.rms_eps);
        }
        const ctx_in = if (self.weights.fuse_scalar_weights == null and layer.ctx_norm_weight != null)
            self.workspace.norm_buf[0..h]
        else
            self.workspace.ctx_hidden[0..h];

        if (layer.k_proj_ctx_bf16) |kb| {
            matVecBf16(kb, ctx_in, kv_dim, h, dst_k[0..kv_dim]);
        } else if (layer.k_proj_ctx) |kf| {
            matVecF32(kf, ctx_in, kv_dim, h, dst_k[0..kv_dim]);
        } else if (layer.k_proj_bf16) |kb| {
            matVecBf16(kb, ctx_in, kv_dim, h, dst_k[0..kv_dim]);
        } else {
            matVecF32(layer.k_proj, ctx_in, kv_dim, h, dst_k[0..kv_dim]);
        }

        if (layer.v_proj_ctx_bf16) |vb| {
            matVecBf16(vb, ctx_in, kv_dim, h, dst_v[0..kv_dim]);
        } else if (layer.v_proj_ctx) |vf| {
            matVecF32(vf, ctx_in, kv_dim, h, dst_v[0..kv_dim]);
        } else if (layer.v_proj_bf16) |vb| {
            matVecBf16(vb, ctx_in, kv_dim, h, dst_v[0..kv_dim]);
        } else {
            matVecF32(layer.v_proj, ctx_in, kv_dim, h, dst_v[0..kv_dim]);
        }

        if (layer.k_norm_ctx_weight orelse layer.k_norm_weight) |kn| {
            applyHeadRmsNorm(dst_k[0..kv_dim], self.config.num_key_value_heads, self.config.head_dim, kn, self.config.rms_eps);
        }
        if (self.config.rope_theta > 0.0) {
            applyRopeNeoXInPlace(dst_k[0..kv_dim], self.config.num_key_value_heads, self.config.head_dim, pos, self.config.rope_theta);
        }
    }

    fn runBlockBackboneOnceForRoot(
        self: *DFlareProposer,
        root_token: u32,
        target_hiddens: []const []const f32,
    ) Error!void {
        const taps: usize = self.config.num_target_taps;
        const h: usize = self.config.hidden_size;
        const w: usize = self.config.block_size;
        const v: usize = self.config.vocab_size;
        const kv_dim: usize = self.config.kvDim();
        const is_real = self.isRealGru();
        if (target_hiddens.len != taps) return error.DimensionMismatch;

        // 0. For real DFlare/DSpark, resolve the anchor token embedding FIRST
        //    (this also registers GraphModel's context tap history provider on the exact target pointer).
        if (is_real) {
            const raw_anchor = try resolveTokEmbed(self.weights.tok_embed, self.weights.tok_embed_hook, root_token, h, self.workspace.tok_embed_tmp);
            if (raw_anchor.ptr != self.workspace.tok_embed_tmp.ptr) {
                @memcpy(self.workspace.tok_embed_tmp[0..h], raw_anchor[0..h]);
            }
            if (self.weights.tok_embed_hook == null) {
                if (self.weights.token_embd_norm) |ten| {
                    applyRmsNormInPlace(self.workspace.tok_embed_tmp[0..h], ten, self.config.rms_eps);
                }
            }
        }

        // 1. Check for full committed-token context history (prompt + accepted decode tokens).
        var use_cached_ctx: bool = false;
        var active_ctx_len: usize = 1;
        var anchor_pos: u32 = if (self.config.rope_theta > 0.0) 1 else 0;
        const max_ctx: usize = self.workspace.max_ctx_len;

        if (is_real) {
            const cb_ctx: ?*anyopaque = if (self.weights.tok_embed_hook) |hk| switch (hk) {
                .callback => |cb| cb.ctx,
                .dense_f32 => null,
            } else null;
            if (lookupContextHistoryProvider(cb_ctx)) |prov_fn| {
                var hist_len: u32 = 0;
                var hist_epoch: u64 = 0;
                if (prov_fn(cb_ctx.?, &hist_len, &hist_epoch, 0, 0, &.{}) and hist_len > 0) {
                    if (hist_epoch != self.ctx_cache_epoch or hist_len < self.ctx_cache_len) {
                        self.ctx_cache_len = 0;
                        self.ctx_cache_epoch = hist_epoch;
                    }
                    const win_start: u32 = if (hist_len > self.workspace.max_ctx_len) hist_len - self.workspace.max_ctx_len else 0;
                    var p: u32 = @max(self.ctx_cache_len, win_start);
                    while (p < hist_len) : (p += 1) {
                        var ok_all = true;
                        for (0..taps) |ti| {
                            var tmp_l: u32 = 0;
                            var tmp_e: u64 = 0;
                            if (!prov_fn(cb_ctx.?, &tmp_l, &tmp_e, p, ti, self.workspace.taps_flat[ti * h ..][0..h])) {
                                ok_all = false;
                                break;
                            }
                        }
                        if (!ok_all) break;
                        const slot: usize = @intCast(p % self.workspace.max_ctx_len);
                        for (self.weights.layers, 0..) |layer, li| {
                            self.projectCtxKvOnePos(layer, li, self.workspace.taps_flat, p, self.workspace.k_buf[0..kv_dim], self.workspace.v_buf[0..kv_dim]);
                            const base_off = (li * max_ctx + slot) * kv_dim;
                            const dst_k = self.workspace.ctx_k_cache[base_off .. base_off + kv_dim];
                            const dst_v = self.workspace.ctx_v_cache[base_off .. base_off + kv_dim];
                            for (0..kv_dim) |d| {
                                dst_k[d] = @floatCast(self.workspace.k_buf[d]);
                                dst_v[d] = @floatCast(self.workspace.v_buf[d]);
                            }
                        }
                    }
                    self.ctx_cache_len = hist_len;
                    active_ctx_len = @min(@as(usize, hist_len), max_ctx);
                    anchor_pos = hist_len;
                    use_cached_ctx = active_ctx_len > 0;
                }
            }
        }

        // 2. Copy current anchor-conditioning `target_hiddens` into `taps_flat` and compute `ctx_hidden` / `ctx_markov` (`f_rnn`).
        for (target_hiddens, 0..) |tap, ti| {
            if (tap.len != h) return error.DimensionMismatch;
            @memcpy(self.workspace.taps_flat[ti * h ..][0..h], tap);
        }

        if (self.weights.fuse_scalar_weights) |fsw| {
            mixScalarTaps(fsw[0..taps], self.workspace.taps_flat[0 .. taps * h], taps, h, self.workspace.ctx_hidden[0..h]);
            const m_row = if (fsw.len >= (self.weights.layers.len + 1) * taps) self.weights.layers.len else 0;
            mixScalarTaps(fsw[m_row * taps ..][0..taps], self.workspace.taps_flat[0 .. taps * h], taps, h, self.workspace.ctx_markov[0..h]);
            if (self.weights.markov_ctx_norm) |mcn| {
                applyRmsNormInPlace(self.workspace.ctx_markov[0..h], mcn, self.config.rms_eps);
            }
        } else if (self.weights.target_fusion_proj) |fp| {
            matVecF32(fp, self.workspace.taps_flat[0 .. taps * h], h, taps * h, self.workspace.ctx_hidden[0..h]);
            @memcpy(self.workspace.ctx_markov[0..h], self.workspace.ctx_hidden[0..h]);
        } else {
            @memcpy(self.workspace.ctx_hidden[0..h], self.workspace.taps_flat[0..h]);
            @memcpy(self.workspace.ctx_markov[0..h], self.workspace.ctx_hidden[0..h]);
        }
        if (self.weights.target_fusion_norm) |fnw| {
            applyRmsNormInPlace(self.workspace.ctx_hidden[0..h], fnw, self.config.rms_eps);
        }

        // 3. Seed noise block `slot_hiddens[0..w-1]`.
        const per_pos_mask = self.weights.mask_embed.len == w * h;
        const tail_pos_mask = (w > 1 and self.weights.mask_embed.len == (w - 1) * h);
        if (is_real) {
            // Exact `dspark.cpp:build_dspark_inp_block`:
            // slot 0 = normed anchor token embedding; slots 1..w-1 = pure mask_embed (NO +ctx_hidden!).
            @memcpy(self.workspace.slot_hiddens[0..h], self.workspace.tok_embed_tmp[0..h]);
            for (1..w) |pos| {
                const dst = self.workspace.slot_hiddens[pos * h ..][0..h];
                const m_slice = if (tail_pos_mask)
                    self.weights.mask_embed[(pos - 1) * h ..][0..h]
                else if (per_pos_mask)
                    self.weights.mask_embed[pos * h ..][0..h]
                else
                    self.weights.mask_embed[0..h];
                @memcpy(dst, m_slice);
            }
        } else {
            for (0..w) |pos| {
                const dst = self.workspace.slot_hiddens[pos * h ..][0..h];
                if (tail_pos_mask) {
                    if (pos == 0) {
                        @memcpy(dst, self.workspace.ctx_hidden[0..h]);
                    } else {
                        const m_slice = self.weights.mask_embed[(pos - 1) * h ..][0..h];
                        for (0..h) |d| dst[d] = self.workspace.ctx_hidden[d] + m_slice[d];
                    }
                } else {
                    const m_slice = if (per_pos_mask)
                        self.weights.mask_embed[pos * h ..][0..h]
                    else
                        self.weights.mask_embed[0..h];
                    for (0..h) |d| {
                        dst[d] = self.workspace.ctx_hidden[d] + m_slice[d];
                    }
                }
            }
        }

        // 4. Execute `3L`/`5L` block-parallel transformer layers with cached context KV (or single-token fallback) + NeoX RoPE.
        for (self.weights.layers, 0..) |layer, li| {
            if (use_cached_ctx) {
                const layer_off = li * max_ctx * kv_dim;
                try runTransformerBlock(
                    layer,
                    self.workspace.slot_hiddens[0 .. w * h],
                    w,
                    h,
                    self.config.intermediate_size,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.config.rms_eps,
                    &.{},
                    &.{},
                    active_ctx_len,
                    self.workspace.norm_buf,
                    self.workspace.q_buf,
                    self.workspace.k_buf[kv_dim..],
                    self.workspace.v_buf[kv_dim..],
                    self.workspace.attn_out,
                    self.workspace.attn_scores,
                    self.workspace.mlp_gate,
                    self.workspace.mlp_up,
                    self.workspace.mlp_down,
                    self.config.rope_theta,
                    anchor_pos,
                    self.workspace.ctx_k_cache[layer_off .. layer_off + active_ctx_len * kv_dim],
                    self.workspace.ctx_v_cache[layer_off .. layer_off + active_ctx_len * kv_dim],
                );
            } else {
                self.projectCtxKvOnePos(layer, li, self.workspace.taps_flat, 0, self.workspace.k_buf[0..kv_dim], self.workspace.v_buf[0..kv_dim]);
                try runTransformerBlock(
                    layer,
                    self.workspace.slot_hiddens[0 .. w * h],
                    w,
                    h,
                    self.config.intermediate_size,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.config.rms_eps,
                    self.workspace.k_buf[0..kv_dim],
                    self.workspace.v_buf[0..kv_dim],
                    1,
                    self.workspace.norm_buf,
                    self.workspace.q_buf,
                    self.workspace.k_buf[kv_dim..],
                    self.workspace.v_buf[kv_dim..],
                    self.workspace.attn_out,
                    self.workspace.attn_scores,
                    self.workspace.mlp_gate,
                    self.workspace.mlp_up,
                    self.workspace.mlp_down,
                    self.config.rope_theta,
                    anchor_pos,
                    null,
                    null,
                );
            }
        }

        if (self.weights.final_norm) |fnw| {
            for (0..w) |pos| {
                applyRmsNormInPlace(self.workspace.slot_hiddens[pos * h ..][0..h], fnw, self.config.rms_eps);
            }
        }

        // 5. For `dflash-Q8` (`markov_w2_vocab_bf16`), pre-score `base_logits = lm_head(slot_hiddens[1..w])` in ONE batched GPU call!
        self.base_logits_ready = false;
        if (is_real and self.weights.markov_w2_vocab_bf16 != null and w > 1 and hasLmHeadHook(self.weights.lm_head)) {
            const n_draft: u8 = @intCast(w - 1);
            try evalLmHead(
                self.weights.lm_head,
                self.workspace.slot_hiddens[h .. w * h],
                n_draft,
                self.config.hidden_size,
                self.config.vocab_size,
                self.workspace.base_logits_buf[0 .. @as(usize, n_draft) * v],
            );
            self.base_logits_ready = true;
        }

        self.backbone_runs += 1;
    }

    /// Evaluates ONE sequential Markov head transition (`markov_w1` / `markov_w2`,
    /// `markov_rank = 256`) at slot `pos` conditioned on `prev_token` and
    /// `parent_rnn_hidden` without re-running the 3L/5L backbone.
    fn evalMarkovStep(
        self: *DFlareProposer,
        pos: u8,
        prev_token: u32,
        parent_rnn: []const f32,
        out_rnn: []f32,
    ) Error!void {
        const h: usize = self.config.hidden_size;
        const mr: usize = self.config.markov_rank;
        const v: usize = self.config.vocab_size;
        if (prev_token >= v) return error.TokenOutOfRange;

        const b_j = self.workspace.slot_hiddens[@as(usize, pos) * h ..][0..h];

        if (self.isRealGru()) {
            // Exact `dspark.cpp:365-380` 3-chunk Inkling2 Markov RNN head (`[3*mr, 2*mr + 2*h]`):
            // joint_in = [parent_rnn (mr) || markov_w1[prev_token] (mr) || b_j (h) || f_rnn (h)]
            const joint_len = 2 * mr + 2 * h;
            @memcpy(self.workspace.markov_concat[0..mr], parent_rnn[0..mr]);
            const w1_rows = self.weights.markov_w1_vocab_rows;
            const tok_clamped: usize = if (prev_token < w1_rows) prev_token else 0;
            const w1_tbl = self.weights.markov_w1_vocab_bf16.?;
            const w1_row = w1_tbl[tok_clamped * mr ..][0..mr];
            for (0..mr) |i| {
                self.workspace.markov_concat[mr + i] = quant.bf16ToF32(w1_row[i]);
            }
            @memcpy(self.workspace.markov_concat[2 * mr .. 2 * mr + h], b_j);
            @memcpy(self.workspace.markov_concat[2 * mr + h .. joint_len], self.workspace.ctx_markov[0..h]);

            const gates = self.workspace.markov_u[0 .. 3 * mr];
            if (self.weights.markov_gru_proj_bf16) |gb| {
                matVecBf16(gb, self.workspace.markov_concat[0..joint_len], 3 * mr, joint_len, gates);
            } else {
                matVecF32(self.weights.markov_gru_proj_f32.?, self.workspace.markov_concat[0..joint_len], 3 * mr, joint_len, gates);
            }
            if (self.weights.markov_gru_bias) |gbias| {
                for (0..3 * mr) |i| gates[i] += gbias[i];
            }
            const markov_vec = self.workspace.markov_out[0..mr];
            for (0..mr) |i| {
                const gate = sigmoidF32(gates[i]);
                const cand = std.math.tanh(gates[mr + i]);
                out_rnn[i] = gate * parent_rnn[i] + (1.0 - gate) * cand;
                markov_vec[i] = std.math.tanh(gates[2 * mr + i]);
            }

            if (self.weights.markov_w2_vocab_bf16) |w2_vocab| {
                // Exact `dspark.cpp:382-401`: `logits = base_logits[pos-1] + w2_vocab @ markov + w2_bias`.
                const v_rows: usize = @min(@as(usize, self.weights.markov_w2_vocab_rows), v);
                matVecBf16(w2_vocab[0 .. v_rows * mr], markov_vec, v_rows, mr, self.workspace.logits_buf[0..v_rows]);
                if (self.weights.markov_w2_vocab_bias) |wb| {
                    for (0..v_rows) |ti| self.workspace.logits_buf[ti] += wb[ti];
                }
                if (self.base_logits_ready and pos >= 1 and pos < self.config.block_size) {
                    const base_row = self.workspace.base_logits_buf[@as(usize, pos - 1) * v ..][0..v_rows];
                    for (0..v_rows) |ti| self.workspace.logits_buf[ti] += base_row[ti];
                }
                if (self.config.final_logit_softcapping > 0.0) {
                    const cap = self.config.final_logit_softcapping;
                    for (0..v_rows) |ti| {
                        self.workspace.logits_buf[ti] = cap * std.math.tanh(self.workspace.logits_buf[ti] / cap);
                    }
                }
                if (v_rows < v) {
                    @memset(self.workspace.logits_buf[v_rows..v], -1e30);
                }
                self.markov_steps += 1;
                return;
            }

            // `mtp.safetensors` `hidden_out`: `lm_head(b_j + hidden_out @ markov + hidden_out_bias)`.
            matVecF32(self.weights.markov_w2, markov_vec, h, mr, self.workspace.head_hidden[0..h]);
            if (self.weights.markov_hidden_bias) |hb| {
                for (0..h) |d| self.workspace.head_hidden[d] += hb[d];
            }
            for (0..h) |d| {
                self.workspace.head_hidden[d] += b_j[d];
            }
            try evalLmHead(
                self.weights.lm_head,
                self.workspace.head_hidden[0..h],
                1,
                self.config.hidden_size,
                self.config.vocab_size,
                self.workspace.logits_buf[0..v],
            );
            const eff_v: usize = if (self.config.unpadded_vocab_size > 0 and self.config.unpadded_vocab_size < v)
                self.config.unpadded_vocab_size
            else
                v;
            if (self.config.final_logit_softcapping > 0.0) {
                const cap = self.config.final_logit_softcapping;
                for (0..eff_v) |ti| {
                    self.workspace.logits_buf[ti] = cap * std.math.tanh(self.workspace.logits_buf[ti] / cap);
                }
            }
            if (eff_v < v) {
                @memset(self.workspace.logits_buf[eff_v..v], -1e30);
            }
            self.markov_steps += 1;
            return;
        }

        const tok_e = try resolveTokEmbed(self.weights.tok_embed, self.weights.tok_embed_hook, prev_token, h, self.workspace.tok_embed_tmp);
        if (self.weights.token_embd_norm) |ten| {
            @memcpy(self.workspace.tok_embed_tmp[0..h], tok_e);
            applyRmsNormInPlace(self.workspace.tok_embed_tmp[0..h], ten, self.config.rms_eps);
        }
        const eff_tok_e = if (self.weights.token_embd_norm != null) self.workspace.tok_embed_tmp[0..h] else tok_e;

        if (self.weights.markov_w1.len == mr * 2 * h) {
            @memcpy(self.workspace.markov_concat[0..h], b_j);
            @memcpy(self.workspace.markov_concat[h .. 2 * h], eff_tok_e);
            matVecF32(self.weights.markov_w1, self.workspace.markov_concat[0 .. 2 * h], mr, 2 * h, self.workspace.markov_u[0..mr]);
        } else {
            for (0..h) |d| {
                self.workspace.markov_concat[d] = b_j[d] + eff_tok_e[d];
            }
            matVecF32(self.weights.markov_w1, self.workspace.markov_concat[0..h], mr, h, self.workspace.markov_u[0..mr]);
        }

        if (self.weights.markov_rnn_proj) |rp| {
            matVecAccumF32(rp, parent_rnn, mr, mr, self.workspace.markov_u[0..mr]);
        } else {
            for (0..mr) |i| {
                self.workspace.markov_u[i] += parent_rnn[i];
            }
        }
        if (self.weights.markov_b1) |b1| {
            for (0..mr) |i| {
                self.workspace.markov_u[i] += b1[i];
            }
        }
        for (0..mr) |i| {
            out_rnn[i] = siluF32(self.workspace.markov_u[i]);
        }

        matVecF32(self.weights.markov_w2, out_rnn, h, mr, self.workspace.head_hidden[0..h]);
        if (self.weights.markov_hidden_bias) |hb| {
            for (0..h) |d| self.workspace.head_hidden[d] += hb[d];
        }
        for (0..h) |d| {
            self.workspace.head_hidden[d] += b_j[d];
        }
        if (self.weights.markov_norm) |mn| {
            applyRmsNormInPlace(self.workspace.head_hidden[0..h], mn, self.config.rms_eps);
        }

        try evalLmHead(
            self.weights.lm_head,
            self.workspace.head_hidden[0..h],
            1,
            self.config.hidden_size,
            self.config.vocab_size,
            self.workspace.logits_buf[0..v],
        );

        self.markov_steps += 1;
    }

    pub fn propose(
        self: *DFlareProposer,
        root_token: u32,
        target_hiddens: []const []const f32,
        sam: ?*const sam_mod.SuffixAutomaton,
        out_tree: *CandidateTree,
    ) Error!DFlareProposeResult {
        return self.proposeIntoConstrained(
            CandidateTree.capacity,
            root_token,
            target_hiddens,
            sam,
            null,
            out_tree,
        );
    }

    pub fn proposeInto(
        self: *DFlareProposer,
        comptime max_nodes: u16,
        root_token: u32,
        target_hiddens: []const []const f32,
        sam: ?*const sam_mod.SuffixAutomaton,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!DFlareProposeResult {
        return self.proposeIntoConstrained(
            max_nodes,
            root_token,
            target_hiddens,
            sam,
            null,
            out_tree,
        );
    }

    /// Populates `DraftTree(max_nodes)` with a linear chain (`top_k_per_pos == 1`,
    /// length `block_size = 8`) or a breadth-first top-k tree (`<= max_tree_nodes`),
    /// executing the `3L`/`5L` backbone ONCE and stepping the low-rank `rnn_hidden`
    /// Markov head sequentially with zero dynamic heap allocations.
    pub fn proposeIntoConstrained(
        self: *DFlareProposer,
        comptime max_nodes: u16,
        root_token: u32,
        target_hiddens: []const []const f32,
        sam: ?*const sam_mod.SuffixAutomaton,
        grammar_opt: ?accept_mod.GrammarTreeAdapter,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!DFlareProposeResult {
        out_tree.reset(root_token);
        const node_limit: u16 = @min(@as(u16, self.config.max_tree_nodes), max_nodes);
        const mr: usize = self.config.markov_rank;
        const v: usize = self.config.vocab_size;
        const w: u8 = self.config.block_size;
        const k: usize = self.config.top_k_per_pos;

        @memset(self.workspace.node_log_probs[0..node_limit], 0.0);
        @memset(self.workspace.node_rnn_hidden[0..mr], 0.0);

        const gram_active = if (grammar_opt) |ga| ga.isActive() else false;
        var node_states: [tree_mod.MAX_SUPPORTED_NODES]grammar_mod.GrammarState = undefined;
        var node_valid: [tree_mod.MAX_SUPPORTED_NODES]bool = @splat(false);
        node_valid[0] = true;
        if (grammar_opt) |ga| {
            if (ga.isActive()) node_states[0] = ga.state.snapshot();
        }

        const l1 = mergeOptionalL1Sam(
            max_nodes,
            out_tree,
            sam,
            self.config.l1_max_draft_len,
            node_limit,
            self.config.l1_seed_step_prob,
            self.config.l1_branch_step_prob,
            self.workspace.l1_draft_buf,
            self.workspace.node_log_probs,
            gram_active,
            grammar_opt,
            &node_states,
            &node_valid,
        );

        const base_backbone = self.backbone_runs;
        const base_markov = self.markov_steps;

        // 1. Run the 3L/5L block-parallel backbone strictly ONCE.
        try self.runBlockBackboneOnceForRoot(root_token, target_hiddens);

        // 2. Decode sequential draft steps through the low-rank Markov head.
        //    For real Inkling2 DFlare/DSpark (`sample_from_anchor = false`), slot 0 is the anchor
        //    token and draft slots are `1..block_size-1` (7 drafted tokens -> M = 8 total tree nodes).
        var frontier_cur: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
        var frontier_next: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
        var cur_len: usize = 1;
        frontier_cur[0] = 0;

        const inv_temp: f32 = 1.0 / self.config.temperature;
        var next_rnn_scratch: [4096]f32 = undefined;
        if (mr > next_rnn_scratch.len) return error.BufferTooSmall;
        const next_rnn = next_rnn_scratch[0..mr];

        const pos_beg: usize = if (self.isRealGru() and w > 1) 1 else 0;
        for (pos_beg..w) |pos_usize| {
            const pos: u8 = @intCast(pos_usize);
            sortFrontierDescending(max_nodes, out_tree, frontier_cur[0..cur_len]);

            var next_len: usize = 0;
            for (frontier_cur[0..cur_len]) |parent| {
                if (gram_active and !node_valid[parent]) continue;
                const parent_tok = out_tree.nodes[parent].token_id;
                const parent_rnn = self.workspace.node_rnn_hidden[@as(usize, parent) * mr ..][0..mr];

                try self.evalMarkovStep(pos, parent_tok, parent_rnn, next_rnn);

                var mask_opt: ?[]const u32 = null;
                if (gram_active) {
                    var parent_ga = grammar_opt.?;
                    var pstate = node_states[parent].snapshot();
                    parent_ga.state = &pstate;
                    mask_opt = parent_ga.fillBitmask(v) catch return error.InvalidLogits;
                }

                const cand_count = try extractDeterministicTopKMasked(
                    self.workspace.logits_buf[0..v],
                    inv_temp,
                    mask_opt,
                    self.workspace.topk_tokens[0..k],
                    self.workspace.topk_probs[0..k],
                );

                for (0..cand_count) |ki| {
                    const tok = self.workspace.topk_tokens[ki];
                    const prob = self.workspace.topk_probs[ki];
                    var child_state: grammar_mod.GrammarState = undefined;
                    if (gram_active) {
                        child_state = node_states[parent].snapshot();
                        if (!grammar_opt.?.advanceTokenOnState(&child_state, tok)) continue;
                    }

                    const child = if (out_tree.findChild(parent, tok)) |existing| blk: {
                        break :blk out_tree.insertChild(parent, tok, prob, .l2) orelse existing;
                    } else blk: {
                        if (out_tree.count >= node_limit) break :blk null;
                        break :blk out_tree.insertChild(parent, tok, prob, .l2);
                    };

                    if (child) |c_idx| {
                        self.workspace.node_log_probs[c_idx] = logProbOf(prob);
                        @memcpy(self.workspace.node_rnn_hidden[@as(usize, c_idx) * mr ..][0..mr], next_rnn);
                        if (gram_active) {
                            node_states[c_idx] = child_state;
                            node_valid[c_idx] = true;
                        }
                        if (next_len < tree_mod.MAX_SUPPORTED_NODES and
                            !containsNode(frontier_next[0..next_len], c_idx))
                        {
                            frontier_next[next_len] = c_idx;
                            next_len += 1;
                        }
                    }
                }
            }
            if (next_len == 0) break;
            @memcpy(frontier_cur[0..next_len], frontier_next[0..next_len]);
            cur_len = next_len;
        }

        return .{
            .tree_nodes = out_tree.count,
            .max_depth = out_tree.max_depth,
            .l1_match_len = l1.match_len,
            .l1_inserted = l1.inserted,
            .l1_is_seed = l1.is_seed,
            .backbone_runs = self.backbone_runs - base_backbone,
            .markov_steps = self.markov_steps - base_markov,
            .expected_tokens = out_tree.expectedTokens(),
        };
    }

    /// Returns token log-prob `ln(step_prob)` recorded for node `idx`.
    pub inline fn nodeLogProb(self: *const DFlareProposer, node_idx: u16) f32 {
        return self.workspace.node_log_probs[node_idx];
    }

    /// Applies `D7` expert-byte-budget submodular pruning (`ADR-0009`) and
    /// remaps `workspace.node_log_probs` to the compacted node indices.
    pub fn pruneWithD7(
        self: *DFlareProposer,
        comptime max_nodes: u16,
        out_tree: *tree_mod.DraftTree(max_nodes),
        cost: tree_mod.PruneCostModel,
        node_experts: ?[]const tree_mod.ExpertBitset,
        cold_experts_by_layer: ?[]const tree_mod.ExpertBitset,
        hard_cap: u16,
    ) Error!tree_mod.DraftTree(max_nodes).PruneSummary {
        return pruneTreeAndRemapLogProbs(max_nodes, out_tree, self.workspace.node_log_probs, cost, node_experts, cold_experts_by_layer, hard_cap);
    }

    /// Simulates/executes `M = 8` batched `Core x 3` verification over `out_tree`
    /// via `D9` lossless greedy acceptance (`T = 0`).
    pub fn verifyCore3BatchGreedy(
        comptime max_nodes: u16,
        tree: *const tree_mod.DraftTree(max_nodes),
        target_argmax: []const u32,
        out_accepted_nodes: []u16,
        out_tokens: []u32,
    ) Error!Core3BatchVerifyResult {
        const acc = try accept_mod.verifyGreedy(max_nodes, tree, target_argmax, out_accepted_nodes, out_tokens);
        return .{
            .batch_size_m = tree.count,
            .core_passes = INKLING2_CORE_PASSES,
            .accept = acc,
        };
    }

    /// Executes `M = 8` batched `Core x 3` verification via `D9` lossless
    /// stochastic rejection sampling (`T > 0`).
    pub fn verifyCore3BatchStochastic(
        comptime max_nodes: u16,
        tree: *const tree_mod.DraftTree(max_nodes),
        target_logits_f32: []const f32,
        draft_probs: ?[]const f32,
        vocab_size: usize,
        temperature: f32,
        rng: std.Random,
        residual_scratch: []f32,
        out_accepted_nodes: []u16,
        out_tokens: []u32,
    ) Error!Core3BatchVerifyResult {
        const acc = try accept_mod.verifyStochasticF32(
            max_nodes,
            tree,
            target_logits_f32,
            vocab_size,
            draft_probs,
            temperature,
            rng,
            residual_scratch,
            out_accepted_nodes,
            out_tokens,
        );
        return .{
            .batch_size_m = tree.count,
            .core_passes = INKLING2_CORE_PASSES,
            .accept = acc,
        };
    }
};

// ============================================================================
// Standalone `arch = dflash` Bundle Loader
// ============================================================================

/// Startup-loaded standalone `DFlare` / `arch = dflash` `.amf` bundle owning decoded
/// draft weights (compact BF16 for large models, FP32 for small models) and a 16 KiB-aligned `DFlareWorkspace`.
pub const LoadedDFlareBundle = struct {
    owned_f32_bufs: [][]f32,
    owned_u16_bufs: [][]u16,
    layers: []TransformerBlockWeights,
    workspace: DFlareWorkspace,
    proposer: DFlareProposer,

    pub const LoadError = Error || amf.ParseError || amf.IoError || error{
        MissingTensor,
        OutOfMemory,
    };

    pub fn loadFromAmfFile(
        allocator: std.mem.Allocator,
        path: [:0]const u8,
        cfg_in: DFlareConfig,
        lm_head_override: ?LmHeadHook,
        budget_override: ?budget_mod.Budget,
    ) LoadError!*LoadedDFlareBundle {
        return loadFromAmfFileWithHooks(allocator, path, cfg_in, null, lm_head_override, budget_override);
    }

    pub fn loadFromAmfFileWithHooks(
        allocator: std.mem.Allocator,
        path: [:0]const u8,
        cfg_in: DFlareConfig,
        tok_embed_override: ?TokEmbedHook,
        lm_head_override: ?LmHeadHook,
        budget_override: ?budget_mod.Budget,
    ) LoadError!*LoadedDFlareBundle {
        var amf_file = try amf.AmfFile.open(allocator, path);
        defer amf_file.close(allocator);
        return loadFromAmfWithHooks(allocator, &amf_file, cfg_in, tok_embed_override, lm_head_override, budget_override);
    }

    pub fn loadFromAmf(
        allocator: std.mem.Allocator,
        amf_file: *const amf.AmfFile,
        cfg_in: DFlareConfig,
        lm_head_override: ?LmHeadHook,
        budget_override: ?budget_mod.Budget,
    ) LoadError!*LoadedDFlareBundle {
        return loadFromAmfWithHooks(allocator, amf_file, cfg_in, null, lm_head_override, budget_override);
    }

    pub fn loadFromAmfWithHooks(
        allocator: std.mem.Allocator,
        amf_file: *const amf.AmfFile,
        cfg_in: DFlareConfig,
        tok_embed_override: ?TokEmbedHook,
        lm_head_override: ?LmHeadHook,
        budget_override: ?budget_mod.Budget,
    ) LoadError!*LoadedDFlareBundle {
        var cfg = cfg_in;
        try cfg.validate();

        const h: usize = cfg.hidden_size;
        const inter: usize = cfg.intermediate_size;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();
        const mr: usize = cfg.markov_rank;
        const v: usize = cfg.vocab_size;
        const use_bf16_layers: bool = (h >= 1024);

        var f32_pool: [256][]f32 = undefined;
        var f32_len: usize = 0;
        errdefer {
            for (0..f32_len) |i| allocator.free(f32_pool[i]);
        }
        var u16_pool: [256][]u16 = undefined;
        var u16_len: usize = 0;
        errdefer {
            for (0..u16_len) |i| allocator.free(u16_pool[i]);
        }

        // Target tap fusion: dense `[h, taps*h]` OR per-layer scalar `[rows, taps]` (`fuse_weight` / `ctx_mix`).
        var fc: ?[]const f32 = null;
        var fuse_scalar: ?[]const f32 = null;
        if (findAnyEntry(amf_file, &.{ "dflare.fc.weight", "fc.weight", "fuse_weight" })) |fe| {
            const vec = try loadEntryIntoF32Pool(allocator, amf_file, fe, &f32_pool, &f32_len);
            if (vec.len == cfg.num_target_taps * h * h) {
                fc = vec;
            } else if (vec.len >= cfg.num_target_taps and (vec.len % cfg.num_target_taps) == 0) {
                // `mtp.safetensors` stores raw pre-softmax `fuse_weight [rows, taps]` logits;
                // apply row-wise softmax to match `dflash-Q8` `ctx_mix.weight` probabilities.
                const taps_us: usize = cfg.num_target_taps;
                const n_rows: usize = vec.len / taps_us;
                for (0..n_rows) |ri| {
                    const row = vec[ri * taps_us ..][0..taps_us];
                    var mx: f32 = -std.math.inf(f32);
                    for (row) |val| {
                        if (val > mx) mx = val;
                    }
                    var sum: f32 = 0.0;
                    for (row) |*val| {
                        val.* = @exp(val.* - mx);
                        sum += val.*;
                    }
                    if (sum > 0.0) {
                        const inv = 1.0 / sum;
                        for (row) |*val| val.* *= inv;
                    }
                }
                fuse_scalar = vec;
            } else {
                return error.DimensionMismatch;
            }
        } else if (amf_file.parsed.find("layers.0.ctx_mix.weight") != null) {
            // `dflash-Q8` GGUF stores per-layer `layers.{l}.ctx_mix.weight [taps]` + `markov_ctx_mix.weight [taps]`.
            const rows: usize = @as(usize, cfg.num_layers) + 1;
            const mix = try allocator.alloc(f32, rows * cfg.num_target_taps);
            if (f32_len >= f32_pool.len) {
                allocator.free(mix);
                return error.Overflow;
            }
            f32_pool[f32_len] = mix;
            f32_len += 1;
            @memset(mix, 1.0 / @as(f32, @floatFromInt(cfg.num_target_taps)));
            var nb: [64]u8 = undefined;
            for (0..cfg.num_layers) |l| {
                const nm = std.fmt.bufPrint(&nb, "layers.{d}.ctx_mix.weight", .{l}) catch return error.Overflow;
                if (amf_file.parsed.find(nm)) |ce| {
                    const cvec = try readEntryAsF32(allocator, amf_file, ce);
                    defer allocator.free(cvec);
                    const ncopy = @min(cvec.len, @as(usize, cfg.num_target_taps));
                    @memcpy(mix[l * cfg.num_target_taps ..][0..ncopy], cvec[0..ncopy]);
                }
            }
            if (amf_file.parsed.find("markov_ctx_mix.weight")) |me| {
                const mvec = try readEntryAsF32(allocator, amf_file, me);
                defer allocator.free(mvec);
                const ncopy = @min(mvec.len, @as(usize, cfg.num_target_taps));
                @memcpy(mix[@as(usize, cfg.num_layers) * cfg.num_target_taps ..][0..ncopy], mvec[0..ncopy]);
            }
            fuse_scalar = mix;
        }
        const fc_norm = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{ "dflare.fc_norm.weight", "fc_norm.weight" }, h, &f32_pool, &f32_len);

        const mask_embed = try loadAmfTensorAnyLen256(allocator, amf_file, &.{ "dflare.mask_embed.weight", "dflare.mask_embed", "mask_embed", "mask_embd.weight" }, &f32_pool, &f32_len);
        const tok_embed_norm = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{"token_embd_norm.weight"}, h, &f32_pool, &f32_len);

        var tok_embed: []const f32 = &.{};
        if (tok_embed_override == null) {
            if (findAnyEntry(amf_file, &.{ "dflare.tok_embed.weight", "token_embd.weight", "embed_tokens.weight" })) |te| {
                const tv = try loadEntryIntoF32Pool(allocator, amf_file, te, &f32_pool, &f32_len);
                if (tv.len != v * h) return error.DimensionMismatch;
                tok_embed = tv;
            } else {
                return error.MissingTensor;
            }
        }
        const final_norm = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{ "dflare.norm.weight", "output_norm.weight", "norm.weight" }, h, &f32_pool, &f32_len);

        const layers = try allocator.alloc(TransformerBlockWeights, cfg.num_layers);
        errdefer allocator.free(layers);

        var nb0: [80]u8 = undefined;
        var nb1: [80]u8 = undefined;
        var nb2: [80]u8 = undefined;
        for (0..cfg.num_layers) |l| {
            const in_norm = try loadLayerF32Opt(allocator, amf_file, l, &.{ "input_layernorm.weight", "attn_norm.weight" }, h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
            const post_norm = try loadLayerF32Opt(allocator, amf_file, l, &.{ "post_attention_layernorm.weight", "post_norm.weight", "ffn_norm.weight" }, h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
            const ctx_norm = try loadLayerF32Opt(allocator, amf_file, l, &.{ "dflare_norm.weight", "ctx_norm.weight" }, h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
            const q_norm = try loadLayerF32Opt(allocator, amf_file, l, &.{ "self_attn.q_norm.weight", "attn.q_norm.weight", "attn_q_norm.weight" }, cfg.head_dim, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
            const k_norm = try loadLayerF32Opt(allocator, amf_file, l, &.{ "self_attn.k_norm.weight", "attn.k_norm.weight", "attn_k_norm.weight" }, cfg.head_dim, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
            const k_norm_ctx = try loadLayerF32Opt(allocator, amf_file, l, &.{ "self_attn.k_norm_ctx.weight", "attn_k_ctx_norm.weight" }, cfg.head_dim, &nb0, &nb1, &nb2, &f32_pool, &f32_len);

            var tb: TransformerBlockWeights = .{
                .in_norm_weight = in_norm,
                .q_norm_weight = q_norm,
                .k_norm_weight = k_norm,
                .k_norm_ctx_weight = k_norm_ctx,
                .ctx_norm_weight = ctx_norm,
                .post_norm_weight = post_norm,
            };
            if (use_bf16_layers) {
                tb.q_proj_bf16 = try loadLayerBf16(allocator, amf_file, l, &.{ "self_attn.q_proj.weight", "attn.q_proj.weight", "attn_q.weight" }, q_dim * h, &nb0, &nb1, &nb2, &u16_pool, &u16_len);
                tb.k_proj_bf16 = try loadLayerBf16(allocator, amf_file, l, &.{ "self_attn.k_proj.weight", "attn.k_proj.weight", "attn_k.weight" }, kv_dim * h, &nb0, &nb1, &nb2, &u16_pool, &u16_len);
                tb.v_proj_bf16 = try loadLayerBf16(allocator, amf_file, l, &.{ "self_attn.v_proj.weight", "attn.v_proj.weight", "attn_v.weight" }, kv_dim * h, &nb0, &nb1, &nb2, &u16_pool, &u16_len);
                tb.o_proj_bf16 = try loadLayerBf16(allocator, amf_file, l, &.{ "self_attn.o_proj.weight", "attn.o_proj.weight", "attn_output.weight" }, h * q_dim, &nb0, &nb1, &nb2, &u16_pool, &u16_len);
                tb.k_proj_ctx_bf16 = try loadLayerBf16Opt(allocator, amf_file, l, &.{ "self_attn.k_proj_ctx.weight", "attn.k_proj_ctx.weight" }, kv_dim * h, &nb0, &nb1, &nb2, &u16_pool, &u16_len);
                tb.v_proj_ctx_bf16 = try loadLayerBf16Opt(allocator, amf_file, l, &.{ "self_attn.v_proj_ctx.weight", "attn.v_proj_ctx.weight" }, kv_dim * h, &nb0, &nb1, &nb2, &u16_pool, &u16_len);
                tb.mlp_gate_bf16 = try loadLayerBf16(allocator, amf_file, l, &.{ "mlp.gate_proj.weight", "ffn_gate.weight" }, inter * h, &nb0, &nb1, &nb2, &u16_pool, &u16_len);
                tb.mlp_up_bf16 = try loadLayerBf16(allocator, amf_file, l, &.{ "mlp.up_proj.weight", "ffn_up.weight" }, inter * h, &nb0, &nb1, &nb2, &u16_pool, &u16_len);
                tb.mlp_down_bf16 = try loadLayerBf16(allocator, amf_file, l, &.{ "mlp.down_proj.weight", "ffn_down.weight" }, h * inter, &nb0, &nb1, &nb2, &u16_pool, &u16_len);
            } else {
                tb.q_proj = try loadLayerF32(allocator, amf_file, l, &.{ "self_attn.q_proj.weight", "attn.q_proj.weight", "attn_q.weight" }, q_dim * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                tb.k_proj = try loadLayerF32(allocator, amf_file, l, &.{ "self_attn.k_proj.weight", "attn.k_proj.weight", "attn_k.weight" }, kv_dim * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                tb.v_proj = try loadLayerF32(allocator, amf_file, l, &.{ "self_attn.v_proj.weight", "attn.v_proj.weight", "attn_v.weight" }, kv_dim * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                tb.o_proj = try loadLayerF32(allocator, amf_file, l, &.{ "self_attn.o_proj.weight", "attn.o_proj.weight", "attn_output.weight" }, h * q_dim, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                tb.k_proj_ctx = try loadLayerF32Opt(allocator, amf_file, l, &.{ "self_attn.k_proj_ctx.weight", "attn.k_proj_ctx.weight" }, kv_dim * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                tb.v_proj_ctx = try loadLayerF32Opt(allocator, amf_file, l, &.{ "self_attn.v_proj_ctx.weight", "attn.v_proj_ctx.weight" }, kv_dim * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                tb.mlp_gate_proj = try loadLayerF32(allocator, amf_file, l, &.{ "mlp.gate_proj.weight", "ffn_gate.weight" }, inter * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                tb.mlp_up_proj = try loadLayerF32(allocator, amf_file, l, &.{ "mlp.up_proj.weight", "ffn_up.weight" }, inter * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                tb.mlp_down_proj = try loadLayerF32(allocator, amf_file, l, &.{ "mlp.down_proj.weight", "ffn_down.weight" }, h * inter, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
            }
            layers[l] = tb;
        }

        // Markov head: detect real 3-gate GRU (`[3*mr, 2*h + 2*mr]`) vs standard simple Markov head!
        const gru_entry = findAnyEntry(amf_file, &.{ "markov_head.rnn_proj.weight", "joint_proj.weight" });
        const is_real_gru: bool = if (gru_entry) |ge| (ge.shape[0] == 3 * mr and ge.shape[1] == 2 * h + 2 * mr) else false;

        var markov_w1: []const f32 = &.{};
        var markov_w1_vocab_bf16: ?[]const u16 = null;
        var markov_w1_vocab_rows: u32 = 0;
        var markov_gru_bf16: ?[]const u16 = null;
        var markov_gru_f32: ?[]const f32 = null;
        var markov_gru_bias: ?[]const f32 = null;
        var markov_rnn: ?[]const f32 = null;
        var markov_b1: ?[]const f32 = null;
        var markov_w2: []const f32 = &.{};
        var markov_hidden_bias: ?[]const f32 = null;
        var markov_w2_vocab_bf16: ?[]const u16 = null;
        var markov_w2_vocab_bias: ?[]const f32 = null;
        var markov_w2_vocab_rows: u32 = 0;
        const markov_ctx_norm = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{ "markov_head.dflare_norm.weight", "markov_ctx_norm.weight" }, h, &f32_pool, &f32_len);
        const markov_norm = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{ "dflare.markov_norm.weight", "markov_norm.weight" }, h, &f32_pool, &f32_len);

        if (is_real_gru) {
            const ge = gru_entry.?;
            if (use_bf16_layers) {
                markov_gru_bf16 = try loadEntryIntoBf16Pool(allocator, amf_file, ge, &u16_pool, &u16_len);
            } else {
                markov_gru_f32 = try loadEntryIntoF32Pool(allocator, amf_file, ge, &f32_pool, &f32_len);
            }
            markov_gru_bias = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{ "markov_head.rnn_proj.bias", "joint_proj.bias" }, 3 * mr, &f32_pool, &f32_len);
            const w1e = findAnyEntry(amf_file, &.{ "markov_head.markov_w1.weight", "markov_w1.weight" }) orelse return error.MissingTensor;
            markov_w1_vocab_bf16 = try loadEntryIntoBf16Pool(allocator, amf_file, w1e, &u16_pool, &u16_len);
            markov_w1_vocab_rows = w1e.shape[0];

            if (findAnyEntry(amf_file, &.{ "markov_head.hidden_out.weight", "dflare.markov_w2.weight" })) |hoe| {
                markov_w2 = try loadEntryIntoF32Pool(allocator, amf_file, hoe, &f32_pool, &f32_len);
                markov_hidden_bias = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{"markov_head.hidden_out.bias"}, h, &f32_pool, &f32_len);
            } else if (amf_file.parsed.find("markov_w2.weight")) |w2e| {
                markov_w2_vocab_bf16 = try loadEntryIntoBf16Pool(allocator, amf_file, w2e, &u16_pool, &u16_len);
                markov_w2_vocab_rows = w2e.shape[0];
                if (amf_file.parsed.find("markov_w2.bias")) |w2b| {
                    markov_w2_vocab_bias = try loadEntryIntoF32Pool(allocator, amf_file, w2b, &f32_pool, &f32_len);
                }
            } else {
                return error.MissingTensor;
            }
        } else {
            markov_w1 = try loadAmfTensorAnyLen256(allocator, amf_file, &.{ "dflare.markov_w1.weight", "markov_head.markov_w1.weight", "markov_w1.weight" }, &f32_pool, &f32_len);
            markov_rnn = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{ "dflare.rnn_proj.weight", "markov_head.rnn_proj.weight", "markov_rnn_proj.weight" }, mr * mr, &f32_pool, &f32_len);
            markov_b1 = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{ "dflare.markov_b1", "markov_head.rnn_proj.bias", "markov_b1" }, mr, &f32_pool, &f32_len);
            markov_w2 = try loadAmfTensorAsF32256(allocator, amf_file, &.{ "dflare.markov_w2.weight", "markov_head.hidden_out.weight", "markov_w2.weight" }, h * mr, &f32_pool, &f32_len);
            markov_hidden_bias = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{"markov_head.hidden_out.bias"}, h, &f32_pool, &f32_len);
        }

        const lm_head: LmHeadHook = lm_head_override orelse blk: {
            if (markov_w2_vocab_bf16 != null) {
                break :blk .{ .dense_f32 = &.{} };
            }
            const lm_w = try loadAmfTensorAsF32256(allocator, amf_file, &.{ "dflare.lm_head.weight", "output.weight", "lm_head.weight" }, v * h, &f32_pool, &f32_len);
            break :blk .{ .dense_f32 = lm_w };
        };

        const owned_f32 = try allocator.alloc([]f32, f32_len);
        errdefer allocator.free(owned_f32);
        @memcpy(owned_f32, f32_pool[0..f32_len]);

        const owned_u16 = try allocator.alloc([]u16, u16_len);
        errdefer allocator.free(owned_u16);
        @memcpy(owned_u16, u16_pool[0..u16_len]);

        if (is_real_gru) {
            if (cfg.rope_theta <= 0.0) cfg.rope_theta = 1_000_000.0;
            if (cfg.unpadded_vocab_size == 0 and markov_w2_vocab_rows > 0 and markov_w2_vocab_rows < v) {
                cfg.unpadded_vocab_size = markov_w2_vocab_rows;
            }
        }

        var ws = try DFlareWorkspace.init(cfg, budget_override);
        errdefer ws.deinit();

        const weights: DFlareWeights = .{
            .target_fusion_proj = fc,
            .fuse_scalar_weights = fuse_scalar,
            .target_fusion_norm = fc_norm,
            .mask_embed = mask_embed,
            .tok_embed = tok_embed,
            .tok_embed_hook = tok_embed_override,
            .token_embd_norm = tok_embed_norm,
            .layers = layers,
            .final_norm = final_norm,
            .markov_w1 = markov_w1,
            .markov_w1_vocab_bf16 = markov_w1_vocab_bf16,
            .markov_w1_vocab_rows = markov_w1_vocab_rows,
            .markov_gru_proj_bf16 = markov_gru_bf16,
            .markov_gru_proj_f32 = markov_gru_f32,
            .markov_gru_bias = markov_gru_bias,
            .markov_ctx_norm = markov_ctx_norm,
            .markov_rnn_proj = markov_rnn,
            .markov_b1 = markov_b1,
            .markov_w2 = markov_w2,
            .markov_hidden_bias = markov_hidden_bias,
            .markov_w2_vocab_bf16 = markov_w2_vocab_bf16,
            .markov_w2_vocab_bias = markov_w2_vocab_bias,
            .markov_w2_vocab_rows = markov_w2_vocab_rows,
            .markov_norm = markov_norm,
            .lm_head = lm_head,
        };

        const bundle = try allocator.create(LoadedDFlareBundle);
        errdefer allocator.destroy(bundle);

        bundle.* = .{
            .owned_f32_bufs = owned_f32,
            .owned_u16_bufs = owned_u16,
            .layers = layers,
            .workspace = ws,
            .proposer = undefined,
        };
        bundle.proposer = try DFlareProposer.init(cfg, weights, &bundle.workspace);
        return bundle;
    }

    pub fn deinit(self: *LoadedDFlareBundle, allocator: std.mem.Allocator) void {
        self.workspace.deinit();
        allocator.free(self.layers);
        for (self.owned_f32_bufs) |b| allocator.free(b);
        allocator.free(self.owned_f32_bufs);
        for (self.owned_u16_bufs) |b| allocator.free(b);
        allocator.free(self.owned_u16_bufs);
        allocator.destroy(self);
    }
};

// ============================================================================
// DFly2 (`Hy4` `5L`, `block_size = 7`, `confidence_head` early-stop)
// ============================================================================

pub const DFly2Config = struct {
    num_layers: u8 = DFLY2_DEFAULT_LAYERS,
    block_size: u8 = DFLY2_BLOCK_SIZE,
    top_k_per_pos: u8 = 1,
    max_tree_nodes: u8 = 32,
    hidden_size: u32,
    intermediate_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    num_target_kv_layers: u8 = DFLY2_DEFAULT_LAYERS,
    max_target_kv_len: u32 = 64,
    /// Dynamic early-stop confidence threshold (`sigmoid(confidence_head) < threshold` truncates tail).
    confidence_threshold: f32 = 0.5,
    /// Minimum draft steps emitted before dynamic truncation may trigger.
    min_draft_steps: u8 = 0,
    rms_eps: f32 = 1e-6,
    temperature: f32 = 1.0,
    l1_seed_step_prob: f32 = 0.90,
    l1_branch_step_prob: f32 = 0.55,
    l1_max_draft_len: u8 = 8,

    pub inline fn qDim(self: DFly2Config) u32 {
        return self.num_attention_heads * self.head_dim;
    }

    pub inline fn kvDim(self: DFly2Config) u32 {
        return self.num_key_value_heads * self.head_dim;
    }

    pub fn validate(self: DFly2Config) Error!void {
        if (self.num_layers == 0 or self.num_layers > MAX_LAYERS) return error.InvalidConfig;
        if (self.block_size == 0 or self.block_size > MAX_BLOCK_SIZE) return error.InvalidConfig;
        if (self.top_k_per_pos == 0 or self.top_k_per_pos > MAX_TOP_K_PER_POS) return error.InvalidConfig;
        if (self.max_tree_nodes < 2 or self.max_tree_nodes > MAX_TREE_NODES) return error.InvalidConfig;
        if (self.hidden_size == 0 or self.intermediate_size == 0 or self.vocab_size == 0) return error.InvalidConfig;
        if (self.num_attention_heads == 0 or self.num_key_value_heads == 0 or self.head_dim == 0) return error.InvalidConfig;
        if ((self.num_attention_heads % self.num_key_value_heads) != 0) return error.InvalidConfig;
        if (self.top_k_per_pos > self.vocab_size) return error.InvalidConfig;
        if (self.num_target_kv_layers == 0 or self.num_target_kv_layers > MAX_LAYERS) return error.InvalidConfig;
        if (self.max_target_kv_len == 0) return error.InvalidConfig;
        if (!std.math.isFinite(self.confidence_threshold) or self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0) {
            return error.InvalidConfig;
        }
        if (self.min_draft_steps > self.block_size) return error.InvalidConfig;
        if (!std.math.isFinite(self.rms_eps) or self.rms_eps <= 0.0) return error.InvalidConfig;
        if (!std.math.isFinite(self.temperature) or self.temperature <= 0.0) return error.InvalidConfig;
        if (self.l1_max_draft_len > MAX_L1_DRAFT_LEN) return error.InvalidConfig;
    }

    /// Parses `dfly2_config.json` with target model fallbacks.
    pub fn fromJson(
        allocator: std.mem.Allocator,
        json_bytes: []const u8,
        fallback_hidden: u32,
        fallback_vocab: u32,
        fallback_max_kv_len: u32,
    ) (Error || error{OutOfMemory})!DFly2Config {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch return error.InvalidConfig;
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidConfig;
        const obj = parsed.value.object;

        const num_layers: u8 = @intCast(jsonU32Opt(obj, "num_hidden_layers") orelse DFLY2_DEFAULT_LAYERS);
        const block_size: u8 = @intCast(jsonU32Opt(obj, "block_size") orelse DFLY2_BLOCK_SIZE);
        const hidden_size: u32 = jsonU32Opt(obj, "hidden_size") orelse fallback_hidden;
        const inter_size: u32 = jsonU32Opt(obj, "intermediate_size") orelse (hidden_size * 2);
        const num_heads: u32 = jsonU32Opt(obj, "num_attention_heads") orelse 4;
        const num_kv_heads: u32 = jsonU32Opt(obj, "num_key_value_heads") orelse num_heads;
        const qk_nope: u32 = jsonU32Opt(obj, "qk_nope_head_dim") orelse 0;
        const qk_rope: u32 = jsonU32Opt(obj, "qk_rope_head_dim") orelse 0;
        const v_dim: u32 = jsonU32Opt(obj, "v_head_dim") orelse 0;
        const head_dim: u32 = jsonU32Opt(obj, "head_dim") orelse (if (qk_nope + qk_rope > 0) @max(qk_nope + qk_rope, v_dim) else (hidden_size / @max(num_heads, 1)));
        const vocab_size: u32 = if (fallback_vocab > 0) fallback_vocab else (jsonU32Opt(obj, "vocab_size") orelse 0);
        const rms_eps: f32 = jsonF32Opt(obj, "rms_norm_eps") orelse 1e-5;
        const conf_thr: f32 = jsonF32Opt(obj, "confidence_threshold") orelse 0.5;
        var num_kv_layers: u8 = num_layers;
        if (obj.get("dfly_target_kv_layers")) |v| {
            if (v == .array and v.array.items.len > 0 and v.array.items.len <= MAX_LAYERS) {
                num_kv_layers = @intCast(v.array.items.len);
            }
        }
        const cfg: DFly2Config = .{
            .num_layers = num_layers,
            .block_size = block_size,
            .hidden_size = hidden_size,
            .intermediate_size = inter_size,
            .num_attention_heads = num_heads,
            .num_key_value_heads = num_kv_heads,
            .head_dim = head_dim,
            .vocab_size = vocab_size,
            .num_target_kv_layers = num_kv_layers,
            .max_target_kv_len = @max(fallback_max_kv_len, @as(u32, 64)),
            .confidence_threshold = conf_thr,
            .rms_eps = rms_eps,
        };
        try cfg.validate();
        return cfg;
    }
};

/// Read-only view of one target layer's KV cache passed into `DFly2Proposer` (`dfly_target_kv_layers`).
pub const TargetKvLayerView = struct {
    k: []const f32, // [kv_len, kv_dim]
    v: []const f32, // [kv_len, kv_dim]
    kv_len: usize,
};

pub const DFly2Weights = struct {
    /// `mtp_fusion`: `[hidden_size, 2 * hidden_size]` fusing `[norm(root_embed) || norm(target_hidden)]`.
    mtp_fusion_proj: []const f32,
    mtp_enorm_weight: ?[]const f32 = null,
    mtp_hnorm_weight: ?[]const f32 = null,
    mtp_fusion_gate: ?[]const f32 = null,
    /// Optional residual `hidden_correction` projection `[hidden_size, hidden_size]`.
    hidden_correction_proj: ?[]const f32 = null,
    hidden_correction_norm: ?[]const f32 = null,
    /// Optional SwiGLU `hidden_correction` (`W23` `synth-hy4`: `gate_proj`, `up_proj`, `down_proj`).
    hidden_correction_embed_norm: ?[]const f32 = null,
    hidden_correction_gate: ?[]const f32 = null,
    hidden_correction_up: ?[]const f32 = null,
    hidden_correction_down: ?[]const f32 = null,
    hidden_correction_inter: u32 = 0,
    mask_embed: []const f32,
    tok_embed: []const f32 = &.{},
    tok_embed_hook: ?TokEmbedHook = null,
    layers: []const TransformerBlockWeights,
    final_norm: ?[]const f32 = null,
    /// `confidence_head` linear weight `[hidden_size]` producing logit for `sigmoid(dot + bias)`.
    confidence_weight: []const f32,
    confidence_bias: f32 = 0.0,
    confidence_norm: ?[]const f32 = null,
    lm_head: LmHeadHook,

    pub fn validate(self: DFly2Weights, cfg: DFly2Config) Error!void {
        const h: usize = cfg.hidden_size;
        const inter: usize = cfg.intermediate_size;
        const v: usize = cfg.vocab_size;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();

        const h2 = std.math.mul(usize, h, 2) catch return error.Overflow;
        const hh2 = std.math.mul(usize, h, h2) catch return error.Overflow;
        const hh = std.math.mul(usize, h, h) catch return error.Overflow;
        const vh = std.math.mul(usize, v, h) catch return error.Overflow;

        if (self.mtp_fusion_proj.len != hh2) return error.InvalidWeights;
        if (self.mtp_enorm_weight) |w| {
            if (w.len != h) return error.InvalidWeights;
        }
        if (self.mtp_hnorm_weight) |w| {
            if (w.len != h) return error.InvalidWeights;
        }
        if (self.hidden_correction_proj) |hc| {
            if (hc.len != hh) return error.InvalidWeights;
        }
        if (self.hidden_correction_norm) |hcn| {
            if (hcn.len != h) return error.InvalidWeights;
        }
        const mask_ok = self.mask_embed.len == h or
            self.mask_embed.len == (std.math.mul(usize, cfg.block_size, h) catch return error.Overflow);
        if (!mask_ok) return error.InvalidWeights;
        if (self.tok_embed_hook == null and self.tok_embed.len != vh) return error.InvalidWeights;
        if (self.layers.len != cfg.num_layers) return error.InvalidWeights;
        for (self.layers) |layer| {
            try layer.validate(h, inter, q_dim, kv_dim, cfg.head_dim);
        }
        if (self.final_norm) |fnw| {
            if (fnw.len != h) return error.InvalidWeights;
        }
        if (self.confidence_weight.len != h) return error.InvalidWeights;
        if (!std.math.isFinite(self.confidence_bias)) return error.InvalidWeights;
        if (self.confidence_norm) |cn| {
            if (cn.len != h) return error.InvalidWeights;
        }
        try validateLmHeadHook(self.lm_head, vh);
    }
};

/// Preallocated 16 KiB-aligned `WiredArena` workspace for `DFly2Proposer`.
pub const DFly2Workspace = struct {
    arena: arena_mod.WiredArena,
    config: DFly2Config,
    mtp_concat: []f32,
    fused_hidden: []f32,
    corr_buf: []f32,
    tok_embed_tmp: []f32,
    slot_hiddens: []f32,
    norm_buf: []f32,
    q_buf: []f32,
    k_buf: []f32,
    v_buf: []f32,
    attn_out: []f32,
    attn_scores: []f32,
    mlp_gate: []f32,
    mlp_up: []f32,
    mlp_down: []f32,
    conf_norm_buf: []f32,
    step_confidences: []f32,
    logits_buf: []f32,
    topk_tokens: []u32,
    topk_probs: []f32,
    node_log_probs: []f32,
    l1_draft_buf: []u32,

    pub fn requiredBytes(cfg: DFly2Config) Error!usize {
        try cfg.validate();
        const h: usize = cfg.hidden_size;
        const inter: usize = cfg.intermediate_size;
        const w: usize = cfg.block_size;
        const v: usize = cfg.vocab_size;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();
        const max_kv: usize = cfg.max_target_kv_len;
        const max_nodes: usize = cfg.max_tree_nodes;

        const chunks = [_]usize{
            try alignUp64((2 * h) * @sizeOf(f32)), // mtp_concat
            try alignUp64(h * @sizeOf(f32)), // fused_hidden
            try alignUp64(h * @sizeOf(f32)), // corr_buf
            try alignUp64(h * @sizeOf(f32)), // tok_embed_tmp
            try alignUp64((w * h) * @sizeOf(f32)), // slot_hiddens
            try alignUp64((w * h) * @sizeOf(f32)), // norm_buf
            try alignUp64((w * q_dim) * @sizeOf(f32)), // q_buf
            try alignUp64((w * kv_dim) * @sizeOf(f32)), // k_buf
            try alignUp64((w * kv_dim) * @sizeOf(f32)), // v_buf
            try alignUp64((w * q_dim) * @sizeOf(f32)), // attn_out
            try alignUp64((max_kv + w) * @sizeOf(f32)), // attn_scores
            try alignUp64((w * inter) * @sizeOf(f32)), // mlp_gate
            try alignUp64((w * inter) * @sizeOf(f32)), // mlp_up
            try alignUp64((w * h) * @sizeOf(f32)), // mlp_down
            try alignUp64(h * @sizeOf(f32)), // conf_norm_buf
            try alignUp64(w * @sizeOf(f32)), // step_confidences
            try alignUp64((w * v) * @sizeOf(f32)), // logits_buf
            try alignUp64((w * cfg.top_k_per_pos) * @sizeOf(u32)), // topk_tokens
            try alignUp64((w * cfg.top_k_per_pos) * @sizeOf(f32)), // topk_probs
            try alignUp64(max_nodes * @sizeOf(f32)), // node_log_probs
            try alignUp64(MAX_L1_DRAFT_LEN * @sizeOf(u32)), // l1_draft_buf
        };
        var total: usize = 0;
        for (chunks) |c| total = std.math.add(usize, total, c) catch return error.Overflow;
        return page.alignUp(total) catch return error.Overflow;
    }

    pub fn init(cfg: DFly2Config, budget_override: ?budget_mod.Budget) Error!DFly2Workspace {
        const cap_bytes = try requiredBytes(cfg);
        const wired_budget = budget_override orelse try defaultWiredBudget();
        try wired_budget.checkPlan(cap_bytes);

        var mem_arena = try arena_mod.WiredArena.init(cap_bytes, wired_budget);
        errdefer mem_arena.deinit();

        const h: usize = cfg.hidden_size;
        const inter: usize = cfg.intermediate_size;
        const w: usize = cfg.block_size;
        const v: usize = cfg.vocab_size;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();
        const max_kv: usize = cfg.max_target_kv_len;
        const max_nodes: usize = cfg.max_tree_nodes;

        const mtp_concat = try allocArenaSlice(f32, &mem_arena, 2 * h);
        const fused_hidden = try allocArenaSlice(f32, &mem_arena, h);
        const corr_buf = try allocArenaSlice(f32, &mem_arena, h);
        const tok_embed_tmp = try allocArenaSlice(f32, &mem_arena, h);
        const slot_hiddens = try allocArenaSlice(f32, &mem_arena, w * h);
        const norm_buf = try allocArenaSlice(f32, &mem_arena, w * h);
        const q_buf = try allocArenaSlice(f32, &mem_arena, w * q_dim);
        const k_buf = try allocArenaSlice(f32, &mem_arena, w * kv_dim);
        const v_buf = try allocArenaSlice(f32, &mem_arena, w * kv_dim);
        const attn_out = try allocArenaSlice(f32, &mem_arena, w * q_dim);
        const attn_scores = try allocArenaSlice(f32, &mem_arena, max_kv + w);
        const mlp_gate = try allocArenaSlice(f32, &mem_arena, w * inter);
        const mlp_up = try allocArenaSlice(f32, &mem_arena, w * inter);
        const mlp_down = try allocArenaSlice(f32, &mem_arena, w * h);
        const conf_norm_buf = try allocArenaSlice(f32, &mem_arena, h);
        const step_confidences = try allocArenaSlice(f32, &mem_arena, w);
        const logits_buf = try allocArenaSlice(f32, &mem_arena, w * v);
        const topk_tokens = try allocArenaSlice(u32, &mem_arena, w * cfg.top_k_per_pos);
        const topk_probs = try allocArenaSlice(f32, &mem_arena, w * cfg.top_k_per_pos);
        const node_log_probs = try allocArenaSlice(f32, &mem_arena, max_nodes);
        const l1_draft_buf = try allocArenaSlice(u32, &mem_arena, MAX_L1_DRAFT_LEN);

        @memset(step_confidences, 0.0);
        @memset(node_log_probs, 0.0);

        return .{
            .arena = mem_arena,
            .config = cfg,
            .mtp_concat = mtp_concat,
            .fused_hidden = fused_hidden,
            .corr_buf = corr_buf,
            .tok_embed_tmp = tok_embed_tmp,
            .slot_hiddens = slot_hiddens,
            .norm_buf = norm_buf,
            .q_buf = q_buf,
            .k_buf = k_buf,
            .v_buf = v_buf,
            .attn_out = attn_out,
            .attn_scores = attn_scores,
            .mlp_gate = mlp_gate,
            .mlp_up = mlp_up,
            .mlp_down = mlp_down,
            .conf_norm_buf = conf_norm_buf,
            .step_confidences = step_confidences,
            .logits_buf = logits_buf,
            .topk_tokens = topk_tokens,
            .topk_probs = topk_probs,
            .node_log_probs = node_log_probs,
            .l1_draft_buf = l1_draft_buf,
        };
    }

    pub fn deinit(self: *DFly2Workspace) void {
        self.mtp_concat = &.{};
        self.fused_hidden = &.{};
        self.corr_buf = &.{};
        self.tok_embed_tmp = &.{};
        self.slot_hiddens = &.{};
        self.norm_buf = &.{};
        self.q_buf = &.{};
        self.k_buf = &.{};
        self.v_buf = &.{};
        self.attn_out = &.{};
        self.attn_scores = &.{};
        self.mlp_gate = &.{};
        self.mlp_up = &.{};
        self.mlp_down = &.{};
        self.conf_norm_buf = &.{};
        self.step_confidences = &.{};
        self.logits_buf = &.{};
        self.topk_tokens = &.{};
        self.topk_probs = &.{};
        self.node_log_probs = &.{};
        self.l1_draft_buf = &.{};
        self.arena.deinit();
    }

    pub fn totalWiredBytes(self: *const DFly2Workspace) u64 {
        return self.arena.bytes().len;
    }
};

pub const DFly2ProposeResult = struct {
    tree_nodes: u16,
    max_depth: u16,
    active_steps: u8,
    truncated_tail_steps: u8,
    l1_match_len: u32,
    l1_inserted: u16,
    l1_is_seed: bool,
    expected_tokens: f32,
};

/// D14 `DFly2` block-parallel speculative proposer (`Hy4` `5L`, `block_size = 7`)
/// consuming `dfly_target_kv_layers` + `hidden_correction` + `mtp_fusion`, with
/// `confidence_head` dynamic low-confidence tail truncation.
pub const DFly2Proposer = struct {
    config: DFly2Config,
    weights: DFly2Weights,
    workspace: *DFly2Workspace,
    rounds_proposed: u64 = 0,

    pub fn init(
        config: DFly2Config,
        weights: DFly2Weights,
        workspace: *DFly2Workspace,
    ) Error!DFly2Proposer {
        try config.validate();
        try weights.validate(config);
        if (workspace.config.block_size != config.block_size or
            workspace.config.hidden_size != config.hidden_size or
            workspace.config.intermediate_size != config.intermediate_size or
            workspace.config.num_attention_heads != config.num_attention_heads or
            workspace.config.num_key_value_heads != config.num_key_value_heads or
            workspace.config.head_dim != config.head_dim or
            workspace.config.vocab_size != config.vocab_size or
            workspace.config.num_target_kv_layers != config.num_target_kv_layers or
            workspace.config.max_target_kv_len < config.max_target_kv_len or
            workspace.config.top_k_per_pos != config.top_k_per_pos or
            workspace.config.max_tree_nodes < config.max_tree_nodes)
        {
            return error.WorkspaceMismatch;
        }
        return .{
            .config = config,
            .weights = weights,
            .workspace = workspace,
        };
    }

    pub fn propose(
        self: *DFly2Proposer,
        root_token: u32,
        last_target_hidden: []const f32,
        dfly_target_kv_layers: []const TargetKvLayerView,
        sam: ?*const sam_mod.SuffixAutomaton,
        out_tree: *CandidateTree,
    ) Error!DFly2ProposeResult {
        return self.proposeIntoConstrained(
            CandidateTree.capacity,
            root_token,
            last_target_hidden,
            dfly_target_kv_layers,
            sam,
            null,
            out_tree,
        );
    }

    pub fn proposeInto(
        self: *DFly2Proposer,
        comptime max_nodes: u16,
        root_token: u32,
        last_target_hidden: []const f32,
        dfly_target_kv_layers: []const TargetKvLayerView,
        sam: ?*const sam_mod.SuffixAutomaton,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!DFly2ProposeResult {
        return self.proposeIntoConstrained(
            max_nodes,
            root_token,
            last_target_hidden,
            dfly_target_kv_layers,
            sam,
            null,
            out_tree,
        );
    }

    /// Evaluates `mtp_fusion` + `hidden_correction` + `5L` block-parallel DFly2
    /// layers over `dfly_target_kv_layers`, truncates low-confidence tail steps
    /// where `sigmoid(confidence_head) < threshold`, and populates `DraftTree`.
    pub fn proposeIntoConstrained(
        self: *DFly2Proposer,
        comptime max_nodes: u16,
        root_token: u32,
        last_target_hidden: []const f32,
        dfly_target_kv_layers: []const TargetKvLayerView,
        sam: ?*const sam_mod.SuffixAutomaton,
        grammar_opt: ?accept_mod.GrammarTreeAdapter,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!DFly2ProposeResult {
        const h: usize = self.config.hidden_size;
        const w: u8 = self.config.block_size;
        const v: usize = self.config.vocab_size;
        const kv_dim: usize = self.config.kvDim();
        const k_top: usize = self.config.top_k_per_pos;

        if (root_token >= v) return error.TokenOutOfRange;
        if (last_target_hidden.len != h) return error.DimensionMismatch;
        const uses_mla = self.weights.layers.len > 0 and self.weights.layers[0].mla != null;
        if (!uses_mla) {
            if (dfly_target_kv_layers.len != self.config.num_target_kv_layers) return error.DimensionMismatch;
            for (dfly_target_kv_layers) |kv| {
                if (kv.kv_len == 0 or kv.kv_len > self.config.max_target_kv_len) return error.DimensionMismatch;
                if (kv.k.len < kv.kv_len * kv_dim or kv.v.len < kv.kv_len * kv_dim) return error.BufferTooSmall;
            }
        }

        out_tree.reset(root_token);
        const node_limit: u16 = @min(@as(u16, self.config.max_tree_nodes), max_nodes);
        @memset(self.workspace.node_log_probs[0..node_limit], 0.0);

        const gram_active = if (grammar_opt) |ga| ga.isActive() else false;
        var node_states: [tree_mod.MAX_SUPPORTED_NODES]grammar_mod.GrammarState = undefined;
        var node_valid: [tree_mod.MAX_SUPPORTED_NODES]bool = @splat(false);
        node_valid[0] = true;
        if (grammar_opt) |ga| {
            if (ga.isActive()) node_states[0] = ga.state.snapshot();
        }

        const l1 = mergeOptionalL1Sam(
            max_nodes,
            out_tree,
            sam,
            self.config.l1_max_draft_len,
            node_limit,
            self.config.l1_seed_step_prob,
            self.config.l1_branch_step_prob,
            self.workspace.l1_draft_buf,
            self.workspace.node_log_probs,
            gram_active,
            grammar_opt,
            &node_states,
            &node_valid,
        );

        // 1. `mtp_fusion`: concat(RMSNorm(tok_embed[root]), RMSNorm(last_target_hidden)) -> fused_hidden.
        const root_e = try resolveTokEmbed(self.weights.tok_embed, self.weights.tok_embed_hook, root_token, h, self.workspace.tok_embed_tmp);
        @memcpy(self.workspace.mtp_concat[0..h], root_e);
        @memcpy(self.workspace.mtp_concat[h .. 2 * h], last_target_hidden);
        applyRmsNormInPlace(self.workspace.mtp_concat[0..h], self.weights.mtp_enorm_weight, self.config.rms_eps);
        applyRmsNormInPlace(self.workspace.mtp_concat[h .. 2 * h], self.weights.mtp_hnorm_weight, self.config.rms_eps);
        matVecF32(self.weights.mtp_fusion_proj, self.workspace.mtp_concat[0 .. 2 * h], h, 2 * h, self.workspace.fused_hidden[0..h]);
        if (self.weights.mtp_fusion_gate) |mg| {
            for (0..h) |d| self.workspace.fused_hidden[d] *= sigmoidF32(mg[d]);
        }

        // 2. `hidden_correction`: optional residual correction (linear OR SwiGLU) before 5L DFly2 block layers.
        if (self.weights.hidden_correction_gate != null and self.weights.hidden_correction_up != null and self.weights.hidden_correction_down != null) {
            const hc_inter: usize = self.weights.hidden_correction_inter;
            if (hc_inter > 0 and hc_inter <= self.workspace.mlp_gate.len) {
                @memcpy(self.workspace.corr_buf[0..h], self.workspace.fused_hidden[0..h]);
                applyRmsNormInPlace(self.workspace.corr_buf[0..h], self.weights.hidden_correction_norm, self.config.rms_eps);
                matVecF32(self.weights.hidden_correction_gate.?, self.workspace.corr_buf[0..h], hc_inter, h, self.workspace.mlp_gate[0..hc_inter]);
                matVecF32(self.weights.hidden_correction_up.?, self.workspace.corr_buf[0..h], hc_inter, h, self.workspace.mlp_up[0..hc_inter]);
                for (0..hc_inter) |i| self.workspace.mlp_gate[i] = siluF32(self.workspace.mlp_gate[i]) * self.workspace.mlp_up[i];
                matVecAccumF32(self.weights.hidden_correction_down.?, self.workspace.mlp_gate[0..hc_inter], h, hc_inter, self.workspace.fused_hidden[0..h]);
            }
        } else if (self.weights.hidden_correction_proj) |hc| {
            @memcpy(self.workspace.corr_buf[0..h], self.workspace.fused_hidden[0..h]);
            applyRmsNormInPlace(self.workspace.corr_buf[0..h], self.weights.hidden_correction_norm, self.config.rms_eps);
            matVecAccumF32(hc, self.workspace.corr_buf[0..h], h, h, self.workspace.fused_hidden[0..h]);
        }

        // 3. Seed positions `0..block_size-1` and run `5L` DFly2 transformer layers consuming `dfly_target_kv_layers`.
        const per_pos_mask = self.weights.mask_embed.len == @as(usize, w) * h;
        for (0..w) |pos| {
            const m_slice = if (per_pos_mask)
                self.weights.mask_embed[pos * h ..][0..h]
            else
                self.weights.mask_embed[0..h];
            const dst = self.workspace.slot_hiddens[pos * h ..][0..h];
            for (0..h) |d| {
                dst[d] = self.workspace.fused_hidden[d] + m_slice[d];
            }
        }

        for (self.weights.layers, 0..) |layer, li| {
            const empty_kv: TargetKvLayerView = .{ .k = &.{}, .v = &.{}, .kv_len = 0 };
            const kv_view = if (dfly_target_kv_layers.len > 0)
                dfly_target_kv_layers[li % dfly_target_kv_layers.len]
            else
                empty_kv;
            try runTransformerBlock(
                layer,
                self.workspace.slot_hiddens[0 .. @as(usize, w) * h],
                w,
                h,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                self.config.rms_eps,
                kv_view.k,
                kv_view.v,
                kv_view.kv_len,
                self.workspace.norm_buf,
                self.workspace.q_buf,
                self.workspace.k_buf,
                self.workspace.v_buf,
                self.workspace.attn_out,
                self.workspace.attn_scores,
                self.workspace.mlp_gate,
                self.workspace.mlp_up,
                self.workspace.mlp_down,
                0.0,
                0,
                null,
                null,
            );
        }

        if (self.weights.final_norm) |fnw| {
            for (0..w) |pos| {
                applyRmsNormInPlace(self.workspace.slot_hiddens[pos * h ..][0..h], fnw, self.config.rms_eps);
            }
        }

        // 4. Evaluate `confidence_head` (`sigmoid`) per step and dynamically truncate
        //    low-confidence tail steps `< confidence_threshold` BEFORE scoring `lm_head`.
        var active_steps: u8 = w;
        for (0..w) |pos| {
            const slot_h = self.workspace.slot_hiddens[pos * h ..][0..h];
            @memcpy(self.workspace.conf_norm_buf[0..h], slot_h);
            if (self.weights.confidence_norm) |cn| {
                applyRmsNormInPlace(self.workspace.conf_norm_buf[0..h], cn, self.config.rms_eps);
            }
            var dot: f32 = self.weights.confidence_bias;
            for (0..h) |d| {
                dot += self.weights.confidence_weight[d] * self.workspace.conf_norm_buf[d];
            }
            const conf = sigmoidF32(dot);
            self.workspace.step_confidences[pos] = conf;
            if (pos >= self.config.min_draft_steps and conf < self.config.confidence_threshold) {
                active_steps = @intCast(pos);
                break;
            }
        }
        const truncated_tail_steps: u8 = w - active_steps;

        // 5. Project ONLY the `active_steps` slots through `lm_head` and build `DraftTree`.
        if (active_steps > 0) {
            const act_usize: usize = active_steps;
            const logits = self.workspace.logits_buf[0 .. act_usize * v];
            try evalLmHead(
                self.weights.lm_head,
                self.workspace.slot_hiddens[0 .. act_usize * h],
                active_steps,
                self.config.hidden_size,
                self.config.vocab_size,
                logits,
            );

            const inv_temp: f32 = 1.0 / self.config.temperature;
            for (0..act_usize) |s| {
                const row = logits[s * v ..][0..v];
                const out_tok = self.workspace.topk_tokens[s * k_top ..][0..k_top];
                const out_prob = self.workspace.topk_probs[s * k_top ..][0..k_top];
                try extractDeterministicTopK(row, inv_temp, out_tok, out_prob);
            }

            var frontier_cur: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
            var frontier_next: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
            var cur_len: usize = 1;
            frontier_cur[0] = 0;

            var masked_toks: [MAX_TOP_K_PER_POS]u32 = undefined;
            var masked_probs: [MAX_TOP_K_PER_POS]f32 = undefined;

            for (0..act_usize) |pos_usize| {
                sortFrontierDescending(max_nodes, out_tree, frontier_cur[0..cur_len]);
                const pos_toks = self.workspace.topk_tokens[pos_usize * k_top ..][0..k_top];
                const pos_probs = self.workspace.topk_probs[pos_usize * k_top ..][0..k_top];
                const pos_logits = logits[pos_usize * v ..][0..v];

                var next_len: usize = 0;
                for (frontier_cur[0..cur_len]) |parent| {
                    if (gram_active and !node_valid[parent]) continue;
                    var cand_count: usize = k_top;
                    var cand_toks: []const u32 = pos_toks;
                    var cand_probs: []const f32 = pos_probs;
                    if (gram_active) {
                        var parent_ga = grammar_opt.?;
                        var pstate = node_states[parent].snapshot();
                        parent_ga.state = &pstate;
                        const mask = parent_ga.fillBitmask(v) catch return error.InvalidLogits;
                        cand_count = try extractDeterministicTopKMasked(
                            pos_logits,
                            inv_temp,
                            mask,
                            masked_toks[0..k_top],
                            masked_probs[0..k_top],
                        );
                        cand_toks = masked_toks[0..cand_count];
                        cand_probs = masked_probs[0..cand_count];
                    }
                    for (0..cand_count) |ki| {
                        const tok = cand_toks[ki];
                        const prob = cand_probs[ki];
                        var child_state: grammar_mod.GrammarState = undefined;
                        if (gram_active) {
                            child_state = node_states[parent].snapshot();
                            if (!grammar_opt.?.advanceTokenOnState(&child_state, tok)) continue;
                        }
                        const child = if (out_tree.findChild(parent, tok)) |existing| blk: {
                            break :blk out_tree.insertChild(parent, tok, prob, .l2) orelse existing;
                        } else blk: {
                            if (out_tree.count >= node_limit) break :blk null;
                            break :blk out_tree.insertChild(parent, tok, prob, .l2);
                        };
                        if (child) |c_idx| {
                            self.workspace.node_log_probs[c_idx] = logProbOf(prob);
                            if (gram_active) {
                                node_states[c_idx] = child_state;
                                node_valid[c_idx] = true;
                            }
                            if (next_len < tree_mod.MAX_SUPPORTED_NODES and
                                !containsNode(frontier_next[0..next_len], c_idx))
                            {
                                frontier_next[next_len] = c_idx;
                                next_len += 1;
                            }
                        }
                    }
                }
                if (next_len == 0) break;
                @memcpy(frontier_cur[0..next_len], frontier_next[0..next_len]);
                cur_len = next_len;
            }
        }

        self.rounds_proposed += 1;

        return .{
            .tree_nodes = out_tree.count,
            .max_depth = out_tree.max_depth,
            .active_steps = active_steps,
            .truncated_tail_steps = truncated_tail_steps,
            .l1_match_len = l1.match_len,
            .l1_inserted = l1.inserted,
            .l1_is_seed = l1.is_seed,
            .expected_tokens = out_tree.expectedTokens(),
        };
    }

    pub inline fn stepConfidence(self: *const DFly2Proposer, pos: u8) f32 {
        return self.workspace.step_confidences[pos];
    }

    pub inline fn nodeLogProb(self: *const DFly2Proposer, node_idx: u16) f32 {
        return self.workspace.node_log_probs[node_idx];
    }

    pub fn pruneWithD7(
        self: *DFly2Proposer,
        comptime max_nodes: u16,
        out_tree: *tree_mod.DraftTree(max_nodes),
        cost: tree_mod.PruneCostModel,
        node_experts: ?[]const tree_mod.ExpertBitset,
        cold_experts_by_layer: ?[]const tree_mod.ExpertBitset,
        hard_cap: u16,
    ) Error!tree_mod.DraftTree(max_nodes).PruneSummary {
        return pruneTreeAndRemapLogProbs(max_nodes, out_tree, self.workspace.node_log_probs, cost, node_experts, cold_experts_by_layer, hard_cap);
    }
};

/// Startup-loaded `Hy4` `DFly2` `.amf` bundle supporting both standard GQA and
/// native `Hy4` MLA (`q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`/
/// `linear_gate`/`learnable_sink_param`) + `confidence_head` + `hidden_correction` + `mtp_fusion`.
pub const LoadedDFly2Bundle = struct {
    owned_f32_bufs: [][]f32,
    layers: []TransformerBlockWeights,
    workspace: DFly2Workspace,
    proposer: DFly2Proposer,

    pub const LoadError = LoadedDFlareBundle.LoadError;

    pub fn loadFromAmfFileWithHooks(
        allocator: std.mem.Allocator,
        path: [:0]const u8,
        cfg_in: DFly2Config,
        tok_embed_override: ?TokEmbedHook,
        lm_head_override: ?LmHeadHook,
        budget_override: ?budget_mod.Budget,
    ) LoadError!*LoadedDFly2Bundle {
        var amf_file = try amf.AmfFile.open(allocator, path);
        defer amf_file.close(allocator);
        return loadFromAmfWithHooks(allocator, &amf_file, cfg_in, tok_embed_override, lm_head_override, budget_override);
    }

    pub fn loadFromAmfWithHooks(
        allocator: std.mem.Allocator,
        amf_file: *const amf.AmfFile,
        cfg_in: DFly2Config,
        tok_embed_override: ?TokEmbedHook,
        lm_head_override: ?LmHeadHook,
        budget_override: ?budget_mod.Budget,
    ) LoadError!*LoadedDFly2Bundle {
        var cfg = cfg_in;
        // Infer intermediate_size from layer 0 mlp.gate_proj or shared_experts if present.
        if (findAnyEntry(amf_file, &.{ "layers.0.mlp.gate_proj.weight", "layers.0.mlp.shared_experts.gate_proj.weight" })) |ge| {
            if (ge.shape[0] > 0) cfg.intermediate_size = ge.shape[0];
        }
        try cfg.validate();

        const h: usize = cfg.hidden_size;
        const inter: usize = cfg.intermediate_size;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();
        const v: usize = cfg.vocab_size;

        var f32_pool: [256][]f32 = undefined;
        var f32_len: usize = 0;
        errdefer {
            for (0..f32_len) |i| allocator.free(f32_pool[i]);
        }

        const eh_proj = try loadAmfTensorAsF32256(
            allocator,
            amf_file,
            &.{ "mtp_fusion.eh_proj.weight", "layers.0.eh_proj.weight", "dfly2.eh_proj.weight" },
            h * 2 * h,
            &f32_pool,
            &f32_len,
        );
        const enorm = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{ "mtp_fusion.enorm.weight", "layers.0.enorm.weight" }, h, &f32_pool, &f32_len);
        const hnorm = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{ "mtp_fusion.hnorm.weight", "layers.0.hnorm.weight" }, h, &f32_pool, &f32_len);
        const mtp_gate = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{"mtp_fusion.gate"}, h, &f32_pool, &f32_len);

        // Optional `hidden_correction` (SwiGLU or linear).
        const hc_norm = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{ "hidden_correction.hidden_norm.weight", "hidden_correction.norm.weight" }, h, &f32_pool, &f32_len);
        const hc_enorm = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{"hidden_correction.embed_norm.weight"}, h, &f32_pool, &f32_len);
        var hc_proj: ?[]const f32 = null;
        var hc_gate: ?[]const f32 = null;
        var hc_up: ?[]const f32 = null;
        var hc_down: ?[]const f32 = null;
        var hc_inter: u32 = 0;
        if (amf_file.parsed.find("hidden_correction.gate_proj.weight")) |hge| {
            hc_inter = hge.shape[0];
            hc_gate = try loadEntryIntoF32Pool(allocator, amf_file, hge, &f32_pool, &f32_len);
            hc_up = try loadAmfTensorAsF32256(allocator, amf_file, &.{"hidden_correction.up_proj.weight"}, @as(usize, hc_inter) * h, &f32_pool, &f32_len);
            hc_down = try loadAmfTensorAsF32256(allocator, amf_file, &.{"hidden_correction.down_proj.weight"}, h * @as(usize, hc_inter), &f32_pool, &f32_len);
        } else {
            hc_proj = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{"hidden_correction.proj.weight"}, h * h, &f32_pool, &f32_len);
        }

        const mask_embed: []const f32 = if (findAnyEntry(amf_file, &.{ "dfly2.mask_embed", "mask_embed", "mask_embed.weight" })) |me|
            try loadEntryIntoF32Pool(allocator, amf_file, me, &f32_pool, &f32_len)
        else blk: {
            const z = try allocator.alloc(f32, h);
            @memset(z, 0.0);
            if (f32_len >= f32_pool.len) {
                allocator.free(z);
                return error.Overflow;
            }
            f32_pool[f32_len] = z;
            f32_len += 1;
            break :blk z;
        };

        var tok_embed: []const f32 = &.{};
        if (tok_embed_override == null) {
            tok_embed = try loadAmfTensorAsF32256(allocator, amf_file, &.{ "dfly2.tok_embed.weight", "embed_tokens.weight" }, v * h, &f32_pool, &f32_len);
        }

        const final_norm = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{ "final_norm.weight", "layers.0.final_layernorm.weight", "norm.weight" }, h, &f32_pool, &f32_len);

        // `confidence_head`: `confidence_head.proj.weight [1, H]` + `bias [1]`.
        var conf_w: []const f32 = undefined;
        var conf_bias: f32 = 0.0;
        if (findAnyEntry(amf_file, &.{ "confidence_head.proj.weight", "confidence_head.weight" })) |cw| {
            conf_w = try loadEntryIntoF32Pool(allocator, amf_file, cw, &f32_pool, &f32_len);
            if (findAnyEntry(amf_file, &.{ "confidence_head.proj.bias", "confidence_head.bias" })) |cb| {
                const bv = try readEntryAsF32(allocator, amf_file, cb);
                defer allocator.free(bv);
                if (bv.len > 0) conf_bias = bv[0];
            }
        } else {
            const cw_zero = try allocator.alloc(f32, h);
            @memset(cw_zero, 0.0);
            if (f32_len >= f32_pool.len) {
                allocator.free(cw_zero);
                return error.Overflow;
            }
            f32_pool[f32_len] = cw_zero;
            f32_len += 1;
            conf_w = cw_zero;
            conf_bias = 4.0;
        }
        const conf_norm = try loadAmfTensorAsF32Opt256(allocator, amf_file, &.{"confidence_head.norm.weight"}, h, &f32_pool, &f32_len);

        const layers = try allocator.alloc(TransformerBlockWeights, cfg.num_layers);
        errdefer allocator.free(layers);

        var nb0: [80]u8 = undefined;
        var nb1: [80]u8 = undefined;
        var nb2: [80]u8 = undefined;
        for (0..cfg.num_layers) |l| {
            const in_norm = try loadLayerF32Opt(allocator, amf_file, l, &.{ "input_layernorm.weight", "attn_norm.weight" }, h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
            const post_norm = try loadLayerF32Opt(allocator, amf_file, l, &.{ "post_attention_layernorm.weight", "post_norm.weight" }, h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
            const gate_p = try loadLayerF32(allocator, amf_file, l, &.{ "mlp.gate_proj.weight", "mlp.shared_experts.gate_proj.weight" }, inter * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
            const up_p = try loadLayerF32(allocator, amf_file, l, &.{ "mlp.up_proj.weight", "mlp.shared_experts.up_proj.weight" }, inter * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
            const down_p = try loadLayerF32(allocator, amf_file, l, &.{ "mlp.down_proj.weight", "mlp.shared_experts.down_proj.weight" }, h * inter, &nb0, &nb1, &nb2, &f32_pool, &f32_len);

            const qa_nm = std.fmt.bufPrint(&nb0, "layers.{d}.self_attn.q_a_proj.weight", .{l}) catch return error.Overflow;
            if (amf_file.parsed.find(qa_nm)) |qae| {
                const q_rank: u32 = qae.shape[0];
                const q_a = try loadEntryIntoF32Pool(allocator, amf_file, qae, &f32_pool, &f32_len);
                const q_a_norm = try loadLayerF32Opt(allocator, amf_file, l, &.{"self_attn.q_a_layernorm.weight"}, q_rank, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                const qbe = amf_file.parsed.find(std.fmt.bufPrint(&nb0, "layers.{d}.self_attn.q_b_proj.weight", .{l}) catch return error.Overflow) orelse return error.MissingTensor;
                const q_b = try loadEntryIntoF32Pool(allocator, amf_file, qbe, &f32_pool, &f32_len);
                const kvae = findAnyEntry(amf_file, &.{
                    std.fmt.bufPrint(&nb0, "layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{l}) catch return error.Overflow,
                    std.fmt.bufPrint(&nb1, "layers.{d}.self_attn.kv_a_mqa.weight", .{l}) catch return error.Overflow,
                }) orelse return error.MissingTensor;
                const kv_a = try loadEntryIntoF32Pool(allocator, amf_file, kvae, &f32_pool, &f32_len);
                const kvbe = amf_file.parsed.find(std.fmt.bufPrint(&nb0, "layers.{d}.self_attn.kv_b_proj.weight", .{l}) catch return error.Overflow) orelse return error.MissingTensor;
                const kv_b = try loadEntryIntoF32Pool(allocator, amf_file, kvbe, &f32_pool, &f32_len);
                const kv_rank: u32 = kvbe.shape[1];
                const qk_rope: u32 = if (kvae.shape[0] > kv_rank) kvae.shape[0] - kv_rank else 0;
                const qk_total: u32 = qbe.shape[0] / cfg.num_attention_heads;
                const qk_nope: u32 = if (qk_total > qk_rope) qk_total - qk_rope else qk_total;
                const kvb_per_h: u32 = kvbe.shape[0] / cfg.num_attention_heads;
                const v_dim: u32 = if (kvb_per_h > qk_nope) kvb_per_h - qk_nope else kvb_per_h;
                const kv_a_norm = try loadLayerF32Opt(allocator, amf_file, l, &.{"self_attn.kv_a_layernorm.weight"}, kv_rank, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                const o_proj = try loadLayerF32(allocator, amf_file, l, &.{"self_attn.o_proj.weight"}, h * cfg.num_attention_heads * v_dim, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                const lin_gate = try loadLayerF32Opt(allocator, amf_file, l, &.{"self_attn.linear_gate.weight"}, cfg.num_attention_heads * v_dim * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                const sink_f32 = try loadLayerF32Opt(allocator, amf_file, l, &.{"self_attn.learnable_sink_param"}, cfg.num_attention_heads, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                const ppn = try loadLayerF32Opt(allocator, amf_file, l, &.{"self_attn.post_proj_norm.weight"}, h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);

                layers[l] = .{
                    .in_norm_weight = in_norm,
                    .post_norm_weight = post_norm,
                    .mlp_gate_proj = gate_p,
                    .mlp_up_proj = up_p,
                    .mlp_down_proj = down_p,
                    .mla = .{
                        .q_rank = q_rank,
                        .kv_rank = kv_rank,
                        .qk_nope = qk_nope,
                        .qk_rope = qk_rope,
                        .v_dim = v_dim,
                        .q_a_proj = q_a,
                        .q_a_norm = q_a_norm,
                        .q_b_proj = q_b,
                        .kv_a_proj = kv_a,
                        .kv_a_norm = kv_a_norm,
                        .kv_b_proj = kv_b,
                        .o_proj = o_proj,
                        .linear_gate = lin_gate,
                        .sink_f32 = sink_f32,
                        .post_proj_norm = ppn,
                    },
                };
            } else {
                const q_p = try loadLayerF32(allocator, amf_file, l, &.{ "self_attn.q_proj.weight", "attn.q_proj.weight" }, q_dim * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                const k_p = try loadLayerF32(allocator, amf_file, l, &.{ "self_attn.k_proj.weight", "attn.k_proj.weight" }, kv_dim * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                const v_p = try loadLayerF32(allocator, amf_file, l, &.{ "self_attn.v_proj.weight", "attn.v_proj.weight" }, kv_dim * h, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                const o_p = try loadLayerF32(allocator, amf_file, l, &.{ "self_attn.o_proj.weight", "attn.o_proj.weight" }, h * q_dim, &nb0, &nb1, &nb2, &f32_pool, &f32_len);
                layers[l] = .{
                    .in_norm_weight = in_norm,
                    .q_proj = q_p,
                    .k_proj = k_p,
                    .v_proj = v_p,
                    .o_proj = o_p,
                    .post_norm_weight = post_norm,
                    .mlp_gate_proj = gate_p,
                    .mlp_up_proj = up_p,
                    .mlp_down_proj = down_p,
                };
            }
        }

        const lm_head: LmHeadHook = lm_head_override orelse blk: {
            const lm_w = try loadAmfTensorAsF32256(allocator, amf_file, &.{ "dfly2.lm_head.weight", "lm_head.weight" }, v * h, &f32_pool, &f32_len);
            break :blk .{ .dense_f32 = lm_w };
        };

        const owned_f32 = try allocator.alloc([]f32, f32_len);
        errdefer allocator.free(owned_f32);
        @memcpy(owned_f32, f32_pool[0..f32_len]);

        var ws = try DFly2Workspace.init(cfg, budget_override);
        errdefer ws.deinit();

        const weights: DFly2Weights = .{
            .mtp_fusion_proj = eh_proj,
            .mtp_enorm_weight = enorm,
            .mtp_hnorm_weight = hnorm,
            .mtp_fusion_gate = mtp_gate,
            .hidden_correction_proj = hc_proj,
            .hidden_correction_norm = hc_norm,
            .hidden_correction_embed_norm = hc_enorm,
            .hidden_correction_gate = hc_gate,
            .hidden_correction_up = hc_up,
            .hidden_correction_down = hc_down,
            .hidden_correction_inter = hc_inter,
            .mask_embed = mask_embed,
            .tok_embed = tok_embed,
            .tok_embed_hook = tok_embed_override,
            .layers = layers,
            .final_norm = final_norm,
            .confidence_weight = conf_w,
            .confidence_bias = conf_bias,
            .confidence_norm = conf_norm,
            .lm_head = lm_head,
        };

        const bundle = try allocator.create(LoadedDFly2Bundle);
        errdefer allocator.destroy(bundle);
        bundle.* = .{
            .owned_f32_bufs = owned_f32,
            .layers = layers,
            .workspace = ws,
            .proposer = undefined,
        };
        bundle.proposer = try DFly2Proposer.init(cfg, weights, &bundle.workspace);
        return bundle;
    }

    pub fn deinit(self: *LoadedDFly2Bundle, allocator: std.mem.Allocator) void {
        self.workspace.deinit();
        allocator.free(self.layers);
        for (self.owned_f32_bufs) |b| allocator.free(b);
        allocator.free(self.owned_f32_bufs);
        allocator.destroy(self);
    }
};

// ============================================================================
// Shared Transformer, Top-K, Tree, D7 Pruning, and AMF Helpers
// ============================================================================

fn pruneTreeAndRemapLogProbs(
    comptime max_nodes: u16,
    out_tree: *tree_mod.DraftTree(max_nodes),
    node_log_probs: []f32,
    cost: tree_mod.PruneCostModel,
    node_experts: ?[]const tree_mod.ExpertBitset,
    cold_experts_by_layer: ?[]const tree_mod.ExpertBitset,
    hard_cap: u16,
) Error!tree_mod.DraftTree(max_nodes).PruneSummary {
    const summary = try out_tree.pruneToOptimalSubtree(cost, node_experts, cold_experts_by_layer, hard_cap);
    var remapped: [tree_mod.MAX_SUPPORTED_NODES]f32 = @splat(0.0);
    for (0..summary.kept_nodes) |new_i| {
        const old_i = summary.remapped_old_indices[new_i];
        if (old_i != tree_mod.NO_NODE) {
            remapped[new_i] = node_log_probs[old_i];
        }
    }
    @memcpy(node_log_probs[0..summary.kept_nodes], remapped[0..summary.kept_nodes]);
    return summary;
}

const L1MergeStats = struct {
    match_len: u32 = 0,
    inserted: u16 = 0,
    is_seed: bool = false,
};

fn mergeOptionalL1Sam(
    comptime max_nodes: u16,
    out_tree: *tree_mod.DraftTree(max_nodes),
    sam: ?*const sam_mod.SuffixAutomaton,
    l1_max_draft_len: u8,
    node_limit: u16,
    l1_seed_step_prob: f32,
    l1_branch_step_prob: f32,
    l1_draft_buf: []u32,
    node_log_probs: []f32,
    gram_active: bool,
    grammar_opt: ?accept_mod.GrammarTreeAdapter,
    node_states: *[tree_mod.MAX_SUPPORTED_NODES]grammar_mod.GrammarState,
    node_valid: *[tree_mod.MAX_SUPPORTED_NODES]bool,
) L1MergeStats {
    const automaton = sam orelse return .{};
    const max_l1: usize = @min(@as(usize, l1_max_draft_len), @as(usize, node_limit - 1));
    if (max_l1 == 0) return .{};

    const q = automaton.queryActive(max_l1, l1_draft_buf[0..max_l1]);
    var valid_len: usize = q.draft_len;
    if (gram_active) {
        const ga = grammar_opt.?;
        var st_walk = node_states[0].snapshot();
        valid_len = 0;
        for (l1_draft_buf[0..q.draft_len]) |tid| {
            if (!ga.advanceTokenOnState(&st_walk, tid)) break;
            valid_len += 1;
        }
    }
    const match_len: u32 = if (valid_len == q.draft_len) q.match_len else @intCast(@min(q.match_len, valid_len));
    if (valid_len == 0) return .{ .match_len = match_len };

    const ins = out_tree.insertL1Continuation(
        @intCast(@max(@as(u32, 1), match_len)),
        l1_draft_buf[0..valid_len],
        l1_seed_step_prob,
        l1_branch_step_prob,
    );
    for (1..out_tree.count) |ni| {
        node_log_probs[ni] = logProbOf(out_tree.nodes[ni].step_prob);
        if (gram_active) {
            const nd = out_tree.nodes[ni];
            if (node_valid[nd.parent]) {
                var cs = node_states[nd.parent].snapshot();
                if (grammar_opt.?.advanceTokenOnState(&cs, nd.token_id)) {
                    node_states[ni] = cs;
                    node_valid[ni] = true;
                }
            }
        }
    }
    return .{
        .match_len = match_len,
        .inserted = ins.inserted_count,
        .is_seed = ins.is_seed,
    };
}

fn hasLmHeadHook(hook: LmHeadHook) bool {
    return switch (hook) {
        .dense_f32 => |w| w.len > 0,
        .dense_f16 => |w| w.len > 0,
        .dense_bf16 => |w| w.len > 0,
        .callback => true,
    };
}

fn applyRopeNeoXInPlace(buf: []f32, num_heads: u32, head_dim: u32, pos: u32, theta: f32) void {
    const hd: usize = head_dim;
    const half: usize = hd / 2;
    const pos_f: f32 = @floatFromInt(pos);
    const inv_hd: f32 = 2.0 / @as(f32, @floatFromInt(hd));
    for (0..num_heads) |hi| {
        const head = buf[hi * hd ..][0..hd];
        for (0..half) |i| {
            const freq: f32 = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(i)) * inv_hd);
            const angle = pos_f * freq;
            const c = @cos(angle);
            const s = @sin(angle);
            const x0 = head[i];
            const x1 = head[i + half];
            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
}

fn runTransformerBlock(
    layer: TransformerBlockWeights,
    hiddens: []f32,
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rms_eps: f32,
    ctx_k: []const f32,
    ctx_v: []const f32,
    ctx_len: usize,
    norm_buf: []f32,
    q_buf: []f32,
    k_buf: []f32,
    v_buf: []f32,
    attn_out: []f32,
    attn_scores: []f32,
    mlp_gate: []f32,
    mlp_up: []f32,
    mlp_down: []f32,
    rope_theta: f32,
    anchor_pos: u32,
    ctx_k_f16: ?[]const f16,
    ctx_v_f16: ?[]const f16,
) Error!void {
    if (layer.mla) |mla| {
        try runMlaTransformerBlock(
            layer,
            mla,
            hiddens,
            seq_len,
            hidden_size,
            intermediate_size,
            num_heads,
            rms_eps,
            ctx_k,
            ctx_len,
            norm_buf,
            attn_out,
            attn_scores,
            mlp_gate,
            mlp_up,
            mlp_down,
        );
        return;
    }

    const q_dim: usize = @as(usize, num_heads) * @as(usize, head_dim);
    const kv_dim: usize = @as(usize, num_kv_heads) * @as(usize, head_dim);
    const hd: usize = head_dim;
    const gqa_group: usize = num_heads / num_kv_heads;
    const inv_sqrt_hd: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const total_kv = ctx_len + seq_len;
    if (attn_scores.len < total_kv) return error.BufferTooSmall;

    for (0..seq_len) |pos| {
        const src = hiddens[pos * hidden_size ..][0..hidden_size];
        const dst = norm_buf[pos * hidden_size ..][0..hidden_size];
        @memcpy(dst, src);
        applyRmsNormInPlace(dst, layer.in_norm_weight, rms_eps);

        if (layer.q_proj_bf16) |qb| {
            matVecBf16(qb, dst, q_dim, hidden_size, q_buf[pos * q_dim ..][0..q_dim]);
        } else {
            matVecF32(layer.q_proj, dst, q_dim, hidden_size, q_buf[pos * q_dim ..][0..q_dim]);
        }
        if (layer.k_proj_bf16) |kb| {
            matVecBf16(kb, dst, kv_dim, hidden_size, k_buf[pos * kv_dim ..][0..kv_dim]);
        } else {
            matVecF32(layer.k_proj, dst, kv_dim, hidden_size, k_buf[pos * kv_dim ..][0..kv_dim]);
        }
        if (layer.v_proj_bf16) |vb| {
            matVecBf16(vb, dst, kv_dim, hidden_size, v_buf[pos * kv_dim ..][0..kv_dim]);
        } else {
            matVecF32(layer.v_proj, dst, kv_dim, hidden_size, v_buf[pos * kv_dim ..][0..kv_dim]);
        }

        if (layer.q_norm_weight) |qn| {
            applyHeadRmsNorm(q_buf[pos * q_dim ..][0..q_dim], num_heads, head_dim, qn, rms_eps);
        }
        if (layer.k_norm_weight) |kn| {
            applyHeadRmsNorm(k_buf[pos * kv_dim ..][0..kv_dim], num_kv_heads, head_dim, kn, rms_eps);
        }
        if (rope_theta > 0.0) {
            const tok_pos: u32 = anchor_pos + @as(u32, @intCast(pos));
            applyRopeNeoXInPlace(q_buf[pos * q_dim ..][0..q_dim], num_heads, head_dim, tok_pos, rope_theta);
            applyRopeNeoXInPlace(k_buf[pos * kv_dim ..][0..kv_dim], num_kv_heads, head_dim, tok_pos, rope_theta);
        }
    }

    // Bidirectional block-parallel attention over `[ctx_kv (0..ctx_len), block_kv (0..seq_len)]`.
    for (0..seq_len) |q_pos| {
        const q_row = q_buf[q_pos * q_dim ..][0..q_dim];
        const out_row = attn_out[q_pos * q_dim ..][0..q_dim];
        for (0..num_heads) |qh| {
            const kvh = qh / gqa_group;
            const q_head = q_row[qh * hd ..][0..hd];
            var max_s: f32 = -std.math.inf(f32);

            if (ctx_k_f16) |ck16| {
                for (0..ctx_len) |kp| {
                    const k_head = ck16[kp * kv_dim + kvh * hd ..][0..hd];
                    var dot: f32 = 0.0;
                    for (0..hd) |d| dot += q_head[d] * @as(f32, @floatCast(k_head[d]));
                    const sc = dot * inv_sqrt_hd;
                    attn_scores[kp] = sc;
                    if (sc > max_s) max_s = sc;
                }
            } else {
                for (0..ctx_len) |kp| {
                    const k_head = ctx_k[kp * kv_dim + kvh * hd ..][0..hd];
                    var dot: f32 = 0.0;
                    for (0..hd) |d| dot += q_head[d] * k_head[d];
                    const sc = dot * inv_sqrt_hd;
                    attn_scores[kp] = sc;
                    if (sc > max_s) max_s = sc;
                }
            }
            for (0..seq_len) |kp| {
                const k_head = k_buf[kp * kv_dim + kvh * hd ..][0..hd];
                var dot: f32 = 0.0;
                for (0..hd) |d| dot += q_head[d] * k_head[d];
                const sc = dot * inv_sqrt_hd;
                attn_scores[ctx_len + kp] = sc;
                if (sc > max_s) max_s = sc;
            }

            var sum_exp: f32 = 0.0;
            for (0..total_kv) |ti| {
                const e = @exp(attn_scores[ti] - max_s);
                attn_scores[ti] = e;
                sum_exp += e;
            }
            const inv_sum: f32 = if (sum_exp > 0.0) 1.0 / sum_exp else 0.0;

            const dst_head = out_row[qh * hd ..][0..hd];
            @memset(dst_head, 0.0);
            if (ctx_v_f16) |cv16| {
                for (0..ctx_len) |kp| {
                    const p = attn_scores[kp] * inv_sum;
                    const v_head = cv16[kp * kv_dim + kvh * hd ..][0..hd];
                    for (0..hd) |d| dst_head[d] += p * @as(f32, @floatCast(v_head[d]));
                }
            } else {
                for (0..ctx_len) |kp| {
                    const p = attn_scores[kp] * inv_sum;
                    const v_head = ctx_v[kp * kv_dim + kvh * hd ..][0..hd];
                    for (0..hd) |d| dst_head[d] += p * v_head[d];
                }
            }
            for (0..seq_len) |kp| {
                const p = attn_scores[ctx_len + kp] * inv_sum;
                const v_head = v_buf[kp * kv_dim + kvh * hd ..][0..hd];
                for (0..hd) |d| dst_head[d] += p * v_head[d];
            }
        }

        // Residual connection after attention projection.
        const h_row = hiddens[q_pos * hidden_size ..][0..hidden_size];
        if (layer.o_proj_bf16) |ob| {
            matVecBf16Accum(ob, out_row, hidden_size, q_dim, h_row);
        } else {
            matVecAccumF32(layer.o_proj, out_row, hidden_size, q_dim, h_row);
        }
    }

    // SwiGLU MLP + residual add across all block slots.
    for (0..seq_len) |pos| {
        const h_row = hiddens[pos * hidden_size ..][0..hidden_size];
        const norm_row = norm_buf[pos * hidden_size ..][0..hidden_size];
        @memcpy(norm_row, h_row);
        applyRmsNormInPlace(norm_row, layer.post_norm_weight, rms_eps);

        const g_row = mlp_gate[pos * intermediate_size ..][0..intermediate_size];
        const u_row = mlp_up[pos * intermediate_size ..][0..intermediate_size];
        if (layer.mlp_gate_bf16) |gb| {
            matVecBf16(gb, norm_row, intermediate_size, hidden_size, g_row);
        } else {
            matVecF32(layer.mlp_gate_proj, norm_row, intermediate_size, hidden_size, g_row);
        }
        if (layer.mlp_up_bf16) |ub| {
            matVecBf16(ub, norm_row, intermediate_size, hidden_size, u_row);
        } else {
            matVecF32(layer.mlp_up_proj, norm_row, intermediate_size, hidden_size, u_row);
        }
        for (0..intermediate_size) |i| {
            g_row[i] = siluF32(g_row[i]) * u_row[i];
        }
        const d_row = mlp_down[pos * hidden_size ..][0..hidden_size];
        if (layer.mlp_down_bf16) |db| {
            matVecBf16(db, g_row, hidden_size, intermediate_size, d_row);
        } else {
            matVecF32(layer.mlp_down_proj, g_row, hidden_size, intermediate_size, d_row);
        }
        for (0..hidden_size) |d| h_row[d] += d_row[d];
    }
}

fn runMlaTransformerBlock(
    layer: TransformerBlockWeights,
    mla: MlaBlockWeights,
    hiddens: []f32,
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: u32,
    rms_eps: f32,
    target_kv_latent: []const f32,
    target_kv_len: usize,
    norm_buf: []f32,
    attn_out: []f32,
    attn_scores: []f32,
    mlp_gate: []f32,
    mlp_up: []f32,
    mlp_down: []f32,
) Error!void {
    const n_q: usize = num_heads;
    const q_rank: usize = mla.q_rank;
    const kv_rank: usize = mla.kv_rank;
    const qk_nope: usize = mla.qk_nope;
    const qk_rope: usize = mla.qk_rope;
    const v_dim: usize = mla.v_dim;
    const qk_dim: usize = qk_nope + qk_rope;
    const inv_sqrt_qk: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(@max(qk_dim, 1))));

    // Stack scratch (small for DFly2 block_size <= 16).
    const max_bs: usize = MAX_BLOCK_SIZE;
    var q_lat: [1024]f32 = undefined;
    var q_all: [max_bs * 2048]f32 = undefined;
    var kv_a_out: [1024]f32 = undefined;
    var k_nope_block: [max_bs * 2048]f32 = undefined;
    var k_rope_block: [max_bs * 256]f32 = undefined;
    var v_block: [max_bs * 2048]f32 = undefined;
    var kv_b_tmp: [4096]f32 = undefined;
    var lin_gate_tmp: [2048]f32 = undefined;
    if (q_rank > q_lat.len or seq_len * n_q * qk_dim > q_all.len or kv_rank + qk_rope > kv_a_out.len or
        seq_len * n_q * qk_nope > k_nope_block.len or seq_len * @max(qk_rope, 1) > k_rope_block.len or
        seq_len * n_q * v_dim > v_block.len or n_q * (qk_nope + v_dim) > kv_b_tmp.len or n_q * v_dim > lin_gate_tmp.len)
    {
        return error.BufferTooSmall;
    }

    for (0..seq_len) |pos| {
        const src = hiddens[pos * hidden_size ..][0..hidden_size];
        const dst = norm_buf[pos * hidden_size ..][0..hidden_size];
        @memcpy(dst, src);
        applyRmsNormInPlace(dst, layer.in_norm_weight, rms_eps);

        matVecF32(mla.q_a_proj, dst, q_rank, hidden_size, q_lat[0..q_rank]);
        applyRmsNormInPlace(q_lat[0..q_rank], mla.q_a_norm, rms_eps);
        matVecF32(mla.q_b_proj, q_lat[0..q_rank], n_q * qk_dim, q_rank, q_all[pos * n_q * qk_dim ..][0 .. n_q * qk_dim]);

        matVecF32(mla.kv_a_proj, dst, kv_rank + qk_rope, hidden_size, kv_a_out[0 .. kv_rank + qk_rope]);
        applyRmsNormInPlace(kv_a_out[0..kv_rank], mla.kv_a_norm, rms_eps);
        if (qk_rope > 0) {
            @memcpy(k_rope_block[pos * qk_rope ..][0..qk_rope], kv_a_out[kv_rank .. kv_rank + qk_rope]);
        }
        matVecF32(mla.kv_b_proj, kv_a_out[0..kv_rank], n_q * (qk_nope + v_dim), kv_rank, kv_b_tmp[0 .. n_q * (qk_nope + v_dim)]);
        for (0..n_q) |hi| {
            const src_h = kv_b_tmp[hi * (qk_nope + v_dim) ..];
            @memcpy(k_nope_block[(pos * n_q + hi) * qk_nope ..][0..qk_nope], src_h[0..qk_nope]);
            @memcpy(v_block[(pos * n_q + hi) * v_dim ..][0..v_dim], src_h[qk_nope .. qk_nope + v_dim]);
        }
    }

    // Limit target KV cross-attention window cleanly to workspace `attn_scores` capacity.
    const use_target_kv: usize = if (target_kv_latent.len >= target_kv_len * kv_rank and target_kv_len > 0)
        @min(target_kv_len, attn_scores.len -| seq_len)
    else
        0;
    const total_kv = use_target_kv + seq_len;

    for (0..seq_len) |q_pos| {
        const q_row = q_all[q_pos * n_q * qk_dim ..][0 .. n_q * qk_dim];
        const out_row = attn_out[q_pos * n_q * v_dim ..][0 .. n_q * v_dim];
        for (0..n_q) |hi| {
            const qh = q_row[hi * qk_dim ..][0..qk_dim];
            var max_s: f32 = if (mla.sink_f32) |sf| sf[hi] else -std.math.inf(f32);

            for (0..use_target_kv) |kp| {
                const c_kv = target_kv_latent[kp * kv_rank ..][0..kv_rank];
                const head_kvb = mla.kv_b_proj[hi * (qk_nope + v_dim) * kv_rank ..];
                var dot: f32 = 0.0;
                for (0..qk_nope) |d| {
                    var k_val: f32 = 0.0;
                    const row_w = head_kvb[d * kv_rank ..][0..kv_rank];
                    for (0..kv_rank) |r| k_val += row_w[r] * c_kv[r];
                    dot += qh[d] * k_val;
                }
                const sc = dot * inv_sqrt_qk;
                attn_scores[kp] = sc;
                if (sc > max_s) max_s = sc;
            }
            for (0..seq_len) |kp| {
                const kn = k_nope_block[(kp * n_q + hi) * qk_nope ..][0..qk_nope];
                var dot: f32 = 0.0;
                for (0..qk_nope) |d| dot += qh[d] * kn[d];
                if (qk_rope > 0) {
                    const kr = k_rope_block[kp * qk_rope ..][0..qk_rope];
                    for (0..qk_rope) |d| dot += qh[qk_nope + d] * kr[d];
                }
                const sc = dot * inv_sqrt_qk;
                attn_scores[use_target_kv + kp] = sc;
                if (sc > max_s) max_s = sc;
            }

            var sum_exp: f32 = if (mla.sink_f32) |sf| @exp(sf[hi] - max_s) else 0.0;
            for (0..total_kv) |ti| {
                const e = @exp(attn_scores[ti] - max_s);
                attn_scores[ti] = e;
                sum_exp += e;
            }
            const inv_sum: f32 = if (sum_exp > 0.0) 1.0 / sum_exp else 0.0;

            const dst_h = out_row[hi * v_dim ..][0..v_dim];
            @memset(dst_h, 0.0);
            for (0..use_target_kv) |kp| {
                const p = attn_scores[kp] * inv_sum;
                const c_kv = target_kv_latent[kp * kv_rank ..][0..kv_rank];
                const head_kvb = mla.kv_b_proj[hi * (qk_nope + v_dim) * kv_rank ..];
                for (0..v_dim) |d| {
                    var v_val: f32 = 0.0;
                    const row_w = head_kvb[(qk_nope + d) * kv_rank ..][0..kv_rank];
                    for (0..kv_rank) |r| v_val += row_w[r] * c_kv[r];
                    dst_h[d] += p * v_val;
                }
            }
            for (0..seq_len) |kp| {
                const p = attn_scores[use_target_kv + kp] * inv_sum;
                const vh = v_block[(kp * n_q + hi) * v_dim ..][0..v_dim];
                for (0..v_dim) |d| dst_h[d] += p * vh[d];
            }
        }
        if (mla.linear_gate) |lg| {
            matVecF32(lg, norm_buf[q_pos * hidden_size ..][0..hidden_size], n_q * v_dim, hidden_size, lin_gate_tmp[0 .. n_q * v_dim]);
            for (0..n_q * v_dim) |i| out_row[i] *= sigmoidF32(lin_gate_tmp[i]);
        }
        const o_tmp = mlp_down[q_pos * hidden_size ..][0..hidden_size];
        matVecF32(mla.o_proj, out_row, hidden_size, n_q * v_dim, o_tmp);
        if (mla.post_proj_norm) |ppn| applyRmsNormInPlace(o_tmp, ppn, rms_eps);
        const h_row = hiddens[q_pos * hidden_size ..][0..hidden_size];
        for (0..hidden_size) |d| h_row[d] += o_tmp[d];
    }

    // SwiGLU MLP + residual add across all block slots.
    for (0..seq_len) |pos| {
        const h_row = hiddens[pos * hidden_size ..][0..hidden_size];
        const norm_row = norm_buf[pos * hidden_size ..][0..hidden_size];
        @memcpy(norm_row, h_row);
        applyRmsNormInPlace(norm_row, layer.post_norm_weight, rms_eps);

        const g_row = mlp_gate[pos * intermediate_size ..][0..intermediate_size];
        const u_row = mlp_up[pos * intermediate_size ..][0..intermediate_size];
        matVecF32(layer.mlp_gate_proj, norm_row, intermediate_size, hidden_size, g_row);
        matVecF32(layer.mlp_up_proj, norm_row, intermediate_size, hidden_size, u_row);
        for (0..intermediate_size) |i| {
            g_row[i] = siluF32(g_row[i]) * u_row[i];
        }
        const d_row = mlp_down[pos * hidden_size ..][0..hidden_size];
        matVecF32(layer.mlp_down_proj, g_row, hidden_size, intermediate_size, d_row);
        for (0..hidden_size) |d| h_row[d] += d_row[d];
    }
}

fn evalLmHead(
    hook: LmHeadHook,
    hiddens: []const f32,
    batch_rows: u8,
    hidden_size: u32,
    vocab_size: u32,
    out_logits: []f32,
) Error!void {
    const h: usize = hidden_size;
    const v: usize = vocab_size;
    switch (hook) {
        .dense_f32 => |mat| {
            for (0..batch_rows) |s| {
                matVecF32(mat, hiddens[s * h ..][0..h], v, h, out_logits[s * v ..][0..v]);
            }
        },
        .dense_f16 => |mat| {
            for (0..batch_rows) |s| {
                const row_h = hiddens[s * h ..][0..h];
                const dst = out_logits[s * v ..][0..v];
                for (0..v) |r| {
                    const m_row = mat[r * h ..][0..h];
                    var acc: f32 = 0.0;
                    for (0..h) |c| acc += @as(f32, @floatCast(m_row[c])) * row_h[c];
                    dst[r] = acc;
                }
            }
        },
        .dense_bf16 => |mat| {
            for (0..batch_rows) |s| {
                const row_h = hiddens[s * h ..][0..h];
                const dst = out_logits[s * v ..][0..v];
                for (0..v) |r| {
                    const m_row = mat[r * h ..][0..h];
                    var acc: f32 = 0.0;
                    for (0..h) |c| acc += quant.bf16ToF32(m_row[c]) * row_h[c];
                    dst[r] = acc;
                }
            }
        },
        .callback => |cb| {
            try cb.eval_fn(cb.ctx, hiddens, batch_rows, hidden_size, vocab_size, out_logits);
        },
    }
}

fn validateLmHeadHook(hook: LmHeadHook, expected_elems: usize) Error!void {
    switch (hook) {
        .dense_f32 => |m| if (m.len != expected_elems) return error.InvalidWeights,
        .dense_f16 => |m| if (m.len != expected_elems) return error.InvalidWeights,
        .dense_bf16 => |m| if (m.len != expected_elems) return error.InvalidWeights,
        .callback => {},
    }
}

fn alignUp64(n: usize) Error!usize {
    const added = std.math.add(usize, n, CACHELINE_ALIGN - 1) catch return error.Overflow;
    return added & ~(CACHELINE_ALIGN - 1);
}

fn defaultWiredBudget() sysinfo.Error!budget_mod.Budget {
    const facts = try sysinfo.Facts.read();
    return budget_mod.Budget.compute(.{
        .memory_bytes = facts.memory_bytes,
        .user_wire_limit = facts.user_wire_limit,
        .global_user_wire_limit = facts.global_user_wire_limit,
        .gpu_working_set_bytes = facts.memory_bytes,
        .configured_cap = null,
        .headroom_bytes = budget_mod.DEFAULT_HEADROOM_BYTES,
    });
}

fn allocArenaSlice(comptime T: type, arena: *arena_mod.WiredArena, count: usize) Error![]T {
    const raw = try arena.allocAligned(std.math.mul(usize, count, @sizeOf(T)) catch return error.Overflow, CACHELINE_ALIGN);
    const typed: []T = @alignCast(std.mem.bytesAsSlice(T, raw));
    return typed[0..count];
}

inline fn sigmoidF32(x: f32) f32 {
    const clamped = std.math.clamp(x, -80.0, 80.0);
    return 1.0 / (1.0 + @exp(-clamped));
}

inline fn siluF32(x: f32) f32 {
    return x * sigmoidF32(x);
}

inline fn logProbOf(prob: f32) f32 {
    return if (prob > 1e-38) @log(prob) else -88.0;
}

fn applyRmsNormInPlace(vec: []f32, weight: ?[]const f32, eps: f32) void {
    var sum_sq: f64 = 0.0;
    for (vec) |x| {
        const xf: f64 = x;
        sum_sq += xf * xf;
    }
    const mean_sq: f64 = sum_sq / @as(f64, @floatFromInt(vec.len));
    const inv_rms: f32 = @floatCast(1.0 / @sqrt(mean_sq + @as(f64, eps)));
    if (weight) |w| {
        for (vec, w) |*x, wi| x.* = x.* * inv_rms * wi;
    } else {
        for (vec) |*x| x.* *= inv_rms;
    }
}

fn applyHeadRmsNorm(vec: []f32, num_heads: u32, head_dim: u32, weight: []const f32, eps: f32) void {
    const hd: usize = head_dim;
    for (0..num_heads) |hi| {
        applyRmsNormInPlace(vec[hi * hd ..][0..hd], weight, eps);
    }
}

fn matVecF32(mat: []const f32, vec: []const f32, rows: usize, cols: usize, out: []f32) void {
    for (0..rows) |r| {
        const row = mat[r * cols ..][0..cols];
        var acc0: f32 = 0.0;
        var acc1: f32 = 0.0;
        var acc2: f32 = 0.0;
        var acc3: f32 = 0.0;
        var c: usize = 0;
        while (c + 4 <= cols) : (c += 4) {
            acc0 += row[c + 0] * vec[c + 0];
            acc1 += row[c + 1] * vec[c + 1];
            acc2 += row[c + 2] * vec[c + 2];
            acc3 += row[c + 3] * vec[c + 3];
        }
        var acc = (acc0 + acc1) + (acc2 + acc3);
        while (c < cols) : (c += 1) acc += row[c] * vec[c];
        out[r] = acc;
    }
}

fn matVecAccumF32(mat: []const f32, vec: []const f32, rows: usize, cols: usize, out: []f32) void {
    for (0..rows) |r| {
        const row = mat[r * cols ..][0..cols];
        var acc0: f32 = 0.0;
        var acc1: f32 = 0.0;
        var acc2: f32 = 0.0;
        var acc3: f32 = 0.0;
        var c: usize = 0;
        while (c + 4 <= cols) : (c += 4) {
            acc0 += row[c + 0] * vec[c + 0];
            acc1 += row[c + 1] * vec[c + 1];
            acc2 += row[c + 2] * vec[c + 2];
            acc3 += row[c + 3] * vec[c + 3];
        }
        var acc = (acc0 + acc1) + (acc2 + acc3);
        while (c < cols) : (c += 1) acc += row[c] * vec[c];
        out[r] += acc;
    }
}

fn containsNode(haystack: []const u16, needle: u16) bool {
    for (haystack) |item| {
        if (item == needle) return true;
    }
    return false;
}

fn sortFrontierDescending(
    comptime max_nodes: u16,
    tree: *const tree_mod.DraftTree(max_nodes),
    frontier: []u16,
) void {
    var i: usize = 1;
    while (i < frontier.len) : (i += 1) {
        const key = frontier[i];
        const key_prob = tree.nodes[key].cum_prob;
        var j: usize = i;
        while (j > 0) : (j -= 1) {
            const prev = frontier[j - 1];
            const prev_prob = tree.nodes[prev].cum_prob;
            const should_swap = (key_prob > prev_prob) or (key_prob == prev_prob and key < prev);
            if (!should_swap) break;
            frontier[j] = prev;
        }
        frontier[j] = key;
    }
}

fn extractDeterministicTopK(
    logits: []const f32,
    inv_temp: f32,
    out_tokens: []u32,
    out_probs: []f32,
) Error!void {
    const got = try extractDeterministicTopKMasked(logits, inv_temp, null, out_tokens, out_probs);
    if (got < out_tokens.len) return error.InvalidLogits;
}

fn extractDeterministicTopKMasked(
    logits: []const f32,
    inv_temp: f32,
    allowed_mask: ?[]const u32,
    out_tokens: []u32,
    out_probs: []f32,
) Error!usize {
    const k = out_tokens.len;
    std.debug.assert(k > 0 and k <= MAX_TOP_K_PER_POS and k <= logits.len);
    std.debug.assert(out_probs.len == k);

    var max_val: f32 = -std.math.inf(f32);
    var allowed_count: usize = 0;
    for (logits, 0..) |val, idx| {
        if (!std.math.isFinite(val)) return error.InvalidLogits;
        if (allowed_mask) |mask| {
            const word = mask[idx >> 5];
            if ((word & (@as(u32, 1) << @as(u5, @intCast(idx & 31)))) == 0) continue;
        }
        allowed_count += 1;
        const scaled = val * inv_temp;
        if (scaled > max_val) max_val = scaled;
    }
    if (allowed_count == 0) return 0;

    var top_vals: [MAX_TOP_K_PER_POS]f32 = @splat(-std.math.inf(f32));
    var top_ids: [MAX_TOP_K_PER_POS]u32 = @splat(std.math.maxInt(u32));
    var denom: f64 = 0.0;

    for (logits, 0..) |val, idx| {
        if (allowed_mask) |mask| {
            const word = mask[idx >> 5];
            if ((word & (@as(u32, 1) << @as(u5, @intCast(idx & 31)))) == 0) continue;
        }
        const tok_id: u32 = @intCast(idx);
        const scaled = val * inv_temp;
        denom += @exp(@as(f64, scaled - max_val));

        const last = k - 1;
        const better_than_last = (scaled > top_vals[last]) or
            (scaled == top_vals[last] and tok_id < top_ids[last]);
        if (!better_than_last) continue;

        var pos: usize = last;
        while (pos > 0) : (pos -= 1) {
            const prev_val = top_vals[pos - 1];
            const prev_id = top_ids[pos - 1];
            const better_than_prev = (scaled > prev_val) or
                (scaled == prev_val and tok_id < prev_id);
            if (!better_than_prev) break;
            top_vals[pos] = prev_val;
            top_ids[pos] = prev_id;
        }
        top_vals[pos] = scaled;
        top_ids[pos] = tok_id;
    }

    if (!(denom > 0.0) or !std.math.isFinite(denom)) return error.InvalidLogits;
    const take: usize = @min(k, allowed_count);
    for (0..take) |i| {
        out_tokens[i] = top_ids[i];
        out_probs[i] = @floatCast(@exp(@as(f64, top_vals[i] - max_val)) / denom);
    }
    return take;
}

fn findAnyEntry(amf_file: *const amf.AmfFile, names: []const []const u8) ?*const amf.TensorEntry {
    for (names) |nm| {
        if (amf_file.parsed.find(nm)) |e| return e;
    }
    return null;
}

fn jsonU32Opt(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = obj.get(key) orelse return null;
    if (v == .integer and v.integer >= 0 and v.integer <= std.math.maxInt(u32)) return @intCast(v.integer);
    return null;
}

fn jsonF32Opt(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .float => @floatCast(v.float),
        .integer => @floatFromInt(v.integer),
        else => null,
    };
}

fn mixScalarTaps(weights: []const f32, taps_flat: []const f32, num_taps: usize, h: usize, out: []f32) void {
    @memset(out[0..h], 0.0);
    for (0..num_taps) |t| {
        const w = weights[t];
        const src = taps_flat[t * h ..][0..h];
        for (0..h) |d| out[d] += w * src[d];
    }
}

fn resolveTokEmbed(
    tok_embed: []const f32,
    hook_opt: ?TokEmbedHook,
    token_id: u32,
    h: usize,
    scratch: []f32,
) Error![]const f32 {
    if (hook_opt) |hk| {
        switch (hk) {
            .dense_f32 => |tbl| {
                const off = @as(usize, token_id) * h;
                if (off + h > tbl.len) return error.TokenOutOfRange;
                return tbl[off..][0..h];
            },
            .callback => |cb| {
                try cb.read_fn(cb.ctx, token_id, scratch[0..h]);
                return scratch[0..h];
            },
        }
    }
    const off = @as(usize, token_id) * h;
    if (off + h > tok_embed.len) return error.TokenOutOfRange;
    return tok_embed[off..][0..h];
}

fn matVecBf16(mat: []const u16, vec: []const f32, rows: usize, cols: usize, out: []f32) void {
    const Vec8f = @Vector(8, f32);
    const Vec8u32 = @Vector(8, u32);
    for (0..rows) |r| {
        const row = mat[r * cols ..][0..cols];
        var vacc: Vec8f = @splat(0.0);
        var c: usize = 0;
        while (c + 8 <= cols) : (c += 8) {
            const raw: [8]u16 = row[c..][0..8].*;
            var bits: Vec8u32 = undefined;
            inline for (0..8) |k| bits[k] = @as(u32, raw[k]) << 16;
            const wf: Vec8f = @bitCast(bits);
            const xf: Vec8f = vec[c..][0..8].*;
            vacc = @mulAdd(Vec8f, wf, xf, vacc);
        }
        var acc: f32 = @reduce(.Add, vacc);
        while (c < cols) : (c += 1) {
            acc += quant.bf16ToF32(row[c]) * vec[c];
        }
        out[r] = acc;
    }
}

fn matVecBf16Accum(mat: []const u16, vec: []const f32, rows: usize, cols: usize, out: []f32) void {
    const Vec8f = @Vector(8, f32);
    const Vec8u32 = @Vector(8, u32);
    for (0..rows) |r| {
        const row = mat[r * cols ..][0..cols];
        var vacc: Vec8f = @splat(0.0);
        var c: usize = 0;
        while (c + 8 <= cols) : (c += 8) {
            const raw: [8]u16 = row[c..][0..8].*;
            var bits: Vec8u32 = undefined;
            inline for (0..8) |k| bits[k] = @as(u32, raw[k]) << 16;
            const wf: Vec8f = @bitCast(bits);
            const xf: Vec8f = vec[c..][0..8].*;
            vacc = @mulAdd(Vec8f, wf, xf, vacc);
        }
        var acc: f32 = @reduce(.Add, vacc);
        while (c < cols) : (c += 1) {
            acc += quant.bf16ToF32(row[c]) * vec[c];
        }
        out[r] += acc;
    }
}

fn readEntryAsF32(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    entry: *const amf.TensorEntry,
) LoadedDFlareBundle.LoadError![]f32 {
    const dt = try entry.dataType();
    var elems: usize = 1;
    for (0..entry.rank) |d| {
        elems = std.math.mul(usize, elems, entry.shape[d]) catch return error.Overflow;
    }
    const blob_len: usize = @intCast(entry.blob_bytes);
    const raw = try allocator.alloc(u8, blob_len);
    defer allocator.free(raw);

    var done: usize = 0;
    while (done < blob_len) {
        const rc = std.c.pread(amf_file.fd, raw.ptr + done, blob_len - done, @intCast(entry.blob_offset + done));
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.ReadFailed;
        }
        if (rc == 0) return error.ShortRead;
        done += @intCast(rc);
    }
    if (std.hash.Crc32.hash(raw) != entry.blob_crc32) return error.BlobChecksumMismatch;

    const out = try allocator.alloc(f32, elems);
    errdefer allocator.free(out);
    switch (dt) {
        .f32 => {
            if (entry.data_bytes < elems * 4) return error.InvalidWeights;
            for (0..elems) |i| {
                out[i] = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
            }
        },
        .f16 => {
            if (entry.data_bytes < elems * 2) return error.InvalidWeights;
            for (0..elems) |i| {
                const bits = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
                out[i] = @floatCast(@as(f16, @bitCast(bits)));
            }
        },
        .bf16 => {
            if (entry.data_bytes < elems * 2) return error.InvalidWeights;
            for (0..elems) |i| {
                out[i] = quant.bf16ToF32(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
            }
        },
        else => {
            const qf_dt = quant_formats.DType.fromAmf(dt) orelse return error.InvalidWeights;
            const rows: usize = entry.shape[0];
            const cols: usize = if (entry.rank >= 2) entry.shape[1] else 1;
            if (rows * cols != elems) return error.InvalidWeights;
            const gs: u16 = if (entry.group_size > 0) entry.group_size else 64;
            const params_slice: []const u8 = if (entry.params_offset > 0 and entry.params_offset <= raw.len)
                raw[@intCast(entry.params_offset)..]
            else
                &.{};
            for (0..rows) |r| {
                quant_formats.dequantRowCpuRef(qf_dt, gs, raw, params_slice, r, rows, cols, out[r * cols ..][0..cols]) catch return error.InvalidWeights;
            }
        },
    }
    return out;
}

fn readEntryAsBf16(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    entry: *const amf.TensorEntry,
) LoadedDFlareBundle.LoadError![]u16 {
    const dt = try entry.dataType();
    var elems: usize = 1;
    for (0..entry.rank) |d| {
        elems = std.math.mul(usize, elems, entry.shape[d]) catch return error.Overflow;
    }
    const blob_len: usize = @intCast(entry.blob_bytes);
    const raw = try allocator.alloc(u8, blob_len);
    defer allocator.free(raw);

    var done: usize = 0;
    while (done < blob_len) {
        const rc = std.c.pread(amf_file.fd, raw.ptr + done, blob_len - done, @intCast(entry.blob_offset + done));
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.ReadFailed;
        }
        if (rc == 0) return error.ShortRead;
        done += @intCast(rc);
    }
    if (std.hash.Crc32.hash(raw) != entry.blob_crc32) return error.BlobChecksumMismatch;

    const out = try allocator.alloc(u16, elems);
    errdefer allocator.free(out);
    switch (dt) {
        .bf16 => {
            if (entry.data_bytes < elems * 2) return error.InvalidWeights;
            for (0..elems) |i| {
                out[i] = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
            }
        },
        .f16 => {
            if (entry.data_bytes < elems * 2) return error.InvalidWeights;
            for (0..elems) |i| {
                const fv: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little))));
                out[i] = @truncate(@as(u32, @bitCast(fv)) >> 16);
            }
        },
        .f32 => {
            if (entry.data_bytes < elems * 4) return error.InvalidWeights;
            for (0..elems) |i| {
                const bits = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
                out[i] = @truncate(bits >> 16);
            }
        },
        else => {
            const qf_dt = quant_formats.DType.fromAmf(dt) orelse return error.InvalidWeights;
            const rows: usize = entry.shape[0];
            const cols: usize = if (entry.rank >= 2) entry.shape[1] else 1;
            if (rows * cols != elems) return error.InvalidWeights;
            const gs: u16 = if (entry.group_size > 0) entry.group_size else 64;
            const params_slice: []const u8 = if (entry.params_offset > 0 and entry.params_offset <= raw.len)
                raw[@intCast(entry.params_offset)..]
            else
                &.{};
            const row_tmp = try allocator.alloc(f32, cols);
            defer allocator.free(row_tmp);
            for (0..rows) |r| {
                quant_formats.dequantRowCpuRef(qf_dt, gs, raw, params_slice, r, rows, cols, row_tmp) catch return error.InvalidWeights;
                for (0..cols) |c| {
                    out[r * cols + c] = @truncate(@as(u32, @bitCast(row_tmp[c])) >> 16);
                }
            }
        },
    }
    return out;
}

fn loadEntryIntoF32Pool(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    entry: *const amf.TensorEntry,
    pool: *[256][]f32,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError![]f32 {
    const vec = try readEntryAsF32(allocator, amf_file, entry);
    errdefer allocator.free(vec);
    if (pool_len.* >= pool.len) return error.Overflow;
    pool[pool_len.*] = vec;
    pool_len.* += 1;
    return vec;
}

fn loadEntryIntoBf16Pool(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    entry: *const amf.TensorEntry,
    pool: *[256][]u16,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError![]u16 {
    const vec = try readEntryAsBf16(allocator, amf_file, entry);
    errdefer allocator.free(vec);
    if (pool_len.* >= pool.len) return error.Overflow;
    pool[pool_len.*] = vec;
    pool_len.* += 1;
    return vec;
}

fn loadAmfTensorAsF32256(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    names: []const []const u8,
    expected_elems: usize,
    pool: *[256][]f32,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError![]f32 {
    const entry = findAnyEntry(amf_file, names) orelse return error.MissingTensor;
    const vec = try loadEntryIntoF32Pool(allocator, amf_file, entry, pool, pool_len);
    if (vec.len != expected_elems) return error.DimensionMismatch;
    return vec;
}

fn loadAmfTensorAsF32Opt256(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    names: []const []const u8,
    expected_elems: usize,
    pool: *[256][]f32,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError!?[]f32 {
    const entry = findAnyEntry(amf_file, names) orelse return null;
    const vec = try loadEntryIntoF32Pool(allocator, amf_file, entry, pool, pool_len);
    if (vec.len != expected_elems) return error.DimensionMismatch;
    return vec;
}

fn loadAmfTensorAnyLen256(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    names: []const []const u8,
    pool: *[256][]f32,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError![]f32 {
    const entry = findAnyEntry(amf_file, names) orelse return error.MissingTensor;
    return loadEntryIntoF32Pool(allocator, amf_file, entry, pool, pool_len);
}

fn findLayerEntry(
    amf_file: *const amf.AmfFile,
    layer: usize,
    suffixes: []const []const u8,
    buf0: *[80]u8,
    buf1: *[80]u8,
    buf2: *[80]u8,
) LoadedDFlareBundle.LoadError!?*const amf.TensorEntry {
    _ = buf2;
    for (suffixes) |sfx| {
        const nm_l = std.fmt.bufPrint(buf0, "layers.{d}.{s}", .{ layer, sfx }) catch return error.Overflow;
        if (amf_file.parsed.find(nm_l)) |e| return e;
        const nm_b = std.fmt.bufPrint(buf1, "blk.{d}.{s}", .{ layer, sfx }) catch return error.Overflow;
        if (amf_file.parsed.find(nm_b)) |e| return e;
    }
    return null;
}

fn loadLayerF32(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    layer: usize,
    suffixes: []const []const u8,
    expected_elems: usize,
    b0: *[80]u8,
    b1: *[80]u8,
    b2: *[80]u8,
    pool: *[256][]f32,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError![]f32 {
    const e = (try findLayerEntry(amf_file, layer, suffixes, b0, b1, b2)) orelse return error.MissingTensor;
    const vec = try loadEntryIntoF32Pool(allocator, amf_file, e, pool, pool_len);
    if (vec.len != expected_elems) return error.DimensionMismatch;
    return vec;
}

fn loadLayerF32Opt(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    layer: usize,
    suffixes: []const []const u8,
    expected_elems: usize,
    b0: *[80]u8,
    b1: *[80]u8,
    b2: *[80]u8,
    pool: *[256][]f32,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError!?[]f32 {
    const e = (try findLayerEntry(amf_file, layer, suffixes, b0, b1, b2)) orelse return null;
    const vec = try loadEntryIntoF32Pool(allocator, amf_file, e, pool, pool_len);
    if (vec.len != expected_elems) return error.DimensionMismatch;
    return vec;
}

fn loadLayerBf16(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    layer: usize,
    suffixes: []const []const u8,
    expected_elems: usize,
    b0: *[80]u8,
    b1: *[80]u8,
    b2: *[80]u8,
    pool: *[256][]u16,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError![]u16 {
    const e = (try findLayerEntry(amf_file, layer, suffixes, b0, b1, b2)) orelse return error.MissingTensor;
    const vec = try loadEntryIntoBf16Pool(allocator, amf_file, e, pool, pool_len);
    if (vec.len != expected_elems) return error.DimensionMismatch;
    return vec;
}

fn loadLayerBf16Opt(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    layer: usize,
    suffixes: []const []const u8,
    expected_elems: usize,
    b0: *[80]u8,
    b1: *[80]u8,
    b2: *[80]u8,
    pool: *[256][]u16,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError!?[]u16 {
    const e = (try findLayerEntry(amf_file, layer, suffixes, b0, b1, b2)) orelse return null;
    const vec = try loadEntryIntoBf16Pool(allocator, amf_file, e, pool, pool_len);
    if (vec.len != expected_elems) return error.DimensionMismatch;
    return vec;
}

// ============================================================================
// Tests
// ============================================================================

fn testBudget(cap_bytes: usize) budget_mod.Budget {
    return budget_mod.Budget.compute(.{
        .memory_bytes = cap_bytes * 8,
        .user_wire_limit = cap_bytes * 2,
        .global_user_wire_limit = cap_bytes * 2,
        .gpu_working_set_bytes = cap_bytes * 2,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
}

fn verifyTreeInvariants(comptime max_nodes: u16, t: *const tree_mod.DraftTree(max_nodes)) !void {
    try std.testing.expect(t.count >= 1 and t.count <= max_nodes);
    const root = t.nodes[0];
    try std.testing.expectEqual(tree_mod.NO_NODE, root.parent);
    try std.testing.expectEqual(@as(u16, 0), root.depth);
    try std.testing.expectEqual(@as(u64, 1), root.ancestor_mask);

    for (1..t.count) |i| {
        const idx: u16 = @intCast(i);
        const node = t.nodes[idx];
        try std.testing.expect(node.parent < idx);
        try std.testing.expectEqual(t.nodes[node.parent].depth + 1, node.depth);
        const self_bit: u64 = @as(u64, 1) << @as(u6, @intCast(idx));
        try std.testing.expectEqual(t.nodes[node.parent].ancestor_mask | self_bit, node.ancestor_mask);
        try std.testing.expect((node.ancestor_mask & 1) == 1);
        try std.testing.expect((node.ancestor_mask & self_bit) != 0);
    }
}

const SyntheticBlockBuffers = struct {
    layers: []TransformerBlockWeights,
    storage: [][]f32,

    fn initZeroed(
        allocator: std.mem.Allocator,
        num_layers: usize,
        h: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
    ) !SyntheticBlockBuffers {
        const layers = try allocator.alloc(TransformerBlockWeights, num_layers);
        errdefer allocator.free(layers);
        const storage = try allocator.alloc([]f32, num_layers * 7);
        var built: usize = 0;
        errdefer {
            for (0..built) |i| allocator.free(storage[i]);
            allocator.free(storage);
        }

        for (0..num_layers) |l| {
            const sizes = [_]usize{ q_dim * h, kv_dim * h, kv_dim * h, h * q_dim, inter * h, inter * h, h * inter };
            var ptrs: [7][]f32 = undefined;
            for (sizes, 0..) |sz, si| {
                const buf = try allocator.alloc(f32, sz);
                @memset(buf, 0.0);
                storage[built] = buf;
                built += 1;
                ptrs[si] = buf;
            }
            // Small deterministic identity-like weights on Q/K/V/O + MLP so signal flows cleanly.
            for (0..@min(q_dim, h)) |d| ptrs[0][d * h + d] = 0.25;
            for (0..@min(kv_dim, h)) |d| {
                ptrs[1][d * h + d] = 0.25;
                ptrs[2][d * h + d] = 0.25;
            }
            for (0..@min(h, q_dim)) |d| ptrs[3][d * q_dim + d] = 0.125;
            for (0..@min(inter, h)) |d| {
                ptrs[4][d * h + d] = 0.2;
                ptrs[5][d * h + d] = 0.2;
                ptrs[6][d * inter + d] = 0.1;
            }
            layers[l] = .{
                .q_proj = ptrs[0],
                .k_proj = ptrs[1],
                .v_proj = ptrs[2],
                .o_proj = ptrs[3],
                .mlp_gate_proj = ptrs[4],
                .mlp_up_proj = ptrs[5],
                .mlp_down_proj = ptrs[6],
            };
        }
        return .{ .layers = layers, .storage = storage };
    }

    fn deinit(self: *SyntheticBlockBuffers, allocator: std.mem.Allocator) void {
        for (self.storage) |b| allocator.free(b);
        allocator.free(self.storage);
        allocator.free(self.layers);
    }
};

fn writeSyntheticDFlareAmf(
    allocator: std.mem.Allocator,
    cfg: DFlareConfig,
    weights: DFlareWeights,
) ![]u8 {
    const h: usize = cfg.hidden_size;
    const inter: usize = cfg.intermediate_size;
    const q_dim: usize = cfg.qDim();
    const kv_dim: usize = cfg.kvDim();
    const mr: usize = cfg.markov_rank;
    const v: usize = cfg.vocab_size;

    const Item = struct {
        name: [47]u8 = @splat(0),
        name_len: usize = 0,
        data: []const f32,
        shape: [4]u32,
        rank: u8,
        kind: amf.TensorKind,
    };

    var items: [128]Item = undefined;
    var count: usize = 0;

    const addItem = struct {
        fn push(arr: *[128]Item, n: *usize, nm: []const u8, data: []const f32, s0: u32, s1: u32, rank: u8, kind: amf.TensorKind) void {
            var it: Item = .{ .data = data, .shape = .{ s0, s1, 0, 0 }, .rank = rank, .kind = kind };
            @memcpy(it.name[0..nm.len], nm);
            it.name_len = nm.len;
            arr[n.*] = it;
            n.* += 1;
        }
    }.push;

    if (weights.target_fusion_proj) |fp| {
        addItem(&items, &count, "dflare.fc.weight", fp, cfg.hidden_size, cfg.num_target_taps * cfg.hidden_size, 2, .dense);
    }
    addItem(&items, &count, "dflare.mask_embed", weights.mask_embed, @intCast(weights.mask_embed.len / h), cfg.hidden_size, 2, .embedding);
    addItem(&items, &count, "dflare.tok_embed.weight", weights.tok_embed, cfg.vocab_size, cfg.hidden_size, 2, .embedding);

    var nbuf: [48]u8 = undefined;
    for (weights.layers, 0..) |ly, li| {
        addItem(&items, &count, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn.q_proj.weight", .{li}), ly.q_proj, @intCast(q_dim), @intCast(h), 2, .dense);
        addItem(&items, &count, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn.k_proj.weight", .{li}), ly.k_proj, @intCast(kv_dim), @intCast(h), 2, .dense);
        addItem(&items, &count, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn.v_proj.weight", .{li}), ly.v_proj, @intCast(kv_dim), @intCast(h), 2, .dense);
        addItem(&items, &count, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn.o_proj.weight", .{li}), ly.o_proj, @intCast(h), @intCast(q_dim), 2, .dense);
        addItem(&items, &count, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj.weight", .{li}), ly.mlp_gate_proj, @intCast(inter), @intCast(h), 2, .dense);
        addItem(&items, &count, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj.weight", .{li}), ly.mlp_up_proj, @intCast(inter), @intCast(h), 2, .dense);
        addItem(&items, &count, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj.weight", .{li}), ly.mlp_down_proj, @intCast(h), @intCast(inter), 2, .dense);
    }

    addItem(&items, &count, "dflare.markov_w1.weight", weights.markov_w1, @intCast(mr), @intCast(weights.markov_w1.len / mr), 2, .dense);
    addItem(&items, &count, "dflare.markov_w2.weight", weights.markov_w2, @intCast(h), @intCast(mr), 2, .dense);
    switch (weights.lm_head) {
        .dense_f32 => |lm| addItem(&items, &count, "dflare.lm_head.weight", lm, @intCast(v), @intCast(h), 2, .lm_head),
        else => return error.InvalidWeights,
    }

    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_raw = count * @sizeOf(amf.TensorEntry);
    const dir_bytes = try page.alignUp(dir_raw);

    const entries = try allocator.alloc(amf.TensorEntry, count);
    defer allocator.free(entries);
    const blobs = try allocator.alloc([]u8, count);
    var built: usize = 0;
    defer {
        for (0..built) |i| allocator.free(blobs[i]);
        allocator.free(blobs);
    }

    var next_off: usize = dir_offset + dir_bytes;
    for (0..count) |i| {
        const it = items[i];
        const data_bytes = it.data.len * 4;
        const blob_bytes = try page.alignUp(data_bytes);
        const blob = try allocator.alloc(u8, blob_bytes);
        errdefer allocator.free(blob);
        @memset(blob, 0);
        for (it.data, 0..) |val, j| {
            std.mem.writeInt(u32, blob[j * 4 ..][0..4], @bitCast(val), .little);
        }
        var e = amf.TensorEntry.zeroed(it.name[0..it.name_len], it.kind, .f32);
        e.rank = it.rank;
        e.shape = it.shape;
        e.data_bytes = data_bytes;
        e.blob_offset = next_off;
        e.blob_bytes = blob_bytes;
        e.blob_crc32 = std.hash.Crc32.hash(blob);
        entries[i] = e;
        blobs[i] = blob;
        built += 1;
        next_off += blob_bytes;
    }

    const image = try allocator.alloc(u8, next_off);
    errdefer allocator.free(image);
    @memset(image, 0);
    for (0..count) |i| {
        @memcpy(image[dir_offset + i * @sizeOf(amf.TensorEntry) ..][0..@sizeOf(amf.TensorEntry)], std.mem.asBytes(&entries[i]));
        @memcpy(image[@intCast(entries[i].blob_offset)..][0..@intCast(entries[i].blob_bytes)], blobs[i]);
    }
    var hdr: amf.Header = .{
        .magic = amf.MAGIC,
        .version = amf.FORMAT_VERSION,
        .flags = 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = amf.CHECKSUM_CRC32_IEEE,
        .tensor_count = @intCast(count),
        .reserved0 = 0,
        .directory_offset = dir_offset,
        .directory_bytes = dir_bytes,
        .directory_crc32 = std.hash.Crc32.hash(image[dir_offset..][0..dir_raw]),
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    const model_name = "synthetic-dflash-d14";
    @memcpy(hdr.model_id[0..model_name.len], model_name);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return image;
}

test "D14 DFlareProposer (3L & 5L, block_size=8, markov_rank=256) sequential Markov conditioning without backbone re-run, D7 pruning, and M=8 Corex3 D9 T=0/T>0 parity" {
    const gpa = std.testing.allocator;
    const h: usize = 16;
    const inter: usize = 32;
    const v: usize = 16;
    const mr: usize = DFLARE_MARKOV_RANK;

    for ([_]DFlareVariant{ .inkling2_mini_3l, .inkling2_small_5l }) |variant| {
        var cfg = if (variant == .inkling2_mini_3l)
            DFlareConfig.miniDefault(@intCast(h), @intCast(inter), 4, 2, 4, @intCast(v))
        else
            DFlareConfig.smallDefault(@intCast(h), @intCast(inter), 4, 2, 4, @intCast(v));
        cfg.top_k_per_pos = 1; // linear chain of block_size = 8
        cfg.max_tree_nodes = 16;

        const need = try DFlareWorkspace.requiredBytes(cfg);
        try std.testing.expectEqual(@as(usize, 0), need % page.APPLE_PAGE_SIZE);
        var ws = try DFlareWorkspace.init(cfg, testBudget(need * 2));
        defer ws.deinit();
        try std.testing.expectEqual(@as(u64, need), ws.totalWiredBytes());

        var bbuf = try SyntheticBlockBuffers.initZeroed(gpa, cfg.num_layers, h, inter, cfg.qDim(), cfg.kvDim());
        defer bbuf.deinit(gpa);

        const taps: usize = cfg.num_target_taps;
        const fc = try gpa.alloc(f32, h * taps * h);
        defer gpa.free(fc);
        @memset(fc, 0.0);
        for (0..h) |d| fc[d * (taps * h) + d] = 1.0;

        var mask_embed: [DFLARE_BLOCK_SIZE * h]f32 = @splat(0.0);
        for (0..DFLARE_BLOCK_SIZE) |pos| mask_embed[pos * h + pos] = 0.5;

        // Encode token `t` uniquely along dimension `t % h` so Markov head cleanly depends on `prev_token`.
        var tok_embed: [v * h]f32 = @splat(0.0);
        for (0..v) |t| tok_embed[t * h + (t % h)] = 2.0;

        // Markov transition: `tok_{j} = (prev_tok + 1) % 8` driven by `tok_embed(prev_tok)`
        // with decaying recurrent `rnn_hidden` (`markov_rnn_proj = 0.05 * I`) via rank-256 bottleneck!
        const mw1 = try gpa.alloc(f32, mr * 2 * h);
        defer gpa.free(mw1);
        @memset(mw1, 0.0);
        for (0..h) |d| {
            // Map second half (`tok_embed`) into channel `d` of `markov_u`.
            mw1[d * (2 * h) + (h + d)] = 3.0;
        }
        const mr_rnn = try gpa.alloc(f32, mr * mr);
        defer gpa.free(mr_rnn);
        @memset(mr_rnn, 0.0);
        for (0..mr) |d| mr_rnn[d * mr + d] = 0.05;

        const mw2 = try gpa.alloc(f32, h * mr);
        defer gpa.free(mw2);
        @memset(mw2, 0.0);
        for (0..8) |d| {
            // Shift channel 0..7 by +1 mod 8 -> sequential chain 1 -> 2 -> ... -> 7 -> 0!
            const next_ch = (d + 1) % 8;
            mw2[next_ch * mr + d] = 4.0;
        }

        var lm_head: [v * h]f32 = @splat(0.0);
        for (0..v) |t| lm_head[t * h + (t % h)] = 5.0;

        const weights: DFlareWeights = .{
            .target_fusion_proj = fc,
            .mask_embed = &mask_embed,
            .tok_embed = &tok_embed,
            .layers = bbuf.layers,
            .markov_w1 = mw1,
            .markov_rnn_proj = mr_rnn,
            .markov_w2 = mw2,
            .lm_head = .{ .dense_f32 = &lm_head },
        };

        var proposer = try DFlareProposer.init(cfg, weights, &ws);
        const tap_vec: [h]f32 = @splat(0.05);
        const taps_slice = [_][]const f32{ &tap_vec, &tap_vec, &tap_vec, &tap_vec };

        var tree: CandidateTree = undefined;
        const res = try proposer.propose(0, taps_slice[0..taps], null, &tree);

        // Exactly 1 backbone run for all 8 Markov steps -> linear chain of depth 8 (9 nodes total)!
        try std.testing.expectEqual(@as(u64, 1), res.backbone_runs);
        try std.testing.expectEqual(@as(u64, DFLARE_BLOCK_SIZE), res.markov_steps);
        try std.testing.expectEqual(@as(u16, DFLARE_BLOCK_SIZE + 1), res.tree_nodes);
        try std.testing.expectEqual(@as(u16, DFLARE_BLOCK_SIZE), res.max_depth);
        try verifyTreeInvariants(CandidateTree.capacity, &tree);

        // Verify sequential Markov chain tokens 1 -> 2 -> 3 -> ... -> 7 -> 0 and valid log-probs.
        for (1..tree.count) |ni| {
            const expected_tok: u32 = @intCast(ni % 8);
            try std.testing.expectEqual(expected_tok, tree.nodes[ni].token_id);
            try std.testing.expect(proposer.nodeLogProb(@intCast(ni)) <= 0.0);
        }

        // Verify D9 lossless greedy (T = 0) across full 8-step M = 9 (or pruned M = 8) Core x 3 verify!
        var target_argmax: [CandidateTree.capacity]u32 = @splat(15);
        for (0..tree.count - 1) |u| {
            target_argmax[u] = tree.nodes[u + 1].token_id;
        }
        target_argmax[tree.count - 1] = 14; // bonus token

        var acc_nodes: [16]u16 = undefined;
        var emitted: [16]u32 = undefined;
        const core3_res = try DFlareProposer.verifyCore3BatchGreedy(
            CandidateTree.capacity,
            &tree,
            target_argmax[0..tree.count],
            &acc_nodes,
            &emitted,
        );
        try std.testing.expectEqual(@as(u8, 3), core3_res.core_passes);
        try std.testing.expectEqual(@as(u16, DFLARE_BLOCK_SIZE), core3_res.accept.draft_accepted);
        try std.testing.expectEqual(@as(u32, 14), core3_res.accept.bonus_token);

        // Wire D7 pruning to hard_cap = 8 (M = 8 batched Core x 3 window!).
        const prune_cost = tree_mod.PruneCostModel.uniformBeta(50.0, 4.0, 2048.0, 0.0, 100.0, 2, 16);
        var node_experts: [16 * 2]tree_mod.ExpertBitset = @splat(.{});
        for (0..tree.count) |ni| {
            node_experts[ni * 2 + 0].set(@intCast(ni % 4));
            node_experts[ni * 2 + 1].set(@intCast(ni % 4));
        }
        const ps = try proposer.pruneWithD7(
            CandidateTree.capacity,
            &tree,
            prune_cost,
            node_experts[0 .. tree.count * 2],
            null,
            8,
        );
        try std.testing.expect(ps.kept_nodes <= 8);
        try verifyTreeInvariants(CandidateTree.capacity, &tree);

        // Verify D9 stochastic acceptance at T = 0.7 preserves target categorical distribution (TVD < 0.02)!
        const small_v: usize = 4;
        var st_tree = tree_mod.DraftTree(16).init(0);
        const c1 = st_tree.insertChild(0, 1, 0.7, .l2).?;
        _ = st_tree.insertChild(c1, 2, 0.6, .l2).?;

        var tgt_logits: [3 * small_v]f32 = @splat(0.0);
        const root_logits = [_]f32{ -0.4, 1.1, 0.2, -0.9 };
        @memcpy(tgt_logits[0..small_v], &root_logits);
        @memcpy(tgt_logits[small_v .. 2 * small_v], &[_]f32{ 0.1, 0.2, 1.3, -0.5 });
        @memcpy(tgt_logits[2 * small_v .. 3 * small_v], &[_]f32{ 0.5, 0.5, 0.5, 0.5 });

        var expected_probs: [small_v]f32 = undefined;
        try accept_mod.softmaxRow(f32, &root_logits, 0.7, &expected_probs);

        var prng = std.Random.DefaultPrng.init(20261125);
        var counts: [small_v]u64 = @splat(0);
        var scratch: [small_v]f32 = undefined;
        var s_acc: [8]u16 = undefined;
        var s_tok: [8]u32 = undefined;
        const rounds: usize = 20_000;
        for (0..rounds) |_| {
            const st_res = try DFlareProposer.verifyCore3BatchStochastic(
                16,
                &st_tree,
                &tgt_logits,
                null,
                small_v,
                0.7,
                prng.random(),
                &scratch,
                &s_acc,
                &s_tok,
            );
            counts[s_tok[0]] += 1;
            _ = st_res;
        }
        var tvd: f64 = 0.0;
        for (0..small_v) |tok| {
            const emp = @as(f64, @floatFromInt(counts[tok])) / @as(f64, @floatFromInt(rounds));
            tvd += @abs(emp - @as(f64, expected_probs[tok]));
        }
        tvd *= 0.5;
        try std.testing.expect(tvd < 0.02);
    }
}

test "D14 DFlareProposer loads standalone arch=dflash AMF bundle and expands multi-branch top-k tree" {
    const gpa = std.testing.allocator;
    const h: usize = 8;
    const inter: usize = 16;
    const v: usize = 12;
    const mr: usize = 16;

    var cfg: DFlareConfig = .{
        .variant = .dflash_standalone,
        .num_layers = 3,
        .block_size = 4,
        .top_k_per_pos = 2,
        .max_tree_nodes = 16,
        .hidden_size = @intCast(h),
        .intermediate_size = @intCast(inter),
        .num_attention_heads = 2,
        .num_key_value_heads = 2,
        .head_dim = 4,
        .vocab_size = @intCast(v),
        .markov_rank = @intCast(mr),
        .num_target_taps = 1,
    };

    var bbuf = try SyntheticBlockBuffers.initZeroed(gpa, cfg.num_layers, h, inter, cfg.qDim(), cfg.kvDim());
    defer bbuf.deinit(gpa);

    var mask_embed: [4 * h]f32 = @splat(0.0);
    for (0..4) |p| mask_embed[p * h + p] = 0.4;
    var tok_embed: [v * h]f32 = @splat(0.0);
    for (0..v) |t| tok_embed[t * h + (t % h)] = 1.0;
    var mw1: [mr * h]f32 = @splat(0.0);
    for (0..h) |d| mw1[d * h + d] = 1.0;
    var mw2: [h * mr]f32 = @splat(0.0);
    for (0..h) |d| mw2[d * mr + d] = 1.0;
    var lm_head: [v * h]f32 = @splat(0.0);
    for (0..v) |t| lm_head[t * h + (t % h)] = 1.5;

    const img = try writeSyntheticDFlareAmf(gpa, cfg, .{
        .mask_embed = &mask_embed,
        .tok_embed = &tok_embed,
        .layers = bbuf.layers,
        .markov_w1 = &mw1,
        .markov_w2 = &mw2,
        .lm_head = .{ .dense_f32 = &lm_head },
    });
    defer gpa.free(img);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "dflash-mini.amf", .data = img });

    var p_buf: [std.fs.max_path_bytes]u8 = undefined;
    const p_raw = try std.fmt.bufPrint(p_buf[0 .. p_buf.len - 1], ".zig-cache/tmp/{s}/dflash-mini.amf", .{tmp.sub_path});
    p_buf[p_raw.len] = 0;

    const bundle = try LoadedDFlareBundle.loadFromAmfFile(
        gpa,
        p_buf[0..p_raw.len :0],
        cfg,
        null,
        testBudget(512 * 1024),
    );
    defer bundle.deinit(gpa);

    const src: DraftSource = .{ .dflash_standalone = &bundle.proposer };
    try std.testing.expectEqual(DraftSourceKind.dflash_standalone, std.meta.activeTag(src));

    const tap0: [h]f32 = @splat(0.2);
    var tree: CandidateTree = undefined;
    const res = try bundle.proposer.propose(0, &.{&tap0}, null, &tree);

    try std.testing.expectEqual(@as(u64, 1), res.backbone_runs);
    try std.testing.expect(res.tree_nodes >= 7 and res.tree_nodes <= cfg.max_tree_nodes);
    try std.testing.expectEqual(@as(u16, 4), res.max_depth);
    try verifyTreeInvariants(CandidateTree.capacity, &tree);
}

test "D14 DFly2Proposer (5L, block_size=7) consumes dfly_target_kv_layers + hidden_correction + mtp_fusion, truncates via confidence_head, and passes D9 verify" {
    const gpa = std.testing.allocator;
    const h: usize = 16;
    const inter: usize = 32;
    const v: usize = 16;
    const kv_len: usize = 4;

    const cfg: DFly2Config = .{
        .num_layers = DFLY2_DEFAULT_LAYERS, // 5L
        .block_size = DFLY2_BLOCK_SIZE, // 7
        .top_k_per_pos = 1,
        .max_tree_nodes = 16,
        .hidden_size = @intCast(h),
        .intermediate_size = @intCast(inter),
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 4,
        .vocab_size = @intCast(v),
        .num_target_kv_layers = 5,
        .max_target_kv_len = 8,
        .confidence_threshold = 0.5,
    };
    const need = try DFly2Workspace.requiredBytes(cfg);
    try std.testing.expectEqual(@as(usize, 0), need % page.APPLE_PAGE_SIZE);
    var ws = try DFly2Workspace.init(cfg, testBudget(need * 2));
    defer ws.deinit();

    var bbuf = try SyntheticBlockBuffers.initZeroed(gpa, cfg.num_layers, h, inter, cfg.qDim(), cfg.kvDim());
    defer bbuf.deinit(gpa);

    const mtp_proj = try gpa.alloc(f32, h * 2 * h);
    defer gpa.free(mtp_proj);
    @memset(mtp_proj, 0.0);
    for (0..h) |d| {
        mtp_proj[d * (2 * h) + d] = 0.5;
        mtp_proj[d * (2 * h) + (h + d)] = 0.5;
    }

    const hc_proj = try gpa.alloc(f32, h * h);
    defer gpa.free(hc_proj);
    @memset(hc_proj, 0.0);
    for (0..h) |d| hc_proj[d * h + d] = 0.25;

    // Steps 0..3 have positive channel-7 drive (high confidence); steps 4..6 have strong negative channel-7 drive (truncated!).
    var mask_embed: [DFLY2_BLOCK_SIZE * h]f32 = @splat(0.0);
    for (0..DFLY2_BLOCK_SIZE) |pos| {
        mask_embed[pos * h + (pos % 6)] = 1.5;
        mask_embed[pos * h + 7] = if (pos < 4) 3.0 else -6.0;
    }

    var tok_embed: [v * h]f32 = @splat(0.0);
    for (0..v) |t| tok_embed[t * h + (t % h)] = 0.5;

    var conf_w: [h]f32 = @splat(0.0);
    conf_w[7] = 2.0;

    var lm_head: [v * h]f32 = @splat(0.0);
    for (0..v) |t| lm_head[t * h + (t % 6)] = 2.0;

    const weights: DFly2Weights = .{
        .mtp_fusion_proj = mtp_proj,
        .hidden_correction_proj = hc_proj,
        .mask_embed = &mask_embed,
        .tok_embed = &tok_embed,
        .layers = bbuf.layers,
        .confidence_weight = &conf_w,
        .confidence_bias = 0.0,
        .lm_head = .{ .dense_f32 = &lm_head },
    };

    var proposer = try DFly2Proposer.init(cfg, weights, &ws);
    const kv_dim = cfg.kvDim();
    var k_bank: [kv_len * 8]f32 = @splat(0.1);
    var v_bank: [kv_len * 8]f32 = @splat(0.2);
    for (0..kv_len) |t| {
        k_bank[t * kv_dim + (t % kv_dim)] = 0.5;
        v_bank[t * kv_dim + (t % kv_dim)] = 0.3;
    }
    const kv_view: TargetKvLayerView = .{
        .k = k_bank[0 .. kv_len * kv_dim],
        .v = v_bank[0 .. kv_len * kv_dim],
        .kv_len = kv_len,
    };
    const kv_layers = [_]TargetKvLayerView{ kv_view, kv_view, kv_view, kv_view, kv_view };
    const tgt_h: [h]f32 = @splat(0.1);

    var tree: CandidateTree = undefined;
    const res = try proposer.propose(1, &tgt_h, &kv_layers, null, &tree);

    // Confidence head must truncate steps 4..6 (< 0.5) while keeping steps 0..3 (>= 0.5)!
    try std.testing.expectEqual(@as(u8, 4), res.active_steps);
    try std.testing.expectEqual(@as(u8, 3), res.truncated_tail_steps);
    try std.testing.expectEqual(@as(u16, 5), res.tree_nodes); // root + 4 active steps
    try std.testing.expectEqual(@as(u16, 4), res.max_depth);
    try std.testing.expect(proposer.stepConfidence(0) >= 0.5);
    try std.testing.expect(proposer.stepConfidence(3) >= 0.5);
    try std.testing.expect(proposer.stepConfidence(4) < 0.5);
    try verifyTreeInvariants(CandidateTree.capacity, &tree);

    // Verify lossless D9 greedy acceptance on the truncated 4-step DFly2 draft tree.
    var target_argmax: [CandidateTree.capacity]u32 = @splat(15);
    for (0..tree.count - 1) |u| {
        target_argmax[u] = tree.nodes[u + 1].token_id;
    }
    target_argmax[tree.count - 1] = 11;

    var acc_nodes: [8]u16 = undefined;
    var emitted: [8]u32 = undefined;
    const acc = try accept_mod.verifyGreedy(CandidateTree.capacity, &tree, target_argmax[0..tree.count], &acc_nodes, &emitted);
    try std.testing.expectEqual(@as(u16, 4), acc.draft_accepted);
    try std.testing.expectEqual(@as(u32, 11), acc.bonus_token);

    // Lowering threshold to 0.0 keeps all block_size = 7 steps (8 nodes total).
    proposer.config.confidence_threshold = 0.0;
    const full_res = try proposer.propose(1, &tgt_h, &kv_layers, null, &tree);
    try std.testing.expectEqual(@as(u8, DFLY2_BLOCK_SIZE), full_res.active_steps);
    try std.testing.expectEqual(@as(u8, 0), full_res.truncated_tail_steps);
    try std.testing.expectEqual(@as(u16, DFLY2_BLOCK_SIZE + 1), full_res.tree_nodes);

    // Wire DFly2 into DraftSource, L1 SAM seeding, and D7 expert-byte-budget pruning.
    const dfly_src: DraftSource = .{ .dfly2_5l = &proposer };
    try std.testing.expectEqual(DraftSourceKind.dfly2_5l, std.meta.activeTag(dfly_src));

    var sam = try sam_mod.SuffixAutomaton.initAlloc(gpa, 64);
    defer sam.deinit(gpa);
    try sam.appendSlice(&.{ 4, 5, 6, 7, 8, 4, 5, 6, 7 });
    const l1_res = try proposer.propose(7, &tgt_h, &kv_layers, &sam, &tree);
    try std.testing.expect(l1_res.l1_match_len >= 4);
    try std.testing.expect(l1_res.l1_is_seed);
    try verifyTreeInvariants(CandidateTree.capacity, &tree);

    const prune_cost = tree_mod.PruneCostModel.uniformBeta(50.0, 4.0, 2048.0, 0.0, 100.0, 1, 16);
    var node_experts: [16]tree_mod.ExpertBitset = @splat(.{});
    for (0..tree.count) |ni| node_experts[ni].set(@intCast(ni % 4));
    const pruned = try proposer.pruneWithD7(
        CandidateTree.capacity,
        &tree,
        prune_cost,
        node_experts[0..tree.count],
        null,
        5,
    );
    try std.testing.expect(pruned.kept_nodes <= 5);
    try verifyTreeInvariants(CandidateTree.capacity, &tree);
}
