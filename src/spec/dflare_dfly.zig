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

/// Shared lightweight block-parallel transformer layer weights used by both
/// `DFlareProposer` (`3L`/`5L`) and `DFly2Proposer` (`5L`).
pub const TransformerBlockWeights = struct {
    in_norm_weight: ?[]const f32 = null,
    q_proj: []const f32, // [q_dim, hidden_size]
    k_proj: []const f32, // [kv_dim, hidden_size]
    v_proj: []const f32, // [kv_dim, hidden_size]
    o_proj: []const f32, // [hidden_size, q_dim]
    q_norm_weight: ?[]const f32 = null, // [head_dim]
    k_norm_weight: ?[]const f32 = null, // [head_dim]
    post_norm_weight: ?[]const f32 = null,
    mlp_gate_proj: []const f32, // [intermediate_size, hidden_size]
    mlp_up_proj: []const f32, // [intermediate_size, hidden_size]
    mlp_down_proj: []const f32, // [hidden_size, intermediate_size]

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
        if (self.q_proj.len != qh or self.k_proj.len != kvh or self.v_proj.len != kvh or self.o_proj.len != hq) {
            return error.InvalidWeights;
        }
        if (self.q_norm_weight) |w| {
            if (w.len != head_dim) return error.InvalidWeights;
        }
        if (self.k_norm_weight) |w| {
            if (w.len != head_dim) return error.InvalidWeights;
        }
        if (self.post_norm_weight) |w| {
            if (w.len != hidden_size) return error.InvalidWeights;
        }
        if (self.mlp_gate_proj.len != ih or self.mlp_up_proj.len != ih or self.mlp_down_proj.len != hi) {
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
};

pub const DFlareWeights = struct {
    /// Multi-layer target tap fusion `[hidden_size, num_target_taps * hidden_size]`.
    /// Optional when `num_target_taps == 1`.
    target_fusion_proj: ?[]const f32 = null,
    target_fusion_norm: ?[]const f32 = null,
    /// Mask embedding: either `[block_size, hidden_size]` or shared `[hidden_size]`.
    mask_embed: []const f32,
    /// Token embeddings `[vocab_size, hidden_size]` for sequential Markov conditioning.
    tok_embed: []const f32,
    /// `3L` / `5L` block-parallel transformer layers.
    layers: []const TransformerBlockWeights,
    final_norm: ?[]const f32 = null,
    /// Low-rank Markov head `markov_w1`: either `[markov_rank, 2 * hidden_size]`
    /// (acting on `[backbone_out[j] || tok_embed(prev_tok)]`) or `[markov_rank, hidden_size]`.
    markov_w1: []const f32,
    /// Optional recurrent transition `[markov_rank, markov_rank]` on `rnn_hidden`.
    markov_rnn_proj: ?[]const f32 = null,
    markov_b1: ?[]const f32 = null,
    /// Output bottleneck expansion `markov_w2`: `[hidden_size, markov_rank]`.
    markov_w2: []const f32,
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
        } else if (taps != 1) {
            return error.InvalidWeights;
        }
        if (self.target_fusion_norm) |fnw| {
            if (fnw.len != h) return error.InvalidWeights;
        }
        const mask_ok = self.mask_embed.len == h or
            self.mask_embed.len == (std.math.mul(usize, cfg.block_size, h) catch return error.Overflow);
        if (!mask_ok) return error.InvalidWeights;

        const vh = std.math.mul(usize, v, h) catch return error.Overflow;
        if (self.tok_embed.len != vh) return error.InvalidWeights;
        if (self.layers.len != cfg.num_layers) return error.InvalidWeights;
        for (self.layers) |layer| {
            try layer.validate(h, inter, q_dim, kv_dim, cfg.head_dim);
        }
        if (self.final_norm) |fnw| {
            if (fnw.len != h) return error.InvalidWeights;
        }

        const mr_h = std.math.mul(usize, mr, h) catch return error.Overflow;
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
        if (self.markov_w2.len != mr_h) return error.InvalidWeights;
        if (self.markov_norm) |mn| {
            if (mn.len != h) return error.InvalidWeights;
        }
        try validateLmHeadHook(self.lm_head, vh);
    }
};

/// Preallocated 16 KiB-aligned `WiredArena` workspace for `DFlareProposer`.
pub const DFlareWorkspace = struct {
    arena: arena_mod.WiredArena,
    config: DFlareConfig,
    taps_flat: []f32,
    ctx_hidden: []f32,
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
    node_rnn_hidden: []f32,
    markov_u: []f32,
    markov_concat: []f32,
    head_hidden: []f32,
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

        const chunks = [_]usize{
            try alignUp64((taps * h) * @sizeOf(f32)), // taps_flat
            try alignUp64(h * @sizeOf(f32)), // ctx_hidden
            try alignUp64((w * h) * @sizeOf(f32)), // slot_hiddens
            try alignUp64((w * h) * @sizeOf(f32)), // norm_buf
            try alignUp64((w * q_dim) * @sizeOf(f32)), // q_buf
            try alignUp64(((w + 1) * kv_dim) * @sizeOf(f32)), // k_buf (ctx + block)
            try alignUp64(((w + 1) * kv_dim) * @sizeOf(f32)), // v_buf
            try alignUp64((w * q_dim) * @sizeOf(f32)), // attn_out
            try alignUp64((w + 1) * @sizeOf(f32)), // attn_scores
            try alignUp64((w * inter) * @sizeOf(f32)), // mlp_gate
            try alignUp64((w * inter) * @sizeOf(f32)), // mlp_up
            try alignUp64((w * h) * @sizeOf(f32)), // mlp_down
            try alignUp64((max_nodes * mr) * @sizeOf(f32)), // node_rnn_hidden
            try alignUp64(mr * @sizeOf(f32)), // markov_u
            try alignUp64((2 * h) * @sizeOf(f32)), // markov_concat
            try alignUp64(h * @sizeOf(f32)), // head_hidden
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

        const taps_flat = try allocArenaSlice(f32, &mem_arena, taps * h);
        const ctx_hidden = try allocArenaSlice(f32, &mem_arena, h);
        const slot_hiddens = try allocArenaSlice(f32, &mem_arena, w * h);
        const norm_buf = try allocArenaSlice(f32, &mem_arena, w * h);
        const q_buf = try allocArenaSlice(f32, &mem_arena, w * q_dim);
        const k_buf = try allocArenaSlice(f32, &mem_arena, (w + 1) * kv_dim);
        const v_buf = try allocArenaSlice(f32, &mem_arena, (w + 1) * kv_dim);
        const attn_out = try allocArenaSlice(f32, &mem_arena, w * q_dim);
        const attn_scores = try allocArenaSlice(f32, &mem_arena, w + 1);
        const mlp_gate = try allocArenaSlice(f32, &mem_arena, w * inter);
        const mlp_up = try allocArenaSlice(f32, &mem_arena, w * inter);
        const mlp_down = try allocArenaSlice(f32, &mem_arena, w * h);
        const node_rnn_hidden = try allocArenaSlice(f32, &mem_arena, max_nodes * mr);
        const markov_u = try allocArenaSlice(f32, &mem_arena, mr);
        const markov_concat = try allocArenaSlice(f32, &mem_arena, 2 * h);
        const head_hidden = try allocArenaSlice(f32, &mem_arena, h);
        const logits_buf = try allocArenaSlice(f32, &mem_arena, v);
        const topk_tokens = try allocArenaSlice(u32, &mem_arena, cfg.top_k_per_pos);
        const topk_probs = try allocArenaSlice(f32, &mem_arena, cfg.top_k_per_pos);
        const node_log_probs = try allocArenaSlice(f32, &mem_arena, max_nodes);
        const l1_draft_buf = try allocArenaSlice(u32, &mem_arena, MAX_L1_DRAFT_LEN);

        @memset(node_log_probs, 0.0);
        @memset(node_rnn_hidden, 0.0);

        return .{
            .arena = mem_arena,
            .config = cfg,
            .taps_flat = taps_flat,
            .ctx_hidden = ctx_hidden,
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
            .node_rnn_hidden = node_rnn_hidden,
            .markov_u = markov_u,
            .markov_concat = markov_concat,
            .head_hidden = head_hidden,
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
        self.node_rnn_hidden = &.{};
        self.markov_u = &.{};
        self.markov_concat = &.{};
        self.head_hidden = &.{};
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
        const taps: usize = self.config.num_target_taps;
        const h: usize = self.config.hidden_size;
        const w: usize = self.config.block_size;
        if (target_hiddens.len != taps) return error.DimensionMismatch;

        for (target_hiddens, 0..) |tap, ti| {
            if (tap.len != h) return error.DimensionMismatch;
            @memcpy(self.workspace.taps_flat[ti * h ..][0..h], tap);
        }

        // 1. Fuse multi-layer target hidden states into `ctx_hidden`.
        if (self.weights.target_fusion_proj) |fp| {
            matVecF32(fp, self.workspace.taps_flat[0 .. taps * h], h, taps * h, self.workspace.ctx_hidden[0..h]);
        } else {
            @memcpy(self.workspace.ctx_hidden[0..h], self.workspace.taps_flat[0..h]);
        }
        if (self.weights.target_fusion_norm) |fnw| {
            applyRmsNormInPlace(self.workspace.ctx_hidden[0..h], fnw, self.config.rms_eps);
        }

        // 2. Seed positions `0..block_size-1` with `ctx_hidden + mask_embed`.
        const per_pos_mask = self.weights.mask_embed.len == w * h;
        for (0..w) |pos| {
            const m_slice = if (per_pos_mask)
                self.weights.mask_embed[pos * h ..][0..h]
            else
                self.weights.mask_embed[0..h];
            const dst = self.workspace.slot_hiddens[pos * h ..][0..h];
            for (0..h) |d| {
                dst[d] = self.workspace.ctx_hidden[d] + m_slice[d];
            }
        }

        // 3. Execute `3L`/`5L` block-parallel transformer layers with `ctx_hidden`
        //    projected as single-token context KV prepended to self-attention.
        const kv_dim: usize = self.config.kvDim();
        for (self.weights.layers) |layer| {
            matVecF32(layer.k_proj, self.workspace.ctx_hidden[0..h], kv_dim, h, self.workspace.k_buf[0..kv_dim]);
            matVecF32(layer.v_proj, self.workspace.ctx_hidden[0..h], kv_dim, h, self.workspace.v_buf[0..kv_dim]);
            if (layer.k_norm_weight) |kn| {
                applyHeadRmsNorm(self.workspace.k_buf[0..kv_dim], self.config.num_key_value_heads, self.config.head_dim, kn, self.config.rms_eps);
            }
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
            );
        }

        if (self.weights.final_norm) |fnw| {
            for (0..w) |pos| {
                applyRmsNormInPlace(self.workspace.slot_hiddens[pos * h ..][0..h], fnw, self.config.rms_eps);
            }
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
        const tok_e = self.weights.tok_embed[@as(usize, prev_token) * h ..][0..h];

        if (self.weights.markov_w1.len == mr * 2 * h) {
            @memcpy(self.workspace.markov_concat[0..h], b_j);
            @memcpy(self.workspace.markov_concat[h .. 2 * h], tok_e);
            matVecF32(self.weights.markov_w1, self.workspace.markov_concat[0 .. 2 * h], mr, 2 * h, self.workspace.markov_u[0..mr]);
        } else {
            for (0..h) |d| {
                self.workspace.markov_concat[d] = b_j[d] + tok_e[d];
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
        try self.runBlockBackboneOnce(target_hiddens);

        // 2. Decode sequential draft steps j = 0..block_size-1 through the low-rank Markov head.
        var frontier_cur: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
        var frontier_next: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
        var cur_len: usize = 1;
        frontier_cur[0] = 0;

        const inv_temp: f32 = 1.0 / self.config.temperature;
        var next_rnn_scratch: [4096]f32 = undefined;
        if (mr > next_rnn_scratch.len) return error.BufferTooSmall;
        const next_rnn = next_rnn_scratch[0..mr];

        for (0..w) |pos_usize| {
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

/// Startup-loaded standalone `arch = dflash` `.amf` bundle owning decoded FP32
/// draft weights and a 16 KiB-aligned `DFlareWorkspace`.
pub const LoadedDFlareBundle = struct {
    owned_f32_bufs: [][]f32,
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
        var amf_file = try amf.AmfFile.open(allocator, path);
        defer amf_file.close(allocator);
        return loadFromAmf(allocator, &amf_file, cfg_in, lm_head_override, budget_override);
    }

    pub fn loadFromAmf(
        allocator: std.mem.Allocator,
        amf_file: *const amf.AmfFile,
        cfg_in: DFlareConfig,
        lm_head_override: ?LmHeadHook,
        budget_override: ?budget_mod.Budget,
    ) LoadError!*LoadedDFlareBundle {
        var cfg = cfg_in;
        cfg.variant = .dflash_standalone;
        try cfg.validate();

        const h: usize = cfg.hidden_size;
        const inter: usize = cfg.intermediate_size;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();
        const mr: usize = cfg.markov_rank;
        const v: usize = cfg.vocab_size;

        // Track up to (8 layers * 11 tensors) + 10 top-level tensors = 98 slices.
        var pool: [128][]f32 = undefined;
        var pool_len: usize = 0;
        errdefer {
            for (0..pool_len) |i| allocator.free(pool[i]);
        }

        const fc = try loadAmfTensorAsF32Opt(allocator, amf_file, &.{ "dflare.fc.weight", "fc.weight" }, cfg.num_target_taps * h * h, &pool, &pool_len);
        const fc_norm = try loadAmfTensorAsF32Opt(allocator, amf_file, &.{ "dflare.fc_norm.weight", "fc_norm.weight" }, h, &pool, &pool_len);

        const mask_embed = try loadAmfTensorAnyLen(allocator, amf_file, &.{ "dflare.mask_embed.weight", "dflare.mask_embed", "mask_embed" }, &pool, &pool_len);
        const tok_embed = try loadAmfTensorAsF32(allocator, amf_file, &.{ "dflare.tok_embed.weight", "token_embd.weight", "embed_tokens.weight" }, v * h, &pool, &pool_len);
        const final_norm = try loadAmfTensorAsF32Opt(allocator, amf_file, &.{ "dflare.norm.weight", "output_norm.weight", "norm.weight" }, h, &pool, &pool_len);

        const layers = try allocator.alloc(TransformerBlockWeights, cfg.num_layers);
        errdefer allocator.free(layers);

        var name_a: [64]u8 = undefined;
        var name_b: [64]u8 = undefined;
        for (0..cfg.num_layers) |l| {
            const in_norm = try loadLayerTensorOpt(allocator, amf_file, l, "input_layernorm.weight", "attn_norm.weight", h, &name_a, &name_b, &pool, &pool_len);
            const q_proj = try loadLayerTensor(allocator, amf_file, l, "attn.q_proj.weight", "attn_q.weight", q_dim * h, &name_a, &name_b, &pool, &pool_len);
            const k_proj = try loadLayerTensor(allocator, amf_file, l, "attn.k_proj.weight", "attn_k.weight", kv_dim * h, &name_a, &name_b, &pool, &pool_len);
            const v_proj = try loadLayerTensor(allocator, amf_file, l, "attn.v_proj.weight", "attn_v.weight", kv_dim * h, &name_a, &name_b, &pool, &pool_len);
            const o_proj = try loadLayerTensor(allocator, amf_file, l, "attn.o_proj.weight", "attn_output.weight", h * q_dim, &name_a, &name_b, &pool, &pool_len);
            const q_norm = try loadLayerTensorOpt(allocator, amf_file, l, "attn.q_norm.weight", "attn_q_norm.weight", cfg.head_dim, &name_a, &name_b, &pool, &pool_len);
            const k_norm = try loadLayerTensorOpt(allocator, amf_file, l, "attn.k_norm.weight", "attn_k_norm.weight", cfg.head_dim, &name_a, &name_b, &pool, &pool_len);
            const post_norm = try loadLayerTensorOpt(allocator, amf_file, l, "post_norm.weight", "ffn_norm.weight", h, &name_a, &name_b, &pool, &pool_len);
            const gate_p = try loadLayerTensor(allocator, amf_file, l, "mlp.gate_proj.weight", "ffn_gate.weight", inter * h, &name_a, &name_b, &pool, &pool_len);
            const up_p = try loadLayerTensor(allocator, amf_file, l, "mlp.up_proj.weight", "ffn_up.weight", inter * h, &name_a, &name_b, &pool, &pool_len);
            const down_p = try loadLayerTensor(allocator, amf_file, l, "mlp.down_proj.weight", "ffn_down.weight", h * inter, &name_a, &name_b, &pool, &pool_len);

            layers[l] = .{
                .in_norm_weight = in_norm,
                .q_proj = q_proj,
                .k_proj = k_proj,
                .v_proj = v_proj,
                .o_proj = o_proj,
                .q_norm_weight = q_norm,
                .k_norm_weight = k_norm,
                .post_norm_weight = post_norm,
                .mlp_gate_proj = gate_p,
                .mlp_up_proj = up_p,
                .mlp_down_proj = down_p,
            };
        }

        const markov_w1 = try loadAmfTensorAnyLen(allocator, amf_file, &.{ "dflare.markov_w1.weight", "markov_w1.weight" }, &pool, &pool_len);
        const markov_rnn = try loadAmfTensorAsF32Opt(allocator, amf_file, &.{ "dflare.rnn_proj.weight", "markov_rnn_proj.weight" }, mr * mr, &pool, &pool_len);
        const markov_b1 = try loadAmfTensorAsF32Opt(allocator, amf_file, &.{ "dflare.markov_b1", "markov_b1" }, mr, &pool, &pool_len);
        const markov_w2 = try loadAmfTensorAsF32(allocator, amf_file, &.{ "dflare.markov_w2.weight", "markov_w2.weight" }, h * mr, &pool, &pool_len);
        const markov_norm = try loadAmfTensorAsF32Opt(allocator, amf_file, &.{ "dflare.markov_norm.weight", "markov_norm.weight" }, h, &pool, &pool_len);

        const lm_head: LmHeadHook = lm_head_override orelse blk: {
            const lm_w = try loadAmfTensorAsF32(allocator, amf_file, &.{ "dflare.lm_head.weight", "output.weight", "lm_head.weight" }, v * h, &pool, &pool_len);
            break :blk .{ .dense_f32 = lm_w };
        };

        const owned = try allocator.alloc([]f32, pool_len);
        errdefer allocator.free(owned);
        @memcpy(owned, pool[0..pool_len]);

        var ws = try DFlareWorkspace.init(cfg, budget_override);
        errdefer ws.deinit();

        const weights: DFlareWeights = .{
            .target_fusion_proj = fc,
            .target_fusion_norm = fc_norm,
            .mask_embed = mask_embed,
            .tok_embed = tok_embed,
            .layers = layers,
            .final_norm = final_norm,
            .markov_w1 = markov_w1,
            .markov_rnn_proj = markov_rnn,
            .markov_b1 = markov_b1,
            .markov_w2 = markov_w2,
            .markov_norm = markov_norm,
            .lm_head = lm_head,
        };

        const bundle = try allocator.create(LoadedDFlareBundle);
        errdefer allocator.destroy(bundle);

        bundle.* = .{
            .owned_f32_bufs = owned,
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
    /// Optional residual `hidden_correction` projection `[hidden_size, hidden_size]`.
    hidden_correction_proj: ?[]const f32 = null,
    hidden_correction_norm: ?[]const f32 = null,
    mask_embed: []const f32,
    tok_embed: []const f32,
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
        if (self.tok_embed.len != vh) return error.InvalidWeights;
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
        if (dfly_target_kv_layers.len != self.config.num_target_kv_layers) return error.DimensionMismatch;
        for (dfly_target_kv_layers) |kv| {
            if (kv.kv_len == 0 or kv.kv_len > self.config.max_target_kv_len) return error.DimensionMismatch;
            if (kv.k.len < kv.kv_len * kv_dim or kv.v.len < kv.kv_len * kv_dim) return error.BufferTooSmall;
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
        const root_e = self.weights.tok_embed[@as(usize, root_token) * h ..][0..h];
        @memcpy(self.workspace.mtp_concat[0..h], root_e);
        @memcpy(self.workspace.mtp_concat[h .. 2 * h], last_target_hidden);
        applyRmsNormInPlace(self.workspace.mtp_concat[0..h], self.weights.mtp_enorm_weight, self.config.rms_eps);
        applyRmsNormInPlace(self.workspace.mtp_concat[h .. 2 * h], self.weights.mtp_hnorm_weight, self.config.rms_eps);
        matVecF32(self.weights.mtp_fusion_proj, self.workspace.mtp_concat[0 .. 2 * h], h, 2 * h, self.workspace.fused_hidden[0..h]);

        // 2. `hidden_correction`: optional residual correction before 5L DFly2 block layers.
        if (self.weights.hidden_correction_proj) |hc| {
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
            const kv_view = dfly_target_kv_layers[li % dfly_target_kv_layers.len];
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
) Error!void {
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

        matVecF32(layer.q_proj, dst, q_dim, hidden_size, q_buf[pos * q_dim ..][0..q_dim]);
        matVecF32(layer.k_proj, dst, kv_dim, hidden_size, k_buf[pos * kv_dim ..][0..kv_dim]);
        matVecF32(layer.v_proj, dst, kv_dim, hidden_size, v_buf[pos * kv_dim ..][0..kv_dim]);

        if (layer.q_norm_weight) |qn| {
            applyHeadRmsNorm(q_buf[pos * q_dim ..][0..q_dim], num_heads, head_dim, qn, rms_eps);
        }
        if (layer.k_norm_weight) |kn| {
            applyHeadRmsNorm(k_buf[pos * kv_dim ..][0..kv_dim], num_kv_heads, head_dim, kn, rms_eps);
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

            for (0..ctx_len) |kp| {
                const k_head = ctx_k[kp * kv_dim + kvh * hd ..][0..hd];
                var dot: f32 = 0.0;
                for (0..hd) |d| dot += q_head[d] * k_head[d];
                const sc = dot * inv_sqrt_hd;
                attn_scores[kp] = sc;
                if (sc > max_s) max_s = sc;
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
            for (0..ctx_len) |kp| {
                const p = attn_scores[kp] * inv_sum;
                const v_head = ctx_v[kp * kv_dim + kvh * hd ..][0..hd];
                for (0..hd) |d| dst_head[d] += p * v_head[d];
            }
            for (0..seq_len) |kp| {
                const p = attn_scores[ctx_len + kp] * inv_sum;
                const v_head = v_buf[kp * kv_dim + kvh * hd ..][0..hd];
                for (0..hd) |d| dst_head[d] += p * v_head[d];
            }
        }

        // Residual connection after attention projection.
        const h_row = hiddens[q_pos * hidden_size ..][0..hidden_size];
        matVecAccumF32(layer.o_proj, out_row, hidden_size, q_dim, h_row);
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

fn readEntryAsF32(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    entry: *const amf.TensorEntry,
) LoadedDFlareBundle.LoadError![]f32 {
    const dt = try entry.dataType();
    if (dt != .f32 and dt != .f16 and dt != .bf16) return error.InvalidWeights;
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
    switch (dt) {
        .f32 => {
            if (entry.data_bytes < elems * 4) {
                allocator.free(out);
                return error.InvalidWeights;
            }
            for (0..elems) |i| {
                out[i] = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
            }
        },
        .f16 => {
            if (entry.data_bytes < elems * 2) {
                allocator.free(out);
                return error.InvalidWeights;
            }
            for (0..elems) |i| {
                const bits = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
                out[i] = @floatCast(@as(f16, @bitCast(bits)));
            }
        },
        .bf16 => {
            if (entry.data_bytes < elems * 2) {
                allocator.free(out);
                return error.InvalidWeights;
            }
            for (0..elems) |i| {
                out[i] = quant.bf16ToF32(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
            }
        },
        else => unreachable,
    }
    return out;
}

fn loadAmfTensorAsF32(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    names: []const []const u8,
    expected_elems: usize,
    pool: *[128][]f32,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError![]f32 {
    const entry = findAnyEntry(amf_file, names) orelse return error.MissingTensor;
    const vec = try readEntryAsF32(allocator, amf_file, entry);
    errdefer allocator.free(vec);
    if (vec.len != expected_elems) return error.DimensionMismatch;
    if (pool_len.* >= pool.len) return error.Overflow;
    pool[pool_len.*] = vec;
    pool_len.* += 1;
    return vec;
}

fn loadAmfTensorAsF32Opt(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    names: []const []const u8,
    expected_elems: usize,
    pool: *[128][]f32,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError!?[]f32 {
    const entry = findAnyEntry(amf_file, names) orelse return null;
    const vec = try readEntryAsF32(allocator, amf_file, entry);
    errdefer allocator.free(vec);
    if (vec.len != expected_elems) return error.DimensionMismatch;
    if (pool_len.* >= pool.len) return error.Overflow;
    pool[pool_len.*] = vec;
    pool_len.* += 1;
    return vec;
}

fn loadAmfTensorAnyLen(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    names: []const []const u8,
    pool: *[128][]f32,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError![]f32 {
    const entry = findAnyEntry(amf_file, names) orelse return error.MissingTensor;
    const vec = try readEntryAsF32(allocator, amf_file, entry);
    errdefer allocator.free(vec);
    if (pool_len.* >= pool.len) return error.Overflow;
    pool[pool_len.*] = vec;
    pool_len.* += 1;
    return vec;
}

fn loadLayerTensor(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    layer: usize,
    suffix_a: []const u8,
    suffix_b: []const u8,
    expected_elems: usize,
    buf_a: *[64]u8,
    buf_b: *[64]u8,
    pool: *[128][]f32,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError![]f32 {
    const nm_a = std.fmt.bufPrint(buf_a, "layers.{d}.{s}", .{ layer, suffix_a }) catch return error.Overflow;
    const nm_b = std.fmt.bufPrint(buf_b, "blk.{d}.{s}", .{ layer, suffix_b }) catch return error.Overflow;
    return loadAmfTensorAsF32(allocator, amf_file, &.{ nm_a, nm_b }, expected_elems, pool, pool_len);
}

fn loadLayerTensorOpt(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    layer: usize,
    suffix_a: []const u8,
    suffix_b: []const u8,
    expected_elems: usize,
    buf_a: *[64]u8,
    buf_b: *[64]u8,
    pool: *[128][]f32,
    pool_len: *usize,
) LoadedDFlareBundle.LoadError!?[]f32 {
    const nm_a = std.fmt.bufPrint(buf_a, "layers.{d}.{s}", .{ layer, suffix_a }) catch return error.Overflow;
    const nm_b = std.fmt.bufPrint(buf_b, "blk.{d}.{s}", .{ layer, suffix_b }) catch return error.Overflow;
    return loadAmfTensorAsF32Opt(allocator, amf_file, &.{ nm_a, nm_b }, expected_elems, pool, pool_len);
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
