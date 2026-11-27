//! D10 DSpark Semi-Autoregressive Speculative Proposer (ADR-0015, ADR-0005).
//!
//! Predicts `block_width` future token distributions in ONE parallel forward
//! step conditioned on `(encoder_summary_vec, last_decoder_hidden)`, merges
//! optional L1 Suffix Automaton (`sam.zig`) prefix seeds, and constructs a
//! valid `CandidateTree` (`tree.zig`, `G5` 64-bit ancestor masks) ready for
//! single-pass lossless `D9` verification (`accept.zig`).
//!
//! Engineering invariants:
//! - Zero dynamic heap allocation on the propose hot path: all intermediate
//!   activations, logits, top-k tables, and L1 buffers live inside a single
//!   16 KiB-aligned `WiredArena` (`DSparkWorkspace`).
//! - Supports compact CPU `fp32`/NEON/SME2-friendly projections (`dense_f32`,
//!   `dense_f16`, `dense_bf16`) and a zero-copy callback hook (`LogitCallback`)
//!   for GPU target `lm_head` logits.
//! - Deterministic candidate selection and tie-breaking (higher logit wins;
//!   exact ties break by ascending `token_id`).

const std = @import("std");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const quant = @import("../quant/spec.zig");
const quant_formats = @import("../quant/formats.zig");
const amf = @import("../format/amf.zig");
const sme = @import("../cpu/sme.zig");
const tree_mod = @import("tree.zig");
const sam_mod = @import("sam.zig");
const grammar_mod = @import("grammar.zig");
const accept_mod = @import("accept.zig");

/// Maximum semi-autoregressive parallel block width (`W`).
pub const MAX_BLOCK_WIDTH: u8 = 16;

/// Maximum top-k branching factor per parallel position.
pub const MAX_TOP_K_PER_POS: u8 = 8;

/// Maximum speculative tree node capacity (`G5` `u64` ancestor bitmask ceiling).
pub const MAX_TREE_NODES: u8 = @intCast(tree_mod.MAX_SUPPORTED_NODES);

/// Maximum L1 Suffix Automaton continuation tokens queried per propose round.
pub const MAX_L1_DRAFT_LEN: usize = 32;

/// Maximum stages in a real-HF `3L DSpark` (`mtp.0..2.*`) proposer.
pub const MAX_DSPARK_3L_LAYERS: usize = 4;

/// Maximum target hidden tap vectors fused by `mtp.0.main_proj`.
pub const MAX_DSPARK_TARGET_TAPS: usize = 4;

/// Maximum mHC parallel residual streams (`hc_mult = 4` on DeepSeek-V4.1).
pub const MAX_HC_MULT: usize = 8;

/// Maximum routed MoE top-k experts per `DSpark` stage (`top_k = 3` on DeepSeek-V4.1).
pub const MAX_DSPARK_ROUTED_TOP_K: usize = 16;

/// Default sliding-window KV capacity per `DSparkAttention` stage.
pub const DEFAULT_DSPARK_WINDOW: u32 = 128;

const CACHELINE_ALIGN: usize = 64;

/// Default 32-node speculative candidate tree (`G5` `u64` ancestor masks).
pub const CandidateTree = tree_mod.DefaultTree;

pub const Error = arena_mod.WiredArena.InitError ||
    arena_mod.WiredArena.AllocError ||
    sysinfo.Error ||
    error{
        InvalidConfig,
        InvalidWeights,
        DimensionMismatch,
        WorkspaceMismatch,
        CallbackFailed,
        InvalidLogits,
        Overflow,
    };

/// Configuration for the DSpark semi-autoregressive proposer (`D10`).
pub const DSparkConfig = struct {
    /// Number of future token positions predicted in parallel per step.
    block_width: u8 = 4,
    /// Number of top candidates expanded per parallel position.
    top_k_per_pos: u8 = 2,
    /// Maximum total nodes in the emitted `CandidateTree` (including root `0`).
    max_tree_nodes: u8 = 32,
    /// Hidden vector width of encoder summary and decoder states.
    hidden_size: u32,
    /// Target vocabulary size.
    vocab_size: u32,
    /// RMSNorm epsilon applied before `lm_head` projection.
    rms_eps: f32 = 1e-6,
    /// Softmax temperature used to score draft candidate probabilities (`> 0`).
    temperature: f32 = 1.0,
    /// Step probability assigned to high-confidence L1 SAM seeds (`match_len >= 4`).
    l1_seed_step_prob: f32 = 0.90,
    /// Step probability assigned to shorter L1 SAM branches (`1 <= match_len < 4`).
    l1_branch_step_prob: f32 = 0.55,
    /// Maximum continuation tokens requested from L1 SAM per round.
    l1_max_draft_len: u8 = 8,

    pub fn validate(self: DSparkConfig) Error!void {
        if (self.block_width == 0 or self.block_width > MAX_BLOCK_WIDTH) {
            return error.InvalidConfig;
        }
        if (self.top_k_per_pos == 0 or self.top_k_per_pos > MAX_TOP_K_PER_POS) {
            return error.InvalidConfig;
        }
        if (self.max_tree_nodes < 2 or self.max_tree_nodes > MAX_TREE_NODES) {
            return error.InvalidConfig;
        }
        if (self.hidden_size == 0 or self.vocab_size == 0) {
            return error.InvalidConfig;
        }
        if (self.top_k_per_pos > self.vocab_size) {
            return error.InvalidConfig;
        }
        if (!std.math.isFinite(self.rms_eps) or self.rms_eps <= 0.0) {
            return error.InvalidConfig;
        }
        if (!std.math.isFinite(self.temperature) or self.temperature <= 0.0) {
            return error.InvalidConfig;
        }
        if (self.l1_max_draft_len > MAX_L1_DRAFT_LEN) {
            return error.InvalidConfig;
        }
    }
};

/// Callback signature allowing GPU target `lm_head` (or tied Metal GEMM) to
/// populate `[block_width, vocab_size]` FP32 logits directly from the parallel
/// `[block_width, hidden_size]` DSpark hidden states with zero copies.
pub const LogitCallback = struct {
    ctx: ?*anyopaque = null,
    eval_fn: *const fn (
        ctx: ?*anyopaque,
        slot_hiddens: []const f32,
        block_width: u8,
        hidden_size: u32,
        vocab_size: u32,
        out_logits: []f32,
    ) Error!void,
};

/// Output vocabulary projection interface (`lm_head`): supports compact CPU
/// weights (`f32`, `f16`, `bf16`) or a GPU-backed callback.
pub const LmHeadHook = union(enum) {
    /// Row-major `[vocab_size, hidden_size]` FP32 weights.
    dense_f32: []const f32,
    /// Row-major `[vocab_size, hidden_size]` FP16 weights.
    dense_f16: []const f16,
    /// Row-major `[vocab_size, hidden_size]` BF16 weights (raw IEEE u16 bits).
    dense_bf16: []const u16,
    /// External (e.g. Metal GPU / tied target) batched logits callback.
    callback: LogitCallback,
};

/// Lightweight position-conditioned parallel weights for DSpark (`D10`).
pub const DSparkWeights = struct {
    /// Optional encoder-summary conditioning projection `[hidden_size, hidden_size]`.
    /// When `null`, `encoder_summary_vec` is added directly when supplied.
    enc_proj: ?[]const f32 = null,
    /// Optional decoder-state conditioning projection `[hidden_size, hidden_size]`.
    /// When `null`, `last_decoder_hidden` is passed through directly.
    dec_proj: ?[]const f32 = null,
    /// Parallel per-slot projections `[block_width, hidden_size, hidden_size]`.
    slot_projs: []const f32,
    /// Parallel per-slot learned position embeddings `[block_width, hidden_size]`.
    slot_embed: []const f32,
    /// Optional shared RMSNorm scale `[hidden_size]` applied before `lm_head`.
    slot_norm_weight: ?[]const f32 = null,
    /// Shared or tied vocabulary projection hook.
    lm_head: LmHeadHook,

    pub fn validate(self: DSparkWeights, cfg: DSparkConfig) Error!void {
        const h: usize = cfg.hidden_size;
        const w: usize = cfg.block_width;
        const v: usize = cfg.vocab_size;

        const hh = std.math.mul(usize, h, h) catch return error.Overflow;
        const wh = std.math.mul(usize, w, h) catch return error.Overflow;
        const whh = std.math.mul(usize, w, hh) catch return error.Overflow;
        const vh = std.math.mul(usize, v, h) catch return error.Overflow;

        if (self.enc_proj) |ep| {
            if (ep.len != hh) return error.InvalidWeights;
        }
        if (self.dec_proj) |dp| {
            if (dp.len != hh) return error.InvalidWeights;
        }
        if (self.slot_projs.len != whh) return error.InvalidWeights;
        if (self.slot_embed.len != wh) return error.InvalidWeights;
        if (self.slot_norm_weight) |nw| {
            if (nw.len != h) return error.InvalidWeights;
        }
        switch (self.lm_head) {
            .dense_f32 => |m| if (m.len != vh) return error.InvalidWeights,
            .dense_f16 => |m| if (m.len != vh) return error.InvalidWeights,
            .dense_bf16 => |m| if (m.len != vh) return error.InvalidWeights,
            .callback => {},
        }
    }
};

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

/// Preallocated 16 KiB-aligned wired workspace for `DSparkProposer`.
///
/// Allocates every scratch tensor once in `init()`, guaranteeing strictly zero
/// dynamic heap allocations across all speculative drafting rounds.
pub const DSparkWorkspace = struct {
    arena: arena_mod.WiredArena,
    config: DSparkConfig,
    base_hidden: []f32,
    slot_input: []f32,
    slot_hiddens: []f32,
    slot_logits: []f32,
    topk_tokens: []u32,
    topk_probs: []f32,
    l1_draft_buf: []u32,

    /// Computes exact 16 KiB-aligned `WiredArena` capacity needed for `cfg`.
    pub fn requiredBytes(cfg: DSparkConfig) Error!usize {
        try cfg.validate();
        const h: usize = cfg.hidden_size;
        const w: usize = cfg.block_width;
        const v: usize = cfg.vocab_size;
        const k: usize = cfg.top_k_per_pos;

        const b_base = try alignUp64(std.math.mul(usize, h, @sizeOf(f32)) catch return error.Overflow);
        const b_slot_in = b_base;
        const wh = std.math.mul(usize, w, h) catch return error.Overflow;
        const b_slot_h = try alignUp64(std.math.mul(usize, wh, @sizeOf(f32)) catch return error.Overflow);
        const wv = std.math.mul(usize, w, v) catch return error.Overflow;
        const b_logits = try alignUp64(std.math.mul(usize, wv, @sizeOf(f32)) catch return error.Overflow);
        const wk = std.math.mul(usize, w, k) catch return error.Overflow;
        const b_topk_tok = try alignUp64(std.math.mul(usize, wk, @sizeOf(u32)) catch return error.Overflow);
        const b_topk_prob = try alignUp64(std.math.mul(usize, wk, @sizeOf(f32)) catch return error.Overflow);
        const b_l1 = try alignUp64(MAX_L1_DRAFT_LEN * @sizeOf(u32));

        var total: usize = 0;
        for ([_]usize{ b_base, b_slot_in, b_slot_h, b_logits, b_topk_tok, b_topk_prob, b_l1 }) |chunk| {
            total = std.math.add(usize, total, chunk) catch return error.Overflow;
        }
        return page.alignUp(total) catch return error.Overflow;
    }

    pub fn init(cfg: DSparkConfig, budget_override: ?budget_mod.Budget) Error!DSparkWorkspace {
        const cap_bytes = try requiredBytes(cfg);
        const wired_budget = budget_override orelse try defaultWiredBudget();
        try wired_budget.checkPlan(cap_bytes);

        var mem_arena = try arena_mod.WiredArena.init(cap_bytes, wired_budget);
        errdefer mem_arena.deinit();

        const h: usize = cfg.hidden_size;
        const w: usize = cfg.block_width;
        const v: usize = cfg.vocab_size;
        const k: usize = cfg.top_k_per_pos;

        const base_raw = try mem_arena.allocAligned(h * @sizeOf(f32), CACHELINE_ALIGN);
        const slot_in_raw = try mem_arena.allocAligned(h * @sizeOf(f32), CACHELINE_ALIGN);
        const slot_h_raw = try mem_arena.allocAligned(w * h * @sizeOf(f32), CACHELINE_ALIGN);
        const logits_raw = try mem_arena.allocAligned(w * v * @sizeOf(f32), CACHELINE_ALIGN);
        const topk_tok_raw = try mem_arena.allocAligned(w * k * @sizeOf(u32), CACHELINE_ALIGN);
        const topk_prob_raw = try mem_arena.allocAligned(w * k * @sizeOf(f32), CACHELINE_ALIGN);
        const l1_raw = try mem_arena.allocAligned(MAX_L1_DRAFT_LEN * @sizeOf(u32), CACHELINE_ALIGN);

        const base_hidden: []f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, base_raw)))[0..h];
        const slot_input: []f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, slot_in_raw)))[0..h];
        const slot_hiddens: []f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, slot_h_raw)))[0 .. w * h];
        const slot_logits: []f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, logits_raw)))[0 .. w * v];
        const topk_tokens: []u32 = @as([]u32, @alignCast(std.mem.bytesAsSlice(u32, topk_tok_raw)))[0 .. w * k];
        const topk_probs: []f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, topk_prob_raw)))[0 .. w * k];
        const l1_draft_buf: []u32 = @as([]u32, @alignCast(std.mem.bytesAsSlice(u32, l1_raw)))[0..MAX_L1_DRAFT_LEN];

        @memset(base_hidden, 0.0);
        @memset(slot_input, 0.0);
        @memset(slot_hiddens, 0.0);
        @memset(slot_logits, 0.0);
        @memset(topk_tokens, 0);
        @memset(topk_probs, 0.0);
        @memset(l1_draft_buf, 0);

        return .{
            .arena = mem_arena,
            .config = cfg,
            .base_hidden = base_hidden,
            .slot_input = slot_input,
            .slot_hiddens = slot_hiddens,
            .slot_logits = slot_logits,
            .topk_tokens = topk_tokens,
            .topk_probs = topk_probs,
            .l1_draft_buf = l1_draft_buf,
        };
    }

    pub fn deinit(self: *DSparkWorkspace) void {
        self.base_hidden = &.{};
        self.slot_input = &.{};
        self.slot_hiddens = &.{};
        self.slot_logits = &.{};
        self.topk_tokens = &.{};
        self.topk_probs = &.{};
        self.l1_draft_buf = &.{};
        self.arena.deinit();
    }

    pub fn totalWiredBytes(self: *const DSparkWorkspace) u64 {
        return self.arena.bytes().len;
    }

    /// Returns the sorted top-k candidate token IDs for parallel slot `pos`.
    pub inline fn slotTokens(self: *const DSparkWorkspace, pos: u8) []const u32 {
        const k: usize = self.config.top_k_per_pos;
        const start = @as(usize, pos) * k;
        return self.topk_tokens[start .. start + k];
    }

    /// Returns the sorted top-k candidate probabilities for parallel slot `pos`.
    pub inline fn slotProbs(self: *const DSparkWorkspace, pos: u8) []const f32 {
        const k: usize = self.config.top_k_per_pos;
        const start = @as(usize, pos) * k;
        return self.topk_probs[start .. start + k];
    }
};

/// Summary statistics from a single `DSparkProposer.propose` round.
pub const ProposeResult = struct {
    /// Total active nodes in `out_tree` (always `>= 1`, including root `0`).
    tree_nodes: u16,
    /// Deepest path depth in `out_tree`.
    max_depth: u16,
    /// L1 SAM matched suffix length (`0` if SAM was absent or missed).
    l1_match_len: u32,
    /// Number of newly allocated nodes contributed by L1 SAM.
    l1_inserted: u16,
    /// True iff L1 SAM produced a high-confidence seed chain (`match_len >= 4`).
    l1_is_seed: bool,
    /// Number of newly allocated nodes contributed by DSpark parallel slots.
    dspark_inserted: u16,
    /// Expected tokens `1 + sum_{v != 0} cum_prob[v]` of the resulting tree.
    expected_tokens: f32,
    /// Mean `DSparkConfidenceHead` sigmoid score across predicted block positions (`1.0` for legacy M7).
    mean_confidence: f32 = 1.0,
    /// Minimum `DSparkConfidenceHead` sigmoid score across predicted block positions (`1.0` for legacy M7).
    min_confidence: f32 = 1.0,
    /// Number of tail block positions gated out by `confidence_threshold` (`0` for legacy M7).
    gated_positions: u8 = 0,
    /// Number of low-rank `markov_head` evaluations executed in this round.
    markov_evals: u32 = 0,
    /// Number of full `[vocab, H]` `lm_head` sweeps avoided by `markov_head`.
    lm_head_sweeps_avoided: u32 = 0,
};

/// Semi-autoregressive DSpark speculative proposer (`D10`).
/// Supports both legacy M7 `dspark.slot.{s}.*` weights and real-HF `DeepSeek-V4.1`
/// `3L DSpark` (`mtp.0..2.*`) cold-tiered / quantized / `markov_head` + `confidence_head` runtimes.
pub const DSparkProposer = struct {
    config: DSparkConfig,
    weights: DSparkWeights,
    workspace: *DSparkWorkspace,
    rounds_proposed: u64 = 0,
    /// Optional real-HF `3L DSpark` (`mtp.0..2.*`) runtime. When non-null, all propose
    /// calls dispatch through the 3L mHC + DSparkAttention + MoE + Markov/Confidence path
    /// with zero hot-path heap allocations.
    real_hf_3l: ?*DSpark3LRuntime = null,

    pub fn init(
        config: DSparkConfig,
        weights: DSparkWeights,
        workspace: *DSparkWorkspace,
    ) Error!DSparkProposer {
        try config.validate();
        try weights.validate(config);
        if (workspace.config.block_width != config.block_width or
            workspace.config.top_k_per_pos != config.top_k_per_pos or
            workspace.config.hidden_size != config.hidden_size or
            workspace.config.vocab_size != config.vocab_size)
        {
            return error.WorkspaceMismatch;
        }
        return .{
            .config = config,
            .weights = weights,
            .workspace = workspace,
            .rounds_proposed = 0,
            .real_hf_3l = null,
        };
    }

    /// Wraps an already-loaded real-HF `3L DSpark` (`mtp.0..2.*`) runtime behind the
    /// standard `DSparkProposer` interface so `SpecEngine.initWithDSpark` works seamlessly.
    pub fn init3L(rt: *DSpark3LRuntime) DSparkProposer {
        return .{
            .config = rt.workspace.config,
            .weights = .{
                .slot_projs = &.{},
                .slot_embed = &.{},
                .lm_head = .{ .dense_f32 = &.{} },
            },
            .workspace = &rt.workspace,
            .rounds_proposed = 0,
            .real_hf_3l = rt,
        };
    }

    pub fn is3L(self: *const DSparkProposer) bool {
        return self.real_hf_3l != null;
    }

    pub fn totalWiredBytes(self: *const DSparkProposer) u64 {
        if (self.real_hf_3l) |rt| return rt.totalWiredBytes();
        return self.workspace.totalWiredBytes();
    }

    pub fn reset3LWindow(self: *DSparkProposer) void {
        if (self.real_hf_3l) |rt| rt.resetWindow();
    }

    pub fn set3LTargetTaps(self: *DSparkProposer, taps: []const []const f32, start_pos: u32) void {
        if (self.real_hf_3l) |rt| rt.setPendingTaps(taps, start_pos);
    }

    pub fn seed3LWindow(self: *DSparkProposer, taps: []const []const f32, start_pos: u32) Error!void {
        if (self.real_hf_3l) |rt| try rt.seedWindow(taps, start_pos);
    }

    pub fn coldStats(self: *const DSparkProposer) ?ColdDSparkStats {
        if (self.real_hf_3l) |rt| return rt.coldStats();
        return null;
    }

    pub fn setColdExpertMinWeight(self: *DSparkProposer, min_weight: f32) void {
        if (self.real_hf_3l) |rt| rt.setColdExpertMinWeight(min_weight);
    }

    pub fn setConfidenceThreshold(self: *DSparkProposer, threshold: f32) void {
        if (self.real_hf_3l) |rt| rt.setConfidenceThreshold(threshold);
    }

    pub fn lastSlotConfidences(self: *const DSparkProposer) []const f32 {
        if (self.real_hf_3l) |rt| return rt.lastSlotConfidences();
        return &.{};
    }

    pub fn targetLayerIds(self: *const DSparkProposer) []const u16 {
        if (self.real_hf_3l) |rt| return rt.config.targetLayerIdsSlice();
        return &.{};
    }

    /// Executes the single-step semi-autoregressive parallel forward pass
    /// conditioned on `(encoder_summary, last_decoder_hidden)`, writing sorted
    /// top-k `(token_id, prob)` pairs for all `block_width` future positions
    /// into `self.workspace`. Zero heap allocations.
    pub fn predictParallelPositions(
        self: *DSparkProposer,
        encoder_summary: ?[]const f32,
        last_decoder_hidden: []const f32,
    ) Error!void {
        const h: usize = self.config.hidden_size;
        const w: usize = self.config.block_width;
        const v: usize = self.config.vocab_size;
        const k: usize = self.config.top_k_per_pos;

        if (last_decoder_hidden.len != h) return error.DimensionMismatch;
        if (encoder_summary) |es| {
            if (es.len != h) return error.DimensionMismatch;
        }

        const base = self.workspace.base_hidden[0..h];

        // 1. Fuse decoder hidden state with optional CED encoder summary vector.
        if (self.weights.dec_proj) |dp| {
            matVecF32(dp, last_decoder_hidden, h, h, base);
        } else {
            @memcpy(base, last_decoder_hidden);
        }

        if (encoder_summary) |es| {
            if (self.weights.enc_proj) |ep| {
                matVecAccumF32(ep, es, h, h, base);
            } else {
                for (0..h) |d| {
                    base[d] += es[d];
                }
            }
        }

        // 2. Compute all `block_width` position-conditioned slot hiddens in one pass.
        const hh = h * h;
        const slot_in = self.workspace.slot_input[0..h];
        for (0..w) |s| {
            const emb = self.weights.slot_embed[s * h ..][0..h];
            for (0..h) |d| {
                slot_in[d] = base[d] + emb[d];
            }
            const proj = self.weights.slot_projs[s * hh ..][0..hh];
            const dst_h = self.workspace.slot_hiddens[s * h ..][0..h];
            matVecF32(proj, slot_in, h, h, dst_h);
            applyRmsNormInPlace(dst_h, self.weights.slot_norm_weight, self.config.rms_eps);
        }

        // 3. Project all `block_width` slots to `[block_width, vocab_size]` logits in one call.
        const logits = self.workspace.slot_logits[0 .. w * v];
        switch (self.weights.lm_head) {
            .dense_f32 => |mat| {
                for (0..w) |s| {
                    const sh = self.workspace.slot_hiddens[s * h ..][0..h];
                    const dst_l = logits[s * v ..][0..v];
                    matVecF32(mat, sh, v, h, dst_l);
                }
            },
            .dense_f16 => |mat| {
                for (0..w) |s| {
                    const sh = self.workspace.slot_hiddens[s * h ..][0..h];
                    const dst_l = logits[s * v ..][0..v];
                    matVecF16Weights(mat, sh, v, h, dst_l);
                }
            },
            .dense_bf16 => |mat| {
                for (0..w) |s| {
                    const sh = self.workspace.slot_hiddens[s * h ..][0..h];
                    const dst_l = logits[s * v ..][0..v];
                    matVecBf16Weights(mat, sh, v, h, dst_l);
                }
            },
            .callback => |cb| {
                try cb.eval_fn(
                    cb.ctx,
                    self.workspace.slot_hiddens[0 .. w * h],
                    self.config.block_width,
                    self.config.hidden_size,
                    self.config.vocab_size,
                    logits,
                );
            },
        }

        // 4. Deterministic top-k extraction + exact softmax probability normalization per slot.
        const inv_temp: f32 = 1.0 / self.config.temperature;
        for (0..w) |s| {
            const row = logits[s * v ..][0..v];
            const out_tok = self.workspace.topk_tokens[s * k ..][0..k];
            const out_prob = self.workspace.topk_probs[s * k ..][0..k];
            try extractDeterministicTopK(row, inv_temp, out_tok, out_prob);
        }
    }

    /// Proposes a speculative candidate tree into `CandidateTree` (`DraftTree(32)`),
    /// merging optional L1 SAM prefix seeds (`ADR-0005`) with one parallel DSpark
    /// semi-autoregressive prediction pass. Zero heap allocations.
    pub fn propose(
        self: *DSparkProposer,
        root_token: u32,
        encoder_summary: ?[]const f32,
        last_decoder_hidden: []const f32,
        sam: ?*const sam_mod.SuffixAutomaton,
        out_tree: *CandidateTree,
    ) Error!ProposeResult {
        return self.proposeInto(
            CandidateTree.capacity,
            root_token,
            encoder_summary,
            last_decoder_hidden,
            sam,
            out_tree,
        );
    }

    /// Generic zero-allocation proposer targeting any `tree.DraftTree(max_nodes)`.
    pub fn proposeInto(
        self: *DSparkProposer,
        comptime max_nodes: u16,
        root_token: u32,
        encoder_summary: ?[]const f32,
        last_decoder_hidden: []const f32,
        sam: ?*const sam_mod.SuffixAutomaton,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!ProposeResult {
        return self.proposeIntoConstrained(
            max_nodes,
            root_token,
            encoder_summary,
            last_decoder_hidden,
            sam,
            null,
            out_tree,
        );
    }

    /// Grammar-aware `D10` tree proposer (`C22`, zero heap alloc): when `grammar`
    /// is non-null and active, truncates L1 chains to valid prefixes and masks
    /// `slot_logits` at every parent frontier node using `GrammarState` bitmasks
    /// so every drafted branch satisfies strict JSON schemas without collapsing depth.
    pub fn proposeIntoConstrained(
        self: *DSparkProposer,
        comptime max_nodes: u16,
        root_token: u32,
        encoder_summary: ?[]const f32,
        last_decoder_hidden: []const f32,
        sam: ?*const sam_mod.SuffixAutomaton,
        grammar_opt: ?accept_mod.GrammarTreeAdapter,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!ProposeResult {
        if (self.real_hf_3l) |rt| {
            const res = try rt.proposeIntoConstrained(
                max_nodes,
                root_token,
                encoder_summary,
                last_decoder_hidden,
                sam,
                grammar_opt,
                out_tree,
            );
            self.rounds_proposed += 1;
            return res;
        }
        out_tree.reset(root_token);

        const node_limit: u16 = @min(@as(u16, self.config.max_tree_nodes), max_nodes);
        const gram_active = if (grammar_opt) |ga| ga.isActive() else false;
        var node_states: [tree_mod.MAX_SUPPORTED_NODES]grammar_mod.GrammarState = undefined;
        var node_valid: [tree_mod.MAX_SUPPORTED_NODES]bool = @splat(false);
        node_valid[0] = true;
        if (grammar_opt) |ga| {
            if (ga.isActive()) node_states[0] = ga.state.snapshot();
        }

        // 1. Optional L1 SAM prefix seed / branch merge (ADR-0005), pruned to grammar-valid prefix.
        var l1_match_len: u32 = 0;
        var l1_inserted: u16 = 0;
        var l1_is_seed: bool = false;
        var seed_tail_node: u16 = 0;

        if (sam) |automaton| {
            const max_l1: usize = @min(
                @as(usize, self.config.l1_max_draft_len),
                @as(usize, node_limit - 1),
            );
            if (max_l1 > 0) {
                const q = automaton.queryActive(max_l1, self.workspace.l1_draft_buf[0..max_l1]);
                var valid_len: usize = q.draft_len;
                if (gram_active) {
                    const ga = grammar_opt.?;
                    var st_walk = node_states[0].snapshot();
                    valid_len = 0;
                    for (self.workspace.l1_draft_buf[0..q.draft_len]) |tid| {
                        if (!ga.advanceTokenOnState(&st_walk, tid)) break;
                        valid_len += 1;
                    }
                }
                l1_match_len = if (valid_len == q.draft_len) q.match_len else @intCast(@min(q.match_len, valid_len));
                if (valid_len > 0) {
                    const ins = out_tree.insertL1Continuation(
                        @intCast(@max(@as(u32, 1), l1_match_len)),
                        self.workspace.l1_draft_buf[0..valid_len],
                        self.config.l1_seed_step_prob,
                        self.config.l1_branch_step_prob,
                    );
                    l1_inserted = ins.inserted_count;
                    l1_is_seed = ins.is_seed;
                    seed_tail_node = ins.seed_tail_node;
                    if (gram_active) {
                        const ga = grammar_opt.?;
                        for (1..out_tree.count) |ni| {
                            const nd = out_tree.nodes[ni];
                            if (node_valid[nd.parent]) {
                                var cs = node_states[nd.parent].snapshot();
                                if (ga.advanceTokenOnState(&cs, nd.token_id)) {
                                    node_states[ni] = cs;
                                    node_valid[ni] = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Single parallel forward pass predicting all `block_width` positions.
        try self.predictParallelPositions(encoder_summary, last_decoder_hidden);

        const before_dspark = out_tree.count;

        // 3. Deterministic semi-autoregressive beam expansion from root `0` across
        //    positions `0..block_width`, strictly bounded by `node_limit`.
        var frontier_cur: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
        var frontier_next: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
        var cur_len: usize = 1;
        frontier_cur[0] = 0;

        const w: u8 = self.config.block_width;
        const k: usize = self.config.top_k_per_pos;
        const v: usize = self.config.vocab_size;
        const inv_temp: f32 = 1.0 / self.config.temperature;
        var masked_toks: [MAX_TOP_K_PER_POS]u32 = undefined;
        var masked_probs: [MAX_TOP_K_PER_POS]f32 = undefined;

        for (0..w) |pos_usize| {
            const pos: u8 = @intCast(pos_usize);
            const pos_tokens = self.workspace.slotTokens(pos);
            const pos_probs = self.workspace.slotProbs(pos);
            const pos_logits = self.workspace.slot_logits[@as(usize, pos) * v ..][0..v];

            // Sort current frontier deterministically by descending `cum_prob`,
            // breaking ties by ascending node index so highest-value paths always
            // receive tree capacity first.
            sortFrontierDescending(max_nodes, out_tree, frontier_cur[0..cur_len]);

            var next_len: usize = 0;
            for (frontier_cur[0..cur_len]) |parent| {
                if (gram_active and !node_valid[parent]) continue;
                var cand_count: usize = k;
                var cand_toks: []const u32 = pos_tokens;
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
                        masked_toks[0..k],
                        masked_probs[0..k],
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
                        // Deduplicate against L1 / prior branch without exceeding node_limit.
                        break :blk out_tree.insertChild(parent, tok, prob, .l2) orelse existing;
                    } else blk: {
                        if (out_tree.count >= node_limit) break :blk null;
                        break :blk out_tree.insertChild(parent, tok, prob, .l2);
                    };

                    if (child) |c_idx| {
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

        // 4. Dual-source ADR-0005 seed continuation: if L1 produced a high-confidence
        //    seed chain and capacity remains, extend `seed_tail_node` with DSpark's
        //    top-1 parallel chain (grammar-masked when C22 is active).
        if (l1_is_seed and seed_tail_node != 0 and out_tree.count < node_limit and (!gram_active or node_valid[seed_tail_node])) {
            var tail: u16 = seed_tail_node;
            for (0..w) |pos_usize| {
                if (out_tree.count >= node_limit) break;
                const pos: u8 = @intCast(pos_usize);
                var tok = self.workspace.slotTokens(pos)[0];
                var prob = self.workspace.slotProbs(pos)[0];
                var next_st: grammar_mod.GrammarState = undefined;
                if (gram_active) {
                    var tail_ga = grammar_opt.?;
                    var tstate = node_states[tail].snapshot();
                    tail_ga.state = &tstate;
                    const mask = tail_ga.fillBitmask(v) catch break;
                    const pos_logits = self.workspace.slot_logits[@as(usize, pos) * v ..][0..v];
                    const got = extractDeterministicTopKMasked(pos_logits, inv_temp, mask, masked_toks[0..1], masked_probs[0..1]) catch break;
                    if (got == 0) break;
                    tok = masked_toks[0];
                    prob = masked_probs[0];
                    next_st = node_states[tail].snapshot();
                    if (!grammar_opt.?.advanceTokenOnState(&next_st, tok)) break;
                }
                const next = out_tree.insertChild(tail, tok, prob, .l2) orelse break;
                if (gram_active) {
                    node_states[next] = next_st;
                    node_valid[next] = true;
                }
                tail = next;
            }
        }

        self.rounds_proposed += 1;

        return .{
            .tree_nodes = out_tree.count,
            .max_depth = out_tree.max_depth,
            .l1_match_len = l1_match_len,
            .l1_inserted = l1_inserted,
            .l1_is_seed = l1_is_seed,
            .dspark_inserted = out_tree.count - before_dspark,
            .expected_tokens = out_tree.expectedTokens(),
        };
    }
};

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
    // Insertion sort on tiny frontier (<= 64 elements): zero heap allocs, deterministic.
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

/// Computes `out[r] = dot(mat[r, 0..cols], vec[0..cols])` with 4-way unrolled FP32 accumulation.
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
        while (c < cols) : (c += 1) {
            acc += row[c] * vec[c];
        }
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
        while (c < cols) : (c += 1) {
            acc += row[c] * vec[c];
        }
        out[r] += acc;
    }
}

fn matVecF16Weights(mat: []const f16, vec: []const f32, rows: usize, cols: usize, out: []f32) void {
    for (0..rows) |r| {
        const row = mat[r * cols ..][0..cols];
        var acc: f32 = 0.0;
        for (0..cols) |c| {
            acc += @as(f32, @floatCast(row[c])) * vec[c];
        }
        out[r] = acc;
    }
}

fn matVecBf16Weights(mat: []const u16, vec: []const f32, rows: usize, cols: usize, out: []f32) void {
    for (0..rows) |r| {
        const row = mat[r * cols ..][0..cols];
        var acc: f32 = 0.0;
        for (0..cols) |c| {
            acc += quant.bf16ToF32(row[c]) * vec[c];
        }
        out[r] = acc;
    }
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
        for (vec, w) |*x, wi| {
            x.* = x.* * inv_rms * wi;
        }
    } else {
        for (vec) |*x| {
            x.* *= inv_rms;
        }
    }
}

/// Extracts top-k `(token_id, softmax_prob)` deterministically from `logits`.
/// Ties in logit value break strictly by smaller `token_id`.
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

        // Maintain sorted top-k buffer in descending `scaled`, ascending `tok_id` on tie.
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

// ============================================================================
// Real-HF `DeepSeek-V4.1-Flash-1017` 3L `DSpark` (`mtp.0..2.*`, `D10`)
// Cold-tiered (`F_NOCACHE` pread + `C27` hot cache), multi-format quantized
// (`MXFP8_G32`, `GGUF_MXFP4`, `GSQ-RCO` `q4`/`q3`/`q2`, `bf16`, `f16`, `f32`),
// optional `D13` `SME2` `sme2_panel64`, `mHC` Sinkhorn-20, `DSparkAttention`,
// `markov_head` low-rank scorer, and `confidence_head` tree/D7 gating.
// ============================================================================

pub const LoadError = Error || amf.ParseError || amf.IoError || sme.Error || std.Thread.SpawnError || error{
    MissingTensor,
    UnsupportedDType,
    InvalidJson,
    OutOfMemory,
};

/// Zero-allocation callback reading one target token embedding row (`[hidden_size]`).
pub const EmbedCallback = struct {
    ctx: ?*anyopaque = null,
    read_fn: *const fn (
        ctx: ?*anyopaque,
        token_id: u32,
        out_f32: []f32,
    ) Error!void,
};

/// Controls how `mtp.2.markov_head` and the optional target `lm_head` combine.
pub const DSpark3LLmHeadMode = enum {
    /// Avoid `block_size` full `[vocab, H]` `lm_head` sweeps when `markov_head` is present;
    /// score block positions autoregressively via low-rank `[vocab, markov_rank]` `markov_head`
    /// conditioned on `mtp.2.norm(x)`.
    markov_low_rank,
    /// Exact `model.py` hybrid scoring: `target_lm_head(mtp.2.norm(x)) + markov_head(prev_tok)`.
    hybrid_target_plus_markov,
    /// Target `lm_head` only (fallback when `markov_head` is absent or explicitly disabled).
    target_lm_head_only,
};

/// Configuration for real-HF `3L DSpark` (`draft_format: "dspark_3l_mtp"`).
pub const DSpark3LConfig = struct {
    num_layers: u8 = 3,
    block_size: u8 = 5,
    top_k_per_pos: u8 = 2,
    max_tree_nodes: u8 = 32,
    hidden_size: u32,
    vocab_size: u32,
    num_attention_heads: u32 = 4,
    head_dim: u32 = 64,
    q_lora_rank: u32 = 64,
    kv_lora_rank: u32 = 64,
    qk_rope_head_dim: u32 = 32,
    o_lora_rank: u32 = 64,
    o_groups: u32 = 2,
    window_size: u32 = DEFAULT_DSPARK_WINDOW,
    n_routed_experts: u32 = 8,
    routed_top_k: u32 = 3,
    moe_intermediate_size: u32 = 128,
    has_shared_expert: bool = true,
    hc_mult: u32 = 4,
    hc_sinkhorn_iters: u32 = 20,
    hc_eps: f32 = 1e-6,
    rms_eps: f32 = 1e-6,
    rope_theta: f32 = 10_000.0,
    routed_scaling_factor: f32 = 1.5,
    swiglu_limit: f32 = 10.0,
    markov_rank: u32 = 256,
    target_layer_ids: [MAX_DSPARK_TARGET_TAPS]u16 = .{ 0, 0, 0, 0 },
    num_target_taps: u8 = 3,
    noise_token_id: u32 = 0,
    temperature: f32 = 1.0,
    /// Optional sigmoid confidence threshold gating deeper tree expansion (`0.0` disables hard cutoff).
    confidence_threshold: f32 = 0.0,
    /// Optional cold-expert routing-weight cutoff (`0.0` runs all selected cold experts).
    cold_expert_min_weight: f32 = 0.0,
    /// Number of 16 KiB-aligned `WiredArena` cold-expert cache slots (`C27` session-local hot cache).
    cold_slot_pool_slots: u32 = 16,
    lm_head_mode: DSpark3LLmHeadMode = .markov_low_rank,
    l1_seed_step_prob: f32 = 0.90,
    l1_branch_step_prob: f32 = 0.55,
    l1_max_draft_len: u8 = 8,

    pub fn targetLayerIdsSlice(self: *const DSpark3LConfig) []const u16 {
        return self.target_layer_ids[0..@min(@as(usize, self.num_target_taps), MAX_DSPARK_TARGET_TAPS)];
    }

    pub fn toLegacyCompatConfig(self: DSpark3LConfig) DSparkConfig {
        return .{
            .block_width = self.block_size,
            .top_k_per_pos = self.top_k_per_pos,
            .max_tree_nodes = self.max_tree_nodes,
            .hidden_size = self.hidden_size,
            .vocab_size = self.vocab_size,
            .rms_eps = self.rms_eps,
            .temperature = self.temperature,
            .l1_seed_step_prob = self.l1_seed_step_prob,
            .l1_branch_step_prob = self.l1_branch_step_prob,
            .l1_max_draft_len = self.l1_max_draft_len,
        };
    }

    pub fn validate(self: DSpark3LConfig) Error!void {
        if (self.num_layers == 0 or self.num_layers > MAX_DSPARK_3L_LAYERS) return error.InvalidConfig;
        if (self.block_size == 0 or self.block_size > MAX_BLOCK_WIDTH) return error.InvalidConfig;
        if (self.top_k_per_pos == 0 or self.top_k_per_pos > MAX_TOP_K_PER_POS) return error.InvalidConfig;
        if (self.max_tree_nodes < 2 or self.max_tree_nodes > MAX_TREE_NODES) return error.InvalidConfig;
        if (self.hidden_size == 0 or self.vocab_size == 0) return error.InvalidConfig;
        if (self.top_k_per_pos > self.vocab_size) return error.InvalidConfig;
        if (self.num_attention_heads == 0 or self.head_dim == 0) return error.InvalidConfig;
        if (self.q_lora_rank == 0 or self.o_lora_rank == 0 or self.o_groups == 0) return error.InvalidConfig;
        if (self.window_size == 0) return error.InvalidConfig;
        if (self.hc_mult == 0 or self.hc_mult > MAX_HC_MULT) return error.InvalidConfig;
        if (self.hc_sinkhorn_iters == 0) return error.InvalidConfig;
        if (self.num_target_taps == 0 or self.num_target_taps > MAX_DSPARK_TARGET_TAPS) return error.InvalidConfig;
        if (self.n_routed_experts > 0 and (self.routed_top_k == 0 or self.routed_top_k > MAX_DSPARK_ROUTED_TOP_K)) {
            return error.InvalidConfig;
        }
        if (!std.math.isFinite(self.rms_eps) or self.rms_eps <= 0.0) return error.InvalidConfig;
        if (!std.math.isFinite(self.hc_eps) or self.hc_eps <= 0.0) return error.InvalidConfig;
        if (!std.math.isFinite(self.temperature) or self.temperature <= 0.0) return error.InvalidConfig;
        if (!std.math.isFinite(self.confidence_threshold) or self.confidence_threshold < 0.0) return error.InvalidConfig;
        if (!std.math.isFinite(self.cold_expert_min_weight) or self.cold_expert_min_weight < 0.0) return error.InvalidConfig;
        if (self.l1_max_draft_len > MAX_L1_DRAFT_LEN) return error.InvalidConfig;
    }

    /// Parses a `dspark_config.json` (or embedded `__meta.config_json`) payload.
    /// Geometry fields omitted from compact `dspark_config.json` files (such as
    /// `q_lora_rank`, `head_dim`, `markov_rank`, `moe_intermediate_size`) are
    /// auto-reconciled from `.amf` tensor shapes during `DSpark3LRuntime.load*`.
    pub fn fromJson(allocator: std.mem.Allocator, json_bytes: []const u8) LoadError!DSpark3LConfig {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidJson,
        };
        defer parsed.deinit();

        const root = switch (parsed.value) {
            .object => |o| o,
            else => return error.InvalidJson,
        };
        const obj: std.json.ObjectMap = if (root.get("text_config")) |tc| switch (tc) {
            .object => |o| o,
            else => root,
        } else root;

        const h = dsparkJsonU32(obj, "hidden_size") orelse dsparkJsonU32(root, "hidden_size") orelse return error.InvalidJson;
        const v = dsparkJsonU32(obj, "vocab_size") orelse dsparkJsonU32(root, "vocab_size") orelse return error.InvalidJson;
        const n_mtp_u32 = dsparkJsonU32(obj, "num_nextn_predict_layers") orelse
            dsparkJsonU32(root, "num_nextn_predict_layers") orelse 3;
        const block_u32 = dsparkJsonU32(obj, "dspark_block_size") orelse
            dsparkJsonU32(root, "dspark_block_size") orelse
            dsparkJsonU32(root, "block_width") orelse 5;
        const top_k_pos = dsparkJsonU32(obj, "top_k_per_pos") orelse
            dsparkJsonU32(root, "top_k_per_pos") orelse 2;
        const max_nodes = dsparkJsonU32(obj, "max_tree_nodes") orelse
            dsparkJsonU32(root, "max_tree_nodes") orelse 32;
        const n_exp = dsparkJsonU32(obj, "dspark_n_routed_experts") orelse
            dsparkJsonU32(root, "dspark_n_routed_experts") orelse
            dsparkJsonU32(obj, "n_routed_experts") orelse 8;
        const r_topk = dsparkJsonU32(obj, "dspark_routed_top_k") orelse
            dsparkJsonU32(root, "dspark_routed_top_k") orelse
            dsparkJsonU32(obj, "dspark_n_activated_experts") orelse 3;
        const n_heads = dsparkJsonU32(obj, "num_attention_heads") orelse 4;
        const hd = dsparkJsonU32(obj, "head_dim") orelse @max(@as(u32, 32), h / @max(n_heads, 1));
        const q_lora = dsparkJsonU32(obj, "q_lora_rank") orelse @min(h, 64);
        const kv_lora = dsparkJsonU32(obj, "kv_lora_rank") orelse hd;
        const qk_rope = dsparkJsonU32(obj, "qk_rope_head_dim") orelse @min(hd / 2, 64);
        const o_lora = dsparkJsonU32(obj, "o_lora_rank") orelse @min(h, 64);
        const o_groups = dsparkJsonU32(obj, "o_groups") orelse 2;
        const win = dsparkJsonU32(obj, "window_size") orelse DEFAULT_DSPARK_WINDOW;
        const moe_i = dsparkJsonU32(obj, "moe_intermediate_size") orelse @max(@as(u32, 32), h / 2);
        const hc_m = dsparkJsonU32(obj, "hc_mult") orelse 4;
        const hc_it = dsparkJsonU32(obj, "hc_sinkhorn_iters") orelse 20;
        const hc_eps = dsparkJsonF32(obj, "hc_eps") orelse 1e-6;
        const rms_eps = dsparkJsonF32(obj, "rms_norm_eps") orelse 1e-6;
        const rope_theta = dsparkJsonF32(obj, "rope_theta") orelse 10_000.0;
        const route_scale = dsparkJsonF32(obj, "routed_scaling_factor") orelse 1.5;
        const swiglu_lim = dsparkJsonF32(obj, "swiglu_limit") orelse 10.0;
        const markov_r = dsparkJsonU32(obj, "dspark_markov_rank") orelse
            dsparkJsonU32(root, "dspark_markov_rank") orelse 256;
        const noise_tok = dsparkJsonU32(obj, "dspark_noise_token_id") orelse
            dsparkJsonU32(root, "dspark_noise_token_id") orelse 0;
        const conf_thresh = dsparkJsonF32(obj, "confidence_threshold") orelse
            dsparkJsonF32(root, "confidence_threshold") orelse 0.0;
        const cold_min_w = dsparkJsonF32(obj, "cold_expert_min_weight") orelse
            dsparkJsonF32(root, "cold_expert_min_weight") orelse 0.0;

        var taps: [MAX_DSPARK_TARGET_TAPS]u16 = .{ 0, 0, 0, 0 };
        var num_taps: u8 = 3;
        if (obj.get("dspark_target_layer_ids") orelse root.get("dspark_target_layer_ids")) |tv| {
            if (tv == .array and tv.array.items.len > 0) {
                const count = @min(tv.array.items.len, MAX_DSPARK_TARGET_TAPS);
                for (0..count) |i| {
                    switch (tv.array.items[i]) {
                        .integer => |iv| {
                            if (iv < 0 or iv > std.math.maxInt(u16)) return error.InvalidConfig;
                            taps[i] = @intCast(iv);
                        },
                        else => return error.InvalidJson,
                    }
                }
                num_taps = @intCast(count);
            }
        }

        return .{
            .num_layers = @intCast(@min(MAX_DSPARK_3L_LAYERS, @max(@as(u32, 1), n_mtp_u32))),
            .block_size = @intCast(@min(MAX_BLOCK_WIDTH, @max(@as(u32, 1), block_u32))),
            .top_k_per_pos = @intCast(@min(MAX_TOP_K_PER_POS, @max(@as(u32, 1), top_k_pos))),
            .max_tree_nodes = @intCast(@min(MAX_TREE_NODES, @max(@as(u32, 2), max_nodes))),
            .hidden_size = h,
            .vocab_size = v,
            .num_attention_heads = n_heads,
            .head_dim = hd,
            .q_lora_rank = q_lora,
            .kv_lora_rank = kv_lora,
            .qk_rope_head_dim = qk_rope,
            .o_lora_rank = o_lora,
            .o_groups = o_groups,
            .window_size = win,
            .n_routed_experts = n_exp,
            .routed_top_k = @min(r_topk, @max(n_exp, 1)),
            .moe_intermediate_size = moe_i,
            .hc_mult = hc_m,
            .hc_sinkhorn_iters = hc_it,
            .hc_eps = hc_eps,
            .rms_eps = rms_eps,
            .rope_theta = rope_theta,
            .routed_scaling_factor = route_scale,
            .swiglu_limit = swiglu_lim,
            .markov_rank = markov_r,
            .target_layer_ids = taps,
            .num_target_taps = num_taps,
            .noise_token_id = noise_tok,
            .confidence_threshold = conf_thresh,
            .cold_expert_min_weight = cold_min_w,
        };
    }
};

/// Options controlling `DSpark3LRuntime` loading, cold-tiering, and target hooks.
pub const DSpark3LLoadOptions = struct {
    embed_hook: ?EmbedCallback = null,
    lm_head_hook: ?LmHeadHook = null,
    lm_head_mode: ?DSpark3LLmHeadMode = null,
    budget_override: ?budget_mod.Budget = null,
    verify_crcs: bool = true,
    /// When `true`, forces all routed experts onto the cold `F_NOCACHE` + `C27`
    /// slot-pool tier even if the `.amf` entries are tagged `.hot_expert`.
    force_cold_experts: bool = false,
    /// Optional cap on hot wired routed experts per stage (`null` follows `.amf` `isCold()`).
    max_hot_experts_per_layer: ?u32 = null,
    /// Optional override for cold-expert slot-pool capacity (`C27` session-local hot cache).
    cold_slot_pool_slots: ?u32 = null,
    /// Optional override for cold-expert minimum routing-weight threshold.
    cold_expert_min_weight: ?f32 = null,
    /// Optional override for `DSparkConfidenceHead` sigmoid gating threshold.
    confidence_threshold: ?f32 = null,
};

/// Telemetry for cold-tiered `3L DSpark` MoE experts (`F_NOCACHE` pread + `C27` hot cache).
pub const ColdDSparkStats = struct {
    hot_experts: u32 = 0,
    cold_experts: u32 = 0,
    pool_slots: u32 = 0,
    slot_pool_bytes: u64 = 0,
    cache_hits: u64 = 0,
    cache_misses: u64 = 0,
    nvme_bytes_read: u64 = 0,
    skipped_low_weight: u64 = 0,
};

/// Compact view over any wired `.amf` 2D tensor (`f32`, `f16`, `bf16`, `mxfp8_g32`,
/// `gguf_mxfp4`, `q4_affine`/`q3_affine`/`q2_affine` `GSQ-RCO`, `sme2_panel64`, etc.).
const QuantMatrix = struct {
    dtype: amf.DType = .f32,
    layout: amf.TensorLayout = .row_major,
    group_size: u16 = 0,
    n: u32 = 0,
    k: u32 = 0,
    data: []const u8 = &.{},
    params: []const u8 = &.{},
    sme_weight: ?sme.SmeLinearWeight = null,

    fn fromEntryAndBlob(entry: *const amf.TensorEntry, blob: []const u8) LoadError!QuantMatrix {
        const dt = try entry.dataType();
        const lay = try entry.tensorLayout();
        const n: u32 = entry.shape[0];
        const k: u32 = if (entry.rank >= 2) entry.shape[1] else 1;
        if (n == 0 or k == 0) return error.InvalidWeights;
        const d_end: usize = @intCast(entry.data_bytes);
        if (d_end > blob.len) return error.InvalidWeights;
        const p_off: usize = @intCast(entry.params_offset);
        const p_len: usize = @intCast(entry.params_bytes);
        if (p_len > 0 and (p_off + p_len > blob.len)) return error.InvalidWeights;
        const sme_w: ?sme.SmeLinearWeight = if (lay == .sme2_panel64)
            try sme.SmeLinearWeight.fromAmfEntry(entry, blob, .temporal)
        else
            null;
        return .{
            .dtype = dt,
            .layout = lay,
            .group_size = entry.group_size,
            .n = n,
            .k = k,
            .data = blob[0..d_end],
            .params = if (p_len > 0) blob[p_off .. p_off + p_len] else &.{},
            .sme_weight = sme_w,
        };
    }
};

const ColdBlobDesc = struct {
    dtype: amf.DType,
    layout: amf.TensorLayout,
    group_size: u16,
    n: u32,
    k: u32,
    blob_offset: u64,
    blob_bytes: u64,
    data_bytes: u64,
    params_offset: u64,
    params_bytes: u64,
    blob_crc32: u32,
    param_kind: amf.QuantParamKind,

    fn fromEntry(entry: *const amf.TensorEntry) LoadError!ColdBlobDesc {
        return .{
            .dtype = try entry.dataType(),
            .layout = try entry.tensorLayout(),
            .group_size = entry.group_size,
            .n = entry.shape[0],
            .k = if (entry.rank >= 2) entry.shape[1] else 1,
            .blob_offset = entry.blob_offset,
            .blob_bytes = entry.blob_bytes,
            .data_bytes = entry.data_bytes,
            .params_offset = entry.params_offset,
            .params_bytes = entry.params_bytes,
            .blob_crc32 = entry.blob_crc32,
            .param_kind = try entry.quantParamKind(),
        };
    }

    fn asQuantMatrix(self: ColdBlobDesc, buf: []const u8) Error!QuantMatrix {
        const d_end: usize = @intCast(self.data_bytes);
        const p_off: usize = @intCast(self.params_offset);
        const p_len: usize = @intCast(self.params_bytes);
        if (d_end > buf.len or (p_len > 0 and p_off + p_len > buf.len)) return error.InvalidWeights;
        var sme_w: ?sme.SmeLinearWeight = null;
        if (self.layout == .sme2_panel64) {
            const pdt = sme.panelDTypeFromAmf(self.dtype) catch return error.InvalidWeights;
            const pk = sme.quantParamKindFromAmf(self.param_kind);
            sme_w = .{
                .dtype = pdt,
                .param_kind = pk,
                .cache_hint = .non_temporal,
                .n = self.n,
                .k = self.k,
                .w_packed = buf[0..d_end],
            };
        }
        return .{
            .dtype = self.dtype,
            .layout = self.layout,
            .group_size = self.group_size,
            .n = self.n,
            .k = self.k,
            .data = buf[0..d_end],
            .params = if (p_len > 0) buf[p_off .. p_off + p_len] else &.{},
            .sme_weight = sme_w,
        };
    }
};

const ExpertTriple = struct {
    w1: QuantMatrix,
    w2: QuantMatrix,
    w3: QuantMatrix,
};

const ExpertRef = union(enum) {
    wired: ExpertTriple,
    cold: struct {
        w1: ColdBlobDesc,
        w2: ColdBlobDesc,
        w3: ColdBlobDesc,
    },
};

const ColdSlot = struct {
    valid: bool = false,
    layer: u8 = 0,
    expert: u16 = 0,
    hits: u32 = 0,
    last_epoch: u64 = 0,
    w1_buf: []u8 = &.{},
    w2_buf: []u8 = &.{},
    w3_buf: []u8 = &.{},
};

/// 16 KiB-aligned `WiredArena` cold-expert slot pool + `C27` session-local hot-expert cache.
/// Zero heap allocations on every cold read (`F_NOCACHE` `pread` or in-memory slice).
const ColdExpertSlotPool = struct {
    arena: ?arena_mod.WiredArena = null,
    slots: []ColdSlot = &.{},
    backing_fd: ?std.c.fd_t = null,
    backing_bytes: ?[]const u8 = null,
    verify_crcs: bool = true,
    epoch: u64 = 0,
    stats: ColdDSparkStats = .{},

    fn deinit(self: *ColdExpertSlotPool, allocator: std.mem.Allocator) void {
        if (self.slots.len > 0) allocator.free(self.slots);
        self.slots = &.{};
        if (self.arena) |*a| a.deinit();
        self.arena = null;
    }

    fn readBlobInto(self: *ColdExpertSlotPool, desc: ColdBlobDesc, dst: []u8) Error!void {
        const len: usize = @intCast(desc.blob_bytes);
        if (dst.len < len) return error.WorkspaceMismatch;
        const target = dst[0..len];
        if (self.backing_fd) |fd| {
            var done: usize = 0;
            while (done < len) {
                const rc = std.c.pread(fd, target.ptr + done, len - done, @intCast(desc.blob_offset + done));
                if (rc < 0) {
                    if (std.c.errno(rc) == .INTR) continue;
                    return error.CallbackFailed;
                }
                if (rc == 0) return error.CallbackFailed;
                done += @intCast(rc);
            }
        } else if (self.backing_bytes) |raw| {
            const start: usize = @intCast(desc.blob_offset);
            if (start + len > raw.len) return error.InvalidWeights;
            @memcpy(target, raw[start .. start + len]);
        } else {
            return error.InvalidWeights;
        }
        if (self.verify_crcs) {
            if (std.hash.Crc32.hash(target) != desc.blob_crc32) return error.InvalidWeights;
        }
        self.stats.nvme_bytes_read += len;
    }

    fn fetchExpert(
        self: *ColdExpertSlotPool,
        layer: u8,
        expert: u16,
        w1: ColdBlobDesc,
        w2: ColdBlobDesc,
        w3: ColdBlobDesc,
        weight: f32,
        min_weight: f32,
    ) Error!?ExpertTriple {
        if (self.slots.len == 0) return error.WorkspaceMismatch;
        self.epoch += 1;

        // 1. C27 session-local hot-expert cache lookup (zero I/O on hit).
        for (self.slots) |*slot| {
            if (slot.valid and slot.layer == layer and slot.expert == expert) {
                slot.hits +|= 1;
                slot.last_epoch = self.epoch;
                self.stats.cache_hits += 1;
                return .{
                    .w1 = try w1.asQuantMatrix(slot.w1_buf),
                    .w2 = try w2.asQuantMatrix(slot.w2_buf),
                    .w3 = try w3.asQuantMatrix(slot.w3_buf),
                };
            }
        }

        // 2. Skip low-weight cold tail experts before issuing NVMe `F_NOCACHE` preads.
        if (min_weight > 0.0 and weight < min_weight) {
            self.stats.skipped_low_weight += 1;
            return null;
        }

        // 3. Evict invalid or lowest `(hits, last_epoch)` slot (C27 LFU/LRU).
        var victim: usize = 0;
        for (self.slots, 0..) |slot, i| {
            if (!slot.valid) {
                victim = i;
                break;
            }
            const cur_v = self.slots[victim];
            if (slot.hits < cur_v.hits or (slot.hits == cur_v.hits and slot.last_epoch < cur_v.last_epoch)) {
                victim = i;
            }
        }
        var slot = &self.slots[victim];
        try self.readBlobInto(w1, slot.w1_buf);
        try self.readBlobInto(w2, slot.w2_buf);
        try self.readBlobInto(w3, slot.w3_buf);
        slot.valid = true;
        slot.layer = layer;
        slot.expert = expert;
        slot.hits = 1;
        slot.last_epoch = self.epoch;
        self.stats.cache_misses += 1;
        return .{
            .w1 = try w1.asQuantMatrix(slot.w1_buf),
            .w2 = try w2.asQuantMatrix(slot.w2_buf),
            .w3 = try w3.asQuantMatrix(slot.w3_buf),
        };
    }
};

const HcBlockWeights = struct {
    /// True when `fn_mat` has real-HF shape `[(2+hc)*hc, hc*H]` and `base` has `(2+hc)*hc` elems.
    /// False when compact synth shape `fn_mat [H, hc]` and `base [hc*hc]`.
    is_full_hf_shape: bool = true,
    fn_mat: QuantMatrix = .{},
    base_f32: []const f32 = &.{},
    scale_f32: []const f32 = &.{},
};

const DSpark3LLayer = struct {
    hc_attn: HcBlockWeights = .{},
    hc_ffn: HcBlockWeights = .{},
    attn_norm: []const f32 = &.{},
    wq_a: QuantMatrix = .{},
    q_norm: []const f32 = &.{},
    wq_b: QuantMatrix = .{},
    wkv: QuantMatrix = .{},
    kv_norm: []const f32 = &.{},
    wo_a: QuantMatrix = .{},
    wo_b: QuantMatrix = .{},
    attn_sink: ?[]const f32 = null,
    ffn_norm: []const f32 = &.{},
    gate_weight: QuantMatrix = .{},
    gate_bias: ?[]const f32 = null,
    shared_w1: ?QuantMatrix = null,
    shared_w2: ?QuantMatrix = null,
    shared_w3: ?QuantMatrix = null,
    experts: []ExpertRef = &.{},
};

const DSpark3LSmeRunner = struct {
    dispatcher: sme.SmeDispatcher,
    act_u16: []u16,
    out_f32: []f32,
};

const DSpark3LAmfSource = union(enum) {
    bytes: struct {
        parsed: *const amf.Parsed,
        raw: []const u8,
    },
    file: *const amf.AmfFile,

    fn parsed(self: DSpark3LAmfSource) *const amf.Parsed {
        return switch (self) {
            .bytes => |b| b.parsed,
            .file => |f| &f.parsed,
        };
    }

    fn loadBlob(
        self: DSpark3LAmfSource,
        entry: *const amf.TensorEntry,
        arena: *arena_mod.WiredArena,
        verify_crc: bool,
    ) LoadError![]align(page.APPLE_PAGE_SIZE) u8 {
        switch (self) {
            .bytes => |b| {
                if (verify_crc) return b.parsed.loadBlobFromBytes(b.raw, entry, arena);
                const start: usize = @intCast(entry.blob_offset);
                const len: usize = @intCast(entry.blob_bytes);
                if (start + len > b.raw.len) return error.BlobOutOfBounds;
                const dst = try arena.allocPages(len);
                @memcpy(dst, b.raw[start .. start + len]);
                return dst;
            },
            .file => |f| {
                if (verify_crc) return f.loadBlob(entry, arena);
                const len: usize = @intCast(entry.blob_bytes);
                const dst = try arena.allocPages(len);
                var done: usize = 0;
                while (done < len) {
                    const rc = std.c.pread(f.fd, dst.ptr + done, len - done, @intCast(entry.blob_offset + done));
                    if (rc < 0) {
                        if (std.c.errno(rc) == .INTR) continue;
                        return error.ReadFailed;
                    }
                    if (rc == 0) return error.ShortRead;
                    done += @intCast(rc);
                }
                return dst;
            },
        }
    }

    fn loadRawAlloc(
        self: DSpark3LAmfSource,
        allocator: std.mem.Allocator,
        entry: *const amf.TensorEntry,
        verify_crc: bool,
    ) LoadError![]u8 {
        const len: usize = @intCast(entry.blob_bytes);
        const buf = try allocator.alloc(u8, len);
        errdefer allocator.free(buf);
        switch (self) {
            .bytes => |b| {
                const start: usize = @intCast(entry.blob_offset);
                if (start + len > b.raw.len) return error.BlobOutOfBounds;
                @memcpy(buf, b.raw[start .. start + len]);
            },
            .file => |f| {
                var done: usize = 0;
                while (done < len) {
                    const rc = std.c.pread(f.fd, buf.ptr + done, len - done, @intCast(entry.blob_offset + done));
                    if (rc < 0) {
                        if (std.c.errno(rc) == .INTR) continue;
                        return error.ReadFailed;
                    }
                    if (rc == 0) return error.ShortRead;
                    done += @intCast(rc);
                }
            },
        }
        if (verify_crc) {
            if (std.hash.Crc32.hash(buf) != entry.blob_crc32) return error.BlobChecksumMismatch;
        }
        return buf;
    }
};

/// Real-HF `DeepSeek-V4.1-Flash-1017` `3L DSpark` (`mtp.0..2.*`) proposer runtime.
///
/// All weights live in a single 16 KiB-aligned `weights_arena` (compact native
/// quantized bytes, never expanded to FP32), cold routed experts stay on NVMe and
/// stream through `cold_pool` (`F_NOCACHE` pread + `C27` hot cache), and all
/// forward-pass scratch lives in `workspace.arena` (`0` hot-path heap allocations).
pub const DSpark3LRuntime = struct {
    config: DSpark3LConfig,
    weights_arena: arena_mod.WiredArena,
    workspace: DSparkWorkspace,
    cold_pool: ColdExpertSlotPool,
    amf_file: ?amf.AmfFile = null,
    owned_bytes_ref: ?[]const u8 = null,
    sme_runner: ?DSpark3LSmeRunner = null,

    // Target callbacks (zero copy, zero alloc).
    embed_hook: ?EmbedCallback = null,
    lm_head_hook: ?LmHeadHook = null,

    // mtp.0 target hidden fusion weights.
    main_proj: QuantMatrix = .{},
    main_norm: []const f32 = &.{},

    // mtp.0..num_layers-1 transformer blocks.
    layers: [MAX_DSPARK_3L_LAYERS]DSpark3LLayer = @splat(.{}),
    wkv_dim: u32 = 0,
    wo_a_in_dim: u32 = 0,
    wo_a_out_dim: u32 = 0,
    wo_b_in_dim: u32 = 0,

    // mtp.{last} output heads.
    final_norm: []const f32 = &.{},
    markov_embed: ?QuantMatrix = null,
    markov_head: ?QuantMatrix = null,
    confidence_proj: ?QuantMatrix = null,

    // Preallocated scratch slices inside `workspace.arena` (0 hot-path allocs).
    window_kv_cache: []f32 = &.{}, // [num_layers, window_size, wkv_dim]
    window_pos: u32 = 0,
    main_concat: []f32 = &.{}, // [num_target_taps * H]
    main_x: []f32 = &.{}, // [H]
    h_streams: []f32 = &.{}, // [block_size, hc_mult, H]
    h_streams_res: []f32 = &.{}, // [block_size, hc_mult, H]
    pre_mix: []f32 = &.{}, // [block_size, hc_mult]
    next_pre_mix: []f32 = &.{}, // [block_size, hc_mult]
    post_mix: []f32 = &.{}, // [block_size, hc_mult]
    comb_mix: []f32 = &.{}, // [block_size, hc_mult * hc_mult]
    hc_flat_buf: []f32 = &.{}, // [hc_mult * H]
    sub_in: []f32 = &.{}, // [block_size, H]
    sub_out: []f32 = &.{}, // [block_size, H]
    q_lora_buf: []f32 = &.{}, // [q_lora_rank]
    q_all: []f32 = &.{}, // [block_size, n_heads * head_dim]
    block_kv: []f32 = &.{}, // [block_size, wkv_dim]
    attn_out: []f32 = &.{}, // [n_heads * head_dim]
    wo_a_out: []f32 = &.{}, // [wo_a_out_dim]
    wo_b_in: []f32 = &.{}, // [wo_b_in_dim]
    moe_scores: []f32 = &.{}, // [n_routed_experts]
    moe_gate: []f32 = &.{}, // [moe_inter]
    moe_up: []f32 = &.{}, // [moe_inter]
    moe_tmp: []f32 = &.{}, // [H]
    x_final: []f32 = &.{}, // [block_size, H]
    markov_embeds: []f32 = &.{}, // [block_size, markov_rank]
    slot_confidences: []f32 = &.{}, // [block_size]
    conf_in: []f32 = &.{}, // [H + markov_rank]
    row_scratch: []f32 = &.{}, // [max_k]

    // Pending multi-tap conditioning handed from `SpecEngine` prior to `proposeInto`.
    pending_taps: [MAX_DSPARK_TARGET_TAPS][]const f32 = @splat(&.{}),
    pending_tap_count: usize = 0,
    pending_start_pos: u32 = 0,
    has_pending_taps: bool = false,

    // Telemetry counters.
    total_markov_evals: u64 = 0,
    total_lm_head_sweeps_avoided: u64 = 0,
    total_confidence_gated_positions: u64 = 0,
    sme2_eval_count: u64 = 0,

    /// Auto-detects whether an `.amf` contains real-HF `3L DSpark` (`mtp.0..2.*`) tensors.
    pub fn isRealHf3LAmf(parsed: *const amf.Parsed) bool {
        return findTensorFlex(parsed, "mtp.0.main_proj") != null;
    }

    pub fn loadFromFile(
        allocator: std.mem.Allocator,
        amf_path: [:0]const u8,
        cfg_opt: ?DSpark3LConfig,
        opts: DSpark3LLoadOptions,
    ) LoadError!*DSpark3LRuntime {
        var file = try amf.AmfFile.open(allocator, amf_path);
        var keep_file = false;
        errdefer if (!keep_file) file.close(allocator);

        var cfg_resolved: ?DSpark3LConfig = cfg_opt;
        if (cfg_resolved == null) {
            if (try file.readMetadataAlloc(allocator, amf.META_CONFIG_JSON)) |meta_json| {
                defer allocator.free(meta_json);
                cfg_resolved = try DSpark3LConfig.fromJson(allocator, meta_json);
            }
        }
        const rt = try loadInternal(allocator, .{ .file = &file }, cfg_resolved, opts);
        if (rt.cold_pool.stats.cold_experts > 0) {
            rt.amf_file = file;
            rt.cold_pool.backing_fd = rt.amf_file.?.fd;
            keep_file = true;
        } else {
            file.close(allocator);
        }
        return rt;
    }

    pub fn loadFromBytes(
        allocator: std.mem.Allocator,
        amf_bytes: []const u8,
        cfg_opt: ?DSpark3LConfig,
        opts: DSpark3LLoadOptions,
    ) LoadError!*DSpark3LRuntime {
        var parsed = try amf.Parsed.parseBytes(allocator, amf_bytes);
        defer parsed.deinit(allocator);

        var cfg_resolved: ?DSpark3LConfig = cfg_opt;
        if (cfg_resolved == null) {
            if (parsed.find(amf.META_CONFIG_JSON)) |me| {
                const start: usize = @intCast(me.blob_offset);
                const blen: usize = @intCast(me.blob_bytes);
                const dlen: usize = @intCast(me.data_bytes);
                if (start + blen > amf_bytes.len or dlen > blen) return error.BlobOutOfBounds;
                if (opts.verify_crcs and std.hash.Crc32.hash(amf_bytes[start .. start + blen]) != me.blob_crc32) {
                    return error.BlobChecksumMismatch;
                }
                cfg_resolved = try DSpark3LConfig.fromJson(allocator, amf_bytes[start .. start + dlen]);
            }
        }
        const rt = try loadInternal(allocator, .{ .bytes = .{ .parsed = &parsed, .raw = amf_bytes } }, cfg_resolved, opts);
        if (rt.cold_pool.stats.cold_experts > 0) {
            rt.owned_bytes_ref = amf_bytes;
            rt.cold_pool.backing_bytes = amf_bytes;
        }
        return rt;
    }

    pub fn deinit(self: *DSpark3LRuntime, allocator: std.mem.Allocator) void {
        if (self.sme_runner) |*sr| sr.dispatcher.deinit();
        self.sme_runner = null;
        for (0..self.config.num_layers) |l| {
            if (self.layers[l].experts.len > 0) allocator.free(self.layers[l].experts);
            self.layers[l].experts = &.{};
        }
        self.cold_pool.deinit(allocator);
        if (self.amf_file) |*f| {
            f.close(allocator);
            self.amf_file = null;
        }
        self.workspace.deinit();
        self.weights_arena.deinit();
        allocator.destroy(self);
    }

    pub fn totalWiredBytes(self: *const DSpark3LRuntime) u64 {
        return @as(u64, self.weights_arena.bytes().len) +
            self.workspace.totalWiredBytes() +
            self.cold_pool.stats.slot_pool_bytes;
    }

    pub fn coldStats(self: *const DSpark3LRuntime) ColdDSparkStats {
        return self.cold_pool.stats;
    }

    pub fn setColdExpertMinWeight(self: *DSpark3LRuntime, min_weight: f32) void {
        if (std.math.isFinite(min_weight) and min_weight >= 0.0) {
            self.config.cold_expert_min_weight = min_weight;
        }
    }

    pub fn setConfidenceThreshold(self: *DSpark3LRuntime, threshold: f32) void {
        if (std.math.isFinite(threshold) and threshold >= 0.0) {
            self.config.confidence_threshold = threshold;
        }
    }

    pub fn lastSlotConfidences(self: *const DSpark3LRuntime) []const f32 {
        return self.slot_confidences[0..self.config.block_size];
    }

    pub fn resetWindow(self: *DSpark3LRuntime) void {
        @memset(self.window_kv_cache, 0.0);
        self.window_pos = 0;
        self.has_pending_taps = false;
        self.pending_tap_count = 0;
        self.pending_start_pos = 0;
    }

    pub fn setPendingTaps(self: *DSpark3LRuntime, taps: []const []const f32, start_pos: u32) void {
        const take = @min(taps.len, MAX_DSPARK_TARGET_TAPS);
        for (0..take) |i| self.pending_taps[i] = taps[i];
        self.pending_tap_count = take;
        self.pending_start_pos = start_pos;
        self.has_pending_taps = (take > 0);
    }

    /// Seeds every stage's `DSparkAttention` sliding-window KV cache from `target_taps`
    /// at sequence position `start_pos` (`model.py:1046-1054`). Zero heap allocations.
    pub fn seedWindow(self: *DSpark3LRuntime, target_taps: []const []const f32, start_pos: u32) Error!void {
        const h: usize = self.config.hidden_size;
        const n_taps: usize = self.config.num_target_taps;
        if (target_taps.len == 0) return error.DimensionMismatch;
        for (0..n_taps) |ti| {
            const src = if (ti < target_taps.len) target_taps[ti] else target_taps[target_taps.len - 1];
            if (src.len != h) return error.DimensionMismatch;
            @memcpy(self.main_concat[ti * h ..][0..h], src);
        }
        try self.evalMatVec(&self.main_proj, self.main_concat[0 .. n_taps * h], self.main_x[0..h], false);
        applyRmsNormInPlace(self.main_x[0..h], self.main_norm, self.config.rms_eps);

        const win: usize = self.config.window_size;
        const wkv_d: usize = self.wkv_dim;
        const slot_idx: usize = @as(usize, start_pos) % win;
        const rd: usize = @min(@as(usize, self.config.qk_rope_head_dim), wkv_d);

        for (0..self.config.num_layers) |l| {
            const layer = &self.layers[l];
            const dst_kv = self.window_kv_cache[(l * win + slot_idx) * wkv_d ..][0..wkv_d];
            try self.evalMatVec(&layer.wkv, self.main_x[0..h], dst_kv, false);
            applyRmsNormInPlace(dst_kv[0..layer.kv_norm.len], layer.kv_norm, self.config.rms_eps);
            if (rd >= 2) {
                applyRopeInterleavedInPlace(dst_kv[wkv_d - rd .. wkv_d], start_pos, self.config.rope_theta, false);
            }
        }
        if (start_pos + 1 > self.window_pos) {
            self.window_pos = start_pos + 1;
        }
    }

    fn evalMatVec(
        self: *DSpark3LRuntime,
        mat: *const QuantMatrix,
        x: []const f32,
        out: []f32,
        accumulate: bool,
    ) Error!void {
        const n: usize = mat.n;
        const k: usize = mat.k;
        if (x.len < k or out.len < n) return error.DimensionMismatch;
        if (mat.layout == .sme2_panel64) {
            if (mat.sme_weight) |sw| {
                if (self.sme_runner) |*sr| {
                    if (k <= sr.act_u16.len and n <= sr.out_f32.len) {
                        const act_bf16 = sw.dtype.activationIsBf16();
                        for (0..k) |ci| {
                            sr.act_u16[ci] = if (act_bf16)
                                quant.f32ToBf16(x[ci])
                            else
                                @bitCast(@as(f16, @floatCast(x[ci])));
                        }
                        const out_bytes = std.mem.sliceAsBytes(sr.out_f32[0..n]);
                        sme.evalSmeLinear(&sr.dispatcher, sw, sr.act_u16[0..k], null, out_bytes, 1, .{
                            .out_dtype = .f32,
                        }) catch return error.CallbackFailed;
                        if (accumulate) {
                            for (0..n) |r| out[r] += sr.out_f32[r];
                        } else {
                            @memcpy(out[0..n], sr.out_f32[0..n]);
                        }
                        self.sme2_eval_count += 1;
                        return;
                    }
                }
            }
            return error.InvalidWeights;
        }
        try matVecQuantF32(mat, x[0..k], out[0..n], self.row_scratch[0..k], accumulate);
    }

    fn computeHcMixes(
        self: *DSpark3LRuntime,
        hc_w: *const HcBlockWeights,
        pos_streams: []const f32, // [hc_mult * H]
        out_pre: []f32,
        out_post: []f32,
        out_comb: []f32,
    ) Error!void {
        const hc: usize = self.config.hc_mult;
        const h: usize = self.config.hidden_size;
        const hc_dim = hc * h;
        const eps = self.config.hc_eps;

        var sum_sq: f64 = 0.0;
        for (pos_streams[0..hc_dim]) |v| {
            const vf: f64 = v;
            sum_sq += vf * vf;
        }
        const rsqrt_val: f32 = @floatCast(1.0 / @sqrt(sum_sq / @as(f64, @floatFromInt(hc_dim)) + @as(f64, self.config.rms_eps)));

        var mixes_buf: [MAX_HC_MULT * (2 + MAX_HC_MULT)]f32 = @splat(0.0);
        const mix_hc = (2 + hc) * hc;

        if (hc_w.is_full_hf_shape) {
            try self.evalMatVec(&hc_w.fn_mat, pos_streams[0..hc_dim], mixes_buf[0..mix_hc], false);
            for (0..mix_hc) |i| mixes_buf[i] *= rsqrt_val;

            const s0 = hc_w.scale_f32[0];
            const s1 = hc_w.scale_f32[1];
            const s2 = hc_w.scale_f32[2];
            for (0..hc) |j| {
                out_pre[j] = sigmoidF32(mixes_buf[j] * s0 + hc_w.base_f32[j]) + eps;
                out_post[j] = 2.0 * sigmoidF32(mixes_buf[hc + j] * s1 + hc_w.base_f32[hc + j]);
            }
            for (0..hc) |j| {
                for (0..hc) |k| {
                    const idx = 2 * hc + j * hc + k;
                    out_comb[j * hc + k] = mixes_buf[idx] * s2 + hc_w.base_f32[idx];
                }
            }
        } else {
            // Compact synth shape: `hc_fn` is `[H, hc_mult]` and `hc_base` is `[hc_mult, hc_mult]`.
            const mean_h = self.hc_flat_buf[0..h];
            @memset(mean_h, 0.0);
            const inv_hc: f32 = 1.0 / @as(f32, @floatFromInt(hc));
            for (0..hc) |s| {
                const st = pos_streams[s * h ..][0..h];
                for (0..h) |d| mean_h[d] += st[d] * inv_hc;
            }
            var proj_s: [MAX_HC_MULT]f32 = @splat(0.0);
            // Compute `proj_s[j] = sum_d mean_h[d] * fn_mat[d, j] * rsqrt`.
            for (0..h) |d| {
                try dequantMatrixRow(&hc_w.fn_mat, d, self.row_scratch[0..hc]);
                const md = mean_h[d] * rsqrt_val;
                for (0..hc) |j| proj_s[j] += md * self.row_scratch[j];
            }
            const s0 = hc_w.scale_f32[0];
            const s1 = hc_w.scale_f32[1];
            const s2 = hc_w.scale_f32[2];
            for (0..hc) |j| {
                out_pre[j] = sigmoidF32(proj_s[j] * s0) + eps;
                out_post[j] = 2.0 * sigmoidF32(proj_s[j] * s1);
            }
            for (0..hc) |j| {
                for (0..hc) |k| {
                    const base_v = if (j * hc + k < hc_w.base_f32.len) hc_w.base_f32[j * hc + k] else 0.0;
                    out_comb[j * hc + k] = base_v + 0.5 * (proj_s[j] + proj_s[k]) * s2;
                }
            }
        }

        // Exact Sinkhorn-20 doubly stochastic projection (`kernel.py:436-459`):
        // 1. row softmax + eps
        for (0..hc) |j| {
            var rmax: f32 = -std.math.inf(f32);
            for (0..hc) |k| {
                const v = out_comb[j * hc + k];
                if (v > rmax) rmax = v;
            }
            var rsum: f32 = 0.0;
            for (0..hc) |k| {
                const ev = @exp(out_comb[j * hc + k] - rmax);
                out_comb[j * hc + k] = ev;
                rsum += ev;
            }
            const inv_r: f32 = if (rsum > 0.0) 1.0 / rsum else 0.0;
            for (0..hc) |k| out_comb[j * hc + k] = out_comb[j * hc + k] * inv_r + eps;
        }
        // 2. initial col normalization
        for (0..hc) |k| {
            var csum: f32 = 0.0;
            for (0..hc) |j| csum += out_comb[j * hc + k];
            const inv_c = 1.0 / (csum + eps);
            for (0..hc) |j| out_comb[j * hc + k] *= inv_c;
        }
        // 3. `hc_sinkhorn_iters - 1` alternating row & col normalizations
        var it: u32 = 1;
        while (it < self.config.hc_sinkhorn_iters) : (it += 1) {
            for (0..hc) |j| {
                var rsum: f32 = 0.0;
                for (0..hc) |k| rsum += out_comb[j * hc + k];
                const inv_r = 1.0 / (rsum + eps);
                for (0..hc) |k| out_comb[j * hc + k] *= inv_r;
            }
            for (0..hc) |k| {
                var csum: f32 = 0.0;
                for (0..hc) |j| csum += out_comb[j * hc + k];
                const inv_c = 1.0 / (csum + eps);
                for (0..hc) |j| out_comb[j * hc + k] *= inv_c;
            }
        }
    }

    fn runStageAttention(
        self: *DSpark3LRuntime,
        stage_idx: usize,
        start_pos: u32,
    ) Error!void {
        const layer = &self.layers[stage_idx];
        const bsz: usize = self.config.block_size;
        const h: usize = self.config.hidden_size;
        const n_heads: usize = self.config.num_attention_heads;
        const hd: usize = self.config.head_dim;
        const q_dim = n_heads * hd;
        const rd: usize = @min(@as(usize, self.config.qk_rope_head_dim), hd);
        const nope_d: usize = hd - rd;
        const wkv_d: usize = self.wkv_dim;
        const win: usize = self.config.window_size;
        const softmax_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

        // 1. Seed `main_x` into this stage's sliding-window KV ring at `start_pos % win`.
        const win_slot: usize = @as(usize, start_pos) % win;
        const win_dst = self.window_kv_cache[(stage_idx * win + win_slot) * wkv_d ..][0..wkv_d];
        try self.evalMatVec(&layer.wkv, self.main_x[0..h], win_dst, false);
        applyRmsNormInPlace(win_dst[0..layer.kv_norm.len], layer.kv_norm, self.config.rms_eps);
        if (rd >= 2 and rd <= wkv_d) {
            applyRopeInterleavedInPlace(win_dst[wkv_d - rd .. wkv_d], start_pos, self.config.rope_theta, false);
        }
        const valid_win: usize = @min(win, @as(usize, @max(self.window_pos, start_pos + 1)));

        // 2. Project Q (`wq_a -> q_norm -> wq_b -> RoPE`) and block KV (`wkv -> kv_norm -> RoPE`).
        for (0..bsz) |p| {
            const x_row = self.sub_in[p * h ..][0..h];
            const pos_p: u32 = start_pos + 1 + @as(u32, @intCast(p));

            try self.evalMatVec(&layer.wq_a, x_row, self.q_lora_buf[0..layer.q_norm.len], false);
            applyRmsNormInPlace(self.q_lora_buf[0..layer.q_norm.len], layer.q_norm, self.config.rms_eps);
            const q_row = self.q_all[p * q_dim ..][0..q_dim];
            try self.evalMatVec(&layer.wq_b, self.q_lora_buf[0..layer.q_norm.len], q_row, false);
            if (rd >= 2) {
                for (0..n_heads) |hi| {
                    applyRopeInterleavedInPlace(q_row[hi * hd + nope_d ..][0..rd], pos_p, self.config.rope_theta, false);
                }
            }

            const kv_row = self.block_kv[p * wkv_d ..][0..wkv_d];
            try self.evalMatVec(&layer.wkv, x_row, kv_row, false);
            applyRmsNormInPlace(kv_row[0..layer.kv_norm.len], layer.kv_norm, self.config.rms_eps);
            if (rd >= 2 and rd <= wkv_d) {
                applyRopeInterleavedInPlace(kv_row[wkv_d - rd .. wkv_d], pos_p, self.config.rope_theta, false);
            }
        }

        // 3. Sparse window + block attention with `attn_sink` + inverse RoPE + grouped `wo_a`/`wo_b`.
        const n_groups: usize = self.config.o_groups;
        const o_lora: usize = self.config.o_lora_rank;
        const total_kv = valid_win + bsz;

        for (0..bsz) |p| {
            const pos_p: u32 = start_pos + 1 + @as(u32, @intCast(p));
            const q_row = self.q_all[p * q_dim ..][0..q_dim];
            const o_row = self.attn_out[0..q_dim];
            @memset(o_row, 0.0);

            for (0..n_heads) |hi| {
                const qh = q_row[hi * hd ..][0..hd];
                const oh = o_row[hi * hd ..][0..hd];

                var max_s: f32 = -1e30;
                for (0..total_kv) |ki| {
                    const kv_vec: []const f32 = if (ki < valid_win)
                        self.window_kv_cache[(stage_idx * win + ki) * wkv_d ..][0..wkv_d]
                    else
                        self.block_kv[(ki - valid_win) * wkv_d ..][0..wkv_d];
                    var dot: f32 = 0.0;
                    const kv_nope_len = @min(nope_d, wkv_d);
                    for (0..kv_nope_len) |d| dot += qh[d] * kv_vec[d];
                    if (rd > 0 and rd <= wkv_d) {
                        const kv_rope = kv_vec[wkv_d - rd .. wkv_d];
                        for (0..rd) |d| dot += qh[nope_d + d] * kv_rope[d];
                    }
                    const sc = dot * softmax_scale;
                    if (sc > max_s) max_s = sc;
                }
                if (layer.attn_sink) |sink| {
                    if (hi < sink.len and sink[hi] > max_s) max_s = sink[hi];
                }

                var sum_exp: f32 = 0.0;
                for (0..total_kv) |ki| {
                    const kv_vec: []const f32 = if (ki < valid_win)
                        self.window_kv_cache[(stage_idx * win + ki) * wkv_d ..][0..wkv_d]
                    else
                        self.block_kv[(ki - valid_win) * wkv_d ..][0..wkv_d];
                    var dot: f32 = 0.0;
                    const kv_nope_len = @min(nope_d, wkv_d);
                    for (0..kv_nope_len) |d| dot += qh[d] * kv_vec[d];
                    if (rd > 0 and rd <= wkv_d) {
                        const kv_rope = kv_vec[wkv_d - rd .. wkv_d];
                        for (0..rd) |d| dot += qh[nope_d + d] * kv_rope[d];
                    }
                    const w = @exp(dot * softmax_scale - max_s);
                    sum_exp += w;
                    for (0..kv_nope_len) |d| oh[d] += w * kv_vec[d];
                    if (rd > 0 and rd <= wkv_d) {
                        const kv_rope = kv_vec[wkv_d - rd .. wkv_d];
                        for (0..rd) |d| oh[nope_d + d] += w * kv_rope[d];
                    }
                }
                if (layer.attn_sink) |sink| {
                    if (hi < sink.len) sum_exp += @exp(sink[hi] - max_s);
                }
                const inv_s: f32 = if (sum_exp > 0.0) 1.0 / sum_exp else 0.0;
                for (0..hd) |d| oh[d] *= inv_s;

                // Inverse RoPE on output tail (`model.py:1070`).
                if (rd >= 2) {
                    applyRopeInterleavedInPlace(oh[nope_d .. nope_d + rd], pos_p, self.config.rope_theta, true);
                }
            }

            // Grouped `wo_a` (`einsum("bsgd,grd->bsgr", o, wo_a)`) -> `wo_b`.
            const group_in = self.wo_a_in_dim;
            const out_per_group = self.wo_a_out_dim / n_groups;
            @memset(self.wo_a_out[0..self.wo_a_out_dim], 0.0);
            for (0..n_groups) |g| {
                const g_off = g * group_in;
                const x_grp: []const f32 = if (g_off + group_in <= q_dim)
                    o_row[g_off .. g_off + group_in]
                else
                    o_row[0..@min(group_in, q_dim)];
                const row_start = g * out_per_group;
                for (0..out_per_group) |r| {
                    try dequantMatrixRow(&layer.wo_a, row_start + r, self.row_scratch[0..group_in]);
                    var acc: f32 = 0.0;
                    for (0..x_grp.len) |c| acc += self.row_scratch[c] * x_grp[c];
                    self.wo_a_out[row_start + r] = acc;
                }
            }

            const y_dst = self.sub_out[p * h ..][0..h];
            if (self.wo_b_in_dim == self.wo_a_out_dim) {
                try self.evalMatVec(&layer.wo_b, self.wo_a_out[0..self.wo_a_out_dim], y_dst, false);
            } else {
                // Synth fixture layout (`wo_a [o_groups*o_lora, group_in]`, `wo_b [H, o_lora]`):
                // reduce across groups into `[o_lora]` before `wo_b`.
                const fold_len = @min(self.wo_b_in_dim, o_lora);
                @memset(self.wo_b_in[0..self.wo_b_in_dim], 0.0);
                const inv_g: f32 = 1.0 / @as(f32, @floatFromInt(n_groups));
                for (0..n_groups) |g| {
                    const src_g = self.wo_a_out[g * out_per_group ..][0..@min(fold_len, out_per_group)];
                    for (src_g, 0..) |val, i| self.wo_b_in[i] += val * inv_g;
                }
                try self.evalMatVec(&layer.wo_b, self.wo_b_in[0..self.wo_b_in_dim], y_dst, false);
            }
        }
    }

    fn evalSwigluExpert(
        self: *DSpark3LRuntime,
        w1: *const QuantMatrix,
        w2: *const QuantMatrix,
        w3: *const QuantMatrix,
        x: []const f32,
        scale_opt: ?f32,
        out_accum: []f32,
    ) Error!void {
        const inter: usize = w1.n;
        const limit = self.config.swiglu_limit;
        try self.evalMatVec(w1, x, self.moe_gate[0..inter], false);
        try self.evalMatVec(w3, x, self.moe_up[0..inter], false);
        for (0..inter) |i| {
            var g = self.moe_gate[i];
            var u = self.moe_up[i];
            if (limit > 0.0) {
                if (g > limit) g = limit;
                if (u > limit) u = limit else if (u < -limit) u = -limit;
            }
            var act = siluF32(g) * u;
            if (scale_opt) |sw| act *= sw;
            self.moe_gate[i] = act;
        }
        try self.evalMatVec(w2, self.moe_gate[0..inter], out_accum, true);
    }

    fn runStageMoe(self: *DSpark3LRuntime, stage_idx: usize) Error!void {
        const layer = &self.layers[stage_idx];
        const bsz: usize = self.config.block_size;
        const h: usize = self.config.hidden_size;
        const n_exp: usize = layer.experts.len;
        const top_k: usize = @min(@as(usize, self.config.routed_top_k), n_exp);

        for (0..bsz) |p| {
            const x_row = self.sub_in[p * h ..][0..h];
            const y_dst = self.sub_out[p * h ..][0..h];
            @memset(y_dst, 0.0);

            if (n_exp > 0 and top_k > 0) {
                // `sqrtsoftplus` + `noaux_tc` routing (`model.py:813-829`).
                try self.evalMatVec(&layer.gate_weight, x_row, self.moe_scores[0..n_exp], false);
                for (0..n_exp) |e| {
                    self.moe_scores[e] = @sqrt(softplusF32(self.moe_scores[e]));
                }

                var sel_idx: [MAX_DSPARK_ROUTED_TOP_K]u16 = @splat(0);
                var sel_key: [MAX_DSPARK_ROUTED_TOP_K]f32 = @splat(-std.math.inf(f32));
                for (0..n_exp) |e| {
                    const b: f32 = if (layer.gate_bias) |gb| (if (e < gb.len) gb[e] else 0.0) else 0.0;
                    const key = self.moe_scores[e] + b;
                    const last = top_k - 1;
                    if (key > sel_key[last] or (key == sel_key[last] and @as(u16, @intCast(e)) < sel_idx[last])) {
                        var pos = last;
                        while (pos > 0) : (pos -= 1) {
                            if (!(key > sel_key[pos - 1] or (key == sel_key[pos - 1] and @as(u16, @intCast(e)) < sel_idx[pos - 1]))) break;
                            sel_key[pos] = sel_key[pos - 1];
                            sel_idx[pos] = sel_idx[pos - 1];
                        }
                        sel_key[pos] = key;
                        sel_idx[pos] = @intCast(e);
                    }
                }

                var w_sum: f32 = 0.0;
                var sel_w: [MAX_DSPARK_ROUTED_TOP_K]f32 = @splat(0.0);
                for (0..top_k) |ki| {
                    sel_w[ki] = self.moe_scores[sel_idx[ki]];
                    w_sum += sel_w[ki];
                }
                const inv_sum: f32 = if (top_k > 1) 1.0 / (w_sum + 1e-20) else 1.0;
                for (0..top_k) |ki| {
                    const ew = sel_w[ki] * inv_sum * self.config.routed_scaling_factor;
                    const eid = sel_idx[ki];
                    switch (layer.experts[eid]) {
                        .wired => |tr| {
                            try self.evalSwigluExpert(&tr.w1, &tr.w2, &tr.w3, x_row, ew, y_dst);
                        },
                        .cold => |cd| {
                            const fetched = try self.cold_pool.fetchExpert(
                                @intCast(stage_idx),
                                eid,
                                cd.w1,
                                cd.w2,
                                cd.w3,
                                ew,
                                self.config.cold_expert_min_weight,
                            );
                            if (fetched) |tr| {
                                try self.evalSwigluExpert(&tr.w1, &tr.w2, &tr.w3, x_row, ew, y_dst);
                            }
                        },
                    }
                }
            }

            // Shared expert (`model.py:905`).
            if (layer.shared_w1 != null and layer.shared_w2 != null and layer.shared_w3 != null) {
                try self.evalSwigluExpert(
                    &layer.shared_w1.?,
                    &layer.shared_w2.?,
                    &layer.shared_w3.?,
                    x_row,
                    null,
                    y_dst,
                );
            }
        }
    }

    /// Runs the complete real-HF `3L DSpark` forward pass (`mtp.0` target fusion ->
    /// 3 stages `mHC` + `DSparkAttention` + MoE -> `mtp.2` `norm` + `markov_head` +
    /// `confidence_head`) with zero dynamic heap allocations.
    pub fn predictBlock(
        self: *DSpark3LRuntime,
        root_token: u32,
        target_taps: []const []const f32,
        start_pos: u32,
    ) Error!struct {
        mean_confidence: f32,
        min_confidence: f32,
        markov_evals: u32,
        lm_head_sweeps_avoided: u32,
    } {
        const h: usize = self.config.hidden_size;
        const bsz: usize = self.config.block_size;
        const v: usize = self.config.vocab_size;
        const k: usize = self.config.top_k_per_pos;
        const hc: usize = self.config.hc_mult;
        const n_taps: usize = self.config.num_target_taps;
        if (target_taps.len == 0) return error.DimensionMismatch;

        // 1. Multi-layer target hidden fusion (`mtp.0.main_proj` -> `mtp.0.main_norm`).
        for (0..n_taps) |ti| {
            const src = if (ti < target_taps.len) target_taps[ti] else target_taps[target_taps.len - 1];
            if (src.len != h) return error.DimensionMismatch;
            @memcpy(self.main_concat[ti * h ..][0..h], src);
        }
        try self.evalMatVec(&self.main_proj, self.main_concat[0 .. n_taps * h], self.main_x[0..h], false);
        applyRmsNormInPlace(self.main_x[0..h], self.main_norm, self.config.rms_eps);

        // 2. Embed draft block `[root_token, noise, ..., noise]` and tile across `hc_mult` streams.
        const pos0_tmp = self.moe_tmp[0..h];
        if (self.embed_hook) |eh| {
            try eh.read_fn(eh.ctx, root_token, pos0_tmp);
        } else {
            @memcpy(pos0_tmp, self.main_x[0..h]);
        }
        for (0..hc) |s| {
            @memcpy(self.h_streams[s * h ..][0..h], pos0_tmp);
        }
        if (bsz > 1) {
            if (self.embed_hook) |eh| {
                try eh.read_fn(eh.ctx, self.config.noise_token_id, pos0_tmp);
            } else {
                for (0..h) |d| pos0_tmp[d] = self.main_x[d] * 0.5;
            }
            for (1..bsz) |p| {
                const p_base = p * hc * h;
                for (0..hc) |s| {
                    @memcpy(self.h_streams[p_base + s * h ..][0..h], pos0_tmp);
                }
            }
        }

        // Initial one-hot `pre_mix` (`make_identity_pre_mix`, `model.py:1162`).
        @memset(self.pre_mix[0 .. bsz * hc], 0.0);
        for (0..bsz) |p| self.pre_mix[p * hc] = 1.0;

        // 3. Execute 3 `DSparkBlock` stages (`mtp.0`, `mtp.1`, `mtp.2`).
        for (0..self.config.num_layers) |l| {
            const layer = &self.layers[l];

            // --- Attention sub-block ---
            @memcpy(self.h_streams_res[0 .. bsz * hc * h], self.h_streams[0 .. bsz * hc * h]);
            for (0..bsz) |p| {
                const st_p = self.h_streams[p * hc * h ..][0 .. hc * h];
                try self.computeHcMixes(
                    &layer.hc_attn,
                    st_p,
                    self.next_pre_mix[p * hc ..][0..hc],
                    self.post_mix[p * hc ..][0..hc],
                    self.comb_mix[p * hc * hc ..][0 .. hc * hc],
                );
                // `hc_pre(x, pre_mix)` -> `attn_norm`
                const sub_p = self.sub_in[p * h ..][0..h];
                hcCollapseStreams(st_p, self.pre_mix[p * hc ..][0..hc], hc, h, sub_p);
                applyRmsNormInPlace(sub_p, layer.attn_norm, self.config.rms_eps);
            }
            try self.runStageAttention(l, start_pos);
            for (0..bsz) |p| {
                hcExpandMixStreams(
                    self.sub_out[p * h ..][0..h],
                    self.h_streams_res[p * hc * h ..][0 .. hc * h],
                    self.post_mix[p * hc ..][0..hc],
                    self.comb_mix[p * hc * hc ..][0 .. hc * hc],
                    hc,
                    h,
                    self.h_streams[p * hc * h ..][0 .. hc * h],
                );
            }
            @memcpy(self.pre_mix[0 .. bsz * hc], self.next_pre_mix[0 .. bsz * hc]);

            // --- MoE FFN sub-block ---
            @memcpy(self.h_streams_res[0 .. bsz * hc * h], self.h_streams[0 .. bsz * hc * h]);
            for (0..bsz) |p| {
                const st_p = self.h_streams[p * hc * h ..][0 .. hc * h];
                try self.computeHcMixes(
                    &layer.hc_ffn,
                    st_p,
                    self.next_pre_mix[p * hc ..][0..hc],
                    self.post_mix[p * hc ..][0..hc],
                    self.comb_mix[p * hc * hc ..][0 .. hc * hc],
                );
                const sub_p = self.sub_in[p * h ..][0..h];
                hcCollapseStreams(st_p, self.pre_mix[p * hc ..][0..hc], hc, h, sub_p);
                applyRmsNormInPlace(sub_p, layer.ffn_norm, self.config.rms_eps);
            }
            try self.runStageMoe(l);
            for (0..bsz) |p| {
                hcExpandMixStreams(
                    self.sub_out[p * h ..][0..h],
                    self.h_streams_res[p * hc * h ..][0 .. hc * h],
                    self.post_mix[p * hc ..][0..hc],
                    self.comb_mix[p * hc * hc ..][0 .. hc * hc],
                    hc,
                    h,
                    self.h_streams[p * hc * h ..][0 .. hc * h],
                );
            }
            @memcpy(self.pre_mix[0 .. bsz * hc], self.next_pre_mix[0 .. bsz * hc]);
        }
        if (start_pos + 1 > self.window_pos) self.window_pos = start_pos + 1;

        // 4. `mtp.2` final collapse + `mtp.2.norm.weight` (`model.py:1147-1148`).
        for (0..bsz) |p| {
            const st_p = self.h_streams[p * hc * h ..][0 .. hc * h];
            const xf_p = self.x_final[p * h ..][0..h];
            hcCollapseStreams(st_p, self.pre_mix[p * hc ..][0..hc], hc, h, xf_p);
            const sh_p = self.workspace.slot_hiddens[p * h ..][0..h];
            @memcpy(sh_p, xf_p);
            applyRmsNormInPlace(sh_p, self.final_norm, self.config.rms_eps);
        }

        // 5. Logits via `markov_head` low-rank scorer & optional hybrid target `lm_head`.
        const logits = self.workspace.slot_logits[0 .. bsz * v];
        const has_markov = (self.markov_embed != null and self.markov_head != null);
        const eff_mode: DSpark3LLmHeadMode = switch (self.config.lm_head_mode) {
            .markov_low_rank => if (has_markov) .markov_low_rank else .target_lm_head_only,
            .hybrid_target_plus_markov => if (has_markov and self.lm_head_hook != null)
                .hybrid_target_plus_markov
            else if (has_markov)
                .markov_low_rank
            else
                .target_lm_head_only,
            .target_lm_head_only => if (self.lm_head_hook != null) .target_lm_head_only else if (has_markov) .markov_low_rank else .target_lm_head_only,
        };

        var markov_evals: u32 = 0;
        var sweeps_avoided: u32 = 0;

        if (eff_mode == .hybrid_target_plus_markov or eff_mode == .target_lm_head_only) {
            const lm_hook = self.lm_head_hook orelse return error.InvalidWeights;
            try evalLmHeadHook(lm_hook, self.workspace.slot_hiddens[0 .. bsz * h], self.config.block_size, self.config.hidden_size, self.config.vocab_size, logits);
        } else {
            @memset(logits, 0.0);
            sweeps_avoided = self.config.block_size;
        }

        const mr: usize = self.config.markov_rank;
        const inv_temp: f32 = 1.0 / self.config.temperature;
        var prev_tok: u32 = if (root_token < v) root_token else 0;

        for (0..bsz) |p| {
            const l_row = logits[p * v ..][0..v];
            const me_dst = self.markov_embeds[p * mr ..][0..mr];
            if (has_markov) {
                const me_mat = &self.markov_embed.?;
                const mh_mat = &self.markov_head.?;
                const safe_tok: usize = if (prev_tok < me_mat.n) prev_tok else 0;
                try dequantMatrixRow(me_mat, safe_tok, me_dst);
                if (eff_mode == .markov_low_rank) {
                    // Fold `mtp.2.norm(x)` softly into the low-rank query so context hidden
                    // states steer low-rank scoring while keeping raw `me_dst` intact for `confidence_head`.
                    const sh_p = self.workspace.slot_hiddens[p * h ..][0..h];
                    const q_low = self.conf_in[0..mr];
                    @memcpy(q_low, me_dst);
                    const inv_folds: f32 = 0.125 / @as(f32, @floatFromInt(@max(@as(usize, 1), (h + mr - 1) / mr)));
                    for (0..h) |d| q_low[d % mr] += sh_p[d] * inv_folds;
                    try self.evalMatVec(mh_mat, q_low, l_row, false);
                } else {
                    // Exact `model.py:1153-1154`: `logits[:, i].add_(markov_head(output_ids[:, i]))`.
                    try self.evalMatVec(mh_mat, me_dst, l_row, true);
                }
                markov_evals += 1;
            } else {
                @memset(me_dst, 0.0);
            }

            const out_tok = self.workspace.topk_tokens[p * k ..][0..k];
            const out_prob = self.workspace.topk_probs[p * k ..][0..k];
            try extractDeterministicTopK(l_row, inv_temp, out_tok, out_prob);
            prev_tok = out_tok[0];
        }

        // 6. `mtp.2.confidence_head.proj` -> sigmoid per block position (`model.py:1098-1100`).
        var conf_sum: f32 = 0.0;
        var conf_min: f32 = 1.0;
        for (0..bsz) |p| {
            const conf_p: f32 = if (self.confidence_proj) |*cp| blk: {
                const cin_len: usize = cp.k;
                @memset(self.conf_in[0..cin_len], 0.0);
                const copy_h = @min(h, cin_len);
                @memcpy(self.conf_in[0..copy_h], self.x_final[p * h ..][0..copy_h]);
                if (cin_len > copy_h) {
                    const copy_m = @min(mr, cin_len - copy_h);
                    @memcpy(self.conf_in[copy_h .. copy_h + copy_m], self.markov_embeds[p * mr ..][0..copy_m]);
                }
                var out1: [1]f32 = .{0.0};
                try self.evalMatVec(cp, self.conf_in[0..cin_len], out1[0..1], false);
                break :blk sigmoidF32(out1[0]);
            } else 1.0;
            self.slot_confidences[p] = conf_p;
            conf_sum += conf_p;
            if (conf_p < conf_min) conf_min = conf_p;
        }
        self.total_markov_evals += markov_evals;
        self.total_lm_head_sweeps_avoided += sweeps_avoided;

        return .{
            .mean_confidence = conf_sum / @as(f32, @floatFromInt(bsz)),
            .min_confidence = conf_min,
            .markov_evals = markov_evals,
            .lm_head_sweeps_avoided = sweeps_avoided,
        };
    }

    pub fn proposeIntoConstrained(
        self: *DSpark3LRuntime,
        comptime max_nodes: u16,
        root_token: u32,
        encoder_summary: ?[]const f32,
        last_decoder_hidden: []const f32,
        sam: ?*const sam_mod.SuffixAutomaton,
        grammar_opt: ?accept_mod.GrammarTreeAdapter,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!ProposeResult {
        out_tree.reset(root_token);

        const node_limit: u16 = @min(@as(u16, self.config.max_tree_nodes), max_nodes);
        const gram_active = if (grammar_opt) |ga| ga.isActive() else false;
        var node_states: [tree_mod.MAX_SUPPORTED_NODES]grammar_mod.GrammarState = undefined;
        var node_valid: [tree_mod.MAX_SUPPORTED_NODES]bool = @splat(false);
        node_valid[0] = true;
        if (grammar_opt) |ga| {
            if (ga.isActive()) node_states[0] = ga.state.snapshot();
        }

        // 1. Optional L1 SAM prefix seed / branch merge (ADR-0005), grammar-pruned.
        var l1_match_len: u32 = 0;
        var l1_inserted: u16 = 0;
        var l1_is_seed: bool = false;
        var seed_tail_node: u16 = 0;

        if (sam) |automaton| {
            const max_l1: usize = @min(
                @as(usize, self.config.l1_max_draft_len),
                @as(usize, node_limit - 1),
            );
            if (max_l1 > 0) {
                const q = automaton.queryActive(max_l1, self.workspace.l1_draft_buf[0..max_l1]);
                var valid_len: usize = q.draft_len;
                if (gram_active) {
                    const ga = grammar_opt.?;
                    var st_walk = node_states[0].snapshot();
                    valid_len = 0;
                    for (self.workspace.l1_draft_buf[0..q.draft_len]) |tid| {
                        if (!ga.advanceTokenOnState(&st_walk, tid)) break;
                        valid_len += 1;
                    }
                }
                l1_match_len = if (valid_len == q.draft_len) q.match_len else @intCast(@min(q.match_len, valid_len));
                if (valid_len > 0) {
                    const ins = out_tree.insertL1Continuation(
                        @intCast(@max(@as(u32, 1), l1_match_len)),
                        self.workspace.l1_draft_buf[0..valid_len],
                        self.config.l1_seed_step_prob,
                        self.config.l1_branch_step_prob,
                    );
                    l1_inserted = ins.inserted_count;
                    l1_is_seed = ins.is_seed;
                    seed_tail_node = ins.seed_tail_node;
                    if (gram_active) {
                        const ga = grammar_opt.?;
                        for (1..out_tree.count) |ni| {
                            const nd = out_tree.nodes[ni];
                            if (node_valid[nd.parent]) {
                                var cs = node_states[nd.parent].snapshot();
                                if (ga.advanceTokenOnState(&cs, nd.token_id)) {
                                    node_states[ni] = cs;
                                    node_valid[ni] = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Build 3-tap conditioning array: prefer pending multi-layer target taps,
        //    falling back cleanly to `(encoder_summary, last_decoder_hidden)`.
        var fallback_taps: [MAX_DSPARK_TARGET_TAPS][]const f32 = @splat(last_decoder_hidden);
        const eff_taps: []const []const f32 = if (self.has_pending_taps and self.pending_tap_count > 0)
            self.pending_taps[0..self.pending_tap_count]
        else blk: {
            if (encoder_summary) |es| fallback_taps[0] = es;
            break :blk fallback_taps[0..self.config.num_target_taps];
        };
        const eff_start_pos: u32 = if (self.has_pending_taps) self.pending_start_pos else self.window_pos;
        self.has_pending_taps = false;

        // 3. Execute 3L DSpark forward pass (zero hot-path allocs).
        const pred_stats = try self.predictBlock(root_token, eff_taps, eff_start_pos);
        const before_dspark = out_tree.count;

        // 4. Confidence-gated deterministic beam expansion (`confidence_head` gates depth
        //    and scales candidate step probabilities feeding `D7` expert-byte pruning).
        var frontier_cur: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
        var frontier_next: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
        var cur_len: usize = 1;
        frontier_cur[0] = 0;

        const w: u8 = self.config.block_size;
        const k: usize = self.config.top_k_per_pos;
        const v: usize = self.config.vocab_size;
        const inv_temp: f32 = 1.0 / self.config.temperature;
        const conf_thresh: f32 = self.config.confidence_threshold;
        var masked_toks: [MAX_TOP_K_PER_POS]u32 = undefined;
        var masked_probs: [MAX_TOP_K_PER_POS]f32 = undefined;
        var gated_positions: u8 = 0;

        for (0..w) |pos_usize| {
            const pos: u8 = @intCast(pos_usize);
            const conf_p: f32 = self.slot_confidences[pos_usize];
            if (conf_thresh > 0.0 and conf_p < conf_thresh and (pos > 0 or out_tree.count > 1)) {
                gated_positions = w - pos;
                break;
            }
            const conf_scale: f32 = @max(@as(f32, 0.05), @min(@as(f32, 1.0), conf_p));
            const pos_tokens = self.workspace.slotTokens(pos);
            const pos_probs = self.workspace.slotProbs(pos);
            const pos_logits = self.workspace.slot_logits[@as(usize, pos) * v ..][0..v];

            sortFrontierDescending(max_nodes, out_tree, frontier_cur[0..cur_len]);

            var next_len: usize = 0;
            for (frontier_cur[0..cur_len]) |parent| {
                if (gram_active and !node_valid[parent]) continue;
                var cand_count: usize = k;
                var cand_toks: []const u32 = pos_tokens;
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
                        masked_toks[0..k],
                        masked_probs[0..k],
                    );
                    cand_toks = masked_toks[0..cand_count];
                    cand_probs = masked_probs[0..cand_count];
                }
                for (0..cand_count) |ki| {
                    const tok = cand_toks[ki];
                    const prob = cand_probs[ki] * conf_scale;
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

        // 5. Dual-source L1 seed tail continuation (also respecting confidence gating).
        if (l1_is_seed and seed_tail_node != 0 and out_tree.count < node_limit and (!gram_active or node_valid[seed_tail_node])) {
            var tail: u16 = seed_tail_node;
            for (0..w) |pos_usize| {
                if (out_tree.count >= node_limit) break;
                const conf_p: f32 = self.slot_confidences[pos_usize];
                if (conf_thresh > 0.0 and conf_p < conf_thresh) break;
                const pos: u8 = @intCast(pos_usize);
                const conf_scale: f32 = @max(@as(f32, 0.05), @min(@as(f32, 1.0), conf_p));
                var tok = self.workspace.slotTokens(pos)[0];
                var prob = self.workspace.slotProbs(pos)[0] * conf_scale;
                var next_st: grammar_mod.GrammarState = undefined;
                if (gram_active) {
                    var tail_ga = grammar_opt.?;
                    var tstate = node_states[tail].snapshot();
                    tail_ga.state = &tstate;
                    const mask = tail_ga.fillBitmask(v) catch break;
                    const pos_logits = self.workspace.slot_logits[@as(usize, pos) * v ..][0..v];
                    const got = extractDeterministicTopKMasked(pos_logits, inv_temp, mask, masked_toks[0..1], masked_probs[0..1]) catch break;
                    if (got == 0) break;
                    tok = masked_toks[0];
                    prob = masked_probs[0] * conf_scale;
                    next_st = node_states[tail].snapshot();
                    if (!grammar_opt.?.advanceTokenOnState(&next_st, tok)) break;
                }
                const next = out_tree.insertChild(tail, tok, prob, .l2) orelse break;
                if (gram_active) {
                    node_states[next] = next_st;
                    node_valid[next] = true;
                }
                tail = next;
            }
        }

        self.total_confidence_gated_positions += gated_positions;
        return .{
            .tree_nodes = out_tree.count,
            .max_depth = out_tree.max_depth,
            .l1_match_len = l1_match_len,
            .l1_inserted = l1_inserted,
            .l1_is_seed = l1_is_seed,
            .dspark_inserted = out_tree.count - before_dspark,
            .expected_tokens = out_tree.expectedTokens(),
            .mean_confidence = pred_stats.mean_confidence,
            .min_confidence = pred_stats.min_confidence,
            .gated_positions = gated_positions,
            .markov_evals = pred_stats.markov_evals,
            .lm_head_sweeps_avoided = pred_stats.lm_head_sweeps_avoided,
        };
    }

    fn loadInternal(
        allocator: std.mem.Allocator,
        source: DSpark3LAmfSource,
        cfg_in: ?DSpark3LConfig,
        opts: DSpark3LLoadOptions,
    ) LoadError!*DSpark3LRuntime {
        const parsed = source.parsed();
        const mp_e = findTensorFlex(parsed, "mtp.0.main_proj") orelse return error.MissingTensor;
        const mn_e = findTensorFlex(parsed, "mtp.0.main_norm") orelse return error.MissingTensor;
        if (mp_e.rank != 2 or mn_e.shape[0] == 0) return error.InvalidWeights;

        const h: u32 = mp_e.shape[0];
        if (mn_e.shape[0] != h or mp_e.shape[1] == 0 or (mp_e.shape[1] % h) != 0) {
            return error.InvalidWeights;
        }
        const inferred_taps: u8 = @intCast(@min(MAX_DSPARK_TARGET_TAPS, mp_e.shape[1] / h));

        // Count `mtp.{L}` stages present in the `.amf`.
        var name_buf: [96]u8 = undefined;
        var num_layers: u8 = 0;
        for (0..MAX_DSPARK_3L_LAYERS) |l| {
            const probe = std.fmt.bufPrint(&name_buf, "mtp.{d}.attn.wq_a", .{l}) catch unreachable;
            if (findTensorFlex(parsed, probe) != null) {
                num_layers = @intCast(l + 1);
            } else {
                break;
            }
        }
        if (num_layers == 0) return error.MissingTensor;

        // Infer geometry directly from `mtp.0` & `mtp.{last}` tensor shapes.
        const wqa0 = findTensorFlex(parsed, "mtp.0.attn.wq_a") orelse return error.MissingTensor;
        const wqb0 = findTensorFlex(parsed, "mtp.0.attn.wq_b") orelse return error.MissingTensor;
        const wkv0 = findTensorFlex(parsed, "mtp.0.attn.wkv") orelse return error.MissingTensor;
        const kvn0 = findTensorFlex(parsed, "mtp.0.attn.kv_norm") orelse return error.MissingTensor;
        const woa0 = findTensorFlex(parsed, "mtp.0.attn.wo_a") orelse return error.MissingTensor;
        const wob0 = findTensorFlex(parsed, "mtp.0.attn.wo_b") orelse return error.MissingTensor;
        const gate0 = findTensorFlex(parsed, "mtp.0.ffn.gate") orelse return error.MissingTensor;
        const hc_fn0 = findTensorFlex(parsed, "mtp.0.hc_attn_fn") orelse return error.MissingTensor;
        const hc_base0 = findTensorFlex(parsed, "mtp.0.hc_attn_base") orelse return error.MissingTensor;

        const last_l: usize = num_layers - 1;
        const mk_emb_e = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.markov_head.embed", .{last_l}) catch unreachable);
        const mk_head_e = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.markov_head.head", .{last_l}) catch unreachable);
        const conf_e = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.confidence_head.proj", .{last_l}) catch unreachable);
        const fnorm_e = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.norm", .{last_l}) catch unreachable) orelse return error.MissingTensor;

        var cfg: DSpark3LConfig = cfg_in orelse .{
            .hidden_size = h,
            .vocab_size = if (mk_emb_e) |me| me.shape[0] else return error.InvalidConfig,
        };
        cfg.num_layers = num_layers;
        cfg.hidden_size = h;
        cfg.num_target_taps = inferred_taps;
        if (mk_emb_e) |me| {
            cfg.vocab_size = me.shape[0];
            cfg.markov_rank = me.shape[1];
        }
        cfg.q_lora_rank = wqa0.shape[0];
        cfg.kv_lora_rank = kvn0.shape[0];
        cfg.n_routed_experts = gate0.shape[0];
        if (cfg.n_routed_experts > 0 and cfg.routed_top_k > cfg.n_routed_experts) {
            cfg.routed_top_k = cfg.n_routed_experts;
        }
        if (findTensorFlex(parsed, "mtp.0.ffn.experts.0.w1")) |ew1| {
            cfg.moe_intermediate_size = ew1.shape[0];
        } else if (findTensorFlex(parsed, "mtp.0.ffn.shared_experts.w1")) |sw1| {
            cfg.moe_intermediate_size = sw1.shape[0];
        }

        // Reconcile `hc_mult` from `hc_attn_fn` / `hc_attn_base`.
        if (hc_fn0.shape[0] == h and hc_fn0.shape[1] > 0 and hc_fn0.shape[1] <= MAX_HC_MULT) {
            cfg.hc_mult = hc_fn0.shape[1];
        } else if (hc_fn0.shape[1] > 0 and (hc_fn0.shape[1] % h) == 0) {
            cfg.hc_mult = hc_fn0.shape[1] / h;
        } else if (hc_base0.rank == 2 and hc_base0.shape[0] > 0 and hc_base0.shape[0] <= MAX_HC_MULT) {
            cfg.hc_mult = hc_base0.shape[0];
        }

        // Reconcile attention geometry (`num_attention_heads`, `head_dim`, `qk_rope_head_dim`, `o_groups`, `o_lora_rank`).
        const q_dim_total = wqb0.shape[0];
        if (findTensorFlex(parsed, "mtp.0.attn.attn_sink")) |se| {
            if (se.shape[0] > 0 and (q_dim_total % se.shape[0]) == 0) {
                cfg.num_attention_heads = se.shape[0];
                cfg.head_dim = q_dim_total / se.shape[0];
            }
        } else if (wkv0.shape[0] == kvn0.shape[0] and (q_dim_total % wkv0.shape[0]) == 0) {
            cfg.head_dim = wkv0.shape[0];
            cfg.num_attention_heads = q_dim_total / cfg.head_dim;
        } else if (kvn0.shape[0] > 0 and (q_dim_total % kvn0.shape[0]) == 0) {
            cfg.head_dim = kvn0.shape[0];
            cfg.num_attention_heads = q_dim_total / cfg.head_dim;
        }
        const wkv_dim = wkv0.shape[0];
        if (wkv_dim > kvn0.shape[0]) {
            cfg.qk_rope_head_dim = wkv_dim - kvn0.shape[0];
        } else if (cfg.qk_rope_head_dim >= cfg.head_dim) {
            cfg.qk_rope_head_dim = cfg.head_dim / 2;
        }
        const wo_a_out = woa0.shape[0];
        const wo_a_in = woa0.shape[1];
        const wo_b_in = wob0.shape[1];
        if (wo_a_in > 0 and (q_dim_total % wo_a_in) == 0) {
            cfg.o_groups = q_dim_total / wo_a_in;
        } else if (wo_b_in > 0 and wo_a_out >= wo_b_in and (wo_a_out % wo_b_in) == 0) {
            cfg.o_groups = wo_a_out / wo_b_in;
        }
        if (cfg.o_groups == 0) cfg.o_groups = 1;
        cfg.o_lora_rank = @max(@as(u32, 1), wo_a_out / cfg.o_groups);

        if (opts.lm_head_mode) |m| cfg.lm_head_mode = m;
        if (opts.cold_slot_pool_slots) |s| cfg.cold_slot_pool_slots = s;
        if (opts.cold_expert_min_weight) |mw| cfg.cold_expert_min_weight = mw;
        if (opts.confidence_threshold) |ct| cfg.confidence_threshold = ct;
        try cfg.validate();

        // --- Pass 1: Size `weights_arena` and cold-expert slot pool ---
        var weights_bytes: u64 = 0;
        var max_mat_k: usize = @max(@as(usize, h) * cfg.num_target_taps, @as(usize, h) * cfg.hc_mult);
        var max_mat_n: usize = h;
        var any_sme_panel = (try mp_e.tensorLayout()) == .sme2_panel64;
        var hot_experts: u32 = 0;
        var cold_experts: u32 = 0;
        var max_w1_blob: usize = 0;
        var max_w2_blob: usize = 0;
        var max_w3_blob: usize = 0;

        weights_bytes += (try page.alignUp(@intCast(mp_e.blob_bytes))) + (try page.alignUp(@as(usize, h) * 4));
        weights_bytes += try page.alignUp(@as(usize, fnorm_e.shape[0]) * 4);
        if (mk_emb_e) |e| {
            weights_bytes += try page.alignUp(@intCast(e.blob_bytes));
            max_mat_k = @max(max_mat_k, e.shape[1]);
        }
        if (mk_head_e) |e| {
            weights_bytes += try page.alignUp(@intCast(e.blob_bytes));
            max_mat_k = @max(max_mat_k, e.shape[1]);
            max_mat_n = @max(max_mat_n, e.shape[0]);
            if ((try e.tensorLayout()) == .sme2_panel64) any_sme_panel = true;
        }
        if (conf_e) |e| {
            weights_bytes += try page.alignUp(@intCast(e.blob_bytes));
            max_mat_k = @max(max_mat_k, e.shape[1]);
        }

        for (0..num_layers) |l| {
            inline for (.{ "hc_attn", "hc_ffn" }) |tag| {
                const fn_e = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.{s}_fn", .{ l, tag }) catch unreachable) orelse return error.MissingTensor;
                const bs_e = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.{s}_base", .{ l, tag }) catch unreachable) orelse return error.MissingTensor;
                const sc_e = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.{s}_scale", .{ l, tag }) catch unreachable) orelse return error.MissingTensor;
                weights_bytes += try page.alignUp(@intCast(fn_e.blob_bytes));
                weights_bytes += try page.alignUp(@as(usize, tensorElemCount(bs_e)) * 4);
                weights_bytes += try page.alignUp(@as(usize, tensorElemCount(sc_e)) * 4);
                max_mat_k = @max(max_mat_k, if (fn_e.rank >= 2) fn_e.shape[1] else fn_e.shape[0]);
                if ((try fn_e.tensorLayout()) == .sme2_panel64) any_sme_panel = true;
            }
            inline for (.{ "attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm" }) |n_tag| {
                const ne = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.{s}", .{ l, n_tag }) catch unreachable) orelse return error.MissingTensor;
                weights_bytes += try page.alignUp(@as(usize, ne.shape[0]) * 4);
            }
            inline for (.{ "attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b", "ffn.gate" }) |m_tag| {
                const me = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.{s}", .{ l, m_tag }) catch unreachable) orelse return error.MissingTensor;
                weights_bytes += try page.alignUp(@intCast(me.blob_bytes));
                max_mat_k = @max(max_mat_k, if (me.rank >= 2) me.shape[1] else 1);
                max_mat_n = @max(max_mat_n, me.shape[0]);
                if ((try me.tensorLayout()) == .sme2_panel64) any_sme_panel = true;
            }
            if (findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.attn.attn_sink", .{l}) catch unreachable)) |se| {
                weights_bytes += try page.alignUp(@as(usize, se.shape[0]) * 4);
            }
            if (findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.gate.bias", .{l}) catch unreachable)) |be| {
                weights_bytes += try page.alignUp(@as(usize, be.shape[0]) * 4);
            }
            if (findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.shared_experts.w1", .{l}) catch unreachable)) |sw1| {
                const sw2 = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.shared_experts.w2", .{l}) catch unreachable) orelse return error.MissingTensor;
                const sw3 = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.shared_experts.w3", .{l}) catch unreachable) orelse return error.MissingTensor;
                weights_bytes += (try page.alignUp(@intCast(sw1.blob_bytes))) +
                    (try page.alignUp(@intCast(sw2.blob_bytes))) +
                    (try page.alignUp(@intCast(sw3.blob_bytes)));
                max_mat_k = @max(max_mat_k, sw2.shape[1]);
                max_mat_n = @max(max_mat_n, sw1.shape[0]);
                if ((try sw1.tensorLayout()) == .sme2_panel64 or (try sw2.tensorLayout()) == .sme2_panel64 or (try sw3.tensorLayout()) == .sme2_panel64) any_sme_panel = true;
            }
            for (0..cfg.n_routed_experts) |e_idx| {
                const ew1 = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.experts.{d}.w1", .{ l, e_idx }) catch unreachable) orelse return error.MissingTensor;
                const ew2 = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.experts.{d}.w2", .{ l, e_idx }) catch unreachable) orelse return error.MissingTensor;
                const ew3 = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.experts.{d}.w3", .{ l, e_idx }) catch unreachable) orelse return error.MissingTensor;
                max_mat_k = @max(max_mat_k, @max(ew1.shape[1], ew2.shape[1]));
                max_mat_n = @max(max_mat_n, @max(ew1.shape[0], ew2.shape[0]));
                if ((try ew1.tensorLayout()) == .sme2_panel64 or (try ew2.tensorLayout()) == .sme2_panel64 or (try ew3.tensorLayout()) == .sme2_panel64) any_sme_panel = true;
                const is_cold = opts.force_cold_experts or
                    (try ew1.tensorKind()).isCold() or
                    (try ew2.tensorKind()).isCold() or
                    (try ew3.tensorKind()).isCold() or
                    (if (opts.max_hot_experts_per_layer) |cap| e_idx >= cap else false);
                if (is_cold) {
                    cold_experts += 1;
                    max_w1_blob = @max(max_w1_blob, @as(usize, @intCast(ew1.blob_bytes)));
                    max_w2_blob = @max(max_w2_blob, @as(usize, @intCast(ew2.blob_bytes)));
                    max_w3_blob = @max(max_w3_blob, @as(usize, @intCast(ew3.blob_bytes)));
                } else {
                    hot_experts += 1;
                    weights_bytes += (try page.alignUp(@intCast(ew1.blob_bytes))) +
                        (try page.alignUp(@intCast(ew2.blob_bytes))) +
                        (try page.alignUp(@intCast(ew3.blob_bytes)));
                }
            }
        }
        const weights_arena_cap = try page.alignUp(@as(usize, @intCast(weights_bytes)) + 4 * page.APPLE_PAGE_SIZE);

        // --- Pass 2: Size unified 16 KiB-aligned `workspace.arena` (0 hot-path allocs) ---
        const compat_cfg = cfg.toLegacyCompatConfig();
        const bsz: usize = cfg.block_size;
        const hc: usize = cfg.hc_mult;
        const q_dim: usize = @as(usize, cfg.num_attention_heads) * cfg.head_dim;
        const moe_inter: usize = @max(@as(usize, cfg.moe_intermediate_size), 1);
        const mr: usize = @max(@as(usize, cfg.markov_rank), 1);
        const conf_in_len: usize = @max(@as(usize, h) + mr, mr);

        var scratch_bytes: usize = 0;
        const chunks = [_]usize{
            try alignUp64(@as(usize, h) * @sizeOf(f32)), // base_hidden
            try alignUp64(@as(usize, h) * @sizeOf(f32)), // slot_input
            try alignUp64(bsz * h * @sizeOf(f32)), // slot_hiddens
            try alignUp64(bsz * @as(usize, cfg.vocab_size) * @sizeOf(f32)), // slot_logits
            try alignUp64(bsz * @as(usize, cfg.top_k_per_pos) * @sizeOf(u32)), // topk_tokens
            try alignUp64(bsz * @as(usize, cfg.top_k_per_pos) * @sizeOf(f32)), // topk_probs
            try alignUp64(MAX_L1_DRAFT_LEN * @sizeOf(u32)), // l1_draft_buf
            try alignUp64(@as(usize, num_layers) * cfg.window_size * wkv_dim * @sizeOf(f32)), // window_kv_cache
            try alignUp64(@as(usize, cfg.num_target_taps) * h * @sizeOf(f32)), // main_concat
            try alignUp64(@as(usize, h) * @sizeOf(f32)), // main_x
            try alignUp64(bsz * hc * h * @sizeOf(f32)), // h_streams
            try alignUp64(bsz * hc * h * @sizeOf(f32)), // h_streams_res
            try alignUp64(bsz * hc * @sizeOf(f32)), // pre_mix
            try alignUp64(bsz * hc * @sizeOf(f32)), // next_pre_mix
            try alignUp64(bsz * hc * @sizeOf(f32)), // post_mix
            try alignUp64(bsz * hc * hc * @sizeOf(f32)), // comb_mix
            try alignUp64(hc * h * @sizeOf(f32)), // hc_flat_buf
            try alignUp64(bsz * h * @sizeOf(f32)), // sub_in
            try alignUp64(bsz * h * @sizeOf(f32)), // sub_out
            try alignUp64(@as(usize, cfg.q_lora_rank) * @sizeOf(f32)), // q_lora_buf
            try alignUp64(bsz * q_dim * @sizeOf(f32)), // q_all
            try alignUp64(bsz * @as(usize, wkv_dim) * @sizeOf(f32)), // block_kv
            try alignUp64(q_dim * @sizeOf(f32)), // attn_out
            try alignUp64(@as(usize, wo_a_out) * @sizeOf(f32)), // wo_a_out
            try alignUp64(@as(usize, @max(wo_b_in, cfg.o_lora_rank)) * @sizeOf(f32)), // wo_b_in
            try alignUp64(@max(@as(usize, cfg.n_routed_experts), 1) * @sizeOf(f32)), // moe_scores
            try alignUp64(moe_inter * @sizeOf(f32)), // moe_gate
            try alignUp64(moe_inter * @sizeOf(f32)), // moe_up
            try alignUp64(@as(usize, h) * @sizeOf(f32)), // moe_tmp
            try alignUp64(bsz * h * @sizeOf(f32)), // x_final
            try alignUp64(bsz * mr * @sizeOf(f32)), // markov_embeds
            try alignUp64(bsz * @sizeOf(f32)), // slot_confidences
            try alignUp64(conf_in_len * @sizeOf(f32)), // conf_in
            try alignUp64(@max(max_mat_k, 16) * @sizeOf(f32)), // row_scratch
            if (any_sme_panel) try alignUp64(@max(max_mat_k, 64) * @sizeOf(u16)) else 64, // sme act_u16
            if (any_sme_panel) try alignUp64(@max(max_mat_n, 16) * @sizeOf(f32)) else 64, // sme out_f32
        };
        for (chunks) |ch| {
            scratch_bytes = std.math.add(usize, scratch_bytes, ch) catch return error.Overflow;
        }
        const scratch_arena_cap = try page.alignUp(scratch_bytes + page.APPLE_PAGE_SIZE);

        // Cold slot pool sizing.
        const pool_slots: usize = if (cold_experts > 0) @min(@as(usize, cold_experts), @max(@as(usize, cfg.cold_slot_pool_slots), 1)) else 0;
        const slot_w1_sz = if (pool_slots > 0) try alignUp64(max_w1_blob) else 0;
        const slot_w2_sz = if (pool_slots > 0) try alignUp64(max_w2_blob) else 0;
        const slot_w3_sz = if (pool_slots > 0) try alignUp64(max_w3_blob) else 0;
        const cold_pool_cap: usize = if (pool_slots > 0)
            try page.alignUp(pool_slots * (slot_w1_sz + slot_w2_sz + slot_w3_sz) + page.APPLE_PAGE_SIZE)
        else
            0;

        const wired_budget = opts.budget_override orelse try defaultWiredBudget();
        try wired_budget.checkPlan(weights_arena_cap + scratch_arena_cap + cold_pool_cap);

        var weights_arena = try arena_mod.WiredArena.init(weights_arena_cap, wired_budget);
        errdefer weights_arena.deinit();

        var scratch_arena = try arena_mod.WiredArena.init(scratch_arena_cap, wired_budget);
        errdefer scratch_arena.deinit();

        var cold_pool: ColdExpertSlotPool = .{
            .verify_crcs = opts.verify_crcs,
            .stats = .{
                .hot_experts = hot_experts,
                .cold_experts = cold_experts,
                .pool_slots = @intCast(pool_slots),
                .slot_pool_bytes = cold_pool_cap,
            },
        };
        errdefer cold_pool.deinit(allocator);
        if (pool_slots > 0) {
            cold_pool.arena = try arena_mod.WiredArena.init(cold_pool_cap, wired_budget);
            cold_pool.slots = try allocator.alloc(ColdSlot, pool_slots);
            for (cold_pool.slots) |*sl| {
                sl.* = .{
                    .w1_buf = try cold_pool.arena.?.allocAligned(slot_w1_sz, CACHELINE_ALIGN),
                    .w2_buf = try cold_pool.arena.?.allocAligned(slot_w2_sz, CACHELINE_ALIGN),
                    .w3_buf = try cold_pool.arena.?.allocAligned(slot_w3_sz, CACHELINE_ALIGN),
                };
            }
        }

        // Carve preallocated slices from `scratch_arena`.
        const allocF32 = struct {
            fn f(a: *arena_mod.WiredArena, n: usize) Error![]f32 {
                const raw = try a.allocAligned(n * @sizeOf(f32), CACHELINE_ALIGN);
                const sl: []f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, raw)))[0..n];
                @memset(sl, 0.0);
                return sl;
            }
        }.f;
        const allocU32 = struct {
            fn f(a: *arena_mod.WiredArena, n: usize) Error![]u32 {
                const raw = try a.allocAligned(n * @sizeOf(u32), CACHELINE_ALIGN);
                const sl: []u32 = @as([]u32, @alignCast(std.mem.bytesAsSlice(u32, raw)))[0..n];
                @memset(sl, 0);
                return sl;
            }
        }.f;

        const base_hidden = try allocF32(&scratch_arena, h);
        const slot_input = try allocF32(&scratch_arena, h);
        const slot_hiddens = try allocF32(&scratch_arena, bsz * h);
        const slot_logits = try allocF32(&scratch_arena, bsz * @as(usize, cfg.vocab_size));
        const topk_tokens = try allocU32(&scratch_arena, bsz * @as(usize, cfg.top_k_per_pos));
        const topk_probs = try allocF32(&scratch_arena, bsz * @as(usize, cfg.top_k_per_pos));
        const l1_draft_buf = try allocU32(&scratch_arena, MAX_L1_DRAFT_LEN);

        const window_kv_cache = try allocF32(&scratch_arena, @as(usize, num_layers) * cfg.window_size * wkv_dim);
        const main_concat = try allocF32(&scratch_arena, @as(usize, cfg.num_target_taps) * h);
        const main_x = try allocF32(&scratch_arena, h);
        const h_streams = try allocF32(&scratch_arena, bsz * hc * h);
        const h_streams_res = try allocF32(&scratch_arena, bsz * hc * h);
        const pre_mix = try allocF32(&scratch_arena, bsz * hc);
        const next_pre_mix = try allocF32(&scratch_arena, bsz * hc);
        const post_mix = try allocF32(&scratch_arena, bsz * hc);
        const comb_mix = try allocF32(&scratch_arena, bsz * hc * hc);
        const hc_flat_buf = try allocF32(&scratch_arena, hc * h);
        const sub_in = try allocF32(&scratch_arena, bsz * h);
        const sub_out = try allocF32(&scratch_arena, bsz * h);
        const q_lora_buf = try allocF32(&scratch_arena, cfg.q_lora_rank);
        const q_all = try allocF32(&scratch_arena, bsz * q_dim);
        const block_kv = try allocF32(&scratch_arena, bsz * @as(usize, wkv_dim));
        const attn_out = try allocF32(&scratch_arena, q_dim);
        const wo_a_out_buf = try allocF32(&scratch_arena, wo_a_out);
        const wo_b_in_buf = try allocF32(&scratch_arena, @max(wo_b_in, cfg.o_lora_rank));
        const moe_scores = try allocF32(&scratch_arena, @max(@as(usize, cfg.n_routed_experts), 1));
        const moe_gate = try allocF32(&scratch_arena, moe_inter);
        const moe_up = try allocF32(&scratch_arena, moe_inter);
        const moe_tmp = try allocF32(&scratch_arena, h);
        const x_final = try allocF32(&scratch_arena, bsz * h);
        const markov_embeds = try allocF32(&scratch_arena, bsz * mr);
        const slot_confidences = try allocF32(&scratch_arena, bsz);
        const conf_in = try allocF32(&scratch_arena, conf_in_len);
        const row_scratch = try allocF32(&scratch_arena, @max(max_mat_k, 16));

        var sme_runner: ?DSpark3LSmeRunner = null;
        errdefer if (sme_runner) |*sr| sr.dispatcher.deinit();
        if (any_sme_panel) {
            const caps = try sme.SmeCaps.probe();
            const act_raw = try scratch_arena.allocAligned(@max(max_mat_k, 64) * @sizeOf(u16), CACHELINE_ALIGN);
            const out_raw = try scratch_arena.allocAligned(@max(max_mat_n, 16) * @sizeOf(f32), CACHELINE_ALIGN);
            sme_runner = .{
                .dispatcher = try sme.SmeDispatcher.init(allocator, .{ .caps_override = caps }),
                .act_u16 = @as([]u16, @alignCast(std.mem.bytesAsSlice(u16, act_raw)))[0..@max(max_mat_k, 64)],
                .out_f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, out_raw)))[0..@max(max_mat_n, 16)],
            };
        }

        // --- Pass 3: Load wired tensors & record cold expert descriptors ---
        const main_proj = try loadQuantMatrixFromSource(source, mp_e, &weights_arena, opts.verify_crcs);
        const main_norm = try loadVecF32FromSource(allocator, source, mn_e, &weights_arena, opts.verify_crcs);
        const final_norm = try loadVecF32FromSource(allocator, source, fnorm_e, &weights_arena, opts.verify_crcs);
        const markov_embed: ?QuantMatrix = if (mk_emb_e) |e| try loadQuantMatrixFromSource(source, e, &weights_arena, opts.verify_crcs) else null;
        const markov_head: ?QuantMatrix = if (mk_head_e) |e| try loadQuantMatrixFromSource(source, e, &weights_arena, opts.verify_crcs) else null;
        const confidence_proj: ?QuantMatrix = if (conf_e) |e| try loadQuantMatrixFromSource(source, e, &weights_arena, opts.verify_crcs) else null;

        var layers: [MAX_DSPARK_3L_LAYERS]DSpark3LLayer = @splat(.{});
        var layers_alloced: usize = 0;
        errdefer for (0..layers_alloced) |li| {
            if (layers[li].experts.len > 0) allocator.free(layers[li].experts);
        };

        for (0..num_layers) |l| {
            var lay: DSpark3LLayer = .{};
            lay.hc_attn = try loadHcBlock(allocator, source, parsed, l, "hc_attn", cfg.hc_mult, h, &weights_arena, opts.verify_crcs);
            lay.hc_ffn = try loadHcBlock(allocator, source, parsed, l, "hc_ffn", cfg.hc_mult, h, &weights_arena, opts.verify_crcs);
            lay.attn_norm = try loadVecF32FromSource(allocator, source, findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.attn_norm", .{l}) catch unreachable) orelse return error.MissingTensor, &weights_arena, opts.verify_crcs);
            lay.wq_a = try loadQuantMatrixFromSource(source, findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.attn.wq_a", .{l}) catch unreachable) orelse return error.MissingTensor, &weights_arena, opts.verify_crcs);
            lay.q_norm = try loadVecF32FromSource(allocator, source, findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.attn.q_norm", .{l}) catch unreachable) orelse return error.MissingTensor, &weights_arena, opts.verify_crcs);
            lay.wq_b = try loadQuantMatrixFromSource(source, findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.attn.wq_b", .{l}) catch unreachable) orelse return error.MissingTensor, &weights_arena, opts.verify_crcs);
            lay.wkv = try loadQuantMatrixFromSource(source, findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.attn.wkv", .{l}) catch unreachable) orelse return error.MissingTensor, &weights_arena, opts.verify_crcs);
            lay.kv_norm = try loadVecF32FromSource(allocator, source, findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.attn.kv_norm", .{l}) catch unreachable) orelse return error.MissingTensor, &weights_arena, opts.verify_crcs);
            lay.wo_a = try loadQuantMatrixFromSource(source, findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.attn.wo_a", .{l}) catch unreachable) orelse return error.MissingTensor, &weights_arena, opts.verify_crcs);
            lay.wo_b = try loadQuantMatrixFromSource(source, findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.attn.wo_b", .{l}) catch unreachable) orelse return error.MissingTensor, &weights_arena, opts.verify_crcs);
            if (findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.attn.attn_sink", .{l}) catch unreachable)) |se| {
                lay.attn_sink = try loadVecF32FromSource(allocator, source, se, &weights_arena, opts.verify_crcs);
            }
            lay.ffn_norm = try loadVecF32FromSource(allocator, source, findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn_norm", .{l}) catch unreachable) orelse return error.MissingTensor, &weights_arena, opts.verify_crcs);
            lay.gate_weight = try loadQuantMatrixFromSource(source, findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.gate", .{l}) catch unreachable) orelse return error.MissingTensor, &weights_arena, opts.verify_crcs);
            if (findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.gate.bias", .{l}) catch unreachable)) |be| {
                lay.gate_bias = try loadVecF32FromSource(allocator, source, be, &weights_arena, opts.verify_crcs);
            }
            if (findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.shared_experts.w1", .{l}) catch unreachable)) |sw1| {
                const sw2 = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.shared_experts.w2", .{l}) catch unreachable) orelse return error.MissingTensor;
                const sw3 = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.shared_experts.w3", .{l}) catch unreachable) orelse return error.MissingTensor;
                lay.shared_w1 = try loadQuantMatrixFromSource(source, sw1, &weights_arena, opts.verify_crcs);
                lay.shared_w2 = try loadQuantMatrixFromSource(source, sw2, &weights_arena, opts.verify_crcs);
                lay.shared_w3 = try loadQuantMatrixFromSource(source, sw3, &weights_arena, opts.verify_crcs);
            }
            const exps = try allocator.alloc(ExpertRef, cfg.n_routed_experts);
            errdefer allocator.free(exps);
            for (0..cfg.n_routed_experts) |e_idx| {
                const ew1 = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.experts.{d}.w1", .{ l, e_idx }) catch unreachable) orelse return error.MissingTensor;
                const ew2 = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.experts.{d}.w2", .{ l, e_idx }) catch unreachable) orelse return error.MissingTensor;
                const ew3 = findTensorFlex(parsed, std.fmt.bufPrint(&name_buf, "mtp.{d}.ffn.experts.{d}.w3", .{ l, e_idx }) catch unreachable) orelse return error.MissingTensor;
                const is_cold = opts.force_cold_experts or
                    (try ew1.tensorKind()).isCold() or
                    (try ew2.tensorKind()).isCold() or
                    (try ew3.tensorKind()).isCold() or
                    (if (opts.max_hot_experts_per_layer) |cap| e_idx >= cap else false);
                if (is_cold) {
                    exps[e_idx] = .{ .cold = .{
                        .w1 = try ColdBlobDesc.fromEntry(ew1),
                        .w2 = try ColdBlobDesc.fromEntry(ew2),
                        .w3 = try ColdBlobDesc.fromEntry(ew3),
                    } };
                } else {
                    exps[e_idx] = .{ .wired = .{
                        .w1 = try loadQuantMatrixFromSource(source, ew1, &weights_arena, opts.verify_crcs),
                        .w2 = try loadQuantMatrixFromSource(source, ew2, &weights_arena, opts.verify_crcs),
                        .w3 = try loadQuantMatrixFromSource(source, ew3, &weights_arena, opts.verify_crcs),
                    } };
                }
            }
            lay.experts = exps;
            layers[l] = lay;
            layers_alloced = l + 1;
        }

        const rt = try allocator.create(DSpark3LRuntime);
        rt.* = .{
            .config = cfg,
            .weights_arena = weights_arena,
            .workspace = .{
                .arena = scratch_arena,
                .config = compat_cfg,
                .base_hidden = base_hidden,
                .slot_input = slot_input,
                .slot_hiddens = slot_hiddens,
                .slot_logits = slot_logits,
                .topk_tokens = topk_tokens,
                .topk_probs = topk_probs,
                .l1_draft_buf = l1_draft_buf,
            },
            .cold_pool = cold_pool,
            .sme_runner = sme_runner,
            .embed_hook = opts.embed_hook,
            .lm_head_hook = opts.lm_head_hook,
            .main_proj = main_proj,
            .main_norm = main_norm,
            .layers = layers,
            .wkv_dim = wkv_dim,
            .wo_a_in_dim = wo_a_in,
            .wo_a_out_dim = wo_a_out,
            .wo_b_in_dim = wo_b_in,
            .final_norm = final_norm,
            .markov_embed = markov_embed,
            .markov_head = markov_head,
            .confidence_proj = confidence_proj,
            .window_kv_cache = window_kv_cache,
            .main_concat = main_concat,
            .main_x = main_x,
            .h_streams = h_streams,
            .h_streams_res = h_streams_res,
            .pre_mix = pre_mix,
            .next_pre_mix = next_pre_mix,
            .post_mix = post_mix,
            .comb_mix = comb_mix,
            .hc_flat_buf = hc_flat_buf,
            .sub_in = sub_in,
            .sub_out = sub_out,
            .q_lora_buf = q_lora_buf,
            .q_all = q_all,
            .block_kv = block_kv,
            .attn_out = attn_out,
            .wo_a_out = wo_a_out_buf,
            .wo_b_in = wo_b_in_buf,
            .moe_scores = moe_scores,
            .moe_gate = moe_gate,
            .moe_up = moe_up,
            .moe_tmp = moe_tmp,
            .x_final = x_final,
            .markov_embeds = markov_embeds,
            .slot_confidences = slot_confidences,
            .conf_in = conf_in,
            .row_scratch = row_scratch,
        };
        return rt;
    }
};

fn dsparkJsonU32(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .integer => |iv| if (iv >= 0 and iv <= std.math.maxInt(u32)) @intCast(iv) else null,
        else => null,
    };
}

fn dsparkJsonF32(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .float => |fv| @floatCast(fv),
        .integer => |iv| @floatFromInt(iv),
        else => null,
    };
}

fn findTensorFlex(parsed: *const amf.Parsed, base: []const u8) ?*const amf.TensorEntry {
    if (parsed.find(base)) |e| return e;
    var buf: [96]u8 = undefined;
    const with_w = std.fmt.bufPrint(&buf, "{s}.weight", .{base}) catch return null;
    return parsed.find(with_w);
}

fn tensorElemCount(entry: *const amf.TensorEntry) usize {
    var total: usize = 1;
    const rank: usize = @min(@as(usize, entry.rank), 4);
    if (rank == 0) return 0;
    for (0..rank) |i| total *= entry.shape[i];
    return total;
}

fn loadQuantMatrixFromSource(
    source: DSpark3LAmfSource,
    entry: *const amf.TensorEntry,
    arena: *arena_mod.WiredArena,
    verify_crc: bool,
) LoadError!QuantMatrix {
    const blob = try source.loadBlob(entry, arena, verify_crc);
    return QuantMatrix.fromEntryAndBlob(entry, blob);
}

fn loadVecF32FromSource(
    allocator: std.mem.Allocator,
    source: DSpark3LAmfSource,
    entry: *const amf.TensorEntry,
    arena: *arena_mod.WiredArena,
    verify_crc: bool,
) LoadError![]const f32 {
    const elems = tensorElemCount(entry);
    if (elems == 0) return error.InvalidWeights;
    const dt = try entry.dataType();
    const raw = try source.loadRawAlloc(allocator, entry, verify_crc);
    defer allocator.free(raw);
    const dst_bytes = try arena.allocAligned(elems * @sizeOf(f32), CACHELINE_ALIGN);
    const out: []f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, dst_bytes)))[0..elems];
    switch (dt) {
        .f32 => {
            if (raw.len < elems * 4) return error.InvalidWeights;
            for (0..elems) |i| out[i] = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        },
        .f16 => {
            if (raw.len < elems * 2) return error.InvalidWeights;
            for (0..elems) |i| out[i] = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little))));
        },
        .bf16 => {
            if (raw.len < elems * 2) return error.InvalidWeights;
            for (0..elems) |i| out[i] = quant.bf16ToF32(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
        },
        else => return error.UnsupportedDType,
    }
    return out;
}

fn loadHcBlock(
    allocator: std.mem.Allocator,
    source: DSpark3LAmfSource,
    parsed: *const amf.Parsed,
    layer_idx: usize,
    tag: []const u8,
    hc_mult: u32,
    hidden_size: u32,
    arena: *arena_mod.WiredArena,
    verify_crc: bool,
) LoadError!HcBlockWeights {
    var nb: [96]u8 = undefined;
    const fn_e = findTensorFlex(parsed, std.fmt.bufPrint(&nb, "mtp.{d}.{s}_fn", .{ layer_idx, tag }) catch unreachable) orelse return error.MissingTensor;
    const bs_e = findTensorFlex(parsed, std.fmt.bufPrint(&nb, "mtp.{d}.{s}_base", .{ layer_idx, tag }) catch unreachable) orelse return error.MissingTensor;
    const sc_e = findTensorFlex(parsed, std.fmt.bufPrint(&nb, "mtp.{d}.{s}_scale", .{ layer_idx, tag }) catch unreachable) orelse return error.MissingTensor;

    const fn_mat = try loadQuantMatrixFromSource(source, fn_e, arena, verify_crc);
    const base_f32 = try loadVecF32FromSource(allocator, source, bs_e, arena, verify_crc);
    const scale_f32 = try loadVecF32FromSource(allocator, source, sc_e, arena, verify_crc);
    if (scale_f32.len < 3) return error.InvalidWeights;
    const mix_hc: u32 = (2 + hc_mult) * hc_mult;
    const is_full = (fn_mat.n == mix_hc and fn_mat.k == hc_mult * hidden_size and base_f32.len >= mix_hc);
    return .{
        .is_full_hf_shape = is_full,
        .fn_mat = fn_mat,
        .base_f32 = base_f32,
        .scale_f32 = scale_f32,
    };
}

fn evalLmHeadHook(
    hook: LmHeadHook,
    slot_hiddens: []const f32,
    block_width: u8,
    hidden_size: u32,
    vocab_size: u32,
    out_logits: []f32,
) Error!void {
    const w: usize = block_width;
    const h: usize = hidden_size;
    const v: usize = vocab_size;
    switch (hook) {
        .dense_f32 => |mat| {
            for (0..w) |s| matVecF32(mat, slot_hiddens[s * h ..][0..h], v, h, out_logits[s * v ..][0..v]);
        },
        .dense_f16 => |mat| {
            for (0..w) |s| matVecF16Weights(mat, slot_hiddens[s * h ..][0..h], v, h, out_logits[s * v ..][0..v]);
        },
        .dense_bf16 => |mat| {
            for (0..w) |s| matVecBf16Weights(mat, slot_hiddens[s * h ..][0..h], v, h, out_logits[s * v ..][0..v]);
        },
        .callback => |cb| {
            try cb.eval_fn(cb.ctx, slot_hiddens[0 .. w * h], block_width, hidden_size, vocab_size, out_logits[0 .. w * v]);
        },
    }
}

fn dequantMatrixRow(mat: *const QuantMatrix, row_idx: usize, out_row: []f32) Error!void {
    quant_formats.dequantRowCpuRef(
        mat.dtype,
        mat.data,
        mat.params,
        mat.n,
        mat.k,
        row_idx,
        mat.group_size,
        out_row,
    ) catch return error.InvalidWeights;
}

/// Zero-allocation multi-format quantized matrix-vector product (`out[r] = dot(W[r], x)`).
/// Specialized hot loops for `f32`, `f16`, `bf16`, `mxfp8_g32`, `gguf_mxfp4`, and full
/// `quant_formats.dequantRowCpuRef` coverage for `GSQ-RCO` (`q4_affine`, `q3_affine`,
/// `q2_affine`) and all remaining `.amf` block formats.
fn matVecQuantF32(
    mat: *const QuantMatrix,
    x: []const f32,
    out: []f32,
    row_scratch: []f32,
    accumulate: bool,
) Error!void {
    const n: usize = mat.n;
    const k: usize = mat.k;
    switch (mat.dtype) {
        .f32 => {
            if (mat.data.len < n * k * 4) return error.InvalidWeights;
            for (0..n) |r| {
                const row_b = mat.data[r * k * 4 ..][0 .. k * 4];
                var acc: f32 = 0.0;
                for (0..k) |c| {
                    const w: f32 = @bitCast(std.mem.readInt(u32, row_b[c * 4 ..][0..4], .little));
                    acc += w * x[c];
                }
                out[r] = if (accumulate) out[r] + acc else acc;
            }
        },
        .f16 => {
            if (mat.data.len < n * k * 2) return error.InvalidWeights;
            for (0..n) |r| {
                const row_b = mat.data[r * k * 2 ..][0 .. k * 2];
                var acc: f32 = 0.0;
                for (0..k) |c| {
                    const w: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, row_b[c * 2 ..][0..2], .little))));
                    acc += w * x[c];
                }
                out[r] = if (accumulate) out[r] + acc else acc;
            }
        },
        .bf16 => {
            if (mat.data.len < n * k * 2) return error.InvalidWeights;
            for (0..n) |r| {
                const row_b = mat.data[r * k * 2 ..][0 .. k * 2];
                var acc: f32 = 0.0;
                for (0..k) |c| {
                    const w = quant.bf16ToF32(std.mem.readInt(u16, row_b[c * 2 ..][0..2], .little));
                    acc += w * x[c];
                }
                out[r] = if (accumulate) out[r] + acc else acc;
            }
        },
        .mxfp8_g32 => {
            if ((k % 32) != 0) return error.InvalidWeights;
            const groups = k / 32;
            if (mat.data.len < n * k or mat.params.len < n * groups) return error.InvalidWeights;
            for (0..n) |r| {
                const row_d = mat.data[r * k ..][0..k];
                const row_s = mat.params[r * groups ..][0..groups];
                var acc: f32 = 0.0;
                for (0..groups) |g| {
                    const sc = quant_formats.ue8m0ToF32(row_s[g]);
                    var g_acc: f32 = 0.0;
                    const base = g * 32;
                    for (0..32) |j| {
                        g_acc += quant_formats.e4m3ToF32(row_d[base + j]) * x[base + j];
                    }
                    acc += g_acc * sc;
                }
                out[r] = if (accumulate) out[r] + acc else acc;
            }
        },
        .gguf_mxfp4 => {
            if ((k % 32) != 0) return error.InvalidWeights;
            const blocks = k / 32;
            const row_bytes = blocks * 17;
            if (mat.data.len < n * row_bytes) return error.InvalidWeights;
            for (0..n) |r| {
                const row_b = mat.data[r * row_bytes ..][0..row_bytes];
                var acc: f32 = 0.0;
                for (0..blocks) |b| {
                    const blk = row_b[b * 17 ..][0..17];
                    const sc = quant_formats.ue8m0ToF32(blk[0]);
                    var b_acc: f32 = 0.0;
                    const base = b * 32;
                    for (0..16) |j| {
                        const byte = blk[1 + j];
                        b_acc += quant_formats.E2M1_TABLE[byte & 0x0F] * x[base + j];
                        b_acc += quant_formats.E2M1_TABLE[byte >> 4] * x[base + 16 + j];
                    }
                    acc += b_acc * sc;
                }
                out[r] = if (accumulate) out[r] + acc else acc;
            }
        },
        else => {
            for (0..n) |r| {
                try dequantMatrixRow(mat, r, row_scratch[0..k]);
                var acc0: f32 = 0.0;
                var acc1: f32 = 0.0;
                var c: usize = 0;
                while (c + 2 <= k) : (c += 2) {
                    acc0 += row_scratch[c] * x[c];
                    acc1 += row_scratch[c + 1] * x[c + 1];
                }
                var acc = acc0 + acc1;
                while (c < k) : (c += 1) acc += row_scratch[c] * x[c];
                out[r] = if (accumulate) out[r] + acc else acc;
            }
        },
    }
}

inline fn sigmoidF32(x: f32) f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + @exp(-x));
    } else {
        const ex = @exp(x);
        return ex / (1.0 + ex);
    }
}

inline fn siluF32(x: f32) f32 {
    return x * sigmoidF32(x);
}

inline fn softplusF32(x: f32) f32 {
    if (x > 20.0) return x;
    if (x < -20.0) return @exp(x);
    return @log(1.0 + @exp(x));
}

fn hcCollapseStreams(streams: []const f32, pre: []const f32, hc: usize, h: usize, out: []f32) void {
    @memset(out[0..h], 0.0);
    for (0..hc) |s| {
        const w = pre[s];
        const st = streams[s * h ..][0..h];
        for (0..h) |d| out[d] += w * st[d];
    }
}

/// Exact `model.py:964-968`:
/// `y[c, d] = post[c] * sub_out[d] + sum_r comb[r, c] * residual[r, d]`.
fn hcExpandMixStreams(
    sub_out: []const f32,
    residual: []const f32,
    post: []const f32,
    comb: []const f32,
    hc: usize,
    h: usize,
    out: []f32,
) void {
    for (0..hc) |c| {
        const dst = out[c * h ..][0..h];
        const pc = post[c];
        for (0..h) |d| dst[d] = pc * sub_out[d];
        for (0..hc) |r| {
            const crc = comb[r * hc + c];
            const res_r = residual[r * h ..][0..h];
            for (0..h) |d| dst[d] += crc * res_r[d];
        }
    }
}

/// Interleaved complex-pair RoPE (`model.py:392-406`).
/// `inverse == true` applies the conjugate rotation used on attention outputs.
fn applyRopeInterleavedInPlace(vec: []f32, pos: u32, theta: f32, inverse: bool) void {
    const half = vec.len / 2;
    const pos_f: f64 = @floatFromInt(pos);
    const dim_f: f64 = @floatFromInt(half * 2);
    const theta_f: f64 = theta;
    for (0..half) |i| {
        const exp_val = @as(f64, @floatFromInt(2 * i)) / dim_f;
        const angle = pos_f / std.math.pow(f64, theta_f, exp_val);
        const c: f32 = @floatCast(@cos(angle));
        const s: f32 = @floatCast(@sin(angle));
        const x0 = vec[2 * i];
        const x1 = vec[2 * i + 1];
        if (!inverse) {
            vec[2 * i] = x0 * c - x1 * s;
            vec[2 * i + 1] = x0 * s + x1 * c;
        } else {
            vec[2 * i] = x0 * c + x1 * s;
            vec[2 * i + 1] = -x0 * s + x1 * c;
        }
    }
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
        // No duplicate token_ids among direct siblings.
        var sib = t.nodes[node.parent].first_child;
        var seen: usize = 0;
        while (sib != tree_mod.NO_NODE) : (sib = t.nodes[sib].next_sibling) {
            if (t.nodes[sib].token_id == node.token_id) seen += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), seen);
    }
}

test "D10 DSpark zero-alloc WiredArena workspace, encoder conditioning, valid G5 masks and D9 greedy verify" {
    const h: u32 = 8;
    const v: u32 = 16;
    const w: u8 = 4;
    const top_k: u8 = 2;

    const cfg: DSparkConfig = .{
        .block_width = w,
        .top_k_per_pos = top_k,
        .max_tree_nodes = 16,
        .hidden_size = h,
        .vocab_size = v,
    };
    const need_bytes = try DSparkWorkspace.requiredBytes(cfg);
    try std.testing.expectEqual(@as(usize, 0), need_bytes % page.APPLE_PAGE_SIZE);

    var ws = try DSparkWorkspace.init(cfg, testBudget(need_bytes * 2));
    defer ws.deinit();
    try std.testing.expectEqual(@as(u64, need_bytes), ws.totalWiredBytes());

    // Build deterministic weights: identity per-slot projections, orthogonal slot embeddings,
    // and lm_head selecting distinct tokens for each slot.
    var slot_projs: [w * h * h]f32 = @splat(0.0);
    for (0..w) |s| {
        for (0..h) |d| {
            slot_projs[s * h * h + d * h + d] = 1.0;
        }
    }
    var slot_embed: [w * h]f32 = @splat(0.0);
    // Slot s emphasizes basis channel (s * 2) and (s * 2 + 1).
    for (0..w) |s| {
        slot_embed[s * h + s * 2] = 3.0;
        slot_embed[s * h + s * 2 + 1] = 1.5;
    }

    var lm_head: [v * h]f32 = @splat(0.0);
    // Map token (2*s + 1) to channel (2*s) and token (2*s + 2) to channel (2*s + 1).
    for (0..w) |s| {
        lm_head[(2 * s + 1) * h + (2 * s)] = 2.0;
        lm_head[(2 * s + 2) * h + (2 * s + 1)] = 2.0;
    }
    // Encoder summary channel 0 pushes token 15 strongly when present.
    lm_head[15 * h + 0] = 4.0;

    var enc_proj: [h * h]f32 = @splat(0.0);
    enc_proj[0 * h + 0] = 1.0;

    const weights: DSparkWeights = .{
        .enc_proj = &enc_proj,
        .dec_proj = null,
        .slot_projs = &slot_projs,
        .slot_embed = &slot_embed,
        .slot_norm_weight = null,
        .lm_head = .{ .dense_f32 = &lm_head },
    };

    var proposer = try DSparkProposer.init(cfg, weights, &ws);
    const dec_h = [_]f32{ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };

    // Without encoder summary: slot 0 top-1 is token 15 (weight 4.0 on ch 0) and top-2 is token 1 (weight 2.0 on ch 0);
    // test determinism across back-to-back calls.
    var tree_a: CandidateTree = undefined;
    var tree_b: CandidateTree = undefined;
    const res_a = try proposer.propose(0, null, &dec_h, null, &tree_a);
    const res_b = try proposer.propose(0, null, &dec_h, null, &tree_b);

    try std.testing.expectEqual(res_a.tree_nodes, res_b.tree_nodes);
    try std.testing.expect(res_a.tree_nodes <= cfg.max_tree_nodes);
    try std.testing.expectEqual(@as(u16, 16), res_a.tree_nodes); // 1 + 2 + 4 + 8 + 1 capped at 16
    try std.testing.expectEqual(@as(u16, 4), res_a.max_depth);
    try verifyTreeInvariants(CandidateTree.capacity, &tree_a);

    for (0..tree_a.count) |i| {
        try std.testing.expectEqual(tree_a.nodes[i].token_id, tree_b.nodes[i].token_id);
        try std.testing.expectEqual(tree_a.nodes[i].parent, tree_b.nodes[i].parent);
        try std.testing.expectEqual(tree_a.nodes[i].ancestor_mask, tree_b.nodes[i].ancestor_mask);
        try std.testing.expectApproxEqAbs(tree_a.nodes[i].step_prob, tree_b.nodes[i].step_prob, 1e-7);
    }

    // With strong negative encoder summary on channel 0: shifts slot 0 away from token 15!
    const enc_summary = [_]f32{ -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    var tree_enc: CandidateTree = undefined;
    _ = try proposer.propose(0, &enc_summary, &dec_h, null, &tree_enc);
    try verifyTreeInvariants(CandidateTree.capacity, &tree_enc);
    try std.testing.expect(ws.slotTokens(0)[0] != tree_a.nodes[1].token_id);

    // Verify DSpark tree passes lossless D9 `accept.verifyGreedy` end-to-end.
    var target_argmax: [CandidateTree.capacity]u32 = @splat(14);
    // Force target model to match the highest-probability path from root down 3 steps.
    var cur: u16 = 0;
    var expected_accepted: [4]u32 = undefined;
    for (0..3) |step| {
        const child = tree_a.nodes[cur].first_child;
        try std.testing.expect(child != tree_mod.NO_NODE);
        target_argmax[cur] = tree_a.nodes[child].token_id;
        expected_accepted[step] = tree_a.nodes[child].token_id;
        cur = child;
    }
    target_argmax[cur] = 13; // bonus token
    expected_accepted[3] = 13;

    var acc_nodes: [8]u16 = undefined;
    var emitted: [8]u32 = undefined;
    const acc = try accept_mod.verifyGreedy(
        CandidateTree.capacity,
        &tree_a,
        target_argmax[0..tree_a.count],
        &acc_nodes,
        &emitted,
    );
    try std.testing.expectEqual(@as(u16, 3), acc.draft_accepted);
    try std.testing.expectEqual(@as(u16, 4), acc.emitted_token_count);
    try std.testing.expectEqualSlices(u32, &expected_accepted, emitted[0..4]);
}

test "D10 DSpark L1 SAM seed merging, GPU logit callback hook, and deterministic tie-breaking" {
    const gpa = std.testing.allocator;
    const h: u32 = 4;
    const v: u32 = 12;
    const w: u8 = 3;

    const cfg: DSparkConfig = .{
        .block_width = w,
        .top_k_per_pos = 2,
        .max_tree_nodes = 24,
        .hidden_size = h,
        .vocab_size = v,
        .l1_max_draft_len = 4,
    };
    const need_bytes = try DSparkWorkspace.requiredBytes(cfg);
    var ws = try DSparkWorkspace.init(cfg, testBudget(need_bytes * 2));
    defer ws.deinit();

    // Custom callback simulating GPU target logits with exact ties to test deterministic tie-breaking.
    const MockGpuCtx = struct {
        calls: usize = 0,
        fn eval(
            ctx: ?*anyopaque,
            slot_hiddens: []const f32,
            block_width: u8,
            hidden_size: u32,
            vocab_size: u32,
            out_logits: []f32,
        ) Error!void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            self.calls += 1;
            if (slot_hiddens.len != @as(usize, block_width) * hidden_size) {
                return error.DimensionMismatch;
            }
            @memset(out_logits, 0.0);
            // Slot 0: exact tie between token 5 and token 2 -> token 2 must rank ahead of token 5!
            out_logits[0 * vocab_size + 5] = 3.0;
            out_logits[0 * vocab_size + 2] = 3.0;
            // Slot 1: token 7 > token 4
            out_logits[1 * vocab_size + 7] = 4.0;
            out_logits[1 * vocab_size + 4] = 2.0;
            // Slot 2: token 9 > token 3
            out_logits[2 * vocab_size + 9] = 4.0;
            out_logits[2 * vocab_size + 3] = 1.0;
        }
    };
    var gpu_ctx: MockGpuCtx = .{};

    var slot_projs: [w * h * h]f32 = @splat(0.0);
    for (0..w) |s| {
        for (0..h) |d| slot_projs[s * h * h + d * h + d] = 1.0;
    }
    const slot_embed: [w * h]f32 = @splat(0.25);

    const weights: DSparkWeights = .{
        .slot_projs = &slot_projs,
        .slot_embed = &slot_embed,
        .lm_head = .{
            .callback = .{
                .ctx = &gpu_ctx,
                .eval_fn = MockGpuCtx.eval,
            },
        },
    };

    var proposer = try DSparkProposer.init(cfg, weights, &ws);

    // Populate L1 SuffixAutomaton with a repeated 5-token pattern followed by [2, 8, 10]
    // so `queryActive` yields a high-confidence `.l1_seed` (`match_len >= 4`) that overlaps
    // slot 0's top-1 token `2` (testing deduplication + seed tail extension!).
    var sam = try sam_mod.SuffixAutomaton.initAlloc(gpa, 64);
    defer sam.deinit(gpa);
    const history = [_]u32{ 1, 3, 5, 7, 9, 2, 8, 10, 11, 1, 3, 5, 7, 9 };
    try sam.appendSlice(&history);

    const dec_h = [_]f32{ 1.0, 0.5, -0.25, 0.75 };
    var out_tree: CandidateTree = undefined;
    const res = try proposer.propose(9, null, &dec_h, &sam, &out_tree);

    try std.testing.expectEqual(@as(usize, 1), gpu_ctx.calls);
    try std.testing.expect(res.l1_is_seed);
    try std.testing.expect(res.l1_match_len >= sam_mod.SEED_MIN_MATCH_LEN);
    try std.testing.expect(res.l1_inserted >= 3);
    try std.testing.expect(res.dspark_inserted > 0);
    try std.testing.expect(res.tree_nodes <= cfg.max_tree_nodes);
    try verifyTreeInvariants(CandidateTree.capacity, &out_tree);

    // Deterministic tie-breaking check on slot 0: token 2 before token 5.
    try std.testing.expectEqual(@as(u32, 2), ws.slotTokens(0)[0]);
    try std.testing.expectEqual(@as(u32, 5), ws.slotTokens(0)[1]);

    // Token 2 under root was deduplicated cleanly and retains `.l1_seed` provenance.
    const child_2 = out_tree.findChild(0, 2).?;
    try std.testing.expectEqual(tree_mod.NodeSource.l1_seed, out_tree.nodes[child_2].source);

    // Also confirm `dense_f16` and `dense_bf16` lm_head paths execute cleanly.
    var lm_f16: [v * h]f16 = @splat(0.0);
    var lm_bf16: [v * h]u16 = @splat(0);
    lm_f16[6 * h + 0] = 2.0;
    lm_bf16[6 * h + 0] = quant.f32ToBf16(2.0);

    var w_f16 = weights;
    w_f16.lm_head = .{ .dense_f16 = &lm_f16 };
    var p_f16 = try DSparkProposer.init(cfg, w_f16, &ws);
    var t_f16: CandidateTree = undefined;
    _ = try p_f16.propose(9, null, &dec_h, null, &t_f16);
    try verifyTreeInvariants(CandidateTree.capacity, &t_f16);
    try std.testing.expectEqual(@as(u32, 6), ws.slotTokens(0)[0]);

    var w_bf16 = weights;
    w_bf16.lm_head = .{ .dense_bf16 = &lm_bf16 };
    var p_bf16 = try DSparkProposer.init(cfg, w_bf16, &ws);
    var t_bf16: CandidateTree = undefined;
    _ = try p_bf16.propose(9, null, &dec_h, null, &t_bf16);
    try verifyTreeInvariants(CandidateTree.capacity, &t_bf16);
    try std.testing.expectEqual(@as(u32, 6), ws.slotTokens(0)[0]);
}

fn readTestFileAlloc(allocator: std.mem.Allocator, path: [:0]const u8) !?[]u8 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(allocator);
    var buf: [4096]u8 = undefined;
    while (true) {
        const rc = std.c.read(fd, &buf, buf.len);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.ReadFailed;
        }
        if (rc == 0) return try list.toOwnedSlice(allocator);
        try list.appendSlice(allocator, buf[0..@intCast(rc)]);
    }
}

test "M9.4 D10 real-HF 3L DSpark loads deepseek41-real-hf-synth/dspark.amf (138 tensors), cold-tier + markov_head + confidence_head + 0 hot-path allocs" {
    const allocator = std.testing.allocator;
    const env_c = std.c.getenv("AMINFER_MODELS_DIR") orelse return error.SkipZigTest;
    const models_dir = std.mem.span(env_c);
    if (models_dir.len == 0) return error.SkipZigTest;
    const amf_path = try std.fs.path.joinZ(allocator, &.{ models_dir, "deepseek41-real-hf-synth", "dspark.amf" });
    defer allocator.free(amf_path);
    const cfg_path = try std.fs.path.joinZ(allocator, &.{ models_dir, "deepseek41-real-hf-synth", "dspark_config.json" });
    defer allocator.free(cfg_path);

    const cfg_bytes = (try readTestFileAlloc(allocator, cfg_path)) orelse return error.SkipZigTest;
    defer allocator.free(cfg_bytes);

    var f = amf.AmfFile.open(allocator, amf_path) catch return error.SkipZigTest;
    try std.testing.expectEqual(@as(usize, 138), f.parsed.entries.len);
    try std.testing.expect(DSpark3LRuntime.isRealHf3LAmf(&f.parsed));
    f.close(allocator);

    const cfg_json = try DSpark3LConfig.fromJson(allocator, cfg_bytes);
    try std.testing.expectEqual(@as(u8, 3), cfg_json.num_layers);
    try std.testing.expectEqual(@as(u8, 5), cfg_json.block_size);
    try std.testing.expectEqual(@as(u32, 256), cfg_json.hidden_size);
    try std.testing.expectEqual(@as(u32, 2048), cfg_json.vocab_size);
    try std.testing.expectEqualSlices(u16, &.{ 0, 2, 3 }, cfg_json.targetLayerIdsSlice());

    // Deterministic target embedding callback (0 allocs).
    const MockEmbed = struct {
        fn read(_: ?*anyopaque, tok: u32, out: []f32) Error!void {
            const tf: f32 = @floatFromInt((tok % 37) + 1);
            for (out, 0..) |*dst, i| {
                dst.* = @sin(tf * 0.13 + @as(f32, @floatFromInt(i)) * 0.07) * 0.25;
            }
        }
    };
    const MockLmHead = struct {
        calls: usize = 0,
        fn eval(ctx: ?*anyopaque, sh: []const f32, bw: u8, hs: u32, vs: u32, out: []f32) Error!void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            self.calls += 1;
            for (0..bw) |s| {
                const row_h = sh[s * hs ..][0..hs];
                var acc: f32 = 0.0;
                for (row_h) |v| acc += v;
                const dst = out[s * vs ..][0..vs];
                for (0..vs) |v_i| {
                    dst[v_i] = @sin(@as(f32, @floatFromInt(v_i + 1)) * 0.03 + acc * 0.1) * 0.5;
                }
            }
        }
    };

    var lm_ctx: MockLmHead = .{};

    // 1. All-hot `markov_low_rank` mode (avoids 5x full [vocab, H] lm_head sweeps).
    var rt_hot = try DSpark3LRuntime.loadFromFile(allocator, amf_path, cfg_json, .{
        .embed_hook = .{ .read_fn = MockEmbed.read },
        .lm_head_hook = .{ .callback = .{ .ctx = @ptrCast(&lm_ctx), .eval_fn = MockLmHead.eval } },
        .lm_head_mode = .markov_low_rank,
        .verify_crcs = true,
    });
    defer rt_hot.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 64), rt_hot.config.markov_rank);
    try std.testing.expectEqual(@as(u32, 24), rt_hot.coldStats().hot_experts);
    try std.testing.expectEqual(@as(u32, 0), rt_hot.coldStats().cold_experts);

    var tap0: [256]f32 = undefined;
    var tap1: [256]f32 = undefined;
    var tap2: [256]f32 = undefined;
    for (0..256) |i| {
        const fi: f32 = @floatFromInt(i);
        tap0[i] = @cos(fi * 0.05) * 0.3;
        tap1[i] = @sin(fi * 0.08) * 0.3;
        tap2[i] = @cos(fi * 0.11 + 0.5) * 0.3;
    }
    const taps = [_][]const f32{ &tap0, &tap1, &tap2 };

    var proposer_hot = DSparkProposer.init3L(rt_hot);
    try std.testing.expect(proposer_hot.is3L());
    try proposer_hot.seed3LWindow(&taps, 0);
    proposer_hot.set3LTargetTaps(&taps, 1);

    var tree_a: CandidateTree = undefined;
    const res_a = try proposer_hot.propose(42, &tap0, &tap2, null, &tree_a);
    try verifyTreeInvariants(CandidateTree.capacity, &tree_a);
    try std.testing.expect(res_a.tree_nodes >= 4);
    try std.testing.expectEqual(@as(u32, 5), res_a.markov_evals);
    try std.testing.expectEqual(@as(u32, 5), res_a.lm_head_sweeps_avoided);
    try std.testing.expectEqual(@as(usize, 0), lm_ctx.calls);
    try std.testing.expect(rt_hot.confidence_proj != null);
    try std.testing.expect(std.math.isFinite(res_a.mean_confidence));
    try std.testing.expect(res_a.mean_confidence >= 0.0 and res_a.mean_confidence <= 1.0);
    try std.testing.expect(res_a.min_confidence >= 0.0 and res_a.min_confidence <= res_a.mean_confidence + 1e-6);

    // Determinism check after reset + same taps.
    proposer_hot.reset3LWindow();
    try proposer_hot.seed3LWindow(&taps, 0);
    proposer_hot.set3LTargetTaps(&taps, 1);
    var tree_b: CandidateTree = undefined;
    const res_b = try proposer_hot.propose(42, &tap0, &tap2, null, &tree_b);
    try std.testing.expectEqual(res_a.tree_nodes, res_b.tree_nodes);
    for (0..tree_a.count) |i| {
        try std.testing.expectEqual(tree_a.nodes[i].token_id, tree_b.nodes[i].token_id);
        try std.testing.expectApproxEqAbs(tree_a.nodes[i].step_prob, tree_b.nodes[i].step_prob, 1e-6);
    }

    // Confidence gating check: raising threshold above `max` gates tail positions cleanly.
    proposer_hot.setConfidenceThreshold(0.9999);
    proposer_hot.reset3LWindow();
    try proposer_hot.seed3LWindow(&taps, 0);
    proposer_hot.set3LTargetTaps(&taps, 1);
    var tree_gated: CandidateTree = undefined;
    const res_gated = try proposer_hot.propose(42, &tap0, &tap2, null, &tree_gated);
    try verifyTreeInvariants(CandidateTree.capacity, &tree_gated);
    try std.testing.expect(res_gated.gated_positions > 0);
    try std.testing.expect(res_gated.max_depth < res_a.max_depth);
    proposer_hot.setConfidenceThreshold(0.0);

    // 2. Hybrid `lm_head` + `markov_head` scoring mode.
    rt_hot.config.lm_head_mode = .hybrid_target_plus_markov;
    proposer_hot.reset3LWindow();
    try proposer_hot.seed3LWindow(&taps, 0);
    proposer_hot.set3LTargetTaps(&taps, 1);
    var tree_hyb: CandidateTree = undefined;
    const res_hyb = try proposer_hot.propose(42, &tap0, &tap2, null, &tree_hyb);
    try verifyTreeInvariants(CandidateTree.capacity, &tree_hyb);
    try std.testing.expectEqual(@as(usize, 1), lm_ctx.calls);
    try std.testing.expectEqual(@as(u32, 5), res_hyb.markov_evals);
    try std.testing.expectEqual(@as(u32, 0), res_hyb.lm_head_sweeps_avoided);

    // 3. Cold-tiered (`F_NOCACHE` pread + `C27` hot cache) matches all-hot bit-for-bit!
    var rt_cold = try DSpark3LRuntime.loadFromFile(allocator, amf_path, cfg_json, .{
        .embed_hook = .{ .read_fn = MockEmbed.read },
        .lm_head_mode = .markov_low_rank,
        .force_cold_experts = true,
        .cold_slot_pool_slots = 12,
        .verify_crcs = true,
    });
    defer rt_cold.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 0), rt_cold.coldStats().hot_experts);
    try std.testing.expectEqual(@as(u32, 24), rt_cold.coldStats().cold_experts);
    try std.testing.expect(rt_cold.weights_arena.bytes().len < rt_hot.weights_arena.bytes().len);

    var proposer_cold = DSparkProposer.init3L(rt_cold);
    try proposer_cold.seed3LWindow(&taps, 0);
    proposer_cold.set3LTargetTaps(&taps, 1);
    var tree_cold: CandidateTree = undefined;
    const res_cold = try proposer_cold.propose(42, &tap0, &tap2, null, &tree_cold);
    try std.testing.expectEqual(res_a.tree_nodes, res_cold.tree_nodes);
    for (0..tree_a.count) |i| {
        try std.testing.expectEqual(tree_a.nodes[i].token_id, tree_cold.nodes[i].token_id);
        try std.testing.expectApproxEqAbs(tree_a.nodes[i].step_prob, tree_cold.nodes[i].step_prob, 1e-6);
    }
    const cs1 = proposer_cold.coldStats().?;
    try std.testing.expect(cs1.cache_misses > 0);
    try std.testing.expect(cs1.nvme_bytes_read > 0);
    // Second round reuses C27 hot-cached experts with cache hits!
    proposer_cold.set3LTargetTaps(&taps, 2);
    _ = try proposer_cold.propose(42, &tap0, &tap2, null, &tree_cold);
    const cs2 = proposer_cold.coldStats().?;
    try std.testing.expect(cs2.cache_hits > cs1.cache_hits);

    // `cold_expert_min_weight` tail skipping on cold misses.
    var rt_skip = try DSpark3LRuntime.loadFromFile(allocator, amf_path, cfg_json, .{
        .embed_hook = .{ .read_fn = MockEmbed.read },
        .lm_head_mode = .markov_low_rank,
        .force_cold_experts = true,
        .cold_slot_pool_slots = 4,
        .cold_expert_min_weight = 10.0,
        .verify_crcs = false,
    });
    defer rt_skip.deinit(allocator);
    var proposer_skip = DSparkProposer.init3L(rt_skip);
    try proposer_skip.seed3LWindow(&taps, 0);
    proposer_skip.set3LTargetTaps(&taps, 1);
    var tree_skip: CandidateTree = undefined;
    _ = try proposer_skip.propose(42, &tap0, &tap2, null, &tree_skip);
    try std.testing.expect(proposer_skip.coldStats().?.skipped_low_weight > 0);
    try std.testing.expectEqual(@as(u64, 0), proposer_skip.coldStats().?.nvme_bytes_read);
}

test "M9.4 D10 3L DSpark GSQ-RCO (q4/q3/q2_affine) + full HF mHC [24, 4*H] + D13 SME2 sme2_panel64 main_proj parity" {
    const allocator = std.testing.allocator;
    const h: u32 = 64;
    const v: u32 = 32;
    const hc: u32 = 4;
    const mix_hc: u32 = (2 + hc) * hc; // 24
    const n_heads: u32 = 2;
    const hd: u32 = 32;
    const q_lora: u32 = 32;
    const o_groups: u32 = 2;
    const o_lora: u32 = 16;
    const moe_i: u32 = 64;
    const n_exp: u32 = 3;
    const mr: u32 = 16;

    const BlobRec = struct {
        entry: amf.TensorEntry,
        bytes: []u8,
    };

    const buildSynth3L = struct {
        fn fillBf16(alloc: std.mem.Allocator, seed: u64, count: usize, scale: f32) ![]u8 {
            var prng = std.Random.DefaultPrng.init(seed);
            const r = prng.random();
            const raw = try alloc.alloc(u8, count * 2);
            for (0..count) |i| {
                const val: f32 = (r.float(f32) * 2.0 - 1.0) * scale;
                std.mem.writeInt(u16, raw[i * 2 ..][0..2], quant.f32ToBf16(val), .little);
            }
            return raw;
        }
        fn fillF32(alloc: std.mem.Allocator, seed: u64, count: usize, scale: f32, bias: f32) ![]u8 {
            var prng = std.Random.DefaultPrng.init(seed);
            const r = prng.random();
            const raw = try alloc.alloc(u8, count * 4);
            for (0..count) |i| {
                const val: f32 = (r.float(f32) * 2.0 - 1.0) * scale + bias;
                std.mem.writeInt(u32, raw[i * 4 ..][0..4], @bitCast(val), .little);
            }
            return raw;
        }
        fn fillAffine(alloc: std.mem.Allocator, seed: u64, dt: amf.DType, n_rows: u32, k_cols: u32) !struct { blob: []u8, data_b: u64, p_off: u64, p_b: u64 } {
            const gs: usize = 64;
            const bw: usize = switch (dt) {
                .q4_affine => 4,
                .q3_affine => 3,
                .q2_affine => 2,
                else => unreachable,
            };
            const row_b = (@as(usize, k_cols) * bw) / 8;
            const data_b: usize = @as(usize, n_rows) * row_b;
            const p_off: usize = (data_b + 63) & ~@as(usize, 63);
            const groups = @as(usize, k_cols) / gs;
            const p_b: usize = @as(usize, n_rows) * groups * @sizeOf(quant.ScaleBias);
            const blob = try alloc.alloc(u8, p_off + p_b);
            @memset(blob, 0);
            var prng = std.Random.DefaultPrng.init(seed);
            const r = prng.random();
            for (0..data_b) |i| blob[i] = r.int(u8);
            const center_bias: f32 = switch (dt) {
                .q4_affine => -0.15,
                .q3_affine => -0.07,
                .q2_affine => -0.03,
                else => unreachable,
            };
            const sbs: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, blob[p_off .. p_off + p_b]));
            for (sbs) |*sb| {
                sb.* = .{ .scale = @floatCast(@as(f32, 0.02)), .bias = @floatCast(center_bias) };
            }
            return .{ .blob = blob, .data_b = data_b, .p_off = p_off, .p_b = p_b };
        }
        fn build(alloc: std.mem.Allocator, pack_main_proj_sme2: bool) ![]u8 {
            var recs: std.ArrayList(BlobRec) = .empty;
            defer {
                for (recs.items) |rec| alloc.free(rec.bytes);
                recs.deinit(alloc);
            }
            const addRaw = struct {
                fn f(a: std.mem.Allocator, list: *std.ArrayList(BlobRec), name: []const u8, kind: amf.TensorKind, dt: amf.DType, shape: [4]u32, rank: u8, layer: u16, exp: u16, bytes: []u8, gs: u16, data_b: u64, p_off: u64, p_b: u64, lay: amf.TensorLayout) !void {
                    var e = amf.TensorEntry.zeroed(name, kind, dt);
                    e.rank = rank;
                    e.shape = shape;
                    e.layer = layer;
                    e.expert = exp;
                    e.group_size = gs;
                    e.quant_bits = switch (dt) {
                        .q2_affine => 2,
                        .q3_affine => 3,
                        .q4_affine => 4,
                        .q5_affine => 5,
                        .q6_affine => 6,
                        .q8_affine => 8,
                        else => 0,
                    };
                    e.data_bytes = data_b;
                    e.params_offset = p_off;
                    e.params_bytes = p_b;
                    e.setLayoutAndParamKind(lay, .affine_scale_bias);
                    try list.append(a, .{ .entry = e, .bytes = bytes });
                }
            }.f;

            // mtp.0.main_proj [h, 3*h] (either row-major bf16 or pre-packed sme2_panel64 bf16).
            const mp_rm = try fillBf16(alloc, 7001, h * (3 * h), 0.15);
            if (pack_main_proj_sme2) {
                defer alloc.free(mp_rm);
                const packed_sz = try sme.packedBlobBytes(.bf16, h, 3 * h);
                const mp_pk = try alloc.alloc(u8, packed_sz);
                const src_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, mp_rm));
                try sme.packDensePanel64(.bf16, src_u16, h, 3 * h, mp_pk);
                try addRaw(alloc, &recs, "mtp.0.main_proj", .dense, .bf16, .{ h, 3 * h, 0, 0 }, 2, 0, amf.NO_EXPERT, mp_pk, 0, packed_sz, 0, 0, .sme2_panel64);
            } else {
                try addRaw(alloc, &recs, "mtp.0.main_proj", .dense, .bf16, .{ h, 3 * h, 0, 0 }, 2, 0, amf.NO_EXPERT, mp_rm, 0, mp_rm.len, 0, 0, .row_major);
            }
            const mn = try fillF32(alloc, 7002, h, 0.05, 1.0);
            try addRaw(alloc, &recs, "mtp.0.main_norm.weight", .norm, .f32, .{ h, 0, 0, 0 }, 1, 0, amf.NO_EXPERT, mn, 0, mn.len, 0, 0, .row_major);

            var nb: [64]u8 = undefined;
            const exp_dts = [_]amf.DType{ .q4_affine, .q3_affine, .q2_affine };
            for (0..3) |l| {
                const l16: u16 = @intCast(l);
                const base_s: u64 = 7100 + l * 100;
                inline for (.{ "hc_attn", "hc_ffn" }, 0..) |tag, ti| {
                    const fn_b = try fillBf16(alloc, base_s + ti * 10 + 1, mix_hc * (hc * h), 0.05);
                    try addRaw(alloc, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.{s}_fn", .{ l, tag }) catch unreachable, .dense, .bf16, .{ mix_hc, hc * h, 0, 0 }, 2, l16, amf.NO_EXPERT, fn_b, 0, fn_b.len, 0, 0, .row_major);
                    const bs_b = try fillF32(alloc, base_s + ti * 10 + 2, mix_hc, 0.1, 0.0);
                    try addRaw(alloc, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.{s}_base", .{ l, tag }) catch unreachable, .dense, .f32, .{ mix_hc, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, bs_b, 0, bs_b.len, 0, 0, .row_major);
                    const sc_b = try fillF32(alloc, base_s + ti * 10 + 3, 3, 0.02, 0.1);
                    try addRaw(alloc, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.{s}_scale", .{ l, tag }) catch unreachable, .dense, .f32, .{ 3, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, sc_b, 0, sc_b.len, 0, 0, .row_major);
                }
                inline for (.{
                    .{ "attn_norm.weight", h },
                    .{ "ffn_norm.weight", h },
                    .{ "attn.q_norm.weight", q_lora },
                    .{ "attn.kv_norm.weight", hd },
                }, 0..) |pair, ni| {
                    const nb_raw = try fillF32(alloc, base_s + 30 + ni, pair[1], 0.05, 1.0);
                    try addRaw(alloc, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.{s}", .{ l, pair[0] }) catch unreachable, .norm, .f32, .{ pair[1], 0, 0, 0 }, 1, l16, amf.NO_EXPERT, nb_raw, 0, nb_raw.len, 0, 0, .row_major);
                }
                const sink_b = try fillF32(alloc, base_s + 39, n_heads, 0.1, -0.5);
                try addRaw(alloc, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.attn.attn_sink", .{l}) catch unreachable, .dense, .f32, .{ n_heads, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, sink_b, 0, sink_b.len, 0, 0, .row_major);
                inline for (.{
                    .{ "attn.wq_a", q_lora, h },
                    .{ "attn.wq_b", n_heads * hd, q_lora },
                    .{ "attn.wkv", hd, h },
                    .{ "attn.wo_a", o_groups * o_lora, (n_heads * hd) / o_groups },
                    .{ "attn.wo_b", h, o_groups * o_lora },
                    .{ "ffn.gate.weight", n_exp, h },
                    .{ "ffn.shared_experts.w1", moe_i, h },
                    .{ "ffn.shared_experts.w2", h, moe_i },
                    .{ "ffn.shared_experts.w3", moe_i, h },
                }, 0..) |spec, mi| {
                    const mb = try fillBf16(alloc, base_s + 40 + mi, spec[1] * spec[2], 0.1);
                    try addRaw(alloc, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.{s}", .{ l, spec[0] }) catch unreachable, .dense, .bf16, .{ spec[1], spec[2], 0, 0 }, 2, l16, amf.NO_EXPERT, mb, 0, mb.len, 0, 0, .row_major);
                }
                const gb = try fillF32(alloc, base_s + 55, n_exp, 0.05, 0.0);
                try addRaw(alloc, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.gate.bias", .{l}) catch unreachable, .router, .f32, .{ n_exp, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, gb, 0, gb.len, 0, 0, .row_major);

                // Mixed GSQ-RCO (`q4_affine` hot, `q3_affine` hot, `q2_affine` cold!).
                for (0..n_exp) |e_i| {
                    const edt = exp_dts[e_i];
                    const ekind: amf.TensorKind = if (e_i == 2) .cold_expert else .hot_expert;
                    const w1 = try fillAffine(alloc, base_s + 60 + e_i * 3 + 0, edt, moe_i, h);
                    try addRaw(alloc, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.experts.{d}.w1.weight", .{ l, e_i }) catch unreachable, ekind, edt, .{ moe_i, h, 0, 0 }, 2, l16, @intCast(e_i), w1.blob, 64, w1.data_b, w1.p_off, w1.p_b, .row_major);
                    const w2 = try fillAffine(alloc, base_s + 60 + e_i * 3 + 1, edt, h, moe_i);
                    try addRaw(alloc, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.experts.{d}.w2.weight", .{ l, e_i }) catch unreachable, ekind, edt, .{ h, moe_i, 0, 0 }, 2, l16, @intCast(e_i), w2.blob, 64, w2.data_b, w2.p_off, w2.p_b, .row_major);
                    const w3 = try fillAffine(alloc, base_s + 60 + e_i * 3 + 2, edt, moe_i, h);
                    try addRaw(alloc, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.experts.{d}.w3.weight", .{ l, e_i }) catch unreachable, ekind, edt, .{ moe_i, h, 0, 0 }, 2, l16, @intCast(e_i), w3.blob, 64, w3.data_b, w3.p_off, w3.p_b, .row_major);
                }
            }
            const fn_last = try fillF32(alloc, 7900, h, 0.05, 1.0);
            try addRaw(alloc, &recs, "mtp.2.norm.weight", .norm, .f32, .{ h, 0, 0, 0 }, 1, 2, amf.NO_EXPERT, fn_last, 0, fn_last.len, 0, 0, .row_major);
            const me_b = try fillBf16(alloc, 7901, v * mr, 0.2);
            try addRaw(alloc, &recs, "mtp.2.markov_head.embed.weight", .embedding, .bf16, .{ v, mr, 0, 0 }, 2, 2, amf.NO_EXPERT, me_b, 0, me_b.len, 0, 0, .row_major);
            const mh_b = try fillBf16(alloc, 7902, v * mr, 0.2);
            try addRaw(alloc, &recs, "mtp.2.markov_head.head.weight", .dense, .bf16, .{ v, mr, 0, 0 }, 2, 2, amf.NO_EXPERT, mh_b, 0, mh_b.len, 0, 0, .row_major);
            const cp_b = try fillF32(alloc, 7903, h + mr, 0.01, 0.0);
            try addRaw(alloc, &recs, "mtp.2.confidence_head.proj.weight", .dense, .f32, .{ 1, h + mr, 0, 0 }, 2, 2, amf.NO_EXPERT, cp_b, 0, cp_b.len, 0, 0, .row_major);

            // Serialize valid 16 KiB-aligned AMF image.
            const dir_bytes = try page.alignUp(recs.items.len * @sizeOf(amf.TensorEntry));
            var total_sz: usize = page.APPLE_PAGE_SIZE + dir_bytes;
            for (recs.items) |*rec| {
                const ab = try page.alignUp(rec.bytes.len);
                rec.entry.blob_offset = total_sz;
                rec.entry.blob_bytes = ab;
                total_sz += ab;
            }
            const img = try alloc.alloc(u8, total_sz);
            @memset(img, 0);
            for (recs.items) |*rec| {
                const off: usize = @intCast(rec.entry.blob_offset);
                const blen: usize = @intCast(rec.entry.blob_bytes);
                @memcpy(img[off .. off + rec.bytes.len], rec.bytes);
                rec.entry.blob_crc32 = std.hash.Crc32.hash(img[off .. off + blen]);
            }
            for (recs.items, 0..) |rec, i| {
                const dst = img[page.APPLE_PAGE_SIZE + i * @sizeOf(amf.TensorEntry) ..][0..@sizeOf(amf.TensorEntry)];
                @memcpy(dst, std.mem.asBytes(&rec.entry));
            }
            const dir_crc = std.hash.Crc32.hash(img[page.APPLE_PAGE_SIZE .. page.APPLE_PAGE_SIZE + recs.items.len * @sizeOf(amf.TensorEntry)]);
            var hdr: amf.Header = std.mem.zeroes(amf.Header);
            hdr.magic = amf.MAGIC;
            hdr.version = amf.FORMAT_VERSION;
            hdr.page_size = page.APPLE_PAGE_SIZE;
            hdr.checksum_algo = amf.CHECKSUM_CRC32_IEEE;
            hdr.tensor_count = @intCast(recs.items.len);
            hdr.directory_offset = page.APPLE_PAGE_SIZE;
            hdr.directory_bytes = dir_bytes;
            hdr.directory_crc32 = dir_crc;
            const mid = "synth-dspark-3l-gsq-rco";
            @memcpy(hdr.model_id[0..mid.len], mid);
            hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
            @memcpy(img[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
            return img;
        }
    }.build;

    const img_rm = try buildSynth3L(allocator, false);
    defer allocator.free(img_rm);
    const img_sme = try buildSynth3L(allocator, true);
    defer allocator.free(img_sme);

    const cfg_3l: DSpark3LConfig = .{
        .num_layers = 3,
        .block_size = 5,
        .top_k_per_pos = 2,
        .max_tree_nodes = 24,
        .hidden_size = h,
        .vocab_size = v,
        .num_attention_heads = n_heads,
        .head_dim = hd,
        .q_lora_rank = q_lora,
        .kv_lora_rank = hd,
        .qk_rope_head_dim = 16,
        .o_lora_rank = o_lora,
        .o_groups = o_groups,
        .n_routed_experts = n_exp,
        .routed_top_k = 3,
        .moe_intermediate_size = moe_i,
        .hc_mult = hc,
        .markov_rank = mr,
        .lm_head_mode = .markov_low_rank,
    };

    var rt_rm = try DSpark3LRuntime.loadFromBytes(allocator, img_rm, cfg_3l, .{ .verify_crcs = true });
    defer rt_rm.deinit(allocator);
    var rt_sme = try DSpark3LRuntime.loadFromBytes(allocator, img_sme, cfg_3l, .{ .verify_crcs = true });
    defer rt_sme.deinit(allocator);

    try std.testing.expect(rt_rm.layers[0].hc_attn.is_full_hf_shape);
    try std.testing.expectEqual(@as(u32, 6), rt_rm.coldStats().hot_experts);
    try std.testing.expectEqual(@as(u32, 3), rt_rm.coldStats().cold_experts);

    var t0: [h]f32 = undefined;
    var t1: [h]f32 = undefined;
    var t2: [h]f32 = undefined;
    for (0..h) |i| {
        const fi: f32 = @floatFromInt(i);
        t0[i] = @sin(fi * 0.13) * 0.4;
        t1[i] = @cos(fi * 0.17) * 0.4;
        t2[i] = @sin(fi * 0.21 + 0.3) * 0.4;
    }
    const taps = [_][]const f32{ &t0, &t1, &t2 };

    var p_rm = DSparkProposer.init3L(rt_rm);
    var p_sme = DSparkProposer.init3L(rt_sme);
    try p_rm.seed3LWindow(&taps, 0);
    try p_sme.seed3LWindow(&taps, 0);
    p_rm.set3LTargetTaps(&taps, 1);
    p_sme.set3LTargetTaps(&taps, 1);

    var tr_rm: CandidateTree = undefined;
    var tr_sme: CandidateTree = undefined;
    const res_rm = try p_rm.propose(3, &t0, &t2, null, &tr_rm);
    const res_sme = try p_sme.propose(3, &t0, &t2, null, &tr_sme);
    try verifyTreeInvariants(CandidateTree.capacity, &tr_rm);
    try verifyTreeInvariants(CandidateTree.capacity, &tr_sme);
    try std.testing.expect(res_rm.mean_confidence > 0.0 and res_rm.mean_confidence < 1.0);
    try std.testing.expectApproxEqAbs(res_rm.mean_confidence, res_sme.mean_confidence, 1e-2);

    try std.testing.expect(rt_sme.sme2_eval_count >= 2);
    try std.testing.expect(rt_rm.coldStats().cache_misses > 0);
    // Row-major vs SME2 panel64 main_proj numerical parity check on slot logits.
    var dot: f64 = 0.0;
    var na: f64 = 0.0;
    var nb_sq: f64 = 0.0;
    for (rt_rm.workspace.slot_logits, rt_sme.workspace.slot_logits) |a, b| {
        dot += @as(f64, a) * @as(f64, b);
        na += @as(f64, a) * @as(f64, a);
        nb_sq += @as(f64, b) * @as(f64, b);
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb_sq) + 1e-12);
    try std.testing.expect(cos >= 0.999);
    try std.testing.expectEqual(rt_rm.workspace.slotTokens(0)[0], rt_sme.workspace.slotTokens(0)[0]);
}
