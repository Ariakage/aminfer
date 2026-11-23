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
};

/// Semi-autoregressive DSpark speculative proposer (`D10`).
pub const DSparkProposer = struct {
    config: DSparkConfig,
    weights: DSparkWeights,
    workspace: *DSparkWorkspace,
    rounds_proposed: u64 = 0,

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
        };
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
