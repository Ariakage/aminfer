//! End-to-end speculative decoding engine combining:
//! - L1 CPU Suffix Automaton (`D1`, `src/spec/sam.zig`),
//! - L2 EAGLE-3 / EAGLE-2 / MTP draft heads with benchmark-gated ANE/GPU placement
//!   (`D2`, `D3`, `D5`, `src/spec/draft.zig`, ADR-0008, ADR-0013),
//! - Dual-source prefix trie merging (`D6`, `src/spec/tree.zig`, ADR-0005),
//! - Expert-byte-budget submodular tree pruning (`D7`, `src/spec/tree.zig`, ADR-0009),
//! - L3 GPU tree-attention verification and KV commit (`G5`, `src/model/qwen3.zig`),
//! - Lossless greedy (`T = 0`) and stochastic (`T > 0`) verification (`D9`, `src/spec/accept.zig`).
//!
//! All scratch state, suffix automaton storage, draft trees, expert bitsets, and
//! residual sampling buffers are preallocated once at startup. The decode round
//! hot path (`stepRound`) performs zero dynamic heap allocation.

const std = @import("std");
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const arena_mod = @import("../mem/arena.zig");
const metal = @import("../bridge/metal.zig");
const moe = @import("../gpu/moe.zig");
const attention = @import("../gpu/attention.zig");
const qwen3 = @import("../model/qwen3.zig");
const sam_mod = @import("sam.zig");
const tree_mod = @import("tree.zig");
const accept_mod = @import("accept.zig");
const draft_mod = @import("draft.zig");

pub const MAX_TREE_NODES: u16 = tree_mod.MAX_SUPPORTED_NODES;
const MAX_PRED_LAYERS: usize = qwen3.MAX_EXPERT_PRED_LAYERS;

pub const SpecConfig = struct {
    mode: draft_mod.DraftMode = .l1_only,
    placement: draft_mod.Placement = .auto,
    /// Hard node cap passed to D7 pruning and L3 verification (`2 <= max_tree_nodes <= 64`).
    max_tree_nodes: u16 = 16,
    /// Maximum speculative depth beyond root `0` (`>= 1`).
    max_draft_depth: u16 = 4,
    /// Number of frontier nodes stepped per EAGLE-2 dynamic expansion level.
    branch_width: u16 = 2,
    /// Top-K candidates fetched per expanded draft node (`1..=MAX_DRAFT_TOP_K`).
    top_k_per_node: u32 = 3,
    /// Maximum L1 suffix automaton continuation length proposed per round.
    l1_max_draft: u16 = 12,
    /// Conditional step probability assigned to L1 seed chains (`match_len >= 4`, ADR-0005).
    l1_seed_step_prob: f32 = 0.88,
    /// Conditional step probability assigned to short L1 branches (`1 <= match_len < 4`).
    l1_branch_step_prob: f32 = 0.45,
    /// Minimum cumulative probability required to expand a frontier node with L2.
    min_frontier_prob: f32 = 0.12,
    /// Sampling temperature: `0.0` = lossless greedy, `> 0.0` = lossless stochastic.
    temperature: f32 = 0.0,
    /// Enable D7 expert-byte-budget submodular pruning (`false` verifies unpruned fixed tree).
    enable_pruning: bool = true,
    /// L1 Suffix Automaton token capacity (`0` defaults to `target.max_seq_len + 64`).
    sam_capacity: u32 = 0,
    /// Optional explicit D7 cost model (`null` auto-calibrates from target geometry).
    cost_model: ?tree_mod.PruneCostModel = null,
};

pub const SpecStats = struct {
    prefill_tokens: u64 = 0,
    rounds: u64 = 0,
    /// Total tokens emitted across decode rounds (sum of `1 + draft_accepted`).
    emitted_tokens: u64 = 0,
    /// Total speculative draft tokens accepted (excluding bonus tokens).
    draft_accepted: u64 = 0,
    /// Total candidate nodes constructed before D7 pruning across all rounds.
    drafted_nodes: u64 = 0,
    /// Total nodes actually sent to L3 GPU verification across all rounds.
    verified_nodes: u64 = 0,
    l1_seed_rounds: u64 = 0,
    l1_branch_rounds: u64 = 0,
    /// Sum of exact hardware-measured unique active experts across all layers & rounds.
    total_unique_experts: u64 = 0,
    /// Total layer passes verified (`rounds * num_hidden_layers` for MoE).
    total_moe_layer_passes: u64 = 0,
    /// Total target algorithmic bytes moved across all verification passes.
    total_bytes_moved: u64 = 0,
    /// Histogram indexed by emitted token count per round (`1..64`, `0` unused).
    accept_hist: [MAX_TREE_NODES + 1]u64 = @splat(0),
    draft_ns: u64 = 0,
    prune_ns: u64 = 0,
    verify_ns: u64 = 0,
    commit_sync_ns: u64 = 0,

    pub fn meanAcceptanceLength(self: *const SpecStats) f64 {
        if (self.rounds == 0) return 0.0;
        return @as(f64, @floatFromInt(self.emitted_tokens)) / @as(f64, @floatFromInt(self.rounds));
    }

    pub fn meanDraftedNodes(self: *const SpecStats) f64 {
        if (self.rounds == 0) return 0.0;
        return @as(f64, @floatFromInt(self.drafted_nodes)) / @as(f64, @floatFromInt(self.rounds));
    }

    pub fn meanVerifiedNodes(self: *const SpecStats) f64 {
        if (self.rounds == 0) return 0.0;
        return @as(f64, @floatFromInt(self.verified_nodes)) / @as(f64, @floatFromInt(self.rounds));
    }

    pub fn meanUniqueExpertsPerLayer(self: *const SpecStats) f64 {
        if (self.total_moe_layer_passes == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_unique_experts)) /
            @as(f64, @floatFromInt(self.total_moe_layer_passes));
    }

    pub fn bytesPerAcceptedToken(self: *const SpecStats) f64 {
        if (self.emitted_tokens == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_bytes_moved)) /
            @as(f64, @floatFromInt(self.emitted_tokens));
    }
};

pub const RoundSummary = struct {
    emitted_count: u16,
    draft_accepted: u16,
    drafted_nodes: u16,
    verified_nodes: u16,
    l1_match_len: u32,
    l1_is_seed: bool,
    avg_unique_experts_per_layer: f64,
    actual_bytes_moved: u64,
};

pub const InitError = error{
    InvalidConfig,
    TreeNotEnabled,
    DraftHeadRequired,
    DraftModeMismatch,
    FeatureTapsRequired,
    BudgetExceeded,
    OutOfMemory,
} || sam_mod.InitError || arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError ||
    budget_mod.PlanError;

pub const RunError = error{
    EmptyPrompt,
    NotPrefilled,
    SequenceFull,
    OutputBufferTooSmall,
    TokenOutOfRange,
    MissingRng,
} || qwen3.StepError || draft_mod.DraftStepError || tree_mod.PruneError || accept_mod.VerifyError;

/// Derives a calibrated M4 hardware cost model for D7 submodular tree pruning
/// (ADR-0009) from the loaded target model geometry and Spike S5 bandwidth curve.
pub fn defaultCostModelForTarget(target: *const qwen3.Qwen3Model) tree_mod.PruneCostModel {
    if (target.config.arch == .qwen3_moe) {
        const n_layers = target.config.num_hidden_layers;
        const sampled: u32 = @min(n_layers, qwen3.MAX_EXPERT_PRED_LAYERS);
        const layer_scale: f64 = @as(f64, @floatFromInt(n_layers)) / @as(f64, @floatFromInt(sampled));
        // Raw token-embedding router predictions overestimate inter-node expert
        // divergence vs contextual hidden states by ~1.35x; apply a 0.72 correlation
        // calibration factor so D7 matches measured 48-layer unique-expert growth.
        const scaled_expert_bytes: f32 = @floatCast(
            @as(f64, @floatFromInt(target.singleExpertBytes())) * layer_scale * 0.72,
        );
        const fixed_bytes: u64 = target.totalLayersNonExpertBytes() + target.global_head_bytes;
        const t_fixed_us: f32 = @floatCast(@as(f64, @floatFromInt(fixed_bytes)) / (75.0 * 1000.0) + 1500.0);

        // Smooth CSR-deduplicated MoE effective bandwidth curve calibrated from M4 Spike S5
        // (avoids an artificial M=1 -> M=2 cliff that starves high-confidence depth-1..4 nodes).
        var beta: [65]f32 = undefined;
        for (0..65) |m| {
            beta[m] = if (m <= 2)
                66.0
            else if (m <= 4)
                62.0
            else if (m <= 8)
                56.0
            else if (m <= 16)
                48.0
            else
                42.0;
        }
        return .{
            .t_fixed_us = t_fixed_us,
            .c_tok_us = 160.0,
            .expert_bytes = scaled_expert_bytes,
            .cold_penalty_bytes = 0.0,
            .beta_gbs_by_m = beta,
            .num_layers = sampled,
            .num_experts = target.config.num_experts,
        };
    }

    // Dense model: U_l(T) is empty; reduces cleanly to a node/latency budget.
    const static_bytes: u64 = target.total_layers_static_bytes + target.global_head_bytes;
    const t_fixed_us: f32 = @floatCast(@as(f64, @floatFromInt(static_bytes)) / (80.0 * 1000.0) + 800.0);
    const c_tok_us: f32 = @max(120.0, t_fixed_us * 0.025);
    return tree_mod.PruneCostModel.dense(t_fixed_us, c_tok_us);
}

pub const SpecEngine = struct {
    target: *qwen3.Qwen3Model,
    draft_head: ?*draft_mod.DraftHead,
    config: SpecConfig,
    cost_model: tree_mod.PruneCostModel,

    sam_arena: arena_mod.WiredArena,
    sam: sam_mod.SuffixAutomaton,

    // Preallocated hot-path scratch buffers (zero dynamic heap allocation on `stepRound`).
    draft_tree: tree_mod.DraftTree(MAX_TREE_NODES),
    eval_tree: tree_mod.DraftTree(MAX_TREE_NODES),
    tree_slot_of_node: [MAX_TREE_NODES]u16,
    slot_expanded: [MAX_TREE_NODES]bool,
    slot_ancestor_masks: [MAX_TREE_NODES]u64,
    node_experts: [MAX_TREE_NODES * MAX_PRED_LAYERS]tree_mod.ExpertBitset,
    verify_tokens: [MAX_TREE_NODES]u32,
    verify_positions: [MAX_TREE_NODES]u32,
    verify_masks: [MAX_TREE_NODES]u64,
    accepted_nodes: [MAX_TREE_NODES]u16,
    batch_tokens: [MAX_TREE_NODES]u32,
    batch_parent_rows: [MAX_TREE_NODES]u32,
    batch_positions: [MAX_TREE_NODES]u32,
    batch_nodes: [MAX_TREE_NODES]u16,
    batch_cands: [MAX_TREE_NODES * draft_mod.MAX_DRAFT_TOP_K]moe.TopKCandidate,
    l1_buf: [MAX_TREE_NODES]u32,
    shifted_tokens: [MAX_TREE_NODES]u32,
    p_res: []f32,

    next_pos: u32,
    last_token: u32,
    has_prefilled: bool,
    stats: SpecStats,

    pub fn init(
        allocator: std.mem.Allocator,
        target: *qwen3.Qwen3Model,
        draft_head: ?*draft_mod.DraftHead,
        config: SpecConfig,
    ) InitError!SpecEngine {
        if (target.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (config.max_tree_nodes < 2 or config.max_tree_nodes > MAX_TREE_NODES) return error.InvalidConfig;
        if (config.max_tree_nodes > target.max_tree_nodes) return error.InvalidConfig;
        if (config.max_draft_depth == 0 or config.max_draft_depth >= MAX_TREE_NODES) return error.InvalidConfig;
        if (config.branch_width == 0 or config.branch_width > MAX_TREE_NODES) return error.InvalidConfig;
        if (config.top_k_per_node == 0 or config.top_k_per_node > draft_mod.MAX_DRAFT_TOP_K) return error.InvalidConfig;
        if (config.l1_max_draft >= MAX_TREE_NODES) return error.InvalidConfig;
        if (!std.math.isFinite(config.temperature) or config.temperature < 0.0) return error.InvalidConfig;

        if (config.mode != .l1_only) {
            const dh = draft_head orelse return error.DraftHeadRequired;
            if (dh.config.format != config.mode) return error.DraftModeMismatch;
            if (!target.enable_feature_taps) return error.FeatureTapsRequired;
        }

        const sam_cap: u32 = if (config.sam_capacity > 0)
            config.sam_capacity
        else
            target.max_seq_len + MAX_TREE_NODES;
        const sam_bytes = try sam_mod.SuffixAutomaton.requiredBytes(sam_cap);
        const arena_bytes = try page.alignUp(sam_bytes);
        const existing_wired: u64 = target.total_wired_bytes + if (draft_head) |dh| dh.totalWiredBytes() else 0;
        if (existing_wired + arena_bytes > target.wired_budget_bytes) {
            return error.BudgetExceeded;
        }
        const rem_limit = target.wired_budget_bytes - existing_wired;
        const sam_budget = budget_mod.Budget.compute(.{
            .memory_bytes = target.wired_budget_bytes + 4 * page.APPLE_PAGE_SIZE,
            .user_wire_limit = rem_limit,
            .global_user_wire_limit = rem_limit,
            .gpu_working_set_bytes = rem_limit,
            .configured_cap = rem_limit,
            .headroom_bytes = 0,
        });
        var sam_arena = try arena_mod.WiredArena.init(arena_bytes, sam_budget);
        errdefer sam_arena.deinit();

        const sam_slice = try sam_arena.allocAligned(sam_bytes, sam_mod.SuffixAutomaton.backing_alignment);
        const sam = try sam_mod.SuffixAutomaton.initFromBuffer(sam_slice, sam_cap);

        const p_res = try allocator.alloc(f32, target.config.vocab_size);
        errdefer allocator.free(p_res);

        const cost = config.cost_model orelse defaultCostModelForTarget(target);

        return .{
            .target = target,
            .draft_head = draft_head,
            .config = config,
            .cost_model = cost,
            .sam_arena = sam_arena,
            .sam = sam,
            .draft_tree = tree_mod.DraftTree(MAX_TREE_NODES).init(0),
            .eval_tree = tree_mod.DraftTree(MAX_TREE_NODES).init(0),
            .tree_slot_of_node = @splat(tree_mod.NO_NODE),
            .slot_expanded = @splat(false),
            .slot_ancestor_masks = @splat(0),
            .node_experts = @splat(tree_mod.ExpertBitset.empty()),
            .verify_tokens = @splat(0),
            .verify_positions = @splat(0),
            .verify_masks = @splat(0),
            .accepted_nodes = @splat(0),
            .batch_tokens = @splat(0),
            .batch_parent_rows = @splat(0),
            .batch_positions = @splat(0),
            .batch_nodes = @splat(0),
            .batch_cands = undefined,
            .l1_buf = @splat(0),
            .shifted_tokens = @splat(0),
            .p_res = p_res,
            .next_pos = 0,
            .last_token = 0,
            .has_prefilled = false,
            .stats = .{},
        };
    }

    pub fn deinit(self: *SpecEngine, allocator: std.mem.Allocator) void {
        allocator.free(self.p_res);
        self.sam_arena.deinit();
    }

    pub fn reset(self: *SpecEngine) void {
        self.target.seq_len = 0;
        if (self.draft_head) |dh| dh.reset();
        self.sam.reset();
        self.next_pos = 0;
        self.last_token = 0;
        self.has_prefilled = false;
        self.stats = .{};
    }

    pub fn totalWiredBytes(self: *const SpecEngine) u64 {
        const draft_bytes: u64 = if (self.draft_head) |dh| dh.totalWiredBytes() else 0;
        return self.target.total_wired_bytes + draft_bytes + self.sam_arena.bytes().len;
    }

    pub fn effectivePlacement(self: *const SpecEngine) draft_mod.Placement {
        if (self.draft_head) |dh| return dh.effective_placement;
        return .gpu;
    }

    fn effectiveMaxSeqLen(self: *const SpecEngine) u32 {
        if (self.draft_head) |dh| return @min(self.target.max_seq_len, dh.max_seq_len);
        return self.target.max_seq_len;
    }

    fn targetFeatureBinding(self: *const SpecEngine, dh: *const draft_mod.DraftHead) metal.BufferBinding {
        return switch (dh.config.format) {
            .eagle3 => self.target.accepted_aux_bind,
            .eagle2, .mtp => self.target.accepted_top_bind,
            .l1_only => unreachable,
        };
    }

    fn sampleFirstToken(self: *SpecEngine, row_idx: usize, greedy_tok: u32, rng: ?std.Random) RunError!u32 {
        if (self.config.temperature == 0.0) return greedy_tok;
        const r = rng orelse return error.MissingRng;
        const v: usize = self.target.config.vocab_size;
        const logits_row = self.target.logits_host[row_idx * v .. (row_idx + 1) * v];
        try accept_mod.softmaxRow(f16, logits_row, self.config.temperature, self.p_res);
        return accept_mod.sampleCategorical(self.p_res, r);
    }

    /// Prefills `prompt_tokens` using chunked causal verification (`prefillChunk`),
    /// populates the L1 Suffix Automaton, and synchronizes the L2 draft head's
    /// KV cache + shifted `(token_{p+1}, target_feature_p)` state with zero CPU
    /// feature staging copies. Returns the first generated token `t_0`.
    pub fn prefill(self: *SpecEngine, prompt_tokens: []const u32, rng: ?std.Random) RunError!u32 {
        if (prompt_tokens.len == 0) return error.EmptyPrompt;
        const max_seq = self.effectiveMaxSeqLen();
        if (prompt_tokens.len > max_seq) return error.PositionOutOfRange;

        for (prompt_tokens) |tid| {
            if (tid >= self.target.config.vocab_size) return error.TokenOutOfRange;
        }

        self.reset();

        const max_chunk: usize = if (self.draft_head) |dh|
            @min(@as(usize, self.target.max_tree_nodes), @as(usize, dh.max_tree_slots))
        else
            @as(usize, self.target.max_tree_nodes);

        var start: usize = 0;
        var first_tok: u32 = 0;
        while (start < prompt_tokens.len) {
            const take: usize = @min(max_chunk, prompt_tokens.len - start);
            const is_last = (start + take == prompt_tokens.len);

            const greedy_tok = try self.target.prefillChunk(
                prompt_tokens[start .. start + take],
                @intCast(start),
                is_last,
            );
            if (is_last) {
                first_tok = try self.sampleFirstToken(take - 1, greedy_tok, rng);
            }

            // Feed prompt tokens into L1 Suffix Automaton.
            for (prompt_tokens[start .. start + take]) |tid| {
                if (self.sam.remainingCapacity() > 0) {
                    self.sam.append(tid) catch |err| switch (err) {
                        error.CapacityExceeded => unreachable,
                    };
                }
            }

            // Synchronize L2 draft head with shifted `(token_{p+1}, target_feature_p)`.
            if (self.draft_head) |dh| {
                for (0..take) |j| {
                    const idx = start + j;
                    self.shifted_tokens[j] = if (idx + 1 < prompt_tokens.len)
                        prompt_tokens[idx + 1]
                    else
                        first_tok;
                }
                const top_k: u32 = if (is_last) self.config.top_k_per_node else 0;
                try dh.syncAccepted(
                    self.target,
                    self.shifted_tokens[0..take],
                    self.targetFeatureBinding(dh),
                    @intCast(start),
                    top_k,
                );
            }

            start += take;
        }

        // Append the emitted first token `t_0` to L1 SAM so round 1 queries the suffix ending at `t_0`.
        if (self.sam.remainingCapacity() > 0) {
            self.sam.append(first_tok) catch |err| switch (err) {
                error.CapacityExceeded => unreachable,
            };
        }

        self.next_pos = @intCast(prompt_tokens.len);
        self.last_token = first_tok;
        self.has_prefilled = true;
        self.stats.prefill_tokens = prompt_tokens.len;
        return first_tok;
    }

    /// Executes one full speculative decode round with zero dynamic heap allocation:
    /// 1) L1 Suffix Automaton query (`D1`, `D6`),
    /// 2) L2 dynamic EAGLE-2/3/MTP expansion with per-level D7 early stopping (`D2`, `D3`, `D5`),
    /// 3) D7 expert-byte-budget submodular pruning (`D7`),
    /// 4) L3 GPU tree-attention verification (`G5`),
    /// 5) Lossless greedy (`T = 0`) or stochastic (`T > 0`) acceptance (`D9`),
    /// 6) Target KV commit + L1 SAM update + L2 draft KV/feature sync.
    ///
    /// Writes `1..=min(out_tokens.len, max_tree_nodes)` newly generated tokens into
    /// `out_tokens[0..summary.emitted_count]`.
    pub fn stepRound(
        self: *SpecEngine,
        rng: ?std.Random,
        out_tokens: []u32,
    ) RunError!RoundSummary {
        if (!self.has_prefilled) return error.NotPrefilled;
        if (out_tokens.len == 0) return error.OutputBufferTooSmall;
        if (self.config.temperature > 0.0 and rng == null) return error.MissingRng;

        const max_seq = self.effectiveMaxSeqLen();
        if (self.next_pos >= max_seq) return error.SequenceFull;

        const base_pos = self.next_pos;
        const avail_seq: u32 = max_seq - base_pos;
        const max_emit: usize = @min(out_tokens.len, @as(usize, avail_seq));
        const max_allowed_depth: u16 = @intCast(@min(
            @as(usize, MAX_TREE_NODES - 1),
            max_emit - 1,
        ));
        const hard_cap: u16 = @min(
            self.config.max_tree_nodes,
            @as(u16, @intCast(@min(max_emit, @as(usize, self.target.max_tree_nodes)))),
        );
        const candidate_cap: u16 = if (self.config.enable_pruning)
            @min(@as(u16, @intCast(self.target.max_tree_nodes)), MAX_TREE_NODES)
        else
            hard_cap;

        const t_draft_start = std.Io.Timestamp.now(std.Options.debug_io, .awake);

        // Reset draft tree to root node `0` = `last_token`.
        self.draft_tree.reset(self.last_token);
        @memset(&self.tree_slot_of_node, tree_mod.NO_NODE);
        @memset(&self.slot_expanded, false);
        @memset(&self.slot_ancestor_masks, 0);
        self.tree_slot_of_node[0] = 0; // root hidden lives in draft `hidden_pool[0]`
        self.slot_expanded[0] = true;

        // 1. L1 Suffix Automaton query (`D1`, `D6`, ADR-0005).
        const l1_cap: usize = @min(
            @as(usize, self.config.l1_max_draft),
            @min(@as(usize, max_allowed_depth), @as(usize, candidate_cap - 1)),
        );
        const l1_q = if (l1_cap > 0)
            self.sam.queryActive(l1_cap, self.l1_buf[0..l1_cap])
        else
            sam_mod.QueryResult{ .match_len = 0, .draft_len = 0, .is_seed = false };

        const l1_res = self.draft_tree.insertL1Continuation(
            l1_q.match_len,
            self.l1_buf[0..l1_q.draft_len],
            self.config.l1_seed_step_prob,
            self.config.l1_branch_step_prob,
        );
        if (l1_res.is_seed) {
            self.stats.l1_seed_rounds += 1;
        } else if (l1_res.chain_len > 0) {
            self.stats.l1_branch_rounds += 1;
        }

        // 2. L2 Draft Head dynamic tree expansion (`D2`, `D3`, `D5`, `D6`, `D7`).
        if (self.draft_head) |dh| {
            if (max_allowed_depth >= 1 and candidate_cap > 1) {
                try self.expandWithDraftHead(
                    dh,
                    base_pos,
                    max_allowed_depth,
                    candidate_cap,
                    hard_cap,
                    l1_res,
                    t_draft_start,
                );
            }
        }

        const t_draft_end = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const draft_ns: u64 = @intCast(@max(@as(i96, 0), t_draft_start.durationTo(t_draft_end).toNanoseconds()));
        const drafted_nodes: u16 = self.draft_tree.count;

        // 3. Final D7 expert-byte-budget submodular pruning (`ADR-0009`).
        const t_prune_start = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        if (self.config.enable_pruning and self.draft_tree.count > 1) {
            const draft_us: f32 = @floatCast(@as(f64, @floatFromInt(draft_ns)) / 1000.0);
            _ = try self.pruneTreeInPlace(&self.draft_tree, draft_us, hard_cap);
        }
        const t_prune_end = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const prune_ns: u64 = @intCast(@max(@as(i96, 0), t_prune_start.durationTo(t_prune_end).toNanoseconds()));

        // 4. L3 GPU Tree Verification (`G5`).
        const t_verify_start = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const verified_nodes: u16 = self.draft_tree.count;
        self.draft_tree.fillTokenIds(self.verify_tokens[0..verified_nodes]);
        self.draft_tree.fillPositions(base_pos, self.verify_positions[0..verified_nodes]);
        self.draft_tree.fillAncestorMasks(self.verify_masks[0..verified_nodes]);

        const vr = try self.target.verifyTree(
            self.verify_tokens[0..verified_nodes],
            self.verify_positions[0..verified_nodes],
            self.verify_masks[0..verified_nodes],
            base_pos,
            true,
        );
        const t_verify_end = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const verify_ns: u64 = @intCast(@max(@as(i96, 0), t_verify_start.durationTo(t_verify_end).toNanoseconds()));

        // 5. Lossless Verification (`D9`): greedy (`T = 0`) or stochastic (`T > 0`).
        const acc = if (self.config.temperature == 0.0)
            try accept_mod.verifyGreedy(
                MAX_TREE_NODES,
                &self.draft_tree,
                vr.argmax_tokens,
                self.accepted_nodes[0..],
                out_tokens,
            )
        else
            try accept_mod.verifyStochastic(
                MAX_TREE_NODES,
                f16,
                &self.draft_tree,
                vr.logits_f16,
                self.target.config.vocab_size,
                null,
                self.config.temperature,
                rng.?,
                self.p_res,
                self.accepted_nodes[0..],
                out_tokens,
            );

        // 6. Commit accepted target KV + gather target features + advance L1 SAM + sync L2 DraftHead.
        const t_commit_start = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const emitted = out_tokens[0..acc.emitted_token_count];

        try self.target.commitTreeKv(base_pos, self.accepted_nodes[0..acc.accepted_node_count]);

        for (emitted) |tok| {
            if (self.sam.remainingCapacity() > 0) {
                self.sam.append(tok) catch |err| switch (err) {
                    error.CapacityExceeded => unreachable,
                };
            }
        }

        if (self.draft_head) |dh| {
            const top_k: u32 = if (base_pos + acc.emitted_token_count < max_seq)
                self.config.top_k_per_node
            else
                0;
            try dh.syncAccepted(
                self.target,
                emitted,
                self.targetFeatureBinding(dh),
                base_pos,
                top_k,
            );
        }

        const t_commit_end = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const commit_sync_ns: u64 = @intCast(@max(@as(i96, 0), t_commit_start.durationTo(t_commit_end).toNanoseconds()));

        // 7. Advance sequence state and accumulate round telemetry.
        self.next_pos = base_pos + acc.emitted_token_count;
        self.last_token = acc.bonus_token;

        self.stats.rounds += 1;
        self.stats.emitted_tokens += acc.emitted_token_count;
        self.stats.draft_accepted += acc.draft_accepted;
        self.stats.drafted_nodes += drafted_nodes;
        self.stats.verified_nodes += verified_nodes;
        self.stats.total_bytes_moved += vr.actual_bytes_moved;
        if (self.target.config.arch == .qwen3_moe) {
            self.stats.total_unique_experts += vr.total_unique_experts;
            self.stats.total_moe_layer_passes += self.target.config.num_hidden_layers;
        }
        const hist_idx = @min(@as(usize, acc.emitted_token_count), MAX_TREE_NODES);
        self.stats.accept_hist[hist_idx] += 1;
        self.stats.draft_ns += draft_ns;
        self.stats.prune_ns += prune_ns;
        self.stats.verify_ns += verify_ns;
        self.stats.commit_sync_ns += commit_sync_ns;

        return .{
            .emitted_count = acc.emitted_token_count,
            .draft_accepted = acc.draft_accepted,
            .drafted_nodes = drafted_nodes,
            .verified_nodes = verified_nodes,
            .l1_match_len = l1_q.match_len,
            .l1_is_seed = l1_res.is_seed,
            .avg_unique_experts_per_layer = vr.avg_unique_experts_per_layer,
            .actual_bytes_moved = vr.actual_bytes_moved,
        };
    }

    fn expandWithDraftHead(
        self: *SpecEngine,
        dh: *draft_mod.DraftHead,
        base_pos: u32,
        max_allowed_depth: u16,
        candidate_cap: u16,
        hard_cap: u16,
        l1_res: tree_mod.InsertL1Result,
        t_draft_start: std.Io.Timestamp,
    ) RunError!void {
        const eff_top_k: u32 = @min(self.config.top_k_per_node, dh.config.draft_vocab_size);
        const root_cands = dh.rootCandidates();
        const root_take = @min(root_cands.len, @as(usize, eff_top_k));

        // Attach cached depth-1 L2 candidates directly onto root `0` (zero extra GPU work;
        // deduplicates automatically with L1 nodes in place per ADR-0005).
        for (root_cands[0..root_take]) |cand| {
            if (self.draft_tree.count >= candidate_cap and
                self.draft_tree.findChild(0, cand.token_id) == null)
            {
                break;
            }
            _ = self.draft_tree.insertChild(0, cand.token_id, cand.prob, .l2);
        }

        const target_depth: u16 = @min(self.config.max_draft_depth, max_allowed_depth);
        if (target_depth <= 1) return;

        // If L1 already delivered a long high-confidence seed chain (`chain_len >= target_depth`),
        // keep the L1 seed + root L2 backups without spending serial `layers.0` steps on every seed token.
        if (l1_res.is_seed and l1_res.chain_len >= target_depth) return;

        var used_slots: u32 = 0;

        // ADR-0005 (D6): For a short L1 seed (`1 <= chain_len < target_depth`), step the draft head
        // through the seed tokens (skipping `lm_head` on intermediate tokens) so L2 extends the seed tail.
        if (l1_res.is_seed and l1_res.chain_len > 0 and l1_res.seed_tail_node != 0) {
            var seed_path: [MAX_TREE_NODES]u16 = undefined;
            var path_len: usize = 0;
            var cur: u16 = l1_res.seed_tail_node;
            while (cur != 0 and cur != tree_mod.NO_NODE) : (cur = self.draft_tree.nodes[cur].parent) {
                seed_path[path_len] = cur;
                path_len += 1;
            }
            // Reverse `seed_path[0..path_len]` so index 0 is depth 1, index `path_len - 1` is `seed_tail_node`.
            var lo: usize = 0;
            var hi: usize = path_len;
            while (lo + 1 < hi) {
                hi -= 1;
                const tmp = seed_path[lo];
                seed_path[lo] = seed_path[hi];
                seed_path[hi] = tmp;
                lo += 1;
            }

            if (used_slots + path_len <= dh.max_tree_slots and path_len > 0) {
                for (0..path_len) |idx| {
                    const node_idx = seed_path[idx];
                    const node = self.draft_tree.nodes[node_idx];
                    const parent_row = self.tree_slot_of_node[node.parent];
                    const slot_idx = used_slots;
                    const parent_mask: u64 = if (parent_row == 0) 0 else self.slot_ancestor_masks[parent_row - 1];
                    self.slot_ancestor_masks[slot_idx] = parent_mask | (@as(u64, 1) << @as(u6, @intCast(slot_idx)));

                    self.batch_tokens[0] = node.token_id;
                    self.batch_parent_rows[0] = parent_row;
                    self.batch_positions[0] = base_pos + node.depth - 1;

                    const is_tail = (idx + 1 == path_len);
                    const step_k: u32 = if (is_tail and node.depth < target_depth) eff_top_k else 0;
                    try dh.stepTreeBatch(
                        self.target,
                        self.batch_tokens[0..1],
                        self.batch_parent_rows[0..1],
                        self.batch_positions[0..1],
                        self.slot_ancestor_masks[0 .. slot_idx + 1],
                        slot_idx,
                        step_k,
                        if (step_k > 0) self.batch_cands[0..step_k] else null,
                    );

                    self.tree_slot_of_node[node_idx] = @intCast(1 + slot_idx);
                    self.slot_expanded[node_idx] = is_tail;
                    used_slots += 1;

                    if (step_k > 0) {
                        for (self.batch_cands[0..step_k]) |cand| {
                            if (self.draft_tree.count >= candidate_cap and
                                self.draft_tree.findChild(node_idx, cand.token_id) == null)
                            {
                                break;
                            }
                            _ = self.draft_tree.insertChild(node_idx, cand.token_id, cand.prob, .l2);
                        }
                    }
                }
            }
        }

        // Evaluate baseline depth-1 score for D7 early stopping (ADR-0009).
        var prev_best_score: f64 = 0.0;
        if (self.config.enable_pruning and self.draft_tree.count > 1) {
            prev_best_score = try self.evaluateCurrentDraftScore(t_draft_start, hard_cap);
        }

        // Dynamic EAGLE-2 frontier expansion (`depth = 2 .. target_depth`).
        var level: u16 = 1;
        while (level < target_depth) : (level += 1) {
            if (self.draft_tree.count >= candidate_cap) break;
            if (used_slots >= dh.max_tree_slots) break;

            const max_batch: usize = @min(
                @as(usize, self.config.branch_width),
                @as(usize, dh.max_tree_slots - used_slots),
            );
            const frontier_count = self.selectFrontier(target_depth, max_batch, level == 1);
            if (frontier_count == 0) break;

            const slot_start = used_slots;
            for (0..frontier_count) |i| {
                const v = self.batch_nodes[i];
                const node = self.draft_tree.nodes[v];
                const parent_row = self.tree_slot_of_node[node.parent];
                const slot_idx = slot_start + @as(u32, @intCast(i));
                const parent_mask: u64 = if (parent_row == 0) 0 else self.slot_ancestor_masks[parent_row - 1];
                self.slot_ancestor_masks[slot_idx] = parent_mask | (@as(u64, 1) << @as(u6, @intCast(slot_idx)));

                self.batch_tokens[i] = node.token_id;
                self.batch_parent_rows[i] = parent_row;
                self.batch_positions[i] = base_pos + node.depth - 1;
            }

            const total_slots = slot_start + @as(u32, @intCast(frontier_count));
            const cand_len = frontier_count * @as(usize, eff_top_k);
            try dh.stepTreeBatch(
                self.target,
                self.batch_tokens[0..frontier_count],
                self.batch_parent_rows[0..frontier_count],
                self.batch_positions[0..frontier_count],
                self.slot_ancestor_masks[0..total_slots],
                slot_start,
                eff_top_k,
                self.batch_cands[0..cand_len],
            );

            for (0..frontier_count) |i| {
                const v = self.batch_nodes[i];
                const slot_idx = slot_start + @as(u32, @intCast(i));
                self.tree_slot_of_node[v] = @intCast(1 + slot_idx);
                self.slot_expanded[v] = true;

                const row_cands = self.batch_cands[i * eff_top_k .. (i + 1) * eff_top_k];
                for (row_cands) |cand| {
                    if (self.draft_tree.count >= candidate_cap and
                        self.draft_tree.findChild(v, cand.token_id) == null)
                    {
                        break;
                    }
                    _ = self.draft_tree.insertChild(v, cand.token_id, cand.prob, .l2);
                }
            }
            used_slots = total_slots;

            // ADR-0009 (D7): Run submodular pruning score check after every draft level;
            // stop expanding early once end-to-end `V(T) / (t_draft + t_verify)` stops improving.
            if (self.config.enable_pruning and level + 1 < target_depth) {
                const new_score = try self.evaluateCurrentDraftScore(t_draft_start, hard_cap);
                if (new_score <= prev_best_score * 1.005) {
                    break;
                }
                prev_best_score = new_score;
            }
        }
    }

    /// Selects up to `max_batch` unexpanded candidate nodes with highest `cum_prob`
    /// whose parent already holds a valid draft hidden state.
    fn selectFrontier(self: *SpecEngine, target_depth: u16, max_batch: usize, allow_fallback: bool) usize {
        var picked: usize = 0;
        var v: u16 = 1;
        while (v < self.draft_tree.count) : (v += 1) {
            if (self.slot_expanded[v]) continue;
            const node = self.draft_tree.nodes[v];
            if (node.depth >= target_depth) continue;
            if (self.tree_slot_of_node[node.parent] == tree_mod.NO_NODE) continue;
            if (node.cum_prob < self.config.min_frontier_prob) continue;

            // Insertion sort into `batch_nodes[0..picked]` by descending `cum_prob`.
            if (picked < max_batch) {
                self.batch_nodes[picked] = v;
                picked += 1;
            } else if (node.cum_prob > self.draft_tree.nodes[self.batch_nodes[picked - 1]].cum_prob) {
                self.batch_nodes[picked - 1] = v;
            } else {
                continue;
            }
            var j: usize = picked - 1;
            while (j > 0) : (j -= 1) {
                const cur_p = self.draft_tree.nodes[self.batch_nodes[j]].cum_prob;
                const prev_p = self.draft_tree.nodes[self.batch_nodes[j - 1]].cum_prob;
                if (cur_p > prev_p) {
                    const tmp = self.batch_nodes[j - 1];
                    self.batch_nodes[j - 1] = self.batch_nodes[j];
                    self.batch_nodes[j] = tmp;
                } else {
                    break;
                }
            }
        }

        // On level 1, if all depth-1 nodes fell below `min_frontier_prob`, expand the single best node
        // so greedy/uncertain distributions still check depth 2 before early-stopping.
        if (picked == 0 and allow_fallback and max_batch > 0) {
            var best_v: u16 = tree_mod.NO_NODE;
            var best_p: f32 = -1.0;
            var u: u16 = 1;
            while (u < self.draft_tree.count) : (u += 1) {
                if (self.slot_expanded[u]) continue;
                const node = self.draft_tree.nodes[u];
                if (node.depth >= target_depth) continue;
                if (self.tree_slot_of_node[node.parent] == tree_mod.NO_NODE) continue;
                if (node.cum_prob > best_p) {
                    best_p = node.cum_prob;
                    best_v = u;
                }
            }
            if (best_v != tree_mod.NO_NODE and best_p > 0.0) {
                self.batch_nodes[0] = best_v;
                picked = 1;
            }
        }
        return picked;
    }

    fn evaluateCurrentDraftScore(
        self: *SpecEngine,
        t_draft_start: std.Io.Timestamp,
        hard_cap: u16,
    ) RunError!f64 {
        const now = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const draft_us: f32 = @floatCast(
            @as(f64, @floatFromInt(@max(@as(i96, 0), t_draft_start.durationTo(now).toNanoseconds()))) / 1000.0,
        );
        self.eval_tree = self.draft_tree;
        const summary = try self.pruneTreeInPlace(&self.eval_tree, draft_us, hard_cap);
        const total_us: f64 = @as(f64, summary.predicted_time_us);
        if (!(total_us > 0.0)) return 0.0;
        return @as(f64, summary.expected_tokens) / total_us;
    }

    fn pruneTreeInPlace(
        self: *SpecEngine,
        t: *tree_mod.DraftTree(MAX_TREE_NODES),
        draft_us: f32,
        hard_cap: u16,
    ) RunError!tree_mod.DraftTree(MAX_TREE_NODES).PruneSummary {
        var cost = self.cost_model;
        cost.t_fixed_us += draft_us;

        if (cost.isMoe()) {
            const m = t.count;
            t.fillTokenIds(self.verify_tokens[0..m]);
            const sampled = try self.target.predictMoEExpertsForTokens(
                self.verify_tokens[0..m],
                self.node_experts[0..],
            );
            cost.num_layers = sampled;
            const bitset_slice = self.node_experts[0 .. @as(usize, m) * sampled];
            return try t.pruneToOptimalSubtree(cost, bitset_slice, null, hard_cap);
        }
        return try t.pruneToOptimalSubtree(cost, null, null, hard_cap);
    }
};

// ============================================================================
// Unit & End-to-End Integration Tests
// ============================================================================

fn verifyLosslessMatch(
    allocator: std.mem.Allocator,
    model_bytes: []const u8,
    model_cfg: qwen3.ModelConfig,
    draft_mode: draft_mod.DraftMode,
    placement: draft_mod.Placement,
    chunk_tokens: u32,
    prompt: []const u32,
    gen_steps: usize,
    enable_pruning: bool,
) !void {
    // 1. Reference sequential greedy decode via `Qwen3Model.stepDecode`.
    var ref_model = try qwen3.Qwen3Model.loadFromBytes(allocator, model_bytes, model_cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = chunk_tokens,
        .verify_crcs = true,
    });
    defer ref_model.deinit(allocator);

    var expected_tokens: [32]u32 = undefined;
    std.debug.assert(gen_steps <= expected_tokens.len);

    var pos: u32 = 0;
    for (prompt[0 .. prompt.len - 1]) |tok| {
        try ref_model.prefillToken(tok, pos);
        pos += 1;
    }
    var cur_tok = try ref_model.stepDecode(prompt[prompt.len - 1], pos);
    pos += 1;
    expected_tokens[0] = cur_tok;
    for (1..gen_steps) |step| {
        cur_tok = try ref_model.stepDecode(cur_tok, pos);
        pos += 1;
        expected_tokens[step] = cur_tok;
    }

    // 2. Speculative decode via `SpecEngine` (`temperature = 0.0`).
    var spec_model = try qwen3.Qwen3Model.loadFromBytes(allocator, model_bytes, model_cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = chunk_tokens,
        .max_tree_nodes = 16,
        .enable_feature_taps = (draft_mode != .l1_only),
        .verify_crcs = true,
    });
    defer spec_model.deinit(allocator);

    var draft_head_storage: draft_mod.DraftHead = undefined;
    var draft_head_ptr: ?*draft_mod.DraftHead = null;
    defer if (draft_head_ptr) |dh| dh.deinit(allocator, &spec_model);

    if (draft_mode != .l1_only) {
        const draft_vocab: u32 = if (draft_mode == .eagle3) @min(model_cfg.vocab_size, 32) else model_cfg.vocab_size;
        const dcfg: draft_mod.DraftConfig = .{
            .format = draft_mode,
            .vocab_size = model_cfg.vocab_size,
            .draft_vocab_size = draft_vocab,
            .hidden_size = model_cfg.hidden_size,
            .target_hidden_size = model_cfg.hidden_size,
            .num_hidden_layers = 1,
            .num_attention_heads = model_cfg.num_attention_heads,
            .num_key_value_heads = model_cfg.num_key_value_heads,
            .head_dim = model_cfg.head_dim,
            .intermediate_size = 128,
            .rms_norm_eps = model_cfg.rms_norm_eps,
            .rope_theta = 10_000.0,
            .max_position_embeddings = 64,
            .norm_before_residual = true,
            .aux_layer_ids = qwen3.defaultEagle3AuxLayerIds(model_cfg.num_hidden_layers),
        };
        const draft_bytes = try draft_mod.buildSyntheticDraftAmfImage(allocator, dcfg, @as(u64, 0xE3A0_1234) +% @backingInt(draft_mode));
        defer allocator.free(draft_bytes);

        draft_head_storage = try draft_mod.DraftHead.loadFromBytes(allocator, &spec_model, draft_bytes, dcfg, .{
            .placement = placement,
            .max_seq_len = 64,
            .chunk_tokens = chunk_tokens,
            .max_tree_slots = 16,
        });
        draft_head_ptr = &draft_head_storage;
    }

    var eng = try SpecEngine.init(allocator, &spec_model, draft_head_ptr, .{
        .mode = draft_mode,
        .placement = placement,
        .max_tree_nodes = 12,
        .max_draft_depth = 4,
        .branch_width = 2,
        .top_k_per_node = 3,
        .l1_max_draft = 8,
        .temperature = 0.0,
        .enable_pruning = enable_pruning,
    });
    defer eng.deinit(allocator);

    var actual_tokens: [32]u32 = undefined;
    actual_tokens[0] = try eng.prefill(prompt, null);
    var produced: usize = 1;
    while (produced < gen_steps) {
        const rem = gen_steps - produced;
        const rs = try eng.stepRound(null, actual_tokens[produced .. produced + rem]);
        try std.testing.expect(rs.emitted_count >= 1);
        try std.testing.expect(rs.emitted_count <= rem);
        produced += rs.emitted_count;
    }

    // Exact 100% bit-for-bit token sequence identity vs `Qwen3Model.stepDecode`!
    try std.testing.expectEqualSlices(u32, expected_tokens[0..gen_steps], actual_tokens[0..gen_steps]);
    try std.testing.expect(eng.stats.rounds >= 1);
    try std.testing.expectEqual(@as(u64, gen_steps - 1), eng.stats.emitted_tokens);
    try std.testing.expect(eng.stats.total_bytes_moved > 0);
}

test "SpecEngine: 100% bit-for-bit lossless match on MoE q3 across l1_only, eagle3, eagle2, mtp and chunk boundary" {
    const allocator = std.testing.allocator;
    const cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_moe,
        .vocab_size = 64,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .moe_intermediate_size = 128,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .max_position_embeddings = 64,
        .tie_word_embeddings = false,
        .norm_topk_prob = true,
    };
    const model_bytes = try qwen3.buildSyntheticAmfImageWithSpec(allocator, cfg, .{
        .linear_dtype = .q3_affine,
        .embed_dtype = .f16,
        .norm_dtype = .f16,
        .router_dtype = .f16,
        .expert_dtype = .q3_affine,
    }, 0x5EC0_0001);
    defer allocator.free(model_bytes);

    // Repetitive prompt exercising L1 seed prefix (`match_len >= 4`), chunk boundary (`chunk_tokens = 8`),
    // and MoE D7 pruning vs unpruned paths.
    const prompt = [_]u32{ 3, 7, 11, 15, 19, 3, 7, 11, 15 };

    try verifyLosslessMatch(allocator, model_bytes, cfg, .l1_only, .gpu, 8, &prompt, 10, true);
    try verifyLosslessMatch(allocator, model_bytes, cfg, .eagle3, .gpu, 8, &prompt, 10, true);
    try verifyLosslessMatch(allocator, model_bytes, cfg, .eagle3, .gpu, 8, &prompt, 10, false);
    try verifyLosslessMatch(allocator, model_bytes, cfg, .eagle2, .gpu, 8, &prompt, 8, true);
    try verifyLosslessMatch(allocator, model_bytes, cfg, .mtp, .gpu, 8, &prompt, 8, true);
    try verifyLosslessMatch(allocator, model_bytes, cfg, .eagle3, .ane, 8, &prompt, 6, true);
}

test "SpecEngine: 100% bit-for-bit lossless match on Dense BF16 across chunk boundary + stochastic T=0.7" {
    const allocator = std.testing.allocator;
    const cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .vocab_size = 64,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .moe_intermediate_size = 0,
        .num_experts = 0,
        .num_experts_per_tok = 0,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .max_position_embeddings = 64,
        .tie_word_embeddings = true,
        .norm_topk_prob = false,
    };
    const model_bytes = try qwen3.buildSyntheticAmfImageWithSpec(allocator, cfg, .{
        .linear_dtype = .bf16,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
        .router_dtype = .bf16,
    }, 0x5EC0_0002);
    defer allocator.free(model_bytes);

    const prompt = [_]u32{ 5, 12, 19, 26, 33, 5, 12, 19, 26 };

    // T = 0 lossless greedy match across `chunk_tokens = 8` boundary.
    try verifyLosslessMatch(allocator, model_bytes, cfg, .l1_only, .gpu, 8, &prompt, 10, true);
    try verifyLosslessMatch(allocator, model_bytes, cfg, .eagle2, .gpu, 8, &prompt, 8, true);

    // T = 0.7 stochastic speculative decode: reproducible with seed, valid tokens, zero errors.
    var model = try qwen3.Qwen3Model.loadFromBytes(allocator, model_bytes, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 8,
        .max_tree_nodes = 12,
        .enable_feature_taps = false,
    });
    defer model.deinit(allocator);

    var eng = try SpecEngine.init(allocator, &model, null, .{
        .mode = .l1_only,
        .max_tree_nodes = 12,
        .temperature = 0.7,
        .enable_pruning = true,
    });
    defer eng.deinit(allocator);

    var prng_a = std.Random.DefaultPrng.init(42);
    var out_a: [10]u32 = undefined;
    out_a[0] = try eng.prefill(&prompt, prng_a.random());
    var produced_a: usize = 1;
    while (produced_a < out_a.len) {
        const rs = try eng.stepRound(prng_a.random(), out_a[produced_a..]);
        produced_a += rs.emitted_count;
    }

    // Re-running with the exact same seed yields bit-for-bit reproducible stochastic output.
    var prng_b = std.Random.DefaultPrng.init(42);
    var out_b: [10]u32 = undefined;
    out_b[0] = try eng.prefill(&prompt, prng_b.random());
    var produced_b: usize = 1;
    while (produced_b < out_b.len) {
        const rs = try eng.stepRound(prng_b.random(), out_b[produced_b..]);
        produced_b += rs.emitted_count;
    }

    for (out_a) |t| try std.testing.expect(t < cfg.vocab_size);
    try std.testing.expectEqualSlices(u32, &out_a, &out_b);
}
