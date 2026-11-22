//! End-to-end speculative decoding engine combining:
//! - L1 CPU Suffix Automaton (`D1`, `src/spec/sam.zig`) + C6 NVMe Direct-I/O
//!   N-Gram Table (`src/spec/ngram_nvme.zig`, ADR-0006),
//! - L2 EAGLE-3 / EAGLE-2 / MTP draft heads with benchmark-gated ANE/GPU placement
//!   (`D2`, `D3`, `D5`, `src/spec/draft.zig`, ADR-0008, ADR-0013),
//! - Standalone same-family SmallDraftModel (`D4`) with target-latency-gated
//!   PEARL asynchronous pipelined pre-drafting (`D8`, ADR-0008),
//! - Dual-source prefix trie merging (`D6`, `src/spec/tree.zig`, ADR-0005),
//! - Expert-byte-budget submodular tree pruning (`D7`, `src/spec/tree.zig`, ADR-0009, ADR-0017),
//! - L3 GPU tree-attention verification and KV commit (`G5`, `src/model/qwen3.zig`),
//! - Lossless greedy (`T = 0`) and stochastic (`T > 0`) verification (`D9`, `src/spec/accept.zig`).
//!
//! All scratch state, suffix automaton storage, draft trees, PEARL worker state,
//! expert bitsets, and residual sampling buffers are preallocated once at startup.
//! The decode round hot path (`stepRound`) performs zero dynamic heap allocation.

const std = @import("std");
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const arena_mod = @import("../mem/arena.zig");
const metal = @import("../bridge/metal.zig");
const quant = @import("../quant/spec.zig");
const moe = @import("../gpu/moe.zig");
const attention = @import("../gpu/attention.zig");
const qwen3 = @import("../model/qwen3.zig");
const graph = @import("../model/graph.zig");
const sam_mod = @import("sam.zig");
const tree_mod = @import("tree.zig");
const accept_mod = @import("accept.zig");
const draft_mod = @import("draft.zig");
const ngram_nvme = @import("ngram_nvme.zig");
const rng_mod = @import("../runtime/rng.zig");
const qos = @import("../platform/qos.zig");
const vmstat = @import("../platform/vmstat.zig");
const sme = @import("../cpu/sme.zig");

pub const MAX_TREE_NODES: u16 = tree_mod.MAX_SUPPORTED_NODES;
const MAX_PRED_LAYERS: usize = qwen3.MAX_EXPERT_PRED_LAYERS;
/// Maximum speculated `(prefix_len, bonus_token)` root hypotheses pre-drafted
/// concurrently by D8 PEARL while `target.verifyTree(T_r)` runs on GPU.
const PEARL_MAX_BONUS_ROOTS: usize = 4;

/// Target-latency-gated PEARL async pipelined small-draft policy (`D8`, ADR-0008).
pub const PearlMode = enum {
    /// Enable automatically when `small_draft != null` and target verify latency
    /// `>= pearl_min_target_us` (default 40 ms, e.g. `Qwen3-14B-4bit`).
    auto,
    /// Disable PEARL pipelining unconditionally.
    off,
    /// Force PEARL async pipelining on whenever `small_draft != null` owns an
    /// independent Metal 4 compute queue.
    on,

    pub fn label(self: PearlMode) []const u8 {
        return switch (self) {
            .auto => "auto",
            .off => "off",
            .on => "on",
        };
    }

    pub fn parse(str: []const u8) error{InvalidPearlMode}!PearlMode {
        if (std.mem.eql(u8, str, "auto")) return .auto;
        if (std.mem.eql(u8, str, "off")) return .off;
        if (std.mem.eql(u8, str, "on")) return .on;
        return error.InvalidPearlMode;
    }
};

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
    /// D8 target-latency-gated PEARL async pipelined small-draft mode (ADR-0008).
    pearl_mode: PearlMode = .auto,
    /// Minimum target verification latency (us) enabling `.auto` PEARL (default 40 ms).
    pearl_min_target_us: f64 = 40_000.0,
    /// Optional C6 NVMe `F_NOCACHE` N-gram table queried when L1 SAM match `< 4`.
    ngram_table: ?*ngram_nvme.NvmeNgramTable = null,
    /// Optional C14 hot-swappable multi-pack `.ang` domain registry.
    ngram_registry: ?*ngram_nvme.DomainNgramRegistry = null,
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
    // C6 NVMe N-gram telemetry
    ngram_queries: u64 = 0,
    ngram_hits: u64 = 0,
    ngram_seed_hits: u64 = 0,
    ngram_cache_hits: u64 = 0,
    ngram_nvme_reads: u64 = 0,
    // D8 PEARL async pipelining telemetry
    pearl_enabled: bool = false,
    pearl_rounds: u64 = 0,
    pearl_hits: u64 = 0,
    pearl_partial_prefix_hits: u64 = 0,
    pearl_rollbacks: u64 = 0,
    pearl_saved_ns: u64 = 0,
    pearl_worker_ns: u64 = 0,
    // D12 SelfSpecRecurrent telemetry
    self_recurrent_draft_steps: u64 = 0,

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
    ngram_match_len: u32 = 0,
    pearl_hit: bool = false,
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
    budget_mod.PlanError || std.Thread.SpawnError || qos.Error;

pub const RunError = error{
    EmptyPrompt,
    NotPrefilled,
    SequenceFull,
    OutputBufferTooSmall,
    TokenOutOfRange,
    MissingRng,
} || qwen3.StepError || graph.GraphStepError || draft_mod.DraftStepError || tree_mod.PruneError ||
    accept_mod.VerifyError || ngram_nvme.RegistryQueryError;

/// Unified speculative target handle supporting both Tier 1 `Qwen3Model` and
/// Tier 2 composable `GraphModel` (`qwen4_exp`, native `LoopLM`, `Step-LoRA`).
pub const TargetModel = union(enum) {
    qwen3: *qwen3.Qwen3Model,
    graph: *graph.GraphModel,

    pub fn vocabSize(self: TargetModel) u32 {
        return switch (self) {
            .qwen3 => |m| m.config.vocab_size,
            .graph => |g| g.config.vocab_size,
        };
    }

    pub fn maxSeqLen(self: TargetModel) u32 {
        return switch (self) {
            .qwen3 => |m| m.max_seq_len,
            .graph => |g| g.max_seq_len,
        };
    }

    pub fn maxTreeNodes(self: TargetModel) u32 {
        return switch (self) {
            .qwen3 => |m| m.max_tree_nodes,
            .graph => |g| g.max_tree_nodes,
        };
    }

    pub fn totalWiredBytes(self: TargetModel) u64 {
        return switch (self) {
            .qwen3 => |m| m.total_wired_bytes,
            .graph => |g| g.total_wired_bytes,
        };
    }

    pub fn wiredBudgetBytes(self: TargetModel) u64 {
        return switch (self) {
            .qwen3 => |m| m.wired_budget_bytes,
            .graph => |g| g.wired_budget_bytes,
        };
    }

    pub fn enableFeatureTaps(self: TargetModel) bool {
        return switch (self) {
            .qwen3 => |m| m.enable_feature_taps,
            .graph => |g| g.enable_feature_taps,
        };
    }

    pub fn isMoe(self: TargetModel) bool {
        return switch (self) {
            .qwen3 => |m| m.config.arch == .qwen3_moe,
            .graph => |g| g.config.axes.ffn_kind == .moe_swiglu,
        };
    }

    pub fn hiddenSize(self: TargetModel) u32 {
        return switch (self) {
            .qwen3 => |m| m.config.hidden_size,
            .graph => |g| g.config.hidden_size,
        };
    }

    pub fn numHiddenLayers(self: TargetModel) u32 {
        return switch (self) {
            .qwen3 => |m| m.config.num_hidden_layers,
            .graph => |g| g.config.num_hidden_layers,
        };
    }

    pub fn logitsHost(self: TargetModel) []const f16 {
        return switch (self) {
            .qwen3 => |m| m.logits_host,
            .graph => |g| g.logits_host,
        };
    }

    pub fn lastLogitsRow(self: TargetModel) usize {
        return switch (self) {
            .qwen3 => |m| m.last_logits_row,
            .graph => |g| g.last_logits_row,
        };
    }

    pub fn prefillChunkCap(self: TargetModel) u32 {
        return switch (self) {
            .qwen3 => |m| m.prefillChunkCap(),
            .graph => |g| g.prefillChunkCap(),
        };
    }

    pub fn acceptedAuxBind(self: TargetModel) metal.BufferBinding {
        return switch (self) {
            .qwen3 => |m| m.accepted_aux_bind,
            .graph => |g| g.accepted_aux_bind,
        };
    }

    pub fn acceptedTopBind(self: TargetModel) metal.BufferBinding {
        return switch (self) {
            .qwen3 => |m| m.accepted_top_bind,
            .graph => |g| g.accepted_top_bind,
        };
    }

    pub fn resetState(self: TargetModel) void {
        switch (self) {
            .qwen3 => |m| m.seq_len = 0,
            .graph => |g| g.resetKvCache(),
        }
    }

    pub fn prefillChunk(self: TargetModel, tokens: []const u32, start_pos: u32, compute_logits: bool) RunError!u32 {
        return switch (self) {
            .qwen3 => |m| try m.prefillChunk(tokens, start_pos, compute_logits),
            .graph => |g| try g.prefillChunk(tokens, start_pos, compute_logits),
        };
    }

    pub fn prefillChunkWithOptions(
        self: TargetModel,
        tokens: []const u32,
        start_pos: u32,
        opts: qwen3.PrefillChunkOptions,
    ) RunError!u32 {
        return switch (self) {
            .qwen3 => |m| try m.prefillChunkWithOptions(tokens, start_pos, opts),
            .graph => |g| try g.prefillChunkWithOptions(tokens, start_pos, opts),
        };
    }

    pub fn verifyTree(
        self: TargetModel,
        tokens: []const u32,
        positions: []const u32,
        masks: []const u64,
        base_pos: u32,
    ) RunError!qwen3.VerifyTreeResult {
        return switch (self) {
            .qwen3 => |m| try m.verifyTree(tokens, positions, masks, base_pos, true),
            .graph => |g| try g.verifyTree(tokens, positions, masks, base_pos),
        };
    }

    pub fn commitTreeKv(self: TargetModel, base_pos: u32, accepted_nodes: []const u16) RunError!void {
        return switch (self) {
            .qwen3 => |m| try m.commitTreeKv(base_pos, accepted_nodes),
            .graph => |g| try g.commitTreeKv(base_pos, accepted_nodes),
        };
    }

    pub fn syncDraftHead(
        self: TargetModel,
        dh: *draft_mod.DraftHead,
        tokens: []const u32,
        feat_bind: metal.BufferBinding,
        base_pos: u32,
        top_k: u32,
    ) draft_mod.DraftStepError!void {
        return switch (self) {
            .qwen3 => |m| dh.syncAccepted(m, tokens, feat_bind, base_pos, top_k),
            .graph => |g| dh.syncAcceptedForGraph(g, tokens, feat_bind, base_pos, top_k),
        };
    }

    pub fn stepDraftHeadTreeBatch(
        self: TargetModel,
        dh: *draft_mod.DraftHead,
        tokens: []const u32,
        parent_rows: []const u32,
        positions: []const u32,
        masks: []const u64,
        slot_start: u32,
        top_k: u32,
        out_cands: ?[]moe.TopKCandidate,
    ) draft_mod.DraftStepError!void {
        return switch (self) {
            .qwen3 => |m| dh.stepTreeBatch(m, tokens, parent_rows, positions, masks, slot_start, top_k, out_cands),
            .graph => |g| dh.stepTreeBatchForGraph(g, tokens, parent_rows, positions, masks, slot_start, top_k, out_cands),
        };
    }
};

pub fn defaultCostModelForGraph(g: *const graph.GraphModel) tree_mod.PruneCostModel {
    const decode_bytes: u64 = g.bytesPerDecodeToken(256);
    const t_fixed_us: f32 = @floatCast(@as(f64, @floatFromInt(decode_bytes)) / (75.0 * 1000.0) + 800.0);
    const c_tok_us: f32 = @max(120.0, t_fixed_us * 0.03);
    return tree_mod.PruneCostModel.dense(t_fixed_us, c_tok_us);
}

/// Calibrated per-scheme effective MoE gather bandwidth `beta_s` (GB/s) for
/// `ADR-0017` heterogeneous GSQ+RCO expert byte-budget tree pruning (`D7`).
pub fn schemeEffectiveBwGbs(dt: quant.WeightDType) f32 {
    return switch (dt) {
        .q2_affine => 60.0,
        .q3_affine => 64.0,
        .q4_affine => 68.0,
        .f16, .bf16 => 78.0,
    };
}

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
        const cold_penalty: f32 = if (target.hasColdExperts()) scaled_expert_bytes * 10.0 else 0.0;
        return .{
            .t_fixed_us = t_fixed_us,
            .c_tok_us = 160.0,
            .expert_bytes = scaled_expert_bytes,
            .cold_penalty_bytes = cold_penalty,
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

// ---------------------------------------------------------------------------
// D8 Target-Latency-Gated PEARL Async Pipelined Small-Draft Worker (ADR-0008)
// ---------------------------------------------------------------------------

const PearlWorkerState = enum(u8) {
    idle = 0,
    running = 1,
    done = 2,
    shutdown = 3,
};

const PearlJob = struct {
    base_pos: u32 = 0,
    committed_before: u32 = 0,
    pending_sync_len: u16 = 0,
    pending_sync_tokens: [MAX_TREE_NODES + 1]u32 = @splat(0),
    lead_len: u16 = 0,
    lead_tokens: [MAX_TREE_NODES]u32 = @splat(0),
    off_gpu: bool = false,
    greedy: bool = true,
    max_draft_depth: u16 = 4,
    branch_width: u16 = 2,
    top_k: u32 = 3,
    min_frontier_prob: f32 = 0.12,
    candidate_cap: u16 = 16,
    max_seq_len: u32 = 0,
};

const PearlResult = struct {
    num_roots: u8 = 0,
    root_prefix_lens: [PEARL_MAX_BONUS_ROOTS]u16 = @splat(0),
    bonus_tokens: [PEARL_MAX_BONUS_ROOTS]u32 = @splat(0),
    trees: [PEARL_MAX_BONUS_ROOTS]tree_mod.DraftTree(MAX_TREE_NODES) = .{
        tree_mod.DraftTree(MAX_TREE_NODES).init(0),
        tree_mod.DraftTree(MAX_TREE_NODES).init(0),
        tree_mod.DraftTree(MAX_TREE_NODES).init(0),
        tree_mod.DraftTree(MAX_TREE_NODES).init(0),
    },
    committed_after_pending: u32 = 0,
    elapsed_ns: u64 = 0,
    err: ?qwen3.StepError = null,
};

/// Heap-pinned persistent `USER_INTERACTIVE` PEARL worker spawned ONCE at
/// `SpecEngine.initWithSmallDraft` (zero per-round thread creation and zero
/// hot-path heap allocation). Overlaps `small_draft` round-`r+1` pre-drafting
/// (off-GPU on ARMv9 SME2 `ldnt1b` when `sme2_panel64`, or on `small_draft`'s
/// independent Metal 4 queue) concurrently while the main thread runs
/// `target.verifyTree(T_r)` on `target.compute`.
const PearlRuntime = struct {
    small_draft: *draft_mod.SmallDraftModel,
    mutex: std.c.pthread_mutex_t,
    work_cond: std.c.pthread_cond_t,
    done_cond: std.c.pthread_cond_t,
    state: PearlWorkerState,
    ready: bool,
    init_err: ?qos.Error,
    stop_expanding: std.atomic.Value(bool) align(128),
    thread: std.Thread,

    job: PearlJob,
    result: PearlResult,

    // Preallocated worker-local scratch (never touched by main thread while `.running`).
    slot_ancestor_masks: [MAX_TREE_NODES]u64,
    node_slots: [PEARL_MAX_BONUS_ROOTS][MAX_TREE_NODES]u16,
    node_expanded: [PEARL_MAX_BONUS_ROOTS][MAX_TREE_NODES]bool,
    batch_tokens: [MAX_TREE_NODES]u32,
    batch_positions: [MAX_TREE_NODES]u32,
    batch_roots: [MAX_TREE_NODES]u8,
    batch_nodes: [MAX_TREE_NODES]u16,
    lead_cands: [MAX_TREE_NODES * draft_mod.MAX_DRAFT_TOP_K]moe.TopKCandidate,
    batch_cands: [MAX_TREE_NODES * draft_mod.MAX_DRAFT_TOP_K]moe.TopKCandidate,

    fn init(allocator: std.mem.Allocator, sd: *draft_mod.SmallDraftModel) InitError!*PearlRuntime {
        const rt = try allocator.create(PearlRuntime);
        errdefer allocator.destroy(rt);

        rt.* = .{
            .small_draft = sd,
            .mutex = std.c.PTHREAD_MUTEX_INITIALIZER,
            .work_cond = std.c.PTHREAD_COND_INITIALIZER,
            .done_cond = std.c.PTHREAD_COND_INITIALIZER,
            .state = .idle,
            .ready = false,
            .init_err = null,
            .stop_expanding = .init(false),
            .thread = undefined,
            .job = .{},
            .result = .{},
            .slot_ancestor_masks = @splat(0),
            .node_slots = @splat(@splat(tree_mod.NO_NODE)),
            .node_expanded = @splat(@splat(false)),
            .batch_tokens = @splat(0),
            .batch_positions = @splat(0),
            .batch_roots = @splat(0),
            .batch_nodes = @splat(0),
            .lead_cands = undefined,
            .batch_cands = undefined,
        };

        rt.thread = try std.Thread.spawn(.{}, workerMain, .{rt});

        // Wait for startup handshake confirming P-core USER_INTERACTIVE QoS.
        _ = std.c.pthread_mutex_lock(&rt.mutex);
        while (!rt.ready) {
            _ = std.c.pthread_cond_wait(&rt.done_cond, &rt.mutex);
        }
        const init_err = rt.init_err;
        _ = std.c.pthread_mutex_unlock(&rt.mutex);

        if (init_err) |err| {
            rt.thread.join();
            _ = std.c.pthread_cond_destroy(&rt.done_cond);
            _ = std.c.pthread_cond_destroy(&rt.work_cond);
            _ = std.c.pthread_mutex_destroy(&rt.mutex);
            allocator.destroy(rt);
            return err;
        }
        return rt;
    }

    fn deinit(self: *PearlRuntime, allocator: std.mem.Allocator) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
        while (self.state == .running) {
            _ = std.c.pthread_cond_wait(&self.done_cond, &self.mutex);
        }
        self.state = .shutdown;
        _ = std.c.pthread_cond_broadcast(&self.work_cond);
        _ = std.c.pthread_mutex_unlock(&self.mutex);

        self.thread.join();
        _ = std.c.pthread_cond_destroy(&self.done_cond);
        _ = std.c.pthread_cond_destroy(&self.work_cond);
        _ = std.c.pthread_mutex_destroy(&self.mutex);
        allocator.destroy(self);
    }

    fn launch(self: *PearlRuntime, job: PearlJob) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
        std.debug.assert(self.state == .idle);
        self.stop_expanding.store(false, .release);
        self.job = job;
        self.result = .{};
        self.state = .running;
        _ = std.c.pthread_cond_signal(&self.work_cond);
        _ = std.c.pthread_mutex_unlock(&self.mutex);
    }

    /// Signals the worker not to start any additional expansion levels because
    /// `target.verifyTree(T_r)` has already completed, preventing critical-path stalls.
    fn requestEarlyStop(self: *PearlRuntime) void {
        self.stop_expanding.store(true, .release);
    }

    fn collectResult(self: *PearlRuntime) PearlResult {
        _ = std.c.pthread_mutex_lock(&self.mutex);
        while (self.state == .running) {
            _ = std.c.pthread_cond_wait(&self.done_cond, &self.mutex);
        }
        const res = self.result;
        if (self.state == .done) {
            self.state = .idle;
        }
        _ = std.c.pthread_mutex_unlock(&self.mutex);
        return res;
    }

    fn waitIdle(self: *PearlRuntime) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
        while (self.state == .running) {
            _ = std.c.pthread_cond_wait(&self.done_cond, &self.mutex);
        }
        if (self.state == .done) {
            self.state = .idle;
        }
        _ = std.c.pthread_mutex_unlock(&self.mutex);
    }

    fn workerMain(self: *PearlRuntime) void {
        var init_err: ?qos.Error = null;
        if (qos.steerPerformanceCore(.user_interactive)) {} else |err| {
            init_err = err;
        }

        _ = std.c.pthread_mutex_lock(&self.mutex);
        self.init_err = init_err;
        self.ready = true;
        _ = std.c.pthread_cond_signal(&self.done_cond);
        if (init_err != null) {
            _ = std.c.pthread_mutex_unlock(&self.mutex);
            return;
        }

        while (true) {
            while (self.state != .running and self.state != .shutdown) {
                _ = std.c.pthread_cond_wait(&self.work_cond, &self.mutex);
            }
            if (self.state == .shutdown) {
                _ = std.c.pthread_mutex_unlock(&self.mutex);
                return;
            }
            _ = std.c.pthread_mutex_unlock(&self.mutex);

            const t0 = vmstat.nowNs() catch 0;
            var res: PearlResult = .{};
            self.executeJob(&res) catch |err| {
                res.err = err;
            };
            const t1 = vmstat.nowNs() catch t0;
            res.elapsed_ns = t1 -| t0;

            _ = std.c.pthread_mutex_lock(&self.mutex);
            self.result = res;
            if (self.state == .running) {
                self.state = .done;
            }
            _ = std.c.pthread_cond_signal(&self.done_cond);
        }
    }

    fn selectFrontierForRoot(
        self: *PearlRuntime,
        tree: *const tree_mod.DraftTree(MAX_TREE_NODES),
        root_idx: usize,
        target_depth: u16,
        max_pick: usize,
        allow_fallback: bool,
        out_start: usize,
    ) usize {
        if (max_pick == 0) return 0;
        var picked: usize = 0;
        var v: u16 = 1;
        while (v < tree.count) : (v += 1) {
            if (self.node_expanded[root_idx][v]) continue;
            const node = tree.nodes[v];
            if (node.depth >= target_depth) continue;
            if (self.node_slots[root_idx][node.parent] == tree_mod.NO_NODE) continue;
            if (node.cum_prob < self.job.min_frontier_prob) continue;

            if (picked < max_pick) {
                self.batch_nodes[out_start + picked] = v;
                self.batch_roots[out_start + picked] = @intCast(root_idx);
                picked += 1;
            } else if (node.cum_prob > tree.nodes[self.batch_nodes[out_start + picked - 1]].cum_prob) {
                self.batch_nodes[out_start + picked - 1] = v;
                self.batch_roots[out_start + picked - 1] = @intCast(root_idx);
            } else {
                continue;
            }

            var j: usize = picked - 1;
            while (j > 0) : (j -= 1) {
                const cur_p = tree.nodes[self.batch_nodes[out_start + j]].cum_prob;
                const prev_p = tree.nodes[self.batch_nodes[out_start + j - 1]].cum_prob;
                if (cur_p > prev_p) {
                    const tmp = self.batch_nodes[out_start + j - 1];
                    self.batch_nodes[out_start + j - 1] = self.batch_nodes[out_start + j];
                    self.batch_nodes[out_start + j] = tmp;
                } else {
                    break;
                }
            }
        }

        if (picked == 0 and allow_fallback) {
            var best_v: u16 = tree_mod.NO_NODE;
            var best_p: f32 = -1.0;
            var u: u16 = 1;
            while (u < tree.count) : (u += 1) {
                if (self.node_expanded[root_idx][u]) continue;
                const node = tree.nodes[u];
                if (node.depth >= target_depth) continue;
                if (self.node_slots[root_idx][node.parent] == tree_mod.NO_NODE) continue;
                if (node.cum_prob > best_p) {
                    best_p = node.cum_prob;
                    best_v = u;
                }
            }
            if (best_v != tree_mod.NO_NODE and best_p > 0.0) {
                self.batch_nodes[out_start] = best_v;
                self.batch_roots[out_start] = @intCast(root_idx);
                picked = 1;
            }
        }
        return picked;
    }

    fn candsForPrefix(
        self: *const PearlRuntime,
        prefix_len: u16,
        got_k_lead: u32,
    ) []const moe.TopKCandidate {
        if (prefix_len == 0) return self.small_draft.rootCandidates();
        const idx: usize = prefix_len - 1;
        return self.lead_cands[idx * got_k_lead .. (idx + 1) * got_k_lead];
    }

    fn appendHypothesis(
        res: *PearlResult,
        max_roots: usize,
        committed_after_pending: u32,
        max_seq_len: u32,
        prefix_len: u16,
        tok: u32,
    ) void {
        if (res.num_roots >= max_roots) return;
        if (committed_after_pending + prefix_len + 1 >= max_seq_len) return;
        for (0..res.num_roots) |i| {
            if (res.root_prefix_lens[i] == prefix_len and res.bonus_tokens[i] == tok) return;
        }
        const r = res.num_roots;
        res.root_prefix_lens[r] = prefix_len;
        res.bonus_tokens[r] = tok;
        res.num_roots += 1;
    }

    fn executeJob(self: *PearlRuntime, res: *PearlResult) qwen3.StepError!void {
        const sd = self.small_draft;
        const job = &self.job;
        const eff_top_k: u32 = @min(job.top_k, sd.model.config.vocab_size);

        // 1. Commit already-accepted ground-truth tokens from a prior PEARL hit into
        //    `sd`'s permanent KV cache (`committed_after_pending == base_pos + 1`).
        //    When `lead_len > 0`, skip `lm_head` (`top_k = 0`) because Step 2 produces
        //    multi-prefix candidates for `k = 1..lead_len` in one batched pass.
        const max_slots: usize = sd.maxTreeSlots();
        const lead_len: u16 = @intCast(@min(@as(usize, job.lead_len), max_slots -| 1));
        if (job.pending_sync_len > 0) {
            try sd.syncAcceptedChunked(
                job.pending_sync_tokens[0..job.pending_sync_len],
                job.committed_before,
                if (lead_len > 0) 0 else eff_top_k,
            );
        }
        const committed_after_pending = sd.seqLen();
        res.committed_after_pending = committed_after_pending;

        // 2. Evaluate `lead_tokens[0..lead_len]` into temporary tree KV slots `0..lead_len`
        //    WITHOUT mutating permanent KV, yielding top-K bonus predictions at EVERY
        //    prefix length `k = 1..lead_len` in ONE single batched forward pass!
        @memset(&self.slot_ancestor_masks, 0);
        var got_k_lead: u32 = 0;
        if (lead_len > 0) {
            for (0..lead_len) |i| {
                self.batch_tokens[i] = job.lead_tokens[i];
                self.batch_positions[i] = committed_after_pending + @as(u32, @intCast(i));
                self.slot_ancestor_masks[i] = if (i == 63)
                    ~@as(u64, 0)
                else
                    (@as(u64, 1) << @as(u6, @intCast(i + 1))) - 1;
            }
            got_k_lead = try sd.stepTreeBatch(
                self.batch_tokens[0..lead_len],
                self.batch_positions[0..lead_len],
                self.slot_ancestor_masks[0..lead_len],
                0,
                eff_top_k,
                self.lead_cands[0 .. @as(usize, lead_len) * @as(usize, eff_top_k)],
            );
        }

        // 3. Select multi-prefix `(prefix_len, bonus_tok)` root hypotheses.
        //    Always complete Steps 1–4 (`res.num_roots > 0` with depth-1 children) before
        //    checking `stop_expanding` in Step 5 (`level >= 1`), so a fast target verify pass
        //    never races ahead of Step 2 and drops all PEARL root hits.
        //    Because batched `stepTreeBatch` reads draft weights ONCE across all
        //    speculated roots (`M = 1..4` costs virtually the same DRAM traffic as `M = 1`),
        //    speculating up to 4 `(prefix_len, bonus_tok)` hypotheses (full lead acceptance
        //    + top partial-prefix rejection fallbacks where `cand != lead_tokens[k]`)
        //    turns partial prefix matches into full PEARL hits at zero extra CB count!
        const full_cands = self.candsForPrefix(lead_len, got_k_lead);
        const rem_for_roots: usize = max_slots -| @as(usize, lead_len);
        const max_roots: usize = @min(PEARL_MAX_BONUS_ROOTS, rem_for_roots);
        if (max_roots == 0) return;

        // Hypothesis 0: full lead path accepted, greedy bonus #1.
        if (full_cands.len > 0) {
            appendHypothesis(res, max_roots, committed_after_pending, job.max_seq_len, lead_len, full_cands[0].token_id);
        }
        // Hypothesis 1 (multi-prefix): rejected at `lead_len - 1` -> #1 non-lead candidate.
        if (max_roots >= 2 and lead_len >= 2) {
            const c_prev = self.candsForPrefix(lead_len - 1, got_k_lead);
            for (c_prev) |c| {
                if (c.token_id != job.lead_tokens[lead_len - 1]) {
                    appendHypothesis(res, max_roots, committed_after_pending, job.max_seq_len, lead_len - 1, c.token_id);
                    break;
                }
            }
        }
        // Hypothesis 2: full lead path accepted, bonus #2.
        if (max_roots >= 2 and full_cands.len >= 2) {
            appendHypothesis(res, max_roots, committed_after_pending, job.max_seq_len, lead_len, full_cands[1].token_id);
        }
        // Hypothesis 3+ (multi-prefix): walk back earlier rejection prefixes `lead_len - 2 .. 1`.
        if (lead_len >= 3) {
            var back: u16 = 2;
            while (res.num_roots < max_roots and back < lead_len) : (back += 1) {
                const k_pref: u16 = lead_len - back;
                const c_k = self.candsForPrefix(k_pref, got_k_lead);
                for (c_k) |c| {
                    if (c.token_id != job.lead_tokens[k_pref]) {
                        appendHypothesis(res, max_roots, committed_after_pending, job.max_seq_len, k_pref, c.token_id);
                        break;
                    }
                }
            }
        }

        const num_roots: usize = res.num_roots;
        if (num_roots == 0) return;

        // 4. Step all speculated `(prefix_len, bonus_tok)` roots in ONE batched `stepTreeBatch`
        //    call at tree slots `lead_len .. lead_len + num_roots`.
        const root_slot_start: u32 = lead_len;
        for (0..num_roots) |r| {
            const k_r = res.root_prefix_lens[r];
            const btok = res.bonus_tokens[r];
            const slot_r: u32 = root_slot_start + @as(u32, @intCast(r));
            const prefix_mask: u64 = if (k_r == 0)
                0
            else if (k_r >= 64)
                ~@as(u64, 0)
            else
                (@as(u64, 1) << @as(u6, @intCast(k_r))) - 1;

            res.trees[r].reset(btok);
            @memset(&self.node_slots[r], tree_mod.NO_NODE);
            @memset(&self.node_expanded[r], false);
            self.node_slots[r][0] = @intCast(slot_r);
            self.node_expanded[r][0] = true;
            self.slot_ancestor_masks[slot_r] = prefix_mask | (@as(u64, 1) << @as(u6, @intCast(slot_r)));
            self.batch_tokens[r] = btok;
            self.batch_positions[r] = committed_after_pending + k_r;
        }

        var used_slots: u32 = root_slot_start + @as(u32, @intCast(num_roots));
        const got_k0 = try sd.stepTreeBatch(
            self.batch_tokens[0..num_roots],
            self.batch_positions[0..num_roots],
            self.slot_ancestor_masks[0..used_slots],
            root_slot_start,
            eff_top_k,
            self.batch_cands[0 .. num_roots * @as(usize, eff_top_k)],
        );
        for (0..num_roots) |r| {
            const row = self.batch_cands[r * got_k0 .. (r + 1) * got_k0];
            for (row) |cand| {
                if (res.trees[r].count >= job.candidate_cap and
                    res.trees[r].findChild(0, cand.token_id) == null)
                {
                    break;
                }
                _ = res.trees[r].insertChild(0, cand.token_id, cand.prob, .l2);
            }
        }

        // 5. Expand depths `2 .. target_depth` across speculated roots concurrently
        //    in shared batched `stepTreeBatch` calls, stopping immediately if
        //    `target.verifyTree(T_r)` has already completed (`stop_expanding`).
        const eff_max_depth: u16 = job.max_draft_depth;
        var level: u16 = 1;
        while (level < eff_max_depth) : (level += 1) {
            if (self.stop_expanding.load(.acquire)) break;
            if (used_slots >= sd.maxTreeSlots()) break;
            const rem_slots: usize = @as(usize, sd.maxTreeSlots() - used_slots);

            var picked: usize = 0;
            for (0..num_roots) |r| {
                if (picked >= rem_slots) break;
                if (res.trees[r].count >= job.candidate_cap) continue;
                const k_r = res.root_prefix_lens[r];
                const r_base_pos = committed_after_pending + k_r;
                if (r_base_pos + 1 >= job.max_seq_len) continue;
                const r_rem_depth: u16 = @intCast(@min(
                    @as(u32, MAX_TREE_NODES - 1),
                    job.max_seq_len - 1 - r_base_pos,
                ));
                const r_target_depth = @min(eff_max_depth, r_rem_depth);
                if (level >= r_target_depth) continue;

                const want: usize = if (r == 0)
                    @as(usize, job.branch_width)
                else
                    1;
                const cap_r = @min(want, rem_slots - picked);
                picked += self.selectFrontierForRoot(&res.trees[r], r, r_target_depth, cap_r, r == 0 and level == 1, picked);
            }
            if (picked == 0) break;

            const slot_start = used_slots;
            for (0..picked) |i| {
                const r_idx: usize = self.batch_roots[i];
                const v = self.batch_nodes[i];
                const node = res.trees[r_idx].nodes[v];
                const parent_slot = self.node_slots[r_idx][node.parent];
                const slot_idx = slot_start + @as(u32, @intCast(i));
                self.slot_ancestor_masks[slot_idx] =
                    self.slot_ancestor_masks[parent_slot] | (@as(u64, 1) << @as(u6, @intCast(slot_idx)));
                self.batch_tokens[i] = node.token_id;
                self.batch_positions[i] = committed_after_pending + res.root_prefix_lens[r_idx] + node.depth;
            }

            const total_slots = slot_start + @as(u32, @intCast(picked));
            const got_k = try sd.stepTreeBatch(
                self.batch_tokens[0..picked],
                self.batch_positions[0..picked],
                self.slot_ancestor_masks[0..total_slots],
                slot_start,
                eff_top_k,
                self.batch_cands[0 .. picked * @as(usize, eff_top_k)],
            );

            for (0..picked) |i| {
                const r_idx: usize = self.batch_roots[i];
                const v = self.batch_nodes[i];
                const slot_idx = slot_start + @as(u32, @intCast(i));
                self.node_slots[r_idx][v] = @intCast(slot_idx);
                self.node_expanded[r_idx][v] = true;

                const row = self.batch_cands[i * got_k .. (i + 1) * got_k];
                for (row) |cand| {
                    if (res.trees[r_idx].count >= job.candidate_cap and
                        res.trees[r_idx].findChild(v, cand.token_id) == null)
                    {
                        break;
                    }
                    _ = res.trees[r_idx].insertChild(v, cand.token_id, cand.prob, .l2);
                }
            }
            used_slots = total_slots;
        }
    }
};

pub const SpecEngine = struct {
    target: TargetModel,
    draft_head: ?*draft_mod.DraftHead,
    small_draft: ?*draft_mod.SmallDraftModel,
    ngram_table: ?*ngram_nvme.NvmeNgramTable,
    ngram_registry: ?*ngram_nvme.DomainNgramRegistry,
    active_ngram_domain: ?[]const u8,
    config: SpecConfig,
    cost_model: tree_mod.PruneCostModel,
    /// Startup-only sampled `[sampled_layers * num_experts]` static byte/BW tables
    /// populated for heterogeneous GSQ+RCO MoE models (`ADR-0017`, empty for uniform).
    expert_bytes_sampled: []u32,
    expert_bw_sampled: []f32,

    sam_arena: arena_mod.WiredArena,
    sam: sam_mod.SuffixAutomaton,

    // Heap-pinned persistent PEARL worker (`D8`, null when `small_draft == null` or `.off`).
    pearl_rt: ?*PearlRuntime,
    pearl_enabled: bool,
    pearl_auto_contended_off: bool,
    pearl_base_verify_us: f64,
    pearl_base_draft_sync_us: f64,
    pearl_probe_rounds: u32,
    pearl_probe_verify_us: [4]f64,
    pearl_probe_hits: u32,
    verify_us_ema: f64,
    has_pearl_ready_tree: bool,
    pearl_ready_tree: tree_mod.DraftTree(MAX_TREE_NODES),
    pearl_pending_sync_len: u16,
    pearl_pending_sync_tokens: [MAX_TREE_NODES + 1]u32,
    pearl_lead_tokens: [MAX_TREE_NODES]u32,
    pearl_rollback_buf: [MAX_TREE_NODES * 2 + 2]u32,

    // Preallocated hot-path scratch buffers (zero dynamic heap allocation on `stepRound`).
    draft_tree: tree_mod.DraftTree(MAX_TREE_NODES),
    eval_tree: tree_mod.DraftTree(MAX_TREE_NODES),
    tree_slot_of_node: [MAX_TREE_NODES]u16,
    slot_expanded: [MAX_TREE_NODES]bool,
    slot_ancestor_masks: [MAX_TREE_NODES]u64,
    node_experts: [MAX_TREE_NODES * MAX_PRED_LAYERS]tree_mod.ExpertBitset,
    cold_masks: [MAX_PRED_LAYERS]tree_mod.ExpertBitset,
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
    ngram_buf: [MAX_TREE_NODES]u32,
    prefetch_tail_buf: [ngram_nvme.MAX_CTX_LEN]u32,
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
        return initInternal(allocator, .{ .qwen3 = target }, draft_head, null, config);
    }

    /// Initializes `SpecEngine` backed by a composable `GraphModel` (`qwen4_exp`,
    /// native `LoopLM`, or `Step-LoRA`), supporting `.l1_only`, `.self_recurrent` (`D12`,
    /// `0` extra draft-model RAM), and `.mtp` / `.eagle2` / `.eagle3` draft heads.
    pub fn initWithGraph(
        allocator: std.mem.Allocator,
        target: *graph.GraphModel,
        draft_head: ?*draft_mod.DraftHead,
        config: SpecConfig,
    ) InitError!SpecEngine {
        return initInternal(allocator, .{ .graph = target }, draft_head, null, config);
    }

    /// Initializes `SpecEngine` paired with a standalone same-family small draft
    /// model (`D4`, e.g. `Qwen3-0.6B` with `Qwen3-14B-4bit`) and optional `D8`
    /// PEARL async pipelining worker thread.
    pub fn initWithSmallDraft(
        allocator: std.mem.Allocator,
        target: *qwen3.Qwen3Model,
        small_draft: *draft_mod.SmallDraftModel,
        config: SpecConfig,
    ) InitError!SpecEngine {
        return initInternal(allocator, .{ .qwen3 = target }, null, small_draft, config);
    }

    fn computeInitialPearlEnabled(
        config: SpecConfig,
        small_draft: ?*draft_mod.SmallDraftModel,
        cost: tree_mod.PruneCostModel,
    ) bool {
        const sd = small_draft orelse return false;
        if (!sd.hasOwnComputeQueue() and !sd.model.hasPureCpuSme2Draft()) return false;
        return switch (config.pearl_mode) {
            .off => false,
            .on => true,
            .auto => @as(f64, cost.t_fixed_us) >= config.pearl_min_target_us,
        };
    }

    fn initInternal(
        allocator: std.mem.Allocator,
        target: TargetModel,
        draft_head: ?*draft_mod.DraftHead,
        small_draft: ?*draft_mod.SmallDraftModel,
        config: SpecConfig,
    ) InitError!SpecEngine {
        if (target.maxTreeNodes() <= 1) return error.TreeNotEnabled;
        if (config.max_tree_nodes < 2 or config.max_tree_nodes > MAX_TREE_NODES) return error.InvalidConfig;
        if (config.max_tree_nodes > target.maxTreeNodes()) return error.InvalidConfig;
        if (config.max_draft_depth == 0 or config.max_draft_depth >= MAX_TREE_NODES) return error.InvalidConfig;
        if (config.branch_width == 0 or config.branch_width > MAX_TREE_NODES) return error.InvalidConfig;
        if (config.top_k_per_node == 0 or config.top_k_per_node > draft_mod.MAX_DRAFT_TOP_K) return error.InvalidConfig;
        if (config.l1_max_draft >= MAX_TREE_NODES) return error.InvalidConfig;
        if (!std.math.isFinite(config.temperature) or config.temperature < 0.0) return error.InvalidConfig;
        if (!std.math.isFinite(config.pearl_min_target_us) or config.pearl_min_target_us < 0.0) return error.InvalidConfig;

        switch (config.mode) {
            .l1_only => {},
            .self_recurrent => {
                if (target != .graph) return error.DraftModeMismatch;
                if (draft_head != null or small_draft != null) return error.DraftModeMismatch;
            },
            .small_draft => {
                if (target != .qwen3) return error.DraftModeMismatch;
                if (small_draft == null) return error.DraftHeadRequired;
                if (draft_head != null) return error.DraftModeMismatch;
            },
            .eagle3, .eagle2, .mtp => {
                const dh = draft_head orelse return error.DraftHeadRequired;
                if (small_draft != null) return error.DraftModeMismatch;
                if (dh.config.format != config.mode) return error.DraftModeMismatch;
                if (!target.enableFeatureTaps()) return error.FeatureTapsRequired;
            },
        }

        const sam_cap: u32 = if (config.sam_capacity > 0)
            config.sam_capacity
        else
            target.maxSeqLen() + MAX_TREE_NODES;
        const sam_bytes = try sam_mod.SuffixAutomaton.requiredBytes(sam_cap);
        const arena_bytes = try page.alignUp(sam_bytes);
        const draft_wired: u64 = if (draft_head) |dh|
            dh.totalWiredBytes()
        else if (small_draft) |sd|
            sd.totalWiredBytes()
        else
            0;
        const existing_wired: u64 = target.totalWiredBytes() + draft_wired;
        if (existing_wired + arena_bytes > target.wiredBudgetBytes()) {
            return error.BudgetExceeded;
        }
        const rem_limit = target.wiredBudgetBytes() - existing_wired;
        const sam_budget = budget_mod.Budget.compute(.{
            .memory_bytes = target.wiredBudgetBytes() + 4 * page.APPLE_PAGE_SIZE,
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

        const p_res = try allocator.alloc(f32, target.vocabSize());
        errdefer allocator.free(p_res);

        var cost = config.cost_model orelse switch (target) {
            .qwen3 => |m| defaultCostModelForTarget(m),
            .graph => |g| defaultCostModelForGraph(g),
        };

        var expert_bytes_sampled: []u32 = &.{};
        var expert_bw_sampled: []f32 = &.{};
        if (config.cost_model == null and target == .qwen3 and target.qwen3.config.arch == .qwen3_moe and target.qwen3.isHeterogeneousMoe()) {
            const tm = target.qwen3;
            const n_layers = tm.config.num_hidden_layers;
            const sampled = cost.num_layers;
            const n_experts = tm.config.num_experts;
            const table_len: usize = @as(usize, sampled) * n_experts;
            expert_bytes_sampled = try allocator.alloc(u32, table_len);
            errdefer allocator.free(expert_bytes_sampled);
            expert_bw_sampled = try allocator.alloc(f32, table_len);
            errdefer allocator.free(expert_bw_sampled);

            for (0..sampled) |s| {
                const l_idx: usize = if (sampled == n_layers)
                    s
                else
                    (s * (n_layers - 1)) / (sampled - 1);
                for (0..n_experts) |e| {
                    const flat = s * n_experts + e;
                    expert_bytes_sampled[flat] = @intCast(tm.expertBytes(l_idx, e));
                    expert_bw_sampled[flat] = schemeEffectiveBwGbs(tm.expertDType(l_idx, e));
                }
            }
            const layer_scale: f64 = @as(f64, @floatFromInt(n_layers)) / @as(f64, @floatFromInt(sampled));
            cost.expert_bytes_table = expert_bytes_sampled;
            cost.expert_bw_gbs_table = expert_bw_sampled;
            cost.expert_table_scale = @floatCast(layer_scale * 0.72);
        }
        errdefer if (expert_bytes_sampled.len > 0) allocator.free(expert_bytes_sampled);
        errdefer if (expert_bw_sampled.len > 0) allocator.free(expert_bw_sampled);

        var pearl_rt: ?*PearlRuntime = null;
        if (small_draft) |sd| {
            if (config.pearl_mode != .off and (sd.hasOwnComputeQueue() or sd.model.hasPureCpuSme2Draft())) {
                pearl_rt = try PearlRuntime.init(allocator, sd);
            }
        }
        errdefer if (pearl_rt) |prt| prt.deinit(allocator);

        const pearl_enabled = computeInitialPearlEnabled(config, small_draft, cost);
        var stats: SpecStats = .{};
        stats.pearl_enabled = pearl_enabled;

        return .{
            .target = target,
            .draft_head = draft_head,
            .small_draft = small_draft,
            .ngram_table = config.ngram_table,
            .ngram_registry = config.ngram_registry,
            .active_ngram_domain = null,
            .config = config,
            .cost_model = cost,
            .expert_bytes_sampled = expert_bytes_sampled,
            .expert_bw_sampled = expert_bw_sampled,
            .sam_arena = sam_arena,
            .sam = sam,
            .pearl_rt = pearl_rt,
            .pearl_enabled = pearl_enabled,
            .pearl_auto_contended_off = false,
            .pearl_base_verify_us = 0.0,
            .pearl_base_draft_sync_us = 0.0,
            .pearl_probe_rounds = 0,
            .pearl_probe_verify_us = @splat(0.0),
            .pearl_probe_hits = 0,
            .verify_us_ema = 0.0,
            .has_pearl_ready_tree = false,
            .pearl_ready_tree = tree_mod.DraftTree(MAX_TREE_NODES).init(0),
            .pearl_pending_sync_len = 0,
            .pearl_pending_sync_tokens = @splat(0),
            .pearl_lead_tokens = @splat(0),
            .pearl_rollback_buf = @splat(0),
            .draft_tree = tree_mod.DraftTree(MAX_TREE_NODES).init(0),
            .eval_tree = tree_mod.DraftTree(MAX_TREE_NODES).init(0),
            .tree_slot_of_node = @splat(tree_mod.NO_NODE),
            .slot_expanded = @splat(false),
            .slot_ancestor_masks = @splat(0),
            .node_experts = @splat(tree_mod.ExpertBitset.empty()),
            .cold_masks = @splat(tree_mod.ExpertBitset.empty()),
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
            .ngram_buf = @splat(0),
            .prefetch_tail_buf = @splat(0),
            .shifted_tokens = @splat(0),
            .p_res = p_res,
            .next_pos = 0,
            .last_token = 0,
            .has_prefilled = false,
            .stats = stats,
        };
    }

    pub fn deinit(self: *SpecEngine, allocator: std.mem.Allocator) void {
        if (self.pearl_rt) |prt| {
            prt.deinit(allocator);
            self.pearl_rt = null;
        }
        if (self.expert_bytes_sampled.len > 0) allocator.free(self.expert_bytes_sampled);
        if (self.expert_bw_sampled.len > 0) allocator.free(self.expert_bw_sampled);
        allocator.free(self.p_res);
        self.sam_arena.deinit();
    }

    pub fn setNgramTable(self: *SpecEngine, tbl: ?*ngram_nvme.NvmeNgramTable) void {
        self.ngram_table = tbl;
    }

    pub fn setActiveNgramDomain(self: *SpecEngine, domain: ?[]const u8) void {
        self.active_ngram_domain = if (domain) |d| (if (d.len > 0) d else null) else null;
    }

    pub fn reset(self: *SpecEngine) void {
        if (self.pearl_rt) |prt| prt.waitIdle();
        self.target.resetState();
        if (self.draft_head) |dh| dh.reset();
        if (self.small_draft) |sd| sd.reset();
        self.sam.reset();
        self.has_pearl_ready_tree = false;
        self.pearl_pending_sync_len = 0;
        self.pearl_pending_sync_tokens = @splat(0);
        self.pearl_auto_contended_off = false;
        self.pearl_base_verify_us = 0.0;
        self.pearl_base_draft_sync_us = 0.0;
        self.pearl_probe_rounds = 0;
        self.pearl_probe_verify_us = @splat(0.0);
        self.pearl_probe_hits = 0;
        self.verify_us_ema = 0.0;
        self.pearl_enabled = computeInitialPearlEnabled(self.config, self.small_draft, self.cost_model);
        self.next_pos = 0;
        self.last_token = 0;
        self.has_prefilled = false;
        self.stats = .{};
        self.stats.pearl_enabled = self.pearl_enabled;
    }

    pub fn totalWiredBytes(self: *const SpecEngine) u64 {
        const draft_bytes: u64 = if (self.draft_head) |dh|
            dh.totalWiredBytes()
        else if (self.small_draft) |sd|
            sd.totalWiredBytes()
        else
            0;
        const ngram_bytes: u64 = (if (self.ngram_table) |nt| nt.totalWiredBytes() else 0) +
            (if (self.ngram_registry) |nr| nr.totalWiredBytes() else 0);
        return self.target.totalWiredBytes() + draft_bytes + self.sam_arena.bytes().len + ngram_bytes;
    }

    pub fn effectivePlacement(self: *const SpecEngine) draft_mod.Placement {
        if (self.draft_head) |dh| return dh.effective_placement;
        if (self.small_draft) |sd| return sd.effective_placement;
        return .gpu;
    }

    pub fn routerPredCalibration(self: *const SpecEngine) qwen3.SmeRouterCalibration {
        return switch (self.target) {
            .qwen3 => |m| m.routerPredCalibration(),
            .graph => std.mem.zeroes(qwen3.SmeRouterCalibration),
        };
    }

    fn effectiveMaxSeqLen(self: *const SpecEngine) u32 {
        if (self.draft_head) |dh| return @min(self.target.maxSeqLen(), dh.max_seq_len);
        if (self.small_draft) |sd| return @min(self.target.maxSeqLen(), sd.maxSeqLen() -| 1);
        return self.target.maxSeqLen();
    }

    fn targetFeatureBinding(self: *const SpecEngine, dh: *const draft_mod.DraftHead) metal.BufferBinding {
        return switch (dh.config.format) {
            .eagle3 => self.target.acceptedAuxBind(),
            .eagle2, .mtp => self.target.acceptedTopBind(),
            .l1_only, .small_draft, .self_recurrent => unreachable,
        };
    }

    fn sampleFirstToken(self: *SpecEngine, row_idx: usize, greedy_tok: u32, rng: ?std.Random) RunError!u32 {
        if (self.config.temperature == 0.0) return greedy_tok;
        const r = rng orelse return error.MissingRng;
        const v: usize = self.target.vocabSize();
        const logits_row = self.target.logitsHost()[row_idx * v .. (row_idx + 1) * v];
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

        const vocab = self.target.vocabSize();
        for (prompt_tokens) |tid| {
            if (tid >= vocab) return error.TokenOutOfRange;
        }

        self.reset();

        const need_top_hidden: bool = if (self.draft_head) |dh|
            (dh.config.format != .eagle3)
        else
            false;
        const max_chunk: usize = if (self.draft_head) |dh|
            if (dh.effective_placement == .cpu_sme2)
                @min(@as(usize, self.target.maxTreeNodes()), @as(usize, dh.max_tree_slots))
            else
                @as(usize, self.target.prefillChunkCap())
        else
            @as(usize, self.target.prefillChunkCap());

        var start: usize = 0;
        var first_tok: u32 = 0;
        while (start < prompt_tokens.len) {
            const take: usize = @min(max_chunk, prompt_tokens.len - start);
            const is_last = (start + take == prompt_tokens.len);

            const greedy_tok = try self.target.prefillChunkWithOptions(
                prompt_tokens[start .. start + take],
                @intCast(start),
                .{
                    .compute_logits = is_last,
                    .preserve_all_top_hidden = need_top_hidden,
                },
            );
            if (is_last) {
                first_tok = try self.sampleFirstToken(self.target.lastLogitsRow(), greedy_tok, rng);
            }

            // Feed prompt tokens into L1 Suffix Automaton.
            for (prompt_tokens[start .. start + take]) |tid| {
                if (self.sam.remainingCapacity() > 0) {
                    self.sam.append(tid) catch |err| switch (err) {
                        error.CapacityExceeded => unreachable,
                    };
                }
            }

            // Synchronize L2 draft head with shifted `(token_{p+1}, target_feature_p)`,
            // slicing across `dh.max_tree_slots` when target ran a wide (`> 64`) prefill chunk.
            if (self.draft_head) |dh| {
                const feat_base = self.targetFeatureBinding(dh);
                const feat_row_bytes: u64 = @as(u64, self.target.hiddenSize()) *
                    (if (dh.config.format == .eagle3) @as(u64, 3) else @as(u64, 1)) * @sizeOf(f16);
                const dh_cap: usize = @min(@as(usize, dh.max_tree_slots), MAX_TREE_NODES);
                var sub: usize = 0;
                while (sub < take) {
                    const sub_take: usize = @min(dh_cap, take - sub);
                    for (0..sub_take) |j| {
                        const idx = start + sub + j;
                        self.shifted_tokens[j] = if (idx + 1 < prompt_tokens.len)
                            prompt_tokens[idx + 1]
                        else
                            first_tok;
                    }
                    const is_final_sub = is_last and (sub + sub_take == take);
                    const top_k: u32 = if (is_final_sub) self.config.top_k_per_node else 0;
                    const sub_feat: metal.BufferBinding = .{
                        .buffer = feat_base.buffer,
                        .offset = feat_base.offset + @as(u64, sub) * feat_row_bytes,
                    };
                    try self.target.syncDraftHead(
                        dh,
                        self.shifted_tokens[0..sub_take],
                        sub_feat,
                        @intCast(start + sub),
                        top_k,
                    );
                    sub += sub_take;
                }
            }

            start += take;
        }

        // Append the emitted first token `t_0` to L1 SAM so round 1 queries the suffix ending at `t_0`.
        if (self.sam.remainingCapacity() > 0) {
            self.sam.append(first_tok) catch |err| switch (err) {
                error.CapacityExceeded => unreachable,
            };
        }

        // C6/C14: Asynchronously prefetch the first round's `.ang` context bucket page.
        if (self.ngram_registry) |nr| {
            _ = try nr.prefetchDomain(self.active_ngram_domain, self.sam.historySlice());
        } else if (self.ngram_table) |nt| {
            _ = try nt.prefetch(self.sam.historySlice());
        }

        // Prefill standalone same-family small draft model (`D4`, zero target hidden-state dep).
        if (self.small_draft) |sd| {
            try sd.prefillAndSeedRoot(prompt_tokens, first_tok, self.config.top_k_per_node);
        }

        self.next_pos = @intCast(prompt_tokens.len);
        self.last_token = first_tok;
        self.has_prefilled = true;
        self.stats.prefill_tokens = prompt_tokens.len;
        return first_tok;
    }

    /// Builds up to `MAX_CTX_LEN` tail tokens combining `sam.historySlice()` and
    /// `extra_tokens` for speculative C6 NVMe N-gram prefetching (zero heap alloc).
    fn buildPrefetchTail(self: *SpecEngine, extra_tokens: []const u32) []const u32 {
        const hist = self.sam.historySlice();
        const total = hist.len + extra_tokens.len;
        const need: usize = @min(total, ngram_nvme.MAX_CTX_LEN);
        if (need == 0) return &.{};
        const from_extra: usize = @min(extra_tokens.len, need);
        const from_hist: usize = need - from_extra;
        if (from_hist > 0) {
            @memcpy(self.prefetch_tail_buf[0..from_hist], hist[hist.len - from_hist .. hist.len]);
        }
        if (from_extra > 0) {
            @memcpy(
                self.prefetch_tail_buf[from_hist..need],
                extra_tokens[extra_tokens.len - from_extra .. extra_tokens.len],
            );
        }
        return self.prefetch_tail_buf[0..need];
    }

    /// Executes one full speculative decode round with zero dynamic heap allocation:
    /// 1) L1 Suffix Automaton (`D1`, `D6`) + optional C6 NVMe `.ang` query when SAM `< 4`,
    /// 2) L2 dynamic EAGLE-2/3/MTP / SmallDraft expansion (or zero-wait D8 PEARL reuse),
    /// 3) D7 expert-byte-budget submodular pruning (`D7`),
    /// 4) L3 GPU tree-attention verification (`G5`) overlapped with C6 NVMe prefetch &
    ///    D8 PEARL round-`r+1` pre-drafting on `small_draft`'s independent compute queue,
    /// 5) Lossless greedy (`T = 0`) or stochastic (`T > 0`) acceptance (`D9`),
    /// 6) Target KV commit + L1 SAM update + L2 draft KV/feature sync or PEARL hit/O(1) rollback.
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
            @as(u16, @intCast(@min(max_emit, @as(usize, self.target.maxTreeNodes())))),
        );
        const candidate_cap: u16 = if (self.config.enable_pruning)
            @min(@as(u16, @intCast(self.target.maxTreeNodes())), MAX_TREE_NODES)
        else
            hard_cap;

        const t_draft_start = std.Io.Timestamp.now(std.Options.debug_io, .awake);

        // Adopt pre-drafted D8 PEARL tree if available, otherwise reset tree to root `0` = `last_token`.
        var used_pearl_tree = false;
        if (self.has_pearl_ready_tree and self.pearl_ready_tree.nodes[0].token_id == self.last_token) {
            self.draft_tree.copyTruncatedFrom(&self.pearl_ready_tree, max_allowed_depth, candidate_cap);
            self.has_pearl_ready_tree = false;
            used_pearl_tree = true;
        } else {
            self.has_pearl_ready_tree = false;
            self.draft_tree.reset(self.last_token);
        }
        @memset(&self.tree_slot_of_node, tree_mod.NO_NODE);
        @memset(&self.slot_expanded, false);
        @memset(&self.slot_ancestor_masks, 0);
        self.tree_slot_of_node[0] = 0; // root hidden lives in draft `hidden_pool[0]`
        self.slot_expanded[0] = true;

        // 1. L1 Suffix Automaton (`D1`, `D6`, ADR-0005) + C6 NVMe N-gram fallback when SAM `< 4`.
        const l1_cap: usize = @min(
            @as(usize, self.config.l1_max_draft),
            @min(@as(usize, max_allowed_depth), @as(usize, candidate_cap - 1)),
        );
        const sam_q = if (l1_cap > 0)
            self.sam.queryActive(l1_cap, self.l1_buf[0..l1_cap])
        else
            sam_mod.QueryResult{ .match_len = 0, .draft_len = 0, .is_seed = false };

        var effective_l1_match_len: u32 = sam_q.match_len;
        var ngram_match_len: u32 = 0;
        var l1_res: tree_mod.InsertL1Result = .{
            .inserted_count = 0,
            .chain_len = 0,
            .seed_tail_node = 0,
            .is_seed = false,
        };

        if (l1_cap > 0 and sam_q.match_len < sam_mod.SEED_MIN_MATCH_LEN and (self.ngram_registry != null or self.ngram_table != null)) {
            const nq: ngram_nvme.QueryResult = if (self.ngram_registry) |nr|
                try nr.queryDomain(self.active_ngram_domain, self.sam.historySlice(), self.ngram_buf[0..l1_cap])
            else blk: {
                const nt = self.ngram_table.?;
                const reads_before = nt.stats.nvme_reads;
                const res = try nt.query(self.sam.historySlice(), self.ngram_buf[0..l1_cap]);
                self.stats.ngram_nvme_reads += nt.stats.nvme_reads -| reads_before;
                break :blk res;
            };
            self.stats.ngram_queries += 1;
            if (nq.cache_hit) self.stats.ngram_cache_hits += 1;

            if (nq.draft_len > 0) {
                ngram_match_len = nq.match_len;
                self.stats.ngram_hits += 1;
                if (nq.is_seed) self.stats.ngram_seed_hits += 1;

                const ngram_seed_prob: f32 = @max(
                    self.config.l1_branch_step_prob,
                    self.config.l1_seed_step_prob * @max(@as(f32, 0.75), nq.step_prob),
                );
                const ngram_branch_prob: f32 = self.config.l1_branch_step_prob * @max(@as(f32, 0.5), nq.step_prob);

                if (nq.match_len > sam_q.match_len) {
                    l1_res = self.draft_tree.insertL1Continuation(
                        nq.match_len,
                        self.ngram_buf[0..nq.draft_len],
                        ngram_seed_prob,
                        ngram_branch_prob,
                    );
                    effective_l1_match_len = nq.match_len;
                    if (sam_q.draft_len > 0) {
                        _ = self.draft_tree.insertL1Continuation(
                            sam_q.match_len,
                            self.l1_buf[0..sam_q.draft_len],
                            self.config.l1_seed_step_prob,
                            self.config.l1_branch_step_prob,
                        );
                    }
                } else {
                    l1_res = self.draft_tree.insertL1Continuation(
                        sam_q.match_len,
                        self.l1_buf[0..sam_q.draft_len],
                        self.config.l1_seed_step_prob,
                        self.config.l1_branch_step_prob,
                    );
                    _ = self.draft_tree.insertL1Continuation(
                        nq.match_len,
                        self.ngram_buf[0..nq.draft_len],
                        ngram_seed_prob,
                        ngram_branch_prob,
                    );
                }
            } else {
                l1_res = self.draft_tree.insertL1Continuation(
                    sam_q.match_len,
                    self.l1_buf[0..sam_q.draft_len],
                    self.config.l1_seed_step_prob,
                    self.config.l1_branch_step_prob,
                );
            }
        } else {
            l1_res = self.draft_tree.insertL1Continuation(
                sam_q.match_len,
                self.l1_buf[0..sam_q.draft_len],
                self.config.l1_seed_step_prob,
                self.config.l1_branch_step_prob,
            );
        }

        if (l1_res.is_seed) {
            self.stats.l1_seed_rounds += 1;
        } else if (l1_res.chain_len > 0) {
            self.stats.l1_branch_rounds += 1;
        }

        // 2. L2 Draft Head or D4 Small Draft Model dynamic tree expansion (`D2`-`D7`).
        //    When D8 PEARL already pre-drafted `draft_tree` during the prior round's
        //    `verifyTree`, skip synchronous small-draft expansion completely!
        if (!used_pearl_tree) {
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
            } else if (self.small_draft) |sd| {
                if (max_allowed_depth >= 1 and candidate_cap > 1) {
                    try self.expandWithSmallDraft(
                        sd,
                        base_pos,
                        max_allowed_depth,
                        candidate_cap,
                        hard_cap,
                        l1_res,
                        t_draft_start,
                    );
                }
            } else if (self.config.mode == .self_recurrent) {
                if (max_allowed_depth >= 1 and candidate_cap > 1) {
                    try self.expandWithSelfRecurrent(
                        self.target.graph,
                        base_pos,
                        max_allowed_depth,
                        candidate_cap,
                    );
                }
            }
        }

        const t_draft_end = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const draft_ns: u64 = @intCast(@max(@as(i96, 0), t_draft_start.durationTo(t_draft_end).toNanoseconds()));
        const drafted_nodes: u16 = self.draft_tree.count;

        // 3. Final D7 expert-byte-budget submodular pruning (`ADR-0009`, `ADR-0017`).
        const t_prune_start = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        if (self.config.enable_pruning and self.draft_tree.count > 1) {
            const draft_us: f32 = @floatCast(@as(f64, @floatFromInt(draft_ns)) / 1000.0);
            _ = try self.pruneTreeInPlace(&self.draft_tree, draft_us, hard_cap);
        } else if (self.draft_tree.count > hard_cap) {
            self.eval_tree.copyTruncatedFrom(&self.draft_tree, max_allowed_depth, hard_cap);
            self.draft_tree = self.eval_tree;
        }
        const t_prune_end = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const prune_ns: u64 = @intCast(@max(@as(i96, 0), t_prune_start.durationTo(t_prune_end).toNanoseconds()));

        // Extract highest-probability lead path of pruned `T_r` for C6 prefetch and D8 PEARL.
        const lead_len: u16 = self.draft_tree.highestProbPath(null, &self.pearl_lead_tokens);

        // C6/C14: Overlapped background NVMe prefetch of the predicted next-round context page.
        if (self.ngram_registry) |nr| {
            const pred_tail = self.buildPrefetchTail(self.pearl_lead_tokens[0..lead_len]);
            if (pred_tail.len >= 2) {
                _ = try nr.prefetchDomain(self.active_ngram_domain, pred_tail);
            }
        } else if (self.ngram_table) |nt| {
            const pred_tail = self.buildPrefetchTail(self.pearl_lead_tokens[0..lead_len]);
            if (pred_tail.len >= nt.header.min_ngram) {
                _ = try nt.prefetch(pred_tail);
            }
        }

        // D8 PEARL: Launch asynchronous round-`r+1` pre-drafting (off-GPU on ARMv9 SME2
        // `ldnt1b` when `sme2_panel64`, or on `small_draft`'s independent Metal 4 queue)
        // right before `target.verifyTree(T_r)`!
        // In `.auto` mode for same-GPU drafts, round 0 runs un-overlapped to sample
        // the uncontended `verify_us` baseline for the ADR-0008 contention gate.
        var pearl_launched = false;
        errdefer if (pearl_launched) {
            if (self.pearl_rt) |prt| prt.waitIdle();
        };
        const off_gpu_draft = if (self.small_draft) |sd| sd.model.hasPureCpuSme2Draft() else false;
        const skip_for_base_probe = (self.config.pearl_mode == .auto and !off_gpu_draft and self.stats.rounds == 0);
        if (self.pearl_enabled and !skip_for_base_probe) {
            if (self.pearl_rt) |prt| {
                if (self.small_draft) |sd| {
                    const committed_before = sd.seqLen();
                    const pending_len: usize = self.pearl_pending_sync_len;
                    const next_base_pos: u32 = base_pos + 1;
                    if (next_base_pos < max_seq and
                        committed_before + pending_len <= sd.maxSeqLen() and
                        committed_before + pending_len == next_base_pos)
                    {
                        var job: PearlJob = .{
                            .base_pos = base_pos,
                            .committed_before = committed_before,
                            .pending_sync_len = self.pearl_pending_sync_len,
                            .lead_len = lead_len,
                            .off_gpu = off_gpu_draft,
                            .greedy = (self.config.temperature == 0.0),
                            .max_draft_depth = self.config.max_draft_depth,
                            .branch_width = self.config.branch_width,
                            .top_k = self.config.top_k_per_node,
                            .min_frontier_prob = self.config.min_frontier_prob,
                            .candidate_cap = candidate_cap,
                            .max_seq_len = max_seq,
                        };
                        if (pending_len > 0) {
                            @memcpy(job.pending_sync_tokens[0..pending_len], self.pearl_pending_sync_tokens[0..pending_len]);
                        }
                        if (lead_len > 0) {
                            @memcpy(job.lead_tokens[0..lead_len], self.pearl_lead_tokens[0..lead_len]);
                        }
                        prt.launch(job);
                        pearl_launched = true;
                        self.stats.pearl_rounds += 1;
                    }
                }
            }
        }

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
        );
        const t_verify_end = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const verify_ns: u64 = @intCast(@max(@as(i96, 0), t_verify_start.durationTo(t_verify_end).toNanoseconds()));
        if (pearl_launched and self.config.temperature == 0.0) {
            self.pearl_rt.?.requestEarlyStop();
        }

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
                self.target.vocabSize(),
                null,
                self.config.temperature,
                rng.?,
                self.p_res,
                self.accepted_nodes[0..],
                out_tokens,
            );

        // 6. Commit accepted target KV + advance L1 SAM + sync L2 DraftHead / D8 PEARL resolve.
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

        // C6/C14: Prefetch actual committed context tail if it diverged from speculated tail.
        if (self.ngram_registry) |nr| {
            _ = try nr.prefetchDomain(self.active_ngram_domain, self.sam.historySlice());
        } else if (self.ngram_table) |nt| {
            _ = try nt.prefetch(self.sam.historySlice());
        }

        var round_pearl_hit = false;
        if (self.draft_head) |dh| {
            const top_k: u32 = if (base_pos + acc.emitted_token_count < max_seq)
                self.config.top_k_per_node
            else
                0;
            try self.target.syncDraftHead(
                dh,
                emitted,
                self.targetFeatureBinding(dh),
                base_pos,
                top_k,
            );
        } else if (pearl_launched) {
            const prt = self.pearl_rt.?;
            const pres = prt.collectResult();
            pearl_launched = false;
            self.stats.pearl_worker_ns += pres.elapsed_ns;
            if (pres.err) |err| return err;

            if (base_pos + acc.emitted_token_count < max_seq) {
                const k_acc: u16 = acc.draft_accepted;
                const prefix_ok = (k_acc <= lead_len) and
                    std.mem.eql(u32, emitted[0..k_acc], self.pearl_lead_tokens[0..k_acc]);
                var matched_root: ?usize = null;
                if (prefix_ok) {
                    for (0..pres.num_roots) |r_idx| {
                        if (pres.root_prefix_lens[r_idx] == k_acc and
                            pres.bonus_tokens[r_idx] == acc.bonus_token)
                        {
                            matched_root = r_idx;
                            break;
                        }
                    }
                }

                if (matched_root) |r_idx| {
                    // Full multi-prefix PEARL hit: reuse pre-drafted `T_{r+1}` directly
                    // on the next round and defer committing `emitted` into next round's
                    // overlapped PEARL pass (`0.0 ms` sync & `0.0 ms` draft on critical path!).
                    self.pearl_ready_tree = pres.trees[r_idx];
                    self.has_pearl_ready_tree = true;
                    self.pearl_pending_sync_len = @intCast(emitted.len);
                    @memcpy(self.pearl_pending_sync_tokens[0..emitted.len], emitted);
                    self.stats.pearl_hits += 1;
                    if (k_acc < lead_len) {
                        self.stats.pearl_partial_prefix_hits += 1;
                    }
                    self.stats.pearl_saved_ns += @min(verify_ns, pres.elapsed_ns);
                    round_pearl_hit = true;
                } else {
                    // Zero-rewind PEARL miss: `lead_tokens` lived in temporary tree slots
                    // only, so `sd.seqLen()` is cleanly `pres.committed_after_pending == base_pos + 1`.
                    const sd = self.small_draft.?;
                    try sd.syncAcceptedChunked(
                        emitted,
                        pres.committed_after_pending,
                        self.config.top_k_per_node,
                    );
                    self.pearl_pending_sync_len = 0;
                    self.has_pearl_ready_tree = false;
                    self.stats.pearl_rollbacks += 1;
                    if (k_acc > 0 and prefix_ok) {
                        self.stats.pearl_partial_prefix_hits += 1;
                    }
                }
            } else {
                self.pearl_pending_sync_len = 0;
                self.has_pearl_ready_tree = false;
            }
        } else if (self.small_draft) |sd| {
            if (base_pos + acc.emitted_token_count < max_seq) {
                const pending_len: usize = self.pearl_pending_sync_len;
                if (pending_len > 0) {
                    const sync_start = sd.seqLen();
                    @memcpy(self.pearl_rollback_buf[0..pending_len], self.pearl_pending_sync_tokens[0..pending_len]);
                    @memcpy(self.pearl_rollback_buf[pending_len .. pending_len + emitted.len], emitted);
                    try sd.syncAcceptedChunked(
                        self.pearl_rollback_buf[0 .. pending_len + emitted.len],
                        sync_start,
                        self.config.top_k_per_node,
                    );
                    self.pearl_pending_sync_len = 0;
                } else {
                    try sd.syncAcceptedChunked(emitted, base_pos + 1, self.config.top_k_per_node);
                }
            } else {
                self.pearl_pending_sync_len = 0;
            }
        }

        const t_commit_end = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const commit_sync_ns: u64 = @intCast(@max(@as(i96, 0), t_commit_start.durationTo(t_commit_end).toNanoseconds()));

        // 7. Advance sequence state, update D8 `.auto` target-latency + contention gate (`ADR-0008`).
        const verify_us: f64 = @as(f64, @floatFromInt(verify_ns)) / 1000.0;
        const draft_sync_us: f64 = @as(f64, @floatFromInt(draft_ns + commit_sync_ns)) / 1000.0;
        const v_norm_denom: f64 = @as(f64, @floatFromInt(@max(@as(u16, 1), verified_nodes)));
        const verify_us_per_node: f64 = verify_us / v_norm_denom;
        self.verify_us_ema = if (self.stats.rounds == 0)
            verify_us
        else
            (0.7 * self.verify_us_ema + 0.3 * verify_us);
        if (self.config.pearl_mode == .auto and self.pearl_rt != null) {
            if (skip_for_base_probe) {
                self.pearl_base_verify_us = verify_us_per_node;
                self.pearl_base_draft_sync_us = draft_sync_us;
            } else if (!off_gpu_draft and !self.pearl_auto_contended_off and self.pearl_enabled and self.pearl_probe_rounds < 4) {
                self.pearl_probe_verify_us[self.pearl_probe_rounds] = verify_us_per_node;
                self.pearl_probe_rounds += 1;
                if (round_pearl_hit) self.pearl_probe_hits += 1;
                if (self.pearl_probe_rounds == 4 and self.pearl_base_verify_us > 0.0) {
                    var sorted = self.pearl_probe_verify_us;
                    std.mem.sort(f64, &sorted, {}, std.sort.asc(f64));
                    const median_contended_per_node = 0.5 * (sorted[1] + sorted[2]);
                    const est_overhead_us = (median_contended_per_node - self.pearl_base_verify_us) * v_norm_denom;
                    const hit_rate = @as(f64, @floatFromInt(self.pearl_probe_hits)) / 4.0;
                    const est_saved_us = hit_rate * self.pearl_base_draft_sync_us;
                    if (est_overhead_us > est_saved_us + 4000.0 and median_contended_per_node > self.pearl_base_verify_us * 1.12) {
                        self.pearl_auto_contended_off = true;
                    }
                }
            }
            self.pearl_enabled = !self.pearl_auto_contended_off and (self.verify_us_ema >= self.config.pearl_min_target_us);
            self.stats.pearl_enabled = self.pearl_enabled;
        }

        self.next_pos = base_pos + acc.emitted_token_count;
        self.last_token = acc.bonus_token;

        self.stats.rounds += 1;
        self.stats.emitted_tokens += acc.emitted_token_count;
        self.stats.draft_accepted += acc.draft_accepted;
        self.stats.drafted_nodes += drafted_nodes;
        self.stats.verified_nodes += verified_nodes;
        self.stats.total_bytes_moved += vr.actual_bytes_moved;
        if (self.target.isMoe()) {
            self.stats.total_unique_experts += vr.total_unique_experts;
            self.stats.total_moe_layer_passes += self.target.numHiddenLayers();
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
            .l1_match_len = effective_l1_match_len,
            .l1_is_seed = l1_res.is_seed,
            .avg_unique_experts_per_layer = vr.avg_unique_experts_per_layer,
            .actual_bytes_moved = vr.actual_bytes_moved,
            .ngram_match_len = ngram_match_len,
            .pearl_hit = used_pearl_tree or round_pearl_hit,
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
                    try self.target.stepDraftHeadTreeBatch(
                        dh,
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
            try self.target.stepDraftHeadTreeBatch(
                dh,
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

    fn expandWithSmallDraft(
        self: *SpecEngine,
        sd: *draft_mod.SmallDraftModel,
        base_pos: u32,
        max_allowed_depth: u16,
        candidate_cap: u16,
        hard_cap: u16,
        l1_res: tree_mod.InsertL1Result,
        t_draft_start: std.Io.Timestamp,
    ) RunError!void {
        const eff_top_k: u32 = @min(self.config.top_k_per_node, sd.model.config.vocab_size);
        const root_cands = sd.rootCandidates();
        const root_take = @min(root_cands.len, @as(usize, eff_top_k));

        // Attach cached depth-1 candidates directly onto root `0` (zero extra GPU work;
        // deduplicates in place with any L1 candidates per ADR-0005).
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
        if (l1_res.is_seed and l1_res.chain_len >= target_depth) return;

        var used_slots: u32 = 0;

        // ADR-0005: Feed an entire L1 `>= 4` seed chain `[s_1..s_c]` in ONE single
        // batched forward pass through the standalone small draft model!
        if (l1_res.is_seed and l1_res.chain_len > 0 and l1_res.seed_tail_node != 0) {
            var seed_path: [MAX_TREE_NODES]u16 = undefined;
            var path_len: usize = 0;
            var cur: u16 = l1_res.seed_tail_node;
            while (cur != 0 and cur != tree_mod.NO_NODE) : (cur = self.draft_tree.nodes[cur].parent) {
                seed_path[path_len] = cur;
                path_len += 1;
            }
            var lo: usize = 0;
            var hi: usize = path_len;
            while (lo + 1 < hi) {
                hi -= 1;
                const tmp = seed_path[lo];
                seed_path[lo] = seed_path[hi];
                seed_path[hi] = tmp;
                lo += 1;
            }

            if (path_len > 0 and path_len <= sd.maxTreeSlots()) {
                for (0..path_len) |idx| {
                    self.batch_tokens[idx] = self.draft_tree.nodes[seed_path[idx]].token_id;
                }
                const got_k = try sd.stepSeedChain(
                    self.batch_tokens[0..path_len],
                    base_pos + 1,
                    &self.slot_ancestor_masks,
                    eff_top_k,
                    self.batch_cands[0..eff_top_k],
                );
                for (0..path_len) |idx| {
                    self.tree_slot_of_node[seed_path[idx]] = @intCast(1 + idx);
                    self.slot_expanded[seed_path[idx]] = (idx + 1 == path_len);
                }
                used_slots = @intCast(path_len);

                const tail_node = l1_res.seed_tail_node;
                if (self.draft_tree.nodes[tail_node].depth < target_depth) {
                    for (self.batch_cands[0..got_k]) |cand| {
                        if (self.draft_tree.count >= candidate_cap and
                            self.draft_tree.findChild(tail_node, cand.token_id) == null)
                        {
                            break;
                        }
                        _ = self.draft_tree.insertChild(tail_node, cand.token_id, cand.prob, .l2);
                    }
                }
            }
        }

        // Baseline depth-1 score for D7 early stopping (ADR-0009).
        var prev_best_score: f64 = 0.0;
        if (self.config.enable_pruning and self.draft_tree.count > 1) {
            prev_best_score = try self.evaluateCurrentDraftScore(t_draft_start, hard_cap);
        }

        // Dynamic EAGLE-2 confidence-ranked frontier expansion (`depth = 2 .. target_depth`).
        var level: u16 = 1;
        while (level < target_depth) : (level += 1) {
            if (self.draft_tree.count >= candidate_cap) break;
            if (used_slots >= sd.maxTreeSlots()) break;

            const max_batch: usize = @min(
                @as(usize, self.config.branch_width),
                @as(usize, sd.maxTreeSlots() - used_slots),
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
                // Standalone small draft has NO 1-token shift: depth-`d` token lives at `base_pos + d`.
                self.batch_positions[i] = base_pos + node.depth;
            }

            const total_slots = slot_start + @as(u32, @intCast(frontier_count));
            const cand_len = frontier_count * @as(usize, eff_top_k);
            const got_k = try sd.stepTreeBatch(
                self.batch_tokens[0..frontier_count],
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

                const row_cands = self.batch_cands[i * got_k .. (i + 1) * got_k];
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

    /// `D12` `SelfSpecRecurrent` shallow tree expansion (`0` extra draft-model RAM,
    /// `0` hot-path heap allocs): drafts an `M = 8..16` candidate tree using a shallow
    /// `r = 1` / `r = 0` pass of `target` without mutating committed KV/GDN state,
    /// ready for one batched `Core x R` verification pass under lossless `D9` acceptance.
    fn expandWithSelfRecurrent(
        self: *SpecEngine,
        g: *graph.GraphModel,
        base_pos: u32,
        max_allowed_depth: u16,
        candidate_cap: u16,
    ) RunError!void {
        const target_depth: u16 = @min(self.config.max_draft_depth, max_allowed_depth);
        if (target_depth == 0 or candidate_cap <= 1) return;

        const eff_top_k: u32 = @min(self.config.top_k_per_node, @min(g.config.vocab_size, draft_mod.MAX_DRAFT_TOP_K));
        var expanded_mask: [MAX_TREE_NODES]bool = @splat(false);

        var level: u16 = 0;
        while (level < target_depth and self.draft_tree.count < candidate_cap) : (level += 1) {
            const cur_n: u16 = self.draft_tree.count;
            if (cur_n == 0 or cur_n > g.max_tree_nodes) break;

            // Collect highest-probability unexpanded frontier nodes at depth == level.
            var frontier: [MAX_TREE_NODES]u16 = undefined;
            var f_count: usize = 0;
            for (0..cur_n) |ni| {
                const node = self.draft_tree.nodes[ni];
                if (expanded_mask[ni] or node.depth != level) continue;
                if (level > 0 and node.cum_prob < self.config.min_frontier_prob) continue;
                frontier[f_count] = @intCast(ni);
                f_count += 1;
            }
            if (f_count == 0) break;

            // Sort frontier descending by `cum_prob` and take top `branch_width`.
            const max_take = @min(f_count, @as(usize, self.config.branch_width));
            var i: usize = 0;
            while (i < max_take) : (i += 1) {
                var best = i;
                var j: usize = i + 1;
                while (j < f_count) : (j += 1) {
                    if (self.draft_tree.nodes[frontier[j]].cum_prob > self.draft_tree.nodes[frontier[best]].cum_prob) {
                        best = j;
                    }
                }
                const tmp = frontier[i];
                frontier[i] = frontier[best];
                frontier[best] = tmp;
            }

            // Evaluate the active tree prefix (`cur_n` nodes) with a single shallow `loops = 1` pass!
            self.draft_tree.fillTokenIds(self.verify_tokens[0..cur_n]);
            self.draft_tree.fillPositions(base_pos, self.verify_positions[0..cur_n]);
            self.draft_tree.fillAncestorMasks(self.verify_masks[0..cur_n]);
            const got_k = try g.draftShallowBatch(
                self.verify_tokens[0..cur_n],
                self.verify_positions[0..cur_n],
                self.verify_masks[0..cur_n],
                base_pos,
                eff_top_k,
                self.batch_cands[0 .. @as(usize, cur_n) * eff_top_k],
            );
            self.stats.self_recurrent_draft_steps += 1;

            var added_any = false;
            for (0..max_take) |fi| {
                const parent_idx = frontier[fi];
                expanded_mask[parent_idx] = true;
                const row_cands = self.batch_cands[@as(usize, parent_idx) * got_k ..][0..got_k];
                for (row_cands) |cand| {
                    if (self.draft_tree.count >= candidate_cap and
                        self.draft_tree.findChild(parent_idx, cand.token_id) == null)
                    {
                        break;
                    }
                    const before_cnt = self.draft_tree.count;
                    _ = self.draft_tree.insertChild(parent_idx, cand.token_id, cand.prob, .l2);
                    if (self.draft_tree.count > before_cnt) added_any = true;
                }
            }
            if (!added_any) break;
        }
    }

    fn pruneTreeInPlace(
        self: *SpecEngine,
        t: *tree_mod.DraftTree(MAX_TREE_NODES),
        draft_us: f32,
        hard_cap: u16,
    ) RunError!tree_mod.DraftTree(MAX_TREE_NODES).PruneSummary {
        var cost = self.cost_model;
        cost.t_fixed_us += draft_us;

        if (cost.isMoe() and self.target == .qwen3) {
            const tm = self.target.qwen3;
            const m = t.count;
            t.fillTokenIds(self.verify_tokens[0..m]);
            const sampled = try tm.predictMoEExpertsForTokens(
                self.verify_tokens[0..m],
                self.node_experts[0..],
            );
            cost.num_layers = sampled;
            const bitset_slice = self.node_experts[0 .. @as(usize, m) * sampled];
            const cold_slice: ?[]const tree_mod.ExpertBitset = if (tm.hasColdExperts()) blk: {
                const cm_layers = tm.fillSampledColdMasks(&self.cold_masks);
                break :blk if (cm_layers == sampled) self.cold_masks[0..cm_layers] else null;
            } else null;
            return try t.pruneToOptimalSubtree(cost, bitset_slice, cold_slice, hard_cap);
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

test "M4 D4 SpecEngine with SmallDraftModel matches stepDecode 100% bit-for-bit on Dense BF16 and MoE Q3" {
    const allocator = std.testing.allocator;

    const dense_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .vocab_size = 64,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .max_position_embeddings = 64,
        .tie_word_embeddings = true,
    };
    const draft_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .vocab_size = 64,
        .hidden_size = 64,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .max_position_embeddings = 64,
        .tie_word_embeddings = true,
    };

    const tgt_bytes = try qwen3.buildSyntheticAmfImageWithSpec(allocator, dense_cfg, .{
        .linear_dtype = .bf16,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
    }, 0xD400_0001);
    defer allocator.free(tgt_bytes);

    // Self-draft (same weights as target -> high tau > 1!) AND smaller distinct draft (lossless!)
    const small_bytes = try qwen3.buildSyntheticAmfImageWithSpec(allocator, draft_cfg, .{
        .linear_dtype = .bf16,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
    }, 0xD400_0002);
    defer allocator.free(small_bytes);

    const prompt = [_]u32{ 2, 9, 14, 21, 33, 2, 9, 14, 21 };

    // Reference sequential decode (`T = 0`)
    var ref_model = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_bytes, dense_cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 8,
    });
    defer ref_model.deinit(allocator);
    var expected: [14]u32 = undefined;
    for (prompt[0 .. prompt.len - 1], 0..) |tok, i| try ref_model.prefillToken(tok, @intCast(i));
    expected[0] = try ref_model.stepDecode(prompt[prompt.len - 1], @intCast(prompt.len - 1));
    for (1..expected.len) |s| {
        expected[s] = try ref_model.stepDecode(expected[s - 1], @intCast(prompt.len - 1 + s));
    }

    // 1. Self-draft (`tgt_bytes` as small_draft) verifies high acceptance `tau > 2.0` + 100% bit-exactness
    {
        var spec_tgt = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_bytes, dense_cfg, .{
            .max_seq_len = 64,
            .chunk_tokens = 8,
            .max_tree_nodes = 16,
        });
        defer spec_tgt.deinit(allocator);
        var sd = try draft_mod.SmallDraftModel.loadFromBytes(allocator, &spec_tgt, tgt_bytes, dense_cfg, .{
            .placement = .gpu,
            .max_seq_len = 64,
            .chunk_tokens = 8,
            .max_tree_slots = 16,
        });
        defer sd.deinit(allocator);

        var eng = try SpecEngine.initWithSmallDraft(allocator, &spec_tgt, &sd, .{
            .mode = .small_draft,
            .max_tree_nodes = 12,
            .max_draft_depth = 4,
            .branch_width = 2,
            .top_k_per_node = 3,
            .temperature = 0.0,
            .enable_pruning = true,
        });
        defer eng.deinit(allocator);

        var actual: [14]u32 = undefined;
        actual[0] = try eng.prefill(&prompt, null);
        var prod: usize = 1;
        while (prod < actual.len) {
            const rs = try eng.stepRound(null, actual[prod..]);
            prod += rs.emitted_count;
        }
        try std.testing.expectEqualSlices(u32, &expected, &actual);
        try std.testing.expect(eng.stats.meanAcceptanceLength() > 2.0);
    }

    // 2. Smaller independent draft (`draft_cfg`) verifies 100% lossless bit-for-bit parity across chunk boundary
    {
        var spec_tgt = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_bytes, dense_cfg, .{
            .max_seq_len = 64,
            .chunk_tokens = 8,
            .max_tree_nodes = 16,
        });
        defer spec_tgt.deinit(allocator);
        var sd = try draft_mod.SmallDraftModel.loadFromBytes(allocator, &spec_tgt, small_bytes, draft_cfg, .{
            .placement = .auto,
            .max_seq_len = 64,
            .chunk_tokens = 8,
            .max_tree_slots = 16,
        });
        defer sd.deinit(allocator);

        var eng = try SpecEngine.initWithSmallDraft(allocator, &spec_tgt, &sd, .{
            .mode = .small_draft,
            .max_tree_nodes = 12,
            .max_draft_depth = 4,
            .branch_width = 2,
            .top_k_per_node = 3,
            .temperature = 0.0,
            .enable_pruning = true,
        });
        defer eng.deinit(allocator);

        var actual: [14]u32 = undefined;
        actual[0] = try eng.prefill(&prompt, null);
        var prod: usize = 1;
        while (prod < actual.len) {
            const rs = try eng.stepRound(null, actual[prod..]);
            prod += rs.emitted_count;
        }
        try std.testing.expectEqualSlices(u32, &expected, &actual);
    }
}

test "M4 SpecEngine on cold-expert NVMe MoE target applies D7 cold penalty and matches all-hot bit-for-bit" {
    const allocator = std.testing.allocator;
    const cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .default_float_dtype = .bf16,
        .vocab_size = 64,
        .hidden_size = 128,
        .num_hidden_layers = 3,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 12,
        .num_experts_per_tok = 2,
        .norm_topk_prob = true,
    };
    const image = try qwen3.buildSyntheticAmfImageWithSpec(allocator, cfg, .{
        .linear_dtype = .q3_affine,
        .embed_dtype = .q3_affine,
        .norm_dtype = .bf16,
        .router_dtype = .q3_affine,
        .expert_dtype = .q3_affine,
    }, 0xC01D_0001);
    defer allocator.free(image);

    const tmp_path: [:0]const u8 = "/Volumes/Akari/.tmp/aminfer-test-m4-spec-cold.amf";
    {
        const fd = std.c.open(tmp_path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.close(fd);
        var written: usize = 0;
        while (written < image.len) {
            const rc = std.c.write(fd, image.ptr + written, image.len - written);
            try std.testing.expect(rc > 0);
            written += @intCast(rc);
        }
    }
    defer _ = std.c.unlink(tmp_path.ptr);

    const prompt = [_]u32{ 4, 8, 12, 16, 20, 4, 8, 12, 16 };

    var ref = try qwen3.Qwen3Model.loadFromFile(allocator, tmp_path, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 16,
    });
    defer ref.deinit(allocator);
    var expected: [10]u32 = undefined;
    for (prompt[0 .. prompt.len - 1], 0..) |tok, i| try ref.prefillToken(tok, @intCast(i));
    expected[0] = try ref.stepDecode(prompt[prompt.len - 1], @intCast(prompt.len - 1));
    for (1..expected.len) |s| {
        expected[s] = try ref.stepDecode(expected[s - 1], @intCast(prompt.len - 1 + s));
    }

    var cold_tgt = try qwen3.Qwen3Model.loadFromFile(allocator, tmp_path, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 16,
        .max_tree_nodes = 12,
        .cold_experts_per_layer = 6,
        .cold_slot_count = 16,
        .enable_ane_predictor = true,
    });
    defer cold_tgt.deinit(allocator);

    var eng = try SpecEngine.init(allocator, &cold_tgt, null, .{
        .mode = .l1_only,
        .max_tree_nodes = 10,
        .enable_pruning = true,
        .temperature = 0.0,
    });
    defer eng.deinit(allocator);
    try std.testing.expect(eng.cost_model.cold_penalty_bytes > 0.0);

    var actual: [10]u32 = undefined;
    actual[0] = try eng.prefill(&prompt, null);
    var prod: usize = 1;
    while (prod < actual.len) {
        const rs = try eng.stepRound(null, actual[prod..]);
        prod += rs.emitted_count;
    }
    try std.testing.expectEqualSlices(u32, &expected, &actual);
}

test "M4.5 C13 SpecEngine SME2 L2-resident router matches GPU router bit-for-bit at T=0" {
    const allocator = std.testing.allocator;
    const cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .default_float_dtype = .bf16,
        .vocab_size = 96,
        .hidden_size = 128,
        .num_hidden_layers = 4,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 16,
        .num_experts_per_tok = 2,
        .norm_topk_prob = true,
    };
    const image = try qwen3.buildSyntheticAmfImageWithSpec(allocator, cfg, .{
        .linear_dtype = .q3_affine,
        .embed_dtype = .q3_affine,
        .norm_dtype = .bf16,
        .router_dtype = .bf16,
        .expert_dtype = .q3_affine,
    }, 0x45C13);
    defer allocator.free(image);

    const prompt = [_]u32{ 2, 9, 17, 25, 2, 9, 17, 25 };

    var tgt_gpu = try qwen3.Qwen3Model.loadFromBytes(allocator, image, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 16,
        .max_tree_nodes = 12,
        .sme_router_mode = .gpu,
    });
    defer tgt_gpu.deinit(allocator);

    var eng_gpu = try SpecEngine.init(allocator, &tgt_gpu, null, .{
        .mode = .l1_only,
        .max_tree_nodes = 10,
        .enable_pruning = true,
        .temperature = 0.0,
    });
    defer eng_gpu.deinit(allocator);
    try std.testing.expectEqual(qwen3.RouterPredBackend.gpu, eng_gpu.routerPredCalibration().backend);

    var out_gpu: [12]u32 = undefined;
    out_gpu[0] = try eng_gpu.prefill(&prompt, null);
    var p_g: usize = 1;
    while (p_g < out_gpu.len) {
        const rs = try eng_gpu.stepRound(null, out_gpu[p_g..]);
        p_g += rs.emitted_count;
    }

    var tgt_sme = try qwen3.Qwen3Model.loadFromBytes(allocator, image, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 16,
        .max_tree_nodes = 12,
        .sme_router_mode = .auto,
    });
    defer tgt_sme.deinit(allocator);

    var eng_sme = try SpecEngine.init(allocator, &tgt_sme, null, .{
        .mode = .l1_only,
        .max_tree_nodes = 10,
        .enable_pruning = true,
        .temperature = 0.0,
    });
    defer eng_sme.deinit(allocator);

    const caps = try sme.SmeCaps.probe();
    if (caps.supportsSme2()) {
        try std.testing.expectEqual(qwen3.RouterPredBackend.sme2, eng_sme.routerPredCalibration().backend);
    }

    var out_sme: [12]u32 = undefined;
    out_sme[0] = try eng_sme.prefill(&prompt, null);
    var p_s: usize = 1;
    while (p_s < out_sme.len) {
        const rs = try eng_sme.stepRound(null, out_sme[p_s..]);
        p_s += rs.emitted_count;
    }
    try std.testing.expectEqualSlices(u32, &out_gpu, &out_sme);
}

test "M4.5 D11 SpecEngine Secure Enclave TRNG seeds replay bit-exact with explicit seed" {
    const allocator = std.testing.allocator;
    const cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = null,
        .default_float_dtype = .bf16,
        .vocab_size = 96,
        .hidden_size = 64,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .tie_word_embeddings = true,
    };
    const image = try qwen3.buildSyntheticAmfImageWithSpec(allocator, cfg, .{
        .linear_dtype = .bf16,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
    }, 0x45D11);
    defer allocator.free(image);

    var tgt = try qwen3.Qwen3Model.loadFromBytes(allocator, image, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 16,
        .max_tree_nodes = 8,
    });
    defer tgt.deinit(allocator);

    var eng = try SpecEngine.init(allocator, &tgt, null, .{
        .mode = .l1_only,
        .max_tree_nodes = 8,
        .enable_pruning = true,
        .temperature = 0.7,
    });
    defer eng.deinit(allocator);

    const prompt = [_]u32{ 3, 11, 23, 37 };

    var rng_hw = try rng_mod.RequestRng.init(null);
    try std.testing.expectEqual(rng_mod.SeedSource.hardware_trng, rng_hw.seedSource());
    var out_hw: [10]u32 = undefined;
    out_hw[0] = try eng.prefill(&prompt, rng_hw.random());
    var p1: usize = 1;
    while (p1 < out_hw.len) {
        const rs = try eng.stepRound(rng_hw.random(), out_hw[p1..]);
        p1 += rs.emitted_count;
    }

    var rng_replay = try rng_mod.RequestRng.init(rng_hw.seed_used);
    try std.testing.expectEqual(rng_mod.SeedSource.explicit, rng_replay.seedSource());
    var out_replay: [10]u32 = undefined;
    out_replay[0] = try eng.prefill(&prompt, rng_replay.random());
    var p2: usize = 1;
    while (p2 < out_replay.len) {
        const rs = try eng.stepRound(rng_replay.random(), out_replay[p2..]);
        p2 += rs.emitted_count;
    }
    try std.testing.expectEqualSlices(u32, &out_hw, &out_replay);
}

test {
    _ = ngram_nvme;
}

test "M5 C6 SpecEngine with F_NOCACHE NvmeNgramTable produces seed hits and 100% bit-exact T=0 tokens" {
    const allocator = std.testing.allocator;
    const cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = null,
        .default_float_dtype = .bf16,
        .vocab_size = 64,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .tie_word_embeddings = true,
    };
    const image = try qwen3.buildSyntheticAmfImageWithSpec(allocator, cfg, .{
        .linear_dtype = .bf16,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
    }, 0xC6_0001);
    defer allocator.free(image);

    // Strictly unique prompt tokens (zero in-context SAM matches!).
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6 };

    // 1. Reference sequential greedy decode (`T = 0`).
    var ref = try qwen3.Qwen3Model.loadFromBytes(allocator, image, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 8,
    });
    defer ref.deinit(allocator);

    var expected: [14]u32 = undefined;
    for (prompt[0 .. prompt.len - 1], 0..) |tok, i| try ref.prefillToken(tok, @intCast(i));
    expected[0] = try ref.stepDecode(prompt[prompt.len - 1], @intCast(prompt.len - 1));
    for (1..expected.len) |s| {
        expected[s] = try ref.stepDecode(expected[s - 1], @intCast(prompt.len - 1 + s));
    }

    // 2. Build an `.ang` v1 table on `/Volumes/Akari/.tmp` indexing the corpus continuation
    //    after `[prompt[3], prompt[4], prompt[5], expected[0]]` -> `expected[1..9]` plus a second hop!
    const tmp_ang: [:0]const u8 = "/Volumes/Akari/.tmp/aminfer-test-m5-c6-engine.ang";
    defer _ = std.c.unlink(tmp_ang.ptr);

    const ctx1 = [_]u32{ prompt[3], prompt[4], prompt[5], expected[0] };
    const ctx2 = [_]u32{ expected[5], expected[6], expected[7], expected[8] };
    const entries = [_]ngram_nvme.BuildEntry{
        .{ .ctx = &ctx1, .cont = expected[1..9], .hit_count = 30, .step_prob = 0.95 },
        .{ .ctx = &ctx2, .cont = expected[9..14], .hit_count = 20, .step_prob = 0.92 },
    };
    try ngram_nvme.writeTableFile(allocator, tmp_ang, &entries, .{
        .num_buckets = 16,
        .vocab_size = cfg.vocab_size,
        .corpus_id = "m5-c6-engine-test",
        .include_table_crc = true,
    });

    var tbl = try ngram_nvme.NvmeNgramTable.open(allocator, tmp_ang, .{
        .cache_pages = 8,
        .max_inflight = 2,
        .io_backend = .kqueue_aio,
        .verify_table_crc = true,
    });
    defer tbl.deinit(allocator);

    // 3. Run SpecEngine WITH `.ang` table attached (`l1_only`, `T = 0`).
    var tgt = try qwen3.Qwen3Model.loadFromBytes(allocator, image, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 8,
        .max_tree_nodes = 16,
    });
    defer tgt.deinit(allocator);

    var eng = try SpecEngine.init(allocator, &tgt, null, .{
        .mode = .l1_only,
        .max_tree_nodes = 12,
        .l1_max_draft = 8,
        .temperature = 0.0,
        .enable_pruning = true,
        .ngram_table = &tbl,
    });
    defer eng.deinit(allocator);

    var actual: [14]u32 = undefined;
    actual[0] = try eng.prefill(&prompt, null);
    var prod: usize = 1;
    while (prod < actual.len) {
        const rs = try eng.stepRound(null, actual[prod..]);
        prod += rs.emitted_count;
    }

    try std.testing.expectEqualSlices(u32, &expected, &actual);
    try std.testing.expect(eng.stats.ngram_queries > 0);
    try std.testing.expect(eng.stats.ngram_hits > 0);
    try std.testing.expect(eng.stats.ngram_seed_hits > 0);
    try std.testing.expect(eng.stats.draft_accepted >= 6);
    try std.testing.expect(eng.stats.meanAcceptanceLength() > 2.0);
    try std.testing.expect(tbl.stats.prefetch_submitted > 0);

    // Optional cross-language Python `ngram.py` -> Zig `NvmeNgramTable` verification
    // when invoked with `AMINFER_TEST_PYTHON_ANG=/path/to/python_built.ang`
    // containing `[100, 101, 102, 103] -> [104, 105, 106, 107]`.
    if (std.c.getenv("AMINFER_TEST_PYTHON_ANG")) |env_ptr| {
        const py_path: [:0]const u8 = std.mem.span(env_ptr);
        const py_hdr = try ngram_nvme.verifyFile(allocator, py_path);
        try std.testing.expect(py_hdr.total_entries > 0);
        try std.testing.expect(py_hdr.hasTableCrc());

        var py_tbl = try ngram_nvme.NvmeNgramTable.open(allocator, py_path, .{
            .cache_pages = 8,
            .max_inflight = 2,
            .io_backend = .kqueue_aio,
            .verify_table_crc = true,
        });
        defer py_tbl.deinit(allocator);

        var py_out: [8]u32 = undefined;
        const py_q = try py_tbl.query(&[_]u32{ 100, 101, 102, 103 }, &py_out);
        try std.testing.expectEqual(@as(u32, 4), py_q.match_len);
        try std.testing.expect(py_q.is_seed);
        try std.testing.expectEqualSlices(u32, &[_]u32{ 104, 105, 106, 107 }, py_out[0..py_q.draft_len]);
    }
}

test "M5 D8 PEARL async pipelined SmallDraft matches pearl=off and stepDecode 100% bit-for-bit across hits, rollbacks, auto gate, and T>0" {
    const allocator = std.testing.allocator;

    const dense_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .vocab_size = 64,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .max_position_embeddings = 64,
        .tie_word_embeddings = false,
    };
    const draft_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .vocab_size = 64,
        .hidden_size = 64,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .max_position_embeddings = 64,
        .tie_word_embeddings = false,
    };

    const tgt_bytes = try qwen3.buildSyntheticAmfImageWithSpec(allocator, dense_cfg, .{
        .linear_dtype = .bf16,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
    }, 0xD800_0001);
    defer allocator.free(tgt_bytes);

    const small_bytes = try qwen3.buildSyntheticAmfImageWithSpec(allocator, draft_cfg, .{
        .linear_dtype = .bf16,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
    }, 0xD800_0002);
    defer allocator.free(small_bytes);

    const prompt = [_]u32{ 3, 8, 15, 22, 29, 36 };

    // Reference sequential greedy decode (`T = 0`).
    var ref = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_bytes, dense_cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 8,
    });
    defer ref.deinit(allocator);
    var expected: [16]u32 = undefined;
    for (prompt[0 .. prompt.len - 1], 0..) |tok, i| try ref.prefillToken(tok, @intCast(i));
    expected[0] = try ref.stepDecode(prompt[prompt.len - 1], @intCast(prompt.len - 1));
    for (1..expected.len) |s| {
        expected[s] = try ref.stepDecode(expected[s - 1], @intCast(prompt.len - 1 + s));
    }

    // 1. Self-draft PEARL `on` vs `off`: exercises full PEARL speculation hits (`pearl_hits > 0`, `pearl_saved_ns > 0`).
    {
        var spec_tgt = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_bytes, dense_cfg, .{
            .max_seq_len = 64,
            .chunk_tokens = 8,
            .max_tree_nodes = 16,
        });
        defer spec_tgt.deinit(allocator);

        var sd = try draft_mod.SmallDraftModel.loadFromBytes(allocator, &spec_tgt, tgt_bytes, dense_cfg, .{
            .placement = .gpu,
            .max_seq_len = 64,
            .chunk_tokens = 8,
            .max_tree_slots = 16,
            .own_compute_queue = true,
        });
        defer sd.deinit(allocator);

        // 1a. PEARL off baseline
        var eng_off = try SpecEngine.initWithSmallDraft(allocator, &spec_tgt, &sd, .{
            .mode = .small_draft,
            .max_tree_nodes = 12,
            .max_draft_depth = 3,
            .branch_width = 2,
            .top_k_per_node = 3,
            .temperature = 0.0,
            .enable_pruning = true,
            .pearl_mode = .off,
        });
        defer eng_off.deinit(allocator);
        try std.testing.expect(!eng_off.stats.pearl_enabled);

        var out_off: [16]u32 = undefined;
        out_off[0] = try eng_off.prefill(&prompt, null);
        var p_off: usize = 1;
        while (p_off < out_off.len) {
            const rs = try eng_off.stepRound(null, out_off[p_off..]);
            p_off += rs.emitted_count;
        }
        try std.testing.expectEqualSlices(u32, &expected, &out_off);
        try std.testing.expectEqual(@as(u64, 0), eng_off.stats.pearl_rounds);

        // 1b. PEARL on (true concurrent hardware overlap!)
        var eng_on = try SpecEngine.initWithSmallDraft(allocator, &spec_tgt, &sd, .{
            .mode = .small_draft,
            .max_tree_nodes = 12,
            .max_draft_depth = 3,
            .branch_width = 2,
            .top_k_per_node = 3,
            .temperature = 0.0,
            .enable_pruning = false,
            .pearl_mode = .on,
        });
        defer eng_on.deinit(allocator);
        try std.testing.expect(eng_on.stats.pearl_enabled);

        var out_on: [16]u32 = undefined;
        out_on[0] = try eng_on.prefill(&prompt, null);
        var p_on: usize = 1;
        while (p_on < out_on.len) {
            const rs = try eng_on.stepRound(null, out_on[p_on..]);
            p_on += rs.emitted_count;
        }
        try std.testing.expectEqualSlices(u32, &expected, &out_on);
        try std.testing.expectEqualSlices(u32, &out_off, &out_on);
        try std.testing.expect(eng_on.stats.pearl_rounds > 0);
        try std.testing.expect(eng_on.stats.pearl_hits > 0);
        try std.testing.expect(eng_on.stats.pearl_saved_ns > 0);
        try std.testing.expect(eng_on.stats.meanAcceptanceLength() > 2.0);
    }

    // 2. Distinct small draft PEARL `on`: exercises O(1) rollback path (`pearl_rollbacks > 0`)
    //    AND `.auto` target-latency gate (`40 ms` default stays off on fast synthetic target,
    //    low threshold gates on dynamically!) + `T = 0.7` stochastic lossless run!
    {
        var spec_tgt = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_bytes, dense_cfg, .{
            .max_seq_len = 64,
            .chunk_tokens = 8,
            .max_tree_nodes = 16,
        });
        defer spec_tgt.deinit(allocator);

        var sd = try draft_mod.SmallDraftModel.loadFromBytes(allocator, &spec_tgt, small_bytes, draft_cfg, .{
            .placement = .gpu,
            .max_seq_len = 64,
            .chunk_tokens = 8,
            .max_tree_slots = 16,
            .own_compute_queue = true,
        });
        defer sd.deinit(allocator);

        // 2a. PEARL on with mismatched small draft -> heavy O(1) KV rollbacks, 100% bit-exact T=0!
        var eng_rb = try SpecEngine.initWithSmallDraft(allocator, &spec_tgt, &sd, .{
            .mode = .small_draft,
            .max_tree_nodes = 12,
            .max_draft_depth = 3,
            .branch_width = 2,
            .top_k_per_node = 3,
            .temperature = 0.0,
            .enable_pruning = false,
            .pearl_mode = .on,
        });
        defer eng_rb.deinit(allocator);

        var out_rb: [16]u32 = undefined;
        out_rb[0] = try eng_rb.prefill(&prompt, null);
        var p_rb: usize = 1;
        while (p_rb < out_rb.len) {
            const rs = try eng_rb.stepRound(null, out_rb[p_rb..]);
            p_rb += rs.emitted_count;
        }
        try std.testing.expectEqualSlices(u32, &expected, &out_rb);
        try std.testing.expect(eng_rb.stats.pearl_rounds > 0);
        try std.testing.expect(eng_rb.stats.pearl_rollbacks > 0);

        // 2b. `.auto` with default 40 ms gate stays OFF on tiny fast synthetic target!
        var eng_auto_off = try SpecEngine.initWithSmallDraft(allocator, &spec_tgt, &sd, .{
            .mode = .small_draft,
            .max_tree_nodes = 12,
            .max_draft_depth = 3,
            .temperature = 0.0,
            .pearl_mode = .auto,
            .pearl_min_target_us = 40_000.0,
        });
        defer eng_auto_off.deinit(allocator);
        try std.testing.expect(!eng_auto_off.stats.pearl_enabled);
        var out_ao: [10]u32 = undefined;
        out_ao[0] = try eng_auto_off.prefill(&prompt, null);
        var p_ao: usize = 1;
        while (p_ao < out_ao.len) {
            const rs = try eng_auto_off.stepRound(null, out_ao[p_ao..]);
            p_ao += rs.emitted_count;
        }
        try std.testing.expectEqualSlices(u32, expected[0..10], &out_ao);
        try std.testing.expect(!eng_auto_off.stats.pearl_enabled);
        try std.testing.expectEqual(@as(u64, 0), eng_auto_off.stats.pearl_rounds);

        // 2c. `.auto` with low threshold (1 us) gates ON dynamically!
        var eng_auto_on = try SpecEngine.initWithSmallDraft(allocator, &spec_tgt, &sd, .{
            .mode = .small_draft,
            .max_tree_nodes = 12,
            .max_draft_depth = 3,
            .temperature = 0.0,
            .pearl_mode = .auto,
            .pearl_min_target_us = 1.0,
        });
        defer eng_auto_on.deinit(allocator);
        var out_an: [10]u32 = undefined;
        out_an[0] = try eng_auto_on.prefill(&prompt, null);
        var p_an: usize = 1;
        while (p_an < out_an.len) {
            const rs = try eng_auto_on.stepRound(null, out_an[p_an..]);
            p_an += rs.emitted_count;
        }
        try std.testing.expectEqualSlices(u32, expected[0..10], &out_an);
        try std.testing.expect(eng_auto_on.stats.pearl_rounds > 0);
        try std.testing.expect(eng_auto_on.stats.pearl_enabled or eng_auto_on.pearl_auto_contended_off);

        // 2d. Stochastic T = 0.7 with PEARL on: valid tokens, zero errors, reproducible with same seed!
        var eng_st = try SpecEngine.initWithSmallDraft(allocator, &spec_tgt, &sd, .{
            .mode = .small_draft,
            .max_tree_nodes = 12,
            .max_draft_depth = 3,
            .temperature = 0.7,
            .enable_pruning = false,
            .pearl_mode = .on,
        });
        defer eng_st.deinit(allocator);

        var rng_a = try rng_mod.RequestRng.init(0xD8_570C);
        var st_a: [12]u32 = undefined;
        st_a[0] = try eng_st.prefill(&prompt, rng_a.random());
        var ps_a: usize = 1;
        while (ps_a < st_a.len) {
            const rs = try eng_st.stepRound(rng_a.random(), st_a[ps_a..]);
            ps_a += rs.emitted_count;
        }
        for (st_a) |t| try std.testing.expect(t < dense_cfg.vocab_size);

        var rng_b = try rng_mod.RequestRng.init(0xD8_570C);
        var st_b: [12]u32 = undefined;
        st_b[0] = try eng_st.prefill(&prompt, rng_b.random());
        var ps_b: usize = 1;
        while (ps_b < st_b.len) {
            const rs = try eng_st.stepRound(rng_b.random(), st_b[ps_b..]);
            ps_b += rs.emitted_count;
        }
        try std.testing.expectEqualSlices(u32, &st_a, &st_b);
    }

    // 3. Off-GPU pure-CPU ARMv9 SME2 (`sme2_panel64`) SmallDraftModel + Multi-Prefix PEARL:
    //    verifies `hasPureCpuSme2Draft() == true`, `--pearl on` & `--pearl auto`, `pearl_hits > 0`,
    //    and 100% bit-exact `T = 0` parity with zero GPU queue submissions from the draft model!
    {
        const sme_draft_bytes = try qwen3.repackAmfBytesToSme2(allocator, tgt_bytes, .all_linear, null);
        defer allocator.free(sme_draft_bytes);

        var spec_tgt = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_bytes, dense_cfg, .{
            .max_seq_len = 64,
            .chunk_tokens = 8,
            .max_tree_nodes = 16,
        });
        defer spec_tgt.deinit(allocator);

        var sd_sme = try draft_mod.SmallDraftModel.loadFromBytes(allocator, &spec_tgt, sme_draft_bytes, dense_cfg, .{
            .placement = .gpu,
            .max_seq_len = 64,
            .chunk_tokens = 8,
            .max_tree_slots = 32,
            .own_compute_queue = true,
        });
        defer sd_sme.deinit(allocator);
        try std.testing.expect(sd_sme.model.hasPureCpuSme2Draft());

        var eng_sme = try SpecEngine.initWithSmallDraft(allocator, &spec_tgt, &sd_sme, .{
            .mode = .small_draft,
            .max_tree_nodes = 12,
            .max_draft_depth = 3,
            .branch_width = 2,
            .top_k_per_node = 3,
            .temperature = 0.0,
            .enable_pruning = false,
            .pearl_mode = .auto,
            .pearl_min_target_us = 1.0,
        });
        defer eng_sme.deinit(allocator);

        var out_sme: [16]u32 = undefined;
        out_sme[0] = try eng_sme.prefill(&prompt, null);
        var p_sme: usize = 1;
        while (p_sme < out_sme.len) {
            const rs = try eng_sme.stepRound(null, out_sme[p_sme..]);
            p_sme += rs.emitted_count;
        }
        try std.testing.expectEqualSlices(u32, &expected, &out_sme);
        try std.testing.expect(eng_sme.stats.pearl_enabled);
        try std.testing.expect(eng_sme.stats.pearl_rounds > 0);
        try std.testing.expect(eng_sme.stats.pearl_hits > 0);
    }
}

test "SpecEngine wires ADR-0017 heterogeneous MoE per-expert byte tables into D7 pruning with T=0 bit-exact parity" {
    const allocator = std.testing.allocator;
    const moe_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .group_size = 64,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 12,
        .num_experts_per_tok = 4,
        .norm_topk_prob = true,
        .tie_word_embeddings = false,
    };
    const mix = [_]quant.WeightDType{ .q2_affine, .q3_affine, .q4_affine };
    const img = try qwen3.buildSyntheticAmfImageWithSpec(allocator, moe_cfg, .{
        .linear_dtype = .q4_affine,
        .embed_dtype = .q4_affine,
        .norm_dtype = .bf16,
        .router_dtype = .q3_affine,
        .expert_dtypes_cycle = &mix,
    }, 0xAD17_D701);
    defer allocator.free(img);

    // Baseline non-speculative greedy tokens.
    var base_model = try qwen3.Qwen3Model.loadFromBytes(allocator, img, moe_cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 8,
        .max_tree_nodes = 1,
    });
    defer base_model.deinit(allocator);
    try std.testing.expect(base_model.isHeterogeneousMoe());

    const prompt = [_]u32{ 5, 19, 41, 77, 12, 5, 19, 41 };
    var expected: [14]u32 = undefined;
    var tok = try base_model.prefillPrompt(&prompt, true);
    expected[0] = tok;
    var pos: u32 = @intCast(prompt.len);
    for (1..expected.len) |i| {
        tok = try base_model.stepDecode(tok, pos);
        expected[i] = tok;
        pos += 1;
    }

    // SpecEngine with D7 pruning auto-wired to heterogeneous per-expert byte & BW tables.
    var spec_model = try qwen3.Qwen3Model.loadFromBytes(allocator, img, moe_cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 8,
        .max_tree_nodes = 16,
    });
    defer spec_model.deinit(allocator);

    var eng = try SpecEngine.init(allocator, &spec_model, null, .{
        .mode = .l1_only,
        .max_tree_nodes = 12,
        .max_draft_depth = 4,
        .enable_pruning = true,
        .temperature = 0.0,
    });
    defer eng.deinit(allocator);

    try std.testing.expect(eng.cost_model.hasPerExpertTable());
    const bytes_tbl = eng.cost_model.expert_bytes_table.?;
    const bw_tbl = eng.cost_model.expert_bw_gbs_table.?;
    try std.testing.expectEqual(@as(usize, 24), bytes_tbl.len);
    try std.testing.expectEqual(@as(usize, 24), bw_tbl.len);
    for (0..moe_cfg.num_hidden_layers) |l| {
        for (0..moe_cfg.num_experts) |e| {
            const idx = l * moe_cfg.num_experts + e;
            try std.testing.expectEqual(@as(u32, @intCast(spec_model.expertBytes(l, e))), bytes_tbl[idx]);
            try std.testing.expectApproxEqAbs(schemeEffectiveBwGbs(spec_model.expertDType(l, e)), bw_tbl[idx], 1e-6);
        }
    }

    var got: [14]u32 = undefined;
    got[0] = try eng.prefill(&prompt, null);
    var p: usize = 1;
    while (p < got.len) {
        const rs = try eng.stepRound(null, got[p..]);
        p += rs.emitted_count;
    }
    try std.testing.expectEqualSlices(u32, &expected, &got);
}

test "D12: SelfSpecRecurrent on native LoopLM (R=3) is T=0 bit-exact with stepDecode & valid at T>0" {
    const allocator = std.testing.allocator;
    const h: u32 = 64;
    const vocab: u32 = 128;
    const n_layers: u32 = 4;
    const layer_attns = try allocator.alloc(graph.AttentionKind, n_layers);
    @memset(layer_attns, .gqa_full);
    var cfg: graph.GraphConfig = .{
        .axes = .{
            .backbone_kind = .decoder_only,
            .layer_attention_kinds = layer_attns,
            .ffn_kind = .dense_swiglu,
            .has_shared_expert = false,
            .residual_kind = .single_stream,
            .residual_branches = 1,
            .memory_lookup_kind = .none,
            .loop_schedule = .{
                .mode = .native_recurrent,
                .start_layer = 1,
                .end_layer = 3,
                .max_loops = 3,
                .inject_x_init = true,
            },
        },
        .vocab_size = vocab,
        .hidden_size = h,
        .num_hidden_layers = n_layers,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .rope_theta = 10_000.0,
        .tie_word_embeddings = true,
    };
    defer cfg.deinit(allocator);

    // Build deterministic dense AMF compatible with `GraphModel`.
    const q3_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .vocab_size = vocab,
        .hidden_size = h,
        .num_hidden_layers = n_layers,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .rope_theta = 10_000.0,
        .tie_word_embeddings = true,
    };
    const img = try draft_mod.buildSyntheticTargetBf16ForTest(allocator, q3_cfg, 88811);
    defer allocator.free(img);

    const prompt = [_]u32{ 3, 14, 27, 45 };
    var ref_model = try graph.GraphModel.loadFromBytes(allocator, img, cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
    });
    defer ref_model.deinit(allocator);

    var expected: [10]u32 = undefined;
    expected[0] = try ref_model.prefillChunk(&prompt, 0, true);
    var pos: u32 = prompt.len;
    for (1..expected.len) |i| {
        expected[i] = try ref_model.stepDecode(expected[i - 1], pos);
        pos += 1;
    }

    var spec_model = try graph.GraphModel.loadFromBytes(allocator, img, cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
    });
    defer spec_model.deinit(allocator);

    var eng = try SpecEngine.initWithGraph(allocator, &spec_model, null, .{
        .mode = .self_recurrent,
        .max_tree_nodes = 12,
        .max_draft_depth = 4,
        .branch_width = 2,
        .top_k_per_node = 2,
        .enable_pruning = false,
        .temperature = 0.0,
    });
    defer eng.deinit(allocator);

    var got: [10]u32 = undefined;
    got[0] = try eng.prefill(&prompt, null);
    var p: usize = 1;
    while (p < got.len) {
        const rs = try eng.stepRound(null, got[p..]);
        p += rs.emitted_count;
    }
    try std.testing.expectEqualSlices(u32, &expected, &got);
    try std.testing.expect(eng.stats.self_recurrent_draft_steps > 0);
    try std.testing.expect(eng.stats.verified_nodes > eng.stats.rounds);

    // Also test stochastic `T > 0` lossless acceptance!
    var eng_stoch = try SpecEngine.initWithGraph(allocator, &spec_model, null, .{
        .mode = .self_recurrent,
        .max_tree_nodes = 8,
        .max_draft_depth = 3,
        .branch_width = 2,
        .top_k_per_node = 2,
        .enable_pruning = false,
        .temperature = 0.7,
    });
    defer eng_stoch.deinit(allocator);
    var prng = std.Random.DefaultPrng.init(42);
    var s_out: [8]u32 = undefined;
    s_out[0] = try eng_stoch.prefill(&prompt, prng.random());
    var sp: usize = 1;
    while (sp < s_out.len) {
        const rs = try eng_stoch.stepRound(prng.random(), s_out[sp..]);
        sp += rs.emitted_count;
    }
    for (s_out) |tid| try std.testing.expect(tid < vocab);
}

test "D13+W9: GraphModel with MTP DraftHead under Placement.cpu_sme2 is T=0 bit-exact vs stepDecode" {
    const allocator = std.testing.allocator;
    const h: u32 = 64;
    const vocab: u32 = 128;
    const n_layers: u32 = 4;
    const layer_attns = try allocator.alloc(graph.AttentionKind, n_layers);
    @memset(layer_attns, .gqa_full);
    var cfg: graph.GraphConfig = .{
        .axes = .{
            .backbone_kind = .decoder_only,
            .layer_attention_kinds = layer_attns,
            .ffn_kind = .dense_swiglu,
            .has_shared_expert = false,
            .residual_kind = .single_stream,
            .residual_branches = 1,
            .memory_lookup_kind = .none,
            .loop_schedule = .{ .mode = .single_pass, .start_layer = 0, .end_layer = n_layers, .max_loops = 1 },
        },
        .vocab_size = vocab,
        .hidden_size = h,
        .num_hidden_layers = n_layers,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .rope_theta = 10_000.0,
        .tie_word_embeddings = true,
    };
    defer cfg.deinit(allocator);

    const q3_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .vocab_size = vocab,
        .hidden_size = h,
        .num_hidden_layers = n_layers,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .rope_theta = 10_000.0,
        .tie_word_embeddings = true,
    };
    const img = try draft_mod.buildSyntheticTargetBf16ForTest(allocator, q3_cfg, 77722);
    defer allocator.free(img);

    const mtp_cfg: draft_mod.DraftConfig = .{
        .format = .mtp,
        .vocab_size = vocab,
        .draft_vocab_size = vocab,
        .hidden_size = h,
        .target_hidden_size = h,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .rope_theta = 10_000.0,
        .share_embed_tokens = true,
        .share_lm_head = true,
    };
    const mtp_img = try draft_mod.buildSyntheticDraftAmfImage(allocator, mtp_cfg, 77733);
    defer allocator.free(mtp_img);

    const prompt = [_]u32{ 8, 21, 55 };
    var ref_model = try graph.GraphModel.loadFromBytes(allocator, img, cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
    });
    defer ref_model.deinit(allocator);
    var expected: [8]u32 = undefined;
    expected[0] = try ref_model.prefillChunk(&prompt, 0, true);
    var pos: u32 = prompt.len;
    for (1..expected.len) |i| {
        expected[i] = try ref_model.stepDecode(expected[i - 1], pos);
        pos += 1;
    }

    var spec_model = try graph.GraphModel.loadFromBytes(allocator, img, cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .enable_feature_taps = true,
    });
    defer spec_model.deinit(allocator);

    var dh = try draft_mod.DraftHead.loadFromBytesForGraph(allocator, &spec_model, mtp_img, mtp_cfg, .{
        .placement = .cpu_sme2,
        .max_seq_len = 128,
        .max_tree_slots = 16,
    });
    defer dh.deinitForGraph(allocator, &spec_model);
    try std.testing.expectEqual(draft_mod.Placement.cpu_sme2, dh.effective_placement);

    var eng = try SpecEngine.initWithGraph(allocator, &spec_model, &dh, .{
        .mode = .mtp,
        .placement = .cpu_sme2,
        .max_tree_nodes = 8,
        .max_draft_depth = 3,
        .enable_pruning = false,
        .temperature = 0.0,
    });
    defer eng.deinit(allocator);

    var got: [8]u32 = undefined;
    got[0] = try eng.prefill(&prompt, null);
    var p: usize = 1;
    while (p < got.len) {
        const rs = try eng.stepRound(null, got[p..]);
        p += rs.emitted_count;
    }
    try std.testing.expectEqualSlices(u32, &expected, &got);
    try std.testing.expect(dh.sme2_eval_count > 0);
}

test "D12+W9+W16: 4-layer qwen4_exp (GDN+QSA+MoE+cold+4-branch GR+PLE+Step-LoRA, R=3) chunk-invariant prefill & 16-tok T=0 bit-exact self-recurrent" {
    const allocator = std.testing.allocator;
    const tmp_dir = try graph.makeTestTempDir(allocator, "qwen4_steplora_r3");
    defer {
        graph.removeTestDir(allocator, tmp_dir);
        allocator.free(tmp_dir);
    }

    var b = try graph.buildSynthQwen4Bundle(allocator, .{
        .num_experts = 8,
        .top_k = 2,
        .cold_experts_per_layer = 2,
        .max_loops = 3,
        .ple_num_shards = 1,
        .include_steplora = true,
        .core_start_layer = 0,
        .tmp_dir = tmp_dir,
    });
    defer {
        if (b.amf_file_path) |p| graph.removeTestFile(p);
        for (b.ple_paths) |p| graph.removeTestFile(p);
        b.deinit(allocator);
    }

    var ple_const = try allocator.alloc([:0]const u8, b.ple_paths.len);
    defer allocator.free(ple_const);
    for (b.ple_paths, 0..) |p, i| ple_const[i] = p;
    const amf_path = b.amf_file_path.?;

    // 21-token prompt forces multi-chunk prefill on chunk=4 (4+4+4+4+4+1) vs single-chunk on chunk=32.
    const prompt = [_]u32{ 5, 19, 41, 7, 102, 13, 64, 28, 91, 3, 55, 118, 22, 77, 10, 48, 83, 14, 66, 39, 97 };

    // 1. Chunk-size-invariant `prefillChunk` (chunk=4 vs chunk=32) + 16-token serial `stepDecode`.
    var ref_chunk4 = try graph.GraphModel.loadFromFile(allocator, amf_path, b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 4,
        .ple_paths = ple_const,
        .cold_slot_count = 32,
        .max_loops_override = 3,
    });
    defer ref_chunk4.deinit(allocator);

    var ref_chunk32 = try graph.GraphModel.loadFromFile(allocator, amf_path, b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 32,
        .ple_paths = ple_const,
        .cold_slot_count = 32,
        .max_loops_override = 3,
    });
    defer ref_chunk32.deinit(allocator);

    var first_c4: u32 = 0;
    var p_off: usize = 0;
    while (p_off < prompt.len) {
        const take: usize = @min(@as(usize, 4), prompt.len - p_off);
        const is_last = (p_off + take == prompt.len);
        first_c4 = try ref_chunk4.prefillChunk(prompt[p_off .. p_off + take], @intCast(p_off), is_last);
        p_off += take;
    }
    const first_c32 = try ref_chunk32.prefillChunk(&prompt, 0, true);
    try std.testing.expectEqual(first_c4, first_c32);
    // Bit-exact f16 logits across chunk=4 vs chunk=32!
    const logits_c4 = ref_chunk4.readLastLogitsF16();
    const logits_c32 = ref_chunk32.readLastLogitsF16();
    for (0..b.cfg.vocab_size) |v| {
        try std.testing.expectEqual(@as(u16, @bitCast(logits_c4[v])), @as(u16, @bitCast(logits_c32[v])));
    }

    var expected: [16]u32 = undefined;
    expected[0] = first_c4;
    var pos: u32 = prompt.len;
    for (1..expected.len) |i| {
        expected[i] = try ref_chunk4.stepDecode(expected[i - 1], pos);
        pos += 1;
    }

    // 2. `SpecEngine` `self_recurrent` (`max_tree_nodes = 32`, `T = 0`) must match 16/16 bit-exact!
    var spec_model = try graph.GraphModel.loadFromFile(allocator, amf_path, b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 32,
        .ple_paths = ple_const,
        .cold_slot_count = 32,
        .max_loops_override = 3,
    });
    defer spec_model.deinit(allocator);

    var eng = try SpecEngine.initWithGraph(allocator, &spec_model, null, .{
        .mode = .self_recurrent,
        .max_tree_nodes = 16,
        .max_draft_depth = 4,
        .enable_pruning = false,
        .temperature = 0.0,
    });
    defer eng.deinit(allocator);

    var got: [16]u32 = undefined;
    got[0] = try eng.prefill(&prompt, null);
    var p_out: usize = 1;
    while (p_out < got.len) {
        const rs = try eng.stepRound(null, got[p_out..]);
        p_out += rs.emitted_count;
    }
    try std.testing.expectEqualSlices(u32, &expected, &got);

    // 3. Verify all ADR-0019 loop isolations & active SME2 Step-LoRA held throughout!
    const iso = spec_model.loopIsolationStats();
    try std.testing.expect(iso.gdn_scratch_evals > 0);
    try std.testing.expect(iso.ple_r0_fills > 0);
    try std.testing.expect(iso.ple_r_ge1_reuses > 0);
    try std.testing.expectEqual(@as(u64, 0), iso.ple_r_ge1_nvme_reads);
    try std.testing.expect(iso.qsa_index_computes > 0);
    try std.testing.expect(iso.qsa_index_reuses > 0);
    try std.testing.expect(iso.moe_r_ge1_pinned_routes > 0);
    try std.testing.expectEqual(@as(u64, 0), iso.moe_r_ge1_cold_stalls);
    try std.testing.expect(iso.steplora_zero_alpha_skips > 0);
    try std.testing.expect(iso.steplora_active_evals > 0);
}
