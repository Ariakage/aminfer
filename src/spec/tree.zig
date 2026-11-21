//! Compile-time specialized dual-source prefix trie and expert-byte-budget
//! submodular tree pruner (status rows D6 & D7, ADR-0005, ADR-0009, ADR-0014).
//!
//! All tree nodes, bitsets, and pruning temporaries live in fixed stack/inline
//! arrays — zero dynamic heap allocation on any speculative round hot path.

const std = @import("std");
const sam_mod = @import("sam.zig");

/// Sentinel for an absent node index (`parent`, `first_child`, `next_sibling`,
/// unused slots in `remapped_old_indices`).
pub const NO_NODE: u16 = std.math.maxInt(u16);

/// Maximum tree node count supported by the `u64` GPU ancestor bitmask.
pub const MAX_SUPPORTED_NODES: u16 = 64;

/// Maximum MoE layer count supported on the stack inside `pruneToOptimalSubtree`.
pub const MAX_PRUNE_LAYERS: usize = 128;

/// Minimum L1 suffix match length promoted to a high-confidence seed prefix (ADR-0005).
pub const SEED_MIN_MATCH_LEN: u32 = sam_mod.SEED_MIN_MATCH_LEN;

/// Provenance of a node inside the dual-source draft trie (ADR-0005).
pub const NodeSource = enum(u8) {
    root = 0,
    l1_seed = 1,
    l1_branch = 2,
    l2 = 3,
};

/// Single node in a compile-time bounded `DraftTree`.
/// Invariant: node `0` is always the root, and `parent < idx` holds for every
/// node `idx > 0` (topological order preserved across insertion and pruning).
pub const Node = struct {
    token_id: u32,
    parent: u16,
    depth: u16,
    /// Bit `j` is `1` iff node `j` is an ancestor of this node or `j == idx`.
    /// Directly consumed by GPU tree-attention verification kernels (G5).
    ancestor_mask: u64,
    /// Conditional step probability `P(this | parent)` in `[0, 1]`.
    step_prob: f32,
    /// Cumulative path probability `prod_{k on path from root} step_prob[k]`.
    cum_prob: f32,
    first_child: u16,
    next_sibling: u16,
    source: NodeSource,
};

/// Result of inserting an L1 suffix automaton continuation into a `DraftTree`.
pub const InsertL1Result = struct {
    /// Number of newly allocated nodes (excluding deduplicated existing nodes).
    inserted_count: u16,
    /// Total length of the attached or merged L1 chain from root `0`.
    chain_len: u16,
    /// Tail node index of an `.l1_seed` chain (`match_len >= 4`), or `0` for
    /// shorter branches / empty continuations (ADR-0005).
    seed_tail_node: u16,
    /// True iff `match_len >= SEED_MIN_MATCH_LEN` and at least one token was attached.
    is_seed: bool,
};

/// Fixed 256-bit expert activation set per `(node, layer)` (ADR-0009).
pub const ExpertBitset = extern struct {
    words: [4]u64 = .{ 0, 0, 0, 0 },

    pub const MAX_EXPERTS: u32 = 256;

    pub inline fn empty() ExpertBitset {
        return .{};
    }

    pub inline fn set(self: *ExpertBitset, expert_id: u32) void {
        std.debug.assert(expert_id < MAX_EXPERTS);
        self.words[expert_id >> 6] |= (@as(u64, 1) << @as(u6, @intCast(expert_id & 63)));
    }

    pub inline fn clear(self: *ExpertBitset, expert_id: u32) void {
        std.debug.assert(expert_id < MAX_EXPERTS);
        self.words[expert_id >> 6] &= ~(@as(u64, 1) << @as(u6, @intCast(expert_id & 63)));
    }

    pub inline fn contains(self: ExpertBitset, expert_id: u32) bool {
        if (expert_id >= MAX_EXPERTS) return false;
        return (self.words[expert_id >> 6] & (@as(u64, 1) << @as(u6, @intCast(expert_id & 63)))) != 0;
    }

    pub inline fn unionWith(self: ExpertBitset, other: ExpertBitset) ExpertBitset {
        return .{ .words = .{
            self.words[0] | other.words[0],
            self.words[1] | other.words[1],
            self.words[2] | other.words[2],
            self.words[3] | other.words[3],
        } };
    }

    pub inline fn unionInPlace(self: *ExpertBitset, other: ExpertBitset) void {
        inline for (0..4) |w| {
            self.words[w] |= other.words[w];
        }
    }

    pub inline fn intersectWith(self: ExpertBitset, other: ExpertBitset) ExpertBitset {
        return .{ .words = .{
            self.words[0] & other.words[0],
            self.words[1] & other.words[1],
            self.words[2] & other.words[2],
            self.words[3] & other.words[3],
        } };
    }

    /// Bits set in `self` that are absent from `other` (`self \ other`).
    pub inline fn difference(self: ExpertBitset, other: ExpertBitset) ExpertBitset {
        return .{ .words = .{
            self.words[0] & ~other.words[0],
            self.words[1] & ~other.words[1],
            self.words[2] & ~other.words[2],
            self.words[3] & ~other.words[3],
        } };
    }

    /// Total number of active experts in this bitset.
    pub inline fn count(self: ExpertBitset) u32 {
        return @as(u32, @popCount(self.words[0])) +
            @as(u32, @popCount(self.words[1])) +
            @as(u32, @popCount(self.words[2])) +
            @as(u32, @popCount(self.words[3]));
    }

    /// Number of newly introduced experts in `self \ other`.
    pub inline fn differenceCount(self: ExpertBitset, other: ExpertBitset) u32 {
        return @as(u32, @popCount(self.words[0] & ~other.words[0])) +
            @as(u32, @popCount(self.words[1] & ~other.words[1])) +
            @as(u32, @popCount(self.words[2] & ~other.words[2])) +
            @as(u32, @popCount(self.words[3] & ~other.words[3]));
    }
};

comptime {
    if (@sizeOf(ExpertBitset) != 32) @compileError("ExpertBitset must be 32 bytes");
}

/// Calibrated hardware verification cost model for D7 submodular pruning
/// (ADR-0009, ADR-0017).
///
/// Predicted verification time (in microseconds):
/// - Uniform experts (`expert_bytes_table == null and expert_bw_gbs_table == null`):
///   ```
///   t(T) = t_fixed_us
///        + |T| * c_tok_us
///        + sum_{l=0..num_layers-1} (unique_hot_l * expert_bytes
///            + unique_cold_l * (expert_bytes + cold_penalty_bytes))
///          / (beta_gbs_by_m[|T|] * 1000.0)
///   ```
/// - Heterogeneous GSQ+RCO experts (`ADR-0017`, `expert_bytes_table` and/or
///   `expert_bw_gbs_table` provided, indexed `[l * num_experts + e]`):
///   ```
///   t(T) = t_fixed_us + |T| * c_tok_us
///        + sum_l sum_{e in U_l(T)} (b_{l,e} * expert_table_scale) / (beta_{l,e} * 1000.0)
///        + (unique_cold_total * cold_penalty_bytes) / (beta_gbs_by_m[|T|] * 1000.0)
///   ```
/// because `1 GB/s = 1000 B/us`. When `num_experts == 0` or neither scalar
/// `expert_bytes` nor `expert_bytes_table` is set, the expert term vanishes and
/// the rule reduces cleanly to a dense node/latency budget.
pub const PruneCostModel = struct {
    t_fixed_us: f32,
    c_tok_us: f32,
    expert_bytes: f32,
    cold_penalty_bytes: f32 = 0.0,
    beta_gbs_by_m: [65]f32,
    num_layers: u32,
    num_experts: u32,
    /// Optional `[num_layers * num_experts]` static per-expert byte sizes (`b_{l,e}`, ADR-0017).
    /// When `null`, every expert uses the uniform `expert_bytes` scalar.
    expert_bytes_table: ?[]const u32 = null,
    /// Optional `[num_layers * num_experts]` calibrated effective bandwidths in GB/s
    /// (`beta_{scheme(l,e)}`, ADR-0017). When `null`, uses `beta_gbs_by_m[|T|]`.
    expert_bw_gbs_table: ?[]const f32 = null,
    /// Scaling multiplier applied to `expert_bytes_table` entries (e.g. sampled-layer
    /// scale * correlation calibration factor; defaults to `1.0`).
    expert_table_scale: f32 = 1.0,

    /// Convenience dense cost model with zero expert bytes.
    pub fn dense(t_fixed_us: f32, c_tok_us: f32) PruneCostModel {
        return .{
            .t_fixed_us = t_fixed_us,
            .c_tok_us = c_tok_us,
            .expert_bytes = 0.0,
            .cold_penalty_bytes = 0.0,
            .beta_gbs_by_m = @splat(1.0),
            .num_layers = 0,
            .num_experts = 0,
        };
    }

    /// Convenience MoE cost model with a constant calibrated effective bandwidth `beta_gbs`.
    pub fn uniformBeta(
        t_fixed_us: f32,
        c_tok_us: f32,
        expert_bytes: f32,
        cold_penalty_bytes: f32,
        beta_gbs: f32,
        num_layers: u32,
        num_experts: u32,
    ) PruneCostModel {
        return .{
            .t_fixed_us = t_fixed_us,
            .c_tok_us = c_tok_us,
            .expert_bytes = expert_bytes,
            .cold_penalty_bytes = cold_penalty_bytes,
            .beta_gbs_by_m = @splat(beta_gbs),
            .num_layers = num_layers,
            .num_experts = num_experts,
        };
    }

    pub inline fn isMoe(self: PruneCostModel) bool {
        return self.num_experts > 0 and self.num_layers > 0 and
            (self.expert_bytes > 0.0 or self.expert_bytes_table != null);
    }

    pub inline fn hasPerExpertTable(self: PruneCostModel) bool {
        return self.expert_bytes_table != null or self.expert_bw_gbs_table != null;
    }
};

pub const PruneError = error{
    EmptyTree,
    InvalidCap,
    InvalidCostModel,
    TooManyLayers,
    MissingExpertBitsets,
    BufferTooSmall,
};

/// Evaluation snapshot of a tree under a `PruneCostModel`.
pub const TreeScore = struct {
    node_count: u16,
    predicted_time_us: f32,
    expected_tokens: f32,
    tokens_per_us: f32,
    avg_unique_experts_per_layer: f32,
};

/// Generates a compile-time-bounded speculative draft prefix trie (`1 <= max_nodes <= 64`,
/// ADR-0014).
pub fn DraftTree(comptime max_nodes: u16) type {
    if (max_nodes == 0 or max_nodes > MAX_SUPPORTED_NODES) {
        @compileError("DraftTree max_nodes must be in 1..=64");
    }

    return struct {
        const Self = @This();

        pub const capacity: u16 = max_nodes;

        pub const PruneSummary = struct {
            kept_nodes: u16,
            predicted_time_us: f32,
            expected_tokens: f32,
            avg_unique_experts_per_layer: f32,
            /// Maps each compacted post-prune node index `new_i in 0..kept_nodes`
            /// back to its pre-prune node index (`NO_NODE` for `>= kept_nodes`).
            remapped_old_indices: [max_nodes]u16,
        };

        nodes: [max_nodes]Node,
        count: u16,
        max_depth: u16,

        /// Initializes a tree containing only root node `0` with `root_token`.
        pub fn init(root_token: u32) Self {
            var self: Self = undefined;
            self.reset(root_token);
            return self;
        }

        /// Resets the tree in-place to a single root node `0`.
        pub fn reset(self: *Self, root_token: u32) void {
            self.nodes[0] = .{
                .token_id = root_token,
                .parent = NO_NODE,
                .depth = 0,
                .ancestor_mask = 1,
                .step_prob = 1.0,
                .cum_prob = 1.0,
                .first_child = NO_NODE,
                .next_sibling = NO_NODE,
                .source = .root,
            };
            self.count = 1;
            self.max_depth = 0;
        }

        pub fn activeSlice(self: *const Self) []const Node {
            return self.nodes[0..self.count];
        }

        /// Looks up a direct child of `parent` matching `token_id`.
        pub fn findChild(self: *const Self, parent: u16, token_id: u32) ?u16 {
            if (parent >= self.count) return null;
            var c = self.nodes[parent].first_child;
            while (c != NO_NODE) : (c = self.nodes[c].next_sibling) {
                if (self.nodes[c].token_id == token_id) return c;
            }
            return null;
        }

        /// Inserts `(token_id, step_prob, source)` as a child of `parent`, or
        /// deduplicates in place if `parent` already has a child with `token_id`
        /// (ADR-0005).
        ///
        /// On deduplication:
        /// - `step_prob` is raised to `@max(existing, step_prob)` and `cum_prob`
        ///   is propagated across the node's entire descendant subtree.
        /// - `source` is upgraded to `.l1_seed` when the incoming source is `.l1_seed`.
        ///
        /// Returns the child's node index, or `null` if `parent >= count` or the
        /// tree is full (`count == max_nodes`) for a new node.
        pub fn insertChild(
            self: *Self,
            parent: u16,
            token_id: u32,
            step_prob: f32,
            source: NodeSource,
        ) ?u16 {
            if (parent >= self.count) return null;
            const prob: f32 = if (std.math.isNan(step_prob)) 0.0 else std.math.clamp(step_prob, 0.0, 1.0);

            if (self.findChild(parent, token_id)) |existing| {
                if (source == .l1_seed) {
                    self.nodes[existing].source = .l1_seed;
                }
                if (prob > self.nodes[existing].step_prob) {
                    self.nodes[existing].step_prob = prob;
                    self.propagateSubtreeCumProb(existing);
                }
                return existing;
            }

            if (self.count >= max_nodes) return null;

            const new_idx: u16 = self.count;
            self.count += 1;

            const p_node = self.nodes[parent];
            const new_depth: u16 = p_node.depth + 1;
            if (new_depth > self.max_depth) self.max_depth = new_depth;

            const bit: u64 = @as(u64, 1) << @as(u6, @intCast(new_idx));
            self.nodes[new_idx] = .{
                .token_id = token_id,
                .parent = parent,
                .depth = new_depth,
                .ancestor_mask = p_node.ancestor_mask | bit,
                .step_prob = prob,
                .cum_prob = p_node.cum_prob * prob,
                .first_child = NO_NODE,
                .next_sibling = NO_NODE,
                .source = source,
            };

            // Append at tail of parent's child list to preserve deterministic candidate order.
            if (self.nodes[parent].first_child == NO_NODE) {
                self.nodes[parent].first_child = new_idx;
            } else {
                var tail = self.nodes[parent].first_child;
                while (self.nodes[tail].next_sibling != NO_NODE) {
                    tail = self.nodes[tail].next_sibling;
                }
                self.nodes[tail].next_sibling = new_idx;
            }

            return new_idx;
        }

        /// Recomputes `cum_prob` for `root_idx` and all of its descendants in
        /// topological order (`parent < child`), with zero recursion or heap allocation.
        fn propagateSubtreeCumProb(self: *Self, root_idx: u16) void {
            if (root_idx == 0) {
                self.nodes[0].cum_prob = 1.0;
            } else {
                const p = self.nodes[root_idx].parent;
                self.nodes[root_idx].cum_prob = self.nodes[p].cum_prob * self.nodes[root_idx].step_prob;
            }
            const root_bit: u64 = @as(u64, 1) << @as(u6, @intCast(root_idx));
            var idx: u16 = root_idx + 1;
            while (idx < self.count) : (idx += 1) {
                if ((self.nodes[idx].ancestor_mask & root_bit) != 0) {
                    const p = self.nodes[idx].parent;
                    self.nodes[idx].cum_prob = self.nodes[p].cum_prob * self.nodes[idx].step_prob;
                }
            }
        }

        /// Attaches an L1 suffix automaton continuation chain starting from root `0`
        /// (ADR-0005):
        /// - When `match_len >= 4`: nodes are tagged `.l1_seed` with `seed_step_prob`
        ///   and `seed_tail_node` returns the chain's last node so L2 can continue
        ///   expanding from it as well as from root `0`.
        /// - When `1 <= match_len < 4`: nodes are tagged `.l1_branch` with
        ///   `branch_step_prob` and `seed_tail_node` is `0`.
        pub fn insertL1Continuation(
            self: *Self,
            match_len: u32,
            draft_tokens: []const u32,
            seed_step_prob: f32,
            branch_step_prob: f32,
        ) InsertL1Result {
            if (match_len == 0 or draft_tokens.len == 0) {
                return .{
                    .inserted_count = 0,
                    .chain_len = 0,
                    .seed_tail_node = 0,
                    .is_seed = false,
                };
            }

            const is_seed = match_len >= SEED_MIN_MATCH_LEN;
            const source: NodeSource = if (is_seed) .l1_seed else .l1_branch;
            const prob: f32 = if (is_seed) seed_step_prob else branch_step_prob;

            var cur: u16 = 0;
            var inserted: u16 = 0;
            var chain_len: u16 = 0;
            for (draft_tokens) |tok| {
                const prev_count = self.count;
                const next = self.insertChild(cur, tok, prob, source) orelse break;
                if (self.count > prev_count) inserted += 1;
                chain_len += 1;
                cur = next;
            }

            return .{
                .inserted_count = inserted,
                .chain_len = chain_len,
                .seed_tail_node = if (is_seed and chain_len > 0) cur else 0,
                .is_seed = is_seed and chain_len > 0,
            };
        }

        /// Expected tokens produced by this tree (`1.0 + sum_{v != 0} cum_prob[v]`).
        pub fn expectedTokens(self: *const Self) f32 {
            var v: f64 = 1.0;
            var i: u16 = 1;
            while (i < self.count) : (i += 1) {
                v += @as(f64, self.nodes[i].cum_prob);
            }
            return @floatCast(v);
        }

        /// Writes `base_pos + depth[i]` for `i in 0..count` for GPU tree RoPE (G5).
        pub fn fillPositions(self: *const Self, base_pos: u32, out_positions: []u32) void {
            std.debug.assert(out_positions.len >= self.count);
            for (0..self.count) |i| {
                out_positions[i] = base_pos + self.nodes[i].depth;
            }
        }

        /// Writes `ancestor_mask[i]` for `i in 0..count` for GPU tree attention (G5).
        pub fn fillAncestorMasks(self: *const Self, out_masks: []u64) void {
            std.debug.assert(out_masks.len >= self.count);
            for (0..self.count) |i| {
                out_masks[i] = self.nodes[i].ancestor_mask;
            }
        }

        /// Writes `token_id[i]` for `i in 0..count` for GPU tree embedding lookup.
        pub fn fillTokenIds(self: *const Self, out_tokens: []u32) void {
            std.debug.assert(out_tokens.len >= self.count);
            for (0..self.count) |i| {
                out_tokens[i] = self.nodes[i].token_id;
            }
        }

        /// Extracts the highest-probability greedy chain starting from root `0`
        /// down to a leaf for `D8` PEARL speculative round-`r+1` pre-drafting
        /// (ADR-0008). Writes `[n_1, ..., n_d]` into `out_nodes` (if non-null)
        /// and their tokens `[tok_1, ..., tok_d]` into `out_tokens`, returning
        /// path length `d` (`0` when only root `0` is present). Zero heap alloc.
        pub fn highestProbPath(
            self: *const Self,
            out_nodes: ?[]u16,
            out_tokens: []u32,
        ) u16 {
            if (self.count <= 1 or out_tokens.len == 0) return 0;
            const max_len: usize = if (out_nodes) |ns| @min(ns.len, out_tokens.len) else out_tokens.len;
            var cur: u16 = 0;
            var len: usize = 0;
            while (len < max_len) {
                var c = self.nodes[cur].first_child;
                if (c == NO_NODE) break;
                var best_c: u16 = c;
                var best_prob: f32 = self.nodes[c].step_prob;
                var best_is_seed: bool = self.nodes[c].source == .l1_seed;
                c = self.nodes[c].next_sibling;
                while (c != NO_NODE) : (c = self.nodes[c].next_sibling) {
                    const p = self.nodes[c].step_prob;
                    const is_seed = self.nodes[c].source == .l1_seed;
                    const better = (p > best_prob) or
                        (p == best_prob and is_seed and !best_is_seed);
                    if (better) {
                        best_c = c;
                        best_prob = p;
                        best_is_seed = is_seed;
                    }
                }
                if (out_nodes) |ns| ns[len] = best_c;
                out_tokens[len] = self.nodes[best_c].token_id;
                len += 1;
                cur = best_c;
            }
            return @intCast(len);
        }

        /// Copies `src` into `self` while strictly enforcing `max_depth_limit`
        /// and `max_nodes_limit` (preserving topological order, ancestor masks,
        /// probabilities, and node provenance). Used by `D8` PEARL when adopting
        /// a pre-drafted tree into a round with a tighter remaining token budget.
        pub fn copyTruncatedFrom(
            self: *Self,
            src: *const Self,
            max_depth_limit: u16,
            max_nodes_limit: u16,
        ) void {
            std.debug.assert(src.count >= 1);
            self.reset(src.nodes[0].token_id);
            const cap = @min(max_nodes_limit, max_nodes);
            if (cap <= 1 or max_depth_limit == 0 or src.count <= 1) return;

            var old_to_new: [max_nodes]u16 = @splat(NO_NODE);
            old_to_new[0] = 0;
            var i: u16 = 1;
            while (i < src.count and self.count < cap) : (i += 1) {
                const n = src.nodes[i];
                if (n.depth > max_depth_limit) continue;
                const new_parent = old_to_new[n.parent];
                if (new_parent == NO_NODE) continue;
                if (self.insertChild(new_parent, n.token_id, n.step_prob, n.source)) |ni| {
                    old_to_new[i] = ni;
                }
            }
        }

        /// Evaluates the current tree's expected tokens, predicted verification time,
        /// throughput score `V(T)/t(T)`, and unique experts/layer under `cost`
        /// without mutating the tree.
        pub fn scoreCurrentTree(
            self: *const Self,
            cost: PruneCostModel,
            node_experts: ?[]const ExpertBitset,
            cold_experts_by_layer: ?[]const ExpertBitset,
        ) PruneError!TreeScore {
            if (self.count == 0) return error.EmptyTree;
            try validateCostModel(cost, self.count, self.count, node_experts, cold_experts_by_layer);

            const use_moe = cost.isMoe();
            var total_expert_bytes: f64 = 0.0;
            var total_direct_us: f64 = 0.0;
            var unique_sum: u64 = 0;

            if (use_moe) {
                const n_layers: usize = cost.num_layers;
                const experts = node_experts orelse return error.MissingExpertBitsets;

                for (0..n_layers) |l| {
                    var u = ExpertBitset.empty();
                    for (0..self.count) |v| {
                        u.unionInPlace(experts[v * n_layers + l]);
                    }
                    const cold_mask: ?ExpertBitset = if (cold_experts_by_layer) |cm| cm[l] else null;
                    const delta = accumulateLayerExperts(cost, l, u, cold_mask);
                    unique_sum += delta.unique_count;
                    total_expert_bytes += delta.scaled_bytes;
                    total_direct_us += delta.direct_time_us;
                }
            }

            const t_us = evalTimeUs(cost, self.count, use_moe, total_expert_bytes, total_direct_us);
            if (!(t_us > 0.0) or !std.math.isFinite(t_us)) return error.InvalidCostModel;
            const v_toks: f64 = @as(f64, self.expectedTokens());
            const avg_unique: f32 = if (use_moe)
                @floatCast(@as(f64, @floatFromInt(unique_sum)) / @as(f64, @floatFromInt(cost.num_layers)))
            else
                0.0;

            return .{
                .node_count = self.count,
                .predicted_time_us = @floatCast(t_us),
                .expected_tokens = @floatCast(v_toks),
                .tokens_per_us = @floatCast(v_toks / t_us),
                .avg_unique_experts_per_layer = avg_unique,
            };
        }

        /// Greedily selects the ancestor-closed subtree `T` (containing root `0`,
        /// `|T| <= hard_cap`) that maximizes `V(T) / t(T)` using exact per-layer
        /// marginal expert-union byte accounting (ADR-0009, ADR-0017), then compacts
        /// the surviving nodes into topological order `0..kept_nodes` and refreshes
        /// `parent`, `first_child`, `next_sibling`, and `ancestor_mask`.
        ///
        /// Performs zero dynamic heap allocation.
        pub fn pruneToOptimalSubtree(
            self: *Self,
            cost: PruneCostModel,
            node_experts: ?[]const ExpertBitset,
            cold_experts_by_layer: ?[]const ExpertBitset,
            hard_cap: u16,
        ) PruneError!PruneSummary {
            if (self.count == 0) return error.EmptyTree;
            if (hard_cap == 0) return error.InvalidCap;

            const eff_cap: u16 = @min(@min(hard_cap, max_nodes), self.count);
            try validateCostModel(cost, self.count, eff_cap, node_experts, cold_experts_by_layer);

            const use_moe = cost.isMoe();
            const use_table = use_moe and cost.hasPerExpertTable();
            const n_layers: usize = if (use_moe) cost.num_layers else 0;
            const has_cold = use_moe and cold_experts_by_layer != null and cost.cold_penalty_bytes > 0.0;
            const eb: f64 = @as(f64, cost.expert_bytes);
            const cpb: f64 = @as(f64, cost.cold_penalty_bytes);

            var union_by_layer: [MAX_PRUNE_LAYERS]ExpertBitset = undefined;
            var cur_expert_bytes: f64 = 0.0;
            var cur_direct_us: f64 = 0.0;
            var cur_unique_sum: u64 = 0;

            if (use_moe) {
                const experts = node_experts orelse return error.MissingExpertBitsets;
                for (0..n_layers) |l| {
                    const b = experts[l]; // root node 0
                    union_by_layer[l] = b;
                    const cold_mask: ?ExpertBitset = if (cold_experts_by_layer) |cm| cm[l] else null;
                    const delta = accumulateLayerExperts(cost, l, b, cold_mask);
                    cur_unique_sum += delta.unique_count;
                    cur_expert_bytes += delta.scaled_bytes;
                    cur_direct_us += delta.direct_time_us;
                }
            }

            var selected_mask: u64 = 1; // Root node 0 is always selected.
            var selected_count: u16 = 1;
            var cur_value: f64 = 1.0; // V({0}) = 1.0 (bonus token)
            var cur_time: f64 = evalTimeUs(cost, 1, use_moe, cur_expert_bytes, cur_direct_us);
            if (!(cur_time > 0.0) or !std.math.isFinite(cur_time)) return error.InvalidCostModel;
            var cur_score: f64 = cur_value / cur_time;

            while (selected_count < eff_cap) {
                var best_node: u16 = NO_NODE;
                var best_score: f64 = cur_score;
                var best_value: f64 = cur_value;
                var best_time: f64 = cur_time;
                var best_expert_bytes: f64 = cur_expert_bytes;
                var best_direct_us: f64 = cur_direct_us;
                var best_delta_unique: u64 = 0;
                var best_cum_prob: f32 = -1.0;

                const next_count: u16 = selected_count + 1;

                var v: u16 = 1;
                while (v < self.count) : (v += 1) {
                    const v_bit: u64 = @as(u64, 1) << @as(u6, @intCast(v));
                    if ((selected_mask & v_bit) != 0) continue;

                    const parent = self.nodes[v].parent;
                    const p_bit: u64 = @as(u64, 1) << @as(u6, @intCast(parent));
                    if ((selected_mask & p_bit) == 0) continue;

                    const v_prob = self.nodes[v].cum_prob;
                    if (!(v_prob > 0.0)) continue;

                    var delta_bytes: f64 = 0.0;
                    var delta_direct_us: f64 = 0.0;
                    var delta_unique: u64 = 0;
                    if (use_moe) {
                        const experts = node_experts.?;
                        const base = @as(usize, v) * n_layers;
                        if (use_table) {
                            for (0..n_layers) |l| {
                                const added = experts[base + l].difference(union_by_layer[l]);
                                if (added.words[0] == 0 and added.words[1] == 0 and added.words[2] == 0 and added.words[3] == 0) {
                                    continue;
                                }
                                const cold_mask: ?ExpertBitset = if (cold_experts_by_layer) |cm| cm[l] else null;
                                const d = accumulateLayerExperts(cost, l, added, cold_mask);
                                delta_unique += d.unique_count;
                                delta_bytes += d.scaled_bytes;
                                delta_direct_us += d.direct_time_us;
                            }
                        } else if (has_cold) {
                            const cold_masks = cold_experts_by_layer.?;
                            for (0..n_layers) |l| {
                                const added = experts[base + l].difference(union_by_layer[l]);
                                const tot = added.count();
                                if (tot > 0) {
                                    delta_unique += tot;
                                    const cold_cnt = added.intersectWith(cold_masks[l]).count();
                                    delta_bytes += @as(f64, @floatFromInt(tot)) * eb +
                                        @as(f64, @floatFromInt(cold_cnt)) * cpb;
                                }
                            }
                        } else {
                            for (0..n_layers) |l| {
                                const tot = experts[base + l].differenceCount(union_by_layer[l]);
                                delta_unique += tot;
                            }
                            delta_bytes = @as(f64, @floatFromInt(delta_unique)) * eb;
                        }
                    }

                    const cand_value = cur_value + @as(f64, v_prob);
                    const cand_expert_bytes = cur_expert_bytes + delta_bytes;
                    const cand_direct_us = cur_direct_us + delta_direct_us;
                    const cand_time = evalTimeUs(cost, next_count, use_moe, cand_expert_bytes, cand_direct_us);
                    if (!(cand_time > 0.0)) continue;
                    const cand_score = cand_value / cand_time;

                    // Require strictly positive marginal gain in V(T)/t(T); break exact
                    // score ties by higher cumulative probability, then lower index.
                    const is_better = (cand_score > best_score) or
                        (best_node != NO_NODE and cand_score == best_score and v_prob > best_cum_prob);
                    if (is_better) {
                        best_node = v;
                        best_score = cand_score;
                        best_value = cand_value;
                        best_time = cand_time;
                        best_expert_bytes = cand_expert_bytes;
                        best_direct_us = cand_direct_us;
                        best_delta_unique = delta_unique;
                        best_cum_prob = v_prob;
                    }
                }

                if (best_node == NO_NODE) break;

                selected_mask |= (@as(u64, 1) << @as(u6, @intCast(best_node)));
                selected_count = next_count;
                cur_value = best_value;
                cur_time = best_time;
                cur_score = best_score;
                cur_expert_bytes = best_expert_bytes;
                cur_direct_us = best_direct_us;
                cur_unique_sum += best_delta_unique;

                if (use_moe) {
                    const experts = node_experts.?;
                    const base = @as(usize, best_node) * n_layers;
                    for (0..n_layers) |l| {
                        union_by_layer[l].unionInPlace(experts[base + l]);
                    }
                }
            }

            // Compact surviving nodes in topological order (`old_idx` ascending
            // preserves `parent < child` since `self.nodes` is already topological).
            var new_nodes: [max_nodes]Node = undefined;
            var old_to_new: [max_nodes]u16 = @splat(NO_NODE);
            var remapped: [max_nodes]u16 = @splat(NO_NODE);
            var new_count: u16 = 0;
            var new_max_depth: u16 = 0;

            var old_idx: u16 = 0;
            while (old_idx < self.count) : (old_idx += 1) {
                const bit: u64 = @as(u64, 1) << @as(u6, @intCast(old_idx));
                if ((selected_mask & bit) == 0) continue;

                const old = self.nodes[old_idx];
                const new_idx: u16 = new_count;
                new_count += 1;

                old_to_new[old_idx] = new_idx;
                remapped[new_idx] = old_idx;

                const new_parent: u16 = if (old_idx == 0) NO_NODE else old_to_new[old.parent];
                std.debug.assert(old_idx == 0 or new_parent < new_idx);

                const new_bit: u64 = @as(u64, 1) << @as(u6, @intCast(new_idx));
                const new_mask: u64 = if (old_idx == 0) 1 else (new_nodes[new_parent].ancestor_mask | new_bit);
                if (old.depth > new_max_depth) new_max_depth = old.depth;

                new_nodes[new_idx] = .{
                    .token_id = old.token_id,
                    .parent = new_parent,
                    .depth = old.depth,
                    .ancestor_mask = new_mask,
                    .step_prob = old.step_prob,
                    .cum_prob = old.cum_prob,
                    .first_child = NO_NODE,
                    .next_sibling = NO_NODE,
                    .source = old.source,
                };

                if (new_parent != NO_NODE) {
                    if (new_nodes[new_parent].first_child == NO_NODE) {
                        new_nodes[new_parent].first_child = new_idx;
                    } else {
                        var tail = new_nodes[new_parent].first_child;
                        while (new_nodes[tail].next_sibling != NO_NODE) {
                            tail = new_nodes[tail].next_sibling;
                        }
                        new_nodes[tail].next_sibling = new_idx;
                    }
                }
            }

            std.debug.assert(new_count == selected_count);
            @memcpy(self.nodes[0..new_count], new_nodes[0..new_count]);
            self.count = new_count;
            self.max_depth = new_max_depth;

            const avg_unique: f32 = if (use_moe)
                @floatCast(@as(f64, @floatFromInt(cur_unique_sum)) / @as(f64, @floatFromInt(cost.num_layers)))
            else
                0.0;

            return .{
                .kept_nodes = new_count,
                .predicted_time_us = @floatCast(cur_time),
                .expected_tokens = @floatCast(cur_value),
                .avg_unique_experts_per_layer = avg_unique,
                .remapped_old_indices = remapped,
            };
        }
    };
}

const ExpertCostDelta = struct {
    unique_count: u64 = 0,
    scaled_bytes: f64 = 0.0,
    direct_time_us: f64 = 0.0,
};

fn accumulateLayerExperts(
    cost: PruneCostModel,
    layer: usize,
    added: ExpertBitset,
    cold_mask: ?ExpertBitset,
) ExpertCostDelta {
    if (!cost.hasPerExpertTable()) {
        const tot = added.count();
        if (tot == 0) return .{};
        var bytes: f64 = @as(f64, @floatFromInt(tot)) * @as(f64, cost.expert_bytes);
        if (cold_mask) |cm| {
            if (cost.cold_penalty_bytes > 0.0) {
                const cold_cnt = added.intersectWith(cm).count();
                bytes += @as(f64, @floatFromInt(cold_cnt)) * @as(f64, cost.cold_penalty_bytes);
            }
        }
        return .{ .unique_count = tot, .scaled_bytes = bytes, .direct_time_us = 0.0 };
    }

    var delta: ExpertCostDelta = .{};
    const n_exp: usize = cost.num_experts;
    const layer_off: usize = layer * n_exp;
    const scale: f64 = @as(f64, cost.expert_table_scale);
    const cpb: f64 = @as(f64, cost.cold_penalty_bytes);
    const has_cold = cold_mask != null and cost.cold_penalty_bytes > 0.0;

    inline for (0..4) |w_idx| {
        var w = added.words[w_idx];
        while (w != 0) {
            const bit: usize = @ctz(w);
            w &= w - 1;
            const e: usize = w_idx * 64 + bit;
            if (e >= n_exp) continue;
            delta.unique_count += 1;
            const eb: f64 = if (cost.expert_bytes_table) |tbl|
                @as(f64, @floatFromInt(tbl[layer_off + e])) * scale
            else
                @as(f64, cost.expert_bytes);
            if (cost.expert_bw_gbs_table) |bw_tbl| {
                delta.direct_time_us += eb / (@as(f64, bw_tbl[layer_off + e]) * 1000.0);
            } else {
                delta.scaled_bytes += eb;
            }
            if (has_cold and cold_mask.?.contains(@intCast(e))) {
                delta.scaled_bytes += cpb;
            }
        }
    }
    return delta;
}

fn validateCostModel(
    cost: PruneCostModel,
    node_count: u16,
    max_m: u16,
    node_experts: ?[]const ExpertBitset,
    cold_experts_by_layer: ?[]const ExpertBitset,
) PruneError!void {
    if (!std.math.isFinite(cost.t_fixed_us) or cost.t_fixed_us < 0.0) return error.InvalidCostModel;
    if (!std.math.isFinite(cost.c_tok_us) or cost.c_tok_us < 0.0) return error.InvalidCostModel;
    if (!std.math.isFinite(cost.expert_bytes) or cost.expert_bytes < 0.0) return error.InvalidCostModel;
    if (!std.math.isFinite(cost.cold_penalty_bytes) or cost.cold_penalty_bytes < 0.0) return error.InvalidCostModel;
    if (!std.math.isFinite(cost.expert_table_scale) or !(cost.expert_table_scale > 0.0)) return error.InvalidCostModel;

    if (cost.isMoe()) {
        if (cost.num_layers > MAX_PRUNE_LAYERS) return error.TooManyLayers;
        const experts = node_experts orelse return error.MissingExpertBitsets;
        const needed = @as(usize, node_count) * @as(usize, cost.num_layers);
        if (experts.len < needed) return error.BufferTooSmall;
        if (cold_experts_by_layer) |cold| {
            if (cold.len < cost.num_layers) return error.BufferTooSmall;
        }
        if (cost.hasPerExpertTable()) {
            if (cost.num_experts > ExpertBitset.MAX_EXPERTS) return error.InvalidCostModel;
            const tbl_needed = @as(usize, cost.num_layers) * @as(usize, cost.num_experts);
            if (cost.expert_bytes_table) |tbl| {
                if (tbl.len < tbl_needed) return error.BufferTooSmall;
            }
            if (cost.expert_bw_gbs_table) |bw_tbl| {
                if (bw_tbl.len < tbl_needed) return error.BufferTooSmall;
                for (bw_tbl[0..tbl_needed]) |b| {
                    if (!std.math.isFinite(b) or !(b > 0.0)) return error.InvalidCostModel;
                }
            }
        }
        var m: u16 = 1;
        while (m <= max_m) : (m += 1) {
            const b = cost.beta_gbs_by_m[m];
            if (!std.math.isFinite(b) or !(b > 0.0)) return error.InvalidCostModel;
        }
    } else {
        if (cost.t_fixed_us == 0.0 and cost.c_tok_us == 0.0) return error.InvalidCostModel;
    }
}

inline fn evalTimeUs(
    cost: PruneCostModel,
    node_count: u16,
    use_moe: bool,
    expert_bytes_total: f64,
    expert_direct_us: f64,
) f64 {
    const base = @as(f64, cost.t_fixed_us) + @as(f64, @floatFromInt(node_count)) * @as(f64, cost.c_tok_us);
    if (!use_moe) return base;
    const bytes_per_us = @as(f64, cost.beta_gbs_by_m[node_count]) * 1000.0;
    return base + expert_direct_us + expert_bytes_total / bytes_per_us;
}

/// Compile-time specialized tree aliases matching GPU `TREE_MAX_NODES` variants (ADR-0014).
pub const SmallTree = DraftTree(16);
pub const DefaultTree = DraftTree(32);
pub const MaxTree = DraftTree(64);

test "DraftTree builds ancestor masks, deduplicates dual-source nodes, and propagates cum_prob" {
    var tree = DefaultTree.init(100);
    try std.testing.expectEqual(@as(u16, 1), tree.count);
    try std.testing.expectEqual(@as(u16, 0), tree.max_depth);
    try std.testing.expectEqual(@as(u64, 1), tree.nodes[0].ancestor_mask);
    try std.testing.expectEqual(@as(f32, 1.0), tree.nodes[0].cum_prob);
    try std.testing.expectEqual(NodeSource.root, tree.nodes[0].source);

    // Insert an L2 chain: root(0) -> 10(1, p=0.4) -> 20(2, p=0.5) -> 30(3, p=0.8)
    const n1 = tree.insertChild(0, 10, 0.4, .l2) orelse return error.TestUnexpectedResult;
    const n2 = tree.insertChild(n1, 20, 0.5, .l2) orelse return error.TestUnexpectedResult;
    const n3 = tree.insertChild(n2, 30, 0.8, .l2) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u16, 1), n1);
    try std.testing.expectEqual(@as(u16, 2), n2);
    try std.testing.expectEqual(@as(u16, 3), n3);
    try std.testing.expectEqual(@as(u64, 0b0011), tree.nodes[n1].ancestor_mask);
    try std.testing.expectEqual(@as(u64, 0b0111), tree.nodes[n2].ancestor_mask);
    try std.testing.expectEqual(@as(u64, 0b1111), tree.nodes[n3].ancestor_mask);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4 * 0.5 * 0.8), tree.nodes[n3].cum_prob, 1e-6);

    // Merge an L1 seed continuation `[10, 20, 40]` with higher step_prob = 0.9:
    // nodes 1 and 2 must deduplicate in place, upgrade to `.l1_seed`, raise
    // `step_prob` to 0.9, propagate `cum_prob` to descendant node 3 (0.9 * 0.9 * 0.8),
    // and attach only ONE new node (40 at index 4)!
    const l1 = tree.insertL1Continuation(5, &[_]u32{ 10, 20, 40 }, 0.9, 0.3);
    try std.testing.expect(l1.is_seed);
    try std.testing.expectEqual(@as(u16, 1), l1.inserted_count);
    try std.testing.expectEqual(@as(u16, 3), l1.chain_len);
    try std.testing.expectEqual(@as(u16, 4), l1.seed_tail_node);
    try std.testing.expectEqual(@as(u16, 5), tree.count);
    try std.testing.expectEqual(NodeSource.l1_seed, tree.nodes[n1].source);
    try std.testing.expectEqual(NodeSource.l1_seed, tree.nodes[n2].source);
    try std.testing.expectEqual(NodeSource.l2, tree.nodes[n3].source);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), tree.nodes[n1].cum_prob, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.81), tree.nodes[n2].cum_prob, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.81 * 0.8), tree.nodes[n3].cum_prob, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.81 * 0.9), tree.nodes[4].cum_prob, 1e-6);
    try std.testing.expectEqual(@as(u64, 0b10111), tree.nodes[4].ancestor_mask);

    // Short L1 match (< 4) inserts as `.l1_branch` and returns `seed_tail_node = 0`.
    const short_l1 = tree.insertL1Continuation(3, &[_]u32{ 99, 98 }, 0.9, 0.25);
    try std.testing.expect(!short_l1.is_seed);
    try std.testing.expectEqual(@as(u16, 0), short_l1.seed_tail_node);
    try std.testing.expectEqual(@as(u16, 2), short_l1.inserted_count);
    const b1 = tree.findChild(0, 99) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(NodeSource.l1_branch, tree.nodes[b1].source);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), tree.nodes[b1].step_prob, 1e-6);
}

test "ExpertBitset union, intersection, difference, and popcount across 256 experts" {
    var a = ExpertBitset.empty();
    var b = ExpertBitset.empty();
    a.set(0);
    a.set(63);
    a.set(64);
    a.set(127);
    a.set(255);

    b.set(63);
    b.set(64);
    b.set(192);

    try std.testing.expectEqual(@as(u32, 5), a.count());
    try std.testing.expectEqual(@as(u32, 3), b.count());
    try std.testing.expect(a.contains(255));
    try std.testing.expect(!a.contains(192));
    try std.testing.expectEqual(@as(u32, 3), a.differenceCount(b));
    try std.testing.expectEqual(@as(u32, 1), b.differenceCount(a));
    try std.testing.expectEqual(@as(u32, 6), a.unionWith(b).count());
    try std.testing.expectEqual(@as(u32, 2), a.intersectWith(b).count());

    a.clear(255);
    try std.testing.expect(!a.contains(255));
    try std.testing.expectEqual(@as(u32, 4), a.count());
}

test "pruneToOptimalSubtree falls back cleanly to a node/latency budget for dense models" {
    var tree = DefaultTree.init(1);
    const high1 = tree.insertChild(0, 10, 0.9, .l2) orelse unreachable;
    const high2 = tree.insertChild(high1, 11, 0.85, .l2) orelse unreachable;
    const low1 = tree.insertChild(0, 20, 0.04, .l1_branch) orelse unreachable;
    _ = tree.insertChild(low1, 21, 0.5, .l1_branch) orelse unreachable;

    // Dense model: t_fixed = 100 us, c_tok = 10 us.
    // Root alone: V = 1.0, t = 110 -> 0.009091 tok/us
    // +high1 (0.9): V = 1.9, t = 120 -> 0.015833 tok/us (keep!)
    // +high2 (0.765): V = 2.665, t = 130 -> 0.020500 tok/us (keep!)
    // +low1 (0.04): V = 2.705, t = 140 -> 0.019321 < 0.020500 (prune low1 and its child!)
    const dense_cost = PruneCostModel.dense(100.0, 10.0);
    const summary = try tree.pruneToOptimalSubtree(dense_cost, null, null, 32);

    try std.testing.expectEqual(@as(u16, 3), summary.kept_nodes);
    try std.testing.expectEqual(@as(u16, 3), tree.count);
    try std.testing.expectEqual(@as(f32, 0.0), summary.avg_unique_experts_per_layer);
    try std.testing.expectApproxEqAbs(@as(f32, 130.0), summary.predicted_time_us, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 + 0.9 + 0.765), summary.expected_tokens, 1e-4);

    // Verify remapping and compacted ancestor masks.
    try std.testing.expectEqual(@as(u16, 0), summary.remapped_old_indices[0]);
    try std.testing.expectEqual(high1, summary.remapped_old_indices[1]);
    try std.testing.expectEqual(high2, summary.remapped_old_indices[2]);
    try std.testing.expectEqual(NO_NODE, summary.remapped_old_indices[3]);
    try std.testing.expectEqual(@as(u64, 0b001), tree.nodes[0].ancestor_mask);
    try std.testing.expectEqual(@as(u64, 0b011), tree.nodes[1].ancestor_mask);
    try std.testing.expectEqual(@as(u64, 0b111), tree.nodes[2].ancestor_mask);
    try std.testing.expectEqual(@as(?u16, null), tree.findChild(0, 20));
}

test "pruneToOptimalSubtree prefers shared experts and penalizes disjoint and cold MoE branches" {
    // 2 layers, 128 experts, 1 MB/expert, beta = 100 GB/s (so each new expert costs 10 us).
    const num_layers: u32 = 2;
    var tree = DefaultTree.init(0);

    // Branch A (node 1 -> node 2): moderate probability, shares experts {0, 1} with root!
    const a1 = tree.insertChild(0, 101, 0.45, .l2) orelse unreachable;
    const a2 = tree.insertChild(a1, 102, 0.80, .l2) orelse unreachable;
    // Branch B (node 3): HIGHER single-step probability (0.50 > 0.45), but activates
    // 8 disjoint hot experts per layer (+160 us total!).
    const b1 = tree.insertChild(0, 201, 0.50, .l2) orelse unreachable;
    // Branch C (node 4): moderate probability (0.45), only 1 new expert per layer,
    // but that expert is COLD with a massive NVMe penalty!
    const c1 = tree.insertChild(0, 301, 0.45, .l2) orelse unreachable;

    var experts: [5 * num_layers]ExpertBitset = @splat(ExpertBitset.empty());
    for (0..num_layers) |l| {
        // Root activates experts 0, 1.
        experts[0 * num_layers + l].set(0);
        experts[0 * num_layers + l].set(1);

        // Branch A reuses experts 0, 1 (zero marginal expert bytes!).
        experts[a1 * num_layers + l].set(0);
        experts[a1 * num_layers + l].set(1);
        experts[a2 * num_layers + l].set(0);
        experts[a2 * num_layers + l].set(1);

        // Branch B introduces 8 brand-new experts {10..17} per layer.
        for (10..18) |e| experts[b1 * num_layers + l].set(@intCast(e));

        // Branch C shares expert 0 and adds 1 cold expert {99} per layer.
        experts[c1 * num_layers + l].set(0);
        experts[c1 * num_layers + l].set(99);
    }

    var cold: [num_layers]ExpertBitset = @splat(ExpertBitset.empty());
    for (0..num_layers) |l| cold[l].set(99);

    const moe_cost = PruneCostModel.uniformBeta(
        100.0, // t_fixed_us
        5.0, // c_tok_us
        1_000_000.0, // expert_bytes (1 MB -> 10 us at 100 GB/s)
        50_000_000.0, // cold_penalty_bytes (50 MB equiv -> 500 us NVMe penalty)
        100.0, // beta_gbs
        num_layers,
        128,
    );

    const pre = try tree.scoreCurrentTree(moe_cost, &experts, &cold);
    const summary = try tree.pruneToOptimalSubtree(moe_cost, &experts, &cold, 32);

    // Branch A (nodes 1, 2) has 0 marginal expert bytes and MUST survive;
    // Branch B (16 new hot experts -> +160 us) and Branch C (cold NVMe -> +1020 us) MUST be pruned!
    try std.testing.expectEqual(@as(u16, 3), summary.kept_nodes);
    try std.testing.expectEqual(@as(u16, 3), tree.count);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), summary.avg_unique_experts_per_layer, 1e-6);
    try std.testing.expect(summary.expected_tokens / summary.predicted_time_us > pre.tokens_per_us);

    // Check compacted indices and links: 0 -> 1 (tok 101) -> 2 (tok 102).
    try std.testing.expectEqual(@as(u32, 101), tree.nodes[1].token_id);
    try std.testing.expectEqual(@as(u32, 102), tree.nodes[2].token_id);
    try std.testing.expectEqual(@as(u16, 0), tree.nodes[1].parent);
    try std.testing.expectEqual(@as(u16, 1), tree.nodes[2].parent);
    try std.testing.expectEqual(@as(u64, 0b011), tree.nodes[1].ancestor_mask);
    try std.testing.expectEqual(@as(u64, 0b111), tree.nodes[2].ancestor_mask);
    try std.testing.expectEqual(@as(u16, 1), tree.nodes[0].first_child);
    try std.testing.expectEqual(NO_NODE, tree.nodes[1].next_sibling);
    try std.testing.expectEqual(@as(u16, 2), tree.nodes[1].first_child);
    try std.testing.expectEqual(NO_NODE, tree.nodes[2].first_child);
    try std.testing.expectEqual(@as(u16, 2), tree.max_depth);
}

test "pruneToOptimalSubtree compacts non-contiguous surviving branches and preserves hard_cap" {
    var tree = MaxTree.init(0);
    // Insert a pruned decoy at index 1, then a high-value branch at index 2 -> 3 -> 4,
    // and another high-value child of root at index 5.
    _ = tree.insertChild(0, 10, 0.001, .l1_branch) orelse unreachable; // idx 1 (pruned)
    const keep_a = tree.insertChild(0, 20, 0.9, .l2) orelse unreachable; // idx 2
    const keep_a2 = tree.insertChild(keep_a, 21, 0.9, .l2) orelse unreachable; // idx 3
    _ = tree.insertChild(keep_a2, 22, 0.9, .l2) orelse unreachable; // idx 4
    const keep_b = tree.insertChild(0, 30, 0.85, .l2) orelse unreachable; // idx 5

    const dense_cost = PruneCostModel.dense(100.0, 2.0);
    // Hard cap of 4 nodes: root(0) + keep_a(2, 0.9) + keep_b(5, 0.85) + keep_a2(3, 0.81).
    // Note: keep_a2 has cum_prob 0.81 < keep_b (0.85), so greedy picks keep_a, keep_b, keep_a2!
    const summary = try tree.pruneToOptimalSubtree(dense_cost, null, null, 4);
    try std.testing.expectEqual(@as(u16, 4), summary.kept_nodes);
    try std.testing.expectEqual(@as(u16, 4), tree.count);

    // Topological compaction order of surviving old indices {0, 2, 3, 5}:
    // new 0 = old 0 (root)
    // new 1 = old 2 (tok 20, parent 0)
    // new 2 = old 3 (tok 21, parent 1)
    // new 3 = old 5 (tok 30, parent 0)
    try std.testing.expectEqualSlices(u16, &[_]u16{ 0, keep_a, keep_a2, keep_b }, summary.remapped_old_indices[0..4]);
    try std.testing.expectEqual(@as(u64, 0b0001), tree.nodes[0].ancestor_mask);
    try std.testing.expectEqual(@as(u64, 0b0011), tree.nodes[1].ancestor_mask);
    try std.testing.expectEqual(@as(u64, 0b0111), tree.nodes[2].ancestor_mask);
    try std.testing.expectEqual(@as(u64, 0b1001), tree.nodes[3].ancestor_mask);
    try std.testing.expectEqual(@as(?u16, 1), tree.findChild(0, 20));
    try std.testing.expectEqual(@as(?u16, 3), tree.findChild(0, 30));
    try std.testing.expectEqual(@as(?u16, 2), tree.findChild(1, 21));
    try std.testing.expectEqual(@as(?u16, null), tree.findChild(0, 10));
    try std.testing.expectEqual(@as(?u16, null), tree.findChild(2, 22));
    try std.testing.expectEqual(@as(u16, 2), tree.max_depth);

    // Verify GPU-ready flat array export helpers.
    var pos: [4]u32 = undefined;
    var masks: [4]u64 = undefined;
    var toks: [4]u32 = undefined;
    tree.fillPositions(512, &pos);
    tree.fillAncestorMasks(&masks);
    tree.fillTokenIds(&toks);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 512, 513, 514, 513 }, &pos);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 0b0001, 0b0011, 0b0111, 0b1001 }, &masks);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 0, 20, 21, 30 }, &toks);
}

test "pruneToOptimalSubtree uses M-dependent beta_gbs_by_m and validates cost model inputs" {
    var tree = SmallTree.init(0);
    const n1 = tree.insertChild(0, 1, 0.7, .l2) orelse unreachable;
    _ = tree.insertChild(n1, 2, 0.6, .l2) orelse unreachable;

    var experts: [3]ExpertBitset = @splat(ExpertBitset.empty());
    experts[0].set(0);
    experts[1].set(0);
    experts[2].set(0);

    var cost = PruneCostModel.uniformBeta(10.0, 2.0, 1_000_000.0, 0.0, 50.0, 1, 8);
    // Bandwidth climbs with tree size M (GEMV -> GEMM regime):
    cost.beta_gbs_by_m[1] = 50.0; // 1 MB / (50 * 1000) = 20 us
    cost.beta_gbs_by_m[2] = 100.0; // 1 MB / (100 * 1000) = 10 us
    cost.beta_gbs_by_m[3] = 200.0; // 1 MB / (200 * 1000) = 5 us

    const summary = try tree.pruneToOptimalSubtree(cost, &experts, null, 16);
    try std.testing.expectEqual(@as(u16, 3), summary.kept_nodes);
    // Exact predicted time at M=3: 10 + 3*2 + 1_000_000 / (200 * 1000) = 21.0 us.
    try std.testing.expectApproxEqAbs(@as(f32, 21.0), summary.predicted_time_us, 1e-4);

    // Error validation paths.
    try std.testing.expectError(error.InvalidCap, tree.pruneToOptimalSubtree(cost, &experts, null, 0));
    try std.testing.expectError(error.MissingExpertBitsets, tree.pruneToOptimalSubtree(cost, null, null, 16));
    try std.testing.expectError(error.BufferTooSmall, tree.pruneToOptimalSubtree(cost, experts[0..2], null, 16));

    var bad_beta = cost;
    bad_beta.beta_gbs_by_m[2] = 0.0;
    try std.testing.expectError(error.InvalidCostModel, tree.pruneToOptimalSubtree(bad_beta, &experts, null, 16));

    var too_many_layers = cost;
    too_many_layers.num_layers = MAX_PRUNE_LAYERS + 1;
    try std.testing.expectError(error.TooManyLayers, tree.pruneToOptimalSubtree(too_many_layers, &experts, null, 16));
}

test "M5 D7 PruneCostModel supports ADR-0017 per-expert byte and bandwidth tables" {
    const num_layers: u32 = 2;
    const num_experts: u32 = 8;

    // 1. Uniform table must match uniform scalar identically.
    var t_scalar = SmallTree.init(0);
    const s1 = t_scalar.insertChild(0, 10, 0.75, .l2) orelse unreachable;
    _ = t_scalar.insertChild(s1, 11, 0.70, .l2) orelse unreachable;
    var t_table = t_scalar;

    var exp_uniform: [3 * num_layers]ExpertBitset = @splat(ExpertBitset.empty());
    for (0..num_layers) |l| {
        exp_uniform[0 * num_layers + l].set(0);
        exp_uniform[1 * num_layers + l].set(1);
        exp_uniform[2 * num_layers + l].set(2);
    }

    const uniform_bytes: [num_layers * num_experts]u32 = @splat(1_000_000);
    const uniform_bw: [num_layers * num_experts]f32 = @splat(80.0);

    const scalar_cost = PruneCostModel.uniformBeta(50.0, 4.0, 1_000_000.0, 0.0, 80.0, num_layers, num_experts);
    var table_cost = scalar_cost;
    table_cost.expert_bytes = 0.0; // Verify `isMoe()` works solely from `expert_bytes_table`.
    table_cost.expert_bytes_table = &uniform_bytes;
    table_cost.expert_bw_gbs_table = &uniform_bw;

    const sum_scalar = try t_scalar.pruneToOptimalSubtree(scalar_cost, &exp_uniform, null, 16);
    const sum_table = try t_table.pruneToOptimalSubtree(table_cost, &exp_uniform, null, 16);
    try std.testing.expectEqual(sum_scalar.kept_nodes, sum_table.kept_nodes);
    try std.testing.expectApproxEqAbs(sum_scalar.predicted_time_us, sum_table.predicted_time_us, 1e-4);
    try std.testing.expectApproxEqAbs(sum_scalar.expected_tokens, sum_table.expected_tokens, 1e-5);

    // 2. Heterogeneous GSQ+RCO: expert 1 is cheap q2_affine (400 KB @ 90 GB/s),
    //    while expert 2 is heavy bf16 (6 MB @ 60 GB/s). Given equal probabilities,
    //    D7 must keep the cheap q2 branch and prune the heavy bf16 branch!
    var t_het = SmallTree.init(0);
    const cheap_branch = t_het.insertChild(0, 100, 0.55, .l2) orelse unreachable;
    const heavy_branch = t_het.insertChild(0, 200, 0.55, .l2) orelse unreachable;

    var exp_het: [3 * num_layers]ExpertBitset = @splat(ExpertBitset.empty());
    for (0..num_layers) |l| {
        exp_het[0 * num_layers + l].set(0);
        exp_het[cheap_branch * num_layers + l].set(1);
        exp_het[heavy_branch * num_layers + l].set(2);
    }

    var het_bytes: [num_layers * num_experts]u32 = @splat(1_000_000);
    var het_bw: [num_layers * num_experts]f32 = @splat(70.0);
    for (0..num_layers) |l| {
        het_bytes[l * num_experts + 0] = 1_000_000;
        het_bw[l * num_experts + 0] = 75.0;
        het_bytes[l * num_experts + 1] = 400_000; // q2_affine
        het_bw[l * num_experts + 1] = 90.0;
        het_bytes[l * num_experts + 2] = 6_000_000; // bf16
        het_bw[l * num_experts + 2] = 60.0;
    }

    var het_cost = PruneCostModel.uniformBeta(100.0, 2.0, 0.0, 0.0, 70.0, num_layers, num_experts);
    het_cost.expert_bytes_table = &het_bytes;
    het_cost.expert_bw_gbs_table = &het_bw;

    const sum_het = try t_het.pruneToOptimalSubtree(het_cost, &exp_het, null, 16);
    try std.testing.expectEqual(@as(u16, 2), sum_het.kept_nodes);
    try std.testing.expect(t_het.findChild(0, 100) != null);
    try std.testing.expectEqual(@as(?u16, null), t_het.findChild(0, 200));

    // 3. Table validation rejects short tables or non-positive bandwidths.
    var bad_short = het_cost;
    bad_short.expert_bytes_table = het_bytes[0 .. num_layers * num_experts - 1];
    try std.testing.expectError(error.BufferTooSmall, t_het.scoreCurrentTree(bad_short, &exp_het, null));

    var bad_bw_arr = het_bw;
    bad_bw_arr[3] = 0.0;
    var bad_bw = het_cost;
    bad_bw.expert_bw_gbs_table = &bad_bw_arr;
    try std.testing.expectError(error.InvalidCostModel, t_het.scoreCurrentTree(bad_bw, &exp_het, null));
}

test "M5 D8 DraftTree highestProbPath and copyTruncatedFrom preserve topology and caps" {
    var tree = DefaultTree.init(42);
    const a1 = tree.insertChild(0, 10, 0.40, .l2) orelse unreachable;
    _ = tree.insertChild(a1, 11, 0.90, .l2) orelse unreachable;
    const b1 = tree.insertChild(0, 20, 0.85, .l2) orelse unreachable;
    const b2 = tree.insertChild(b1, 21, 0.30, .l2) orelse unreachable;
    const b2_hi = tree.insertChild(b1, 22, 0.75, .l1_seed) orelse unreachable;
    const b3 = tree.insertChild(b2_hi, 23, 0.80, .l1_seed) orelse unreachable;
    _ = b2;

    var nodes_out: [8]u16 = undefined;
    var toks_out: [8]u32 = undefined;
    const path_len = tree.highestProbPath(&nodes_out, &toks_out);
    try std.testing.expectEqual(@as(u16, 3), path_len);
    try std.testing.expectEqualSlices(u16, &[_]u16{ b1, b2_hi, b3 }, nodes_out[0..path_len]);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 20, 22, 23 }, toks_out[0..path_len]);

    // Truncate to depth <= 2 and cap <= 4 nodes.
    var trunc = DefaultTree.init(0);
    trunc.copyTruncatedFrom(&tree, 2, 4);
    try std.testing.expectEqual(@as(u32, 42), trunc.nodes[0].token_id);
    try std.testing.expectEqual(@as(u16, 4), trunc.count);
    try std.testing.expect(trunc.max_depth <= 2);
    // Root (0) + a1 (depth 1) + a2 (depth 2) + b1 (depth 1) = 4 nodes; b3 (depth 3) excluded.
    try std.testing.expect(trunc.findChild(0, 10) != null);
    try std.testing.expect(trunc.findChild(0, 20) != null);
}
