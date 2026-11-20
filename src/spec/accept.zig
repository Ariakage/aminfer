//! Lossless greedy (`T = 0`) and stochastic (`T > 0`) speculative tree
//! acceptance (status row D9, docs/architecture/speculative-decoding.md).
//!
//! Zero dynamic heap allocation on the verification hot path: callers supply
//! preallocated output buffers and a `vocab_size` `f32` residual scratch slice.

const std = @import("std");
const tree_mod = @import("tree.zig");

pub const VerifyError = error{
    EmptyTree,
    EmptyVocab,
    InvalidTargetLength,
    InvalidDraftLength,
    InvalidDraftProbs,
    InvalidTemperature,
    InvalidLogits,
    ScratchTooSmall,
    OutputBufferTooSmall,
    TokenOutOfRange,
    Overflow,
};

/// Result of verifying a `DraftTree` against target model predictions.
pub const AcceptResult = struct {
    /// Number of accepted tree nodes written to `out_accepted_nodes`
    /// (`[0, c_1, ..., c_k]`, always `1 + draft_accepted >= 1`).
    /// Node `0` is always accepted at index `0` because its KV was computed at `base_pos`.
    accepted_node_count: u16,
    /// Number of newly produced tokens written to `out_tokens`
    /// (`[c_1.tok, ..., c_k.tok, bonus_tok]`, always `1 + draft_accepted >= 1`).
    emitted_token_count: u16,
    /// Number of draft child tokens accepted before the bonus token (`k >= 0`).
    draft_accepted: u16,
    /// Tree index of the deepest accepted node (`out_accepted_nodes[accepted_node_count - 1]`).
    last_accepted_node: u16,
    /// Bonus token selected/sampled at `last_accepted_node` (`out_tokens[emitted_token_count - 1]`).
    bonus_token: u32,
};

/// Lossless greedy (`T = 0`) tree verification.
///
/// Walks from root node `0`: at current node `u`, reads `t = target_argmax[u]`.
/// If `u` has a child `c` with `token_id == t`, accepts `c` and advances `u = c`;
/// otherwise emits `t` as the bonus token and stops.
///
/// Always produces `1 + k` accepted tree nodes (`[0, c_1, ..., c_k]`) whose KV
/// rows get committed at `base_pos .. base_pos + k`, and `1 + k` newly emitted
/// tokens (`[c_1.tok, ..., c_k.tok, bonus_tok]`), identical to sequential greedy
/// decoding.
pub fn verifyGreedy(
    comptime max_nodes: u16,
    tree: *const tree_mod.DraftTree(max_nodes),
    target_argmax: []const u32,
    out_accepted_nodes: []u16,
    out_tokens: []u32,
) VerifyError!AcceptResult {
    if (tree.count == 0) return error.EmptyTree;
    if (target_argmax.len < tree.count) return error.InvalidTargetLength;

    const max_out: usize = @as(usize, tree.max_depth) + 1;
    if (out_accepted_nodes.len < max_out or out_tokens.len < max_out) {
        return error.OutputBufferTooSmall;
    }

    var u: u16 = 0;
    out_accepted_nodes[0] = 0;
    var draft_accepted: u16 = 0;

    while (true) {
        const t = target_argmax[u];
        if (tree.findChild(u, t)) |child| {
            out_tokens[draft_accepted] = t;
            draft_accepted += 1;
            out_accepted_nodes[draft_accepted] = child;
            u = child;
        } else {
            out_tokens[draft_accepted] = t;
            const total: u16 = draft_accepted + 1;
            return .{
                .accepted_node_count = total,
                .emitted_token_count = total,
                .draft_accepted = draft_accepted,
                .last_accepted_node = u,
                .bonus_token = t,
            };
        }
    }
}

/// Computes numerically stable `softmax(logits / temperature)` in `f64`
/// intermediate precision and writes normalized `f32` probabilities into `out_probs`.
pub fn softmaxRow(
    comptime LogitT: type,
    logits: []const LogitT,
    temperature: f32,
    out_probs: []f32,
) VerifyError!void {
    _ = try softmaxRowWithSum(LogitT, logits, temperature, out_probs);
}

fn softmaxRowWithSum(
    comptime LogitT: type,
    logits: []const LogitT,
    temperature: f32,
    out_probs: []f32,
) VerifyError!f64 {
    comptime {
        if (LogitT != f16 and LogitT != f32) {
            @compileError("softmaxRow LogitT must be f16 or f32");
        }
    }
    if (logits.len == 0) return error.EmptyVocab;
    if (out_probs.len < logits.len) return error.ScratchTooSmall;
    if (!std.math.isFinite(temperature) or !(temperature > 0.0)) {
        return error.InvalidTemperature;
    }

    var max_val: f32 = -std.math.inf(f32);
    for (logits) |raw| {
        const v: f32 = @floatCast(raw);
        if (std.math.isNan(v) or (std.math.isInf(v) and v > 0.0)) {
            return error.InvalidLogits;
        }
        if (v > max_val) max_val = v;
    }
    if (max_val == -std.math.inf(f32)) return error.InvalidLogits;

    const inv_t: f64 = 1.0 / @as(f64, temperature);
    const max_f64: f64 = @as(f64, max_val);
    var sum: f64 = 0.0;

    for (logits, 0..) |raw, i| {
        const v: f64 = @floatCast(raw);
        const shifted = (v - max_f64) * inv_t;
        const e: f64 = if (shifted < -80.0) 0.0 else @exp(shifted);
        out_probs[i] = @floatCast(e);
        sum += e;
    }

    if (!(sum > 0.0) or !std.math.isFinite(sum)) return error.InvalidLogits;

    const inv_sum: f64 = 1.0 / sum;
    var norm_sum: f64 = 0.0;
    for (0..logits.len) |i| {
        const p_f32: f32 = @floatCast(@as(f64, out_probs[i]) * inv_sum);
        out_probs[i] = p_f32;
        norm_sum += @as(f64, p_f32);
    }
    return norm_sum;
}

/// Samples a token index from a non-negative probability slice (normalized or
/// unnormalized) using `f64` cumulative summation. Never returns a zero-mass token
/// as long as at least one entry is strictly positive.
pub fn sampleCategorical(probs: []const f32, rng: std.Random) u32 {
    std.debug.assert(probs.len > 0);
    var total: f64 = 0.0;
    for (probs) |p| {
        if (p > 0.0) total += @as(f64, p);
    }
    if (!(total > 0.0)) return 0;

    const u: f64 = rng.float(f64) * total;
    var acc: f64 = 0.0;
    var last_pos: u32 = 0;
    for (probs, 0..) |p, idx| {
        if (p > 0.0) {
            last_pos = @intCast(idx);
            acc += @as(f64, p);
            if (u < acc) return @intCast(idx);
        }
    }
    return last_pos;
}

/// Lossless stochastic (`temperature > 0`) multi-candidate recursive rejection
/// sampling over sibling candidates in a `DraftTree` (`speculative-decoding.md`).
///
/// Two mathematically exact regimes are supported with zero dynamic heap allocation:
///
/// 1. **Deterministic / top-k / dual-source candidate sets (`draft_probs == null`)**:
///    Used whenever sibling candidates come from deterministic sources (L1 suffix
///    automaton, EAGLE-2/3 top-k beam expansion, or merged L1+L2 trees with
///    arbitrary `step_prob`s). Each distinct sibling `c_j` with token `x_j` is
///    treated as a point-mass proposal: `c_j` is accepted with conditional
///    probability `p_res[x_j] / rem`, and upon rejection `p_res[x_j]` is zeroed
///    and subtracted from `rem`. This yields `P(accept c_j) = p_target(x_j)`
///    exactly (total acceptance `sum_j p_target(x_j)`) and samples any bonus token
///    from the remaining mass, preserving `p_target = softmax(logits / T)` exactly
///    regardless of tree shape or `step_prob` skew.
///
/// 2. **Stochastic without-replacement draft distributions (`draft_probs != null`)**:
///    `draft_probs` has flat shape `[tree.count * vocab_size]`, where sibling
///    children `c_1, ..., c_m` of node `u` were drawn without replacement from
///    `q_u = draft_probs[u * vocab_size ..][0..vocab_size]`. At stage `j`,
///    `q^{(j-1)}(x) = q_u[x] / q_rem` over unrejected tokens; `c_j` is accepted
///    with probability `min(1, p_res[x_j] / q^{(j-1)}(x_j))`, and on rejection
///    `p_res` is updated to `norm(max(0, p_res - q^{(j-1)}))` and `q_rem -= q_u[x_j]`.
///    Because rejection forces `p_res[x_j] == 0` for all previously rejected
///    siblings, `max(0, p_res[x] - q_u[x] / q_rem)` stays exact in-place inside
///    the single `p_res` scratch slice. Any sibling with `q_u[x_j] == 0` falls
///    back cleanly to exact point-mass verification.
pub fn verifyStochastic(
    comptime max_nodes: u16,
    comptime LogitT: type,
    tree: *const tree_mod.DraftTree(max_nodes),
    target_logits: []const LogitT,
    vocab_size: usize,
    draft_probs: ?[]const f32,
    temperature: f32,
    rng: std.Random,
    p_res: []f32,
    out_accepted_nodes: []u16,
    out_tokens: []u32,
) VerifyError!AcceptResult {
    comptime {
        if (LogitT != f16 and LogitT != f32) {
            @compileError("verifyStochastic LogitT must be f16 or f32");
        }
    }
    if (tree.count == 0) return error.EmptyTree;
    if (vocab_size == 0) return error.EmptyVocab;
    if (!std.math.isFinite(temperature) or !(temperature > 0.0)) {
        return error.InvalidTemperature;
    }

    const total_logits = std.math.mul(usize, @as(usize, tree.count), vocab_size) catch return error.Overflow;
    if (target_logits.len < total_logits) return error.InvalidTargetLength;
    if (draft_probs) |dp| {
        if (dp.len < total_logits) return error.InvalidDraftLength;
    }
    if (p_res.len < vocab_size) return error.ScratchTooSmall;

    const max_out: usize = @as(usize, tree.max_depth) + 1;
    if (out_accepted_nodes.len < max_out or out_tokens.len < max_out) {
        return error.OutputBufferTooSmall;
    }

    const residual = p_res[0..vocab_size];

    var u: u16 = 0;
    out_accepted_nodes[0] = 0;
    var draft_accepted: u16 = 0;

    while (true) {
        const row_start = @as(usize, u) * vocab_size;
        const row_logits = target_logits[row_start .. row_start + vocab_size];
        const norm_sum = try softmaxRowWithSum(LogitT, row_logits, temperature, residual);

        const first_child = tree.nodes[u].first_child;
        if (first_child == tree_mod.NO_NODE) {
            const bonus = sampleCategorical(residual, rng);
            out_tokens[draft_accepted] = bonus;
            const total: u16 = draft_accepted + 1;
            return .{
                .accepted_node_count = total,
                .emitted_token_count = total,
                .draft_accepted = draft_accepted,
                .last_accepted_node = u,
                .bonus_token = bonus,
            };
        }

        // Validate every child token ID against `vocab_size` before mutating `residual`.
        {
            var chk = first_child;
            while (chk != tree_mod.NO_NODE) : (chk = tree.nodes[chk].next_sibling) {
                if (tree.nodes[chk].token_id >= vocab_size) return error.TokenOutOfRange;
            }
        }

        var accepted_child: ?u16 = null;

        if (draft_probs) |dp| {
            const q_u = dp[row_start .. row_start + vocab_size];
            var q_rem: f64 = 0.0;
            for (q_u) |qv| {
                if (!std.math.isFinite(qv) or qv < 0.0) return error.InvalidDraftProbs;
                q_rem += @as(f64, qv);
            }

            var c = first_child;
            while (c != tree_mod.NO_NODE) : (c = tree.nodes[c].next_sibling) {
                const tok: usize = tree.nodes[c].token_id;
                const p_tok: f64 = @as(f64, residual[tok]);
                const q_raw: f64 = @as(f64, q_u[tok]);

                if (q_raw > 0.0 and q_rem > 0.0) {
                    const q_eff: f64 = @min(1.0, q_raw / q_rem);
                    const alpha: f64 = if (p_tok >= q_eff) 1.0 else p_tok / q_eff;
                    if (rng.float(f64) < alpha) {
                        accepted_child = c;
                        break;
                    }

                    // Rejected: p_res <- norm(max(0, p_res - q^{(j-1)})), q_rem -= q_raw.
                    const inv_q_rem: f64 = 1.0 / q_rem;
                    var z: f64 = 0.0;
                    for (0..vocab_size) |x| {
                        if (x == tok) {
                            residual[x] = 0.0;
                            continue;
                        }
                        const cur_p: f64 = @as(f64, residual[x]);
                        if (cur_p <= 0.0) {
                            residual[x] = 0.0;
                            continue;
                        }
                        const diff = cur_p - @as(f64, q_u[x]) * inv_q_rem;
                        if (diff > 0.0) {
                            residual[x] = @floatCast(diff);
                            z += diff;
                        } else {
                            residual[x] = 0.0;
                        }
                    }

                    q_rem -= q_raw;
                    if (q_rem < 0.0) q_rem = 0.0;

                    if (z > 0.0) {
                        const inv_z: f64 = 1.0 / z;
                        for (0..vocab_size) |x| {
                            if (residual[x] > 0.0) {
                                residual[x] = @floatCast(@as(f64, residual[x]) * inv_z);
                            }
                        }
                    } else {
                        // Extreme underflow guard: restore target distribution before bonus draw.
                        _ = try softmaxRowWithSum(LogitT, row_logits, temperature, residual);
                        break;
                    }
                } else {
                    // Exact point-mass fallback for deterministic candidates with zero draft mass.
                    var rem: f64 = 0.0;
                    for (residual) |pv| rem += @as(f64, pv);
                    if (rem > 0.0 and p_tok > 0.0) {
                        if (rng.float(f64) * rem < p_tok) {
                            accepted_child = c;
                            break;
                        }
                        residual[tok] = 0.0;
                        const new_rem = rem - p_tok;
                        if (new_rem > 0.0) {
                            const inv_rem: f64 = 1.0 / new_rem;
                            for (0..vocab_size) |x| {
                                if (residual[x] > 0.0) {
                                    residual[x] = @floatCast(@as(f64, residual[x]) * inv_rem);
                                }
                            }
                        } else {
                            _ = try softmaxRowWithSum(LogitT, row_logits, temperature, residual);
                            break;
                        }
                    }
                }
            }
        } else {
            // Exact point-mass recursive rejection sampling over deterministic candidates.
            var rem: f64 = norm_sum;
            var c = first_child;
            while (c != tree_mod.NO_NODE) : (c = tree.nodes[c].next_sibling) {
                const tok: usize = tree.nodes[c].token_id;
                const p_tok: f64 = @as(f64, residual[tok]);
                if (rem > 0.0 and p_tok > 0.0) {
                    if (rng.float(f64) * rem < p_tok) {
                        accepted_child = c;
                        break;
                    }
                    rem -= p_tok;
                    if (rem < 0.0) rem = 0.0;
                    residual[tok] = 0.0;
                }
            }
            if (accepted_child == null and !(rem > 0.0)) {
                _ = try softmaxRowWithSum(LogitT, row_logits, temperature, residual);
            }
        }

        if (accepted_child) |child| {
            out_tokens[draft_accepted] = tree.nodes[child].token_id;
            draft_accepted += 1;
            out_accepted_nodes[draft_accepted] = child;
            u = child;
        } else {
            const bonus = sampleCategorical(residual, rng);
            out_tokens[draft_accepted] = bonus;
            const total: u16 = draft_accepted + 1;
            return .{
                .accepted_node_count = total,
                .emitted_token_count = total,
                .draft_accepted = draft_accepted,
                .last_accepted_node = u,
                .bonus_token = bonus,
            };
        }
    }
}

/// Convenience wrapper for `verifyStochastic` with `f16` GPU target logits.
pub fn verifyStochasticF16(
    comptime max_nodes: u16,
    tree: *const tree_mod.DraftTree(max_nodes),
    target_logits_f16: []const f16,
    vocab_size: usize,
    draft_probs: ?[]const f32,
    temperature: f32,
    rng: std.Random,
    p_res: []f32,
    out_accepted_nodes: []u16,
    out_tokens: []u32,
) VerifyError!AcceptResult {
    return verifyStochastic(
        max_nodes,
        f16,
        tree,
        target_logits_f16,
        vocab_size,
        draft_probs,
        temperature,
        rng,
        p_res,
        out_accepted_nodes,
        out_tokens,
    );
}

/// Convenience wrapper for `verifyStochastic` with `f32` target logits.
pub fn verifyStochasticF32(
    comptime max_nodes: u16,
    tree: *const tree_mod.DraftTree(max_nodes),
    target_logits_f32: []const f32,
    vocab_size: usize,
    draft_probs: ?[]const f32,
    temperature: f32,
    rng: std.Random,
    p_res: []f32,
    out_accepted_nodes: []u16,
    out_tokens: []u32,
) VerifyError!AcceptResult {
    return verifyStochastic(
        max_nodes,
        f32,
        tree,
        target_logits_f32,
        vocab_size,
        draft_probs,
        temperature,
        rng,
        p_res,
        out_accepted_nodes,
        out_tokens,
    );
}

fn sampleWithoutReplacement(probs: []const f32, k: usize, rng: std.Random, out: []u32) usize {
    var rem: [64]f32 = undefined;
    std.debug.assert(probs.len <= rem.len);
    @memcpy(rem[0..probs.len], probs);
    var drawn: usize = 0;
    while (drawn < k) {
        var total: f64 = 0.0;
        for (rem[0..probs.len]) |p| {
            if (p > 0.0) total += @as(f64, p);
        }
        if (!(total > 0.0)) break;
        const tok = sampleCategorical(rem[0..probs.len], rng);
        out[drawn] = tok;
        rem[tok] = 0.0;
        drawn += 1;
    }
    return drawn;
}

test "verifyGreedy handles 0-accepted fallback, chains, multi-branch trees, and full-depth matches" {
    var tree = tree_mod.DefaultTree.init(100);
    // Build a branching tree:
    //   0 (100)
    //   ├── 1 (10) -> 3 (30) -> 5 (50)
    //   └── 2 (20) -> 4 (40)
    const n1 = tree.insertChild(0, 10, 0.7, .l1_seed) orelse unreachable;
    const n2 = tree.insertChild(0, 20, 0.3, .l2) orelse unreachable;
    const n3 = tree.insertChild(n1, 30, 0.8, .l1_seed) orelse unreachable;
    const n4 = tree.insertChild(n2, 40, 0.6, .l2) orelse unreachable;
    const n5 = tree.insertChild(n3, 50, 0.9, .l1_seed) orelse unreachable;

    var acc_nodes: [8]u16 = undefined;
    var out_toks: [8]u32 = undefined;

    // Case 1: 0 draft tokens match -> root 0 accepted + bonus token 99.
    const argmax_miss = [_]u32{ 99, 30, 40, 50, 77, 88 };
    const r0 = try verifyGreedy(32, &tree, &argmax_miss, &acc_nodes, &out_toks);
    try std.testing.expectEqual(@as(u16, 0), r0.draft_accepted);
    try std.testing.expectEqual(@as(u16, 1), r0.accepted_node_count);
    try std.testing.expectEqual(@as(u16, 1), r0.emitted_token_count);
    try std.testing.expectEqual(@as(u16, 0), r0.last_accepted_node);
    try std.testing.expectEqual(@as(u32, 99), r0.bonus_token);
    try std.testing.expectEqualSlices(u16, &[_]u16{0}, acc_nodes[0..r0.accepted_node_count]);
    try std.testing.expectEqualSlices(u32, &[_]u32{99}, out_toks[0..r0.emitted_token_count]);

    // Case 2: Second branch matches partially: 0 -> 2 (20) -> 4 (40) -> bonus 77.
    const argmax_branch = [_]u32{ 20, 30, 40, 50, 77, 88 };
    const r1 = try verifyGreedy(32, &tree, &argmax_branch, &acc_nodes, &out_toks);
    try std.testing.expectEqual(@as(u16, 2), r1.draft_accepted);
    try std.testing.expectEqual(@as(u16, 3), r1.accepted_node_count);
    try std.testing.expectEqual(@as(u16, 3), r1.emitted_token_count);
    try std.testing.expectEqual(n4, r1.last_accepted_node);
    try std.testing.expectEqual(@as(u32, 77), r1.bonus_token);
    try std.testing.expectEqualSlices(u16, &[_]u16{ 0, n2, n4 }, acc_nodes[0..r1.accepted_node_count]);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 20, 40, 77 }, out_toks[0..r1.emitted_token_count]);

    // Case 3: Full-depth match along first branch: 0 -> 1 (10) -> 3 (30) -> 5 (50) -> bonus 88.
    const argmax_full = [_]u32{ 10, 30, 40, 50, 77, 88 };
    const r2 = try verifyGreedy(32, &tree, &argmax_full, &acc_nodes, &out_toks);
    try std.testing.expectEqual(@as(u16, 3), r2.draft_accepted);
    try std.testing.expectEqual(@as(u16, 4), r2.accepted_node_count);
    try std.testing.expectEqual(@as(u16, 4), r2.emitted_token_count);
    try std.testing.expectEqual(n5, r2.last_accepted_node);
    try std.testing.expectEqual(@as(u32, 88), r2.bonus_token);
    try std.testing.expectEqualSlices(u16, &[_]u16{ 0, n1, n3, n5 }, acc_nodes[0..r2.accepted_node_count]);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 10, 30, 50, 88 }, out_toks[0..r2.emitted_token_count]);
}

test "verifyGreedy produces 100% bit-exact sequence identical to sequential greedy oracle" {
    // Deterministic autoregressive oracle depending on prefix hash + position.
    const Oracle = struct {
        fn nextToken(prefix_hash: u64, pos: u32) u32 {
            var x = prefix_hash ^ (@as(u64, pos) *% 0x9E3779B97F4A7C15);
            x = (x ^ (x >> 30)) *% 0xbf58476d1ce4e5b9;
            x = (x ^ (x >> 27)) *% 0x94d049bb133111eb;
            return @intCast((x ^ (x >> 31)) % 32);
        }

        fn advanceHash(h: u64, tok: u32) u64 {
            return (h *% 1315423911) ^ (@as(u64, tok) +% 0x517cc1b727220a95);
        }
    };

    const total_steps: usize = 200;
    var seq_tokens: [total_steps]u32 = undefined;
    var h_seq: u64 = 0xCAFE_BABE;
    var cur_tok: u32 = 7;
    for (0..total_steps) |step| {
        h_seq = Oracle.advanceHash(h_seq, cur_tok);
        cur_tok = Oracle.nextToken(h_seq, @intCast(step));
        seq_tokens[step] = cur_tok;
    }

    // Speculative decode using branching DraftTrees mixing true oracle paths and noise.
    var prng = std.Random.DefaultPrng.init(0xD9_6EEED);
    const rng = prng.random();

    var spec_tokens: [total_steps + 16]u32 = undefined;
    var spec_len: usize = 0;
    var h_spec: u64 = 0xCAFE_BABE;
    var root_tok: u32 = 7;

    var tree = tree_mod.DefaultTree.init(root_tok);
    var node_hashes: [32]u64 = undefined;
    var target_argmax: [32]u32 = undefined;
    var acc_nodes: [32]u16 = undefined;
    var round_toks: [32]u32 = undefined;

    while (spec_len < total_steps) {
        tree.reset(root_tok);
        node_hashes[0] = Oracle.advanceHash(h_spec, root_tok);

        // Build a branching tree of depth <= 4 with mixed matching and decoy drafts.
        var parent_idx: u16 = 0;
        while (parent_idx < tree.count and tree.count < 16) : (parent_idx += 1) {
            if (tree.nodes[parent_idx].depth >= 4) continue;
            const pos: u32 = @intCast(spec_len + tree.nodes[parent_idx].depth);
            const true_next = Oracle.nextToken(node_hashes[parent_idx], pos);
            // Insert a decoy sibling and sometimes the true oracle token.
            const decoy = (true_next + 1 + rng.uintLessThan(u32, 31)) % 32;
            if (tree.insertChild(parent_idx, decoy, 0.4, .l2)) |c| {
                node_hashes[c] = Oracle.advanceHash(node_hashes[parent_idx], decoy);
            }
            if (rng.uintLessThan(u32, 3) != 0) {
                if (tree.insertChild(parent_idx, true_next, 0.6, .l1_seed)) |c| {
                    node_hashes[c] = Oracle.advanceHash(node_hashes[parent_idx], true_next);
                }
            }
        }

        // Evaluate target oracle at every node in the tree.
        for (0..tree.count) |i| {
            const pos: u32 = @intCast(spec_len + tree.nodes[i].depth);
            target_argmax[i] = Oracle.nextToken(node_hashes[i], pos);
        }

        const res = try verifyGreedy(32, &tree, target_argmax[0..tree.count], &acc_nodes, &round_toks);
        for (0..res.emitted_token_count) |k| {
            spec_tokens[spec_len] = round_toks[k];
            spec_len += 1;
        }
        // Advance prefix hash through all committed KV nodes (`[0, c_1, ..., c_k]`).
        h_spec = node_hashes[res.last_accepted_node];
        root_tok = res.bonus_token;
    }

    try std.testing.expectEqualSlices(u32, seq_tokens[0..total_steps], spec_tokens[0..total_steps]);
}

test "verifyStochastic preserves exact target distribution over 50,000 rounds (skewed stochastic draft q_u, TVD < 0.015)" {
    const V = 8;
    const rounds: usize = 50_000;
    const temperature: f32 = 0.8;

    // Target logits at depth 0 (`u = 0`) and conditional target logits at depth 1
    // indexed by first emitted token `t0 in 0..V`.
    const root_logits_f32 = [V]f32{ 1.8, 1.1, 0.7, 0.2, -0.3, -0.8, -1.2, 0.4 };
    var cond_logits_f32: [V][V]f32 = undefined;
    for (0..V) |t0| {
        for (0..V) |t1| {
            const phase: f32 = @floatFromInt(@as(i32, @intCast((t0 * 3 + t1 * 5) % 8)) - 3);
            cond_logits_f32[t0][t1] = phase * 0.45;
        }
    }

    // Deliberately heavily skewed draft distributions (poorly aligned with target!).
    const q_root = [V]f32{ 0.03, 0.05, 0.07, 0.35, 0.25, 0.15, 0.08, 0.02 };
    const q_child = [V]f32{ 0.02, 0.28, 0.04, 0.06, 0.30, 0.20, 0.05, 0.05 };

    // Exact ground-truth target probabilities.
    var p_root: [V]f32 = undefined;
    try softmaxRow(f32, &root_logits_f32, temperature, &p_root);
    var p_cond: [V][V]f32 = undefined;
    for (0..V) |t0| {
        try softmaxRow(f32, &cond_logits_f32[t0], temperature, &p_cond[t0]);
    }

    var prng = std.Random.DefaultPrng.init(0x570C_A571_C001);
    const rng = prng.random();

    var counts_t0: [V]u64 = @splat(0);
    var counts_joint: [V * V]u64 = @splat(0);

    var tree = tree_mod.SmallTree.init(0);
    var target_logits_f16: [16 * V]f16 = undefined;
    var draft_probs: [16 * V]f32 = undefined;
    var p_res: [V]f32 = undefined;
    var acc_nodes: [16]u16 = undefined;
    var out_toks: [16]u32 = undefined;

    for (0..rounds) |_| {
        tree.reset(0);

        // Sample 3 distinct depth-1 children without replacement from skewed `q_root`.
        var d1_toks: [3]u32 = undefined;
        const d1_n = sampleWithoutReplacement(&q_root, 3, rng, &d1_toks);
        var d1_nodes: [3]u16 = undefined;
        for (0..d1_n) |i| {
            d1_nodes[i] = tree.insertChild(0, d1_toks[i], q_root[d1_toks[i]], .l2) orelse unreachable;
            // Sample 2 distinct depth-2 children without replacement from skewed `q_child`.
            var d2_toks: [2]u32 = undefined;
            const d2_n = sampleWithoutReplacement(&q_child, 2, rng, &d2_toks);
            for (0..d2_n) |j| {
                _ = tree.insertChild(d1_nodes[i], d2_toks[j], q_child[d2_toks[j]], .l2) orelse unreachable;
            }
        }

        // Fill f16 target logits and per-node draft distributions.
        for (0..V) |v| {
            target_logits_f16[v] = @floatCast(root_logits_f32[v]);
            draft_probs[v] = q_root[v];
        }
        var idx: u16 = 1;
        while (idx < tree.count) : (idx += 1) {
            const base = @as(usize, idx) * V;
            const tok = tree.nodes[idx].token_id;
            if (tree.nodes[idx].depth == 1) {
                for (0..V) |v| {
                    target_logits_f16[base + v] = @floatCast(cond_logits_f32[tok][v]);
                    draft_probs[base + v] = q_child[v];
                }
            } else {
                // Depth-2 leaves have no children; logits still valid.
                for (0..V) |v| {
                    target_logits_f16[base + v] = @floatCast(root_logits_f32[v]);
                    draft_probs[base + v] = q_child[v];
                }
            }
        }

        const res = try verifyStochasticF16(
            16,
            &tree,
            target_logits_f16[0 .. @as(usize, tree.count) * V],
            V,
            draft_probs[0 .. @as(usize, tree.count) * V],
            temperature,
            rng,
            &p_res,
            &acc_nodes,
            &out_toks,
        );

        const t0 = out_toks[0];
        counts_t0[t0] += 1;

        // If speculative verification already produced `t1` (when depth-1 was accepted),
        // use `out_toks[1]`; otherwise sample `t1` from exact `p_cond[t0]` (next round).
        const t1: u32 = if (res.emitted_token_count >= 2)
            out_toks[1]
        else
            sampleCategorical(&p_cond[t0], rng);
        counts_joint[t0 * V + t1] += 1;
    }

    // Recompute exact f16-quantized target reference probabilities for TVD check.
    var root_f16: [V]f16 = undefined;
    for (0..V) |v| root_f16[v] = @floatCast(root_logits_f32[v]);
    var p_root_ref: [V]f32 = undefined;
    try softmaxRow(f16, &root_f16, temperature, &p_root_ref);

    var tvd_t0: f64 = 0.0;
    for (0..V) |v| {
        const emp = @as(f64, @floatFromInt(counts_t0[v])) / @as(f64, rounds);
        tvd_t0 += @abs(emp - @as(f64, p_root_ref[v]));
    }
    tvd_t0 *= 0.5;
    try std.testing.expect(tvd_t0 < 0.015);

    var tvd_joint: f64 = 0.0;
    for (0..V) |t0| {
        var c_f16: [V]f16 = undefined;
        for (0..V) |t1| c_f16[t1] = @floatCast(cond_logits_f32[t0][t1]);
        var p_c_ref: [V]f32 = undefined;
        try softmaxRow(f16, &c_f16, temperature, &p_c_ref);
        for (0..V) |t1| {
            const emp = @as(f64, @floatFromInt(counts_joint[t0 * V + t1])) / @as(f64, rounds);
            const exact = @as(f64, p_root_ref[t0]) * @as(f64, p_c_ref[t1]);
            tvd_joint += @abs(emp - exact);
        }
    }
    tvd_joint *= 0.5;
    try std.testing.expect(tvd_joint < 0.015);
}

test "verifyStochastic preserves exact target distribution over 50,000 rounds (deterministic branching tree, draft_probs = null, TVD < 0.015)" {
    const V = 8;
    const rounds: usize = 50_000;
    const temperature: f32 = 0.7;

    // Fixed dual-source branching tree with deliberately skewed `step_prob`s.
    var tree = tree_mod.SmallTree.init(0);
    const n1 = tree.insertChild(0, 3, 0.95, .l1_seed) orelse unreachable;
    const n2 = tree.insertChild(0, 0, 0.02, .l2) orelse unreachable;
    const n3 = tree.insertChild(0, 6, 0.01, .l1_branch) orelse unreachable;
    _ = tree.insertChild(n1, 2, 0.90, .l1_seed) orelse unreachable;
    _ = tree.insertChild(n1, 5, 0.05, .l2) orelse unreachable;
    _ = tree.insertChild(n2, 1, 0.80, .l2) orelse unreachable;
    _ = tree.insertChild(n3, 7, 0.60, .l1_branch) orelse unreachable;

    const root_logits = [V]f32{ 1.5, 0.4, -0.2, 0.9, -0.7, 0.1, 1.2, -1.0 };
    var cond_logits: [V][V]f32 = undefined;
    for (0..V) |t0| {
        for (0..V) |t1| {
            const d: f32 = @floatFromInt(@as(i32, @intCast((t0 * 7 + t1 * 3) % 8)) - 4);
            cond_logits[t0][t1] = d * 0.35;
        }
    }

    var target_logits: [16 * V]f32 = undefined;
    @memcpy(target_logits[0..V], &root_logits);
    var idx: u16 = 1;
    while (idx < tree.count) : (idx += 1) {
        const base = @as(usize, idx) * V;
        const tok = tree.nodes[idx].token_id;
        if (tree.nodes[idx].depth == 1) {
            @memcpy(target_logits[base .. base + V], &cond_logits[tok]);
        } else {
            @memcpy(target_logits[base .. base + V], &root_logits);
        }
    }

    var p_root: [V]f32 = undefined;
    try softmaxRow(f32, &root_logits, temperature, &p_root);
    var p_cond: [V][V]f32 = undefined;
    for (0..V) |t0| {
        try softmaxRow(f32, &cond_logits[t0], temperature, &p_cond[t0]);
    }

    var prng = std.Random.DefaultPrng.init(0xDE7E_4111_571C);
    const rng = prng.random();

    var counts_t0: [V]u64 = @splat(0);
    var counts_joint: [V * V]u64 = @splat(0);
    var p_res: [V]f32 = undefined;
    var acc_nodes: [16]u16 = undefined;
    var out_toks: [16]u32 = undefined;

    for (0..rounds) |_| {
        const res = try verifyStochasticF32(
            16,
            &tree,
            target_logits[0 .. @as(usize, tree.count) * V],
            V,
            null,
            temperature,
            rng,
            &p_res,
            &acc_nodes,
            &out_toks,
        );
        const t0 = out_toks[0];
        counts_t0[t0] += 1;
        const t1: u32 = if (res.emitted_token_count >= 2)
            out_toks[1]
        else
            sampleCategorical(&p_cond[t0], rng);
        counts_joint[t0 * V + t1] += 1;
    }

    var tvd_t0: f64 = 0.0;
    for (0..V) |v| {
        const emp = @as(f64, @floatFromInt(counts_t0[v])) / @as(f64, rounds);
        tvd_t0 += @abs(emp - @as(f64, p_root[v]));
    }
    tvd_t0 *= 0.5;
    try std.testing.expect(tvd_t0 < 0.015);

    var tvd_joint: f64 = 0.0;
    for (0..V) |t0| {
        for (0..V) |t1| {
            const emp = @as(f64, @floatFromInt(counts_joint[t0 * V + t1])) / @as(f64, rounds);
            const exact = @as(f64, p_root[t0]) * @as(f64, p_cond[t0][t1]);
            tvd_joint += @abs(emp - exact);
        }
    }
    tvd_joint *= 0.5;
    try std.testing.expect(tvd_joint < 0.015);
}

test "verifyGreedy and verifyStochastic validate buffer lengths, temperatures, logits, and token ranges" {
    var tree = tree_mod.SmallTree.init(0);
    _ = tree.insertChild(0, 10, 0.5, .l2) orelse unreachable;

    var acc: [1]u16 = undefined;
    var out: [1]u32 = undefined;
    const argmax = [_]u32{ 10, 0 };
    try std.testing.expectError(error.OutputBufferTooSmall, verifyGreedy(16, &tree, &argmax, &acc, &out));
    try std.testing.expectError(error.InvalidTargetLength, verifyGreedy(16, &tree, argmax[0..1], &acc, &out));

    var acc2: [4]u16 = undefined;
    var out2: [4]u32 = undefined;
    var p_res: [4]f32 = undefined;
    const logits = [_]f32{ 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0 };
    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();

    try std.testing.expectError(error.InvalidTemperature, verifyStochasticF32(16, &tree, &logits, 4, null, 0.0, rng, &p_res, &acc2, &out2));
    try std.testing.expectError(error.InvalidTemperature, verifyStochasticF32(16, &tree, &logits, 4, null, -0.5, rng, &p_res, &acc2, &out2));
    try std.testing.expectError(error.TokenOutOfRange, verifyStochasticF32(16, &tree, &logits, 4, null, 1.0, rng, &p_res, &acc2, &out2));
    try std.testing.expectError(error.ScratchTooSmall, verifyStochasticF32(16, &tree, &logits, 4, null, 1.0, rng, p_res[0..2], &acc2, &out2));

    var bad_logits = logits;
    bad_logits[1] = std.math.nan(f32);
    var tree_ok = tree_mod.SmallTree.init(0);
    _ = tree_ok.insertChild(0, 1, 0.5, .l2) orelse unreachable;
    try std.testing.expectError(error.InvalidLogits, verifyStochasticF32(16, &tree_ok, &bad_logits, 4, null, 1.0, rng, &p_res, &acc2, &out2));

    // Masked `-inf` logits are valid as long as at least one token is finite;
    // all `-inf` logits must fail with `error.InvalidLogits`.
    const masked_logits = [_]f32{ -std.math.inf(f32), 2.0, -std.math.inf(f32), -std.math.inf(f32), 0.0, 1.0, 0.0, 0.0 };
    const r_masked = try verifyStochasticF32(16, &tree_ok, &masked_logits, 4, null, 1.0, rng, &p_res, &acc2, &out2);
    try std.testing.expectEqual(@as(u32, 1), out2[0]);
    try std.testing.expectEqual(@as(u16, 1), r_masked.draft_accepted);

    const all_neg_inf = [_]f32{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), 0.0, 0.0, 0.0, 0.0 };
    try std.testing.expectError(error.InvalidLogits, verifyStochasticF32(16, &tree_ok, &all_neg_inf, 4, null, 1.0, rng, &p_res, &acc2, &out2));
}

test "verifyStochastic single-child chain matches Leviathan rejection sampling (50,000 rounds, TVD < 0.015)" {
    const V = 8;
    const rounds: usize = 50_000;
    const temperature: f32 = 1.0;

    const target_row = [V]f32{ 1.2, 0.8, 0.3, -0.1, -0.5, 0.6, -1.0, 0.2 };
    const draft_row = [V]f32{ 0.05, 0.05, 0.25, 0.25, 0.15, 0.05, 0.15, 0.05 };

    var p_exact: [V]f32 = undefined;
    try softmaxRow(f32, &target_row, temperature, &p_exact);

    var prng = std.Random.DefaultPrng.init(0x1E71_A70A);
    const rng = prng.random();

    var counts: [V]u64 = @splat(0);
    var tree = tree_mod.SmallTree.init(0);
    var logits: [2 * V]f32 = undefined;
    var draft_probs: [2 * V]f32 = undefined;
    @memcpy(logits[0..V], &target_row);
    @memcpy(logits[V .. 2 * V], &target_row);
    @memcpy(draft_probs[0..V], &draft_row);
    @memcpy(draft_probs[V .. 2 * V], &draft_row);

    var p_res: [V]f32 = undefined;
    var acc_nodes: [4]u16 = undefined;
    var out_toks: [4]u32 = undefined;

    for (0..rounds) |_| {
        tree.reset(0);
        const c = sampleCategorical(&draft_row, rng);
        _ = tree.insertChild(0, c, draft_row[c], .l2) orelse unreachable;

        _ = try verifyStochasticF32(16, &tree, &logits, V, &draft_probs, temperature, rng, &p_res, &acc_nodes, &out_toks);
        counts[out_toks[0]] += 1;
    }

    var tvd: f64 = 0.0;
    for (0..V) |v| {
        const emp = @as(f64, @floatFromInt(counts[v])) / @as(f64, rounds);
        tvd += @abs(emp - @as(f64, p_exact[v]));
    }
    tvd *= 0.5;
    try std.testing.expect(tvd < 0.015);
}
