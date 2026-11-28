//! Lossless greedy (`T = 0`) and stochastic (`T > 0`) speculative tree
//! acceptance (status row D9, docs/architecture/speculative-decoding.md).
//!
//! Zero dynamic heap allocation on the verification hot path: callers supply
//! preallocated output buffers and a `vocab_size` `f32` residual scratch slice.

const std = @import("std");
const tree_mod = @import("tree.zig");
pub const grammar = @import("grammar.zig");
pub const sampling = @import("sampling.zig");

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

// ---------------------------------------------------------------------------
// M7.5 C22 Grammar-Constrained & Shared-Sampler Speculative Tree Verification
// ---------------------------------------------------------------------------

pub const ConstrainedVerifyError = VerifyError || sampling.SampleError;

/// Zero-allocation grammar hook for `G5`/`D9` speculative tree verification.
/// When `program == null`, all verification routines take the unconstrained
/// zero-overhead bypass path.
pub const GrammarTreeAdapter = struct {
    program: ?*const grammar.GrammarProgram = null,
    state: *grammar.GrammarState,
    vocab_index: ?*const grammar.VocabIndex = null,
    token_slices: ?[]const ?[]const u8 = null,
    token_bytes_fn: ?*const fn (ctx: ?*const anyopaque, token_id: u32) ?[]const u8 = null,
    token_bytes_ctx: ?*const anyopaque = null,
    eos_token_ids: []const u32 = &.{},
    mask_scratch: []u32 = &.{},

    pub inline fn isActive(self: GrammarTreeAdapter) bool {
        return self.program != null;
    }

    pub fn fillBitmask(self: GrammarTreeAdapter, vocab_size: usize) ConstrainedVerifyError![]const u32 {
        const words = grammar.bitmaskWordCount(vocab_size);
        if (self.mask_scratch.len < words) return error.BitmaskTooSmall;
        const mask_out = self.mask_scratch[0..words];
        if (self.vocab_index) |vi| {
            vi.fillAllowedTokenBitmask(self.program, self.state, mask_out);
        } else if (self.token_slices) |slices| {
            grammar.fillAllowedTokenBitmaskDirect(
                self.program,
                self.state,
                slices[0..@min(slices.len, vocab_size)],
                self.eos_token_ids,
                mask_out,
            );
        } else {
            grammar.allowAllTokens(mask_out, vocab_size);
        }
        const rem: u5 = @intCast(vocab_size & 31);
        if (rem != 0 and words > 0) {
            mask_out[words - 1] &= (@as(u32, 1) << rem) - 1;
        }
        return mask_out;
    }

    pub fn advanceTokenOnState(
        self: GrammarTreeAdapter,
        st: *grammar.GrammarState,
        token_id: u32,
    ) bool {
        const prog = self.program orelse return true;
        for (self.eos_token_ids) |eid| {
            if (eid == token_id) return st.isAccepting();
        }
        if (self.vocab_index) |vi| {
            for (vi.eos_tokens[0..vi.eos_count]) |eid| {
                if (eid == token_id) return st.isAccepting();
            }
        }
        if (self.token_slices) |slices| {
            if (token_id >= slices.len) return false;
            const raw = slices[token_id] orelse return false;
            if (raw.len == 0) return false;
            return prog.advanceBytes(st, raw);
        }
        if (self.token_bytes_fn) |get_fn| {
            const raw = get_fn(self.token_bytes_ctx, token_id) orelse return false;
            if (raw.len == 0) return false;
            return prog.advanceBytes(st, raw);
        }
        return false;
    }
};

/// Propagates `GrammarState` across all branches of a `DraftTree` (`G5`) using O(1)
/// value `snapshot()` / `restore()` at each parent node (`0` heap allocations).
/// When `program == null`, marks every active tree node valid immediately.
pub fn propagateGrammarAcrossTree(
    comptime max_nodes: u16,
    tree: *const tree_mod.DraftTree(max_nodes),
    program_opt: ?*const grammar.GrammarProgram,
    root_state: grammar.GrammarState,
    token_slices: []const ?[]const u8,
    eos_token_ids: []const u32,
    out_node_states: *[max_nodes]grammar.GrammarState,
    out_node_valid: *[max_nodes]bool,
) void {
    @memset(out_node_valid, false);
    if (tree.count == 0) return;

    const prog = program_opt orelse {
        for (0..tree.count) |i| {
            out_node_valid[i] = true;
        }
        return;
    };

    out_node_states[0] = root_state.snapshot();
    out_node_valid[0] = true;

    // Nodes in `DraftTree` are stored in topological order (`parent < child`).
    var idx: u16 = 1;
    while (idx < tree.count) : (idx += 1) {
        const node = tree.nodes[idx];
        const parent = node.parent;
        if (parent >= idx or !out_node_valid[parent]) {
            out_node_valid[idx] = false;
            continue;
        }

        var branch_state: grammar.GrammarState = undefined;
        branch_state.restore(out_node_states[parent]);

        var ok = false;
        var is_eos = false;
        for (eos_token_ids) |eid| {
            if (eid == node.token_id) {
                is_eos = true;
                ok = branch_state.isAccepting();
                break;
            }
        }
        if (!is_eos and node.token_id < token_slices.len) {
            if (token_slices[node.token_id]) |raw| {
                if (raw.len > 0) {
                    ok = prog.advanceBytes(&branch_state, raw);
                }
            }
        }

        if (ok) {
            out_node_states[idx] = branch_state;
            out_node_valid[idx] = true;
        } else {
            out_node_valid[idx] = false;
        }
    }
}

/// Grammar- and sampler-aware speculative tree verification (`G5`/`D9` + `C22` + `C18`).
///
/// - **Zero-overhead bypass**: when `grammar_adapter == null` (or `.program == null`) and
///   `params` has no penalties, `logit_bias`, `top_k`, `min_p`, `top_p < 1`, or bitmask:
///   - `temperature == 0` performs direct greedy `argmax` tree walk (`0` softmax, `0` RNG);
///   - `temperature > 0` delegates directly to `verifyStochastic`!
/// - **Constrained mode**: evaluates the `C22` allowed-token bitmask at each visited node `u`,
///   applies penalties/`logit_bias`, rejects any grammar-invalid draft sibling with zero
///   target probability mass, advances `GrammarState` via O(1) snapshot/advance on accepted
///   draft children and the final bonus token, and updates `penalty_tracker` in lockstep.
pub fn verifyConstrained(
    comptime max_nodes: u16,
    comptime LogitT: type,
    tree: *const tree_mod.DraftTree(max_nodes),
    target_logits: []const LogitT,
    vocab_size: usize,
    draft_probs: ?[]const f32,
    params: sampling.SamplingParams,
    penalty_tracker: ?*sampling.DefaultPenaltyTracker,
    grammar_adapter: ?GrammarTreeAdapter,
    rng: std.Random,
    scratch: sampling.SamplerScratch,
    out_accepted_nodes: []u16,
    out_tokens: []u32,
) ConstrainedVerifyError!AcceptResult {
    comptime {
        if (LogitT != f16 and LogitT != f32) {
            @compileError("verifyConstrained LogitT must be f16 or f32");
        }
    }
    if (tree.count == 0) return error.EmptyTree;
    if (vocab_size == 0) return error.EmptyVocab;
    try params.validate();

    const max_out: usize = @as(usize, tree.max_depth) + 1;
    if (out_accepted_nodes.len < max_out or out_tokens.len < max_out) {
        return error.OutputBufferTooSmall;
    }
    const total_logits = std.math.mul(usize, @as(usize, tree.count), vocab_size) catch return error.Overflow;
    if (target_logits.len < total_logits) return error.InvalidTargetLength;

    const gram_active = if (grammar_adapter) |ga| ga.isActive() else false;
    const unconstrained_stochastic = !gram_active and
        !params.hasPenalties() and
        params.logit_bias.len == 0 and
        params.allowed_token_mask == null and
        params.top_k == 0 and
        params.min_p == 0.0 and
        params.top_p == 1.0;

    // 1. Strict zero-overhead unconstrained fast paths!
    if (!gram_active and params.isUnconstrainedGreedy()) {
        var u: u16 = 0;
        out_accepted_nodes[0] = 0;
        var draft_accepted: u16 = 0;
        while (true) {
            const row = target_logits[@as(usize, u) * vocab_size ..][0..vocab_size];
            const sampled = try sampling.sampleToken(LogitT, row, .{}, &.{}, rng, .{});
            const t = sampled.token_id;
            if (penalty_tracker) |pt| pt.recordToken(t);
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

    if (unconstrained_stochastic and params.temperature > 0.0) {
        const res = try verifyStochastic(
            max_nodes,
            LogitT,
            tree,
            target_logits,
            vocab_size,
            draft_probs,
            params.temperature,
            rng,
            scratch.logits_f32,
            out_accepted_nodes,
            out_tokens,
        );
        if (penalty_tracker) |pt| {
            pt.recordSlice(out_tokens[0..res.emitted_token_count]);
        }
        return res;
    }

    if (scratch.logits_f32.len < vocab_size) return error.ScratchTooSmall;
    if (draft_probs) |dp| {
        if (dp.len < total_logits) return error.InvalidDraftLength;
    }

    var u: u16 = 0;
    out_accepted_nodes[0] = 0;
    var draft_accepted: u16 = 0;

    while (true) {
        const row_start = @as(usize, u) * vocab_size;
        const row_logits = target_logits[row_start .. row_start + vocab_size];

        var step_params = params;
        if (gram_active) {
            step_params.allowed_token_mask = try grammar_adapter.?.fillBitmask(vocab_size);
        }
        const seen = if (penalty_tracker) |pt| pt.activeEntries() else &[_]sampling.TokenFreqEntry{};

        // Constrained / penalized greedy (`T == 0`)
        if (step_params.temperature == 0.0) {
            const sampled = try sampling.sampleToken(LogitT, row_logits, step_params, seen, rng, scratch);
            const t = sampled.token_id;
            if (gram_active) {
                const ga = grammar_adapter.?;
                const snap = ga.state.snapshot();
                if (!ga.advanceTokenOnState(ga.state, t)) {
                    ga.state.restore(snap);
                    return error.EmptyAllowedSet;
                }
            }
            if (penalty_tracker) |pt| pt.recordToken(t);

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
            continue;
        }

        // Constrained / filtered stochastic (`T > 0`) target distribution in `residual` (`scratch.logits_f32`)!
        const residual = scratch.logits_f32[0..vocab_size];
        const max_l = try sampling.prepareConstrainedLogits(LogitT, row_logits, step_params, seen, residual);
        const inv_t: f64 = 1.0 / @as(f64, step_params.temperature);
        const max_f32: f64 = @floatCast(std.math.floatMax(f32));
        const max_f64: f64 = std.math.clamp(@as(f64, max_l), -max_f32, max_f32);

        // Build target distribution `p_target` applying `top_k`, `min_p`, `top_p`, and `allowed_token_mask`.
        var max_exp: f64 = 0.0;
        var argmax_id: usize = 0;
        for (0..vocab_size) |idx| {
            const l = residual[idx];
            if (l == max_l) argmax_id = idx;
            if (l == -std.math.inf(f32) or !std.math.isFinite(l)) {
                residual[idx] = 0.0;
                continue;
            }
            const clamped_l = std.math.clamp(@as(f64, l), -max_f32, max_f32);
            const shifted = (clamped_l - max_f64) * inv_t;
            const e: f64 = if (shifted < -80.0) 0.0 else @exp(shifted);
            residual[idx] = @floatCast(e);
            if (e > max_exp) max_exp = e;
        }

        if (step_params.min_p > 0.0 and max_exp > 0.0) {
            const thresh: f32 = @floatCast(@as(f64, step_params.min_p) * max_exp);
            for (0..vocab_size) |idx| {
                if (idx != argmax_id and residual[idx] < thresh) residual[idx] = 0.0;
            }
        }

        if ((step_params.top_k > 0 and step_params.top_k < vocab_size) or step_params.top_p < 1.0) {
            if (scratch.cand_ids.len < vocab_size) return error.ScratchTooSmall;
            var cnt: usize = 0;
            for (0..vocab_size) |idx| {
                if (residual[idx] > 0.0) {
                    scratch.cand_ids[cnt] = @intCast(idx);
                    cnt += 1;
                }
            }
            const Ctx = struct {
                s: []const f32,
                pub fn lessThan(ctx: @This(), a: u32, b: u32) bool {
                    const va = ctx.s[a];
                    const vb = ctx.s[b];
                    if (va != vb) return va > vb;
                    return a < b;
                }
            };
            std.mem.sort(u32, scratch.cand_ids[0..cnt], Ctx{ .s = residual }, Ctx.lessThan);
            var keep = cnt;
            if (step_params.top_k > 0 and step_params.top_k < keep) {
                keep = @intCast(step_params.top_k);
            }
            if (step_params.top_p < 1.0 and keep > 1) {
                var sum_all: f64 = 0.0;
                for (scratch.cand_ids[0..keep]) |id| sum_all += @as(f64, residual[id]);
                const target_mass = @as(f64, step_params.top_p) * sum_all;
                var cum: f64 = 0.0;
                var kp: usize = 0;
                for (scratch.cand_ids[0..keep]) |id| {
                    cum += @as(f64, residual[id]);
                    kp += 1;
                    if (cum >= target_mass) break;
                }
                keep = @max(kp, 1);
            }
            for (scratch.cand_ids[keep..cnt]) |drop_id| {
                residual[drop_id] = 0.0;
            }
        }

        var sum_e: f64 = 0.0;
        for (residual) |e| sum_e += @as(f64, e);
        if (!(sum_e > 0.0)) return error.EmptyAllowedSet;
        const inv_sum_e: f64 = 1.0 / sum_e;
        var norm_sum: f64 = 0.0;
        for (residual) |*e| {
            const p: f32 = @floatCast(@as(f64, e.*) * inv_sum_e);
            e.* = p;
            norm_sum += @as(f64, p);
        }

        const first_child = tree.nodes[u].first_child;
        if (first_child == tree_mod.NO_NODE) {
            const bonus = sampleCategorical(residual, rng);
            if (gram_active) {
                _ = grammar_adapter.?.advanceTokenOnState(grammar_adapter.?.state, bonus);
            }
            if (penalty_tracker) |pt| pt.recordToken(bonus);
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
                    if (alpha > 0.0 and rng.float(f64) < alpha) {
                        accepted_child = c;
                        break;
                    }
                    const inv_q_rem: f64 = 1.0 / q_rem;
                    var z: f64 = 0.0;
                    for (0..vocab_size) |x| {
                        if (x == tok or residual[x] <= 0.0) {
                            residual[x] = 0.0;
                            continue;
                        }
                        const diff = @as(f64, residual[x]) - @as(f64, q_u[x]) * inv_q_rem;
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
                        for (residual) |*rv| {
                            if (rv.* > 0.0) rv.* = @floatCast(@as(f64, rv.*) * inv_z);
                        }
                    } else {
                        residual[argmax_id] = 1.0;
                        break;
                    }
                } else {
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
                            for (residual) |*rv| {
                                if (rv.* > 0.0) rv.* = @floatCast(@as(f64, rv.*) * inv_rem);
                            }
                        } else {
                            residual[argmax_id] = 1.0;
                            break;
                        }
                    }
                }
            }
        } else {
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
                residual[argmax_id] = 1.0;
            }
        }

        if (accepted_child) |child| {
            const ctok = tree.nodes[child].token_id;
            if (gram_active) {
                const ga = grammar_adapter.?;
                const snap = ga.state.snapshot();
                if (!ga.advanceTokenOnState(ga.state, ctok)) {
                    ga.state.restore(snap);
                    return error.EmptyAllowedSet;
                }
            }
            if (penalty_tracker) |pt| pt.recordToken(ctok);
            out_tokens[draft_accepted] = ctok;
            draft_accepted += 1;
            out_accepted_nodes[draft_accepted] = child;
            u = child;
        } else {
            const bonus = sampleCategorical(residual, rng);
            if (gram_active) {
                _ = grammar_adapter.?.advanceTokenOnState(grammar_adapter.?.state, bonus);
            }
            if (penalty_tracker) |pt| pt.recordToken(bonus);
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

test {
    _ = grammar;
    _ = sampling;
}

test "C22 speculative tree branch propagation and verifyConstrained enforce json_schema bitmask losslessly" {
    const vocab = [_]?[]const u8{
        null, // 0: EOS
        "{", // 1
        "\"ok\":", // 2
        "true", // 3
        "false", // 4
        "}", // 5
        "invalid", // 6
        "123", // 7
    };
    const V = vocab.len;
    const eos = [_]u32{0};

    var prog: grammar.GrammarProgram = undefined;
    try grammar.compileJsonSchema(&prog,
        \\{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}
    );

    // 1. Multi-branch DraftTree (`G5`) O(1) snapshot/restore pruning check:
    //    root(0) -> branch A: `{`(1) -> `"ok":`(2) -> `true`(3) -> `}`(5) [ALL VALID]
    //            -> branch B: `{`(1) -> `"ok":`(2) -> `123`(7) -> `}`(5)  [7 & 5 INVALID: ok must be boolean!]
    //            -> branch C: `invalid`(6)                                [INVALID at depth 1]
    var tree = tree_mod.SmallTree.init(0);
    const n_brace = tree.insertChild(0, 1, 0.8, .l1_seed).?;
    const n_bad_root = tree.insertChild(0, 6, 0.2, .l2).?;
    const n_key = tree.insertChild(n_brace, 2, 0.9, .l1_seed).?;
    const n_bad_int = tree.insertChild(n_key, 7, 0.6, .l2).?;
    const n_bad_close = tree.insertChild(n_bad_int, 5, 0.9, .l2).?;
    const n_true = tree.insertChild(n_key, 3, 0.4, .l1_branch).?;
    const n_good_close = tree.insertChild(n_true, 5, 0.9, .l1_branch).?;

    var node_states: [16]grammar.GrammarState = undefined;
    var node_valid: [16]bool = undefined;
    propagateGrammarAcrossTree(16, &tree, &prog, prog.initialState(), &vocab, &eos, &node_states, &node_valid);

    try std.testing.expect(node_valid[n_brace]);
    try std.testing.expect(!node_valid[n_bad_root]);
    try std.testing.expect(node_valid[n_key]);
    try std.testing.expect(!node_valid[n_bad_int]);
    try std.testing.expect(!node_valid[n_bad_close]);
    try std.testing.expect(node_valid[n_true]);
    try std.testing.expect(node_valid[n_good_close]);
    try std.testing.expect(node_states[n_good_close].isAccepting());

    // 2. Even if raw target logits strongly prefer grammar-invalid tokens (`invalid`=6 and `123`=7),
    //    `verifyConstrained` masks them out, rejects `n_bad_int`(7), accepts `n_true`(3) -> `n_good_close`(5),
    //    and emits EOS(0) as the bonus token!
    var logits: [16 * V]f32 = @splat(0.0);
    // At root(0): token 6 ("invalid") has highest raw logit 10.0, valid `{`(1) has 5.0 -> picks 1!
    logits[0 * V + 6] = 10.0;
    logits[0 * V + 1] = 5.0;
    // At n_brace: `"ok":`(2) has 6.0, `invalid`(6) has 9.0 -> picks 2!
    logits[@as(usize, n_brace) * V + 6] = 9.0;
    logits[@as(usize, n_brace) * V + 2] = 6.0;
    // At n_key: `123`(7) has 12.0 (invalid for boolean!), `true`(3) has 4.0 -> picks 3!
    logits[@as(usize, n_key) * V + 7] = 12.0;
    logits[@as(usize, n_key) * V + 3] = 4.0;
    // At n_true: `}`(5) has 5.0 -> picks 5!
    logits[@as(usize, n_true) * V + 5] = 5.0;
    // At n_good_close (accepting!): EOS(0) has 3.0, `invalid`(6) has 8.0 -> picks EOS(0)!
    logits[@as(usize, n_good_close) * V + 6] = 8.0;
    logits[@as(usize, n_good_close) * V + 0] = 3.0;

    var st = prog.initialState();
    var mask_buf: [1]u32 = .{0};
    var scratch_f32: [V]f32 = undefined;
    var scratch_ids: [V]u32 = undefined;
    var acc_nodes: [16]u16 = undefined;
    var out_toks: [16]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(99);

    const res = try verifyConstrained(
        16,
        f32,
        &tree,
        &logits,
        V,
        null,
        .{ .temperature = 0.0 },
        null,
        .{
            .program = &prog,
            .state = &st,
            .token_slices = &vocab,
            .eos_token_ids = &eos,
            .mask_scratch = &mask_buf,
        },
        prng.random(),
        .{ .logits_f32 = &scratch_f32, .cand_ids = &scratch_ids },
        &acc_nodes,
        &out_toks,
    );

    try std.testing.expectEqual(@as(u16, 4), res.draft_accepted);
    try std.testing.expectEqual(@as(u16, 5), res.emitted_token_count);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 3, 5, 0 }, out_toks[0..res.emitted_token_count]);
    try std.testing.expect(st.isAccepting());

    // 3. Stochastic distribution preservation under C22 grammar mask (`{"ok":` -> only `true`(3) or `false`(4) allowed!).
    var st_bool_base = prog.initialState();
    try std.testing.expect(prog.advanceBytes(&st_bool_base, "{\"ok\":"));

    const row = [V]f32{ 2.0, 3.0, 1.0, 1.4, 0.6, 5.0, 9.0, 7.0 };
    // Exact masked probabilities for tokens 3 (`1.4`) and 4 (`0.6`):
    const e3 = @exp(1.4);
    const e4 = @exp(0.6);
    const p3_exact = e3 / (e3 + e4);
    const p4_exact = e4 / (e3 + e4);

    var counts: [V]u64 = @splat(0);
    const rounds: usize = 20_000;
    var bool_tree = tree_mod.SmallTree.init(2);
    _ = bool_tree.insertChild(0, 7, 0.7, .l2); // grammar-invalid sibling `123`!
    _ = bool_tree.insertChild(0, 4, 0.3, .l1_seed); // grammar-valid sibling `false`!
    var two_rows: [3 * V]f32 = @splat(0.0);
    @memcpy(two_rows[0..V], &row);
    // At child `4` (`false`), allow `}`(5) for the bonus token
    two_rows[2 * V + 5] = 1.0;

    for (0..rounds) |_| {
        var st_iter = st_bool_base.snapshot();
        const r_st = try verifyConstrained(
            16,
            f32,
            &bool_tree,
            &two_rows,
            V,
            null,
            .{ .temperature = 1.0 },
            null,
            .{
                .program = &prog,
                .state = &st_iter,
                .token_slices = &vocab,
                .eos_token_ids = &eos,
                .mask_scratch = &mask_buf,
            },
            prng.random(),
            .{ .logits_f32 = &scratch_f32, .cand_ids = &scratch_ids },
            &acc_nodes,
            &out_toks,
        );
        _ = r_st;
        counts[out_toks[0]] += 1;
    }

    // Zero grammar-invalid tokens ever accepted!
    for (0..V) |tok_i| {
        if (tok_i != 3 and tok_i != 4) {
            try std.testing.expectEqual(@as(u64, 0), counts[tok_i]);
        }
    }
    const emp3 = @as(f64, @floatFromInt(counts[3])) / @as(f64, rounds);
    const emp4 = @as(f64, @floatFromInt(counts[4])) / @as(f64, rounds);
    const tvd = 0.5 * (@abs(emp3 - p3_exact) + @abs(emp4 - p4_exact));
    try std.testing.expect(tvd < 0.015);
}

test "F12.2 GrammarTreeAdapter.fillBitmask clamps mismatched token_slices and F20.2 verifyConstrained handles extreme logits without NaN" {
    var prog: grammar.GrammarProgram = undefined;
    try grammar.compileJsonSchema(&prog, "{\"type\":\"object\"}");
    var st = prog.initialState();

    // 1. slices.len = 32 < vocab_size = 64 AND slices.len = 96 > vocab_size = 64
    var slices32: [32]?[]const u8 = @splat("x");
    slices32[0] = "{";
    var slices96: [96]?[]const u8 = @splat("x");
    slices96[0] = "{";
    var mask_scratch: [4]u32 = .{ 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF };

    const ga32 = GrammarTreeAdapter{
        .program = &prog,
        .state = &st,
        .token_slices = &slices32,
        .mask_scratch = &mask_scratch,
    };
    const m64_from_32 = try ga32.fillBitmask(64);
    try std.testing.expectEqual(@as(usize, 2), m64_from_32.len);
    try std.testing.expectEqual(@as(u32, 0x0000_0001), m64_from_32[0]);
    try std.testing.expectEqual(@as(u32, 0), m64_from_32[1]);

    const ga96 = GrammarTreeAdapter{
        .program = &prog,
        .state = &st,
        .token_slices = &slices96,
        .mask_scratch = &mask_scratch,
    };
    const m50_from_96 = try ga96.fillBitmask(50);
    try std.testing.expectEqual(@as(usize, 2), m50_from_96.len);
    try std.testing.expectEqual(@as(u32, 0x0000_0001), m50_from_96[0]);
    try std.testing.expectEqual(@as(u32, 0), m50_from_96[1]);

    // 2. Huge logits + bias + small repetition_penalty = 1e-4 must not produce spurious EmptyAllowedSet!
    var tree = tree_mod.DraftTree(16).init(0);
    _ = tree.insertChild(0, 0, 0.9, .l2).?;
    slices32[1] = "}";
    const target_logits = [_]f32{
        3.0e38, 1.0,    0.5, -1.0,
        1.0,    3.0e38, 0.5, -1.0,
    };
    var pt = sampling.DefaultPenaltyTracker.init();
    pt.recordToken(0);
    var work_logits: [4]f32 = undefined;
    var cand_ids: [4]u32 = undefined;
    var out_nodes: [4]u16 = undefined;
    var out_toks: [4]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(123);

    var st2 = prog.initialState();
    const ga_tiny = GrammarTreeAdapter{
        .program = &prog,
        .state = &st2,
        .token_slices = slices32[0..4],
        .mask_scratch = &mask_scratch,
    };
    const res = try verifyConstrained(
        16,
        f32,
        &tree,
        &target_logits,
        4,
        null,
        .{
            .temperature = 1.0,
            .repetition_penalty = 1e-4,
            .logit_bias = &.{.{ .token_id = 0, .bias = 2.0e38 }},
        },
        &pt,
        ga_tiny,
        prng.random(),
        .{ .logits_f32 = &work_logits, .cand_ids = &cand_ids },
        &out_nodes,
        &out_toks,
    );
    try std.testing.expect(res.emitted_token_count >= 1);
    try std.testing.expectEqual(@as(u32, 0), out_toks[0]);
}
