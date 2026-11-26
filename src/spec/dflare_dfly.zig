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
const vmstat = @import("../platform/vmstat.zig");
const quant = @import("../quant/spec.zig");
const quant_formats = @import("../gpu/quant_formats.zig");
const dequant = @import("../gpu/dequant.zig");
const moe = @import("../gpu/moe.zig");
const metal = @import("../bridge/metal.zig");
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
    metal.WrapError ||
    dequant.InitError ||
    dequant.EncodeError ||
    quant_formats.InitError ||
    quant_formats.EncodeError ||
    quant.ShapeError ||
    error{
        InvalidConfig,
        InvalidWeights,
        DimensionMismatch,
        WorkspaceMismatch,
        CallbackFailed,
        InvalidLogits,
        Overflow,
        UnsupportedDType,
        OutOfMemory,
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
        std.mem.indexOf(u8, json_bytes, "\"dflash_config\"") != null or
        std.mem.indexOf(u8, json_bytes, "\"DFlashDraftModel\"") != null or
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

/// Raw zero-copy 2D matrix view (native `.f32`, `.bf16`, `.f16`, `.gguf_q8_0`,
/// `.q4_affine`, or any `W21`/`W22` quantized `.amf` format) backed by `WiredArena`.
pub const RawLinearMat = struct {
    data: []const u8,
    params: []const u8 = &.{},
    amf_dtype: amf.DType,
    group_size: u16 = 64,
    tensor_scale_per_row: bool = false,
    n: u32,
    k: u32,

    pub fn matches(self: RawLinearMat, expected_n: usize, expected_k: usize) bool {
        if (self.n != expected_n or self.k != expected_k) return false;
        return switch (self.amf_dtype) {
            .f32 => self.data.len >= expected_n * expected_k * 4,
            .f16, .bf16 => self.data.len >= expected_n * expected_k * 2,
            else => blk: {
                const qf_dt = quant_formats.DType.fromAmf(self.amf_dtype) orelse break :blk false;
                qf_dt.validateShape(expected_n, expected_k, self.group_size) catch break :blk false;
                const wb = qf_dt.packedWeightBytes(expected_n, expected_k, self.group_size) catch break :blk false;
                const pb = qf_dt.paramsBytes(expected_n, expected_k, self.group_size) catch break :blk false;
                if (self.data.len < wb) break :blk false;
                if (pb > 0 and self.params.len < pb) break :blk false;
                break :blk true;
            },
        };
    }
};

/// Metal 4 GPU-resident linear matrix binding (`D16`, ADR-0026).
pub const GpuLinearBinding = struct {
    amf_dtype: amf.DType,
    group_size: u16 = 64,
    tensor_scale_per_row: bool = false,
    weight: metal.BufferBinding,
    params: ?metal.BufferBinding = null,
    n: u32,
    k: u32,
};

pub const GpuTapDType = enum(u8) {
    f16 = 0,
    f32 = 1,
};

/// GPU-resident target hidden taps passed directly from target verification / decode
/// into `DFlareProposer` without host `f16 -> f32` staging copies (`D16`, ADR-0026).
pub const GpuTargetTaps = struct {
    dtype: GpuTapDType = .f16,
    taps: []const metal.BufferBinding,
};

pub const GpuHistoryTapDType = enum(u8) {
    f32 = 0,
    f16 = 1,
    bf16 = 2,
};

/// GPU-resident circular ring of committed target context taps (`[cap, num_taps, hidden]`).
pub const GpuContextHistory = struct {
    taps_ring: metal.BufferBinding,
    dtype: GpuHistoryTapDType = .f16,
    cap: u32,
    len: u32,
    epoch: u64,
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
    /// Optional zero-copy raw `.amf` projections (`f16`, `gguf_q8_0`, `q4_affine`, `W21`/`W22`).
    q_proj_raw: ?RawLinearMat = null,
    k_proj_raw: ?RawLinearMat = null,
    v_proj_raw: ?RawLinearMat = null,
    o_proj_raw: ?RawLinearMat = null,
    /// Optional dedicated context KV projections (`layers.{l}.self_attn.{k,v}_proj_ctx`).
    k_proj_ctx: ?[]const f32 = null,
    v_proj_ctx: ?[]const f32 = null,
    k_proj_ctx_bf16: ?[]const u16 = null,
    v_proj_ctx_bf16: ?[]const u16 = null,
    k_proj_ctx_raw: ?RawLinearMat = null,
    v_proj_ctx_raw: ?RawLinearMat = null,
    k_norm_ctx_weight: ?[]const f32 = null,
    ctx_norm_weight: ?[]const f32 = null,
    q_norm_weight: ?[]const f32 = null, // [head_dim]
    k_norm_weight: ?[]const f32 = null, // [head_dim]
    /// Optional per-Q-head `attention_sink_bias` (`[num_attention_heads]`, MiMo-V2.6 `DFlash`).
    sink_bias: ?[]const f32 = null,
    post_norm_weight: ?[]const f32 = null,
    mlp_gate_proj: []const f32 = &.{}, // [intermediate_size, hidden_size]
    mlp_up_proj: []const f32 = &.{}, // [intermediate_size, hidden_size]
    mlp_down_proj: []const f32 = &.{}, // [hidden_size, intermediate_size]
    mlp_gate_bf16: ?[]const u16 = null,
    mlp_up_bf16: ?[]const u16 = null,
    mlp_down_bf16: ?[]const u16 = null,
    mlp_gate_raw: ?RawLinearMat = null,
    mlp_up_raw: ?RawLinearMat = null,
    mlp_down_raw: ?RawLinearMat = null,
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
            const q_ok = (self.q_proj.len == qh) or (if (self.q_proj_bf16) |b| b.len == qh else false) or (if (self.q_proj_raw) |r| r.matches(q_dim, hidden_size) else false);
            const k_ok = (self.k_proj.len == kvh) or (if (self.k_proj_bf16) |b| b.len == kvh else false) or (if (self.k_proj_raw) |r| r.matches(kv_dim, hidden_size) else false);
            const v_ok = (self.v_proj.len == kvh) or (if (self.v_proj_bf16) |b| b.len == kvh else false) or (if (self.v_proj_raw) |r| r.matches(kv_dim, hidden_size) else false);
            const o_ok = (self.o_proj.len == hq) or (if (self.o_proj_bf16) |b| b.len == hq else false) or (if (self.o_proj_raw) |r| r.matches(hidden_size, q_dim) else false);
            if (!q_ok or !k_ok or !v_ok or !o_ok) {
                return error.InvalidWeights;
            }
            if (self.k_proj_ctx_raw) |r| if (!r.matches(kv_dim, hidden_size)) return error.InvalidWeights;
            if (self.v_proj_ctx_raw) |r| if (!r.matches(kv_dim, hidden_size)) return error.InvalidWeights;
            if (self.q_norm_weight) |w| {
                if (w.len != head_dim) return error.InvalidWeights;
            }
            if (self.k_norm_weight) |w| {
                if (w.len != head_dim) return error.InvalidWeights;
            }
            if (self.sink_bias) |sb| {
                const num_q_heads = q_dim / head_dim;
                if (sb.len != num_q_heads) return error.InvalidWeights;
            }
        }
        if (self.post_norm_weight) |w| {
            if (w.len != hidden_size) return error.InvalidWeights;
        }
        const g_ok = (self.mlp_gate_proj.len == ih) or (if (self.mlp_gate_bf16) |b| b.len == ih else false) or (if (self.mlp_gate_raw) |r| r.matches(intermediate_size, hidden_size) else false);
        const u_ok = (self.mlp_up_proj.len == ih) or (if (self.mlp_up_bf16) |b| b.len == ih else false) or (if (self.mlp_up_raw) |r| r.matches(intermediate_size, hidden_size) else false);
        const d_ok = (self.mlp_down_proj.len == hi) or (if (self.mlp_down_bf16) |b| b.len == hi else false) or (if (self.mlp_down_raw) |r| r.matches(hidden_size, intermediate_size) else false);
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
    /// `0` enables direct parallel block-diffusion scoring through target `lm_head` (`D15` `DFlash`),
    /// while `> 0` runs the `Inkling 2` / `dflash-Q8` low-rank Markov head.
    markov_rank: u32 = DFLARE_MARKOV_RANK,
    num_target_taps: u8 = 3,
    target_layer_ids: [MAX_TARGET_TAPS]u16 = .{ 2, 11, 20, 0, 0, 0, 0, 0 },
    rms_eps: f32 = 1e-6,
    rope_theta: f32 = 0.0,
    /// Optional partial NeoX RoPE dimension (`0` => full `head_dim`; `64` on MiMo-V2.6 `DFlash`).
    rope_dim: u32 = 0,
    /// Optional attention value scale (`1.0` default; `0.612` on MiMo-V2.6 `DFlash`).
    attention_value_scale: f32 = 1.0,
    /// Optional causal sliding window over prior accepted context KV (`0` => full ctx ring; `1024` on MiMo-V2.6 `DFlash`).
    sliding_window: u32 = 0,
    /// Optional mask token ID (`151675` on MiMo-V2.6 `DFlash`) used to initialize slots `1..B-1` via `embed(mask_token_id)`.
    mask_token_id: ?u32 = null,
    /// Direct block-diffusion refinement passes (`1` = single-pass `8..16` block; `2` = optional 2-iteration refinement).
    refinement_iters: u8 = 1,
    /// Probability threshold for retaining predicted tokens in iteration 2 of direct block-diffusion refinement.
    refine_confidence_threshold: f32 = 0.5,
    /// When `> 0.0`, forks a runner-up top-2 branch at uncertain direct block-diffusion slots (`top1_prob < threshold`)
    /// up to `max_tree_nodes` (e.g. `16`).
    fork_top2_threshold: f32 = 0.0,
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

    pub inline fn effectiveRopeDim(self: DFlareConfig) u32 {
        return if (self.rope_dim > 0) self.rope_dim else self.head_dim;
    }

    pub inline fn isDirectBlockDiffusion(self: DFlareConfig) bool {
        return self.markov_rank == 0;
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
        const rdim = self.effectiveRopeDim();
        if (rdim == 0 or rdim > self.head_dim or (rdim % 2) != 0) return error.InvalidConfig;
        if (!std.math.isFinite(self.attention_value_scale) or self.attention_value_scale <= 0.0) return error.InvalidConfig;
        if (self.refinement_iters == 0 or self.refinement_iters > 2) return error.InvalidConfig;
        if (!std.math.isFinite(self.refine_confidence_threshold) or self.refine_confidence_threshold < 0.0 or self.refine_confidence_threshold > 1.0) return error.InvalidConfig;
        if (!std.math.isFinite(self.fork_top2_threshold) or self.fork_top2_threshold < 0.0 or self.fork_top2_threshold > 1.0) return error.InvalidConfig;
        if (self.mask_token_id) |mtid| {
            if (mtid >= self.vocab_size) return error.InvalidConfig;
        }
        if (self.num_target_taps == 0 or self.num_target_taps > MAX_TARGET_TAPS) return error.InvalidConfig;
        if (!std.math.isFinite(self.rms_eps) or self.rms_eps <= 0.0) return error.InvalidConfig;
        if (!std.math.isFinite(self.temperature) or self.temperature <= 0.0) return error.InvalidConfig;
        if (self.l1_max_draft_len > MAX_L1_DRAFT_LEN) return error.InvalidConfig;
    }

    /// Parses `dflare_config.json`, HF `dflash/config.json`, or GGUF `arch=dflash` config JSON with target fallbacks.
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
        const df_obj: ?std.json.ObjectMap = if (obj.get("dflash_config")) |dv| (if (dv == .object) dv.object else null) else null;

        const is_dflash_arch: bool = blk: {
            if (df_obj != null) break :blk true;
            if (obj.get("draft_kind")) |dk| {
                if (dk == .string and std.mem.eql(u8, dk.string, "dflash")) break :blk true;
            }
            if (obj.get("architectures")) |av| {
                if (av == .array and av.array.items.len > 0 and av.array.items[0] == .string) {
                    if (std.mem.indexOf(u8, av.array.items[0].string, "DFlash") != null) break :blk true;
                }
            }
            break :blk false;
        };

        const num_layers: u8 = @intCast(jsonU32Opt(obj, "num_hidden_layers") orelse (if (is_dflash_arch) @as(u32, 5) else 3));
        const hidden_size: u32 = jsonU32Opt(obj, "hidden_size") orelse fallback_hidden;
        const inter_size: u32 = jsonU32Opt(obj, "intermediate_size") orelse (hidden_size * 4);
        const num_heads: u32 = jsonU32Opt(obj, "num_attention_heads") orelse 8;
        const num_kv_heads: u32 = jsonU32Opt(obj, "num_key_value_heads") orelse num_heads;
        const head_dim: u32 = jsonU32Opt(obj, "head_dim") orelse (hidden_size / @max(num_heads, 1));
        const vocab_size: u32 = if (fallback_vocab > 0) fallback_vocab else (jsonU32Opt(obj, "vocab_size") orelse 0);
        const block_size_raw: u32 = jsonU32Opt(obj, "block_size") orelse
            (if (df_obj) |dfo| jsonU32Opt(dfo, "block_size") else null) orelse DFLARE_BLOCK_SIZE;
        const block_size: u8 = @intCast(block_size_raw);
        const markov_rank_opt = jsonU32Opt(obj, "markov_rank");
        const markov_rank: u32 = markov_rank_opt orelse (if (is_dflash_arch) @as(u32, 0) else DFLARE_MARKOV_RANK);
        const rms_eps: f32 = jsonF32Opt(obj, "rms_norm_eps") orelse 1e-6;
        const rope_theta: f32 = jsonF32Opt(obj, "rope_theta") orelse (if (is_dflash_arch) @as(f32, 10_000.0) else 0.0);
        const explicit_rope_dim = jsonU32Opt(obj, "rope_dim") orelse jsonU32Opt(obj, "rotary_dim");
        const partial_rotary_factor: f32 = jsonF32Opt(obj, "partial_rotary_factor") orelse 1.0;
        const rope_dim: u32 = if (explicit_rope_dim) |rd|
            rd
        else if (partial_rotary_factor > 0.0 and partial_rotary_factor < 1.0) blk: {
            const raw: u32 = @intFromFloat(@floor(@as(f32, @floatFromInt(head_dim)) * partial_rotary_factor));
            break :blk raw & ~@as(u32, 1);
        } else head_dim;
        const v_scale: f32 = jsonF32Opt(obj, "attention_value_scale") orelse
            (if (df_obj) |dfo| jsonF32Opt(dfo, "attention_value_scale") else null) orelse 1.0;
        const sliding_window: u32 = jsonU32Opt(obj, "sliding_window") orelse
            jsonU32Opt(obj, "sliding_window_size") orelse 0;
        const mask_token_id: ?u32 = jsonU32Opt(obj, "mask_token_id") orelse
            (if (df_obj) |dfo| jsonU32Opt(dfo, "mask_token_id") else null);
        const refinement_iters: u8 = @intCast(jsonU32Opt(obj, "refinement_iters") orelse 1);
        const refine_conf: f32 = jsonF32Opt(obj, "refine_confidence_threshold") orelse 0.5;
        const fork_top2: f32 = jsonF32Opt(obj, "fork_top2_threshold") orelse 0.0;
        const unpadded_vocab: u32 = jsonU32Opt(obj, "unpadded_vocab_size") orelse 0;
        const final_softcap: f32 = jsonF32Opt(obj, "final_logit_softcapping") orelse 0.0;

        var tap_ids: [MAX_TARGET_TAPS]u16 = .{ 5, 23, 41, 48, 0, 0, 0, 0 };
        var num_taps: u8 = 4;
        const tap_arr_opt = obj.get("target_layer_ids") orelse (if (df_obj) |dfo| dfo.get("target_layer_ids") else null);
        if (tap_arr_opt) |v| {
            if (v == .array and v.array.items.len > 0 and v.array.items.len <= MAX_TARGET_TAPS) {
                num_taps = @intCast(v.array.items.len);
                for (v.array.items, 0..) |item, i| {
                    if (item == .integer and item.integer >= 0) {
                        tap_ids[i] = @intCast(item.integer);
                    }
                }
            }
        }
        const variant: DFlareVariant = if (is_dflash_arch)
            .dflash_standalone
        else if (num_layers >= 5)
            .inkling2_small_5l
        else
            .inkling2_mini_3l;

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
            .rope_dim = rope_dim,
            .attention_value_scale = v_scale,
            .sliding_window = sliding_window,
            .mask_token_id = mask_token_id,
            .refinement_iters = refinement_iters,
            .refine_confidence_threshold = refine_conf,
            .fork_top2_threshold = fork_top2,
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
    target_fusion_proj_raw: ?RawLinearMat = null,
    /// Optional per-layer scalar tap mixing weights `[rows, num_target_taps]`
    /// (real `Inkling-2-Mini` `fuse_weight [4, 4]` & `dflash-Q8` `ctx_mix`).
    fuse_scalar_weights: ?[]const f32 = null,
    target_fusion_norm: ?[]const f32 = null,
    /// Mask embedding: `[block_size, hidden_size]`, `[(block_size - 1), hidden_size]`, shared `[hidden_size]`,
    /// or empty when `cfg.mask_token_id != null` (resolved from `embed(mask_token_id)`).
    mask_embed: []const f32 = &.{},
    /// Token embeddings `[vocab_size, hidden_size]` for sequential Markov conditioning (or empty when `tok_embed_hook` is set).
    tok_embed: []const f32 = &.{},
    tok_embed_raw: ?RawLinearMat = null,
    tok_embed_hook: ?TokEmbedHook = null,
    token_embd_norm: ?[]const f32 = null,
    /// `3L` / `5L` block-parallel transformer layers.
    layers: []const TransformerBlockWeights,
    final_norm: ?[]const f32 = null,
    /// Low-rank Markov head `markov_w1`: `[markov_rank, 2*H]`, `[markov_rank, H]`, or empty when `markov_w1_vocab_bf16` is set.
    markov_w1: []const f32 = &.{},
    markov_w1_raw: ?RawLinearMat = null,
    /// Optional token-ID low-rank embedding table `[vocab_rows, markov_rank]` BF16 (real `Inkling-2-Mini`).
    markov_w1_vocab_bf16: ?[]const u16 = null,
    markov_w1_vocab_raw: ?RawLinearMat = null,
    markov_w1_vocab_rows: u32 = 0,
    /// Optional 3-gate GRU joint projection `[3 * markov_rank, 2*H + 2*markov_rank]` (`768 x 5120`).
    markov_gru_proj_bf16: ?[]const u16 = null,
    markov_gru_proj_f32: ?[]const f32 = null,
    markov_gru_proj_raw: ?RawLinearMat = null,
    markov_gru_bias: ?[]const f32 = null,
    markov_ctx_norm: ?[]const f32 = null,
    /// Optional recurrent transition `[markov_rank, markov_rank]` on `rnn_hidden`.
    markov_rnn_proj: ?[]const f32 = null,
    markov_rnn_proj_raw: ?RawLinearMat = null,
    markov_b1: ?[]const f32 = null,
    /// Output bottleneck expansion `markov_w2`: `[hidden_size, markov_rank]` (or empty when `markov_w2_vocab_bf16` is set).
    markov_w2: []const f32 = &.{},
    markov_w2_raw: ?RawLinearMat = null,
    markov_hidden_bias: ?[]const f32 = null,
    /// Optional direct low-rank vocab logits head `[vocab_rows, markov_rank]` (`dflash-Q8` GGUF).
    markov_w2_vocab_bf16: ?[]const u16 = null,
    markov_w2_vocab_raw: ?RawLinearMat = null,
    markov_w2_vocab_bias: ?[]const f32 = null,
    markov_w2_vocab_rows: u32 = 0,
    markov_norm: ?[]const f32 = null,
    lm_head: LmHeadHook,
    lm_head_raw: ?RawLinearMat = null,
    /// Optional zero-copy `WiredArena` backing slice owning the loaded `.amf` tensors.
    wired_arena_bytes: ?[]align(page.APPLE_PAGE_SIZE) u8 = null,

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
        } else if (self.target_fusion_proj_raw) |fpr| {
            if (!fpr.matches(h, taps * h)) return error.InvalidWeights;
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
            (cfg.block_size > 1 and self.mask_embed.len == (std.math.mul(usize, cfg.block_size - 1, h) catch return error.Overflow)) or
            (self.mask_embed.len == 0 and cfg.mask_token_id != null);
        if (!mask_ok) return error.InvalidWeights;

        const vh = std.math.mul(usize, v, h) catch return error.Overflow;
        if (self.tok_embed_hook == null) {
            const tok_ok = (self.tok_embed.len == vh) or (if (self.tok_embed_raw) |tr| tr.matches(v, h) else false);
            if (!tok_ok) return error.InvalidWeights;
        }
        if (self.layers.len != cfg.num_layers) return error.InvalidWeights;
        for (self.layers) |layer| {
            try layer.validate(h, inter, q_dim, kv_dim, cfg.head_dim);
        }
        if (self.final_norm) |fnw| {
            if (fnw.len != h) return error.InvalidWeights;
        }

        if (mr == 0) {
            // Direct parallel block-diffusion mode (`D15` `DFlash`): no Markov head; scores via target `lm_head`.
            if (self.lm_head_raw) |lhr| {
                if (!lhr.matches(v, h)) return error.InvalidWeights;
            } else {
                try validateLmHeadHook(self.lm_head, vh);
            }
            return;
        }

        const has_gru = (self.markov_gru_proj_bf16 != null or self.markov_gru_proj_f32 != null or self.markov_gru_proj_raw != null);
        const mr_h = std.math.mul(usize, mr, h) catch return error.Overflow;
        if (!has_gru) {
            const mr_2h = std.math.mul(usize, mr_h, 2) catch return error.Overflow;
            const w1_ok = (self.markov_w1.len == mr_h or self.markov_w1.len == mr_2h) or
                (if (self.markov_w1_raw) |r| (r.matches(mr, h) or r.matches(mr, 2 * h)) else false);
            if (!w1_ok) {
                return error.InvalidWeights;
            }
            if (self.markov_rnn_proj) |rp| {
                const mr2 = std.math.mul(usize, mr, mr) catch return error.Overflow;
                if (rp.len != mr2) return error.InvalidWeights;
            } else if (self.markov_rnn_proj_raw) |rpr| {
                if (!rpr.matches(mr, mr)) return error.InvalidWeights;
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
            } else if (self.markov_gru_proj_raw) |gr| {
                if (!gr.matches(3 * mr, joint_in)) return error.InvalidWeights;
            }
            const has_w1v = (self.markov_w1_vocab_bf16 != null or self.markov_w1_vocab_raw != null);
            if (!has_w1v or self.markov_w1_vocab_rows == 0) return error.InvalidWeights;
        }
        const has_w2v = (self.markov_w2_vocab_bf16 != null or self.markov_w2_vocab_raw != null);
        if (!has_w2v) {
            const w2_ok = (self.markov_w2.len == mr_h) or (if (self.markov_w2_raw) |r| r.matches(h, mr) else false);
            if (!w2_ok) return error.InvalidWeights;
        }
        if (self.markov_norm) |mn| {
            if (mn.len != h) return error.InvalidWeights;
        }
        if (!has_w2v and self.lm_head_raw == null) {
            try validateLmHeadHook(self.lm_head, vh);
        } else if (self.lm_head_raw) |lhr| {
            if (!lhr.matches(v, h)) return error.InvalidWeights;
        }
    }
};

/// Preallocated 16 KiB-aligned `WiredArena` workspace for `DFlareProposer`
/// (shared zero-copy CPU + Metal 4 GPU physical pages).
pub const DFlareWorkspace = struct {
    arena: arena_mod.WiredArena,
    config: DFlareConfig,
    max_ctx_len: u32,
    ctx_batch_cap: u32,
    taps_flat: []f32,
    ctx_hidden: []f32,
    ctx_markov: []f32,
    tok_embed_tmp: []f32,
    mask_embed_tmp: []f32,
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
    // GPU zero-copy slices inside `self.arena` (`64`-byte aligned):
    gpu_taps_flat_f16: []f16,
    gpu_ctx_taps_f32: []f32,
    gpu_ctx_taps_f16: []f16,
    gpu_ctx_in_f16: []f16,
    gpu_ctx_k_tmp_f16: []f16,
    gpu_ctx_v_tmp_f16: []f16,
    gpu_ctx_hidden_f16: []f16,
    gpu_ctx_markov_f16: []f16,
    gpu_ctx_markov_f32: []f32,
    gpu_anchor_f16: []f16,
    gpu_slot_hiddens_f16: []f16,
    gpu_norm_f16: []f16,
    gpu_q_f16: []f16,
    gpu_k_f16: []f16,
    gpu_v_f16: []f16,
    gpu_attn_out_f16: []f16,
    gpu_mlp_gate_f16: []f16,
    gpu_mlp_up_f16: []f16,
    gpu_mlp_down_f16: []f16,
    gpu_markov_concat_f16: []f16,
    gpu_markov_concat_f32: []f32,
    gpu_markov_gates_f16: []f16,
    gpu_markov_gates_f32: []f32,
    gpu_markov_rnn_ping_f32: []f32,
    gpu_markov_vec_f16: []f16,
    gpu_markov_vec_f32: []f32,
    gpu_head_hidden_f16: []f16,
    gpu_base_logits_f16: []f16,
    gpu_logits_f16: []f16,
    gpu_logits_f32: []f32,
    gpu_topk_cands: []moe.TopKCandidate,
    gpu_token_chain_u32: []u32,
    gpu_grammar_mask_u32: []u32,
    gpu_w1_row_f32: []f32,
    cpu_quant_row_scratch: []f32,

    pub fn requiredBytes(cfg: DFlareConfig) Error!usize {
        try cfg.validate();
        const h: usize = cfg.hidden_size;
        const inter: usize = cfg.intermediate_size;
        const w: usize = cfg.block_size;
        const v: usize = cfg.vocab_size;
        const mr: usize = @max(@as(usize, cfg.markov_rank), 1);
        const topk_cap: usize = @max(@as(usize, cfg.top_k_per_pos), 2);
        const taps: usize = cfg.num_target_taps;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();
        const max_nodes: usize = cfg.max_tree_nodes;
        const max_ctx: usize = if (cfg.sliding_window > 0) @min(@as(usize, cfg.sliding_window), 1024) else if (cfg.hidden_size >= 1024) 1024 else 64;
        const ctx_batch_cap: usize = @min(max_ctx, 64);
        const ctx_kv_elems: usize = @as(usize, cfg.num_layers) * max_ctx * kv_dim;
        const mask_words: usize = (v + 31) / 32;
        const row_scratch_elems: usize = @max(taps * h, @max(h, @max(inter, @max(q_dim, 2 * h + 2 * mr))));

        const chunks = [_]usize{
            try alignUp64((taps * h) * @sizeOf(f32)), // taps_flat
            try alignUp64(h * @sizeOf(f32)), // ctx_hidden
            try alignUp64(h * @sizeOf(f32)), // ctx_markov
            try alignUp64(h * @sizeOf(f32)), // tok_embed_tmp
            try alignUp64(h * @sizeOf(f32)), // mask_embed_tmp
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
            try alignUp64((3 * mr) * @sizeOf(f32)), // markov_u
            try alignUp64(mr * @sizeOf(f32)), // markov_out
            try alignUp64((2 * h + 2 * mr) * @sizeOf(f32)), // markov_concat
            try alignUp64(h * @sizeOf(f32)), // head_hidden
            try alignUp64((w * v) * @sizeOf(f32)), // base_logits_buf
            try alignUp64(v * @sizeOf(f32)), // logits_buf
            try alignUp64(topk_cap * @sizeOf(u32)), // topk_tokens
            try alignUp64(topk_cap * @sizeOf(f32)), // topk_probs
            try alignUp64(max_nodes * @sizeOf(f32)), // node_log_probs
            try alignUp64(MAX_L1_DRAFT_LEN * @sizeOf(u32)), // l1_draft_buf
            try alignUp64((taps * h) * @sizeOf(f16)), // gpu_taps_flat_f16
            try alignUp64((ctx_batch_cap * taps * h) * @sizeOf(f32)), // gpu_ctx_taps_f32
            try alignUp64((ctx_batch_cap * taps * h) * @sizeOf(f16)), // gpu_ctx_taps_f16
            try alignUp64((ctx_batch_cap * h) * @sizeOf(f16)), // gpu_ctx_in_f16
            try alignUp64((ctx_batch_cap * kv_dim) * @sizeOf(f16)), // gpu_ctx_k_tmp_f16
            try alignUp64((ctx_batch_cap * kv_dim) * @sizeOf(f16)), // gpu_ctx_v_tmp_f16
            try alignUp64(h * @sizeOf(f16)), // gpu_ctx_hidden_f16
            try alignUp64(h * @sizeOf(f16)), // gpu_ctx_markov_f16
            try alignUp64(h * @sizeOf(f32)), // gpu_ctx_markov_f32
            try alignUp64(h * @sizeOf(f16)), // gpu_anchor_f16
            try alignUp64((w * h) * @sizeOf(f16)), // gpu_slot_hiddens_f16
            try alignUp64((w * h) * @sizeOf(f16)), // gpu_norm_f16
            try alignUp64((w * q_dim) * @sizeOf(f16)), // gpu_q_f16
            try alignUp64((w * kv_dim) * @sizeOf(f16)), // gpu_k_f16
            try alignUp64((w * kv_dim) * @sizeOf(f16)), // gpu_v_f16
            try alignUp64((w * q_dim) * @sizeOf(f16)), // gpu_attn_out_f16
            try alignUp64((w * inter) * @sizeOf(f16)), // gpu_mlp_gate_f16
            try alignUp64((w * inter) * @sizeOf(f16)), // gpu_mlp_up_f16
            try alignUp64((w * h) * @sizeOf(f16)), // gpu_mlp_down_f16
            try alignUp64((2 * h + 2 * mr) * @sizeOf(f16)), // gpu_markov_concat_f16
            try alignUp64((2 * h + 2 * mr) * @sizeOf(f32)), // gpu_markov_concat_f32
            try alignUp64((3 * mr) * @sizeOf(f16)), // gpu_markov_gates_f16
            try alignUp64((3 * mr) * @sizeOf(f32)), // gpu_markov_gates_f32
            try alignUp64((2 * mr) * @sizeOf(f32)), // gpu_markov_rnn_ping_f32
            try alignUp64(mr * @sizeOf(f16)), // gpu_markov_vec_f16
            try alignUp64(mr * @sizeOf(f32)), // gpu_markov_vec_f32
            try alignUp64(h * @sizeOf(f16)), // gpu_head_hidden_f16
            try alignUp64((w * v) * @sizeOf(f16)), // gpu_base_logits_f16
            try alignUp64(v * @sizeOf(f16)), // gpu_logits_f16
            try alignUp64(v * @sizeOf(f32)), // gpu_logits_f32
            try alignUp64((w * MAX_TOP_K_PER_POS) * @sizeOf(moe.TopKCandidate)), // gpu_topk_cands
            try alignUp64((w + 1) * @sizeOf(u32)), // gpu_token_chain_u32
            try alignUp64(mask_words * @sizeOf(u32)), // gpu_grammar_mask_u32
            try alignUp64(mr * @sizeOf(f32)), // gpu_w1_row_f32
            try alignUp64(row_scratch_elems * @sizeOf(f32)), // cpu_quant_row_scratch
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
        const mr: usize = @max(@as(usize, cfg.markov_rank), 1);
        const topk_cap: usize = @max(@as(usize, cfg.top_k_per_pos), 2);
        const taps: usize = cfg.num_target_taps;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();
        const max_nodes: usize = cfg.max_tree_nodes;
        const max_ctx: usize = if (cfg.sliding_window > 0) @min(@as(usize, cfg.sliding_window), 1024) else if (cfg.hidden_size >= 1024) 1024 else 64;
        const ctx_batch_cap: usize = @min(max_ctx, 64);
        const ctx_kv_elems: usize = @as(usize, cfg.num_layers) * max_ctx * kv_dim;
        const mask_words: usize = (v + 31) / 32;
        const row_scratch_elems: usize = @max(taps * h, @max(h, @max(inter, @max(q_dim, 2 * h + 2 * mr))));

        const taps_flat = try allocArenaSlice(f32, &mem_arena, taps * h);
        const ctx_hidden = try allocArenaSlice(f32, &mem_arena, h);
        const ctx_markov = try allocArenaSlice(f32, &mem_arena, h);
        const tok_embed_tmp = try allocArenaSlice(f32, &mem_arena, h);
        const mask_embed_tmp = try allocArenaSlice(f32, &mem_arena, h);
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
        const topk_tokens = try allocArenaSlice(u32, &mem_arena, topk_cap);
        const topk_probs = try allocArenaSlice(f32, &mem_arena, topk_cap);
        const node_log_probs = try allocArenaSlice(f32, &mem_arena, max_nodes);
        const l1_draft_buf = try allocArenaSlice(u32, &mem_arena, MAX_L1_DRAFT_LEN);

        const gpu_taps_flat_f16 = try allocArenaSlice(f16, &mem_arena, taps * h);
        const gpu_ctx_taps_f32 = try allocArenaSlice(f32, &mem_arena, ctx_batch_cap * taps * h);
        const gpu_ctx_taps_f16 = try allocArenaSlice(f16, &mem_arena, ctx_batch_cap * taps * h);
        const gpu_ctx_in_f16 = try allocArenaSlice(f16, &mem_arena, ctx_batch_cap * h);
        const gpu_ctx_k_tmp_f16 = try allocArenaSlice(f16, &mem_arena, ctx_batch_cap * kv_dim);
        const gpu_ctx_v_tmp_f16 = try allocArenaSlice(f16, &mem_arena, ctx_batch_cap * kv_dim);
        const gpu_ctx_hidden_f16 = try allocArenaSlice(f16, &mem_arena, h);
        const gpu_ctx_markov_f16 = try allocArenaSlice(f16, &mem_arena, h);
        const gpu_ctx_markov_f32 = try allocArenaSlice(f32, &mem_arena, h);
        const gpu_anchor_f16 = try allocArenaSlice(f16, &mem_arena, h);
        const gpu_slot_hiddens_f16 = try allocArenaSlice(f16, &mem_arena, w * h);
        const gpu_norm_f16 = try allocArenaSlice(f16, &mem_arena, w * h);
        const gpu_q_f16 = try allocArenaSlice(f16, &mem_arena, w * q_dim);
        const gpu_k_f16 = try allocArenaSlice(f16, &mem_arena, w * kv_dim);
        const gpu_v_f16 = try allocArenaSlice(f16, &mem_arena, w * kv_dim);
        const gpu_attn_out_f16 = try allocArenaSlice(f16, &mem_arena, w * q_dim);
        const gpu_mlp_gate_f16 = try allocArenaSlice(f16, &mem_arena, w * inter);
        const gpu_mlp_up_f16 = try allocArenaSlice(f16, &mem_arena, w * inter);
        const gpu_mlp_down_f16 = try allocArenaSlice(f16, &mem_arena, w * h);
        const gpu_markov_concat_f16 = try allocArenaSlice(f16, &mem_arena, 2 * h + 2 * mr);
        const gpu_markov_concat_f32 = try allocArenaSlice(f32, &mem_arena, 2 * h + 2 * mr);
        const gpu_markov_gates_f16 = try allocArenaSlice(f16, &mem_arena, 3 * mr);
        const gpu_markov_gates_f32 = try allocArenaSlice(f32, &mem_arena, 3 * mr);
        const gpu_markov_rnn_ping_f32 = try allocArenaSlice(f32, &mem_arena, 2 * mr);
        const gpu_markov_vec_f16 = try allocArenaSlice(f16, &mem_arena, mr);
        const gpu_markov_vec_f32 = try allocArenaSlice(f32, &mem_arena, mr);
        const gpu_head_hidden_f16 = try allocArenaSlice(f16, &mem_arena, h);
        const gpu_base_logits_f16 = try allocArenaSlice(f16, &mem_arena, w * v);
        const gpu_logits_f16 = try allocArenaSlice(f16, &mem_arena, v);
        const gpu_logits_f32 = try allocArenaSlice(f32, &mem_arena, v);
        const gpu_topk_cands = try allocArenaSlice(moe.TopKCandidate, &mem_arena, w * MAX_TOP_K_PER_POS);
        const gpu_token_chain_u32 = try allocArenaSlice(u32, &mem_arena, w + 1);
        const gpu_grammar_mask_u32 = try allocArenaSlice(u32, &mem_arena, mask_words);
        const gpu_w1_row_f32 = try allocArenaSlice(f32, &mem_arena, mr);
        const cpu_quant_row_scratch = try allocArenaSlice(f32, &mem_arena, row_scratch_elems);

        @memset(mask_embed_tmp, 0.0);
        @memset(node_log_probs, 0.0);
        @memset(node_rnn_hidden, 0.0);
        @memset(ctx_k_cache, 0.0);
        @memset(ctx_v_cache, 0.0);
        @memset(gpu_markov_rnn_ping_f32, 0.0);
        @memset(gpu_token_chain_u32, 0);

        return .{
            .arena = mem_arena,
            .config = cfg,
            .max_ctx_len = @intCast(max_ctx),
            .ctx_batch_cap = @intCast(ctx_batch_cap),
            .taps_flat = taps_flat,
            .ctx_hidden = ctx_hidden,
            .ctx_markov = ctx_markov,
            .tok_embed_tmp = tok_embed_tmp,
            .mask_embed_tmp = mask_embed_tmp,
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
            .gpu_taps_flat_f16 = gpu_taps_flat_f16,
            .gpu_ctx_taps_f32 = gpu_ctx_taps_f32,
            .gpu_ctx_taps_f16 = gpu_ctx_taps_f16,
            .gpu_ctx_in_f16 = gpu_ctx_in_f16,
            .gpu_ctx_k_tmp_f16 = gpu_ctx_k_tmp_f16,
            .gpu_ctx_v_tmp_f16 = gpu_ctx_v_tmp_f16,
            .gpu_ctx_hidden_f16 = gpu_ctx_hidden_f16,
            .gpu_ctx_markov_f16 = gpu_ctx_markov_f16,
            .gpu_ctx_markov_f32 = gpu_ctx_markov_f32,
            .gpu_anchor_f16 = gpu_anchor_f16,
            .gpu_slot_hiddens_f16 = gpu_slot_hiddens_f16,
            .gpu_norm_f16 = gpu_norm_f16,
            .gpu_q_f16 = gpu_q_f16,
            .gpu_k_f16 = gpu_k_f16,
            .gpu_v_f16 = gpu_v_f16,
            .gpu_attn_out_f16 = gpu_attn_out_f16,
            .gpu_mlp_gate_f16 = gpu_mlp_gate_f16,
            .gpu_mlp_up_f16 = gpu_mlp_up_f16,
            .gpu_mlp_down_f16 = gpu_mlp_down_f16,
            .gpu_markov_concat_f16 = gpu_markov_concat_f16,
            .gpu_markov_concat_f32 = gpu_markov_concat_f32,
            .gpu_markov_gates_f16 = gpu_markov_gates_f16,
            .gpu_markov_gates_f32 = gpu_markov_gates_f32,
            .gpu_markov_rnn_ping_f32 = gpu_markov_rnn_ping_f32,
            .gpu_markov_vec_f16 = gpu_markov_vec_f16,
            .gpu_markov_vec_f32 = gpu_markov_vec_f32,
            .gpu_head_hidden_f16 = gpu_head_hidden_f16,
            .gpu_base_logits_f16 = gpu_base_logits_f16,
            .gpu_logits_f16 = gpu_logits_f16,
            .gpu_logits_f32 = gpu_logits_f32,
            .gpu_topk_cands = gpu_topk_cands,
            .gpu_token_chain_u32 = gpu_token_chain_u32,
            .gpu_grammar_mask_u32 = gpu_grammar_mask_u32,
            .gpu_w1_row_f32 = gpu_w1_row_f32,
            .cpu_quant_row_scratch = cpu_quant_row_scratch,
        };
    }

    pub fn deinit(self: *DFlareWorkspace) void {
        self.taps_flat = &.{};
        self.ctx_hidden = &.{};
        self.ctx_markov = &.{};
        self.tok_embed_tmp = &.{};
        self.mask_embed_tmp = &.{};
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
        self.gpu_taps_flat_f16 = &.{};
        self.gpu_ctx_taps_f32 = &.{};
        self.gpu_ctx_taps_f16 = &.{};
        self.gpu_ctx_in_f16 = &.{};
        self.gpu_ctx_k_tmp_f16 = &.{};
        self.gpu_ctx_v_tmp_f16 = &.{};
        self.gpu_ctx_hidden_f16 = &.{};
        self.gpu_ctx_markov_f16 = &.{};
        self.gpu_ctx_markov_f32 = &.{};
        self.gpu_anchor_f16 = &.{};
        self.gpu_slot_hiddens_f16 = &.{};
        self.gpu_norm_f16 = &.{};
        self.gpu_q_f16 = &.{};
        self.gpu_k_f16 = &.{};
        self.gpu_v_f16 = &.{};
        self.gpu_attn_out_f16 = &.{};
        self.gpu_mlp_gate_f16 = &.{};
        self.gpu_mlp_up_f16 = &.{};
        self.gpu_mlp_down_f16 = &.{};
        self.gpu_markov_concat_f16 = &.{};
        self.gpu_markov_concat_f32 = &.{};
        self.gpu_markov_gates_f16 = &.{};
        self.gpu_markov_gates_f32 = &.{};
        self.gpu_markov_rnn_ping_f32 = &.{};
        self.gpu_markov_vec_f16 = &.{};
        self.gpu_markov_vec_f32 = &.{};
        self.gpu_head_hidden_f16 = &.{};
        self.gpu_base_logits_f16 = &.{};
        self.gpu_logits_f16 = &.{};
        self.gpu_logits_f32 = &.{};
        self.gpu_topk_cands = &.{};
        self.gpu_token_chain_u32 = &.{};
        self.gpu_grammar_mask_u32 = &.{};
        self.gpu_w1_row_f32 = &.{};
        self.cpu_quant_row_scratch = &.{};
        self.arena.deinit();
    }

    pub fn totalWiredBytes(self: *const DFlareWorkspace) u64 {
        return self.arena.bytes().len;
    }

    inline fn sliceBinding(self: *const DFlareWorkspace, buf: metal.Buffer, ptr: anytype) metal.BufferBinding {
        const byte_ptr: [*]const u8 = @ptrCast(ptr.ptr);
        return .{ .buffer = buf, .offset = self.arena.offsetOf(byte_ptr) };
    }
};

// Metal 4 uniform payloads (exact ABI match with `src/kernels/dflare_gpu.metal`)
const PackTapsParams = extern struct {
    num_taps: u32,
    hidden_size: u32,
    src_is_f16: u32,
    dst_is_f16: u32,
};

const MixTapsParams = extern struct {
    num_pos: u32,
    num_taps: u32,
    hidden_size: u32,
    taps_dtype: u32,
    ring_cap: u32,
    ring_start: u32,
    has_mix_w: u32,
    apply_norm: u32,
    has_norm_w: u32,
    write_f32: u32,
    rms_eps: f32,
    _pad0: u32 = 0,
};

const SeedBlockParams = extern struct {
    block_size: u32,
    hidden_size: u32,
    is_real_gru: u32,
    mask_rows: u32,
    anchor_is_f16: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

const DFlareRmsNormParams = extern struct {
    num_rows: u32,
    dim: u32,
    x_is_f32: u32,
    has_weight: u32,
    write_f32: u32,
    eps: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

const DFlareQkNormRopeParams = extern struct {
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    anchor_pos: u32,
    has_q_norm: u32,
    has_k_norm: u32,
    rope_dim: u32 = 0,
    rms_eps: f32,
    rope_theta: f32,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

const DFlareCtxKvStoreParams = extern struct {
    num_pos: u32,
    num_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_ctx: u32,
    layer_elem_off: u32,
    has_k_norm: u32,
    rope_zero_pos: u32,
    rms_eps: f32,
    rope_theta: f32,
    rope_dim: u32 = 0,
    _pad1: u32 = 0,
};

const DFlareBlockGqaParams = extern struct {
    seq_len: u32,
    active_ctx_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    layer_elem_off: u32,
    scale: f32,
    v_scale: f32 = 1.0,
    max_ctx: u32 = 0,
    anchor_pos: u32 = 0,
    sliding_window: u32 = 0,
    has_sink_bias: u32 = 0,
};

const DFlarePackCtxTapsParams = extern struct {
    num_pos: u32,
    num_taps: u32,
    hidden_size: u32,
    taps_dtype: u32,
    ring_cap: u32,
    ring_start: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

const DFlareElemParams = extern struct {
    count: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

const DFlareGemvF32Params = extern struct {
    m: u32,
    n: u32,
    k: u32,
    has_residual: u32,
    x_is_f32: u32,
    y_is_f32: u32,
    res_is_f32: u32,
    _pad0: u32 = 0,
};

const DFlareMarkovAssembleParams = extern struct {
    hidden_size: u32,
    markov_rank: u32,
    slot_pos: u32,
    w1_rows: u32,
    w1_dtype: u32,
    use_tok_buf: u32,
    tok_slot: u32,
    imm_tok: u32,
    write_f32: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

const DFlareMarkovGruUpdateParams = extern struct {
    markov_rank: u32,
    gates_is_f32: u32,
    has_bias: u32,
    _pad0: u32 = 0,
};

const DFlareSimpleMarkovPrepParams = extern struct {
    hidden_size: u32,
    vocab_size: u32,
    slot_pos: u32,
    use_tok_buf: u32,
    tok_slot: u32,
    imm_tok: u32,
    tok_dtype: u32,
    has_tok_norm: u32,
    concat_2h: u32,
    write_f32: u32,
    rms_eps: f32,
    _pad0: u32 = 0,
};

const DFlareSimpleMarkovSiluParams = extern struct {
    markov_rank: u32,
    u_is_f32: u32,
    add_parent_rnn: u32,
    has_b1: u32,
};

const DFlareMarkovCombineParams = extern struct {
    hidden_size: u32,
    w2_is_f32: u32,
    has_bias: u32,
    has_norm: u32,
    slot_pos: u32,
    rms_eps: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

const DFlareW2VocabParams = extern struct {
    vocab_size: u32,
    v_rows: u32,
    markov_rank: u32,
    w2_dtype: u32,
    has_bias: u32,
    has_base_logits: u32,
    base_row_elem_off: u32,
    softcap: f32,
};

const DFlareLogitsEpilogueParams = extern struct {
    vocab_size: u32,
    v_rows: u32,
    has_bias: u32,
    has_base_logits: u32,
    base_row_elem_off: u32,
    in_is_f16: u32,
    softcap: f32,
    _pad0: u32 = 0,
};

const DFlareTopKParams = extern struct {
    eff_vocab_size: u32,
    top_k: u32,
    logits_is_f16: u32,
    has_mask: u32,
    out_cand_offset: u32,
    write_next_tok: u32,
    next_tok_slot: u32,
    aligned_vec4: u32,
    inv_temp: f32,
    softcap: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

comptime {
    if (@sizeOf(PackTapsParams) != 16) @compileError("PackTapsParams size");
    if (@sizeOf(MixTapsParams) != 48) @compileError("MixTapsParams size");
    if (@sizeOf(SeedBlockParams) != 32) @compileError("SeedBlockParams size");
    if (@sizeOf(DFlareRmsNormParams) != 32) @compileError("DFlareRmsNormParams size");
    if (@sizeOf(DFlareQkNormRopeParams) != 48) @compileError("DFlareQkNormRopeParams size");
    if (@sizeOf(DFlareCtxKvStoreParams) != 48) @compileError("DFlareCtxKvStoreParams size");
    if (@sizeOf(DFlareBlockGqaParams) != 48) @compileError("DFlareBlockGqaParams size");
    if (@sizeOf(DFlarePackCtxTapsParams) != 32) @compileError("DFlarePackCtxTapsParams size");
    if (@sizeOf(DFlareElemParams) != 16) @compileError("DFlareElemParams size");
    if (@sizeOf(DFlareGemvF32Params) != 32) @compileError("DFlareGemvF32Params size");
    if (@sizeOf(DFlareMarkovAssembleParams) != 48) @compileError("DFlareMarkovAssembleParams size");
    if (@sizeOf(DFlareMarkovGruUpdateParams) != 16) @compileError("DFlareMarkovGruUpdateParams size");
    if (@sizeOf(DFlareSimpleMarkovPrepParams) != 48) @compileError("DFlareSimpleMarkovPrepParams size");
    if (@sizeOf(DFlareSimpleMarkovSiluParams) != 16) @compileError("DFlareSimpleMarkovSiluParams size");
    if (@sizeOf(DFlareMarkovCombineParams) != 32) @compileError("DFlareMarkovCombineParams size");
    if (@sizeOf(DFlareW2VocabParams) != 32) @compileError("DFlareW2VocabParams size");
    if (@sizeOf(DFlareLogitsEpilogueParams) != 32) @compileError("DFlareLogitsEpilogueParams size");
    if (@sizeOf(DFlareTopKParams) != 48) @compileError("DFlareTopKParams size");
}

/// Startup-compiled Metal 4 compute pipelines for `src/kernels/dflare_gpu.metal`.
pub const DFlareGpuPipelines = struct {
    pack_taps8: metal.Pipeline,
    pack_ctx_taps_f16: metal.Pipeline,
    mix_taps_rmsnorm: metal.Pipeline,
    seed_block: metal.Pipeline,
    rmsnorm_rows: metal.Pipeline,
    qk_norm_rope: metal.Pipeline,
    ctx_kv_norm_rope_store: metal.Pipeline,
    block_gqa_attn: metal.Pipeline,
    swiglu_mul: metal.Pipeline,
    gemv_batch_f32: metal.Pipeline,
    markov_gru_assemble: metal.Pipeline,
    markov_gru_update: metal.Pipeline,
    simple_markov_prep: metal.Pipeline,
    simple_markov_silu: metal.Pipeline,
    markov_head_combine: metal.Pipeline,
    markov_w2_vocab_score: metal.Pipeline,
    logits_epilogue: metal.Pipeline,
    top1_logits: metal.Pipeline,
    topk_logits: metal.Pipeline,

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!DFlareGpuPipelines {
        const pack_taps8 = try metal.Pipeline.init(device, library, "dflare_pack_taps8");
        errdefer pack_taps8.close();
        const pack_ctx_taps_f16 = try metal.Pipeline.init(device, library, "dflare_pack_ctx_taps_f16");
        errdefer pack_ctx_taps_f16.close();
        const mix_taps_rmsnorm = try metal.Pipeline.init(device, library, "dflare_mix_taps_rmsnorm");
        errdefer mix_taps_rmsnorm.close();
        const seed_block = try metal.Pipeline.init(device, library, "dflare_seed_block");
        errdefer seed_block.close();
        const rmsnorm_rows = try metal.Pipeline.init(device, library, "dflare_rmsnorm_rows");
        errdefer rmsnorm_rows.close();
        const qk_norm_rope = try metal.Pipeline.init(device, library, "dflare_qk_norm_rope");
        errdefer qk_norm_rope.close();
        const ctx_kv_norm_rope_store = try metal.Pipeline.init(device, library, "dflare_ctx_kv_norm_rope_store");
        errdefer ctx_kv_norm_rope_store.close();
        const block_gqa_attn = try metal.Pipeline.init(device, library, "dflare_block_gqa_attn");
        errdefer block_gqa_attn.close();
        const swiglu_mul = try metal.Pipeline.init(device, library, "dflare_swiglu_mul");
        errdefer swiglu_mul.close();
        const gemv_batch_f32 = try metal.Pipeline.init(device, library, "dflare_gemv_batch_f32");
        errdefer gemv_batch_f32.close();
        const markov_gru_assemble = try metal.Pipeline.init(device, library, "dflare_markov_gru_assemble");
        errdefer markov_gru_assemble.close();
        const markov_gru_update = try metal.Pipeline.init(device, library, "dflare_markov_gru_update");
        errdefer markov_gru_update.close();
        const simple_markov_prep = try metal.Pipeline.init(device, library, "dflare_simple_markov_prep");
        errdefer simple_markov_prep.close();
        const simple_markov_silu = try metal.Pipeline.init(device, library, "dflare_simple_markov_silu");
        errdefer simple_markov_silu.close();
        const markov_head_combine = try metal.Pipeline.init(device, library, "dflare_markov_head_combine");
        errdefer markov_head_combine.close();
        const markov_w2_vocab_score = try metal.Pipeline.init(device, library, "dflare_markov_w2_vocab_score");
        errdefer markov_w2_vocab_score.close();
        const logits_epilogue = try metal.Pipeline.init(device, library, "dflare_logits_epilogue");
        errdefer logits_epilogue.close();
        const top1_logits = try metal.Pipeline.init(device, library, "dflare_top1_logits");
        errdefer top1_logits.close();
        const topk_logits = try metal.Pipeline.init(device, library, "dflare_topk_logits");
        errdefer topk_logits.close();

        return .{
            .pack_taps8 = pack_taps8,
            .pack_ctx_taps_f16 = pack_ctx_taps_f16,
            .mix_taps_rmsnorm = mix_taps_rmsnorm,
            .seed_block = seed_block,
            .rmsnorm_rows = rmsnorm_rows,
            .qk_norm_rope = qk_norm_rope,
            .ctx_kv_norm_rope_store = ctx_kv_norm_rope_store,
            .block_gqa_attn = block_gqa_attn,
            .swiglu_mul = swiglu_mul,
            .gemv_batch_f32 = gemv_batch_f32,
            .markov_gru_assemble = markov_gru_assemble,
            .markov_gru_update = markov_gru_update,
            .simple_markov_prep = simple_markov_prep,
            .simple_markov_silu = simple_markov_silu,
            .markov_head_combine = markov_head_combine,
            .markov_w2_vocab_score = markov_w2_vocab_score,
            .logits_epilogue = logits_epilogue,
            .top1_logits = top1_logits,
            .topk_logits = topk_logits,
        };
    }

    pub fn close(self: DFlareGpuPipelines) void {
        self.topk_logits.close();
        self.top1_logits.close();
        self.logits_epilogue.close();
        self.markov_w2_vocab_score.close();
        self.markov_head_combine.close();
        self.simple_markov_silu.close();
        self.simple_markov_prep.close();
        self.markov_gru_update.close();
        self.markov_gru_assemble.close();
        self.gemv_batch_f32.close();
        self.swiglu_mul.close();
        self.block_gqa_attn.close();
        self.ctx_kv_norm_rope_store.close();
        self.qk_norm_rope.close();
        self.rmsnorm_rows.close();
        self.seed_block.close();
        self.mix_taps_rmsnorm.close();
        self.pack_ctx_taps_f16.close();
        self.pack_taps8.close();
    }
};

/// Options for attaching Metal 4 GPU acceleration (`D16`, ADR-0026) to `DFlareProposer`.
pub const DFlareGpuAttachOptions = struct {
    device: metal.Device,
    library: metal.Library,
    /// Optional borrowed Metal 4 compute ring (e.g., target `GraphModel.compute`).
    /// When `null`, `DFlareProposer` creates and owns its own `metal.Compute`.
    compute: ?*metal.Compute = null,
    /// Optional borrowed `DequantPipelines` (e.g., target `GraphModel.dequant_pipes`).
    dequant_pipes: ?*const dequant.DequantPipelines = null,
    /// Optional borrowed `QuantFormatPipelines` (e.g., target `GraphModel.qf_pipes`).
    qf_pipes: ?*const quant_formats.QuantFormatPipelines = null,
    /// Optional GPU binding for target `lm_head` so `3L`/`5L` Markov scoring and
    /// `dflash` `base_logits` stay inside the same `MTL4CommandBuffer`.
    target_lm_head: ?GpuLinearBinding = null,
    /// Optional GPU binding for target `embed_tokens`.
    target_embed_tokens: ?GpuLinearBinding = null,
    /// Optional target embedding RMSNorm weight `[hidden_size]` `f32`.
    target_embed_norm_f32: ?[]const f32 = null,
    timeout_ms: u64 = 30_000,
};

const GpuLayerBindings = struct {
    in_norm_w: ?metal.BufferBinding = null,
    q_proj: GpuLinearBinding,
    k_proj: GpuLinearBinding,
    v_proj: GpuLinearBinding,
    o_proj: GpuLinearBinding,
    k_proj_ctx: ?GpuLinearBinding = null,
    v_proj_ctx: ?GpuLinearBinding = null,
    ctx_norm_w: ?metal.BufferBinding = null,
    k_norm_ctx_w: ?metal.BufferBinding = null,
    q_norm_w: ?metal.BufferBinding = null,
    k_norm_w: ?metal.BufferBinding = null,
    sink_bias_w: ?metal.BufferBinding = null,
    post_norm_w: ?metal.BufferBinding = null,
    mlp_gate: GpuLinearBinding,
    mlp_up: GpuLinearBinding,
    mlp_down: GpuLinearBinding,
};

/// Zero-copy `WiredArena` Metal 4 runtime context bound to `DFlareProposer`.
pub const DFlareGpuContext = struct {
    device: metal.Device,
    library: metal.Library,
    compute_ptr: *metal.Compute,
    owned_compute: ?metal.Compute = null,
    dequant_ptr: *const dequant.DequantPipelines,
    owned_dequant: ?dequant.DequantPipelines = null,
    borrowed_qf: ?*const quant_formats.QuantFormatPipelines = null,
    owned_qf: ?quant_formats.QuantFormatPipelines = null,
    pipes: DFlareGpuPipelines,
    ws_buf: metal.Buffer,
    weight_buf: ?metal.Buffer = null,
    staging_arena: ?arena_mod.WiredArena = null,
    staging_buf: ?metal.Buffer = null,
    resident_bufs: [32]metal.Buffer = undefined,
    resident_count: usize = 0,
    timeout_ms: u64 = 30_000,

    layers: [MAX_LAYERS]GpuLayerBindings = undefined,
    num_layers: usize = 0,
    target_fusion_proj: ?GpuLinearBinding = null,
    fuse_scalar_weights: ?metal.BufferBinding = null,
    fuse_rows: u32 = 0,
    target_fusion_norm: ?metal.BufferBinding = null,
    mask_embed: metal.BufferBinding = undefined,
    mask_rows: u32 = 1,
    tok_embed: ?metal.BufferBinding = null,
    tok_embed_dtype: u32 = 0,
    token_embd_norm: ?metal.BufferBinding = null,
    final_norm: ?metal.BufferBinding = null,
    markov_w1: ?GpuLinearBinding = null,
    markov_w1_vocab: ?metal.BufferBinding = null,
    markov_w1_vocab_dtype: u32 = 0,
    markov_w1_vocab_rows: u32 = 0,
    markov_w1_vocab_raw: ?RawLinearMat = null,
    markov_gru_proj: ?GpuLinearBinding = null,
    markov_gru_bias: ?metal.BufferBinding = null,
    markov_ctx_norm: ?metal.BufferBinding = null,
    markov_rnn_proj: ?GpuLinearBinding = null,
    markov_b1: ?metal.BufferBinding = null,
    markov_w2: ?GpuLinearBinding = null,
    markov_hidden_bias: ?metal.BufferBinding = null,
    markov_w2_vocab: ?GpuLinearBinding = null,
    markov_w2_vocab_bias: ?metal.BufferBinding = null,
    markov_w2_vocab_rows: u32 = 0,
    markov_norm: ?metal.BufferBinding = null,
    lm_head: ?GpuLinearBinding = null,
    target_embed_norm: ?metal.BufferBinding = null,

    fn qfPipesFor(self: *const DFlareGpuContext, dtype: amf.DType, group_size: u16) ?*const quant_formats.QuantFormatPipelines {
        const qf_dt = quant_formats.DType.fromAmf(dtype) orelse return null;
        if (self.borrowed_qf) |bqf| {
            if (bqf.pipelineFor(qf_dt, group_size, .gemv) != null and
                bqf.pipelineFor(qf_dt, group_size, .gemv_batch) != null)
            {
                return bqf;
            }
        }
        if (self.owned_qf) |*oqf| {
            if (oqf.pipelineFor(qf_dt, group_size, .gemv) != null and
                oqf.pipelineFor(qf_dt, group_size, .gemv_batch) != null)
            {
                return oqf;
            }
        }
        return null;
    }

    fn ensureResident(self: *DFlareGpuContext, buf: metal.Buffer) Error!void {
        if (self.owned_compute == null) return;
        for (self.resident_bufs[0..self.resident_count]) |existing| {
            if (existing.handle == buf.handle) return;
        }
        if (self.resident_count >= self.resident_bufs.len) return error.Overflow;
        try self.compute_ptr.makeResident(&.{buf});
        self.resident_bufs[self.resident_count] = buf;
        self.resident_count += 1;
    }

    pub fn close(self: *DFlareGpuContext) void {
        if (self.owned_compute == null and self.resident_count > 0) {
            self.compute_ptr.removeResident(self.resident_bufs[0..self.resident_count]) catch |err| {
                std.debug.panic("DFlareGpuContext removeResident failed: {}", .{err});
            };
            self.resident_count = 0;
        }
        self.pipes.close();
        if (self.owned_qf) |*oqf| oqf.close();
        if (self.owned_dequant) |*odq| odq.close();
        if (self.staging_buf) |sb| sb.close();
        if (self.staging_arena) |*sa| sa.deinit();
        if (self.weight_buf) |wb| wb.close();
        self.ws_buf.close();
        if (self.owned_compute) |oc| oc.close();
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

/// D14 / D16 `DFlare` block-parallel speculative proposer (`Inkling 2` `3L`/`5L`,
/// `block_size = 8`, `markov_rank = 256` sequential `rnn_hidden` Markov head).
/// Executes on Metal 4 GPU when attached via `attachGpu` and falls back to
/// the scalar CPU path automatically when no GPU context is attached.
pub const DFlareProposer = struct {
    config: DFlareConfig,
    weights: DFlareWeights,
    workspace: *DFlareWorkspace,
    gpu_ctx: ?*DFlareGpuContext = null,
    backbone_runs: u64 = 0,
    markov_steps: u64 = 0,
    ctx_cache_len: u32 = 0,
    ctx_cache_epoch: u64 = 0,
    base_logits_ready: bool = false,

    inline fn isRealGru(self: *const DFlareProposer) bool {
        return self.weights.markov_gru_proj_bf16 != null or
            self.weights.markov_gru_proj_f32 != null or
            self.weights.markov_gru_proj_raw != null;
    }

    pub inline fn isDirectBlockDiffusion(self: *const DFlareProposer) bool {
        return self.config.markov_rank == 0;
    }

    inline fn usesAnchorMaskSeeding(self: *const DFlareProposer) bool {
        return self.isRealGru() or self.isDirectBlockDiffusion();
    }

    fn resolveEffectiveMaskEmbed(self: *DFlareProposer) Error![]const f32 {
        if (self.weights.mask_embed.len > 0) return self.weights.mask_embed;
        const mtid = self.config.mask_token_id orelse return error.InvalidWeights;
        const h: usize = self.config.hidden_size;
        const raw = try resolveTokEmbed(
            self.weights.tok_embed,
            self.weights.tok_embed_raw,
            self.weights.tok_embed_hook,
            mtid,
            h,
            self.workspace.mask_embed_tmp,
        );
        if (raw.ptr != self.workspace.mask_embed_tmp.ptr) {
            @memcpy(self.workspace.mask_embed_tmp[0..h], raw[0..h]);
        }
        if (self.weights.tok_embed_hook == null) {
            if (self.weights.token_embd_norm) |ten| {
                applyRmsNormInPlace(self.workspace.mask_embed_tmp[0..h], ten, self.config.rms_eps);
            }
        }
        return self.workspace.mask_embed_tmp[0..h];
    }

    inline fn hasMarkovW2Vocab(self: *const DFlareProposer) bool {
        return self.weights.markov_w2_vocab_bf16 != null or self.weights.markov_w2_vocab_raw != null;
    }

    pub inline fn hasGpu(self: *const DFlareProposer) bool {
        return self.gpu_ctx != null;
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

    /// Attaches a Metal 4 GPU execution context (`D16`, ADR-0026) over the
    /// `WiredArena`-backed weights and workspace. Safe to re-call; call
    /// `detachGpu()` before `workspace.deinit()` or `LoadedDFlareBundle.deinit()`.
    pub fn attachGpu(self: *DFlareProposer, opts: DFlareGpuAttachOptions) Error!void {
        self.detachGpu();
        if (self.config.head_dim > 256) return error.InvalidConfig;

        const ctx = try std.heap.c_allocator.create(DFlareGpuContext);
        errdefer std.heap.c_allocator.destroy(ctx);

        const ws_buf = try metal.Buffer.wrap(opts.device, self.workspace.arena.bytes());
        errdefer ws_buf.close();

        var weight_buf: ?metal.Buffer = null;
        errdefer if (weight_buf) |wb| wb.close();
        if (self.weights.wired_arena_bytes) |wab| {
            weight_buf = try metal.Buffer.wrap(opts.device, wab);
        }

        var owned_compute: ?metal.Compute = null;
        errdefer if (owned_compute) |oc| oc.close();
        if (opts.compute == null) {
            owned_compute = try metal.Compute.initWithFrames(opts.device, 2);
        }

        var owned_dequant: ?dequant.DequantPipelines = null;
        errdefer if (owned_dequant) |*odq| odq.close();
        if (opts.dequant_pipes == null) {
            owned_dequant = try dequant.DequantPipelines.init(opts.device, opts.library);
        }

        const pipes = try DFlareGpuPipelines.init(opts.device, opts.library);
        errdefer pipes.close();

        // Collect missing QuantFormatPipelines specs across all DFlare weights + target hooks.
        var missing_specs: [quant_formats.ALL_FORMAT_SPECS.len]quant_formats.FormatSpec = undefined;
        var missing_len: usize = 0;
        const h: u32 = self.config.hidden_size;
        const inter: u32 = self.config.intermediate_size;
        const q_dim: u32 = self.config.qDim();
        const kv_dim: u32 = self.config.kvDim();
        const mr: u32 = self.config.markov_rank;

        try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, self.weights.target_fusion_proj_raw);
        for (self.weights.layers) |ly| {
            try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, ly.q_proj_raw);
            try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, ly.k_proj_raw);
            try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, ly.v_proj_raw);
            try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, ly.o_proj_raw);
            try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, ly.k_proj_ctx_raw);
            try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, ly.v_proj_ctx_raw);
            try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, ly.mlp_gate_raw);
            try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, ly.mlp_up_raw);
            try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, ly.mlp_down_raw);
        }
        try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, self.weights.markov_w1_raw);
        try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, self.weights.markov_gru_proj_raw);
        try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, self.weights.markov_rnn_proj_raw);
        try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, self.weights.markov_w2_raw);
        try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, self.weights.markov_w2_vocab_raw);
        try collectMissingQfSpec(opts.qf_pipes, &missing_specs, &missing_len, self.weights.lm_head_raw);
        if (opts.target_lm_head) |tlh| {
            try collectMissingQfSpecDType(opts.qf_pipes, &missing_specs, &missing_len, tlh.amf_dtype, tlh.group_size);
        }

        var owned_qf: ?quant_formats.QuantFormatPipelines = null;
        errdefer if (owned_qf) |*oqf| oqf.close();
        if (missing_len > 0) {
            owned_qf = try quant_formats.QuantFormatPipelines.initWithOptions(opts.device, opts.library, .{
                .specs = missing_specs[0..missing_len],
                .include_moe = false,
                .include_act_quant = false,
            });
        }

        // Pass 1: measure staging bytes needed for any host slices NOT inside `wired_arena_bytes`.
        const wired_opt = self.weights.wired_arena_bytes;
        var stage_bytes: usize = 0;
        if (self.weights.mask_embed.len > 0) {
            stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(self.weights.mask_embed));
        }
        if (self.weights.fuse_scalar_weights) |fsw| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(fsw));
        if (self.weights.target_fusion_norm) |fnw| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(fnw));
        if (self.weights.target_fusion_proj_raw) |r| {
            stage_bytes += stageMeasureRaw(wired_opt, r);
        } else if (self.weights.target_fusion_proj) |fp| {
            stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(fp));
        }
        if (self.weights.token_embd_norm) |ten| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(ten));
        if (self.weights.final_norm) |fnw| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(fnw));
        if (opts.target_embed_norm_f32) |tenf| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(tenf));

        if (opts.target_embed_tokens == null and self.weights.tok_embed_hook == null) {
            if (self.weights.tok_embed_raw) |tr| {
                stage_bytes += stageMeasureRaw(wired_opt, tr);
            } else if (self.weights.tok_embed.len > 0) {
                stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(self.weights.tok_embed));
            }
        } else if (self.weights.tok_embed_hook) |hk| {
            if (hk == .dense_f32) stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(hk.dense_f32));
        }

        for (self.weights.layers) |ly| {
            if (ly.in_norm_weight) |w| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(w));
            if (ly.ctx_norm_weight) |w| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(w));
            if (ly.k_norm_ctx_weight) |w| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(w));
            if (ly.q_norm_weight) |w| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(w));
            if (ly.k_norm_weight) |w| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(w));
            if (ly.sink_bias) |w| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(w));
            if (ly.post_norm_weight) |w| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(w));
            stage_bytes += stageMeasureLayerProj(wired_opt, ly.q_proj, ly.q_proj_bf16, ly.q_proj_raw);
            stage_bytes += stageMeasureLayerProj(wired_opt, ly.k_proj, ly.k_proj_bf16, ly.k_proj_raw);
            stage_bytes += stageMeasureLayerProj(wired_opt, ly.v_proj, ly.v_proj_bf16, ly.v_proj_raw);
            stage_bytes += stageMeasureLayerProj(wired_opt, ly.o_proj, ly.o_proj_bf16, ly.o_proj_raw);
            stage_bytes += stageMeasureLayerProj(wired_opt, ly.k_proj_ctx orelse &.{}, ly.k_proj_ctx_bf16, ly.k_proj_ctx_raw);
            stage_bytes += stageMeasureLayerProj(wired_opt, ly.v_proj_ctx orelse &.{}, ly.v_proj_ctx_bf16, ly.v_proj_ctx_raw);
            stage_bytes += stageMeasureLayerProj(wired_opt, ly.mlp_gate_proj, ly.mlp_gate_bf16, ly.mlp_gate_raw);
            stage_bytes += stageMeasureLayerProj(wired_opt, ly.mlp_up_proj, ly.mlp_up_bf16, ly.mlp_up_raw);
            stage_bytes += stageMeasureLayerProj(wired_opt, ly.mlp_down_proj, ly.mlp_down_bf16, ly.mlp_down_raw);
        }

        if (self.weights.markov_gru_bias) |b| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(b));
        if (self.weights.markov_ctx_norm) |b| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(b));
        if (self.weights.markov_b1) |b| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(b));
        if (self.weights.markov_hidden_bias) |b| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(b));
        if (self.weights.markov_w2_vocab_bias) |b| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(b));
        if (self.weights.markov_norm) |b| stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(b));
        stage_bytes += stageMeasureLayerProj(wired_opt, self.weights.markov_w1, null, self.weights.markov_w1_raw);
        stage_bytes += stageMeasureLayerProj(wired_opt, &.{}, self.weights.markov_w1_vocab_bf16, self.weights.markov_w1_vocab_raw);
        stage_bytes += stageMeasureLayerProj(wired_opt, self.weights.markov_gru_proj_f32 orelse &.{}, self.weights.markov_gru_proj_bf16, self.weights.markov_gru_proj_raw);
        stage_bytes += stageMeasureLayerProj(wired_opt, self.weights.markov_rnn_proj orelse &.{}, null, self.weights.markov_rnn_proj_raw);
        stage_bytes += stageMeasureLayerProj(wired_opt, self.weights.markov_w2, null, self.weights.markov_w2_raw);
        stage_bytes += stageMeasureLayerProj(wired_opt, &.{}, self.weights.markov_w2_vocab_bf16, self.weights.markov_w2_vocab_raw);

        if (opts.target_lm_head == null) {
            if (self.weights.lm_head_raw) |lhr| {
                stage_bytes += stageMeasureRaw(wired_opt, lhr);
            } else switch (self.weights.lm_head) {
                .dense_f32 => |m| if (m.len > 0) {
                    stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(m));
                },
                .dense_f16 => |m| if (m.len > 0) {
                    stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(m));
                },
                .dense_bf16 => |m| if (m.len > 0) {
                    stage_bytes += stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(m));
                },
                .callback => {},
            }
        }

        var staging_arena: ?arena_mod.WiredArena = null;
        errdefer if (staging_arena) |*sa| sa.deinit();
        var staging_buf: ?metal.Buffer = null;
        errdefer if (staging_buf) |sb| sb.close();
        if (stage_bytes > 0) {
            const aligned_stage = try page.alignUp(stage_bytes);
            const st_budget = try defaultWiredBudget();
            staging_arena = try arena_mod.WiredArena.init(aligned_stage, st_budget);
            staging_buf = try metal.Buffer.wrap(opts.device, staging_arena.?.bytes());
        }

        ctx.* = .{
            .device = opts.device,
            .library = opts.library,
            .compute_ptr = undefined,
            .owned_compute = owned_compute,
            .dequant_ptr = undefined,
            .owned_dequant = owned_dequant,
            .borrowed_qf = opts.qf_pipes,
            .owned_qf = owned_qf,
            .pipes = pipes,
            .ws_buf = ws_buf,
            .weight_buf = weight_buf,
            .staging_arena = staging_arena,
            .staging_buf = staging_buf,
            .timeout_ms = opts.timeout_ms,
            .num_layers = self.config.num_layers,
        };
        ctx.compute_ptr = if (ctx.owned_compute) |*oc| oc else opts.compute.?;
        ctx.dequant_ptr = if (ctx.owned_dequant) |*odq| odq else opts.dequant_pipes.?;
        // Ownership moved into `ctx`; clear local errdefers and guard remaining setup with `errdefer ctx.close()`.
        owned_compute = null;
        owned_dequant = null;
        owned_qf = null;
        staging_arena = null;
        staging_buf = null;
        weight_buf = null;
        errdefer ctx.close();

        // Pass 2: bind zero-copy `weight_buf` or stage into `staging_buf`.
        const bind_ctx = BindHelper{
            .wired = wired_opt,
            .weight_buf = ctx.weight_buf,
            .st_arena = if (ctx.staging_arena) |*sa| sa else null,
            .st_buf = ctx.staging_buf,
        };

        if (self.weights.mask_embed.len > 0) {
            ctx.mask_embed = try bind_ctx.bindSlice(std.mem.sliceAsBytes(self.weights.mask_embed));
            ctx.mask_rows = @intCast(self.weights.mask_embed.len / h);
        } else {
            ctx.mask_embed = self.workspace.sliceBinding(ctx.ws_buf, self.workspace.mask_embed_tmp);
            ctx.mask_rows = 1;
        }
        if (self.weights.fuse_scalar_weights) |fsw| {
            ctx.fuse_scalar_weights = try bind_ctx.bindSlice(std.mem.sliceAsBytes(fsw));
            ctx.fuse_rows = @intCast(fsw.len / self.config.num_target_taps);
        }
        if (self.weights.target_fusion_norm) |fnw| ctx.target_fusion_norm = try bind_ctx.bindSlice(std.mem.sliceAsBytes(fnw));
        if (self.weights.target_fusion_proj_raw) |fpr| {
            ctx.target_fusion_proj = try bind_ctx.bindRaw(fpr);
        } else if (self.weights.target_fusion_proj) |fp| {
            ctx.target_fusion_proj = try bind_ctx.bindF32Mat(fp, h, self.config.num_target_taps * h);
        }
        if (self.weights.token_embd_norm) |ten| ctx.token_embd_norm = try bind_ctx.bindSlice(std.mem.sliceAsBytes(ten));
        if (self.weights.final_norm) |fnw| ctx.final_norm = try bind_ctx.bindSlice(std.mem.sliceAsBytes(fnw));
        if (opts.target_embed_norm_f32) |tenf| ctx.target_embed_norm = try bind_ctx.bindSlice(std.mem.sliceAsBytes(tenf));

        if (opts.target_embed_tokens) |tet| {
            ctx.tok_embed = tet.weight;
            ctx.tok_embed_dtype = switch (tet.amf_dtype) {
                .f32 => 0,
                .f16 => 1,
                .bf16 => 2,
                else => 0,
            };
        } else if (self.weights.tok_embed_hook) |hk| {
            if (hk == .dense_f32) {
                ctx.tok_embed = try bind_ctx.bindSlice(std.mem.sliceAsBytes(hk.dense_f32));
                ctx.tok_embed_dtype = 0;
            }
        } else if (self.weights.tok_embed_raw) |tr| {
            if (tr.amf_dtype == .f32 or tr.amf_dtype == .f16 or tr.amf_dtype == .bf16) {
                ctx.tok_embed = try bind_ctx.bindSlice(tr.data);
                ctx.tok_embed_dtype = switch (tr.amf_dtype) {
                    .f32 => 0,
                    .f16 => 1,
                    .bf16 => 2,
                    else => 0,
                };
            }
        } else if (self.weights.tok_embed.len > 0) {
            ctx.tok_embed = try bind_ctx.bindSlice(std.mem.sliceAsBytes(self.weights.tok_embed));
            ctx.tok_embed_dtype = 0;
        }

        for (self.weights.layers, 0..) |ly, li| {
            ctx.layers[li] = .{
                .in_norm_w = if (ly.in_norm_weight) |w| try bind_ctx.bindSlice(std.mem.sliceAsBytes(w)) else null,
                .q_proj = try bind_ctx.bindLayerProj(ly.q_proj, ly.q_proj_bf16, ly.q_proj_raw, q_dim, h),
                .k_proj = try bind_ctx.bindLayerProj(ly.k_proj, ly.k_proj_bf16, ly.k_proj_raw, kv_dim, h),
                .v_proj = try bind_ctx.bindLayerProj(ly.v_proj, ly.v_proj_bf16, ly.v_proj_raw, kv_dim, h),
                .o_proj = try bind_ctx.bindLayerProj(ly.o_proj, ly.o_proj_bf16, ly.o_proj_raw, h, q_dim),
                .k_proj_ctx = try bind_ctx.bindLayerProjOpt(ly.k_proj_ctx, ly.k_proj_ctx_bf16, ly.k_proj_ctx_raw, kv_dim, h),
                .v_proj_ctx = try bind_ctx.bindLayerProjOpt(ly.v_proj_ctx, ly.v_proj_ctx_bf16, ly.v_proj_ctx_raw, kv_dim, h),
                .ctx_norm_w = if (ly.ctx_norm_weight) |w| try bind_ctx.bindSlice(std.mem.sliceAsBytes(w)) else null,
                .k_norm_ctx_w = if (ly.k_norm_ctx_weight) |w| try bind_ctx.bindSlice(std.mem.sliceAsBytes(w)) else null,
                .q_norm_w = if (ly.q_norm_weight) |w| try bind_ctx.bindSlice(std.mem.sliceAsBytes(w)) else null,
                .k_norm_w = if (ly.k_norm_weight) |w| try bind_ctx.bindSlice(std.mem.sliceAsBytes(w)) else null,
                .sink_bias_w = if (ly.sink_bias) |w| try bind_ctx.bindSlice(std.mem.sliceAsBytes(w)) else null,
                .post_norm_w = if (ly.post_norm_weight) |w| try bind_ctx.bindSlice(std.mem.sliceAsBytes(w)) else null,
                .mlp_gate = try bind_ctx.bindLayerProj(ly.mlp_gate_proj, ly.mlp_gate_bf16, ly.mlp_gate_raw, inter, h),
                .mlp_up = try bind_ctx.bindLayerProj(ly.mlp_up_proj, ly.mlp_up_bf16, ly.mlp_up_raw, inter, h),
                .mlp_down = try bind_ctx.bindLayerProj(ly.mlp_down_proj, ly.mlp_down_bf16, ly.mlp_down_raw, h, inter),
            };
        }

        if (mr > 0) {
            if (self.weights.markov_gru_bias) |b| ctx.markov_gru_bias = try bind_ctx.bindSlice(std.mem.sliceAsBytes(b));
            if (self.weights.markov_ctx_norm) |b| ctx.markov_ctx_norm = try bind_ctx.bindSlice(std.mem.sliceAsBytes(b));
            if (self.weights.markov_b1) |b| ctx.markov_b1 = try bind_ctx.bindSlice(std.mem.sliceAsBytes(b));
            if (self.weights.markov_hidden_bias) |b| ctx.markov_hidden_bias = try bind_ctx.bindSlice(std.mem.sliceAsBytes(b));
            if (self.weights.markov_w2_vocab_bias) |b| ctx.markov_w2_vocab_bias = try bind_ctx.bindSlice(std.mem.sliceAsBytes(b));
            if (self.weights.markov_norm) |b| ctx.markov_norm = try bind_ctx.bindSlice(std.mem.sliceAsBytes(b));

            if (self.weights.markov_w1_raw) |r| {
                ctx.markov_w1 = try bind_ctx.bindRaw(r);
            } else if (self.weights.markov_w1.len > 0) {
                const w1_k: u32 = @intCast(self.weights.markov_w1.len / mr);
                ctx.markov_w1 = try bind_ctx.bindF32Mat(self.weights.markov_w1, mr, w1_k);
            }

            if (self.weights.markov_w1_vocab_raw) |r| {
                ctx.markov_w1_vocab_rows = self.weights.markov_w1_vocab_rows;
                if (r.amf_dtype == .bf16 or r.amf_dtype == .f16 or r.amf_dtype == .f32 or (r.amf_dtype == .gguf_q8_0 and (mr % 32) == 0)) {
                    ctx.markov_w1_vocab = try bind_ctx.bindSlice(r.data);
                    ctx.markov_w1_vocab_dtype = switch (r.amf_dtype) {
                        .bf16 => 0,
                        .f16 => 1,
                        .f32 => 2,
                        .gguf_q8_0 => 3,
                        else => 0,
                    };
                } else {
                    ctx.markov_w1_vocab_raw = r;
                    ctx.markov_w1_vocab_dtype = 2;
                }
            } else if (self.weights.markov_w1_vocab_bf16) |wb| {
                ctx.markov_w1_vocab = try bind_ctx.bindSlice(std.mem.sliceAsBytes(wb));
                ctx.markov_w1_vocab_dtype = 0;
                ctx.markov_w1_vocab_rows = self.weights.markov_w1_vocab_rows;
            }

            ctx.markov_gru_proj = try bind_ctx.bindLayerProjOpt(self.weights.markov_gru_proj_f32, self.weights.markov_gru_proj_bf16, self.weights.markov_gru_proj_raw, 3 * mr, 2 * h + 2 * mr);
            ctx.markov_rnn_proj = try bind_ctx.bindLayerProjOpt(self.weights.markov_rnn_proj, null, self.weights.markov_rnn_proj_raw, mr, mr);
            if (self.weights.markov_w2_raw) |r| {
                ctx.markov_w2 = try bind_ctx.bindRaw(r);
            } else if (self.weights.markov_w2.len > 0) {
                ctx.markov_w2 = try bind_ctx.bindF32Mat(self.weights.markov_w2, h, mr);
            }

            if (self.weights.markov_w2_vocab_raw) |r| {
                ctx.markov_w2_vocab = try bind_ctx.bindRaw(r);
                ctx.markov_w2_vocab_rows = self.weights.markov_w2_vocab_rows;
            } else if (self.weights.markov_w2_vocab_bf16) |w2b| {
                ctx.markov_w2_vocab = try bind_ctx.bindBf16Mat(w2b, self.weights.markov_w2_vocab_rows, mr);
                ctx.markov_w2_vocab_rows = self.weights.markov_w2_vocab_rows;
            }
        }

        if (opts.target_lm_head) |tlh| {
            ctx.lm_head = tlh;
        } else if (self.weights.lm_head_raw) |lhr| {
            ctx.lm_head = try bind_ctx.bindRaw(lhr);
        } else switch (self.weights.lm_head) {
            .dense_f32 => |m| if (m.len > 0) {
                ctx.lm_head = try bind_ctx.bindF32Mat(m, self.config.vocab_size, h);
            },
            .dense_f16 => |m| if (m.len > 0) {
                ctx.lm_head = .{ .amf_dtype = .f16, .weight = try bind_ctx.bindSlice(std.mem.sliceAsBytes(m)), .n = self.config.vocab_size, .k = h };
            },
            .dense_bf16 => |m| if (m.len > 0) {
                ctx.lm_head = try bind_ctx.bindBf16Mat(m, self.config.vocab_size, h);
            },
            .callback => {},
        }

        // Make owned/wrapped buffers resident on `compute_ptr`.
        ctx.resident_bufs[ctx.resident_count] = ctx.ws_buf;
        ctx.resident_count += 1;
        if (ctx.weight_buf) |wb| {
            ctx.resident_bufs[ctx.resident_count] = wb;
            ctx.resident_count += 1;
        }
        if (ctx.staging_buf) |sb| {
            ctx.resident_bufs[ctx.resident_count] = sb;
            ctx.resident_count += 1;
        }
        if (ctx.owned_qf) |*oqf| {
            ctx.resident_bufs[ctx.resident_count] = oqf.codebook_buf;
            ctx.resident_count += 1;
        }
        // Only register external buffers when we own `Compute` (`opts.compute == null`).
        if (ctx.owned_compute != null) {
            if (opts.target_lm_head) |tlh| {
                appendUniqueResident(&ctx.resident_bufs, &ctx.resident_count, tlh.weight.buffer);
                if (tlh.params) |pb| appendUniqueResident(&ctx.resident_bufs, &ctx.resident_count, pb.buffer);
            }
            if (opts.target_embed_tokens) |tet| {
                appendUniqueResident(&ctx.resident_bufs, &ctx.resident_count, tet.weight.buffer);
                if (tet.params) |pb| appendUniqueResident(&ctx.resident_bufs, &ctx.resident_count, pb.buffer);
            }
        }
        try ctx.compute_ptr.makeResident(ctx.resident_bufs[0..ctx.resident_count]);

        self.gpu_ctx = ctx;
    }

    pub fn detachGpu(self: *DFlareProposer) void {
        if (self.gpu_ctx) |ctx| {
            ctx.close();
            std.heap.c_allocator.destroy(ctx);
            self.gpu_ctx = null;
        }
    }

    /// Updates or attaches a GPU target `lm_head` binding after `attachGpu`.
    pub fn setTargetLmHeadGpu(self: *DFlareProposer, binding: ?GpuLinearBinding) Error!void {
        const gpu = self.gpu_ctx orelse return error.InvalidWeights;
        if (binding) |tlh| {
            if (dequant.usesQuantFormats(tlh.amf_dtype, tlh.group_size) and gpu.qfPipesFor(tlh.amf_dtype, tlh.group_size) == null) {
                var specs: [quant_formats.ALL_FORMAT_SPECS.len]quant_formats.FormatSpec = undefined;
                var n_specs: usize = 0;
                if (gpu.owned_qf) |*oqf| {
                    for (quant_formats.ALL_FORMAT_SPECS) |s| {
                        if (oqf.pipelineFor(s.dtype, s.group_size, .gemv) != null) {
                            specs[n_specs] = s;
                            n_specs += 1;
                        }
                    }
                }
                try collectMissingQfSpecDType(gpu.borrowed_qf, &specs, &n_specs, tlh.amf_dtype, tlh.group_size);
                const new_qf = try quant_formats.QuantFormatPipelines.initWithOptions(gpu.device, gpu.library, .{
                    .specs = specs[0..n_specs],
                    .include_moe = false,
                    .include_act_quant = false,
                });
                if (gpu.owned_qf) |*old_qf| {
                    gpu.compute_ptr.removeResident(&.{old_qf.codebook_buf}) catch |err| {
                        std.debug.panic("removeResident(old_qf) failed: {}", .{err});
                    };
                    var wi: usize = 0;
                    while (wi < gpu.resident_count) {
                        if (gpu.resident_bufs[wi].handle == old_qf.codebook_buf.handle) {
                            gpu.resident_bufs[wi] = gpu.resident_bufs[gpu.resident_count - 1];
                            gpu.resident_count -= 1;
                        } else {
                            wi += 1;
                        }
                    }
                    old_qf.close();
                }
                gpu.owned_qf = new_qf;
                if (gpu.resident_count < gpu.resident_bufs.len) {
                    gpu.resident_bufs[gpu.resident_count] = new_qf.codebook_buf;
                    gpu.resident_count += 1;
                }
                try gpu.compute_ptr.makeResident(&.{new_qf.codebook_buf});
            }
            if (gpu.owned_compute != null) {
                try gpu.compute_ptr.makeResident(&.{tlh.weight.buffer});
                appendUniqueResident(&gpu.resident_bufs, &gpu.resident_count, tlh.weight.buffer);
                if (tlh.params) |pb| {
                    try gpu.compute_ptr.makeResident(&.{pb.buffer});
                    appendUniqueResident(&gpu.resident_bufs, &gpu.resident_count, pb.buffer);
                }
            }
        }
        gpu.lm_head = binding;
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

    /// Runs the `3L` / `5L` block-parallel backbone ONCE from GPU-resident target hidden taps.
    pub fn runBlockBackboneOnceGpuTaps(
        self: *DFlareProposer,
        root_token: u32,
        gpu_taps: GpuTargetTaps,
        gpu_ctx_history: ?GpuContextHistory,
    ) Error!void {
        const gpu = self.gpu_ctx orelse return error.InvalidWeights;
        if (gpu_taps.taps.len != self.config.num_target_taps) return error.DimensionMismatch;
        try self.runBlockBackboneGpuImpl(gpu, root_token, null, gpu_taps, gpu_ctx_history, true);
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

        if (layer.k_proj_ctx_bf16 != null or layer.k_proj_ctx != null or layer.k_proj_ctx_raw != null) {
            matVecLayer(
                layer.k_proj_ctx orelse &.{},
                layer.k_proj_ctx_bf16,
                layer.k_proj_ctx_raw,
                ctx_in,
                kv_dim,
                h,
                dst_k[0..kv_dim],
                false,
                self.workspace.cpu_quant_row_scratch,
            );
        } else {
            matVecLayer(
                layer.k_proj,
                layer.k_proj_bf16,
                layer.k_proj_raw,
                ctx_in,
                kv_dim,
                h,
                dst_k[0..kv_dim],
                false,
                self.workspace.cpu_quant_row_scratch,
            );
        }

        if (layer.v_proj_ctx_bf16 != null or layer.v_proj_ctx != null or layer.v_proj_ctx_raw != null) {
            matVecLayer(
                layer.v_proj_ctx orelse &.{},
                layer.v_proj_ctx_bf16,
                layer.v_proj_ctx_raw,
                ctx_in,
                kv_dim,
                h,
                dst_v[0..kv_dim],
                false,
                self.workspace.cpu_quant_row_scratch,
            );
        } else {
            matVecLayer(
                layer.v_proj,
                layer.v_proj_bf16,
                layer.v_proj_raw,
                ctx_in,
                kv_dim,
                h,
                dst_v[0..kv_dim],
                false,
                self.workspace.cpu_quant_row_scratch,
            );
        }

        if (layer.k_norm_ctx_weight orelse layer.k_norm_weight) |kn| {
            applyHeadRmsNorm(dst_k[0..kv_dim], self.config.num_key_value_heads, self.config.head_dim, kn, self.config.rms_eps);
        }
        if (self.config.rope_theta > 0.0) {
            applyRopeNeoXInPlace(dst_k[0..kv_dim], self.config.num_key_value_heads, self.config.head_dim, self.config.effectiveRopeDim(), pos, self.config.rope_theta);
        }
    }

    fn runBlockBackboneOnceForRoot(
        self: *DFlareProposer,
        root_token: u32,
        target_hiddens: []const []const f32,
    ) Error!void {
        return self.runBlockBackboneOnceForRootImpl(root_token, target_hiddens, false);
    }

    fn runBlockBackboneOnceForRootImpl(
        self: *DFlareProposer,
        root_token: u32,
        target_hiddens: []const []const f32,
        preseeded_slots: bool,
    ) Error!void {
        if (self.gpu_ctx) |gpu| {
            return self.runBlockBackboneGpuImpl(gpu, root_token, target_hiddens, null, null, true);
        }
        const taps: usize = self.config.num_target_taps;
        const h: usize = self.config.hidden_size;
        const w: usize = self.config.block_size;
        const v: usize = self.config.vocab_size;
        const kv_dim: usize = self.config.kvDim();
        const is_real = self.isRealGru();
        const anchor_mask_mode = self.usesAnchorMaskSeeding();
        if (target_hiddens.len != taps) return error.DimensionMismatch;

        // 0. For real DFlare/DFlash, resolve anchor token embedding (and mask_token_id embedding if needed)
        //    FIRST (this also registers GraphModel's context tap history provider on the exact target pointer).
        var eff_mask_embed: []const f32 = self.weights.mask_embed;
        if (anchor_mask_mode and !preseeded_slots) {
            const raw_anchor = try resolveTokEmbed(self.weights.tok_embed, self.weights.tok_embed_raw, self.weights.tok_embed_hook, root_token, h, self.workspace.tok_embed_tmp);
            if (raw_anchor.ptr != self.workspace.tok_embed_tmp.ptr) {
                @memcpy(self.workspace.tok_embed_tmp[0..h], raw_anchor[0..h]);
            }
            if (self.weights.tok_embed_hook == null) {
                if (self.weights.token_embd_norm) |ten| {
                    applyRmsNormInPlace(self.workspace.tok_embed_tmp[0..h], ten, self.config.rms_eps);
                }
            }
            eff_mask_embed = try self.resolveEffectiveMaskEmbed();
        } else if (!preseeded_slots and eff_mask_embed.len == 0) {
            eff_mask_embed = try self.resolveEffectiveMaskEmbed();
        }

        // 1. Check for full committed-token context history (prompt + accepted decode tokens).
        var use_cached_ctx: bool = false;
        var active_ctx_len: usize = 1;
        var anchor_pos: u32 = if (self.config.rope_theta > 0.0) 1 else 0;
        const max_ctx: usize = self.workspace.max_ctx_len;

        if (anchor_mask_mode) {
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
                        // Fuse multi-tap target hidden states ONCE per context position when using `fc` + `hidden_norm` (`DFlash`).
                        if (self.weights.fuse_scalar_weights == null) {
                            if (self.weights.target_fusion_proj != null or self.weights.target_fusion_proj_raw != null) {
                                matVecLayer(
                                    self.weights.target_fusion_proj orelse &.{},
                                    null,
                                    self.weights.target_fusion_proj_raw,
                                    self.workspace.taps_flat[0 .. taps * h],
                                    h,
                                    taps * h,
                                    self.workspace.ctx_hidden[0..h],
                                    false,
                                    self.workspace.cpu_quant_row_scratch,
                                );
                            } else {
                                @memcpy(self.workspace.ctx_hidden[0..h], self.workspace.taps_flat[0..h]);
                            }
                            if (self.weights.target_fusion_norm) |fnw| {
                                applyRmsNormInPlace(self.workspace.ctx_hidden[0..h], fnw, self.config.rms_eps);
                            }
                        }
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
        } else if (self.weights.target_fusion_proj != null or self.weights.target_fusion_proj_raw != null) {
            matVecLayer(
                self.weights.target_fusion_proj orelse &.{},
                null,
                self.weights.target_fusion_proj_raw,
                self.workspace.taps_flat[0 .. taps * h],
                h,
                taps * h,
                self.workspace.ctx_hidden[0..h],
                false,
                self.workspace.cpu_quant_row_scratch,
            );
            @memcpy(self.workspace.ctx_markov[0..h], self.workspace.ctx_hidden[0..h]);
        } else {
            @memcpy(self.workspace.ctx_hidden[0..h], self.workspace.taps_flat[0..h]);
            @memcpy(self.workspace.ctx_markov[0..h], self.workspace.ctx_hidden[0..h]);
        }
        if (self.weights.target_fusion_norm) |fnw| {
            applyRmsNormInPlace(self.workspace.ctx_hidden[0..h], fnw, self.config.rms_eps);
        }

        // 3. Seed noise block `slot_hiddens[0..w-1]` (skipped when caller pre-populated refined slot embeddings).
        if (!preseeded_slots) {
            const per_pos_mask = eff_mask_embed.len == w * h;
            const tail_pos_mask = (w > 1 and eff_mask_embed.len == (w - 1) * h);
            if (anchor_mask_mode) {
                // Exact `dflash.py` / `dspark.cpp:build_dspark_inp_block`:
                // slot 0 = anchor token embedding; slots 1..w-1 = pure mask_embed / embed(mask_token_id).
                @memcpy(self.workspace.slot_hiddens[0..h], self.workspace.tok_embed_tmp[0..h]);
                for (1..w) |pos| {
                    const dst = self.workspace.slot_hiddens[pos * h ..][0..h];
                    const m_slice = if (tail_pos_mask)
                        eff_mask_embed[(pos - 1) * h ..][0..h]
                    else if (per_pos_mask)
                        eff_mask_embed[pos * h ..][0..h]
                    else
                        eff_mask_embed[0..h];
                    @memcpy(dst, m_slice);
                }
            } else {
                for (0..w) |pos| {
                    const dst = self.workspace.slot_hiddens[pos * h ..][0..h];
                    if (tail_pos_mask) {
                        if (pos == 0) {
                            @memcpy(dst, self.workspace.ctx_hidden[0..h]);
                        } else {
                            const m_slice = eff_mask_embed[(pos - 1) * h ..][0..h];
                            for (0..h) |d| dst[d] = self.workspace.ctx_hidden[d] + m_slice[d];
                        }
                    } else {
                        const m_slice = if (per_pos_mask)
                            eff_mask_embed[pos * h ..][0..h]
                        else
                            eff_mask_embed[0..h];
                        for (0..h) |d| {
                            dst[d] = self.workspace.ctx_hidden[d] + m_slice[d];
                        }
                    }
                }
            }
        }

        // 4. Execute `3L`/`5L` block-parallel transformer layers with cached context KV (or single-token fallback) + Partial NeoX RoPE + SWA + sink_bias + v_scale.
        const block_extras: BlockAttnExtras = .{
            .rope_dim = self.config.effectiveRopeDim(),
            .sliding_window = self.config.sliding_window,
            .max_ctx = max_ctx,
            .v_scale = self.config.attention_value_scale,
        };
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
                    self.workspace.ctx_k_cache[layer_off .. layer_off + max_ctx * kv_dim],
                    self.workspace.ctx_v_cache[layer_off .. layer_off + max_ctx * kv_dim],
                    self.workspace.cpu_quant_row_scratch,
                    block_extras,
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
                    self.workspace.cpu_quant_row_scratch,
                    block_extras,
                );
            }
        }

        if (self.weights.final_norm) |fnw| {
            for (0..w) |pos| {
                applyRmsNormInPlace(self.workspace.slot_hiddens[pos * h ..][0..h], fnw, self.config.rms_eps);
            }
        }

        // 5. For `dflash-Q8` (`markov_w2_vocab`), pre-score `base_logits = lm_head(slot_hiddens[1..w])` in ONE batched GPU call!
        self.base_logits_ready = false;
        if (is_real and self.hasMarkovW2Vocab() and w > 1 and (hasLmHeadHook(self.weights.lm_head) or self.weights.lm_head_raw != null)) {
            const n_draft: u8 = @intCast(w - 1);
            try evalLmHead(
                self.weights.lm_head,
                self.weights.lm_head_raw,
                self.workspace.slot_hiddens[h .. w * h],
                n_draft,
                self.config.hidden_size,
                self.config.vocab_size,
                self.workspace.base_logits_buf[0 .. @as(usize, n_draft) * v],
                self.workspace.cpu_quant_row_scratch,
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
            if (self.weights.markov_w1_vocab_bf16) |w1_tbl| {
                const w1_row = w1_tbl[tok_clamped * mr ..][0..mr];
                for (0..mr) |i| {
                    self.workspace.markov_concat[mr + i] = quant.bf16ToF32(w1_row[i]);
                }
            } else if (self.weights.markov_w1_vocab_raw) |w1r| {
                try readRawRowF32(w1r, tok_clamped, self.workspace.markov_concat[mr .. 2 * mr]);
            } else {
                return error.InvalidWeights;
            }
            @memcpy(self.workspace.markov_concat[2 * mr .. 2 * mr + h], b_j);
            @memcpy(self.workspace.markov_concat[2 * mr + h .. joint_len], self.workspace.ctx_markov[0..h]);

            const gates = self.workspace.markov_u[0 .. 3 * mr];
            matVecLayer(
                self.weights.markov_gru_proj_f32 orelse &.{},
                self.weights.markov_gru_proj_bf16,
                self.weights.markov_gru_proj_raw,
                self.workspace.markov_concat[0..joint_len],
                3 * mr,
                joint_len,
                gates,
                false,
                self.workspace.cpu_quant_row_scratch,
            );
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

            if (self.hasMarkovW2Vocab()) {
                // Exact `dspark.cpp:382-401`: `logits = base_logits[pos-1] + w2_vocab @ markov + w2_bias`.
                const v_rows: usize = @min(@as(usize, self.weights.markov_w2_vocab_rows), v);
                matVecLayer(
                    &.{},
                    self.weights.markov_w2_vocab_bf16,
                    self.weights.markov_w2_vocab_raw,
                    markov_vec,
                    v_rows,
                    mr,
                    self.workspace.logits_buf[0..v_rows],
                    false,
                    self.workspace.cpu_quant_row_scratch,
                );
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
            matVecLayer(
                self.weights.markov_w2,
                null,
                self.weights.markov_w2_raw,
                markov_vec,
                h,
                mr,
                self.workspace.head_hidden[0..h],
                false,
                self.workspace.cpu_quant_row_scratch,
            );
            if (self.weights.markov_hidden_bias) |hb| {
                for (0..h) |d| self.workspace.head_hidden[d] += hb[d];
            }
            for (0..h) |d| {
                self.workspace.head_hidden[d] += b_j[d];
            }
            try evalLmHead(
                self.weights.lm_head,
                self.weights.lm_head_raw,
                self.workspace.head_hidden[0..h],
                1,
                self.config.hidden_size,
                self.config.vocab_size,
                self.workspace.logits_buf[0..v],
                self.workspace.cpu_quant_row_scratch,
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

        const tok_e = try resolveTokEmbed(self.weights.tok_embed, self.weights.tok_embed_raw, self.weights.tok_embed_hook, prev_token, h, self.workspace.tok_embed_tmp);
        if (self.weights.token_embd_norm) |ten| {
            @memcpy(self.workspace.tok_embed_tmp[0..h], tok_e);
            applyRmsNormInPlace(self.workspace.tok_embed_tmp[0..h], ten, self.config.rms_eps);
        }
        const eff_tok_e = if (self.weights.token_embd_norm != null) self.workspace.tok_embed_tmp[0..h] else tok_e;

        const w1_cols = if (self.weights.markov_w1_raw) |r| r.k else (self.weights.markov_w1.len / mr);
        if (w1_cols == 2 * h) {
            @memcpy(self.workspace.markov_concat[0..h], b_j);
            @memcpy(self.workspace.markov_concat[h .. 2 * h], eff_tok_e);
            matVecLayer(
                self.weights.markov_w1,
                null,
                self.weights.markov_w1_raw,
                self.workspace.markov_concat[0 .. 2 * h],
                mr,
                2 * h,
                self.workspace.markov_u[0..mr],
                false,
                self.workspace.cpu_quant_row_scratch,
            );
        } else {
            for (0..h) |d| {
                self.workspace.markov_concat[d] = b_j[d] + eff_tok_e[d];
            }
            matVecLayer(
                self.weights.markov_w1,
                null,
                self.weights.markov_w1_raw,
                self.workspace.markov_concat[0..h],
                mr,
                h,
                self.workspace.markov_u[0..mr],
                false,
                self.workspace.cpu_quant_row_scratch,
            );
        }

        if (self.weights.markov_rnn_proj != null or self.weights.markov_rnn_proj_raw != null) {
            matVecLayer(
                self.weights.markov_rnn_proj orelse &.{},
                null,
                self.weights.markov_rnn_proj_raw,
                parent_rnn,
                mr,
                mr,
                self.workspace.markov_u[0..mr],
                true,
                self.workspace.cpu_quant_row_scratch,
            );
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

        matVecLayer(
            self.weights.markov_w2,
            null,
            self.weights.markov_w2_raw,
            out_rnn,
            h,
            mr,
            self.workspace.head_hidden[0..h],
            false,
            self.workspace.cpu_quant_row_scratch,
        );
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
            self.weights.lm_head_raw,
            self.workspace.head_hidden[0..h],
            1,
            self.config.hidden_size,
            self.config.vocab_size,
            self.workspace.logits_buf[0..v],
            self.workspace.cpu_quant_row_scratch,
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

    /// Executes `DFlare` drafting directly from GPU-resident target hidden taps
    /// (and optional GPU-resident circular context tap ring) with zero host tap copies.
    pub fn proposeWithGpuTaps(
        self: *DFlareProposer,
        root_token: u32,
        gpu_taps: GpuTargetTaps,
        gpu_ctx_history: ?GpuContextHistory,
        sam: ?*const sam_mod.SuffixAutomaton,
        out_tree: *CandidateTree,
    ) Error!DFlareProposeResult {
        return self.proposeIntoConstrainedGpu(
            CandidateTree.capacity,
            root_token,
            gpu_taps,
            gpu_ctx_history,
            sam,
            null,
            out_tree,
        );
    }

    pub fn proposeIntoWithGpuTaps(
        self: *DFlareProposer,
        comptime max_nodes: u16,
        root_token: u32,
        gpu_taps: GpuTargetTaps,
        gpu_ctx_history: ?GpuContextHistory,
        sam: ?*const sam_mod.SuffixAutomaton,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!DFlareProposeResult {
        return self.proposeIntoConstrainedGpu(
            max_nodes,
            root_token,
            gpu_taps,
            gpu_ctx_history,
            sam,
            null,
            out_tree,
        );
    }

    pub fn proposeIntoConstrainedGpu(
        self: *DFlareProposer,
        comptime max_nodes: u16,
        root_token: u32,
        gpu_taps: GpuTargetTaps,
        gpu_ctx_history: ?GpuContextHistory,
        sam: ?*const sam_mod.SuffixAutomaton,
        grammar_opt: ?accept_mod.GrammarTreeAdapter,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!DFlareProposeResult {
        const gpu = self.gpu_ctx orelse return error.InvalidWeights;
        if (gpu_taps.taps.len != self.config.num_target_taps) return error.DimensionMismatch;
        return self.proposeIntoConstrainedGpuImpl(max_nodes, gpu, root_token, null, gpu_taps, gpu_ctx_history, sam, grammar_opt, out_tree);
    }

    /// Populates `DraftTree(max_nodes)` with a linear chain (`top_k_per_pos == 1`,
    /// length `block_size = 8`) or a breadth-first top-k tree (`<= max_tree_nodes`),
    /// executing the `3L`/`5L` backbone ONCE and stepping the low-rank `rnn_hidden`
    /// Markov head sequentially with zero dynamic heap allocations.
    /// Automatically uses Metal 4 GPU when `attachGpu` has been called.
    pub fn proposeIntoConstrained(
        self: *DFlareProposer,
        comptime max_nodes: u16,
        root_token: u32,
        target_hiddens: []const []const f32,
        sam: ?*const sam_mod.SuffixAutomaton,
        grammar_opt: ?accept_mod.GrammarTreeAdapter,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!DFlareProposeResult {
        if (self.gpu_ctx) |gpu| {
            return self.proposeIntoConstrainedGpuImpl(max_nodes, gpu, root_token, target_hiddens, null, null, sam, grammar_opt, out_tree);
        }
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

        if (self.isDirectBlockDiffusion()) {
            try self.proposeDirectBlockDiffusionCpu(
                max_nodes,
                root_token,
                target_hiddens,
                node_limit,
                gram_active,
                grammar_opt,
                &node_states,
                &node_valid,
                out_tree,
            );
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

    /// Inserts direct block-diffusion predictions (`slots 1..B-1`, each with up to 2 candidates)
    /// into `out_tree` as a primary `B-1`-deep chain plus optional top-2 sibling forks at uncertain
    /// block slots up to `node_limit` (e.g. `16` tree nodes). Zero dynamic heap allocations.
    fn insertDirectBlockIntoTree(
        self: *DFlareProposer,
        comptime max_nodes: u16,
        draft_slots: usize,
        slot_cands: []const [2]moe.TopKCandidate,
        slot_cand_counts: []const u8,
        node_limit: u16,
        gram_active: bool,
        grammar_opt: ?accept_mod.GrammarTreeAdapter,
        node_states: *[tree_mod.MAX_SUPPORTED_NODES]grammar_mod.GrammarState,
        node_valid: *[tree_mod.MAX_SUPPORTED_NODES]bool,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) void {
        const v: usize = self.config.vocab_size;
        var parent_of_slot: [MAX_BLOCK_SIZE]u16 = @splat(tree_mod.NO_NODE);
        var parent: u16 = 0;

        // 1. Primary greedy/top-1 block chain `root -> slot 1 -> ... -> slot B-1`.
        for (0..draft_slots) |si| {
            if (gram_active and !node_valid[parent]) break;
            if (slot_cand_counts[si] == 0) break;
            const c0 = slot_cands[si][0];
            if (c0.token_id == std.math.maxInt(u32) or c0.token_id >= v) break;
            var child_state: grammar_mod.GrammarState = undefined;
            if (gram_active) {
                child_state = node_states[parent].snapshot();
                if (!grammar_opt.?.advanceTokenOnState(&child_state, c0.token_id)) break;
            }
            parent_of_slot[si] = parent;
            const child = if (out_tree.findChild(parent, c0.token_id)) |existing| blk: {
                break :blk out_tree.insertChild(parent, c0.token_id, c0.prob, .l2) orelse existing;
            } else blk: {
                if (out_tree.count >= node_limit) break :blk null;
                break :blk out_tree.insertChild(parent, c0.token_id, c0.prob, .l2);
            };
            if (child) |c_idx| {
                self.workspace.node_log_probs[c_idx] = c0.log_prob;
                if (gram_active) {
                    node_states[c_idx] = child_state;
                    node_valid[c_idx] = true;
                }
                parent = c_idx;
            } else break;
        }

        // 2. Optional top-2 fork at uncertain block slots up to `node_limit` (`<= 16` nodes).
        const want_fork = (self.config.fork_top2_threshold > 0.0 or self.config.top_k_per_pos >= 2);
        if (!want_fork or out_tree.count >= node_limit) return;

        const ForkCand = struct {
            parent: u16,
            cand: moe.TopKCandidate,
            mass: f32,
        };
        var forks: [MAX_BLOCK_SIZE]ForkCand = undefined;
        var fork_len: usize = 0;
        for (0..draft_slots) |si| {
            const p_node = parent_of_slot[si];
            if (p_node == tree_mod.NO_NODE) continue;
            if (slot_cand_counts[si] < 2) continue;
            const c0 = slot_cands[si][0];
            const c1 = slot_cands[si][1];
            if (c1.token_id == std.math.maxInt(u32) or c1.token_id >= v or c1.token_id == c0.token_id) continue;
            if (self.config.fork_top2_threshold > 0.0 and c0.prob >= self.config.fork_top2_threshold) continue;
            forks[fork_len] = .{
                .parent = p_node,
                .cand = c1,
                .mass = out_tree.nodes[p_node].cum_prob * c1.prob,
            };
            fork_len += 1;
        }
        // Deterministic descending selection sort by joint mass (tie-break by lower token_id).
        var fi: usize = 0;
        while (fi < fork_len) : (fi += 1) {
            var best = fi;
            var fj: usize = fi + 1;
            while (fj < fork_len) : (fj += 1) {
                if (forks[fj].mass > forks[best].mass or
                    (forks[fj].mass == forks[best].mass and forks[fj].cand.token_id < forks[best].cand.token_id))
                {
                    best = fj;
                }
            }
            const tmp = forks[fi];
            forks[fi] = forks[best];
            forks[best] = tmp;
        }
        for (forks[0..fork_len]) |fk| {
            if (out_tree.count >= node_limit) break;
            if (out_tree.findChild(fk.parent, fk.cand.token_id) != null) continue;
            var child_state: grammar_mod.GrammarState = undefined;
            if (gram_active) {
                if (!node_valid[fk.parent]) continue;
                child_state = node_states[fk.parent].snapshot();
                if (!grammar_opt.?.advanceTokenOnState(&child_state, fk.cand.token_id)) continue;
            }
            if (out_tree.insertChild(fk.parent, fk.cand.token_id, fk.cand.prob, .l2)) |c_idx| {
                self.workspace.node_log_probs[c_idx] = fk.cand.log_prob;
                if (gram_active) {
                    node_states[c_idx] = child_state;
                    node_valid[c_idx] = true;
                }
            }
        }
    }

    fn proposeDirectBlockDiffusionCpu(
        self: *DFlareProposer,
        comptime max_nodes: u16,
        root_token: u32,
        target_hiddens: []const []const f32,
        node_limit: u16,
        gram_active: bool,
        grammar_opt: ?accept_mod.GrammarTreeAdapter,
        node_states: *[tree_mod.MAX_SUPPORTED_NODES]grammar_mod.GrammarState,
        node_valid: *[tree_mod.MAX_SUPPORTED_NODES]bool,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!void {
        const h: usize = self.config.hidden_size;
        const w: usize = self.config.block_size;
        const v: usize = self.config.vocab_size;
        const eff_v: usize = if (self.config.unpadded_vocab_size > 0 and self.config.unpadded_vocab_size < v) self.config.unpadded_vocab_size else v;
        const draft_slots: usize = if (w > 1) w - 1 else 1;
        const slot_offset: usize = if (w > 1) 1 else 0;
        const eff_k: usize = if (self.config.fork_top2_threshold > 0.0 or self.config.top_k_per_pos >= 2) 2 else 1;
        const inv_temp: f32 = 1.0 / self.config.temperature;

        var slot_cands: [MAX_BLOCK_SIZE][2]moe.TopKCandidate = undefined;
        var slot_counts: [MAX_BLOCK_SIZE]u8 = @splat(0);

        const iters: u8 = self.config.refinement_iters;
        for (0..iters) |iter| {
            if (iter == 0) {
                try self.runBlockBackboneOnceForRootImpl(root_token, target_hiddens, false);
            } else {
                // Iteration 2 refinement: retain high-confidence predicted token embeddings, re-mask uncertain slots.
                const eff_mask = try self.resolveEffectiveMaskEmbed();
                const anchor_e = try resolveTokEmbed(self.weights.tok_embed, self.weights.tok_embed_raw, self.weights.tok_embed_hook, root_token, h, self.workspace.tok_embed_tmp);
                @memcpy(self.workspace.slot_hiddens[0..h], anchor_e[0..h]);
                if (self.weights.tok_embed_hook == null) {
                    if (self.weights.token_embd_norm) |ten| applyRmsNormInPlace(self.workspace.slot_hiddens[0..h], ten, self.config.rms_eps);
                }
                for (0..draft_slots) |si| {
                    const dst = self.workspace.slot_hiddens[(slot_offset + si) * h ..][0..h];
                    if (slot_counts[si] > 0 and slot_cands[si][0].prob >= self.config.refine_confidence_threshold and slot_cands[si][0].token_id < v) {
                        const te = try resolveTokEmbed(self.weights.tok_embed, self.weights.tok_embed_raw, self.weights.tok_embed_hook, slot_cands[si][0].token_id, h, self.workspace.tok_embed_tmp);
                        @memcpy(dst, te[0..h]);
                        if (self.weights.tok_embed_hook == null) {
                            if (self.weights.token_embd_norm) |ten| applyRmsNormInPlace(dst, ten, self.config.rms_eps);
                        }
                    } else {
                        const m_slice = if (eff_mask.len == w * h)
                            eff_mask[(slot_offset + si) * h ..][0..h]
                        else if (w > 1 and eff_mask.len == (w - 1) * h)
                            eff_mask[si * h ..][0..h]
                        else
                            eff_mask[0..h];
                        @memcpy(dst, m_slice);
                    }
                }
                try self.runBlockBackboneOnceForRootImpl(root_token, target_hiddens, true);
            }

            // Score all `draft_slots` in ONE batched `lm_head` call.
            try evalLmHead(
                self.weights.lm_head,
                self.weights.lm_head_raw,
                self.workspace.slot_hiddens[slot_offset * h .. (slot_offset + draft_slots) * h],
                @intCast(draft_slots),
                self.config.hidden_size,
                self.config.vocab_size,
                self.workspace.base_logits_buf[0 .. draft_slots * v],
                self.workspace.cpu_quant_row_scratch,
            );

            for (0..draft_slots) |si| {
                const row = self.workspace.base_logits_buf[si * v ..][0..eff_v];
                if (self.config.final_logit_softcapping > 0.0) {
                    const cap = self.config.final_logit_softcapping;
                    for (row) |*lv| lv.* = std.math.tanh(lv.* / cap) * cap;
                }
                const got = try extractDeterministicTopKMasked(
                    row,
                    inv_temp,
                    null,
                    self.workspace.topk_tokens[0..eff_k],
                    self.workspace.topk_probs[0..eff_k],
                );
                slot_counts[si] = @intCast(got);
                for (0..got) |ki| {
                    const p = self.workspace.topk_probs[ki];
                    slot_cands[si][ki] = .{
                        .token_id = self.workspace.topk_tokens[ki],
                        .prob = p,
                        .log_prob = logProbOf(p),
                    };
                }
            }
        }

        self.insertDirectBlockIntoTree(
            max_nodes,
            draft_slots,
            slot_cands[0..draft_slots],
            slot_counts[0..draft_slots],
            node_limit,
            gram_active,
            grammar_opt,
            node_states,
            node_valid,
            out_tree,
        );
    }

    // ========================================================================
    // Metal 4 GPU Execution Path (`D16`, ADR-0026)
    // ========================================================================

    fn encodeGpuLinear(
        gpu: *const DFlareGpuContext,
        frame: *metal.Compute.Frame,
        lin: GpuLinearBinding,
        x: metal.BufferBinding,
        y: metal.BufferBinding,
        m: u32,
        residual: ?metal.BufferBinding,
        y_is_f32_if_f32: bool,
        insert_barrier_before: bool,
    ) Error!void {
        if (lin.amf_dtype == .f32) {
            const p: DFlareGemvF32Params = .{
                .m = m,
                .n = lin.n,
                .k = lin.k,
                .has_residual = if (residual != null) 1 else 0,
                .x_is_f32 = 0,
                .y_is_f32 = if (y_is_f32_if_f32) 1 else 0,
                .res_is_f32 = 0,
            };
            const gx: u64 = (@as(u64, lin.n) + 3) / 4;
            try frame.encodeThreadgroups(
                gpu.pipes.gemv_batch_f32,
                &.{ x, lin.weight, y, residual orelse y },
                std.mem.asBytes(&p),
                .{ gx, m, 1 },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
            return;
        }
        const qf_ptr = gpu.qfPipesFor(lin.amf_dtype, lin.group_size);
        try gpu.dequant_ptr.encodeLinearUnified(frame, .{
            .amf_dtype = lin.amf_dtype,
            .group_size = lin.group_size,
            .tensor_scale_per_row = lin.tensor_scale_per_row,
            .weight = lin.weight,
            .params = lin.params,
            .x = x,
            .y = y,
            .residual = residual,
            .m = m,
            .n = lin.n,
            .k = lin.k,
            .mode = if (m == 1) .gemv else .batched,
            .qf_pipes = qf_ptr,
        }, insert_barrier_before);
    }

    fn encodeGpuContextKvBatch(
        self: *const DFlareProposer,
        gpu: *const DFlareGpuContext,
        frame: *metal.Compute.Frame,
        taps_src: metal.BufferBinding,
        taps_dtype: u32,
        ring_cap: u32,
        ring_start: u32,
        num_pos: u32,
        kv_start_pos: u32,
        rope_zero_pos: bool,
        use_ctx_hidden_direct: bool,
        first_barrier: bool,
    ) Error!void {
        const ws = self.workspace;
        const wb = gpu.ws_buf;
        const h: u32 = self.config.hidden_size;
        const taps: u32 = self.config.num_target_taps;
        const kv_dim: u32 = self.config.kvDim();
        const max_ctx: u32 = ws.max_ctx_len;
        const ctx_taps_f16_b = ws.sliceBinding(wb, ws.gpu_ctx_taps_f16);
        const ctx_in_b = ws.sliceBinding(wb, ws.gpu_ctx_in_f16);
        const k_tmp_b = ws.sliceBinding(wb, ws.gpu_ctx_k_tmp_f16);
        const v_tmp_b = ws.sliceBinding(wb, ws.gpu_ctx_v_tmp_f16);
        const ctx_k_b = ws.sliceBinding(wb, ws.ctx_k_cache);
        const ctx_v_b = ws.sliceBinding(wb, ws.ctx_v_cache);
        const ctx_h_b = ws.sliceBinding(wb, ws.gpu_ctx_hidden_f16);
        const dummy_b = ctx_h_b;

        // `DFlash` (`fc.weight [H, taps*H]` + `hidden_norm.weight`): compute fused context hidden
        // ONCE per new context position and share across all `num_layers` layers (`dflash.py:253`).
        const use_shared_fc_ctx = (!use_ctx_hidden_direct and gpu.fuse_scalar_weights == null and gpu.target_fusion_proj != null);
        var shared_ctx_ready = false;
        if (use_shared_fc_ctx) {
            const pcp: DFlarePackCtxTapsParams = .{
                .num_pos = num_pos,
                .num_taps = taps,
                .hidden_size = h,
                .taps_dtype = taps_dtype,
                .ring_cap = ring_cap,
                .ring_start = ring_start,
            };
            const row_cols: u32 = taps * h;
            try frame.encodeThreads(
                gpu.pipes.pack_ctx_taps_f16,
                &.{ taps_src, ctx_taps_f16_b },
                std.mem.asBytes(&pcp),
                .{ row_cols, num_pos, 1 },
                .{ @min(row_cols, 256), 1, 1 },
                first_barrier,
            );
            try encodeGpuLinear(gpu, frame, gpu.target_fusion_proj.?, ctx_taps_f16_b, ctx_in_b, num_pos, null, false, true);
            if (gpu.target_fusion_norm) |fnw| {
                const np: DFlareRmsNormParams = .{
                    .num_rows = num_pos,
                    .dim = h,
                    .x_is_f32 = 0,
                    .has_weight = 1,
                    .write_f32 = 0,
                    .eps = self.config.rms_eps,
                };
                try frame.encodeThreadgroups(
                    gpu.pipes.rmsnorm_rows,
                    &.{ ctx_in_b, fnw, ctx_in_b, dummy_b },
                    std.mem.asBytes(&np),
                    .{ num_pos, 1, 1 },
                    .{ 256, 1, 1 },
                    true,
                );
            }
            shared_ctx_ready = true;
        }

        for (0..self.config.num_layers) |li| {
            const gl = gpu.layers[li];
            const barrier = if (li == 0 and !shared_ctx_ready) first_barrier else true;
            var kv_input = ctx_in_b;
            var kv_barrier = true;

            if (shared_ctx_ready) {
                if (gl.ctx_norm_w) |cnw| {
                    const np: DFlareRmsNormParams = .{
                        .num_rows = num_pos,
                        .dim = h,
                        .x_is_f32 = 0,
                        .has_weight = 1,
                        .write_f32 = 0,
                        .eps = self.config.rms_eps,
                    };
                    const tmp_norm_b = ws.sliceBinding(wb, ws.gpu_norm_f16);
                    if (num_pos <= self.config.block_size) {
                        try frame.encodeThreadgroups(
                            gpu.pipes.rmsnorm_rows,
                            &.{ ctx_in_b, cnw, tmp_norm_b, dummy_b },
                            std.mem.asBytes(&np),
                            .{ num_pos, 1, 1 },
                            .{ 256, 1, 1 },
                            barrier,
                        );
                        kv_input = tmp_norm_b;
                    } else {
                        try frame.encodeThreadgroups(
                            gpu.pipes.rmsnorm_rows,
                            &.{ ctx_in_b, cnw, ctx_in_b, dummy_b },
                            std.mem.asBytes(&np),
                            .{ num_pos, 1, 1 },
                            .{ 256, 1, 1 },
                            barrier,
                        );
                    }
                } else {
                    kv_input = ctx_in_b;
                    kv_barrier = barrier;
                }
            } else if (use_ctx_hidden_direct) {
                if (gl.ctx_norm_w) |cnw| {
                    const np: DFlareRmsNormParams = .{
                        .num_rows = 1,
                        .dim = h,
                        .x_is_f32 = 0,
                        .has_weight = 1,
                        .write_f32 = 0,
                        .eps = self.config.rms_eps,
                    };
                    try frame.encodeThreadgroups(
                        gpu.pipes.rmsnorm_rows,
                        &.{ ctx_h_b, cnw, ctx_in_b, dummy_b },
                        std.mem.asBytes(&np),
                        .{ 1, 1, 1 },
                        .{ 256, 1, 1 },
                        barrier,
                    );
                } else {
                    kv_input = ctx_h_b;
                    kv_barrier = barrier;
                }
            } else {
                const fsw_opt = gpu.fuse_scalar_weights;
                const mix_b: metal.BufferBinding = if (fsw_opt) |fsw| blk: {
                    const row_idx: u64 = if ((li + 1) * taps <= gpu.fuse_rows * taps) li else 0;
                    break :blk .{ .buffer = fsw.buffer, .offset = fsw.offset + row_idx * taps * @sizeOf(f32) };
                } else dummy_b;
                const norm_opt = gl.ctx_norm_w orelse if (fsw_opt == null) gpu.target_fusion_norm else null;
                const mp: MixTapsParams = .{
                    .num_pos = num_pos,
                    .num_taps = taps,
                    .hidden_size = h,
                    .taps_dtype = taps_dtype,
                    .ring_cap = ring_cap,
                    .ring_start = ring_start,
                    .has_mix_w = if (fsw_opt != null) 1 else 0,
                    .apply_norm = if (norm_opt != null) 1 else 0,
                    .has_norm_w = if (norm_opt != null) 1 else 0,
                    .write_f32 = 0,
                    .rms_eps = self.config.rms_eps,
                };
                try frame.encodeThreadgroups(
                    gpu.pipes.mix_taps_rmsnorm,
                    &.{ taps_src, mix_b, norm_opt orelse dummy_b, ctx_in_b, dummy_b },
                    std.mem.asBytes(&mp),
                    .{ num_pos, 1, 1 },
                    .{ 256, 1, 1 },
                    barrier,
                );
            }

            try encodeGpuLinear(gpu, frame, gl.k_proj_ctx orelse gl.k_proj, kv_input, k_tmp_b, num_pos, null, false, kv_barrier);
            try encodeGpuLinear(gpu, frame, gl.v_proj_ctx orelse gl.v_proj, kv_input, v_tmp_b, num_pos, null, false, false);

            const kn_w = gl.k_norm_ctx_w orelse gl.k_norm_w;
            const sp: DFlareCtxKvStoreParams = .{
                .num_pos = num_pos,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .start_pos = kv_start_pos,
                .max_ctx = max_ctx,
                .layer_elem_off = @intCast(li * max_ctx * kv_dim),
                .has_k_norm = if (kn_w != null) 1 else 0,
                .rope_zero_pos = if (rope_zero_pos) 1 else 0,
                .rms_eps = self.config.rms_eps,
                .rope_theta = self.config.rope_theta,
                .rope_dim = self.config.effectiveRopeDim(),
            };
            try frame.encodeThreadgroups(
                gpu.pipes.ctx_kv_norm_rope_store,
                &.{ k_tmp_b, v_tmp_b, kn_w orelse dummy_b, ctx_k_b, ctx_v_b },
                std.mem.asBytes(&sp),
                .{ self.config.num_key_value_heads, num_pos, 1 },
                .{ 128, 1, 1 },
                true,
            );
        }
    }

    fn encodeGpuBackboneIntoFrame(
        self: *DFlareProposer,
        gpu: *const DFlareGpuContext,
        frame: *metal.Compute.Frame,
        anchor_pos: u32,
        active_ctx_len: u32,
        need_ctx_kv_single: bool,
        gpu_taps_opt: ?GpuTargetTaps,
        first_barrier: bool,
        preseeded_slots: bool,
    ) Error!void {
        const ws = self.workspace;
        const wb = gpu.ws_buf;
        const h: u32 = self.config.hidden_size;
        const w: u32 = self.config.block_size;
        const inter: u32 = self.config.intermediate_size;
        const q_dim: u32 = self.config.qDim();
        const kv_dim: u32 = self.config.kvDim();
        const taps: u32 = self.config.num_target_taps;
        const anchor_mask_mode = self.usesAnchorMaskSeeding();

        const taps_f32_b = ws.sliceBinding(wb, ws.taps_flat);
        const taps_f16_b = ws.sliceBinding(wb, ws.gpu_taps_flat_f16);
        const ctx_h_b = ws.sliceBinding(wb, ws.gpu_ctx_hidden_f16);
        const ctx_mf16_b = ws.sliceBinding(wb, ws.gpu_ctx_markov_f16);
        const ctx_mf32_b = ws.sliceBinding(wb, ws.gpu_ctx_markov_f32);
        const anchor_b = ws.sliceBinding(wb, ws.gpu_anchor_f16);
        const slot_b = ws.sliceBinding(wb, ws.gpu_slot_hiddens_f16);
        const norm_b = ws.sliceBinding(wb, ws.gpu_norm_f16);
        const q_b = ws.sliceBinding(wb, ws.gpu_q_f16);
        const k_b = ws.sliceBinding(wb, ws.gpu_k_f16);
        const v_b = ws.sliceBinding(wb, ws.gpu_v_f16);
        const ctx_k_b = ws.sliceBinding(wb, ws.ctx_k_cache);
        const ctx_v_b = ws.sliceBinding(wb, ws.ctx_v_cache);
        const attn_out_b = ws.sliceBinding(wb, ws.gpu_attn_out_f16);
        const mlp_gate_b = ws.sliceBinding(wb, ws.gpu_mlp_gate_f16);
        const mlp_up_b = ws.sliceBinding(wb, ws.gpu_mlp_up_f16);
        const mlp_down_b = ws.sliceBinding(wb, ws.gpu_mlp_down_f16);
        const dummy_b = ctx_h_b;

        var taps_src_b = taps_f32_b;
        var taps_dtype: u32 = 0;
        var cur_barrier = first_barrier;

        if (gpu_taps_opt) |gt| {
            var pack_binds: [9]metal.BufferBinding = undefined;
            for (0..8) |ti| {
                pack_binds[ti] = if (ti < gt.taps.len) gt.taps[ti] else gt.taps[0];
            }
            pack_binds[8] = taps_f16_b;
            const pp: PackTapsParams = .{
                .num_taps = taps,
                .hidden_size = h,
                .src_is_f16 = if (gt.dtype == .f16) 1 else 0,
                .dst_is_f16 = 1,
            };
            try frame.encodeThreads(
                gpu.pipes.pack_taps8,
                &pack_binds,
                std.mem.asBytes(&pp),
                .{ h, taps, 1 },
                .{ @min(h, 256), 1, 1 },
                cur_barrier,
            );
            taps_src_b = taps_f16_b;
            taps_dtype = 1;
            cur_barrier = true;
        }

        // Compute `ctx_hidden` and `ctx_markov` (`f_rnn`).
        if (gpu.fuse_scalar_weights) |fsw| {
            const mp0: MixTapsParams = .{
                .num_pos = 1,
                .num_taps = taps,
                .hidden_size = h,
                .taps_dtype = taps_dtype,
                .ring_cap = 0,
                .ring_start = 0,
                .has_mix_w = 1,
                .apply_norm = if (gpu.target_fusion_norm != null) 1 else 0,
                .has_norm_w = if (gpu.target_fusion_norm != null) 1 else 0,
                .write_f32 = 0,
                .rms_eps = self.config.rms_eps,
            };
            try frame.encodeThreadgroups(
                gpu.pipes.mix_taps_rmsnorm,
                &.{ taps_src_b, fsw, gpu.target_fusion_norm orelse dummy_b, ctx_h_b, dummy_b },
                std.mem.asBytes(&mp0),
                .{ 1, 1, 1 },
                .{ 256, 1, 1 },
                cur_barrier,
            );
            const m_row: u64 = if (gpu.fuse_rows >= self.config.num_layers + 1) self.config.num_layers else 0;
            const m_fsw: metal.BufferBinding = .{ .buffer = fsw.buffer, .offset = fsw.offset + m_row * taps * @sizeOf(f32) };
            const mp1: MixTapsParams = .{
                .num_pos = 1,
                .num_taps = taps,
                .hidden_size = h,
                .taps_dtype = taps_dtype,
                .ring_cap = 0,
                .ring_start = 0,
                .has_mix_w = 1,
                .apply_norm = if (gpu.markov_ctx_norm != null) 1 else 0,
                .has_norm_w = if (gpu.markov_ctx_norm != null) 1 else 0,
                .write_f32 = 1,
                .rms_eps = self.config.rms_eps,
            };
            try frame.encodeThreadgroups(
                gpu.pipes.mix_taps_rmsnorm,
                &.{ taps_src_b, m_fsw, gpu.markov_ctx_norm orelse dummy_b, ctx_mf16_b, ctx_mf32_b },
                std.mem.asBytes(&mp1),
                .{ 1, 1, 1 },
                .{ 256, 1, 1 },
                false,
            );
        } else if (gpu.target_fusion_proj) |fp| {
            // For `target_fusion_proj`, input is `taps_f16_b` (already packed if `gpu_taps_opt`, else converted on host).
            try encodeGpuLinear(gpu, frame, fp, taps_f16_b, ctx_h_b, 1, null, false, cur_barrier);
            const copy_p: MixTapsParams = .{
                .num_pos = 1,
                .num_taps = 1,
                .hidden_size = h,
                .taps_dtype = 1,
                .ring_cap = 0,
                .ring_start = 0,
                .has_mix_w = 0,
                .apply_norm = 0,
                .has_norm_w = 0,
                .write_f32 = 1,
                .rms_eps = self.config.rms_eps,
            };
            try frame.encodeThreadgroups(
                gpu.pipes.mix_taps_rmsnorm,
                &.{ ctx_h_b, dummy_b, dummy_b, ctx_mf16_b, ctx_mf32_b },
                std.mem.asBytes(&copy_p),
                .{ 1, 1, 1 },
                .{ 256, 1, 1 },
                true,
            );
            if (gpu.target_fusion_norm) |fnw| {
                const np: DFlareRmsNormParams = .{
                    .num_rows = 1,
                    .dim = h,
                    .x_is_f32 = 0,
                    .has_weight = 1,
                    .write_f32 = 0,
                    .eps = self.config.rms_eps,
                };
                try frame.encodeThreadgroups(
                    gpu.pipes.rmsnorm_rows,
                    &.{ ctx_h_b, fnw, ctx_h_b, dummy_b },
                    std.mem.asBytes(&np),
                    .{ 1, 1, 1 },
                    .{ 256, 1, 1 },
                    true,
                );
            }
        } else {
            const cp_m: MixTapsParams = .{
                .num_pos = 1,
                .num_taps = 1,
                .hidden_size = h,
                .taps_dtype = taps_dtype,
                .ring_cap = 0,
                .ring_start = 0,
                .has_mix_w = 0,
                .apply_norm = 0,
                .has_norm_w = 0,
                .write_f32 = 1,
                .rms_eps = self.config.rms_eps,
            };
            try frame.encodeThreadgroups(
                gpu.pipes.mix_taps_rmsnorm,
                &.{ taps_src_b, dummy_b, dummy_b, ctx_mf16_b, ctx_mf32_b },
                std.mem.asBytes(&cp_m),
                .{ 1, 1, 1 },
                .{ 256, 1, 1 },
                cur_barrier,
            );
            const cp_h: MixTapsParams = .{
                .num_pos = 1,
                .num_taps = 1,
                .hidden_size = h,
                .taps_dtype = taps_dtype,
                .ring_cap = 0,
                .ring_start = 0,
                .has_mix_w = 0,
                .apply_norm = if (gpu.target_fusion_norm != null) 1 else 0,
                .has_norm_w = if (gpu.target_fusion_norm != null) 1 else 0,
                .write_f32 = 0,
                .rms_eps = self.config.rms_eps,
            };
            try frame.encodeThreadgroups(
                gpu.pipes.mix_taps_rmsnorm,
                &.{ taps_src_b, dummy_b, gpu.target_fusion_norm orelse dummy_b, ctx_h_b, dummy_b },
                std.mem.asBytes(&cp_h),
                .{ 1, 1, 1 },
                .{ 256, 1, 1 },
                false,
            );
        }

        // Single-position fallback context KV if no context history provider was active.
        if (need_ctx_kv_single) {
            try self.encodeGpuContextKvBatch(
                gpu,
                frame,
                taps_src_b,
                taps_dtype,
                0,
                0,
                1,
                0,
                true,
                gpu.fuse_scalar_weights == null,
                true,
            );
        }

        // Seed `slot_hiddens[0..w-1]` (skipped when caller pre-populated refined slot embeddings).
        if (!preseeded_slots) {
            const seed_p: SeedBlockParams = .{
                .block_size = w,
                .hidden_size = h,
                .is_real_gru = if (anchor_mask_mode) 1 else 0,
                .mask_rows = gpu.mask_rows,
                .anchor_is_f16 = 1,
            };
            try frame.encodeThreads(
                gpu.pipes.seed_block,
                &.{ anchor_b, ctx_h_b, gpu.mask_embed, slot_b },
                std.mem.asBytes(&seed_p),
                .{ h, w, 1 },
                .{ @min(h, 256), 1, 1 },
                true,
            );
        }

        // Execute all `num_layers` (`3L` / `5L`) block-parallel transformer layers.
        const max_ctx: u32 = ws.max_ctx_len;
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.config.head_dim)));
        for (0..self.config.num_layers) |li| {
            const gl = gpu.layers[li];
            const in_np: DFlareRmsNormParams = .{
                .num_rows = w,
                .dim = h,
                .x_is_f32 = 0,
                .has_weight = if (gl.in_norm_w != null) 1 else 0,
                .write_f32 = 0,
                .eps = self.config.rms_eps,
            };
            try frame.encodeThreadgroups(
                gpu.pipes.rmsnorm_rows,
                &.{ slot_b, gl.in_norm_w orelse dummy_b, norm_b, dummy_b },
                std.mem.asBytes(&in_np),
                .{ w, 1, 1 },
                .{ 256, 1, 1 },
                true,
            );

            try encodeGpuLinear(gpu, frame, gl.q_proj, norm_b, q_b, w, null, false, true);
            try encodeGpuLinear(gpu, frame, gl.k_proj, norm_b, k_b, w, null, false, false);
            try encodeGpuLinear(gpu, frame, gl.v_proj, norm_b, v_b, w, null, false, false);

            if (gl.q_norm_w != null or gl.k_norm_w != null or self.config.rope_theta > 0.0) {
                const qp: DFlareQkNormRopeParams = .{
                    .seq_len = w,
                    .num_q_heads = self.config.num_attention_heads,
                    .num_kv_heads = self.config.num_key_value_heads,
                    .head_dim = self.config.head_dim,
                    .anchor_pos = anchor_pos,
                    .has_q_norm = if (gl.q_norm_w != null) 1 else 0,
                    .has_k_norm = if (gl.k_norm_w != null) 1 else 0,
                    .rope_dim = self.config.effectiveRopeDim(),
                    .rms_eps = self.config.rms_eps,
                    .rope_theta = self.config.rope_theta,
                };
                try frame.encodeThreadgroups(
                    gpu.pipes.qk_norm_rope,
                    &.{ q_b, k_b, gl.q_norm_w orelse dummy_b, gl.k_norm_w orelse dummy_b },
                    std.mem.asBytes(&qp),
                    .{ self.config.num_attention_heads + self.config.num_key_value_heads, w, 1 },
                    .{ 128, 1, 1 },
                    true,
                );
            }

            const gqa_p: DFlareBlockGqaParams = .{
                .seq_len = w,
                .active_ctx_len = active_ctx_len,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .layer_elem_off = @intCast(li * max_ctx * kv_dim),
                .scale = attn_scale,
                .v_scale = self.config.attention_value_scale,
                .max_ctx = max_ctx,
                .anchor_pos = anchor_pos,
                .sliding_window = self.config.sliding_window,
                .has_sink_bias = if (gl.sink_bias_w != null) 1 else 0,
            };
            try frame.encodeThreadgroups(
                gpu.pipes.block_gqa_attn,
                &.{ q_b, ctx_k_b, ctx_v_b, k_b, v_b, attn_out_b, gl.sink_bias_w orelse dummy_b },
                std.mem.asBytes(&gqa_p),
                .{ self.config.num_attention_heads, w, 1 },
                .{ 128, 1, 1 },
                true,
            );

            // `o_proj` + fused residual into `slot_hiddens`.
            try encodeGpuLinear(gpu, frame, gl.o_proj, attn_out_b, slot_b, w, slot_b, false, true);

            // Post-attention RMSNorm + SwiGLU MLP + fused residual.
            const post_np: DFlareRmsNormParams = .{
                .num_rows = w,
                .dim = h,
                .x_is_f32 = 0,
                .has_weight = if (gl.post_norm_w != null) 1 else 0,
                .write_f32 = 0,
                .eps = self.config.rms_eps,
            };
            try frame.encodeThreadgroups(
                gpu.pipes.rmsnorm_rows,
                &.{ slot_b, gl.post_norm_w orelse dummy_b, norm_b, dummy_b },
                std.mem.asBytes(&post_np),
                .{ w, 1, 1 },
                .{ 256, 1, 1 },
                true,
            );

            try encodeGpuLinear(gpu, frame, gl.mlp_gate, norm_b, mlp_gate_b, w, null, false, true);
            try encodeGpuLinear(gpu, frame, gl.mlp_up, norm_b, mlp_up_b, w, null, false, false);

            const sw_cnt: u32 = w * inter;
            const sw_p: DFlareElemParams = .{ .count = sw_cnt };
            try frame.encodeThreads(
                gpu.pipes.swiglu_mul,
                &.{ mlp_gate_b, mlp_up_b },
                std.mem.asBytes(&sw_p),
                .{ sw_cnt, 1, 1 },
                .{ @min(sw_cnt, 256), 1, 1 },
                true,
            );

            try encodeGpuLinear(gpu, frame, gl.mlp_down, mlp_gate_b, slot_b, w, slot_b, false, true);
            _ = mlp_down_b;
            _ = q_dim;
        }

        if (gpu.final_norm) |fnw| {
            const fnp: DFlareRmsNormParams = .{
                .num_rows = w,
                .dim = h,
                .x_is_f32 = 0,
                .has_weight = 1,
                .write_f32 = 0,
                .eps = self.config.rms_eps,
            };
            try frame.encodeThreadgroups(
                gpu.pipes.rmsnorm_rows,
                &.{ slot_b, fnw, slot_b, dummy_b },
                std.mem.asBytes(&fnp),
                .{ w, 1, 1 },
                .{ 256, 1, 1 },
                true,
            );
        }
    }

    fn encodeGpuMarkovHeadStep(
        self: *const DFlareProposer,
        gpu: *const DFlareGpuContext,
        frame: *metal.Compute.Frame,
        pos: u8,
        use_tok_buf: bool,
        tok_slot: u32,
        imm_tok: u32,
        parent_rnn_b: metal.BufferBinding,
        out_rnn_b: metal.BufferBinding,
        score_on_gpu: bool,
        first_barrier: bool,
    ) Error!struct { logits_b: metal.BufferBinding, logits_is_f16: bool, softcap_done: bool, eff_vocab: u32 } {
        const ws = self.workspace;
        const wb = gpu.ws_buf;
        const h: u32 = self.config.hidden_size;
        const mr: u32 = self.config.markov_rank;
        const v: u32 = self.config.vocab_size;

        const slot_b = ws.sliceBinding(wb, ws.gpu_slot_hiddens_f16);
        const ctx_mf16_b = ws.sliceBinding(wb, ws.gpu_ctx_markov_f16);
        const tok_chain_b = ws.sliceBinding(wb, ws.gpu_token_chain_u32);
        const concat_f16_b = ws.sliceBinding(wb, ws.gpu_markov_concat_f16);
        const concat_f32_b = ws.sliceBinding(wb, ws.gpu_markov_concat_f32);
        const gates_f16_b = ws.sliceBinding(wb, ws.gpu_markov_gates_f16);
        const gates_f32_b = ws.sliceBinding(wb, ws.gpu_markov_gates_f32);
        const mvec_f16_b = ws.sliceBinding(wb, ws.gpu_markov_vec_f16);
        const mvec_f32_b = ws.sliceBinding(wb, ws.gpu_markov_vec_f32);
        const head_h_b = ws.sliceBinding(wb, ws.gpu_head_hidden_f16);
        const base_log_b = ws.sliceBinding(wb, ws.gpu_base_logits_f16);
        const logits_f16_b = ws.sliceBinding(wb, ws.gpu_logits_f16);
        const logits_f32_b = ws.sliceBinding(wb, ws.gpu_logits_f32);
        const w1_row_b = ws.sliceBinding(wb, ws.gpu_w1_row_f32);
        const dummy_b = mvec_f32_b;

        if (self.isRealGru()) {
            const w1_b = gpu.markov_w1_vocab orelse w1_row_b;
            const w1_rows_eff: u32 = if (gpu.markov_w1_vocab != null) gpu.markov_w1_vocab_rows else 1;
            const ap: DFlareMarkovAssembleParams = .{
                .hidden_size = h,
                .markov_rank = mr,
                .slot_pos = pos,
                .w1_rows = w1_rows_eff,
                .w1_dtype = gpu.markov_w1_vocab_dtype,
                .use_tok_buf = if (use_tok_buf and gpu.markov_w1_vocab != null) 1 else 0,
                .tok_slot = tok_slot,
                .imm_tok = if (gpu.markov_w1_vocab != null) imm_tok else 0,
                .write_f32 = 0,
            };
            const joint_len: u32 = 2 * mr + 2 * h;
            try frame.encodeThreads(
                gpu.pipes.markov_gru_assemble,
                &.{ parent_rnn_b, w1_b, slot_b, ctx_mf16_b, tok_chain_b, concat_f16_b, concat_f32_b },
                std.mem.asBytes(&ap),
                .{ joint_len, 1, 1 },
                .{ @min(joint_len, 256), 1, 1 },
                first_barrier,
            );

            const gp = gpu.markov_gru_proj.?;
            const gates_is_f32 = (gp.amf_dtype == .f32);
            try encodeGpuLinear(
                gpu,
                frame,
                gp,
                concat_f16_b,
                if (gates_is_f32) gates_f32_b else gates_f16_b,
                1,
                null,
                gates_is_f32,
                true,
            );

            const up_p: DFlareMarkovGruUpdateParams = .{
                .markov_rank = mr,
                .gates_is_f32 = if (gates_is_f32) 1 else 0,
                .has_bias = if (gpu.markov_gru_bias != null) 1 else 0,
            };
            try frame.encodeThreads(
                gpu.pipes.markov_gru_update,
                &.{
                    if (gates_is_f32) gates_f32_b else gates_f16_b,
                    gpu.markov_gru_bias orelse dummy_b,
                    parent_rnn_b,
                    out_rnn_b,
                    mvec_f16_b,
                    mvec_f32_b,
                },
                std.mem.asBytes(&up_p),
                .{ mr, 1, 1 },
                .{ @min(mr, 256), 1, 1 },
                true,
            );

            if (gpu.markov_w2_vocab) |w2v| {
                const v_rows: u32 = @min(gpu.markov_w2_vocab_rows, v);
                const has_base: u32 = if (self.base_logits_ready and pos >= 1 and pos < self.config.block_size) 1 else 0;
                const base_off: u32 = if (has_base != 0) @as(u32, pos - 1) * v else 0;
                if (w2v.amf_dtype == .bf16 or w2v.amf_dtype == .f16 or w2v.amf_dtype == .f32) {
                    const w2_dt: u32 = switch (w2v.amf_dtype) {
                        .bf16 => 0,
                        .f16 => 1,
                        .f32 => 2,
                        else => 0,
                    };
                    const wp: DFlareW2VocabParams = .{
                        .vocab_size = v,
                        .v_rows = v_rows,
                        .markov_rank = mr,
                        .w2_dtype = w2_dt,
                        .has_bias = if (gpu.markov_w2_vocab_bias != null) 1 else 0,
                        .has_base_logits = has_base,
                        .base_row_elem_off = base_off,
                        .softcap = self.config.final_logit_softcapping,
                    };
                    try frame.encodeThreadgroups(
                        gpu.pipes.markov_w2_vocab_score,
                        &.{ mvec_f32_b, w2v.weight, gpu.markov_w2_vocab_bias orelse dummy_b, base_log_b, logits_f32_b },
                        std.mem.asBytes(&wp),
                        .{ (v + 3) / 4, 1, 1 },
                        .{ 128, 1, 1 },
                        true,
                    );
                } else {
                    try encodeGpuLinear(gpu, frame, w2v, mvec_f16_b, logits_f16_b, 1, null, false, true);
                    const ep: DFlareLogitsEpilogueParams = .{
                        .vocab_size = v,
                        .v_rows = v_rows,
                        .has_bias = if (gpu.markov_w2_vocab_bias != null) 1 else 0,
                        .has_base_logits = has_base,
                        .base_row_elem_off = base_off,
                        .in_is_f16 = 1,
                        .softcap = self.config.final_logit_softcapping,
                    };
                    try frame.encodeThreads(
                        gpu.pipes.logits_epilogue,
                        &.{ logits_f16_b, gpu.markov_w2_vocab_bias orelse dummy_b, base_log_b, logits_f32_b },
                        std.mem.asBytes(&ep),
                        .{ v, 1, 1 },
                        .{ 256, 1, 1 },
                        true,
                    );
                }
                return .{ .logits_b = logits_f32_b, .logits_is_f16 = false, .softcap_done = true, .eff_vocab = v_rows };
            }

            // `mtp.safetensors` `hidden_out`: `b_j + hidden_out @ markov_vec + bias` -> `lm_head`.
            const w2_lin = gpu.markov_w2.?;
            const w2_is_f32 = (w2_lin.amf_dtype == .f32);
            const w2_out_b = if (w2_is_f32) concat_f32_b else concat_f16_b;
            try encodeGpuLinear(gpu, frame, w2_lin, mvec_f16_b, w2_out_b, 1, null, w2_is_f32, true);

            const cp: DFlareMarkovCombineParams = .{
                .hidden_size = h,
                .w2_is_f32 = if (w2_is_f32) 1 else 0,
                .has_bias = if (gpu.markov_hidden_bias != null) 1 else 0,
                .has_norm = 0,
                .slot_pos = pos,
                .rms_eps = self.config.rms_eps,
            };
            try frame.encodeThreadgroups(
                gpu.pipes.markov_head_combine,
                &.{ w2_out_b, gpu.markov_hidden_bias orelse dummy_b, slot_b, dummy_b, head_h_b },
                std.mem.asBytes(&cp),
                .{ 1, 1, 1 },
                .{ 256, 1, 1 },
                true,
            );

            const eff_v: u32 = if (self.config.unpadded_vocab_size > 0 and self.config.unpadded_vocab_size < v)
                self.config.unpadded_vocab_size
            else
                v;
            if (score_on_gpu) {
                if (gpu.lm_head) |lmh| {
                    try encodeGpuLinear(gpu, frame, lmh, head_h_b, logits_f16_b, 1, null, false, true);
                    return .{ .logits_b = logits_f16_b, .logits_is_f16 = true, .softcap_done = false, .eff_vocab = eff_v };
                }
            }
            return .{ .logits_b = head_h_b, .logits_is_f16 = true, .softcap_done = false, .eff_vocab = eff_v };
        }

        // Non-GRU simple SiLU Markov head.
        const tok_b = gpu.tok_embed orelse return error.InvalidWeights;
        const w1_lin = gpu.markov_w1.?;
        const concat_2h: u32 = if (w1_lin.k == 2 * h) 1 else 0;
        const has_tnorm: u32 = if (self.weights.tok_embed_hook == null and gpu.token_embd_norm != null) 1 else 0;
        const sp: DFlareSimpleMarkovPrepParams = .{
            .hidden_size = h,
            .vocab_size = v,
            .slot_pos = pos,
            .use_tok_buf = if (use_tok_buf) 1 else 0,
            .tok_slot = tok_slot,
            .imm_tok = imm_tok,
            .tok_dtype = gpu.tok_embed_dtype,
            .has_tok_norm = has_tnorm,
            .concat_2h = concat_2h,
            .write_f32 = 0,
            .rms_eps = self.config.rms_eps,
        };
        try frame.encodeThreadgroups(
            gpu.pipes.simple_markov_prep,
            &.{ tok_b, gpu.token_embd_norm orelse dummy_b, slot_b, tok_chain_b, concat_f16_b, concat_f32_b },
            std.mem.asBytes(&sp),
            .{ 1, 1, 1 },
            .{ 256, 1, 1 },
            first_barrier,
        );

        const u_is_f32 = (w1_lin.amf_dtype == .f32);
        const u_out_b = if (u_is_f32) gates_f32_b else gates_f16_b;
        try encodeGpuLinear(gpu, frame, w1_lin, concat_f16_b, u_out_b, 1, null, u_is_f32, true);

        const add_parent_direct = (gpu.markov_rnn_proj == null);
        if (gpu.markov_rnn_proj) |rp_lin| {
            // Note: `parent_rnn_b` is `f32`, while `encodeGpuLinear` expects `f16` x; copy/cast via `mix_taps_rmsnorm` (`f32 -> f16`) or use `mvec_f16_b`!
            const cast_p: MixTapsParams = .{
                .num_pos = 1,
                .num_taps = 1,
                .hidden_size = mr,
                .taps_dtype = 0,
                .ring_cap = 0,
                .ring_start = 0,
                .has_mix_w = 0,
                .apply_norm = 0,
                .has_norm_w = 0,
                .write_f32 = 0,
                .rms_eps = self.config.rms_eps,
            };
            try frame.encodeThreadgroups(
                gpu.pipes.mix_taps_rmsnorm,
                &.{ parent_rnn_b, dummy_b, dummy_b, mvec_f16_b, dummy_b },
                std.mem.asBytes(&cast_p),
                .{ 1, 1, 1 },
                .{ 256, 1, 1 },
                false,
            );
            // Accumulate `rp @ mvec_f16_b` into `u_out_b` (`gates_f32_b` or `gates_f16_b`) via residual!
            if (rp_lin.amf_dtype == .f32 and u_is_f32) {
                const gp_acc: DFlareGemvF32Params = .{
                    .m = 1,
                    .n = rp_lin.n,
                    .k = rp_lin.k,
                    .has_residual = 1,
                    .x_is_f32 = 0,
                    .y_is_f32 = 1,
                    .res_is_f32 = 1,
                };
                try frame.encodeThreadgroups(
                    gpu.pipes.gemv_batch_f32,
                    &.{ mvec_f16_b, rp_lin.weight, gates_f32_b, gates_f32_b },
                    std.mem.asBytes(&gp_acc),
                    .{ (rp_lin.n + 3) / 4, 1, 1 },
                    .{ 128, 1, 1 },
                    true,
                );
            } else {
                // Convert `u_out_b` -> `gates_f16_b` if needed, or run `encodeGpuLinear` with residual `gates_f16_b`!
                if (u_is_f32) {
                    const cast_u: MixTapsParams = .{
                        .num_pos = 1,
                        .num_taps = 1,
                        .hidden_size = mr,
                        .taps_dtype = 0,
                        .ring_cap = 0,
                        .ring_start = 0,
                        .has_mix_w = 0,
                        .apply_norm = 0,
                        .has_norm_w = 0,
                        .write_f32 = 0,
                        .rms_eps = self.config.rms_eps,
                    };
                    try frame.encodeThreadgroups(
                        gpu.pipes.mix_taps_rmsnorm,
                        &.{ gates_f32_b, dummy_b, dummy_b, gates_f16_b, dummy_b },
                        std.mem.asBytes(&cast_u),
                        .{ 1, 1, 1 },
                        .{ 256, 1, 1 },
                        true,
                    );
                }
                try encodeGpuLinear(gpu, frame, rp_lin, mvec_f16_b, gates_f16_b, 1, gates_f16_b, false, true);
            }
        }

        const silu_u_is_f32: u32 = if (u_is_f32 and (gpu.markov_rnn_proj == null or gpu.markov_rnn_proj.?.amf_dtype == .f32)) 1 else 0;
        const silu_p: DFlareSimpleMarkovSiluParams = .{
            .markov_rank = mr,
            .u_is_f32 = silu_u_is_f32,
            .add_parent_rnn = if (add_parent_direct) 1 else 0,
            .has_b1 = if (gpu.markov_b1 != null) 1 else 0,
        };
        try frame.encodeThreads(
            gpu.pipes.simple_markov_silu,
            &.{
                if (silu_u_is_f32 != 0) gates_f32_b else gates_f16_b,
                parent_rnn_b,
                gpu.markov_b1 orelse dummy_b,
                out_rnn_b,
                mvec_f16_b,
            },
            std.mem.asBytes(&silu_p),
            .{ mr, 1, 1 },
            .{ @min(mr, 256), 1, 1 },
            true,
        );

        const w2_lin = gpu.markov_w2.?;
        const w2_is_f32 = (w2_lin.amf_dtype == .f32);
        const w2_out_b = if (w2_is_f32) concat_f32_b else concat_f16_b;
        try encodeGpuLinear(gpu, frame, w2_lin, mvec_f16_b, w2_out_b, 1, null, w2_is_f32, true);

        const cp: DFlareMarkovCombineParams = .{
            .hidden_size = h,
            .w2_is_f32 = if (w2_is_f32) 1 else 0,
            .has_bias = if (gpu.markov_hidden_bias != null) 1 else 0,
            .has_norm = if (gpu.markov_norm != null) 1 else 0,
            .slot_pos = pos,
            .rms_eps = self.config.rms_eps,
        };
        try frame.encodeThreadgroups(
            gpu.pipes.markov_head_combine,
            &.{ w2_out_b, gpu.markov_hidden_bias orelse dummy_b, slot_b, gpu.markov_norm orelse dummy_b, head_h_b },
            std.mem.asBytes(&cp),
            .{ 1, 1, 1 },
            .{ 256, 1, 1 },
            true,
        );

        if (score_on_gpu) {
            if (gpu.lm_head) |lmh| {
                try encodeGpuLinear(gpu, frame, lmh, head_h_b, logits_f16_b, 1, null, false, true);
                return .{ .logits_b = logits_f16_b, .logits_is_f16 = true, .softcap_done = true, .eff_vocab = v };
            }
        }
        return .{ .logits_b = head_h_b, .logits_is_f16 = true, .softcap_done = true, .eff_vocab = v };
    }

    fn encodeGpuTopK(
        gpu: *const DFlareGpuContext,
        ws: *const DFlareWorkspace,
        frame: *metal.Compute.Frame,
        logits_b: metal.BufferBinding,
        logits_is_f16: bool,
        eff_vocab: u32,
        top_k: u32,
        inv_temp: f32,
        softcap: f32,
        has_mask: bool,
        out_cand_offset: u32,
        write_next_tok: bool,
        next_tok_slot: u32,
    ) Error!void {
        const wb = gpu.ws_buf;
        const mask_b = ws.sliceBinding(wb, ws.gpu_grammar_mask_u32);
        const cands_b = ws.sliceBinding(wb, ws.gpu_topk_cands);
        const tok_chain_b = ws.sliceBinding(wb, ws.gpu_token_chain_u32);
        const align_req: u64 = if (logits_is_f16) 8 else 16;
        const kp: DFlareTopKParams = .{
            .eff_vocab_size = eff_vocab,
            .top_k = top_k,
            .logits_is_f16 = if (logits_is_f16) 1 else 0,
            .has_mask = if (has_mask) 1 else 0,
            .out_cand_offset = out_cand_offset,
            .write_next_tok = if (write_next_tok) 1 else 0,
            .next_tok_slot = next_tok_slot,
            .aligned_vec4 = if ((logits_b.offset % align_req) == 0) 1 else 0,
            .inv_temp = inv_temp,
            .softcap = softcap,
        };
        if (top_k == 1) {
            try frame.encodeThreadgroups(
                gpu.pipes.top1_logits,
                &.{ logits_b, mask_b, cands_b, tok_chain_b },
                std.mem.asBytes(&kp),
                .{ 1, 1, 1 },
                .{ 256, 1, 1 },
                true,
            );
        } else {
            try frame.encodeThreadgroups(
                gpu.pipes.topk_logits,
                &.{ logits_b, mask_b, cands_b, tok_chain_b },
                std.mem.asBytes(&kp),
                .{ 1, 1, 1 },
                .{ 128, 1, 1 },
                true,
            );
        }
    }

    fn prepareAnchorAndContextGpu(
        self: *DFlareProposer,
        gpu: *DFlareGpuContext,
        root_token: u32,
        host_taps_opt: ?[]const []const f32,
        gpu_taps_opt: ?GpuTargetTaps,
        gpu_ctx_hist_opt: ?GpuContextHistory,
    ) Error!struct {
        use_cached_ctx: bool,
        active_ctx_len: u32,
        anchor_pos: u32,
        final_chunk_pos: u32,
        final_chunk_len: u32,
        ring_cap: u32,
        ring_start: u32,
        taps_dtype: u32,
    } {
        if (gpu_taps_opt) |gt| {
            for (gt.taps) |tb| try gpu.ensureResident(tb.buffer);
        }
        if (gpu_ctx_hist_opt) |gch| {
            try gpu.ensureResident(gch.taps_ring.buffer);
        }
        const ws = self.workspace;
        const h: usize = self.config.hidden_size;
        const taps: usize = self.config.num_target_taps;
        const max_ctx: u32 = ws.max_ctx_len;
        const anchor_mask_mode = self.usesAnchorMaskSeeding();

        // 0. Resolve anchor token embedding -> `ws.gpu_anchor_f16` (and mask_token_id embedding if needed).
        if (anchor_mask_mode) {
            const raw_anchor = try resolveTokEmbed(self.weights.tok_embed, self.weights.tok_embed_raw, self.weights.tok_embed_hook, root_token, h, ws.tok_embed_tmp);
            if (raw_anchor.ptr != ws.tok_embed_tmp.ptr) {
                @memcpy(ws.tok_embed_tmp[0..h], raw_anchor[0..h]);
            }
            if (self.weights.tok_embed_hook == null) {
                if (self.weights.token_embd_norm) |ten| {
                    applyRmsNormInPlace(ws.tok_embed_tmp[0..h], ten, self.config.rms_eps);
                }
            }
            for (0..h) |d| ws.gpu_anchor_f16[d] = @floatCast(ws.tok_embed_tmp[d]);
            if (self.weights.mask_embed.len == 0) {
                _ = try self.resolveEffectiveMaskEmbed();
            }
        } else if (self.weights.mask_embed.len == 0) {
            _ = try self.resolveEffectiveMaskEmbed();
        }

        // Copy host taps into `ws.taps_flat` and `ws.gpu_taps_flat_f16`.
        if (host_taps_opt) |ht| {
            if (ht.len != taps) return error.DimensionMismatch;
            for (ht, 0..) |tap, ti| {
                if (tap.len != h) return error.DimensionMismatch;
                @memcpy(ws.taps_flat[ti * h ..][0..h], tap);
            }
            if (gpu.target_fusion_proj != null) {
                for (0..taps * h) |i| ws.gpu_taps_flat_f16[i] = @floatCast(ws.taps_flat[i]);
            }
        }

        var use_cached_ctx: bool = false;
        var active_ctx_len: u32 = 1;
        var anchor_pos: u32 = if (self.config.rope_theta > 0.0) 1 else 0;
        var final_chunk_pos: u32 = 0;
        var final_chunk_len: u32 = 0;
        var ring_cap: u32 = 0;
        var ring_start: u32 = 0;
        var taps_dtype: u32 = 0;

        if (anchor_mask_mode) {
            if (gpu_ctx_hist_opt) |gch| {
                if (gch.len > 0 and gch.cap > 0) {
                    if (gch.epoch != self.ctx_cache_epoch or gch.len < self.ctx_cache_len) {
                        self.ctx_cache_len = 0;
                        self.ctx_cache_epoch = gch.epoch;
                    }
                    const win_start: u32 = if (gch.len > max_ctx) gch.len - max_ctx else 0;
                    var p: u32 = @max(self.ctx_cache_len, win_start);
                    const gch_dtype: u32 = @backingInt(gch.dtype);
                    while (p < gch.len) {
                        const rem = gch.len - p;
                        const take: u32 = @min(rem, ws.ctx_batch_cap);
                        if (p + take < gch.len) {
                            // Early catch-up chunk (> ctx_batch_cap): commit & wait so `gpu_ctx_in_f16` stays in bounds.
                            var cf = try gpu.compute_ptr.beginFrame(gpu.timeout_ms);
                            errdefer cf.abort();
                            try self.encodeGpuContextKvBatch(
                                gpu,
                                &cf,
                                gch.taps_ring,
                                gch_dtype,
                                gch.cap,
                                p % gch.cap,
                                take,
                                p,
                                false,
                                false,
                                false,
                            );
                            const sig = try cf.commit();
                            try gpu.compute_ptr.wait(sig, gpu.timeout_ms);
                            p += take;
                        } else {
                            final_chunk_pos = p;
                            final_chunk_len = take;
                            ring_cap = gch.cap;
                            ring_start = p % gch.cap;
                            taps_dtype = gch_dtype;
                            break;
                        }
                    }
                    self.ctx_cache_len = gch.len;
                    active_ctx_len = @min(gch.len, max_ctx);
                    anchor_pos = gch.len;
                    use_cached_ctx = active_ctx_len > 0;
                }
            } else {
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
                        const win_start: u32 = if (hist_len > max_ctx) hist_len - max_ctx else 0;
                        var p: u32 = @max(self.ctx_cache_len, win_start);
                        const wb = gpu.ws_buf;
                        const ctx_taps_b = ws.sliceBinding(wb, ws.gpu_ctx_taps_f32);
                        while (p < hist_len) {
                            const rem = hist_len - p;
                            const take: u32 = @min(rem, ws.ctx_batch_cap);
                            var actual: u32 = 0;
                            while (actual < take) : (actual += 1) {
                                var ok_all = true;
                                for (0..taps) |ti| {
                                    var tmp_l: u32 = 0;
                                    var tmp_e: u64 = 0;
                                    const dst = ws.gpu_ctx_taps_f32[(@as(usize, actual) * taps + ti) * h ..][0..h];
                                    if (!prov_fn(cb_ctx.?, &tmp_l, &tmp_e, p + actual, ti, dst)) {
                                        ok_all = false;
                                        break;
                                    }
                                }
                                if (!ok_all) break;
                            }
                            if (actual == 0) break;
                            if (p + actual < hist_len) {
                                // Early catch-up chunk (> ctx_batch_cap): commit & wait before overwriting staging buffer.
                                var cf = try gpu.compute_ptr.beginFrame(gpu.timeout_ms);
                                errdefer cf.abort();
                                try self.encodeGpuContextKvBatch(gpu, &cf, ctx_taps_b, 0, 0, 0, actual, p, false, false, false);
                                const sig = try cf.commit();
                                try gpu.compute_ptr.wait(sig, gpu.timeout_ms);
                                p += actual;
                            } else {
                                // Final chunk stays in the main draft MTL4CommandBuffer!
                                final_chunk_pos = p;
                                final_chunk_len = actual;
                                p += actual;
                                break;
                            }
                        }
                        self.ctx_cache_len = hist_len;
                        active_ctx_len = @min(hist_len, max_ctx);
                        anchor_pos = hist_len;
                        use_cached_ctx = active_ctx_len > 0;
                    }
                }
            }
        }

        return .{
            .use_cached_ctx = use_cached_ctx,
            .active_ctx_len = active_ctx_len,
            .anchor_pos = anchor_pos,
            .final_chunk_pos = final_chunk_pos,
            .final_chunk_len = final_chunk_len,
            .ring_cap = ring_cap,
            .ring_start = ring_start,
            .taps_dtype = taps_dtype,
        };
    }

    fn runBlockBackboneGpuImpl(
        self: *DFlareProposer,
        gpu: *DFlareGpuContext,
        root_token: u32,
        host_taps_opt: ?[]const []const f32,
        gpu_taps_opt: ?GpuTargetTaps,
        gpu_ctx_hist_opt: ?GpuContextHistory,
        sync_host_outputs: bool,
    ) Error!void {
        const ws = self.workspace;
        const wb = gpu.ws_buf;
        const h: usize = self.config.hidden_size;
        const w: usize = self.config.block_size;
        const v: usize = self.config.vocab_size;

        const prep = try self.prepareAnchorAndContextGpu(gpu, root_token, host_taps_opt, gpu_taps_opt, gpu_ctx_hist_opt);
        var frame = try gpu.compute_ptr.beginFrame(gpu.timeout_ms);
        errdefer frame.abort();

        var first_b = false;
        if (prep.final_chunk_len > 0) {
            const taps_b = if (gpu_ctx_hist_opt) |gch| gch.taps_ring else ws.sliceBinding(wb, ws.gpu_ctx_taps_f32);
            try self.encodeGpuContextKvBatch(
                gpu,
                &frame,
                taps_b,
                prep.taps_dtype,
                prep.ring_cap,
                prep.ring_start,
                prep.final_chunk_len,
                prep.final_chunk_pos,
                false,
                false,
                false,
            );
            first_b = true;
        }

        try self.encodeGpuBackboneIntoFrame(
            gpu,
            &frame,
            prep.anchor_pos,
            prep.active_ctx_len,
            !prep.use_cached_ctx,
            gpu_taps_opt,
            first_b,
            false,
        );

        self.base_logits_ready = false;
        const need_base = self.isRealGru() and self.hasMarkovW2Vocab() and w > 1;
        var base_computed_gpu = false;
        if (need_base) {
            if (gpu.lm_head) |lmh| {
                const slot_draft_b = metal.BufferBinding{
                    .buffer = wb,
                    .offset = ws.sliceBinding(wb, ws.gpu_slot_hiddens_f16).offset + h * @sizeOf(f16),
                };
                try encodeGpuLinear(gpu, &frame, lmh, slot_draft_b, ws.sliceBinding(wb, ws.gpu_base_logits_f16), @intCast(w - 1), null, false, true);
                self.base_logits_ready = true;
                base_computed_gpu = true;
            }
        }

        const sig = try frame.commit();
        try gpu.compute_ptr.wait(sig, gpu.timeout_ms);

        if (sync_host_outputs or (need_base and !base_computed_gpu and (hasLmHeadHook(self.weights.lm_head) or self.weights.lm_head_raw != null))) {
            for (0..w * h) |i| ws.slot_hiddens[i] = @floatCast(ws.gpu_slot_hiddens_f16[i]);
            @memcpy(ws.ctx_markov[0..h], ws.gpu_ctx_markov_f32[0..h]);
            for (0..h) |d| ws.ctx_hidden[d] = @floatCast(ws.gpu_ctx_hidden_f16[d]);
        }
        if (need_base and !base_computed_gpu and (hasLmHeadHook(self.weights.lm_head) or self.weights.lm_head_raw != null)) {
            const n_draft: u8 = @intCast(w - 1);
            try evalLmHead(
                self.weights.lm_head,
                self.weights.lm_head_raw,
                ws.slot_hiddens[h .. w * h],
                n_draft,
                self.config.hidden_size,
                self.config.vocab_size,
                ws.base_logits_buf[0 .. @as(usize, n_draft) * v],
                ws.cpu_quant_row_scratch,
            );
            for (0..@as(usize, n_draft) * v) |i| ws.gpu_base_logits_f16[i] = @floatCast(ws.base_logits_buf[i]);
            self.base_logits_ready = true;
        }
        self.backbone_runs += 1;
    }

    fn proposeIntoConstrainedGpuImpl(
        self: *DFlareProposer,
        comptime max_nodes: u16,
        gpu: *DFlareGpuContext,
        root_token: u32,
        host_taps_opt: ?[]const []const f32,
        gpu_taps_opt: ?GpuTargetTaps,
        gpu_ctx_hist_opt: ?GpuContextHistory,
        sam: ?*const sam_mod.SuffixAutomaton,
        grammar_opt: ?accept_mod.GrammarTreeAdapter,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!DFlareProposeResult {
        out_tree.reset(root_token);
        const ws = self.workspace;
        const wb = gpu.ws_buf;
        const node_limit: u16 = @min(@as(u16, self.config.max_tree_nodes), max_nodes);
        const mr: usize = self.config.markov_rank;
        const v: usize = self.config.vocab_size;
        const w: u8 = self.config.block_size;
        const k: usize = self.config.top_k_per_pos;

        @memset(ws.node_log_probs[0..node_limit], 0.0);
        @memset(ws.node_rnn_hidden[0..mr], 0.0);

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
            ws.l1_draft_buf,
            ws.node_log_probs,
            gram_active,
            grammar_opt,
            &node_states,
            &node_valid,
        );

        const base_backbone = self.backbone_runs;
        const base_markov = self.markov_steps;
        if (self.isDirectBlockDiffusion()) {
            try self.proposeDirectBlockDiffusionGpu(
                max_nodes,
                gpu,
                root_token,
                host_taps_opt,
                gpu_taps_opt,
                gpu_ctx_hist_opt,
                node_limit,
                gram_active,
                grammar_opt,
                &node_states,
                &node_valid,
                out_tree,
            );
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

        const pos_beg: usize = if (self.isRealGru() and w > 1) 1 else 0;
        const inv_temp: f32 = 1.0 / self.config.temperature;

        const has_gpu_w2_vocab = (gpu.markov_w2_vocab != null);
        const has_gpu_lm_head = (gpu.lm_head != null);
        const has_any_lm_head = hasLmHeadHook(self.weights.lm_head) or self.weights.lm_head_raw != null;
        const can_gpu_score = has_gpu_w2_vocab or has_gpu_lm_head;
        const can_single_frame_chain = (k == 1) and
            (!gram_active) and
            (gpu.markov_w1_vocab_raw == null) and
            (has_gpu_w2_vocab or has_gpu_lm_head) and
            (!has_gpu_w2_vocab or has_gpu_lm_head or !has_any_lm_head);

        const chain_end: usize = @min(@as(usize, w), pos_beg + @as(usize, node_limit -| 1));
        if (can_single_frame_chain and pos_beg < chain_end) {
            // SINGLE-FRAME FULLY-CHAINED METAL 4 PATH:
            // Context KV update + 3L/5L backbone + base_logits + all Markov steps + Top-1 reduction
            // execute in ONE `MTL4CommandBuffer` with ZERO intermediate CPU-GPU flushes!
            const prep = try self.prepareAnchorAndContextGpu(gpu, root_token, host_taps_opt, gpu_taps_opt, gpu_ctx_hist_opt);
            ws.gpu_token_chain_u32[0] = root_token;
            @memset(ws.gpu_markov_rnn_ping_f32[0..mr], 0.0);

            var frame = try gpu.compute_ptr.beginFrame(gpu.timeout_ms);
            errdefer frame.abort();

            var first_b = false;
            if (prep.final_chunk_len > 0) {
                const taps_b = if (gpu_ctx_hist_opt) |gch| gch.taps_ring else ws.sliceBinding(wb, ws.gpu_ctx_taps_f32);
                try self.encodeGpuContextKvBatch(
                    gpu,
                    &frame,
                    taps_b,
                    prep.taps_dtype,
                    prep.ring_cap,
                    prep.ring_start,
                    prep.final_chunk_len,
                    prep.final_chunk_pos,
                    false,
                    false,
                    false,
                );
                first_b = true;
            }

            try self.encodeGpuBackboneIntoFrame(
                gpu,
                &frame,
                prep.anchor_pos,
                prep.active_ctx_len,
                !prep.use_cached_ctx,
                gpu_taps_opt,
                first_b,
                false,
            );

            self.base_logits_ready = false;
            if (self.isRealGru() and has_gpu_w2_vocab and chain_end > 1) {
                if (gpu.lm_head) |lmh| {
                    const slot_draft_b = metal.BufferBinding{
                        .buffer = wb,
                        .offset = ws.sliceBinding(wb, ws.gpu_slot_hiddens_f16).offset + self.config.hidden_size * @sizeOf(f16),
                    };
                    try encodeGpuLinear(gpu, &frame, lmh, slot_draft_b, ws.sliceBinding(wb, ws.gpu_base_logits_f16), @intCast(chain_end - 1), null, false, true);
                    self.base_logits_ready = true;
                }
            }

            const ping_base_b = ws.sliceBinding(wb, ws.gpu_markov_rnn_ping_f32);
            const ping_stride: u64 = mr * @sizeOf(f32);
            for (pos_beg..chain_end) |pos_usize| {
                const step_idx: u32 = @intCast(pos_usize - pos_beg);
                const in_ping: metal.BufferBinding = .{ .buffer = wb, .offset = ping_base_b.offset + (step_idx & 1) * ping_stride };
                const out_ping: metal.BufferBinding = .{ .buffer = wb, .offset = ping_base_b.offset + ((step_idx + 1) & 1) * ping_stride };
                const scored = try self.encodeGpuMarkovHeadStep(
                    gpu,
                    &frame,
                    @intCast(pos_usize),
                    true,
                    step_idx,
                    0,
                    in_ping,
                    out_ping,
                    true,
                    true,
                );
                const softcap_arg: f32 = if (scored.softcap_done) 0.0 else self.config.final_logit_softcapping;
                try encodeGpuTopK(
                    gpu,
                    ws,
                    &frame,
                    scored.logits_b,
                    scored.logits_is_f16,
                    scored.eff_vocab,
                    1,
                    inv_temp,
                    softcap_arg,
                    false,
                    step_idx,
                    pos_usize + 1 < chain_end,
                    step_idx + 1,
                );
            }

            const sig = try frame.commit();
            try gpu.compute_ptr.wait(sig, gpu.timeout_ms);
            self.backbone_runs += 1;

            var parent: u16 = 0;
            for (pos_beg..chain_end) |pos_usize| {
                const step_idx = pos_usize - pos_beg;
                const cand = ws.gpu_topk_cands[step_idx];
                self.markov_steps += 1;
                if (cand.token_id == std.math.maxInt(u32) or cand.token_id >= v) break;
                const child = if (out_tree.findChild(parent, cand.token_id)) |existing| blk: {
                    break :blk out_tree.insertChild(parent, cand.token_id, cand.prob, .l2) orelse existing;
                } else blk: {
                    if (out_tree.count >= node_limit) break :blk null;
                    break :blk out_tree.insertChild(parent, cand.token_id, cand.prob, .l2);
                };
                if (child) |c_idx| {
                    ws.node_log_probs[c_idx] = cand.log_prob;
                    parent = c_idx;
                } else break;
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

        // General GPU path for `top_k > 1`, `gram_active`, or host `LmHeadHook.callback` fallback.
        try self.runBlockBackboneGpuImpl(gpu, root_token, host_taps_opt, gpu_taps_opt, gpu_ctx_hist_opt, !can_gpu_score);

        var frontier_cur: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
        var frontier_next: [tree_mod.MAX_SUPPORTED_NODES]u16 = undefined;
        var cur_len: usize = 1;
        frontier_cur[0] = 0;

        const ping_base_b = ws.sliceBinding(wb, ws.gpu_markov_rnn_ping_f32);
        const in_ping_b = ping_base_b;
        const out_ping_b: metal.BufferBinding = .{ .buffer = wb, .offset = ping_base_b.offset + mr * @sizeOf(f32) };

        for (pos_beg..w) |pos_usize| {
            const pos: u8 = @intCast(pos_usize);
            sortFrontierDescending(max_nodes, out_tree, frontier_cur[0..cur_len]);

            var next_len: usize = 0;
            for (frontier_cur[0..cur_len]) |parent| {
                if (gram_active and !node_valid[parent]) continue;
                const parent_tok = out_tree.nodes[parent].token_id;
                if (parent_tok >= v) return error.TokenOutOfRange;
                const parent_rnn = ws.node_rnn_hidden[@as(usize, parent) * mr ..][0..mr];
                @memcpy(ws.gpu_markov_rnn_ping_f32[0..mr], parent_rnn);

                if (gpu.markov_w1_vocab_raw) |w1r| {
                    if (gpu.markov_w1_vocab == null) {
                        const tok_clamped: usize = if (parent_tok < self.weights.markov_w1_vocab_rows) parent_tok else 0;
                        try readRawRowF32(w1r, tok_clamped, ws.gpu_w1_row_f32[0..mr]);
                    }
                }

                var mask_opt: ?[]const u32 = null;
                if (gram_active) {
                    var parent_ga = grammar_opt.?;
                    var pstate = node_states[parent].snapshot();
                    parent_ga.state = &pstate;
                    mask_opt = parent_ga.fillBitmask(v) catch return error.InvalidLogits;
                    if (mask_opt) |mwords| {
                        const copy_w = @min(mwords.len, ws.gpu_grammar_mask_u32.len);
                        @memcpy(ws.gpu_grammar_mask_u32[0..copy_w], mwords[0..copy_w]);
                        if (copy_w < ws.gpu_grammar_mask_u32.len) {
                            @memset(ws.gpu_grammar_mask_u32[copy_w..], 0);
                        }
                    }
                }

                var frame = try gpu.compute_ptr.beginFrame(gpu.timeout_ms);
                errdefer frame.abort();
                const scored = try self.encodeGpuMarkovHeadStep(
                    gpu,
                    &frame,
                    pos,
                    false,
                    0,
                    parent_tok,
                    in_ping_b,
                    out_ping_b,
                    can_gpu_score,
                    false,
                );
                if (can_gpu_score) {
                    const softcap_arg: f32 = if (scored.softcap_done) 0.0 else self.config.final_logit_softcapping;
                    try encodeGpuTopK(
                        gpu,
                        ws,
                        &frame,
                        scored.logits_b,
                        scored.logits_is_f16,
                        scored.eff_vocab,
                        @intCast(k),
                        inv_temp,
                        softcap_arg,
                        gram_active and mask_opt != null,
                        0,
                        false,
                        0,
                    );
                }
                const sig = try frame.commit();
                try gpu.compute_ptr.wait(sig, gpu.timeout_ms);

                const next_rnn = ws.gpu_markov_rnn_ping_f32[mr .. 2 * mr];
                self.markov_steps += 1;

                var cand_count: usize = 0;
                if (can_gpu_score) {
                    for (0..k) |ki| {
                        const c = ws.gpu_topk_cands[ki];
                        if (c.token_id == std.math.maxInt(u32) or c.token_id >= v) break;
                        ws.topk_tokens[cand_count] = c.token_id;
                        ws.topk_probs[cand_count] = c.prob;
                        cand_count += 1;
                    }
                } else {
                    const h: usize = self.config.hidden_size;
                    for (0..h) |d| ws.head_hidden[d] = @floatCast(ws.gpu_head_hidden_f16[d]);
                    try evalLmHead(
                        self.weights.lm_head,
                        self.weights.lm_head_raw,
                        ws.head_hidden[0..h],
                        1,
                        self.config.hidden_size,
                        self.config.vocab_size,
                        ws.logits_buf[0..v],
                        ws.cpu_quant_row_scratch,
                    );
                    const eff_v: usize = scored.eff_vocab;
                    if (self.isRealGru() and self.config.final_logit_softcapping > 0.0) {
                        const cap = self.config.final_logit_softcapping;
                        for (0..eff_v) |ti| ws.logits_buf[ti] = cap * std.math.tanh(ws.logits_buf[ti] / cap);
                    }
                    if (eff_v < v) @memset(ws.logits_buf[eff_v..v], -1e30);
                    cand_count = try extractDeterministicTopKMasked(
                        ws.logits_buf[0..v],
                        inv_temp,
                        mask_opt,
                        ws.topk_tokens[0..k],
                        ws.topk_probs[0..k],
                    );
                }

                for (0..cand_count) |ki| {
                    const tok = ws.topk_tokens[ki];
                    const prob = ws.topk_probs[ki];
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
                        ws.node_log_probs[c_idx] = logProbOf(prob);
                        @memcpy(ws.node_rnn_hidden[@as(usize, c_idx) * mr ..][0..mr], next_rnn);
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

    /// Single-frame batched Metal 4 GPU path for `D15` `DFlash` direct parallel block-diffusion
    /// (`markov_rank == 0`, `block_size = 8..16`, `0` scalar CPU loops on GPU path, optional
    /// 2-iteration refinement and top-2 uncertain-slot branching up to `max_tree_nodes`).
    fn proposeDirectBlockDiffusionGpu(
        self: *DFlareProposer,
        comptime max_nodes: u16,
        gpu: *DFlareGpuContext,
        root_token: u32,
        host_taps_opt: ?[]const []const f32,
        gpu_taps_opt: ?GpuTargetTaps,
        gpu_ctx_hist_opt: ?GpuContextHistory,
        node_limit: u16,
        gram_active: bool,
        grammar_opt: ?accept_mod.GrammarTreeAdapter,
        node_states: *[tree_mod.MAX_SUPPORTED_NODES]grammar_mod.GrammarState,
        node_valid: *[tree_mod.MAX_SUPPORTED_NODES]bool,
        out_tree: *tree_mod.DraftTree(max_nodes),
    ) Error!void {
        const ws = self.workspace;
        const wb = gpu.ws_buf;
        const h: usize = self.config.hidden_size;
        const w: usize = self.config.block_size;
        const v: usize = self.config.vocab_size;
        const eff_v: u32 = if (self.config.unpadded_vocab_size > 0 and self.config.unpadded_vocab_size < v) self.config.unpadded_vocab_size else self.config.vocab_size;
        const draft_slots: usize = if (w > 1) w - 1 else 1;
        const slot_offset: usize = if (w > 1) 1 else 0;
        const eff_k: u32 = if (self.config.fork_top2_threshold > 0.0 or self.config.top_k_per_pos >= 2) 2 else 1;
        const inv_temp: f32 = 1.0 / self.config.temperature;

        const prep = try self.prepareAnchorAndContextGpu(gpu, root_token, host_taps_opt, gpu_taps_opt, gpu_ctx_hist_opt);
        var slot_cands: [MAX_BLOCK_SIZE][2]moe.TopKCandidate = undefined;
        var slot_counts: [MAX_BLOCK_SIZE]u8 = @splat(0);

        const iters: u8 = self.config.refinement_iters;
        for (0..iters) |iter| {
            const is_refine = (iter > 0);
            if (is_refine) {
                const eff_mask = try self.resolveEffectiveMaskEmbed();
                for (0..h) |d| ws.gpu_slot_hiddens_f16[d] = ws.gpu_anchor_f16[d];
                for (0..draft_slots) |si| {
                    const dst = ws.gpu_slot_hiddens_f16[(slot_offset + si) * h ..][0..h];
                    if (slot_counts[si] > 0 and slot_cands[si][0].prob >= self.config.refine_confidence_threshold and slot_cands[si][0].token_id < v) {
                        const te = try resolveTokEmbed(self.weights.tok_embed, self.weights.tok_embed_raw, self.weights.tok_embed_hook, slot_cands[si][0].token_id, h, ws.tok_embed_tmp);
                        if (te.ptr != ws.tok_embed_tmp.ptr) @memcpy(ws.tok_embed_tmp[0..h], te[0..h]);
                        if (self.weights.tok_embed_hook == null) {
                            if (self.weights.token_embd_norm) |ten| applyRmsNormInPlace(ws.tok_embed_tmp[0..h], ten, self.config.rms_eps);
                        }
                        for (0..h) |d| dst[d] = @floatCast(ws.tok_embed_tmp[d]);
                    } else {
                        const m_slice = if (eff_mask.len == w * h)
                            eff_mask[(slot_offset + si) * h ..][0..h]
                        else if (w > 1 and eff_mask.len == (w - 1) * h)
                            eff_mask[si * h ..][0..h]
                        else
                            eff_mask[0..h];
                        for (0..h) |d| dst[d] = @floatCast(m_slice[d]);
                    }
                }
            }

            var frame = try gpu.compute_ptr.beginFrame(gpu.timeout_ms);
            errdefer frame.abort();

            var first_b = false;
            if (!is_refine and prep.final_chunk_len > 0) {
                const taps_b = if (gpu_ctx_hist_opt) |gch| gch.taps_ring else ws.sliceBinding(wb, ws.gpu_ctx_taps_f32);
                try self.encodeGpuContextKvBatch(
                    gpu,
                    &frame,
                    taps_b,
                    prep.taps_dtype,
                    prep.ring_cap,
                    prep.ring_start,
                    prep.final_chunk_len,
                    prep.final_chunk_pos,
                    false,
                    false,
                    false,
                );
                first_b = true;
            }

            try self.encodeGpuBackboneIntoFrame(
                gpu,
                &frame,
                prep.anchor_pos,
                prep.active_ctx_len,
                !is_refine and !prep.use_cached_ctx,
                gpu_taps_opt,
                first_b,
                is_refine,
            );

            if (gpu.lm_head) |lmh| {
                const slot_draft_b: metal.BufferBinding = .{
                    .buffer = wb,
                    .offset = ws.sliceBinding(wb, ws.gpu_slot_hiddens_f16).offset + @as(u64, slot_offset) * h * @sizeOf(f16),
                };
                const base_logits_b = ws.sliceBinding(wb, ws.gpu_base_logits_f16);
                try encodeGpuLinear(gpu, &frame, lmh, slot_draft_b, base_logits_b, @intCast(draft_slots), null, false, true);
                const row_bytes: u64 = @as(u64, v) * @sizeOf(f16);
                for (0..draft_slots) |si| {
                    const row_b: metal.BufferBinding = .{
                        .buffer = wb,
                        .offset = base_logits_b.offset + @as(u64, si) * row_bytes,
                    };
                    try encodeGpuTopK(
                        gpu,
                        ws,
                        &frame,
                        row_b,
                        true,
                        eff_v,
                        eff_k,
                        inv_temp,
                        self.config.final_logit_softcapping,
                        false,
                        @intCast(si * eff_k),
                        false,
                        0,
                    );
                }
                const sig = try frame.commit();
                try gpu.compute_ptr.wait(sig, gpu.timeout_ms);
                self.backbone_runs += 1;

                for (0..draft_slots) |si| {
                    var cnt: u8 = 0;
                    for (0..eff_k) |ki| {
                        const c = ws.gpu_topk_cands[si * eff_k + ki];
                        if (c.token_id == std.math.maxInt(u32) or c.token_id >= v) continue;
                        slot_cands[si][cnt] = c;
                        cnt += 1;
                    }
                    slot_counts[si] = cnt;
                }
            } else {
                // Host `LmHeadHook.callback` fallback when no GPU `lm_head` binding was provided.
                const sig = try frame.commit();
                try gpu.compute_ptr.wait(sig, gpu.timeout_ms);
                self.backbone_runs += 1;

                for (0..w * h) |i| ws.slot_hiddens[i] = @floatCast(ws.gpu_slot_hiddens_f16[i]);
                try evalLmHead(
                    self.weights.lm_head,
                    self.weights.lm_head_raw,
                    ws.slot_hiddens[slot_offset * h .. (slot_offset + draft_slots) * h],
                    @intCast(draft_slots),
                    self.config.hidden_size,
                    self.config.vocab_size,
                    ws.base_logits_buf[0 .. draft_slots * v],
                    ws.cpu_quant_row_scratch,
                );
                for (0..draft_slots) |si| {
                    const row = ws.base_logits_buf[si * v ..][0..eff_v];
                    if (self.config.final_logit_softcapping > 0.0) {
                        const cap = self.config.final_logit_softcapping;
                        for (row) |*lv| lv.* = std.math.tanh(lv.* / cap) * cap;
                    }
                    const got = try extractDeterministicTopKMasked(
                        row,
                        inv_temp,
                        null,
                        ws.topk_tokens[0..eff_k],
                        ws.topk_probs[0..eff_k],
                    );
                    slot_counts[si] = @intCast(got);
                    for (0..got) |ki| {
                        const p = ws.topk_probs[ki];
                        slot_cands[si][ki] = .{
                            .token_id = ws.topk_tokens[ki],
                            .prob = p,
                            .log_prob = logProbOf(p),
                        };
                    }
                }
            }
        }

        self.insertDirectBlockIntoTree(
            max_nodes,
            draft_slots,
            slot_cands[0..draft_slots],
            slot_counts[0..draft_slots],
            node_limit,
            gram_active,
            grammar_opt,
            node_states,
            node_valid,
            out_tree,
        );
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

/// Startup-loaded standalone `DFlare` / `arch = dflash` `.amf` bundle owning
/// zero-copy `WiredArena`-backed weights (native `.bf16`/`.f16`/`.f32`/`.gguf_q8_0`/affine/W21/W22)
/// and a 16 KiB-aligned `DFlareWorkspace`.
pub const LoadedDFlareBundle = struct {
    weight_arena: ?arena_mod.WiredArena = null,
    owned_f32_bufs: [][]f32,
    owned_u16_bufs: [][]u16,
    layers: []TransformerBlockWeights,
    workspace: DFlareWorkspace,
    proposer: DFlareProposer,

    pub const LoadError = Error || amf.ParseError || amf.IoError || error{
        MissingTensor,
        OutOfMemory,
    };

    const LayerEntryRefs = struct {
        in_norm: ?*const amf.TensorEntry = null,
        post_norm: ?*const amf.TensorEntry = null,
        ctx_norm: ?*const amf.TensorEntry = null,
        q_norm: ?*const amf.TensorEntry = null,
        k_norm: ?*const amf.TensorEntry = null,
        k_norm_ctx: ?*const amf.TensorEntry = null,
        sink_bias: ?*const amf.TensorEntry = null,
        q_proj: *const amf.TensorEntry = undefined,
        k_proj: *const amf.TensorEntry = undefined,
        v_proj: *const amf.TensorEntry = undefined,
        o_proj: *const amf.TensorEntry = undefined,
        k_proj_ctx: ?*const amf.TensorEntry = null,
        v_proj_ctx: ?*const amf.TensorEntry = null,
        mlp_gate: *const amf.TensorEntry = undefined,
        mlp_up: *const amf.TensorEntry = undefined,
        mlp_down: *const amf.TensorEntry = undefined,
        ctx_mix: ?*const amf.TensorEntry = null,
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
        if (cfg.num_layers > 64) return error.InvalidConfig;

        const h: usize = cfg.hidden_size;
        const inter: usize = cfg.intermediate_size;
        const q_dim: usize = cfg.qDim();
        const kv_dim: usize = cfg.kvDim();
        const mr: usize = cfg.markov_rank;
        const v: usize = cfg.vocab_size;

        // Pass 1: resolve every `.amf` tensor entry, validate shapes/dtypes, and compute exact `WiredArena` size.
        var weight_bytes: usize = 0;
        var max_small_blob: usize = 0;

        var fc_entry_raw: ?*const amf.TensorEntry = null;
        var fuse_scalar_entry: ?*const amf.TensorEntry = null;
        var fuse_scalar_elems: usize = 0;
        var has_ctx_mix: bool = false;
        var markov_ctx_mix_entry: ?*const amf.TensorEntry = null;
        if (findAnyEntry(amf_file, &.{ "dflare.fc.weight", "fc.weight", "fuse_weight" })) |fe| {
            const fc_elems = try entryElems(fe);
            if (fc_elems == cfg.num_target_taps * h * h) {
                fc_entry_raw = fe;
                try planRawBytes(&weight_bytes, fe, h, cfg.num_target_taps * h, true);
            } else if (fc_elems >= cfg.num_target_taps and (fc_elems % cfg.num_target_taps) == 0) {
                fuse_scalar_entry = fe;
                fuse_scalar_elems = try planF32Bytes(&weight_bytes, &max_small_blob, fe, null);
            } else {
                return error.DimensionMismatch;
            }
        } else if (amf_file.parsed.find("layers.0.ctx_mix.weight") != null) {
            has_ctx_mix = true;
            const rows: usize = @as(usize, cfg.num_layers) + 1;
            const mix_elems = std.math.mul(usize, rows, cfg.num_target_taps) catch return error.Overflow;
            weight_bytes = std.math.add(usize, weight_bytes, try alignUp64(mix_elems * @sizeOf(f32))) catch return error.Overflow;
            markov_ctx_mix_entry = amf_file.parsed.find("markov_ctx_mix.weight");
            if (markov_ctx_mix_entry) |me| {
                max_small_blob = @max(max_small_blob, @as(usize, @intCast(me.blob_bytes)));
            }
        }

        const fc_norm_entry = findAnyEntry(amf_file, &.{ "dflare.fc_norm.weight", "fc_norm.weight", "hidden_norm.weight" });
        if (fc_norm_entry) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, h);

        const mask_entry = findAnyEntry(amf_file, &.{ "dflare.mask_embed.weight", "dflare.mask_embed", "mask_embed", "mask_embd.weight" });
        var mask_elems: usize = 0;
        if (mask_entry) |me| {
            mask_elems = try planF32Bytes(&weight_bytes, &max_small_blob, me, null);
        } else if (cfg.mask_token_id != null) {
            weight_bytes = std.math.add(usize, weight_bytes, try alignUp64(h * @sizeOf(f32))) catch return error.Overflow;
        } else {
            return error.MissingTensor;
        }

        const tok_embed_norm_entry = findAnyEntry(amf_file, &.{"token_embd_norm.weight"});
        if (tok_embed_norm_entry) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, h);

        var tok_embed_entry: ?*const amf.TensorEntry = null;
        if (tok_embed_override == null) {
            const te = findAnyEntry(amf_file, &.{ "dflare.tok_embed.weight", "token_embd.weight", "embed_tokens.weight" }) orelse return error.MissingTensor;
            tok_embed_entry = te;
            try planRawBytes(&weight_bytes, te, v, h, true);
        }

        const final_norm_entry = findAnyEntry(amf_file, &.{ "dflare.norm.weight", "output_norm.weight", "norm.weight" });
        if (final_norm_entry) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, h);

        var layer_refs: [64]LayerEntryRefs = undefined;
        var nb0: [80]u8 = undefined;
        var nb1: [80]u8 = undefined;
        var nb2: [80]u8 = undefined;
        for (0..cfg.num_layers) |l| {
            var refs: LayerEntryRefs = .{};
            refs.in_norm = try findLayerEntry(amf_file, l, &.{ "input_layernorm.weight", "attn_norm.weight" }, &nb0, &nb1, &nb2);
            if (refs.in_norm) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, h);
            refs.post_norm = try findLayerEntry(amf_file, l, &.{ "post_attention_layernorm.weight", "post_norm.weight", "ffn_norm.weight" }, &nb0, &nb1, &nb2);
            if (refs.post_norm) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, h);
            refs.ctx_norm = try findLayerEntry(amf_file, l, &.{ "dflare_norm.weight", "ctx_norm.weight" }, &nb0, &nb1, &nb2);
            if (refs.ctx_norm) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, h);
            refs.q_norm = try findLayerEntry(amf_file, l, &.{ "self_attn.q_norm.weight", "attn.q_norm.weight", "attn_q_norm.weight" }, &nb0, &nb1, &nb2);
            if (refs.q_norm) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, cfg.head_dim);
            refs.k_norm = try findLayerEntry(amf_file, l, &.{ "self_attn.k_norm.weight", "attn.k_norm.weight", "attn_k_norm.weight" }, &nb0, &nb1, &nb2);
            if (refs.k_norm) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, cfg.head_dim);
            refs.k_norm_ctx = try findLayerEntry(amf_file, l, &.{ "self_attn.k_norm_ctx.weight", "attn_k_ctx_norm.weight" }, &nb0, &nb1, &nb2);
            if (refs.k_norm_ctx) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, cfg.head_dim);
            refs.sink_bias = try findLayerEntry(amf_file, l, &.{ "self_attn.sink_bias", "self_attn.attention_sink_bias", "attn_sinks.weight" }, &nb0, &nb1, &nb2);
            if (refs.sink_bias) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, cfg.num_attention_heads);

            refs.q_proj = (try findLayerEntry(amf_file, l, &.{ "self_attn.q_proj.weight", "attn.q_proj.weight", "attn_q.weight" }, &nb0, &nb1, &nb2)) orelse return error.MissingTensor;
            try planRawBytes(&weight_bytes, refs.q_proj, q_dim, h, false);
            refs.k_proj = (try findLayerEntry(amf_file, l, &.{ "self_attn.k_proj.weight", "attn.k_proj.weight", "attn_k.weight" }, &nb0, &nb1, &nb2)) orelse return error.MissingTensor;
            try planRawBytes(&weight_bytes, refs.k_proj, kv_dim, h, false);
            refs.v_proj = (try findLayerEntry(amf_file, l, &.{ "self_attn.v_proj.weight", "attn.v_proj.weight", "attn_v.weight" }, &nb0, &nb1, &nb2)) orelse return error.MissingTensor;
            try planRawBytes(&weight_bytes, refs.v_proj, kv_dim, h, false);
            refs.o_proj = (try findLayerEntry(amf_file, l, &.{ "self_attn.o_proj.weight", "attn.o_proj.weight", "attn_output.weight" }, &nb0, &nb1, &nb2)) orelse return error.MissingTensor;
            try planRawBytes(&weight_bytes, refs.o_proj, h, q_dim, false);

            refs.k_proj_ctx = try findLayerEntry(amf_file, l, &.{ "self_attn.k_proj_ctx.weight", "attn.k_proj_ctx.weight" }, &nb0, &nb1, &nb2);
            if (refs.k_proj_ctx) |e| try planRawBytes(&weight_bytes, e, kv_dim, h, false);
            refs.v_proj_ctx = try findLayerEntry(amf_file, l, &.{ "self_attn.v_proj_ctx.weight", "attn.v_proj_ctx.weight" }, &nb0, &nb1, &nb2);
            if (refs.v_proj_ctx) |e| try planRawBytes(&weight_bytes, e, kv_dim, h, false);

            refs.mlp_gate = (try findLayerEntry(amf_file, l, &.{ "mlp.gate_proj.weight", "ffn_gate.weight" }, &nb0, &nb1, &nb2)) orelse return error.MissingTensor;
            try planRawBytes(&weight_bytes, refs.mlp_gate, inter, h, false);
            refs.mlp_up = (try findLayerEntry(amf_file, l, &.{ "mlp.up_proj.weight", "ffn_up.weight" }, &nb0, &nb1, &nb2)) orelse return error.MissingTensor;
            try planRawBytes(&weight_bytes, refs.mlp_up, inter, h, false);
            refs.mlp_down = (try findLayerEntry(amf_file, l, &.{ "mlp.down_proj.weight", "ffn_down.weight" }, &nb0, &nb1, &nb2)) orelse return error.MissingTensor;
            try planRawBytes(&weight_bytes, refs.mlp_down, h, inter, false);

            if (has_ctx_mix) {
                const nm = std.fmt.bufPrint(&nb0, "layers.{d}.ctx_mix.weight", .{l}) catch return error.Overflow;
                refs.ctx_mix = amf_file.parsed.find(nm);
                if (refs.ctx_mix) |ce| max_small_blob = @max(max_small_blob, @as(usize, @intCast(ce.blob_bytes)));
            }
            layer_refs[l] = refs;
        }

        // Markov head entries (omitted in `D15` direct block-diffusion `DFlash` where `markov_rank == 0`).
        const gru_entry = findAnyEntry(amf_file, &.{ "markov_head.rnn_proj.weight", "joint_proj.weight" });
        const any_markov_w1 = findAnyEntry(amf_file, &.{ "dflare.markov_w1.weight", "markov_head.markov_w1.weight", "markov_w1.weight" });
        var eff_mr: usize = mr;
        if (eff_mr == 0 and any_markov_w1 != null) {
            // Auto-detect `dflash-Q8` / `Inkling 2` Markov rank when JSON omitted `markov_rank`.
            const w1e = any_markov_w1.?;
            if (gru_entry != null and w1e.rank >= 2 and w1e.shape[1] > 0) {
                eff_mr = w1e.shape[1];
                cfg.markov_rank = @intCast(eff_mr);
            } else if (w1e.rank >= 2 and w1e.shape[0] > 0) {
                eff_mr = w1e.shape[0];
                cfg.markov_rank = @intCast(eff_mr);
            }
        } else if (eff_mr > 0 and gru_entry == null and any_markov_w1 == null and cfg.variant == .dflash_standalone) {
            eff_mr = 0;
            cfg.markov_rank = 0;
        }
        const is_direct_dflash: bool = (eff_mr == 0);
        const is_real_gru: bool = if (!is_direct_dflash) (if (gru_entry) |ge| (ge.shape[0] == 3 * eff_mr and ge.shape[1] == 2 * h + 2 * eff_mr) else false) else false;

        const markov_ctx_norm_entry = if (!is_direct_dflash) findAnyEntry(amf_file, &.{ "markov_head.dflare_norm.weight", "markov_ctx_norm.weight" }) else null;
        if (markov_ctx_norm_entry) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, h);
        const markov_norm_entry = if (!is_direct_dflash) findAnyEntry(amf_file, &.{ "dflare.markov_norm.weight", "markov_norm.weight" }) else null;
        if (markov_norm_entry) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, h);

        var markov_gru_bias_entry: ?*const amf.TensorEntry = null;
        var markov_w1_vocab_entry: ?*const amf.TensorEntry = null;
        var markov_w1_vocab_rows: u32 = 0;
        var markov_hidden_out_entry: ?*const amf.TensorEntry = null;
        var markov_hidden_bias_entry: ?*const amf.TensorEntry = null;
        var markov_w2_vocab_entry: ?*const amf.TensorEntry = null;
        var markov_w2_vocab_bias_entry: ?*const amf.TensorEntry = null;
        var markov_w2_vocab_rows: u32 = 0;

        var markov_w1_entry: ?*const amf.TensorEntry = null;
        var markov_w1_k: usize = h;
        var markov_rnn_entry: ?*const amf.TensorEntry = null;
        var markov_b1_entry: ?*const amf.TensorEntry = null;
        var markov_w2_entry: ?*const amf.TensorEntry = null;

        if (!is_direct_dflash) {
            if (is_real_gru) {
                try planRawBytes(&weight_bytes, gru_entry.?, 3 * eff_mr, 2 * h + 2 * eff_mr, false);
                markov_gru_bias_entry = findAnyEntry(amf_file, &.{ "markov_head.rnn_proj.bias", "joint_proj.bias" });
                if (markov_gru_bias_entry) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, 3 * eff_mr);

                const w1e = any_markov_w1 orelse return error.MissingTensor;
                if (w1e.rank < 2 or w1e.shape[0] == 0 or w1e.shape[1] != eff_mr) return error.DimensionMismatch;
                markov_w1_vocab_entry = w1e;
                markov_w1_vocab_rows = w1e.shape[0];
                try planRawBytes(&weight_bytes, w1e, w1e.shape[0], eff_mr, false);

                if (findAnyEntry(amf_file, &.{ "markov_head.hidden_out.weight", "dflare.markov_w2.weight" })) |hoe| {
                    markov_hidden_out_entry = hoe;
                    try planRawBytes(&weight_bytes, hoe, h, eff_mr, false);
                    markov_hidden_bias_entry = findAnyEntry(amf_file, &.{"markov_head.hidden_out.bias"});
                    if (markov_hidden_bias_entry) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, h);
                } else if (amf_file.parsed.find("markov_w2.weight")) |w2e| {
                    if (w2e.rank < 2 or w2e.shape[0] == 0 or w2e.shape[1] != eff_mr) return error.DimensionMismatch;
                    markov_w2_vocab_entry = w2e;
                    markov_w2_vocab_rows = w2e.shape[0];
                    try planRawBytes(&weight_bytes, w2e, w2e.shape[0], eff_mr, false);
                    markov_w2_vocab_bias_entry = amf_file.parsed.find("markov_w2.bias");
                    if (markov_w2_vocab_bias_entry) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, w2e.shape[0]);
                } else {
                    return error.MissingTensor;
                }
            } else {
                const w1e = any_markov_w1 orelse return error.MissingTensor;
                const w1_elems = try entryElems(w1e);
                if (w1_elems == eff_mr * 2 * h) {
                    markov_w1_k = 2 * h;
                } else if (w1_elems == eff_mr * h) {
                    markov_w1_k = h;
                } else {
                    return error.DimensionMismatch;
                }
                markov_w1_entry = w1e;
                try planRawBytes(&weight_bytes, w1e, eff_mr, markov_w1_k, true);

                markov_rnn_entry = findAnyEntry(amf_file, &.{ "dflare.rnn_proj.weight", "markov_head.rnn_proj.weight", "markov_rnn_proj.weight" });
                if (markov_rnn_entry) |e| try planRawBytes(&weight_bytes, e, eff_mr, eff_mr, false);

                markov_b1_entry = findAnyEntry(amf_file, &.{ "dflare.markov_b1", "markov_head.rnn_proj.bias", "markov_b1" });
                if (markov_b1_entry) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, eff_mr);

                const w2e = findAnyEntry(amf_file, &.{ "dflare.markov_w2.weight", "markov_head.hidden_out.weight", "markov_w2.weight" }) orelse return error.MissingTensor;
                markov_w2_entry = w2e;
                try planRawBytes(&weight_bytes, w2e, h, eff_mr, false);

                markov_hidden_bias_entry = findAnyEntry(amf_file, &.{"markov_head.hidden_out.bias"});
                if (markov_hidden_bias_entry) |e| _ = try planF32Bytes(&weight_bytes, &max_small_blob, e, h);
            }
        }

        var lm_head_entry: ?*const amf.TensorEntry = null;
        if (lm_head_override == null and markov_w2_vocab_entry == null) {
            const lme = findAnyEntry(amf_file, &.{ "dflare.lm_head.weight", "output.weight", "lm_head.weight" }) orelse return error.MissingTensor;
            lm_head_entry = lme;
            try planRawBytes(&weight_bytes, lme, v, h, true);
        }

        if (is_real_gru) {
            if (cfg.rope_theta <= 0.0) cfg.rope_theta = 1_000_000.0;
            if (cfg.unpadded_vocab_size == 0 and markov_w2_vocab_rows > 0 and markov_w2_vocab_rows < v) {
                cfg.unpadded_vocab_size = markov_w2_vocab_rows;
            }
        }

        // Single 16 KiB-aligned `WiredArena` for all draft weights (`0` per-tensor heap allocations).
        const weight_cap = try page.alignUp(@max(weight_bytes, page.APPLE_PAGE_SIZE));
        const wired_budget = budget_override orelse try defaultWiredBudget();
        try wired_budget.checkPlan(weight_cap);
        var weight_arena = try arena_mod.WiredArena.init(weight_cap, wired_budget);
        errdefer weight_arena.deinit();

        var empty_small_scratch: [0]u8 = .{};
        const small_scratch: []u8 = if (max_small_blob > 0)
            try allocator.alloc(u8, max_small_blob)
        else
            &empty_small_scratch;
        defer if (max_small_blob > 0) allocator.free(small_scratch);

        // Pass 2: pread directly into `weight_arena`.
        var fc_raw: ?RawLinearMat = null;
        var fuse_scalar: ?[]const f32 = null;
        if (fc_entry_raw) |fe| {
            fc_raw = try readRawEntryIntoArena(&weight_arena, amf_file, fe, h, cfg.num_target_taps * h, true);
        } else if (fuse_scalar_entry) |fe| {
            const vec = try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, fe, fuse_scalar_elems);
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
        } else if (has_ctx_mix) {
            const rows: usize = @as(usize, cfg.num_layers) + 1;
            const mix = try allocArenaSlice(f32, &weight_arena, rows * cfg.num_target_taps);
            @memset(mix, 1.0 / @as(f32, @floatFromInt(cfg.num_target_taps)));
            var ctx_tmp: [128]f32 = undefined;
            for (0..cfg.num_layers) |l| {
                if (layer_refs[l].ctx_mix) |ce| {
                    const ce_elems = try entryElems(ce);
                    if (ce_elems > ctx_tmp.len) return error.DimensionMismatch;
                    try readF32EntryIntoSlice(small_scratch, amf_file, ce, ctx_tmp[0..ce_elems]);
                    const ncopy = @min(ce_elems, @as(usize, cfg.num_target_taps));
                    @memcpy(mix[l * cfg.num_target_taps ..][0..ncopy], ctx_tmp[0..ncopy]);
                }
            }
            if (markov_ctx_mix_entry) |me| {
                const me_elems = try entryElems(me);
                if (me_elems > ctx_tmp.len) return error.DimensionMismatch;
                try readF32EntryIntoSlice(small_scratch, amf_file, me, ctx_tmp[0..me_elems]);
                const ncopy = @min(me_elems, @as(usize, cfg.num_target_taps));
                @memcpy(mix[@as(usize, cfg.num_layers) * cfg.num_target_taps ..][0..ncopy], ctx_tmp[0..ncopy]);
            }
            fuse_scalar = mix;
        }

        const fc_norm = if (fc_norm_entry) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, h) else null;
        const tok_embed_norm = if (tok_embed_norm_entry) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, h) else null;
        const tok_embed_raw: ?RawLinearMat = if (tok_embed_entry) |te|
            try readRawEntryIntoArena(&weight_arena, amf_file, te, v, h, true)
        else
            null;
        const mask_embed: []const f32 = if (mask_entry) |me|
            try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, me, mask_elems)
        else blk: {
            const mtid = cfg.mask_token_id orelse return error.MissingTensor;
            const dst = try allocArenaSlice(f32, &weight_arena, h);
            const raw = try resolveTokEmbed(&.{}, tok_embed_raw, tok_embed_override, mtid, h, dst);
            if (raw.ptr != dst.ptr) @memcpy(dst, raw[0..h]);
            if (tok_embed_override == null) {
                if (tok_embed_norm) |ten| applyRmsNormInPlace(dst, ten, cfg.rms_eps);
            }
            break :blk dst;
        };
        const final_norm = if (final_norm_entry) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, h) else null;

        const layers = try allocator.alloc(TransformerBlockWeights, cfg.num_layers);
        errdefer allocator.free(layers);

        for (0..cfg.num_layers) |l| {
            const refs = layer_refs[l];
            layers[l] = .{
                .in_norm_weight = if (refs.in_norm) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, h) else null,
                .q_norm_weight = if (refs.q_norm) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, cfg.head_dim) else null,
                .k_norm_weight = if (refs.k_norm) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, cfg.head_dim) else null,
                .k_norm_ctx_weight = if (refs.k_norm_ctx) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, cfg.head_dim) else null,
                .sink_bias = if (refs.sink_bias) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, cfg.num_attention_heads) else null,
                .ctx_norm_weight = if (refs.ctx_norm) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, h) else null,
                .post_norm_weight = if (refs.post_norm) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, h) else null,
                .q_proj_raw = try readRawEntryIntoArena(&weight_arena, amf_file, refs.q_proj, q_dim, h, false),
                .k_proj_raw = try readRawEntryIntoArena(&weight_arena, amf_file, refs.k_proj, kv_dim, h, false),
                .v_proj_raw = try readRawEntryIntoArena(&weight_arena, amf_file, refs.v_proj, kv_dim, h, false),
                .o_proj_raw = try readRawEntryIntoArena(&weight_arena, amf_file, refs.o_proj, h, q_dim, false),
                .k_proj_ctx_raw = if (refs.k_proj_ctx) |e| try readRawEntryIntoArena(&weight_arena, amf_file, e, kv_dim, h, false) else null,
                .v_proj_ctx_raw = if (refs.v_proj_ctx) |e| try readRawEntryIntoArena(&weight_arena, amf_file, e, kv_dim, h, false) else null,
                .mlp_gate_raw = try readRawEntryIntoArena(&weight_arena, amf_file, refs.mlp_gate, inter, h, false),
                .mlp_up_raw = try readRawEntryIntoArena(&weight_arena, amf_file, refs.mlp_up, inter, h, false),
                .mlp_down_raw = try readRawEntryIntoArena(&weight_arena, amf_file, refs.mlp_down, h, inter, false),
            };
        }

        const markov_ctx_norm = if (markov_ctx_norm_entry) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, h) else null;
        const markov_norm = if (markov_norm_entry) |e| try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, h) else null;

        var markov_w1_raw: ?RawLinearMat = null;
        var markov_w1_vocab_raw: ?RawLinearMat = null;
        var markov_gru_proj_raw: ?RawLinearMat = null;
        var markov_gru_bias: ?[]const f32 = null;
        var markov_rnn_proj_raw: ?RawLinearMat = null;
        var markov_b1: ?[]const f32 = null;
        var markov_w2_raw: ?RawLinearMat = null;
        var markov_hidden_bias: ?[]const f32 = null;
        var markov_w2_vocab_raw: ?RawLinearMat = null;
        var markov_w2_vocab_bias: ?[]const f32 = null;

        if (!is_direct_dflash) {
            if (is_real_gru) {
                markov_gru_proj_raw = try readRawEntryIntoArena(&weight_arena, amf_file, gru_entry.?, 3 * eff_mr, 2 * h + 2 * eff_mr, false);
                if (markov_gru_bias_entry) |e| markov_gru_bias = try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, 3 * eff_mr);
                markov_w1_vocab_raw = try readRawEntryIntoArena(&weight_arena, amf_file, markov_w1_vocab_entry.?, markov_w1_vocab_rows, eff_mr, false);
                if (markov_hidden_out_entry) |hoe| {
                    markov_w2_raw = try readRawEntryIntoArena(&weight_arena, amf_file, hoe, h, eff_mr, false);
                    if (markov_hidden_bias_entry) |e| markov_hidden_bias = try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, h);
                } else if (markov_w2_vocab_entry) |w2e| {
                    markov_w2_vocab_raw = try readRawEntryIntoArena(&weight_arena, amf_file, w2e, markov_w2_vocab_rows, eff_mr, false);
                    if (markov_w2_vocab_bias_entry) |e| markov_w2_vocab_bias = try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, markov_w2_vocab_rows);
                }
            } else {
                markov_w1_raw = try readRawEntryIntoArena(&weight_arena, amf_file, markov_w1_entry.?, eff_mr, markov_w1_k, true);
                if (markov_rnn_entry) |e| markov_rnn_proj_raw = try readRawEntryIntoArena(&weight_arena, amf_file, e, eff_mr, eff_mr, false);
                if (markov_b1_entry) |e| markov_b1 = try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, eff_mr);
                markov_w2_raw = try readRawEntryIntoArena(&weight_arena, amf_file, markov_w2_entry.?, h, eff_mr, false);
                if (markov_hidden_bias_entry) |e| markov_hidden_bias = try readF32EntryIntoArena(&weight_arena, small_scratch, amf_file, e, h);
            }
        }

        const lm_head_raw: ?RawLinearMat = if (lm_head_entry) |lme|
            try readRawEntryIntoArena(&weight_arena, amf_file, lme, v, h, true)
        else
            null;
        const lm_head: LmHeadHook = lm_head_override orelse .{ .dense_f32 = &.{} };

        const owned_f32 = try allocator.alloc([]f32, 0);
        errdefer allocator.free(owned_f32);
        const owned_u16 = try allocator.alloc([]u16, 0);
        errdefer allocator.free(owned_u16);

        var ws = try DFlareWorkspace.init(cfg, budget_override);
        errdefer ws.deinit();

        const weights: DFlareWeights = .{
            .target_fusion_proj_raw = fc_raw,
            .fuse_scalar_weights = fuse_scalar,
            .target_fusion_norm = fc_norm,
            .mask_embed = mask_embed,
            .tok_embed_raw = tok_embed_raw,
            .tok_embed_hook = tok_embed_override,
            .token_embd_norm = tok_embed_norm,
            .layers = layers,
            .final_norm = final_norm,
            .markov_w1_raw = markov_w1_raw,
            .markov_w1_vocab_raw = markov_w1_vocab_raw,
            .markov_w1_vocab_rows = markov_w1_vocab_rows,
            .markov_gru_proj_raw = markov_gru_proj_raw,
            .markov_gru_bias = markov_gru_bias,
            .markov_ctx_norm = markov_ctx_norm,
            .markov_rnn_proj_raw = markov_rnn_proj_raw,
            .markov_b1 = markov_b1,
            .markov_w2_raw = markov_w2_raw,
            .markov_hidden_bias = markov_hidden_bias,
            .markov_w2_vocab_raw = markov_w2_vocab_raw,
            .markov_w2_vocab_bias = markov_w2_vocab_bias,
            .markov_w2_vocab_rows = markov_w2_vocab_rows,
            .markov_norm = markov_norm,
            .lm_head = lm_head,
            .lm_head_raw = lm_head_raw,
            .wired_arena_bytes = weight_arena.bytes(),
        };

        const bundle = try allocator.create(LoadedDFlareBundle);
        errdefer allocator.destroy(bundle);

        bundle.* = .{
            .weight_arena = weight_arena,
            .owned_f32_bufs = owned_f32,
            .owned_u16_bufs = owned_u16,
            .layers = layers,
            .workspace = ws,
            .proposer = undefined,
        };
        // Ownership of `weight_arena`, `layers`, `ws`, `owned_f32`, `owned_u16` moved into `bundle`.
        bundle.proposer = DFlareProposer.init(cfg, weights, &bundle.workspace) catch |err| {
            bundle.deinit(allocator);
            return err;
        };
        return bundle;
    }

    pub fn totalWiredBytes(self: *const LoadedDFlareBundle) u64 {
        const wa_bytes: u64 = if (self.weight_arena) |wa| @intCast(wa.bytes().len) else 0;
        return self.workspace.totalWiredBytes() + wa_bytes;
    }

    pub fn deinit(self: *LoadedDFlareBundle, allocator: std.mem.Allocator) void {
        self.proposer.detachGpu();
        self.workspace.deinit();
        if (self.weight_arena) |*wa| wa.deinit();
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
        const root_e = try resolveTokEmbed(self.weights.tok_embed, null, self.weights.tok_embed_hook, root_token, h, self.workspace.tok_embed_tmp);
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
                null,
                .{},
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
                null,
                self.workspace.slot_hiddens[0 .. act_usize * h],
                active_steps,
                self.config.hidden_size,
                self.config.vocab_size,
                logits,
                null,
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

const BlockAttnExtras = struct {
    rope_dim: u32 = 0,
    sliding_window: u32 = 0,
    max_ctx: usize = 0,
    v_scale: f32 = 1.0,
};

fn applyRopeNeoXInPlace(buf: []f32, num_heads: u32, head_dim: u32, rope_dim: u32, pos: u32, theta: f32) void {
    const hd: usize = head_dim;
    const eff_rope: usize = if (rope_dim > 0 and rope_dim <= head_dim) rope_dim else head_dim;
    const half: usize = eff_rope / 2;
    if (half == 0) return;
    const pos_f: f32 = @floatFromInt(pos);
    const inv_rd: f32 = 2.0 / @as(f32, @floatFromInt(eff_rope));
    for (0..num_heads) |hi| {
        const head = buf[hi * hd ..][0..hd];
        for (0..half) |i| {
            const freq: f32 = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(i)) * inv_rd);
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
    quant_row_scratch: ?[]f32,
    extras: BlockAttnExtras,
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
    const total_kv_cap = ctx_len + seq_len;
    if (attn_scores.len < total_kv_cap) return error.BufferTooSmall;
    var empty_row_scratch: [0]f32 = .{};
    const scratch: []f32 = quant_row_scratch orelse &empty_row_scratch;
    const eff_rope_dim: u32 = if (extras.rope_dim > 0) extras.rope_dim else head_dim;
    const v_scale: f32 = extras.v_scale;
    const max_ctx_ring: usize = if (extras.max_ctx > 0) extras.max_ctx else ctx_len;

    for (0..seq_len) |pos| {
        const src = hiddens[pos * hidden_size ..][0..hidden_size];
        const dst = norm_buf[pos * hidden_size ..][0..hidden_size];
        @memcpy(dst, src);
        applyRmsNormInPlace(dst, layer.in_norm_weight, rms_eps);

        matVecLayer(layer.q_proj, layer.q_proj_bf16, layer.q_proj_raw, dst, q_dim, hidden_size, q_buf[pos * q_dim ..][0..q_dim], false, scratch);
        matVecLayer(layer.k_proj, layer.k_proj_bf16, layer.k_proj_raw, dst, kv_dim, hidden_size, k_buf[pos * kv_dim ..][0..kv_dim], false, scratch);
        matVecLayer(layer.v_proj, layer.v_proj_bf16, layer.v_proj_raw, dst, kv_dim, hidden_size, v_buf[pos * kv_dim ..][0..kv_dim], false, scratch);

        if (layer.q_norm_weight) |qn| {
            applyHeadRmsNorm(q_buf[pos * q_dim ..][0..q_dim], num_heads, head_dim, qn, rms_eps);
        }
        if (layer.k_norm_weight) |kn| {
            applyHeadRmsNorm(k_buf[pos * kv_dim ..][0..kv_dim], num_kv_heads, head_dim, kn, rms_eps);
        }
        if (rope_theta > 0.0) {
            const tok_pos: u32 = anchor_pos + @as(u32, @intCast(pos));
            applyRopeNeoXInPlace(q_buf[pos * q_dim ..][0..q_dim], num_heads, head_dim, eff_rope_dim, tok_pos, rope_theta);
            applyRopeNeoXInPlace(k_buf[pos * kv_dim ..][0..kv_dim], num_kv_heads, head_dim, eff_rope_dim, tok_pos, rope_theta);
        }
    }

    // Bidirectional block-parallel attention over `[ctx_kv (causal SWA over circular ring), block_kv (0..seq_len)]`,
    // plus optional `sink_bias` softmax column and `attention_value_scale`.
    const ctx_base: usize = if (anchor_pos >= ctx_len) (@as(usize, anchor_pos) - ctx_len) else 0;
    for (0..seq_len) |q_pos| {
        const q_row = q_buf[q_pos * q_dim ..][0..q_dim];
        const out_row = attn_out[q_pos * q_dim ..][0..q_dim];
        var eff_ctx_start: usize = ctx_base;
        var eff_ctx_len: usize = ctx_len;
        if (extras.sliding_window > 0 and anchor_pos >= ctx_len) {
            const q_log_pos: usize = @as(usize, anchor_pos) + q_pos;
            const swa: usize = extras.sliding_window;
            const min_k_pos: usize = if (q_log_pos + 1 > swa) (q_log_pos + 1 - swa) else 0;
            eff_ctx_start = @max(ctx_base, min_k_pos);
            eff_ctx_len = if (anchor_pos > eff_ctx_start) (@as(usize, anchor_pos) - eff_ctx_start) else 0;
        }
        const total_kv = eff_ctx_len + seq_len;

        for (0..num_heads) |qh| {
            const kvh = qh / gqa_group;
            const q_head = q_row[qh * hd ..][0..hd];
            var max_s: f32 = -std.math.inf(f32);

            if (ctx_k_f16) |ck16| {
                for (0..eff_ctx_len) |kp| {
                    const slot = if (max_ctx_ring > 0) ((eff_ctx_start + kp) % max_ctx_ring) else (eff_ctx_start + kp);
                    const k_head = ck16[slot * kv_dim + kvh * hd ..][0..hd];
                    var dot: f32 = 0.0;
                    for (0..hd) |d| dot += q_head[d] * @as(f32, @floatCast(k_head[d]));
                    const sc = dot * inv_sqrt_hd;
                    attn_scores[kp] = sc;
                    if (sc > max_s) max_s = sc;
                }
            } else {
                for (0..eff_ctx_len) |kp| {
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
                attn_scores[eff_ctx_len + kp] = sc;
                if (sc > max_s) max_s = sc;
            }
            const sink_opt: ?f32 = if (layer.sink_bias) |sb| sb[qh] else null;
            if (sink_opt) |sv| {
                if (sv > max_s) max_s = sv;
            }

            var sum_exp: f32 = 0.0;
            for (0..total_kv) |ti| {
                const e = @exp(attn_scores[ti] - max_s);
                attn_scores[ti] = e;
                sum_exp += e;
            }
            if (sink_opt) |sv| {
                sum_exp += @exp(sv - max_s);
            }
            const inv_sum: f32 = if (sum_exp > 0.0) (1.0 / sum_exp) * v_scale else 0.0;

            const dst_head = out_row[qh * hd ..][0..hd];
            @memset(dst_head, 0.0);
            if (ctx_v_f16) |cv16| {
                for (0..eff_ctx_len) |kp| {
                    const slot = if (max_ctx_ring > 0) ((eff_ctx_start + kp) % max_ctx_ring) else (eff_ctx_start + kp);
                    const p = attn_scores[kp] * inv_sum;
                    const v_head = cv16[slot * kv_dim + kvh * hd ..][0..hd];
                    for (0..hd) |d| dst_head[d] += p * @as(f32, @floatCast(v_head[d]));
                }
            } else {
                for (0..eff_ctx_len) |kp| {
                    const p = attn_scores[kp] * inv_sum;
                    const v_head = ctx_v[kp * kv_dim + kvh * hd ..][0..hd];
                    for (0..hd) |d| dst_head[d] += p * v_head[d];
                }
            }
            for (0..seq_len) |kp| {
                const p = attn_scores[eff_ctx_len + kp] * inv_sum;
                const v_head = v_buf[kp * kv_dim + kvh * hd ..][0..hd];
                for (0..hd) |d| dst_head[d] += p * v_head[d];
            }
        }

        // Residual connection after attention projection.
        const h_row = hiddens[q_pos * hidden_size ..][0..hidden_size];
        matVecLayer(layer.o_proj, layer.o_proj_bf16, layer.o_proj_raw, out_row, hidden_size, q_dim, h_row, true, scratch);
    }

    // SwiGLU MLP + residual add across all block slots.
    for (0..seq_len) |pos| {
        const h_row = hiddens[pos * hidden_size ..][0..hidden_size];
        const norm_row = norm_buf[pos * hidden_size ..][0..hidden_size];
        @memcpy(norm_row, h_row);
        applyRmsNormInPlace(norm_row, layer.post_norm_weight, rms_eps);

        const g_row = mlp_gate[pos * intermediate_size ..][0..intermediate_size];
        const u_row = mlp_up[pos * intermediate_size ..][0..intermediate_size];
        matVecLayer(layer.mlp_gate_proj, layer.mlp_gate_bf16, layer.mlp_gate_raw, norm_row, intermediate_size, hidden_size, g_row, false, scratch);
        matVecLayer(layer.mlp_up_proj, layer.mlp_up_bf16, layer.mlp_up_raw, norm_row, intermediate_size, hidden_size, u_row, false, scratch);
        for (0..intermediate_size) |i| {
            g_row[i] = siluF32(g_row[i]) * u_row[i];
        }
        const d_row = mlp_down[pos * hidden_size ..][0..hidden_size];
        matVecLayer(layer.mlp_down_proj, layer.mlp_down_bf16, layer.mlp_down_raw, g_row, hidden_size, intermediate_size, d_row, false, scratch);
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
    lm_head_raw: ?RawLinearMat,
    hiddens: []const f32,
    batch_rows: u8,
    hidden_size: u32,
    vocab_size: u32,
    out_logits: []f32,
    row_scratch: ?[]f32,
) Error!void {
    const h: usize = hidden_size;
    const v: usize = vocab_size;
    if (!hasLmHeadHook(hook)) {
        if (lm_head_raw) |lhr| {
            var empty_row_scratch: [0]f32 = .{};
            const scratch: []f32 = row_scratch orelse &empty_row_scratch;
            for (0..batch_rows) |s| {
                matVecRaw(lhr, hiddens[s * h ..][0..h], v, h, out_logits[s * v ..][0..v], false, scratch);
            }
            return;
        }
    }
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
    tok_embed_raw: ?RawLinearMat,
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
    if (tok_embed_raw) |tr| {
        if (token_id >= tr.n) return error.TokenOutOfRange;
        try readRawRowF32(tr, token_id, scratch[0..h]);
        return scratch[0..h];
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

fn collectMissingQfSpec(
    borrowed: ?*const quant_formats.QuantFormatPipelines,
    out_specs: *[quant_formats.ALL_FORMAT_SPECS.len]quant_formats.FormatSpec,
    out_len: *usize,
    raw_opt: ?RawLinearMat,
) Error!void {
    const r = raw_opt orelse return;
    return collectMissingQfSpecDType(borrowed, out_specs, out_len, r.amf_dtype, r.group_size);
}

fn collectMissingQfSpecDType(
    borrowed: ?*const quant_formats.QuantFormatPipelines,
    out_specs: *[quant_formats.ALL_FORMAT_SPECS.len]quant_formats.FormatSpec,
    out_len: *usize,
    amf_dtype: amf.DType,
    group_size: u16,
) Error!void {
    if (!dequant.usesQuantFormats(amf_dtype, group_size)) return;
    const qf_dt = quant_formats.DType.fromAmf(amf_dtype) orelse return error.InvalidWeights;
    const eff_gs = qf_dt.effectiveGroupSize(group_size);
    if (borrowed) |bp| {
        if (bp.pipelineFor(qf_dt, eff_gs, .gemv) != null) return;
    }
    for (out_specs[0..out_len.*]) |existing| {
        if (existing.dtype == qf_dt and existing.group_size == eff_gs) return;
    }
    if (out_len.* >= out_specs.len) return error.Overflow;
    out_specs[out_len.*] = .{ .dtype = qf_dt, .group_size = eff_gs };
    out_len.* += 1;
}

fn isSubSliceOfWired(wired_opt: ?[]align(page.APPLE_PAGE_SIZE) u8, bytes: []const u8) bool {
    const w = wired_opt orelse return false;
    if (bytes.len == 0) return false;
    const w_start = @intFromPtr(w.ptr);
    const w_end = w_start + w.len;
    const b_start = @intFromPtr(bytes.ptr);
    const b_end = b_start + bytes.len;
    return b_start >= w_start and b_end <= w_end;
}

fn stageMeasureSlice(wired_opt: ?[]align(page.APPLE_PAGE_SIZE) u8, bytes: []const u8) usize {
    if (bytes.len == 0 or isSubSliceOfWired(wired_opt, bytes)) return 0;
    return alignUp64(bytes.len) catch std.math.maxInt(usize) / 2;
}

fn stageMeasureRaw(wired_opt: ?[]align(page.APPLE_PAGE_SIZE) u8, r: RawLinearMat) usize {
    const d_sz = stageMeasureSlice(wired_opt, r.data);
    const p_sz = if (r.params.len > 0) stageMeasureSlice(wired_opt, r.params) else 0;
    return d_sz + p_sz;
}

fn stageMeasureLayerProj(
    wired_opt: ?[]align(page.APPLE_PAGE_SIZE) u8,
    f32_s: []const f32,
    bf16_opt: ?[]const u16,
    raw_opt: ?RawLinearMat,
) usize {
    if (raw_opt) |r| return stageMeasureRaw(wired_opt, r);
    if (bf16_opt) |b| return stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(b));
    if (f32_s.len > 0) return stageMeasureSlice(wired_opt, std.mem.sliceAsBytes(f32_s));
    return 0;
}

fn appendUniqueResident(arr: *[32]metal.Buffer, count: *usize, buf: metal.Buffer) void {
    for (arr[0..count.*]) |existing| {
        if (existing.handle == buf.handle) return;
    }
    if (count.* < arr.len) {
        arr[count.*] = buf;
        count.* += 1;
    }
}

const BindHelper = struct {
    wired: ?[]align(page.APPLE_PAGE_SIZE) u8,
    weight_buf: ?metal.Buffer,
    st_arena: ?*arena_mod.WiredArena,
    st_buf: ?metal.Buffer,

    fn bindSlice(self: BindHelper, bytes: []const u8) Error!metal.BufferBinding {
        if (bytes.len == 0) return error.InvalidWeights;
        if (isSubSliceOfWired(self.wired, bytes)) {
            const wb = self.weight_buf orelse return error.InvalidWeights;
            const off = @intFromPtr(bytes.ptr) - @intFromPtr(self.wired.?.ptr);
            return .{ .buffer = wb, .offset = off };
        }
        const sa = self.st_arena orelse return error.InvalidWeights;
        const sb = self.st_buf orelse return error.InvalidWeights;
        const dst = try sa.allocAligned(bytes.len, CACHELINE_ALIGN);
        @memcpy(dst, bytes);
        return .{ .buffer = sb, .offset = sa.offsetOf(dst.ptr) };
    }

    fn bindRaw(self: BindHelper, r: RawLinearMat) Error!GpuLinearBinding {
        const wb = try self.bindSlice(r.data);
        const pb: ?metal.BufferBinding = if (r.params.len > 0) try self.bindSlice(r.params) else null;
        return .{
            .amf_dtype = r.amf_dtype,
            .group_size = r.group_size,
            .tensor_scale_per_row = r.tensor_scale_per_row,
            .weight = wb,
            .params = pb,
            .n = r.n,
            .k = r.k,
        };
    }

    fn bindF32Mat(self: BindHelper, slice: []const f32, n: usize, k: usize) Error!GpuLinearBinding {
        if (slice.len != n * k) return error.InvalidWeights;
        return .{
            .amf_dtype = .f32,
            .group_size = 64,
            .weight = try self.bindSlice(std.mem.sliceAsBytes(slice)),
            .n = @intCast(n),
            .k = @intCast(k),
        };
    }

    fn bindBf16Mat(self: BindHelper, slice: []const u16, n: usize, k: usize) Error!GpuLinearBinding {
        if (slice.len != n * k) return error.InvalidWeights;
        return .{
            .amf_dtype = .bf16,
            .group_size = 64,
            .weight = try self.bindSlice(std.mem.sliceAsBytes(slice)),
            .n = @intCast(n),
            .k = @intCast(k),
        };
    }

    fn bindLayerProj(
        self: BindHelper,
        f32_s: []const f32,
        bf16_opt: ?[]const u16,
        raw_opt: ?RawLinearMat,
        n: usize,
        k: usize,
    ) Error!GpuLinearBinding {
        if (raw_opt) |r| {
            if (!r.matches(n, k)) return error.InvalidWeights;
            return self.bindRaw(r);
        }
        if (bf16_opt) |b| return self.bindBf16Mat(b, n, k);
        if (f32_s.len == 0) return error.InvalidWeights;
        return self.bindF32Mat(f32_s, n, k);
    }

    fn bindLayerProjOpt(
        self: BindHelper,
        f32_opt: ?[]const f32,
        bf16_opt: ?[]const u16,
        raw_opt: ?RawLinearMat,
        n: usize,
        k: usize,
    ) Error!?GpuLinearBinding {
        const f32_len: usize = if (f32_opt) |s| s.len else 0;
        if (raw_opt == null and bf16_opt == null and f32_len == 0) return null;
        return try self.bindLayerProj(f32_opt orelse &.{}, bf16_opt, raw_opt, n, k);
    }
};

fn readRawRowF32(raw: RawLinearMat, row: usize, out_f32: []f32) Error!void {
    const cols: usize = raw.k;
    if (row >= raw.n or out_f32.len < cols) return error.DimensionMismatch;
    switch (raw.amf_dtype) {
        .f32 => {
            const byte_off = row * cols * 4;
            if (byte_off + cols * 4 > raw.data.len) return error.InvalidWeights;
            const row_bytes = raw.data[byte_off..][0 .. cols * 4];
            for (0..cols) |c| {
                out_f32[c] = @bitCast(std.mem.readInt(u32, row_bytes[c * 4 ..][0..4], .little));
            }
        },
        .f16 => {
            const byte_off = row * cols * 2;
            if (byte_off + cols * 2 > raw.data.len) return error.InvalidWeights;
            const row_bytes = raw.data[byte_off..][0 .. cols * 2];
            for (0..cols) |c| {
                out_f32[c] = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, row_bytes[c * 2 ..][0..2], .little))));
            }
        },
        .bf16 => {
            const byte_off = row * cols * 2;
            if (byte_off + cols * 2 > raw.data.len) return error.InvalidWeights;
            const row_bytes = raw.data[byte_off..][0 .. cols * 2];
            for (0..cols) |c| {
                out_f32[c] = quant.bf16ToF32(std.mem.readInt(u16, row_bytes[c * 2 ..][0..2], .little));
            }
        },
        else => {
            const qf_dt = quant_formats.DType.fromAmf(raw.amf_dtype) orelse return error.InvalidWeights;
            quant_formats.dequantRowCpuRef(qf_dt, raw.group_size, raw.data, raw.params, row, raw.n, cols, out_f32[0..cols]) catch return error.InvalidWeights;
        },
    }
}

fn matVecRaw(
    raw: RawLinearMat,
    vec: []const f32,
    rows: usize,
    cols: usize,
    out: []f32,
    accum: bool,
    row_scratch: []f32,
) void {
    switch (raw.amf_dtype) {
        .f32 => {
            if (@intFromPtr(raw.data.ptr) % @alignOf(f32) == 0) {
                const mat: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, raw.data[0 .. rows * cols * 4]));
                if (accum) {
                    matVecAccumF32(mat, vec, rows, cols, out);
                } else {
                    matVecF32(mat, vec, rows, cols, out);
                }
                return;
            }
            for (0..rows) |r| {
                const row_bytes = raw.data[r * cols * 4 ..][0 .. cols * 4];
                var acc: f32 = 0.0;
                for (0..cols) |c| {
                    const wf: f32 = @bitCast(std.mem.readInt(u32, row_bytes[c * 4 ..][0..4], .little));
                    acc += wf * vec[c];
                }
                if (accum) out[r] += acc else out[r] = acc;
            }
        },
        .bf16 => {
            if (@intFromPtr(raw.data.ptr) % @alignOf(u16) == 0) {
                const mat: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw.data[0 .. rows * cols * 2]));
                if (accum) {
                    matVecBf16Accum(mat, vec, rows, cols, out);
                } else {
                    matVecBf16(mat, vec, rows, cols, out);
                }
                return;
            }
            for (0..rows) |r| {
                const row_bytes = raw.data[r * cols * 2 ..][0 .. cols * 2];
                var acc: f32 = 0.0;
                for (0..cols) |c| {
                    acc += quant.bf16ToF32(std.mem.readInt(u16, row_bytes[c * 2 ..][0..2], .little)) * vec[c];
                }
                if (accum) out[r] += acc else out[r] = acc;
            }
        },
        .f16 => {
            for (0..rows) |r| {
                const row_bytes = raw.data[r * cols * 2 ..][0 .. cols * 2];
                var acc: f32 = 0.0;
                for (0..cols) |c| {
                    const wf: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, row_bytes[c * 2 ..][0..2], .little))));
                    acc += wf * vec[c];
                }
                if (accum) out[r] += acc else out[r] = acc;
            }
        },
        else => {
            const qf_dt = quant_formats.DType.fromAmf(raw.amf_dtype) orelse std.debug.panic("matVecRaw unsupported amf_dtype={s}", .{@tagName(raw.amf_dtype)});
            if (row_scratch.len < cols) std.debug.panic("matVecRaw scratch too small: {} < {}", .{ row_scratch.len, cols });
            const scratch = row_scratch[0..cols];
            for (0..rows) |r| {
                quant_formats.dequantRowCpuRef(qf_dt, raw.group_size, raw.data, raw.params, r, rows, cols, scratch) catch |err| {
                    std.debug.panic("matVecRaw dequant failed: {}", .{err});
                };
                var acc: f32 = 0.0;
                for (0..cols) |c| acc += scratch[c] * vec[c];
                if (accum) out[r] += acc else out[r] = acc;
            }
        },
    }
}

fn matVecLayer(
    f32_w: []const f32,
    bf16_w: ?[]const u16,
    raw_w: ?RawLinearMat,
    vec: []const f32,
    rows: usize,
    cols: usize,
    out: []f32,
    accum: bool,
    row_scratch: []f32,
) void {
    if (raw_w) |rw| {
        matVecRaw(rw, vec, rows, cols, out, accum, row_scratch);
        return;
    }
    if (bf16_w) |bw| {
        if (accum) {
            matVecBf16Accum(bw, vec, rows, cols, out);
        } else {
            matVecBf16(bw, vec, rows, cols, out);
        }
        return;
    }
    if (accum) {
        matVecAccumF32(f32_w, vec, rows, cols, out);
    } else {
        matVecF32(f32_w, vec, rows, cols, out);
    }
}

fn entryElems(entry: *const amf.TensorEntry) LoadedDFlareBundle.LoadError!usize {
    var elems: usize = 1;
    for (0..entry.rank) |d| {
        elems = std.math.mul(usize, elems, entry.shape[d]) catch return error.Overflow;
    }
    return elems;
}

fn validateRawShapeAndType(
    entry: *const amf.TensorEntry,
    expected_n: usize,
    expected_k: usize,
    allow_rank1: bool,
) LoadedDFlareBundle.LoadError!usize {
    const dt = try entry.dataType();
    if (entry.rank == 2) {
        if (entry.shape[0] != expected_n or entry.shape[1] != expected_k) return error.DimensionMismatch;
    } else if (entry.rank == 1 and allow_rank1) {
        const elems = try entryElems(entry);
        if (elems != expected_n * expected_k) return error.DimensionMismatch;
    } else {
        return error.DimensionMismatch;
    }
    const gs: u16 = if (entry.group_size > 0) entry.group_size else 64;
    switch (dt) {
        .f32 => {
            const need = expected_n * expected_k * 4;
            if (entry.data_bytes < need or entry.blob_bytes < need) return error.InvalidWeights;
        },
        .f16, .bf16 => {
            const need = expected_n * expected_k * 2;
            if (entry.data_bytes < need or entry.blob_bytes < need) return error.InvalidWeights;
        },
        else => {
            const qf_dt = quant_formats.DType.fromAmf(dt) orelse return error.InvalidWeights;
            qf_dt.validateShape(@intCast(expected_n), @intCast(expected_k), gs) catch return error.DimensionMismatch;
            const wb = qf_dt.packedWeightBytes(@intCast(expected_n), @intCast(expected_k), gs) catch return error.InvalidWeights;
            const pb = qf_dt.paramsBytes(@intCast(expected_n), @intCast(expected_k), gs) catch return error.InvalidWeights;
            if (entry.data_bytes < wb or entry.blob_bytes < entry.data_bytes) return error.InvalidWeights;
            if (pb > 0 and (entry.params_bytes < pb or entry.params_offset + entry.params_bytes > entry.blob_bytes)) return error.InvalidWeights;
        },
    }
    const payload_end: u64 = if (entry.params_bytes > 0)
        @max(entry.data_bytes, entry.params_offset + entry.params_bytes)
    else
        entry.data_bytes;
    if (payload_end > entry.blob_bytes) return error.InvalidWeights;
    return @intCast(payload_end);
}

fn planRawBytes(
    total_bytes: *usize,
    entry: *const amf.TensorEntry,
    expected_n: usize,
    expected_k: usize,
    allow_rank1: bool,
) LoadedDFlareBundle.LoadError!void {
    const payload_len = try validateRawShapeAndType(entry, expected_n, expected_k, allow_rank1);
    const aligned = try alignUp64(payload_len);
    total_bytes.* = std.math.add(usize, total_bytes.*, aligned) catch return error.Overflow;
}

fn planF32Bytes(
    total_bytes: *usize,
    max_small_blob: *usize,
    entry: *const amf.TensorEntry,
    expected_elems: ?usize,
) LoadedDFlareBundle.LoadError!usize {
    const dt = try entry.dataType();
    const elems = try entryElems(entry);
    if (expected_elems) |ee| {
        if (elems != ee) return error.DimensionMismatch;
    }
    switch (dt) {
        .f32 => if (entry.data_bytes < elems * 4) return error.InvalidWeights,
        .f16, .bf16 => if (entry.data_bytes < elems * 2) return error.InvalidWeights,
        else => {
            if (quant_formats.DType.fromAmf(dt) == null) return error.InvalidWeights;
        },
    }
    const blob_len: usize = @intCast(entry.blob_bytes);
    max_small_blob.* = @max(max_small_blob.*, blob_len);
    const aligned = try alignUp64(std.math.mul(usize, elems, @sizeOf(f32)) catch return error.Overflow);
    total_bytes.* = std.math.add(usize, total_bytes.*, aligned) catch return error.Overflow;
    return elems;
}

fn preadExact(fd: @FieldType(amf.AmfFile, "fd"), dst: []u8, file_off: u64) LoadedDFlareBundle.LoadError!void {
    var done: usize = 0;
    while (done < dst.len) {
        const rc = std.c.pread(fd, dst.ptr + done, dst.len - done, @intCast(file_off + done));
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.ReadFailed;
        }
        if (rc == 0) return error.ShortRead;
        done += @intCast(rc);
    }
}

fn verifyTrailingCrc(
    fd: @FieldType(amf.AmfFile, "fd"),
    hasher: *std.hash.Crc32,
    entry: *const amf.TensorEntry,
    payload_len: usize,
) LoadedDFlareBundle.LoadError!void {
    const blob_len: usize = @intCast(entry.blob_bytes);
    var rem: usize = blob_len - payload_len;
    var off: u64 = entry.blob_offset + payload_len;
    var tail_buf: [4096]u8 = undefined;
    while (rem > 0) {
        const chunk = @min(rem, tail_buf.len);
        try preadExact(fd, tail_buf[0..chunk], off);
        hasher.update(tail_buf[0..chunk]);
        off += chunk;
        rem -= chunk;
    }
    if (hasher.final() != entry.blob_crc32) return error.BlobChecksumMismatch;
}

fn readRawEntryIntoArena(
    arena: *arena_mod.WiredArena,
    amf_file: *const amf.AmfFile,
    entry: *const amf.TensorEntry,
    expected_n: usize,
    expected_k: usize,
    allow_rank1: bool,
) LoadedDFlareBundle.LoadError!RawLinearMat {
    const dt = try entry.dataType();
    const payload_len = try validateRawShapeAndType(entry, expected_n, expected_k, allow_rank1);
    const dst = try allocArenaSlice(u8, arena, payload_len);
    try preadExact(amf_file.fd, dst, entry.blob_offset);
    var hasher = std.hash.Crc32.init();
    hasher.update(dst);
    try verifyTrailingCrc(amf_file.fd, &hasher, entry, payload_len);

    const data_len: usize = @intCast(entry.data_bytes);
    const params: []const u8 = if (entry.params_bytes > 0 and entry.params_offset + entry.params_bytes <= dst.len)
        dst[@intCast(entry.params_offset)..][0..@intCast(entry.params_bytes)]
    else
        &.{};
    const gs: u16 = if (entry.group_size > 0) entry.group_size else 64;
    return .{
        .data = dst[0..data_len],
        .params = params,
        .amf_dtype = dt,
        .group_size = gs,
        .tensor_scale_per_row = dequant.isTensorScalePerRow(dt, @intCast(expected_n), @intCast(params.len)),
        .n = @intCast(expected_n),
        .k = @intCast(expected_k),
    };
}

fn decodeBlobToF32(
    dt: amf.DType,
    entry: *const amf.TensorEntry,
    raw: []const u8,
    out: []f32,
) LoadedDFlareBundle.LoadError!void {
    const elems = out.len;
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
                out[i] = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little))));
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
}

fn readF32EntryIntoSlice(
    scratch_buf: []u8,
    amf_file: *const amf.AmfFile,
    entry: *const amf.TensorEntry,
    out: []f32,
) LoadedDFlareBundle.LoadError!void {
    const dt = try entry.dataType();
    const blob_len: usize = @intCast(entry.blob_bytes);
    if (scratch_buf.len < blob_len) return error.BufferTooSmall;
    const raw = scratch_buf[0..blob_len];
    try preadExact(amf_file.fd, raw, entry.blob_offset);
    if (std.hash.Crc32.hash(raw) != entry.blob_crc32) return error.BlobChecksumMismatch;
    try decodeBlobToF32(dt, entry, raw, out);
}

fn readF32EntryIntoArena(
    arena: *arena_mod.WiredArena,
    scratch_buf: []u8,
    amf_file: *const amf.AmfFile,
    entry: *const amf.TensorEntry,
    expected_elems: usize,
) LoadedDFlareBundle.LoadError![]f32 {
    const elems = try entryElems(entry);
    if (elems != expected_elems) return error.DimensionMismatch;
    const out = try allocArenaSlice(f32, arena, elems);
    try readF32EntryIntoSlice(scratch_buf, amf_file, entry, out);
    return out;
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

    const total_size = try page.alignUp(next_off);
    const image = try allocator.alloc(u8, total_size);
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

const TypedAmfItem = struct {
    name: [64]u8 = @splat(0),
    name_len: usize = 0,
    blob: []u8,
    data_bytes: u64,
    dtype: amf.DType,
    quant_bits: u8 = 0,
    group_size: u16 = 0,
    shape: [4]u32,
    rank: u8,
    kind: amf.TensorKind,
};

fn fillDetWeight(dst: []f32, seed: u64, scale: f32) void {
    var st = seed;
    for (dst) |*v| {
        st = st *% 6364136223846793005 +% 1442695040888963407;
        const bits: i32 = @as(i32, @intCast((st >> 40) & 0xFF)) - 128;
        v.* = @as(f32, @floatFromInt(bits)) * (scale / 128.0);
    }
}

fn fillDetNorm(dst: []f32, seed: u64) void {
    var st = seed;
    for (dst) |*v| {
        st = st *% 6364136223846793005 +% 1442695040888963407;
        const frac = @as(f32, @floatFromInt((st >> 40) & 0xFF)) / 255.0;
        v.* = 0.85 + 0.3 * frac;
    }
}

fn pushTypedF32(
    allocator: std.mem.Allocator,
    items: *std.ArrayListUnmanaged(TypedAmfItem),
    nm: []const u8,
    vals: []const f32,
    s0: u32,
    s1: u32,
    rank: u8,
    kind: amf.TensorKind,
) !void {
    const data_bytes = vals.len * 4;
    const blob = try allocator.alloc(u8, try page.alignUp(@max(data_bytes, page.APPLE_PAGE_SIZE)));
    errdefer allocator.free(blob);
    @memset(blob, 0);
    for (vals, 0..) |val, i| {
        std.mem.writeInt(u32, blob[i * 4 ..][0..4], @bitCast(val), .little);
    }
    var it: TypedAmfItem = .{
        .blob = blob,
        .data_bytes = data_bytes,
        .dtype = .f32,
        .quant_bits = 0,
        .group_size = 0,
        .shape = .{ s0, s1, 0, 0 },
        .rank = rank,
        .kind = kind,
    };
    @memcpy(it.name[0..nm.len], nm);
    it.name_len = nm.len;
    try items.append(allocator, it);
}

fn pushTypedBf16(
    allocator: std.mem.Allocator,
    items: *std.ArrayListUnmanaged(TypedAmfItem),
    nm: []const u8,
    vals: []const f32,
    s0: u32,
    s1: u32,
    rank: u8,
    kind: amf.TensorKind,
) !void {
    const data_bytes = vals.len * 2;
    const blob = try allocator.alloc(u8, try page.alignUp(@max(data_bytes, page.APPLE_PAGE_SIZE)));
    errdefer allocator.free(blob);
    @memset(blob, 0);
    for (vals, 0..) |val, i| {
        const b16: u16 = @truncate(@as(u32, @bitCast(val)) >> 16);
        std.mem.writeInt(u16, blob[i * 2 ..][0..2], b16, .little);
    }
    var it: TypedAmfItem = .{
        .blob = blob,
        .data_bytes = data_bytes,
        .dtype = .bf16,
        .quant_bits = 0,
        .group_size = 0,
        .shape = .{ s0, s1, 0, 0 },
        .rank = rank,
        .kind = kind,
    };
    @memcpy(it.name[0..nm.len], nm);
    it.name_len = nm.len;
    try items.append(allocator, it);
}

fn pushTypedF16(
    allocator: std.mem.Allocator,
    items: *std.ArrayListUnmanaged(TypedAmfItem),
    nm: []const u8,
    vals: []const f32,
    s0: u32,
    s1: u32,
    rank: u8,
    kind: amf.TensorKind,
) !void {
    const data_bytes = vals.len * 2;
    const blob = try allocator.alloc(u8, try page.alignUp(@max(data_bytes, page.APPLE_PAGE_SIZE)));
    errdefer allocator.free(blob);
    @memset(blob, 0);
    for (vals, 0..) |val, i| {
        const h16: u16 = @bitCast(@as(f16, @floatCast(val)));
        std.mem.writeInt(u16, blob[i * 2 ..][0..2], h16, .little);
    }
    var it: TypedAmfItem = .{
        .blob = blob,
        .data_bytes = data_bytes,
        .dtype = .f16,
        .quant_bits = 0,
        .group_size = 0,
        .shape = .{ s0, s1, 0, 0 },
        .rank = rank,
        .kind = kind,
    };
    @memcpy(it.name[0..nm.len], nm);
    it.name_len = nm.len;
    try items.append(allocator, it);
}

fn pushTypedGgufQ8_0(
    allocator: std.mem.Allocator,
    items: *std.ArrayListUnmanaged(TypedAmfItem),
    nm: []const u8,
    vals: []const f32,
    rows: u32,
    cols: u32,
    kind: amf.TensorKind,
) !void {
    std.debug.assert(cols % 32 == 0 and vals.len == @as(usize, rows) * cols);
    const blocks_per_row = cols / 32;
    const data_bytes: usize = @as(usize, rows) * blocks_per_row * 34;
    const blob = try allocator.alloc(u8, try page.alignUp(@max(data_bytes, page.APPLE_PAGE_SIZE)));
    errdefer allocator.free(blob);
    @memset(blob, 0);
    for (0..rows) |r| {
        for (0..blocks_per_row) |b| {
            const src_blk = vals[r * cols + b * 32 ..][0..32];
            var max_abs: f32 = 0.0;
            for (src_blk) |x| max_abs = @max(max_abs, @abs(x));
            const scale_f32: f32 = if (max_abs > 0.0) max_abs / 127.0 else 1.0 / 127.0;
            const scale_f16: f16 = @floatCast(scale_f32);
            const eff_scale: f32 = @floatCast(scale_f16);
            const dst_blk = blob[(r * blocks_per_row + b) * 34 ..][0..34];
            std.mem.writeInt(u16, dst_blk[0..2], @bitCast(scale_f16), .little);
            for (0..32) |j| {
                const q_f = if (eff_scale > 0.0) std.math.round(src_blk[j] / eff_scale) else 0.0;
                const q_i8: i8 = @intFromFloat(std.math.clamp(q_f, -127.0, 127.0));
                dst_blk[2 + j] = @bitCast(q_i8);
            }
        }
    }
    var it: TypedAmfItem = .{
        .blob = blob,
        .data_bytes = data_bytes,
        .dtype = .gguf_q8_0,
        .quant_bits = 8,
        .group_size = 32,
        .shape = .{ rows, cols, 0, 0 },
        .rank = 2,
        .kind = kind,
    };
    @memcpy(it.name[0..nm.len], nm);
    it.name_len = nm.len;
    try items.append(allocator, it);
}

fn assembleTypedAmf(allocator: std.mem.Allocator, items: []const TypedAmfItem) ![]u8 {
    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_raw = items.len * @sizeOf(amf.TensorEntry);
    const dir_bytes = try page.alignUp(dir_raw);

    const entries = try allocator.alloc(amf.TensorEntry, items.len);
    defer allocator.free(entries);

    var next_off: usize = dir_offset + dir_bytes;
    for (items, 0..) |it, i| {
        var e = amf.TensorEntry.zeroed(it.name[0..it.name_len], it.kind, it.dtype);
        e.rank = it.rank;
        e.shape = it.shape;
        e.quant_bits = it.quant_bits;
        e.group_size = it.group_size;
        e.data_bytes = it.data_bytes;
        e.blob_offset = next_off;
        e.blob_bytes = it.blob.len;
        e.blob_crc32 = std.hash.Crc32.hash(it.blob);
        entries[i] = e;
        next_off += it.blob.len;
    }

    const total_size = try page.alignUp(next_off);
    const image = try allocator.alloc(u8, total_size);
    errdefer allocator.free(image);
    @memset(image, 0);
    for (items, 0..) |it, i| {
        @memcpy(image[dir_offset + i * @sizeOf(amf.TensorEntry) ..][0..@sizeOf(amf.TensorEntry)], std.mem.asBytes(&entries[i]));
        @memcpy(image[@intCast(entries[i].blob_offset)..][0..it.blob.len], it.blob);
    }
    var hdr: amf.Header = .{
        .magic = amf.MAGIC,
        .version = amf.FORMAT_VERSION,
        .flags = 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = amf.CHECKSUM_CRC32_IEEE,
        .tensor_count = @intCast(items.len),
        .reserved0 = 0,
        .directory_offset = dir_offset,
        .directory_bytes = dir_bytes,
        .directory_crc32 = std.hash.Crc32.hash(image[dir_offset..][0..dir_raw]),
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    const model_name = "synthetic-d16-gpu";
    @memcpy(hdr.model_id[0..model_name.len], model_name);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return image;
}

test "D16 Metal 4 GPU 3L DFlare (bf16 + f16 + 3-gate GRU + fuse_weight + GpuTargetTaps + GpuContextHistory) parity and zero steady-state malloc" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();

    const h: usize = 64;
    const inter: usize = 128;
    const nh: u32 = 4;
    const nkvh: u32 = 2;
    const hd: u32 = 16;
    const q_dim: usize = @as(usize, nh) * hd;
    const kv_dim: usize = @as(usize, nkvh) * hd;
    const mr: usize = 32;
    const v: usize = 48;
    const taps: usize = 4;
    const w: u8 = 8;

    const cfg: DFlareConfig = .{
        .variant = .inkling2_mini_3l,
        .num_layers = 3,
        .block_size = w,
        .top_k_per_pos = 1,
        .max_tree_nodes = 16,
        .hidden_size = @intCast(h),
        .intermediate_size = @intCast(inter),
        .num_attention_heads = nh,
        .num_key_value_heads = nkvh,
        .head_dim = hd,
        .vocab_size = @intCast(v),
        .markov_rank = @intCast(mr),
        .num_target_taps = @intCast(taps),
        .rms_eps = 1e-5,
        .rope_theta = 10_000.0,
    };

    var items: std.ArrayListUnmanaged(TypedAmfItem) = .empty;
    defer {
        for (items.items) |it| gpa.free(it.blob);
        items.deinit(gpa);
    }

    const scratch = try gpa.alloc(f32, @max(v * h, 3 * mr * (2 * h + 2 * mr)));
    defer gpa.free(scratch);

    // Pre-softmax `fuse_weight [4, 4]` (BF16).
    const fuse_w = scratch[0 .. (cfg.num_layers + 1) * taps];
    fillDetWeight(fuse_w, 101, 1.5);
    try pushTypedBf16(gpa, &items, "fuse_weight", fuse_w, cfg.num_layers + 1, @intCast(taps), 2, .norm);

    const mask_w = scratch[0 .. (w - 1) * h];
    fillDetWeight(mask_w, 102, 0.25);
    try pushTypedBf16(gpa, &items, "dflare.mask_embed.weight", mask_w, w - 1, @intCast(h), 2, .embedding);

    const tok_w = scratch[0 .. v * h];
    fillDetWeight(tok_w, 103, 0.35);
    try pushTypedBf16(gpa, &items, "dflare.tok_embed.weight", tok_w, @intCast(v), @intCast(h), 2, .embedding);

    const norm_h = scratch[0..h];
    fillDetNorm(norm_h, 104);
    try pushTypedBf16(gpa, &items, "fc_norm.weight", norm_h, @intCast(h), 0, 1, .norm);
    fillDetNorm(norm_h, 105);
    try pushTypedBf16(gpa, &items, "dflare.norm.weight", norm_h, @intCast(h), 0, 1, .norm);
    fillDetNorm(norm_h, 106);
    try pushTypedBf16(gpa, &items, "markov_head.dflare_norm.weight", norm_h, @intCast(h), 0, 1, .norm);
    fillDetNorm(norm_h, 107);
    try pushTypedBf16(gpa, &items, "dflare.markov_norm.weight", norm_h, @intCast(h), 0, 1, .norm);

    var nbuf: [64]u8 = undefined;
    for (0..cfg.num_layers) |l| {
        const sbase: u64 = 1000 + l * 100;
        fillDetNorm(scratch[0..h], sbase + 1);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l}), scratch[0..h], @intCast(h), 0, 1, .norm);
        fillDetNorm(scratch[0..h], sbase + 2);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l}), scratch[0..h], @intCast(h), 0, 1, .norm);
        fillDetNorm(scratch[0..h], sbase + 3);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.dflare_norm.weight", .{l}), scratch[0..h], @intCast(h), 0, 1, .norm);
        fillDetNorm(scratch[0..hd], sbase + 4);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}), scratch[0..hd], hd, 0, 1, .norm);
        fillDetNorm(scratch[0..hd], sbase + 5);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}), scratch[0..hd], hd, 0, 1, .norm);
        fillDetNorm(scratch[0..hd], sbase + 6);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm_ctx.weight", .{l}), scratch[0..hd], hd, 0, 1, .norm);

        fillDetWeight(scratch[0 .. q_dim * h], sbase + 10, 0.12);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj.weight", .{l}), scratch[0 .. q_dim * h], @intCast(q_dim), @intCast(h), 2, .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 11, 0.12);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), 2, .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 12, 0.12);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), 2, .dense);
        fillDetWeight(scratch[0 .. h * q_dim], sbase + 13, 0.12);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj.weight", .{l}), scratch[0 .. h * q_dim], @intCast(h), @intCast(q_dim), 2, .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 14, 0.12);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj_ctx.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), 2, .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 15, 0.12);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj_ctx.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), 2, .dense);

        fillDetWeight(scratch[0 .. inter * h], sbase + 20, 0.10);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj.weight", .{l}), scratch[0 .. inter * h], @intCast(inter), @intCast(h), 2, .dense);
        fillDetWeight(scratch[0 .. inter * h], sbase + 21, 0.10);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj.weight", .{l}), scratch[0 .. inter * h], @intCast(inter), @intCast(h), 2, .dense);
        fillDetWeight(scratch[0 .. h * inter], sbase + 22, 0.10);
        try pushTypedBf16(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj.weight", .{l}), scratch[0 .. h * inter], @intCast(h), @intCast(inter), 2, .dense);
    }

    const joint_in = 2 * h + 2 * mr;
    fillDetWeight(scratch[0 .. 3 * mr * joint_in], 501, 0.14);
    try pushTypedBf16(gpa, &items, "markov_head.rnn_proj.weight", scratch[0 .. 3 * mr * joint_in], @intCast(3 * mr), @intCast(joint_in), 2, .dense);
    fillDetWeight(scratch[0 .. 3 * mr], 502, 0.08);
    try pushTypedBf16(gpa, &items, "markov_head.rnn_proj.bias", scratch[0 .. 3 * mr], @intCast(3 * mr), 0, 1, .norm);
    fillDetWeight(scratch[0 .. v * mr], 503, 0.25);
    try pushTypedBf16(gpa, &items, "markov_head.markov_w1.weight", scratch[0 .. v * mr], @intCast(v), @intCast(mr), 2, .dense);
    fillDetWeight(scratch[0 .. h * mr], 504, 0.22);
    try pushTypedBf16(gpa, &items, "markov_head.hidden_out.weight", scratch[0 .. h * mr], @intCast(h), @intCast(mr), 2, .dense);
    fillDetWeight(scratch[0..h], 505, 0.05);
    try pushTypedBf16(gpa, &items, "markov_head.hidden_out.bias", scratch[0..h], @intCast(h), 0, 1, .norm);

    // Distinct diagonal boost on `f16` `lm_head` so greedy argmax has clear margins > FP16 noise.
    fillDetWeight(scratch[0 .. v * h], 506, 0.18);
    for (0..v) |tok| scratch[tok * h + (tok % h)] += 0.35;
    try pushTypedF16(gpa, &items, "dflare.lm_head.weight", scratch[0 .. v * h], @intCast(v), @intCast(h), 2, .lm_head);

    const img = try assembleTypedAmf(gpa, items.items);
    defer gpa.free(img);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "dflare-3l-d16.amf", .data = img });
    var p_buf: [std.fs.max_path_bytes]u8 = undefined;
    const p_raw = try std.fmt.bufPrint(p_buf[0 .. p_buf.len - 1], ".zig-cache/tmp/{s}/dflare-3l-d16.amf", .{tmp.sub_path});
    p_buf[p_raw.len] = 0;

    const bundle_cpu = try LoadedDFlareBundle.loadFromAmfFile(gpa, p_buf[0..p_raw.len :0], cfg, null, null);
    defer bundle_cpu.deinit(gpa);
    const bundle_gpu = try LoadedDFlareBundle.loadFromAmfFile(gpa, p_buf[0..p_raw.len :0], cfg, null, null);
    defer bundle_gpu.deinit(gpa);

    try bundle_gpu.proposer.attachGpu(.{ .device = device, .library = library });
    const gctx = bundle_gpu.proposer.gpu_ctx orelse return error.InvalidWeights;
    try std.testing.expect(gctx.weight_buf != null);
    try std.testing.expect(gctx.staging_buf == null); // 0-byte staging copy!

    // Prepare host & GPU context tap history (len grows 2 -> 5 -> 7 across 3 rounds).
    const max_hist: usize = 7;
    var hist_f32: [max_hist * taps * h]f32 = undefined;
    fillDetWeight(&hist_f32, 701, 0.45);
    const hist_cap: usize = 8;
    const tap_arena_bytes = try page.alignUp(hist_cap * taps * h * @sizeOf(f16) + taps * h * @sizeOf(f16));
    var tap_arena = try arena_mod.WiredArena.init(tap_arena_bytes, testBudget(tap_arena_bytes + page.APPLE_PAGE_SIZE));
    defer tap_arena.deinit();
    const ring_f16 = try allocArenaSlice(f16, &tap_arena, hist_cap * taps * h);
    const step_taps_f16 = try allocArenaSlice(f16, &tap_arena, taps * h);
    for (hist_f32, 0..) |val, i| ring_f16[i] = @floatCast(val);
    const tap_buf = try metal.Buffer.wrap(device, tap_arena.bytes());
    defer tap_buf.close();

    const HistBox = struct {
        data: []const f32,
        len: u32 = 2,
        epoch: u64 = 1,
        fn read(raw_ctx: *const anyopaque, out_len: *u32, out_epoch: *u64, pos: u32, tap_idx: usize, dst: []f32) bool {
            const box: *const @This() = @ptrCast(@alignCast(raw_ctx));
            out_len.* = box.len;
            out_epoch.* = box.epoch;
            if (dst.len == 0) return true;
            if (pos >= box.len or tap_idx >= taps or dst.len < h) return false;
            @memcpy(dst[0..h], box.data[(@as(usize, pos) * taps + tap_idx) * h ..][0..h]);
            return true;
        }
    };
    var hbox: HistBox = .{ .data = &hist_f32 };
    var tok_f32: [v * h]f32 = undefined;
    for (0..v) |ti| {
        const raw = bundle_cpu.proposer.weights.tok_embed_raw.?;
        try readRawRowF32(raw, ti, tok_f32[ti * h ..][0..h]);
    }
    const DualCtx = struct {
        tbl: []const f32,
        hist: *const HistBox,
        fn readTok(c: ?*anyopaque, tok_id: u32, dst: []f32) Error!void {
            const dc: *@This() = @ptrCast(@alignCast(c.?));
            if (tok_id >= v) return error.TokenOutOfRange;
            @memcpy(dst[0..h], dc.tbl[@as(usize, tok_id) * h ..][0..h]);
        }
        fn readHist(c: *const anyopaque, out_len: *u32, out_epoch: *u64, pos: u32, tap_idx: usize, dst: []f32) bool {
            const dc: *const @This() = @ptrCast(@alignCast(c));
            return HistBox.read(dc.hist, out_len, out_epoch, pos, tap_idx, dst);
        }
    };
    var dual_ctx: DualCtx = .{ .tbl = &tok_f32, .hist = &hbox };
    registerContextHistoryProvider(&dual_ctx, DualCtx.readHist);
    defer unregisterContextHistoryProvider(&dual_ctx);
    bundle_cpu.proposer.weights.tok_embed_hook = .{ .callback = .{ .ctx = &dual_ctx, .read_fn = DualCtx.readTok } };
    bundle_gpu.proposer.weights.tok_embed_hook = .{ .callback = .{ .ctx = &dual_ctx, .read_fn = DualCtx.readTok } };

    var cpu_tree: CandidateTree = undefined;
    var gpu_tree: CandidateTree = undefined;
    const round_lens = [_]u32{ 2, 3, 5, 7 };
    var gpu_taps_views: [taps]metal.BufferBinding = undefined;
    for (0..taps) |ti| {
        gpu_taps_views[ti] = .{
            .buffer = tap_buf,
            .offset = tap_arena.offsetOf(std.mem.sliceAsBytes(step_taps_f16[ti * h ..][0..h]).ptr),
        };
    }

    var warm_gpu_blocks: i64 = 0;
    for (round_lens, 0..) |cur_len, ri| {
        hbox.len = cur_len;
        const pos_off = (cur_len - 1) * taps * h;
        var tap_slices: [taps][]const f32 = undefined;
        for (0..taps) |ti| {
            tap_slices[ti] = hist_f32[pos_off + ti * h ..][0..h];
            for (0..h) |d| step_taps_f16[ti * h + d] = @floatCast(tap_slices[ti][d]);
        }

        const root_tok: u32 = @intCast((ri * 5 + 3) % v);
        const cpu_snap0 = try vmstat.Snapshot.capture();
        const cpu_res = try bundle_cpu.proposer.propose(root_tok, &tap_slices, null, &cpu_tree);
        const cpu_delta = vmstat.heapDelta(cpu_snap0, try vmstat.Snapshot.capture());

        const snap_before = try vmstat.Snapshot.capture();
        const gpu_res = try bundle_gpu.proposer.proposeWithGpuTaps(
            root_tok,
            .{ .dtype = .f16, .taps = &gpu_taps_views },
            .{
                .taps_ring = .{ .buffer = tap_buf, .offset = tap_arena.offsetOf(std.mem.sliceAsBytes(ring_f16).ptr) },
                .dtype = .f16,
                .cap = hist_cap,
                .len = cur_len,
                .epoch = 1,
            },
            null,
            &gpu_tree,
        );
        const gpu_delta = vmstat.heapDelta(snap_before, try vmstat.Snapshot.capture());
        // `propose` and `proposeWithGpuTaps` take no `std.mem.Allocator` (0 Zig heap allocs by
        // construction). `malloc_zone_statistics(null, ...)` is process-wide, so in the 495-test
        // multi-subsystem binary background Metal/CoreML/VideoToolbox libdispatch threads can
        // asynchronously reclaim or allocate a small number of OS blocks; clamp the warm baseline
        // at 0 and allow bounded background-thread slack while still catching unbounded leaks.
        const max_bg_driver_blocks: i64 = 32;
        if (std.c.getenv("MTL_DEBUG_LAYER") == null and std.c.getenv("MTL_SHADER_VALIDATION") == null) {
            try std.testing.expect(cpu_delta.blocks_in_use <= max_bg_driver_blocks);
            if (ri == 2) {
                warm_gpu_blocks = @max(@as(i64, 0), gpu_delta.blocks_in_use);
                try std.testing.expect(warm_gpu_blocks <= max_bg_driver_blocks);
            } else if (ri >= 3) {
                // O(1) constant Metal 4 runtime submit record only; 0 per-dispatch or Zig heap growth.
                try std.testing.expect(gpu_delta.blocks_in_use <= warm_gpu_blocks + max_bg_driver_blocks);
            }
        }

        try std.testing.expectEqual(@as(u64, 1), gpu_res.backbone_runs);
        try std.testing.expectEqual(cpu_res.tree_nodes, gpu_res.tree_nodes);
        try std.testing.expectEqual(cur_len, bundle_gpu.proposer.ctx_cache_len);
        for (0..cpu_tree.count) |ni| {
            try std.testing.expectEqual(cpu_tree.nodes[ni].token_id, gpu_tree.nodes[ni].token_id);
            try std.testing.expect(@abs(cpu_tree.nodes[ni].step_prob - gpu_tree.nodes[ni].step_prob) < 1e-2);
        }
    }

    // Verify multi-branch `top_k = 2` GPU tree expansion + hidden-state parity on host-tap path.
    var cfg_k2 = cfg;
    cfg_k2.top_k_per_pos = 2;
    const bundle_cpu_k2 = try LoadedDFlareBundle.loadFromAmfFile(gpa, p_buf[0..p_raw.len :0], cfg_k2, null, null);
    defer bundle_cpu_k2.deinit(gpa);
    const bundle_gpu_k2 = try LoadedDFlareBundle.loadFromAmfFile(gpa, p_buf[0..p_raw.len :0], cfg_k2, null, null);
    defer bundle_gpu_k2.deinit(gpa);
    try bundle_gpu_k2.proposer.attachGpu(.{ .device = device, .library = library });
    bundle_cpu_k2.proposer.weights.tok_embed_hook = .{ .callback = .{ .ctx = &dual_ctx, .read_fn = DualCtx.readTok } };
    bundle_gpu_k2.proposer.weights.tok_embed_hook = .{ .callback = .{ .ctx = &dual_ctx, .read_fn = DualCtx.readTok } };
    hbox.epoch = 2;
    hbox.len = 4;
    var tap_slices_k2: [taps][]const f32 = undefined;
    for (0..taps) |ti| tap_slices_k2[ti] = hist_f32[3 * taps * h + ti * h ..][0..h];
    _ = try bundle_cpu_k2.proposer.propose(7, &tap_slices_k2, null, &cpu_tree);
    _ = try bundle_gpu_k2.proposer.propose(7, &tap_slices_k2, null, &gpu_tree);
    try std.testing.expectEqual(cpu_tree.count, gpu_tree.count);
    for (0..cpu_tree.count) |ni| {
        try std.testing.expectEqual(cpu_tree.nodes[ni].token_id, gpu_tree.nodes[ni].token_id);
        try std.testing.expect(@abs(cpu_tree.nodes[ni].step_prob - gpu_tree.nodes[ni].step_prob) < 1e-2);
    }
    // Check backbone hidden state parity (`slot_hiddens` max abs diff < 1e-2) via `runBlockBackboneOnceGpuTaps`.
    try bundle_cpu_k2.proposer.runBlockBackboneOnceForRoot(7, &tap_slices_k2);
    try bundle_gpu_k2.proposer.runBlockBackboneOnceGpuTaps(
        7,
        .{ .dtype = .f16, .taps = &gpu_taps_views },
        .{
            .taps_ring = .{ .buffer = tap_buf, .offset = tap_arena.offsetOf(std.mem.sliceAsBytes(ring_f16).ptr) },
            .dtype = .f16,
            .cap = hist_cap,
            .len = 4,
            .epoch = 2,
        },
    );
    var max_h_diff: f32 = 0.0;
    for (bundle_cpu_k2.proposer.workspace.slot_hiddens[0 .. w * h], bundle_gpu_k2.proposer.workspace.slot_hiddens[0 .. w * h]) |a, b| {
        max_h_diff = @max(max_h_diff, @abs(a - b));
    }
    try std.testing.expect(max_h_diff < 1e-2);
}

test "D16 Metal 4 GPU arch=dflash (gguf_q8_0 layers + markov_w2_vocab + final_logit_softcapping + target_lm_head GPU binding) parity" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();

    const h: usize = 64;
    const inter: usize = 128;
    const nh: u32 = 4;
    const nkvh: u32 = 2;
    const hd: u32 = 16;
    const q_dim: usize = @as(usize, nh) * hd;
    const kv_dim: usize = @as(usize, nkvh) * hd;
    const mr: usize = 32;
    const v: usize = 48;
    const unpadded_v: usize = 40;
    const taps: usize = 4;
    const w: u8 = 8;

    const cfg: DFlareConfig = .{
        .variant = .dflash_standalone,
        .num_layers = 3,
        .block_size = w,
        .top_k_per_pos = 1,
        .max_tree_nodes = 16,
        .hidden_size = @intCast(h),
        .intermediate_size = @intCast(inter),
        .num_attention_heads = nh,
        .num_key_value_heads = nkvh,
        .head_dim = hd,
        .vocab_size = @intCast(v),
        .unpadded_vocab_size = @intCast(unpadded_v),
        .final_logit_softcapping = 30.0,
        .markov_rank = @intCast(mr),
        .num_target_taps = @intCast(taps),
        .rms_eps = 1e-5,
        .rope_theta = 1_000_000.0,
    };

    var items: std.ArrayListUnmanaged(TypedAmfItem) = .empty;
    defer {
        for (items.items) |it| gpa.free(it.blob);
        items.deinit(gpa);
    }

    const scratch = try gpa.alloc(f32, @max(v * h, 3 * mr * (2 * h + 2 * mr)));
    defer gpa.free(scratch);

    const mask_w = scratch[0 .. (w - 1) * h];
    fillDetWeight(mask_w, 202, 0.25);
    try pushTypedF32(gpa, &items, "mask_embd.weight", mask_w, w - 1, @intCast(h), 2, .embedding);

    const tok_w = scratch[0 .. v * h];
    fillDetWeight(tok_w, 203, 0.35);
    try pushTypedF16(gpa, &items, "token_embd.weight", tok_w, @intCast(v), @intCast(h), 2, .embedding);

    fillDetNorm(scratch[0..h], 204);
    try pushTypedF32(gpa, &items, "token_embd_norm.weight", scratch[0..h], @intCast(h), 0, 1, .norm);
    fillDetNorm(scratch[0..h], 205);
    try pushTypedF32(gpa, &items, "output_norm.weight", scratch[0..h], @intCast(h), 0, 1, .norm);
    fillDetNorm(scratch[0..h], 206);
    try pushTypedF32(gpa, &items, "markov_ctx_norm.weight", scratch[0..h], @intCast(h), 0, 1, .norm);
    fillDetNorm(scratch[0..h], 207);
    try pushTypedF32(gpa, &items, "markov_norm.weight", scratch[0..h], @intCast(h), 0, 1, .norm);

    var nbuf: [64]u8 = undefined;
    for (0..cfg.num_layers) |l| {
        const sbase: u64 = 2000 + l * 100;
        fillDetWeight(scratch[0..taps], sbase + 0, 0.5);
        for (scratch[0..taps]) |*val| val.* = 0.25 + 0.1 * @abs(val.*);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.ctx_mix.weight", .{l}), scratch[0..taps], @intCast(taps), 0, 1, .norm);
        fillDetNorm(scratch[0..h], sbase + 1);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn_norm.weight", .{l}), scratch[0..h], @intCast(h), 0, 1, .norm);
        fillDetNorm(scratch[0..h], sbase + 2);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.ffn_norm.weight", .{l}), scratch[0..h], @intCast(h), 0, 1, .norm);
        fillDetNorm(scratch[0..h], sbase + 3);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.ctx_norm.weight", .{l}), scratch[0..h], @intCast(h), 0, 1, .norm);
        fillDetNorm(scratch[0..hd], sbase + 4);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn_q_norm.weight", .{l}), scratch[0..hd], hd, 0, 1, .norm);
        fillDetNorm(scratch[0..hd], sbase + 5);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn_k_norm.weight", .{l}), scratch[0..hd], hd, 0, 1, .norm);
        fillDetNorm(scratch[0..hd], sbase + 6);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn_k_ctx_norm.weight", .{l}), scratch[0..hd], hd, 0, 1, .norm);

        fillDetWeight(scratch[0 .. q_dim * h], sbase + 10, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj.weight", .{l}), scratch[0 .. q_dim * h], @intCast(q_dim), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 11, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 12, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. h * q_dim], sbase + 13, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj.weight", .{l}), scratch[0 .. h * q_dim], @intCast(h), @intCast(q_dim), .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 14, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj_ctx.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 15, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj_ctx.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), .dense);

        fillDetWeight(scratch[0 .. inter * h], sbase + 20, 0.10);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj.weight", .{l}), scratch[0 .. inter * h], @intCast(inter), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. inter * h], sbase + 21, 0.10);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj.weight", .{l}), scratch[0 .. inter * h], @intCast(inter), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. h * inter], sbase + 22, 0.10);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj.weight", .{l}), scratch[0 .. h * inter], @intCast(h), @intCast(inter), .dense);
    }
    fillDetWeight(scratch[0..taps], 2399, 0.5);
    for (scratch[0..taps]) |*val| val.* = 0.25 + 0.1 * @abs(val.*);
    try pushTypedF32(gpa, &items, "markov_ctx_mix.weight", scratch[0..taps], @intCast(taps), 0, 1, .norm);

    const joint_in = 2 * h + 2 * mr;
    fillDetWeight(scratch[0 .. 3 * mr * joint_in], 601, 0.14);
    try pushTypedBf16(gpa, &items, "joint_proj.weight", scratch[0 .. 3 * mr * joint_in], @intCast(3 * mr), @intCast(joint_in), 2, .dense);
    fillDetWeight(scratch[0 .. 3 * mr], 602, 0.08);
    try pushTypedF32(gpa, &items, "joint_proj.bias", scratch[0 .. 3 * mr], @intCast(3 * mr), 0, 1, .norm);
    fillDetWeight(scratch[0 .. v * mr], 603, 0.22);
    try pushTypedBf16(gpa, &items, "markov_w1.weight", scratch[0 .. v * mr], @intCast(v), @intCast(mr), 2, .dense);
    fillDetWeight(scratch[0 .. unpadded_v * mr], 604, 0.25);
    for (0..unpadded_v) |tok| scratch[tok * mr + (tok % mr)] += 0.45;
    try pushTypedBf16(gpa, &items, "markov_w2.weight", scratch[0 .. unpadded_v * mr], @intCast(unpadded_v), @intCast(mr), 2, .dense);
    fillDetWeight(scratch[0..unpadded_v], 605, 0.10);
    try pushTypedF32(gpa, &items, "markov_w2.bias", scratch[0..unpadded_v], @intCast(unpadded_v), 0, 1, .norm);

    const img = try assembleTypedAmf(gpa, items.items);
    defer gpa.free(img);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "dflash-q8-d16.amf", .data = img });
    var p_buf: [std.fs.max_path_bytes]u8 = undefined;
    const p_raw = try std.fmt.bufPrint(p_buf[0 .. p_buf.len - 1], ".zig-cache/tmp/{s}/dflash-q8-d16.amf", .{tmp.sub_path});
    p_buf[p_raw.len] = 0;

    // External target `lm_head [v, h]` shared cleanly as BF16 on GPU and F32 on CPU.
    const tgt_arena_bytes = try page.alignUp(v * h * @sizeOf(u16) + 8 * taps * h * @sizeOf(f16) + taps * h * @sizeOf(f16));
    var ext_arena = try arena_mod.WiredArena.init(tgt_arena_bytes, testBudget(tgt_arena_bytes + page.APPLE_PAGE_SIZE));
    defer ext_arena.deinit();
    const lm_bf16 = try allocArenaSlice(u16, &ext_arena, v * h);
    var lm_f32: [v * h]f32 = undefined;
    fillDetWeight(&lm_f32, 606, 0.16);
    for (0..v) |tok| lm_f32[tok * h + (tok % h)] += 0.30;
    for (lm_f32, 0..) |fv, i| lm_bf16[i] = @truncate(@as(u32, @bitCast(fv)) >> 16);

    const hist_cap: usize = 8;
    const ring_f16 = try allocArenaSlice(f16, &ext_arena, hist_cap * taps * h);
    const step_taps_f16 = try allocArenaSlice(f16, &ext_arena, taps * h);
    var hist_f32: [6 * taps * h]f32 = undefined;
    fillDetWeight(&hist_f32, 709, 0.40);
    for (hist_f32, 0..) |val, i| ring_f16[i] = @floatCast(val);

    const ext_buf = try metal.Buffer.wrap(device, ext_arena.bytes());
    defer ext_buf.close();

    const bundle_cpu = try LoadedDFlareBundle.loadFromAmfFile(gpa, p_buf[0..p_raw.len :0], cfg, .{ .dense_f32 = &lm_f32 }, null);
    defer bundle_cpu.deinit(gpa);
    const bundle_gpu = try LoadedDFlareBundle.loadFromAmfFile(gpa, p_buf[0..p_raw.len :0], cfg, null, null);
    defer bundle_gpu.deinit(gpa);

    try bundle_gpu.proposer.attachGpu(.{
        .device = device,
        .library = library,
        .target_lm_head = .{
            .amf_dtype = .bf16,
            .weight = .{ .buffer = ext_buf, .offset = ext_arena.offsetOf(std.mem.sliceAsBytes(lm_bf16).ptr) },
            .n = @intCast(v),
            .k = @intCast(h),
        },
    });
    const gctx = bundle_gpu.proposer.gpu_ctx orelse return error.InvalidWeights;
    try std.testing.expect(gctx.weight_buf != null);
    try std.testing.expect(gctx.staging_buf == null);

    var tok_f32: [v * h]f32 = undefined;
    for (0..v) |ti| {
        try readRawRowF32(bundle_cpu.proposer.weights.tok_embed_raw.?, ti, tok_f32[ti * h ..][0..h]);
    }
    const CpuDualCtx = struct {
        tbl: []const f32,
        norm: ?[]const f32,
        eps: f32,
        hist: []const f32,
        len: u32 = 2,
        fn readTok(c: ?*anyopaque, tok_id: u32, dst: []f32) Error!void {
            const dc: *@This() = @ptrCast(@alignCast(c.?));
            if (tok_id >= v) return error.TokenOutOfRange;
            @memcpy(dst[0..h], dc.tbl[@as(usize, tok_id) * h ..][0..h]);
            if (dc.norm) |nw| applyRmsNormInPlace(dst[0..h], nw, dc.eps);
        }
        fn readHist(c: *const anyopaque, out_len: *u32, out_epoch: *u64, pos: u32, tap_idx: usize, dst: []f32) bool {
            const dc: *const @This() = @ptrCast(@alignCast(c));
            out_len.* = dc.len;
            out_epoch.* = 1;
            if (dst.len == 0) return true;
            if (pos >= dc.len or tap_idx >= taps or dst.len < h) return false;
            @memcpy(dst[0..h], dc.hist[(@as(usize, pos) * taps + tap_idx) * h ..][0..h]);
            return true;
        }
    };
    var dual_ctx: CpuDualCtx = .{
        .tbl = &tok_f32,
        .norm = bundle_cpu.proposer.weights.token_embd_norm,
        .eps = cfg.rms_eps,
        .hist = &hist_f32,
    };
    registerContextHistoryProvider(&dual_ctx, CpuDualCtx.readHist);
    defer unregisterContextHistoryProvider(&dual_ctx);
    bundle_cpu.proposer.weights.tok_embed_hook = .{ .callback = .{ .ctx = &dual_ctx, .read_fn = CpuDualCtx.readTok } };

    var gpu_taps_views: [taps]metal.BufferBinding = undefined;
    for (0..taps) |ti| {
        gpu_taps_views[ti] = .{
            .buffer = ext_buf,
            .offset = ext_arena.offsetOf(std.mem.sliceAsBytes(step_taps_f16[ti * h ..][0..h]).ptr),
        };
    }

    var cpu_tree: CandidateTree = undefined;
    var gpu_tree: CandidateTree = undefined;
    const round_lens = [_]u32{ 2, 3, 4, 5, 6 };
    var warm_gpu_blocks: i64 = 0;
    for (round_lens, 0..) |cur_len, ri| {
        dual_ctx.len = cur_len;
        const pos_off = (cur_len - 1) * taps * h;
        var tap_slices: [taps][]const f32 = undefined;
        for (0..taps) |ti| {
            tap_slices[ti] = hist_f32[pos_off + ti * h ..][0..h];
            for (0..h) |d| step_taps_f16[ti * h + d] = @floatCast(tap_slices[ti][d]);
        }

        const root_tok: u32 = @intCast((ri * 7 + 1) % unpadded_v);
        const cpu_snap0 = try vmstat.Snapshot.capture();
        const cpu_res = try bundle_cpu.proposer.propose(root_tok, &tap_slices, null, &cpu_tree);
        const cpu_delta = vmstat.heapDelta(cpu_snap0, try vmstat.Snapshot.capture());

        const snap_before = try vmstat.Snapshot.capture();
        const gpu_res = try bundle_gpu.proposer.proposeWithGpuTaps(
            root_tok,
            .{ .dtype = .f16, .taps = &gpu_taps_views },
            .{
                .taps_ring = .{ .buffer = ext_buf, .offset = ext_arena.offsetOf(std.mem.sliceAsBytes(ring_f16).ptr) },
                .dtype = .f16,
                .cap = hist_cap,
                .len = cur_len,
                .epoch = 1,
            },
            null,
            &gpu_tree,
        );
        const gpu_delta = vmstat.heapDelta(snap_before, try vmstat.Snapshot.capture());
        // Same process-wide `malloc_zone_statistics` background-thread guard as the 3L DFlare test.
        const max_bg_driver_blocks: i64 = 32;
        if (std.c.getenv("MTL_DEBUG_LAYER") == null and std.c.getenv("MTL_SHADER_VALIDATION") == null) {
            try std.testing.expect(cpu_delta.blocks_in_use <= max_bg_driver_blocks);
            if (ri == 2) {
                warm_gpu_blocks = @max(@as(i64, 0), gpu_delta.blocks_in_use);
                try std.testing.expect(warm_gpu_blocks <= max_bg_driver_blocks);
            } else if (ri >= 3) {
                try std.testing.expect(gpu_delta.blocks_in_use <= warm_gpu_blocks + max_bg_driver_blocks);
            }
        }

        try std.testing.expectEqual(@as(u64, 1), gpu_res.backbone_runs);
        try std.testing.expectEqual(cpu_res.tree_nodes, gpu_res.tree_nodes);
        for (0..cpu_tree.count) |ni| {
            try std.testing.expectEqual(cpu_tree.nodes[ni].token_id, gpu_tree.nodes[ni].token_id);
            try std.testing.expect(@abs(cpu_tree.nodes[ni].step_prob - gpu_tree.nodes[ni].step_prob) < 1e-2);
        }
    }
}

test "D16 Metal 4 GPU arch=dflash gguf_q8_0 markov_w1 stays zero-copy single-frame with CPU parity" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();

    const h: usize = 64;
    const inter: usize = 128;
    const nh: u32 = 4;
    const nkvh: u32 = 2;
    const hd: u32 = 16;
    const q_dim: usize = @as(usize, nh) * hd;
    const kv_dim: usize = @as(usize, nkvh) * hd;
    const mr: usize = 64; // multiple of 32 (2 GGUF Q8_0 blocks per vocab row)
    const v: usize = 48;
    const unpadded_v: usize = 40;
    const taps: usize = 4;
    const w: u8 = 8;

    const cfg: DFlareConfig = .{
        .variant = .dflash_standalone,
        .num_layers = 2,
        .block_size = w,
        .top_k_per_pos = 1,
        .max_tree_nodes = 4, // exercises M=4 chain_end cap as well!
        .hidden_size = @intCast(h),
        .intermediate_size = @intCast(inter),
        .num_attention_heads = nh,
        .num_key_value_heads = nkvh,
        .head_dim = hd,
        .vocab_size = @intCast(v),
        .unpadded_vocab_size = @intCast(unpadded_v),
        .final_logit_softcapping = 30.0,
        .markov_rank = @intCast(mr),
        .num_target_taps = @intCast(taps),
        .rms_eps = 1e-5,
        .rope_theta = 1_000_000.0,
    };

    var items: std.ArrayListUnmanaged(TypedAmfItem) = .empty;
    defer {
        for (items.items) |it| gpa.free(it.blob);
        items.deinit(gpa);
    }

    const scratch = try gpa.alloc(f32, @max(v * h, 3 * mr * (2 * h + 2 * mr)));
    defer gpa.free(scratch);

    fillDetWeight(scratch[0 .. (w - 1) * h], 302, 0.25);
    try pushTypedF32(gpa, &items, "mask_embd.weight", scratch[0 .. (w - 1) * h], w - 1, @intCast(h), 2, .embedding);
    fillDetWeight(scratch[0 .. v * h], 303, 0.35);
    try pushTypedF16(gpa, &items, "token_embd.weight", scratch[0 .. v * h], @intCast(v), @intCast(h), 2, .embedding);
    fillDetNorm(scratch[0..h], 304);
    try pushTypedF32(gpa, &items, "token_embd_norm.weight", scratch[0..h], @intCast(h), 0, 1, .norm);
    fillDetNorm(scratch[0..h], 305);
    try pushTypedF32(gpa, &items, "output_norm.weight", scratch[0..h], @intCast(h), 0, 1, .norm);
    fillDetNorm(scratch[0..h], 306);
    try pushTypedF32(gpa, &items, "markov_ctx_norm.weight", scratch[0..h], @intCast(h), 0, 1, .norm);
    fillDetNorm(scratch[0..h], 307);
    try pushTypedF32(gpa, &items, "markov_norm.weight", scratch[0..h], @intCast(h), 0, 1, .norm);

    var nbuf: [64]u8 = undefined;
    for (0..cfg.num_layers) |l| {
        const sbase: u64 = 3000 + l * 100;
        fillDetWeight(scratch[0..taps], sbase + 0, 0.5);
        for (scratch[0..taps]) |*val| val.* = 0.25 + 0.1 * @abs(val.*);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.ctx_mix.weight", .{l}), scratch[0..taps], @intCast(taps), 0, 1, .norm);
        fillDetNorm(scratch[0..h], sbase + 1);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn_norm.weight", .{l}), scratch[0..h], @intCast(h), 0, 1, .norm);
        fillDetNorm(scratch[0..h], sbase + 2);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.ffn_norm.weight", .{l}), scratch[0..h], @intCast(h), 0, 1, .norm);
        fillDetNorm(scratch[0..h], sbase + 3);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.ctx_norm.weight", .{l}), scratch[0..h], @intCast(h), 0, 1, .norm);
        fillDetNorm(scratch[0..hd], sbase + 4);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn_q_norm.weight", .{l}), scratch[0..hd], hd, 0, 1, .norm);
        fillDetNorm(scratch[0..hd], sbase + 5);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn_k_norm.weight", .{l}), scratch[0..hd], hd, 0, 1, .norm);
        fillDetNorm(scratch[0..hd], sbase + 6);
        try pushTypedF32(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn_k_ctx_norm.weight", .{l}), scratch[0..hd], hd, 0, 1, .norm);

        fillDetWeight(scratch[0 .. q_dim * h], sbase + 10, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj.weight", .{l}), scratch[0 .. q_dim * h], @intCast(q_dim), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 11, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 12, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. h * q_dim], sbase + 13, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj.weight", .{l}), scratch[0 .. h * q_dim], @intCast(h), @intCast(q_dim), .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 14, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj_ctx.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. kv_dim * h], sbase + 15, 0.12);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj_ctx.weight", .{l}), scratch[0 .. kv_dim * h], @intCast(kv_dim), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. inter * h], sbase + 20, 0.10);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj.weight", .{l}), scratch[0 .. inter * h], @intCast(inter), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. inter * h], sbase + 21, 0.10);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj.weight", .{l}), scratch[0 .. inter * h], @intCast(inter), @intCast(h), .dense);
        fillDetWeight(scratch[0 .. h * inter], sbase + 22, 0.10);
        try pushTypedGgufQ8_0(gpa, &items, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj.weight", .{l}), scratch[0 .. h * inter], @intCast(h), @intCast(inter), .dense);
    }
    fillDetWeight(scratch[0..taps], 3399, 0.5);
    for (scratch[0..taps]) |*val| val.* = 0.25 + 0.1 * @abs(val.*);
    try pushTypedF32(gpa, &items, "markov_ctx_mix.weight", scratch[0..taps], @intCast(taps), 0, 1, .norm);

    const joint_in = 2 * h + 2 * mr;
    fillDetWeight(scratch[0 .. 3 * mr * joint_in], 701, 0.14);
    try pushTypedBf16(gpa, &items, "joint_proj.weight", scratch[0 .. 3 * mr * joint_in], @intCast(3 * mr), @intCast(joint_in), 2, .dense);
    fillDetWeight(scratch[0 .. 3 * mr], 702, 0.08);
    try pushTypedF32(gpa, &items, "joint_proj.bias", scratch[0 .. 3 * mr], @intCast(3 * mr), 0, 1, .norm);
    fillDetWeight(scratch[0 .. v * mr], 703, 0.35);
    try pushTypedGgufQ8_0(gpa, &items, "markov_w1.weight", scratch[0 .. v * mr], @intCast(v), @intCast(mr), .dense);
    fillDetWeight(scratch[0 .. unpadded_v * mr], 704, 0.25);
    for (0..unpadded_v) |tok| scratch[tok * mr + (tok % mr)] += 0.45;
    try pushTypedBf16(gpa, &items, "markov_w2.weight", scratch[0 .. unpadded_v * mr], @intCast(unpadded_v), @intCast(mr), 2, .dense);
    fillDetWeight(scratch[0..unpadded_v], 705, 0.10);
    try pushTypedF32(gpa, &items, "markov_w2.bias", scratch[0..unpadded_v], @intCast(unpadded_v), 0, 1, .norm);

    const img = try assembleTypedAmf(gpa, items.items);
    defer gpa.free(img);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "dflash-q8-w1.amf", .data = img });
    var p_buf: [std.fs.max_path_bytes]u8 = undefined;
    const p_raw = try std.fmt.bufPrint(p_buf[0 .. p_buf.len - 1], ".zig-cache/tmp/{s}/dflash-q8-w1.amf", .{tmp.sub_path});
    p_buf[p_raw.len] = 0;

    const tgt_arena_bytes = try page.alignUp(v * h * @sizeOf(u16) + 4 * taps * h * @sizeOf(f16) + taps * h * @sizeOf(f16));
    var ext_arena = try arena_mod.WiredArena.init(tgt_arena_bytes, testBudget(tgt_arena_bytes + page.APPLE_PAGE_SIZE));
    defer ext_arena.deinit();
    const lm_bf16 = try allocArenaSlice(u16, &ext_arena, v * h);
    var lm_f32: [v * h]f32 = undefined;
    fillDetWeight(&lm_f32, 706, 0.16);
    for (0..v) |tok| lm_f32[tok * h + (tok % h)] += 0.30;
    for (lm_f32, 0..) |fv, i| lm_bf16[i] = @truncate(@as(u32, @bitCast(fv)) >> 16);

    const step_taps_f16 = try allocArenaSlice(f16, &ext_arena, taps * h);
    var step_taps_f32: [taps * h]f32 = undefined;
    fillDetWeight(&step_taps_f32, 707, 0.40);
    for (step_taps_f32, 0..) |val, i| step_taps_f16[i] = @floatCast(val);

    const ext_buf = try metal.Buffer.wrap(device, ext_arena.bytes());
    defer ext_buf.close();

    const bundle_cpu = try LoadedDFlareBundle.loadFromAmfFile(gpa, p_buf[0..p_raw.len :0], cfg, .{ .dense_f32 = &lm_f32 }, null);
    defer bundle_cpu.deinit(gpa);
    const bundle_gpu = try LoadedDFlareBundle.loadFromAmfFile(gpa, p_buf[0..p_raw.len :0], cfg, null, null);
    defer bundle_gpu.deinit(gpa);

    try bundle_gpu.proposer.attachGpu(.{
        .device = device,
        .library = library,
        .target_lm_head = .{
            .amf_dtype = .bf16,
            .weight = .{ .buffer = ext_buf, .offset = ext_arena.offsetOf(std.mem.sliceAsBytes(lm_bf16).ptr) },
            .n = @intCast(v),
            .k = @intCast(h),
        },
    });
    const gctx = bundle_gpu.proposer.gpu_ctx orelse return error.InvalidWeights;
    try std.testing.expect(gctx.weight_buf != null);
    try std.testing.expect(gctx.staging_buf == null);
    try std.testing.expect(gctx.markov_w1_vocab != null);
    try std.testing.expect(gctx.markov_w1_vocab_raw == null);
    try std.testing.expectEqual(@as(u32, 3), gctx.markov_w1_vocab_dtype);

    var tap_slices: [taps][]const f32 = undefined;
    var gpu_taps_views: [taps]metal.BufferBinding = undefined;
    const base_taps_off = ext_arena.offsetOf(std.mem.sliceAsBytes(step_taps_f16).ptr);
    for (0..taps) |ti| {
        tap_slices[ti] = step_taps_f32[ti * h ..][0..h];
        gpu_taps_views[ti] = .{ .buffer = ext_buf, .offset = base_taps_off + ti * h * @sizeOf(f16) };
    }

    var cpu_tree = CandidateTree.init(0);
    var gpu_tree = CandidateTree.init(0);
    const cpu_res = try bundle_cpu.proposer.propose(5, &tap_slices, null, &cpu_tree);
    const gpu_res = try bundle_gpu.proposer.proposeWithGpuTaps(5, .{ .dtype = .f16, .taps = &gpu_taps_views }, null, null, &gpu_tree);

    try std.testing.expectEqual(@as(u16, 4), gpu_res.tree_nodes);
    try std.testing.expectEqual(@as(u64, 3), gpu_res.markov_steps); // M=4 runs 3 Markov steps, not 7!
    try std.testing.expectEqual(cpu_res.tree_nodes, gpu_res.tree_nodes);
    for (0..cpu_tree.count) |ni| {
        try std.testing.expectEqual(cpu_tree.nodes[ni].token_id, gpu_tree.nodes[ni].token_id);
        try std.testing.expect(@abs(cpu_tree.nodes[ni].step_prob - gpu_tree.nodes[ni].step_prob) < 1e-2);
    }
}

test "D16 Metal 4 GPU simple non-GRU DFlare (dense fc + markov_w1/rnn_proj/w2 + top_k=1 and top_k=2) parity" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();

    const h: usize = 32;
    const inter: usize = 64;
    const nh: u32 = 2;
    const nkvh: u32 = 2;
    const hd: u32 = 16;
    const q_dim: usize = @as(usize, nh) * hd;
    const kv_dim: usize = @as(usize, nkvh) * hd;
    const mr: usize = 32;
    const v: usize = 24;
    const taps: usize = 2;
    const w: u8 = 4;

    const cfg: DFlareConfig = .{
        .variant = .dflash_standalone,
        .num_layers = 3,
        .block_size = w,
        .top_k_per_pos = 1,
        .max_tree_nodes = 15,
        .hidden_size = @intCast(h),
        .intermediate_size = @intCast(inter),
        .num_attention_heads = nh,
        .num_key_value_heads = nkvh,
        .head_dim = hd,
        .vocab_size = @intCast(v),
        .markov_rank = @intCast(mr),
        .num_target_taps = @intCast(taps),
        .rms_eps = 1e-5,
    };

    var l_q: [3 * q_dim * h]f32 = undefined;
    var l_k: [3 * kv_dim * h]f32 = undefined;
    var l_v: [3 * kv_dim * h]f32 = undefined;
    var l_o: [3 * h * q_dim]f32 = undefined;
    var l_gate: [3 * inter * h]f32 = undefined;
    var l_up: [3 * inter * h]f32 = undefined;
    var l_down: [3 * h * inter]f32 = undefined;
    var layers: [3]TransformerBlockWeights = undefined;
    for (0..3) |li| {
        fillDetWeight(l_q[li * q_dim * h ..][0 .. q_dim * h], 3000 + li * 10 + 1, 0.14);
        fillDetWeight(l_k[li * kv_dim * h ..][0 .. kv_dim * h], 3000 + li * 10 + 2, 0.14);
        fillDetWeight(l_v[li * kv_dim * h ..][0 .. kv_dim * h], 3000 + li * 10 + 3, 0.14);
        fillDetWeight(l_o[li * h * q_dim ..][0 .. h * q_dim], 3000 + li * 10 + 4, 0.14);
        fillDetWeight(l_gate[li * inter * h ..][0 .. inter * h], 3000 + li * 10 + 5, 0.12);
        fillDetWeight(l_up[li * inter * h ..][0 .. inter * h], 3000 + li * 10 + 6, 0.12);
        fillDetWeight(l_down[li * h * inter ..][0 .. h * inter], 3000 + li * 10 + 7, 0.12);
        layers[li] = .{
            .q_proj = l_q[li * q_dim * h ..][0 .. q_dim * h],
            .k_proj = l_k[li * kv_dim * h ..][0 .. kv_dim * h],
            .v_proj = l_v[li * kv_dim * h ..][0 .. kv_dim * h],
            .o_proj = l_o[li * h * q_dim ..][0 .. h * q_dim],
            .mlp_gate_proj = l_gate[li * inter * h ..][0 .. inter * h],
            .mlp_up_proj = l_up[li * inter * h ..][0 .. inter * h],
            .mlp_down_proj = l_down[li * h * inter ..][0 .. h * inter],
        };
    }

    var fc: [h * taps * h]f32 = undefined;
    fillDetWeight(&fc, 801, 0.18);
    var mask_embed: [w * h]f32 = undefined;
    fillDetWeight(&mask_embed, 802, 0.30);
    var tok_embed: [v * h]f32 = undefined;
    fillDetWeight(&tok_embed, 803, 0.35);
    var mw1: [mr * 2 * h]f32 = undefined;
    fillDetWeight(&mw1, 804, 0.20);
    var mrnn: [mr * mr]f32 = undefined;
    fillDetWeight(&mrnn, 805, 0.22);
    var mb1: [mr]f32 = undefined;
    fillDetWeight(&mb1, 806, 0.05);
    var mw2: [h * mr]f32 = undefined;
    fillDetWeight(&mw2, 807, 0.22);
    var lm_head: [v * h]f32 = undefined;
    fillDetWeight(&lm_head, 808, 0.18);
    for (0..v) |tok| lm_head[tok * h + (tok % h)] += 0.40;

    const weights: DFlareWeights = .{
        .target_fusion_proj = &fc,
        .mask_embed = &mask_embed,
        .tok_embed = &tok_embed,
        .layers = &layers,
        .markov_w1 = &mw1,
        .markov_rnn_proj = &mrnn,
        .markov_b1 = &mb1,
        .markov_w2 = &mw2,
        .lm_head = .{ .dense_f32 = &lm_head },
    };

    var ws_cpu = try DFlareWorkspace.init(cfg, null);
    defer ws_cpu.deinit();
    var ws_gpu = try DFlareWorkspace.init(cfg, null);
    defer ws_gpu.deinit();

    var cpu_proposer = try DFlareProposer.init(cfg, weights, &ws_cpu);
    var gpu_proposer = try DFlareProposer.init(cfg, weights, &ws_gpu);
    try gpu_proposer.attachGpu(.{ .device = device, .library = library });
    defer gpu_proposer.detachGpu();

    var tap0: [h]f32 = undefined;
    var tap1: [h]f32 = undefined;
    fillDetWeight(&tap0, 901, 0.45);
    fillDetWeight(&tap1, 902, 0.45);
    var cpu_tree: CandidateTree = undefined;
    var gpu_tree: CandidateTree = undefined;

    _ = try cpu_proposer.propose(2, &.{ &tap0, &tap1 }, null, &cpu_tree);
    _ = try gpu_proposer.propose(2, &.{ &tap0, &tap1 }, null, &gpu_tree);
    try std.testing.expectEqual(cpu_tree.count, gpu_tree.count);
    for (0..cpu_tree.count) |ni| {
        try std.testing.expectEqual(cpu_tree.nodes[ni].token_id, gpu_tree.nodes[ni].token_id);
        try std.testing.expect(@abs(cpu_tree.nodes[ni].step_prob - gpu_tree.nodes[ni].step_prob) < 1e-2);
    }

    var cfg_k2 = cfg;
    cfg_k2.top_k_per_pos = 2;
    var ws_cpu_k2 = try DFlareWorkspace.init(cfg_k2, null);
    defer ws_cpu_k2.deinit();
    var ws_gpu_k2 = try DFlareWorkspace.init(cfg_k2, null);
    defer ws_gpu_k2.deinit();
    var cpu_proposer_k2 = try DFlareProposer.init(cfg_k2, weights, &ws_cpu_k2);
    var gpu_proposer_k2 = try DFlareProposer.init(cfg_k2, weights, &ws_gpu_k2);
    try gpu_proposer_k2.attachGpu(.{ .device = device, .library = library });
    defer gpu_proposer_k2.detachGpu();

    _ = try cpu_proposer_k2.propose(5, &.{ &tap0, &tap1 }, null, &cpu_tree);
    _ = try gpu_proposer_k2.propose(5, &.{ &tap0, &tap1 }, null, &gpu_tree);
    try std.testing.expectEqual(cpu_tree.count, gpu_tree.count);
    for (0..cpu_tree.count) |ni| {
        try std.testing.expectEqual(cpu_tree.nodes[ni].token_id, gpu_tree.nodes[ni].token_id);
        try std.testing.expect(@abs(cpu_tree.nodes[ni].step_prob - gpu_tree.nodes[ni].step_prob) < 1e-2);
    }
}

test "M9.1 D15 DFlash 5L direct block-diffusion (B=8 & B=16, markov_rank=0, 5-tap fc, rope_dim=64, SWA, v_scale=0.612, sink_bias, mask_token_id) GPU==CPU parity" {
    const allocator = std.testing.allocator;

    // 1. Verify DFlareConfig.fromJson parses MiMo-V2.6 dflash/config.json schema cleanly.
    const mimo_dflash_json =
        \\{
        \\  "architectures": ["DFlashDraftModel"],
        \\  "hidden_size": 4096,
        \\  "intermediate_size": 16384,
        \\  "num_hidden_layers": 5,
        \\  "num_attention_heads": 64,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 128,
        \\  "partial_rotary_factor": 0.5,
        \\  "rope_theta": 10000.0,
        \\  "sliding_window": 1024,
        \\  "attention_value_scale": 0.612,
        \\  "vocab_size": 151936,
        \\  "rms_norm_eps": 1e-6,
        \\  "block_size": 16,
        \\  "dflash_config": {
        \\    "mask_token_id": 151675,
        \\    "target_layer_ids": [0, 11, 23, 35, 47]
        \\  }
        \\}
    ;
    try std.testing.expectEqual(@as(?DraftKindDetected, .dflare), detectDraftKindFromJson(mimo_dflash_json));
    const parsed_cfg = try DFlareConfig.fromJson(allocator, mimo_dflash_json, 4096, 151936);
    try std.testing.expectEqual(@as(u32, 5), parsed_cfg.num_layers);
    try std.testing.expectEqual(@as(u8, 16), parsed_cfg.block_size);
    try std.testing.expectEqual(@as(u32, 0), parsed_cfg.markov_rank);
    try std.testing.expectEqual(@as(u32, 5), parsed_cfg.num_target_taps);
    try std.testing.expectEqual(@as(u32, 64), parsed_cfg.effectiveRopeDim());
    try std.testing.expectEqual(@as(u32, 1024), parsed_cfg.sliding_window);
    try std.testing.expectEqual(@as(?u32, 151675), parsed_cfg.mask_token_id);
    try std.testing.expectApproxEqAbs(@as(f32, 0.612), parsed_cfg.attention_value_scale, 1e-5);

    // 2. Exercise 5-layer direct block-diffusion (markov_rank=0) with 5-tap fc + hidden_norm,
    //    q_norm, k_norm, sink_bias, v_scale=0.612, partial RoPE (64/128), circular SWA wraparound,
    //    and mask_token_id seeding across BOTH B=8 and B=16 (plus 2-iter refinement + top-2 fork)!
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();

    const h: usize = 128;
    const inter: usize = 256;
    const nh: u32 = 2;
    const nkvh: u32 = 1;
    const hd: u32 = 128;
    const q_dim: usize = @as(usize, nh) * hd;
    const kv_dim: usize = @as(usize, nkvh) * hd;
    const v: usize = 32;
    const taps: usize = 5;
    const num_layers: usize = 5;

    const l_q = try allocator.alloc(f32, num_layers * q_dim * h);
    defer allocator.free(l_q);
    const l_k = try allocator.alloc(f32, num_layers * kv_dim * h);
    defer allocator.free(l_k);
    const l_v = try allocator.alloc(f32, num_layers * kv_dim * h);
    defer allocator.free(l_v);
    const l_o = try allocator.alloc(f32, num_layers * h * q_dim);
    defer allocator.free(l_o);
    const l_gate = try allocator.alloc(f32, num_layers * inter * h);
    defer allocator.free(l_gate);
    const l_up = try allocator.alloc(f32, num_layers * inter * h);
    defer allocator.free(l_up);
    const l_down = try allocator.alloc(f32, num_layers * h * inter);
    defer allocator.free(l_down);
    const l_in_norm = try allocator.alloc(f32, num_layers * h);
    defer allocator.free(l_in_norm);
    const l_post_norm = try allocator.alloc(f32, num_layers * h);
    defer allocator.free(l_post_norm);
    const l_q_norm = try allocator.alloc(f32, num_layers * hd);
    defer allocator.free(l_q_norm);
    const l_k_norm = try allocator.alloc(f32, num_layers * hd);
    defer allocator.free(l_k_norm);
    const l_sink = try allocator.alloc(f32, num_layers * nh);
    defer allocator.free(l_sink);

    var layers: [num_layers]TransformerBlockWeights = undefined;
    for (0..num_layers) |li| {
        fillDetWeight(l_q[li * q_dim * h ..][0 .. q_dim * h], 7100 + li * 16 + 1, 0.08);
        fillDetWeight(l_k[li * kv_dim * h ..][0 .. kv_dim * h], 7100 + li * 16 + 2, 0.08);
        fillDetWeight(l_v[li * kv_dim * h ..][0 .. kv_dim * h], 7100 + li * 16 + 3, 0.08);
        fillDetWeight(l_o[li * h * q_dim ..][0 .. h * q_dim], 7100 + li * 16 + 4, 0.08);
        fillDetWeight(l_gate[li * inter * h ..][0 .. inter * h], 7100 + li * 16 + 5, 0.07);
        fillDetWeight(l_up[li * inter * h ..][0 .. inter * h], 7100 + li * 16 + 6, 0.07);
        fillDetWeight(l_down[li * h * inter ..][0 .. h * inter], 7100 + li * 16 + 7, 0.07);
        for (0..h) |d| {
            l_in_norm[li * h + d] = 1.0 + 0.05 * @sin(@as(f32, @floatFromInt(li * 31 + d)));
            l_post_norm[li * h + d] = 1.0 + 0.05 * @cos(@as(f32, @floatFromInt(li * 17 + d)));
        }
        for (0..hd) |d| {
            l_q_norm[li * hd + d] = 1.0 + 0.04 * @sin(@as(f32, @floatFromInt(li * 13 + d)));
            l_k_norm[li * hd + d] = 1.0 + 0.04 * @cos(@as(f32, @floatFromInt(li * 11 + d)));
        }
        for (0..nh) |hi| {
            l_sink[li * nh + hi] = -0.35 + 0.4 * @as(f32, @floatFromInt(hi)) + 0.1 * @as(f32, @floatFromInt(li));
        }
        layers[li] = .{
            .in_norm_weight = l_in_norm[li * h ..][0..h],
            .q_proj = l_q[li * q_dim * h ..][0 .. q_dim * h],
            .k_proj = l_k[li * kv_dim * h ..][0 .. kv_dim * h],
            .v_proj = l_v[li * kv_dim * h ..][0 .. kv_dim * h],
            .q_norm_weight = l_q_norm[li * hd ..][0..hd],
            .k_norm_weight = l_k_norm[li * hd ..][0..hd],
            .sink_bias = l_sink[li * nh ..][0..nh],
            .o_proj = l_o[li * h * q_dim ..][0 .. h * q_dim],
            .post_norm_weight = l_post_norm[li * h ..][0..h],
            .mlp_gate_proj = l_gate[li * inter * h ..][0 .. inter * h],
            .mlp_up_proj = l_up[li * inter * h ..][0 .. inter * h],
            .mlp_down_proj = l_down[li * h * inter ..][0 .. h * inter],
        };
    }

    const fc = try allocator.alloc(f32, h * taps * h);
    defer allocator.free(fc);
    fillDetWeight(fc, 7501, 0.06);
    var hidden_norm: [h]f32 = undefined;
    var final_norm: [h]f32 = undefined;
    for (0..h) |d| {
        hidden_norm[d] = 1.0 + 0.05 * @sin(@as(f32, @floatFromInt(d)));
        final_norm[d] = 1.0 + 0.05 * @cos(@as(f32, @floatFromInt(d)));
    }
    const tok_embed = try allocator.alloc(f32, v * h);
    defer allocator.free(tok_embed);
    fillDetWeight(tok_embed, 7502, 0.30);
    const lm_head = try allocator.alloc(f32, v * h);
    defer allocator.free(lm_head);
    fillDetWeight(lm_head, 7503, 0.14);
    for (0..v) |tok| lm_head[tok * h + (tok % h)] += 1.5 + 0.07 * @as(f32, @floatFromInt(tok));

    const weights: DFlareWeights = .{
        .target_fusion_proj = fc,
        .target_fusion_norm = &hidden_norm,
        .final_norm = &final_norm,
        .mask_embed = &.{}, // Resolved from tok_embed[mask_token_id]
        .tok_embed = tok_embed,
        .layers = &layers,
        .lm_head = .{ .dense_f32 = lm_head },
    };

    var tap_storage: [taps][h]f32 = undefined;
    var tap_slices: [taps][]const f32 = undefined;
    for (0..taps) |ti| {
        fillDetWeight(&tap_storage[ti], 7600 + ti, 0.40);
        tap_slices[ti] = &tap_storage[ti];
    }

    // Build 10 positions of deterministic 5-tap target hidden history so circular SWA (`sliding_window=6`)
    // wraps around across multiple rounds!
    const hist_cap: usize = 12;
    const hist_f32 = try allocator.alloc(f32, hist_cap * taps * h);
    defer allocator.free(hist_f32);
    fillDetWeight(hist_f32, 7700, 0.40);

    const HistState = struct {
        data: []const f32,
        len: u32,
        epoch: u64,
        h_dim: usize,
        n_taps: usize,
        fn read(c: *const anyopaque, out_len: *u32, out_epoch: *u64, pos: u32, tap_idx: usize, dst: []f32) bool {
            const self_p: *const @This() = @ptrCast(@alignCast(c));
            out_len.* = self_p.len;
            out_epoch.* = self_p.epoch;
            if (pos >= self_p.len or tap_idx >= self_p.n_taps or dst.len < self_p.h_dim) return false;
            const off = (@as(usize, pos) * self_p.n_taps + tap_idx) * self_p.h_dim;
            @memcpy(dst[0..self_p.h_dim], self_p.data[off..][0..self_p.h_dim]);
            return true;
        }
    };
    const DualHookCtx = struct {
        tok: []const f32,
        h_dim: usize,
        v_len: usize,
        hist: *HistState,
        fn readTok(c: ?*anyopaque, token_id: u32, dst: []f32) Error!void {
            const dc: *const @This() = @ptrCast(@alignCast(c.?));
            if (token_id >= dc.v_len or dst.len < dc.h_dim) return error.TokenOutOfRange;
            @memcpy(dst[0..dc.h_dim], dc.tok[@as(usize, token_id) * dc.h_dim ..][0..dc.h_dim]);
        }
        fn readHist(c: *const anyopaque, out_len: *u32, out_epoch: *u64, pos: u32, tap_idx: usize, dst: []f32) bool {
            const dc: *const @This() = @ptrCast(@alignCast(c));
            return HistState.read(dc.hist, out_len, out_epoch, pos, tap_idx, dst);
        }
    };

    var hist_state: HistState = .{ .data = hist_f32, .len = 0, .epoch = 1, .h_dim = h, .n_taps = taps };
    var dual_ctx: DualHookCtx = .{ .tok = tok_embed, .h_dim = h, .v_len = v, .hist = &hist_state };
    registerContextHistoryProvider(&dual_ctx, DualHookCtx.readHist);
    defer unregisterContextHistoryProvider(&dual_ctx);

    var weights_hooked = weights;
    weights_hooked.tok_embed = &.{};
    weights_hooked.tok_embed_hook = .{ .callback = .{ .ctx = &dual_ctx, .read_fn = DualHookCtx.readTok } };

    for ([_]u8{ 8, 16 }) |block_w| {
        const cfg: DFlareConfig = .{
            .variant = .dflash_standalone,
            .num_layers = num_layers,
            .block_size = block_w,
            .top_k_per_pos = 1,
            .max_tree_nodes = 16,
            .hidden_size = @intCast(h),
            .intermediate_size = @intCast(inter),
            .num_attention_heads = nh,
            .num_key_value_heads = nkvh,
            .head_dim = hd,
            .rope_dim = 64,
            .rope_theta = 10_000.0,
            .attention_value_scale = 0.612,
            .sliding_window = 6,
            .vocab_size = @intCast(v),
            .markov_rank = 0,
            .num_target_taps = @intCast(taps),
            .mask_token_id = 31,
            .rms_eps = 1e-6,
        };

        var ws_cpu = try DFlareWorkspace.init(cfg, null);
        defer ws_cpu.deinit();
        var ws_gpu = try DFlareWorkspace.init(cfg, null);
        defer ws_gpu.deinit();

        var cpu_p = try DFlareProposer.init(cfg, weights_hooked, &ws_cpu);
        var gpu_p = try DFlareProposer.init(cfg, weights_hooked, &ws_gpu);
        try gpu_p.attachGpu(.{ .device = device, .library = library });
        defer gpu_p.detachGpu();

        // Advance through 3, 7, 10 context positions (exceeds sliding_window=6 -> circular SWA KV wraparound!).
        for ([_]u32{ 3, 7, 10 }) |ctx_len| {
            hist_state.len = ctx_len;
            const pos_off = (@as(usize, ctx_len) - 1) * taps * h;
            for (0..taps) |ti| tap_slices[ti] = hist_f32[pos_off + ti * h ..][0..h];

            var cpu_tree: CandidateTree = undefined;
            var gpu_tree: CandidateTree = undefined;
            const cpu_res = try cpu_p.propose(4, &tap_slices, null, &cpu_tree);
            const gpu_res = try gpu_p.propose(4, &tap_slices, null, &gpu_tree);

            try std.testing.expectEqual(@as(u16, block_w), cpu_res.tree_nodes);
            try std.testing.expectEqual(cpu_res.tree_nodes, gpu_res.tree_nodes);

            // Assert cosine similarity >= 0.9999 across ALL B block hidden vectors (CPU vs Metal 4 GPU)!
            var dot_sum: f64 = 0.0;
            var na_sum: f64 = 0.0;
            var nb_sum: f64 = 0.0;
            for (0..@as(usize, block_w) * h) |idx| {
                const ca: f64 = ws_cpu.slot_hiddens[idx];
                const gb: f64 = @floatCast(ws_gpu.gpu_slot_hiddens_f16[idx]);
                dot_sum += ca * gb;
                na_sum += ca * ca;
                nb_sum += gb * gb;
            }
            const cos_sim = dot_sum / (@sqrt(na_sum) * @sqrt(nb_sum));
            try std.testing.expect(cos_sim >= 0.9999);

            // Also verify full-vocab logit parity across all draft slots (`f32` CPU vs `f16` Metal 4 GPU).
            const draft_slots = @as(usize, block_w) - 1;
            var max_logit_diff: f32 = 0.0;
            for (0..draft_slots * v) |li_idx| {
                const c_l = ws_cpu.base_logits_buf[li_idx];
                const g_l: f32 = @floatCast(ws_gpu.gpu_base_logits_f16[li_idx]);
                max_logit_diff = @max(max_logit_diff, @abs(c_l - g_l));
            }
            try std.testing.expect(max_logit_diff < 2e-2);

            for (0..cpu_tree.count) |ni| {
                try std.testing.expectEqual(cpu_tree.nodes[ni].token_id, gpu_tree.nodes[ni].token_id);
                try std.testing.expect(@abs(cpu_tree.nodes[ni].step_prob - gpu_tree.nodes[ni].step_prob) < 1e-2);
            }
        }
    }

    // 3. Verify 2-iteration refinement + top-2 fork at uncertain slots (<= 16 tree nodes) GPU == CPU parity!
    const refine_cfg: DFlareConfig = .{
        .variant = .dflash_standalone,
        .num_layers = 2,
        .block_size = 8,
        .top_k_per_pos = 2,
        .max_tree_nodes = 16,
        .hidden_size = @intCast(h),
        .intermediate_size = @intCast(inter),
        .num_attention_heads = nh,
        .num_key_value_heads = nkvh,
        .head_dim = hd,
        .rope_dim = 64,
        .attention_value_scale = 0.612,
        .sliding_window = 6,
        .vocab_size = @intCast(v),
        .markov_rank = 0,
        .num_target_taps = @intCast(taps),
        .mask_token_id = 31,
        .refinement_iters = 2,
        .refine_confidence_threshold = 0.05,
        .fork_top2_threshold = 0.99,
    };
    var refine_weights = weights;
    refine_weights.layers = layers[0..2];
    var ws_rc = try DFlareWorkspace.init(refine_cfg, null);
    defer ws_rc.deinit();
    var ws_rg = try DFlareWorkspace.init(refine_cfg, null);
    defer ws_rg.deinit();
    var p_rc = try DFlareProposer.init(refine_cfg, refine_weights, &ws_rc);
    var p_rg = try DFlareProposer.init(refine_cfg, refine_weights, &ws_rg);
    try p_rg.attachGpu(.{ .device = device, .library = library });
    defer p_rg.detachGpu();

    var r_cpu_tree: CandidateTree = undefined;
    var r_gpu_tree: CandidateTree = undefined;
    _ = try p_rc.propose(7, &tap_slices, null, &r_cpu_tree);
    _ = try p_rg.propose(7, &tap_slices, null, &r_gpu_tree);
    try std.testing.expectEqual(@as(u64, 2), p_rc.backbone_runs);
    try std.testing.expectEqual(@as(u64, 2), p_rg.backbone_runs);
    try std.testing.expect(r_gpu_tree.count > 8 and r_gpu_tree.count <= 16);
    try std.testing.expectEqual(r_cpu_tree.count, r_gpu_tree.count);
    for (0..r_cpu_tree.count) |ni| {
        try std.testing.expectEqual(r_cpu_tree.nodes[ni].token_id, r_gpu_tree.nodes[ni].token_id);
    }

    // 4. Verify D7 cold-expert block pruning reduces unique cold experts and remaps node_log_probs.
    const d7_layers: u32 = 2;
    var node_exps: [16 * d7_layers]tree_mod.ExpertBitset = @splat(tree_mod.ExpertBitset.empty());
    var cold_masks: [d7_layers]tree_mod.ExpertBitset = @splat(tree_mod.ExpertBitset.empty());
    for (0..d7_layers) |l| {
        for (8..16) |e| cold_masks[l].set(@intCast(e));
    }
    var pre_cold: u32 = 0;
    for (0..r_gpu_tree.count) |ni| {
        const depth = r_gpu_tree.nodes[ni].depth;
        for (0..d7_layers) |l| {
            if (depth <= 2) {
                node_exps[ni * d7_layers + l].set(0); // hot
            } else {
                const cold_e: u32 = 8 + @as(u32, @intCast(ni % 8));
                node_exps[ni * d7_layers + l].set(cold_e);
            }
        }
    }
    for (0..d7_layers) |l| {
        var u_bs = tree_mod.ExpertBitset.empty();
        for (0..r_gpu_tree.count) |ni| u_bs = u_bs.unionWith(node_exps[ni * d7_layers + l]);
        pre_cold += u_bs.intersectWith(cold_masks[l]).count();
    }
    try std.testing.expect(pre_cold > 0);
    const pre_nodes = r_gpu_tree.count;
    const d7_cost = tree_mod.PruneCostModel.uniformBeta(100.0, 2.0, 300_000.0, 6_000_000.0, 60.0, d7_layers, 16);
    const ps = try p_rg.pruneWithD7(
        32,
        &r_gpu_tree,
        d7_cost,
        node_exps[0 .. @as(usize, pre_nodes) * d7_layers],
        &cold_masks,
        16,
    );
    try std.testing.expect(ps.kept_nodes < pre_nodes);
    try std.testing.expect(ps.kept_nodes >= 1);
    try std.testing.expectEqual(@as(f32, 0.0), p_rg.nodeLogProb(0));
}
