//! Metal 4 fused operator suite (`G25`, ADR-0024 & ADR-0025) and CPU FP32
//! parity reference harness for `Inkling 2` (`inkling2`, `inkling2_loop`) and
//! `Hy4` (`hy_v4`, `hy_v4_vl`, `hy_v4_flash`).
//!
//! `Inkling 2` operators:
//! - `ami_inkling2_rmsnorm_sconv4`: register-fused `RMSNorm + k=4` causal
//!   depthwise short conv (static `conv_w` and dynamic per-token `weight_proj`),
//!   with in-place 3-step circular conv ring update (`M = 1`) and causal scan
//!   (`M > 1`) with zero intermediate global-memory roundtrips.
//! - `ami_inkling2_rel_gqa_attn`: single-pass register-resident GQA / 1024-token
//!   `HybridKvRing` SWA kernel (`M = 1` decode and `M >= 2` causal/tree block
//!   verify) fusing `qk_l2_clip` (`22.627`), `log_scaling` (`1 + log(pos + 1)`),
//!   `d_rel = 16` (`rel_extent = 1024`) + `rel_l2_clip` (`8.0`), `attn_sink_proj`,
//!   attention softcap, and elementwise `gate_attn` output gating.
//! - `ami_inkling2_moe_route`, `ami_inkling2_moe_swiglu_softcap`,
//!   `ami_inkling2_moe_route_softcap_epilogue`: router top-k (`E <= 1024`,
//!   `top_k <= 16`) with `bias + route_scale` (`8.0`) + logit softcap (`4.0`),
//!   fused SwiGLU + MLP softcap (`25.0`/`30.0`), and weighted combine epilogue.
//! - `ami_inkling2_attn_res`: 6-layer `attn_res` cross-layer block residual
//!   query combination (`RMSNorm` + learned query-weighted residual aggregation).
//!
//! `Hy4` operators:
//! - `IndexCache` (`full` vs `shared` zero-recompute reuse) + `IndexPool`
//!   (`ami_hy4_index_pool_compress` + `ami_hy4_index_pool_topk` 2-stage
//!   `compress_ratio = 4` -> `group_topk` -> `candidate_topk` selector).
//! - `ami_hy4_latent_moe_route`: sigmoid router + `e_score_correction_bias`
//!   top-k selection + normalized weights * `routed_scaling_factor` (`3.998`).
//! - `ami_hy4_swiglu_clamped_linear_gate`: SwiGLU with `swiglu_limit` (`10.0`
//!   gate/up clamp) + optional elementwise `linear_gate`.
//! - `ami_hy4_combine_sandwich_norm`: `LatentMoE` `combine_output_norm` +
//!   `sandwich_norm` / `post_proj_norm` fused RMSNorm + residual epilogue.
//! - `ami_hy4_mla_sink_attn`: sparse MLA decode/verify/prefill attention over
//!   `IndexCache`/`IndexPool` selected token indices + SWA + `learnable_sink_param`.
//!
//! Zero dynamic heap allocations on the hot path; every wired region is 16 KiB
//! aligned (`APPLE_PAGE_SIZE = 16384`).

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const amf = @import("../format/amf.zig");
const quant = @import("../quant/spec.zig");
const attention = @import("attention.zig");
const hybrid = @import("hybrid.zig");

pub const APPLE_PAGE_SIZE: usize = page.APPLE_PAGE_SIZE;
pub const MAX_INKLING2_HEAD_DIM: u32 = 256;
pub const MAX_INKLING2_REL_DIM: u32 = 32;
pub const DEFAULT_REL_DIM: u32 = 16;
pub const DEFAULT_REL_EXTENT: u32 = 1024;
pub const INKLING2_SCONV_K: u32 = 4;
pub const INKLING2_SCONV_RING_SLOTS: u32 = 3;
pub const INKLING2_DEFAULT_QK_L2_CLIP: f32 = 22.627;
pub const INKLING2_DEFAULT_REL_L2_CLIP: f32 = 8.0;
pub const INKLING2_DEFAULT_ROUTE_SCALE: f32 = 8.0;
pub const INKLING2_DEFAULT_ROUTER_SOFTCAP: f32 = 4.0;
pub const INKLING2_DEFAULT_MLP_SOFTCAP: f32 = 25.0;
pub const MAX_ATTN_RES_BLOCKS: u32 = 8;
pub const INKLING2_DEFAULT_ATTN_RES_BLOCKS: u32 = 6;
pub const MAX_EXPERTS: u32 = 1024;
pub const MAX_TOP_K: u32 = 16;
pub const MAX_HY4_KV_LORA_RANK: u32 = 2048;
pub const HY4_DEFAULT_COMPRESS_RATIO: u32 = 4;
pub const HY4_DEFAULT_GROUP_TOPK: u32 = 512;
pub const HY4_DEFAULT_CANDIDATE_TOPK: u32 = 2048;
pub const HY4_DEFAULT_ROUTED_SCALING_FACTOR: f32 = 3.998;
pub const HY4_DEFAULT_SWIGLU_LIMIT: f32 = 10.0;
pub const HY4_INDEX_CACHE_PERIOD: u32 = 4;
pub const MAX_TREE_NODES: u32 = attention.MAX_TREE_NODES;

pub const ArenaError = error{
    ZeroDimension,
    InvalidConfig,
    InvalidLayerIndex,
    Overflow,
} || arena_mod.WiredArena.AllocError;

pub const EncodeError = metal.Error || error{
    ZeroDimension,
    InvalidConfig,
    InvalidHeadDim,
    InvalidNodeCount,
    InvalidLayerIndex,
    UnsupportedExperts,
    UnsupportedTopK,
    MissingInputs,
    Overflow,
};

// ===========================================================================
// Uniform Payloads (exact ABI match with `src/kernels/inkling2_hy4.metal`)
// ===========================================================================

pub const Sconv4Params = extern struct {
    m: u32,
    hidden_dim: u32,
    state_pos: u32,
    write_state: u32,
    is_dynamic: u32,
    has_bias: u32,
    apply_rms_norm: u32,
    apply_silu: u32,
    conv_w_f16: u32,
    serial_tiles: u32,
    rms_eps: f32,
    _pad0: u32 = 0,
};

pub const Inkling2RelGqaParams = extern struct {
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    window_size: u32,
    ring_capacity: u32,
    chunk_tokens: u32,
    causal_q_start_pos: u32,
    d_rel: u32,
    rel_extent: u32,
    rel_per_head: u32,
    use_log_scaling: u32,
    has_sink: u32,
    sink_per_token: u32,
    has_gate_attn: u32,
    gate_mode: u32,
    has_tree: u32,
    scale: f32,
    qk_l2_clip: f32,
    rel_l2_clip: f32,
    rel_scale: f32,
    log_scale_alpha: f32,
    attn_softcap: f32,
};

pub const Inkling2MoeRouteParams = extern struct {
    m: u32,
    num_experts: u32,
    top_k: u32,
    logits_f16: u32,
    has_bias: u32,
    bias_f16: u32,
    bias_mode: u32,
    norm_topk_prob: u32,
    route_scale: f32,
    router_softcap: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

pub const Inkling2MoeSwigluParams = extern struct {
    num_rows: u32,
    intermediate_size: u32,
    interleaved_gate_up: u32,
    softcap: f32,
};

pub const Inkling2MoeEpilogueParams = extern struct {
    m: u32,
    hidden_dim: u32,
    top_k: u32,
    has_residual: u32,
    weights_f16: u32,
    mlp_softcap: f32,
    res_scale: f32,
    _pad0: u32 = 0,
};

pub const Inkling2AttnResParams = extern struct {
    m: u32,
    hidden_dim: u32,
    num_blocks: u32,
    block_stride_elems: u32,
    apply_rms_norm: u32,
    has_norm_w: u32,
    query_per_token: u32,
    rms_eps: f32,
    scale: f32,
    normalize_values: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
    block_slots: [MAX_ATTN_RES_BLOCKS]u32 = @splat(0),
};

pub const Hy4IndexCompressParams = extern struct {
    kv_len: u32,
    idx_dim: u32,
    num_k_heads: u32,
    compress_ratio: u32,
    pool_mode: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

pub const Hy4IndexPoolTopkParams = extern struct {
    num_nodes: u32,
    num_q_heads: u32,
    num_k_heads: u32,
    idx_dim: u32,
    kv_len: u32,
    causal_q_start_pos: u32,
    compress_ratio: u32,
    group_topk: u32,
    candidate_topk: u32,
    score_mode: u32,
    has_head_weights: u32,
    swa_window: u32,
    max_groups_stride: u32,
    max_cand_stride: u32,
    scale: f32,
    _pad0: u32 = 0,
};

pub const Hy4LatentMoeRouteParams = extern struct {
    m: u32,
    num_experts: u32,
    top_k: u32,
    logits_f16: u32,
    has_bias: u32,
    bias_f16: u32,
    norm_topk_prob: u32,
    _pad0: u32 = 0,
    routed_scaling_factor: f32,
    eps: f32,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

pub const Hy4SwigluClampedParams = extern struct {
    num_rows: u32,
    intermediate_size: u32,
    interleaved_gate_up: u32,
    clamp_mode: u32,
    has_linear_gate: u32,
    linear_gate_mode: u32,
    swiglu_limit: f32,
    _pad0: u32 = 0,
};

pub const Hy4CombineSandwichParams = extern struct {
    m: u32,
    hidden_dim: u32,
    top_k: u32,
    has_shared: u32,
    has_residual: u32,
    apply_combine_norm: u32,
    apply_post_norm: u32,
    weights_f16: u32,
    rms_eps: f32,
    res_scale: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

pub const Hy4MlaSinkAttnParams = extern struct {
    num_nodes: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    qk_rope_dim: u32,
    kv_len: u32,
    causal_q_start_pos: u32,
    swa_window: u32,
    candidate_topk: u32,
    has_sink: u32,
    has_linear_gate: u32,
    linear_gate_mode: u32,
    has_tree: u32,
    scale: f32,
    attn_softcap: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

comptime {
    if (@sizeOf(Sconv4Params) != 48) @compileError("Sconv4Params must be 48 bytes");
    if (@sizeOf(Inkling2RelGqaParams) != 96) @compileError("Inkling2RelGqaParams must be 96 bytes");
    if (@sizeOf(Inkling2MoeRouteParams) != 48) @compileError("Inkling2MoeRouteParams must be 48 bytes");
    if (@sizeOf(Inkling2MoeSwigluParams) != 16) @compileError("Inkling2MoeSwigluParams must be 16 bytes");
    if (@sizeOf(Inkling2MoeEpilogueParams) != 32) @compileError("Inkling2MoeEpilogueParams must be 32 bytes");
    if (@sizeOf(Inkling2AttnResParams) != 80) @compileError("Inkling2AttnResParams must be 80 bytes");
    if (@sizeOf(Hy4IndexCompressParams) != 32) @compileError("Hy4IndexCompressParams must be 32 bytes");
    if (@sizeOf(Hy4IndexPoolTopkParams) != 64) @compileError("Hy4IndexPoolTopkParams must be 64 bytes");
    if (@sizeOf(Hy4LatentMoeRouteParams) != 48) @compileError("Hy4LatentMoeRouteParams must be 48 bytes");
    if (@sizeOf(Hy4SwigluClampedParams) != 32) @compileError("Hy4SwigluClampedParams must be 32 bytes");
    if (@sizeOf(Hy4CombineSandwichParams) != 48) @compileError("Hy4CombineSandwichParams must be 48 bytes");
    if (@sizeOf(Hy4MlaSinkAttnParams) != 64) @compileError("Hy4MlaSinkAttnParams must be 64 bytes");
}

// ===========================================================================
// Preallocated 16 KiB-Aligned Wired Rings, Arenas & Indexers
// ===========================================================================

pub const SconvRingSpec = struct {
    num_layers: u32,
    hidden_dim: u32,
};

/// 16 KiB-aligned wired 3-step circular convolution state ring per layer.
pub const SconvRingArena = struct {
    spec: SconvRingSpec,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    base_off: u64,
    layer_stride_bytes: u64,
    layer_pos: []u32,

    pub fn layerStrideBytes(hidden_dim: u32) ArenaError!u64 {
        if (hidden_dim == 0) return error.ZeroDimension;
        const elems = std.math.mul(usize, INKLING2_SCONV_RING_SLOTS, hidden_dim) catch return error.Overflow;
        const raw = std.math.mul(usize, elems, @sizeOf(f16)) catch return error.Overflow;
        return @intCast(amf.align64(raw));
    }

    pub fn requiredBytes(spec: SconvRingSpec) ArenaError!usize {
        if (spec.num_layers == 0 or spec.hidden_dim == 0) return error.ZeroDimension;
        const stride = try layerStrideBytes(spec.hidden_dim);
        const state_bytes = std.math.mul(usize, spec.num_layers, @intCast(stride)) catch return error.Overflow;
        const pos_bytes = amf.align64(std.math.mul(usize, spec.num_layers, @sizeOf(u32)) catch return error.Overflow);
        const total = std.math.add(usize, state_bytes, pos_bytes) catch return error.Overflow;
        return page.alignUp(total) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: SconvRingSpec) ArenaError!SconvRingArena {
        const total = try requiredBytes(spec);
        const stride = try layerStrideBytes(spec.hidden_dim);
        const backing = try arena.allocPages(total);
        @memset(backing, 0);

        const state_bytes = @as(usize, spec.num_layers) * @as(usize, @intCast(stride));
        const pos_end = state_bytes + @as(usize, spec.num_layers) * @sizeOf(u32);
        const layer_pos: []u32 = @alignCast(std.mem.bytesAsSlice(u32, backing[state_bytes..pos_end]));

        return .{
            .spec = spec,
            .backing = backing,
            .base_off = @intCast(arena.offsetOf(backing.ptr)),
            .layer_stride_bytes = stride,
            .layer_pos = layer_pos,
        };
    }

    pub fn bytes(self: *const SconvRingArena) []align(page.APPLE_PAGE_SIZE) u8 {
        return self.backing;
    }

    pub fn layerBinding(self: *const SconvRingArena, layer_idx: u32, buf: metal.Buffer) ArenaError!metal.BufferBinding {
        if (layer_idx >= self.spec.num_layers) return error.InvalidLayerIndex;
        return .{
            .buffer = buf,
            .offset = self.base_off + @as(u64, layer_idx) * self.layer_stride_bytes,
        };
    }

    pub fn layerSlice(self: *const SconvRingArena, layer_idx: u32) ArenaError![]f16 {
        if (layer_idx >= self.spec.num_layers) return error.InvalidLayerIndex;
        const off: usize = @intCast(@as(u64, layer_idx) * self.layer_stride_bytes);
        const len: usize = @as(usize, INKLING2_SCONV_RING_SLOTS) * self.spec.hidden_dim * @sizeOf(f16);
        return @alignCast(std.mem.bytesAsSlice(f16, self.backing[off .. off + len]));
    }

    pub fn statePos(self: *const SconvRingArena, layer_idx: u32) ArenaError!u32 {
        if (layer_idx >= self.spec.num_layers) return error.InvalidLayerIndex;
        return self.layer_pos[layer_idx];
    }

    pub fn advanceStatePos(self: *SconvRingArena, layer_idx: u32, tokens: u32) ArenaError!void {
        if (layer_idx >= self.spec.num_layers) return error.InvalidLayerIndex;
        self.layer_pos[layer_idx] = (self.layer_pos[layer_idx] + tokens) % INKLING2_SCONV_RING_SLOTS;
    }
};

pub const AttnResRingSpec = struct {
    max_blocks: u32 = INKLING2_DEFAULT_ATTN_RES_BLOCKS,
    max_tokens: u32,
    hidden_dim: u32,
};

/// 16 KiB-aligned circular cross-layer block residual ring (`<= 8` blocks, default 6).
pub const AttnResRingArena = struct {
    spec: AttnResRingSpec,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    base_off: u64,
    block_stride_elems: u32,
    count: u32,
    head: u32,

    pub fn validateSpec(spec: AttnResRingSpec) ArenaError!void {
        if (spec.max_blocks == 0 or spec.max_tokens == 0 or spec.hidden_dim == 0) return error.ZeroDimension;
        if (spec.max_blocks > MAX_ATTN_RES_BLOCKS) return error.InvalidConfig;
    }

    pub fn requiredBytes(spec: AttnResRingSpec) ArenaError!usize {
        try validateSpec(spec);
        const stride_elems = std.math.mul(usize, spec.max_tokens, spec.hidden_dim) catch return error.Overflow;
        const total_elems = std.math.mul(usize, spec.max_blocks, stride_elems) catch return error.Overflow;
        const raw = std.math.mul(usize, total_elems, @sizeOf(f16)) catch return error.Overflow;
        return page.alignUp(raw) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: AttnResRingSpec) ArenaError!AttnResRingArena {
        const total = try requiredBytes(spec);
        const stride_elems = std.math.mul(u32, spec.max_tokens, spec.hidden_dim) catch return error.Overflow;
        const backing = try arena.allocPages(total);
        @memset(backing, 0);
        return .{
            .spec = spec,
            .backing = backing,
            .base_off = @intCast(arena.offsetOf(backing.ptr)),
            .block_stride_elems = stride_elems,
            .count = 0,
            .head = 0,
        };
    }

    pub fn bytes(self: *const AttnResRingArena) []align(page.APPLE_PAGE_SIZE) u8 {
        return self.backing;
    }

    pub fn binding(self: *const AttnResRingArena, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.base_off };
    }

    pub fn blockSlice(self: *const AttnResRingArena, phys_slot: u32) ArenaError![]f16 {
        if (phys_slot >= self.spec.max_blocks) return error.InvalidLayerIndex;
        const start_elems = @as(usize, phys_slot) * self.block_stride_elems;
        const byte_start = start_elems * @sizeOf(f16);
        const byte_len = @as(usize, self.block_stride_elems) * @sizeOf(f16);
        return @alignCast(std.mem.bytesAsSlice(f16, self.backing[byte_start .. byte_start + byte_len]));
    }

    /// Reserves the next physical slot in circular order `[0..max_blocks)` and advances ring state.
    pub fn pushSlot(self: *AttnResRingArena) u32 {
        const slot = self.head;
        self.head = (self.head + 1) % self.spec.max_blocks;
        if (self.count < self.spec.max_blocks) {
            self.count += 1;
        }
        return slot;
    }

    /// Returns active physical slots ordered oldest -> newest in `out_slots[0..num_active]`.
    pub fn activeSlots(self: *const AttnResRingArena, out_slots: *[MAX_ATTN_RES_BLOCKS]u32) u32 {
        out_slots.* = @splat(0);
        if (self.count == 0) return 0;
        const oldest = if (self.count < self.spec.max_blocks) 0 else self.head;
        for (0..self.count) |i| {
            out_slots[i] = (oldest + @as(u32, @intCast(i))) % self.spec.max_blocks;
        }
        return self.count;
    }
};

/// Hy4 `IndexCache` layer mode (`full` runs indexer and updates cache; `shared`
/// skips indexer completely and reuses cached indices from prior `full` layer).
pub const IndexCacheMode = enum(u8) {
    full = 0,
    shared = 1,

    pub fn forPeriod4(layer_idx: u32) IndexCacheMode {
        return if ((layer_idx % HY4_INDEX_CACHE_PERIOD) == 0) .full else .shared;
    }
};

pub const IndexPoolMode = enum(u32) {
    mean = 0,
    max = 1,
};

pub const IndexScoreMode = enum(u32) {
    relu_weighted_sum = 0,
    dot_sum_scaled = 1,
    dot_max_scaled = 2,
};

pub const GateActivationMode = enum(u32) {
    sigmoid = 0,
    silu = 1,
    raw_mul = 2,
};

pub const SwigluClampMode = enum(u32) {
    gate_and_up = 0,
    product = 1,
    both = 2,
};

pub const IndexPoolSpec = struct {
    max_nodes: u32,
    max_kv_len: u32,
    num_k_heads: u32 = 1,
    idx_dim: u32,
    compress_ratio: u32 = HY4_DEFAULT_COMPRESS_RATIO,
    group_topk: u32 = HY4_DEFAULT_GROUP_TOPK,
    candidate_topk: u32 = HY4_DEFAULT_CANDIDATE_TOPK,
};

/// 16 KiB-aligned wired `IndexPool` + `IndexCache` buffer holding compressed
/// key blocks, radix-select scratch, and cached `[max_nodes, candidate_topk]`
/// token indices shared across `.shared` Hy4 layers.
pub const IndexPoolBuffer = struct {
    spec: IndexPoolSpec,
    max_groups: u32,
    max_cand: u32,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    base_off: u64,
    k_pool_off: u64,
    scratch_group_scores_off: u64,
    scratch_cand_scores_off: u64,
    scratch_sel_groups_off: u64,
    scratch_cand_ids_off: u64,
    selected_tokens_off: u64,
    selected_counts_off: u64,
    k_pool: []f16,
    selected_tokens: []u32,
    selected_counts: []u32,

    const Layout = struct {
        max_groups: u32,
        max_cand: u32,
        k_pool_bytes: usize,
        g_scores_bytes: usize,
        c_scores_bytes: usize,
        sel_groups_bytes: usize,
        cand_ids_bytes: usize,
        sel_tokens_bytes: usize,
        sel_counts_bytes: usize,
        total_aligned: usize,
    };

    fn computeLayout(spec: IndexPoolSpec) ArenaError!Layout {
        if (spec.max_nodes == 0 or spec.max_kv_len == 0 or spec.num_k_heads == 0 or
            spec.idx_dim == 0 or spec.compress_ratio == 0 or spec.candidate_topk == 0)
        {
            return error.ZeroDimension;
        }
        const max_groups = (spec.max_kv_len + spec.compress_ratio - 1) / spec.compress_ratio;
        const max_cand_from_groups = if (spec.group_topk > 0)
            (std.math.mul(u32, spec.group_topk, spec.compress_ratio) catch return error.Overflow)
        else
            spec.max_kv_len;
        const max_cand = @min(spec.max_kv_len, @max(max_cand_from_groups, spec.candidate_topk));

        const k_stride = std.math.mul(usize, spec.num_k_heads, spec.idx_dim) catch return error.Overflow;
        const k_pool_elems = std.math.mul(usize, max_groups, k_stride) catch return error.Overflow;
        const k_pool_bytes = amf.align64(std.math.mul(usize, k_pool_elems, @sizeOf(f16)) catch return error.Overflow);

        const g_elems = std.math.mul(usize, spec.max_nodes, max_groups) catch return error.Overflow;
        const g_scores_bytes = amf.align64(std.math.mul(usize, g_elems, @sizeOf(f32)) catch return error.Overflow);

        const c_elems = std.math.mul(usize, spec.max_nodes, max_cand) catch return error.Overflow;
        const c_scores_bytes = amf.align64(std.math.mul(usize, c_elems, @sizeOf(f32)) catch return error.Overflow);
        const cand_ids_bytes = amf.align64(std.math.mul(usize, c_elems, @sizeOf(u32)) catch return error.Overflow);

        const sg_elems = std.math.mul(usize, spec.max_nodes, @max(spec.group_topk, 1)) catch return error.Overflow;
        const sel_groups_bytes = amf.align64(std.math.mul(usize, sg_elems, @sizeOf(u32)) catch return error.Overflow);

        const st_elems = std.math.mul(usize, spec.max_nodes, spec.candidate_topk) catch return error.Overflow;
        const sel_tokens_bytes = amf.align64(std.math.mul(usize, st_elems, @sizeOf(u32)) catch return error.Overflow);
        const sel_counts_bytes = amf.align64(std.math.mul(usize, spec.max_nodes, @sizeOf(u32)) catch return error.Overflow);

        var total: usize = 0;
        for ([_]usize{
            k_pool_bytes,
            g_scores_bytes,
            c_scores_bytes,
            sel_groups_bytes,
            cand_ids_bytes,
            sel_tokens_bytes,
            sel_counts_bytes,
        }) |sz| {
            total = std.math.add(usize, total, sz) catch return error.Overflow;
        }
        return .{
            .max_groups = max_groups,
            .max_cand = max_cand,
            .k_pool_bytes = k_pool_bytes,
            .g_scores_bytes = g_scores_bytes,
            .c_scores_bytes = c_scores_bytes,
            .sel_groups_bytes = sel_groups_bytes,
            .cand_ids_bytes = cand_ids_bytes,
            .sel_tokens_bytes = sel_tokens_bytes,
            .sel_counts_bytes = sel_counts_bytes,
            .total_aligned = page.alignUp(total) catch return error.Overflow,
        };
    }

    pub fn requiredBytes(spec: IndexPoolSpec) ArenaError!usize {
        return (try computeLayout(spec)).total_aligned;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: IndexPoolSpec) ArenaError!IndexPoolBuffer {
        const l = try computeLayout(spec);
        const backing = try arena.allocPages(l.total_aligned);
        @memset(backing, 0);
        const base: u64 = @intCast(arena.offsetOf(backing.ptr));

        var cur: usize = 0;
        const k_pool_off = base + cur;
        const k_pool_len = @as(usize, l.max_groups) * spec.num_k_heads * spec.idx_dim * @sizeOf(f16);
        const k_pool: []f16 = @alignCast(std.mem.bytesAsSlice(f16, backing[cur .. cur + k_pool_len]));
        cur += l.k_pool_bytes;

        const g_scores_off = base + cur;
        cur += l.g_scores_bytes;

        const c_scores_off = base + cur;
        cur += l.c_scores_bytes;

        const sel_groups_off = base + cur;
        cur += l.sel_groups_bytes;

        const cand_ids_off = base + cur;
        cur += l.cand_ids_bytes;

        const sel_tokens_off = base + cur;
        const sel_tokens_len = @as(usize, spec.max_nodes) * spec.candidate_topk * @sizeOf(u32);
        const selected_tokens: []u32 = @alignCast(std.mem.bytesAsSlice(u32, backing[cur .. cur + sel_tokens_len]));
        cur += l.sel_tokens_bytes;

        const sel_counts_off = base + cur;
        const sel_counts_len = @as(usize, spec.max_nodes) * @sizeOf(u32);
        const selected_counts: []u32 = @alignCast(std.mem.bytesAsSlice(u32, backing[cur .. cur + sel_counts_len]));

        return .{
            .spec = spec,
            .max_groups = l.max_groups,
            .max_cand = l.max_cand,
            .backing = backing,
            .base_off = base,
            .k_pool_off = k_pool_off,
            .scratch_group_scores_off = g_scores_off,
            .scratch_cand_scores_off = c_scores_off,
            .scratch_sel_groups_off = sel_groups_off,
            .scratch_cand_ids_off = cand_ids_off,
            .selected_tokens_off = sel_tokens_off,
            .selected_counts_off = sel_counts_off,
            .k_pool = k_pool,
            .selected_tokens = selected_tokens,
            .selected_counts = selected_counts,
        };
    }

    pub fn bytes(self: *const IndexPoolBuffer) []align(page.APPLE_PAGE_SIZE) u8 {
        return self.backing;
    }

    pub fn kPoolBinding(self: *const IndexPoolBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.k_pool_off };
    }

    pub fn scratchGroupScoresBinding(self: *const IndexPoolBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.scratch_group_scores_off };
    }

    pub fn scratchCandScoresBinding(self: *const IndexPoolBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.scratch_cand_scores_off };
    }

    pub fn scratchSelGroupsBinding(self: *const IndexPoolBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.scratch_sel_groups_off };
    }

    pub fn scratchCandIdsBinding(self: *const IndexPoolBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.scratch_cand_ids_off };
    }

    pub fn selectedTokensBinding(self: *const IndexPoolBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.selected_tokens_off };
    }

    pub fn selectedCountsBinding(self: *const IndexPoolBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.selected_counts_off };
    }
};

// ===========================================================================
// Typed Kernel Arguments & Pipelines
// ===========================================================================

pub const RmsNormSconv4Args = struct {
    x: metal.BufferBinding,
    norm_w: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .f16,
    conv_w: ?metal.BufferBinding = null,
    conv_w_f16: bool = false,
    dyn_w: ?metal.BufferBinding = null,
    conv_bias: ?metal.BufferBinding = null,
    conv_state_in: metal.BufferBinding,
    conv_state_out: metal.BufferBinding,
    out: metal.BufferBinding,
    m: u32,
    hidden_dim: u32,
    state_pos: u32 = 0,
    write_state: bool = true,
    is_dynamic: bool = false,
    apply_rms_norm: bool = true,
    apply_silu: bool = false,
    rms_eps: f32 = 1e-6,
};

pub const Inkling2RelGqaAttnArgs = struct {
    q: metal.BufferBinding,
    q_rel: ?metal.BufferBinding = null,
    rel_emb: ?metal.BufferBinding = null,
    kv_ring: *const hybrid.HybridKvRing,
    kv_ring_buf: metal.Buffer,
    k_tree: ?metal.BufferBinding = null,
    v_tree: ?metal.BufferBinding = null,
    ancestor_masks: ?metal.BufferBinding = null,
    sink_logits: ?metal.BufferBinding = null,
    sink_per_token: bool = false,
    gate_attn: ?metal.BufferBinding = null,
    gate_mode: GateActivationMode = .sigmoid,
    out: metal.BufferBinding,
    num_nodes: u32,
    num_q_heads: u32,
    kv_len: u32,
    causal_q_start_pos: ?u32 = null,
    d_rel: u32 = DEFAULT_REL_DIM,
    rel_extent: u32 = DEFAULT_REL_EXTENT,
    rel_per_head: bool = false,
    use_log_scaling: bool = false,
    log_scale_alpha: f32 = 1.0,
    scale: ?f32 = null,
    qk_l2_clip: f32 = INKLING2_DEFAULT_QK_L2_CLIP,
    rel_l2_clip: f32 = INKLING2_DEFAULT_REL_L2_CLIP,
    rel_scale: f32 = 1.0,
    attn_softcap: f32 = 0.0,
};

pub const Inkling2MoeRouteArgs = struct {
    logits: metal.BufferBinding,
    logits_f16: bool = false,
    bias: ?metal.BufferBinding = null,
    bias_f16: bool = false,
    bias_add_to_logits: bool = true,
    topk_experts: metal.BufferBinding,
    topk_weights: metal.BufferBinding,
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: bool = true,
    route_scale: f32 = INKLING2_DEFAULT_ROUTE_SCALE,
    router_softcap: f32 = INKLING2_DEFAULT_ROUTER_SOFTCAP,
};

pub const Inkling2MoeSwigluArgs = struct {
    gate_in: metal.BufferBinding,
    up_in: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    num_rows: u32,
    intermediate_size: u32,
    interleaved_gate_up: bool = false,
    softcap: f32 = INKLING2_DEFAULT_MLP_SOFTCAP,
};

pub const Inkling2MoeEpilogueArgs = struct {
    expert_out: metal.BufferBinding,
    topk_weights: metal.BufferBinding,
    weights_f16: bool = false,
    residual: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    m: u32,
    hidden_dim: u32,
    top_k: u32,
    mlp_softcap: f32 = INKLING2_DEFAULT_MLP_SOFTCAP,
    res_scale: f32 = 1.0,
};

pub const Inkling2AttnResArgs = struct {
    ring: *const AttnResRingArena,
    ring_buf: metal.Buffer,
    query: metal.BufferBinding,
    norm_w: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .f16,
    out: metal.BufferBinding,
    m: u32,
    num_blocks_override: ?u32 = null,
    apply_rms_norm: bool = true,
    normalize_values: bool = false,
    query_per_token: bool = false,
    rms_eps: f32 = 1e-6,
    scale: f32 = 1.0,
};

pub const Hy4IndexCompressArgs = struct {
    k_fine: metal.BufferBinding,
    index_pool: *const IndexPoolBuffer,
    index_metal_buf: metal.Buffer,
    kv_len: u32,
    pool_mode: IndexPoolMode = .mean,
};

pub const Hy4IndexPoolTopkArgs = struct {
    q_idx: metal.BufferBinding,
    head_weights: ?metal.BufferBinding = null,
    k_fine: metal.BufferBinding,
    index_pool: *const IndexPoolBuffer,
    index_metal_buf: metal.Buffer,
    num_nodes: u32,
    num_q_heads: u32,
    kv_len: u32,
    causal_q_start_pos: ?u32 = null,
    score_mode: IndexScoreMode = .relu_weighted_sum,
    swa_window: u32 = 0,
    scale: ?f32 = null,
};

pub const Hy4IndexLayerArgs = struct {
    mode: IndexCacheMode,
    compress_keys: bool = true,
    pool_mode: IndexPoolMode = .mean,
    q_idx: metal.BufferBinding,
    head_weights: ?metal.BufferBinding = null,
    k_fine: metal.BufferBinding,
    index_pool: *const IndexPoolBuffer,
    index_metal_buf: metal.Buffer,
    num_nodes: u32,
    num_q_heads: u32,
    kv_len: u32,
    causal_q_start_pos: ?u32 = null,
    score_mode: IndexScoreMode = .relu_weighted_sum,
    swa_window: u32 = 0,
    scale: ?f32 = null,
};

pub const Hy4IndexStepDispatches = struct {
    compress_dispatches: u32 = 0,
    topk_dispatches: u32 = 0,

    pub fn total(self: Hy4IndexStepDispatches) u32 {
        return self.compress_dispatches + self.topk_dispatches;
    }
};

pub const Hy4LatentMoeRouteArgs = struct {
    logits: metal.BufferBinding,
    logits_f16: bool = false,
    e_score_correction_bias: ?metal.BufferBinding = null,
    bias_f16: bool = false,
    topk_experts: metal.BufferBinding,
    topk_weights: metal.BufferBinding,
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: bool = true,
    routed_scaling_factor: f32 = HY4_DEFAULT_ROUTED_SCALING_FACTOR,
    eps: f32 = 1e-20,
};

pub const Hy4SwigluClampedArgs = struct {
    gate_in: metal.BufferBinding,
    up_in: ?metal.BufferBinding = null,
    linear_gate: ?metal.BufferBinding = null,
    linear_gate_mode: GateActivationMode = .sigmoid,
    out: metal.BufferBinding,
    num_rows: u32,
    intermediate_size: u32,
    interleaved_gate_up: bool = false,
    clamp_mode: SwigluClampMode = .gate_and_up,
    swiglu_limit: f32 = HY4_DEFAULT_SWIGLU_LIMIT,
};

pub const Hy4CombineSandwichArgs = struct {
    expert_or_x: metal.BufferBinding,
    topk_weights: ?metal.BufferBinding = null,
    weights_f16: bool = false,
    shared_out: ?metal.BufferBinding = null,
    residual: ?metal.BufferBinding = null,
    combine_norm_w: ?metal.BufferBinding = null,
    post_norm_w: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .f16,
    out: metal.BufferBinding,
    m: u32,
    hidden_dim: u32,
    top_k: u32 = 0,
    rms_eps: f32 = 1e-5,
    res_scale: f32 = 1.0,
};

pub const Hy4MlaSinkAttnArgs = struct {
    q_nope: metal.BufferBinding,
    q_pe: ?metal.BufferBinding = null,
    kv_latent: metal.BufferBinding,
    kv_rope: ?metal.BufferBinding = null,
    index_pool: *const IndexPoolBuffer,
    index_metal_buf: metal.Buffer,
    kv_tree_latent: ?metal.BufferBinding = null,
    kv_tree_rope: ?metal.BufferBinding = null,
    ancestor_masks: ?metal.BufferBinding = null,
    sink_params: ?metal.BufferBinding = null,
    linear_gate: ?metal.BufferBinding = null,
    linear_gate_mode: GateActivationMode = .sigmoid,
    out: metal.BufferBinding,
    num_nodes: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    qk_rope_dim: u32 = 0,
    kv_len: u32,
    causal_q_start_pos: ?u32 = null,
    swa_window: u32 = 0,
    scale: ?f32 = null,
    attn_softcap: f32 = 0.0,
};

fn sameBinding(a: metal.BufferBinding, b: metal.BufferBinding) bool {
    return a.buffer.handle == b.buffer.handle and a.offset == b.offset;
}

pub const Inkling2Hy4Pipelines = struct {
    rmsnorm_sconv4: [2]metal.Pipeline,
    rel_gqa_attn_d64: metal.Pipeline,
    rel_gqa_attn_d128: metal.Pipeline,
    rel_gqa_attn_dyn: metal.Pipeline,
    inkling2_moe_route: metal.Pipeline,
    inkling2_moe_swiglu_softcap: metal.Pipeline,
    inkling2_moe_route_softcap_epilogue: metal.Pipeline,
    inkling2_attn_res: [2]metal.Pipeline,
    hy4_index_pool_compress: metal.Pipeline,
    hy4_index_pool_topk: metal.Pipeline,
    hy4_latent_moe_route: metal.Pipeline,
    hy4_swiglu_clamped_linear_gate: metal.Pipeline,
    hy4_combine_sandwich_norm: [2]metal.Pipeline,
    hy4_mla_sink_attn_r512: metal.Pipeline,
    hy4_mla_sink_attn_dyn: metal.Pipeline,

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!Inkling2Hy4Pipelines {
        const sconv_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_rmsnorm_sconv4", &.{
            .{ .index = 0, .value = .{ .bool = false } },
        });
        errdefer sconv_f16.close();

        const sconv_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_rmsnorm_sconv4", &.{
            .{ .index = 0, .value = .{ .bool = true } },
        });
        errdefer sconv_bf16.close();

        const rel_d64 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_rel_gqa_attn", &.{
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer rel_d64.close();

        const rel_d128 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_rel_gqa_attn", &.{
            .{ .index = 1, .value = .{ .u32 = 128 } },
        });
        errdefer rel_d128.close();

        const rel_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_rel_gqa_attn", &.{
            .{ .index = 1, .value = .{ .u32 = 0 } },
        });
        errdefer rel_dyn.close();

        const ink_route = try metal.Pipeline.init(device, library, "ami_inkling2_moe_route");
        errdefer ink_route.close();

        const ink_swiglu = try metal.Pipeline.init(device, library, "ami_inkling2_moe_swiglu_softcap");
        errdefer ink_swiglu.close();

        const ink_epilogue = try metal.Pipeline.init(device, library, "ami_inkling2_moe_route_softcap_epilogue");
        errdefer ink_epilogue.close();

        const attn_res_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_attn_res", &.{
            .{ .index = 0, .value = .{ .bool = false } },
        });
        errdefer attn_res_f16.close();

        const attn_res_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_attn_res", &.{
            .{ .index = 0, .value = .{ .bool = true } },
        });
        errdefer attn_res_bf16.close();

        const idx_compress = try metal.Pipeline.init(device, library, "ami_hy4_index_pool_compress");
        errdefer idx_compress.close();

        const idx_topk = try metal.Pipeline.init(device, library, "ami_hy4_index_pool_topk");
        errdefer idx_topk.close();

        const hy4_route = try metal.Pipeline.init(device, library, "ami_hy4_latent_moe_route");
        errdefer hy4_route.close();

        const hy4_swiglu = try metal.Pipeline.init(device, library, "ami_hy4_swiglu_clamped_linear_gate");
        errdefer hy4_swiglu.close();

        const hy4_comb_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_hy4_combine_sandwich_norm", &.{
            .{ .index = 0, .value = .{ .bool = false } },
        });
        errdefer hy4_comb_f16.close();

        const hy4_comb_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_hy4_combine_sandwich_norm", &.{
            .{ .index = 0, .value = .{ .bool = true } },
        });
        errdefer hy4_comb_bf16.close();

        const hy4_mla_512 = try metal.Pipeline.initSpecialized(device, library, "ami_hy4_mla_sink_attn", &.{
            .{ .index = 2, .value = .{ .u32 = 512 } },
        });
        errdefer hy4_mla_512.close();

        const hy4_mla_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_hy4_mla_sink_attn", &.{
            .{ .index = 2, .value = .{ .u32 = 0 } },
        });
        errdefer hy4_mla_dyn.close();

        return .{
            .rmsnorm_sconv4 = .{ sconv_f16, sconv_bf16 },
            .rel_gqa_attn_d64 = rel_d64,
            .rel_gqa_attn_d128 = rel_d128,
            .rel_gqa_attn_dyn = rel_dyn,
            .inkling2_moe_route = ink_route,
            .inkling2_moe_swiglu_softcap = ink_swiglu,
            .inkling2_moe_route_softcap_epilogue = ink_epilogue,
            .inkling2_attn_res = .{ attn_res_f16, attn_res_bf16 },
            .hy4_index_pool_compress = idx_compress,
            .hy4_index_pool_topk = idx_topk,
            .hy4_latent_moe_route = hy4_route,
            .hy4_swiglu_clamped_linear_gate = hy4_swiglu,
            .hy4_combine_sandwich_norm = .{ hy4_comb_f16, hy4_comb_bf16 },
            .hy4_mla_sink_attn_r512 = hy4_mla_512,
            .hy4_mla_sink_attn_dyn = hy4_mla_dyn,
        };
    }

    pub fn close(self: Inkling2Hy4Pipelines) void {
        self.hy4_mla_sink_attn_dyn.close();
        self.hy4_mla_sink_attn_r512.close();
        self.hy4_combine_sandwich_norm[1].close();
        self.hy4_combine_sandwich_norm[0].close();
        self.hy4_swiglu_clamped_linear_gate.close();
        self.hy4_latent_moe_route.close();
        self.hy4_index_pool_topk.close();
        self.hy4_index_pool_compress.close();
        self.inkling2_attn_res[1].close();
        self.inkling2_attn_res[0].close();
        self.inkling2_moe_route_softcap_epilogue.close();
        self.inkling2_moe_swiglu_softcap.close();
        self.inkling2_moe_route.close();
        self.rel_gqa_attn_dyn.close();
        self.rel_gqa_attn_d128.close();
        self.rel_gqa_attn_d64.close();
        self.rmsnorm_sconv4[1].close();
        self.rmsnorm_sconv4[0].close();
    }

    pub fn encodeRmsNormSconv4(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: RmsNormSconv4Args,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_dim == 0) return error.ZeroDimension;
        if (args.apply_rms_norm and args.norm_w == null) return error.MissingInputs;
        if (!args.is_dynamic and args.conv_w == null) return error.MissingInputs;
        if (args.is_dynamic and args.dyn_w == null) return error.MissingInputs;

        const in_place_multi_tile = args.m > 64 and args.write_state and sameBinding(args.conv_state_in, args.conv_state_out);
        const params: Sconv4Params = .{
            .m = args.m,
            .hidden_dim = args.hidden_dim,
            .state_pos = args.state_pos % INKLING2_SCONV_RING_SLOTS,
            .write_state = if (args.write_state) 1 else 0,
            .is_dynamic = if (args.is_dynamic) 1 else 0,
            .has_bias = if (args.conv_bias != null) 1 else 0,
            .apply_rms_norm = if (args.apply_rms_norm) 1 else 0,
            .apply_silu = if (args.apply_silu) 1 else 0,
            .conv_w_f16 = if (args.conv_w_f16) 1 else 0,
            .serial_tiles = if (in_place_multi_tile) 1 else 0,
            .rms_eps = args.rms_eps,
        };
        const groups_x: u64 = if (in_place_multi_tile) 1 else (@as(u64, args.m) + 63) / 64;
        const pipe = self.rmsnorm_sconv4[@backingInt(args.norm_dtype)];
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.x,
                args.norm_w orelse args.x,
                args.conv_w orelse args.x,
                args.dyn_w orelse args.x,
                args.conv_bias orelse args.x,
                args.conv_state_in,
                args.conv_state_out,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ groups_x, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeInkling2RelGqaAttn(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2RelGqaAttnArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const has_any_tree = (args.k_tree != null or args.v_tree != null or args.ancestor_masks != null);
        if (has_any_tree and (args.k_tree == null or args.v_tree == null or args.ancestor_masks == null)) {
            return error.MissingInputs;
        }
        if (has_any_tree and args.num_nodes > MAX_TREE_NODES) return error.InvalidNodeCount;
        if (args.num_nodes == 0 or args.num_q_heads == 0) return error.ZeroDimension;
        if ((args.num_q_heads % args.kv_ring.num_kv_heads) != 0) return error.InvalidConfig;
        if (args.kv_ring.head_dim == 0 or args.kv_ring.head_dim > MAX_INKLING2_HEAD_DIM or (args.kv_ring.head_dim % 4) != 0) {
            return error.InvalidHeadDim;
        }
        if (args.d_rel > MAX_INKLING2_REL_DIM) return error.InvalidConfig;
        if (args.d_rel > 0 and (args.q_rel == null or args.rel_emb == null or args.rel_extent == 0)) {
            return error.MissingInputs;
        }

        const scale = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.kv_ring.head_dim))));
        const params: Inkling2RelGqaParams = .{
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.kv_ring.num_kv_heads,
            .head_dim = args.kv_ring.head_dim,
            .kv_len = args.kv_len,
            .window_size = args.kv_ring.window_size,
            .ring_capacity = args.kv_ring.ring_capacity,
            .chunk_tokens = args.kv_ring.chunk_tokens,
            .causal_q_start_pos = args.causal_q_start_pos orelse 0xFFFF_FFFF,
            .d_rel = args.d_rel,
            .rel_extent = args.rel_extent,
            .rel_per_head = if (args.rel_per_head) 1 else 0,
            .use_log_scaling = if (args.use_log_scaling) 1 else 0,
            .has_sink = if (args.sink_logits != null) 1 else 0,
            .sink_per_token = if (args.sink_per_token) 1 else 0,
            .has_gate_attn = if (args.gate_attn != null) 1 else 0,
            .gate_mode = @backingInt(args.gate_mode),
            .has_tree = if (has_any_tree) 1 else 0,
            .scale = scale,
            .qk_l2_clip = args.qk_l2_clip,
            .rel_l2_clip = args.rel_l2_clip,
            .rel_scale = args.rel_scale,
            .log_scale_alpha = args.log_scale_alpha,
            .attn_softcap = args.attn_softcap,
        };
        const gqa_ratio = args.num_q_heads / args.kv_ring.num_kv_heads;
        const q_groups_per_kv: u64 = (@as(u64, gqa_ratio) + 7) / 8;
        const pipe = switch (args.kv_ring.head_dim) {
            64 => self.rel_gqa_attn_d64,
            128 => self.rel_gqa_attn_d128,
            else => self.rel_gqa_attn_dyn,
        };
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.q,
                args.q_rel orelse args.q,
                args.rel_emb orelse args.q,
                args.kv_ring.tileTableBinding(args.kv_ring_buf),
                args.k_tree orelse args.q,
                args.v_tree orelse args.q,
                args.ancestor_masks orelse args.q,
                args.sink_logits orelse args.q,
                args.gate_attn orelse args.q,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ @as(u64, args.kv_ring.num_kv_heads) * q_groups_per_kv, args.num_nodes, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeInkling2MoeRoute(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2MoeRouteArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0) return error.ZeroDimension;
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExperts;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;

        const params: Inkling2MoeRouteParams = .{
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .logits_f16 = if (args.logits_f16) 1 else 0,
            .has_bias = if (args.bias != null) 1 else 0,
            .bias_f16 = if (args.bias_f16) 1 else 0,
            .bias_mode = if (args.bias_add_to_logits) 0 else 1,
            .norm_topk_prob = if (args.norm_topk_prob) 1 else 0,
            .route_scale = args.route_scale,
            .router_softcap = args.router_softcap,
        };
        try frame.encodeThreadgroups(
            self.inkling2_moe_route,
            &.{
                args.logits,
                args.bias orelse args.logits,
                args.topk_experts,
                args.topk_weights,
            },
            std.mem.asBytes(&params),
            .{ args.m, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeInkling2MoeSwigluSoftcap(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2MoeSwigluArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_rows == 0 or args.intermediate_size == 0) return error.ZeroDimension;
        if (!args.interleaved_gate_up and args.up_in == null) return error.MissingInputs;

        const params: Inkling2MoeSwigluParams = .{
            .num_rows = args.num_rows,
            .intermediate_size = args.intermediate_size,
            .interleaved_gate_up = if (args.interleaved_gate_up) 1 else 0,
            .softcap = args.softcap,
        };
        const tg_x: u64 = @min(@as(u64, args.intermediate_size), 256);
        try frame.encodeThreads(
            self.inkling2_moe_swiglu_softcap,
            &.{
                args.gate_in,
                args.up_in orelse args.gate_in,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.intermediate_size, args.num_rows, 1 },
            .{ tg_x, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeInkling2MoeEpilogue(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2MoeEpilogueArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_dim == 0 or args.top_k == 0) return error.ZeroDimension;
        if (args.top_k > MAX_TOP_K) return error.UnsupportedTopK;

        const params: Inkling2MoeEpilogueParams = .{
            .m = args.m,
            .hidden_dim = args.hidden_dim,
            .top_k = args.top_k,
            .has_residual = if (args.residual != null) 1 else 0,
            .weights_f16 = if (args.weights_f16) 1 else 0,
            .mlp_softcap = args.mlp_softcap,
            .res_scale = args.res_scale,
        };
        const tg_x: u64 = @min(@as(u64, args.hidden_dim), 256);
        try frame.encodeThreads(
            self.inkling2_moe_route_softcap_epilogue,
            &.{
                args.expert_out,
                args.topk_weights,
                args.residual orelse args.expert_out,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.hidden_dim, args.m, 1 },
            .{ tg_x, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeInkling2AttnRes(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2AttnResArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.m > args.ring.spec.max_tokens) return error.InvalidNodeCount;
        var slots: [MAX_ATTN_RES_BLOCKS]u32 = @splat(0);
        const active_cnt = args.ring.activeSlots(&slots);
        const num_blocks = args.num_blocks_override orelse active_cnt;
        if (num_blocks == 0 or num_blocks > active_cnt or num_blocks > MAX_ATTN_RES_BLOCKS) {
            return error.InvalidConfig;
        }

        const params: Inkling2AttnResParams = .{
            .m = args.m,
            .hidden_dim = args.ring.spec.hidden_dim,
            .num_blocks = num_blocks,
            .block_stride_elems = args.ring.block_stride_elems,
            .apply_rms_norm = if (args.apply_rms_norm) 1 else 0,
            .has_norm_w = if (args.norm_w != null) 1 else 0,
            .query_per_token = if (args.query_per_token) 1 else 0,
            .rms_eps = args.rms_eps,
            .scale = args.scale,
            .normalize_values = if (args.normalize_values) 1 else 0,
            .block_slots = slots,
        };
        const pipe = self.inkling2_attn_res[@backingInt(args.norm_dtype)];
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.ring.binding(args.ring_buf),
                args.query,
                args.norm_w orelse args.query,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.m, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeHy4IndexCompress(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Hy4IndexCompressArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.kv_len == 0) return error.ZeroDimension;
        if (args.kv_len > args.index_pool.spec.max_kv_len) return error.InvalidConfig;

        const spec = args.index_pool.spec;
        const num_groups: u64 = (@as(u64, args.kv_len) + spec.compress_ratio - 1) / spec.compress_ratio;
        const k_stride: u64 = @as(u64, spec.num_k_heads) * spec.idx_dim;
        const params: Hy4IndexCompressParams = .{
            .kv_len = args.kv_len,
            .idx_dim = spec.idx_dim,
            .num_k_heads = spec.num_k_heads,
            .compress_ratio = spec.compress_ratio,
            .pool_mode = @backingInt(args.pool_mode),
        };
        try frame.encodeThreads(
            self.hy4_index_pool_compress,
            &.{
                args.k_fine,
                args.index_pool.kPoolBinding(args.index_metal_buf),
            },
            std.mem.asBytes(&params),
            .{ k_stride, num_groups, 1 },
            .{ @min(k_stride, 64), 4, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeHy4IndexPoolTopk(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Hy4IndexPoolTopkArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const spec = args.index_pool.spec;
        if (args.num_nodes == 0 or args.num_nodes > spec.max_nodes) return error.InvalidNodeCount;
        if (args.num_q_heads == 0) return error.ZeroDimension;
        if (args.kv_len > spec.max_kv_len) return error.InvalidConfig;

        const scale = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(spec.idx_dim))));
        const params: Hy4IndexPoolTopkParams = .{
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .num_k_heads = spec.num_k_heads,
            .idx_dim = spec.idx_dim,
            .kv_len = args.kv_len,
            .causal_q_start_pos = args.causal_q_start_pos orelse 0xFFFF_FFFF,
            .compress_ratio = spec.compress_ratio,
            .group_topk = spec.group_topk,
            .candidate_topk = spec.candidate_topk,
            .score_mode = @backingInt(args.score_mode),
            .has_head_weights = if (args.head_weights != null) 1 else 0,
            .swa_window = args.swa_window,
            .max_groups_stride = args.index_pool.max_groups,
            .max_cand_stride = args.index_pool.max_cand,
            .scale = scale,
        };
        try frame.encodeThreadgroups(
            self.hy4_index_pool_topk,
            &.{
                args.q_idx,
                args.head_weights orelse args.q_idx,
                args.k_fine,
                args.index_pool.kPoolBinding(args.index_metal_buf),
                args.index_pool.scratchGroupScoresBinding(args.index_metal_buf),
                args.index_pool.scratchCandScoresBinding(args.index_metal_buf),
                args.index_pool.scratchSelGroupsBinding(args.index_metal_buf),
                args.index_pool.scratchCandIdsBinding(args.index_metal_buf),
                args.index_pool.selectedTokensBinding(args.index_metal_buf),
                args.index_pool.selectedCountsBinding(args.index_metal_buf),
            },
            std.mem.asBytes(&params),
            .{ args.num_nodes, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Unified `IndexCache` period-4 (`full` vs `shared`) + `IndexPool` layer dispatcher:
    /// - `.full`: runs key compression (when `compress_keys` and `kv_len > 0`) and `ami_hy4_index_pool_topk`.
    /// - `.shared`: performs 0 dispatches and reuses `index_pool` cached indices.
    pub fn encodeHy4IndexLayer(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Hy4IndexLayerArgs,
        insert_barrier_before: bool,
    ) EncodeError!Hy4IndexStepDispatches {
        var counts: Hy4IndexStepDispatches = .{};
        if (args.mode == .shared) return counts;

        var barrier = insert_barrier_before;
        if (args.compress_keys and args.kv_len > 0) {
            try self.encodeHy4IndexCompress(frame, .{
                .k_fine = args.k_fine,
                .index_pool = args.index_pool,
                .index_metal_buf = args.index_metal_buf,
                .kv_len = args.kv_len,
                .pool_mode = args.pool_mode,
            }, barrier);
            counts.compress_dispatches += 1;
            barrier = true;
        }
        try self.encodeHy4IndexPoolTopk(frame, .{
            .q_idx = args.q_idx,
            .head_weights = args.head_weights,
            .k_fine = args.k_fine,
            .index_pool = args.index_pool,
            .index_metal_buf = args.index_metal_buf,
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .kv_len = args.kv_len,
            .causal_q_start_pos = args.causal_q_start_pos,
            .score_mode = args.score_mode,
            .swa_window = args.swa_window,
            .scale = args.scale,
        }, barrier);
        counts.topk_dispatches += 1;
        return counts;
    }

    pub fn encodeHy4LatentMoeRoute(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Hy4LatentMoeRouteArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0) return error.ZeroDimension;
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExperts;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;

        const params: Hy4LatentMoeRouteParams = .{
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .logits_f16 = if (args.logits_f16) 1 else 0,
            .has_bias = if (args.e_score_correction_bias != null) 1 else 0,
            .bias_f16 = if (args.bias_f16) 1 else 0,
            .norm_topk_prob = if (args.norm_topk_prob) 1 else 0,
            .routed_scaling_factor = args.routed_scaling_factor,
            .eps = args.eps,
        };
        try frame.encodeThreadgroups(
            self.hy4_latent_moe_route,
            &.{
                args.logits,
                args.e_score_correction_bias orelse args.logits,
                args.topk_experts,
                args.topk_weights,
            },
            std.mem.asBytes(&params),
            .{ args.m, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeHy4SwigluClampedLinearGate(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Hy4SwigluClampedArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_rows == 0 or args.intermediate_size == 0) return error.ZeroDimension;
        if (!args.interleaved_gate_up and args.up_in == null) return error.MissingInputs;

        const params: Hy4SwigluClampedParams = .{
            .num_rows = args.num_rows,
            .intermediate_size = args.intermediate_size,
            .interleaved_gate_up = if (args.interleaved_gate_up) 1 else 0,
            .clamp_mode = @backingInt(args.clamp_mode),
            .has_linear_gate = if (args.linear_gate != null) 1 else 0,
            .linear_gate_mode = @backingInt(args.linear_gate_mode),
            .swiglu_limit = args.swiglu_limit,
        };
        const tg_x: u64 = @min(@as(u64, args.intermediate_size), 256);
        try frame.encodeThreads(
            self.hy4_swiglu_clamped_linear_gate,
            &.{
                args.gate_in,
                args.up_in orelse args.gate_in,
                args.linear_gate orelse args.gate_in,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.intermediate_size, args.num_rows, 1 },
            .{ tg_x, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeHy4CombineSandwichNorm(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Hy4CombineSandwichArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_dim == 0) return error.ZeroDimension;
        if (args.top_k > MAX_TOP_K) return error.UnsupportedTopK;
        if (args.top_k > 0 and args.topk_weights == null) return error.MissingInputs;

        const params: Hy4CombineSandwichParams = .{
            .m = args.m,
            .hidden_dim = args.hidden_dim,
            .top_k = args.top_k,
            .has_shared = if (args.shared_out != null) 1 else 0,
            .has_residual = if (args.residual != null) 1 else 0,
            .apply_combine_norm = if (args.combine_norm_w != null) 1 else 0,
            .apply_post_norm = if (args.post_norm_w != null) 1 else 0,
            .weights_f16 = if (args.weights_f16) 1 else 0,
            .rms_eps = args.rms_eps,
            .res_scale = args.res_scale,
        };
        const pipe = self.hy4_combine_sandwich_norm[@backingInt(args.norm_dtype)];
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.expert_or_x,
                args.topk_weights orelse args.expert_or_x,
                args.shared_out orelse args.expert_or_x,
                args.residual orelse args.expert_or_x,
                args.combine_norm_w orelse args.expert_or_x,
                args.post_norm_w orelse args.expert_or_x,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.m, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeHy4MlaSinkAttn(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Hy4MlaSinkAttnArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const has_any_tree = (args.kv_tree_latent != null or args.kv_tree_rope != null or args.ancestor_masks != null);
        if (has_any_tree and (args.kv_tree_latent == null or args.ancestor_masks == null)) {
            return error.MissingInputs;
        }
        if (has_any_tree and args.qk_rope_dim > 0 and args.kv_tree_rope == null) {
            return error.MissingInputs;
        }
        if (args.num_nodes == 0 or args.num_nodes > args.index_pool.spec.max_nodes) return error.InvalidNodeCount;
        if (has_any_tree and args.num_nodes > MAX_TREE_NODES) return error.InvalidNodeCount;
        if (args.num_q_heads == 0 or args.kv_lora_rank == 0) return error.ZeroDimension;
        if (args.kv_lora_rank > MAX_HY4_KV_LORA_RANK or (args.kv_lora_rank % 2) != 0) return error.InvalidHeadDim;
        if (args.qk_rope_dim > 256) return error.InvalidHeadDim;
        if (args.qk_rope_dim > 0 and (args.q_pe == null or args.kv_rope == null)) return error.MissingInputs;

        const scale = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.kv_lora_rank + args.qk_rope_dim))));
        const params: Hy4MlaSinkAttnParams = .{
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .kv_lora_rank = args.kv_lora_rank,
            .qk_rope_dim = args.qk_rope_dim,
            .kv_len = args.kv_len,
            .causal_q_start_pos = args.causal_q_start_pos orelse 0xFFFF_FFFF,
            .swa_window = args.swa_window,
            .candidate_topk = args.index_pool.spec.candidate_topk,
            .has_sink = if (args.sink_params != null) 1 else 0,
            .has_linear_gate = if (args.linear_gate != null) 1 else 0,
            .linear_gate_mode = @backingInt(args.linear_gate_mode),
            .has_tree = if (has_any_tree) 1 else 0,
            .scale = scale,
            .attn_softcap = args.attn_softcap,
        };
        const pipe = if (args.kv_lora_rank == 512) self.hy4_mla_sink_attn_r512 else self.hy4_mla_sink_attn_dyn;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.q_nope,
                args.q_pe orelse args.q_nope,
                args.kv_latent,
                args.kv_rope orelse args.q_nope,
                args.index_pool.selectedTokensBinding(args.index_metal_buf),
                args.index_pool.selectedCountsBinding(args.index_metal_buf),
                args.kv_tree_latent orelse args.q_nope,
                args.kv_tree_rope orelse args.q_nope,
                args.ancestor_masks orelse args.q_nope,
                args.sink_params orelse args.q_nope,
                args.linear_gate orelse args.q_nope,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.num_q_heads, args.num_nodes, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }
};

// ===========================================================================
// Standalone CPU FP32 Reference Implementations
// ===========================================================================

fn sigmoidF32(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn siluF32(x: f32) f32 {
    return x * sigmoidF32(x);
}

fn softcapF32(x: f32, cap: f32) f32 {
    if (cap <= 0.0) return x;
    return cap * std.math.tanh(std.math.clamp(x / cap, -20.0, 20.0));
}

fn applyGateF32(val: f32, g: f32, mode: GateActivationMode) f32 {
    return switch (mode) {
        .sigmoid => val * sigmoidF32(g),
        .silu => val * siluF32(g),
        .raw_mul => val * g,
    };
}

fn loadNormWeightF32(raw_u16: []const u16, idx: usize, dtype: quant.FloatDType) f32 {
    return switch (dtype) {
        .f16 => @floatCast(@as(f16, @bitCast(raw_u16[idx]))),
        .bf16 => quant.bf16ToF32(raw_u16[idx]),
    };
}

/// CPU FP32 reference for `ami_inkling2_rmsnorm_sconv4` (both static `conv_w`
/// and dynamic `dyn_w`, updating `conv_state` ring when `write_state == true`).
pub fn rmsnormSconv4CpuRef(
    allocator: std.mem.Allocator,
    x: []const f16,
    norm_w_u16: ?[]const u16,
    norm_dtype: quant.FloatDType,
    conv_w_f32: ?[]const f32,
    conv_w_f16: ?[]const f16,
    dyn_w: ?[]const f16,
    conv_bias: ?[]const f32,
    conv_state: []f16,
    out: []f32,
    m: u32,
    hidden_dim: u32,
    state_pos: u32,
    write_state: bool,
    apply_rms_norm: bool,
    apply_silu: bool,
    rms_eps: f32,
) !void {
    const d: usize = hidden_dim;
    const norm_buf = try allocator.alloc(f32, @as(usize, m) * d);
    defer allocator.free(norm_buf);

    for (0..m) |t| {
        var inv_rms: f32 = 1.0;
        if (apply_rms_norm) {
            var ss: f64 = 0.0;
            for (0..d) |c| {
                const xv: f32 = @floatCast(x[t * d + c]);
                ss += @as(f64, xv) * @as(f64, xv);
            }
            inv_rms = @floatCast(1.0 / @sqrt(ss / @as(f64, @floatFromInt(d)) + @as(f64, rms_eps)));
        }
        for (0..d) |c| {
            const xv: f32 = @floatCast(x[t * d + c]);
            const nw: f32 = if (apply_rms_norm) loadNormWeightF32(norm_w_u16.?, c, norm_dtype) else 1.0;
            norm_buf[t * d + c] = xv * inv_rms * nw;
        }
    }

    const s0 = (state_pos + 0) % INKLING2_SCONV_RING_SLOTS;
    const s1 = (state_pos + 1) % INKLING2_SCONV_RING_SLOTS;
    const s2 = (state_pos + 2) % INKLING2_SCONV_RING_SLOTS;

    for (0..d) |c| {
        var h_m3: f32 = @floatCast(conv_state[@as(usize, s0) * d + c]);
        var h_m2: f32 = @floatCast(conv_state[@as(usize, s1) * d + c]);
        var h_m1: f32 = @floatCast(conv_state[@as(usize, s2) * d + c]);
        const b: f32 = if (conv_bias) |cb| cb[c] else 0.0;

        for (0..m) |t| {
            const h_cur = norm_buf[t * d + c];
            var w0: f32 = 0.0;
            var w1: f32 = 0.0;
            var w2: f32 = 0.0;
            var w3: f32 = 0.0;
            if (dyn_w) |dw| {
                const off = (t * d + c) * 4;
                w0 = @floatCast(dw[off + 0]);
                w1 = @floatCast(dw[off + 1]);
                w2 = @floatCast(dw[off + 2]);
                w3 = @floatCast(dw[off + 3]);
            } else if (conv_w_f16) |cw16| {
                const off = c * 4;
                w0 = @floatCast(cw16[off + 0]);
                w1 = @floatCast(cw16[off + 1]);
                w2 = @floatCast(cw16[off + 2]);
                w3 = @floatCast(cw16[off + 3]);
            } else if (conv_w_f32) |cw32| {
                const off = c * 4;
                w0 = cw32[off + 0];
                w1 = cw32[off + 1];
                w2 = cw32[off + 2];
                w3 = cw32[off + 3];
            }
            var acc = b + w0 * h_m3 + w1 * h_m2 + w2 * h_m1 + w3 * h_cur;
            if (apply_silu) acc = siluF32(acc);
            out[t * d + c] = acc;

            h_m3 = h_m2;
            h_m2 = h_m1;
            h_m1 = h_cur;
        }

        if (write_state) {
            const dst0 = (state_pos + m + 0) % INKLING2_SCONV_RING_SLOTS;
            const dst1 = (state_pos + m + 1) % INKLING2_SCONV_RING_SLOTS;
            const dst2 = (state_pos + m + 2) % INKLING2_SCONV_RING_SLOTS;
            conv_state[@as(usize, dst0) * d + c] = @floatCast(h_m3);
            conv_state[@as(usize, dst1) * d + c] = @floatCast(h_m2);
            conv_state[@as(usize, dst2) * d + c] = @floatCast(h_m1);
        }
    }
}

fn l2ClipVecF32(vec: []f32, clip_norm: f32) void {
    if (clip_norm <= 0.0) return;
    var ss: f64 = 0.0;
    for (vec) |v| ss += @as(f64, v) * @as(f64, v);
    const norm: f32 = @floatCast(@sqrt(ss));
    if (norm > clip_norm) {
        const fac = clip_norm / norm;
        for (vec) |*v| v.* *= fac;
    }
}

/// CPU FP32 reference for `ami_inkling2_rel_gqa_attn`.
pub fn inkling2RelGqaAttnCpuRef(
    allocator: std.mem.Allocator,
    q: []const f16,
    k_hist: []const f16,
    v_hist: []const f16,
    q_rel: ?[]const f16,
    rel_emb: ?[]const f16,
    k_tree: ?[]const f16,
    v_tree: ?[]const f16,
    ancestor_masks: ?[]const u64,
    sink_logits: ?[]const f32,
    sink_per_token: bool,
    gate_attn: ?[]const f16,
    gate_mode: GateActivationMode,
    out: []f32,
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    window_size: u32,
    causal_q_start_pos: ?u32,
    d_rel: u32,
    rel_extent: u32,
    rel_per_head: bool,
    use_log_scaling: bool,
    log_scale_alpha: f32,
    scale: f32,
    qk_l2_clip: f32,
    rel_l2_clip: f32,
    rel_scale: f32,
    attn_softcap: f32,
) !void {
    const gqa_ratio = num_q_heads / num_kv_heads;
    const q_buf = try allocator.alloc(f32, head_dim);
    defer allocator.free(q_buf);
    const k_buf = try allocator.alloc(f32, head_dim);
    defer allocator.free(k_buf);
    const qr_buf = try allocator.alloc(f32, @max(d_rel, 1));
    defer allocator.free(qr_buf);
    const rk_buf = try allocator.alloc(f32, @max(d_rel, 1));
    defer allocator.free(rk_buf);
    const acc = try allocator.alloc(f32, head_dim);
    defer allocator.free(acc);

    const has_tree = ancestor_masks != null;

    for (0..num_nodes) |node| {
        const anc_mask: u64 = if (has_tree) ancestor_masks.?[node] else 0;
        const tree_anc: u32 = if (has_tree) @intCast(@popCount(anc_mask)) else 0;
        const tree_depth: u32 = if (tree_anc > 0) tree_anc - 1 else 0;
        const q_pos: u32 = if (has_tree)
            kv_len + tree_depth
        else if (causal_q_start_pos) |qs|
            qs + @as(u32, @intCast(node))
        else if (kv_len > 0)
            kv_len - 1
        else
            0;

        const eff_kv_len: u32 = if (has_tree)
            kv_len
        else if (causal_q_start_pos != null)
            @min(kv_len, q_pos + 1)
        else
            kv_len;

        const win_start: u32 = if (window_size > 0 and (q_pos + 1) > window_size)
            (q_pos + 1) - window_size
        else
            0;

        const log_mult: f32 = if (use_log_scaling)
            1.0 + log_scale_alpha * std.math.log(f32, std.math.e, @as(f32, @floatFromInt(q_pos)) + 1.0)
        else
            1.0;
        const scale_eff = scale * log_mult;

        for (0..num_q_heads) |qh| {
            const kv_head = qh / gqa_ratio;
            const q_off = (node * num_q_heads + qh) * head_dim;
            for (0..head_dim) |d| q_buf[d] = @floatCast(q[q_off + d]);
            l2ClipVecF32(q_buf, qk_l2_clip);

            if (d_rel > 0) {
                const qr_off = (node * num_q_heads + qh) * d_rel;
                for (0..d_rel) |r| qr_buf[r] = @floatCast(q_rel.?[qr_off + r]);
                l2ClipVecF32(qr_buf[0..d_rel], rel_l2_clip);
            }
            const rel_h_off: usize = if (rel_per_head) qh * rel_extent else 0;

            var max_score: f32 = -std.math.inf(f32);
            var sum_exp: f32 = 0.0;
            if (sink_logits) |sl| {
                const s_idx = if (sink_per_token) node * num_q_heads + qh else qh;
                max_score = sl[s_idx];
                sum_exp = 1.0;
            }
            @memset(acc, 0.0);

            const stepScoreAndUpdate = struct {
                fn run(
                    qb: []const f32,
                    kb: []f32,
                    qrb: []const f32,
                    rkb: []f32,
                    k_src: []const f16,
                    v_src: []const f16,
                    rel_table: ?[]const f16,
                    rel_off: usize,
                    qp: u32,
                    kp: u32,
                    hd: u32,
                    dr: u32,
                    re: u32,
                    qkl2: f32,
                    rl2: f32,
                    se: f32,
                    rs: f32,
                    cap: f32,
                    m_ptr: *f32,
                    l_ptr: *f32,
                    a_buf: []f32,
                ) void {
                    for (0..hd) |d| kb[d] = @floatCast(k_src[d]);
                    l2ClipVecF32(kb, qkl2);
                    var qk_dot: f64 = 0.0;
                    for (0..hd) |d| qk_dot += @as(f64, qb[d]) * @as(f64, kb[d]);
                    var rel_sc: f32 = 0.0;
                    if (dr > 0 and re > 0) {
                        const dist: u32 = if (qp >= kp) @min(qp - kp, re - 1) else 0;
                        const rt = rel_table.?[(rel_off + dist) * dr ..][0..dr];
                        for (0..dr) |r| rkb[r] = @floatCast(rt[r]);
                        l2ClipVecF32(rkb[0..dr], rl2);
                        var rd: f64 = 0.0;
                        for (0..dr) |r| rd += @as(f64, qrb[r]) * @as(f64, rkb[r]);
                        rel_sc = @as(f32, @floatCast(rd)) * rs;
                    }
                    const sc = softcapF32(@as(f32, @floatCast(qk_dot)) * se + rel_sc, cap);
                    const new_m = @max(m_ptr.*, sc);
                    const corr: f32 = if (l_ptr.* > 0.0) @exp(m_ptr.* - new_m) else 0.0;
                    const pw: f32 = @exp(sc - new_m);
                    l_ptr.* = l_ptr.* * corr + pw;
                    for (0..hd) |d| {
                        const vv: f32 = @floatCast(v_src[d]);
                        a_buf[d] = a_buf[d] * corr + pw * vv;
                    }
                    m_ptr.* = new_m;
                }
            }.run;

            var t: u32 = win_start;
            while (t < eff_kv_len) : (t += 1) {
                const kv_off = (@as(usize, t) * num_kv_heads + kv_head) * head_dim;
                stepScoreAndUpdate(
                    q_buf,
                    k_buf,
                    qr_buf,
                    rk_buf,
                    k_hist[kv_off..][0..head_dim],
                    v_hist[kv_off..][0..head_dim],
                    rel_emb,
                    rel_h_off,
                    q_pos,
                    t,
                    head_dim,
                    d_rel,
                    rel_extent,
                    qk_l2_clip,
                    rel_l2_clip,
                    scale_eff,
                    rel_scale,
                    attn_softcap,
                    &max_score,
                    &sum_exp,
                    acc,
                );
            }

            if (has_tree) {
                for (0..num_nodes) |j| {
                    if (((anc_mask >> @intCast(j)) & 1) == 0) continue;
                    const j_anc: u32 = @intCast(@popCount(ancestor_masks.?[j]));
                    const k_pos_j = kv_len + (if (j_anc > 0) j_anc - 1 else 0);
                    if (window_size > 0 and k_pos_j < win_start) continue;
                    const kv_off = (j * num_kv_heads + kv_head) * head_dim;
                    stepScoreAndUpdate(
                        q_buf,
                        k_buf,
                        qr_buf,
                        rk_buf,
                        k_tree.?[kv_off..][0..head_dim],
                        v_tree.?[kv_off..][0..head_dim],
                        rel_emb,
                        rel_h_off,
                        q_pos,
                        k_pos_j,
                        head_dim,
                        d_rel,
                        rel_extent,
                        qk_l2_clip,
                        rel_l2_clip,
                        scale_eff,
                        rel_scale,
                        attn_softcap,
                        &max_score,
                        &sum_exp,
                        acc,
                    );
                }
            }

            const inv_l: f32 = if (sum_exp > 0.0) 1.0 / sum_exp else 0.0;
            for (0..head_dim) |d| {
                var ov = acc[d] * inv_l;
                if (gate_attn) |ga| {
                    ov = applyGateF32(ov, @floatCast(ga[q_off + d]), gate_mode);
                }
                out[q_off + d] = ov;
            }
        }
    }
}

/// CPU FP32 reference for `ami_inkling2_moe_route`.
pub fn inkling2MoeRouteCpuRef(
    allocator: std.mem.Allocator,
    logits_f32: ?[]const f32,
    logits_f16: ?[]const f16,
    bias_f32: ?[]const f32,
    bias_f16: ?[]const f16,
    bias_add_to_logits: bool,
    topk_experts: []u32,
    topk_weights: []f32,
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: bool,
    route_scale: f32,
    router_softcap: f32,
) !void {
    const probs = try allocator.alloc(f32, num_experts);
    defer allocator.free(probs);
    const picked = try allocator.alloc(bool, num_experts);
    defer allocator.free(picked);

    for (0..m) |tok| {
        const base = tok * num_experts;
        var max_v: f32 = -std.math.inf(f32);
        for (0..num_experts) |e| {
            var v: f32 = if (logits_f16) |l16| @floatCast(l16[base + e]) else logits_f32.?[base + e];
            v = softcapF32(v, router_softcap);
            if (bias_add_to_logits) {
                if (bias_f16) |b16| {
                    v += @floatCast(b16[e]);
                } else if (bias_f32) |b32| {
                    v += b32[e];
                }
            }
            probs[e] = v;
            picked[e] = false;
            max_v = @max(max_v, v);
        }

        var sum_exp: f32 = 0.0;
        for (0..num_experts) |e| {
            const ex = @exp(probs[e] - max_v);
            probs[e] = ex;
            sum_exp += ex;
        }
        const inv_sum: f32 = if (sum_exp > 0.0) 1.0 / sum_exp else 0.0;
        for (0..num_experts) |e| probs[e] *= inv_sum;

        var w_sum: f32 = 0.0;
        const out_base = tok * top_k;
        for (0..top_k) |k| {
            var best_s: f32 = -std.math.inf(f32);
            var best_e: u32 = 0xFFFF_FFFF;
            for (0..num_experts) |e| {
                if (picked[e]) continue;
                var s = probs[e];
                if (!bias_add_to_logits) {
                    if (bias_f16) |b16| {
                        s += @floatCast(b16[e]);
                    } else if (bias_f32) |b32| {
                        s += b32[e];
                    }
                }
                if (s > best_s or (s == best_s and @as(u32, @intCast(e)) < best_e)) {
                    best_s = s;
                    best_e = @intCast(e);
                }
            }
            picked[best_e] = true;
            topk_experts[out_base + k] = best_e;
            topk_weights[out_base + k] = probs[best_e];
            w_sum += probs[best_e];
        }

        const norm_fac: f32 = if (norm_topk_prob and w_sum > 0.0) 1.0 / w_sum else 1.0;
        for (0..top_k) |k| {
            topk_weights[out_base + k] = topk_weights[out_base + k] * norm_fac * route_scale;
        }
    }
}

/// CPU FP32 reference for `ami_inkling2_moe_swiglu_softcap`.
pub fn inkling2MoeSwigluSoftcapCpuRef(
    gate_in: []const f16,
    up_in: ?[]const f16,
    out: []f32,
    num_rows: u32,
    intermediate_size: u32,
    interleaved_gate_up: bool,
    softcap: f32,
) void {
    for (0..num_rows) |r| {
        for (0..intermediate_size) |c| {
            const g: f32 = if (interleaved_gate_up)
                @floatCast(gate_in[r * (2 * intermediate_size) + c])
            else
                @floatCast(gate_in[r * intermediate_size + c]);
            const u: f32 = if (interleaved_gate_up)
                @floatCast(gate_in[r * (2 * intermediate_size) + intermediate_size + c])
            else
                @floatCast(up_in.?[r * intermediate_size + c]);
            out[r * intermediate_size + c] = softcapF32(siluF32(g) * u, softcap);
        }
    }
}

/// CPU FP32 reference for `ami_inkling2_moe_route_softcap_epilogue`.
pub fn inkling2MoeEpilogueCpuRef(
    expert_out: []const f16,
    topk_weights_f32: ?[]const f32,
    topk_weights_f16: ?[]const f16,
    residual: ?[]const f16,
    out: []f32,
    m: u32,
    hidden_dim: u32,
    top_k: u32,
    mlp_softcap: f32,
    res_scale: f32,
) void {
    for (0..m) |tok| {
        const w_base = tok * top_k;
        for (0..hidden_dim) |c| {
            var acc: f32 = 0.0;
            for (0..top_k) |k| {
                const w: f32 = if (topk_weights_f16) |w16|
                    @floatCast(w16[w_base + k])
                else
                    topk_weights_f32.?[w_base + k];
                const ev: f32 = @floatCast(expert_out[(w_base + k) * hidden_dim + c]);
                acc += w * ev;
            }
            acc = softcapF32(acc, mlp_softcap);
            if (residual) |res| {
                acc = @as(f32, @floatCast(res[tok * hidden_dim + c])) + res_scale * acc;
            }
            out[tok * hidden_dim + c] = acc;
        }
    }
}

/// CPU FP32 reference for `ami_inkling2_attn_res`.
pub fn inkling2AttnResCpuRef(
    block_states: []const f16,
    block_slots: []const u32,
    block_stride_elems: u32,
    query: []const f16,
    norm_w_u16: ?[]const u16,
    norm_dtype: quant.FloatDType,
    out: []f32,
    m: u32,
    hidden_dim: u32,
    apply_rms_norm: bool,
    normalize_values: bool,
    query_per_token: bool,
    rms_eps: f32,
    scale: f32,
) void {
    const num_blocks = block_slots.len;
    var logits: [MAX_ATTN_RES_BLOCKS]f32 = @splat(0.0);
    var alphas: [MAX_ATTN_RES_BLOCKS]f32 = @splat(0.0);
    var inv_rms_b: [MAX_ATTN_RES_BLOCKS]f32 = @splat(1.0);

    for (0..m) |tok| {
        const q_base = if (query_per_token) tok * hidden_dim else 0;
        const tok_off = tok * hidden_dim;
        var max_l: f32 = -std.math.inf(f32);

        for (0..num_blocks) |b| {
            const vb = block_states[@as(usize, block_slots[b]) * block_stride_elems + tok_off ..][0..hidden_dim];
            var ss: f64 = 0.0;
            var dt: f64 = 0.0;
            for (0..hidden_dim) |c| {
                const v: f32 = @floatCast(vb[c]);
                const qv: f32 = @floatCast(query[q_base + c]);
                const nw: f32 = if (norm_w_u16) |nu| loadNormWeightF32(nu, c, norm_dtype) else 1.0;
                ss += @as(f64, v) * @as(f64, v);
                dt += @as(f64, v * nw) * @as(f64, qv);
            }
            const inv_rms: f32 = if (apply_rms_norm or normalize_values)
                @floatCast(1.0 / @sqrt(ss / @as(f64, @floatFromInt(hidden_dim)) + @as(f64, rms_eps)))
            else
                1.0;
            inv_rms_b[b] = inv_rms;
            logits[b] = (if (apply_rms_norm) (@as(f32, @floatCast(dt)) * inv_rms) else @as(f32, @floatCast(dt))) * scale;
            max_l = @max(max_l, logits[b]);
        }

        var exp_sum: f32 = 0.0;
        for (0..num_blocks) |b| {
            alphas[b] = @exp(logits[b] - max_l);
            exp_sum += alphas[b];
        }
        const inv_e: f32 = if (exp_sum > 0.0) 1.0 / exp_sum else 0.0;
        for (0..num_blocks) |b| alphas[b] *= inv_e;

        for (0..hidden_dim) |c| {
            var acc: f32 = 0.0;
            for (0..num_blocks) |b| {
                const raw_v: f32 = @floatCast(block_states[@as(usize, block_slots[b]) * block_stride_elems + tok_off + c]);
                const v = if (normalize_values) raw_v * inv_rms_b[b] else raw_v;
                acc += alphas[b] * v;
            }
            out[tok_off + c] = acc;
        }
    }
}

/// CPU FP32 reference for `ami_hy4_index_pool_compress`.
pub fn hy4IndexPoolCompressCpuRef(
    k_fine: []const f16,
    k_pool_out: []f16,
    kv_len: u32,
    num_k_heads: u32,
    idx_dim: u32,
    compress_ratio: u32,
    pool_mode: IndexPoolMode,
) void {
    const stride: usize = @as(usize, num_k_heads) * idx_dim;
    const num_groups = (kv_len + compress_ratio - 1) / compress_ratio;
    for (0..num_groups) |grp| {
        const t0 = grp * compress_ratio;
        const t1 = @min(t0 + compress_ratio, kv_len);
        const cnt: f32 = @floatFromInt(t1 - t0);
        for (0..stride) |ch| {
            if (pool_mode == .mean) {
                var acc: f32 = 0.0;
                for (t0..t1) |t| acc += @as(f32, @floatCast(k_fine[t * stride + ch]));
                k_pool_out[grp * stride + ch] = @floatCast(acc / cnt);
            } else {
                var mx: f32 = -std.math.inf(f32);
                for (t0..t1) |t| mx = @max(mx, @as(f32, @floatCast(k_fine[t * stride + ch])));
                k_pool_out[grp * stride + ch] = @floatCast(mx);
            }
        }
    }
}

const ScoredItem = struct {
    score: f32,
    idx: u32,
};

fn scoredItemDesc(_: void, a: ScoredItem, b: ScoredItem) bool {
    if (a.score != b.score) return a.score > b.score;
    return a.idx < b.idx;
}

fn idxAsc(_: void, a: ScoredItem, b: ScoredItem) bool {
    return a.idx < b.idx;
}

fn scoreKeyCpu(
    q_node: []const f16,
    head_weights: ?[]const f16,
    k_vec: []const f16,
    num_q_heads: u32,
    num_k_heads: u32,
    idx_dim: u32,
    score_mode: IndexScoreMode,
    scale: f32,
) f32 {
    const q_per_k = if (num_k_heads > 0) @max(num_q_heads / num_k_heads, 1) else num_q_heads;
    var agg: f32 = if (score_mode == .dot_max_scaled) -std.math.inf(f32) else 0.0;
    for (0..num_q_heads) |h| {
        const kh: usize = if (num_k_heads > 1) h / q_per_k else 0;
        const qh = q_node[h * idx_dim ..][0..idx_dim];
        const khv = k_vec[kh * idx_dim ..][0..idx_dim];
        var dot_acc: f64 = 0.0;
        for (0..idx_dim) |d| {
            dot_acc += @as(f64, @floatCast(qh[d])) * @as(f64, @floatCast(khv[d]));
        }
        const hd: f32 = @floatCast(dot_acc);
        switch (score_mode) {
            .relu_weighted_sum => {
                const hw: f32 = if (head_weights) |w| @floatCast(w[h]) else 1.0;
                agg += hw * @max(@as(f32, 0.0), hd) * scale;
            },
            .dot_sum_scaled => {
                const hw: f32 = if (head_weights) |w| @floatCast(w[h]) else 1.0;
                agg += hw * hd * scale;
            },
            .dot_max_scaled => {
                agg = @max(agg, hd * scale);
            },
        }
    }
    return agg;
}

/// CPU FP32 reference for `ami_hy4_index_pool_topk`.
pub fn hy4IndexPoolTopkCpuRef(
    allocator: std.mem.Allocator,
    q_idx: []const f16,
    head_weights: ?[]const f16,
    k_fine: []const f16,
    k_pool: []const f16,
    selected_tokens: []u32,
    selected_counts: []u32,
    num_nodes: u32,
    num_q_heads: u32,
    num_k_heads: u32,
    idx_dim: u32,
    kv_len: u32,
    causal_q_start_pos: ?u32,
    compress_ratio: u32,
    group_topk: u32,
    candidate_topk: u32,
    score_mode: IndexScoreMode,
    swa_window: u32,
    scale: f32,
) !void {
    const k_stride: usize = @as(usize, num_k_heads) * idx_dim;
    for (0..num_nodes) |node| {
        const out_slice = selected_tokens[node * candidate_topk ..][0..candidate_topk];
        @memset(out_slice, 0xFFFF_FFFF);

        const eff_kv_len: u32 = if (causal_q_start_pos) |qs|
            @min(kv_len, qs + @as(u32, @intCast(node)) + 1)
        else
            kv_len;
        const sparse_end: u32 = if (swa_window > 0 and eff_kv_len > swa_window)
            eff_kv_len - swa_window
        else if (swa_window > 0)
            0
        else
            eff_kv_len;

        if (sparse_end == 0 or candidate_topk == 0) {
            selected_counts[node] = 0;
            continue;
        }
        if (sparse_end <= candidate_topk) {
            for (0..sparse_end) |i| out_slice[i] = @intCast(i);
            selected_counts[node] = sparse_end;
            continue;
        }

        const q_node = q_idx[node * num_q_heads * idx_dim ..][0 .. num_q_heads * idx_dim];
        const hw_node: ?[]const f16 = if (head_weights) |hw| hw[node * num_q_heads ..][0..num_q_heads] else null;
        const num_groups = (sparse_end + compress_ratio - 1) / compress_ratio;

        var cand_ids = std.ArrayList(u32).empty;
        defer cand_ids.deinit(allocator);

        if (group_topk > 0 and num_groups > group_topk) {
            const g_items = try allocator.alloc(ScoredItem, num_groups);
            defer allocator.free(g_items);
            for (0..num_groups) |g| {
                g_items[g] = .{
                    .score = scoreKeyCpu(q_node, hw_node, k_pool[g * k_stride ..][0..k_stride], num_q_heads, num_k_heads, idx_dim, score_mode, scale),
                    .idx = @intCast(g),
                };
            }
            std.sort.pdq(ScoredItem, g_items, {}, scoredItemDesc);
            std.sort.pdq(ScoredItem, g_items[0..group_topk], {}, idxAsc);
            for (g_items[0..group_topk]) |gi| {
                const t0 = gi.idx * compress_ratio;
                const t1 = @min(t0 + compress_ratio, sparse_end);
                for (t0..t1) |t| try cand_ids.append(allocator, @intCast(t));
            }
        } else {
            for (0..sparse_end) |t| try cand_ids.append(allocator, @intCast(t));
        }

        if (cand_ids.items.len <= candidate_topk) {
            for (cand_ids.items, 0..) |tid, i| out_slice[i] = tid;
            selected_counts[node] = @intCast(cand_ids.items.len);
            continue;
        }

        const c_items = try allocator.alloc(ScoredItem, cand_ids.items.len);
        defer allocator.free(c_items);
        for (cand_ids.items, 0..) |t, ci| {
            c_items[ci] = .{
                .score = scoreKeyCpu(q_node, hw_node, k_fine[@as(usize, t) * k_stride ..][0..k_stride], num_q_heads, num_k_heads, idx_dim, score_mode, scale),
                .idx = t,
            };
        }
        std.sort.pdq(ScoredItem, c_items, {}, scoredItemDesc);
        std.sort.pdq(ScoredItem, c_items[0..candidate_topk], {}, idxAsc);
        for (0..candidate_topk) |i| out_slice[i] = c_items[i].idx;
        selected_counts[node] = candidate_topk;
    }
}

/// CPU FP32 reference for `ami_hy4_latent_moe_route`.
pub fn hy4LatentMoeRouteCpuRef(
    allocator: std.mem.Allocator,
    logits_f32: ?[]const f32,
    logits_f16: ?[]const f16,
    bias_f32: ?[]const f32,
    bias_f16: ?[]const f16,
    topk_experts: []u32,
    topk_weights: []f32,
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
    eps: f32,
) !void {
    const scores = try allocator.alloc(f32, num_experts);
    defer allocator.free(scores);
    const picked = try allocator.alloc(bool, num_experts);
    defer allocator.free(picked);

    for (0..m) |tok| {
        const base = tok * num_experts;
        for (0..num_experts) |e| {
            const x: f32 = if (logits_f16) |l16| @floatCast(l16[base + e]) else logits_f32.?[base + e];
            scores[e] = sigmoidF32(x);
            picked[e] = false;
        }

        var w_sum: f32 = 0.0;
        const out_base = tok * top_k;
        for (0..top_k) |k| {
            var best_s: f32 = -std.math.inf(f32);
            var best_e: u32 = 0xFFFF_FFFF;
            for (0..num_experts) |e| {
                if (picked[e]) continue;
                var s = scores[e];
                if (bias_f16) |b16| {
                    s += @floatCast(b16[e]);
                } else if (bias_f32) |b32| {
                    s += b32[e];
                }
                if (s > best_s or (s == best_s and @as(u32, @intCast(e)) < best_e)) {
                    best_s = s;
                    best_e = @intCast(e);
                }
            }
            picked[best_e] = true;
            topk_experts[out_base + k] = best_e;
            topk_weights[out_base + k] = scores[best_e];
            w_sum += scores[best_e];
        }

        const norm_fac: f32 = if (norm_topk_prob) 1.0 / (w_sum + eps) else 1.0;
        for (0..top_k) |k| {
            topk_weights[out_base + k] = topk_weights[out_base + k] * norm_fac * routed_scaling_factor;
        }
    }
}

/// CPU FP32 reference for `ami_hy4_swiglu_clamped_linear_gate`.
pub fn hy4SwigluClampedLinearGateCpuRef(
    gate_in: []const f16,
    up_in: ?[]const f16,
    linear_gate: ?[]const f16,
    linear_gate_mode: GateActivationMode,
    out: []f32,
    num_rows: u32,
    intermediate_size: u32,
    interleaved_gate_up: bool,
    clamp_mode: SwigluClampMode,
    swiglu_limit: f32,
) void {
    for (0..num_rows) |r| {
        for (0..intermediate_size) |c| {
            const idx = r * intermediate_size + c;
            var g: f32 = if (interleaved_gate_up)
                @floatCast(gate_in[r * (2 * intermediate_size) + c])
            else
                @floatCast(gate_in[idx]);
            var u: f32 = if (interleaved_gate_up)
                @floatCast(gate_in[r * (2 * intermediate_size) + intermediate_size + c])
            else
                @floatCast(up_in.?[idx]);

            if (swiglu_limit > 0.0 and (clamp_mode == .gate_and_up or clamp_mode == .both)) {
                g = std.math.clamp(g, -swiglu_limit, swiglu_limit);
                u = std.math.clamp(u, -swiglu_limit, swiglu_limit);
            }
            var v = siluF32(g) * u;
            if (swiglu_limit > 0.0 and (clamp_mode == .product or clamp_mode == .both)) {
                v = std.math.clamp(v, -swiglu_limit, swiglu_limit);
            }
            if (linear_gate) |lg| {
                v = applyGateF32(v, @floatCast(lg[idx]), linear_gate_mode);
            }
            out[idx] = v;
        }
    }
}

/// CPU FP32 reference for `ami_hy4_combine_sandwich_norm`.
pub fn hy4CombineSandwichNormCpuRef(
    allocator: std.mem.Allocator,
    expert_or_x: []const f16,
    topk_weights_f32: ?[]const f32,
    topk_weights_f16: ?[]const f16,
    shared_out: ?[]const f16,
    residual: ?[]const f16,
    combine_norm_w: ?[]const u16,
    post_norm_w: ?[]const u16,
    norm_dtype: quant.FloatDType,
    out: []f32,
    m: u32,
    hidden_dim: u32,
    top_k: u32,
    rms_eps: f32,
    res_scale: f32,
) !void {
    const tmp = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(tmp);

    for (0..m) |tok| {
        const tok_off = tok * hidden_dim;
        const w_base = tok * top_k;
        var ss: f64 = 0.0;
        for (0..hidden_dim) |c| {
            var val: f32 = 0.0;
            if (top_k > 0) {
                for (0..top_k) |k| {
                    const w: f32 = if (topk_weights_f16) |w16|
                        @floatCast(w16[w_base + k])
                    else
                        topk_weights_f32.?[w_base + k];
                    val += w * @as(f32, @floatCast(expert_or_x[(w_base + k) * hidden_dim + c]));
                }
            } else {
                val = @floatCast(expert_or_x[tok_off + c]);
            }
            if (shared_out) |sh| val += @floatCast(sh[tok_off + c]);
            tmp[c] = val;
            ss += @as(f64, val) * @as(f64, val);
        }

        if (combine_norm_w) |cnw| {
            const inv1: f32 = @floatCast(1.0 / @sqrt(ss / @as(f64, @floatFromInt(hidden_dim)) + @as(f64, rms_eps)));
            for (0..hidden_dim) |c| {
                tmp[c] = tmp[c] * inv1 * loadNormWeightF32(cnw, c, norm_dtype);
            }
        }
        if (residual) |res| {
            for (0..hidden_dim) |c| {
                tmp[c] = @as(f32, @floatCast(res[tok_off + c])) + res_scale * tmp[c];
            }
        }
        if (post_norm_w) |pnw| {
            var ss2: f64 = 0.0;
            for (0..hidden_dim) |c| ss2 += @as(f64, tmp[c]) * @as(f64, tmp[c]);
            const inv2: f32 = @floatCast(1.0 / @sqrt(ss2 / @as(f64, @floatFromInt(hidden_dim)) + @as(f64, rms_eps)));
            for (0..hidden_dim) |c| {
                out[tok_off + c] = tmp[c] * inv2 * loadNormWeightF32(pnw, c, norm_dtype);
            }
        } else {
            @memcpy(out[tok_off..][0..hidden_dim], tmp);
        }
    }
}

/// CPU FP32 reference for `ami_hy4_mla_sink_attn`.
pub fn hy4MlaSinkAttnCpuRef(
    allocator: std.mem.Allocator,
    q_nope: []const f16,
    q_pe: ?[]const f16,
    kv_latent: []const f16,
    kv_rope: ?[]const f16,
    selected_tokens: []const u32,
    selected_counts: []const u32,
    kv_tree_latent: ?[]const f16,
    kv_tree_rope: ?[]const f16,
    ancestor_masks: ?[]const u64,
    sink_params: ?[]const f32,
    linear_gate: ?[]const f16,
    linear_gate_mode: GateActivationMode,
    out: []f32,
    num_nodes: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    qk_rope_dim: u32,
    kv_len: u32,
    causal_q_start_pos: ?u32,
    swa_window: u32,
    candidate_topk: u32,
    scale: f32,
    attn_softcap: f32,
) !void {
    const acc = try allocator.alloc(f32, kv_lora_rank);
    defer allocator.free(acc);
    const has_tree = ancestor_masks != null;

    for (0..num_nodes) |node| {
        const eff_kv_len: u32 = if (has_tree)
            kv_len
        else if (causal_q_start_pos) |qs|
            @min(kv_len, qs + @as(u32, @intCast(node)) + 1)
        else
            kv_len;
        const swa_lo: u32 = if (swa_window > 0)
            (if (eff_kv_len > swa_window) eff_kv_len - swa_window else 0)
        else
            eff_kv_len;
        const sparse_cap: u32 = if (swa_window > 0) swa_lo else eff_kv_len;
        const sel_cnt = if (candidate_topk > 0) @min(selected_counts[node], candidate_topk) else 0;
        const sel_slice = selected_tokens[node * candidate_topk ..][0..sel_cnt];

        for (0..num_q_heads) |qh| {
            const q_off = (node * num_q_heads + qh) * kv_lora_rank;
            const q_pe_off = (node * num_q_heads + qh) * qk_rope_dim;
            var max_s: f32 = if (sink_params) |sp| sp[qh] else -std.math.inf(f32);
            var sum_exp: f32 = if (sink_params != null) 1.0 else 0.0;
            @memset(acc, 0.0);

            const accumTok = struct {
                fn run(
                    qn: []const f16,
                    qp: ?[]const f16,
                    kl: []const f16,
                    kr: ?[]const f16,
                    d_lat: u32,
                    d_rope: u32,
                    sc_scale: f32,
                    cap: f32,
                    m_ptr: *f32,
                    l_ptr: *f32,
                    a_buf: []f32,
                ) void {
                    var dot_p: f64 = 0.0;
                    for (0..d_lat) |d| dot_p += @as(f64, @floatCast(qn[d])) * @as(f64, @floatCast(kl[d]));
                    for (0..d_rope) |d| dot_p += @as(f64, @floatCast(qp.?[d])) * @as(f64, @floatCast(kr.?[d]));
                    const sc = softcapF32(@as(f32, @floatCast(dot_p)) * sc_scale, cap);
                    const new_m = @max(m_ptr.*, sc);
                    const corr: f32 = if (l_ptr.* > 0.0) @exp(m_ptr.* - new_m) else 0.0;
                    const pw: f32 = @exp(sc - new_m);
                    l_ptr.* = l_ptr.* * corr + pw;
                    for (0..d_lat) |d| {
                        a_buf[d] = a_buf[d] * corr + pw * @as(f32, @floatCast(kl[d]));
                    }
                    m_ptr.* = new_m;
                }
            }.run;

            const qn_row = q_nope[q_off..][0..kv_lora_rank];
            const qp_row: ?[]const f16 = if (qk_rope_dim > 0) q_pe.?[q_pe_off..][0..qk_rope_dim] else null;

            for (sel_slice) |t| {
                if (t == 0xFFFF_FFFF or t >= sparse_cap) continue;
                const kr_row: ?[]const f16 = if (qk_rope_dim > 0) kv_rope.?[@as(usize, t) * qk_rope_dim ..][0..qk_rope_dim] else null;
                accumTok(qn_row, qp_row, kv_latent[@as(usize, t) * kv_lora_rank ..][0..kv_lora_rank], kr_row, kv_lora_rank, qk_rope_dim, scale, attn_softcap, &max_s, &sum_exp, acc);
            }
            var t: u32 = swa_lo;
            while (t < eff_kv_len) : (t += 1) {
                const kr_row: ?[]const f16 = if (qk_rope_dim > 0) kv_rope.?[@as(usize, t) * qk_rope_dim ..][0..qk_rope_dim] else null;
                accumTok(qn_row, qp_row, kv_latent[@as(usize, t) * kv_lora_rank ..][0..kv_lora_rank], kr_row, kv_lora_rank, qk_rope_dim, scale, attn_softcap, &max_s, &sum_exp, acc);
            }
            if (has_tree) {
                const mask = ancestor_masks.?[node];
                for (0..num_nodes) |j| {
                    if (((mask >> @intCast(j)) & 1) == 0) continue;
                    const kr_row: ?[]const f16 = if (qk_rope_dim > 0) kv_tree_rope.?[j * qk_rope_dim ..][0..qk_rope_dim] else null;
                    accumTok(qn_row, qp_row, kv_tree_latent.?[j * kv_lora_rank ..][0..kv_lora_rank], kr_row, kv_lora_rank, qk_rope_dim, scale, attn_softcap, &max_s, &sum_exp, acc);
                }
            }

            const inv_l: f32 = if (sum_exp > 0.0) 1.0 / sum_exp else 0.0;
            for (0..kv_lora_rank) |d| {
                var ov = acc[d] * inv_l;
                if (linear_gate) |lg| {
                    ov = applyGateF32(ov, @floatCast(lg[q_off + d]), linear_gate_mode);
                }
                out[q_off + d] = ov;
            }
        }
    }
}

// ===========================================================================
// Tests: Alignment, Zero-Alloc Hot Path, and CPU FP32 Parity across
// M = 1, M = 8, M = 33 (incl. 1152 width), 1024-tok HybridKvRing SWA,
// IndexCache period-4 reuse, and IndexPool compress_ratio = 4
// ===========================================================================

fn makeTestArena(min_bytes: usize) !arena_mod.WiredArena {
    const cap = try page.alignUp(min_bytes);
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = cap * 4,
        .user_wire_limit = cap * 2,
        .global_user_wire_limit = cap * 2,
        .gpu_working_set_bytes = cap * 2,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    return arena_mod.WiredArena.init(cap, budget);
}

fn synthF32(seed: usize, scale: f32) f32 {
    const x: u64 = (@as(u64, seed) *% 6364136223846793005) +% 1442695040888963407;
    const bucket: i32 = @as(i32, @intCast((x >> 33) % 41)) - 20;
    return @as(f32, @floatFromInt(bucket)) * scale;
}

fn fillSynthF16(slice: []f16, seed_base: usize, scale: f32) void {
    for (slice, 0..) |*dst, i| dst.* = @floatCast(synthF32(seed_base + i, scale));
}

fn fillSynthNormU16(slice: []u16, seed_base: usize, dtype: quant.FloatDType) void {
    for (slice, 0..) |*dst, i| {
        const v = 0.85 + @abs(synthF32(seed_base + i, 0.02));
        dst.* = switch (dtype) {
            .f16 => @bitCast(@as(f16, @floatCast(v))),
            .bf16 => quant.f32ToBf16(v),
        };
    }
}

fn expectF16MatchesF32(expected: []const f32, actual_f16: []const f16, abs_tol: f32, rel_tol: f32) !void {
    try std.testing.expectEqual(expected.len, actual_f16.len);
    for (expected, actual_f16) |exp_v, act_h| {
        const act_v: f32 = @floatCast(act_h);
        try std.testing.expectApproxEqAbs(exp_v, act_v, @max(abs_tol, @abs(exp_v) * rel_tol));
    }
}

const TestMetalCtx = struct {
    device: metal.Device,
    library: metal.Library,
    pipes: Inkling2Hy4Pipelines,

    fn init() !TestMetalCtx {
        const device = try metal.Device.open();
        errdefer device.close();
        const library = try metal.Library.openEmbedded(device);
        errdefer library.close();
        const pipes = try Inkling2Hy4Pipelines.init(device, library);
        return .{ .device = device, .library = library, .pipes = pipes };
    }

    fn deinit(self: TestMetalCtx) void {
        self.pipes.close();
        self.library.close();
        self.device.close();
    }
};

test "gpu.inkling2_hy4. all 18 Metal 4 PSOs compile and wired arenas enforce 16 KiB alignment" {
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();

    const all_pipes = [_]metal.Pipeline{
        ctx.pipes.rmsnorm_sconv4[0],
        ctx.pipes.rmsnorm_sconv4[1],
        ctx.pipes.rel_gqa_attn_d64,
        ctx.pipes.rel_gqa_attn_d128,
        ctx.pipes.rel_gqa_attn_dyn,
        ctx.pipes.inkling2_moe_route,
        ctx.pipes.inkling2_moe_swiglu_softcap,
        ctx.pipes.inkling2_moe_route_softcap_epilogue,
        ctx.pipes.inkling2_attn_res[0],
        ctx.pipes.inkling2_attn_res[1],
        ctx.pipes.hy4_index_pool_compress,
        ctx.pipes.hy4_index_pool_topk,
        ctx.pipes.hy4_latent_moe_route,
        ctx.pipes.hy4_swiglu_clamped_linear_gate,
        ctx.pipes.hy4_combine_sandwich_norm[0],
        ctx.pipes.hy4_combine_sandwich_norm[1],
        ctx.pipes.hy4_mla_sink_attn_r512,
        ctx.pipes.hy4_mla_sink_attn_dyn,
    };
    for (all_pipes) |p| {
        try std.testing.expect(p.maxThreadsPerThreadgroup() >= 256);
    }

    const sconv_spec: SconvRingSpec = .{ .num_layers = 4, .hidden_dim = 1152 };
    const attn_res_spec: AttnResRingSpec = .{ .max_blocks = 6, .max_tokens = 33, .hidden_dim = 1152 };
    const idx_spec: IndexPoolSpec = .{
        .max_nodes = 33,
        .max_kv_len = 2048,
        .num_k_heads = 1,
        .idx_dim = 128,
        .compress_ratio = 4,
        .group_topk = 512,
        .candidate_topk = 2048,
    };
    const b0 = try SconvRingArena.requiredBytes(sconv_spec);
    const b1 = try AttnResRingArena.requiredBytes(attn_res_spec);
    const b2 = try IndexPoolBuffer.requiredBytes(idx_spec);
    try std.testing.expect(std.mem.isAligned(b0, APPLE_PAGE_SIZE));
    try std.testing.expect(std.mem.isAligned(b1, APPLE_PAGE_SIZE));
    try std.testing.expect(std.mem.isAligned(b2, APPLE_PAGE_SIZE));

    var arena = try makeTestArena(b0 + b1 + b2 + APPLE_PAGE_SIZE);
    defer arena.deinit();

    const sconv_ring = try SconvRingArena.init(&arena, sconv_spec);
    const attn_res_ring = try AttnResRingArena.init(&arena, attn_res_spec);
    const idx_pool = try IndexPoolBuffer.init(&arena, idx_spec);
    try page.checkShareable(sconv_ring.bytes().ptr, sconv_ring.bytes().len);
    try page.checkShareable(attn_res_ring.bytes().ptr, attn_res_ring.bytes().len);
    try page.checkShareable(idx_pool.bytes().ptr, idx_pool.bytes().len);
}

test "gpu.inkling2_hy4. rmsnorm_sconv4 static & dynamic parity across M=1 ring, M=8, and M=33 (1152 width)" {
    const gpa = std.testing.allocator;
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();

    const hidden_dim: u32 = 1152; // non-256-divisible width
    const m_cases = [_]u32{ 1, 8, 33 };
    const max_m: u32 = 33;

    const ring_bytes = try SconvRingArena.requiredBytes(.{ .num_layers = 2, .hidden_dim = hidden_dim });
    var arena = try makeTestArena(ring_bytes + 4 * 1024 * 1024);
    defer arena.deinit();

    var sconv_ring = try SconvRingArena.init(&arena, .{ .num_layers = 2, .hidden_dim = hidden_dim });
    const x_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const nw_raw = try arena.allocAligned(@as(usize, hidden_dim) * @sizeOf(u16), 64);
    const cw_raw = try arena.allocAligned(@as(usize, hidden_dim) * 4 * @sizeOf(f32), 64);
    const dw_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * 4 * @sizeOf(f16), 64);
    const bias_raw = try arena.allocAligned(@as(usize, hidden_dim) * @sizeOf(f32), 64);
    const out_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const nw_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, nw_raw));
    const cw_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, cw_raw));
    const dw_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, dw_raw));
    const bias_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bias_raw));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));

    fillSynthF16(x_f16, 17, 0.08);
    fillSynthNormU16(nw_u16, 101, .bf16);
    for (cw_f32, 0..) |*v, i| v.* = synthF32(311 + i, 0.06);
    fillSynthF16(dw_f16, 521, 0.05);
    for (bias_f32, 0..) |*v, i| v.* = synthF32(701 + i, 0.03);

    const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(ctx.device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    // 1) Multi-step M=1 in-place circular 3-step ring decode (static weights, bf16 norm).
    const l0_state = try sconv_ring.layerSlice(0);
    fillSynthF16(l0_state, 907, 0.05);
    const ref_state0 = try gpa.dupe(f16, l0_state);
    defer gpa.free(ref_state0);
    const ref_out = try gpa.alloc(f32, @as(usize, max_m) * hidden_dim);
    defer gpa.free(ref_out);

    for (0..4) |step| {
        const x_off = step * hidden_dim;
        try rmsnormSconv4CpuRef(
            gpa,
            x_f16[x_off..][0..hidden_dim],
            nw_u16,
            .bf16,
            cw_f32,
            null,
            null,
            bias_f32,
            ref_state0,
            ref_out[0..hidden_dim],
            1,
            hidden_dim,
            @intCast(step % INKLING2_SCONV_RING_SLOTS),
            true,
            true,
            false,
            1e-6,
        );
        const l0_bind = try sconv_ring.layerBinding(0, gpu_buf);
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try ctx.pipes.encodeRmsNormSconv4(&f, .{
            .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) + x_off * @sizeOf(f16) },
            .norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(nw_raw.ptr) },
            .norm_dtype = .bf16,
            .conv_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(cw_raw.ptr) },
            .conv_bias = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bias_raw.ptr) },
            .conv_state_in = l0_bind,
            .conv_state_out = l0_bind,
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .m = 1,
            .hidden_dim = hidden_dim,
            .state_pos = try sconv_ring.statePos(0),
            .write_state = true,
            .is_dynamic = false,
        }, false);
        try compute.wait(try f.commit(), 5000);
        try sconv_ring.advanceStatePos(0, 1);

        try expectF16MatchesF32(ref_out[0..hidden_dim], out_f16[0..hidden_dim], 0.01, 0.015);
    }
    for (ref_state0, l0_state) |r_h, g_h| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(r_h)), @as(f32, @floatCast(g_h)), 0.01);
    }

    // 2) M=1, M=8, M=33 across static and dynamic weight_proj modes (f16 norm).
    fillSynthNormU16(nw_u16, 1301, .f16);
    for (m_cases) |m_val| {
        for ([_]bool{ false, true }) |is_dyn| {
            const l1_state = try sconv_ring.layerSlice(1);
            fillSynthF16(l1_state, 1501 + m_val, 0.06);
            const ref_state1 = try gpa.dupe(f16, l1_state);
            defer gpa.free(ref_state1);

            const total_elems = @as(usize, m_val) * hidden_dim;
            try rmsnormSconv4CpuRef(
                gpa,
                x_f16[0..total_elems],
                nw_u16,
                .f16,
                if (is_dyn) null else cw_f32,
                null,
                if (is_dyn) dw_f16[0 .. total_elems * 4] else null,
                bias_f32,
                ref_state1,
                ref_out[0..total_elems],
                m_val,
                hidden_dim,
                1,
                true,
                true,
                is_dyn,
                1e-6,
            );

            const l1_bind = try sconv_ring.layerBinding(1, gpu_buf);
            var f = try compute.beginFrame(5000);
            errdefer f.abort();
            try ctx.pipes.encodeRmsNormSconv4(&f, .{
                .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
                .norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(nw_raw.ptr) },
                .norm_dtype = .f16,
                .conv_w = if (is_dyn) null else .{ .buffer = gpu_buf, .offset = arena.offsetOf(cw_raw.ptr) },
                .dyn_w = if (is_dyn) .{ .buffer = gpu_buf, .offset = arena.offsetOf(dw_raw.ptr) } else null,
                .conv_bias = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bias_raw.ptr) },
                .conv_state_in = l1_bind,
                .conv_state_out = l1_bind,
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
                .m = m_val,
                .hidden_dim = hidden_dim,
                .state_pos = 1,
                .write_state = true,
                .is_dynamic = is_dyn,
                .apply_silu = is_dyn,
            }, false);
            try compute.wait(try f.commit(), 5000);

            try expectF16MatchesF32(ref_out[0..total_elems], out_f16[0..total_elems], 0.015, 0.015);
            for (ref_state1, l1_state) |r_h, g_h| {
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(r_h)), @as(f32, @floatCast(g_h)), 0.01);
            }
        }
    }
}

test "gpu.inkling2_hy4. rel_gqa_attn fuses clip, log_scaling, d_rel=16, sink, softcap, gate_attn across M=1,8,33 and 1024-tok SWA ring" {
    const gpa = std.testing.allocator;
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();

    const num_q_heads: u32 = 8;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 128;
    const d_rel: u32 = DEFAULT_REL_DIM;
    const rel_extent: u32 = DEFAULT_REL_EXTENT;
    const kv_len: u32 = 1100; // exceeds 1024 -> exercises 1024-token circular HybridKvRing SWA wrap!
    const max_nodes: u32 = 33;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const kv_table_bytes = try hybrid.HybridKvRing.tableRequiredBytes(8);
    var arena = try makeTestArena(kv_table_bytes + 4 * 1024 * 1024);
    defer arena.deinit();

    var ring = try hybrid.HybridKvRing.init(gpa, ctx.device, &arena, .{
        .mode = .sliding_window_128,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .num_tiles = 8, // 8 * 128 = 1024-token circular SWA ring
    });
    defer ring.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 1024), ring.window_size);
    try std.testing.expectEqual(@as(u32, 1024), ring.ring_capacity);

    const q_raw = try arena.allocAligned(@as(usize, max_nodes) * num_q_heads * head_dim * @sizeOf(f16), 64);
    const q_rel_raw = try arena.allocAligned(@as(usize, max_nodes) * num_q_heads * d_rel * @sizeOf(f16), 64);
    const rel_emb_raw = try arena.allocAligned(@as(usize, rel_extent) * d_rel * @sizeOf(f16), 64);
    const k_tree_raw = try arena.allocAligned(@as(usize, 8) * num_kv_heads * head_dim * @sizeOf(f16), 64);
    const v_tree_raw = try arena.allocAligned(@as(usize, 8) * num_kv_heads * head_dim * @sizeOf(f16), 64);
    const masks_raw = try arena.allocAligned(@as(usize, 8) * @sizeOf(u64), 64);
    const sink_raw = try arena.allocAligned(@as(usize, max_nodes) * num_q_heads * @sizeOf(f32), 64);
    const gate_raw = try arena.allocAligned(@as(usize, max_nodes) * num_q_heads * head_dim * @sizeOf(f16), 64);
    const out_raw = try arena.allocAligned(@as(usize, max_nodes) * num_q_heads * head_dim * @sizeOf(f16), 64);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
    const q_rel_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_rel_raw));
    const rel_emb_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, rel_emb_raw));
    const k_tree_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_tree_raw));
    const v_tree_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_tree_raw));
    const masks_u64: []u64 = @alignCast(std.mem.bytesAsSlice(u64, masks_raw));
    const sink_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, sink_raw));
    const gate_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate_raw));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));

    // Use large scale (2.8) on Q, K, and rel vectors so qk_l2_clip (22.627) and rel_l2_clip (8.0) actively trigger!
    fillSynthF16(q_f16, 19, 0.25);
    fillSynthF16(q_rel_f16, 43, 0.45);
    fillSynthF16(rel_emb_f16, 89, 0.40);
    fillSynthF16(k_tree_f16, 137, 0.25);
    fillSynthF16(v_tree_f16, 179, 0.09);
    for (sink_f32, 0..) |*s, i| s.* = synthF32(223 + i, 0.08);
    fillSynthF16(gate_f16, 269, 0.12);

    const k_hist = try gpa.alloc(f16, @as(usize, kv_len) * num_kv_heads * head_dim);
    defer gpa.free(k_hist);
    const v_hist = try gpa.alloc(f16, @as(usize, kv_len) * num_kv_heads * head_dim);
    defer gpa.free(v_hist);
    fillSynthF16(k_hist, 313, 0.24);
    fillSynthF16(v_hist, 401, 0.09);

    // Populate the 1024-token circular HybridKvRing tiles via IOSurface lock.
    for (0..ring.k_tiles.len) |tile_i| {
        var kg = try ring.k_tiles[tile_i].lock(false);
        errdefer kg.release();
        var vg = try ring.v_tiles[tile_i].lock(false);
        errdefer vg.release();
        const k_stride = kg.bytes_per_row / @sizeOf(f16);
        const v_stride = vg.bytes_per_row / @sizeOf(f16);
        const k_dst: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
        const v_dst: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vg.bytes));

        for (0..kv_len) |t| {
            const slot = try ring.slotForPos(@intCast(t));
            if (slot.tile_idx != tile_i) continue;
            for (0..num_kv_heads) |kvh| {
                const row = kvh * ring.chunk_tokens + slot.slot_in_chunk;
                const src_off = (t * num_kv_heads + kvh) * head_dim;
                @memcpy(k_dst[row * k_stride ..][0..head_dim], k_hist[src_off..][0..head_dim]);
                @memcpy(v_dst[row * v_stride ..][0..head_dim], v_hist[src_off..][0..head_dim]);
            }
        }
        try vg.unlock();
        try kg.unlock();
    }

    // 8-node DFlare branching verification tree masks.
    const tree_masks = [8]u64{
        (1 << 0),
        (1 << 0) | (1 << 1),
        (1 << 0) | (1 << 1) | (1 << 2),
        (1 << 0) | (1 << 1) | (1 << 3),
        (1 << 0) | (1 << 4),
        (1 << 0) | (1 << 4) | (1 << 5),
        (1 << 0) | (1 << 4) | (1 << 5) | (1 << 6),
        (1 << 0) | (1 << 7),
    };
    @memcpy(masks_u64, &tree_masks);

    const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(ctx.device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try ring.makeResident(compute);

    const ref_out = try gpa.alloc(f32, @as(usize, max_nodes) * num_q_heads * head_dim);
    defer gpa.free(ref_out);

    const Case = struct { m: u32, tree: bool, causal_start: ?u32, win_override: ?u32 };
    const cases = [_]Case{
        .{ .m = 1, .tree = false, .causal_start = null, .win_override = null },
        .{ .m = 8, .tree = true, .causal_start = null, .win_override = null },
        // Wrapped circular SWA causal M=33 prefill: window_size=960 + (M-1)=32 = 992 <= 1024 ring_capacity,
        // ensuring zero circular overwrite across all 33 queries [1067..1099] inside the 1024-tok ring.
        .{ .m = 33, .tree = false, .causal_start = kv_len - 33, .win_override = 960 },
    };

    for (cases) |c| {
        const out_elems = @as(usize, c.m) * num_q_heads * head_dim;
        const orig_win = ring.window_size;
        if (c.win_override) |w| ring.window_size = w;
        defer ring.window_size = orig_win;

        try inkling2RelGqaAttnCpuRef(
            gpa,
            q_f16[0..out_elems],
            k_hist,
            v_hist,
            q_rel_f16[0 .. @as(usize, c.m) * num_q_heads * d_rel],
            rel_emb_f16,
            if (c.tree) k_tree_f16 else null,
            if (c.tree) v_tree_f16 else null,
            if (c.tree) masks_u64 else null,
            sink_f32,
            true,
            gate_f16[0..out_elems],
            .sigmoid,
            ref_out[0..out_elems],
            c.m,
            num_q_heads,
            num_kv_heads,
            head_dim,
            kv_len,
            ring.window_size,
            c.causal_start,
            d_rel,
            rel_extent,
            false,
            true,
            1.0,
            scale,
            INKLING2_DEFAULT_QK_L2_CLIP,
            INKLING2_DEFAULT_REL_L2_CLIP,
            0.5,
            25.0,
        );

        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try ctx.pipes.encodeInkling2RelGqaAttn(&f, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .q_rel = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_rel_raw.ptr) },
            .rel_emb = .{ .buffer = gpu_buf, .offset = arena.offsetOf(rel_emb_raw.ptr) },
            .kv_ring = &ring,
            .kv_ring_buf = gpu_buf,
            .k_tree = if (c.tree) .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_tree_raw.ptr) } else null,
            .v_tree = if (c.tree) .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_tree_raw.ptr) } else null,
            .ancestor_masks = if (c.tree) .{ .buffer = gpu_buf, .offset = arena.offsetOf(masks_raw.ptr) } else null,
            .sink_logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sink_raw.ptr) },
            .sink_per_token = true,
            .gate_attn = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gate_raw.ptr) },
            .gate_mode = .sigmoid,
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .num_nodes = c.m,
            .num_q_heads = num_q_heads,
            .kv_len = kv_len,
            .causal_q_start_pos = c.causal_start,
            .d_rel = d_rel,
            .rel_extent = rel_extent,
            .use_log_scaling = true,
            .log_scale_alpha = 1.0,
            .scale = scale,
            .qk_l2_clip = INKLING2_DEFAULT_QK_L2_CLIP,
            .rel_l2_clip = INKLING2_DEFAULT_REL_L2_CLIP,
            .rel_scale = 0.5,
            .attn_softcap = 25.0,
        }, false);
        try compute.wait(try f.commit(), 5000);

        try expectF16MatchesF32(ref_out[0..out_elems], out_f16[0..out_elems], 0.015, 0.015);
    }
}

test "gpu.inkling2_hy4. Inkling2 MoE route (E=128/384), SwiGLU softcap (1152 width), epilogue, and 6-layer attn_res parity" {
    const gpa = std.testing.allocator;
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();

    const max_m: u32 = 33;
    const max_e: u32 = 384;
    const top_k: u32 = 8;
    const hidden_dim: u32 = 1152; // non-256-divisible width

    const attn_res_bytes = try AttnResRingArena.requiredBytes(.{ .max_blocks = 6, .max_tokens = max_m, .hidden_dim = hidden_dim });
    var arena = try makeTestArena(attn_res_bytes + 8 * 1024 * 1024);
    defer arena.deinit();

    var attn_res_ring = try AttnResRingArena.init(&arena, .{ .max_blocks = 6, .max_tokens = max_m, .hidden_dim = hidden_dim });
    for (0..8) |step| {
        const slot = attn_res_ring.pushSlot();
        const blk_slice = try attn_res_ring.blockSlice(slot);
        fillSynthF16(blk_slice, 100 + step * 500, 0.07);
    }

    const logits_raw = try arena.allocAligned(@as(usize, max_m) * max_e * @sizeOf(f32), 64);
    const bias_raw = try arena.allocAligned(@as(usize, max_e) * @sizeOf(f32), 64);
    const exp_ids_raw = try arena.allocAligned(@as(usize, max_m) * top_k * @sizeOf(u32), 64);
    const exp_w_raw = try arena.allocAligned(@as(usize, max_m) * top_k * @sizeOf(f32), 64);
    const gate_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const up_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const swiglu_out_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const expert_out_raw = try arena.allocAligned(@as(usize, max_m) * top_k * hidden_dim * @sizeOf(f16), 64);
    const res_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const epi_out_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const query_raw = try arena.allocAligned(@as(usize, hidden_dim) * @sizeOf(f16), 64);
    const nw_raw = try arena.allocAligned(@as(usize, hidden_dim) * @sizeOf(u16), 64);
    const ar_out_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);

    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
    const bias_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bias_raw));
    const exp_ids: []u32 = @alignCast(std.mem.bytesAsSlice(u32, exp_ids_raw));
    const exp_w: []f32 = @alignCast(std.mem.bytesAsSlice(f32, exp_w_raw));
    const gate_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate_raw));
    const up_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, up_raw));
    const swiglu_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, swiglu_out_raw));
    const expert_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, expert_out_raw));
    const res_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, res_raw));
    const epi_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, epi_out_raw));
    const query_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, query_raw));
    const nw_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, nw_raw));
    const ar_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, ar_out_raw));

    for (logits_f32, 0..) |*v, i| v.* = synthF32(23 + i, 0.35);
    for (bias_f32, 0..) |*v, i| v.* = synthF32(59 + i, 0.08);
    fillSynthF16(gate_f16, 97, 0.45);
    fillSynthF16(up_f16, 131, 0.45);
    fillSynthF16(expert_out_f16, 173, 0.12);
    fillSynthF16(res_f16, 211, 0.09);
    fillSynthF16(query_f16, 251, 0.10);
    fillSynthNormU16(nw_u16, 293, .bf16);

    const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(ctx.device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const ref_ids = try gpa.alloc(u32, @as(usize, max_m) * top_k);
    defer gpa.free(ref_ids);
    const ref_w = try gpa.alloc(f32, @as(usize, max_m) * top_k);
    defer gpa.free(ref_w);
    const ref_buf = try gpa.alloc(f32, @as(usize, max_m) * hidden_dim);
    defer gpa.free(ref_buf);

    var slots: [MAX_ATTN_RES_BLOCKS]u32 = @splat(0);
    const active_blocks = attn_res_ring.activeSlots(&slots);
    try std.testing.expectEqual(@as(u32, 6), active_blocks);

    for ([_]u32{ 1, 8, 33 }) |m_val| {
        for ([_]u32{ 128, 384 }) |num_e| {
            try inkling2MoeRouteCpuRef(
                gpa,
                logits_f32[0 .. @as(usize, m_val) * num_e],
                null,
                bias_f32[0..num_e],
                null,
                true,
                ref_ids[0 .. @as(usize, m_val) * top_k],
                ref_w[0 .. @as(usize, m_val) * top_k],
                m_val,
                num_e,
                top_k,
                true,
                INKLING2_DEFAULT_ROUTE_SCALE,
                INKLING2_DEFAULT_ROUTER_SOFTCAP,
            );
            var f = try compute.beginFrame(5000);
            errdefer f.abort();
            try ctx.pipes.encodeInkling2MoeRoute(&f, .{
                .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) },
                .bias = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bias_raw.ptr) },
                .topk_experts = .{ .buffer = gpu_buf, .offset = arena.offsetOf(exp_ids_raw.ptr) },
                .topk_weights = .{ .buffer = gpu_buf, .offset = arena.offsetOf(exp_w_raw.ptr) },
                .m = m_val,
                .num_experts = num_e,
                .top_k = top_k,
            }, false);
            try compute.wait(try f.commit(), 5000);

            for (0..@as(usize, m_val) * top_k) |i| {
                try std.testing.expectEqual(ref_ids[i], exp_ids[i]);
                try std.testing.expectApproxEqAbs(ref_w[i], exp_w[i], 1e-4);
            }
        }

        const elem_cnt = @as(usize, m_val) * hidden_dim;
        inkling2MoeSwigluSoftcapCpuRef(gate_f16[0..elem_cnt], up_f16[0..elem_cnt], ref_buf[0..elem_cnt], m_val, hidden_dim, false, 30.0);
        {
            var f = try compute.beginFrame(5000);
            errdefer f.abort();
            try ctx.pipes.encodeInkling2MoeSwigluSoftcap(&f, .{
                .gate_in = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gate_raw.ptr) },
                .up_in = .{ .buffer = gpu_buf, .offset = arena.offsetOf(up_raw.ptr) },
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(swiglu_out_raw.ptr) },
                .num_rows = m_val,
                .intermediate_size = hidden_dim,
                .softcap = 30.0,
            }, false);
            try compute.wait(try f.commit(), 5000);
            try expectF16MatchesF32(ref_buf[0..elem_cnt], swiglu_out_f16[0..elem_cnt], 0.02, 0.015);
        }

        inkling2MoeEpilogueCpuRef(
            expert_out_f16[0 .. elem_cnt * top_k],
            exp_w[0 .. @as(usize, m_val) * top_k],
            null,
            res_f16[0..elem_cnt],
            ref_buf[0..elem_cnt],
            m_val,
            hidden_dim,
            top_k,
            25.0,
            1.0,
        );
        {
            var f = try compute.beginFrame(5000);
            errdefer f.abort();
            try ctx.pipes.encodeInkling2MoeEpilogue(&f, .{
                .expert_out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(expert_out_raw.ptr) },
                .topk_weights = .{ .buffer = gpu_buf, .offset = arena.offsetOf(exp_w_raw.ptr) },
                .residual = .{ .buffer = gpu_buf, .offset = arena.offsetOf(res_raw.ptr) },
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(epi_out_raw.ptr) },
                .m = m_val,
                .hidden_dim = hidden_dim,
                .top_k = top_k,
                .mlp_softcap = 25.0,
            }, false);
            try compute.wait(try f.commit(), 5000);
            try expectF16MatchesF32(ref_buf[0..elem_cnt], epi_out_f16[0..elem_cnt], 0.02, 0.015);
        }

        for ([_]bool{ false, true }) |norm_vals| {
            inkling2AttnResCpuRef(
                @alignCast(std.mem.bytesAsSlice(f16, attn_res_ring.bytes())),
                slots[0..active_blocks],
                attn_res_ring.block_stride_elems,
                query_f16,
                nw_u16,
                .bf16,
                ref_buf[0..elem_cnt],
                m_val,
                hidden_dim,
                true,
                norm_vals,
                false,
                1e-6,
                1.0,
            );
            {
                var f = try compute.beginFrame(5000);
                errdefer f.abort();
                try ctx.pipes.encodeInkling2AttnRes(&f, .{
                    .ring = &attn_res_ring,
                    .ring_buf = gpu_buf,
                    .query = .{ .buffer = gpu_buf, .offset = arena.offsetOf(query_raw.ptr) },
                    .norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(nw_raw.ptr) },
                    .norm_dtype = .bf16,
                    .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(ar_out_raw.ptr) },
                    .m = m_val,
                    .normalize_values = norm_vals,
                }, false);
                try compute.wait(try f.commit(), 5000);
                try expectF16MatchesF32(ref_buf[0..elem_cnt], ar_out_f16[0..elem_cnt], 0.01, 0.015);
            }
        }
    }
}

test "gpu.inkling2_hy4. Hy4 IndexCache period-4 reuse, IndexPool compress_ratio=4 (tiny & 512/2048), and MLA sink attn parity" {
    const gpa = std.testing.allocator;
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();

    // Part A: Tiny/synthetic dims across M=1, M=8 (tree verify), M=33 (causal prefill),
    // verifying IndexCache period-4 (full vs shared zero-dispatch) + Hy4 MLA Sink Attn!
    {
        const max_nodes: u32 = 33;
        const kv_len: u32 = 160;
        const swa_window: u32 = 32;
        const num_idx_heads: u32 = 4;
        const idx_dim: u32 = 64;
        const compress_ratio: u32 = 4;
        const group_topk: u32 = 8;
        const candidate_topk: u32 = 16;
        const num_q_heads: u32 = 8;
        const kv_lora_rank: u32 = 512;
        const qk_rope_dim: u32 = 64;
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_lora_rank + qk_rope_dim)));
        const idx_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(idx_dim)));

        const idx_bytes = try IndexPoolBuffer.requiredBytes(.{
            .max_nodes = max_nodes,
            .max_kv_len = kv_len,
            .num_k_heads = 1,
            .idx_dim = idx_dim,
            .compress_ratio = compress_ratio,
            .group_topk = group_topk,
            .candidate_topk = candidate_topk,
        });
        var arena = try makeTestArena(idx_bytes + 4 * 1024 * 1024);
        defer arena.deinit();

        const idx_pool = try IndexPoolBuffer.init(&arena, .{
            .max_nodes = max_nodes,
            .max_kv_len = kv_len,
            .num_k_heads = 1,
            .idx_dim = idx_dim,
            .compress_ratio = compress_ratio,
            .group_topk = group_topk,
            .candidate_topk = candidate_topk,
        });

        const q_idx_raw = try arena.allocAligned(@as(usize, max_nodes) * num_idx_heads * idx_dim * @sizeOf(f16), 64);
        const hw_raw = try arena.allocAligned(@as(usize, max_nodes) * num_idx_heads * @sizeOf(f16), 64);
        const k_fine_raw = try arena.allocAligned(@as(usize, kv_len) * idx_dim * @sizeOf(f16), 64);
        const q_nope_raw = try arena.allocAligned(@as(usize, max_nodes) * num_q_heads * kv_lora_rank * @sizeOf(f16), 64);
        const q_pe_raw = try arena.allocAligned(@as(usize, max_nodes) * num_q_heads * qk_rope_dim * @sizeOf(f16), 64);
        const kv_lat_raw = try arena.allocAligned(@as(usize, kv_len) * kv_lora_rank * @sizeOf(f16), 64);
        const kv_rope_raw = try arena.allocAligned(@as(usize, kv_len) * qk_rope_dim * @sizeOf(f16), 64);
        const tree_lat_raw = try arena.allocAligned(@as(usize, 8) * kv_lora_rank * @sizeOf(f16), 64);
        const tree_rope_raw = try arena.allocAligned(@as(usize, 8) * qk_rope_dim * @sizeOf(f16), 64);
        const masks_raw = try arena.allocAligned(@as(usize, 8) * @sizeOf(u64), 64);
        const sink_raw = try arena.allocAligned(@as(usize, num_q_heads) * @sizeOf(f32), 64);
        const lg_raw = try arena.allocAligned(@as(usize, max_nodes) * num_q_heads * kv_lora_rank * @sizeOf(f16), 64);
        const out_raw = try arena.allocAligned(@as(usize, max_nodes) * num_q_heads * kv_lora_rank * @sizeOf(f16), 64);

        const q_idx_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_idx_raw));
        const hw_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, hw_raw));
        const k_fine_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_fine_raw));
        const q_nope_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_nope_raw));
        const q_pe_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_pe_raw));
        const kv_lat_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kv_lat_raw));
        const kv_rope_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kv_rope_raw));
        const tree_lat_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, tree_lat_raw));
        const tree_rope_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, tree_rope_raw));
        const masks_u64: []u64 = @alignCast(std.mem.bytesAsSlice(u64, masks_raw));
        const sink_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, sink_raw));
        const lg_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, lg_raw));
        const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));

        fillSynthF16(q_idx_f16, 11, 0.14);
        fillSynthF16(hw_f16, 37, 0.08);
        fillSynthF16(k_fine_f16, 71, 0.13);
        fillSynthF16(q_nope_f16, 109, 0.07);
        fillSynthF16(q_pe_f16, 151, 0.08);
        fillSynthF16(kv_lat_f16, 193, 0.07);
        fillSynthF16(kv_rope_f16, 241, 0.08);
        fillSynthF16(tree_lat_f16, 281, 0.07);
        fillSynthF16(tree_rope_f16, 331, 0.08);
        for (sink_f32, 0..) |*s, i| s.* = synthF32(389 + i, 0.10);
        fillSynthF16(lg_f16, 421, 0.11);

        const tree_masks = [8]u64{
            (1 << 0),
            (1 << 0) | (1 << 1),
            (1 << 0) | (1 << 1) | (1 << 2),
            (1 << 0) | (1 << 1) | (1 << 3),
            (1 << 0) | (1 << 4),
            (1 << 0) | (1 << 4) | (1 << 5),
            (1 << 0) | (1 << 4) | (1 << 5) | (1 << 6),
            (1 << 0) | (1 << 7),
        };
        @memcpy(masks_u64, &tree_masks);

        const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(ctx.device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        const ref_k_pool = try gpa.alloc(f16, idx_pool.k_pool.len);
        defer gpa.free(ref_k_pool);
        const ref_sel = try gpa.alloc(u32, @as(usize, max_nodes) * candidate_topk);
        defer gpa.free(ref_sel);
        const ref_cnt = try gpa.alloc(u32, max_nodes);
        defer gpa.free(ref_cnt);
        const ref_attn = try gpa.alloc(f32, @as(usize, max_nodes) * num_q_heads * kv_lora_rank);
        defer gpa.free(ref_attn);

        // Verify period-4 IndexCache: Layer 0 = .full (2 dispatches), Layers 1..3 = .shared (0 dispatches)!
        try std.testing.expectEqual(IndexCacheMode.full, IndexCacheMode.forPeriod4(0));
        try std.testing.expectEqual(IndexCacheMode.shared, IndexCacheMode.forPeriod4(1));
        try std.testing.expectEqual(IndexCacheMode.shared, IndexCacheMode.forPeriod4(2));
        try std.testing.expectEqual(IndexCacheMode.shared, IndexCacheMode.forPeriod4(3));
        try std.testing.expectEqual(IndexCacheMode.full, IndexCacheMode.forPeriod4(4));

        const AttnCase = struct { m: u32, tree: bool, causal_start: ?u32, pool_mode: IndexPoolMode };
        const cases = [_]AttnCase{
            .{ .m = 1, .tree = false, .causal_start = null, .pool_mode = .mean },
            .{ .m = 8, .tree = true, .causal_start = null, .pool_mode = .max },
            .{ .m = 33, .tree = false, .causal_start = kv_len - 33, .pool_mode = .mean },
        };

        for (cases) |c| {
            hy4IndexPoolCompressCpuRef(k_fine_f16, ref_k_pool, kv_len, 1, idx_dim, compress_ratio, c.pool_mode);
            try hy4IndexPoolTopkCpuRef(
                gpa,
                q_idx_f16[0 .. @as(usize, c.m) * num_idx_heads * idx_dim],
                hw_f16[0 .. @as(usize, c.m) * num_idx_heads],
                k_fine_f16,
                ref_k_pool,
                ref_sel[0 .. @as(usize, c.m) * candidate_topk],
                ref_cnt[0..c.m],
                c.m,
                num_idx_heads,
                1,
                idx_dim,
                kv_len,
                c.causal_start,
                compress_ratio,
                group_topk,
                candidate_topk,
                .relu_weighted_sum,
                swa_window,
                idx_scale,
            );

            // Full layer 0 runs compress + topk; shared layer 1 emits 0 indexer dispatches!
            var f0 = try compute.beginFrame(5000);
            errdefer f0.abort();
            const d0 = try ctx.pipes.encodeHy4IndexLayer(&f0, .{
                .mode = IndexCacheMode.forPeriod4(0),
                .compress_keys = true,
                .pool_mode = c.pool_mode,
                .q_idx = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_idx_raw.ptr) },
                .head_weights = .{ .buffer = gpu_buf, .offset = arena.offsetOf(hw_raw.ptr) },
                .k_fine = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_fine_raw.ptr) },
                .index_pool = &idx_pool,
                .index_metal_buf = gpu_buf,
                .num_nodes = c.m,
                .num_q_heads = num_idx_heads,
                .kv_len = kv_len,
                .causal_q_start_pos = c.causal_start,
                .swa_window = swa_window,
                .scale = idx_scale,
            }, false);
            try std.testing.expectEqual(@as(u32, 1), d0.compress_dispatches);
            try std.testing.expectEqual(@as(u32, 1), d0.topk_dispatches);

            const d1 = try ctx.pipes.encodeHy4IndexLayer(&f0, .{
                .mode = IndexCacheMode.forPeriod4(1),
                .compress_keys = true,
                .pool_mode = c.pool_mode,
                .q_idx = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_idx_raw.ptr) },
                .head_weights = .{ .buffer = gpu_buf, .offset = arena.offsetOf(hw_raw.ptr) },
                .k_fine = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_fine_raw.ptr) },
                .index_pool = &idx_pool,
                .index_metal_buf = gpu_buf,
                .num_nodes = c.m,
                .num_q_heads = num_idx_heads,
                .kv_len = kv_len,
                .causal_q_start_pos = c.causal_start,
                .swa_window = swa_window,
                .scale = idx_scale,
            }, true);
            try std.testing.expectEqual(@as(u32, 0), d1.total());

            try ctx.pipes.encodeHy4MlaSinkAttn(&f0, .{
                .q_nope = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_nope_raw.ptr) },
                .q_pe = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_pe_raw.ptr) },
                .kv_latent = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kv_lat_raw.ptr) },
                .kv_rope = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kv_rope_raw.ptr) },
                .index_pool = &idx_pool,
                .index_metal_buf = gpu_buf,
                .kv_tree_latent = if (c.tree) .{ .buffer = gpu_buf, .offset = arena.offsetOf(tree_lat_raw.ptr) } else null,
                .kv_tree_rope = if (c.tree) .{ .buffer = gpu_buf, .offset = arena.offsetOf(tree_rope_raw.ptr) } else null,
                .ancestor_masks = if (c.tree) .{ .buffer = gpu_buf, .offset = arena.offsetOf(masks_raw.ptr) } else null,
                .sink_params = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sink_raw.ptr) },
                .linear_gate = .{ .buffer = gpu_buf, .offset = arena.offsetOf(lg_raw.ptr) },
                .linear_gate_mode = .sigmoid,
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
                .num_nodes = c.m,
                .num_q_heads = num_q_heads,
                .kv_lora_rank = kv_lora_rank,
                .qk_rope_dim = qk_rope_dim,
                .kv_len = kv_len,
                .causal_q_start_pos = c.causal_start,
                .swa_window = swa_window,
                .scale = attn_scale,
            }, true);
            try compute.wait(try f0.commit(), 5000);

            for (0..c.m) |node| {
                try std.testing.expectEqual(ref_cnt[node], idx_pool.selected_counts[node]);
                const cnt = ref_cnt[node];
                for (0..cnt) |si| {
                    try std.testing.expectEqual(ref_sel[node * candidate_topk + si], idx_pool.selected_tokens[node * candidate_topk + si]);
                }
            }

            const out_elems = @as(usize, c.m) * num_q_heads * kv_lora_rank;
            try hy4MlaSinkAttnCpuRef(
                gpa,
                q_nope_f16[0..out_elems],
                q_pe_f16[0 .. @as(usize, c.m) * num_q_heads * qk_rope_dim],
                kv_lat_f16,
                kv_rope_f16,
                idx_pool.selected_tokens,
                idx_pool.selected_counts,
                if (c.tree) tree_lat_f16 else null,
                if (c.tree) tree_rope_f16 else null,
                if (c.tree) masks_u64 else null,
                sink_f32,
                lg_f16[0..out_elems],
                .sigmoid,
                ref_attn[0..out_elems],
                c.m,
                num_q_heads,
                kv_lora_rank,
                qk_rope_dim,
                kv_len,
                c.causal_start,
                swa_window,
                candidate_topk,
                attn_scale,
                0.0,
            );
            try expectF16MatchesF32(ref_attn[0..out_elems], out_f16[0..out_elems], 0.01, 0.015);
        }
    }

    // Part B: Full Hy4-flash IndexPool dimensions (`group_topk = 512`, `candidate_topk = 2048`,
    // `compress_ratio = 4`, `idx_dim = 128`, `num_q_heads = 32`) on `kv_len = 3072` (768 groups -> 512 -> 2048)!
    {
        const kv_len: u32 = 3072;
        const num_q_heads: u32 = 32;
        const idx_dim: u32 = 128;
        const idx_bytes = try IndexPoolBuffer.requiredBytes(.{
            .max_nodes = 1,
            .max_kv_len = kv_len,
            .num_k_heads = 1,
            .idx_dim = idx_dim,
            .compress_ratio = HY4_DEFAULT_COMPRESS_RATIO,
            .group_topk = HY4_DEFAULT_GROUP_TOPK,
            .candidate_topk = HY4_DEFAULT_CANDIDATE_TOPK,
        });
        var arena = try makeTestArena(idx_bytes + 2 * 1024 * 1024);
        defer arena.deinit();

        const idx_pool = try IndexPoolBuffer.init(&arena, .{
            .max_nodes = 1,
            .max_kv_len = kv_len,
            .num_k_heads = 1,
            .idx_dim = idx_dim,
            .compress_ratio = HY4_DEFAULT_COMPRESS_RATIO,
            .group_topk = HY4_DEFAULT_GROUP_TOPK,
            .candidate_topk = HY4_DEFAULT_CANDIDATE_TOPK,
        });
        const q_raw = try arena.allocAligned(@as(usize, num_q_heads) * idx_dim * @sizeOf(f16), 64);
        const hw_raw = try arena.allocAligned(@as(usize, num_q_heads) * @sizeOf(f16), 64);
        const kf_raw = try arena.allocAligned(@as(usize, kv_len) * idx_dim * @sizeOf(f16), 64);
        const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
        const hw_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, hw_raw));
        const kf_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kf_raw));
        fillSynthF16(q_f16, 17, 0.10);
        fillSynthF16(hw_f16, 53, 0.06);
        fillSynthF16(kf_f16, 97, 0.10);

        const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(ctx.device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        const disp = try ctx.pipes.encodeHy4IndexLayer(&f, .{
            .mode = .full,
            .compress_keys = true,
            .pool_mode = .mean,
            .q_idx = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .head_weights = .{ .buffer = gpu_buf, .offset = arena.offsetOf(hw_raw.ptr) },
            .k_fine = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kf_raw.ptr) },
            .index_pool = &idx_pool,
            .index_metal_buf = gpu_buf,
            .num_nodes = 1,
            .num_q_heads = num_q_heads,
            .kv_len = kv_len,
            .score_mode = .relu_weighted_sum,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(idx_dim))),
        }, false);
        try std.testing.expectEqual(@as(u32, 2), disp.total());
        try compute.wait(try f.commit(), 5000);

        const ref_k_pool = try gpa.alloc(f16, idx_pool.k_pool.len);
        defer gpa.free(ref_k_pool);
        const ref_sel = try gpa.alloc(u32, HY4_DEFAULT_CANDIDATE_TOPK);
        defer gpa.free(ref_sel);
        var ref_cnt: [1]u32 = .{0};
        hy4IndexPoolCompressCpuRef(kf_f16, ref_k_pool, kv_len, 1, idx_dim, HY4_DEFAULT_COMPRESS_RATIO, .mean);
        try hy4IndexPoolTopkCpuRef(
            gpa,
            q_f16,
            hw_f16,
            kf_f16,
            ref_k_pool,
            ref_sel,
            &ref_cnt,
            1,
            num_q_heads,
            1,
            idx_dim,
            kv_len,
            null,
            HY4_DEFAULT_COMPRESS_RATIO,
            HY4_DEFAULT_GROUP_TOPK,
            HY4_DEFAULT_CANDIDATE_TOPK,
            .relu_weighted_sum,
            0,
            1.0 / @sqrt(@as(f32, @floatFromInt(idx_dim))),
        );
        try std.testing.expectEqual(ref_cnt[0], idx_pool.selected_counts[0]);
        for (0..ref_cnt[0]) |i| {
            try std.testing.expectEqual(ref_sel[i], idx_pool.selected_tokens[i]);
        }
    }
}

test "gpu.inkling2_hy4. Hy4 LatentMoE route (E=512), SwiGLU clamped (10.0 + linear_gate), and combine_sandwich_norm parity across M=1,8,33" {
    const gpa = std.testing.allocator;
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();

    const max_m: u32 = 33;
    const num_experts: u32 = 512;
    const top_k: u32 = 8;
    const hidden_dim: u32 = 1152; // non-256-divisible width

    var arena = try makeTestArena(8 * 1024 * 1024);
    defer arena.deinit();

    const logits_raw = try arena.allocAligned(@as(usize, max_m) * num_experts * @sizeOf(f32), 64);
    const bias_raw = try arena.allocAligned(@as(usize, num_experts) * @sizeOf(f32), 64);
    const exp_ids_raw = try arena.allocAligned(@as(usize, max_m) * top_k * @sizeOf(u32), 64);
    const exp_w_raw = try arena.allocAligned(@as(usize, max_m) * top_k * @sizeOf(f32), 64);
    const gate_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const up_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const lg_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const swiglu_out_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const exp_out_raw = try arena.allocAligned(@as(usize, max_m) * top_k * hidden_dim * @sizeOf(f16), 64);
    const shared_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const res_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);
    const cnw_raw = try arena.allocAligned(@as(usize, hidden_dim) * @sizeOf(u16), 64);
    const pnw_raw = try arena.allocAligned(@as(usize, hidden_dim) * @sizeOf(u16), 64);
    const comb_out_raw = try arena.allocAligned(@as(usize, max_m) * hidden_dim * @sizeOf(f16), 64);

    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
    const bias_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bias_raw));
    const exp_ids: []u32 = @alignCast(std.mem.bytesAsSlice(u32, exp_ids_raw));
    const exp_w: []f32 = @alignCast(std.mem.bytesAsSlice(f32, exp_w_raw));
    const gate_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate_raw));
    const up_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, up_raw));
    const lg_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, lg_raw));
    const swiglu_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, swiglu_out_raw));
    const exp_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, exp_out_raw));
    const shared_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, shared_raw));
    const res_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, res_raw));
    const cnw_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, cnw_raw));
    const pnw_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, pnw_raw));
    const comb_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, comb_out_raw));

    for (logits_f32, 0..) |*v, i| v.* = synthF32(13 + i, 0.20);
    for (bias_f32, 0..) |*v, i| v.* = synthF32(67 + i, 0.04);
    // Include values exceeding +-10.0 to exercise swiglu_limit = 10.0 gate/up clamping!
    fillSynthF16(gate_f16, 107, 0.85);
    fillSynthF16(up_f16, 151, 0.85);
    fillSynthF16(lg_f16, 197, 0.12);
    fillSynthF16(exp_out_f16, 239, 0.08);
    fillSynthF16(shared_f16, 283, 0.07);
    fillSynthF16(res_f16, 331, 0.09);
    fillSynthNormU16(cnw_u16, 373, .bf16);
    fillSynthNormU16(pnw_u16, 409, .bf16);

    const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(ctx.device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const ref_ids = try gpa.alloc(u32, @as(usize, max_m) * top_k);
    defer gpa.free(ref_ids);
    const ref_w = try gpa.alloc(f32, @as(usize, max_m) * top_k);
    defer gpa.free(ref_w);
    const ref_buf = try gpa.alloc(f32, @as(usize, max_m) * hidden_dim);
    defer gpa.free(ref_buf);

    for ([_]u32{ 1, 8, 33 }) |m_val| {
        try hy4LatentMoeRouteCpuRef(
            gpa,
            logits_f32[0 .. @as(usize, m_val) * num_experts],
            null,
            bias_f32,
            null,
            ref_ids[0 .. @as(usize, m_val) * top_k],
            ref_w[0 .. @as(usize, m_val) * top_k],
            m_val,
            num_experts,
            top_k,
            true,
            HY4_DEFAULT_ROUTED_SCALING_FACTOR,
            1e-20,
        );
        {
            var f = try compute.beginFrame(5000);
            errdefer f.abort();
            try ctx.pipes.encodeHy4LatentMoeRoute(&f, .{
                .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) },
                .e_score_correction_bias = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bias_raw.ptr) },
                .topk_experts = .{ .buffer = gpu_buf, .offset = arena.offsetOf(exp_ids_raw.ptr) },
                .topk_weights = .{ .buffer = gpu_buf, .offset = arena.offsetOf(exp_w_raw.ptr) },
                .m = m_val,
                .num_experts = num_experts,
                .top_k = top_k,
            }, false);
            try compute.wait(try f.commit(), 5000);
            for (0..@as(usize, m_val) * top_k) |i| {
                try std.testing.expectEqual(ref_ids[i], exp_ids[i]);
                try std.testing.expectApproxEqAbs(ref_w[i], exp_w[i], 1e-4);
            }
        }

        const elem_cnt = @as(usize, m_val) * hidden_dim;
        hy4SwigluClampedLinearGateCpuRef(
            gate_f16[0..elem_cnt],
            up_f16[0..elem_cnt],
            lg_f16[0..elem_cnt],
            .sigmoid,
            ref_buf[0..elem_cnt],
            m_val,
            hidden_dim,
            false,
            .gate_and_up,
            HY4_DEFAULT_SWIGLU_LIMIT,
        );
        {
            var f = try compute.beginFrame(5000);
            errdefer f.abort();
            try ctx.pipes.encodeHy4SwigluClampedLinearGate(&f, .{
                .gate_in = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gate_raw.ptr) },
                .up_in = .{ .buffer = gpu_buf, .offset = arena.offsetOf(up_raw.ptr) },
                .linear_gate = .{ .buffer = gpu_buf, .offset = arena.offsetOf(lg_raw.ptr) },
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(swiglu_out_raw.ptr) },
                .num_rows = m_val,
                .intermediate_size = hidden_dim,
            }, false);
            try compute.wait(try f.commit(), 5000);
            try expectF16MatchesF32(ref_buf[0..elem_cnt], swiglu_out_f16[0..elem_cnt], 0.03, 0.015);
        }

        try hy4CombineSandwichNormCpuRef(
            gpa,
            exp_out_f16[0 .. elem_cnt * top_k],
            exp_w[0 .. @as(usize, m_val) * top_k],
            null,
            shared_f16[0..elem_cnt],
            res_f16[0..elem_cnt],
            cnw_u16,
            pnw_u16,
            .bf16,
            ref_buf[0..elem_cnt],
            m_val,
            hidden_dim,
            top_k,
            1e-5,
            1.0,
        );
        {
            var f = try compute.beginFrame(5000);
            errdefer f.abort();
            try ctx.pipes.encodeHy4CombineSandwichNorm(&f, .{
                .expert_or_x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(exp_out_raw.ptr) },
                .topk_weights = .{ .buffer = gpu_buf, .offset = arena.offsetOf(exp_w_raw.ptr) },
                .shared_out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(shared_raw.ptr) },
                .residual = .{ .buffer = gpu_buf, .offset = arena.offsetOf(res_raw.ptr) },
                .combine_norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(cnw_raw.ptr) },
                .post_norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(pnw_raw.ptr) },
                .norm_dtype = .bf16,
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(comb_out_raw.ptr) },
                .m = m_val,
                .hidden_dim = hidden_dim,
                .top_k = top_k,
            }, false);
            try compute.wait(try f.commit(), 5000);
            try expectF16MatchesF32(ref_buf[0..elem_cnt], comb_out_f16[0..elem_cnt], 0.02, 0.015);
        }
    }
}

test "gpu.inkling2_hy4. hot-path encoding performs zero heap allocations" {
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();

    const sconv_bytes = try SconvRingArena.requiredBytes(.{ .num_layers = 1, .hidden_dim = 1152 });
    const idx_bytes = try IndexPoolBuffer.requiredBytes(.{
        .max_nodes = 8,
        .max_kv_len = 64,
        .num_k_heads = 1,
        .idx_dim = 64,
        .compress_ratio = 4,
        .group_topk = 4,
        .candidate_topk = 8,
    });
    var arena = try makeTestArena(sconv_bytes + idx_bytes + 512 * 1024);
    defer arena.deinit();

    var sconv_ring = try SconvRingArena.init(&arena, .{ .num_layers = 1, .hidden_dim = 1152 });
    const idx_pool = try IndexPoolBuffer.init(&arena, .{
        .max_nodes = 8,
        .max_kv_len = 64,
        .num_k_heads = 1,
        .idx_dim = 64,
        .compress_ratio = 4,
        .group_topk = 4,
        .candidate_topk = 8,
    });
    const x_raw = try arena.allocAligned(8 * 1152 * @sizeOf(f16), 64);
    const nw_raw = try arena.allocAligned(1152 * @sizeOf(f16), 64);
    const cw_raw = try arena.allocAligned(1152 * 4 * @sizeOf(f32), 64);
    const out_raw = try arena.allocAligned(8 * 1152 * @sizeOf(f16), 64);
    const q_raw = try arena.allocAligned(8 * 4 * 64 * @sizeOf(f16), 64);
    const kf_raw = try arena.allocAligned(64 * 64 * @sizeOf(f16), 64);

    const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(ctx.device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    // Verify with FailingAllocator that neither arena methods nor GPU encoding touch the heap!
    var fail_alloc = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    _ = fail_alloc.allocator();

    var f = try compute.beginFrame(5000);
    errdefer f.abort();
    const sconv_bind = try sconv_ring.layerBinding(0, gpu_buf);
    try ctx.pipes.encodeRmsNormSconv4(&f, .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(nw_raw.ptr) },
        .conv_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(cw_raw.ptr) },
        .conv_state_in = sconv_bind,
        .conv_state_out = sconv_bind,
        .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
        .m = 8,
        .hidden_dim = 1152,
        .state_pos = try sconv_ring.statePos(0),
    }, false);
    try sconv_ring.advanceStatePos(0, 8);
    const d_full = try ctx.pipes.encodeHy4IndexLayer(&f, .{
        .mode = .full,
        .q_idx = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
        .k_fine = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kf_raw.ptr) },
        .index_pool = &idx_pool,
        .index_metal_buf = gpu_buf,
        .num_nodes = 8,
        .num_q_heads = 4,
        .kv_len = 64,
    }, true);
    try std.testing.expectEqual(@as(u32, 2), d_full.total());
    try compute.wait(try f.commit(), 5000);
}

test "gpu.inkling2_hy4. d64/d192 rel_gqa_attn, sconv M=65 multi-tile, D=2560 sandwich_norm, and NoPE/dyn MLA parity" {
    const gpa = std.testing.allocator;
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();

    // 1) sconv M=65 multi-tile in-place (serial_tiles=1)
    {
        const m_val: u32 = 65;
        const d: u32 = 1152;
        const ring_bytes = try SconvRingArena.requiredBytes(.{ .num_layers = 1, .hidden_dim = d });
        var arena = try makeTestArena(ring_bytes + 1024 * 1024);
        defer arena.deinit();

        var sconv_ring = try SconvRingArena.init(&arena, .{ .num_layers = 1, .hidden_dim = d });
        const st = try sconv_ring.layerSlice(0);
        fillSynthF16(st, 17, 0.05);
        const ref_st = try gpa.dupe(f16, st);
        defer gpa.free(ref_st);

        const x_raw = try arena.allocAligned(@as(usize, m_val) * d * @sizeOf(f16), 64);
        const nw_raw = try arena.allocAligned(@as(usize, d) * @sizeOf(u16), 64);
        const cw_raw = try arena.allocAligned(@as(usize, d) * 4 * @sizeOf(f32), 64);
        const out_raw = try arena.allocAligned(@as(usize, m_val) * d * @sizeOf(f16), 64);
        const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
        const nw_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, nw_raw));
        const cw_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, cw_raw));
        const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
        fillSynthF16(x_f16, 41, 0.08);
        fillSynthNormU16(nw_u16, 73, .f16);
        for (cw_f32, 0..) |*v, i| v.* = synthF32(103 + i, 0.05);

        const ref_out = try gpa.alloc(f32, @as(usize, m_val) * d);
        defer gpa.free(ref_out);
        try rmsnormSconv4CpuRef(gpa, x_f16, nw_u16, .f16, cw_f32, null, null, null, ref_st, ref_out, m_val, d, 0, true, true, false, 1e-6);

        const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(ctx.device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});
        const l0_bind = try sconv_ring.layerBinding(0, gpu_buf);
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try ctx.pipes.encodeRmsNormSconv4(&f, .{
            .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
            .norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(nw_raw.ptr) },
            .conv_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(cw_raw.ptr) },
            .conv_state_in = l0_bind,
            .conv_state_out = l0_bind,
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .m = m_val,
            .hidden_dim = d,
        }, false);
        try compute.wait(try f.commit(), 5000);
        try expectF16MatchesF32(ref_out, out_f16, 0.01, 0.015);
    }

    // 2) rel_gqa_attn d64 and d192 (dyn PSO)
    for ([_]u32{ 64, 192 }) |hd| {
        const num_q: u32 = 4;
        const num_kv: u32 = 2;
        const kv_len: u32 = 24;
        const m_val: u32 = 4;
        const kv_table_bytes = try hybrid.HybridKvRing.tableRequiredBytes(1);
        var arena = try makeTestArena(kv_table_bytes + 512 * 1024);
        defer arena.deinit();

        var ring = try hybrid.HybridKvRing.init(gpa, ctx.device, &arena, .{
            .mode = .sliding_window_128,
            .num_kv_heads = num_kv,
            .head_dim = hd,
            .num_tiles = 1,
        });
        defer ring.deinit(gpa);

        const q_raw = try arena.allocAligned(@as(usize, m_val) * num_q * hd * @sizeOf(f16), 64);
        const qr_raw = try arena.allocAligned(@as(usize, m_val) * num_q * 16 * @sizeOf(f16), 64);
        const re_raw = try arena.allocAligned(@as(usize, 64) * 16 * @sizeOf(f16), 64);
        const out_raw = try arena.allocAligned(@as(usize, m_val) * num_q * hd * @sizeOf(f16), 64);
        const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
        const qr_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, qr_raw));
        const re_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, re_raw));
        const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
        fillSynthF16(q_f16, 11 + hd, 0.09);
        fillSynthF16(qr_f16, 29 + hd, 0.15);
        fillSynthF16(re_f16, 53 + hd, 0.15);

        const k_hist = try gpa.alloc(f16, @as(usize, kv_len) * num_kv * hd);
        defer gpa.free(k_hist);
        const v_hist = try gpa.alloc(f16, @as(usize, kv_len) * num_kv * hd);
        defer gpa.free(v_hist);
        fillSynthF16(k_hist, 79 + hd, 0.09);
        fillSynthF16(v_hist, 107 + hd, 0.09);

        {
            var kg = try ring.k_tiles[0].lock(false);
            errdefer kg.release();
            var vg = try ring.v_tiles[0].lock(false);
            errdefer vg.release();
            const ks = kg.bytes_per_row / @sizeOf(f16);
            const vs = vg.bytes_per_row / @sizeOf(f16);
            const kd: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
            const vd: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vg.bytes));
            for (0..kv_len) |t| {
                for (0..num_kv) |kvh| {
                    const row = kvh * ring.chunk_tokens + t;
                    const src = (t * num_kv + kvh) * hd;
                    @memcpy(kd[row * ks ..][0..hd], k_hist[src..][0..hd]);
                    @memcpy(vd[row * vs ..][0..hd], v_hist[src..][0..hd]);
                }
            }
            try vg.unlock();
            try kg.unlock();
        }

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const ref_out = try gpa.alloc(f32, @as(usize, m_val) * num_q * hd);
        defer gpa.free(ref_out);
        try inkling2RelGqaAttnCpuRef(gpa, q_f16, k_hist, v_hist, qr_f16, re_f16, null, null, null, null, false, null, .sigmoid, ref_out, m_val, num_q, num_kv, hd, kv_len, ring.window_size, kv_len - m_val, 16, 64, false, false, 1.0, scale, 22.627, 8.0, 1.0, 0.0);

        const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(ctx.device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});
        try ring.makeResident(compute);

        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try ctx.pipes.encodeInkling2RelGqaAttn(&f, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .q_rel = .{ .buffer = gpu_buf, .offset = arena.offsetOf(qr_raw.ptr) },
            .rel_emb = .{ .buffer = gpu_buf, .offset = arena.offsetOf(re_raw.ptr) },
            .kv_ring = &ring,
            .kv_ring_buf = gpu_buf,
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .num_nodes = m_val,
            .num_q_heads = num_q,
            .kv_len = kv_len,
            .causal_q_start_pos = kv_len - m_val,
            .d_rel = 16,
            .rel_extent = 64,
            .scale = scale,
        }, false);
        try compute.wait(try f.commit(), 5000);
        try expectF16MatchesF32(ref_out, out_f16, 0.01, 0.015);
    }

    // 3) D=2560 (> 2048 fallback) sandwich_norm + NoPE MLA (kv_lora_rank=1024 dyn PSO, qk_rope_dim=0)
    {
        const d: u32 = 2560;
        const m_val: u32 = 2;
        const d_lat: u32 = 1024;
        const idx_bytes = try IndexPoolBuffer.requiredBytes(.{
            .max_nodes = m_val,
            .max_kv_len = 16,
            .idx_dim = 32,
            .compress_ratio = 4,
            .group_topk = 2,
            .candidate_topk = 8,
        });
        var arena = try makeTestArena(idx_bytes + 1024 * 1024);
        defer arena.deinit();

        const idx_pool = try IndexPoolBuffer.init(&arena, .{
            .max_nodes = m_val,
            .max_kv_len = 16,
            .idx_dim = 32,
            .compress_ratio = 4,
            .group_topk = 2,
            .candidate_topk = 8,
        });
        for (0..m_val) |node| {
            idx_pool.selected_counts[node] = 4;
            for (0..4) |si| idx_pool.selected_tokens[node * 8 + si] = @intCast(si * 2);
        }

        const x_raw = try arena.allocAligned(@as(usize, m_val) * d * @sizeOf(f16), 64);
        const res_raw = try arena.allocAligned(@as(usize, m_val) * d * @sizeOf(f16), 64);
        const cnw_raw = try arena.allocAligned(@as(usize, d) * @sizeOf(u16), 64);
        const pnw_raw = try arena.allocAligned(@as(usize, d) * @sizeOf(u16), 64);
        const c_out_raw = try arena.allocAligned(@as(usize, m_val) * d * @sizeOf(f16), 64);
        const q_raw = try arena.allocAligned(@as(usize, m_val) * 4 * d_lat * @sizeOf(f16), 64);
        const kv_raw = try arena.allocAligned(@as(usize, 16) * d_lat * @sizeOf(f16), 64);
        const sink_raw = try arena.allocAligned(4 * @sizeOf(f32), 64);
        const mla_out_raw = try arena.allocAligned(@as(usize, m_val) * 4 * d_lat * @sizeOf(f16), 64);

        const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
        const res_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, res_raw));
        const cnw_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, cnw_raw));
        const pnw_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, pnw_raw));
        const c_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, c_out_raw));
        const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
        const kv_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kv_raw));
        const sink_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, sink_raw));
        const mla_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, mla_out_raw));

        fillSynthF16(x_f16, 301, 0.08);
        fillSynthF16(res_f16, 353, 0.08);
        fillSynthNormU16(cnw_u16, 401, .f16);
        fillSynthNormU16(pnw_u16, 457, .f16);
        fillSynthF16(q_f16, 503, 0.07);
        fillSynthF16(kv_f16, 557, 0.07);
        for (sink_f32, 0..) |*s, i| s.* = synthF32(601 + i, 0.08);

        const ref_c = try gpa.alloc(f32, @as(usize, m_val) * d);
        defer gpa.free(ref_c);
        try hy4CombineSandwichNormCpuRef(gpa, x_f16, null, null, null, res_f16, cnw_u16, pnw_u16, .f16, ref_c, m_val, d, 0, 1e-5, 1.0);

        const mla_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_lat)));
        const ref_m = try gpa.alloc(f32, @as(usize, m_val) * 4 * d_lat);
        defer gpa.free(ref_m);
        try hy4MlaSinkAttnCpuRef(gpa, q_f16, null, kv_f16, null, idx_pool.selected_tokens, idx_pool.selected_counts, null, null, null, sink_f32, null, .sigmoid, ref_m, m_val, 4, d_lat, 0, 16, null, 0, 8, mla_scale, 0.0);

        const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(ctx.device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try ctx.pipes.encodeHy4CombineSandwichNorm(&f, .{
            .expert_or_x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
            .residual = .{ .buffer = gpu_buf, .offset = arena.offsetOf(res_raw.ptr) },
            .combine_norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(cnw_raw.ptr) },
            .post_norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(pnw_raw.ptr) },
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(c_out_raw.ptr) },
            .m = m_val,
            .hidden_dim = d,
        }, false);
        try ctx.pipes.encodeHy4MlaSinkAttn(&f, .{
            .q_nope = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .kv_latent = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kv_raw.ptr) },
            .index_pool = &idx_pool,
            .index_metal_buf = gpu_buf,
            .sink_params = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sink_raw.ptr) },
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(mla_out_raw.ptr) },
            .num_nodes = m_val,
            .num_q_heads = 4,
            .kv_lora_rank = d_lat,
            .qk_rope_dim = 0,
            .kv_len = 16,
            .scale = mla_scale,
        }, false);
        try compute.wait(try f.commit(), 5000);

        try expectF16MatchesF32(ref_c, c_out_f16, 0.01, 0.015);
        try expectF16MatchesF32(ref_m, mla_out_f16, 0.01, 0.015);
    }
}
