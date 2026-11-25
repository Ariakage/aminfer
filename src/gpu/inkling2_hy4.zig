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
const sme = @import("../cpu/sme.zig");
const attention = @import("attention.zig");
const dequant = @import("dequant.zig");
const hybrid = @import("hybrid.zig");
const moe = @import("moe.zig");
const quant_formats = @import("quant_formats.zig");

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
    PipelineNotInitialized,
    InvalidArgument,
    UnsupportedExpertCount,
    WorkspaceTooSmall,
    InvalidGroupSize,
    UnalignedK,
};

pub const LayerEncodeError = EncodeError || moe.EncodeError || dequant.EncodeError || quant_formats.ShapeError;

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
    post_residual_scale: f32 = 0.0,
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
    scoring_sigmoid: u32 = 0,
    _pad1: u32 = 0,
};

pub const Inkling2MoeSwigluParams = extern struct {
    num_rows: u32,
    intermediate_size: u32,
    interleaved_gate_up: u32,
    softcap: f32,
    gate_softcap: f32 = 0.0,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
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

pub const Inkling2ScaledL2ClipParams = extern struct {
    m: u32,
    num_heads: u32,
    head_dim: u32,
    has_scale_vec: u32,
    use_log_scaling: u32,
    has_positions: u32,
    base_pos: u32,
    log_floor: u32,
    threshold: f32,
    log_alpha: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

pub const Inkling2UnitRmsNormParams = extern struct {
    m: u32,
    num_heads: u32,
    head_dim: u32,
    rms_eps: f32,
};

pub const Inkling2F16ToF32Params = extern struct {
    count: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

pub const Inkling2AttnPostProjParams = extern struct {
    m: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    d_rel: u32,
    has_q_scale_vec: u32,
    has_rel_scale_vec: u32,
    has_rel: u32,
    do_v_norm: u32,
    has_sink: u32,
    use_log_scaling: u32,
    has_positions: u32,
    base_pos: u32,
    log_floor: u32,
    q_clip: f32,
    k_clip: f32,
    rel_clip: f32,
    log_alpha: f32,
    rms_eps: f32,
    _pad0: u32 = 0,
};

pub const Inkling2DynSconvAssembleParams = extern struct {
    m: u32,
    kv_dim: u32,
    num_kv_heads: u32,
    head_dim: u32,
    proj_n: u32,
    has_static_w: u32,
    static_w_f16: u32,
    weight_scale: f32,
};

pub const Inkling2MsgBankNormParams = extern struct {
    m: u32,
    hidden_dim: u32,
    num_ring_blocks: u32,
    append_cur: u32,
    block_stride_elems: u32,
    rms_eps: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    block_slots: [8]u32,
};

pub const Inkling2CrossPassMsgParams = extern struct {
    m: u32,
    hidden_dim: u32,
    bank_len: u32,
    q_rows: u32,
    kv_rows: u32,
    loop_layer_idx: u32,
    has_kv_q: u32,
    queries_f16: u32,
    scale: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

pub const Inkling2MessageGateParams = extern struct {
    m: u32,
    state_dim: u32,
    down_in_dim: u32,
    gate_rank: u32,
    has_norm_w: u32,
    has_forget_bias: u32,
    has_gain_bias: u32,
    msg_f16: u32,
    gain_act: u32,
    rms_eps: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

pub const Inkling2GatherRowsParams = extern struct {
    count: u32,
    row_elems: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

pub const Inkling2ScatterRowsParams = extern struct {
    count: u32,
    row_elems: u32,
    slot_stride: u32,
    _pad0: u32 = 0,
};

pub const Inkling2MoeRouteCsrParams = extern struct {
    m: u32,
    num_experts: u32,
    top_k: u32,
    has_bias: u32,
    bias_f16: u32,
    bias_mode: u32,
    norm_topk_prob: u32,
    scoring_sigmoid: u32,
    is_heterogeneous: u32,
    bucket_unique_stride: u32,
    bucket_offsets_stride: u32,
    gate_up_groups_x_q: u32,
    gate_up_groups_x_f: u32,
    down_groups_x: u32,
    down_groups_x_q8: u32,
    route_scale: f32,
    router_softcap: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

pub const Inkling2MoeGateUpSoftcapParams = extern struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_groups: u32,
    row_bytes: u32,
    group_size: u32,
    pair_interleaved: u32,
    scale_stride: u32,
    gate_outer_scale_off: u32,
    up_outer_scale_off: u32,
    outer_scale_fallback: f32,
    gate_softcap: f32,
    up_softcap: f32,
};

comptime {
    if (@sizeOf(Sconv4Params) != 48) @compileError("Sconv4Params must be 48 bytes");
    if (@sizeOf(Inkling2RelGqaParams) != 96) @compileError("Inkling2RelGqaParams must be 96 bytes");
    if (@sizeOf(Inkling2MoeRouteParams) != 48) @compileError("Inkling2MoeRouteParams must be 48 bytes");
    if (@sizeOf(Inkling2MoeSwigluParams) != 32) @compileError("Inkling2MoeSwigluParams must be 32 bytes");
    if (@sizeOf(Inkling2MoeEpilogueParams) != 32) @compileError("Inkling2MoeEpilogueParams must be 32 bytes");
    if (@sizeOf(Inkling2AttnResParams) != 80) @compileError("Inkling2AttnResParams must be 80 bytes");
    if (@sizeOf(Hy4IndexCompressParams) != 32) @compileError("Hy4IndexCompressParams must be 32 bytes");
    if (@sizeOf(Hy4IndexPoolTopkParams) != 64) @compileError("Hy4IndexPoolTopkParams must be 64 bytes");
    if (@sizeOf(Hy4LatentMoeRouteParams) != 48) @compileError("Hy4LatentMoeRouteParams must be 48 bytes");
    if (@sizeOf(Hy4SwigluClampedParams) != 32) @compileError("Hy4SwigluClampedParams must be 32 bytes");
    if (@sizeOf(Hy4CombineSandwichParams) != 48) @compileError("Hy4CombineSandwichParams must be 48 bytes");
    if (@sizeOf(Hy4MlaSinkAttnParams) != 64) @compileError("Hy4MlaSinkAttnParams must be 64 bytes");
    if (@sizeOf(Inkling2ScaledL2ClipParams) != 48) @compileError("Inkling2ScaledL2ClipParams must be 48 bytes");
    if (@sizeOf(Inkling2UnitRmsNormParams) != 16) @compileError("Inkling2UnitRmsNormParams must be 16 bytes");
    if (@sizeOf(Inkling2F16ToF32Params) != 16) @compileError("Inkling2F16ToF32Params must be 16 bytes");
    if (@sizeOf(Inkling2AttnPostProjParams) != 80) @compileError("Inkling2AttnPostProjParams must be 80 bytes");
    if (@sizeOf(Inkling2DynSconvAssembleParams) != 32) @compileError("Inkling2DynSconvAssembleParams must be 32 bytes");
    if (@sizeOf(Inkling2MsgBankNormParams) != 64) @compileError("Inkling2MsgBankNormParams must be 64 bytes");
    if (@sizeOf(Inkling2CrossPassMsgParams) != 48) @compileError("Inkling2CrossPassMsgParams must be 48 bytes");
    if (@sizeOf(Inkling2MessageGateParams) != 48) @compileError("Inkling2MessageGateParams must be 48 bytes");
    if (@sizeOf(Inkling2GatherRowsParams) != 16) @compileError("Inkling2GatherRowsParams must be 16 bytes");
    if (@sizeOf(Inkling2ScatterRowsParams) != 16) @compileError("Inkling2ScatterRowsParams must be 16 bytes");
    if (@sizeOf(Inkling2MoeRouteCsrParams) != 80) @compileError("Inkling2MoeRouteCsrParams must be 80 bytes");
    if (@sizeOf(Inkling2MoeGateUpSoftcapParams) != 48) @compileError("Inkling2MoeGateUpSoftcapParams must be 48 bytes");
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
    post_residual_scale: f32 = 0.0,
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
    rel_zero_oob: bool = false,
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
    scoring_sigmoid: bool = false,
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
    pair_interleaved: bool = false,
    softcap: f32 = INKLING2_DEFAULT_MLP_SOFTCAP,
    gate_softcap: f32 = 0.0,
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

pub const DeinterleaveError = error{
    ZeroDimension,
    UnalignedDimension,
    InvalidDType,
    BufferTooSmall,
    Overflow,
};

pub const DeinterleavedWeight = struct {
    raw: []u8,
    u16_slice: []u16,
    arena_offset: u64,
    rows: u32,
    cols: u32,
    dtype: quant.FloatDType,

    pub fn binding(self: DeinterleavedWeight, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.arena_offset };
    }
};

fn panel64GroupBytes(dtype: sme.PanelDType) DeinterleaveError!usize {
    return switch (dtype) {
        .f16, .bf16 => 2048,
        .q4_affine => sme.QUANT_HEADER_BYTES + 512,
        .q3_affine => sme.QUANT_HEADER_BYTES + 384,
        .q2_affine => sme.QUANT_HEADER_BYTES + 256,
        else => error.InvalidDType,
    };
}

inline fn panel64ConvertOut(src_f16: f32, out_dtype: quant.FloatDType) u16 {
    return quant.f32ToFloatBits(out_dtype, src_f16);
}

/// Deinterleaves an `sme2_panel64` matrix `[n, k]` into contiguous row-major `f16` or `bf16`
/// `[n, k]` once at model startup so Metal GPU kernels execute directly on existing `.amf` weights
/// with zero repacking (`< 2 ms` for `Inkling-2-Mini` ~3.4 MiB controller blobs).
pub fn deinterleavePanel64ToRowMajor(
    dtype: sme.PanelDType,
    src_panel64: []const u8,
    n: usize,
    k: usize,
    out_dtype: quant.FloatDType,
    dst_u16: []u16,
) DeinterleaveError!void {
    if (n == 0 or k == 0) return error.ZeroDimension;
    if ((k % sme.GROUP_SIZE) != 0) return error.UnalignedDimension;
    const need_src = sme.packedBlobBytes(dtype, n, k) catch |err| switch (err) {
        error.ZeroDimension => return error.ZeroDimension,
        error.UnalignedK => return error.UnalignedDimension,
        error.Overflow => return error.Overflow,
        else => return error.InvalidDType,
    };
    const nk = std.math.mul(usize, n, k) catch return error.Overflow;
    if (src_panel64.len < need_src or dst_u16.len < nk) return error.BufferTooSmall;

    const panels = (n + sme.PANEL_COLS - 1) / sme.PANEL_COLS;
    const groups = k / sme.GROUP_SIZE;
    const g_bytes = try panel64GroupBytes(dtype);
    var in_off: usize = 0;

    switch (dtype) {
        .f16, .bf16 => {
            const src_fdt: quant.FloatDType = if (dtype == .bf16) .bf16 else .f16;
            const same_fdt = (src_fdt == out_dtype);
            for (0..panels) |p| {
                for (0..groups) |g| {
                    const k_base = g * sme.GROUP_SIZE;
                    const grp = src_panel64[in_off .. in_off + g_bytes];
                    for (0..32) |kp| {
                        const k0 = k_base + 2 * kp;
                        const k1 = k0 + 1;
                        const chunk_off = kp * 64;
                        for (0..sme.PANEL_COLS) |c| {
                            const col = p * sme.PANEL_COLS + c;
                            if (col >= n) break;
                            const v0 = std.mem.readInt(u16, grp[chunk_off + c * 4 + 0 ..][0..2], .little);
                            const v1 = std.mem.readInt(u16, grp[chunk_off + c * 4 + 2 ..][0..2], .little);
                            const row_off = col * k;
                            if (same_fdt) {
                                dst_u16[row_off + k0] = v0;
                                dst_u16[row_off + k1] = v1;
                            } else {
                                dst_u16[row_off + k0] = panel64ConvertOut(quant.floatBitsToF32(src_fdt, v0), out_dtype);
                                dst_u16[row_off + k1] = panel64ConvertOut(quant.floatBitsToF32(src_fdt, v1), out_dtype);
                            }
                        }
                    }
                    in_off += g_bytes;
                }
            }
        },
        .q4_affine, .q3_affine, .q2_affine => {
            for (0..panels) |p| {
                for (0..groups) |g| {
                    const k_base = g * sme.GROUP_SIZE;
                    const grp = src_panel64[in_off .. in_off + g_bytes];
                    var scales: [sme.PANEL_COLS]f32 = undefined;
                    var biases: [sme.PANEL_COLS]f32 = undefined;
                    for (0..sme.PANEL_COLS) |c| {
                        const s_bits = std.mem.readInt(u16, grp[c * 4 ..][0..2], .little);
                        const b_bits = std.mem.readInt(u16, grp[64 + c * 4 ..][0..2], .little);
                        scales[c] = @floatCast(@as(f16, @bitCast(s_bits)));
                        biases[c] = @floatCast(@as(f16, @bitCast(b_bits)));
                    }
                    const code_src = grp[sme.QUANT_HEADER_BYTES..];
                    switch (dtype) {
                        .q4_affine => {
                            for (0..32) |kp| {
                                const k0 = k_base + 2 * kp;
                                const k1 = k0 + 1;
                                for (0..sme.PANEL_COLS) |c| {
                                    const col = p * sme.PANEL_COLS + c;
                                    if (col >= n) break;
                                    const byte = code_src[kp * 16 + c];
                                    const q0: f32 = @floatFromInt(byte & 0x0F);
                                    const q1: f32 = @floatFromInt(byte >> 4);
                                    const row_off = col * k;
                                    dst_u16[row_off + k0] = panel64ConvertOut(q0 * scales[c] + biases[c], out_dtype);
                                    dst_u16[row_off + k1] = panel64ConvertOut(q1 * scales[c] + biases[c], out_dtype);
                                }
                            }
                        },
                        .q3_affine => {
                            for (0..2) |half| {
                                const kp_base = half * 16;
                                const h_off = half * 192;
                                for (0..64) |i| {
                                    const c = i % 16;
                                    const col = p * sme.PANEL_COLS + c;
                                    if (col >= n) continue;
                                    const b0 = code_src[h_off + 0 + i];
                                    const b1 = code_src[h_off + 64 + i];
                                    const b2 = code_src[h_off + 128 + i];
                                    const sub_kp = i / 16;
                                    const sc = scales[c];
                                    const bs = biases[c];
                                    const row_off = col * k;
                                    const q_vals = [4][2]u8{
                                        .{ b0 & 0x07, (b0 >> 4) & 0x07 },
                                        .{ b1 & 0x07, (b1 >> 4) & 0x07 },
                                        .{ b2 & 0x07, (b2 >> 4) & 0x07 },
                                        .{
                                            ((b0 >> 3) & 1) | (((b1 >> 3) & 1) << 1) | (((b2 >> 3) & 1) << 2),
                                            ((b0 >> 7) & 1) | (((b1 >> 7) & 1) << 1) | (((b2 >> 7) & 1) << 2),
                                        },
                                    };
                                    inline for (0..4) |q_idx| {
                                        const kp = kp_base + q_idx * 4 + sub_kp;
                                        const k0 = k_base + 2 * kp;
                                        const k1 = k0 + 1;
                                        dst_u16[row_off + k0] = panel64ConvertOut(@as(f32, @floatFromInt(q_vals[q_idx][0])) * sc + bs, out_dtype);
                                        dst_u16[row_off + k1] = panel64ConvertOut(@as(f32, @floatFromInt(q_vals[q_idx][1])) * sc + bs, out_dtype);
                                    }
                                }
                            }
                        },
                        .q2_affine => {
                            for (0..32) |kp| {
                                const k0 = k_base + 2 * kp;
                                const k1 = k0 + 1;
                                for (0..8) |b| {
                                    const c0 = 2 * b;
                                    const c1 = c0 + 1;
                                    const col0 = p * sme.PANEL_COLS + c0;
                                    const byte = code_src[kp * 8 + b];
                                    if (col0 < n) {
                                        const row0 = col0 * k;
                                        dst_u16[row0 + k0] = panel64ConvertOut(@as(f32, @floatFromInt(byte & 0x03)) * scales[c0] + biases[c0], out_dtype);
                                        dst_u16[row0 + k1] = panel64ConvertOut(@as(f32, @floatFromInt((byte >> 2) & 0x03)) * scales[c0] + biases[c0], out_dtype);
                                    }
                                    const col1 = col0 + 1;
                                    if (col1 < n) {
                                        const row1 = col1 * k;
                                        dst_u16[row1 + k0] = panel64ConvertOut(@as(f32, @floatFromInt((byte >> 4) & 0x03)) * scales[c1] + biases[c1], out_dtype);
                                        dst_u16[row1 + k1] = panel64ConvertOut(@as(f32, @floatFromInt((byte >> 6) & 0x03)) * scales[c1] + biases[c1], out_dtype);
                                    }
                                }
                            }
                        },
                        else => unreachable,
                    }
                    in_off += g_bytes;
                }
            }
        },
        else => return error.InvalidDType,
    }
}

pub const DeinterleaveArenaError = DeinterleaveError || arena_mod.WiredArena.AllocError;

/// Allocates 64-byte-aligned storage in `arena` and deinterleaves `src_panel64` `[n, k]`
/// into contiguous row-major `f16`/`bf16` weights once at model startup.
pub fn deinterleavePanel64ToArena(
    arena: *arena_mod.WiredArena,
    dtype: sme.PanelDType,
    src_panel64: []const u8,
    n: usize,
    k: usize,
    out_dtype: quant.FloatDType,
) DeinterleaveArenaError!DeinterleavedWeight {
    if (n == 0 or k == 0) return error.ZeroDimension;
    const nk = std.math.mul(usize, n, k) catch return error.Overflow;
    const bytes = std.math.mul(usize, nk, @sizeOf(u16)) catch return error.Overflow;
    const raw = try arena.allocAligned(bytes, 64);
    const u16_slice: []u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
    try deinterleavePanel64ToRowMajor(dtype, src_panel64, n, k, out_dtype, u16_slice);
    return .{
        .raw = raw,
        .u16_slice = u16_slice,
        .arena_offset = arena.offsetOf(raw.ptr),
        .rows = @intCast(n),
        .cols = @intCast(k),
        .dtype = out_dtype,
    };
}

pub const Inkling2ScaledL2ClipArgs = struct {
    buf: metal.BufferBinding,
    scale_vec: ?metal.BufferBinding = null,
    positions: ?metal.BufferBinding = null,
    m: u32,
    num_heads: u32,
    head_dim: u32,
    threshold: f32 = INKLING2_DEFAULT_QK_L2_CLIP,
    use_log_scaling: bool = false,
    base_pos: u32 = 0,
    log_floor: u32 = 0,
    log_alpha: f32 = 0.0,
};

pub const Inkling2UnitRmsNormArgs = struct {
    buf: metal.BufferBinding,
    m: u32,
    num_heads: u32,
    head_dim: u32,
    rms_eps: f32 = 1e-6,
};

pub const Inkling2F16ToF32Args = struct {
    src: metal.BufferBinding,
    dst: metal.BufferBinding,
    count: u32,
};

pub const Inkling2AttnPostProjArgs = struct {
    q: metal.BufferBinding,
    k: metal.BufferBinding,
    q_rel: ?metal.BufferBinding = null,
    v: ?metal.BufferBinding = null,
    q_scale_vec: ?metal.BufferBinding = null,
    rel_scale_vec: ?metal.BufferBinding = null,
    positions: ?metal.BufferBinding = null,
    sink_f16: ?metal.BufferBinding = null,
    sink_f32: ?metal.BufferBinding = null,
    m: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    d_rel: u32 = 0,
    q_clip: f32 = INKLING2_DEFAULT_QK_L2_CLIP,
    k_clip: f32 = INKLING2_DEFAULT_QK_L2_CLIP,
    rel_clip: f32 = INKLING2_DEFAULT_REL_L2_CLIP,
    use_log_scaling: bool = false,
    base_pos: u32 = 0,
    log_floor: u32 = 0,
    log_alpha: f32 = 0.0,
    do_v_norm: bool = false,
    rms_eps: f32 = 1e-6,
};

pub const Inkling2DynSconvAssembleArgs = struct {
    proj: metal.BufferBinding,
    static_w: ?metal.BufferBinding = null,
    static_w_f16: bool = false,
    dyn_w: metal.BufferBinding,
    m: u32,
    kv_dim: u32,
    num_kv_heads: u32 = 0,
    head_dim: u32 = 0,
    proj_n: u32,
    /// Optional scale applied directly into assembled `dyn_w` weights (defaults
    /// to `1.0` when `encodeRmsNormSconv4(..., post_residual_scale = 1.0 / 3.0)` applies `/ 3.0`).
    weight_scale: f32 = 1.0,
};

pub const Inkling2MsgBankNormArgs = struct {
    ring: *const AttnResRingArena,
    ring_buf: metal.Buffer,
    cur: ?metal.BufferBinding = null,
    msg_bank_out: metal.BufferBinding,
    m: u32,
    start_slot: u32 = 0,
    /// When null, uses `ring.activeSlots(...)`; when non-null, uses `[start_slot .. end_slot)`.
    end_slot: ?u32 = null,
    append_cur: bool = true,
    rms_eps: f32 = 1e-6,
};

pub const Inkling2CrossPassMsgArgs = struct {
    msg_bank: metal.BufferBinding,
    q_src: metal.BufferBinding,
    kv_q_src: ?metal.BufferBinding = null,
    queries_f16: bool = false,
    out_msgs: metal.BufferBinding,
    m: u32,
    hidden_dim: u32,
    bank_len: u32,
    q_rows: u32,
    kv_rows: u32 = 0,
    loop_layer_idx: u32,
    scale: ?f32 = null,
};

pub const Inkling2MessageGateGainAct = enum(u32) {
    tanh = 0,
    silu = 1,
};

pub const Inkling2MessageGateArgs = struct {
    cur: metal.BufferBinding,
    msg: metal.BufferBinding,
    msg_f16: bool = false,
    down_weight: metal.BufferBinding,
    forget_weight: metal.BufferBinding,
    gain_weight: metal.BufferBinding,
    weight_dtype: quant.FloatDType = .bf16,
    norm_weight: ?metal.BufferBinding = null,
    forget_bias: ?metal.BufferBinding = null,
    gain_bias: ?metal.BufferBinding = null,
    z_scratch: metal.BufferBinding,
    out: metal.BufferBinding,
    m: u32,
    state_dim: u32,
    down_in_dim: u32,
    gate_rank: u32 = 64,
    gain_act: Inkling2MessageGateGainAct = .tanh,
    rms_eps: f32 = 1e-6,
};

pub const Inkling2GatherRowsArgs = struct {
    src: metal.BufferBinding,
    indices: metal.BufferBinding,
    dst: metal.BufferBinding,
    count: u32,
    row_elems: u32,
};

pub const Inkling2ScatterRowsArgs = struct {
    src: metal.BufferBinding,
    slots_or_tok: metal.BufferBinding,
    slot_k_indices: ?metal.BufferBinding = null,
    dst: metal.BufferBinding,
    count: u32,
    row_elems: u32,
    slot_stride: u32 = 0,
};

pub const Inkling2MoeRouteCsrArgs = struct {
    logits: metal.BufferBinding,
    logits_f16: bool = true,
    bias: ?metal.BufferBinding = null,
    bias_f16: bool = false,
    bias_add_to_logits: bool = false,
    scoring_sigmoid: bool = true,
    workspace: *const moe.MoeWorkspace,
    workspace_buf: metal.Buffer,
    m: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32 = 0,
    intermediate_size: u32 = 0,
    norm_topk_prob: bool = true,
    route_scale: f32 = INKLING2_DEFAULT_ROUTE_SCALE,
    router_softcap: f32 = 0.0,
    is_heterogeneous: bool = false,
    expert_schemes: ?metal.BufferBinding = null,
};

pub const Inkling2MoeGateUpSoftcapArgs = struct {
    amf_dtype: amf.DType,
    group_size: u16 = 64,
    tensor_scale_per_row: bool = false,
    nvfp4_outer_scale: ?f32 = null,
    pair_interleaved: bool = true,
    x: metal.BufferBinding,
    weights_base: metal.BufferBinding,
    expert_table: metal.BufferBinding,
    workspace: *const moe.MoeWorkspace,
    workspace_buf: metal.Buffer,
    qf_pipes: ?*const quant_formats.QuantFormatPipelines = null,
    m: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    scheme_bucket: ?quant.WeightDType = null,
    max_active_override: ?u32 = null,
    use_indirect: bool = false,
    gate_softcap: f32 = INKLING2_DEFAULT_ROUTER_SOFTCAP,
    up_softcap: f32 = INKLING2_DEFAULT_MLP_SOFTCAP,
};

pub const Inkling2LinearDesc = struct {
    amf_dtype: amf.DType,
    group_size: u16 = 64,
    tensor_scale_per_row: bool = false,
    weight: metal.BufferBinding,
    params: ?metal.BufferBinding = null,
    n: u32,
    k: u32,
};

pub const Inkling2MoeRouterSource = union(enum) {
    logits: struct {
        binding: metal.BufferBinding,
        f16: bool = true,
    },
    linear: Inkling2LinearDesc,
};

pub const Inkling2MoeIndirectLayerArgs = struct {
    moe_pipes: *const moe.MoePipelines,
    qf_pipes: ?*const quant_formats.QuantFormatPipelines = null,
    dequant_pipes: ?*const dequant.DequantPipelines = null,
    x: metal.BufferBinding,
    router: Inkling2MoeRouterSource,
    bias: ?metal.BufferBinding = null,
    bias_f16: bool = false,
    bias_add_to_logits: bool = false,
    scoring_sigmoid: bool = true,
    norm_topk_prob: bool = true,
    route_scale: f32 = INKLING2_DEFAULT_ROUTE_SCALE,
    router_softcap: f32 = 0.0,
    gate_softcap: f32 = INKLING2_DEFAULT_ROUTER_SOFTCAP,
    up_softcap: f32 = INKLING2_DEFAULT_MLP_SOFTCAP,
    pair_interleaved: bool = true,
    bucket_pair_interleaved: ?[moe.NUM_SCHEME_BUCKETS]bool = null,
    weights_base: metal.BufferBinding,
    expert_table: metal.BufferBinding,
    workspace: *const moe.MoeWorkspace,
    workspace_buf: metal.Buffer,
    is_heterogeneous: bool = false,
    /// Per-expert bucket ID (`0..moe.NUM_SCHEME_BUCKETS - 1`) indexing `scheme_counts` and `scheme_descs`.
    expert_schemes: ?metal.BufferBinding = null,
    scheme_counts: [moe.NUM_SCHEME_BUCKETS]u32 = .{ 0, 0, 0, 0, 0 },
    scheme_descs: [moe.NUM_SCHEME_BUCKETS]moe.MoeSchemeDesc = .{ .{}, .{}, .{}, .{}, .{} },
    residual: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    m: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    act_quant_policy: quant_formats.ActQuantPolicy = .none,
    act_scratch: ?metal.BufferBinding = null,
    act_scratch_bytes: u64 = 0,
    router_tile_f16: ?metal.BufferBinding = null,
    router_tile_f16_bytes: u64 = 0,
};

pub const Inkling2GroupedExpertMlpArgs = struct {
    dequant_pipes: *const dequant.DequantPipelines,
    qf_pipes: ?*const quant_formats.QuantFormatPipelines = null,
    x: metal.BufferBinding,
    w1: Inkling2LinearDesc,
    w_up: ?Inkling2LinearDesc = null,
    w2: Inkling2LinearDesc,
    /// Total matched slots (`M_e >= 1`) assigned to this single active expert.
    m_e: u32,
    hidden_size: u32,
    intermediate_size: u32,
    /// Fast path when `m_e == 1`: token row in `x` and destination slot binding/row.
    single_token_idx: u32 = 0,
    /// When `m_e > 1`: `[m_e]` u32 token row indices into `x`.
    token_indices_dev: ?metal.BufferBinding = null,
    /// When `m_e > 1`: destination row offsets `[m_e]` (or token indices if `slot_stride > 0`).
    dst_slots_or_tok_dev: ?metal.BufferBinding = null,
    slot_k_indices_dev: ?metal.BufferBinding = null,
    slot_stride: u32 = 0,
    gathered_x: ?metal.BufferBinding = null,
    gate_up_buf: metal.BufferBinding,
    up_buf: ?metal.BufferBinding = null,
    act_buf: metal.BufferBinding,
    down_tmp_buf: ?metal.BufferBinding = null,
    expert_out: metal.BufferBinding,
    gate_softcap: f32 = INKLING2_DEFAULT_ROUTER_SOFTCAP,
    up_softcap: f32 = INKLING2_DEFAULT_MLP_SOFTCAP,
    tile_f16: ?metal.BufferBinding = null,
    tile_f16_bytes: u64 = 0,
    act_quant_policy: quant_formats.ActQuantPolicy = .none,
    act_scratch: ?metal.BufferBinding = null,
    act_scratch_bytes: u64 = 0,
};

fn sameBinding(a: metal.BufferBinding, b: metal.BufferBinding) bool {
    return a.buffer.handle == b.buffer.handle and a.offset == b.offset;
}

pub const SOFTCAP_QUANT_SLOTS: usize = 38;

fn softcapKernelNameFor(dtype: quant_formats.DType) [:0]const u8 {
    if (dtype.isAffine()) return "ami_inkling2_moe_gate_up_softcap_affine";
    if (dtype.isFpFamily()) return "ami_inkling2_moe_gate_up_softcap_fp";
    if (dtype.isGgufBlk()) return "ami_inkling2_moe_gate_up_softcap_gguf_blk";
    if (dtype.isGgufK()) return "ami_inkling2_moe_gate_up_softcap_gguf_k";
    return "ami_inkling2_moe_gate_up_softcap_iq";
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
    inkling2_scaled_l2_clip: metal.Pipeline,
    inkling2_unit_rms_norm: metal.Pipeline,
    inkling2_f16_to_f32: metal.Pipeline,
    inkling2_attn_post_proj: metal.Pipeline,
    inkling2_assemble_dyn_sconv_w: metal.Pipeline,
    inkling2_msg_bank_norm: metal.Pipeline,
    inkling2_cross_pass_msgs: metal.Pipeline,
    inkling2_msg_gate_down: [2]metal.Pipeline,
    inkling2_msg_gate_up: [2]metal.Pipeline,
    inkling2_gather_rows_f16: metal.Pipeline,
    inkling2_scatter_rows_f16: metal.Pipeline,
    inkling2_moe_route_csr: [2]metal.Pipeline,
    inkling2_moe_gate_up_softcap_float: [2]metal.Pipeline,
    inkling2_moe_gate_up_softcap_quant: [SOFTCAP_QUANT_SLOTS]?metal.Pipeline,

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

        const ink_l2_clip = try metal.Pipeline.init(device, library, "ami_inkling2_scaled_l2_clip_per_head");
        errdefer ink_l2_clip.close();

        const ink_unit_rms = try metal.Pipeline.init(device, library, "ami_inkling2_unit_rms_norm_per_head");
        errdefer ink_unit_rms.close();

        const ink_f16_to_f32 = try metal.Pipeline.init(device, library, "ami_inkling2_f16_to_f32");
        errdefer ink_f16_to_f32.close();

        const ink_attn_post = try metal.Pipeline.init(device, library, "ami_inkling2_attn_post_proj");
        errdefer ink_attn_post.close();

        const ink_dyn_sconv = try metal.Pipeline.init(device, library, "ami_inkling2_assemble_dyn_sconv_w");
        errdefer ink_dyn_sconv.close();

        const ink_msg_bank_norm = try metal.Pipeline.init(device, library, "ami_inkling2_msg_bank_norm");
        errdefer ink_msg_bank_norm.close();

        const ink_cross_pass = try metal.Pipeline.init(device, library, "ami_inkling2_cross_pass_msgs");
        errdefer ink_cross_pass.close();

        const ink_gate_dn_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_msg_gate_down", &.{
            .{ .index = 0, .value = .{ .bool = false } },
        });
        errdefer ink_gate_dn_f16.close();

        const ink_gate_dn_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_msg_gate_down", &.{
            .{ .index = 0, .value = .{ .bool = true } },
        });
        errdefer ink_gate_dn_bf16.close();

        const ink_gate_up_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_msg_gate_up", &.{
            .{ .index = 0, .value = .{ .bool = false } },
        });
        errdefer ink_gate_up_f16.close();

        const ink_gate_up_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_msg_gate_up", &.{
            .{ .index = 0, .value = .{ .bool = true } },
        });
        errdefer ink_gate_up_bf16.close();

        const ink_gather = try metal.Pipeline.init(device, library, "ami_inkling2_gather_rows_f16");
        errdefer ink_gather.close();

        const ink_scatter = try metal.Pipeline.init(device, library, "ami_inkling2_scatter_rows_f16");
        errdefer ink_scatter.close();

        const ink_route_csr_f32 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_moe_route_csr", &.{
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer ink_route_csr_f32.close();

        const ink_route_csr_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_moe_route_csr", &.{
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer ink_route_csr_f16.close();

        const softcap_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_moe_gate_up_softcap_float", &.{
            .{ .index = 0, .value = .{ .bool = false } },
        });
        errdefer softcap_f16.close();

        const softcap_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_inkling2_moe_gate_up_softcap_float", &.{
            .{ .index = 0, .value = .{ .bool = true } },
        });
        errdefer softcap_bf16.close();

        var softcap_quant: [SOFTCAP_QUANT_SLOTS]?metal.Pipeline = @splat(null);
        errdefer for (softcap_quant) |maybe_p| {
            if (maybe_p) |p| p.close();
        };
        inline for (@typeInfo(quant_formats.DType).@"enum".field_values) |fv| {
            const dt: quant_formats.DType = @fromBackingInt(fv);
            softcap_quant[fv] = try metal.Pipeline.initSpecialized(
                device,
                library,
                softcapKernelNameFor(dt),
                &.{.{ .index = 4, .value = .{ .u32 = fv } }},
            );
        }

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
            .inkling2_scaled_l2_clip = ink_l2_clip,
            .inkling2_unit_rms_norm = ink_unit_rms,
            .inkling2_f16_to_f32 = ink_f16_to_f32,
            .inkling2_attn_post_proj = ink_attn_post,
            .inkling2_assemble_dyn_sconv_w = ink_dyn_sconv,
            .inkling2_msg_bank_norm = ink_msg_bank_norm,
            .inkling2_cross_pass_msgs = ink_cross_pass,
            .inkling2_msg_gate_down = .{ ink_gate_dn_f16, ink_gate_dn_bf16 },
            .inkling2_msg_gate_up = .{ ink_gate_up_f16, ink_gate_up_bf16 },
            .inkling2_gather_rows_f16 = ink_gather,
            .inkling2_scatter_rows_f16 = ink_scatter,
            .inkling2_moe_route_csr = .{ ink_route_csr_f32, ink_route_csr_f16 },
            .inkling2_moe_gate_up_softcap_float = .{ softcap_f16, softcap_bf16 },
            .inkling2_moe_gate_up_softcap_quant = softcap_quant,
        };
    }

    pub fn close(self: Inkling2Hy4Pipelines) void {
        for (self.inkling2_moe_gate_up_softcap_quant) |maybe_p| {
            if (maybe_p) |p| p.close();
        }
        self.inkling2_moe_gate_up_softcap_float[1].close();
        self.inkling2_moe_gate_up_softcap_float[0].close();
        self.inkling2_moe_route_csr[1].close();
        self.inkling2_moe_route_csr[0].close();
        self.inkling2_scatter_rows_f16.close();
        self.inkling2_gather_rows_f16.close();
        self.inkling2_msg_gate_up[1].close();
        self.inkling2_msg_gate_up[0].close();
        self.inkling2_msg_gate_down[1].close();
        self.inkling2_msg_gate_down[0].close();
        self.inkling2_cross_pass_msgs.close();
        self.inkling2_msg_bank_norm.close();
        self.inkling2_assemble_dyn_sconv_w.close();
        self.inkling2_attn_post_proj.close();
        self.inkling2_f16_to_f32.close();
        self.inkling2_unit_rms_norm.close();
        self.inkling2_scaled_l2_clip.close();
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
            .post_residual_scale = args.post_residual_scale,
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
            .rel_per_head = (@as(u32, if (args.rel_per_head) 1 else 0)) | (@as(u32, if (args.rel_zero_oob) 2 else 0)),
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
            .scoring_sigmoid = if (args.scoring_sigmoid) 1 else 0,
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
        if (!args.interleaved_gate_up and !args.pair_interleaved and args.up_in == null) return error.MissingInputs;

        const params: Inkling2MoeSwigluParams = .{
            .num_rows = args.num_rows,
            .intermediate_size = args.intermediate_size,
            .interleaved_gate_up = if (args.pair_interleaved) 2 else if (args.interleaved_gate_up) 1 else 0,
            .softcap = args.softcap,
            .gate_softcap = args.gate_softcap,
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

    pub fn encodeInkling2ScaledL2ClipPerHead(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2ScaledL2ClipArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_heads == 0 or args.head_dim == 0) return error.ZeroDimension;
        const params: Inkling2ScaledL2ClipParams = .{
            .m = args.m,
            .num_heads = args.num_heads,
            .head_dim = args.head_dim,
            .has_scale_vec = if (args.scale_vec != null) 1 else 0,
            .use_log_scaling = if (args.use_log_scaling) 1 else 0,
            .has_positions = if (args.positions != null) 1 else 0,
            .base_pos = args.base_pos,
            .log_floor = args.log_floor,
            .threshold = args.threshold,
            .log_alpha = args.log_alpha,
        };
        try frame.encodeThreadgroups(
            self.inkling2_scaled_l2_clip,
            &.{
                args.buf,
                args.scale_vec orelse args.buf,
                args.positions orelse args.buf,
            },
            std.mem.asBytes(&params),
            .{ args.num_heads, args.m, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeInkling2UnitRmsNormPerHead(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2UnitRmsNormArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_heads == 0 or args.head_dim == 0) return error.ZeroDimension;
        const params: Inkling2UnitRmsNormParams = .{
            .m = args.m,
            .num_heads = args.num_heads,
            .head_dim = args.head_dim,
            .rms_eps = args.rms_eps,
        };
        try frame.encodeThreadgroups(
            self.inkling2_unit_rms_norm,
            &.{args.buf},
            std.mem.asBytes(&params),
            .{ args.num_heads, args.m, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeInkling2F16ToF32(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2F16ToF32Args,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.count == 0) return error.ZeroDimension;
        const params: Inkling2F16ToF32Params = .{ .count = args.count };
        try frame.encodeThreads(
            self.inkling2_f16_to_f32,
            &.{ args.src, args.dst },
            std.mem.asBytes(&params),
            .{ args.count, 1, 1 },
            .{ @min(args.count, 256), 1, 1 },
            insert_barrier_before,
        );
    }

    /// Single-dispatch post-projection attention prep for `M >= 1` tokens with zero CPU sync:
    /// per-head Q ScaledL2Clip + optional LogScale, K ScaledL2Clip, optional R ScaledL2Clip +
    /// optional LogScale, optional V unweighted RMSNorm, and optional `attn_sinks` f16->f32.
    pub fn encodeInkling2AttnPostProj(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2AttnPostProjArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_q_heads == 0 or args.num_kv_heads == 0 or args.head_dim == 0) {
            return error.ZeroDimension;
        }
        const has_rel = (args.q_rel != null);
        if (has_rel and args.d_rel == 0) return error.ZeroDimension;
        if (args.do_v_norm and args.v == null) return error.MissingInputs;
        const has_sink = (args.sink_f16 != null or args.sink_f32 != null);
        if (has_sink and (args.sink_f16 == null or args.sink_f32 == null)) return error.MissingInputs;

        const gx: u64 = @as(u64, args.num_q_heads) +
            args.num_kv_heads +
            (if (has_rel) args.num_q_heads else 0) +
            (if (args.do_v_norm) args.num_kv_heads else 0) +
            (if (has_sink) @as(u64, 1) else 0);
        const params: Inkling2AttnPostProjParams = .{
            .m = args.m,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .d_rel = args.d_rel,
            .has_q_scale_vec = if (args.q_scale_vec != null) 1 else 0,
            .has_rel_scale_vec = if (args.rel_scale_vec != null) 1 else 0,
            .has_rel = if (has_rel) 1 else 0,
            .do_v_norm = if (args.do_v_norm) 1 else 0,
            .has_sink = if (has_sink) 1 else 0,
            .use_log_scaling = if (args.use_log_scaling) 1 else 0,
            .has_positions = if (args.positions != null) 1 else 0,
            .base_pos = args.base_pos,
            .log_floor = args.log_floor,
            .q_clip = args.q_clip,
            .k_clip = args.k_clip,
            .rel_clip = args.rel_clip,
            .log_alpha = args.log_alpha,
            .rms_eps = args.rms_eps,
        };
        try frame.encodeThreadgroups(
            self.inkling2_attn_post_proj,
            &.{
                args.q,
                args.k,
                args.q_rel orelse args.q,
                args.v orelse args.q,
                args.q_scale_vec orelse args.q,
                args.rel_scale_vec orelse args.q,
                args.positions orelse args.q,
                args.sink_f16 orelse args.q,
                args.sink_f32 orelse args.q,
            },
            std.mem.asBytes(&params),
            .{ gx, args.m, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Assembles per-token lag-reversed `[t-3, t-2, t-1, t]` dynamic `sconv4`
    /// filter weights directly on GPU buffers for global layers (`0` CPU sync).
    pub fn encodeInkling2DynSconvAssemble(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2DynSconvAssembleArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.kv_dim == 0 or args.proj_n == 0) return error.ZeroDimension;
        if (args.proj_n == args.num_kv_heads * 4 and args.head_dim == 0) return error.InvalidHeadDim;
        const params: Inkling2DynSconvAssembleParams = .{
            .m = args.m,
            .kv_dim = args.kv_dim,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .proj_n = args.proj_n,
            .has_static_w = if (args.static_w != null) 1 else 0,
            .static_w_f16 = if (args.static_w_f16) 1 else 0,
            .weight_scale = args.weight_scale,
        };
        try frame.encodeThreads(
            self.inkling2_assemble_dyn_sconv_w,
            &.{
                args.proj,
                args.static_w orelse args.proj,
                args.dyn_w,
            },
            std.mem.asBytes(&params),
            .{ args.kv_dim, args.m, 1 },
            .{ @min(args.kv_dim, 256), 1, 1 },
            insert_barrier_before,
        );
    }

    /// Normalizes residual-stream blocks in `ring` (+ optional `cur` hidden states) into `msg_bank_out` (`[bank_len, m, hidden_dim]` `f32`).
    pub fn encodeInkling2MsgBankNorm(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2MsgBankNormArgs,
        insert_barrier_before: bool,
    ) EncodeError!u32 {
        if (args.m == 0 or args.m > args.ring.spec.max_tokens) return error.InvalidNodeCount;
        if (args.append_cur and args.cur == null) return error.MissingInputs;

        var slots: [MAX_ATTN_RES_BLOCKS]u32 = @splat(0);
        var num_ring_blocks: u32 = 0;
        const max_ring: u32 = if (args.append_cur) MAX_ATTN_RES_BLOCKS - 1 else MAX_ATTN_RES_BLOCKS;
        if (args.end_slot) |end_s| {
            if (end_s < args.start_slot) return error.InvalidConfig;
            var slot = args.start_slot;
            while (slot < end_s and num_ring_blocks < max_ring) : (slot += 1) {
                if (slot >= args.ring.spec.max_blocks) return error.InvalidConfig;
                slots[num_ring_blocks] = slot;
                num_ring_blocks += 1;
            }
        } else {
            const active = args.ring.activeSlots(&slots);
            num_ring_blocks = @min(active, max_ring);
        }
        const total_blocks = num_ring_blocks + (if (args.append_cur) @as(u32, 1) else 0);
        if (total_blocks == 0 or total_blocks > MAX_ATTN_RES_BLOCKS) return error.InvalidConfig;

        const params: Inkling2MsgBankNormParams = .{
            .m = args.m,
            .hidden_dim = args.ring.spec.hidden_dim,
            .num_ring_blocks = num_ring_blocks,
            .append_cur = if (args.append_cur) 1 else 0,
            .block_stride_elems = args.ring.block_stride_elems,
            .rms_eps = args.rms_eps,
            .block_slots = slots,
        };
        const ring_bind = args.ring.binding(args.ring_buf);
        try frame.encodeThreadgroups(
            self.inkling2_msg_bank_norm,
            &.{
                ring_bind,
                args.cur orelse ring_bind,
                args.msg_bank_out,
            },
            std.mem.asBytes(&params),
            .{ total_blocks, args.m, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
        return total_blocks;
    }

    /// Computes 4 attended cross-pass messages `[msg_q, msg_k, msg_v, msg_mlp]` (`[4, m, hidden_dim]` `f32`) on GPU.
    pub fn encodeInkling2CrossPassMsgs(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2CrossPassMsgArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_dim == 0 or args.bank_len == 0 or args.q_rows == 0) {
            return error.ZeroDimension;
        }
        if (args.bank_len > MAX_ATTN_RES_BLOCKS) return error.InvalidConfig;
        const scale = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.hidden_dim))));
        const params: Inkling2CrossPassMsgParams = .{
            .m = args.m,
            .hidden_dim = args.hidden_dim,
            .bank_len = args.bank_len,
            .q_rows = args.q_rows,
            .kv_rows = if (args.kv_rows > 0) args.kv_rows else args.q_rows,
            .loop_layer_idx = args.loop_layer_idx,
            .has_kv_q = if (args.kv_q_src != null) 1 else 0,
            .queries_f16 = if (args.queries_f16) 1 else 0,
            .scale = scale,
        };
        try frame.encodeThreadgroups(
            self.inkling2_cross_pass_msgs,
            &.{
                args.msg_bank,
                args.q_src,
                args.kv_q_src orelse args.q_src,
                args.out_msgs,
            },
            std.mem.asBytes(&params),
            .{ 4, args.m, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Two-stage GPU `stepMessageGate` (`gate_rank = 64` or generic):
    /// Stage 1: `z = down_weight @ [UnweightedRMSNorm(cur), UnweightedRMSNorm(msg)]` -> `[m, gate_rank] f32`;
    /// Stage 2: `out = (1 - tanh(forget @ z)) * cur + act(gain @ z) * msg` -> `[m, state_dim] f16`.
    pub fn encodeInkling2MessageGate(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2MessageGateArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.state_dim == 0 or args.down_in_dim == 0 or args.gate_rank == 0) {
            return error.ZeroDimension;
        }
        if (args.down_in_dim != args.state_dim and args.down_in_dim != args.state_dim * 2) {
            return error.InvalidConfig;
        }
        const dt_idx = @backingInt(args.weight_dtype);
        const params: Inkling2MessageGateParams = .{
            .m = args.m,
            .state_dim = args.state_dim,
            .down_in_dim = args.down_in_dim,
            .gate_rank = args.gate_rank,
            .has_norm_w = if (args.norm_weight != null) 1 else 0,
            .has_forget_bias = if (args.forget_bias != null) 1 else 0,
            .has_gain_bias = if (args.gain_bias != null) 1 else 0,
            .msg_f16 = if (args.msg_f16) 1 else 0,
            .gain_act = @backingInt(args.gain_act),
            .rms_eps = args.rms_eps,
        };
        const gx_down: u64 = (@as(u64, args.gate_rank) + 3) / 4;
        const gy: u64 = (@as(u64, args.m) + 3) / 4;
        try frame.encodeThreadgroups(
            self.inkling2_msg_gate_down[dt_idx],
            &.{
                args.cur,
                args.msg,
                args.down_weight,
                args.norm_weight orelse args.z_scratch,
                args.z_scratch,
            },
            std.mem.asBytes(&params),
            .{ gx_down, gy, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );

        const gx_up: u64 = (@as(u64, args.state_dim) + 7) / 8;
        try frame.encodeThreadgroups(
            self.inkling2_msg_gate_up[dt_idx],
            &.{
                args.z_scratch,
                args.forget_weight,
                args.gain_weight,
                args.forget_bias orelse args.z_scratch,
                args.gain_bias orelse args.z_scratch,
                args.cur,
                args.msg,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ gx_up, gy, 1 },
            .{ 128, 1, 1 },
            true,
        );
    }

    pub fn encodeInkling2GatherRows(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2GatherRowsArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.count == 0 or args.row_elems == 0) return error.ZeroDimension;
        const params: Inkling2GatherRowsParams = .{
            .count = args.count,
            .row_elems = args.row_elems,
        };
        const gx: u64 = (@as(u64, args.row_elems) + 255) / 256;
        try frame.encodeThreadgroups(
            self.inkling2_gather_rows_f16,
            &.{ args.src, args.indices, args.dst },
            std.mem.asBytes(&params),
            .{ gx, args.count, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeInkling2ScatterRows(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2ScatterRowsArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.count == 0 or args.row_elems == 0) return error.ZeroDimension;
        if (args.slot_stride > 0 and args.slot_k_indices == null) return error.MissingInputs;
        const params: Inkling2ScatterRowsParams = .{
            .count = args.count,
            .row_elems = args.row_elems,
            .slot_stride = args.slot_stride,
        };
        const gx: u64 = (@as(u64, args.row_elems) + 255) / 256;
        try frame.encodeThreadgroups(
            self.inkling2_scatter_rows_f16,
            &.{
                args.src,
                args.slots_or_tok,
                args.slot_k_indices orelse args.slots_or_tok,
                args.dst,
            },
            std.mem.asBytes(&params),
            .{ gx, args.count, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Inkling 2 top-K CSR router populating `MoeWorkspace` schedule + 5 scheme
    /// bucket headers (`MTLDispatchThreadgroupsIndirectArguments`) with 0 CPU sync.
    pub fn encodeInkling2MoeRouteCsr(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2MoeRouteCsrArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0) return error.ZeroDimension;
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExperts;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;
        const ws = args.workspace;
        if (args.m > ws.spec.max_tokens or
            args.num_experts > ws.spec.num_experts or
            args.top_k > ws.spec.top_k)
        {
            return error.WorkspaceTooSmall;
        }
        if (args.is_heterogeneous and args.expert_schemes == null) return error.MissingInputs;

        const wb = args.workspace_buf;
        const eff_inter = if (args.intermediate_size > 0) args.intermediate_size else ws.spec.intermediate_size;
        const eff_hidden = if (args.hidden_size > 0) args.hidden_size else ws.spec.hidden_size;
        const params: Inkling2MoeRouteCsrParams = .{
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .has_bias = if (args.bias != null) 1 else 0,
            .bias_f16 = if (args.bias_f16) 1 else 0,
            .bias_mode = if (args.bias_add_to_logits) 0 else 1,
            .norm_topk_prob = if (args.norm_topk_prob) 1 else 0,
            .scoring_sigmoid = if (args.scoring_sigmoid) 1 else 0,
            .is_heterogeneous = if (args.is_heterogeneous) 1 else 0,
            .bucket_unique_stride = ws.bucket_unique_stride_u32,
            .bucket_offsets_stride = ws.bucket_offsets_stride_u32,
            .gate_up_groups_x_q = moe.gateUpGroupsXQ(eff_inter),
            .gate_up_groups_x_f = moe.gateUpGroupsXF(eff_inter),
            .down_groups_x = moe.downGroupsX(eff_hidden),
            .down_groups_x_q8 = moe.downGroupsXQ8(eff_hidden),
            .route_scale = args.route_scale,
            .router_softcap = args.router_softcap,
        };
        const pipe = self.inkling2_moe_route_csr[@intFromBool(args.logits_f16)];
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.logits,
                args.bias orelse args.logits,
                .{ .buffer = wb, .offset = ws.schedule_header_off },
                .{ .buffer = wb, .offset = ws.unique_experts_off },
                .{ .buffer = wb, .offset = ws.expert_token_offsets_off },
                .{ .buffer = wb, .offset = ws.token_indices_off },
                .{ .buffer = wb, .offset = ws.routing_weights_off },
                .{ .buffer = wb, .offset = ws.slot_k_indices_off },
                .{ .buffer = wb, .offset = ws.scratch_topk_experts_off },
                .{ .buffer = wb, .offset = ws.scratch_topk_weights_off },
                args.expert_schemes orelse args.logits,
                .{ .buffer = wb, .offset = ws.bucket_headers_off },
                .{ .buffer = wb, .offset = ws.bucket_unique_experts_off },
                .{ .buffer = wb, .offset = ws.bucket_expert_token_offsets_off },
            },
            std.mem.asBytes(&params),
            .{ 1, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Dispatches fused `Inkling 2` Gate+Up + SoftcappedSwiGLU (`pair_interleaved` or split)
    /// across any `.amf` `DType` (`0..37`, excluding `f32`/`u32`), directly or via GPU indirect args.
    pub fn encodeInkling2MoeGateUpSoftcap(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2MoeGateUpSoftcapArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0 or
            args.hidden_size == 0 or args.intermediate_size == 0)
        {
            return error.ZeroDimension;
        }
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExperts;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;
        const ws = args.workspace;
        if (args.m > ws.spec.max_tokens or
            args.num_experts > ws.spec.num_experts or
            args.top_k > ws.spec.top_k or
            args.intermediate_size > ws.spec.intermediate_size)
        {
            return error.WorkspaceTooSmall;
        }
        const cap_active: u32 = @min(args.max_active_override orelse args.num_experts, args.num_experts);
        const max_active: u32 = @min(args.m * args.top_k, cap_active);
        if (max_active == 0) return;

        const wb = args.workspace_buf;
        const hdr_off = if (args.scheme_bucket) |s| ws.bucketHeaderOffset(s) else ws.schedule_header_off;
        const unique_off = if (args.scheme_bucket) |s| ws.bucketUniqueOffset(s) else ws.unique_experts_off;
        const offsets_off = if (args.scheme_bucket) |s| ws.bucketOffsetsOffset(s) else ws.expert_token_offsets_off;

        const hdr_b = metal.BufferBinding{ .buffer = wb, .offset = hdr_off };
        const uniq_b = metal.BufferBinding{ .buffer = wb, .offset = unique_off };
        const offs_b = metal.BufferBinding{ .buffer = wb, .offset = offsets_off };
        const tok_b = metal.BufferBinding{ .buffer = wb, .offset = ws.token_indices_off };
        const inter_b = metal.BufferBinding{ .buffer = wb, .offset = ws.inter_off };

        const base_bindings = [_]metal.BufferBinding{
            args.x,
            args.weights_base,
            args.expert_table,
            hdr_b,
            uniq_b,
            offs_b,
            tok_b,
            inter_b,
        };

        switch (args.amf_dtype) {
            .f16, .bf16 => {
                const pipe = self.inkling2_moe_gate_up_softcap_float[@intFromBool(args.amf_dtype == .bf16)];
                const row_bytes = std.math.mul(u32, args.hidden_size, 2) catch return error.Overflow;
                const params: Inkling2MoeGateUpSoftcapParams = .{
                    .hidden_size = args.hidden_size,
                    .intermediate_size = args.intermediate_size,
                    .num_groups = 0,
                    .row_bytes = row_bytes,
                    .group_size = 0,
                    .pair_interleaved = if (args.pair_interleaved) 1 else 0,
                    .scale_stride = 0,
                    .gate_outer_scale_off = 0xFFFF_FFFF,
                    .up_outer_scale_off = 0xFFFF_FFFF,
                    .outer_scale_fallback = 1.0,
                    .gate_softcap = args.gate_softcap,
                    .up_softcap = args.up_softcap,
                };
                if (args.use_indirect) {
                    const ind = metal.BufferBinding{
                        .buffer = wb,
                        .offset = ws.scheduleIndirectOffset(args.scheme_bucket, moe.INDIRECT_GATE_UP_F_OFFSET),
                    };
                    try frame.encodeThreadgroupsIndirect(pipe, &base_bindings, std.mem.asBytes(&params), ind, .{ 128, 1, 1 }, insert_barrier_before);
                } else {
                    const gx = moe.gateUpGroupsXF(args.intermediate_size);
                    try frame.encodeThreadgroups(pipe, &base_bindings, std.mem.asBytes(&params), .{ gx, max_active, 1 }, .{ 128, 1, 1 }, insert_barrier_before);
                }
            },
            .f32, .u32 => return error.InvalidArgument,
            else => {
                const qf_dt = quant_formats.DType.fromAmf(args.amf_dtype) orelse return error.InvalidArgument;
                const pipe = self.inkling2_moe_gate_up_softcap_quant[@backingInt(qf_dt)] orelse return error.PipelineNotInitialized;
                const row_check: usize = if (args.pair_interleaved)
                    std.math.mul(usize, args.intermediate_size, 2) catch return error.Overflow
                else
                    args.intermediate_size;
                qf_dt.validateShape(row_check, args.hidden_size, args.group_size) catch |err| return switch (err) {
                    error.ZeroDimension => error.ZeroDimension,
                    error.InvalidGroupSize => error.InvalidGroupSize,
                    error.UnalignedK => error.UnalignedK,
                    error.Overflow => error.Overflow,
                };
                const num_groups = qf_dt.groupsPerRow(args.hidden_size, args.group_size) catch |err| return switch (err) {
                    error.ZeroDimension => error.ZeroDimension,
                    error.InvalidGroupSize => error.InvalidGroupSize,
                    error.UnalignedK => error.UnalignedK,
                    error.Overflow => error.Overflow,
                };
                const row_bytes = qf_dt.rowWeightBytes(args.hidden_size, args.group_size) catch |err| return switch (err) {
                    error.ZeroDimension => error.ZeroDimension,
                    error.InvalidGroupSize => error.InvalidGroupSize,
                    error.UnalignedK => error.UnalignedK,
                    error.Overflow => error.Overflow,
                };
                const eff_gs = qf_dt.effectiveGroupSize(args.group_size);
                const scale_stride: u32 = switch (qf_dt) {
                    .fp8_e4m3_block => @intCast((@as(usize, args.hidden_size) + 127) / 128),
                    .fp8_e4m3_tensor => if (args.tensor_scale_per_row) 1 else 0,
                    else => 0,
                };
                var gate_outer_off: u32 = 0xFFFF_FFFF;
                var up_outer_off: u32 = 0xFFFF_FFFF;
                var outer_fallback: f32 = 1.0;
                if (qf_dt == .nvfp4_g16) {
                    if (args.nvfp4_outer_scale) |ov| {
                        outer_fallback = ov;
                    } else {
                        const off = quant_formats.DType.nvfp4OuterScaleOffset(row_check, args.hidden_size) catch return error.UnalignedK;
                        gate_outer_off = @intCast(off);
                        up_outer_off = @intCast(off);
                    }
                }
                const params: Inkling2MoeGateUpSoftcapParams = .{
                    .hidden_size = args.hidden_size,
                    .intermediate_size = args.intermediate_size,
                    .num_groups = @intCast(num_groups),
                    .row_bytes = @intCast(row_bytes),
                    .group_size = eff_gs,
                    .pair_interleaved = if (args.pair_interleaved) 1 else 0,
                    .scale_stride = scale_stride,
                    .gate_outer_scale_off = gate_outer_off,
                    .up_outer_scale_off = up_outer_off,
                    .outer_scale_fallback = outer_fallback,
                    .gate_softcap = args.gate_softcap,
                    .up_softcap = args.up_softcap,
                };
                const ind = metal.BufferBinding{
                    .buffer = wb,
                    .offset = ws.scheduleIndirectOffset(args.scheme_bucket, moe.INDIRECT_GATE_UP_Q_OFFSET),
                };
                const gx = moe.gateUpGroupsXQ(args.intermediate_size);

                if (qf_dt.needsCodebook()) {
                    const qf = args.qf_pipes orelse return error.PipelineNotInitialized;
                    const cb = metal.BufferBinding{ .buffer = qf.codebook_buf, .offset = qf_dt.codebookOffset() };
                    const iq_bindings = [_]metal.BufferBinding{
                        args.x,
                        args.weights_base,
                        args.expert_table,
                        hdr_b,
                        uniq_b,
                        offs_b,
                        tok_b,
                        inter_b,
                        cb,
                    };
                    if (args.use_indirect) {
                        try frame.encodeThreadgroupsIndirect(pipe, &iq_bindings, std.mem.asBytes(&params), ind, .{ 128, 1, 1 }, insert_barrier_before);
                    } else {
                        try frame.encodeThreadgroups(pipe, &iq_bindings, std.mem.asBytes(&params), .{ gx, max_active, 1 }, .{ 128, 1, 1 }, insert_barrier_before);
                    }
                } else {
                    if (args.use_indirect) {
                        try frame.encodeThreadgroupsIndirect(pipe, &base_bindings, std.mem.asBytes(&params), ind, .{ 128, 1, 1 }, insert_barrier_before);
                    } else {
                        try frame.encodeThreadgroups(pipe, &base_bindings, std.mem.asBytes(&params), .{ gx, max_active, 1 }, .{ 128, 1, 1 }, insert_barrier_before);
                    }
                }
            },
        }
    }

    /// Complete zero-CPU-readback GPU-indirect `Inkling 2` MoE FFN layer for `cold == 0`
    /// (`M >= 1`, `top_k <= 16`, `num_experts <= 1024`), supporting both uniform and
    /// up to `moe.NUM_SCHEME_BUCKETS` heterogeneous buckets across `f16`/`bf16`, `GSQ-RCO v2`,
    /// `W21` FP8/MXFP8/NVFP4, and `W22` GGUF Q/K/IQ formats with SoftcappedSwiGLU.
    pub fn encodeInkling2MoeLayerIndirect(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2MoeIndirectLayerArgs,
        insert_barrier_before: bool,
    ) LayerEncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0 or
            args.hidden_size == 0 or args.intermediate_size == 0)
        {
            return error.ZeroDimension;
        }
        var logits_bind: metal.BufferBinding = undefined;
        var logits_f16: bool = false;
        var route_barrier = insert_barrier_before;
        switch (args.router) {
            .logits => |l| {
                logits_bind = l.binding;
                logits_f16 = l.f16;
            },
            .linear => |lin| {
                if (lin.n != args.num_experts or lin.k != args.hidden_size) return error.InvalidConfig;
                const dp = args.dequant_pipes orelse return error.PipelineNotInitialized;
                const r_mode: dequant.UnifiedLinearMode = if (args.m == 1)
                    .gemv
                else if (args.m < 16 or args.router_tile_f16 == null)
                    .batched
                else
                    .gemm32;
                logits_bind = args.workspace.routerLogitsBinding(args.workspace_buf);
                try dp.encodeLinearUnified(frame, .{
                    .amf_dtype = lin.amf_dtype,
                    .group_size = lin.group_size,
                    .tensor_scale_per_row = lin.tensor_scale_per_row,
                    .weight = lin.weight,
                    .params = lin.params,
                    .x = args.x,
                    .y = logits_bind,
                    .m = args.m,
                    .n = lin.n,
                    .k = lin.k,
                    .mode = r_mode,
                    .qf_pipes = args.qf_pipes,
                    .tile_f16 = args.router_tile_f16,
                    .tile_f16_bytes = args.router_tile_f16_bytes,
                }, insert_barrier_before);
                logits_f16 = true;
                route_barrier = true;
            },
        }

        try self.encodeInkling2MoeRouteCsr(frame, .{
            .logits = logits_bind,
            .logits_f16 = logits_f16,
            .bias = args.bias,
            .bias_f16 = args.bias_f16,
            .bias_add_to_logits = args.bias_add_to_logits,
            .scoring_sigmoid = args.scoring_sigmoid,
            .workspace = args.workspace,
            .workspace_buf = args.workspace_buf,
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
            .norm_topk_prob = args.norm_topk_prob,
            .route_scale = args.route_scale,
            .router_softcap = args.router_softcap,
            .is_heterogeneous = args.is_heterogeneous,
            .expert_schemes = args.expert_schemes,
        }, route_barrier);

        var any_gu = false;
        if (args.is_heterogeneous) {
            const num_buckets = moe.countActiveSchemeBuckets(args.scheme_counts);
            if (num_buckets == 0) return error.InvalidConfig;
            const use_indirect = moe.shouldUseIndirectDispatch(args.m, num_buckets);
            var first_gu = true;
            for (0..moe.NUM_SCHEME_BUCKETS) |s_idx| {
                const cap = args.scheme_counts[s_idx];
                if (cap == 0) continue;
                const scheme: quant.WeightDType = @fromBackingInt(@intCast(s_idx));
                const desc = args.scheme_descs[s_idx];
                const pair_i = if (args.bucket_pair_interleaved) |bp| bp[s_idx] else args.pair_interleaved;
                var x_eff = args.x;
                var gu_barrier = first_gu;
                const aq_gu = args.act_quant_policy.resolveForTensor(desc.gate_up_dtype, args.hidden_size);
                if (aq_gu != .none) {
                    const need = std.math.mul(u64, @as(u64, args.m) * args.hidden_size, @sizeOf(f16)) catch return error.Overflow;
                    if (args.qf_pipes) |qf| {
                        if (args.act_scratch) |scratch| {
                            if (args.act_scratch_bytes >= need) {
                                try qf.encodeActQuantEmulate(frame, aq_gu, .{
                                    .x_in = args.x,
                                    .x_out = scratch,
                                    .m = args.m,
                                    .k = args.hidden_size,
                                }, true);
                                x_eff = scratch;
                                gu_barrier = true;
                            }
                        }
                    }
                }
                try self.encodeInkling2MoeGateUpSoftcap(frame, .{
                    .amf_dtype = desc.gate_up_dtype,
                    .group_size = desc.gate_up_group_size,
                    .tensor_scale_per_row = desc.gate_up_tensor_scale_per_row,
                    .pair_interleaved = pair_i,
                    .x = x_eff,
                    .weights_base = args.weights_base,
                    .expert_table = args.expert_table,
                    .workspace = args.workspace,
                    .workspace_buf = args.workspace_buf,
                    .qf_pipes = args.qf_pipes,
                    .m = args.m,
                    .num_experts = args.num_experts,
                    .top_k = args.top_k,
                    .hidden_size = args.hidden_size,
                    .intermediate_size = args.intermediate_size,
                    .scheme_bucket = scheme,
                    .max_active_override = cap,
                    .use_indirect = use_indirect,
                    .gate_softcap = args.gate_softcap,
                    .up_softcap = args.up_softcap,
                }, gu_barrier);
                first_gu = false;
                any_gu = true;
            }
        } else {
            const desc = args.scheme_descs[0];
            const use_indirect = moe.shouldUseIndirectDispatch(args.m, 1);
            var x_eff = args.x;
            const aq_gu = args.act_quant_policy.resolveForTensor(desc.gate_up_dtype, args.hidden_size);
            if (aq_gu != .none) {
                const need = std.math.mul(u64, @as(u64, args.m) * args.hidden_size, @sizeOf(f16)) catch return error.Overflow;
                if (args.qf_pipes) |qf| {
                    if (args.act_scratch) |scratch| {
                        if (args.act_scratch_bytes >= need) {
                            try qf.encodeActQuantEmulate(frame, aq_gu, .{
                                .x_in = args.x,
                                .x_out = scratch,
                                .m = args.m,
                                .k = args.hidden_size,
                            }, true);
                            x_eff = scratch;
                        }
                    }
                }
            }
            try self.encodeInkling2MoeGateUpSoftcap(frame, .{
                .amf_dtype = desc.gate_up_dtype,
                .group_size = desc.gate_up_group_size,
                .tensor_scale_per_row = desc.gate_up_tensor_scale_per_row,
                .pair_interleaved = args.pair_interleaved,
                .x = x_eff,
                .weights_base = args.weights_base,
                .expert_table = args.expert_table,
                .workspace = args.workspace,
                .workspace_buf = args.workspace_buf,
                .qf_pipes = args.qf_pipes,
                .m = args.m,
                .num_experts = args.num_experts,
                .top_k = args.top_k,
                .hidden_size = args.hidden_size,
                .intermediate_size = args.intermediate_size,
                .scheme_bucket = null,
                .max_active_override = args.num_experts,
                .use_indirect = use_indirect,
                .gate_softcap = args.gate_softcap,
                .up_softcap = args.up_softcap,
            }, true);
            any_gu = true;
        }
        if (!any_gu) return error.InvalidConfig;

        try args.moe_pipes.encodeDownAndReduceMultiFormat(frame, .{
            .base = .{
                .x = args.x,
                .router = .{ .logits = .{
                    .binding = logits_bind,
                    .dtype = if (logits_f16) .f16 else .f32,
                } },
                .weights_base = args.weights_base,
                .expert_table = args.expert_table,
                .workspace = args.workspace,
                .workspace_buf = args.workspace_buf,
                .residual = args.residual,
                .out = args.out,
                .m = args.m,
                .num_experts = args.num_experts,
                .top_k = args.top_k,
                .hidden_size = args.hidden_size,
                .intermediate_size = args.intermediate_size,
                .norm_topk_prob = args.norm_topk_prob,
            },
            .is_heterogeneous = args.is_heterogeneous,
            .expert_schemes = args.expert_schemes,
            .scheme_counts = args.scheme_counts,
            .scheme_descs = args.scheme_descs,
            .qf_pipes = args.qf_pipes,
            .act_quant_policy = args.act_quant_policy,
            .act_scratch = args.act_scratch,
            .act_scratch_bytes = args.act_scratch_bytes,
        }, true);
    }

    /// Executes a single active `Inkling 2` expert (`cold > 0` / `DFlare` verify) across
    /// all `M_e >= 1` tokens routed to it, streaming `w1` (`pair_interleaved` or split
    /// gate + `w_up`) and `w2` weights ONCE per expert rather than `M_e` separate GEMVs.
    pub fn encodeInkling2GroupedExpertMlp(
        self: Inkling2Hy4Pipelines,
        frame: *metal.Compute.Frame,
        args: Inkling2GroupedExpertMlpArgs,
        insert_barrier_before: bool,
    ) LayerEncodeError!void {
        if (args.m_e == 0 or args.hidden_size == 0 or args.intermediate_size == 0) {
            return error.ZeroDimension;
        }
        const expected_w1_n = if (args.w_up == null)
            std.math.mul(u32, args.intermediate_size, 2) catch return error.Overflow
        else
            args.intermediate_size;
        if (args.w1.k != args.hidden_size or args.w1.n != expected_w1_n) return error.InvalidConfig;
        if (args.w_up) |wu| {
            if (wu.k != args.hidden_size or wu.n != args.intermediate_size) return error.InvalidConfig;
        }
        if (args.w2.k != args.intermediate_size or args.w2.n != args.hidden_size) return error.InvalidConfig;

        var x_in = args.x;
        var barrier = insert_barrier_before;
        if (args.m_e == 1) {
            const x_off = std.math.mul(u64, args.single_token_idx, @as(u64, args.hidden_size) * @sizeOf(f16)) catch return error.Overflow;
            x_in = .{ .buffer = args.x.buffer, .offset = args.x.offset + x_off };
        } else {
            const gx = args.gathered_x orelse return error.MissingInputs;
            const t_idx = args.token_indices_dev orelse return error.MissingInputs;
            try self.encodeInkling2GatherRows(frame, .{
                .src = args.x,
                .indices = t_idx,
                .dst = gx,
                .count = args.m_e,
                .row_elems = args.hidden_size,
            }, insert_barrier_before);
            x_in = gx;
            barrier = true;
        }

        const lin_mode: dequant.UnifiedLinearMode = if (args.m_e == 1)
            .gemv
        else if (args.m_e < 16 or args.tile_f16 == null)
            .batched
        else
            .gemm32;

        try args.dequant_pipes.encodeLinearUnified(frame, .{
            .amf_dtype = args.w1.amf_dtype,
            .group_size = args.w1.group_size,
            .tensor_scale_per_row = args.w1.tensor_scale_per_row,
            .weight = args.w1.weight,
            .params = args.w1.params,
            .x = x_in,
            .y = args.gate_up_buf,
            .m = args.m_e,
            .n = args.w1.n,
            .k = args.w1.k,
            .mode = lin_mode,
            .qf_pipes = args.qf_pipes,
            .tile_f16 = args.tile_f16,
            .tile_f16_bytes = args.tile_f16_bytes,
            .act_quant_policy = args.act_quant_policy,
            .act_scratch = args.act_scratch,
            .act_scratch_bytes = args.act_scratch_bytes,
        }, barrier);

        if (args.w_up) |wu| {
            const ub = args.up_buf orelse return error.MissingInputs;
            const aq_w1 = args.act_quant_policy.resolveForTensor(args.w1.amf_dtype, args.w1.k);
            const up_barrier = (aq_w1 != .none);
            try args.dequant_pipes.encodeLinearUnified(frame, .{
                .amf_dtype = wu.amf_dtype,
                .group_size = wu.group_size,
                .tensor_scale_per_row = wu.tensor_scale_per_row,
                .weight = wu.weight,
                .params = wu.params,
                .x = x_in,
                .y = ub,
                .m = args.m_e,
                .n = wu.n,
                .k = wu.k,
                .mode = lin_mode,
                .qf_pipes = args.qf_pipes,
                .tile_f16 = args.tile_f16,
                .tile_f16_bytes = args.tile_f16_bytes,
                .act_quant_policy = args.act_quant_policy,
                .act_scratch = args.act_scratch,
                .act_scratch_bytes = args.act_scratch_bytes,
            }, up_barrier);
            try self.encodeInkling2MoeSwigluSoftcap(frame, .{
                .gate_in = args.gate_up_buf,
                .up_in = ub,
                .out = args.act_buf,
                .num_rows = args.m_e,
                .intermediate_size = args.intermediate_size,
                .pair_interleaved = false,
                .softcap = args.up_softcap,
                .gate_softcap = args.gate_softcap,
            }, true);
        } else {
            try self.encodeInkling2MoeSwigluSoftcap(frame, .{
                .gate_in = args.gate_up_buf,
                .out = args.act_buf,
                .num_rows = args.m_e,
                .intermediate_size = args.intermediate_size,
                .pair_interleaved = true,
                .softcap = args.up_softcap,
                .gate_softcap = args.gate_softcap,
            }, true);
        }

        if (args.m_e == 1) {
            try args.dequant_pipes.encodeLinearUnified(frame, .{
                .amf_dtype = args.w2.amf_dtype,
                .group_size = args.w2.group_size,
                .tensor_scale_per_row = args.w2.tensor_scale_per_row,
                .weight = args.w2.weight,
                .params = args.w2.params,
                .x = args.act_buf,
                .y = args.expert_out,
                .m = 1,
                .n = args.w2.n,
                .k = args.w2.k,
                .mode = .gemv,
                .qf_pipes = args.qf_pipes,
                .tile_f16 = args.tile_f16,
                .tile_f16_bytes = args.tile_f16_bytes,
                .act_quant_policy = args.act_quant_policy,
                .act_scratch = args.act_scratch,
                .act_scratch_bytes = args.act_scratch_bytes,
            }, true);
        } else {
            const dn_tmp = args.down_tmp_buf orelse return error.MissingInputs;
            const dst_slots = args.dst_slots_or_tok_dev orelse return error.MissingInputs;
            try args.dequant_pipes.encodeLinearUnified(frame, .{
                .amf_dtype = args.w2.amf_dtype,
                .group_size = args.w2.group_size,
                .tensor_scale_per_row = args.w2.tensor_scale_per_row,
                .weight = args.w2.weight,
                .params = args.w2.params,
                .x = args.act_buf,
                .y = dn_tmp,
                .m = args.m_e,
                .n = args.w2.n,
                .k = args.w2.k,
                .mode = lin_mode,
                .qf_pipes = args.qf_pipes,
                .tile_f16 = args.tile_f16,
                .tile_f16_bytes = args.tile_f16_bytes,
                .act_quant_policy = args.act_quant_policy,
                .act_scratch = args.act_scratch,
                .act_scratch_bytes = args.act_scratch_bytes,
            }, true);
            try self.encodeInkling2ScatterRows(frame, .{
                .src = dn_tmp,
                .slots_or_tok = dst_slots,
                .slot_k_indices = args.slot_k_indices_dev,
                .dst = args.expert_out,
                .count = args.m_e,
                .row_elems = args.hidden_size,
                .slot_stride = args.slot_stride,
            }, true);
        }
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

pub fn inkling2ScaledL2ClipPerHeadCpuRef(
    in_buf: []const f16,
    scale_vec: ?[]const f32,
    positions: ?[]const u32,
    m: u32,
    num_heads: u32,
    head_dim: u32,
    threshold: f32,
    use_log_scaling: bool,
    base_pos: u32,
    log_floor: u32,
    log_alpha: f32,
    out_f32: []f32,
) void {
    const hd: usize = head_dim;
    const nh_total: usize = num_heads;
    const floor_f: f32 = if (log_floor > 0) @floatFromInt(log_floor) else 1.0;
    for (0..m) |tok| {
        const tau: f32 = if (use_log_scaling and log_alpha != 0.0) blk: {
            const pos: u32 = if (positions) |lp| lp[tok] else (base_pos + @as(u32, @intCast(tok)));
            const ratio: f32 = @max(1.0, (@as(f32, @floatFromInt(pos)) + 1.0) / floor_f);
            break :blk 1.0 + log_alpha * @log(ratio);
        } else 1.0;
        for (0..nh_total) |nh| {
            const base = (tok * nh_total + nh) * hd;
            const sv_off = nh * hd;
            var sq: f32 = 0.0;
            for (0..hd) |i| {
                var fv: f32 = @floatCast(in_buf[base + i]);
                if (scale_vec) |sv| fv *= sv[sv_off + i];
                sq += fv * fv;
            }
            const norm = @sqrt(sq);
            const clip: f32 = if (threshold > 0.0) @min(1.0, threshold / (norm + 1e-6)) else 1.0;
            const fac: f32 = clip * tau;
            for (0..hd) |i| {
                var fv: f32 = @floatCast(in_buf[base + i]);
                if (scale_vec) |sv| fv *= sv[sv_off + i];
                out_f32[base + i] = fv * fac;
            }
        }
    }
}

pub fn inkling2UnitRmsNormPerHeadCpuRef(
    in_buf: []const f16,
    m: u32,
    num_heads: u32,
    head_dim: u32,
    rms_eps: f32,
    out_f32: []f32,
) void {
    const hd: usize = head_dim;
    const nh_total: usize = num_heads;
    for (0..m) |tok| {
        for (0..nh_total) |nh| {
            const base = (tok * nh_total + nh) * hd;
            var sq: f32 = 0.0;
            for (0..hd) |i| {
                const v: f32 = @floatCast(in_buf[base + i]);
                sq += v * v;
            }
            const inv_rms: f32 = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(head_dim)) + rms_eps);
            for (0..hd) |i| {
                out_f32[base + i] = @as(f32, @floatCast(in_buf[base + i])) * inv_rms;
            }
        }
    }
}

pub fn inkling2DynSconvAssembleCpuRef(
    proj_f16: []const f16,
    static_w_f16: ?[]const f16,
    static_w_f32: ?[]const f32,
    m: u32,
    kv_dim: u32,
    num_kv_heads: u32,
    head_dim: u32,
    proj_n: u32,
    weight_scale: f32,
    out_f32: []f32,
) void {
    const kvd: usize = kv_dim;
    const n_kvh: usize = num_kv_heads;
    const hd: usize = head_dim;
    for (0..m) |tok| {
        for (0..kvd) |c| {
            const p_off: usize = if (proj_n == kv_dim * 4)
                (tok * kvd + c) * 4
            else if (proj_n == num_kv_heads * 4 and hd > 0)
                (tok * n_kvh + (c / hd)) * 4
            else
                tok * 4;
            var w0: f32 = @floatCast(proj_f16[p_off + 3]);
            var w1: f32 = @floatCast(proj_f16[p_off + 2]);
            var w2: f32 = @floatCast(proj_f16[p_off + 1]);
            var w3: f32 = @floatCast(proj_f16[p_off + 0]);
            if (static_w_f16) |sf16| {
                w0 += @floatCast(sf16[c * 4 + 0]);
                w1 += @floatCast(sf16[c * 4 + 1]);
                w2 += @floatCast(sf16[c * 4 + 2]);
                w3 += @floatCast(sf16[c * 4 + 3]);
            } else if (static_w_f32) |sf32| {
                w0 += sf32[c * 4 + 0];
                w1 += sf32[c * 4 + 1];
                w2 += sf32[c * 4 + 2];
                w3 += sf32[c * 4 + 3];
            }
            const dst = (tok * kvd + c) * 4;
            out_f32[dst + 0] = w0 * weight_scale;
            out_f32[dst + 1] = w1 * weight_scale;
            out_f32[dst + 2] = w2 * weight_scale;
            out_f32[dst + 3] = w3 * weight_scale;
        }
    }
}

pub fn inkling2MsgBankNormCpuRef(
    block_states: []const f16,
    block_slots: []const u32,
    block_stride_elems: u32,
    cur_f16: ?[]const f16,
    m: u32,
    hidden_dim: u32,
    rms_eps: f32,
    out_bank_f32: []f32,
) usize {
    const h: usize = hidden_dim;
    const m_us: usize = m;
    const mh: usize = m_us * h;
    var bank_len: usize = 0;

    for (block_slots) |slot| {
        const src = block_states[@as(usize, slot) * block_stride_elems ..][0..mh];
        const dst = out_bank_f32[bank_len * mh ..][0..mh];
        for (0..m_us) |tok| {
            var sq: f32 = 0.0;
            for (0..h) |c| {
                const v: f32 = @floatCast(src[tok * h + c]);
                dst[tok * h + c] = v;
                sq += v * v;
            }
            const inv_rms: f32 = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(hidden_dim)) + rms_eps);
            for (0..h) |c| dst[tok * h + c] *= inv_rms;
        }
        bank_len += 1;
    }

    if (cur_f16) |cur| {
        const dst = out_bank_f32[bank_len * mh ..][0..mh];
        for (0..m_us) |tok| {
            var sq: f32 = 0.0;
            for (0..h) |c| {
                const v: f32 = @floatCast(cur[tok * h + c]);
                dst[tok * h + c] = v;
                sq += v * v;
            }
            const inv_rms: f32 = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(hidden_dim)) + rms_eps);
            for (0..h) |c| dst[tok * h + c] *= inv_rms;
        }
        bank_len += 1;
    }
    return bank_len;
}

pub fn inkling2CrossPassMsgsCpuRef(
    bank_norm_f32: []const f32,
    bank_len: usize,
    q_src_f32: []const f32,
    kv_q_src_f32: ?[]const f32,
    loop_layer_idx: usize,
    m: u32,
    hidden_dim: u32,
    scale: f32,
    out_msgs: []f32,
) void {
    const h: usize = hidden_dim;
    const m_us: usize = m;
    const mh: usize = m_us * h;
    const q_rows: usize = @max(@as(usize, 1), q_src_f32.len / h);
    const kv_slice = kv_q_src_f32 orelse q_src_f32;
    const kv_rows: usize = @max(@as(usize, 1), kv_slice.len / h);

    const queries = [4][]const f32{
        q_src_f32[((2 * loop_layer_idx + 0) % q_rows) * h ..][0..h],
        kv_slice[((2 * loop_layer_idx + 0) % kv_rows) * h ..][0..h],
        kv_slice[((2 * loop_layer_idx + 1) % kv_rows) * h ..][0..h],
        q_src_f32[((2 * loop_layer_idx + 1) % q_rows) * h ..][0..h],
    };
    var scores: [MAX_ATTN_RES_BLOCKS]f32 = undefined;
    var weights: [MAX_ATTN_RES_BLOCKS]f32 = undefined;
    for (0..4) |qi| {
        const qv = queries[qi];
        for (0..m_us) |tok| {
            var max_s: f32 = -std.math.inf(f32);
            for (0..bank_len) |b| {
                const bv = bank_norm_f32[b * mh + tok * h ..][0..h];
                var dot_v: f32 = 0.0;
                for (0..h) |c| dot_v += qv[c] * bv[c];
                const s = dot_v * scale;
                scores[b] = s;
                if (s > max_s) max_s = s;
            }
            var exp_sum: f32 = 0.0;
            for (0..bank_len) |b| {
                const e = @exp(scores[b] - max_s);
                weights[b] = e;
                exp_sum += e;
            }
            const inv_sum: f32 = if (exp_sum > 0.0) 1.0 / exp_sum else 0.0;
            const dst = out_msgs[qi * mh + tok * h ..][0..h];
            for (0..h) |c| {
                var acc: f32 = 0.0;
                for (0..bank_len) |b| {
                    acc += weights[b] * inv_sum * bank_norm_f32[b * mh + tok * h + c];
                }
                dst[c] = acc;
            }
        }
    }
}

pub fn inkling2MessageGateCpuRef(
    allocator: std.mem.Allocator,
    cur_f16: []const f16,
    msg_f32: []const f32,
    down_u16: []const u16,
    forget_u16: []const u16,
    gain_u16: []const u16,
    weight_dtype: quant.FloatDType,
    norm_w: ?[]const f32,
    forget_bias: ?[]const f32,
    gain_bias: ?[]const f32,
    m: u32,
    state_dim: u32,
    down_in_dim: u32,
    gate_rank: u32,
    gain_act: Inkling2MessageGateGainAct,
    rms_eps: f32,
    out_f32: []f32,
) !void {
    const m_us: usize = m;
    const sd: usize = state_dim;
    const din: usize = down_in_dim;
    const gr: usize = gate_rank;
    const z = try allocator.alloc(f32, m_us * gr);
    defer allocator.free(z);
    const x_norm = try allocator.alloc(f32, din);
    defer allocator.free(x_norm);

    for (0..m_us) |tok| {
        var sq_c: f32 = 0.0;
        for (0..sd) |c| {
            const vc: f32 = @floatCast(cur_f16[tok * sd + c]);
            sq_c += vc * vc;
        }
        const inv_c: f32 = 1.0 / @sqrt(sq_c / @as(f32, @floatFromInt(state_dim)) + rms_eps);
        for (0..sd) |c| {
            const nw = if (norm_w) |nwa| nwa[c] else 1.0;
            x_norm[c] = @as(f32, @floatCast(cur_f16[tok * sd + c])) * inv_c * nw;
        }
        if (din == sd * 2) {
            var sq_m: f32 = 0.0;
            for (0..sd) |c| {
                const vm = msg_f32[tok * sd + c];
                sq_m += vm * vm;
            }
            const inv_m: f32 = 1.0 / @sqrt(sq_m / @as(f32, @floatFromInt(state_dim)) + rms_eps);
            for (0..sd) |c| {
                const nw = if (norm_w) |nwa| nwa[sd + c] else 1.0;
                x_norm[sd + c] = msg_f32[tok * sd + c] * inv_m * nw;
            }
        }
        for (0..gr) |r| {
            var acc: f32 = 0.0;
            for (0..din) |c| {
                acc += x_norm[c] * quant.floatBitsToF32(weight_dtype, down_u16[r * din + c]);
            }
            z[tok * gr + r] = acc;
        }
        for (0..sd) |r| {
            var f_acc: f32 = if (forget_bias) |fb| fb[r] else 0.0;
            var g_acc: f32 = if (gain_bias) |gb| gb[r] else 0.0;
            for (0..gr) |k| {
                const zv = z[tok * gr + k];
                f_acc += zv * quant.floatBitsToF32(weight_dtype, forget_u16[r * gr + k]);
                g_acc += zv * quant.floatBitsToF32(weight_dtype, gain_u16[r * gr + k]);
            }
            const fg = std.math.tanh(f_acc);
            const mg = switch (gain_act) {
                .tanh => std.math.tanh(g_acc),
                .silu => siluF32(g_acc),
            };
            const cv: f32 = @floatCast(cur_f16[tok * sd + r]);
            out_f32[tok * sd + r] = (1.0 - fg) * cv + mg * msg_f32[tok * sd + r];
        }
    }
}

fn dequantExpertRowCpuRef(
    dtype: amf.DType,
    group_size: u16,
    qweight: []const u8,
    params: []const u8,
    row: usize,
    total_rows: usize,
    k: usize,
    out_f32: []f32,
) !void {
    switch (dtype) {
        .f16, .bf16 => {
            const fdt: quant.FloatDType = if (dtype == .bf16) .bf16 else .f16;
            const u16_s: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, qweight));
            for (0..k) |c| out_f32[c] = quant.floatBitsToF32(fdt, u16_s[row * k + c]);
        },
        else => {
            const qf_dt = quant_formats.DType.fromAmf(dtype) orelse return error.InvalidArgument;
            try quant_formats.dequantRowCpuRef(qf_dt, group_size, qweight, params, row, total_rows, k, out_f32);
        },
    }
}

pub const MoeExpertCpuDesc = struct {
    gate_up_dtype: amf.DType,
    gate_up_group_size: u16 = 64,
    down_dtype: amf.DType,
    down_group_size: u16 = 64,
    pair_interleaved: bool = true,
    gate_qw: []const u8,
    gate_params: []const u8,
    up_qw: []const u8,
    up_params: []const u8,
    down_qw: []const u8,
    down_params: []const u8,
};

pub fn inkling2MoeLayerSoftcapCpuRefWithRounding(
    allocator: std.mem.Allocator,
    x_f16: []const f16,
    logits_f32: []const f32,
    bias_f32: ?[]const f32,
    bias_add_to_logits: bool,
    scoring_sigmoid: bool,
    norm_topk_prob: bool,
    route_scale: f32,
    router_softcap: f32,
    gate_softcap: f32,
    up_softcap: f32,
    experts: []const MoeExpertCpuDesc,
    residual_f16: ?[]const f16,
    round_gate_up_f16: bool,
    m: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    out_f32: []f32,
) !void {
    const m_us: usize = m;
    const ne: usize = experts.len;
    const tk: usize = top_k;
    const h: usize = hidden_size;
    const inter: usize = intermediate_size;

    const scores = try allocator.alloc(f32, ne);
    defer allocator.free(scores);
    const sel_probs = try allocator.alloc(f32, ne);
    defer allocator.free(sel_probs);
    const w_row = try allocator.alloc(f32, @max(h, inter));
    defer allocator.free(w_row);
    const gate_acts = try allocator.alloc(f32, inter);
    defer allocator.free(gate_acts);

    for (0..m_us) |tok| {
        const x_row = x_f16[tok * h ..][0..h];
        const l_row = logits_f32[tok * ne ..][0..ne];

        var max_v: f32 = -std.math.inf(f32);
        for (0..ne) |e| {
            var v = softcapF32(l_row[e], router_softcap);
            if (bias_f32 != null and bias_add_to_logits) v += bias_f32.?[e];
            scores[e] = v;
            if (v > max_v) max_v = v;
        }
        var exp_sum: f32 = 0.0;
        for (0..ne) |e| {
            if (scoring_sigmoid) {
                sel_probs[e] = sigmoidF32(scores[e]);
            } else {
                const ex = @exp(scores[e] - max_v);
                sel_probs[e] = ex;
                exp_sum += ex;
            }
        }
        if (!scoring_sigmoid) {
            const inv_s: f32 = if (exp_sum > 0.0) 1.0 / exp_sum else 0.0;
            for (0..ne) |e| sel_probs[e] *= inv_s;
        }

        var used: [MAX_EXPERTS]bool = @splat(false);
        var topk_e: [MAX_TOP_K]usize = undefined;
        var topk_w: [MAX_TOP_K]f32 = undefined;
        var w_sum: f32 = 0.0;
        for (0..tk) |ki| {
            var best_s: f32 = -std.math.inf(f32);
            var best_e: usize = 0;
            for (0..ne) |e| {
                if (used[e]) continue;
                var sel_s = sel_probs[e];
                if (bias_f32 != null and !bias_add_to_logits) sel_s += bias_f32.?[e];
                if (sel_s > best_s) {
                    best_s = sel_s;
                    best_e = e;
                }
            }
            used[best_e] = true;
            topk_e[ki] = best_e;
            topk_w[ki] = sel_probs[best_e];
            w_sum += sel_probs[best_e];
        }
        const norm_fac: f32 = if (norm_topk_prob and w_sum > 0.0) 1.0 / w_sum else 1.0;

        for (0..h) |d| {
            out_f32[tok * h + d] = if (residual_f16) |rf| @floatCast(rf[tok * h + d]) else 0.0;
        }

        for (0..tk) |ki| {
            const eid = topk_e[ki];
            const rw = topk_w[ki] * norm_fac * route_scale;
            const exp_d = experts[eid];
            const gu_total_rows = if (exp_d.pair_interleaved) inter * 2 else inter;
            for (0..inter) |r| {
                const g_row = if (exp_d.pair_interleaved) r * 2 else r;
                const u_row = if (exp_d.pair_interleaved) r * 2 + 1 else r;
                try dequantExpertRowCpuRef(exp_d.gate_up_dtype, exp_d.gate_up_group_size, exp_d.gate_qw, exp_d.gate_params, g_row, gu_total_rows, h, w_row[0..h]);
                var g_acc: f32 = 0.0;
                for (0..h) |c| g_acc += @as(f32, @floatCast(x_row[c])) * w_row[c];

                const u_qw = if (exp_d.pair_interleaved) exp_d.gate_qw else exp_d.up_qw;
                const u_pr = if (exp_d.pair_interleaved) exp_d.gate_params else exp_d.up_params;
                try dequantExpertRowCpuRef(exp_d.gate_up_dtype, exp_d.gate_up_group_size, u_qw, u_pr, u_row, gu_total_rows, h, w_row[0..h]);
                var u_acc: f32 = 0.0;
                for (0..h) |c| u_acc += @as(f32, @floatCast(x_row[c])) * w_row[c];

                const g_eff: f32 = if (round_gate_up_f16) @floatCast(@as(f16, @floatCast(g_acc))) else g_acc;
                const u_eff: f32 = if (round_gate_up_f16) @floatCast(@as(f16, @floatCast(u_acc))) else u_acc;
                const act = if (gate_softcap > 0.0)
                    softcapF32(g_eff, gate_softcap) * sigmoidF32(g_eff) * softcapF32(u_eff, up_softcap)
                else
                    softcapF32(siluF32(g_eff) * u_eff, up_softcap);
                gate_acts[r] = @as(f32, @floatCast(@as(f16, @floatCast(act))));
            }
            for (0..h) |d| {
                try dequantExpertRowCpuRef(exp_d.down_dtype, exp_d.down_group_size, exp_d.down_qw, exp_d.down_params, d, h, inter, w_row[0..inter]);
                var d_acc: f32 = 0.0;
                for (0..inter) |r| d_acc += gate_acts[r] * w_row[r];
                out_f32[tok * h + d] += rw * d_acc;
            }
        }
    }
}

pub fn inkling2MoeLayerSoftcapCpuRef(
    allocator: std.mem.Allocator,
    x_f16: []const f16,
    logits_f32: []const f32,
    bias_f32: ?[]const f32,
    bias_add_to_logits: bool,
    scoring_sigmoid: bool,
    norm_topk_prob: bool,
    route_scale: f32,
    router_softcap: f32,
    gate_softcap: f32,
    up_softcap: f32,
    experts: []const MoeExpertCpuDesc,
    residual_f16: ?[]const f16,
    m: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    out_f32: []f32,
) !void {
    return inkling2MoeLayerSoftcapCpuRefWithRounding(
        allocator,
        x_f16,
        logits_f32,
        bias_f32,
        bias_add_to_logits,
        scoring_sigmoid,
        norm_topk_prob,
        route_scale,
        router_softcap,
        gate_softcap,
        up_softcap,
        experts,
        residual_f16,
        false,
        m,
        top_k,
        hidden_size,
        intermediate_size,
        out_f32,
    );
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
        ctx.pipes.inkling2_scaled_l2_clip,
        ctx.pipes.inkling2_unit_rms_norm,
        ctx.pipes.inkling2_f16_to_f32,
        ctx.pipes.inkling2_attn_post_proj,
        ctx.pipes.inkling2_assemble_dyn_sconv_w,
        ctx.pipes.inkling2_msg_bank_norm,
        ctx.pipes.inkling2_cross_pass_msgs,
        ctx.pipes.inkling2_msg_gate_down[0],
        ctx.pipes.inkling2_msg_gate_down[1],
        ctx.pipes.inkling2_msg_gate_up[0],
        ctx.pipes.inkling2_msg_gate_up[1],
        ctx.pipes.inkling2_gather_rows_f16,
        ctx.pipes.inkling2_scatter_rows_f16,
        ctx.pipes.inkling2_moe_route_csr[0],
        ctx.pipes.inkling2_moe_route_csr[1],
        ctx.pipes.inkling2_moe_gate_up_softcap_float[0],
        ctx.pipes.inkling2_moe_gate_up_softcap_float[1],
    };
    for (all_pipes) |p| {
        try std.testing.expect(p.maxThreadsPerThreadgroup() >= 256);
    }
    inline for (@typeInfo(quant_formats.DType).@"enum".field_values) |fv| {
        const p = ctx.pipes.inkling2_moe_gate_up_softcap_quant[fv].?;
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

const TestMoeBundle = struct {
    dequant_pipes: dequant.DequantPipelines,
    moe_pipes: moe.MoePipelines,
    qf_pipes: quant_formats.QuantFormatPipelines,

    fn init(device: metal.Device, library: metal.Library) !TestMoeBundle {
        const dp = try dequant.DequantPipelines.init(device, library);
        errdefer dp.close();
        const mp = try moe.MoePipelines.init(device, library);
        errdefer mp.close();
        const qfp = try quant_formats.QuantFormatPipelines.init(device, library);
        errdefer qfp.close();
        return .{ .dequant_pipes = dp, .moe_pipes = mp, .qf_pipes = qfp };
    }

    fn deinit(self: TestMoeBundle) void {
        self.qf_pipes.close();
        self.moe_pipes.close();
        self.dequant_pipes.close();
    }
};

fn packTestAffineGroup(bits: u8, gs: usize, codes: []const u8, out: []u8) void {
    @memset(out, 0);
    switch (bits) {
        2 => for (0..gs / 4) |i| {
            out[i] = codes[4 * i] | (codes[4 * i + 1] << 2) | (codes[4 * i + 2] << 4) | (codes[4 * i + 3] << 6);
        },
        4 => for (0..gs / 2) |i| {
            out[i] = codes[2 * i] | (codes[2 * i + 1] << 4);
        },
        8 => @memcpy(out[0..gs], codes[0..gs]),
        else => {
            var bit_pos: usize = 0;
            for (0..gs) |i| {
                var val: u32 = codes[i];
                for (0..bits) |_| {
                    if ((val & 1) != 0) {
                        out[bit_pos >> 3] |= @as(u8, 1) << @intCast(bit_pos & 7);
                    }
                    val >>= 1;
                    bit_pos += 1;
                }
            }
        },
    }
}

fn fillSynthMoeMatrix(
    dtype: amf.DType,
    group_size: u16,
    rows: usize,
    cols: usize,
    qweight: []u8,
    params: []u8,
    seed: usize,
) !void {
    if (dtype == .f16 or dtype == .bf16) {
        const fdt: quant.FloatDType = if (dtype == .bf16) .bf16 else .f16;
        const u16_s: []u16 = @alignCast(std.mem.bytesAsSlice(u16, qweight[0 .. rows * cols * 2]));
        for (0..rows) |r| {
            for (0..cols) |c| {
                const v = synthF32(seed + r * 31 + c * 17, 0.035);
                u16_s[r * cols + c] = quant.f32ToFloatBits(fdt, v);
            }
        }
        return;
    }

    const qf_dt = quant_formats.DType.fromAmf(dtype) orelse return error.InvalidArgument;
    if (qf_dt.isAffine()) {
        const eff_gs: usize = qf_dt.effectiveGroupSize(group_size);
        const bits = qf_dt.affineBits().?;
        const q_mask: u8 = @intCast((@as(u16, 1) << @intCast(bits)) - 1);
        const grp_bytes = (eff_gs * bits) / 8;
        const grps = cols / eff_gs;
        const rb = grps * grp_bytes;
        var codes: [128]u8 = undefined;
        const sb: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, params[0 .. rows * grps * @sizeOf(quant.ScaleBias)]));
        for (0..rows) |r| {
            for (0..grps) |g| {
                for (0..eff_gs) |j| {
                    codes[j] = @intCast((seed + r * 13 + g * 7 + j * 5 + 3) & q_mask);
                }
                packTestAffineGroup(bits, eff_gs, codes[0..eff_gs], qweight[r * rb + g * grp_bytes ..][0..grp_bytes]);
                const s: f32 = 0.025 + 0.008 * @as(f32, @floatFromInt((r + g) % 6));
                const b: f32 = -0.15 - 0.04 * @as(f32, @floatFromInt((r * 3 + g) % 5));
                sb[r * grps + g] = .{ .scale = @floatCast(s), .bias = @floatCast(b) };
            }
        }
        return;
    }

    if (qf_dt == .nvfp4_g16) {
        const grps = cols / 16;
        const rb = cols / 2;
        const valid_e4m3 = [_]u8{ 0x24, 0x28, 0x2C, 0x30, 0x34, 0x38, 0x3C, 0x40 };
        for (0..rows) |r| {
            for (0..rb) |b| {
                const lo: u8 = @intCast((seed + r * 7 + b * 3) & 0xF);
                const hi: u8 = @intCast((seed + r * 5 + b * 11 + 1) & 0xF);
                qweight[r * rb + b] = lo | (hi << 4);
            }
            for (0..grps) |g| {
                params[r * grps + g] = valid_e4m3[(seed + r + g) % valid_e4m3.len];
            }
        }
        const outer_off = try quant_formats.DType.nvfp4OuterScaleOffset(rows, cols);
        const outer_ptr: *align(1) f32 = @ptrCast(params[outer_off..].ptr);
        outer_ptr.* = 0.375;
        return;
    }

    const rb = try qf_dt.rowWeightBytes(cols, group_size);
    const bb = qf_dt.blockBytes(group_size);
    const blks = try qf_dt.groupsPerRow(cols, group_size);
    for (0..rows) |r| {
        for (0..blks) |b| {
            const blk = qweight[r * rb + b * bb ..][0..bb];
            for (blk, 0..) |*byte, i| {
                byte.* = @intCast((seed + r * 29 + b * 17 + i * 13) & 0xFF);
            }
            const d_h: f16 = @floatCast(0.03125 + 0.0078125 * @as(f32, @floatFromInt((seed + r + b) % 5)));
            const m_h: f16 = @floatCast(-0.0625 - 0.015625 * @as(f32, @floatFromInt((seed + r * 3 + b) % 4)));
            const d_u16: u16 = @bitCast(d_h);
            const m_u16: u16 = @bitCast(m_h);
            switch (qf_dt) {
                .gguf_q4_0, .gguf_q5_0, .gguf_q8_0, .gguf_iq4_nl, .gguf_q2_0, .gguf_iq1_s, .gguf_iq2_xxs, .gguf_iq2_xs, .gguf_iq2_s, .gguf_iq3_xxs, .gguf_iq3_s, .gguf_iq4_xs => {
                    std.mem.writeInt(u16, blk[0..2], d_u16, .little);
                },
                .gguf_q4_1, .gguf_q5_1, .gguf_q4_k, .gguf_q5_k => {
                    std.mem.writeInt(u16, blk[0..2], d_u16, .little);
                    std.mem.writeInt(u16, blk[2..4], m_u16, .little);
                },
                .gguf_q2_k => {
                    std.mem.writeInt(u16, blk[80..82], d_u16, .little);
                    std.mem.writeInt(u16, blk[82..84], m_u16, .little);
                },
                .gguf_q3_k => std.mem.writeInt(u16, blk[108..110], d_u16, .little),
                .gguf_q6_k => std.mem.writeInt(u16, blk[208..210], d_u16, .little),
                else => return error.InvalidArgument,
            }
        }
    }
}

fn weightBytesFor(dtype: amf.DType, group_size: u16, rows: usize, cols: usize) !usize {
    if (dtype == .f16 or dtype == .bf16) return rows * cols * @sizeOf(u16);
    const qf_dt = quant_formats.DType.fromAmf(dtype) orelse return error.InvalidArgument;
    return qf_dt.packedWeightBytes(rows, cols, group_size);
}

fn paramsBytesFor(dtype: amf.DType, group_size: u16, rows: usize, cols: usize) !usize {
    if (dtype == .f16 or dtype == .bf16) return 0;
    const qf_dt = quant_formats.DType.fromAmf(dtype) orelse return error.InvalidArgument;
    return qf_dt.paramsBytes(rows, cols, group_size);
}

const SynthExpertAlloc = struct {
    cpu_desc: MoeExpertCpuDesc,
    ew: moe.ExpertWeights,
};

fn allocSynthMoeExpert(
    arena: *arena_mod.WiredArena,
    base_off: u64,
    desc: moe.MoeSchemeDesc,
    pair_interleaved: bool,
    hidden_size: u32,
    intermediate_size: u32,
    seed: usize,
) !SynthExpertAlloc {
    const h: usize = hidden_size;
    const inter: usize = intermediate_size;
    const gu_rows = if (pair_interleaved) inter * 2 else inter;

    const g_qw_b = try weightBytesFor(desc.gate_up_dtype, desc.gate_up_group_size, gu_rows, h);
    const g_pr_b = try paramsBytesFor(desc.gate_up_dtype, desc.gate_up_group_size, gu_rows, h);
    const g_qw = try arena.allocAligned(g_qw_b, 64);
    const g_pr: []u8 = if (g_pr_b > 0) try arena.allocAligned(g_pr_b, 64) else &.{};
    try fillSynthMoeMatrix(desc.gate_up_dtype, desc.gate_up_group_size, gu_rows, h, g_qw, g_pr, seed);

    var u_qw: []u8 = &.{};
    var u_pr: []u8 = &.{};
    if (!pair_interleaved) {
        const u_qw_b = try weightBytesFor(desc.gate_up_dtype, desc.gate_up_group_size, inter, h);
        const u_pr_b = try paramsBytesFor(desc.gate_up_dtype, desc.gate_up_group_size, inter, h);
        u_qw = try arena.allocAligned(u_qw_b, 64);
        u_pr = if (u_pr_b > 0) try arena.allocAligned(u_pr_b, 64) else &.{};
        try fillSynthMoeMatrix(desc.gate_up_dtype, desc.gate_up_group_size, inter, h, u_qw, u_pr, seed + 1009);
    }

    const d_qw_b = try weightBytesFor(desc.down_dtype, desc.down_group_size, h, inter);
    const d_pr_b = try paramsBytesFor(desc.down_dtype, desc.down_group_size, h, inter);
    const d_qw = try arena.allocAligned(d_qw_b, 64);
    const d_pr: []u8 = if (d_pr_b > 0) try arena.allocAligned(d_pr_b, 64) else &.{};
    try fillSynthMoeMatrix(desc.down_dtype, desc.down_group_size, h, inter, d_qw, d_pr, seed + 2029);

    return .{
        .cpu_desc = .{
            .gate_up_dtype = desc.gate_up_dtype,
            .gate_up_group_size = desc.gate_up_group_size,
            .down_dtype = desc.down_dtype,
            .down_group_size = desc.down_group_size,
            .pair_interleaved = pair_interleaved,
            .gate_qw = g_qw,
            .gate_params = g_pr,
            .up_qw = u_qw,
            .up_params = u_pr,
            .down_qw = d_qw,
            .down_params = d_pr,
        },
        .ew = .{
            .gate_qweight_offset = arena.offsetOf(g_qw.ptr) - base_off,
            .gate_sb_offset = if (g_pr.len > 0) arena.offsetOf(g_pr.ptr) - base_off else 0,
            .up_qweight_offset = if (u_qw.len > 0) arena.offsetOf(u_qw.ptr) - base_off else 0,
            .up_sb_offset = if (u_pr.len > 0) arena.offsetOf(u_pr.ptr) - base_off else 0,
            .down_qweight_offset = arena.offsetOf(d_qw.ptr) - base_off,
            .down_sb_offset = if (d_pr.len > 0) arena.offsetOf(d_pr.ptr) - base_off else 0,
        },
    };
}

test "gpu.inkling2_hy4. G29 attn post-proj ScaledL2Clip LogScale V-RMSNorm sink parity (M=1, M=8)" {
    const gpa = std.testing.allocator;
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();

    const m_cases = [_]u32{ 1, 8 };
    const nq: u32 = 18;
    const nkv: u32 = 2;
    const hd: u32 = 128;
    const d_rel: u32 = 16;

    for (m_cases) |m_val| {
        const q_elems = @as(usize, m_val) * nq * hd;
        const kv_elems = @as(usize, m_val) * nkv * hd;
        const r_elems = @as(usize, m_val) * nq * d_rel;
        const s_elems = @as(usize, m_val) * nq;

        var arena = try makeTestArena(2 * 1024 * 1024);
        defer arena.deinit();

        const q_fused_raw = try arena.allocAligned(q_elems * @sizeOf(f16), 64);
        const k_fused_raw = try arena.allocAligned(kv_elems * @sizeOf(f16), 64);
        const r_fused_raw = try arena.allocAligned(r_elems * @sizeOf(f16), 64);
        const v_fused_raw = try arena.allocAligned(kv_elems * @sizeOf(f16), 64);
        const q_solo_raw = try arena.allocAligned(q_elems * @sizeOf(f16), 64);
        const v_solo_raw = try arena.allocAligned(kv_elems * @sizeOf(f16), 64);
        const q_sv_raw = try arena.allocAligned(@as(usize, nq) * hd * @sizeOf(f32), 64);
        const r_sv_raw = try arena.allocAligned(@as(usize, nq) * d_rel * @sizeOf(f32), 64);
        const pos_raw = try arena.allocAligned(@as(usize, m_val) * @sizeOf(u32), 64);
        const sink16_raw = try arena.allocAligned(s_elems * @sizeOf(f16), 64);
        const sink32_fused_raw = try arena.allocAligned(s_elems * @sizeOf(f32), 64);
        const sink32_solo_raw = try arena.allocAligned(s_elems * @sizeOf(f32), 64);

        const q_fused: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_fused_raw));
        const k_fused: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_fused_raw));
        const r_fused: []f16 = @alignCast(std.mem.bytesAsSlice(f16, r_fused_raw));
        const v_fused: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_fused_raw));
        const q_solo: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_solo_raw));
        const v_solo: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_solo_raw));
        const q_sv: []f32 = @alignCast(std.mem.bytesAsSlice(f32, q_sv_raw));
        const r_sv: []f32 = @alignCast(std.mem.bytesAsSlice(f32, r_sv_raw));
        const pos_s: []u32 = @alignCast(std.mem.bytesAsSlice(u32, pos_raw));
        const sink16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, sink16_raw));
        const sink32_fused: []f32 = @alignCast(std.mem.bytesAsSlice(f32, sink32_fused_raw));
        const sink32_solo: []f32 = @alignCast(std.mem.bytesAsSlice(f32, sink32_solo_raw));

        fillSynthF16(q_fused, 101 + m_val, 0.28);
        fillSynthF16(k_fused, 203 + m_val, 0.28);
        fillSynthF16(r_fused, 307 + m_val, 0.32);
        fillSynthF16(v_fused, 409 + m_val, 0.12);
        fillSynthF16(sink16, 503 + m_val, 0.15);
        @memcpy(q_solo, q_fused);
        @memcpy(v_solo, v_fused);
        for (q_sv, 0..) |*s, i| s.* = 0.8 + @abs(synthF32(601 + i, 0.03));
        for (r_sv, 0..) |*s, i| s.* = 0.9 + @abs(synthF32(701 + i, 0.02));
        for (pos_s, 0..) |*p, t| p.* = 128 * (@as(u32, @intCast(t)) + 2) + 17;

        const ref_q = try gpa.alloc(f32, q_elems);
        defer gpa.free(ref_q);
        const ref_k = try gpa.alloc(f32, kv_elems);
        defer gpa.free(ref_k);
        const ref_r = try gpa.alloc(f32, r_elems);
        defer gpa.free(ref_r);
        const ref_v = try gpa.alloc(f32, kv_elems);
        defer gpa.free(ref_v);

        inkling2ScaledL2ClipPerHeadCpuRef(q_fused, q_sv, pos_s, m_val, nq, hd, 22.627, true, 0, 256, 0.125, ref_q);
        inkling2ScaledL2ClipPerHeadCpuRef(k_fused, null, null, m_val, nkv, hd, 22.627, false, 0, 0, 0.0, ref_k);
        inkling2ScaledL2ClipPerHeadCpuRef(r_fused, r_sv, pos_s, m_val, nq, d_rel, 8.0, true, 0, 256, 0.125, ref_r);
        inkling2UnitRmsNormPerHeadCpuRef(v_fused, m_val, nkv, hd, 1e-6, ref_v);

        const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(ctx.device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try ctx.pipes.encodeInkling2AttnPostProj(&f, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_fused_raw.ptr) },
            .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_fused_raw.ptr) },
            .q_rel = .{ .buffer = gpu_buf, .offset = arena.offsetOf(r_fused_raw.ptr) },
            .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_fused_raw.ptr) },
            .q_scale_vec = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_sv_raw.ptr) },
            .rel_scale_vec = .{ .buffer = gpu_buf, .offset = arena.offsetOf(r_sv_raw.ptr) },
            .positions = .{ .buffer = gpu_buf, .offset = arena.offsetOf(pos_raw.ptr) },
            .sink_f16 = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sink16_raw.ptr) },
            .sink_f32 = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sink32_fused_raw.ptr) },
            .m = m_val,
            .num_q_heads = nq,
            .num_kv_heads = nkv,
            .head_dim = hd,
            .d_rel = d_rel,
            .q_clip = 22.627,
            .k_clip = 22.627,
            .rel_clip = 8.0,
            .use_log_scaling = true,
            .log_floor = 256,
            .log_alpha = 0.125,
            .do_v_norm = true,
            .rms_eps = 1e-6,
        }, false);
        try ctx.pipes.encodeInkling2ScaledL2ClipPerHead(&f, .{
            .buf = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_solo_raw.ptr) },
            .scale_vec = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_sv_raw.ptr) },
            .positions = .{ .buffer = gpu_buf, .offset = arena.offsetOf(pos_raw.ptr) },
            .m = m_val,
            .num_heads = nq,
            .head_dim = hd,
            .threshold = 22.627,
            .use_log_scaling = true,
            .log_floor = 256,
            .log_alpha = 0.125,
        }, false);
        try ctx.pipes.encodeInkling2UnitRmsNormPerHead(&f, .{
            .buf = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_solo_raw.ptr) },
            .m = m_val,
            .num_heads = nkv,
            .head_dim = hd,
            .rms_eps = 1e-6,
        }, false);
        try ctx.pipes.encodeInkling2F16ToF32(&f, .{
            .src = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sink16_raw.ptr) },
            .dst = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sink32_solo_raw.ptr) },
            .count = @intCast(s_elems),
        }, false);
        try compute.wait(try f.commit(), 5000);

        try expectF16MatchesF32(ref_q, q_fused, 0.01, 0.01);
        try expectF16MatchesF32(ref_q, q_solo, 0.01, 0.01);
        try expectF16MatchesF32(ref_k, k_fused, 0.01, 0.01);
        try expectF16MatchesF32(ref_r, r_fused, 0.01, 0.01);
        try expectF16MatchesF32(ref_v, v_fused, 0.005, 0.005);
        try expectF16MatchesF32(ref_v, v_solo, 0.005, 0.005);
        for (sink16, 0..) |sh, idx| {
            const sv: f32 = @floatCast(sh);
            try std.testing.expectEqual(sv, sink32_fused[idx]);
            try std.testing.expectEqual(sv, sink32_solo[idx]);
        }
    }
}

test "gpu.inkling2_hy4. G29 dyn_sconv lag-reversed assembly + sconv4 parity (M=1, M=8)" {
    const gpa = std.testing.allocator;
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();

    const Case = struct { m: u32, proj_n: u32, static_w_f16: ?bool };
    const cases = [_]Case{
        .{ .m = 1, .proj_n = 256 * 4, .static_w_f16 = null },
        .{ .m = 8, .proj_n = 256 * 4, .static_w_f16 = false },
        .{ .m = 8, .proj_n = 2 * 4, .static_w_f16 = true },
        .{ .m = 1, .proj_n = 4, .static_w_f16 = false },
    };

    const kv_dim: u32 = 256;
    const nkv: u32 = 2;
    const hd: u32 = 128;

    for (cases) |c| {
        var arena = try makeTestArena(2 * 1024 * 1024);
        defer arena.deinit();

        const sring_spec: SconvRingSpec = .{ .num_layers = 1, .hidden_dim = kv_dim };
        var sring = try SconvRingArena.init(&arena, sring_spec);
        const x_raw = try arena.allocAligned(@as(usize, c.m) * kv_dim * @sizeOf(f16), 64);
        const proj_raw = try arena.allocAligned(@as(usize, c.m) * c.proj_n * @sizeOf(f16), 64);
        const dyn_w_raw = try arena.allocAligned(@as(usize, c.m) * kv_dim * 4 * @sizeOf(f16), 64);
        const static_f32_raw = try arena.allocAligned(@as(usize, kv_dim) * 4 * @sizeOf(f32), 64);
        const static_f16_raw = try arena.allocAligned(@as(usize, kv_dim) * 4 * @sizeOf(f16), 64);
        const out_raw = try arena.allocAligned(@as(usize, c.m) * kv_dim * @sizeOf(f16), 64);

        const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
        const proj_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, proj_raw));
        const dyn_w_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, dyn_w_raw));
        const static_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, static_f32_raw));
        const static_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, static_f16_raw));
        const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));

        fillSynthF16(x_f16, 17 + c.m, 0.08);
        fillSynthF16(proj_f16, 43 + c.proj_n, 0.05);
        for (static_f32, 0..) |*sv, idx| {
            const v = synthF32(97 + idx, 0.04);
            sv.* = v;
            static_f16[idx] = @floatCast(v);
        }
        const ring_slice = try sring.layerSlice(0);
        fillSynthF16(ring_slice, 131, 0.06);

        const state_cpu = try gpa.alloc(f16, ring_slice.len);
        defer gpa.free(state_cpu);
        @memcpy(state_cpu, ring_slice);

        const ref_dyn_w = try gpa.alloc(f32, @as(usize, c.m) * kv_dim * 4);
        defer gpa.free(ref_dyn_w);
        const sw16_opt: ?[]const f16 = if (c.static_w_f16 == true) static_f16 else null;
        const sw32_opt: ?[]const f32 = if (c.static_w_f16 == false) static_f32 else null;
        inkling2DynSconvAssembleCpuRef(proj_f16, sw16_opt, sw32_opt, c.m, kv_dim, nkv, hd, c.proj_n, 1.0, ref_dyn_w);

        const ref_dyn_w_f16 = try gpa.alloc(f16, ref_dyn_w.len);
        defer gpa.free(ref_dyn_w_f16);
        for (ref_dyn_w, 0..) |v, idx| ref_dyn_w_f16[idx] = @floatCast(v);

        const ref_sconv_out = try gpa.alloc(f32, @as(usize, c.m) * kv_dim);
        defer gpa.free(ref_sconv_out);
        try rmsnormSconv4CpuRef(
            gpa,
            x_f16,
            null,
            .f16,
            null,
            null,
            ref_dyn_w_f16,
            null,
            state_cpu,
            ref_sconv_out,
            c.m,
            kv_dim,
            0,
            true,
            false,
            false,
            1e-6,
        );
        for (ref_sconv_out, 0..) |*rv, idx| {
            rv.* = (rv.* + @as(f32, @floatCast(x_f16[idx]))) * (1.0 / 3.0);
        }

        const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(ctx.device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        const static_bind: ?metal.BufferBinding = if (c.static_w_f16) |is_f16|
            .{ .buffer = gpu_buf, .offset = arena.offsetOf(if (is_f16) static_f16_raw.ptr else static_f32_raw.ptr) }
        else
            null;
        const state_b = try sring.layerBinding(0, gpu_buf);

        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try ctx.pipes.encodeInkling2DynSconvAssemble(&f, .{
            .proj = .{ .buffer = gpu_buf, .offset = arena.offsetOf(proj_raw.ptr) },
            .static_w = static_bind,
            .static_w_f16 = c.static_w_f16 orelse false,
            .dyn_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(dyn_w_raw.ptr) },
            .m = c.m,
            .kv_dim = kv_dim,
            .num_kv_heads = nkv,
            .head_dim = hd,
            .proj_n = c.proj_n,
        }, false);
        try ctx.pipes.encodeRmsNormSconv4(&f, .{
            .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
            .dyn_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(dyn_w_raw.ptr) },
            .conv_state_in = state_b,
            .conv_state_out = state_b,
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .m = c.m,
            .hidden_dim = kv_dim,
            .state_pos = 0,
            .write_state = true,
            .is_dynamic = true,
            .apply_rms_norm = false,
            .apply_silu = false,
            .post_residual_scale = 1.0 / 3.0,
        }, true);
        try compute.wait(try f.commit(), 5000);

        try expectF16MatchesF32(ref_dyn_w, dyn_w_f16, 0.004, 0.005);
        try expectF16MatchesF32(ref_sconv_out, out_f16, 0.006, 0.008);
    }
}

test "gpu.inkling2_hy4. G29 sme2_panel64 deinterleave + cross-pass msg bank + stepMessageGate parity (M=1, M=8)" {
    const gpa = std.testing.allocator;
    const ink_ctrl = @import("../cpu/inkling2_ctrl.zig");
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();

    // 1. Exact roundtrip of deinterleavePanel64ToRowMajor across all 5 controller PanelDTypes.
    {
        const cases = [_]struct { dt: sme.PanelDType, n: usize, k: usize, out_dt: quant.FloatDType }{
            .{ .dt = .f16, .n = 21, .k = 128, .out_dt = .f16 },
            .{ .dt = .bf16, .n = 64, .k = 128, .out_dt = .bf16 },
            .{ .dt = .q4_affine, .n = 21, .k = 128, .out_dt = .f16 },
            .{ .dt = .q3_affine, .n = 33, .k = 128, .out_dt = .bf16 },
            .{ .dt = .q2_affine, .n = 19, .k = 128, .out_dt = .f16 },
        };
        for (cases) |tc| {
            const nk = tc.n * tc.k;
            const blob_bytes = try sme.packedBlobBytes(tc.dt, tc.n, tc.k);
            const blob = try gpa.alloc(u8, blob_bytes);
            defer gpa.free(blob);
            const dst_u16 = try gpa.alloc(u16, nk);
            defer gpa.free(dst_u16);
            const exp_f32 = try gpa.alloc(f32, nk);
            defer gpa.free(exp_f32);

            if (!tc.dt.isQuantized()) {
                const src_u16 = try gpa.alloc(u16, nk);
                defer gpa.free(src_u16);
                const fdt: quant.FloatDType = if (tc.dt == .bf16) .bf16 else .f16;
                for (src_u16, 0..) |*w, i| {
                    const v = synthF32(701 + i, 0.04);
                    w.* = quant.f32ToFloatBits(fdt, v);
                    exp_f32[i] = quant.floatBitsToF32(fdt, w.*);
                }
                try sme.packDensePanel64(tc.dt, src_u16, tc.n, tc.k, blob);
            } else {
                const bits: u8 = switch (tc.dt) {
                    .q4_affine => 4,
                    .q3_affine => 3,
                    .q2_affine => 2,
                    else => unreachable,
                };
                const q_mask: u8 = @intCast((@as(u16, 1) << @intCast(bits)) - 1);
                const grps = tc.k / sme.GROUP_SIZE;
                const codes = try gpa.alloc(u8, nk);
                defer gpa.free(codes);
                const sbs = try gpa.alloc(quant.ScaleBias, tc.n * grps);
                defer gpa.free(sbs);
                for (0..tc.n) |r| {
                    for (0..grps) |g| {
                        const sb_val: quant.ScaleBias = .{
                            .scale = @floatCast(0.03 + 0.01 * @as(f32, @floatFromInt((r + g) % 5))),
                            .bias = @floatCast(-0.15 + 0.03 * @as(f32, @floatFromInt((r + g * 2) % 6))),
                        };
                        sbs[r * grps + g] = sb_val;
                        const s_f32: f32 = @floatCast(sb_val.scale);
                        const b_f32: f32 = @floatCast(sb_val.bias);
                        for (0..sme.GROUP_SIZE) |j| {
                            const code: u8 = @intCast((r * 7 + g * 13 + j * 5 + 1) & q_mask);
                            codes[r * tc.k + g * sme.GROUP_SIZE + j] = code;
                            exp_f32[r * tc.k + g * sme.GROUP_SIZE + j] = @as(f32, @floatFromInt(code)) * s_f32 + b_f32;
                        }
                    }
                }
                try sme.packQuantAffinePanel64(tc.dt, codes, sbs, tc.n, tc.k, blob);
            }
            try deinterleavePanel64ToRowMajor(tc.dt, blob, tc.n, tc.k, tc.out_dt, dst_u16);
            for (dst_u16, 0..) |act_u16, i| {
                const act_f32 = quant.floatBitsToF32(tc.out_dt, act_u16);
                try std.testing.expectApproxEqAbs(exp_f32[i], act_f32, 0.02);
            }
        }
    }

    // 2. Full cross-pass msg_bank_norm -> cross_pass_msgs -> stepMessageGate GPU vs live Inkling2LoopController parity!
    const m_cases = [_]u32{ 1, 8 };
    const hidden_dim: u32 = 256;
    const gate_rank: u32 = ink_ctrl.GATE_RANK;
    const bank_blocks: u32 = 4;

    var disp = try sme.SmeDispatcher.init(gpa, .{});
    defer disp.deinit();

    const budget_over = budget_mod.Budget.compute(.{
        .memory_bytes = 64 * 1024 * 1024,
        .user_wire_limit = 32 * 1024 * 1024,
        .global_user_wire_limit = 32 * 1024 * 1024,
        .gpu_working_set_bytes = 32 * 1024 * 1024,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });

    const dn_in_dim = hidden_dim * 2;
    const dn_blob_b = try sme.packedBlobBytes(.bf16, gate_rank, dn_in_dim);
    const fg_blob_b = try sme.packedBlobBytes(.bf16, hidden_dim, gate_rank);
    const dn_src = try gpa.alloc(u16, gate_rank * dn_in_dim);
    defer gpa.free(dn_src);
    const fg_src = try gpa.alloc(u16, hidden_dim * gate_rank);
    defer gpa.free(fg_src);
    const gn_src = try gpa.alloc(u16, hidden_dim * gate_rank);
    defer gpa.free(gn_src);
    const dn_blob = try gpa.alloc(u8, dn_blob_b);
    defer gpa.free(dn_blob);
    const fg_blob = try gpa.alloc(u8, fg_blob_b);
    defer gpa.free(fg_blob);
    const gn_blob = try gpa.alloc(u8, fg_blob_b);
    defer gpa.free(gn_blob);

    for (dn_src, 0..) |*v, i| v.* = quant.f32ToBf16(synthF32(11 + i, 0.01));
    for (fg_src, 0..) |*v, i| v.* = quant.f32ToBf16(synthF32(313 + i, 0.01));
    for (gn_src, 0..) |*v, i| v.* = quant.f32ToBf16(synthF32(617 + i, 0.01));
    try sme.packDensePanel64(.bf16, dn_src, gate_rank, dn_in_dim, dn_blob);
    try sme.packDensePanel64(.bf16, fg_src, hidden_dim, gate_rank, fg_blob);
    try sme.packDensePanel64(.bf16, gn_src, hidden_dim, gate_rank, gn_blob);

    const dn16_src = try gpa.alloc(u16, gate_rank * dn_in_dim);
    defer gpa.free(dn16_src);
    const fg16_src = try gpa.alloc(u16, hidden_dim * gate_rank);
    defer gpa.free(fg16_src);
    const gn16_src = try gpa.alloc(u16, hidden_dim * gate_rank);
    defer gpa.free(gn16_src);
    const dn16_blob = try gpa.alloc(u8, dn_blob_b);
    defer gpa.free(dn16_blob);
    const fg16_blob = try gpa.alloc(u8, fg_blob_b);
    defer gpa.free(fg16_blob);
    const gn16_blob = try gpa.alloc(u8, fg_blob_b);
    defer gpa.free(gn16_blob);
    for (dn16_src, 0..) |*v, i| v.* = quant.f32ToFloatBits(.f16, synthF32(11 + i, 0.01));
    for (fg16_src, 0..) |*v, i| v.* = quant.f32ToFloatBits(.f16, synthF32(313 + i, 0.01));
    for (gn16_src, 0..) |*v, i| v.* = quant.f32ToFloatBits(.f16, synthF32(617 + i, 0.01));
    try sme.packDensePanel64(.f16, dn16_src, gate_rank, dn_in_dim, dn16_blob);
    try sme.packDensePanel64(.f16, fg16_src, hidden_dim, gate_rank, fg16_blob);
    try sme.packDensePanel64(.f16, gn16_src, hidden_dim, gate_rank, gn16_blob);

    var ctrl_cfg = ink_ctrl.ControllerConfig.uniform(dn_in_dim, hidden_dim, .bf16, .tanh, 8);
    ctrl_cfg.budget_override = budget_over;
    const b_in: ink_ctrl.BranchWeightInput = .{
        .down_weight = .{ .sme2_panel64 = dn_blob },
        .forget_weight = .{ .sme2_panel64 = fg_blob },
        .gain_weight = .{ .sme2_panel64 = gn_blob },
    };
    var ctrl = try ink_ctrl.Inkling2LoopController.init(ctrl_cfg, .{ b_in, b_in, b_in, b_in });
    defer ctrl.deinit();

    var ctrl16_cfg = ink_ctrl.ControllerConfig.uniform(dn_in_dim, hidden_dim, .f16, .tanh, 8);
    ctrl16_cfg.budget_override = budget_over;
    const b16_in: ink_ctrl.BranchWeightInput = .{
        .down_weight = .{ .sme2_panel64 = dn16_blob },
        .forget_weight = .{ .sme2_panel64 = fg16_blob },
        .gain_weight = .{ .sme2_panel64 = gn16_blob },
    };
    var ctrl16 = try ink_ctrl.Inkling2LoopController.init(ctrl16_cfg, .{ b16_in, b16_in, b16_in, b16_in });
    defer ctrl16.deinit();

    for (m_cases) |m_val| {
        const mh: usize = @as(usize, m_val) * hidden_dim;
        var arena = try makeTestArena(4 * 1024 * 1024);
        defer arena.deinit();

        var ring = try AttnResRingArena.init(&arena, .{
            .max_blocks = bank_blocks,
            .max_tokens = m_val,
            .hidden_dim = hidden_dim,
        });
        for (0..3) |bs| {
            _ = ring.pushSlot();
            const sl = try ring.blockSlice(@intCast(bs));
            fillSynthF16(sl, 100 + bs * 41, 0.08);
        }

        const cur_raw = try arena.allocAligned(mh * @sizeOf(f16), 64);
        const bank_raw = try arena.allocAligned(@as(usize, bank_blocks) * mh * @sizeOf(f32), 64);
        const q_src_raw = try arena.allocAligned(4 * hidden_dim * @sizeOf(f32), 64);
        const kv_q_src_raw = try arena.allocAligned(4 * hidden_dim * @sizeOf(f32), 64);
        const msgs_raw = try arena.allocAligned(4 * mh * @sizeOf(f32), 64);
        const z_raw = try arena.allocAligned(@as(usize, m_val) * gate_rank * @sizeOf(f32), 64);
        const gate_out_raw = try arena.allocAligned(mh * @sizeOf(f16), 64);
        const gate16_out_raw = try arena.allocAligned(mh * @sizeOf(f16), 64);
        const silu_out_raw = try arena.allocAligned(mh * @sizeOf(f16), 64);
        const norm_w_raw = try arena.allocAligned(dn_in_dim * @sizeOf(f32), 64);
        const f_bias_raw = try arena.allocAligned(hidden_dim * @sizeOf(f32), 64);
        const g_bias_raw = try arena.allocAligned(hidden_dim * @sizeOf(f32), 64);

        const dn_de = try deinterleavePanel64ToArena(&arena, .bf16, dn_blob, gate_rank, dn_in_dim, .bf16);
        const fg_de = try deinterleavePanel64ToArena(&arena, .bf16, fg_blob, hidden_dim, gate_rank, .bf16);
        const gn_de = try deinterleavePanel64ToArena(&arena, .bf16, gn_blob, hidden_dim, gate_rank, .bf16);
        const dn_de16 = try deinterleavePanel64ToArena(&arena, .f16, dn16_blob, gate_rank, dn_in_dim, .f16);
        const fg_de16 = try deinterleavePanel64ToArena(&arena, .f16, fg16_blob, hidden_dim, gate_rank, .f16);
        const gn_de16 = try deinterleavePanel64ToArena(&arena, .f16, gn16_blob, hidden_dim, gate_rank, .f16);
        const dn_f16 = try deinterleavePanel64ToArena(&arena, .bf16, dn_blob, gate_rank, dn_in_dim, .f16);
        const fg_f16 = try deinterleavePanel64ToArena(&arena, .bf16, fg_blob, hidden_dim, gate_rank, .f16);
        const gn_f16 = try deinterleavePanel64ToArena(&arena, .bf16, gn_blob, hidden_dim, gate_rank, .f16);

        const cur_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, cur_raw));
        const bank_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bank_raw));
        const q_src_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, q_src_raw));
        const kv_q_src_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, kv_q_src_raw));
        const msgs_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, msgs_raw));
        const gate_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate_out_raw));
        const gate16_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate16_out_raw));
        const silu_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, silu_out_raw));
        const norm_w_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, norm_w_raw));
        const f_bias_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, f_bias_raw));
        const g_bias_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, g_bias_raw));

        fillSynthF16(cur_f16, 401 + m_val, 0.08);
        for (q_src_f32, 0..) |*v, i| v.* = synthF32(503 + i, 0.09);
        for (kv_q_src_f32, 0..) |*v, i| v.* = synthF32(709 + i, 0.09);
        for (norm_w_f32, 0..) |*v, i| v.* = 0.85 + @abs(synthF32(809 + i, 0.02));
        for (f_bias_f32, 0..) |*v, i| v.* = synthF32(907 + i, 0.02);
        for (g_bias_f32, 0..) |*v, i| v.* = synthF32(953 + i, 0.02);

        // CPU references (bank -> cross-pass -> both MessageGateCpuRef & live Inkling2LoopController).
        const ring_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, ring.bytes()));
        const ref_bank = try gpa.alloc(f32, bank_blocks * mh);
        defer gpa.free(ref_bank);
        const ref_msgs = try gpa.alloc(f32, 4 * mh);
        defer gpa.free(ref_msgs);
        const ref_ctrl = try gpa.alloc(f32, mh);
        defer gpa.free(ref_ctrl);
        const ref_ctrl16 = try gpa.alloc(f32, mh);
        defer gpa.free(ref_ctrl16);
        const ref_gate_cpu = try gpa.alloc(f32, mh);
        defer gpa.free(ref_gate_cpu);
        const ref_gate16_cpu = try gpa.alloc(f32, mh);
        defer gpa.free(ref_gate16_cpu);
        const ref_silu_cpu = try gpa.alloc(f32, mh);
        defer gpa.free(ref_silu_cpu);
        const cur_cpu_f32 = try gpa.alloc(f32, mh);
        defer gpa.free(cur_cpu_f32);
        for (cur_f16, 0..) |cv, i| cur_cpu_f32[i] = @floatCast(cv);

        const bank_len = inkling2MsgBankNormCpuRef(ring_f16, &[_]u32{ 0, 1, 2 }, ring.block_stride_elems, cur_f16, m_val, hidden_dim, 1e-6, ref_bank);
        try std.testing.expectEqual(@as(usize, 4), bank_len);
        inkling2CrossPassMsgsCpuRef(ref_bank, bank_len, q_src_f32, kv_q_src_f32, 1, m_val, hidden_dim, 1.0 / @sqrt(@as(f32, @floatFromInt(hidden_dim))), ref_msgs);
        try ctrl.stepMessageGate(&disp, .attn, cur_cpu_f32, ref_msgs[0..mh], ref_ctrl, m_val, null);
        try ctrl16.stepMessageGate(&disp, .attn, cur_cpu_f32, ref_msgs[0..mh], ref_ctrl16, m_val, null);
        try inkling2MessageGateCpuRef(gpa, cur_f16, ref_msgs[0..mh], dn_src, fg_src, gn_src, .bf16, null, null, null, m_val, hidden_dim, dn_in_dim, gate_rank, .tanh, 1e-6, ref_gate_cpu);
        try inkling2MessageGateCpuRef(gpa, cur_f16, ref_msgs[0..mh], dn16_src, fg16_src, gn16_src, .f16, null, null, null, m_val, hidden_dim, dn_in_dim, gate_rank, .tanh, 1e-6, ref_gate16_cpu);
        try inkling2MessageGateCpuRef(gpa, cur_f16, ref_msgs[0..mh], dn_f16.u16_slice, fg_f16.u16_slice, gn_f16.u16_slice, .f16, norm_w_f32, f_bias_f32, g_bias_f32, m_val, hidden_dim, dn_in_dim, gate_rank, .silu, 1e-6, ref_silu_cpu);

        const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(ctx.device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        const gpu_bank_len = try ctx.pipes.encodeInkling2MsgBankNorm(&f, .{
            .ring = &ring,
            .ring_buf = gpu_buf,
            .cur = .{ .buffer = gpu_buf, .offset = arena.offsetOf(cur_raw.ptr) },
            .msg_bank_out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bank_raw.ptr) },
            .m = m_val,
            .rms_eps = 1e-6,
        }, false);
        try std.testing.expectEqual(@as(u32, 4), gpu_bank_len);
        try ctx.pipes.encodeInkling2CrossPassMsgs(&f, .{
            .msg_bank = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bank_raw.ptr) },
            .q_src = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_src_raw.ptr) },
            .kv_q_src = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kv_q_src_raw.ptr) },
            .out_msgs = .{ .buffer = gpu_buf, .offset = arena.offsetOf(msgs_raw.ptr) },
            .m = m_val,
            .hidden_dim = hidden_dim,
            .bank_len = gpu_bank_len,
            .q_rows = 4,
            .kv_rows = 4,
            .loop_layer_idx = 1,
        }, true);
        try ctx.pipes.encodeInkling2MessageGate(&f, .{
            .cur = .{ .buffer = gpu_buf, .offset = arena.offsetOf(cur_raw.ptr) },
            .msg = .{ .buffer = gpu_buf, .offset = arena.offsetOf(msgs_raw.ptr) },
            .down_weight = dn_de.binding(gpu_buf),
            .forget_weight = fg_de.binding(gpu_buf),
            .gain_weight = gn_de.binding(gpu_buf),
            .weight_dtype = .bf16,
            .z_scratch = .{ .buffer = gpu_buf, .offset = arena.offsetOf(z_raw.ptr) },
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gate_out_raw.ptr) },
            .m = m_val,
            .state_dim = hidden_dim,
            .down_in_dim = dn_in_dim,
            .gate_rank = gate_rank,
            .gain_act = .tanh,
            .rms_eps = 1e-6,
        }, true);
        try ctx.pipes.encodeInkling2MessageGate(&f, .{
            .cur = .{ .buffer = gpu_buf, .offset = arena.offsetOf(cur_raw.ptr) },
            .msg = .{ .buffer = gpu_buf, .offset = arena.offsetOf(msgs_raw.ptr) },
            .down_weight = dn_de16.binding(gpu_buf),
            .forget_weight = fg_de16.binding(gpu_buf),
            .gain_weight = gn_de16.binding(gpu_buf),
            .weight_dtype = .f16,
            .z_scratch = .{ .buffer = gpu_buf, .offset = arena.offsetOf(z_raw.ptr) },
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gate16_out_raw.ptr) },
            .m = m_val,
            .state_dim = hidden_dim,
            .down_in_dim = dn_in_dim,
            .gate_rank = gate_rank,
            .gain_act = .tanh,
            .rms_eps = 1e-6,
        }, true);
        try ctx.pipes.encodeInkling2MessageGate(&f, .{
            .cur = .{ .buffer = gpu_buf, .offset = arena.offsetOf(cur_raw.ptr) },
            .msg = .{ .buffer = gpu_buf, .offset = arena.offsetOf(msgs_raw.ptr) },
            .down_weight = dn_f16.binding(gpu_buf),
            .forget_weight = fg_f16.binding(gpu_buf),
            .gain_weight = gn_f16.binding(gpu_buf),
            .weight_dtype = .f16,
            .norm_weight = .{ .buffer = gpu_buf, .offset = arena.offsetOf(norm_w_raw.ptr) },
            .forget_bias = .{ .buffer = gpu_buf, .offset = arena.offsetOf(f_bias_raw.ptr) },
            .gain_bias = .{ .buffer = gpu_buf, .offset = arena.offsetOf(g_bias_raw.ptr) },
            .z_scratch = .{ .buffer = gpu_buf, .offset = arena.offsetOf(z_raw.ptr) },
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(silu_out_raw.ptr) },
            .m = m_val,
            .state_dim = hidden_dim,
            .down_in_dim = dn_in_dim,
            .gate_rank = gate_rank,
            .gain_act = .silu,
            .rms_eps = 1e-6,
        }, true);
        try compute.wait(try f.commit(), 5000);

        for (ref_bank, bank_f32[0..ref_bank.len]) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 0.002);
        for (ref_msgs, msgs_f32[0..ref_msgs.len]) |ev, av| try std.testing.expectApproxEqAbs(ev, av, 0.002);
        try expectF16MatchesF32(ref_gate_cpu, gate_out_f16, 0.006, 0.01);
        try expectF16MatchesF32(ref_gate16_cpu, gate16_out_f16, 0.006, 0.01);
        // `Inkling2LoopController` quantizes intermediate `x_norm` and `z_out` to bf16/f16,
        // whereas GPU kernels keep them in fp32 (`z_scratch`); verify agreement within activation-truncation budget.
        try expectF16MatchesF32(ref_ctrl, gate_out_f16, 0.06, 0.06);
        try expectF16MatchesF32(ref_ctrl16, gate16_out_f16, 0.03, 0.03);
        try expectF16MatchesF32(ref_silu_cpu, silu_out_f16, 0.006, 0.01);
    }

    // 3. Startup `Inkling-2-Mini` (~3.4 MiB mixed q4/bf16) deinterleave roundtrip.
    {
        const h_mini: usize = 2304;
        const r_mini: usize = 64;
        const down_elems = r_mini * (h_mini * 2);
        const gate_elems = h_mini * r_mini;
        const dn_b = try sme.packedBlobBytes(.bf16, r_mini, h_mini * 2);
        const fg_b = try sme.packedBlobBytes(.q4_affine, h_mini, r_mini);
        const blob_dn = try gpa.alloc(u8, dn_b);
        defer gpa.free(blob_dn);
        const blob_fg = try gpa.alloc(u8, fg_b);
        defer gpa.free(blob_fg);
        const dst_dn = try gpa.alloc(u16, down_elems);
        defer gpa.free(dst_dn);
        const dst_fg = try gpa.alloc(u16, gate_elems);
        defer gpa.free(dst_fg);

        const src_dn = try gpa.alloc(u16, down_elems);
        defer gpa.free(src_dn);
        for (src_dn, 0..) |*v, i| v.* = quant.f32ToBf16(synthF32(i, 0.01));
        try sme.packDensePanel64(.bf16, src_dn, r_mini, h_mini * 2, blob_dn);

        const fg_codes = try gpa.alloc(u8, gate_elems);
        defer gpa.free(fg_codes);
        const fg_sbs = try gpa.alloc(quant.ScaleBias, h_mini);
        defer gpa.free(fg_sbs);
        for (0..h_mini) |r| {
            fg_sbs[r] = .{ .scale = 0.05, .bias = -0.3 };
            for (0..r_mini) |c| fg_codes[r * r_mini + c] = @intCast((r + c) & 0xF);
        }
        try sme.packQuantAffinePanel64(.q4_affine, fg_codes, fg_sbs, h_mini, r_mini, blob_fg);

        for (0..4) |_| {
            try deinterleavePanel64ToRowMajor(.bf16, blob_dn, r_mini, h_mini * 2, .bf16, dst_dn);
            try deinterleavePanel64ToRowMajor(.q4_affine, blob_fg, h_mini, r_mini, .bf16, dst_fg);
        }
        try std.testing.expectEqualSlices(u16, src_dn, dst_dn);
        for (0..h_mini) |r| {
            const exp_v = @as(f32, @floatFromInt(fg_codes[r * r_mini + 7])) * 0.05 - 0.3;
            try std.testing.expectApproxEqAbs(exp_v, quant.bf16ToF32(dst_fg[r * r_mini + 7]), 0.02);
        }
    }
}

fn verifyIndirectSoftcapMoeCase(
    ctx: TestMetalCtx,
    bundle: TestMoeBundle,
    m_val: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    is_hetero: bool,
    expert_bucket_ids: []const u8,
    scheme_descs: [moe.NUM_SCHEME_BUCKETS]moe.MoeSchemeDesc,
    pair_interleaved: bool,
    use_linear_router: bool,
    with_residual: bool,
    seed: usize,
) !void {
    const gpa = std.testing.allocator;
    const h: usize = hidden_size;

    var scheme_counts: [moe.NUM_SCHEME_BUCKETS]u32 = @splat(0);
    for (expert_bucket_ids) |bid| scheme_counts[bid] += 1;

    const ws_spec: moe.WorkspaceSpec = .{
        .max_tokens = m_val,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
    };
    const ws_bytes = try moe.MoeWorkspace.requiredBytes(ws_spec);
    var arena = try makeTestArena(ws_bytes + 16 * 1024 * 1024);
    defer arena.deinit();

    const ws = try moe.MoeWorkspace.init(&arena, ws_spec);
    const x_raw = try arena.allocAligned(@as(usize, m_val) * h * @sizeOf(f16), 64);
    const res_raw = try arena.allocAligned(@as(usize, m_val) * h * @sizeOf(f16), 64);
    const out_raw = try arena.allocAligned(@as(usize, m_val) * h * @sizeOf(f16), 64);
    const logits_raw = try arena.allocAligned(@as(usize, m_val) * num_experts * @sizeOf(f16), 64);
    const router_w_raw = try arena.allocAligned(@as(usize, num_experts) * h * @sizeOf(f16), 64);
    const bias_raw = try arena.allocAligned(@as(usize, num_experts) * @sizeOf(f32), 64);
    const table_raw = try arena.allocAligned(@as(usize, num_experts) * @sizeOf(moe.ExpertWeights), 64);
    const schemes_raw = try arena.allocAligned(amf.align64(num_experts), 64);
    @memset(out_raw, 0);
    @memcpy(schemes_raw[0..num_experts], expert_bucket_ids);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const res_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, res_raw));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
    const logits_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, logits_raw));
    const router_w_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, router_w_raw));
    const bias_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bias_raw));
    const table: []moe.ExpertWeights = @alignCast(std.mem.bytesAsSlice(moe.ExpertWeights, table_raw));

    fillSynthF16(x_f16, seed + 17, 0.08);
    fillSynthF16(res_f16, seed + 37, 0.06);
    fillSynthF16(router_w_f16, seed + 53, 0.06);
    for (bias_f32, 0..) |*b, e| b.* = synthF32(seed + 83 + e, 0.04);

    // Build synthetic logits; if hetero test, push one bucket completely unselected at runtime!
    const ref_logits = try gpa.alloc(f32, @as(usize, m_val) * num_experts);
    defer gpa.free(ref_logits);
    if (use_linear_router) {
        for (0..m_val) |t| {
            for (0..num_experts) |e| {
                var acc: f32 = 0.0;
                for (0..h) |c| {
                    acc += @as(f32, @floatCast(x_f16[t * h + c])) * @as(f32, @floatCast(router_w_f16[e * h + c]));
                }
                ref_logits[t * num_experts + e] = @as(f32, @floatCast(@as(f16, @floatCast(acc))));
            }
        }
    } else {
        for (0..m_val) |t| {
            for (0..num_experts) |e| {
                const suppress = (is_hetero and expert_bucket_ids[e] == 4);
                const val = if (suppress) -20.0 else (synthF32(seed + t * 97 + e * 13, 0.12) + @as(f32, @floatFromInt(e)) * 0.03);
                logits_f16[t * num_experts + e] = @floatCast(val);
                ref_logits[t * num_experts + e] = @as(f32, @floatCast(logits_f16[t * num_experts + e]));
            }
        }
    }

    const weights_anchor = try arena.allocAligned(64, 64);
    const weights_base_off = arena.offsetOf(weights_anchor.ptr);
    const cpu_experts = try gpa.alloc(MoeExpertCpuDesc, num_experts);
    defer gpa.free(cpu_experts);
    for (0..num_experts) |e| {
        const bid = if (is_hetero) expert_bucket_ids[e] else 0;
        const alloc_res = try allocSynthMoeExpert(&arena, weights_base_off, scheme_descs[bid], pair_interleaved, hidden_size, intermediate_size, seed + e * 41);
        cpu_experts[e] = alloc_res.cpu_desc;
        table[e] = alloc_res.ew;
    }

    const ref_out = try gpa.alloc(f32, @as(usize, m_val) * h);
    defer gpa.free(ref_out);
    try inkling2MoeLayerSoftcapCpuRef(
        gpa,
        x_f16,
        ref_logits,
        bias_f32,
        false,
        true,
        true,
        INKLING2_DEFAULT_ROUTE_SCALE,
        0.0,
        INKLING2_DEFAULT_ROUTER_SOFTCAP,
        INKLING2_DEFAULT_MLP_SOFTCAP,
        cpu_experts,
        if (with_residual) res_f16 else null,
        m_val,
        top_k,
        hidden_size,
        intermediate_size,
        ref_out,
    );

    const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(ctx.device, 2);
    defer compute.close();
    try compute.makeResident(&.{ gpu_buf, bundle.qf_pipes.codebook_buf });

    var f = try compute.beginFrame(5000);
    errdefer f.abort();
    try ctx.pipes.encodeInkling2MoeLayerIndirect(&f, .{
        .moe_pipes = &bundle.moe_pipes,
        .qf_pipes = &bundle.qf_pipes,
        .dequant_pipes = &bundle.dequant_pipes,
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .router = if (use_linear_router) .{ .linear = .{
            .amf_dtype = .f16,
            .weight = .{ .buffer = gpu_buf, .offset = arena.offsetOf(router_w_raw.ptr) },
            .n = num_experts,
            .k = hidden_size,
        } } else .{ .logits = .{
            .binding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .f16 = true,
        } },
        .bias = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bias_raw.ptr) },
        .bias_f16 = false,
        .bias_add_to_logits = false,
        .scoring_sigmoid = true,
        .norm_topk_prob = true,
        .route_scale = INKLING2_DEFAULT_ROUTE_SCALE,
        .router_softcap = 0.0,
        .gate_softcap = INKLING2_DEFAULT_ROUTER_SOFTCAP,
        .up_softcap = INKLING2_DEFAULT_MLP_SOFTCAP,
        .pair_interleaved = pair_interleaved,
        .weights_base = .{ .buffer = gpu_buf, .offset = weights_base_off },
        .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
        .workspace = &ws,
        .workspace_buf = gpu_buf,
        .is_heterogeneous = is_hetero,
        .expert_schemes = if (is_hetero) .{ .buffer = gpu_buf, .offset = arena.offsetOf(schemes_raw.ptr) } else null,
        .scheme_counts = scheme_counts,
        .scheme_descs = scheme_descs,
        .residual = if (with_residual) .{ .buffer = gpu_buf, .offset = arena.offsetOf(res_raw.ptr) } else null,
        .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
        .m = m_val,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
    }, false);
    try compute.wait(try f.commit(), 5000);

    try expectF16MatchesF32(ref_out, out_f16, 0.08, 0.04);
}

test "gpu.inkling2_hy4. G29 Inkling 2 indirect Softcap MoE parity across affine, NVFP4, IQ/GGUF, and 5-bucket hetero (M=1, M=8)" {
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();
    const bundle = try TestMoeBundle.init(ctx.device, ctx.library);
    defer bundle.deinit();

    var zero_bids: [16]u8 = @splat(0);
    var default_descs: [moe.NUM_SCHEME_BUCKETS]moe.MoeSchemeDesc = .{ .{}, .{}, .{}, .{}, .{} };

    // 1. Uniform q3_affine pair_interleaved (M=1 linear router + M=8 precomputed logits).
    default_descs[0] = moe.MoeSchemeDesc.uniform(.q3_affine, 64, false);
    try verifyIndirectSoftcapMoeCase(ctx, bundle, 1, 16, 8, 256, 128, false, &zero_bids, default_descs, true, true, true, 101);
    try verifyIndirectSoftcapMoeCase(ctx, bundle, 8, 16, 8, 256, 128, false, &zero_bids, default_descs, true, false, false, 203);

    // 2. Uniform nvfp4_g16 (outer_scale != 1.0, both pair_interleaved=true and split).
    default_descs[0] = moe.MoeSchemeDesc.uniform(.nvfp4_g16, 16, false);
    try verifyIndirectSoftcapMoeCase(ctx, bundle, 1, 16, 8, 256, 128, false, &zero_bids, default_descs, true, false, true, 307);
    try verifyIndirectSoftcapMoeCase(ctx, bundle, 8, 16, 8, 256, 128, false, &zero_bids, default_descs, false, false, true, 409);

    // 3. Uniform gguf_iq3_xxs Gate+Up + gguf_iq4_nl DownProj (`INDIRECT_DOWN_Q8_OFFSET` path!).
    default_descs[0] = .{
        .gate_up_dtype = .gguf_iq3_xxs,
        .gate_up_group_size = 256,
        .down_dtype = .gguf_iq4_nl,
        .down_group_size = 32,
    };
    try verifyIndirectSoftcapMoeCase(ctx, bundle, 1, 16, 8, 256, 128, false, &zero_bids, default_descs, true, false, true, 509);
    try verifyIndirectSoftcapMoeCase(ctx, bundle, 8, 16, 8, 256, 128, false, &zero_bids, default_descs, true, false, false, 613);

    // 4. Heterogeneous 5-bucket GSQ+UD mix (bucket 4 intentionally suppressed -> 0 runtime hits).
    const hetero_bids = [16]u8{ 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1 };
    const hetero_descs = [moe.NUM_SCHEME_BUCKETS]moe.MoeSchemeDesc{
        moe.MoeSchemeDesc.uniform(.q4_affine, 64, false),
        moe.MoeSchemeDesc.uniform(.q3_affine, 64, false),
        .{ .gate_up_dtype = .nvfp4_g16, .gate_up_group_size = 16, .down_dtype = .gguf_iq4_nl, .down_group_size = 32 },
        .{ .gate_up_dtype = .gguf_iq3_xxs, .gate_up_group_size = 256, .down_dtype = .gguf_q8_0, .down_group_size = 32 },
        moe.MoeSchemeDesc.uniform(.f16, 0, false),
    };
    try verifyIndirectSoftcapMoeCase(ctx, bundle, 8, 16, 8, 256, 128, true, &hetero_bids, hetero_descs, true, false, true, 719);
}

test "gpu.inkling2_hy4. G29 grouped multi-row expert MLP parity (M=8, M_e=1 and M_e=4)" {
    const gpa = std.testing.allocator;
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();
    const bundle = try TestMoeBundle.init(ctx.device, ctx.library);
    defer bundle.deinit();

    const m_total: u32 = 8;
    const top_k: u32 = 4;
    const h: u32 = 256;
    const inter: u32 = 128;

    const Cases = [_]struct {
        gu_dt: amf.DType,
        gu_gs: u16,
        dn_dt: amf.DType,
        dn_gs: u16,
        pair_i: bool,
        m_e: u32,
    }{
        .{ .gu_dt = .q4_affine, .gu_gs = 64, .dn_dt = .q4_affine, .dn_gs = 64, .pair_i = true, .m_e = 4 },
        .{ .gu_dt = .nvfp4_g16, .gu_gs = 16, .dn_dt = .nvfp4_g16, .dn_gs = 16, .pair_i = false, .m_e = 4 },
        .{ .gu_dt = .gguf_iq3_xxs, .gu_gs = 256, .dn_dt = .gguf_iq4_nl, .dn_gs = 32, .pair_i = true, .m_e = 1 },
    };

    for (Cases, 0..) |tc, ci| {
        var arena = try makeTestArena(4 * 1024 * 1024);
        defer arena.deinit();

        const x_raw = try arena.allocAligned(@as(usize, m_total) * h * @sizeOf(f16), 64);
        const gx_raw = try arena.allocAligned(@as(usize, tc.m_e) * h * @sizeOf(f16), 64);
        const gu_raw = try arena.allocAligned(@as(usize, tc.m_e) * inter * 2 * @sizeOf(f16), 64);
        const up_raw = try arena.allocAligned(@as(usize, tc.m_e) * inter * @sizeOf(f16), 64);
        const act_raw = try arena.allocAligned(@as(usize, tc.m_e) * inter * @sizeOf(f16), 64);
        const dn_tmp_raw = try arena.allocAligned(@as(usize, tc.m_e) * h * @sizeOf(f16), 64);
        const exp_out_raw = try arena.allocAligned(@as(usize, m_total) * top_k * h * @sizeOf(f16), 64);
        const t_idx_raw = try arena.allocAligned(@as(usize, @max(tc.m_e, 4)) * @sizeOf(u32), 64);
        const k_idx_raw = try arena.allocAligned(@as(usize, @max(tc.m_e, 4)) * @sizeOf(u32), 64);
        @memset(exp_out_raw, 0);

        const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
        const exp_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, exp_out_raw));
        const t_idx: []u32 = @alignCast(std.mem.bytesAsSlice(u32, t_idx_raw));
        const k_idx: []u32 = @alignCast(std.mem.bytesAsSlice(u32, k_idx_raw));
        fillSynthF16(x_f16, 811 + ci * 31, 0.08);
        t_idx[0] = 1;
        t_idx[1] = 3;
        t_idx[2] = 5;
        t_idx[3] = 7;
        k_idx[0] = 0;
        k_idx[1] = 2;
        k_idx[2] = 1;
        k_idx[3] = 3;

        const desc = moe.MoeSchemeDesc{
            .gate_up_dtype = tc.gu_dt,
            .gate_up_group_size = tc.gu_gs,
            .down_dtype = tc.dn_dt,
            .down_group_size = tc.dn_gs,
        };
        const dummy_anchor = try arena.allocAligned(64, 64);
        const synth_e = try allocSynthMoeExpert(&arena, arena.offsetOf(dummy_anchor.ptr), desc, tc.pair_i, h, inter, 907 + ci * 53);

        // Compute exact expected slot outputs via CPU ref (`top_k = 1`, deterministic 1-expert routing).
        const x_sub = try gpa.alloc(f16, @as(usize, tc.m_e) * h);
        defer gpa.free(x_sub);
        const logits_sub = try gpa.alloc(f32, tc.m_e);
        defer gpa.free(logits_sub);
        const ref_sub = try gpa.alloc(f32, @as(usize, tc.m_e) * h);
        defer gpa.free(ref_sub);
        for (0..tc.m_e) |i| {
            const tok_i = if (tc.m_e == 1) @as(usize, 5) else t_idx[i];
            @memcpy(x_sub[i * h ..][0..h], x_f16[tok_i * h ..][0..h]);
            logits_sub[i] = 1.0;
        }
        try inkling2MoeLayerSoftcapCpuRefWithRounding(
            gpa,
            x_sub,
            logits_sub,
            null,
            false,
            true,
            true,
            1.0,
            0.0,
            INKLING2_DEFAULT_ROUTER_SOFTCAP,
            INKLING2_DEFAULT_MLP_SOFTCAP,
            &.{synth_e.cpu_desc},
            null,
            true,
            tc.m_e,
            1,
            h,
            inter,
            ref_sub,
        );

        const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(ctx.device, 2);
        defer compute.close();
        try compute.makeResident(&.{ gpu_buf, bundle.qf_pipes.codebook_buf });

        const gu_rows: u32 = if (tc.pair_i) inter * 2 else inter;
        const base_off = arena.offsetOf(dummy_anchor.ptr);
        const w1_desc: Inkling2LinearDesc = .{
            .amf_dtype = tc.gu_dt,
            .group_size = tc.gu_gs,
            .weight = .{ .buffer = gpu_buf, .offset = base_off + synth_e.ew.gate_qweight_offset },
            .params = if (synth_e.ew.gate_sb_offset > 0) .{ .buffer = gpu_buf, .offset = base_off + synth_e.ew.gate_sb_offset } else null,
            .n = gu_rows,
            .k = h,
        };
        const wu_desc: ?Inkling2LinearDesc = if (!tc.pair_i) .{
            .amf_dtype = tc.gu_dt,
            .group_size = tc.gu_gs,
            .weight = .{ .buffer = gpu_buf, .offset = base_off + synth_e.ew.up_qweight_offset },
            .params = if (synth_e.ew.up_sb_offset > 0) .{ .buffer = gpu_buf, .offset = base_off + synth_e.ew.up_sb_offset } else null,
            .n = inter,
            .k = h,
        } else null;
        const w2_desc: Inkling2LinearDesc = .{
            .amf_dtype = tc.dn_dt,
            .group_size = tc.dn_gs,
            .weight = .{ .buffer = gpu_buf, .offset = base_off + synth_e.ew.down_qweight_offset },
            .params = if (synth_e.ew.down_sb_offset > 0) .{ .buffer = gpu_buf, .offset = base_off + synth_e.ew.down_sb_offset } else null,
            .n = h,
            .k = inter,
        };

        const out_bind: metal.BufferBinding = if (tc.m_e == 1)
            .{ .buffer = gpu_buf, .offset = arena.offsetOf(exp_out_raw.ptr) + (@as(u64, 5 * top_k + 2)) * h * @sizeOf(f16) }
        else
            .{ .buffer = gpu_buf, .offset = arena.offsetOf(exp_out_raw.ptr) };

        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try ctx.pipes.encodeInkling2GroupedExpertMlp(&f, .{
            .dequant_pipes = &bundle.dequant_pipes,
            .qf_pipes = &bundle.qf_pipes,
            .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
            .w1 = w1_desc,
            .w_up = wu_desc,
            .w2 = w2_desc,
            .m_e = tc.m_e,
            .hidden_size = h,
            .intermediate_size = inter,
            .single_token_idx = 5,
            .token_indices_dev = if (tc.m_e > 1) .{ .buffer = gpu_buf, .offset = arena.offsetOf(t_idx_raw.ptr) } else null,
            .dst_slots_or_tok_dev = if (tc.m_e > 1) .{ .buffer = gpu_buf, .offset = arena.offsetOf(t_idx_raw.ptr) } else null,
            .slot_k_indices_dev = if (tc.m_e > 1) .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_idx_raw.ptr) } else null,
            .slot_stride = top_k,
            .gathered_x = if (tc.m_e > 1) .{ .buffer = gpu_buf, .offset = arena.offsetOf(gx_raw.ptr) } else null,
            .gate_up_buf = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gu_raw.ptr) },
            .up_buf = if (!tc.pair_i) .{ .buffer = gpu_buf, .offset = arena.offsetOf(up_raw.ptr) } else null,
            .act_buf = .{ .buffer = gpu_buf, .offset = arena.offsetOf(act_raw.ptr) },
            .down_tmp_buf = if (tc.m_e > 1) .{ .buffer = gpu_buf, .offset = arena.offsetOf(dn_tmp_raw.ptr) } else null,
            .expert_out = out_bind,
        }, false);
        try compute.wait(try f.commit(), 5000);

        for (0..tc.m_e) |i| {
            const slot: usize = if (tc.m_e == 1) (5 * top_k + 2) else (@as(usize, t_idx[i]) * top_k + k_idx[i]);
            try expectF16MatchesF32(ref_sub[i * h ..][0..h], exp_out_f16[slot * h ..][0..h], 0.02, 0.02);
        }
    }
}

test "gpu.inkling2_hy4. G29 zero hot-path heap allocations across all new sublayer & MoE encoders" {
    const ctx = try TestMetalCtx.init();
    defer ctx.deinit();
    const bundle = try TestMoeBundle.init(ctx.device, ctx.library);
    defer bundle.deinit();

    const m_val: u32 = 8;
    const h: u32 = 256;
    const inter: u32 = 128;
    const num_experts: u32 = 16;
    const top_k: u32 = 8;

    const ws_spec: moe.WorkspaceSpec = .{
        .max_tokens = m_val,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = h,
        .intermediate_size = inter,
    };
    const ws_bytes = try moe.MoeWorkspace.requiredBytes(ws_spec);
    var arena = try makeTestArena(ws_bytes + 8 * 1024 * 1024);
    defer arena.deinit();

    const ws = try moe.MoeWorkspace.init(&arena, ws_spec);
    var ring = try AttnResRingArena.init(&arena, .{ .max_blocks = 4, .max_tokens = m_val, .hidden_dim = h });
    _ = ring.pushSlot();
    const dummy_b = try arena.allocAligned(2 * 1024 * 1024, 64);
    @memset(dummy_b, 0);
    const dummy_pos = try arena.allocAligned(@as(usize, m_val) * @sizeOf(u32), 64);
    @memset(dummy_pos, 0);

    const anchor = try arena.allocAligned(64, 64);
    const synth_e = try allocSynthMoeExpert(
        &arena,
        arena.offsetOf(anchor.ptr),
        moe.MoeSchemeDesc.uniform(.q4_affine, 64, false),
        true,
        h,
        inter,
        111,
    );
    const table_raw = try arena.allocAligned(@as(usize, num_experts) * @sizeOf(moe.ExpertWeights), 64);
    const table: []moe.ExpertWeights = @alignCast(std.mem.bytesAsSlice(moe.ExpertWeights, table_raw));
    for (table) |*slot| slot.* = synth_e.ew;

    const gpu_buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(ctx.device, 2);
    defer compute.close();
    try compute.makeResident(&.{ gpu_buf, bundle.qf_pipes.codebook_buf });

    var fail_alloc = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    _ = fail_alloc.allocator();

    const d_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(dummy_b.ptr) };
    const pos_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(dummy_pos.ptr) };
    var f = try compute.beginFrame(5000);
    errdefer f.abort();
    try ctx.pipes.encodeInkling2AttnPostProj(&f, .{
        .q = d_bind,
        .k = d_bind,
        .v = d_bind,
        .positions = pos_bind,
        .sink_f16 = d_bind,
        .sink_f32 = d_bind,
        .m = m_val,
        .num_q_heads = 2,
        .num_kv_heads = 2,
        .head_dim = 128,
        .use_log_scaling = true,
        .log_floor = 256,
        .log_alpha = 0.125,
        .do_v_norm = true,
    }, false);
    try ctx.pipes.encodeInkling2DynSconvAssemble(&f, .{
        .proj = d_bind,
        .dyn_w = d_bind,
        .m = m_val,
        .kv_dim = h,
        .proj_n = h * 4,
    }, false);
    const bl = try ctx.pipes.encodeInkling2MsgBankNorm(&f, .{
        .ring = &ring,
        .ring_buf = gpu_buf,
        .cur = d_bind,
        .msg_bank_out = d_bind,
        .m = m_val,
    }, true);
    try ctx.pipes.encodeInkling2CrossPassMsgs(&f, .{
        .msg_bank = d_bind,
        .q_src = d_bind,
        .out_msgs = d_bind,
        .m = m_val,
        .hidden_dim = h,
        .bank_len = bl,
        .q_rows = 4,
        .loop_layer_idx = 0,
    }, true);
    try ctx.pipes.encodeInkling2MessageGate(&f, .{
        .cur = d_bind,
        .msg = d_bind,
        .down_weight = d_bind,
        .forget_weight = d_bind,
        .gain_weight = d_bind,
        .z_scratch = d_bind,
        .out = d_bind,
        .m = m_val,
        .state_dim = h,
        .down_in_dim = h * 2,
    }, true);
    try ctx.pipes.encodeInkling2MoeLayerIndirect(&f, .{
        .moe_pipes = &bundle.moe_pipes,
        .qf_pipes = &bundle.qf_pipes,
        .dequant_pipes = &bundle.dequant_pipes,
        .x = d_bind,
        .router = .{ .logits = .{ .binding = d_bind, .f16 = true } },
        .weights_base = .{ .buffer = gpu_buf, .offset = arena.offsetOf(anchor.ptr) },
        .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
        .workspace = &ws,
        .workspace_buf = gpu_buf,
        .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(dummy_b.ptr) + 1024 * 1024 },
        .m = m_val,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = h,
        .intermediate_size = inter,
    }, true);
    try compute.wait(try f.commit(), 5000);
    try std.testing.expectEqual(@as(usize, 0), fail_alloc.allocated_bytes);
}
