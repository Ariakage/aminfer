//! GPU MoE router, deduplicated active-expert fused 3-bit / 4-bit affine
//! SwiGLU FFN (G3), dense SwiGLU utilities, and single-dispatch greedy argmax.
//! M2.5 (G12, ADR-0016) adds BF16 and MLX-quantized routers, native FP16 /
//! BF16 deduplicated MoE SwiGLU, and GPU embedding lookup.
//!
//! All pipelines are compiled once at startup using Metal 4 function constants
//! (`Q_BITS`, `Q_GROUP_SIZE`, `MOE_TOP_K`, `LOGITS_F16`, `ROUTER_W_F16`,
//! `ROUTER_W_BF16`, `MOE_W_BF16`, `EMBED_KIND`) and
//! execute strictly over preallocated wired workspaces with zero hot-path
//! dynamic heap allocation (ADR-0009, ADR-0011, ADR-0014).

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const quant = @import("../quant/spec.zig");
const quant_formats = @import("quant_formats.zig");
const dequant = @import("dequant.zig");
const amf = @import("../format/amf.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const vmstat = @import("../platform/vmstat.zig");

/// Maximum supported number of MoE experts per layer (G7, ADR-0015).
pub const MAX_EXPERTS: u32 = 1024;
/// Maximum supported number of active experts per token (`top_k`).
pub const MAX_TOP_K: u32 = 16;
/// Maximum batched rows for `encodeArgmaxRows` and `encodeTopKLogitsRows`.
/// Maximum rows in batched GPU argmax / top-K logits (`encodeArgmaxRows`,
/// `encodeTopKLogitsRows`), supporting wide prefill chunks up to 1024 (`G15`).
pub const MAX_LOGIT_ROWS: u32 = 1024;

/// Maximum wide-chunk GPU prefill batch width (`G15`, ADR-0020).
pub const MAX_WIDE_PREFILL_TOKENS: u32 = 1024;
/// Number of distinct `WeightDType` scheme buckets supported by G13 heterogeneous MoE (ADR-0017).
pub const NUM_SCHEME_BUCKETS: usize = quant.WeightDType.COUNT;
/// Byte stride between adjacent `ScheduleHeader` records inside `bucket_headers`.
pub const BUCKET_HEADER_STRIDE: usize = 64;
/// Byte offsets of the four G20 `metal.IndirectDispatchArgs` (`[3]u32`, 12 B each)
/// embedded inside every 64-byte schedule slot (`schedule_header_off` and
/// `bucket_headers_off + s * BUCKET_HEADER_STRIDE`, ADR-0023 §3).
pub const INDIRECT_GATE_UP_Q_OFFSET: u64 = 16;
pub const INDIRECT_GATE_UP_F_OFFSET: u64 = 28;
pub const INDIRECT_DOWN_OFFSET: u64 = 40;
pub const INDIRECT_DOWN_Q8_OFFSET: u64 = 52;

/// M7.8 G20 static-grid crossover guard (ADR-0023 §3):
/// When `M == 1` and all routed experts in the layer share a single homogeneous
/// `DType` (`num_scheme_buckets <= 1`), `active_bucket_pairs == top_k` is
/// statically known with zero empty threadgroups; keep direct CPU-encoded
/// `dispatchThreadgroups` so `M = 1` decode pays zero indirect fetch stall.
/// Activate GPU indirect dispatch ONLY when `M > 1` or `num_scheme_buckets > 1`.
pub inline fn shouldUseIndirectDispatch(m: u32, num_scheme_buckets: u32) bool {
    return m > 1 or num_scheme_buckets > 1;
}

/// Counts how many `WeightDType` scheme buckets are statically populated in a layer.
pub inline fn countActiveSchemeBuckets(scheme_counts: [NUM_SCHEME_BUCKETS]u32) u32 {
    var active: u32 = 0;
    for (scheme_counts) |cnt| {
        if (cnt > 0) active += 1;
    }
    return active;
}

pub inline fn gateUpGroupsXQ(intermediate_size: u32) u32 {
    return @intCast((@as(u64, intermediate_size) + 7) / 8);
}

pub inline fn gateUpGroupsXF(intermediate_size: u32) u32 {
    return @intCast((@as(u64, intermediate_size) + 15) / 16);
}

pub inline fn downGroupsX(hidden_size: u32) u32 {
    return @intCast((@as(u64, hidden_size) + 15) / 16);
}

pub inline fn downGroupsXQ8(hidden_size: u32) u32 {
    return @intCast((@as(u64, hidden_size) + 7) / 8);
}

inline fn writeScheduleSlotIndirect(
    slot_base: [*]u8,
    active_experts: u32,
    intermediate_size: u32,
    hidden_size: u32,
) void {
    const gu_q: *metal.IndirectDispatchArgs = @ptrCast(@alignCast(slot_base + INDIRECT_GATE_UP_Q_OFFSET));
    const gu_f: *metal.IndirectDispatchArgs = @ptrCast(@alignCast(slot_base + INDIRECT_GATE_UP_F_OFFSET));
    const dn: *metal.IndirectDispatchArgs = @ptrCast(@alignCast(slot_base + INDIRECT_DOWN_OFFSET));
    const dn8: *metal.IndirectDispatchArgs = @ptrCast(@alignCast(slot_base + INDIRECT_DOWN_Q8_OFFSET));
    gu_q.* = metal.IndirectDispatchArgs.init(gateUpGroupsXQ(intermediate_size), active_experts, 1);
    gu_f.* = metal.IndirectDispatchArgs.init(gateUpGroupsXF(intermediate_size), active_experts, 1);
    dn.* = metal.IndirectDispatchArgs.init(downGroupsX(hidden_size), active_experts, 1);
    dn8.* = metal.IndirectDispatchArgs.init(downGroupsXQ8(hidden_size), active_experts, 1);
}

/// Device-memory expert weight offset table entry (48 bytes, zero padding).
/// All offsets are 64-bit byte offsets from the bound resident weights buffer,
/// supporting multi-GiB `WiredArena` `MTLBuffer`s without 32-bit overflow.
/// Every offset must be at least 4-byte aligned (`.amf` blobs are 16 KiB
/// aligned and `params_offset` is 64-byte aligned). For the FP16 / BF16 MoE
/// path (`encodeMoELayerFloat`) the `*_qweight_offset` fields hold the byte
/// offsets of row-major `[N, K]` 16-bit matrices (at least 8-byte aligned) and
/// the `*_sb_offset` fields are ignored (set them to 0).
pub const ExpertWeights = extern struct {
    gate_qweight_offset: u64,
    gate_sb_offset: u64,
    up_qweight_offset: u64,
    up_sb_offset: u64,
    down_qweight_offset: u64,
    down_sb_offset: u64,
};

/// Header written by `ami_moe_route_topk` at the start of the deduplicated
/// active-expert schedule (16 bytes, zero padding).
pub const ScheduleHeader = extern struct {
    num_unique_experts: u32,
    total_assignments: u32,
    m: u32,
    top_k: u32,
};

pub const RouterProjParams = extern struct {
    m: u32,
    num_experts: u32,
    hidden_size: u32,
    reserved: u32 = 0,
};

pub const RouteParams = extern struct {
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: u32,
    gate_up_groups_x_q: u32 = 0,
    gate_up_groups_x_f: u32 = 0,
    down_groups_x: u32 = 0,
    down_groups_x_q8: u32 = 0,
};

pub const MoeGateUpParams = extern struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_groups: u32,
    row_bytes: u32,
    swiglu_limit: f32 = 0.0,
    _pad_sw0: u32 = 0,
    _pad_sw1: u32 = 0,
    _pad_sw2: u32 = 0,
};

pub const MoeDownParams = extern struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_groups: u32,
    row_bytes: u32,
    top_k: u32,
    reserved: u32 = 0,
};

pub const MoeReduceParams = extern struct {
    m: u32,
    hidden_size: u32,
    top_k: u32,
    add_residual: u32,
};

pub const ElemParams = extern struct {
    count: u32,
    swiglu_limit: f32 = 0.0,
};

pub const ArgmaxParams = extern struct {
    vocab_size: u32,
    aligned_half4: u32,
};

/// Uniform payload matching `struct RouterQParams` in `moe_ffn.metal`.
pub const RouterQParams = extern struct {
    m: u32,
    num_experts: u32,
    hidden_size: u32,
    num_groups: u32,
    row_bytes: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

/// Uniform payload matching `struct EmbedParams` in `moe_ffn.metal`.
pub const EmbedParams = extern struct {
    m: u32,
    vocab_size: u32,
    hidden_size: u32,
    row_bytes: u32,
    num_groups: u32,
    use_token_buffer: u32,
    token_id: u32,
    _pad: u32 = 0,
};

/// Uniform payload matching `struct ArgmaxRowsParams` in `moe_ffn.metal`.
pub const ArgmaxRowsParams = extern struct {
    num_rows: u32,
    vocab_size: u32,
    aligned_vec4: u32,
    _pad: u32 = 0,
};

/// Output candidate written by `encodeTopKLogitsRows` (`[M, top_k]`).
/// Matches `struct TopKCandidate` in `moe_ffn.metal` (16 bytes).
pub const TopKCandidate = extern struct {
    token_id: u32,
    prob: f32,
    log_prob: f32,
    _pad: u32 = 0,
};

/// Uniform payload matching `struct TopKLogitsRowsParams` in `moe_ffn.metal`.
pub const TopKLogitsRowsParams = extern struct {
    num_rows: u32,
    vocab_size: u32,
    top_k: u32,
    aligned_vec4: u32,
};

/// Uniform payload matching `struct MoePerSlotParams` in `moe_ffn.metal`.
pub const MoePerSlotParams = extern struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_groups: u32,
    row_bytes: u32,
    m: u32,
    top_k: u32,
    swiglu_limit: f32 = 0.0,
    _pad_sw: u32 = 0,
};

/// Uniform payload matching `struct CopyStridedParams` in `moe_ffn.metal`.
pub const CopyStridedParams = extern struct {
    num_rows: u32,
    copy_cols: u32,
    src_stride: u32,
    src_col_off: u32,
    dst_stride: u32,
    dst_col_off: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

/// Uniform payload matching `struct GatherRowsParams` in `moe_ffn.metal`.
pub const GatherRowsParams = extern struct {
    count: u32,
    row_elems: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

/// Uniform payload matching `struct BucketRouteParams` in `moe_ffn.metal` (G13/G20, ADR-0017/ADR-0023).
pub const BucketRouteParams = extern struct {
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: u32,
    bucket_unique_stride: u32,
    bucket_offsets_stride: u32,
    gate_up_groups_x_q: u32 = 0,
    gate_up_groups_x_f: u32 = 0,
    down_groups_x: u32 = 0,
    down_groups_x_q8: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

/// Uniform payload matching `struct MoeSharedGateParams` in `moe_ffn.metal` (G7).
pub const MoeSharedGateParams = extern struct {
    m: u32,
    hidden_size: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

/// Uniform payload matching `struct MoeSharedGateUpParams` in `moe_ffn.metal` (G7).
pub const MoeSharedGateUpParams = extern struct {
    gate_qweight_offset: u64,
    gate_sb_offset: u64,
    up_qweight_offset: u64,
    up_sb_offset: u64,
    m: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_groups: u32,
    row_bytes: u32,
    swiglu_limit: f32 = 0.0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

/// Uniform payload matching `struct MoeSharedDownParams` in `moe_ffn.metal` (G7).
pub const MoeSharedDownParams = extern struct {
    down_qweight_offset: u64,
    down_sb_offset: u64,
    m: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_groups: u32,
    row_bytes: u32,
    use_gate: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

comptime {
    if (@sizeOf(ExpertWeights) != 48) @compileError("ExpertWeights must be 48 bytes");
    if (@sizeOf(ScheduleHeader) != 16) @compileError("ScheduleHeader must be 16 bytes");
    if (@sizeOf(RouterProjParams) != 16) @compileError("RouterProjParams must be 16 bytes");
    if (@sizeOf(RouteParams) != 32) @compileError("RouteParams must be 32 bytes");
    if (@sizeOf(MoeGateUpParams) != 32) @compileError("MoeGateUpParams must be 32 bytes");
    if (@sizeOf(MoeDownParams) != 24) @compileError("MoeDownParams must be 24 bytes");
    if (@sizeOf(MoeReduceParams) != 16) @compileError("MoeReduceParams must be 16 bytes");
    if (@sizeOf(ElemParams) != 8) @compileError("ElemParams must be 8 bytes");
    if (@sizeOf(ArgmaxParams) != 8) @compileError("ArgmaxParams must be 8 bytes");
    if (@sizeOf(RouterQParams) != 32) @compileError("RouterQParams must be 32 bytes");
    if (@sizeOf(EmbedParams) != 32) @compileError("EmbedParams must be 32 bytes");
    if (@sizeOf(ArgmaxRowsParams) != 16) @compileError("ArgmaxRowsParams must be 16 bytes");
    if (@sizeOf(TopKCandidate) != 16) @compileError("TopKCandidate must be 16 bytes");
    if (@sizeOf(TopKLogitsRowsParams) != 16) @compileError("TopKLogitsRowsParams must be 16 bytes");
    if (@sizeOf(MoePerSlotParams) != 32) @compileError("MoePerSlotParams must be 32 bytes");
    if (@sizeOf(CopyStridedParams) != 32) @compileError("CopyStridedParams must be 32 bytes");
    if (@sizeOf(GatherRowsParams) != 16) @compileError("GatherRowsParams must be 16 bytes");
    if (@sizeOf(BucketRouteParams) != 48) @compileError("BucketRouteParams must be 48 bytes");
    if (@sizeOf(MoeSharedGateParams) != 16) @compileError("MoeSharedGateParams must be 16 bytes");
    if (@sizeOf(MoeSharedGateUpParams) != 64) @compileError("MoeSharedGateUpParams must be 64 bytes");
    if (@sizeOf(MoeSharedDownParams) != 48) @compileError("MoeSharedDownParams must be 48 bytes");
}

pub const LogitsDType = enum {
    f16,
    f32,
};

pub const RouterWeightDType = enum {
    f16,
    f32,
    bf16,
};

pub const WorkspaceSpec = struct {
    max_tokens: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    /// Optional dedicated shared-expert SwiGLU intermediate width (G7).
    /// When 0, shared-expert layers with `shared_intermediate_size <= top_k * intermediate_size`
    /// reuse `workspace.inter` after routed `DownProj` completes (zero extra bytes).
    shared_intermediate_size: u32 = 0,
};

pub const WorkspaceError = error{
    ZeroDimension,
    UnsupportedExpertCount,
    UnsupportedTopK,
    Overflow,
} || arena_mod.WiredArena.AllocError;

pub const EncodeError = metal.Error || quant.ShapeError || quant_formats.EncodeError || error{
    UnsupportedExpertCount,
    UnsupportedTopK,
    WorkspaceTooSmall,
};

fn validateWorkspaceSpec(spec: WorkspaceSpec) WorkspaceError!void {
    if (spec.max_tokens == 0 or spec.num_experts == 0 or spec.top_k == 0 or
        spec.hidden_size == 0 or spec.intermediate_size == 0)
    {
        return error.ZeroDimension;
    }
    if (spec.num_experts > MAX_EXPERTS) return error.UnsupportedExpertCount;
    if (spec.top_k > MAX_TOP_K or spec.top_k > spec.num_experts) return error.UnsupportedTopK;
}

/// Preallocated wired workspace for deduplicated MoE routing and SwiGLU FFN.
/// Carved once at startup from a `WiredArena`; zero heap allocation at runtime.
pub const MoeWorkspace = struct {
    spec: WorkspaceSpec,

    router_logits_off: u64,
    schedule_header_off: u64,
    unique_experts_off: u64,
    expert_token_offsets_off: u64,
    token_indices_off: u64,
    routing_weights_off: u64,
    slot_k_indices_off: u64,
    scratch_topk_experts_off: u64,
    scratch_topk_weights_off: u64,
    inter_off: u64,
    down_out_off: u64,
    // Preallocated G13 per-scheme CSR buckets (ADR-0017; zero hot-path alloc).
    bucket_headers_off: u64,
    bucket_unique_experts_off: u64,
    bucket_expert_token_offsets_off: u64,
    bucket_unique_stride_u32: u32,
    bucket_offsets_stride_u32: u32,
    // Preallocated G7 shared-expert scalar gate & optional dedicated intermediate scratch.
    shared_gate_scores_off: u64,
    shared_inter_off: u64,

    router_logits: []f32,
    schedule_header_raw: []u8,
    schedule_header: *ScheduleHeader,
    unique_experts: []u32,
    expert_token_offsets: []u32,
    token_indices: []u32,
    routing_weights: []f32,
    slot_k_indices: []u32,
    scratch_topk_experts: []u32,
    scratch_topk_weights: []f32,
    inter: []f16,
    down_out: []f32,
    bucket_headers_raw: []u8,
    bucket_unique_experts: []u32,
    bucket_expert_token_offsets: []u32,
    shared_gate_scores: []f32,
    shared_inter: []f16,

    /// Tight upper bound on arena bytes needed to carve this workspace (including
    /// 64-byte alignment padding per sub-allocation).
    pub fn requiredBytes(spec: WorkspaceSpec) WorkspaceError!usize {
        try validateWorkspaceSpec(spec);
        const max_tok: usize = spec.max_tokens;
        const n_exp: usize = spec.num_experts;
        const k: usize = spec.top_k;
        const hidden: usize = spec.hidden_size;
        const inter_dim: usize = spec.intermediate_size;
        const shared_inter_dim: usize = spec.shared_intermediate_size;

        const assignments = std.math.mul(usize, max_tok, k) catch return error.Overflow;
        const logits_elems = std.math.mul(usize, max_tok, n_exp) catch return error.Overflow;
        const inter_elems = std.math.mul(usize, assignments, inter_dim) catch return error.Overflow;
        const down_elems = std.math.mul(usize, assignments, hidden) catch return error.Overflow;
        const bucket_u_stride = amf.align64(std.math.mul(usize, n_exp, @sizeOf(u32)) catch return error.Overflow);
        const bucket_o_stride = amf.align64(std.math.mul(usize, n_exp + 1, @sizeOf(u32)) catch return error.Overflow);
        const shared_gate_bytes = std.math.mul(usize, max_tok, @sizeOf(f32)) catch return error.Overflow;
        const shared_inter_bytes = if (shared_inter_dim > 0)
            std.math.mul(usize, std.math.mul(usize, max_tok, shared_inter_dim) catch return error.Overflow, @sizeOf(f16)) catch return error.Overflow
        else
            0;

        const sizes = [_]usize{
            std.math.mul(usize, logits_elems, @sizeOf(f32)) catch return error.Overflow,
            BUCKET_HEADER_STRIDE,
            std.math.mul(usize, n_exp, @sizeOf(u32)) catch return error.Overflow,
            std.math.mul(usize, n_exp + 1, @sizeOf(u32)) catch return error.Overflow,
            std.math.mul(usize, assignments, @sizeOf(u32)) catch return error.Overflow,
            std.math.mul(usize, assignments, @sizeOf(f32)) catch return error.Overflow,
            std.math.mul(usize, assignments, @sizeOf(u32)) catch return error.Overflow,
            std.math.mul(usize, assignments, @sizeOf(u32)) catch return error.Overflow,
            std.math.mul(usize, assignments, @sizeOf(f32)) catch return error.Overflow,
            std.math.mul(usize, inter_elems, @sizeOf(f16)) catch return error.Overflow,
            std.math.mul(usize, down_elems, @sizeOf(f32)) catch return error.Overflow,
            NUM_SCHEME_BUCKETS * BUCKET_HEADER_STRIDE,
            NUM_SCHEME_BUCKETS * bucket_u_stride,
            NUM_SCHEME_BUCKETS * bucket_o_stride,
            shared_gate_bytes,
            shared_inter_bytes,
        };

        // Account for up to 64 bytes of initial arena alignment plus 64-byte rounding per slice.
        var total: usize = 64;
        for (sizes) |sz| {
            if (sz == 0) continue;
            const aligned = amf.align64(sz);
            total = std.math.add(usize, total, aligned) catch return error.Overflow;
        }
        return total;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: WorkspaceSpec) WorkspaceError!MoeWorkspace {
        try validateWorkspaceSpec(spec);
        const max_tok: usize = spec.max_tokens;
        const n_exp: usize = spec.num_experts;
        const k: usize = spec.top_k;
        const hidden: usize = spec.hidden_size;
        const inter_dim: usize = spec.intermediate_size;
        const shared_inter_dim: usize = spec.shared_intermediate_size;

        const assignments = std.math.mul(usize, max_tok, k) catch return error.Overflow;
        const logits_bytes = std.math.mul(usize, std.math.mul(usize, max_tok, n_exp) catch return error.Overflow, @sizeOf(f32)) catch return error.Overflow;
        const uniq_bytes = std.math.mul(usize, n_exp, @sizeOf(u32)) catch return error.Overflow;
        const offs_bytes = std.math.mul(usize, n_exp + 1, @sizeOf(u32)) catch return error.Overflow;
        const tok_idx_bytes = std.math.mul(usize, assignments, @sizeOf(u32)) catch return error.Overflow;
        const rw_bytes = std.math.mul(usize, assignments, @sizeOf(f32)) catch return error.Overflow;
        const slot_k_bytes = tok_idx_bytes;
        const scratch_e_bytes = tok_idx_bytes;
        const scratch_w_bytes = rw_bytes;
        const inter_bytes = std.math.mul(usize, std.math.mul(usize, assignments, inter_dim) catch return error.Overflow, @sizeOf(f16)) catch return error.Overflow;
        const down_bytes = std.math.mul(usize, std.math.mul(usize, assignments, hidden) catch return error.Overflow, @sizeOf(f32)) catch return error.Overflow;
        const bucket_u_stride_bytes = amf.align64(uniq_bytes);
        const bucket_o_stride_bytes = amf.align64(offs_bytes);
        const bucket_hdrs_bytes = NUM_SCHEME_BUCKETS * BUCKET_HEADER_STRIDE;
        const bucket_uniq_bytes = NUM_SCHEME_BUCKETS * bucket_u_stride_bytes;
        const bucket_offs_bytes = NUM_SCHEME_BUCKETS * bucket_o_stride_bytes;
        const shared_gate_bytes = std.math.mul(usize, max_tok, @sizeOf(f32)) catch return error.Overflow;
        const shared_inter_bytes = if (shared_inter_dim > 0)
            std.math.mul(usize, std.math.mul(usize, max_tok, shared_inter_dim) catch return error.Overflow, @sizeOf(f16)) catch return error.Overflow
        else
            0;

        const router_logits_raw = try arena.allocAligned(logits_bytes, 64);
        const sched_hdr_raw = try arena.allocAligned(BUCKET_HEADER_STRIDE, 64);
        const uniq_raw = try arena.allocAligned(uniq_bytes, 64);
        const offs_raw = try arena.allocAligned(offs_bytes, 64);
        const tok_idx_raw = try arena.allocAligned(tok_idx_bytes, 64);
        const rw_raw = try arena.allocAligned(rw_bytes, 64);
        const slot_k_raw = try arena.allocAligned(slot_k_bytes, 64);
        const scratch_e_raw = try arena.allocAligned(scratch_e_bytes, 64);
        const scratch_w_raw = try arena.allocAligned(scratch_w_bytes, 64);
        const inter_raw = try arena.allocAligned(inter_bytes, 64);
        const down_raw = try arena.allocAligned(down_bytes, 64);
        const bucket_hdrs_raw = try arena.allocAligned(bucket_hdrs_bytes, 64);
        const bucket_uniq_raw = try arena.allocAligned(bucket_uniq_bytes, 64);
        const bucket_offs_raw = try arena.allocAligned(bucket_offs_bytes, 64);
        const shared_gate_raw = try arena.allocAligned(shared_gate_bytes, 64);
        const shared_inter_raw: ?[]u8 = if (shared_inter_bytes > 0)
            try arena.allocAligned(shared_inter_bytes, 64)
        else
            null;

        @memset(sched_hdr_raw, 0);
        @memset(bucket_hdrs_raw, 0);
        @memset(shared_gate_raw, 0);

        const inter_off = arena.offsetOf(inter_raw.ptr);
        const shared_inter_off = if (shared_inter_raw) |sr| arena.offsetOf(sr.ptr) else inter_off;
        const shared_inter_slice: []f16 = if (shared_inter_raw) |sr| @alignCast(std.mem.bytesAsSlice(f16, sr)) else &.{};

        return .{
            .spec = spec,
            .router_logits_off = arena.offsetOf(router_logits_raw.ptr),
            .schedule_header_off = arena.offsetOf(sched_hdr_raw.ptr),
            .unique_experts_off = arena.offsetOf(uniq_raw.ptr),
            .expert_token_offsets_off = arena.offsetOf(offs_raw.ptr),
            .token_indices_off = arena.offsetOf(tok_idx_raw.ptr),
            .routing_weights_off = arena.offsetOf(rw_raw.ptr),
            .slot_k_indices_off = arena.offsetOf(slot_k_raw.ptr),
            .scratch_topk_experts_off = arena.offsetOf(scratch_e_raw.ptr),
            .scratch_topk_weights_off = arena.offsetOf(scratch_w_raw.ptr),
            .inter_off = inter_off,
            .down_out_off = arena.offsetOf(down_raw.ptr),
            .bucket_headers_off = arena.offsetOf(bucket_hdrs_raw.ptr),
            .bucket_unique_experts_off = arena.offsetOf(bucket_uniq_raw.ptr),
            .bucket_expert_token_offsets_off = arena.offsetOf(bucket_offs_raw.ptr),
            .bucket_unique_stride_u32 = @intCast(bucket_u_stride_bytes / @sizeOf(u32)),
            .bucket_offsets_stride_u32 = @intCast(bucket_o_stride_bytes / @sizeOf(u32)),
            .shared_gate_scores_off = arena.offsetOf(shared_gate_raw.ptr),
            .shared_inter_off = shared_inter_off,
            .router_logits = @alignCast(std.mem.bytesAsSlice(f32, router_logits_raw)),
            .schedule_header_raw = sched_hdr_raw,
            .schedule_header = @ptrCast(@alignCast(sched_hdr_raw.ptr)),
            .unique_experts = @alignCast(std.mem.bytesAsSlice(u32, uniq_raw)),
            .expert_token_offsets = @alignCast(std.mem.bytesAsSlice(u32, offs_raw)),
            .token_indices = @alignCast(std.mem.bytesAsSlice(u32, tok_idx_raw)),
            .routing_weights = @alignCast(std.mem.bytesAsSlice(f32, rw_raw)),
            .slot_k_indices = @alignCast(std.mem.bytesAsSlice(u32, slot_k_raw)),
            .scratch_topk_experts = @alignCast(std.mem.bytesAsSlice(u32, scratch_e_raw)),
            .scratch_topk_weights = @alignCast(std.mem.bytesAsSlice(f32, scratch_w_raw)),
            .inter = @alignCast(std.mem.bytesAsSlice(f16, inter_raw)),
            .down_out = @alignCast(std.mem.bytesAsSlice(f32, down_raw)),
            .bucket_headers_raw = bucket_hdrs_raw,
            .bucket_unique_experts = @alignCast(std.mem.bytesAsSlice(u32, bucket_uniq_raw)),
            .bucket_expert_token_offsets = @alignCast(std.mem.bytesAsSlice(u32, bucket_offs_raw)),
            .shared_gate_scores = @alignCast(std.mem.bytesAsSlice(f32, shared_gate_raw)),
            .shared_inter = shared_inter_slice,
        };
    }

    pub fn routerLogitsBinding(self: *const MoeWorkspace, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.router_logits_off };
    }

    pub fn bucketHeaderOffset(self: *const MoeWorkspace, scheme: quant.WeightDType) u64 {
        return self.bucket_headers_off + @as(u64, @backingInt(scheme)) * BUCKET_HEADER_STRIDE;
    }

    pub fn bucketUniqueOffset(self: *const MoeWorkspace, scheme: quant.WeightDType) u64 {
        return self.bucket_unique_experts_off + @as(u64, @backingInt(scheme)) * @as(u64, self.bucket_unique_stride_u32) * @sizeOf(u32);
    }

    pub fn bucketOffsetsOffset(self: *const MoeWorkspace, scheme: quant.WeightDType) u64 {
        return self.bucket_expert_token_offsets_off + @as(u64, @backingInt(scheme)) * @as(u64, self.bucket_offsets_stride_u32) * @sizeOf(u32);
    }

    pub fn bucketHeader(self: *const MoeWorkspace, scheme: quant.WeightDType) *const ScheduleHeader {
        const off: usize = @as(usize, @backingInt(scheme)) * BUCKET_HEADER_STRIDE;
        return @ptrCast(@alignCast(self.bucket_headers_raw.ptr + off));
    }

    pub fn bucketHeaderMut(self: *MoeWorkspace, scheme: quant.WeightDType) *ScheduleHeader {
        const off: usize = @as(usize, @backingInt(scheme)) * BUCKET_HEADER_STRIDE;
        return @ptrCast(@alignCast(self.bucket_headers_raw.ptr + off));
    }

    pub fn scheduleIndirectOffset(self: *const MoeWorkspace, scheme: ?quant.WeightDType, field_off: u64) u64 {
        const base = if (scheme) |s| self.bucketHeaderOffset(s) else self.schedule_header_off;
        return base + field_off;
    }

    pub fn scheduleIndirectGateUpQ(self: *const MoeWorkspace) *const metal.IndirectDispatchArgs {
        return @ptrCast(@alignCast(self.schedule_header_raw.ptr + INDIRECT_GATE_UP_Q_OFFSET));
    }

    pub fn scheduleIndirectGateUpF(self: *const MoeWorkspace) *const metal.IndirectDispatchArgs {
        return @ptrCast(@alignCast(self.schedule_header_raw.ptr + INDIRECT_GATE_UP_F_OFFSET));
    }

    pub fn scheduleIndirectDown(self: *const MoeWorkspace) *const metal.IndirectDispatchArgs {
        return @ptrCast(@alignCast(self.schedule_header_raw.ptr + INDIRECT_DOWN_OFFSET));
    }

    pub fn scheduleIndirectDownQ8(self: *const MoeWorkspace) *const metal.IndirectDispatchArgs {
        return @ptrCast(@alignCast(self.schedule_header_raw.ptr + INDIRECT_DOWN_Q8_OFFSET));
    }

    pub fn bucketIndirectGateUpQ(self: *const MoeWorkspace, scheme: quant.WeightDType) *const metal.IndirectDispatchArgs {
        const off: usize = @as(usize, @backingInt(scheme)) * BUCKET_HEADER_STRIDE + INDIRECT_GATE_UP_Q_OFFSET;
        return @ptrCast(@alignCast(self.bucket_headers_raw.ptr + off));
    }

    pub fn bucketIndirectGateUpF(self: *const MoeWorkspace, scheme: quant.WeightDType) *const metal.IndirectDispatchArgs {
        const off: usize = @as(usize, @backingInt(scheme)) * BUCKET_HEADER_STRIDE + INDIRECT_GATE_UP_F_OFFSET;
        return @ptrCast(@alignCast(self.bucket_headers_raw.ptr + off));
    }

    pub fn bucketIndirectDown(self: *const MoeWorkspace, scheme: quant.WeightDType) *const metal.IndirectDispatchArgs {
        const off: usize = @as(usize, @backingInt(scheme)) * BUCKET_HEADER_STRIDE + INDIRECT_DOWN_OFFSET;
        return @ptrCast(@alignCast(self.bucket_headers_raw.ptr + off));
    }

    pub fn bucketIndirectDownQ8(self: *const MoeWorkspace, scheme: quant.WeightDType) *const metal.IndirectDispatchArgs {
        const off: usize = @as(usize, @backingInt(scheme)) * BUCKET_HEADER_STRIDE + INDIRECT_DOWN_Q8_OFFSET;
        return @ptrCast(@alignCast(self.bucket_headers_raw.ptr + off));
    }

    /// Zero-allocation CPU-side CSR scheme partitioner (G13, ADR-0017).
    /// Re-partitions the active experts currently in `self` into the 5 disjoint
    /// preallocated per-scheme CSR buckets sharing `token_indices`/`routing_weights`/
    /// `slot_k_indices`, matching `ami_moe_route_topk_bucketed` identically.
    pub fn rebuildSchemeBucketsCpu(self: *MoeWorkspace, expert_dtypes: []const quant.WeightDType) void {
        const num_u: usize = self.schedule_header.num_unique_experts;
        const total_a: usize = self.schedule_header.total_assignments;
        const m_val: u32 = self.schedule_header.m;
        const top_k_val: u32 = self.schedule_header.top_k;
        std.debug.assert(num_u <= MAX_EXPERTS);
        std.debug.assert(total_a <= self.token_indices.len);

        var orig_u: [MAX_EXPERTS]u32 = undefined;
        var orig_off: [MAX_EXPERTS + 1]u32 = undefined;
        @memcpy(orig_u[0..num_u], self.unique_experts[0..num_u]);
        @memcpy(orig_off[0 .. num_u + 1], self.expert_token_offsets[0 .. num_u + 1]);

        // Stage the pre-partition CSR assignment arrays without heap allocation
        // or large stack frames (macOS worker threads have 512 KiB stacks):
        // reuse dead `self.down_out` scratch (populated only later by DownProj)
        // whenever `down_out.len >= 3 * total_a`, falling back to small stack
        // scratch only for tiny synthetic workspaces.
        var stack_tok: [64 * MAX_TOP_K]u32 = undefined;
        var stack_rw: [64 * MAX_TOP_K]f32 = undefined;
        var stack_sk: [64 * MAX_TOP_K]u32 = undefined;

        var orig_tok: []u32 = undefined;
        var orig_rw: []f32 = undefined;
        var orig_sk: []u32 = undefined;
        if (self.down_out.len >= total_a * 3) {
            const u32_pool: []u32 = @ptrCast(self.down_out);
            orig_tok = u32_pool[0..total_a];
            orig_rw = self.down_out[total_a .. 2 * total_a];
            orig_sk = u32_pool[2 * total_a .. 3 * total_a];
        } else {
            std.debug.assert(total_a <= stack_tok.len);
            orig_tok = stack_tok[0..total_a];
            orig_rw = stack_rw[0..total_a];
            orig_sk = stack_sk[0..total_a];
        }

        @memcpy(orig_tok, self.token_indices[0..total_a]);
        @memcpy(orig_rw, self.routing_weights[0..total_a]);
        @memcpy(orig_sk, self.slot_k_indices[0..total_a]);

        var run_u: u32 = 0;
        var run_a: u32 = 0;
        for (0..NUM_SCHEME_BUCKETS) |s_idx| {
            const scheme: quant.WeightDType = @fromBackingInt(@intCast(s_idx));
            const u_base: usize = s_idx * self.bucket_unique_stride_u32;
            const o_base: usize = s_idx * self.bucket_offsets_stride_u32;
            var u_s: u32 = 0;
            var a_s: u32 = 0;
            for (0..num_u) |ui| {
                const eid = orig_u[ui];
                if (eid >= expert_dtypes.len or expert_dtypes[eid] != scheme) continue;
                const slot_start = run_a;
                self.bucket_unique_experts[u_base + u_s] = eid;
                self.bucket_expert_token_offsets[o_base + u_s] = slot_start;
                self.unique_experts[run_u] = eid;
                self.expert_token_offsets[run_u] = slot_start;
                const s0 = orig_off[ui];
                const s1 = orig_off[ui + 1];
                for (s0..s1) |ai| {
                    self.token_indices[run_a] = orig_tok[ai];
                    self.routing_weights[run_a] = orig_rw[ai];
                    self.slot_k_indices[run_a] = orig_sk[ai];
                    run_a += 1;
                    a_s += 1;
                }
                u_s += 1;
                run_u += 1;
            }
            self.bucket_expert_token_offsets[o_base + u_s] = run_a;
            self.bucketHeaderMut(scheme).* = .{
                .num_unique_experts = u_s,
                .total_assignments = a_s,
                .m = m_val,
                .top_k = top_k_val,
            };
            writeScheduleSlotIndirect(
                self.bucket_headers_raw.ptr + s_idx * BUCKET_HEADER_STRIDE,
                u_s,
                self.spec.intermediate_size,
                self.spec.hidden_size,
            );
        }
        self.expert_token_offsets[run_u] = run_a;
        self.schedule_header.num_unique_experts = run_u;
        self.schedule_header.total_assignments = run_a;
        writeScheduleSlotIndirect(
            self.schedule_header_raw.ptr,
            run_u,
            self.spec.intermediate_size,
            self.spec.hidden_size,
        );
    }
};

pub const RouterProjArgs = struct {
    x: metal.BufferBinding,
    router_weights: metal.BufferBinding,
    logits: metal.BufferBinding,
    m: u32,
    num_experts: u32,
    hidden_size: u32,
};

/// MLX affine group-64 quantized router projection (`logits` is `f32 [m, E]`).
pub const RouterProjQuantArgs = struct {
    x: metal.BufferBinding,
    qweight: metal.BufferBinding,
    scale_bias: metal.BufferBinding,
    logits: metal.BufferBinding,
    m: u32,
    num_experts: u32,
    hidden_size: u32,
};

pub const RouteTopKArgs = struct {
    logits: metal.BufferBinding,
    workspace: *const MoeWorkspace,
    workspace_buf: metal.Buffer,
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: bool = true,
    /// Optional layer dimensions for G20 indirect dispatch args (0 defaults to `workspace.spec`).
    hidden_size: u32 = 0,
    intermediate_size: u32 = 0,
};

/// Arguments for G13 scheme-bucketed routing (`ami_moe_route_topk_bucketed`).
pub const RouteTopKBucketedArgs = struct {
    logits: metal.BufferBinding,
    /// Device-resident `[num_experts]u8` mapping `expert_id -> @backingInt(quant.WeightDType)`.
    expert_schemes: metal.BufferBinding,
    workspace: *const MoeWorkspace,
    workspace_buf: metal.Buffer,
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: bool = true,
    /// Optional layer dimensions for G20 indirect dispatch args (0 defaults to `workspace.spec`).
    hidden_size: u32 = 0,
    intermediate_size: u32 = 0,
};

pub const GateUpSwigluArgs = struct {
    x: metal.BufferBinding,
    weights_base: metal.BufferBinding,
    expert_table: metal.BufferBinding,
    workspace: *const MoeWorkspace,
    workspace_buf: metal.Buffer,
    m: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    /// When set, reads the per-scheme CSR bucket from `workspace` (G13, ADR-0017).
    scheme_bucket: ?quant.WeightDType = null,
    /// Optional tighter grid bound for a scheme bucket (defaults to `num_experts`).
    max_active_override: ?u32 = null,
    /// Optional G20 indirect dispatch override; when null, applies `shouldUseIndirectDispatch`.
    use_indirect: ?bool = null,
    /// Optional SwiGLU clamp (`0.0` = disabled; `> 0.0` = `silu(min(gate, limit)) * clamp(up, -limit, limit)`).
    swiglu_limit: f32 = 0.0,
};

pub const DownProjArgs = struct {
    weights_base: metal.BufferBinding,
    expert_table: metal.BufferBinding,
    workspace: *const MoeWorkspace,
    workspace_buf: metal.Buffer,
    m: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    /// When set, reads the per-scheme CSR bucket from `workspace` (G13, ADR-0017).
    scheme_bucket: ?quant.WeightDType = null,
    /// Optional tighter grid bound for a scheme bucket (defaults to `num_experts`).
    max_active_override: ?u32 = null,
    /// Optional G20 indirect dispatch override; when null, applies `shouldUseIndirectDispatch`.
    use_indirect: ?bool = null,
};

pub const ReduceArgs = struct {
    workspace: *const MoeWorkspace,
    workspace_buf: metal.Buffer,
    residual: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    m: u32,
    hidden_size: u32,
    top_k: u32,
};

pub const RouterSource = union(enum) {
    weights: struct {
        binding: metal.BufferBinding,
        dtype: RouterWeightDType,
    },
    logits: struct {
        binding: metal.BufferBinding,
        dtype: LogitsDType,
    },
    /// MLX affine 3-bit / 4-bit group-64 router weights `[num_experts, hidden]`
    /// (e.g. real `Qwen3-30B-A3B-3bit` `mlp.gate`), producing `f32` logits in
    /// the workspace.
    quant_weights: struct {
        qweight: metal.BufferBinding,
        scale_bias: metal.BufferBinding,
        bits: quant.QuantBits,
    },
    /// Any `W21`/`W22` quantized router weight (`q2..q8`, `fp8`, `mxfp8`, `nvfp4`,
    /// or GGUF block format), writing `f16` logits into the workspace.
    quant_format: struct {
        dtype: quant_formats.DType,
        group_size: u16 = 64,
        tensor_scale_per_row: bool = false,
        qweight: metal.BufferBinding,
        params: ?metal.BufferBinding = null,
        qf_pipes: *const quant_formats.QuantFormatPipelines,
    },
};

/// Descriptor for one MoE expert quantization scheme bucket (uniform or
/// heterogeneous, supporting asymmetric `gate/up` vs `down` dtypes as seen in
/// dynamic GGUFs such as `Inkling-2-Mini` `UD-*`).
pub const MoeSchemeDesc = struct {
    gate_up_dtype: amf.DType = .q4_affine,
    gate_up_group_size: u16 = 64,
    gate_up_tensor_scale_per_row: bool = false,
    down_dtype: amf.DType = .q4_affine,
    down_group_size: u16 = 64,
    down_tensor_scale_per_row: bool = false,

    pub fn uniform(dtype: amf.DType, group_size: u16, tensor_scale_per_row: bool) MoeSchemeDesc {
        return .{
            .gate_up_dtype = dtype,
            .gate_up_group_size = group_size,
            .gate_up_tensor_scale_per_row = tensor_scale_per_row,
            .down_dtype = dtype,
            .down_group_size = group_size,
            .down_tensor_scale_per_row = tensor_scale_per_row,
        };
    }

    pub fn eql(a: MoeSchemeDesc, b: MoeSchemeDesc) bool {
        return a.gate_up_dtype == b.gate_up_dtype and
            a.gate_up_group_size == b.gate_up_group_size and
            a.gate_up_tensor_scale_per_row == b.gate_up_tensor_scale_per_row and
            a.down_dtype == b.down_dtype and
            a.down_group_size == b.down_group_size and
            a.down_tensor_scale_per_row == b.down_tensor_scale_per_row;
    }

    pub fn usesQuantFormatsGateUp(self: MoeSchemeDesc) bool {
        return dequant.usesQuantFormats(self.gate_up_dtype, self.gate_up_group_size);
    }

    pub fn usesQuantFormatsDown(self: MoeSchemeDesc) bool {
        return dequant.usesQuantFormats(self.down_dtype, self.down_group_size);
    }

    pub fn usesAnyQuantFormats(self: MoeSchemeDesc) bool {
        return self.usesQuantFormatsGateUp() or self.usesQuantFormatsDown();
    }
};

/// Multi-format MoE layer dispatch arguments covering all `W21`/`W22` `.amf`
/// dtypes (`0..37`), optional heterogeneous scheme buckets (`0..NUM_SCHEME_BUCKETS-1`),
/// and optional `G24` activation quantization emulation.
pub const MoELayerMultiFormatArgs = struct {
    base: MoELayerArgs,
    /// When `true`, routes via `ami_moe_route_topk_bucketed` using `expert_schemes`
    /// and `scheme_counts[0..NUM_SCHEME_BUCKETS]` mapped to `scheme_descs`.
    is_heterogeneous: bool = false,
    expert_schemes: ?metal.BufferBinding = null,
    scheme_counts: [NUM_SCHEME_BUCKETS]u32 = .{ 0, 0, 0, 0, 0 },
    /// Per-bucket descriptors (`scheme_descs[0]` is used when `is_heterogeneous == false`).
    scheme_descs: [NUM_SCHEME_BUCKETS]MoeSchemeDesc = .{
        .{}, .{}, .{}, .{}, .{},
    },
    qf_pipes: ?*const quant_formats.QuantFormatPipelines = null,
    act_quant_policy: quant_formats.ActQuantPolicy = .none,
    act_scratch: ?metal.BufferBinding = null,
    act_scratch_bytes: u64 = 0,
    shared_expert_amf_dtype: ?amf.DType = null,
    shared_expert_group_size: u16 = 64,
    shared_expert_tensor_scale_per_row: bool = false,
};

pub const MoELayerArgs = struct {
    x: metal.BufferBinding,
    router: RouterSource,
    weights_base: metal.BufferBinding,
    expert_table: metal.BufferBinding,
    workspace: *const MoeWorkspace,
    workspace_buf: metal.Buffer,
    /// Optional residual `[m, hidden_size]` added into `out` during reduction
    /// (may alias `out` for in-place residual addition).
    residual: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    m: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    norm_topk_prob: bool = true,
    /// Optional unconditional shared expert (G7, ADR-0015). Offsets are
    /// relative to `shared_weights_base orelse weights_base`.
    shared_expert: ?ExpertWeights = null,
    /// Optional separate buffer holding `shared_expert` weights (defaults to
    /// `weights_base` when null).
    shared_weights_base: ?metal.BufferBinding = null,
    /// Storage dtype of `shared_expert` weights. When null, inherits the
    /// layer's weight dtype (`encodeMoELayer` / `encodeMoELayerFloat`) or
    /// defaults to `.q4_affine` in `encodeMoELayerHetero`.
    shared_expert_dtype: ?quant.WeightDType = null,
    /// Intermediate width of `shared_expert` SwiGLU. When 0, defaults to
    /// `intermediate_size`.
    shared_intermediate_size: u32 = 0,
    /// Optional `[hidden_size]` scalar sigmoid gate weights (`sigmoid(x · w)`)
    /// applied per token to the shared expert output before accumulating into `out`.
    shared_gate_weight: ?metal.BufferBinding = null,
    /// Storage dtype of `shared_gate_weight` (`.f16`, `.bf16`, or `.f32`).
    shared_gate_dtype: RouterWeightDType = .f16,
    /// Optional SwiGLU clamp (`0.0` = disabled; `> 0.0` = `silu(min(gate, limit)) * clamp(up, -limit, limit)`).
    swiglu_limit: f32 = 0.0,
};

/// Alias spelling used by M2.5 integration code.
pub const MoeLayerArgs = MoELayerArgs;

/// Arguments for G13 heterogeneous per-expert scheme-bucketed MoE FFN execution (ADR-0017).
pub const MoELayerHeteroArgs = struct {
    base: MoELayerArgs,
    /// Device-resident `[num_experts]u8` mapping `expert_id -> @backingInt(quant.WeightDType)`.
    expert_schemes: metal.BufferBinding,
    /// Static expert counts per `WeightDType` bucket in this layer (`> 0` launches that bucket).
    scheme_counts: [NUM_SCHEME_BUCKETS]u32,
};

/// Token ID source for `encodeEmbedLookup`.
pub const EmbedTokens = union(enum) {
    /// Single token ID streamed through the uniform ring (decode, `m = 1`).
    token_id: u32,
    /// GPU-resident `u32 [m]` token-ID buffer (prefill / batched lookup).
    token_ids: metal.BufferBinding,
};

/// GPU embedding lookup writing `f16 [m, hidden_size]` into `out`.
pub const EmbedLookupArgs = struct {
    /// `[vocab_size, hidden_size]` table (`f16` / `bf16`), or packed MLX
    /// affine group-64 codes (`q4_affine` / `q3_affine`).
    weight: metal.BufferBinding,
    /// `half2(scale, bias)` group parameters; required for quantized tables.
    scale_bias: ?metal.BufferBinding = null,
    tokens: EmbedTokens,
    out: metal.BufferBinding,
    m: u32 = 1,
    vocab_size: u32,
    hidden_size: u32,
};

pub const SiluMulArgs = struct {
    gate: metal.BufferBinding,
    up: metal.BufferBinding,
    out: metal.BufferBinding,
    count: u32,
    swiglu_limit: f32 = 0.0,
};

pub const ResidualAddArgs = struct {
    x: metal.BufferBinding,
    delta: metal.BufferBinding,
    count: u32,
};

pub const ArgmaxArgs = struct {
    logits: metal.BufferBinding,
    out_token_id: metal.BufferBinding,
    vocab_size: u32,
};

/// Batched `[num_rows, vocab_size]` greedy argmax arguments (`1 <= num_rows <= 64`).
pub const ArgmaxRowsArgs = struct {
    logits: metal.BufferBinding,
    out_token_ids: metal.BufferBinding,
    num_rows: u32,
    vocab_size: u32,
    logits_dtype: LogitsDType = .f16,
};

/// Batched `[num_rows, vocab_size]` fused FP32 softmax + top-k extraction
/// arguments (`1 <= num_rows <= 64`, `1 <= top_k <= 16`).
pub const TopKLogitsRowsArgs = struct {
    logits: metal.BufferBinding,
    /// `[num_rows, top_k]` `TopKCandidate` output buffer in wired shared RAM.
    out_candidates: metal.BufferBinding,
    num_rows: u32,
    vocab_size: u32,
    top_k: u32,
    logits_dtype: LogitsDType = .f16,
};

/// Strided 2D `f16` copy arguments (`ami_copy_strided_f16`).
pub const CopyStridedArgs = struct {
    src: metal.BufferBinding,
    dst: metal.BufferBinding,
    num_rows: u32,
    copy_cols: u32,
    src_stride: u32,
    src_col_off: u32 = 0,
    dst_stride: u32,
    dst_col_off: u32 = 0,
};

/// Indexed `f16` row gather arguments (`ami_gather_rows_f16`).
pub const GatherRowsArgs = struct {
    src: metal.BufferBinding,
    indices: metal.BufferBinding,
    dst: metal.BufferBinding,
    count: u32,
    row_elems: u32,
};

/// Startup-compiled specialized Metal 4 pipelines for MoE routing,
/// deduplicated 2-bit / 3-bit / 4-bit / float expert SwiGLU FFN, G13
/// scheme-bucketed heterogeneous routing, dense SwiGLU utilities,
/// and GPU greedy argmax.
pub const MoePipelines = struct {
    router_proj_f16: metal.Pipeline,
    router_proj_f32: metal.Pipeline,
    route_topk_f32_k8: metal.Pipeline,
    route_topk_f16_k8: metal.Pipeline,
    route_topk_f32_dyn: metal.Pipeline,
    route_topk_f16_dyn: metal.Pipeline,
    route_topk_bucketed_f32_k8: metal.Pipeline,
    route_topk_bucketed_f16_k8: metal.Pipeline,
    route_topk_bucketed_f32_dyn: metal.Pipeline,
    route_topk_bucketed_f16_dyn: metal.Pipeline,
    gate_up_swiglu_q4: metal.Pipeline,
    gate_up_swiglu_q3: metal.Pipeline,
    gate_up_swiglu_q2: metal.Pipeline,
    down_proj_q4: metal.Pipeline,
    down_proj_q3: metal.Pipeline,
    down_proj_q2: metal.Pipeline,
    reduce_f16_k8: metal.Pipeline,
    reduce_f16_dyn: metal.Pipeline,
    silu_mul_f16: metal.Pipeline,
    residual_add_f16: metal.Pipeline,
    argmax_f16: metal.Pipeline,
    router_proj_bf16: metal.Pipeline,
    router_proj_q4: metal.Pipeline,
    router_proj_q3: metal.Pipeline,
    router_proj_q2: metal.Pipeline,
    gate_up_swiglu_f16: metal.Pipeline,
    gate_up_swiglu_bf16: metal.Pipeline,
    down_proj_f16: metal.Pipeline,
    down_proj_bf16: metal.Pipeline,
    embed_f16: metal.Pipeline,
    embed_bf16: metal.Pipeline,
    embed_q4: metal.Pipeline,
    embed_q3: metal.Pipeline,
    embed_q2: metal.Pipeline,
    argmax_rows_f16: metal.Pipeline,
    argmax_rows_f32: metal.Pipeline,
    topk_logits_rows_f16: metal.Pipeline,
    topk_logits_rows_f32: metal.Pipeline,
    gate_up_swiglu_perslot_q4: metal.Pipeline,
    gate_up_swiglu_perslot_q3: metal.Pipeline,
    gate_up_swiglu_perslot_q2: metal.Pipeline,
    down_proj_perslot_q4: metal.Pipeline,
    down_proj_perslot_q3: metal.Pipeline,
    down_proj_perslot_q2: metal.Pipeline,
    copy_strided_f16: metal.Pipeline,
    gather_rows_f16: metal.Pipeline,
    // G7 shared expert pipelines (scalar sigmoid gate + GateUpSwiGLU + DownAccum).
    shared_gate_f16: metal.Pipeline,
    shared_gate_f32: metal.Pipeline,
    shared_gate_bf16: metal.Pipeline,
    shared_gate_up_q4: metal.Pipeline,
    shared_gate_up_q3: metal.Pipeline,
    shared_gate_up_q2: metal.Pipeline,
    shared_gate_up_f16: metal.Pipeline,
    shared_gate_up_bf16: metal.Pipeline,
    shared_down_accum_q4: metal.Pipeline,
    shared_down_accum_q3: metal.Pipeline,
    shared_down_accum_q2: metal.Pipeline,
    shared_down_accum_f16: metal.Pipeline,
    shared_down_accum_bf16: metal.Pipeline,

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!MoePipelines {
        const router_proj_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_router_proj", &.{
            .{ .index = 4, .value = .{ .bool = true } },
            .{ .index = 5, .value = .{ .bool = false } },
        });
        errdefer router_proj_f16.close();

        const router_proj_f32 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_router_proj", &.{
            .{ .index = 4, .value = .{ .bool = false } },
            .{ .index = 5, .value = .{ .bool = false } },
        });
        errdefer router_proj_f32.close();

        const route_topk_f32_k8 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_route_topk", &.{
            .{ .index = 2, .value = .{ .u32 = 8 } },
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer route_topk_f32_k8.close();

        const route_topk_f16_k8 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_route_topk", &.{
            .{ .index = 2, .value = .{ .u32 = 8 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer route_topk_f16_k8.close();

        const route_topk_f32_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_moe_route_topk", &.{
            .{ .index = 2, .value = .{ .u32 = 0 } },
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer route_topk_f32_dyn.close();

        const route_topk_f16_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_moe_route_topk", &.{
            .{ .index = 2, .value = .{ .u32 = 0 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer route_topk_f16_dyn.close();

        const route_topk_bucketed_f32_k8 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_route_topk_bucketed", &.{
            .{ .index = 2, .value = .{ .u32 = 8 } },
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer route_topk_bucketed_f32_k8.close();

        const route_topk_bucketed_f16_k8 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_route_topk_bucketed", &.{
            .{ .index = 2, .value = .{ .u32 = 8 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer route_topk_bucketed_f16_k8.close();

        const route_topk_bucketed_f32_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_moe_route_topk_bucketed", &.{
            .{ .index = 2, .value = .{ .u32 = 0 } },
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer route_topk_bucketed_f32_dyn.close();

        const route_topk_bucketed_f16_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_moe_route_topk_bucketed", &.{
            .{ .index = 2, .value = .{ .u32 = 0 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer route_topk_bucketed_f16_dyn.close();

        const gate_up_swiglu_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_gate_up_swiglu", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gate_up_swiglu_q4.close();

        const gate_up_swiglu_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_gate_up_swiglu", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gate_up_swiglu_q3.close();

        const gate_up_swiglu_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_gate_up_swiglu", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gate_up_swiglu_q2.close();

        const down_proj_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_down_proj", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer down_proj_q4.close();

        const down_proj_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_down_proj", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer down_proj_q3.close();

        const down_proj_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_down_proj", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer down_proj_q2.close();

        const reduce_f16_k8 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_reduce_f16", &.{
            .{ .index = 2, .value = .{ .u32 = 8 } },
        });
        errdefer reduce_f16_k8.close();

        const reduce_f16_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_moe_reduce_f16", &.{
            .{ .index = 2, .value = .{ .u32 = 0 } },
        });
        errdefer reduce_f16_dyn.close();

        const silu_mul_f16 = try metal.Pipeline.init(device, library, "ami_silu_mul_f16");
        errdefer silu_mul_f16.close();

        const residual_add_f16 = try metal.Pipeline.init(device, library, "ami_residual_add_f16");
        errdefer residual_add_f16.close();

        const argmax_f16 = try metal.Pipeline.init(device, library, "ami_argmax_f16");
        errdefer argmax_f16.close();

        const router_proj_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_router_proj", &.{
            .{ .index = 4, .value = .{ .bool = false } },
            .{ .index = 5, .value = .{ .bool = true } },
        });
        errdefer router_proj_bf16.close();

        const router_proj_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_router_proj_q", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer router_proj_q4.close();

        const router_proj_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_router_proj_q", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer router_proj_q3.close();

        const router_proj_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_router_proj_q", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer router_proj_q2.close();

        const gate_up_swiglu_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_gate_up_swiglu_float", &.{
            .{ .index = 6, .value = .{ .bool = false } },
        });
        errdefer gate_up_swiglu_f16.close();

        const gate_up_swiglu_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_gate_up_swiglu_float", &.{
            .{ .index = 6, .value = .{ .bool = true } },
        });
        errdefer gate_up_swiglu_bf16.close();

        const down_proj_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_down_proj_float", &.{
            .{ .index = 6, .value = .{ .bool = false } },
        });
        errdefer down_proj_f16.close();

        const down_proj_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_down_proj_float", &.{
            .{ .index = 6, .value = .{ .bool = true } },
        });
        errdefer down_proj_bf16.close();

        const embed_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_embed_lookup", &.{
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 7, .value = .{ .u32 = @backingInt(quant.WeightDType.f16) } },
        });
        errdefer embed_f16.close();

        const embed_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_embed_lookup", &.{
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 7, .value = .{ .u32 = @backingInt(quant.WeightDType.bf16) } },
        });
        errdefer embed_bf16.close();

        const embed_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_embed_lookup", &.{
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 7, .value = .{ .u32 = @backingInt(quant.WeightDType.q4_affine) } },
        });
        errdefer embed_q4.close();

        const embed_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_embed_lookup", &.{
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 7, .value = .{ .u32 = @backingInt(quant.WeightDType.q3_affine) } },
        });
        errdefer embed_q3.close();

        const embed_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_embed_lookup", &.{
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 7, .value = .{ .u32 = @backingInt(quant.WeightDType.q2_affine) } },
        });
        errdefer embed_q2.close();

        const argmax_rows_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_argmax_rows", &.{
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer argmax_rows_f16.close();

        const argmax_rows_f32 = try metal.Pipeline.initSpecialized(device, library, "ami_argmax_rows", &.{
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer argmax_rows_f32.close();

        const topk_logits_rows_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_topk_logits_rows", &.{
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer topk_logits_rows_f16.close();

        const topk_logits_rows_f32 = try metal.Pipeline.initSpecialized(device, library, "ami_topk_logits_rows", &.{
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer topk_logits_rows_f32.close();

        const gate_up_swiglu_perslot_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_gate_up_swiglu_perslot", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gate_up_swiglu_perslot_q4.close();

        const gate_up_swiglu_perslot_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_gate_up_swiglu_perslot", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gate_up_swiglu_perslot_q3.close();

        const gate_up_swiglu_perslot_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_gate_up_swiglu_perslot", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gate_up_swiglu_perslot_q2.close();

        const down_proj_perslot_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_down_proj_perslot", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer down_proj_perslot_q4.close();

        const down_proj_perslot_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_down_proj_perslot", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer down_proj_perslot_q3.close();

        const down_proj_perslot_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_down_proj_perslot", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer down_proj_perslot_q2.close();

        const copy_strided_f16 = try metal.Pipeline.init(device, library, "ami_copy_strided_f16");
        errdefer copy_strided_f16.close();

        const gather_rows_f16 = try metal.Pipeline.init(device, library, "ami_gather_rows_f16");
        errdefer gather_rows_f16.close();

        const shared_gate_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_gate", &.{
            .{ .index = 4, .value = .{ .bool = true } },
            .{ .index = 5, .value = .{ .bool = false } },
        });
        errdefer shared_gate_f16.close();

        const shared_gate_f32 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_gate", &.{
            .{ .index = 4, .value = .{ .bool = false } },
            .{ .index = 5, .value = .{ .bool = false } },
        });
        errdefer shared_gate_f32.close();

        const shared_gate_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_gate", &.{
            .{ .index = 4, .value = .{ .bool = false } },
            .{ .index = 5, .value = .{ .bool = true } },
        });
        errdefer shared_gate_bf16.close();

        const shared_gate_up_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_gate_up_swiglu", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer shared_gate_up_q4.close();

        const shared_gate_up_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_gate_up_swiglu", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer shared_gate_up_q3.close();

        const shared_gate_up_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_gate_up_swiglu", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer shared_gate_up_q2.close();

        const shared_gate_up_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_gate_up_swiglu_float", &.{
            .{ .index = 6, .value = .{ .bool = false } },
        });
        errdefer shared_gate_up_f16.close();

        const shared_gate_up_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_gate_up_swiglu_float", &.{
            .{ .index = 6, .value = .{ .bool = true } },
        });
        errdefer shared_gate_up_bf16.close();

        const shared_down_accum_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_down_accum", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer shared_down_accum_q4.close();

        const shared_down_accum_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_down_accum", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer shared_down_accum_q3.close();

        const shared_down_accum_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_down_accum", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer shared_down_accum_q2.close();

        const shared_down_accum_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_down_accum_float", &.{
            .{ .index = 6, .value = .{ .bool = false } },
        });
        errdefer shared_down_accum_f16.close();

        const shared_down_accum_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_shared_down_accum_float", &.{
            .{ .index = 6, .value = .{ .bool = true } },
        });
        errdefer shared_down_accum_bf16.close();

        return .{
            .router_proj_f16 = router_proj_f16,
            .router_proj_f32 = router_proj_f32,
            .route_topk_f32_k8 = route_topk_f32_k8,
            .route_topk_f16_k8 = route_topk_f16_k8,
            .route_topk_f32_dyn = route_topk_f32_dyn,
            .route_topk_f16_dyn = route_topk_f16_dyn,
            .route_topk_bucketed_f32_k8 = route_topk_bucketed_f32_k8,
            .route_topk_bucketed_f16_k8 = route_topk_bucketed_f16_k8,
            .route_topk_bucketed_f32_dyn = route_topk_bucketed_f32_dyn,
            .route_topk_bucketed_f16_dyn = route_topk_bucketed_f16_dyn,
            .gate_up_swiglu_q4 = gate_up_swiglu_q4,
            .gate_up_swiglu_q3 = gate_up_swiglu_q3,
            .gate_up_swiglu_q2 = gate_up_swiglu_q2,
            .down_proj_q4 = down_proj_q4,
            .down_proj_q3 = down_proj_q3,
            .down_proj_q2 = down_proj_q2,
            .reduce_f16_k8 = reduce_f16_k8,
            .reduce_f16_dyn = reduce_f16_dyn,
            .silu_mul_f16 = silu_mul_f16,
            .residual_add_f16 = residual_add_f16,
            .argmax_f16 = argmax_f16,
            .router_proj_bf16 = router_proj_bf16,
            .router_proj_q4 = router_proj_q4,
            .router_proj_q3 = router_proj_q3,
            .router_proj_q2 = router_proj_q2,
            .gate_up_swiglu_f16 = gate_up_swiglu_f16,
            .gate_up_swiglu_bf16 = gate_up_swiglu_bf16,
            .down_proj_f16 = down_proj_f16,
            .down_proj_bf16 = down_proj_bf16,
            .embed_f16 = embed_f16,
            .embed_bf16 = embed_bf16,
            .embed_q4 = embed_q4,
            .embed_q3 = embed_q3,
            .embed_q2 = embed_q2,
            .argmax_rows_f16 = argmax_rows_f16,
            .argmax_rows_f32 = argmax_rows_f32,
            .topk_logits_rows_f16 = topk_logits_rows_f16,
            .topk_logits_rows_f32 = topk_logits_rows_f32,
            .gate_up_swiglu_perslot_q4 = gate_up_swiglu_perslot_q4,
            .gate_up_swiglu_perslot_q3 = gate_up_swiglu_perslot_q3,
            .gate_up_swiglu_perslot_q2 = gate_up_swiglu_perslot_q2,
            .down_proj_perslot_q4 = down_proj_perslot_q4,
            .down_proj_perslot_q3 = down_proj_perslot_q3,
            .down_proj_perslot_q2 = down_proj_perslot_q2,
            .copy_strided_f16 = copy_strided_f16,
            .gather_rows_f16 = gather_rows_f16,
            .shared_gate_f16 = shared_gate_f16,
            .shared_gate_f32 = shared_gate_f32,
            .shared_gate_bf16 = shared_gate_bf16,
            .shared_gate_up_q4 = shared_gate_up_q4,
            .shared_gate_up_q3 = shared_gate_up_q3,
            .shared_gate_up_q2 = shared_gate_up_q2,
            .shared_gate_up_f16 = shared_gate_up_f16,
            .shared_gate_up_bf16 = shared_gate_up_bf16,
            .shared_down_accum_q4 = shared_down_accum_q4,
            .shared_down_accum_q3 = shared_down_accum_q3,
            .shared_down_accum_q2 = shared_down_accum_q2,
            .shared_down_accum_f16 = shared_down_accum_f16,
            .shared_down_accum_bf16 = shared_down_accum_bf16,
        };
    }

    pub fn close(self: MoePipelines) void {
        self.shared_down_accum_bf16.close();
        self.shared_down_accum_f16.close();
        self.shared_down_accum_q2.close();
        self.shared_down_accum_q3.close();
        self.shared_down_accum_q4.close();
        self.shared_gate_up_bf16.close();
        self.shared_gate_up_f16.close();
        self.shared_gate_up_q2.close();
        self.shared_gate_up_q3.close();
        self.shared_gate_up_q4.close();
        self.shared_gate_bf16.close();
        self.shared_gate_f32.close();
        self.shared_gate_f16.close();
        self.gather_rows_f16.close();
        self.copy_strided_f16.close();
        self.down_proj_perslot_q2.close();
        self.down_proj_perslot_q3.close();
        self.down_proj_perslot_q4.close();
        self.gate_up_swiglu_perslot_q2.close();
        self.gate_up_swiglu_perslot_q3.close();
        self.gate_up_swiglu_perslot_q4.close();
        self.topk_logits_rows_f32.close();
        self.topk_logits_rows_f16.close();
        self.argmax_rows_f32.close();
        self.argmax_rows_f16.close();
        self.embed_q2.close();
        self.embed_q3.close();
        self.embed_q4.close();
        self.embed_bf16.close();
        self.embed_f16.close();
        self.down_proj_bf16.close();
        self.down_proj_f16.close();
        self.gate_up_swiglu_bf16.close();
        self.gate_up_swiglu_f16.close();
        self.router_proj_q2.close();
        self.router_proj_q3.close();
        self.router_proj_q4.close();
        self.router_proj_bf16.close();
        self.argmax_f16.close();
        self.residual_add_f16.close();
        self.silu_mul_f16.close();
        self.reduce_f16_dyn.close();
        self.reduce_f16_k8.close();
        self.down_proj_q2.close();
        self.down_proj_q3.close();
        self.down_proj_q4.close();
        self.gate_up_swiglu_q2.close();
        self.gate_up_swiglu_q3.close();
        self.gate_up_swiglu_q4.close();
        self.route_topk_bucketed_f16_dyn.close();
        self.route_topk_bucketed_f32_dyn.close();
        self.route_topk_bucketed_f16_k8.close();
        self.route_topk_bucketed_f32_k8.close();
        self.route_topk_f16_dyn.close();
        self.route_topk_f32_dyn.close();
        self.route_topk_f16_k8.close();
        self.route_topk_f32_k8.close();
        self.router_proj_f32.close();
        self.router_proj_f16.close();
    }

    pub fn encodeRouterProj(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        weight_dtype: RouterWeightDType,
        args: RouterProjArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.hidden_size == 0) return error.ZeroDimension;
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExpertCount;

        const pipe = switch (weight_dtype) {
            .f16 => self.router_proj_f16,
            .f32 => self.router_proj_f32,
            .bf16 => self.router_proj_bf16,
        };
        const params: RouterProjParams = .{
            .m = args.m,
            .num_experts = args.num_experts,
            .hidden_size = args.hidden_size,
        };
        const groups_x: u64 = (@as(u64, args.num_experts) + 3) / 4;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.router_weights, args.logits },
            std.mem.asBytes(&params),
            .{ groups_x, args.m, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeRouteTopK(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        logits_dtype: LogitsDType,
        args: RouteTopKArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0) return error.ZeroDimension;
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExpertCount;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;
        if (args.m > args.workspace.spec.max_tokens or
            args.num_experts > args.workspace.spec.num_experts or
            args.top_k > args.workspace.spec.top_k)
        {
            return error.WorkspaceTooSmall;
        }

        const pipe = switch (logits_dtype) {
            .f32 => if (args.top_k == 8) self.route_topk_f32_k8 else self.route_topk_f32_dyn,
            .f16 => if (args.top_k == 8) self.route_topk_f16_k8 else self.route_topk_f16_dyn,
        };
        const ws = args.workspace;
        const wb = args.workspace_buf;
        const eff_inter = if (args.intermediate_size > 0) args.intermediate_size else ws.spec.intermediate_size;
        const eff_hidden = if (args.hidden_size > 0) args.hidden_size else ws.spec.hidden_size;
        const params: RouteParams = .{
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .norm_topk_prob = if (args.norm_topk_prob) 1 else 0,
            .gate_up_groups_x_q = gateUpGroupsXQ(eff_inter),
            .gate_up_groups_x_f = gateUpGroupsXF(eff_inter),
            .down_groups_x = downGroupsX(eff_hidden),
            .down_groups_x_q8 = downGroupsXQ8(eff_hidden),
        };
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.logits,
                .{ .buffer = wb, .offset = ws.schedule_header_off },
                .{ .buffer = wb, .offset = ws.unique_experts_off },
                .{ .buffer = wb, .offset = ws.expert_token_offsets_off },
                .{ .buffer = wb, .offset = ws.token_indices_off },
                .{ .buffer = wb, .offset = ws.routing_weights_off },
                .{ .buffer = wb, .offset = ws.slot_k_indices_off },
                .{ .buffer = wb, .offset = ws.scratch_topk_experts_off },
                .{ .buffer = wb, .offset = ws.scratch_topk_weights_off },
            },
            std.mem.asBytes(&params),
            .{ 1, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes G13 scheme-bucketed softmax + top-K + disjoint per-scheme CSR
    /// routing (`ami_moe_route_topk_bucketed`, ADR-0017). Zero heap allocation
    /// and zero CPU-GPU sync: populates both the unified CSR schedule and the
    /// 5 per-`WeightDType` scheme buckets in preallocated `workspace`.
    pub fn encodeRouteTopKBucketed(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        logits_dtype: LogitsDType,
        args: RouteTopKBucketedArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0) return error.ZeroDimension;
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExpertCount;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;
        if (args.m > args.workspace.spec.max_tokens or
            args.num_experts > args.workspace.spec.num_experts or
            args.top_k > args.workspace.spec.top_k)
        {
            return error.WorkspaceTooSmall;
        }

        const pipe = switch (logits_dtype) {
            .f32 => if (args.top_k == 8) self.route_topk_bucketed_f32_k8 else self.route_topk_bucketed_f32_dyn,
            .f16 => if (args.top_k == 8) self.route_topk_bucketed_f16_k8 else self.route_topk_bucketed_f16_dyn,
        };
        const ws = args.workspace;
        const wb = args.workspace_buf;
        const eff_inter = if (args.intermediate_size > 0) args.intermediate_size else ws.spec.intermediate_size;
        const eff_hidden = if (args.hidden_size > 0) args.hidden_size else ws.spec.hidden_size;
        const params: BucketRouteParams = .{
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .norm_topk_prob = if (args.norm_topk_prob) 1 else 0,
            .bucket_unique_stride = ws.bucket_unique_stride_u32,
            .bucket_offsets_stride = ws.bucket_offsets_stride_u32,
            .gate_up_groups_x_q = gateUpGroupsXQ(eff_inter),
            .gate_up_groups_x_f = gateUpGroupsXF(eff_inter),
            .down_groups_x = downGroupsX(eff_hidden),
            .down_groups_x_q8 = downGroupsXQ8(eff_hidden),
        };
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.logits,
                .{ .buffer = wb, .offset = ws.schedule_header_off },
                .{ .buffer = wb, .offset = ws.unique_experts_off },
                .{ .buffer = wb, .offset = ws.expert_token_offsets_off },
                .{ .buffer = wb, .offset = ws.token_indices_off },
                .{ .buffer = wb, .offset = ws.routing_weights_off },
                .{ .buffer = wb, .offset = ws.slot_k_indices_off },
                .{ .buffer = wb, .offset = ws.scratch_topk_experts_off },
                .{ .buffer = wb, .offset = ws.scratch_topk_weights_off },
                args.expert_schemes,
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

    pub fn encodeGateUpSwiglu(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: GateUpSwigluArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0 or
            args.hidden_size == 0 or args.intermediate_size == 0)
        {
            return error.ZeroDimension;
        }
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExpertCount;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;
        if (args.m > args.workspace.spec.max_tokens or
            args.num_experts > args.workspace.spec.num_experts or
            args.top_k > args.workspace.spec.top_k or
            args.intermediate_size > args.workspace.spec.intermediate_size)
        {
            return error.WorkspaceTooSmall;
        }

        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.gate_up_swiglu_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.gate_up_swiglu_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.gate_up_swiglu_q2 };
            },
        };

        const params: MoeGateUpParams = .{
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
            .num_groups = @intCast(num_groups),
            .row_bytes = @intCast(row_bytes),
            .swiglu_limit = args.swiglu_limit,
        };
        const groups_x: u64 = (@as(u64, args.intermediate_size) + 7) / 8;
        const cap_active: u64 = if (args.max_active_override) |cap| @min(@as(u64, cap), args.num_experts) else args.num_experts;
        const max_active: u64 = @min(@as(u64, args.m) * args.top_k, cap_active);
        if (max_active == 0) return;
        const ws = args.workspace;
        const wb = args.workspace_buf;
        const hdr_off = if (args.scheme_bucket) |s| ws.bucketHeaderOffset(s) else ws.schedule_header_off;
        const unique_off = if (args.scheme_bucket) |s| ws.bucketUniqueOffset(s) else ws.unique_experts_off;
        const offsets_off = if (args.scheme_bucket) |s| ws.bucketOffsetsOffset(s) else ws.expert_token_offsets_off;
        const bindings = [_]metal.BufferBinding{
            args.x,
            args.weights_base,
            args.expert_table,
            .{ .buffer = wb, .offset = hdr_off },
            .{ .buffer = wb, .offset = unique_off },
            .{ .buffer = wb, .offset = offsets_off },
            .{ .buffer = wb, .offset = ws.token_indices_off },
            .{ .buffer = wb, .offset = ws.inter_off },
        };
        const use_indirect = args.use_indirect orelse
            shouldUseIndirectDispatch(args.m, if (args.scheme_bucket != null) 2 else 1);
        if (use_indirect) {
            try frame.encodeThreadgroupsIndirect(
                pipe,
                &bindings,
                std.mem.asBytes(&params),
                .{ .buffer = wb, .offset = ws.scheduleIndirectOffset(args.scheme_bucket, INDIRECT_GATE_UP_Q_OFFSET) },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
        } else {
            try frame.encodeThreadgroups(
                pipe,
                &bindings,
                std.mem.asBytes(&params),
                .{ groups_x, max_active, 1 },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
        }
    }

    pub fn encodeDownProj(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: DownProjArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0 or
            args.hidden_size == 0 or args.intermediate_size == 0)
        {
            return error.ZeroDimension;
        }
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExpertCount;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;
        if (args.m > args.workspace.spec.max_tokens or
            args.num_experts > args.workspace.spec.num_experts or
            args.top_k > args.workspace.spec.top_k or
            args.hidden_size > args.workspace.spec.hidden_size or
            args.intermediate_size > args.workspace.spec.intermediate_size)
        {
            return error.WorkspaceTooSmall;
        }

        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.intermediate_size), try Ops.rowWeightBytes(args.intermediate_size), self.down_proj_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.intermediate_size), try Ops.rowWeightBytes(args.intermediate_size), self.down_proj_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.intermediate_size), try Ops.rowWeightBytes(args.intermediate_size), self.down_proj_q2 };
            },
        };

        const params: MoeDownParams = .{
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
            .num_groups = @intCast(num_groups),
            .row_bytes = @intCast(row_bytes),
            .top_k = args.top_k,
        };
        const groups_x: u64 = (@as(u64, args.hidden_size) + 15) / 16;
        const cap_active: u64 = if (args.max_active_override) |cap| @min(@as(u64, cap), args.num_experts) else args.num_experts;
        const max_active: u64 = @min(@as(u64, args.m) * args.top_k, cap_active);
        if (max_active == 0) return;
        const ws = args.workspace;
        const wb = args.workspace_buf;
        const hdr_off = if (args.scheme_bucket) |s| ws.bucketHeaderOffset(s) else ws.schedule_header_off;
        const unique_off = if (args.scheme_bucket) |s| ws.bucketUniqueOffset(s) else ws.unique_experts_off;
        const offsets_off = if (args.scheme_bucket) |s| ws.bucketOffsetsOffset(s) else ws.expert_token_offsets_off;
        const bindings = [_]metal.BufferBinding{
            .{ .buffer = wb, .offset = ws.inter_off },
            args.weights_base,
            args.expert_table,
            .{ .buffer = wb, .offset = hdr_off },
            .{ .buffer = wb, .offset = unique_off },
            .{ .buffer = wb, .offset = offsets_off },
            .{ .buffer = wb, .offset = ws.token_indices_off },
            .{ .buffer = wb, .offset = ws.routing_weights_off },
            .{ .buffer = wb, .offset = ws.slot_k_indices_off },
            .{ .buffer = wb, .offset = ws.down_out_off },
        };
        const use_indirect = args.use_indirect orelse
            shouldUseIndirectDispatch(args.m, if (args.scheme_bucket != null) 2 else 1);
        if (use_indirect) {
            try frame.encodeThreadgroupsIndirect(
                pipe,
                &bindings,
                std.mem.asBytes(&params),
                .{ .buffer = wb, .offset = ws.scheduleIndirectOffset(args.scheme_bucket, INDIRECT_DOWN_OFFSET) },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
        } else {
            try frame.encodeThreadgroups(
                pipe,
                &bindings,
                std.mem.asBytes(&params),
                .{ groups_x, max_active, 1 },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
        }
    }

    pub fn encodeReduce(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: ReduceArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_size == 0 or args.top_k == 0) return error.ZeroDimension;
        if (args.top_k > MAX_TOP_K) return error.UnsupportedTopK;
        if (args.m > args.workspace.spec.max_tokens or
            args.hidden_size > args.workspace.spec.hidden_size or
            args.top_k > args.workspace.spec.top_k)
        {
            return error.WorkspaceTooSmall;
        }

        const total: u64 = @as(u64, args.m) * args.hidden_size;
        const groups_x: u64 = (total + 255) / 256;
        const pipe = if (args.top_k == 8) self.reduce_f16_k8 else self.reduce_f16_dyn;
        const params: MoeReduceParams = .{
            .m = args.m,
            .hidden_size = args.hidden_size,
            .top_k = args.top_k,
            .add_residual = if (args.residual != null) 1 else 0,
        };
        const resid_binding = args.residual orelse args.out;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                .{ .buffer = args.workspace_buf, .offset = args.workspace.down_out_off },
                resid_binding,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ groups_x, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes a complete deduplicated MoE FFN layer into `frame`:
    /// optional router projection -> `ami_moe_route_topk` ->
    /// `ami_moe_gate_up_swiglu` -> `ami_moe_down_proj` -> optional shared expert
    /// GateUp -> `ami_moe_reduce_f16` -> optional shared expert DownAccum.
    pub fn encodeMoELayer(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: MoELayerArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try self.encodeRouting(frame, args, insert_barrier_before);

        // M7.8 G20 M=1 static-grid guard: homogeneous M==1 stays on direct dispatch.
        const use_indirect = shouldUseIndirectDispatch(args.m, 1);

        try self.encodeGateUpSwiglu(frame, bits, .{
            .x = args.x,
            .weights_base = args.weights_base,
            .expert_table = args.expert_table,
            .workspace = args.workspace,
            .workspace_buf = args.workspace_buf,
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
            .use_indirect = use_indirect,
            .swiglu_limit = args.swiglu_limit,
        }, true);

        try self.encodeDownProj(frame, bits, .{
            .weights_base = args.weights_base,
            .expert_table = args.expert_table,
            .workspace = args.workspace,
            .workspace_buf = args.workspace_buf,
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
            .use_indirect = use_indirect,
        }, true);

        const default_shared_dt = quant.WeightDType.fromQuant(bits);
        try self.encodeReduceAndOptionalShared(frame, args, default_shared_dt);
    }

    /// Encodes the optional shared-expert pre-reduce stages (`ami_moe_shared_gate`,
    /// `ami_moe_shared_gate_up_swiglu*`), then `ami_moe_reduce_f16`, then
    /// `ami_moe_shared_down_accum*` accumulating into `args.out`.
    /// Running shared Gate/GateUp BEFORE `encodeReduce` keeps `args.x` untouched
    /// even if `args.x` aliases `args.out`, and safely reuses `workspace.inter`
    /// after routed `DownProj` when `workspace.spec.shared_intermediate_size == 0`.
    fn encodeReduceAndOptionalShared(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: MoELayerArgs,
        default_shared_dtype: quant.WeightDType,
    ) EncodeError!void {
        if (args.shared_expert) |se| {
            const se_dtype = args.shared_expert_dtype orelse default_shared_dtype;
            const se_inter = if (args.shared_intermediate_size > 0) args.shared_intermediate_size else args.intermediate_size;
            const se_weights = args.shared_weights_base orelse args.weights_base;
            try self.encodeSharedExpertPreReduce(frame, se_dtype, se, se_weights, args, se_inter, true);
            try self.encodeReduce(frame, .{
                .workspace = args.workspace,
                .workspace_buf = args.workspace_buf,
                .residual = args.residual,
                .out = args.out,
                .m = args.m,
                .hidden_size = args.hidden_size,
                .top_k = args.top_k,
            }, true);
            try self.encodeSharedExpertDownAccum(frame, se_dtype, se, se_weights, args, se_inter, true);
        } else {
            try self.encodeReduce(frame, .{
                .workspace = args.workspace,
                .workspace_buf = args.workspace_buf,
                .residual = args.residual,
                .out = args.out,
                .m = args.m,
                .hidden_size = args.hidden_size,
                .top_k = args.top_k,
            }, true);
        }
    }

    fn resolveSharedInterOffset(workspace: *const MoeWorkspace, m: u32, se_inter: u32) EncodeError!u64 {
        if (m == 0 or se_inter == 0) return error.ZeroDimension;
        if (m > workspace.spec.max_tokens) return error.WorkspaceTooSmall;
        if (workspace.spec.shared_intermediate_size > 0) {
            if (se_inter > workspace.spec.shared_intermediate_size) return error.WorkspaceTooSmall;
            return workspace.shared_inter_off;
        }
        // Fallback: reuse `workspace.inter` (capacity `max_tokens * top_k * intermediate_size` f16)
        // after routed `DownProj` has completed.
        const needed: usize = @as(usize, m) * @as(usize, se_inter);
        if (needed > workspace.inter.len) return error.WorkspaceTooSmall;
        return workspace.inter_off;
    }

    fn encodeSharedExpertPreReduce(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        se_dtype: quant.WeightDType,
        se: ExpertWeights,
        se_weights: metal.BufferBinding,
        args: MoELayerArgs,
        se_inter: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_size == 0 or se_inter == 0) return error.ZeroDimension;
        const inter_off = try resolveSharedInterOffset(args.workspace, args.m, se_inter);
        const wb = args.workspace_buf;

        var gu_barrier = insert_barrier_before;
        if (args.shared_gate_weight) |gw| {
            const g_pipe = switch (args.shared_gate_dtype) {
                .f16 => self.shared_gate_f16,
                .f32 => self.shared_gate_f32,
                .bf16 => self.shared_gate_bf16,
            };
            const g_params: MoeSharedGateParams = .{
                .m = args.m,
                .hidden_size = args.hidden_size,
            };
            const g_groups_x: u64 = (@as(u64, args.m) + 3) / 4;
            try frame.encodeThreadgroups(
                g_pipe,
                &.{
                    args.x,
                    gw,
                    .{ .buffer = wb, .offset = args.workspace.shared_gate_scores_off },
                },
                std.mem.asBytes(&g_params),
                .{ g_groups_x, 1, 1 },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
            gu_barrier = false;
        }

        const inter_binding: metal.BufferBinding = .{ .buffer = wb, .offset = inter_off };
        switch (se_dtype) {
            .q4_affine, .q3_affine, .q2_affine => {
                const num_groups, const row_bytes, const pipe = switch (se_dtype) {
                    .q4_affine => blk: {
                        const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                        break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.shared_gate_up_q4 };
                    },
                    .q3_affine => blk: {
                        const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                        break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.shared_gate_up_q3 };
                    },
                    .q2_affine => blk: {
                        const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                        break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.shared_gate_up_q2 };
                    },
                    else => unreachable,
                };
                const params: MoeSharedGateUpParams = .{
                    .gate_qweight_offset = se.gate_qweight_offset,
                    .gate_sb_offset = se.gate_sb_offset,
                    .up_qweight_offset = se.up_qweight_offset,
                    .up_sb_offset = se.up_sb_offset,
                    .m = args.m,
                    .hidden_size = args.hidden_size,
                    .intermediate_size = se_inter,
                    .num_groups = @intCast(num_groups),
                    .row_bytes = @intCast(row_bytes),
                    .swiglu_limit = args.swiglu_limit,
                };
                const groups_x: u64 = (@as(u64, se_inter) + 7) / 8;
                const groups_y: u64 = (@as(u64, args.m) + 3) / 4;
                try frame.encodeThreadgroups(
                    pipe,
                    &.{ args.x, se_weights, inter_binding },
                    std.mem.asBytes(&params),
                    .{ groups_x, groups_y, 1 },
                    .{ 128, 1, 1 },
                    gu_barrier,
                );
            },
            .f16, .bf16 => {
                if (args.hidden_size % 4 != 0) return error.UnalignedK;
                const pipe = if (se_dtype == .bf16) self.shared_gate_up_bf16 else self.shared_gate_up_f16;
                const params: MoeSharedGateUpParams = .{
                    .gate_qweight_offset = se.gate_qweight_offset,
                    .gate_sb_offset = 0,
                    .up_qweight_offset = se.up_qweight_offset,
                    .up_sb_offset = 0,
                    .m = args.m,
                    .hidden_size = args.hidden_size,
                    .intermediate_size = se_inter,
                    .num_groups = 0,
                    .row_bytes = 0,
                    .swiglu_limit = args.swiglu_limit,
                };
                const groups_x: u64 = (@as(u64, se_inter) + 3) / 4;
                try frame.encodeThreadgroups(
                    pipe,
                    &.{ args.x, se_weights, inter_binding },
                    std.mem.asBytes(&params),
                    .{ groups_x, args.m, 1 },
                    .{ 128, 1, 1 },
                    gu_barrier,
                );
            },
        }
    }

    fn encodeSharedExpertDownAccum(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        se_dtype: quant.WeightDType,
        se: ExpertWeights,
        se_weights: metal.BufferBinding,
        args: MoELayerArgs,
        se_inter: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const inter_off = try resolveSharedInterOffset(args.workspace, args.m, se_inter);
        const wb = args.workspace_buf;
        const inter_binding: metal.BufferBinding = .{ .buffer = wb, .offset = inter_off };
        const gate_scores_binding: metal.BufferBinding = .{ .buffer = wb, .offset = args.workspace.shared_gate_scores_off };
        const use_gate: u32 = if (args.shared_gate_weight != null) 1 else 0;

        switch (se_dtype) {
            .q4_affine, .q3_affine, .q2_affine => {
                const num_groups, const row_bytes, const pipe = switch (se_dtype) {
                    .q4_affine => blk: {
                        const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                        break :blk .{ try Ops.groupsPerRow(se_inter), try Ops.rowWeightBytes(se_inter), self.shared_down_accum_q4 };
                    },
                    .q3_affine => blk: {
                        const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                        break :blk .{ try Ops.groupsPerRow(se_inter), try Ops.rowWeightBytes(se_inter), self.shared_down_accum_q3 };
                    },
                    .q2_affine => blk: {
                        const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                        break :blk .{ try Ops.groupsPerRow(se_inter), try Ops.rowWeightBytes(se_inter), self.shared_down_accum_q2 };
                    },
                    else => unreachable,
                };
                const params: MoeSharedDownParams = .{
                    .down_qweight_offset = se.down_qweight_offset,
                    .down_sb_offset = se.down_sb_offset,
                    .m = args.m,
                    .hidden_size = args.hidden_size,
                    .intermediate_size = se_inter,
                    .num_groups = @intCast(num_groups),
                    .row_bytes = @intCast(row_bytes),
                    .use_gate = use_gate,
                };
                const groups_x: u64 = (@as(u64, args.hidden_size) + 7) / 8;
                const groups_y: u64 = (@as(u64, args.m) + 3) / 4;
                try frame.encodeThreadgroups(
                    pipe,
                    &.{ inter_binding, se_weights, gate_scores_binding, args.out },
                    std.mem.asBytes(&params),
                    .{ groups_x, groups_y, 1 },
                    .{ 128, 1, 1 },
                    insert_barrier_before,
                );
            },
            .f16, .bf16 => {
                if (se_inter % 4 != 0) return error.UnalignedK;
                const pipe = if (se_dtype == .bf16) self.shared_down_accum_bf16 else self.shared_down_accum_f16;
                const params: MoeSharedDownParams = .{
                    .down_qweight_offset = se.down_qweight_offset,
                    .down_sb_offset = 0,
                    .m = args.m,
                    .hidden_size = args.hidden_size,
                    .intermediate_size = se_inter,
                    .num_groups = 0,
                    .row_bytes = 0,
                    .use_gate = use_gate,
                };
                const groups_x: u64 = (@as(u64, args.hidden_size) + 3) / 4;
                try frame.encodeThreadgroups(
                    pipe,
                    &.{ inter_binding, se_weights, gate_scores_binding, args.out },
                    std.mem.asBytes(&params),
                    .{ groups_x, args.m, 1 },
                    .{ 128, 1, 1 },
                    insert_barrier_before,
                );
            },
        }
    }

    /// Standalone unconditional shared-expert SwiGLU accumulation into `out`
    /// (`out += [sigmoid(x · gate_w)] * shared_expert(x)`), using `workspace`
    /// scratch with zero CPU-GPU syncs and zero heap allocations (G7).
    pub fn encodeSharedExpertAccum(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.WeightDType,
        args: MoELayerArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const se = args.shared_expert orelse return error.InvalidArgument;
        const se_inter = if (args.shared_intermediate_size > 0) args.shared_intermediate_size else args.intermediate_size;
        const se_weights = args.shared_weights_base orelse args.weights_base;
        try self.encodeSharedExpertPreReduce(frame, dtype, se, se_weights, args, se_inter, insert_barrier_before);
        try self.encodeSharedExpertDownAccum(frame, dtype, se, se_weights, args, se_inter, true);
    }

    /// Encodes the optional router projection (f32 / f16 / bf16 / quantized
    /// weights) followed by softmax + top-k + CSR deduplication.
    fn encodeRouting(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: MoELayerArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const logits_binding, const logits_dtype, const route_barrier = switch (args.router) {
            .weights => |w| blk: {
                const lb = args.workspace.routerLogitsBinding(args.workspace_buf);
                try self.encodeRouterProj(frame, w.dtype, .{
                    .x = args.x,
                    .router_weights = w.binding,
                    .logits = lb,
                    .m = args.m,
                    .num_experts = args.num_experts,
                    .hidden_size = args.hidden_size,
                }, insert_barrier_before);
                break :blk .{ lb, LogitsDType.f32, true };
            },
            .quant_weights => |q| blk: {
                const lb = args.workspace.routerLogitsBinding(args.workspace_buf);
                try self.encodeRouterProjQuant(frame, q.bits, .{
                    .x = args.x,
                    .qweight = q.qweight,
                    .scale_bias = q.scale_bias,
                    .logits = lb,
                    .m = args.m,
                    .num_experts = args.num_experts,
                    .hidden_size = args.hidden_size,
                }, insert_barrier_before);
                break :blk .{ lb, LogitsDType.f32, true };
            },
            .quant_format => |qf_r| blk: {
                const lb = args.workspace.routerLogitsBinding(args.workspace_buf);
                if (args.m == 1) {
                    try qf_r.qf_pipes.encodeGemv(frame, qf_r.dtype, qf_r.group_size, .{
                        .x = args.x,
                        .qweight = qf_r.qweight,
                        .params = qf_r.params,
                        .y = lb,
                        .n = args.num_experts,
                        .k = args.hidden_size,
                        .tensor_scale_per_row = qf_r.tensor_scale_per_row,
                    }, insert_barrier_before);
                } else {
                    try qf_r.qf_pipes.encodeGemvBatch(frame, qf_r.dtype, qf_r.group_size, .{
                        .x = args.x,
                        .qweight = qf_r.qweight,
                        .params = qf_r.params,
                        .y = lb,
                        .m = args.m,
                        .n = args.num_experts,
                        .k = args.hidden_size,
                        .tensor_scale_per_row = qf_r.tensor_scale_per_row,
                    }, insert_barrier_before);
                }
                break :blk .{ lb, LogitsDType.f16, true };
            },
            .logits => |l| .{ l.binding, l.dtype, insert_barrier_before },
        };

        try self.encodeRouteTopK(frame, logits_dtype, .{
            .logits = logits_binding,
            .workspace = args.workspace,
            .workspace_buf = args.workspace_buf,
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .norm_topk_prob = args.norm_topk_prob,
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
        }, route_barrier);
    }

    /// Encodes the optional router projection (f32 / f16 / bf16 / quantized)
    /// followed by G13 scheme-bucketed softmax + top-k + CSR deduplication.
    fn encodeRoutingBucketed(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: MoELayerHeteroArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const base = args.base;
        const logits_binding, const logits_dtype, const route_barrier = switch (base.router) {
            .weights => |w| blk: {
                const lb = base.workspace.routerLogitsBinding(base.workspace_buf);
                try self.encodeRouterProj(frame, w.dtype, .{
                    .x = base.x,
                    .router_weights = w.binding,
                    .logits = lb,
                    .m = base.m,
                    .num_experts = base.num_experts,
                    .hidden_size = base.hidden_size,
                }, insert_barrier_before);
                break :blk .{ lb, LogitsDType.f32, true };
            },
            .quant_weights => |q| blk: {
                const lb = base.workspace.routerLogitsBinding(base.workspace_buf);
                try self.encodeRouterProjQuant(frame, q.bits, .{
                    .x = base.x,
                    .qweight = q.qweight,
                    .scale_bias = q.scale_bias,
                    .logits = lb,
                    .m = base.m,
                    .num_experts = base.num_experts,
                    .hidden_size = base.hidden_size,
                }, insert_barrier_before);
                break :blk .{ lb, LogitsDType.f32, true };
            },
            .quant_format => |qf_r| blk: {
                const lb = base.workspace.routerLogitsBinding(base.workspace_buf);
                if (base.m == 1) {
                    try qf_r.qf_pipes.encodeGemv(frame, qf_r.dtype, qf_r.group_size, .{
                        .x = base.x,
                        .qweight = qf_r.qweight,
                        .params = qf_r.params,
                        .y = lb,
                        .n = base.num_experts,
                        .k = base.hidden_size,
                        .tensor_scale_per_row = qf_r.tensor_scale_per_row,
                    }, insert_barrier_before);
                } else {
                    try qf_r.qf_pipes.encodeGemvBatch(frame, qf_r.dtype, qf_r.group_size, .{
                        .x = base.x,
                        .qweight = qf_r.qweight,
                        .params = qf_r.params,
                        .y = lb,
                        .m = base.m,
                        .n = base.num_experts,
                        .k = base.hidden_size,
                        .tensor_scale_per_row = qf_r.tensor_scale_per_row,
                    }, insert_barrier_before);
                }
                break :blk .{ lb, LogitsDType.f16, true };
            },
            .logits => |l| .{ l.binding, l.dtype, insert_barrier_before },
        };

        try self.encodeRouteTopKBucketed(frame, logits_dtype, .{
            .logits = logits_binding,
            .expert_schemes = args.expert_schemes,
            .workspace = base.workspace,
            .workspace_buf = base.workspace_buf,
            .m = base.m,
            .num_experts = base.num_experts,
            .top_k = base.top_k,
            .norm_topk_prob = base.norm_topk_prob,
            .hidden_size = base.hidden_size,
            .intermediate_size = base.intermediate_size,
        }, route_barrier);
    }

    /// Encodes G13 per-scheme Gate+Up+SwiGLU and DownProj dispatches for a
    /// heterogeneous MoE layer whose CSR buckets in `base.workspace` have
    /// already been populated by `encodeRouteTopKBucketed` or
    /// `workspace.rebuildSchemeBucketsCpu`. Zero heap allocation.
    ///
    /// When `skip_empty_from_host` is true (e.g., `ColdRuntime` after GPU
    /// routing sync), reads each bucket's live `num_unique_experts` from
    /// shared RAM and omits runtime-empty buckets completely.
    /// Otherwise enforces the M7.8 G20 static-grid guard: keeps direct dispatch
    /// only when `M == 1` and `num_scheme_buckets <= 1`, and uses GPU indirect
    /// dispatch whenever `M > 1` or `num_scheme_buckets > 1`.
    pub fn encodeGateUpDownHetero(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        base: MoELayerArgs,
        scheme_counts: [NUM_SCHEME_BUCKETS]u32,
        skip_empty_from_host: bool,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const num_scheme_buckets = countActiveSchemeBuckets(scheme_counts);
        const use_indirect = if (skip_empty_from_host)
            false
        else
            shouldUseIndirectDispatch(base.m, num_scheme_buckets);

        var bucket_caps: [NUM_SCHEME_BUCKETS]u32 = undefined;
        for (0..NUM_SCHEME_BUCKETS) |s_idx| {
            const scheme: quant.WeightDType = @fromBackingInt(@intCast(s_idx));
            bucket_caps[s_idx] = if (skip_empty_from_host)
                base.workspace.bucketHeader(scheme).num_unique_experts
            else
                scheme_counts[s_idx];
        }

        // Phase A: specialized Gate+Up+SwiGLU per active scheme bucket.
        // All buckets write disjoint CSR slot rows of `workspace.inter`.
        var first_gu = true;
        for (0..NUM_SCHEME_BUCKETS) |s_idx| {
            const cap = bucket_caps[s_idx];
            if (cap == 0) continue;
            const scheme: quant.WeightDType = @fromBackingInt(@intCast(s_idx));
            const barrier = if (first_gu) insert_barrier_before else false;
            first_gu = false;
            const gu_args: GateUpSwigluArgs = .{
                .x = base.x,
                .weights_base = base.weights_base,
                .expert_table = base.expert_table,
                .workspace = base.workspace,
                .workspace_buf = base.workspace_buf,
                .m = base.m,
                .num_experts = base.num_experts,
                .top_k = base.top_k,
                .hidden_size = base.hidden_size,
                .intermediate_size = base.intermediate_size,
                .scheme_bucket = scheme,
                .max_active_override = cap,
                .use_indirect = use_indirect,
                .swiglu_limit = base.swiglu_limit,
            };
            switch (scheme) {
                .f16 => try self.encodeGateUpSwigluFloat(frame, .f16, gu_args, barrier),
                .bf16 => try self.encodeGateUpSwigluFloat(frame, .bf16, gu_args, barrier),
                .q4_affine => try self.encodeGateUpSwiglu(frame, .q4, gu_args, barrier),
                .q3_affine => try self.encodeGateUpSwiglu(frame, .q3, gu_args, barrier),
                .q2_affine => try self.encodeGateUpSwiglu(frame, .q2, gu_args, barrier),
            }
        }

        // Phase B: specialized DownProj per active scheme bucket.
        // First bucket waits on all Gate+Up buckets via intra-encoder barrier;
        // subsequent buckets write disjoint `(tok, k_idx)` slots of `down_out`.
        var first_dn = true;
        for (0..NUM_SCHEME_BUCKETS) |s_idx| {
            const cap = bucket_caps[s_idx];
            if (cap == 0) continue;
            const scheme: quant.WeightDType = @fromBackingInt(@intCast(s_idx));
            const barrier = if (first_dn) true else false;
            first_dn = false;
            const dn_args: DownProjArgs = .{
                .weights_base = base.weights_base,
                .expert_table = base.expert_table,
                .workspace = base.workspace,
                .workspace_buf = base.workspace_buf,
                .m = base.m,
                .num_experts = base.num_experts,
                .top_k = base.top_k,
                .hidden_size = base.hidden_size,
                .intermediate_size = base.intermediate_size,
                .scheme_bucket = scheme,
                .max_active_override = cap,
                .use_indirect = use_indirect,
            };
            switch (scheme) {
                .f16 => try self.encodeDownProjFloat(frame, .f16, dn_args, barrier),
                .bf16 => try self.encodeDownProjFloat(frame, .bf16, dn_args, barrier),
                .q4_affine => try self.encodeDownProj(frame, .q4, dn_args, barrier),
                .q3_affine => try self.encodeDownProj(frame, .q3, dn_args, barrier),
                .q2_affine => try self.encodeDownProj(frame, .q2, dn_args, barrier),
            }
        }
    }

    /// Encodes a complete G13 heterogeneous MoE FFN layer into `frame` with
    /// zero dynamic heap allocation and zero CPU-GPU synchronization:
    /// optional router projection -> `ami_moe_route_topk_bucketed` ->
    /// one specialized Gate+Up+SwiGLU & DownProj pipeline per present scheme
    /// bucket -> optional shared expert -> `ami_moe_reduce_f16` -> optional
    /// shared expert accumulation into `out`.
    pub fn encodeMoELayerHetero(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: MoELayerHeteroArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try self.encodeRoutingBucketed(frame, args, insert_barrier_before);
        try self.encodeGateUpDownHetero(frame, args.base, args.scheme_counts, false, true);
        try self.encodeReduceAndOptionalShared(frame, args.base, .q4_affine);
    }

    fn encodeBucketGateUpOne(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        base: MoELayerArgs,
        x_in: metal.BufferBinding,
        desc: MoeSchemeDesc,
        scheme_bucket: ?quant.WeightDType,
        cap: u32,
        use_indirect: bool,
        qf_pipes: ?*const quant_formats.QuantFormatPipelines,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (!desc.usesQuantFormatsGateUp()) {
            const gu_args: GateUpSwigluArgs = .{
                .x = x_in,
                .weights_base = base.weights_base,
                .expert_table = base.expert_table,
                .workspace = base.workspace,
                .workspace_buf = base.workspace_buf,
                .m = base.m,
                .num_experts = base.num_experts,
                .top_k = base.top_k,
                .hidden_size = base.hidden_size,
                .intermediate_size = base.intermediate_size,
                .scheme_bucket = scheme_bucket,
                .max_active_override = cap,
                .use_indirect = use_indirect,
                .swiglu_limit = base.swiglu_limit,
            };
            switch (desc.gate_up_dtype) {
                .f16 => try self.encodeGateUpSwigluFloat(frame, .f16, gu_args, insert_barrier_before),
                .bf16 => try self.encodeGateUpSwigluFloat(frame, .bf16, gu_args, insert_barrier_before),
                .q4_affine => try self.encodeGateUpSwiglu(frame, .q4, gu_args, insert_barrier_before),
                .q3_affine => try self.encodeGateUpSwiglu(frame, .q3, gu_args, insert_barrier_before),
                .q2_affine => try self.encodeGateUpSwiglu(frame, .q2, gu_args, insert_barrier_before),
                else => return error.InvalidArgument,
            }
            return;
        }
        const qf = qf_pipes orelse return error.PipelineNotInitialized;
        const qf_dt = quant_formats.DType.fromAmf(desc.gate_up_dtype) orelse return error.InvalidArgument;
        const cap_active: u32 = @min(cap, base.num_experts);
        const max_active: u32 = @min(base.m * base.top_k, cap_active);
        if (max_active == 0) return;
        const ws = base.workspace;
        const wb = base.workspace_buf;
        const hdr_off = if (scheme_bucket) |s| ws.bucketHeaderOffset(s) else ws.schedule_header_off;
        const unique_off = if (scheme_bucket) |s| ws.bucketUniqueOffset(s) else ws.unique_experts_off;
        const offsets_off = if (scheme_bucket) |s| ws.bucketOffsetsOffset(s) else ws.expert_token_offsets_off;
        try qf.encodeMoeGateUpSwiglu(frame, qf_dt, desc.gate_up_group_size, .{
            .x = x_in,
            .weights_base = base.weights_base,
            .expert_table = base.expert_table,
            .sched_hdr = .{ .buffer = wb, .offset = hdr_off },
            .unique_experts = .{ .buffer = wb, .offset = unique_off },
            .expert_token_offsets = .{ .buffer = wb, .offset = offsets_off },
            .token_indices = .{ .buffer = wb, .offset = ws.token_indices_off },
            .inter = .{ .buffer = wb, .offset = ws.inter_off },
            .num_unique_experts = max_active,
            .hidden_size = base.hidden_size,
            .intermediate_size = base.intermediate_size,
            .tensor_scale_per_row = desc.gate_up_tensor_scale_per_row,
            .indirect_args = if (use_indirect) .{ .buffer = wb, .offset = ws.scheduleIndirectOffset(scheme_bucket, INDIRECT_GATE_UP_Q_OFFSET) } else null,
            .swiglu_limit = base.swiglu_limit,
        }, insert_barrier_before);
    }

    fn encodeBucketDownOne(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        base: MoELayerArgs,
        inter_in: metal.BufferBinding,
        desc: MoeSchemeDesc,
        scheme_bucket: ?quant.WeightDType,
        cap: u32,
        use_indirect: bool,
        qf_pipes: ?*const quant_formats.QuantFormatPipelines,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (!desc.usesQuantFormatsDown()) {
            const dn_args: DownProjArgs = .{
                .weights_base = base.weights_base,
                .expert_table = base.expert_table,
                .workspace = base.workspace,
                .workspace_buf = base.workspace_buf,
                .m = base.m,
                .num_experts = base.num_experts,
                .top_k = base.top_k,
                .hidden_size = base.hidden_size,
                .intermediate_size = base.intermediate_size,
                .scheme_bucket = scheme_bucket,
                .max_active_override = cap,
                .use_indirect = use_indirect,
            };
            switch (desc.down_dtype) {
                .f16 => try self.encodeDownProjFloat(frame, .f16, dn_args, insert_barrier_before),
                .bf16 => try self.encodeDownProjFloat(frame, .bf16, dn_args, insert_barrier_before),
                .q4_affine => try self.encodeDownProj(frame, .q4, dn_args, insert_barrier_before),
                .q3_affine => try self.encodeDownProj(frame, .q3, dn_args, insert_barrier_before),
                .q2_affine => try self.encodeDownProj(frame, .q2, dn_args, insert_barrier_before),
                else => return error.InvalidArgument,
            }
            return;
        }
        const qf = qf_pipes orelse return error.PipelineNotInitialized;
        const qf_dt = quant_formats.DType.fromAmf(desc.down_dtype) orelse return error.InvalidArgument;
        const cap_active: u32 = @min(cap, base.num_experts);
        const max_active: u32 = @min(base.m * base.top_k, cap_active);
        if (max_active == 0) return;
        const ws = base.workspace;
        const wb = base.workspace_buf;
        const hdr_off = if (scheme_bucket) |s| ws.bucketHeaderOffset(s) else ws.schedule_header_off;
        const unique_off = if (scheme_bucket) |s| ws.bucketUniqueOffset(s) else ws.unique_experts_off;
        const offsets_off = if (scheme_bucket) |s| ws.bucketOffsetsOffset(s) else ws.expert_token_offsets_off;
        try qf.encodeMoeDownProj(frame, qf_dt, desc.down_group_size, .{
            .inter = inter_in,
            .weights_base = base.weights_base,
            .expert_table = base.expert_table,
            .sched_hdr = .{ .buffer = wb, .offset = hdr_off },
            .unique_experts = .{ .buffer = wb, .offset = unique_off },
            .expert_token_offsets = .{ .buffer = wb, .offset = offsets_off },
            .token_indices = .{ .buffer = wb, .offset = ws.token_indices_off },
            .routing_weights = .{ .buffer = wb, .offset = ws.routing_weights_off },
            .slot_k_indices = .{ .buffer = wb, .offset = ws.slot_k_indices_off },
            .down_out = .{ .buffer = wb, .offset = ws.down_out_off },
            .num_unique_experts = max_active,
            .hidden_size = base.hidden_size,
            .intermediate_size = base.intermediate_size,
            .top_k = base.top_k,
            .tensor_scale_per_row = desc.down_tensor_scale_per_row,
            .indirect_args = if (use_indirect) .{ .buffer = wb, .offset = ws.scheduleIndirectOffset(scheme_bucket, INDIRECT_DOWN_Q8_OFFSET) } else null,
        }, insert_barrier_before);
    }

    /// Encodes a complete multi-format MoE FFN layer across ALL `W21`/`W22` `.amf`
    /// `DType`s (`0..37`, uniform or up to `NUM_SCHEME_BUCKETS` heterogeneous buckets,
    /// including asymmetric `gate_up_dtype != down_dtype`) and optional `G24`
    /// activation-quantization emulation. Zero hot-path heap allocation.
    pub fn encodeMoELayerMultiFormat(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: MoELayerMultiFormatArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const base = args.base;
        if (args.is_heterogeneous) {
            const es = args.expert_schemes orelse return error.InvalidArgument;
            try self.encodeRoutingBucketed(frame, .{
                .base = base,
                .expert_schemes = es,
                .scheme_counts = args.scheme_counts,
            }, insert_barrier_before);

            const num_buckets = countActiveSchemeBuckets(args.scheme_counts);
            const use_indirect = shouldUseIndirectDispatch(base.m, num_buckets);

            var first_gu = true;
            for (0..NUM_SCHEME_BUCKETS) |s_idx| {
                const cap = args.scheme_counts[s_idx];
                if (cap == 0) continue;
                const scheme: quant.WeightDType = @fromBackingInt(@intCast(s_idx));
                const desc = args.scheme_descs[s_idx];
                var x_eff = base.x;
                var gu_barrier = first_gu;
                const aq_gu = args.act_quant_policy.resolveForTensor(desc.gate_up_dtype, base.hidden_size);
                if (aq_gu != .none) {
                    const need = std.math.mul(u64, @as(u64, base.m) * base.hidden_size, @sizeOf(f16)) catch return error.Overflow;
                    if (args.qf_pipes) |qf| {
                        if (args.act_scratch) |scratch| {
                            if (args.act_scratch_bytes >= need) {
                                try qf.encodeActQuantEmulate(frame, aq_gu, .{
                                    .x_in = base.x,
                                    .x_out = scratch,
                                    .m = base.m,
                                    .k = base.hidden_size,
                                }, true);
                                x_eff = scratch;
                                gu_barrier = true;
                            }
                        }
                    }
                }
                try self.encodeBucketGateUpOne(frame, base, x_eff, desc, scheme, cap, use_indirect, args.qf_pipes, gu_barrier);
                first_gu = false;
            }

            try self.encodeDownAndReduceMultiFormat(frame, args, true);
        } else {
            try self.encodeRouting(frame, base, insert_barrier_before);
            const desc = args.scheme_descs[0];
            const use_indirect = shouldUseIndirectDispatch(base.m, 1);
            var x_eff = base.x;
            var gu_barrier = true;
            const aq_gu = args.act_quant_policy.resolveForTensor(desc.gate_up_dtype, base.hidden_size);
            if (aq_gu != .none) {
                const need = std.math.mul(u64, @as(u64, base.m) * base.hidden_size, @sizeOf(f16)) catch return error.Overflow;
                if (args.qf_pipes) |qf| {
                    if (args.act_scratch) |scratch| {
                        if (args.act_scratch_bytes >= need) {
                            try qf.encodeActQuantEmulate(frame, aq_gu, .{
                                .x_in = base.x,
                                .x_out = scratch,
                                .m = base.m,
                                .k = base.hidden_size,
                            }, true);
                            x_eff = scratch;
                            gu_barrier = true;
                        }
                    }
                }
            }
            try self.encodeBucketGateUpOne(frame, base, x_eff, desc, null, base.num_experts, use_indirect, args.qf_pipes, gu_barrier);
            try self.encodeDownAndReduceMultiFormat(frame, args, true);
        }
    }

    /// Encodes the multi-format DownProj stage (all populated heterogeneous buckets
    /// or single homogeneous pool), optional `G24` intermediate activation quant
    /// emulation, `ami_moe_reduce_f16`, and optional shared expert accumulation.
    /// Shared by `encodeMoELayerMultiFormat` and `Inkling2Hy4Pipelines.encodeInkling2MoeLayerIndirect`.
    pub fn encodeDownAndReduceMultiFormat(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: MoELayerMultiFormatArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const base = args.base;
        if (args.is_heterogeneous) {
            const num_buckets = countActiveSchemeBuckets(args.scheme_counts);
            const use_indirect = shouldUseIndirectDispatch(base.m, num_buckets);
            var first_dn = insert_barrier_before;
            for (0..NUM_SCHEME_BUCKETS) |s_idx| {
                const cap = args.scheme_counts[s_idx];
                if (cap == 0) continue;
                const scheme: quant.WeightDType = @fromBackingInt(@intCast(s_idx));
                const desc = args.scheme_descs[s_idx];
                const inter_bind = metal.BufferBinding{ .buffer = base.workspace_buf, .offset = base.workspace.inter_off };
                try self.encodeBucketDownOne(frame, base, inter_bind, desc, scheme, cap, use_indirect, args.qf_pipes, first_dn);
                first_dn = false;
            }
        } else {
            const desc = args.scheme_descs[0];
            const use_indirect = shouldUseIndirectDispatch(base.m, 1);
            var inter_eff = metal.BufferBinding{ .buffer = base.workspace_buf, .offset = base.workspace.inter_off };
            var dn_barrier = insert_barrier_before;
            const aq_dn = args.act_quant_policy.resolveForTensor(desc.down_dtype, base.intermediate_size);
            if (aq_dn != .none and desc.usesQuantFormatsDown()) {
                const rows = base.m * base.top_k;
                const need = std.math.mul(u64, @as(u64, rows) * base.intermediate_size, @sizeOf(f16)) catch return error.Overflow;
                if (args.qf_pipes) |qf| {
                    if (args.act_scratch) |scratch| {
                        if (args.act_scratch_bytes >= need) {
                            try qf.encodeActQuantEmulate(frame, aq_dn, .{
                                .x_in = inter_eff,
                                .x_out = scratch,
                                .m = rows,
                                .k = base.intermediate_size,
                            }, dn_barrier);
                            inter_eff = scratch;
                            dn_barrier = true;
                        }
                    }
                }
            }
            try self.encodeBucketDownOne(frame, base, inter_eff, desc, null, base.num_experts, use_indirect, args.qf_pipes, dn_barrier);
        }

        // Shared expert + routed reduction.
        const se_amf = args.shared_expert_amf_dtype orelse args.scheme_descs[0].gate_up_dtype;
        if (base.shared_expert != null and dequant.usesQuantFormats(se_amf, args.shared_expert_group_size)) {
            try self.encodeReduce(frame, .{
                .workspace = base.workspace,
                .workspace_buf = base.workspace_buf,
                .residual = base.residual,
                .out = base.out,
                .m = base.m,
                .hidden_size = base.hidden_size,
                .top_k = base.top_k,
            }, true);
            const se = base.shared_expert.?;
            const qf = args.qf_pipes orelse return error.PipelineNotInitialized;
            const qf_dt = quant_formats.DType.fromAmf(se_amf) orelse return error.InvalidArgument;
            const se_inter = if (base.shared_intermediate_size > 0) base.shared_intermediate_size else base.intermediate_size;
            const se_weights = base.shared_weights_base orelse base.weights_base;
            const gate_off = try resolveSharedInterOffset(base.workspace, base.m, se_inter);
            const gate_elems: usize = @as(usize, base.m) * se_inter;
            const up_bind: metal.BufferBinding = if (gate_elems <= base.workspace.down_out.len)
                .{ .buffer = base.workspace_buf, .offset = base.workspace.down_out_off }
            else if (gate_off == base.workspace.inter_off and 2 * gate_elems <= base.workspace.inter.len)
                .{ .buffer = base.workspace_buf, .offset = gate_off + @as(u64, gate_elems) * @sizeOf(f16) }
            else
                return error.WorkspaceTooSmall;
            const gate_bind = metal.BufferBinding{ .buffer = base.workspace_buf, .offset = gate_off };
            const w_gate = metal.BufferBinding{ .buffer = se_weights.buffer, .offset = se_weights.offset + se.gate_qweight_offset };
            const p_gate: ?metal.BufferBinding = if (se.gate_sb_offset != 0) .{ .buffer = se_weights.buffer, .offset = se_weights.offset + se.gate_sb_offset } else null;
            const w_up = metal.BufferBinding{ .buffer = se_weights.buffer, .offset = se_weights.offset + se.up_qweight_offset };
            const p_up: ?metal.BufferBinding = if (se.up_sb_offset != 0) .{ .buffer = se_weights.buffer, .offset = se_weights.offset + se.up_sb_offset } else null;
            const w_dn = metal.BufferBinding{ .buffer = se_weights.buffer, .offset = se_weights.offset + se.down_qweight_offset };
            const p_dn: ?metal.BufferBinding = if (se.down_sb_offset != 0) .{ .buffer = se_weights.buffer, .offset = se_weights.offset + se.down_sb_offset } else null;

            if (base.m == 1) {
                try qf.encodeGemv(frame, qf_dt, args.shared_expert_group_size, .{
                    .x = base.x,
                    .qweight = w_gate,
                    .params = p_gate,
                    .y = gate_bind,
                    .n = se_inter,
                    .k = base.hidden_size,
                    .tensor_scale_per_row = args.shared_expert_tensor_scale_per_row,
                }, true);
                try qf.encodeGemv(frame, qf_dt, args.shared_expert_group_size, .{
                    .x = base.x,
                    .qweight = w_up,
                    .params = p_up,
                    .y = up_bind,
                    .n = se_inter,
                    .k = base.hidden_size,
                    .tensor_scale_per_row = args.shared_expert_tensor_scale_per_row,
                }, false);
            } else {
                try qf.encodeGemvBatch(frame, qf_dt, args.shared_expert_group_size, .{
                    .x = base.x,
                    .qweight = w_gate,
                    .params = p_gate,
                    .y = gate_bind,
                    .m = base.m,
                    .n = se_inter,
                    .k = base.hidden_size,
                    .tensor_scale_per_row = args.shared_expert_tensor_scale_per_row,
                }, true);
                try qf.encodeGemvBatch(frame, qf_dt, args.shared_expert_group_size, .{
                    .x = base.x,
                    .qweight = w_up,
                    .params = p_up,
                    .y = up_bind,
                    .m = base.m,
                    .n = se_inter,
                    .k = base.hidden_size,
                    .tensor_scale_per_row = args.shared_expert_tensor_scale_per_row,
                }, false);
            }
            try self.encodeSiluMul(frame, .{
                .gate = gate_bind,
                .up = up_bind,
                .out = gate_bind,
                .count = @intCast(gate_elems),
                .swiglu_limit = base.swiglu_limit,
            }, true);
            if (base.m == 1) {
                try qf.encodeGemv(frame, qf_dt, args.shared_expert_group_size, .{
                    .x = gate_bind,
                    .qweight = w_dn,
                    .params = p_dn,
                    .y = base.out,
                    .residual = base.out,
                    .n = base.hidden_size,
                    .k = se_inter,
                    .tensor_scale_per_row = args.shared_expert_tensor_scale_per_row,
                }, true);
            } else {
                try qf.encodeGemvBatch(frame, qf_dt, args.shared_expert_group_size, .{
                    .x = gate_bind,
                    .qweight = w_dn,
                    .params = p_dn,
                    .y = base.out,
                    .residual = base.out,
                    .m = base.m,
                    .n = base.hidden_size,
                    .k = se_inter,
                    .tensor_scale_per_row = args.shared_expert_tensor_scale_per_row,
                }, true);
            }
        } else {
            const default_se = dequant.legacyWeightDTypeForAmf(args.scheme_descs[0].gate_up_dtype);
            try self.encodeReduceAndOptionalShared(frame, base, default_se);
        }
    }

    /// Encodes an MLX affine 2-bit / 3-bit / 4-bit group-64 router projection
    /// writing `f32` logits `[m, num_experts]`. Zero heap allocation.
    pub fn encodeRouterProjQuant(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: RouterProjQuantArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.hidden_size == 0) return error.ZeroDimension;
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExpertCount;
        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.router_proj_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.router_proj_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.router_proj_q2 };
            },
        };
        const params: RouterQParams = .{
            .m = args.m,
            .num_experts = args.num_experts,
            .hidden_size = args.hidden_size,
            .num_groups = @intCast(num_groups),
            .row_bytes = @intCast(row_bytes),
        };
        const groups_x: u64 = (@as(u64, args.num_experts) + 15) / 16;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.qweight, args.scale_bias, args.logits },
            std.mem.asBytes(&params),
            .{ groups_x, args.m, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes the deduplicated FP16 / BF16 expert gate + up + SwiGLU
    /// (`ExpertWeights.*_qweight_offset` = byte offsets of `[N, K]` matrices).
    pub fn encodeGateUpSwigluFloat(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.FloatDType,
        args: GateUpSwigluArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0 or
            args.hidden_size == 0 or args.intermediate_size == 0)
        {
            return error.ZeroDimension;
        }
        if (args.hidden_size % 4 != 0 or args.intermediate_size % 4 != 0) return error.UnalignedK;
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExpertCount;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;
        if (args.m > args.workspace.spec.max_tokens or
            args.num_experts > args.workspace.spec.num_experts or
            args.top_k > args.workspace.spec.top_k or
            args.intermediate_size > args.workspace.spec.intermediate_size)
        {
            return error.WorkspaceTooSmall;
        }
        const pipe = switch (dtype) {
            .f16 => self.gate_up_swiglu_f16,
            .bf16 => self.gate_up_swiglu_bf16,
        };
        const params: MoeGateUpParams = .{
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
            .num_groups = 0,
            .row_bytes = args.hidden_size * 2,
            .swiglu_limit = args.swiglu_limit,
        };
        const groups_x: u64 = (@as(u64, args.intermediate_size) + 15) / 16;
        const cap_active: u64 = if (args.max_active_override) |cap| @min(@as(u64, cap), args.num_experts) else args.num_experts;
        const max_active: u64 = @min(@as(u64, args.m) * args.top_k, cap_active);
        if (max_active == 0) return;
        const ws = args.workspace;
        const wb = args.workspace_buf;
        const hdr_off = if (args.scheme_bucket) |s| ws.bucketHeaderOffset(s) else ws.schedule_header_off;
        const unique_off = if (args.scheme_bucket) |s| ws.bucketUniqueOffset(s) else ws.unique_experts_off;
        const offsets_off = if (args.scheme_bucket) |s| ws.bucketOffsetsOffset(s) else ws.expert_token_offsets_off;
        const bindings = [_]metal.BufferBinding{
            args.x,
            args.weights_base,
            args.expert_table,
            .{ .buffer = wb, .offset = hdr_off },
            .{ .buffer = wb, .offset = unique_off },
            .{ .buffer = wb, .offset = offsets_off },
            .{ .buffer = wb, .offset = ws.token_indices_off },
            .{ .buffer = wb, .offset = ws.inter_off },
        };
        const use_indirect = args.use_indirect orelse
            shouldUseIndirectDispatch(args.m, if (args.scheme_bucket != null) 2 else 1);
        if (use_indirect) {
            try frame.encodeThreadgroupsIndirect(
                pipe,
                &bindings,
                std.mem.asBytes(&params),
                .{ .buffer = wb, .offset = ws.scheduleIndirectOffset(args.scheme_bucket, INDIRECT_GATE_UP_F_OFFSET) },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
        } else {
            try frame.encodeThreadgroups(
                pipe,
                &bindings,
                std.mem.asBytes(&params),
                .{ groups_x, max_active, 1 },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
        }
    }

    /// Encodes the deduplicated FP16 / BF16 expert down projection scaled by
    /// routing weights into the fp32 `down_out` workspace.
    pub fn encodeDownProjFloat(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.FloatDType,
        args: DownProjArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0 or
            args.hidden_size == 0 or args.intermediate_size == 0)
        {
            return error.ZeroDimension;
        }
        if (args.hidden_size % 4 != 0 or args.intermediate_size % 4 != 0) return error.UnalignedK;
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExpertCount;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;
        if (args.m > args.workspace.spec.max_tokens or
            args.num_experts > args.workspace.spec.num_experts or
            args.top_k > args.workspace.spec.top_k or
            args.hidden_size > args.workspace.spec.hidden_size or
            args.intermediate_size > args.workspace.spec.intermediate_size)
        {
            return error.WorkspaceTooSmall;
        }
        const pipe = switch (dtype) {
            .f16 => self.down_proj_f16,
            .bf16 => self.down_proj_bf16,
        };
        const params: MoeDownParams = .{
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
            .num_groups = 0,
            .row_bytes = args.intermediate_size * 2,
            .top_k = args.top_k,
        };
        const groups_x: u64 = (@as(u64, args.hidden_size) + 15) / 16;
        const cap_active: u64 = if (args.max_active_override) |cap| @min(@as(u64, cap), args.num_experts) else args.num_experts;
        const max_active: u64 = @min(@as(u64, args.m) * args.top_k, cap_active);
        if (max_active == 0) return;
        const ws = args.workspace;
        const wb = args.workspace_buf;
        const hdr_off = if (args.scheme_bucket) |s| ws.bucketHeaderOffset(s) else ws.schedule_header_off;
        const unique_off = if (args.scheme_bucket) |s| ws.bucketUniqueOffset(s) else ws.unique_experts_off;
        const offsets_off = if (args.scheme_bucket) |s| ws.bucketOffsetsOffset(s) else ws.expert_token_offsets_off;
        const bindings = [_]metal.BufferBinding{
            .{ .buffer = wb, .offset = ws.inter_off },
            args.weights_base,
            args.expert_table,
            .{ .buffer = wb, .offset = hdr_off },
            .{ .buffer = wb, .offset = unique_off },
            .{ .buffer = wb, .offset = offsets_off },
            .{ .buffer = wb, .offset = ws.token_indices_off },
            .{ .buffer = wb, .offset = ws.routing_weights_off },
            .{ .buffer = wb, .offset = ws.slot_k_indices_off },
            .{ .buffer = wb, .offset = ws.down_out_off },
        };
        const use_indirect = args.use_indirect orelse
            shouldUseIndirectDispatch(args.m, if (args.scheme_bucket != null) 2 else 1);
        if (use_indirect) {
            try frame.encodeThreadgroupsIndirect(
                pipe,
                &bindings,
                std.mem.asBytes(&params),
                .{ .buffer = wb, .offset = ws.scheduleIndirectOffset(args.scheme_bucket, INDIRECT_DOWN_OFFSET) },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
        } else {
            try frame.encodeThreadgroups(
                pipe,
                &bindings,
                std.mem.asBytes(&params),
                .{ groups_x, max_active, 1 },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
        }
    }

    /// Encodes a complete deduplicated FP16 / BF16 MoE FFN layer into `frame`:
    /// optional router projection (f32 / f16 / bf16 / quantized) ->
    /// `ami_moe_route_topk` -> `ami_moe_gate_up_swiglu_float` ->
    /// `ami_moe_down_proj_float` -> `ami_moe_reduce_f16`. Zero heap allocation.
    pub fn encodeMoELayerFloat(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.FloatDType,
        args: MoELayerArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.hidden_size % 4 != 0 or args.intermediate_size % 4 != 0) return error.UnalignedK;
        try self.encodeRouting(frame, args, insert_barrier_before);

        // M7.8 G20 M=1 static-grid guard: homogeneous M==1 stays on direct dispatch.
        const use_indirect = shouldUseIndirectDispatch(args.m, 1);

        try self.encodeGateUpSwigluFloat(frame, dtype, .{
            .x = args.x,
            .weights_base = args.weights_base,
            .expert_table = args.expert_table,
            .workspace = args.workspace,
            .workspace_buf = args.workspace_buf,
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
            .use_indirect = use_indirect,
            .swiglu_limit = args.swiglu_limit,
        }, true);

        try self.encodeDownProjFloat(frame, dtype, .{
            .weights_base = args.weights_base,
            .expert_table = args.expert_table,
            .workspace = args.workspace,
            .workspace_buf = args.workspace_buf,
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
            .use_indirect = use_indirect,
        }, true);

        try self.encodeReduceAndOptionalShared(frame, args, quant.WeightDType.fromFloat(dtype));
    }

    /// Encodes a GPU embedding lookup (`out[m, :] = table[token[m], :]`) for
    /// `f16`, `bf16`, `q4_affine`, `q3_affine`, or `q2_affine` tables, writing `f16
    /// [m, hidden_size]` with zero CPU copies. Out-of-range token IDs yield zero
    /// rows. Zero heap allocation.
    pub fn encodeEmbedLookup(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.WeightDType,
        args: EmbedLookupArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.vocab_size == 0 or args.hidden_size == 0) return error.ZeroDimension;
        const pipe, const row_bytes: usize, const num_groups: usize = switch (dtype) {
            .f16 => .{ self.embed_f16, @as(usize, args.hidden_size) * 2, 0 },
            .bf16 => .{ self.embed_bf16, @as(usize, args.hidden_size) * 2, 0 },
            .q4_affine => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ self.embed_q4, try Ops.rowWeightBytes(args.hidden_size), try Ops.groupsPerRow(args.hidden_size) };
            },
            .q3_affine => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ self.embed_q3, try Ops.rowWeightBytes(args.hidden_size), try Ops.groupsPerRow(args.hidden_size) };
            },
            .q2_affine => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ self.embed_q2, try Ops.rowWeightBytes(args.hidden_size), try Ops.groupsPerRow(args.hidden_size) };
            },
        };
        const sb_binding: metal.BufferBinding = switch (dtype) {
            .f16, .bf16 => args.scale_bias orelse args.weight,
            .q4_affine, .q3_affine, .q2_affine => args.scale_bias orelse return error.InvalidArgument,
        };
        const use_buf: bool, const tok_id: u32, const tok_binding: metal.BufferBinding = switch (args.tokens) {
            .token_id => |t| blk: {
                if (args.m != 1) return error.InvalidArgument;
                break :blk .{ false, t, args.weight };
            },
            .token_ids => |b| .{ true, 0, b },
        };
        const params: EmbedParams = .{
            .m = args.m,
            .vocab_size = args.vocab_size,
            .hidden_size = args.hidden_size,
            .row_bytes = std.math.cast(u32, row_bytes) orelse return error.Overflow,
            .num_groups = @intCast(num_groups),
            .use_token_buffer = if (use_buf) 1 else 0,
            .token_id = tok_id,
        };
        const vec_cols: u64 = (@as(u64, args.hidden_size) + 7) / 8;
        try frame.encodeThreads(
            pipe,
            &.{ args.weight, sb_binding, tok_binding, args.out },
            std.mem.asBytes(&params),
            .{ vec_cols, args.m, 1 },
            .{ @min(vec_cols, 64), 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeSiluMul(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: SiluMulArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.count == 0) return error.ZeroDimension;
        const params: ElemParams = .{ .count = args.count, .swiglu_limit = args.swiglu_limit };
        const groups_x: u64 = (@as(u64, args.count) + 255) / 256;
        try frame.encodeThreadgroups(
            self.silu_mul_f16,
            &.{ args.gate, args.up, args.out },
            std.mem.asBytes(&params),
            .{ groups_x, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeResidualAdd(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: ResidualAddArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.count == 0) return error.ZeroDimension;
        const params: ElemParams = .{ .count = args.count };
        const groups_x: u64 = (@as(u64, args.count) + 255) / 256;
        try frame.encodeThreadgroups(
            self.residual_add_f16,
            &.{ args.x, args.delta },
            std.mem.asBytes(&params),
            .{ groups_x, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeArgmax(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: ArgmaxArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.vocab_size == 0) return error.ZeroDimension;
        const max_tg = self.argmax_f16.maxThreadsPerThreadgroup();
        const aligned_tg: u32 = @min(@as(u32, 1024), max_tg & ~@as(u32, 31));
        const tg_size: u64 = if (aligned_tg >= 32) aligned_tg else 32;
        const params: ArgmaxParams = .{
            .vocab_size = args.vocab_size,
            .aligned_half4 = if ((args.logits.offset % 8) == 0) 1 else 0,
        };
        try frame.encodeThreadgroups(
            self.argmax_f16,
            &.{ args.logits, args.out_token_id },
            std.mem.asBytes(&params),
            .{ 1, 1, 1 },
            .{ tg_size, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes batched `M`-row greedy argmax (`[num_rows, vocab_size]` f16/f32 ->
    /// `[num_rows]u32`, `1 <= num_rows <= MAX_LOGIT_ROWS`) with deterministic smallest-index
    /// tie breaking. Zero heap allocation on the hot path.
    pub fn encodeArgmaxRows(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: ArgmaxRowsArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_rows == 0 or args.vocab_size == 0) return error.ZeroDimension;
        if (args.num_rows > MAX_LOGIT_ROWS) return error.InvalidArgument;

        const pipe, const vec_align: u64 = switch (args.logits_dtype) {
            .f16 => .{ self.argmax_rows_f16, 8 },
            .f32 => .{ self.argmax_rows_f32, 16 },
        };
        const max_tg = pipe.maxThreadsPerThreadgroup();
        const aligned_tg: u32 = @min(@as(u32, 1024), max_tg & ~@as(u32, 31));
        const tg_size: u64 = if (aligned_tg >= 32) aligned_tg else 32;
        const can_vec4 = (args.logits.offset % vec_align) == 0 and (args.vocab_size % 4) == 0;
        const params: ArgmaxRowsParams = .{
            .num_rows = args.num_rows,
            .vocab_size = args.vocab_size,
            .aligned_vec4 = if (can_vec4) 1 else 0,
        };
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.logits, args.out_token_ids },
            std.mem.asBytes(&params),
            .{ args.num_rows, 1, 1 },
            .{ tg_size, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes single-dispatch fused FP32 softmax + top-K candidate extraction
    /// (`[num_rows, vocab_size]` -> `[num_rows, top_k]` `TopKCandidate`,
    /// `1 <= num_rows <= MAX_LOGIT_ROWS`, `1 <= top_k <= 16`). One 256-thread threadgroup
    /// per row; zero heap allocation on the hot path.
    pub fn encodeTopKLogitsRows(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: TopKLogitsRowsArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_rows == 0 or args.vocab_size == 0) return error.ZeroDimension;
        if (args.num_rows > MAX_LOGIT_ROWS) return error.InvalidArgument;
        if (args.top_k == 0 or args.top_k > MAX_TOP_K or args.top_k > args.vocab_size) {
            return error.UnsupportedTopK;
        }

        const pipe, const vec_align: u64 = switch (args.logits_dtype) {
            .f16 => .{ self.topk_logits_rows_f16, 8 },
            .f32 => .{ self.topk_logits_rows_f32, 16 },
        };
        const can_vec4 = (args.logits.offset % vec_align) == 0 and (args.vocab_size % 4) == 0;
        const params: TopKLogitsRowsParams = .{
            .num_rows = args.num_rows,
            .vocab_size = args.vocab_size,
            .top_k = args.top_k,
            .aligned_vec4 = if (can_vec4) 1 else 0,
        };
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.logits, args.out_candidates },
            std.mem.asBytes(&params),
            .{ args.num_rows, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes non-deduplicated per-slot 3-bit / 4-bit MoE gate+up+SwiGLU
    /// (`Spike S5` comparison path against CSR-deduplicated `encodeGateUpSwiglu`).
    pub fn encodeGateUpSwigluPerSlot(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: GateUpSwigluArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0 or
            args.hidden_size == 0 or args.intermediate_size == 0)
        {
            return error.ZeroDimension;
        }
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExpertCount;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;
        if (args.m > args.workspace.spec.max_tokens or
            args.num_experts > args.workspace.spec.num_experts or
            args.top_k > args.workspace.spec.top_k or
            args.intermediate_size > args.workspace.spec.intermediate_size)
        {
            return error.WorkspaceTooSmall;
        }

        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.gate_up_swiglu_perslot_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.gate_up_swiglu_perslot_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.hidden_size), try Ops.rowWeightBytes(args.hidden_size), self.gate_up_swiglu_perslot_q2 };
            },
        };
        const params: MoePerSlotParams = .{
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
            .num_groups = @intCast(num_groups),
            .row_bytes = @intCast(row_bytes),
            .m = args.m,
            .top_k = args.top_k,
            .swiglu_limit = args.swiglu_limit,
        };
        const groups_x: u64 = (@as(u64, args.intermediate_size) + 7) / 8;
        const total_slots: u64 = @as(u64, args.m) * args.top_k;
        const ws = args.workspace;
        const wb = args.workspace_buf;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.x,
                args.weights_base,
                args.expert_table,
                .{ .buffer = wb, .offset = ws.scratch_topk_experts_off },
                .{ .buffer = wb, .offset = ws.inter_off },
            },
            std.mem.asBytes(&params),
            .{ groups_x, total_slots, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes non-deduplicated per-slot 3-bit / 4-bit MoE down projection
    /// (`Spike S5` comparison path against CSR-deduplicated `encodeDownProj`).
    pub fn encodeDownProjPerSlot(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: DownProjArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0 or
            args.hidden_size == 0 or args.intermediate_size == 0)
        {
            return error.ZeroDimension;
        }
        if (args.num_experts > MAX_EXPERTS) return error.UnsupportedExpertCount;
        if (args.top_k > MAX_TOP_K or args.top_k > args.num_experts) return error.UnsupportedTopK;
        if (args.m > args.workspace.spec.max_tokens or
            args.num_experts > args.workspace.spec.num_experts or
            args.top_k > args.workspace.spec.top_k or
            args.hidden_size > args.workspace.spec.hidden_size or
            args.intermediate_size > args.workspace.spec.intermediate_size)
        {
            return error.WorkspaceTooSmall;
        }

        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.intermediate_size), try Ops.rowWeightBytes(args.intermediate_size), self.down_proj_perslot_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.intermediate_size), try Ops.rowWeightBytes(args.intermediate_size), self.down_proj_perslot_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.intermediate_size), try Ops.rowWeightBytes(args.intermediate_size), self.down_proj_perslot_q2 };
            },
        };
        const params: MoePerSlotParams = .{
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
            .num_groups = @intCast(num_groups),
            .row_bytes = @intCast(row_bytes),
            .m = args.m,
            .top_k = args.top_k,
        };
        const groups_x: u64 = (@as(u64, args.hidden_size) + 15) / 16;
        const total_slots: u64 = @as(u64, args.m) * args.top_k;
        const ws = args.workspace;
        const wb = args.workspace_buf;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                .{ .buffer = wb, .offset = ws.inter_off },
                args.weights_base,
                args.expert_table,
                .{ .buffer = wb, .offset = ws.scratch_topk_experts_off },
                .{ .buffer = wb, .offset = ws.scratch_topk_weights_off },
                .{ .buffer = wb, .offset = ws.down_out_off },
            },
            std.mem.asBytes(&params),
            .{ groups_x, total_slots, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes a complete non-deduplicated per-slot MoE FFN layer (Spike S5).
    pub fn encodeMoELayerPerSlot(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: MoELayerArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try self.encodeRouting(frame, args, insert_barrier_before);

        try self.encodeGateUpSwigluPerSlot(frame, bits, .{
            .x = args.x,
            .weights_base = args.weights_base,
            .expert_table = args.expert_table,
            .workspace = args.workspace,
            .workspace_buf = args.workspace_buf,
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
        }, true);

        try self.encodeDownProjPerSlot(frame, bits, .{
            .weights_base = args.weights_base,
            .expert_table = args.expert_table,
            .workspace = args.workspace,
            .workspace_buf = args.workspace_buf,
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .hidden_size = args.hidden_size,
            .intermediate_size = args.intermediate_size,
        }, true);

        try self.encodeReduce(frame, .{
            .workspace = args.workspace,
            .workspace_buf = args.workspace_buf,
            .residual = args.residual,
            .out = args.out,
            .m = args.m,
            .hidden_size = args.hidden_size,
            .top_k = args.top_k,
        }, true);
    }

    /// Encodes a strided 2D `f16` copy (`ami_copy_strided_f16`). Zero heap allocation.
    pub fn encodeCopyStrided(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: CopyStridedArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_rows == 0 or args.copy_cols == 0 or args.src_stride == 0 or args.dst_stride == 0) {
            return error.ZeroDimension;
        }
        if (args.src_col_off + args.copy_cols > args.src_stride or
            args.dst_col_off + args.copy_cols > args.dst_stride)
        {
            return error.InvalidArgument;
        }
        const params: CopyStridedParams = .{
            .num_rows = args.num_rows,
            .copy_cols = args.copy_cols,
            .src_stride = args.src_stride,
            .src_col_off = args.src_col_off,
            .dst_stride = args.dst_stride,
            .dst_col_off = args.dst_col_off,
        };
        const groups_x: u64 = (@as(u64, args.copy_cols) + 255) / 256;
        try frame.encodeThreadgroups(
            self.copy_strided_f16,
            &.{ args.src, args.dst },
            std.mem.asBytes(&params),
            .{ groups_x, args.num_rows, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes an indexed `f16` row gather (`ami_gather_rows_f16`). Zero heap allocation.
    pub fn encodeGatherRows(
        self: MoePipelines,
        frame: *metal.Compute.Frame,
        args: GatherRowsArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.count == 0 or args.row_elems == 0) return error.ZeroDimension;
        const params: GatherRowsParams = .{
            .count = args.count,
            .row_elems = args.row_elems,
        };
        const groups_x: u64 = (@as(u64, args.row_elems) + 255) / 256;
        try frame.encodeThreadgroups(
            self.gather_rows_f16,
            &.{ args.src, args.indices, args.dst },
            std.mem.asBytes(&params),
            .{ groups_x, args.count, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }
};

/// Algorithmic minimum weight + scale_bias bytes read for a single MoE expert
/// (`gate + up + down`) stored in `dtype` (G13 / ADR-0017).
pub fn singleExpertBytesForDType(
    dtype: quant.WeightDType,
    hidden_size: u32,
    intermediate_size: u32,
) quant.ShapeError!u64 {
    return switch (dtype) {
        .f16, .bf16 => blk: {
            const elem_pairs = std.math.mul(u64, hidden_size, intermediate_size) catch return error.Overflow;
            // 3 matrices (gate, up, down) * 2 bytes/elem = 6 bytes/elem pair.
            break :blk std.math.mul(u64, elem_pairs, 6) catch return error.Overflow;
        },
        .q4_affine => try expertLayerBytes(.q4, 1, hidden_size, intermediate_size),
        .q3_affine => try expertLayerBytes(.q3, 1, hidden_size, intermediate_size),
        .q2_affine => try expertLayerBytes(.q2, 1, hidden_size, intermediate_size),
    };
}

/// Algorithmic minimum weight + scale_bias bytes read by one deduplicated MoE
/// layer when `unique_experts` distinct experts are active (`gate + up + down`).
pub fn expertLayerBytes(
    bits: quant.QuantBits,
    unique_experts: u32,
    hidden_size: u32,
    intermediate_size: u32,
) quant.ShapeError!u64 {
    if (unique_experts == 0) return 0;
    const per_expert: u64 = switch (bits) {
        .q4 => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
            const gu_w = try Ops.packedWeightBytes(intermediate_size, hidden_size);
            const gu_sb = try Ops.scaleBiasBytes(intermediate_size, hidden_size);
            const dn_w = try Ops.packedWeightBytes(hidden_size, intermediate_size);
            const dn_sb = try Ops.scaleBiasBytes(hidden_size, intermediate_size);
            break :blk @as(u64, 2) * (gu_w + gu_sb) + dn_w + dn_sb;
        },
        .q3 => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
            const gu_w = try Ops.packedWeightBytes(intermediate_size, hidden_size);
            const gu_sb = try Ops.scaleBiasBytes(intermediate_size, hidden_size);
            const dn_w = try Ops.packedWeightBytes(hidden_size, intermediate_size);
            const dn_sb = try Ops.scaleBiasBytes(hidden_size, intermediate_size);
            break :blk @as(u64, 2) * (gu_w + gu_sb) + dn_w + dn_sb;
        },
        .q2 => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
            const gu_w = try Ops.packedWeightBytes(intermediate_size, hidden_size);
            const gu_sb = try Ops.scaleBiasBytes(intermediate_size, hidden_size);
            const dn_w = try Ops.packedWeightBytes(hidden_size, intermediate_size);
            const dn_sb = try Ops.scaleBiasBytes(hidden_size, intermediate_size);
            break :blk @as(u64, 2) * (gu_w + gu_sb) + dn_w + dn_sb;
        },
    };
    return std.math.mul(u64, per_expert, unique_experts) catch return error.Overflow;
}

// ---------------------------------------------------------------------------
// CPU fp32 reference implementations (used for tests & model verification)
// ---------------------------------------------------------------------------

pub fn siluF32(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub fn siluMulRef(gate: []const f16, up: []const f16, out: []f32) void {
    std.debug.assert(gate.len == up.len and gate.len == out.len);
    for (0..gate.len) |i| {
        out[i] = siluF32(@floatCast(gate[i])) * @as(f32, @floatCast(up[i]));
    }
}

pub fn residualAddRef(x: []const f16, delta: []const f16, out: []f32) void {
    std.debug.assert(x.len == delta.len and x.len == out.len);
    for (0..x.len) |i| {
        out[i] = @as(f32, @floatCast(x[i])) + @as(f32, @floatCast(delta[i]));
    }
}

pub fn argmaxRef(logits: []const f16) u32 {
    std.debug.assert(logits.len > 0);
    var best_v: f32 = -std.math.inf(f32);
    var best_i: u32 = 0xFFFF_FFFF;
    for (logits, 0..) |lv, idx| {
        const v: f32 = @floatCast(lv);
        const ui: u32 = @intCast(idx);
        if (v > best_v or (v == best_v and ui < best_i)) {
            best_v = v;
            best_i = ui;
        }
    }
    return if (best_i == 0xFFFF_FFFF) 0 else best_i;
}

/// CPU reference greedy argmax over `[num_rows, vocab_size]` `f16` or `f32` logits.
pub fn argmaxRowsRef(
    logits_dtype: LogitsDType,
    logits_raw: []const u8,
    out_token_ids: []u32,
    num_rows: usize,
    vocab_size: usize,
) void {
    std.debug.assert(vocab_size > 0 and out_token_ids.len >= num_rows);
    for (0..num_rows) |row| {
        var best_v: f32 = -std.math.inf(f32);
        var best_i: u32 = 0xFFFF_FFFF;
        switch (logits_dtype) {
            .f16 => {
                const slice: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, logits_raw));
                const r = slice[row * vocab_size ..][0..vocab_size];
                for (r, 0..) |lv, idx| {
                    const v: f32 = @floatCast(lv);
                    const ui: u32 = @intCast(idx);
                    if (v > best_v or (v == best_v and ui < best_i)) {
                        best_v = v;
                        best_i = ui;
                    }
                }
            },
            .f32 => {
                const slice: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
                const r = slice[row * vocab_size ..][0..vocab_size];
                for (r, 0..) |v, idx| {
                    const ui: u32 = @intCast(idx);
                    if (v > best_v or (v == best_v and ui < best_i)) {
                        best_v = v;
                        best_i = ui;
                    }
                }
            },
        }
        out_token_ids[row] = if (best_i == 0xFFFF_FFFF) 0 else best_i;
    }
}

/// CPU fp64/fp32 reference row-wise softmax + exact top-K extraction
/// (`[num_rows, vocab_size]` -> `[num_rows, top_k]` `TopKCandidate`).
pub fn topKLogitsRowsRef(
    logits_dtype: LogitsDType,
    logits_raw: []const u8,
    out_cands: []TopKCandidate,
    num_rows: usize,
    vocab_size: usize,
    top_k: usize,
) void {
    std.debug.assert(vocab_size > 0 and top_k > 0 and top_k <= MAX_TOP_K and top_k <= vocab_size);
    std.debug.assert(out_cands.len >= num_rows * top_k);

    for (0..num_rows) |row| {
        var top_v: [MAX_TOP_K]f32 = @splat(-std.math.inf(f32));
        var top_i: [MAX_TOP_K]u32 = @splat(0xFFFF_FFFF);

        var max_v: f64 = -std.math.inf(f64);
        for (0..vocab_size) |idx| {
            const v: f32 = switch (logits_dtype) {
                .f16 => @floatCast(@as([]const f16, @alignCast(std.mem.bytesAsSlice(f16, logits_raw)))[row * vocab_size + idx]),
                .f32 => @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, logits_raw)))[row * vocab_size + idx],
            };
            const fv: f64 = v;
            if (fv > max_v) max_v = fv;

            const ui: u32 = @intCast(idx);
            if (v > top_v[top_k - 1] or (v == top_v[top_k - 1] and ui < top_i[top_k - 1])) {
                var pos: usize = top_k - 1;
                while (pos > 0) : (pos -= 1) {
                    if (v > top_v[pos - 1] or (v == top_v[pos - 1] and ui < top_i[pos - 1])) {
                        top_v[pos] = top_v[pos - 1];
                        top_i[pos] = top_i[pos - 1];
                    } else break;
                }
                top_v[pos] = v;
                top_i[pos] = ui;
            }
        }

        var sum_exp: f64 = 0.0;
        for (0..vocab_size) |idx| {
            const v: f32 = switch (logits_dtype) {
                .f16 => @floatCast(@as([]const f16, @alignCast(std.mem.bytesAsSlice(f16, logits_raw)))[row * vocab_size + idx]),
                .f32 => @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, logits_raw)))[row * vocab_size + idx],
            };
            if (std.math.isFinite(v)) {
                sum_exp += @exp(@as(f64, v) - max_v);
            }
        }
        const log_sum: f64 = @log(sum_exp);
        for (0..top_k) |k_idx| {
            const lp: f64 = (@as(f64, top_v[k_idx]) - max_v) - log_sum;
            out_cands[row * top_k + k_idx] = .{
                .token_id = top_i[k_idx],
                .prob = @floatCast(@exp(lp)),
                .log_prob = @floatCast(lp),
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Spike S5: Unique Active Experts vs Tree Size & Dedup vs Per-Slot MoE (ADR-0009)
// ---------------------------------------------------------------------------

pub const DEFAULT_SPIKE_S5_M = [_]u32{ 1, 2, 4, 8, 16, 32 };

pub const SpikeS5Routing = enum {
    correlated,
    uniform,

    pub fn label(self: SpikeS5Routing) []const u8 {
        return switch (self) {
            .correlated => "correlated",
            .uniform => "uniform",
        };
    }
};

pub const SpikeS5Config = struct {
    bits: quant.QuantBits = .q3,
    num_experts: u32 = 128,
    top_k: u32 = 8,
    hidden_size: u32 = 2048,
    intermediate_size: u32 = 768,
    m_values: []const u32 = &DEFAULT_SPIKE_S5_M,
    /// Number of shifted routing banks spanning all experts so the active DRAM
    /// working set exceeds M4 L2/SLC (>= 64 MiB).
    num_routing_banks: u32 = 16,
    min_working_set_bytes: usize = 64 * 1024 * 1024,
    dvfs_warmup_batches: u32 = 12,
    warmup_batches: u32 = 4,
    timed_batches: u32 = 6,
    dispatches_per_batch: u32 = 8,
    timeout_ms: u64 = 30_000,
};

pub const SpikeS5CaseResult = struct {
    routing: SpikeS5Routing,
    m: u32,
    mean_unique_experts: f64,
    total_slots: u32,
    unique_bytes_per_dispatch: f64,
    slot_bytes_per_dispatch: u64,
    dedup_latency_us: f64,
    perslot_latency_us: f64,
    /// Effective bandwidth on unique expert weight bytes for CSR-deduplicated MoE (GB/s).
    dedup_beta_gbs: f64,
    /// Effective bandwidth on unique expert weight bytes for per-slot MoE (GB/s, apples-to-apples).
    perslot_beta_unique_gbs: f64,
    /// Raw slot throughput for per-slot MoE (`M * top_k * expert_bytes / elapsed_ns`, GB/s).
    perslot_beta_raw_gbs: f64,
    /// `perslot_latency_us / dedup_latency_us` (> 1.0 means dedup is faster).
    dedup_speedup_vs_perslot: f64,
    bytes_saved_pct: f64,
};

pub const SpikeS5Error = error{
    InvalidConfig,
    NonFiniteOutput,
    OutOfMemory,
} || EncodeError || WorkspaceError || vmstat.Error || vmstat.SwapError || sysinfo.Error ||
    metal.WrapError || arena_mod.WiredArena.InitError;

pub const SpikeS5Report = struct {
    config: SpikeS5Config,
    pool_bytes: u64,
    results: []SpikeS5CaseResult,
    /// Calibrated CSR-deduplicated MoE effective bandwidth (GB/s) for `D7` pruning.
    beta_dedup_gbs: f64,
    /// Per-tree-size calibrated deduplicated MoE bandwidth table `[0..64]` (GB/s)
    /// ready for `PruneCostModel.beta_gbs_by_m` (ADR-0009).
    beta_dedup_by_m: [65]f32,
    /// Per-tree-size per-slot MoE effective unique bandwidth table `[0..64]` (GB/s).
    beta_perslot_by_m: [65]f32,
    /// Smallest tested `M` where CSR-deduplicated beats per-slot under correlated routing (0 if never).
    crossover_m_correlated: u32,
    /// Smallest tested `M` where CSR-deduplicated beats per-slot under uniform routing (0 if never).
    crossover_m_uniform: u32,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn deinit(self: *SpikeS5Report, allocator: std.mem.Allocator) void {
        allocator.free(self.results);
        self.results = &.{};
    }

    pub fn write(self: *const SpikeS5Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        const bits_label: []const u8 = switch (self.config.bits) {
            .q2 => "q2",
            .q3 => "q3",
            .q4 => "q4",
        };
        try out.print(
            \\Spike S5 Unique Active Experts vs Tree Size & CSR-Dedup vs Per-Slot MoE
            \\s5.geometry                      {s}, E={d}, top_k={d}, H={d}, I={d}, pool={d:.1} MiB
            \\s5.beta_dedup_gbs                {d:.2}
            \\s5.crossover_m.correlated        {d}
            \\s5.crossover_m.uniform           {d}
            \\routing      M    |U(T)|  slots  saved%  dedup(us)  slot(us)  dedup_GB/s  slot_uGB/s  slot_rGB/s  speedup
            \\
        , .{
            bits_label,
            self.config.num_experts,
            self.config.top_k,
            self.config.hidden_size,
            self.config.intermediate_size,
            @as(f64, @floatFromInt(self.pool_bytes)) / (1024.0 * 1024.0),
            self.beta_dedup_gbs,
            self.crossover_m_correlated,
            self.crossover_m_uniform,
        });
        for (self.results) |r| {
            try out.print(
                \\{s: <11} {d: >2}  {d: >7.1}  {d: >5}  {d: >6.1}  {d: >9.2}  {d: >8.2}  {d: >10.2}  {d: >10.2}  {d: >10.2}  {d: >6.2}x
                \\
            , .{
                r.routing.label(),
                r.m,
                r.mean_unique_experts,
                r.total_slots,
                r.bytes_saved_pct,
                r.dedup_latency_us,
                r.perslot_latency_us,
                r.dedup_beta_gbs,
                r.perslot_beta_unique_gbs,
                r.perslot_beta_raw_gbs,
                r.dedup_speedup_vs_perslot,
            });
        }
        try out.print(
            \\s5.swapouts_delta                {d}
            \\s5.pageouts_delta                {d}
            \\
        , .{
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
        });
    }
};

fn splitMix64(state: *u64) u64 {
    state.* +%= 0x9E3779B97F4A7C15;
    var z = state.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

fn randUnitF32(state: *u64) f32 {
    const bits: u32 = @truncate(splitMix64(state) >> 40); // 24-bit mantissa
    return @as(f32, @floatFromInt(bits)) / 16_777_216.0;
}

fn fillSpikeS5RoutingBank(
    routing: SpikeS5Routing,
    logits: []f32,
    m: u32,
    num_experts: u32,
    bank: usize,
) void {
    var rng: u64 = 0xA5A5_1234_5678_0001 ^ (@as(u64, bank) *% 0x9E37_79B9) ^ (@as(u64, m) << 32) ^
        (@as(u64, @backingInt(routing)) << 48);
    const shift: u32 = @intCast((bank * @max(@as(usize, 1), num_experts / 16)) % num_experts);

    for (0..m) |tok| {
        // Approximate tree depth for correlated branch drift: log2(tok + 1).
        const depth_f: f32 = @floatFromInt(31 - @clz(@as(u32, @intCast(tok + 1))));
        for (0..num_experts) |e| {
            const val: f32 = switch (routing) {
                .uniform => randUnitF32(&rng) * 4.0 - 2.0,
                .correlated => blk: {
                    const rank: u32 = (@as(u32, @intCast(e)) + num_experts - shift) % num_experts;
                    const zipf_base: f32 = 3.5 / (1.0 + 0.28 * @as(f32, @floatFromInt(rank)));
                    // Centered triangular-like noise (sum of 3 uniforms - 1.5).
                    const n_raw = (randUnitF32(&rng) + randUnitF32(&rng) + randUnitF32(&rng)) - 1.5;
                    const drift = 0.45 + 0.08 * depth_f;
                    break :blk zipf_base + n_raw * drift;
                },
            };
            logits[tok * num_experts + e] = val;
        }
    }
}

fn countUniqueExpertsCpu(logits: []const f32, m: u32, num_experts: u32, top_k: u32) u32 {
    var seen: [MAX_EXPERTS]bool = @splat(false);
    var vals: [MAX_EXPERTS]f32 = undefined;
    var unique_cnt: u32 = 0;

    for (0..m) |tok| {
        const row = logits[tok * num_experts ..][0..num_experts];
        @memcpy(vals[0..num_experts], row);
        for (0..top_k) |_| {
            var best_v: f32 = -std.math.inf(f32);
            var best_e: u32 = 0xFFFF_FFFF;
            for (0..num_experts) |e| {
                const ue: u32 = @intCast(e);
                if (vals[e] > best_v or (vals[e] == best_v and ue < best_e)) {
                    best_v = vals[e];
                    best_e = ue;
                }
            }
            if (best_e < num_experts) {
                vals[best_e] = -std.math.inf(f32);
                if (!seen[best_e]) {
                    seen[best_e] = true;
                    unique_cnt += 1;
                }
            }
        }
    }
    return unique_cnt;
}

fn fillInterpolatedBetaTable(table: *[65]f32, m_vals: []const u32, betas: []const f64) void {
    if (m_vals.len == 0) {
        @memset(table, 60.0);
        return;
    }
    for (0..65) |m_idx| {
        const m_u: u32 = @intCast(@max(m_idx, 1));
        if (m_u <= m_vals[0]) {
            table[m_idx] = @floatCast(betas[0]);
            continue;
        }
        if (m_u >= m_vals[m_vals.len - 1]) {
            table[m_idx] = @floatCast(betas[m_vals.len - 1]);
            continue;
        }
        var found = false;
        for (0..m_vals.len - 1) |i| {
            if (m_u >= m_vals[i] and m_u <= m_vals[i + 1]) {
                const m0: f64 = @floatFromInt(m_vals[i]);
                const m1: f64 = @floatFromInt(m_vals[i + 1]);
                const t: f64 = if (m1 > m0) (@as(f64, @floatFromInt(m_u)) - m0) / (m1 - m0) else 0.0;
                table[m_idx] = @floatCast(betas[i] + t * (betas[i + 1] - betas[i]));
                found = true;
                break;
            }
        }
        if (!found) {
            table[m_idx] = @floatCast(betas[m_vals.len - 1]);
        }
    }
}

/// Runs Spike S5 on real Apple Silicon hardware (ADR-0009):
/// a) Unique active experts per layer `|U(T)|` vs tree size `M` under correlated
///    and uniform routing, and
/// b) Effective bandwidth `beta` (GB/s) and latency (us) of CSR-deduplicated vs
///    non-deduplicated per-slot MoE over rotating wired pools (`>= 64 MiB`,
///    `swapouts_delta = 0`), returning calibrated `beta_dedup_gbs` and `[65]f32`
///    lookup tables for `D7` expert-byte-budget pruning.
pub fn runSpikeS5(allocator: std.mem.Allocator, config: SpikeS5Config) SpikeS5Error!SpikeS5Report {
    if (config.m_values.len == 0 or config.num_experts == 0 or config.num_experts > MAX_EXPERTS or
        config.top_k == 0 or config.top_k > MAX_TOP_K or config.top_k > config.num_experts or
        config.hidden_size == 0 or config.intermediate_size == 0 or
        config.num_routing_banks == 0 or config.timed_batches == 0 or config.dispatches_per_batch == 0)
    {
        return error.InvalidConfig;
    }

    var max_m: u32 = 0;
    for (config.m_values) |mv| {
        if (mv == 0 or mv > MAX_LOGIT_ROWS) return error.InvalidConfig;
        if (mv > max_m) max_m = mv;
    }

    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const dev_info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, dev_info, .{});

    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const vm_before = try vmstat.Snapshot.capture();

    // Compute per-expert quantized blob sizes (16 KiB-aligned per projection).
    const gu_w_bytes: usize, const gu_sb_bytes: usize, const dn_w_bytes: usize, const dn_sb_bytes: usize = switch (config.bits) {
        .q2 => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
            break :blk .{
                try Ops.packedWeightBytes(config.intermediate_size, config.hidden_size),
                try Ops.scaleBiasBytes(config.intermediate_size, config.hidden_size),
                try Ops.packedWeightBytes(config.hidden_size, config.intermediate_size),
                try Ops.scaleBiasBytes(config.hidden_size, config.intermediate_size),
            };
        },
        .q3 => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
            break :blk .{
                try Ops.packedWeightBytes(config.intermediate_size, config.hidden_size),
                try Ops.scaleBiasBytes(config.intermediate_size, config.hidden_size),
                try Ops.packedWeightBytes(config.hidden_size, config.intermediate_size),
                try Ops.scaleBiasBytes(config.hidden_size, config.intermediate_size),
            };
        },
        .q4 => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
            break :blk .{
                try Ops.packedWeightBytes(config.intermediate_size, config.hidden_size),
                try Ops.scaleBiasBytes(config.intermediate_size, config.hidden_size),
                try Ops.packedWeightBytes(config.hidden_size, config.intermediate_size),
                try Ops.scaleBiasBytes(config.hidden_size, config.intermediate_size),
            };
        },
    };

    const gu_sb_off = amf.align64(gu_w_bytes);
    const gu_blob_bytes = try page.alignUp(gu_sb_off + gu_sb_bytes);
    const dn_sb_off = amf.align64(dn_w_bytes);
    const dn_blob_bytes = try page.alignUp(dn_sb_off + dn_sb_bytes);
    const per_expert_bytes = 2 * gu_blob_bytes + dn_blob_bytes;
    const single_pool_bytes = @as(usize, config.num_experts) * per_expert_bytes;
    const num_pools: usize = if (config.min_working_set_bytes == 0)
        1
    else
        @max(@as(usize, 1), (config.min_working_set_bytes + single_pool_bytes - 1) / single_pool_bytes);
    const total_pool_bytes = single_pool_bytes * num_pools;

    const ws_spec: WorkspaceSpec = .{
        .max_tokens = max_m,
        .num_experts = config.num_experts,
        .top_k = config.top_k,
        .hidden_size = config.hidden_size,
        .intermediate_size = config.intermediate_size,
    };
    const ws_bytes = try MoeWorkspace.requiredBytes(ws_spec);
    const table_bytes = amf.align64(@as(usize, config.num_experts) * @sizeOf(ExpertWeights));
    const all_tables_bytes = table_bytes * num_pools;
    const x_bytes = amf.align64(@as(usize, max_m) * config.hidden_size * @sizeOf(f16));
    const bank_logits_bytes = amf.align64(@as(usize, max_m) * config.num_experts * @sizeOf(f32));
    const all_logits_bytes = bank_logits_bytes * config.num_routing_banks;

    const arena_cap = try page.alignUp(
        total_pool_bytes + ws_bytes + all_tables_bytes + x_bytes * 2 + all_logits_bytes + 8 * page.APPLE_PAGE_SIZE,
    );
    var arena = try arena_mod.WiredArena.init(arena_cap, wired_budget);
    defer arena.deinit();

    // Reserve a non-zero page prefix so every expert offset is non-zero.
    _ = try arena.allocPages(page.APPLE_PAGE_SIZE);

    const tables_raw = try arena.allocAligned(all_tables_bytes, 64);
    const table_offs = try allocator.alloc(u64, num_pools);
    defer allocator.free(table_offs);

    // Initialize 4 template expert blobs once, then replicate across all experts/pools
    // to keep CPU setup fast while touching distinct physical wired DRAM pages.
    var tmpl_g: [4][]u8 = undefined;
    var tmpl_u: [4][]u8 = undefined;
    var tmpl_d: [4][]u8 = undefined;

    for (0..num_pools) |p_idx| {
        const tbl_off_in_buf = p_idx * table_bytes;
        const tbl_slice: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(
            ExpertWeights,
            tables_raw[tbl_off_in_buf .. tbl_off_in_buf + @as(usize, config.num_experts) * @sizeOf(ExpertWeights)],
        ));
        table_offs[p_idx] = arena.offsetOf(tables_raw.ptr) + tbl_off_in_buf;

        for (0..config.num_experts) |e| {
            const g_blob = try arena.allocPages(gu_blob_bytes);
            const u_blob = try arena.allocPages(gu_blob_bytes);
            const d_blob = try arena.allocPages(dn_blob_bytes);

            if (p_idx == 0 and e < 4) {
                @memset(g_blob, 0);
                @memset(u_blob, 0);
                @memset(d_blob, 0);
                switch (config.bits) {
                    .q2 => {
                        try fillSyntheticQuantMatrix(.{ .bits = .q2, .group_size = 64 }, g_blob[0..gu_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, g_blob[gu_sb_off .. gu_sb_off + gu_sb_bytes])), config.intermediate_size, config.hidden_size, e * 3 + 1);
                        try fillSyntheticQuantMatrix(.{ .bits = .q2, .group_size = 64 }, u_blob[0..gu_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, u_blob[gu_sb_off .. gu_sb_off + gu_sb_bytes])), config.intermediate_size, config.hidden_size, e * 3 + 2);
                        try fillSyntheticQuantMatrix(.{ .bits = .q2, .group_size = 64 }, d_blob[0..dn_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, d_blob[dn_sb_off .. dn_sb_off + dn_sb_bytes])), config.hidden_size, config.intermediate_size, e * 3 + 3);
                    },
                    .q3 => {
                        try fillSyntheticQuantMatrix(.{ .bits = .q3, .group_size = 64 }, g_blob[0..gu_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, g_blob[gu_sb_off .. gu_sb_off + gu_sb_bytes])), config.intermediate_size, config.hidden_size, e * 3 + 1);
                        try fillSyntheticQuantMatrix(.{ .bits = .q3, .group_size = 64 }, u_blob[0..gu_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, u_blob[gu_sb_off .. gu_sb_off + gu_sb_bytes])), config.intermediate_size, config.hidden_size, e * 3 + 2);
                        try fillSyntheticQuantMatrix(.{ .bits = .q3, .group_size = 64 }, d_blob[0..dn_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, d_blob[dn_sb_off .. dn_sb_off + dn_sb_bytes])), config.hidden_size, config.intermediate_size, e * 3 + 3);
                    },
                    .q4 => {
                        try fillSyntheticQuantMatrix(.{ .bits = .q4, .group_size = 64 }, g_blob[0..gu_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, g_blob[gu_sb_off .. gu_sb_off + gu_sb_bytes])), config.intermediate_size, config.hidden_size, e * 3 + 1);
                        try fillSyntheticQuantMatrix(.{ .bits = .q4, .group_size = 64 }, u_blob[0..gu_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, u_blob[gu_sb_off .. gu_sb_off + gu_sb_bytes])), config.intermediate_size, config.hidden_size, e * 3 + 2);
                        try fillSyntheticQuantMatrix(.{ .bits = .q4, .group_size = 64 }, d_blob[0..dn_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, d_blob[dn_sb_off .. dn_sb_off + dn_sb_bytes])), config.hidden_size, config.intermediate_size, e * 3 + 3);
                    },
                }
                tmpl_g[e] = g_blob;
                tmpl_u[e] = u_blob;
                tmpl_d[e] = d_blob;
            } else {
                const src = e & 3;
                @memcpy(g_blob, tmpl_g[src]);
                @memcpy(u_blob, tmpl_u[src]);
                @memcpy(d_blob, tmpl_d[src]);
            }

            const g_off = arena.offsetOf(g_blob.ptr);
            const u_off = arena.offsetOf(u_blob.ptr);
            const d_off = arena.offsetOf(d_blob.ptr);
            tbl_slice[e] = .{
                .gate_qweight_offset = g_off,
                .gate_sb_offset = g_off + gu_sb_off,
                .up_qweight_offset = u_off,
                .up_sb_offset = u_off + gu_sb_off,
                .down_qweight_offset = d_off,
                .down_sb_offset = d_off + dn_sb_off,
            };
        }
    }

    const ws = try MoeWorkspace.init(&arena, ws_spec);
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const out_raw = try arena.allocAligned(x_bytes, 64);
    const logits_banks_raw = try arena.allocAligned(all_logits_bytes, 64);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw[0 .. @as(usize, max_m) * config.hidden_size * @sizeOf(f16)]));
    const out_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw[0 .. @as(usize, max_m) * config.hidden_size * @sizeOf(f16)]));
    for (x_f16, 0..) |*v, idx| {
        const c: i32 = @as(i32, @intCast((idx * 17 + 11) % 31)) - 15;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.03125);
    }
    @memset(out_raw, 0);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    // Single frame in flight because all batches reuse one MoeWorkspace;
    // dispatches within each batch are serialized by intra-encoder barriers.
    var compute = try metal.Compute.initWithFrames(device, 1);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const x_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) };
    const out_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) };
    const w_base = metal.BufferBinding{ .buffer = gpu_buf, .offset = 0 };
    const logits_base_off = arena.offsetOf(logits_banks_raw.ptr);

    const routings = [_]SpikeS5Routing{ .correlated, .uniform };
    const results = try allocator.alloc(SpikeS5CaseResult, routings.len * config.m_values.len);
    errdefer allocator.free(results);

    const corr_dedup_betas = try allocator.alloc(f64, config.m_values.len);
    defer allocator.free(corr_dedup_betas);
    const corr_perslot_betas = try allocator.alloc(f64, config.m_values.len);
    defer allocator.free(corr_perslot_betas);

    const one_expert_bytes = try expertLayerBytes(config.bits, 1, config.hidden_size, config.intermediate_size);

    var first_case = true;
    var crossover_corr: u32 = 0;
    var crossover_unif: u32 = 0;

    for (routings, 0..) |routing, ri| {
        for (config.m_values, 0..) |m_val, mi| {
            // Populate all routing banks for this (routing, M) and compute exact mean |U(T)|.
            var total_u: u64 = 0;
            for (0..config.num_routing_banks) |b| {
                const off = b * bank_logits_bytes;
                const b_slice: []f32 = @alignCast(std.mem.bytesAsSlice(
                    f32,
                    logits_banks_raw[off .. off + @as(usize, m_val) * config.num_experts * @sizeOf(f32)],
                ));
                fillSpikeS5RoutingBank(routing, b_slice, m_val, config.num_experts, b);
                total_u += countUniqueExpertsCpu(b_slice, m_val, config.num_experts, config.top_k);
            }
            const mean_u = @as(f64, @floatFromInt(total_u)) / @as(f64, @floatFromInt(config.num_routing_banks));
            const total_slots = m_val * config.top_k;
            const unique_bytes_per_disp = mean_u * @as(f64, @floatFromInt(one_expert_bytes));
            const slot_bytes_per_disp = one_expert_bytes * total_slots;

            // Measure CSR-deduplicated vs per-slot back-to-back over identical banks.
            var dedup_ns: u64 = 0;
            var perslot_ns: u64 = 0;
            const total_disps: u64 = @as(u64, config.timed_batches) * config.dispatches_per_batch;

            inline for (.{ true, false }) |is_dedup| {
                const warmups = config.warmup_batches + if (first_case and is_dedup) config.dvfs_warmup_batches else 0;
                var bank_cursor: usize = 0;
                var wsig: u64 = 0;
                for (0..warmups) |_| {
                    var frame = try compute.beginFrame(config.timeout_ms);
                    errdefer frame.abort();
                    for (0..config.dispatches_per_batch) |d| {
                        const bank = bank_cursor % config.num_routing_banks;
                        const pool = bank_cursor % num_pools;
                        bank_cursor += 1;
                        const layer_args: MoELayerArgs = .{
                            .x = x_bind,
                            .router = .{ .logits = .{
                                .binding = .{ .buffer = gpu_buf, .offset = logits_base_off + @as(u64, bank) * bank_logits_bytes },
                                .dtype = .f32,
                            } },
                            .weights_base = w_base,
                            .expert_table = .{ .buffer = gpu_buf, .offset = table_offs[pool] },
                            .workspace = &ws,
                            .workspace_buf = gpu_buf,
                            .out = out_bind,
                            .m = m_val,
                            .num_experts = config.num_experts,
                            .top_k = config.top_k,
                            .hidden_size = config.hidden_size,
                            .intermediate_size = config.intermediate_size,
                            .norm_topk_prob = true,
                        };
                        if (is_dedup) {
                            try pipes.encodeMoELayer(&frame, config.bits, layer_args, d > 0);
                        } else {
                            try pipes.encodeMoELayerPerSlot(&frame, config.bits, layer_args, d > 0);
                        }
                    }
                    wsig = try frame.commit();
                }
                if (wsig != 0) try compute.wait(wsig, config.timeout_ms);

                var lsig: u64 = 0;
                const t0 = try vmstat.nowNs();
                for (0..config.timed_batches) |_| {
                    var frame = try compute.beginFrame(config.timeout_ms);
                    errdefer frame.abort();
                    for (0..config.dispatches_per_batch) |d| {
                        const bank = bank_cursor % config.num_routing_banks;
                        const pool = bank_cursor % num_pools;
                        bank_cursor += 1;
                        const layer_args: MoELayerArgs = .{
                            .x = x_bind,
                            .router = .{ .logits = .{
                                .binding = .{ .buffer = gpu_buf, .offset = logits_base_off + @as(u64, bank) * bank_logits_bytes },
                                .dtype = .f32,
                            } },
                            .weights_base = w_base,
                            .expert_table = .{ .buffer = gpu_buf, .offset = table_offs[pool] },
                            .workspace = &ws,
                            .workspace_buf = gpu_buf,
                            .out = out_bind,
                            .m = m_val,
                            .num_experts = config.num_experts,
                            .top_k = config.top_k,
                            .hidden_size = config.hidden_size,
                            .intermediate_size = config.intermediate_size,
                            .norm_topk_prob = true,
                        };
                        if (is_dedup) {
                            try pipes.encodeMoELayer(&frame, config.bits, layer_args, d > 0);
                        } else {
                            try pipes.encodeMoELayerPerSlot(&frame, config.bits, layer_args, d > 0);
                        }
                    }
                    lsig = try frame.commit();
                }
                try compute.wait(lsig, config.timeout_ms);
                const t1 = try vmstat.nowNs();

                if (!std.math.isFinite(out_f16[0])) return error.NonFiniteOutput;
                if (is_dedup) {
                    dedup_ns = t1 - t0;
                } else {
                    perslot_ns = t1 - t0;
                }
            }
            first_case = false;

            const dedup_us = (@as(f64, @floatFromInt(dedup_ns)) / @as(f64, @floatFromInt(total_disps))) / 1000.0;
            const perslot_us = (@as(f64, @floatFromInt(perslot_ns)) / @as(f64, @floatFromInt(total_disps))) / 1000.0;
            const dedup_beta = if (dedup_ns > 0) (unique_bytes_per_disp * @as(f64, @floatFromInt(total_disps))) / @as(f64, @floatFromInt(dedup_ns)) else 0.0;
            const perslot_beta_u = if (perslot_ns > 0) (unique_bytes_per_disp * @as(f64, @floatFromInt(total_disps))) / @as(f64, @floatFromInt(perslot_ns)) else 0.0;
            const perslot_beta_r = if (perslot_ns > 0) (@as(f64, @floatFromInt(slot_bytes_per_disp)) * @as(f64, @floatFromInt(total_disps))) / @as(f64, @floatFromInt(perslot_ns)) else 0.0;
            const speedup = if (dedup_us > 0.0) perslot_us / dedup_us else 1.0;
            const saved_pct = (1.0 - (mean_u / @as(f64, @floatFromInt(total_slots)))) * 100.0;

            results[ri * config.m_values.len + mi] = .{
                .routing = routing,
                .m = m_val,
                .mean_unique_experts = mean_u,
                .total_slots = total_slots,
                .unique_bytes_per_dispatch = unique_bytes_per_disp,
                .slot_bytes_per_dispatch = slot_bytes_per_disp,
                .dedup_latency_us = dedup_us,
                .perslot_latency_us = perslot_us,
                .dedup_beta_gbs = dedup_beta,
                .perslot_beta_unique_gbs = perslot_beta_u,
                .perslot_beta_raw_gbs = perslot_beta_r,
                .dedup_speedup_vs_perslot = speedup,
                .bytes_saved_pct = saved_pct,
            };

            if (routing == .correlated) {
                corr_dedup_betas[mi] = dedup_beta;
                corr_perslot_betas[mi] = perslot_beta_u;
                if (crossover_corr == 0 and dedup_us <= perslot_us) {
                    crossover_corr = m_val;
                }
            } else {
                if (crossover_unif == 0 and dedup_us <= perslot_us) {
                    crossover_unif = m_val;
                }
            }
        }
    }

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    var beta_dedup_by_m: [65]f32 = undefined;
    var beta_perslot_by_m: [65]f32 = undefined;
    fillInterpolatedBetaTable(&beta_dedup_by_m, config.m_values, corr_dedup_betas);
    fillInterpolatedBetaTable(&beta_perslot_by_m, config.m_values, corr_perslot_betas);

    // Representative calibrated scalar beta_dedup_gbs (at M=8 if present, else median index).
    var rep_beta = corr_dedup_betas[corr_dedup_betas.len / 2];
    for (config.m_values, 0..) |mv, mi| {
        if (mv == 8) {
            rep_beta = corr_dedup_betas[mi];
            break;
        }
    }

    return .{
        .config = config,
        .pool_bytes = total_pool_bytes,
        .results = results,
        .beta_dedup_gbs = rep_beta,
        .beta_dedup_by_m = beta_dedup_by_m,
        .beta_perslot_by_m = beta_perslot_by_m,
        .crossover_m_correlated = crossover_corr,
        .crossover_m_uniform = crossover_unif,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

pub const RouteRefResult = struct {
    num_unique: u32,
    topk_experts: []u32,
    topk_weights: []f32,
    unique_experts: []u32,
    expert_token_offsets: []u32,

    pub fn deinit(self: *RouteRefResult, allocator: std.mem.Allocator) void {
        allocator.free(self.topk_experts);
        allocator.free(self.topk_weights);
        allocator.free(self.unique_experts);
        allocator.free(self.expert_token_offsets);
    }
};

/// CPU fp32 reference softmax, top-k selection (ties broken by smallest
/// expert_id), optional top-k renormalization, and deduplicated expert list.
pub fn routeTopKRef(
    allocator: std.mem.Allocator,
    logits_f32: []const f32,
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: bool,
) !RouteRefResult {
    std.debug.assert(num_experts > 0 and num_experts <= MAX_EXPERTS);
    std.debug.assert(top_k > 0 and top_k <= num_experts and top_k <= MAX_TOP_K);
    std.debug.assert(logits_f32.len >= @as(usize, m) * num_experts);

    const assignments: usize = @as(usize, m) * top_k;
    const topk_experts = try allocator.alloc(u32, assignments);
    errdefer allocator.free(topk_experts);
    const topk_weights = try allocator.alloc(f32, assignments);
    errdefer allocator.free(topk_weights);
    const unique_experts = try allocator.alloc(u32, num_experts);
    errdefer allocator.free(unique_experts);
    const expert_token_offsets = try allocator.alloc(u32, num_experts + 1);
    errdefer allocator.free(expert_token_offsets);

    var counts: [MAX_EXPERTS]u32 = @splat(0);
    var probs: [MAX_EXPERTS]f32 = undefined;

    for (0..m) |tok| {
        const row = logits_f32[tok * num_experts ..][0..num_experts];
        var max_v: f32 = -std.math.inf(f32);
        for (row) |v| max_v = @max(max_v, v);
        var sum_exp: f32 = 0.0;
        for (0..num_experts) |e| {
            const ex = @exp(row[e] - max_v);
            probs[e] = ex;
            sum_exp += ex;
        }
        const inv_sum = 1.0 / sum_exp;
        for (0..num_experts) |e| probs[e] *= inv_sum;

        var w_sum: f32 = 0.0;
        for (0..top_k) |k_idx| {
            var best_v: f32 = -std.math.inf(f32);
            var best_e: u32 = 0xFFFF_FFFF;
            for (0..num_experts) |e| {
                const ue: u32 = @intCast(e);
                if (probs[e] > best_v or (probs[e] == best_v and ue < best_e)) {
                    best_v = probs[e];
                    best_e = ue;
                }
            }
            probs[best_e] = -std.math.inf(f32);
            topk_experts[tok * top_k + k_idx] = best_e;
            topk_weights[tok * top_k + k_idx] = best_v;
            w_sum += best_v;
            counts[best_e] += 1;
        }
        if (norm_topk_prob and w_sum > 0.0) {
            const renorm = 1.0 / w_sum;
            for (0..top_k) |k_idx| {
                topk_weights[tok * top_k + k_idx] *= renorm;
            }
        }
    }

    var u: u32 = 0;
    var cursor: u32 = 0;
    for (0..num_experts) |e| {
        if (counts[e] > 0) {
            unique_experts[u] = @intCast(e);
            expert_token_offsets[u] = cursor;
            cursor += counts[e];
            u += 1;
        }
    }
    expert_token_offsets[u] = cursor;

    return .{
        .num_unique = u,
        .topk_experts = topk_experts,
        .topk_weights = topk_weights,
        .unique_experts = unique_experts,
        .expert_token_offsets = expert_token_offsets,
    };
}

/// Full CPU fp32 reference for an affine-quantized MoE layer over an arena
/// byte buffer and `ExpertWeights` offset table.
pub fn moeLayerRef(
    comptime spec: quant.QuantSpec,
    allocator: std.mem.Allocator,
    x: []const f16,
    logits_f32: []const f32,
    weights_base: []const u8,
    expert_table: []const ExpertWeights,
    residual: ?[]const f16,
    out_f32: []f32,
    m: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    norm_topk_prob: bool,
) !RouteRefResult {
    const Ops = quant.QuantOps(spec);
    var route = try routeTopKRef(allocator, logits_f32, m, num_experts, top_k, norm_topk_prob);
    errdefer route.deinit(allocator);

    const gu_w_bytes = try Ops.packedWeightBytes(intermediate_size, hidden_size);
    const gu_sb_bytes = try Ops.scaleBiasBytes(intermediate_size, hidden_size);
    const dn_w_bytes = try Ops.packedWeightBytes(hidden_size, intermediate_size);
    const dn_sb_bytes = try Ops.scaleBiasBytes(hidden_size, intermediate_size);

    const gate_buf = try allocator.alloc(f32, intermediate_size);
    defer allocator.free(gate_buf);
    const up_buf = try allocator.alloc(f32, intermediate_size);
    defer allocator.free(up_buf);
    const inter_f16 = try allocator.alloc(f16, intermediate_size);
    defer allocator.free(inter_f16);
    const down_buf = try allocator.alloc(f32, hidden_size);
    defer allocator.free(down_buf);

    for (0..@as(usize, m) * hidden_size) |i| {
        out_f32[i] = if (residual) |res| @as(f32, @floatCast(res[i])) else 0.0;
    }

    for (0..m) |tok| {
        const x_tok = x[tok * hidden_size ..][0..hidden_size];
        const out_tok = out_f32[tok * hidden_size ..][0..hidden_size];
        for (0..top_k) |k_idx| {
            const e = route.topk_experts[tok * top_k + k_idx];
            const rw = route.topk_weights[tok * top_k + k_idx];
            const ew = expert_table[e];

            const gate_qw = weights_base[@intCast(ew.gate_qweight_offset)..][0..gu_w_bytes];
            const gate_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                quant.ScaleBias,
                weights_base[@intCast(ew.gate_sb_offset)..][0..gu_sb_bytes],
            ));
            const up_qw = weights_base[@intCast(ew.up_qweight_offset)..][0..gu_w_bytes];
            const up_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                quant.ScaleBias,
                weights_base[@intCast(ew.up_sb_offset)..][0..gu_sb_bytes],
            ));
            const down_qw = weights_base[@intCast(ew.down_qweight_offset)..][0..dn_w_bytes];
            const down_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                quant.ScaleBias,
                weights_base[@intCast(ew.down_sb_offset)..][0..dn_sb_bytes],
            ));

            try Ops.matmulRef(x_tok, gate_qw, gate_sb, gate_buf, 1, intermediate_size, hidden_size);
            try Ops.matmulRef(x_tok, up_qw, up_sb, up_buf, 1, intermediate_size, hidden_size);
            for (0..intermediate_size) |i| {
                inter_f16[i] = @floatCast(siluF32(gate_buf[i]) * up_buf[i]);
            }
            try Ops.matmulRef(inter_f16, down_qw, down_sb, down_buf, 1, hidden_size, intermediate_size);
            for (0..hidden_size) |i| {
                out_tok[i] += down_buf[i] * rw;
            }
        }
    }

    return route;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn makeTestArena(min_bytes: usize) !arena_mod.WiredArena {
    const cap = try page.alignUp(min_bytes + page.APPLE_PAGE_SIZE);
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = cap,
        .global_user_wire_limit = cap,
        .gpu_working_set_bytes = cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    return arena_mod.WiredArena.init(cap, budget);
}

fn fillSyntheticQuantMatrix(
    comptime spec: quant.QuantSpec,
    qweight: []u8,
    scale_bias: []quant.ScaleBias,
    n: usize,
    k: usize,
    seed: usize,
) !void {
    const Ops = quant.QuantOps(spec);
    const groups = try Ops.groupsPerRow(k);
    const row_bytes = try Ops.rowWeightBytes(k);
    const q_mask: u8 = (@as(u8, 1) << @intCast(spec.bits.bitWidth())) - 1;
    var codes: [spec.group_size]u8 = undefined;
    for (0..n) |row| {
        for (0..groups) |g| {
            for (0..spec.group_size) |j| {
                const v = (seed * 31 + row * 13 + g * 7 + j * 5 + 3) & q_mask;
                codes[j] = @intCast(v);
            }
            const g_off = row * row_bytes + g * Ops.groupWeightBytes();
            Ops.packGroup(&codes, qweight[g_off..][0..Ops.groupWeightBytes()]);
            const s: f32 = 0.02 + 0.005 * @as(f32, @floatFromInt((seed + row + g) % 7));
            const b: f32 = -0.08 - 0.02 * @as(f32, @floatFromInt((seed * 3 + row + g) % 5));
            scale_bias[row * groups + g] = .{ .scale = @floatCast(s), .bias = @floatCast(b) };
        }
    }
}

test "ABI structs, workspace byte sizing, and expertLayerBytes validate invariants" {
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(ExpertWeights));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(ScheduleHeader));

    // Qwen3-30B-A3B 3-bit expert geometry: hidden=2048, inter=768, 8 active experts.
    // gate/up: 768 * (2048*3/8 + 32*4) = 768 * (768 + 128) = 688,128 B each
    // down:    2048 * (768*3/8 + 12*4) = 2048 * (288 + 48) = 688,128 B
    // 1 expert = 2,064,384 B; 8 experts = 16,515,072 B.
    const bytes_8 = try expertLayerBytes(.q3, 8, 2048, 768);
    try std.testing.expectEqual(@as(u64, 16_515_072), bytes_8);
    try std.testing.expectEqual(@as(u64, 0), try expertLayerBytes(.q4, 0, 2048, 768));

    const ws_bytes = try MoeWorkspace.requiredBytes(.{
        .max_tokens = 8,
        .num_experts = 128,
        .top_k = 8,
        .hidden_size = 2048,
        .intermediate_size = 768,
    });
    try std.testing.expect(ws_bytes > 0);
    try std.testing.expectError(error.UnsupportedExpertCount, MoeWorkspace.requiredBytes(.{
        .max_tokens = 1,
        .num_experts = MAX_EXPERTS + 1,
        .top_k = 8,
        .hidden_size = 64,
        .intermediate_size = 64,
    }));
    try std.testing.expectError(error.UnsupportedTopK, MoeWorkspace.requiredBytes(.{
        .max_tokens = 1,
        .num_experts = 8,
        .top_k = 9,
        .hidden_size = 64,
        .intermediate_size = 64,
    }));
}

fn verifyRouteTopK(
    logits_dtype: LogitsDType,
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: bool,
    force_overlap: bool,
) !void {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const ws_spec: WorkspaceSpec = .{
        .max_tokens = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = 64,
        .intermediate_size = 64,
    };
    const logits_elem_bytes: usize = if (logits_dtype == .f16) @sizeOf(f16) else @sizeOf(f32);
    const logits_bytes: usize = @as(usize, m) * num_experts * logits_elem_bytes;
    var arena = try makeTestArena(try MoeWorkspace.requiredBytes(ws_spec) + logits_bytes + 512);
    defer arena.deinit();

    const ws = try MoeWorkspace.init(&arena, ws_spec);
    const logits_raw = try arena.allocAligned(logits_bytes, 64);

    const logits_ref = try gpa.alloc(f32, @as(usize, m) * num_experts);
    defer gpa.free(logits_ref);

    for (0..m) |tok| {
        for (0..num_experts) |e| {
            var v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((tok * 19 + e * 7) % 29)) - 14)) * 0.125;
            if (force_overlap) {
                // Deliberately bias all tokens toward a shared small pool of experts
                // plus 1-2 token-specific experts so U is guaranteed strictly < M * top_k.
                if (e < top_k) {
                    v = 4.0 - 0.1 * @as(f32, @floatFromInt(e)) + 0.05 * @as(f32, @floatFromInt(tok % 3));
                } else if (e == (top_k + (tok % 4)) % num_experts) {
                    v = 4.5;
                }
            }
            if (logits_dtype == .f16) {
                const h: f16 = @floatCast(v);
                const slice: []f16 = @alignCast(std.mem.bytesAsSlice(f16, logits_raw));
                slice[tok * num_experts + e] = h;
                logits_ref[tok * num_experts + e] = @floatCast(h);
            } else {
                const slice: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
                slice[tok * num_experts + e] = v;
                logits_ref[tok * num_experts + e] = v;
            }
        }
    }

    var ref = try routeTopKRef(gpa, logits_ref, m, num_experts, top_k, norm_topk_prob);
    defer ref.deinit(gpa);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeRouteTopK(&frame, logits_dtype, .{
        .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) },
        .workspace = &ws,
        .workspace_buf = gpu_buf,
        .m = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .norm_topk_prob = norm_topk_prob,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    // 1. Header & deduplication invariants.
    const u_got = ws.schedule_header.num_unique_experts;
    try std.testing.expectEqual(ref.num_unique, u_got);
    try std.testing.expectEqual(m * top_k, ws.schedule_header.total_assignments);
    try std.testing.expectEqual(m, ws.schedule_header.m);
    try std.testing.expectEqual(top_k, ws.schedule_header.top_k);
    if (force_overlap and m > 1) {
        try std.testing.expect(u_got < m * top_k);
    }

    // 2. Every unique expert appears strictly once in ascending order, offsets sum to M * top_k.
    try std.testing.expectEqual(@as(u32, 0), ws.expert_token_offsets[0]);
    try std.testing.expectEqual(m * top_k, ws.expert_token_offsets[u_got]);
    for (0..u_got) |ui| {
        const exp_id = ws.unique_experts[ui];
        try std.testing.expectEqual(ref.unique_experts[ui], exp_id);
        if (ui > 0) {
            try std.testing.expect(ws.unique_experts[ui - 1] < exp_id);
        }
        try std.testing.expectEqual(ref.expert_token_offsets[ui], ws.expert_token_offsets[ui]);
        try std.testing.expect(ws.expert_token_offsets[ui + 1] > ws.expert_token_offsets[ui]);
    }

    // 3. Reconstruct per-token (expert_id, weight) from the deduplicated CSR schedule
    //    and verify exact match with CPU fp32 reference.
    const seen = try gpa.alloc(bool, @as(usize, m) * top_k);
    defer gpa.free(seen);
    @memset(seen, false);

    for (0..u_got) |ui| {
        const exp_id = ws.unique_experts[ui];
        const s0 = ws.expert_token_offsets[ui];
        const s1 = ws.expert_token_offsets[ui + 1];
        for (s0..s1) |s| {
            const tok = ws.token_indices[s];
            const k_idx = ws.slot_k_indices[s];
            const rw = ws.routing_weights[s];
            try std.testing.expect(tok < m);
            try std.testing.expect(k_idx < top_k);
            const flat = @as(usize, tok) * top_k + k_idx;
            try std.testing.expect(!seen[flat]);
            seen[flat] = true;
            try std.testing.expectEqual(ref.topk_experts[flat], exp_id);
            try std.testing.expectApproxEqAbs(ref.topk_weights[flat], rw, 1e-5);
        }
    }
    for (seen) |ok| try std.testing.expect(ok);

    if (norm_topk_prob) {
        for (0..m) |tok| {
            var sum: f32 = 0.0;
            for (0..u_got) |ui| {
                for (ws.expert_token_offsets[ui]..ws.expert_token_offsets[ui + 1]) |s| {
                    if (ws.token_indices[s] == tok) sum += ws.routing_weights[s];
                }
            }
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
        }
    }
}

test "ami_moe_route_topk deduplicates shared experts across M=8 tokens and matches CPU fp32 reference" {
    // Qwen3-30B-A3B geometry: 128 experts, top_k=8 (compile-time specialized), M=1 and M=8 with shared experts.
    try verifyRouteTopK(.f32, 1, 128, 8, true, false);
    try verifyRouteTopK(.f16, 8, 128, 8, true, true);
    try verifyRouteTopK(.f32, 8, 128, 8, false, true);
    // Odd num_experts, dynamic top_k, and full 256-expert upper bound.
    try verifyRouteTopK(.f32, 5, 17, 3, true, true);
    try verifyRouteTopK(.f16, 8, 256, 8, true, true);
}

test "ami_moe_router_proj computes fp16 and fp32 router logits accurately on standard and odd shapes" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const cases = [_]struct { dtype: RouterWeightDType, m: u32, num_experts: u32, hidden: u32 }{
        .{ .dtype = .f16, .m = 1, .num_experts = 128, .hidden = 256 },
        .{ .dtype = .f16, .m = 8, .num_experts = 128, .hidden = 192 },
        .{ .dtype = .f32, .m = 5, .num_experts = 17, .hidden = 67 },
    };

    for (cases) |tc| {
        const x_bytes: usize = @as(usize, tc.m) * tc.hidden * @sizeOf(f16);
        const w_elem: usize = if (tc.dtype == .f16) @sizeOf(f16) else @sizeOf(f32);
        const w_bytes: usize = @as(usize, tc.num_experts) * tc.hidden * w_elem;
        const l_bytes: usize = @as(usize, tc.m) * tc.num_experts * @sizeOf(f32);

        var arena = try makeTestArena(x_bytes + w_bytes + l_bytes + 512);
        defer arena.deinit();

        const x_raw = try arena.allocAligned(x_bytes, 64);
        const w_raw = try arena.allocAligned(w_bytes, 64);
        const l_raw = try arena.allocAligned(l_bytes, 64);
        @memset(l_raw, 0);

        const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
        const l_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, l_raw));
        for (x_f16, 0..) |*v, i| {
            const c: i32 = @as(i32, @intCast((i * 11 + 3) % 23)) - 11;
            v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.0625);
        }

        const w_ref = try gpa.alloc(f32, @as(usize, tc.num_experts) * tc.hidden);
        defer gpa.free(w_ref);
        for (0..w_ref.len) |i| {
            const c: i32 = @as(i32, @intCast((i * 7 + 5) % 19)) - 9;
            const val: f32 = @as(f32, @floatFromInt(c)) * 0.03125;
            if (tc.dtype == .f16) {
                const h: f16 = @floatCast(val);
                const wf16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, w_raw));
                wf16[i] = h;
                w_ref[i] = @floatCast(h);
            } else {
                const wf32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, w_raw));
                wf32[i] = val;
                w_ref[i] = val;
            }
        }

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeRouterProj(&frame, tc.dtype, .{
            .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
            .router_weights = .{ .buffer = gpu_buf, .offset = arena.offsetOf(w_raw.ptr) },
            .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(l_raw.ptr) },
            .m = tc.m,
            .num_experts = tc.num_experts,
            .hidden_size = tc.hidden,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        for (0..tc.m) |tok| {
            for (0..tc.num_experts) |e| {
                var expected: f32 = 0.0;
                for (0..tc.hidden) |k| {
                    expected += @as(f32, @floatCast(x_f16[tok * tc.hidden + k])) * w_ref[e * tc.hidden + k];
                }
                try std.testing.expectApproxEqAbs(expected, l_f32[tok * tc.num_experts + e], 1e-3);
            }
        }
    }
}

fn verifyFullMoELayer(
    comptime spec: quant.QuantSpec,
    m: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    use_router_weights: bool,
    with_residual: bool,
) !void {
    const Ops = quant.QuantOps(spec);
    const gpa = std.testing.allocator;

    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const gu_w_bytes = try Ops.packedWeightBytes(intermediate_size, hidden_size);
    const gu_sb_off = amf.align64(gu_w_bytes);
    const gu_sb_bytes = try Ops.scaleBiasBytes(intermediate_size, hidden_size);
    const gu_blob_bytes = try page.alignUp(gu_sb_off + gu_sb_bytes);

    const dn_w_bytes = try Ops.packedWeightBytes(hidden_size, intermediate_size);
    const dn_sb_off = amf.align64(dn_w_bytes);
    const dn_sb_bytes = try Ops.scaleBiasBytes(hidden_size, intermediate_size);
    const dn_blob_bytes = try page.alignUp(dn_sb_off + dn_sb_bytes);

    const per_expert_bytes = 2 * gu_blob_bytes + dn_blob_bytes;
    const all_experts_bytes = @as(usize, num_experts) * per_expert_bytes;

    const ws_spec: WorkspaceSpec = .{
        .max_tokens = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
    };
    const ws_bytes = try MoeWorkspace.requiredBytes(ws_spec);
    const table_bytes = @as(usize, num_experts) * @sizeOf(ExpertWeights);
    const x_bytes = @as(usize, m) * hidden_size * @sizeOf(f16);
    const out_bytes = x_bytes;
    const router_w_bytes = @as(usize, num_experts) * hidden_size * @sizeOf(f16);
    const logits_bytes = @as(usize, m) * num_experts * @sizeOf(f32);

    const total_need = all_experts_bytes + ws_bytes + table_bytes + x_bytes + out_bytes +
        router_w_bytes + logits_bytes + 8 * page.APPLE_PAGE_SIZE;
    var arena = try makeTestArena(total_need);
    defer arena.deinit();

    // Reserve a non-zero prefix page so every ExpertWeights offset is non-zero.
    _ = try arena.allocPages(page.APPLE_PAGE_SIZE);

    const table_raw = try arena.allocAligned(table_bytes, 64);
    const expert_table: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(ExpertWeights, table_raw));

    for (0..num_experts) |e| {
        const g_blob = try arena.allocPages(gu_blob_bytes);
        const u_blob = try arena.allocPages(gu_blob_bytes);
        const d_blob = try arena.allocPages(dn_blob_bytes);
        @memset(g_blob, 0);
        @memset(u_blob, 0);
        @memset(d_blob, 0);

        try fillSyntheticQuantMatrix(
            spec,
            g_blob[0..gu_w_bytes],
            @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, g_blob[gu_sb_off .. gu_sb_off + gu_sb_bytes])),
            intermediate_size,
            hidden_size,
            e * 3 + 1,
        );
        try fillSyntheticQuantMatrix(
            spec,
            u_blob[0..gu_w_bytes],
            @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, u_blob[gu_sb_off .. gu_sb_off + gu_sb_bytes])),
            intermediate_size,
            hidden_size,
            e * 3 + 2,
        );
        try fillSyntheticQuantMatrix(
            spec,
            d_blob[0..dn_w_bytes],
            @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, d_blob[dn_sb_off .. dn_sb_off + dn_sb_bytes])),
            hidden_size,
            intermediate_size,
            e * 3 + 3,
        );

        const g_off = arena.offsetOf(g_blob.ptr);
        const u_off = arena.offsetOf(u_blob.ptr);
        const d_off = arena.offsetOf(d_blob.ptr);
        expert_table[e] = .{
            .gate_qweight_offset = g_off,
            .gate_sb_offset = g_off + gu_sb_off,
            .up_qweight_offset = u_off,
            .up_sb_offset = u_off + gu_sb_off,
            .down_qweight_offset = d_off,
            .down_sb_offset = d_off + dn_sb_off,
        };
    }

    const ws = try MoeWorkspace.init(&arena, ws_spec);
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const out_raw = try arena.allocAligned(out_bytes, 64);
    const router_w_raw = try arena.allocAligned(router_w_bytes, 64);
    const logits_raw = try arena.allocAligned(logits_bytes, 64);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
    const router_w_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, router_w_raw));
    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));

    for (x_f16, 0..) |*v, idx| {
        const c: i32 = @as(i32, @intCast((idx * 17 + 11) % 31)) - 15;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.03125);
    }
    const resid_copy = try gpa.alloc(f16, x_f16.len);
    defer gpa.free(resid_copy);
    for (out_f16, 0..) |*v, idx| {
        const c: i32 = @as(i32, @intCast((idx * 13 + 7) % 19)) - 9;
        v.* = if (with_residual) @floatCast(@as(f32, @floatFromInt(c)) * 0.0625) else @floatCast(0.0);
        resid_copy[idx] = v.*;
    }

    const ref_logits = try gpa.alloc(f32, @as(usize, m) * num_experts);
    defer gpa.free(ref_logits);

    if (use_router_weights) {
        for (0..num_experts) |e| {
            for (0..hidden_size) |k| {
                // Keep router logits distinct and overlapping across tokens.
                const c: i32 = @as(i32, @intCast((e * 11 + k * 3 + 1) % 17)) - 8;
                const boost: f32 = if (e < top_k + 2) 0.05 else 0.0;
                router_w_f16[e * hidden_size + k] = @floatCast(@as(f32, @floatFromInt(c)) * 0.015625 + boost);
            }
        }
        for (0..m) |tok| {
            for (0..num_experts) |e| {
                var acc: f32 = 0.0;
                for (0..hidden_size) |k| {
                    acc += @as(f32, @floatCast(x_f16[tok * hidden_size + k])) *
                        @as(f32, @floatCast(router_w_f16[e * hidden_size + k]));
                }
                ref_logits[tok * num_experts + e] = acc;
            }
        }
    } else {
        for (0..m) |tok| {
            for (0..num_experts) |e| {
                var v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((tok * 13 + e * 5) % 23)) - 11)) * 0.125;
                if (e < top_k) {
                    v = 3.5 - 0.1 * @as(f32, @floatFromInt(e));
                } else if (e == (top_k + (tok % 3)) % num_experts) {
                    v = 4.0;
                }
                logits_f32[tok * num_experts + e] = v;
                ref_logits[tok * num_experts + e] = v;
            }
        }
    }

    const ref_out = try gpa.alloc(f32, @as(usize, m) * hidden_size);
    defer gpa.free(ref_out);
    var route_ref = try moeLayerRef(
        spec,
        gpa,
        x_f16,
        ref_logits,
        arena.bytes(),
        expert_table,
        if (with_residual) resid_copy else null,
        ref_out,
        m,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        true,
    );
    defer route_ref.deinit(gpa);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const out_binding: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) };
    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeMoELayer(&frame, spec.bits, .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .router = if (use_router_weights)
            .{ .weights = .{ .binding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(router_w_raw.ptr) }, .dtype = .f16 } }
        else
            .{ .logits = .{ .binding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) }, .dtype = .f32 } },
        .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
        .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
        .workspace = &ws,
        .workspace_buf = gpu_buf,
        .residual = if (with_residual) out_binding else null,
        .out = out_binding,
        .m = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .norm_topk_prob = true,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    try std.testing.expectEqual(route_ref.num_unique, ws.schedule_header.num_unique_experts);
    if (!use_router_weights and m > 1) {
        try std.testing.expect(ws.schedule_header.num_unique_experts < m * top_k);
    }

    for (ref_out, 0..) |expected, idx| {
        const got: f32 = @floatCast(out_f16[idx]);
        const tol: f32 = @max(0.02, @abs(expected) * 0.02);
        try std.testing.expectApproxEqAbs(expected, got, tol);
    }
}

test "deduplicated 3-bit MoE layer matches CPU fp32 reference on M=1 decode, M=8 batch and odd dims" {
    // M=1 decode with top_k=8 (Qwen3-30B-A3B top_k) and odd group count (inter=192 = 3*64).
    try verifyFullMoELayer(.{ .bits = .q3, .group_size = 64 }, 1, 16, 8, 256, 192, true, false);
    // M=8 multi-token batch with shared experts (U < M * top_k) and fused residual add.
    try verifyFullMoELayer(.{ .bits = .q3, .group_size = 64 }, 8, 32, 8, 192, 128, false, true);
    // Odd M, odd num_experts, dynamic top_k=3.
    try verifyFullMoELayer(.{ .bits = .q3, .group_size = 64 }, 5, 17, 3, 192, 128, true, true);
}

test "deduplicated 4-bit MoE layer matches CPU fp32 reference on M=1 decode, M=8 batch and odd dims" {
    try verifyFullMoELayer(.{ .bits = .q4, .group_size = 64 }, 1, 16, 8, 256, 192, true, true);
    try verifyFullMoELayer(.{ .bits = .q4, .group_size = 64 }, 8, 32, 8, 192, 128, false, false);
    try verifyFullMoELayer(.{ .bits = .q4, .group_size = 64 }, 5, 17, 3, 192, 128, true, false);
}

test "ami_silu_mul_f16 matches CPU fp32 reference across odd sizes and Qwen3-14B intermediate size" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const counts = [_]u32{ 1, 7, 13, 257, 17_408 };
    for (counts) |count| {
        const bytes: usize = @as(usize, count) * @sizeOf(f16);
        var arena = try makeTestArena(bytes * 3 + 512);
        defer arena.deinit();

        const gate_raw = try arena.allocAligned(bytes, 64);
        const up_raw = try arena.allocAligned(bytes, 64);
        const out_raw = try arena.allocAligned(bytes, 64);
        @memset(out_raw, 0);

        const gate_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate_raw));
        const up_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, up_raw));
        const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));

        for (0..count) |i| {
            const g: i32 = @as(i32, @intCast((i * 29 + 5) % 41)) - 20;
            const u: i32 = @as(i32, @intCast((i * 17 + 9) % 31)) - 15;
            gate_f16[i] = @floatCast(@as(f32, @floatFromInt(g)) * 0.25);
            up_f16[i] = @floatCast(@as(f32, @floatFromInt(u)) * 0.125);
        }

        const ref = try gpa.alloc(f32, count);
        defer gpa.free(ref);
        siluMulRef(gate_f16, up_f16, ref);

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeSiluMul(&frame, .{
            .gate = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gate_raw.ptr) },
            .up = .{ .buffer = gpu_buf, .offset = arena.offsetOf(up_raw.ptr) },
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .count = count,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        for (ref, 0..) |expected, i| {
            const got: f32 = @floatCast(out_f16[i]);
            try std.testing.expectApproxEqAbs(expected, got, @max(0.005, @abs(expected) * 0.005));
        }
    }
}

test "ami_residual_add_f16 matches CPU fp32 reference across odd sizes and Qwen3-14B hidden size" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const counts = [_]u32{ 1, 7, 257, 5_120 };
    for (counts) |count| {
        const bytes: usize = @as(usize, count) * @sizeOf(f16);
        var arena = try makeTestArena(bytes * 2 + 512);
        defer arena.deinit();

        const x_raw = try arena.allocAligned(bytes, 64);
        const delta_raw = try arena.allocAligned(bytes, 64);
        const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
        const delta_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, delta_raw));

        for (0..count) |i| {
            const a: i32 = @as(i32, @intCast((i * 19 + 3) % 37)) - 18;
            const b: i32 = @as(i32, @intCast((i * 11 + 7) % 29)) - 14;
            x_f16[i] = @floatCast(@as(f32, @floatFromInt(a)) * 0.25);
            delta_f16[i] = @floatCast(@as(f32, @floatFromInt(b)) * 0.125);
        }

        const ref = try gpa.alloc(f32, count);
        defer gpa.free(ref);
        residualAddRef(x_f16, delta_f16, ref);

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeResidualAdd(&frame, .{
            .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
            .delta = .{ .buffer = gpu_buf, .offset = arena.offsetOf(delta_raw.ptr) },
            .count = count,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        for (ref, 0..) |expected, i| {
            const got: f32 = @floatCast(x_f16[i]);
            try std.testing.expectApproxEqAbs(expected, got, @max(0.005, @abs(expected) * 0.005));
        }
    }
}

test "ami_argmax_f16 finds exact greedy token ID across edge sizes, ties, and Qwen3 vocab 151936" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const cases = [_]struct { vocab: u32, winner: u32, tie_at: ?u32, all_neg: bool, unaligned_off: bool }{
        .{ .vocab = 1, .winner = 0, .tie_at = null, .all_neg = true, .unaligned_off = false },
        .{ .vocab = 7, .winner = 6, .tie_at = null, .all_neg = false, .unaligned_off = false },
        .{ .vocab = 31, .winner = 0, .tie_at = 19, .all_neg = false, .unaligned_off = false },
        .{ .vocab = 33, .winner = 32, .tie_at = null, .all_neg = true, .unaligned_off = true },
        .{ .vocab = 1025, .winner = 777, .tie_at = 1024, .all_neg = false, .unaligned_off = false },
        // Full Qwen3 vocabulary size (151,936) with peak near end, plus tie-breaking and odd +1.
        .{ .vocab = 151_936, .winner = 149_823, .tie_at = 151_935, .all_neg = false, .unaligned_off = false },
        .{ .vocab = 151_936, .winner = 0, .tie_at = null, .all_neg = true, .unaligned_off = false },
        .{ .vocab = 151_937, .winner = 151_936, .tie_at = null, .all_neg = false, .unaligned_off = false },
    };

    for (cases) |tc| {
        const extra: usize = if (tc.unaligned_off) 2 else 0;
        const logits_bytes: usize = @as(usize, tc.vocab) * @sizeOf(f16) + extra;
        var arena = try makeTestArena(logits_bytes + 256);
        defer arena.deinit();

        const logits_raw = try arena.allocAligned(logits_bytes, 64);
        const out_raw = try arena.allocAligned(@sizeOf(u32), 64);
        const out_slot: *u32 = @ptrCast(@alignCast(out_raw.ptr));
        out_slot.* = 0xDEAD_BEEF;

        const logits_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, logits_raw[extra..]));
        for (logits_f16, 0..) |*v, idx| {
            if (tc.all_neg) {
                v.* = @floatCast(-10.0 - @as(f32, @floatFromInt(idx % 97)) * 0.125);
            } else {
                const c: i32 = @as(i32, @intCast((idx * 37 + 13) % 61)) - 30;
                v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.25);
            }
        }
        const peak: f16 = if (tc.all_neg) @floatCast(-1.25) else @floatCast(19.5);
        logits_f16[tc.winner] = peak;
        if (tc.tie_at) |t| logits_f16[t] = peak;

        const expected = argmaxRef(logits_f16);
        try std.testing.expectEqual(tc.winner, expected);

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeArgmax(&frame, .{
            .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) + extra },
            .out_token_id = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .vocab_size = tc.vocab,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        try std.testing.expectEqual(expected, out_slot.*);
    }
}

test "ami_moe_route_topk breaks exact ties deterministically by smallest expert ID" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const num_experts: u32 = 128;
    const top_k: u32 = 8;
    const m: u32 = 4;
    const ws_spec: WorkspaceSpec = .{
        .max_tokens = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = 64,
        .intermediate_size = 64,
    };
    const logits_bytes: usize = @as(usize, m) * num_experts * @sizeOf(f32);
    var arena = try makeTestArena(try MoeWorkspace.requiredBytes(ws_spec) + logits_bytes + 512);
    defer arena.deinit();

    const ws = try MoeWorkspace.init(&arena, ws_spec);
    const logits_raw = try arena.allocAligned(logits_bytes, 64);
    @memset(logits_raw, 0); // All identical 0.0f logits -> experts 0..7 must win every token.

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeRouteTopK(&frame, .f32, .{
        .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) },
        .workspace = &ws,
        .workspace_buf = gpu_buf,
        .m = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .norm_topk_prob = true,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    // All 4 tokens share the exact same 8 experts -> U = 8 < 4 * 8 = 32.
    try std.testing.expectEqual(@as(u32, 8), ws.schedule_header.num_unique_experts);
    for (0..8) |u| {
        try std.testing.expectEqual(@as(u32, @intCast(u)), ws.unique_experts[u]);
        try std.testing.expectEqual(@as(u32, @intCast(u * m)), ws.expert_token_offsets[u]);
    }
    try std.testing.expectEqual(@as(u32, m * top_k), ws.expert_token_offsets[8]);
    for (ws.routing_weights[0 .. m * top_k]) |w| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.125), w, 1e-5);
    }
}

test "deduplicated MoE handles >16 tokens per expert (M=20) and odd non-multiple-of-16 output rows" {
    // M=20 with 4 experts and top_k=2 guarantees >16 tokens hit the same expert
    // (exercising both wave wrap in route_topk and T_BLK=16 tiling in gate_up/down).
    try verifyFullMoELayer(.{ .bits = .q3, .group_size = 64 }, 20, 4, 2, 128, 64, false, true);
    try verifyFullMoELayer(.{ .bits = .q4, .group_size = 64 }, 20, 4, 2, 128, 64, false, false);

    // Standalone odd output rows: intermediate_size = 37 (not multiple of 8) in gate_up_swiglu,
    // and hidden_size = 37 (not multiple of 16) in down_proj + reduce_f16 across M=1 and M=5.
    const gpa = std.testing.allocator;
    const Q3 = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    for ([_]u32{ 1, 5 }) |m| {
        const num_experts: u32 = 4;
        const top_k: u32 = 2;
        const k_in: u32 = 128; // group-64 aligned K for gate_up and down
        const n_odd: u32 = 37; // odd N (neither multiple of 8 nor 16)

        const w_bytes = try Q3.packedWeightBytes(n_odd, k_in);
        const sb_off = amf.align64(w_bytes);
        const sb_bytes = try Q3.scaleBiasBytes(n_odd, k_in);
        const blob_bytes = try page.alignUp(sb_off + sb_bytes);

        const ws_spec: WorkspaceSpec = .{
            .max_tokens = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = k_in, // max of k_in and n_odd
            .intermediate_size = k_in,
        };
        const total = @as(usize, num_experts) * 3 * blob_bytes +
            try MoeWorkspace.requiredBytes(ws_spec) +
            @as(usize, num_experts) * @sizeOf(ExpertWeights) +
            @as(usize, m) * k_in * @sizeOf(f16) +
            @as(usize, m) * n_odd * @sizeOf(f16) +
            @as(usize, m) * num_experts * @sizeOf(f32) +
            8 * page.APPLE_PAGE_SIZE;
        var arena = try makeTestArena(total);
        defer arena.deinit();

        const table_raw = try arena.allocAligned(@as(usize, num_experts) * @sizeOf(ExpertWeights), 64);
        const table: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(ExpertWeights, table_raw));
        for (0..num_experts) |e| {
            const gb = try arena.allocPages(blob_bytes);
            const ub = try arena.allocPages(blob_bytes);
            const db = try arena.allocPages(blob_bytes);
            @memset(gb, 0);
            @memset(ub, 0);
            @memset(db, 0);
            try fillSyntheticQuantMatrix(.{ .bits = .q3, .group_size = 64 }, gb[0..w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, gb[sb_off .. sb_off + sb_bytes])), n_odd, k_in, e * 5 + 1);
            try fillSyntheticQuantMatrix(.{ .bits = .q3, .group_size = 64 }, ub[0..w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, ub[sb_off .. sb_off + sb_bytes])), n_odd, k_in, e * 5 + 2);
            try fillSyntheticQuantMatrix(.{ .bits = .q3, .group_size = 64 }, db[0..w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, db[sb_off .. sb_off + sb_bytes])), n_odd, k_in, e * 5 + 3);
            table[e] = .{
                .gate_qweight_offset = arena.offsetOf(gb.ptr),
                .gate_sb_offset = arena.offsetOf(gb.ptr) + sb_off,
                .up_qweight_offset = arena.offsetOf(ub.ptr),
                .up_sb_offset = arena.offsetOf(ub.ptr) + sb_off,
                .down_qweight_offset = arena.offsetOf(db.ptr),
                .down_sb_offset = arena.offsetOf(db.ptr) + sb_off,
            };
        }

        const ws = try MoeWorkspace.init(&arena, ws_spec);
        const x_raw = try arena.allocAligned(@as(usize, m) * k_in * @sizeOf(f16), 64);
        const out_raw = try arena.allocAligned(@as(usize, m) * n_odd * @sizeOf(f16), 64);
        const logits_raw = try arena.allocAligned(@as(usize, m) * num_experts * @sizeOf(f32), 64);
        const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
        const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
        const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));

        for (x_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(i % 15)) - 7)) * 0.0625);
        for (logits_f32, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % num_experts)) * 0.5;
        @memset(out_raw, 0);

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        // 1. Route + GateUp with odd intermediate_size = 37
        var frame1 = try compute.beginFrame(5000);
        errdefer frame1.abort();
        try pipes.encodeRouteTopK(&frame1, .f32, .{
            .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .workspace = &ws,
            .workspace_buf = gpu_buf,
            .m = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .norm_topk_prob = true,
        }, false);
        try pipes.encodeGateUpSwiglu(&frame1, .q3, .{
            .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
            .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
            .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
            .workspace = &ws,
            .workspace_buf = gpu_buf,
            .m = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = k_in,
            .intermediate_size = n_odd,
        }, true);
        const sig1 = try frame1.commit();
        try compute.wait(sig1, 5000);

        // Verify odd-N gate_up_swiglu output against CPU reference.
        var g_ref: [n_odd]f32 = undefined;
        var u_ref: [n_odd]f32 = undefined;
        const u_cnt = ws.schedule_header.num_unique_experts;
        for (0..u_cnt) |ui| {
            const e = ws.unique_experts[ui];
            const ew = table[e];
            const g_qw = arena.bytes()[@intCast(ew.gate_qweight_offset)..][0..w_bytes];
            const g_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, arena.bytes()[@intCast(ew.gate_sb_offset)..][0..sb_bytes]));
            const u_qw = arena.bytes()[@intCast(ew.up_qweight_offset)..][0..w_bytes];
            const u_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, arena.bytes()[@intCast(ew.up_sb_offset)..][0..sb_bytes]));
            for (ws.expert_token_offsets[ui]..ws.expert_token_offsets[ui + 1]) |s| {
                const tok = ws.token_indices[s];
                try Q3.matmulRef(x_f16[tok * k_in ..][0..k_in], g_qw, g_sb, &g_ref, 1, n_odd, k_in);
                try Q3.matmulRef(x_f16[tok * k_in ..][0..k_in], u_qw, u_sb, &u_ref, 1, n_odd, k_in);
                for (0..n_odd) |col| {
                    const expected = siluF32(g_ref[col]) * u_ref[col];
                    const got: f32 = @floatCast(ws.inter[s * n_odd + col]);
                    try std.testing.expectApproxEqAbs(expected, got, @max(0.015, @abs(expected) * 0.015));
                }
            }
        }

        // 2. Fill inter with [m * top_k, k_in] and test DownProj + Reduce with odd hidden_size = 37.
        for (0..@as(usize, m) * top_k * k_in) |i| {
            ws.inter[i] = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + 1) % 13)) - 6)) * 0.0625);
        }
        var frame2 = try compute.beginFrame(5000);
        errdefer frame2.abort();
        try pipes.encodeDownProj(&frame2, .q3, .{
            .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
            .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
            .workspace = &ws,
            .workspace_buf = gpu_buf,
            .m = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = n_odd,
            .intermediate_size = k_in,
        }, false);
        try pipes.encodeReduce(&frame2, .{
            .workspace = &ws,
            .workspace_buf = gpu_buf,
            .residual = null,
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .m = m,
            .hidden_size = n_odd,
            .top_k = top_k,
        }, true);
        const sig2 = try frame2.commit();
        try compute.wait(sig2, 5000);

        const ref_out = try gpa.alloc(f32, @as(usize, m) * n_odd);
        defer gpa.free(ref_out);
        @memset(ref_out, 0.0);
        var d_ref: [n_odd]f32 = undefined;
        for (0..u_cnt) |ui| {
            const e = ws.unique_experts[ui];
            const ew = table[e];
            const d_qw = arena.bytes()[@intCast(ew.down_qweight_offset)..][0..w_bytes];
            const d_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, arena.bytes()[@intCast(ew.down_sb_offset)..][0..sb_bytes]));
            for (ws.expert_token_offsets[ui]..ws.expert_token_offsets[ui + 1]) |s| {
                const tok = ws.token_indices[s];
                const rw = ws.routing_weights[s];
                try Q3.matmulRef(ws.inter[s * k_in ..][0..k_in], d_qw, d_sb, &d_ref, 1, n_odd, k_in);
                for (0..n_odd) |col| {
                    ref_out[tok * n_odd + col] += d_ref[col] * rw;
                }
            }
        }
        for (ref_out, 0..) |expected, idx| {
            const got: f32 = @floatCast(out_f16[idx]);
            try std.testing.expectApproxEqAbs(expected, got, @max(0.015, @abs(expected) * 0.015));
        }
    }
}

test "Qwen3-30B-A3B 3-bit MoE geometry (hidden=2048, inter=768, E=128, top_k=8) passes M=1 and M=8" {
    const spec: quant.QuantSpec = .{ .bits = .q3, .group_size = 64 };
    const Ops = quant.QuantOps(spec);
    const gpa = std.testing.allocator;

    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const num_experts: u32 = 128;
    const top_k: u32 = 8;
    const hidden_size: u32 = 2048;
    const intermediate_size: u32 = 768;
    const unique_backing_experts: usize = 4; // alias 128 table entries across 4 real Q3 weight sets (~8 MiB)

    const gu_w_bytes = try Ops.packedWeightBytes(intermediate_size, hidden_size);
    const gu_sb_off = amf.align64(gu_w_bytes);
    const gu_sb_bytes = try Ops.scaleBiasBytes(intermediate_size, hidden_size);
    const gu_blob_bytes = try page.alignUp(gu_sb_off + gu_sb_bytes);

    const dn_w_bytes = try Ops.packedWeightBytes(hidden_size, intermediate_size);
    const dn_sb_off = amf.align64(dn_w_bytes);
    const dn_sb_bytes = try Ops.scaleBiasBytes(hidden_size, intermediate_size);
    const dn_blob_bytes = try page.alignUp(dn_sb_off + dn_sb_bytes);

    const max_m: u32 = 8;
    const ws_spec: WorkspaceSpec = .{
        .max_tokens = max_m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
    };

    const total = unique_backing_experts * (2 * gu_blob_bytes + dn_blob_bytes) +
        try MoeWorkspace.requiredBytes(ws_spec) +
        @as(usize, num_experts) * @sizeOf(ExpertWeights) +
        @as(usize, max_m) * hidden_size * @sizeOf(f16) * 2 +
        @as(usize, max_m) * num_experts * @sizeOf(f32) +
        8 * page.APPLE_PAGE_SIZE;
    var arena = try makeTestArena(total);
    defer arena.deinit();

    var backing: [unique_backing_experts]ExpertWeights = undefined;
    for (0..unique_backing_experts) |b| {
        const gb = try arena.allocPages(gu_blob_bytes);
        const ub = try arena.allocPages(gu_blob_bytes);
        const db = try arena.allocPages(dn_blob_bytes);
        @memset(gb, 0);
        @memset(ub, 0);
        @memset(db, 0);
        try fillSyntheticQuantMatrix(spec, gb[0..gu_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, gb[gu_sb_off .. gu_sb_off + gu_sb_bytes])), intermediate_size, hidden_size, b * 3 + 1);
        try fillSyntheticQuantMatrix(spec, ub[0..gu_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, ub[gu_sb_off .. gu_sb_off + gu_sb_bytes])), intermediate_size, hidden_size, b * 3 + 2);
        try fillSyntheticQuantMatrix(spec, db[0..dn_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, db[dn_sb_off .. dn_sb_off + dn_sb_bytes])), hidden_size, intermediate_size, b * 3 + 3);
        backing[b] = .{
            .gate_qweight_offset = arena.offsetOf(gb.ptr),
            .gate_sb_offset = arena.offsetOf(gb.ptr) + gu_sb_off,
            .up_qweight_offset = arena.offsetOf(ub.ptr),
            .up_sb_offset = arena.offsetOf(ub.ptr) + gu_sb_off,
            .down_qweight_offset = arena.offsetOf(db.ptr),
            .down_sb_offset = arena.offsetOf(db.ptr) + dn_sb_off,
        };
    }

    const table_raw = try arena.allocAligned(@as(usize, num_experts) * @sizeOf(ExpertWeights), 64);
    const expert_table: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(ExpertWeights, table_raw));
    for (0..num_experts) |e| expert_table[e] = backing[e % unique_backing_experts];

    const ws = try MoeWorkspace.init(&arena, ws_spec);
    const x_raw = try arena.allocAligned(@as(usize, max_m) * hidden_size * @sizeOf(f16), 64);
    const out_raw = try arena.allocAligned(@as(usize, max_m) * hidden_size * @sizeOf(f16), 64);
    const logits_raw = try arena.allocAligned(@as(usize, max_m) * num_experts * @sizeOf(f32), 64);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));

    for (x_f16, 0..) |*v, idx| {
        const c: i32 = @as(i32, @intCast((idx * 13 + 7) % 21)) - 10;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.015625);
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    for ([_]u32{ 1, 8 }) |m| {
        @memset(out_raw, 0);
        for (0..m) |tok| {
            for (0..num_experts) |e| {
                var v: f32 = -1.0 + 0.01 * @as(f32, @floatFromInt((tok * 11 + e) % 31));
                if (e >= 100 and e < 107) v = 4.0 - 0.05 * @as(f32, @floatFromInt(e - 100));
                if (e == 110 + (tok % 4)) v = 4.5;
                logits_f32[tok * num_experts + e] = v;
            }
        }

        const ref_out = try gpa.alloc(f32, @as(usize, m) * hidden_size);
        defer gpa.free(ref_out);
        var route_ref = try moeLayerRef(
            spec,
            gpa,
            x_f16[0 .. @as(usize, m) * hidden_size],
            logits_f32[0 .. @as(usize, m) * num_experts],
            arena.bytes(),
            expert_table,
            null,
            ref_out,
            m,
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            true,
        );
        defer route_ref.deinit(gpa);

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeMoELayer(&frame, .q3, .{
            .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
            .router = .{ .logits = .{ .binding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) }, .dtype = .f32 } },
            .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
            .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
            .workspace = &ws,
            .workspace_buf = gpu_buf,
            .residual = null,
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .m = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .norm_topk_prob = true,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        try std.testing.expectEqual(route_ref.num_unique, ws.schedule_header.num_unique_experts);
        if (m == 8) {
            // 7 shared experts + 4 token-specific experts = 11 unique experts << 64 assignments!
            try std.testing.expectEqual(@as(u32, 11), ws.schedule_header.num_unique_experts);
        }
        for (ref_out, 0..) |expected, idx| {
            const got: f32 = @floatCast(out_f16[idx]);
            try std.testing.expectApproxEqAbs(expected, got, @max(0.03, @abs(expected) * 0.02));
        }
    }
}

test "encode helpers validate dimensions, alignment, and workspace capacity" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const ws_spec: WorkspaceSpec = .{
        .max_tokens = 2,
        .num_experts = 8,
        .top_k = 2,
        .hidden_size = 64,
        .intermediate_size = 64,
    };
    var arena = try makeTestArena(try MoeWorkspace.requiredBytes(ws_spec) + 1024);
    defer arena.deinit();
    const ws = try MoeWorkspace.init(&arena, ws_spec);
    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    var frame = try compute.beginFrame(5000);
    defer frame.abort();
    const b0: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = 0 };

    try std.testing.expectError(error.ZeroDimension, pipes.encodeSiluMul(&frame, .{ .gate = b0, .up = b0, .out = b0, .count = 0 }, false));
    try std.testing.expectError(error.ZeroDimension, pipes.encodeResidualAdd(&frame, .{ .x = b0, .delta = b0, .count = 0 }, false));
    try std.testing.expectError(error.ZeroDimension, pipes.encodeArgmax(&frame, .{ .logits = b0, .out_token_id = b0, .vocab_size = 0 }, false));
    try std.testing.expectError(error.UnsupportedExpertCount, pipes.encodeRouteTopK(&frame, .f32, .{
        .logits = b0,
        .workspace = &ws,
        .workspace_buf = gpu_buf,
        .m = 1,
        .num_experts = MAX_EXPERTS + 1,
        .top_k = 2,
    }, false));
    try std.testing.expectError(error.WorkspaceTooSmall, pipes.encodeRouteTopK(&frame, .f32, .{
        .logits = b0,
        .workspace = &ws,
        .workspace_buf = gpu_buf,
        .m = 3,
        .num_experts = 8,
        .top_k = 2,
    }, false));
    try std.testing.expectError(error.UnalignedK, pipes.encodeGateUpSwiglu(&frame, .q3, .{
        .x = b0,
        .weights_base = b0,
        .expert_table = b0,
        .workspace = &ws,
        .workspace_buf = gpu_buf,
        .m = 1,
        .num_experts = 8,
        .top_k = 2,
        .hidden_size = 65,
        .intermediate_size = 64,
    }, false));
}

// ---------------------------------------------------------------------------
// M2.5 CPU fp32 references: BF16 / quantized routers, FP16 / BF16 MoE layer,
// and embedding lookup (G12, ADR-0016)
// ---------------------------------------------------------------------------

/// Algorithmic minimum weight bytes read by one deduplicated FP16 / BF16 MoE
/// layer when `unique_experts` distinct experts are active.
pub fn expertLayerBytesFloat(unique_experts: u32, hidden_size: u32, intermediate_size: u32) quant.ShapeError!u64 {
    const per_expert = std.math.mul(u64, @as(u64, 3 * 2) * hidden_size, intermediate_size) catch return error.Overflow;
    return std.math.mul(u64, per_expert, unique_experts) catch return error.Overflow;
}

/// CPU fp32 reference router projection over `f16` / `bf16` weights
/// (`w_bits` holds raw 16-bit patterns of `[num_experts, hidden]`).
pub fn routerProjFloatRef(
    dtype: quant.FloatDType,
    x: []const f16,
    w_bits: []const u16,
    logits_f32: []f32,
    m: u32,
    num_experts: u32,
    hidden_size: u32,
) (quant.ShapeError || error{BufferTooSmall})!void {
    return quant.denseGemmCpuRef(dtype, x, w_bits, logits_f32, m, num_experts, hidden_size);
}

/// CPU fp32 reference router projection over MLX affine group-64 weights.
pub fn routerProjQuantRef(
    comptime spec: quant.QuantSpec,
    x: []const f16,
    qweight: []const u8,
    scale_bias: []const quant.ScaleBias,
    logits_f32: []f32,
    m: u32,
    num_experts: u32,
    hidden_size: u32,
) (quant.ShapeError || error{BufferTooSmall})!void {
    return quant.QuantOps(spec).matmulRef(x, qweight, scale_bias, logits_f32, m, num_experts, hidden_size);
}

fn floatMatrixAt(weights_base: []const u8, offset: u64, n: usize, k: usize) []const u16 {
    const off: usize = @intCast(offset);
    return @alignCast(std.mem.bytesAsSlice(u16, weights_base[off..][0 .. n * k * @sizeOf(u16)]));
}

/// Full CPU fp32 reference for an FP16 / BF16 MoE layer over an arena byte
/// buffer and `ExpertWeights` byte-offset table (`*_sb_offset` ignored).
pub fn moeLayerFloatRef(
    dtype: quant.FloatDType,
    allocator: std.mem.Allocator,
    x: []const f16,
    logits_f32: []const f32,
    weights_base: []const u8,
    expert_table: []const ExpertWeights,
    residual: ?[]const f16,
    out_f32: []f32,
    m: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    norm_topk_prob: bool,
) !RouteRefResult {
    var route = try routeTopKRef(allocator, logits_f32, m, num_experts, top_k, norm_topk_prob);
    errdefer route.deinit(allocator);

    const gate_buf = try allocator.alloc(f32, intermediate_size);
    defer allocator.free(gate_buf);
    const up_buf = try allocator.alloc(f32, intermediate_size);
    defer allocator.free(up_buf);
    const inter_f16 = try allocator.alloc(f16, intermediate_size);
    defer allocator.free(inter_f16);
    const down_buf = try allocator.alloc(f32, hidden_size);
    defer allocator.free(down_buf);

    for (0..@as(usize, m) * hidden_size) |i| {
        out_f32[i] = if (residual) |res| @as(f32, @floatCast(res[i])) else 0.0;
    }

    for (0..m) |tok| {
        const x_tok = x[tok * hidden_size ..][0..hidden_size];
        const out_tok = out_f32[tok * hidden_size ..][0..hidden_size];
        for (0..top_k) |k_idx| {
            const e = route.topk_experts[tok * top_k + k_idx];
            const rw = route.topk_weights[tok * top_k + k_idx];
            const ew = expert_table[e];
            const gate_w = floatMatrixAt(weights_base, ew.gate_qweight_offset, intermediate_size, hidden_size);
            const up_w = floatMatrixAt(weights_base, ew.up_qweight_offset, intermediate_size, hidden_size);
            const down_w = floatMatrixAt(weights_base, ew.down_qweight_offset, hidden_size, intermediate_size);
            try quant.denseGemvCpuRef(dtype, x_tok, gate_w, gate_buf, intermediate_size, hidden_size);
            try quant.denseGemvCpuRef(dtype, x_tok, up_w, up_buf, intermediate_size, hidden_size);
            for (0..intermediate_size) |i| {
                inter_f16[i] = @floatCast(siluF32(gate_buf[i]) * up_buf[i]);
            }
            try quant.denseGemvCpuRef(dtype, inter_f16, down_w, down_buf, hidden_size, intermediate_size);
            for (0..hidden_size) |i| out_tok[i] += down_buf[i] * rw;
        }
    }
    return route;
}

/// CPU fp32 reference embedding lookup. `table` holds raw `f16` / `bf16`
/// `[vocab, hidden]` elements or packed MLX affine group-64 codes; quantized
/// tables require `scale_bias`. Out-of-range token IDs produce zero rows.
pub fn embedLookupRef(
    dtype: quant.WeightDType,
    table: []const u8,
    scale_bias: ?[]const quant.ScaleBias,
    token_ids: []const u32,
    out: []f32,
    vocab_size: u32,
    hidden_size: u32,
) (quant.ShapeError || error{ BufferTooSmall, MissingScaleBias })!void {
    const h: usize = hidden_size;
    if (out.len < token_ids.len * h) return error.BufferTooSmall;
    for (token_ids, 0..) |tok, row| {
        const dst = out[row * h ..][0..h];
        if (tok >= vocab_size) {
            @memset(dst, 0.0);
            continue;
        }
        switch (dtype) {
            .f16, .bf16 => {
                const fdt: quant.FloatDType = if (dtype == .f16) .f16 else .bf16;
                const row_bits: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, table[tok * h * 2 ..][0 .. h * 2]));
                for (0..h) |i| dst[i] = quant.floatBitsToF32(fdt, row_bits[i]);
            },
            inline .q4_affine, .q3_affine, .q2_affine => |q| {
                const bits: quant.QuantBits = switch (q) {
                    .q4_affine => .q4,
                    .q3_affine => .q3,
                    .q2_affine => .q2,
                    else => unreachable,
                };
                const Ops = quant.QuantOps(.{ .bits = bits, .group_size = 64 });
                const sbs = scale_bias orelse return error.MissingScaleBias;
                const groups = try Ops.groupsPerRow(h);
                const row_bytes = try Ops.rowWeightBytes(h);
                const gb = Ops.groupWeightBytes();
                var deq: [64]f32 = undefined;
                for (0..groups) |g| {
                    Ops.dequantizeGroup(table[tok * row_bytes + g * gb ..][0..gb], sbs[tok * groups + g], &deq);
                    @memcpy(dst[g * 64 ..][0..64], &deq);
                }
            },
        }
    }
}

fn verifyRouterFloat(dtype: RouterWeightDType, m: u32, num_experts: u32, hidden: u32) !void {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const fdt: quant.FloatDType = if (dtype == .bf16) .bf16 else .f16;
    const x_bytes: usize = @as(usize, m) * hidden * @sizeOf(f16);
    const w_bytes: usize = @as(usize, num_experts) * hidden * @sizeOf(u16);
    const l_bytes: usize = @as(usize, m) * num_experts * @sizeOf(f32);
    var arena = try makeTestArena(x_bytes + w_bytes + l_bytes + 512);
    defer arena.deinit();
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const w_raw = try arena.allocAligned(w_bytes, 64);
    const l_raw = try arena.allocAligned(l_bytes, 64);
    @memset(l_raw, 0);
    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const w_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, w_raw));
    const l_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, l_raw));
    for (x_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 11 + 3) % 23)) - 11;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.0625);
    }
    for (w_bits, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 7 + 5) % 19)) - 9;
        v.* = quant.f32ToFloatBits(fdt, @as(f32, @floatFromInt(c)) * 0.03125 + @as(f32, @floatFromInt(i % 3)) / 300.0);
    }
    const ref = try gpa.alloc(f32, @as(usize, m) * num_experts);
    defer gpa.free(ref);
    try routerProjFloatRef(fdt, x_f16, w_bits, ref, m, num_experts, hidden);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeRouterProj(&frame, dtype, .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .router_weights = .{ .buffer = gpu_buf, .offset = arena.offsetOf(w_raw.ptr) },
        .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(l_raw.ptr) },
        .m = m,
        .num_experts = num_experts,
        .hidden_size = hidden,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);
    for (ref, 0..) |expected, i| {
        try std.testing.expectApproxEqAbs(expected, l_f32[i], @max(1e-3, @abs(expected) * 1e-4));
    }
}

test "ami_moe_router_proj computes bf16 router logits on standard and odd shapes" {
    try verifyRouterFloat(.bf16, 1, 128, 256);
    try verifyRouterFloat(.bf16, 8, 128, 192);
    try verifyRouterFloat(.bf16, 5, 17, 67);
    try verifyRouterFloat(.f16, 3, 33, 2048);
}

fn verifyRouterQuant(comptime spec: quant.QuantSpec, m: u32, num_experts: u32, hidden: u32) !void {
    const Ops = quant.QuantOps(spec);
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const qw_bytes = try Ops.packedWeightBytes(num_experts, hidden);
    const sb_bytes = try Ops.scaleBiasBytes(num_experts, hidden);
    const x_bytes: usize = @as(usize, m) * hidden * @sizeOf(f16);
    const l_bytes: usize = @as(usize, m) * num_experts * @sizeOf(f32);
    var arena = try makeTestArena(qw_bytes + sb_bytes + x_bytes + l_bytes + 1024);
    defer arena.deinit();
    const qw_raw = try arena.allocAligned(qw_bytes, 64);
    const sb_raw = try arena.allocAligned(sb_bytes, 64);
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const l_raw = try arena.allocAligned(l_bytes, 64);
    @memset(l_raw, 0);
    const sbs: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sb_raw));
    try fillSyntheticQuantMatrix(spec, qw_raw, sbs, num_experts, hidden, 5);
    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    for (x_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 13 + 1) % 29)) - 14;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.03125);
    }
    const l_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, l_raw));
    const ref = try gpa.alloc(f32, @as(usize, m) * num_experts);
    defer gpa.free(ref);
    try routerProjQuantRef(spec, x_f16, qw_raw, sbs, ref, m, num_experts, hidden);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeRouterProjQuant(&frame, spec.bits, .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .qweight = .{ .buffer = gpu_buf, .offset = arena.offsetOf(qw_raw.ptr) },
        .scale_bias = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sb_raw.ptr) },
        .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(l_raw.ptr) },
        .m = m,
        .num_experts = num_experts,
        .hidden_size = hidden,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);
    for (ref, 0..) |expected, i| {
        try std.testing.expectApproxEqAbs(expected, l_f32[i], @max(2e-3, @abs(expected) * 1e-3));
    }
}

test "ami_moe_router_proj_q computes q3 and q4 router logits matching CPU fp32 reference" {
    // Real Qwen3-30B-A3B-3bit router geometry: q3_affine [128, 2048].
    try verifyRouterQuant(.{ .bits = .q3, .group_size = 64 }, 1, 128, 2048);
    try verifyRouterQuant(.{ .bits = .q3, .group_size = 64 }, 8, 128, 2048);
    try verifyRouterQuant(.{ .bits = .q4, .group_size = 64 }, 1, 128, 2048);
    try verifyRouterQuant(.{ .bits = .q4, .group_size = 64 }, 3, 17, 192);
    try verifyRouterQuant(.{ .bits = .q3, .group_size = 64 }, 5, 33, 320);
}

const FloatRouterMode = enum { logits, bf16_weights, q3_weights };

fn verifyFullMoELayerFloat(
    dtype: quant.FloatDType,
    m: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    router_mode: FloatRouterMode,
    with_residual: bool,
) !void {
    const gpa = std.testing.allocator;
    const Q3 = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });

    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const gu_bytes: usize = @as(usize, intermediate_size) * hidden_size * @sizeOf(u16);
    const dn_bytes: usize = gu_bytes;
    const gu_blob = try page.alignUp(gu_bytes);
    const dn_blob = try page.alignUp(dn_bytes);
    const all_experts_bytes = @as(usize, num_experts) * (2 * gu_blob + dn_blob);

    const ws_spec: WorkspaceSpec = .{
        .max_tokens = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
    };
    const ws_bytes = try MoeWorkspace.requiredBytes(ws_spec);
    const table_bytes = @as(usize, num_experts) * @sizeOf(ExpertWeights);
    const x_bytes = @as(usize, m) * hidden_size * @sizeOf(f16);
    const router_bytes = @as(usize, num_experts) * hidden_size * @sizeOf(u16);
    const logits_bytes = @as(usize, m) * num_experts * @sizeOf(f32);
    var arena = try makeTestArena(all_experts_bytes + ws_bytes + table_bytes + 2 * x_bytes +
        2 * router_bytes + logits_bytes + 8 * page.APPLE_PAGE_SIZE);
    defer arena.deinit();
    _ = try arena.allocPages(page.APPLE_PAGE_SIZE);

    const table_raw = try arena.allocAligned(table_bytes, 64);
    const expert_table: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(ExpertWeights, table_raw));
    for (0..num_experts) |e| {
        const blobs = [_][]align(page.APPLE_PAGE_SIZE) u8{
            try arena.allocPages(gu_blob),
            try arena.allocPages(gu_blob),
            try arena.allocPages(dn_blob),
        };
        for (blobs, 0..) |blob, which| {
            const bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, blob[0..gu_bytes]));
            for (bits, 0..) |*v, i| {
                const c: i32 = @as(i32, @intCast((e * 37 + which * 11 + i * 7 + 3) % 41)) - 20;
                v.* = quant.f32ToFloatBits(dtype, @as(f32, @floatFromInt(c)) * 0.004 + @as(f32, @floatFromInt(i % 3)) / 900.0);
            }
        }
        expert_table[e] = .{
            .gate_qweight_offset = arena.offsetOf(blobs[0].ptr),
            .gate_sb_offset = 0,
            .up_qweight_offset = arena.offsetOf(blobs[1].ptr),
            .up_sb_offset = 0,
            .down_qweight_offset = arena.offsetOf(blobs[2].ptr),
            .down_sb_offset = 0,
        };
    }

    const ws = try MoeWorkspace.init(&arena, ws_spec);
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const out_raw = try arena.allocAligned(x_bytes, 64);
    const router_raw = try arena.allocAligned(router_bytes, 64);
    const router_sb_raw = try arena.allocAligned(router_bytes, 64);
    const logits_raw = try arena.allocAligned(logits_bytes, 64);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
    for (x_f16, 0..) |*v, idx| {
        const c: i32 = @as(i32, @intCast((idx * 17 + 11) % 31)) - 15;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.03125);
    }
    const resid_copy = try gpa.alloc(f16, x_f16.len);
    defer gpa.free(resid_copy);
    for (out_f16, 0..) |*v, idx| {
        const c: i32 = @as(i32, @intCast((idx * 13 + 7) % 19)) - 9;
        v.* = if (with_residual) @floatCast(@as(f32, @floatFromInt(c)) * 0.0625) else @floatCast(0.0);
        resid_copy[idx] = v.*;
    }

    const ref_logits = try gpa.alloc(f32, @as(usize, m) * num_experts);
    defer gpa.free(ref_logits);
    var q3_sbs: []quant.ScaleBias = &.{};
    switch (router_mode) {
        .logits => {
            const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
            for (0..m) |tok| {
                for (0..num_experts) |e| {
                    var v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((tok * 13 + e * 5) % 23)) - 11)) * 0.125;
                    if (e < top_k) {
                        v = 3.5 - 0.1 * @as(f32, @floatFromInt(e));
                    } else if (e == (top_k + (tok % 3)) % num_experts) {
                        v = 4.0;
                    }
                    logits_f32[tok * num_experts + e] = v;
                    ref_logits[tok * num_experts + e] = v;
                }
            }
        },
        .bf16_weights => {
            const rw: []u16 = @alignCast(std.mem.bytesAsSlice(u16, router_raw));
            for (0..num_experts) |e| {
                for (0..hidden_size) |k| {
                    const c: i32 = @as(i32, @intCast((e * 11 + k * 3 + 1) % 17)) - 8;
                    const boost: f32 = if (e < top_k + 2) 0.05 else 0.0;
                    rw[e * hidden_size + k] = quant.f32ToBf16(@as(f32, @floatFromInt(c)) * 0.015625 + boost);
                }
            }
            try routerProjFloatRef(.bf16, x_f16, rw, ref_logits, m, num_experts, hidden_size);
        },
        .q3_weights => {
            const qw_bytes = try Q3.packedWeightBytes(num_experts, hidden_size);
            const sb_bytes = try Q3.scaleBiasBytes(num_experts, hidden_size);
            q3_sbs = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, router_sb_raw[0..sb_bytes]));
            try fillSyntheticQuantMatrix(.{ .bits = .q3, .group_size = 64 }, router_raw[0..qw_bytes], q3_sbs, num_experts, hidden_size, 9);
            try routerProjQuantRef(.{ .bits = .q3, .group_size = 64 }, x_f16, router_raw[0..qw_bytes], q3_sbs, ref_logits, m, num_experts, hidden_size);
        },
    }

    const ref_out = try gpa.alloc(f32, @as(usize, m) * hidden_size);
    defer gpa.free(ref_out);
    var route_ref = try moeLayerFloatRef(
        dtype,
        gpa,
        x_f16,
        ref_logits,
        arena.bytes(),
        expert_table,
        if (with_residual) resid_copy else null,
        ref_out,
        m,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        true,
    );
    defer route_ref.deinit(gpa);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const router_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(router_raw.ptr) };
    const out_binding: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) };
    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeMoELayerFloat(&frame, dtype, .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .router = switch (router_mode) {
            .logits => .{ .logits = .{ .binding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) }, .dtype = .f32 } },
            .bf16_weights => .{ .weights = .{ .binding = router_bind, .dtype = .bf16 } },
            .q3_weights => .{ .quant_weights = .{
                .qweight = router_bind,
                .scale_bias = .{ .buffer = gpu_buf, .offset = arena.offsetOf(router_sb_raw.ptr) },
                .bits = .q3,
            } },
        },
        .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
        .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
        .workspace = &ws,
        .workspace_buf = gpu_buf,
        .residual = if (with_residual) out_binding else null,
        .out = out_binding,
        .m = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .norm_topk_prob = true,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    try std.testing.expectEqual(route_ref.num_unique, ws.schedule_header.num_unique_experts);
    if (router_mode == .logits and m > 1) {
        try std.testing.expect(ws.schedule_header.num_unique_experts < m * top_k);
    }
    for (ref_out, 0..) |expected, idx| {
        const got: f32 = @floatCast(out_f16[idx]);
        try std.testing.expectApproxEqAbs(expected, got, @max(0.01, @abs(expected) * 0.01));
    }
}

test "deduplicated FP16 MoE layer matches CPU fp32 reference on M=1 decode and M=8 batch" {
    try verifyFullMoELayerFloat(.f16, 1, 16, 8, 256, 192, .bf16_weights, false);
    try verifyFullMoELayerFloat(.f16, 8, 32, 8, 192, 128, .logits, true);
    try verifyFullMoELayerFloat(.f16, 8, 16, 8, 256, 96, .q3_weights, true);
    try verifyFullMoELayerFloat(.f16, 5, 17, 3, 68, 36, .logits, false);
}

test "deduplicated BF16 MoE layer matches CPU fp32 reference on M=1 decode and M=8 batch" {
    try verifyFullMoELayerFloat(.bf16, 1, 16, 8, 256, 192, .q3_weights, true);
    try verifyFullMoELayerFloat(.bf16, 8, 32, 8, 192, 128, .logits, true);
    try verifyFullMoELayerFloat(.bf16, 8, 16, 4, 128, 64, .bf16_weights, false);
    try verifyFullMoELayerFloat(.bf16, 5, 17, 3, 68, 36, .logits, true);
}

test "FP16 / BF16 MoE encoders reject K not divisible by 4" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();
    const ws_spec: WorkspaceSpec = .{ .max_tokens = 1, .num_experts = 4, .top_k = 2, .hidden_size = 66, .intermediate_size = 64 };
    var arena = try makeTestArena(try MoeWorkspace.requiredBytes(ws_spec));
    defer arena.deinit();
    const ws = try MoeWorkspace.init(&arena, ws_spec);
    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    var frame = try compute.beginFrame(5000);
    defer frame.abort();
    const b: metal.BufferBinding = .{ .buffer = buf };
    try std.testing.expectError(error.UnalignedK, pipes.encodeMoELayerFloat(&frame, .bf16, .{
        .x = b,
        .router = .{ .logits = .{ .binding = b, .dtype = .f32 } },
        .weights_base = b,
        .expert_table = b,
        .workspace = &ws,
        .workspace_buf = buf,
        .out = b,
        .m = 1,
        .num_experts = 4,
        .top_k = 2,
        .hidden_size = 66,
        .intermediate_size = 64,
    }, false));
    try std.testing.expectEqual(@as(u64, 3 * 2 * 2048 * 768 * 8), try expertLayerBytesFloat(8, 2048, 768));
}

const EmbedTokenMode = enum { uniform, buffer };

fn verifyEmbedLookup(dtype: quant.WeightDType, vocab: u32, hidden: u32, m: u32, mode: EmbedTokenMode) !void {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const table_bytes: usize, const sb_bytes: usize = switch (dtype) {
        .f16, .bf16 => .{ @as(usize, vocab) * hidden * 2, 0 },
        .q4_affine => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
            break :blk .{ try Ops.packedWeightBytes(vocab, hidden), try Ops.scaleBiasBytes(vocab, hidden) };
        },
        .q3_affine => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
            break :blk .{ try Ops.packedWeightBytes(vocab, hidden), try Ops.scaleBiasBytes(vocab, hidden) };
        },
        .q2_affine => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
            break :blk .{ try Ops.packedWeightBytes(vocab, hidden), try Ops.scaleBiasBytes(vocab, hidden) };
        },
    };
    const out_bytes: usize = @as(usize, m) * hidden * @sizeOf(f16);
    var arena = try makeTestArena(try page.alignUp(table_bytes) + sb_bytes + out_bytes + 4096);
    defer arena.deinit();
    const table_raw = try arena.allocPages(try page.alignUp(table_bytes));
    const sb_raw = try arena.allocAligned(@max(sb_bytes, 64), 64);
    const ids_raw = try arena.allocAligned(@as(usize, m) * @sizeOf(u32), 64);
    const out_raw = try arena.allocAligned(out_bytes, 64);
    @memset(out_raw, 0xFF);

    var sbs: ?[]quant.ScaleBias = null;
    switch (dtype) {
        .f16, .bf16 => {
            const fdt: quant.FloatDType = if (dtype == .f16) .f16 else .bf16;
            const bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, table_raw[0..table_bytes]));
            for (bits, 0..) |*v, i| {
                const c: i32 = @as(i32, @intCast((i * 31 + 17) % 61)) - 30;
                v.* = quant.f32ToFloatBits(fdt, @as(f32, @floatFromInt(c)) * 0.01 + @as(f32, @floatFromInt(i % 7)) / 7000.0);
            }
        },
        .q4_affine => {
            const s: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sb_raw[0..sb_bytes]));
            try fillSyntheticQuantMatrix(.{ .bits = .q4, .group_size = 64 }, table_raw[0..table_bytes], s, vocab, hidden, 3);
            sbs = s;
        },
        .q3_affine => {
            const s: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sb_raw[0..sb_bytes]));
            try fillSyntheticQuantMatrix(.{ .bits = .q3, .group_size = 64 }, table_raw[0..table_bytes], s, vocab, hidden, 4);
            sbs = s;
        },
        .q2_affine => {
            const s: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sb_raw[0..sb_bytes]));
            try fillSyntheticQuantMatrix(.{ .bits = .q2, .group_size = 64 }, table_raw[0..table_bytes], s, vocab, hidden, 5);
            sbs = s;
        },
    }

    const ids: []u32 = @alignCast(std.mem.bytesAsSlice(u32, ids_raw));
    for (ids, 0..) |*id, i| id.* = @intCast((i * 97 + 13) % vocab);
    if (mode == .uniform) {
        std.debug.assert(m == 1);
        ids[0] = vocab - 1;
    } else if (m > 2) {
        ids[1] = vocab + 5; // out of range -> zero row
        ids[2] = ids[0]; // duplicate token
    }

    const ref = try gpa.alloc(f32, @as(usize, m) * hidden);
    defer gpa.free(ref);
    try embedLookupRef(dtype, table_raw, sbs, ids, ref, vocab, hidden);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeEmbedLookup(&frame, dtype, .{
        .weight = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
        .scale_bias = if (sbs != null) .{ .buffer = gpu_buf, .offset = arena.offsetOf(sb_raw.ptr) } else null,
        .tokens = switch (mode) {
            .uniform => .{ .token_id = ids[0] },
            .buffer => .{ .token_ids = .{ .buffer = gpu_buf, .offset = arena.offsetOf(ids_raw.ptr) } },
        },
        .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
        .m = m,
        .vocab_size = vocab,
        .hidden_size = hidden,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    const out_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
    for (ref, 0..) |expected, i| {
        const got: f32 = @floatCast(out_f16[i]);
        switch (dtype) {
            // f16 copies are bit-exact; bf16 -> f16 is a single RNE rounding.
            .f16, .bf16 => try std.testing.expectEqual(@as(f32, @floatCast(@as(f16, @floatCast(expected)))), got),
            .q4_affine, .q3_affine, .q2_affine => try std.testing.expectApproxEqAbs(expected, got, @max(1e-3, @abs(expected) * 2e-3)),
        }
    }
}

test "GPU embedding lookup matches CPU reference for f16, bf16, q4 and q3 tables" {
    try verifyEmbedLookup(.f16, 1000, 2048, 1, .uniform);
    try verifyEmbedLookup(.f16, 300, 100, 8, .buffer);
    try verifyEmbedLookup(.bf16, 1000, 1024, 1, .uniform);
    try verifyEmbedLookup(.bf16, 300, 1030, 8, .buffer);
    try verifyEmbedLookup(.q4_affine, 257, 640, 8, .buffer);
    try verifyEmbedLookup(.q4_affine, 151, 5120, 1, .uniform);
    try verifyEmbedLookup(.q3_affine, 151, 2048, 1, .uniform);
    try verifyEmbedLookup(.q3_affine, 97, 192, 5, .buffer);
}

test "GPU embedding lookup rejects quantized tables without scale_bias and multi-token uniforms" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();
    var arena = try makeTestArena(page.APPLE_PAGE_SIZE);
    defer arena.deinit();
    _ = try arena.allocPages(page.APPLE_PAGE_SIZE);
    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    var frame = try compute.beginFrame(5000);
    defer frame.abort();
    const b: metal.BufferBinding = .{ .buffer = buf };
    try std.testing.expectError(error.InvalidArgument, pipes.encodeEmbedLookup(&frame, .q3_affine, .{
        .weight = b,
        .tokens = .{ .token_id = 0 },
        .out = b,
        .vocab_size = 8,
        .hidden_size = 64,
    }, false));
    try std.testing.expectError(error.InvalidArgument, pipes.encodeEmbedLookup(&frame, .f16, .{
        .weight = b,
        .tokens = .{ .token_id = 0 },
        .out = b,
        .m = 2,
        .vocab_size = 8,
        .hidden_size = 64,
    }, false));
    try std.testing.expectError(error.UnalignedK, pipes.encodeEmbedLookup(&frame, .q4_affine, .{
        .weight = b,
        .scale_bias = b,
        .tokens = .{ .token_id = 0 },
        .out = b,
        .vocab_size = 8,
        .hidden_size = 100,
    }, false));
}

test "ami_argmax_rows finds exact greedy token IDs across M=1..64, f16/f32, ties, odd and 151936 vocabs" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const cases = [_]struct { dtype: LogitsDType, m: u32, vocab: u32, unaligned: bool }{
        .{ .dtype = .f16, .m = 1, .vocab = 1, .unaligned = false },
        .{ .dtype = .f16, .m = 8, .vocab = 37, .unaligned = true },
        .{ .dtype = .f16, .m = 16, .vocab = 1025, .unaligned = false },
        .{ .dtype = .f16, .m = 64, .vocab = 2048, .unaligned = false },
        .{ .dtype = .f16, .m = 4, .vocab = 151_936, .unaligned = false },
        .{ .dtype = .f32, .m = 8, .vocab = 37, .unaligned = true },
        .{ .dtype = .f32, .m = 16, .vocab = 2048, .unaligned = false },
        .{ .dtype = .f32, .m = 4, .vocab = 64_000, .unaligned = false },
    };

    for (cases) |tc| {
        const elem_bytes: usize = if (tc.dtype == .f16) @sizeOf(f16) else @sizeOf(f32);
        const extra: usize = if (tc.unaligned) elem_bytes else 0;
        const data_bytes: usize = @as(usize, tc.m) * tc.vocab * elem_bytes;
        const out_bytes: usize = @as(usize, tc.m) * @sizeOf(u32);
        var arena = try makeTestArena(data_bytes + extra + out_bytes + 512);
        defer arena.deinit();

        const logits_raw = try arena.allocAligned(data_bytes + extra, 64);
        const out_raw = try arena.allocAligned(out_bytes, 64);
        const out_ids: []u32 = @alignCast(std.mem.bytesAsSlice(u32, out_raw));
        @memset(out_ids, 0xDEAD_BEEF);

        const payload = logits_raw[extra .. extra + data_bytes];
        for (0..tc.m) |row| {
            const winner: usize = (row * 997 + 13) % tc.vocab;
            const tie_idx: usize = @min(tc.vocab - 1, winner + 7);
            const all_neg = (row & 1) == 1;
            for (0..tc.vocab) |i| {
                var val: f32 = if (all_neg)
                    -10.0 - @as(f32, @floatFromInt((row * 17 + i) % 83)) * 0.125
                else
                    @as(f32, @floatFromInt(@as(i32, @intCast((row * 31 + i * 19 + 5) % 61)) - 30)) * 0.25;
                if (i == winner or (tc.vocab > 8 and i == tie_idx)) {
                    val = if (all_neg) -1.25 else 21.5;
                }
                if (tc.dtype == .f16) {
                    const sl: []f16 = @alignCast(std.mem.bytesAsSlice(f16, payload));
                    sl[row * tc.vocab + i] = @floatCast(val);
                } else {
                    const sl: []f32 = @alignCast(std.mem.bytesAsSlice(f32, payload));
                    sl[row * tc.vocab + i] = val;
                }
            }
        }

        const ref_ids = try gpa.alloc(u32, tc.m);
        defer gpa.free(ref_ids);
        argmaxRowsRef(tc.dtype, payload, ref_ids, tc.m, tc.vocab);

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.init(device);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeArgmaxRows(&frame, .{
            .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) + extra },
            .out_token_ids = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .num_rows = tc.m,
            .vocab_size = tc.vocab,
            .logits_dtype = tc.dtype,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        for (0..tc.m) |row| {
            try std.testing.expectEqual(ref_ids[row], out_ids[row]);
        }
    }
}

test "ami_topk_logits_rows computes exact FP32 softmax probabilities and top-k across M=1..64 and 151936 vocab" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const cases = [_]struct { dtype: LogitsDType, m: u32, vocab: u32, top_k: u32 }{
        .{ .dtype = .f16, .m = 1, .vocab = 1, .top_k = 1 },
        .{ .dtype = .f16, .m = 8, .vocab = 37, .top_k = 4 },
        .{ .dtype = .f16, .m = 16, .vocab = 1025, .top_k = 8 },
        .{ .dtype = .f16, .m = 64, .vocab = 2048, .top_k = 16 },
        .{ .dtype = .f16, .m = 4, .vocab = 64_000, .top_k = 8 },
        .{ .dtype = .f16, .m = 4, .vocab = 151_936, .top_k = 16 },
        .{ .dtype = .f32, .m = 8, .vocab = 257, .top_k = 8 },
    };

    for (cases) |tc| {
        const elem_bytes: usize = if (tc.dtype == .f16) @sizeOf(f16) else @sizeOf(f32);
        const data_bytes: usize = @as(usize, tc.m) * tc.vocab * elem_bytes;
        const out_bytes: usize = @as(usize, tc.m) * tc.top_k * @sizeOf(TopKCandidate);
        var arena = try makeTestArena(data_bytes + out_bytes + 512);
        defer arena.deinit();

        const logits_raw = try arena.allocAligned(data_bytes, 64);
        const out_raw = try arena.allocAligned(out_bytes, 64);
        @memset(out_raw, 0);
        const out_cands: []const TopKCandidate = @alignCast(std.mem.bytesAsSlice(TopKCandidate, out_raw));

        for (0..tc.m) |row| {
            // Keep background logits moderate and plant distinct top peaks + ties so
            // both exact tie breaking and full-vocab softmax normalization are exercised.
            for (0..tc.vocab) |i| {
                const c: i32 = @as(i32, @intCast((row * 29 + i * 17 + 7) % 41)) - 20;
                const val: f32 = @as(f32, @floatFromInt(c)) * 0.125;
                if (tc.dtype == .f16) {
                    const sl: []f16 = @alignCast(std.mem.bytesAsSlice(f16, logits_raw));
                    sl[row * tc.vocab + i] = @floatCast(val);
                } else {
                    const sl: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
                    sl[row * tc.vocab + i] = val;
                }
            }
            const num_peaks = @min(tc.top_k + 2, tc.vocab);
            for (0..num_peaks) |p| {
                const idx = (row * 131 + p * 7919 + 3) % tc.vocab;
                // Deliberately include an exact tie between p=1 and p=2.
                const peak_val: f32 = if (p == 2) 10.5 else (11.0 - 0.5 * @as(f32, @floatFromInt(p)));
                if (tc.dtype == .f16) {
                    const sl: []f16 = @alignCast(std.mem.bytesAsSlice(f16, logits_raw));
                    sl[row * tc.vocab + idx] = @floatCast(peak_val);
                } else {
                    const sl: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
                    sl[row * tc.vocab + idx] = peak_val;
                }
            }
        }

        const ref_cands = try gpa.alloc(TopKCandidate, @as(usize, tc.m) * tc.top_k);
        defer gpa.free(ref_cands);
        topKLogitsRowsRef(tc.dtype, logits_raw, ref_cands, tc.m, tc.vocab, tc.top_k);

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.init(device);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeTopKLogitsRows(&frame, .{
            .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .out_candidates = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .num_rows = tc.m,
            .vocab_size = tc.vocab,
            .top_k = tc.top_k,
            .logits_dtype = tc.dtype,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        for (0..tc.m) |row| {
            var prob_sum: f32 = 0.0;
            for (0..tc.top_k) |k_idx| {
                const idx = row * tc.top_k + k_idx;
                const exp_c = ref_cands[idx];
                const got_c = out_cands[idx];
                try std.testing.expectEqual(exp_c.token_id, got_c.token_id);
                try std.testing.expectApproxEqAbs(exp_c.prob, got_c.prob, @max(1e-5, exp_c.prob * 1e-4));
                try std.testing.expectApproxEqAbs(exp_c.log_prob, got_c.log_prob, @max(1e-4, @abs(exp_c.log_prob) * 1e-4));
                prob_sum += got_c.prob;
            }
            try std.testing.expect(prob_sum > 0.0 and prob_sum <= 1.0001);
        }
    }
}

fn verifyPerSlotMatchesDedup(
    comptime spec: quant.QuantSpec,
    m: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
) !void {
    const Ops = quant.QuantOps(spec);
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    const gu_w_bytes = try Ops.packedWeightBytes(intermediate_size, hidden_size);
    const gu_sb_off = amf.align64(gu_w_bytes);
    const gu_sb_bytes = try Ops.scaleBiasBytes(intermediate_size, hidden_size);
    const gu_blob = try page.alignUp(gu_sb_off + gu_sb_bytes);
    const dn_w_bytes = try Ops.packedWeightBytes(hidden_size, intermediate_size);
    const dn_sb_off = amf.align64(dn_w_bytes);
    const dn_sb_bytes = try Ops.scaleBiasBytes(hidden_size, intermediate_size);
    const dn_blob = try page.alignUp(dn_sb_off + dn_sb_bytes);

    const ws_spec: WorkspaceSpec = .{
        .max_tokens = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
    };
    const ws_bytes = try MoeWorkspace.requiredBytes(ws_spec);
    const table_bytes = @as(usize, num_experts) * @sizeOf(ExpertWeights);
    const x_bytes = @as(usize, m) * hidden_size * @sizeOf(f16);
    const logits_bytes = @as(usize, m) * num_experts * @sizeOf(f32);
    var arena = try makeTestArena(@as(usize, num_experts) * (2 * gu_blob + dn_blob) + ws_bytes + table_bytes + x_bytes * 3 + logits_bytes + 8 * page.APPLE_PAGE_SIZE);
    defer arena.deinit();
    _ = try arena.allocPages(page.APPLE_PAGE_SIZE);

    const table_raw = try arena.allocAligned(table_bytes, 64);
    const expert_table: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(ExpertWeights, table_raw));
    for (0..num_experts) |e| {
        const gb = try arena.allocPages(gu_blob);
        const ub = try arena.allocPages(gu_blob);
        const db = try arena.allocPages(dn_blob);
        @memset(gb, 0);
        @memset(ub, 0);
        @memset(db, 0);
        try fillSyntheticQuantMatrix(spec, gb[0..gu_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, gb[gu_sb_off .. gu_sb_off + gu_sb_bytes])), intermediate_size, hidden_size, e * 3 + 1);
        try fillSyntheticQuantMatrix(spec, ub[0..gu_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, ub[gu_sb_off .. gu_sb_off + gu_sb_bytes])), intermediate_size, hidden_size, e * 3 + 2);
        try fillSyntheticQuantMatrix(spec, db[0..dn_w_bytes], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, db[dn_sb_off .. dn_sb_off + dn_sb_bytes])), hidden_size, intermediate_size, e * 3 + 3);
        const go = arena.offsetOf(gb.ptr);
        const uo = arena.offsetOf(ub.ptr);
        const do = arena.offsetOf(db.ptr);
        expert_table[e] = .{
            .gate_qweight_offset = go,
            .gate_sb_offset = go + gu_sb_off,
            .up_qweight_offset = uo,
            .up_sb_offset = uo + gu_sb_off,
            .down_qweight_offset = do,
            .down_sb_offset = do + dn_sb_off,
        };
    }

    const ws = try MoeWorkspace.init(&arena, ws_spec);
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const out_dedup_raw = try arena.allocAligned(x_bytes, 64);
    const out_slot_raw = try arena.allocAligned(x_bytes, 64);
    const logits_raw = try arena.allocAligned(logits_bytes, 64);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const out_dedup: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_dedup_raw));
    const out_slot: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_slot_raw));
    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));

    for (x_f16, 0..) |*v, idx| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((idx * 17 + 11) % 31)) - 15)) * 0.03125);
    fillSpikeS5RoutingBank(.correlated, logits_f32, m, num_experts, 0);

    const ref_out = try gpa.alloc(f32, x_f16.len);
    defer gpa.free(ref_out);
    var rref = try moeLayerRef(spec, gpa, x_f16, logits_f32, arena.bytes(), expert_table, null, ref_out, m, num_experts, top_k, hidden_size, intermediate_size, true);
    defer rref.deinit(gpa);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const base_args: MoELayerArgs = .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .router = .{ .logits = .{ .binding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) }, .dtype = .f32 } },
        .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
        .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
        .workspace = &ws,
        .workspace_buf = gpu_buf,
        .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_dedup_raw.ptr) },
        .m = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
    };

    var f1 = try compute.beginFrame(5000);
    errdefer f1.abort();
    try pipes.encodeMoELayer(&f1, spec.bits, base_args, false);
    const s1 = try f1.commit();
    try compute.wait(s1, 5000);

    var slot_args = base_args;
    slot_args.out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_slot_raw.ptr) };
    var f2 = try compute.beginFrame(5000);
    errdefer f2.abort();
    try pipes.encodeMoELayerPerSlot(&f2, spec.bits, slot_args, false);
    const s2 = try f2.commit();
    try compute.wait(s2, 5000);

    for (ref_out, 0..) |expected, idx| {
        const d_got: f32 = @floatCast(out_dedup[idx]);
        const s_got: f32 = @floatCast(out_slot[idx]);
        const tol: f32 = @max(0.02, @abs(expected) * 0.02);
        try std.testing.expectApproxEqAbs(expected, d_got, tol);
        try std.testing.expectApproxEqAbs(expected, s_got, tol);
        try std.testing.expectApproxEqAbs(d_got, s_got, @max(0.015, @abs(d_got) * 0.015));
    }
}

test "non-deduplicated per-slot 3-bit and 4-bit MoE matches deduplicated CSR MoE and CPU ref" {
    try verifyPerSlotMatchesDedup(.{ .bits = .q3, .group_size = 64 }, 1, 16, 8, 256, 192);
    try verifyPerSlotMatchesDedup(.{ .bits = .q3, .group_size = 64 }, 8, 32, 8, 192, 128);
    try verifyPerSlotMatchesDedup(.{ .bits = .q4, .group_size = 64 }, 1, 16, 8, 256, 192);
    try verifyPerSlotMatchesDedup(.{ .bits = .q4, .group_size = 64 }, 8, 32, 8, 192, 128);
}

test "Spike S5 benchmark smoke test exercises correlated/uniform routing, dedup vs per-slot, and beta table" {
    const gpa = std.testing.allocator;
    const ms = [_]u32{ 1, 4, 8 };
    var rep = try runSpikeS5(gpa, .{
        .bits = .q3,
        .num_experts = 32,
        .top_k = 8,
        .hidden_size = 256,
        .intermediate_size = 192,
        .m_values = &ms,
        .num_routing_banks = 4,
        .min_working_set_bytes = 0,
        .dvfs_warmup_batches = 0,
        .warmup_batches = 1,
        .timed_batches = 2,
        .dispatches_per_batch = 2,
        .timeout_ms = 10_000,
    });
    defer rep.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 6), rep.results.len);
    try std.testing.expect(rep.beta_dedup_gbs > 0.0);
    for (rep.results) |r| {
        try std.testing.expect(r.mean_unique_experts >= 8.0);
        try std.testing.expect(r.dedup_latency_us > 0.0);
        try std.testing.expect(r.perslot_latency_us > 0.0);
        try std.testing.expect(r.dedup_beta_gbs > 0.0);
        try std.testing.expect(r.perslot_beta_unique_gbs > 0.0);
    }
    // Correlated routing at M=8 should have strictly fewer unique experts than uniform.
    try std.testing.expect(rep.results[2].mean_unique_experts <= rep.results[5].mean_unique_experts);
    for (rep.beta_dedup_by_m) |b| try std.testing.expect(b > 0.0);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try rep.write(&aw.writer);
    const rendered = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, rendered, "s5.beta_dedup_gbs") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "correlated") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "uniform") != null);
}

fn fillSyntheticFloatMatrix(
    dtype: quant.FloatDType,
    raw: []u8,
    n: usize,
    k: usize,
    seed: usize,
) void {
    const bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
    std.debug.assert(bits.len >= n * k);
    for (0..n) |row| {
        for (0..k) |col| {
            const c: i32 = @as(i32, @intCast((seed * 29 + row * 17 + col * 11 + 5) % 31)) - 15;
            const v: f32 = @as(f32, @floatFromInt(c)) * 0.015;
            bits[row * k + col] = quant.f32ToFloatBits(dtype, v);
        }
    }
}

fn floatMatmulRowRef(
    dtype: quant.FloatDType,
    x: []const f16,
    w_raw: []const u8,
    out: []f32,
    n: usize,
    k: usize,
) void {
    const bits: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, w_raw));
    for (0..n) |row| {
        var acc: f32 = 0.0;
        const r = bits[row * k ..][0..k];
        for (0..k) |col| {
            const xv: f32 = @floatCast(x[col]);
            const wv: f32 = quant.floatBitsToF32(dtype, r[col]);
            acc += xv * wv;
        }
        out[row] = acc;
    }
}

fn verifyHeteroMoELayer(
    expert_dtypes: []const quant.WeightDType,
    m: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    with_residual: bool,
) !void {
    const gpa = std.testing.allocator;
    const num_experts: u32 = @intCast(expert_dtypes.len);
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    var scheme_counts: [NUM_SCHEME_BUCKETS]u32 = @splat(0);
    var total_weight_bytes: usize = 0;
    for (expert_dtypes) |dt| {
        scheme_counts[@backingInt(dt)] += 1;
        const eb = try singleExpertBytesForDType(dt, hidden_size, intermediate_size);
        total_weight_bytes += try page.alignUp(@intCast(eb)) + 3 * page.APPLE_PAGE_SIZE;
    }

    const ws_spec: WorkspaceSpec = .{
        .max_tokens = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
    };
    const ws_bytes = try MoeWorkspace.requiredBytes(ws_spec);
    const x_bytes: usize = @as(usize, m) * hidden_size * @sizeOf(f16);
    const logits_bytes: usize = @as(usize, m) * num_experts * @sizeOf(f32);
    const table_bytes: usize = @as(usize, num_experts) * @sizeOf(ExpertWeights);
    const schemes_bytes: usize = amf.align64(num_experts);

    var arena = try makeTestArena(
        total_weight_bytes + ws_bytes * 2 + x_bytes * 4 + logits_bytes + table_bytes + schemes_bytes + 4 * page.APPLE_PAGE_SIZE,
    );
    defer arena.deinit();
    _ = try arena.allocPages(page.APPLE_PAGE_SIZE);

    const ws_gpu = try MoeWorkspace.init(&arena, ws_spec);
    var ws_cpu = try MoeWorkspace.init(&arena, ws_spec);
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const resid_raw = try arena.allocAligned(x_bytes, 64);
    const out_gpu_raw = try arena.allocAligned(x_bytes, 64);
    const out_cpu_raw = try arena.allocAligned(x_bytes, 64);
    const logits_raw = try arena.allocAligned(logits_bytes, 64);
    const table_raw = try arena.allocAligned(table_bytes, 64);
    const schemes_raw = try arena.allocAligned(schemes_bytes, 64);
    @memset(out_gpu_raw, 0);
    @memset(out_cpu_raw, 0);
    @memset(schemes_raw, 0);
    for (expert_dtypes, 0..) |dt, e| schemes_raw[e] = @backingInt(dt);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const resid_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, resid_raw));
    for (x_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 13 + 7) % 25)) - 12;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.04);
    }
    for (resid_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 19 + 3) % 21)) - 10;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.05);
    }

    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
    for (0..m) |tok| {
        for (0..num_experts) |e| {
            const c: i32 = @as(i32, @intCast((tok * 23 + e * 11 + 5) % 37)) - 18;
            logits_f32[tok * num_experts + e] = @as(f32, @floatFromInt(c)) * 0.15;
        }
    }

    const tbl: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(ExpertWeights, table_raw));
    for (expert_dtypes, 0..) |dt, e| {
        switch (dt) {
            .f16, .bf16 => {
                const fdt: quant.FloatDType = if (dt == .f16) .f16 else .bf16;
                const gu_bytes: usize = @as(usize, intermediate_size) * hidden_size * 2;
                const dn_bytes: usize = @as(usize, hidden_size) * intermediate_size * 2;
                const g_blob = try arena.allocAligned(gu_bytes, 64);
                const u_blob = try arena.allocAligned(gu_bytes, 64);
                const d_blob = try arena.allocAligned(dn_bytes, 64);
                fillSyntheticFloatMatrix(fdt, g_blob, intermediate_size, hidden_size, e * 3 + 1);
                fillSyntheticFloatMatrix(fdt, u_blob, intermediate_size, hidden_size, e * 3 + 2);
                fillSyntheticFloatMatrix(fdt, d_blob, hidden_size, intermediate_size, e * 3 + 3);
                tbl[e] = .{
                    .gate_qweight_offset = arena.offsetOf(g_blob.ptr),
                    .gate_sb_offset = 0,
                    .up_qweight_offset = arena.offsetOf(u_blob.ptr),
                    .up_sb_offset = 0,
                    .down_qweight_offset = arena.offsetOf(d_blob.ptr),
                    .down_sb_offset = 0,
                };
            },
            inline .q4_affine, .q3_affine, .q2_affine => |q| {
                const bits: quant.QuantBits = switch (q) {
                    .q4_affine => .q4,
                    .q3_affine => .q3,
                    .q2_affine => .q2,
                    else => unreachable,
                };
                const Ops = quant.QuantOps(.{ .bits = bits, .group_size = 64 });
                const gu_w = try Ops.packedWeightBytes(intermediate_size, hidden_size);
                const gu_sb = try Ops.scaleBiasBytes(intermediate_size, hidden_size);
                const dn_w = try Ops.packedWeightBytes(hidden_size, intermediate_size);
                const dn_sb = try Ops.scaleBiasBytes(hidden_size, intermediate_size);
                const g_w_raw = try arena.allocAligned(gu_w, 64);
                const g_sb_raw = try arena.allocAligned(gu_sb, 64);
                const u_w_raw = try arena.allocAligned(gu_w, 64);
                const u_sb_raw = try arena.allocAligned(gu_sb, 64);
                const d_w_raw = try arena.allocAligned(dn_w, 64);
                const d_sb_raw = try arena.allocAligned(dn_sb, 64);
                try fillSyntheticQuantMatrix(.{ .bits = bits, .group_size = 64 }, g_w_raw, @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, g_sb_raw)), intermediate_size, hidden_size, e * 3 + 1);
                try fillSyntheticQuantMatrix(.{ .bits = bits, .group_size = 64 }, u_w_raw, @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, u_sb_raw)), intermediate_size, hidden_size, e * 3 + 2);
                try fillSyntheticQuantMatrix(.{ .bits = bits, .group_size = 64 }, d_w_raw, @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, d_sb_raw)), hidden_size, intermediate_size, e * 3 + 3);
                tbl[e] = .{
                    .gate_qweight_offset = arena.offsetOf(g_w_raw.ptr),
                    .gate_sb_offset = arena.offsetOf(g_sb_raw.ptr),
                    .up_qweight_offset = arena.offsetOf(u_w_raw.ptr),
                    .up_sb_offset = arena.offsetOf(u_sb_raw.ptr),
                    .down_qweight_offset = arena.offsetOf(d_w_raw.ptr),
                    .down_sb_offset = arena.offsetOf(d_sb_raw.ptr),
                };
            },
        }
    }

    // CPU fp32 reference over heterogeneous per-expert schemes.
    var route_ref = try routeTopKRef(gpa, logits_f32, m, num_experts, top_k, true);
    defer route_ref.deinit(gpa);
    const ref_out = try gpa.alloc(f32, @as(usize, m) * hidden_size);
    defer gpa.free(ref_out);
    if (with_residual) {
        for (resid_f16, 0..) |r, i| ref_out[i] = @floatCast(r);
    } else {
        @memset(ref_out, 0.0);
    }
    const gate_buf = try gpa.alloc(f32, intermediate_size);
    defer gpa.free(gate_buf);
    const up_buf = try gpa.alloc(f32, intermediate_size);
    defer gpa.free(up_buf);
    const inter_f16 = try gpa.alloc(f16, intermediate_size);
    defer gpa.free(inter_f16);
    const down_buf = try gpa.alloc(f32, hidden_size);
    defer gpa.free(down_buf);

    const wbase = arena.bytes();
    for (0..m) |tok| {
        const x_tok = x_f16[tok * hidden_size ..][0..hidden_size];
        const out_tok = ref_out[tok * hidden_size ..][0..hidden_size];
        for (0..top_k) |k_idx| {
            const e = route_ref.topk_experts[tok * top_k + k_idx];
            const rw = route_ref.topk_weights[tok * top_k + k_idx];
            const ew = tbl[e];
            switch (expert_dtypes[e]) {
                .f16, .bf16 => {
                    const fdt: quant.FloatDType = if (expert_dtypes[e] == .f16) .f16 else .bf16;
                    const gu_bytes: usize = @as(usize, intermediate_size) * hidden_size * 2;
                    const dn_bytes: usize = @as(usize, hidden_size) * intermediate_size * 2;
                    floatMatmulRowRef(fdt, x_tok, wbase[@intCast(ew.gate_qweight_offset)..][0..gu_bytes], gate_buf, intermediate_size, hidden_size);
                    floatMatmulRowRef(fdt, x_tok, wbase[@intCast(ew.up_qweight_offset)..][0..gu_bytes], up_buf, intermediate_size, hidden_size);
                    for (0..intermediate_size) |i| inter_f16[i] = @floatCast(siluF32(gate_buf[i]) * up_buf[i]);
                    floatMatmulRowRef(fdt, inter_f16, wbase[@intCast(ew.down_qweight_offset)..][0..dn_bytes], down_buf, hidden_size, intermediate_size);
                },
                inline .q4_affine, .q3_affine, .q2_affine => |q| {
                    const bits: quant.QuantBits = switch (q) {
                        .q4_affine => .q4,
                        .q3_affine => .q3,
                        .q2_affine => .q2,
                        else => unreachable,
                    };
                    const Ops = quant.QuantOps(.{ .bits = bits, .group_size = 64 });
                    const gu_w = try Ops.packedWeightBytes(intermediate_size, hidden_size);
                    const gu_sb = try Ops.scaleBiasBytes(intermediate_size, hidden_size);
                    const dn_w = try Ops.packedWeightBytes(hidden_size, intermediate_size);
                    const dn_sb = try Ops.scaleBiasBytes(hidden_size, intermediate_size);
                    try Ops.matmulRef(x_tok, wbase[@intCast(ew.gate_qweight_offset)..][0..gu_w], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, wbase[@intCast(ew.gate_sb_offset)..][0..gu_sb])), gate_buf, 1, intermediate_size, hidden_size);
                    try Ops.matmulRef(x_tok, wbase[@intCast(ew.up_qweight_offset)..][0..gu_w], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, wbase[@intCast(ew.up_sb_offset)..][0..gu_sb])), up_buf, 1, intermediate_size, hidden_size);
                    for (0..intermediate_size) |i| inter_f16[i] = @floatCast(siluF32(gate_buf[i]) * up_buf[i]);
                    try Ops.matmulRef(inter_f16, wbase[@intCast(ew.down_qweight_offset)..][0..dn_w], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, wbase[@intCast(ew.down_sb_offset)..][0..dn_sb])), down_buf, 1, hidden_size, intermediate_size);
                },
            }
            for (0..hidden_size) |i| out_tok[i] += down_buf[i] * rw;
        }
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const base_gpu: MoELayerArgs = .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .router = .{ .logits = .{ .binding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) }, .dtype = .f32 } },
        .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
        .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
        .workspace = &ws_gpu,
        .workspace_buf = gpu_buf,
        .residual = if (with_residual) .{ .buffer = gpu_buf, .offset = arena.offsetOf(resid_raw.ptr) } else null,
        .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_gpu_raw.ptr) },
        .m = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .norm_topk_prob = true,
    };

    // Path 1: Zero-sync GPU scheme-bucketed routing + heterogeneous FFN.
    var f1 = try compute.beginFrame(5000);
    errdefer f1.abort();
    try pipes.encodeMoELayerHetero(&f1, .{
        .base = base_gpu,
        .expert_schemes = .{ .buffer = gpu_buf, .offset = arena.offsetOf(schemes_raw.ptr) },
        .scheme_counts = scheme_counts,
    }, false);
    const s1 = try f1.commit();
    try compute.wait(s1, 5000);

    // Path 2: Standard route_topk -> CPU zero-alloc `rebuildSchemeBucketsCpu` ->
    // `encodeGateUpDownHetero(..., skip_empty_from_host = true)` + reduce.
    var f2 = try compute.beginFrame(5000);
    errdefer f2.abort();
    try pipes.encodeRouteTopK(&f2, .f32, .{
        .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) },
        .workspace = &ws_cpu,
        .workspace_buf = gpu_buf,
        .m = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .norm_topk_prob = true,
    }, false);
    const s2 = try f2.commit();
    try compute.wait(s2, 5000);

    ws_cpu.rebuildSchemeBucketsCpu(expert_dtypes);

    // Verify GPU bucketed routing and CPU rebuildSchemeBucketsCpu produce identical buckets.
    var total_bucket_u: u32 = 0;
    var total_bucket_a: u32 = 0;
    for (0..NUM_SCHEME_BUCKETS) |s_idx| {
        const scheme: quant.WeightDType = @fromBackingInt(@intCast(s_idx));
        const bh_gpu = ws_gpu.bucketHeader(scheme).*;
        const bh_cpu = ws_cpu.bucketHeader(scheme).*;
        try std.testing.expectEqual(bh_gpu.num_unique_experts, bh_cpu.num_unique_experts);
        try std.testing.expectEqual(bh_gpu.total_assignments, bh_cpu.total_assignments);
        total_bucket_u += bh_gpu.num_unique_experts;
        total_bucket_a += bh_gpu.total_assignments;
        const u_base = s_idx * ws_gpu.bucket_unique_stride_u32;
        const o_base = s_idx * ws_gpu.bucket_offsets_stride_u32;
        for (0..bh_gpu.num_unique_experts) |ui| {
            const eid = ws_gpu.bucket_unique_experts[u_base + ui];
            try std.testing.expectEqual(scheme, expert_dtypes[eid]);
            try std.testing.expectEqual(eid, ws_cpu.bucket_unique_experts[u_base + ui]);
            try std.testing.expectEqual(ws_gpu.bucket_expert_token_offsets[o_base + ui], ws_cpu.bucket_expert_token_offsets[o_base + ui]);
        }
        try std.testing.expectEqual(ws_gpu.bucket_expert_token_offsets[o_base + bh_gpu.num_unique_experts], ws_cpu.bucket_expert_token_offsets[o_base + bh_cpu.num_unique_experts]);
    }
    try std.testing.expectEqual(route_ref.num_unique, total_bucket_u);
    try std.testing.expectEqual(m * top_k, total_bucket_a);

    var base_cpu = base_gpu;
    base_cpu.workspace = &ws_cpu;
    base_cpu.out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_cpu_raw.ptr) };
    var f3 = try compute.beginFrame(5000);
    errdefer f3.abort();
    try pipes.encodeGateUpDownHetero(&f3, base_cpu, scheme_counts, true, false);
    try pipes.encodeReduce(&f3, .{
        .workspace = &ws_cpu,
        .workspace_buf = gpu_buf,
        .residual = base_cpu.residual,
        .out = base_cpu.out,
        .m = m,
        .hidden_size = hidden_size,
        .top_k = top_k,
    }, true);
    const s3 = try f3.commit();
    try compute.wait(s3, 5000);

    const out_gpu: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_gpu_raw));
    const out_cpu: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_cpu_raw));
    for (ref_out, 0..) |expected, idx| {
        const g_val: f32 = @floatCast(out_gpu[idx]);
        const c_val: f32 = @floatCast(out_cpu[idx]);
        const tol: f32 = @max(0.02, @abs(expected) * 0.02);
        try std.testing.expectApproxEqAbs(expected, g_val, tol);
        try std.testing.expectApproxEqAbs(expected, c_val, tol);
        try std.testing.expectApproxEqAbs(g_val, c_val, @max(0.005, @abs(g_val) * 0.005));
    }
}

test "q2_affine Metal 4 kernels (MoE, router, embed, perslot) match CPU fp32 reference" {
    // 2-bit full deduplicated MoE & per-slot parity (M=1 decode and M=8 batch).
    try verifyPerSlotMatchesDedup(.{ .bits = .q2, .group_size = 64 }, 1, 16, 8, 256, 192);
    try verifyPerSlotMatchesDedup(.{ .bits = .q2, .group_size = 64 }, 8, 32, 8, 192, 128);
    // 2-bit router projection.
    try verifyRouterQuant(.{ .bits = .q2, .group_size = 64 }, 1, 128, 2048);
    try verifyRouterQuant(.{ .bits = .q2, .group_size = 64 }, 5, 33, 320);
    // 2-bit GPU embedding lookup (uniform single-token & batched buffer).
    try verifyEmbedLookup(.q2_affine, 151, 2048, 1, .uniform);
    try verifyEmbedLookup(.q2_affine, 97, 192, 5, .buffer);
    // 2-bit exact expert byte count check (Qwen3-30B-A3B geometry: hidden=2048, inter=768).
    // gate/up: 768 * (2048*2/8 + 32*4) = 768 * (512 + 128) = 491,520 B each
    // down:    2048 * (768*2/8 + 12*4) = 2048 * (192 + 48) = 491,520 B
    // 1 expert = 1,474,560 B; 8 experts = 11,796,480 B.
    try std.testing.expectEqual(@as(u64, 1_474_560), try singleExpertBytesForDType(.q2_affine, 2048, 768));
    try std.testing.expectEqual(@as(u64, 11_796_480), try expertLayerBytes(.q2, 8, 2048, 768));
}

test "G13 heterogeneous scheme-bucketed MoE mixes {q2,q3,q4,f16,bf16} experts matching CPU fp32 ref" {
    // Mix all 5 schemes in one layer (E=20, top_k=6, M=1 decode and M=8 batch).
    const all5 = [_]quant.WeightDType{
        .q2_affine, .q3_affine, .q4_affine, .f16,       .bf16,
        .q2_affine, .q2_affine, .q3_affine, .q4_affine, .bf16,
        .f16,       .q3_affine, .q2_affine, .q4_affine, .q2_affine,
        .bf16,      .q3_affine, .q2_affine, .f16,       .q4_affine,
    };
    try verifyHeteroMoELayer(&all5, 1, 6, 256, 192, false);
    try verifyHeteroMoELayer(&all5, 8, 6, 256, 192, true);

    // Realistic GSQ 3-scheme MoE layer {q2_affine, q3_affine, q4_affine} with E=32, top_k=8 (Qwen3-style).
    var gsq3: [32]quant.WeightDType = undefined;
    for (&gsq3, 0..) |*dt, e| {
        dt.* = if (e < 4) .q4_affine else if (e < 14) .q3_affine else .q2_affine;
    }
    try verifyHeteroMoELayer(&gsq3, 1, 8, 256, 192, true);
    try verifyHeteroMoELayer(&gsq3, 8, 8, 192, 128, false);
}

fn evalSharedExpertCpuRef(
    arena_bytes: []const u8,
    x_f16: []const f16,
    se: ExpertWeights,
    se_dtype: quant.WeightDType,
    gate_w_raw: ?[]const u8,
    gate_dtype: RouterWeightDType,
    m: u32,
    hidden_size: u32,
    shared_inter: u32,
    out_accum: []f32,
    gpa: std.mem.Allocator,
) !void {
    const g_out = try gpa.alloc(f32, shared_inter);
    defer gpa.free(g_out);
    const u_out = try gpa.alloc(f32, shared_inter);
    defer gpa.free(u_out);
    const inter_f16 = try gpa.alloc(f16, shared_inter);
    defer gpa.free(inter_f16);
    const d_out = try gpa.alloc(f32, hidden_size);
    defer gpa.free(d_out);

    for (0..m) |tok| {
        const x_row = x_f16[tok * hidden_size ..][0..hidden_size];
        const gate_scale: f32 = if (gate_w_raw) |gw| blk: {
            var dot_acc: f32 = 0.0;
            switch (gate_dtype) {
                .f16 => {
                    const w: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, gw));
                    for (0..hidden_size) |k| dot_acc += @as(f32, @floatCast(x_row[k])) * @as(f32, @floatCast(w[k]));
                },
                .bf16 => {
                    const w: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, gw));
                    for (0..hidden_size) |k| dot_acc += @as(f32, @floatCast(x_row[k])) * quant.bf16ToF32(w[k]);
                },
                .f32 => {
                    const w: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, gw));
                    for (0..hidden_size) |k| dot_acc += @as(f32, @floatCast(x_row[k])) * w[k];
                },
            }
            break :blk 1.0 / (1.0 + @exp(-dot_acc));
        } else 1.0;

        switch (se_dtype) {
            .f16, .bf16 => {
                const fdt: quant.FloatDType = if (se_dtype == .f16) .f16 else .bf16;
                const gu_bytes = @as(usize, shared_inter) * hidden_size * 2;
                const dn_bytes = @as(usize, hidden_size) * shared_inter * 2;
                floatMatmulRowRef(fdt, x_row, arena_bytes[@intCast(se.gate_qweight_offset)..][0..gu_bytes], g_out, shared_inter, hidden_size);
                floatMatmulRowRef(fdt, x_row, arena_bytes[@intCast(se.up_qweight_offset)..][0..gu_bytes], u_out, shared_inter, hidden_size);
                for (0..shared_inter) |i| {
                    inter_f16[i] = @floatCast((g_out[i] / (1.0 + @exp(-g_out[i]))) * u_out[i]);
                }
                floatMatmulRowRef(fdt, inter_f16, arena_bytes[@intCast(se.down_qweight_offset)..][0..dn_bytes], d_out, hidden_size, shared_inter);
            },
            inline .q4_affine, .q3_affine, .q2_affine => |q| {
                const bits: quant.QuantBits = switch (q) {
                    .q4_affine => .q4,
                    .q3_affine => .q3,
                    .q2_affine => .q2,
                    else => unreachable,
                };
                const Ops = quant.QuantOps(.{ .bits = bits, .group_size = 64 });
                const gu_w_bytes = try Ops.packedWeightBytes(shared_inter, hidden_size);
                const gu_sb_bytes = try Ops.scaleBiasBytes(shared_inter, hidden_size);
                const dn_w_bytes = try Ops.packedWeightBytes(hidden_size, shared_inter);
                const dn_sb_bytes = try Ops.scaleBiasBytes(hidden_size, shared_inter);
                const g_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    arena_bytes[@intCast(se.gate_sb_offset)..][0..gu_sb_bytes],
                ));
                const u_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    arena_bytes[@intCast(se.up_sb_offset)..][0..gu_sb_bytes],
                ));
                const d_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    arena_bytes[@intCast(se.down_sb_offset)..][0..dn_sb_bytes],
                ));
                try Ops.matmulRef(x_row, arena_bytes[@intCast(se.gate_qweight_offset)..][0..gu_w_bytes], g_sb, g_out, 1, shared_inter, hidden_size);
                try Ops.matmulRef(x_row, arena_bytes[@intCast(se.up_qweight_offset)..][0..gu_w_bytes], u_sb, u_out, 1, shared_inter, hidden_size);
                for (0..shared_inter) |i| {
                    inter_f16[i] = @floatCast((g_out[i] / (1.0 + @exp(-g_out[i]))) * u_out[i]);
                }
                try Ops.matmulRef(inter_f16, arena_bytes[@intCast(se.down_qweight_offset)..][0..dn_w_bytes], d_sb, d_out, 1, hidden_size, shared_inter);
            },
        }

        for (0..hidden_size) |h| {
            out_accum[tok * hidden_size + h] += d_out[h] * gate_scale;
        }
    }
}

fn fillSyntheticExpertForDType(
    arena: *arena_mod.WiredArena,
    dt: quant.WeightDType,
    hidden_size: u32,
    intermediate_size: u32,
    seed: usize,
) !ExpertWeights {
    switch (dt) {
        .f16, .bf16 => {
            const fdt: quant.FloatDType = if (dt == .f16) .f16 else .bf16;
            const gu_bytes: usize = @as(usize, intermediate_size) * hidden_size * 2;
            const dn_bytes: usize = @as(usize, hidden_size) * intermediate_size * 2;
            const g_blob = try arena.allocAligned(gu_bytes, 64);
            const u_blob = try arena.allocAligned(gu_bytes, 64);
            const d_blob = try arena.allocAligned(dn_bytes, 64);
            fillSyntheticFloatMatrix(fdt, g_blob, intermediate_size, hidden_size, seed * 3 + 1);
            fillSyntheticFloatMatrix(fdt, u_blob, intermediate_size, hidden_size, seed * 3 + 2);
            fillSyntheticFloatMatrix(fdt, d_blob, hidden_size, intermediate_size, seed * 3 + 3);
            return .{
                .gate_qweight_offset = arena.offsetOf(g_blob.ptr),
                .gate_sb_offset = 0,
                .up_qweight_offset = arena.offsetOf(u_blob.ptr),
                .up_sb_offset = 0,
                .down_qweight_offset = arena.offsetOf(d_blob.ptr),
                .down_sb_offset = 0,
            };
        },
        inline .q4_affine, .q3_affine, .q2_affine => |q| {
            const bits: quant.QuantBits = switch (q) {
                .q4_affine => .q4,
                .q3_affine => .q3,
                .q2_affine => .q2,
                else => unreachable,
            };
            const Ops = quant.QuantOps(.{ .bits = bits, .group_size = 64 });
            const gu_w_bytes = try Ops.packedWeightBytes(intermediate_size, hidden_size);
            const gu_sb_bytes = try Ops.scaleBiasBytes(intermediate_size, hidden_size);
            const dn_w_bytes = try Ops.packedWeightBytes(hidden_size, intermediate_size);
            const dn_sb_bytes = try Ops.scaleBiasBytes(hidden_size, intermediate_size);

            const g_qw = try arena.allocAligned(gu_w_bytes, 64);
            const g_sb_raw = try arena.allocAligned(gu_sb_bytes, 64);
            const u_qw = try arena.allocAligned(gu_w_bytes, 64);
            const u_sb_raw = try arena.allocAligned(gu_sb_bytes, 64);
            const d_qw = try arena.allocAligned(dn_w_bytes, 64);
            const d_sb_raw = try arena.allocAligned(dn_sb_bytes, 64);

            const g_sb: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, g_sb_raw));
            const u_sb: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, u_sb_raw));
            const d_sb: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, d_sb_raw));

            try fillSyntheticQuantMatrix(.{ .bits = bits, .group_size = 64 }, g_qw, g_sb, intermediate_size, hidden_size, seed * 3 + 1);
            try fillSyntheticQuantMatrix(.{ .bits = bits, .group_size = 64 }, u_qw, u_sb, intermediate_size, hidden_size, seed * 3 + 2);
            try fillSyntheticQuantMatrix(.{ .bits = bits, .group_size = 64 }, d_qw, d_sb, hidden_size, intermediate_size, seed * 3 + 3);

            return .{
                .gate_qweight_offset = arena.offsetOf(g_qw.ptr),
                .gate_sb_offset = arena.offsetOf(g_sb_raw.ptr),
                .up_qweight_offset = arena.offsetOf(u_qw.ptr),
                .up_sb_offset = arena.offsetOf(u_sb_raw.ptr),
                .down_qweight_offset = arena.offsetOf(d_qw.ptr),
                .down_sb_offset = arena.offsetOf(d_sb_raw.ptr),
            };
        },
    }
}

fn verifyHighExpertWithShared(
    expert_dtypes: []const quant.WeightDType,
    m: u32,
    top_k: u32,
    hidden_size: u32,
    intermediate_size: u32,
    shared_dtype: quant.WeightDType,
    shared_inter: u32,
    dedicated_shared_ws: bool,
    with_shared_gate: bool,
    shared_gate_dtype: RouterWeightDType,
    with_residual: bool,
    use_hetero: bool,
) !void {
    const gpa = std.testing.allocator;
    const num_experts: u32 = @intCast(expert_dtypes.len);
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    var scheme_counts: [NUM_SCHEME_BUCKETS]u32 = @splat(0);
    var total_weight_bytes: usize = 0;
    for (expert_dtypes) |dt| {
        scheme_counts[@backingInt(dt)] += 1;
        const eb = try singleExpertBytesForDType(dt, hidden_size, intermediate_size);
        total_weight_bytes += amf.align64(@intCast(eb)) + 512;
    }
    const shared_eb = try singleExpertBytesForDType(shared_dtype, hidden_size, shared_inter);
    total_weight_bytes += amf.align64(@intCast(shared_eb)) + 1024;

    const ws_spec: WorkspaceSpec = .{
        .max_tokens = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .shared_intermediate_size = if (dedicated_shared_ws) shared_inter else 0,
    };
    const ws_bytes = try MoeWorkspace.requiredBytes(ws_spec);
    const x_bytes: usize = @as(usize, m) * hidden_size * @sizeOf(f16);
    const logits_bytes: usize = @as(usize, m) * num_experts * @sizeOf(f32);
    const table_bytes: usize = @as(usize, num_experts) * @sizeOf(ExpertWeights);
    const schemes_bytes: usize = amf.align64(num_experts);
    const gate_w_bytes: usize = amf.align64(@as(usize, hidden_size) * @sizeOf(f32));

    var arena = try makeTestArena(
        total_weight_bytes + ws_bytes + x_bytes * 3 + logits_bytes + table_bytes + schemes_bytes + gate_w_bytes + 8 * page.APPLE_PAGE_SIZE,
    );
    defer arena.deinit();
    _ = try arena.allocPages(page.APPLE_PAGE_SIZE);

    const ws = try MoeWorkspace.init(&arena, ws_spec);
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const resid_raw = try arena.allocAligned(x_bytes, 64);
    const out_raw = try arena.allocAligned(x_bytes, 64);
    const logits_raw = try arena.allocAligned(logits_bytes, 64);
    const table_raw = try arena.allocAligned(table_bytes, 64);
    const schemes_raw = try arena.allocAligned(schemes_bytes, 64);
    const gate_w_raw = try arena.allocAligned(gate_w_bytes, 64);
    @memset(out_raw, 0);
    @memset(schemes_raw, 0);
    @memset(gate_w_raw, 0);
    for (expert_dtypes, 0..) |dt, e| schemes_raw[e] = @backingInt(dt);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const resid_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, resid_raw));
    for (x_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 13 + 7) % 25)) - 12;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.04);
    }
    for (resid_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 19 + 3) % 21)) - 10;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.05);
    }
    switch (shared_gate_dtype) {
        .f16 => {
            const w: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate_w_raw[0 .. @as(usize, hidden_size) * 2]));
            for (w, 0..) |*v, i| {
                const c: i32 = @as(i32, @intCast((i * 7 + 5) % 19)) - 9;
                v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.06);
            }
        },
        .bf16 => {
            const w: []u16 = @alignCast(std.mem.bytesAsSlice(u16, gate_w_raw[0 .. @as(usize, hidden_size) * 2]));
            for (w, 0..) |*v, i| {
                const c: i32 = @as(i32, @intCast((i * 7 + 5) % 19)) - 9;
                v.* = quant.f32ToBf16(@as(f32, @floatFromInt(c)) * 0.06);
            }
        },
        .f32 => {
            const w: []f32 = @alignCast(std.mem.bytesAsSlice(f32, gate_w_raw[0 .. @as(usize, hidden_size) * 4]));
            for (w, 0..) |*v, i| {
                const c: i32 = @as(i32, @intCast((i * 7 + 5) % 19)) - 9;
                v.* = @as(f32, @floatFromInt(c)) * 0.06;
            }
        },
    }

    // Spread logits across all 512..1024 experts so high indices (>= 256) are guaranteed active,
    // while also forcing overlapping experts across tokens when m > 1.
    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
    for (0..m) |tok| {
        for (0..num_experts) |e| {
            const base_v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((tok * 29 + e * 13 + 7) % 41)) - 20)) * 0.1;
            const boost: f32 = if (e == num_experts - 1 or e == num_experts / 2 + (tok % 3) or e == 300 + tok) 4.5 else 0.0;
            logits_f32[tok * num_experts + e] = base_v + boost;
        }
    }

    const tbl: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(ExpertWeights, table_raw));
    for (expert_dtypes, 0..) |dt, e| {
        tbl[e] = try fillSyntheticExpertForDType(&arena, dt, hidden_size, intermediate_size, e + 1);
    }
    const se_weights = try fillSyntheticExpertForDType(&arena, shared_dtype, hidden_size, shared_inter, num_experts + 77);

    // CPU reference: routed MoE + optional residual + shared_expert.
    var route_ref = try routeTopKRef(gpa, logits_f32, m, num_experts, top_k, true);
    defer route_ref.deinit(gpa);

    const ref_out = try gpa.alloc(f32, @as(usize, m) * hidden_size);
    defer gpa.free(ref_out);
    for (0..m) |tok| {
        for (0..hidden_size) |h| {
            ref_out[tok * hidden_size + h] = if (with_residual) @floatCast(resid_f16[tok * hidden_size + h]) else 0.0;
        }
    }
    const g_tmp = try gpa.alloc(f32, intermediate_size);
    defer gpa.free(g_tmp);
    const u_tmp = try gpa.alloc(f32, intermediate_size);
    defer gpa.free(u_tmp);
    const inter_tmp = try gpa.alloc(f16, intermediate_size);
    defer gpa.free(inter_tmp);
    const d_tmp = try gpa.alloc(f32, hidden_size);
    defer gpa.free(d_tmp);
    for (0..m) |tok| {
        const x_row = x_f16[tok * hidden_size ..][0..hidden_size];
        for (0..top_k) |k_idx| {
            const eid = route_ref.topk_experts[tok * top_k + k_idx];
            const rw = route_ref.topk_weights[tok * top_k + k_idx];
            const ew = tbl[eid];
            switch (expert_dtypes[eid]) {
                .f16, .bf16 => {
                    const fdt: quant.FloatDType = if (expert_dtypes[eid] == .f16) .f16 else .bf16;
                    const gu_bytes = @as(usize, intermediate_size) * hidden_size * 2;
                    const dn_bytes = @as(usize, hidden_size) * intermediate_size * 2;
                    floatMatmulRowRef(fdt, x_row, arena.bytes()[@intCast(ew.gate_qweight_offset)..][0..gu_bytes], g_tmp, intermediate_size, hidden_size);
                    floatMatmulRowRef(fdt, x_row, arena.bytes()[@intCast(ew.up_qweight_offset)..][0..gu_bytes], u_tmp, intermediate_size, hidden_size);
                    for (0..intermediate_size) |i| inter_tmp[i] = @floatCast((g_tmp[i] / (1.0 + @exp(-g_tmp[i]))) * u_tmp[i]);
                    floatMatmulRowRef(fdt, inter_tmp, arena.bytes()[@intCast(ew.down_qweight_offset)..][0..dn_bytes], d_tmp, hidden_size, intermediate_size);
                },
                inline .q4_affine, .q3_affine, .q2_affine => |q| {
                    const bits: quant.QuantBits = switch (q) {
                        .q4_affine => .q4,
                        .q3_affine => .q3,
                        .q2_affine => .q2,
                        else => unreachable,
                    };
                    const Ops = quant.QuantOps(.{ .bits = bits, .group_size = 64 });
                    const gu_w_bytes = try Ops.packedWeightBytes(intermediate_size, hidden_size);
                    const gu_sb_bytes = try Ops.scaleBiasBytes(intermediate_size, hidden_size);
                    const dn_w_bytes = try Ops.packedWeightBytes(hidden_size, intermediate_size);
                    const dn_sb_bytes = try Ops.scaleBiasBytes(hidden_size, intermediate_size);
                    const g_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, arena.bytes()[@intCast(ew.gate_sb_offset)..][0..gu_sb_bytes]));
                    const u_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, arena.bytes()[@intCast(ew.up_sb_offset)..][0..gu_sb_bytes]));
                    const d_sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, arena.bytes()[@intCast(ew.down_sb_offset)..][0..dn_sb_bytes]));
                    try Ops.matmulRef(x_row, arena.bytes()[@intCast(ew.gate_qweight_offset)..][0..gu_w_bytes], g_sb, g_tmp, 1, intermediate_size, hidden_size);
                    try Ops.matmulRef(x_row, arena.bytes()[@intCast(ew.up_qweight_offset)..][0..gu_w_bytes], u_sb, u_tmp, 1, intermediate_size, hidden_size);
                    for (0..intermediate_size) |i| inter_tmp[i] = @floatCast((g_tmp[i] / (1.0 + @exp(-g_tmp[i]))) * u_tmp[i]);
                    try Ops.matmulRef(inter_tmp, arena.bytes()[@intCast(ew.down_qweight_offset)..][0..dn_w_bytes], d_sb, d_tmp, 1, hidden_size, intermediate_size);
                },
            }
            for (0..hidden_size) |h| ref_out[tok * hidden_size + h] += rw * d_tmp[h];
        }
    }

    const gate_slice_opt: ?[]const u8 = if (with_shared_gate)
        gate_w_raw[0 .. @as(usize, hidden_size) * (if (shared_gate_dtype == .f32) @as(usize, 4) else @as(usize, 2))]
    else
        null;
    try evalSharedExpertCpuRef(
        arena.bytes(),
        x_f16,
        se_weights,
        shared_dtype,
        gate_slice_opt,
        shared_gate_dtype,
        m,
        hidden_size,
        shared_inter,
        ref_out,
        gpa,
    );

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const base_args: MoELayerArgs = .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .router = .{ .logits = .{ .binding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) }, .dtype = .f32 } },
        .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
        .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
        .workspace = &ws,
        .workspace_buf = gpu_buf,
        .residual = if (with_residual) .{ .buffer = gpu_buf, .offset = arena.offsetOf(resid_raw.ptr) } else null,
        .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
        .m = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .norm_topk_prob = true,
        .shared_expert = se_weights,
        .shared_expert_dtype = shared_dtype,
        .shared_intermediate_size = shared_inter,
        .shared_gate_weight = if (with_shared_gate) .{ .buffer = gpu_buf, .offset = arena.offsetOf(gate_w_raw.ptr) } else null,
        .shared_gate_dtype = shared_gate_dtype,
    };

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    if (use_hetero) {
        try pipes.encodeMoELayerHetero(&frame, .{
            .base = base_args,
            .expert_schemes = .{ .buffer = gpu_buf, .offset = arena.offsetOf(schemes_raw.ptr) },
            .scheme_counts = scheme_counts,
        }, false);
    } else {
        try pipes.encodeMoELayer(&frame, .q3, base_args, false);
    }
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    // Verify routing activated experts with IDs >= 256.
    var saw_high_expert = false;
    for (0..ws.schedule_header.num_unique_experts) |ui| {
        if (ws.unique_experts[ui] >= 256) saw_high_expert = true;
    }
    try std.testing.expect(saw_high_expert);
    try std.testing.expectEqual(route_ref.num_unique, ws.schedule_header.num_unique_experts);

    const out_gpu: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
    for (ref_out, 0..) |expected, idx| {
        const g_val: f32 = @floatCast(out_gpu[idx]);
        const tol: f32 = @max(0.025, @abs(expected) * 0.025);
        try std.testing.expectApproxEqAbs(expected, g_val, tol);
    }
}

test "G7 MoE E=512 top-10 + shared_expert (1-bucket, 3-bucket {q2,q3,q4}, gated/ungated) matches CPU fp32 ref" {
    // 1) Pure routing parity for E=512 top-10 and E=1024 top-12 (f32 and f16 logits).
    try verifyRouteTopK(.f32, 1, 512, 10, true, false);
    try verifyRouteTopK(.f16, 8, 512, 10, true, true);
    try verifyRouteTopK(.f32, 4, 1024, 12, true, true);

    // 2) Uniform q3 E=512 top-10 + ungated q4 shared_expert (reusing workspace.inter).
    var uniform_q3: [512]quant.WeightDType = @splat(.q3_affine);
    try verifyHighExpertWithShared(&uniform_q3, 1, 10, 128, 64, .q4_affine, 64, false, false, .f16, true, false);
    try verifyHighExpertWithShared(&uniform_q3, 4, 10, 128, 64, .q4_affine, 128, false, true, .bf16, true, false);

    // 3) 3-bucket {q2, q3, q4} G13 E=512 top-10 + sigmoid-gated bf16 shared_expert (Qwen3.8 profile).
    var hetero_512: [512]quant.WeightDType = undefined;
    for (&hetero_512, 0..) |*dt, e| {
        dt.* = if (e < 64) .q4_affine else if (e < 256) .q3_affine else .q2_affine;
    }
    try verifyHighExpertWithShared(&hetero_512, 1, 10, 128, 64, .bf16, 128, true, true, .f16, true, true);
    try verifyHighExpertWithShared(&hetero_512, 8, 10, 128, 64, .q4_affine, 128, true, true, .f32, false, true);
}

test "G15: wide MoeWorkspace (M=512, M=1024), 3-bucket hetero + shared_expert, rebuildSchemeBucketsCpu, and wide argmax/topK rows" {
    const gpa = std.testing.allocator;

    // 1) Wide CSR routing parity at M = 512 (E=64, top-8) and M = 1024 (E=32, top-4).
    try verifyRouteTopK(.f32, 512, 64, 8, true, false);
    try verifyRouteTopK(.f16, 1024, 32, 4, true, true);

    // 2) Full M = 512 heterogeneous 3-bucket {q2, q3, q4} MoE + gated bf16 shared_expert
    //    (verifies zero u16 slot overflow and full GPU vs CPU fp32 parity at M = 512).
    const m_wide: u32 = 512;
    const num_experts: u32 = 32;
    const top_k: u32 = 4;
    const hidden_size: u32 = 128;
    const intermediate_size: u32 = 64;
    var expert_dtypes: [num_experts]quant.WeightDType = undefined;
    var scheme_counts: [NUM_SCHEME_BUCKETS]u32 = @splat(0);
    for (&expert_dtypes, 0..) |*dt, e| {
        dt.* = if (e < 8) .q4_affine else if (e < 20) .q3_affine else .q2_affine;
        scheme_counts[@backingInt(dt.*)] += 1;
    }

    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try MoePipelines.init(device, library);
    defer pipes.close();

    var total_weight_bytes: usize = 0;
    for (expert_dtypes) |dt| {
        total_weight_bytes += amf.align64(@intCast(try singleExpertBytesForDType(dt, hidden_size, intermediate_size))) + 512;
    }
    const shared_eb = try singleExpertBytesForDType(.bf16, hidden_size, intermediate_size);
    total_weight_bytes += amf.align64(@intCast(shared_eb)) + 1024;

    const ws_spec: WorkspaceSpec = .{
        .max_tokens = m_wide,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .shared_intermediate_size = intermediate_size,
    };
    const ws_bytes = try MoeWorkspace.requiredBytes(ws_spec);
    const x_bytes = @as(usize, m_wide) * hidden_size * @sizeOf(f16);
    const logits_bytes = @as(usize, m_wide) * num_experts * @sizeOf(f32);
    const vocab_wide: u32 = 256;
    const vocab_logits_bytes = @as(usize, m_wide) * vocab_wide * @sizeOf(f16);
    const argmax_out_bytes = @as(usize, m_wide) * @sizeOf(u32);
    const topk_k: u32 = 4;
    const topk_out_bytes = @as(usize, m_wide) * topk_k * @sizeOf(TopKCandidate);

    const arena_cap = try page.alignUp(
        total_weight_bytes + 2 * ws_bytes + 3 * x_bytes + logits_bytes +
            vocab_logits_bytes + argmax_out_bytes + topk_out_bytes + 32768,
    ) + page.APPLE_PAGE_SIZE;
    var arena = try makeTestArena(arena_cap);
    defer arena.deinit();

    var ws_gpu = try MoeWorkspace.init(&arena, ws_spec);
    var ws_cpu = try MoeWorkspace.init(&arena, ws_spec);
    const table_raw = try arena.allocAligned(@as(usize, num_experts) * @sizeOf(ExpertWeights), 64);
    const table: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(ExpertWeights, table_raw));
    for (0..num_experts) |e| {
        table[e] = try fillSyntheticExpertForDType(&arena, expert_dtypes[e], hidden_size, intermediate_size, e + 1);
    }
    const shared_ew = try fillSyntheticExpertForDType(&arena, .bf16, hidden_size, intermediate_size, 999);
    const shared_gate_raw = try arena.allocAligned(@as(usize, hidden_size) * @sizeOf(f16), 64);
    const shared_gate_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, shared_gate_raw));
    for (shared_gate_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 11 + 3) % 15)) - 7)) * 0.03125);

    const schemes_raw = try arena.allocAligned(num_experts, 64);
    for (0..num_experts) |e| schemes_raw[e] = @backingInt(expert_dtypes[e]);

    const x_raw = try arena.allocAligned(x_bytes, 64);
    const out_gpu_raw = try arena.allocAligned(x_bytes, 64);
    const out_cpu_raw = try arena.allocAligned(x_bytes, 64);
    const logits_raw = try arena.allocAligned(logits_bytes, 64);
    const vlogits_raw = try arena.allocAligned(vocab_logits_bytes, 64);
    const argmax_raw = try arena.allocAligned(argmax_out_bytes, 64);
    const topk_raw = try arena.allocAligned(topk_out_bytes, 64);
    @memset(out_gpu_raw, 0);
    @memset(out_cpu_raw, 0);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
    const vlogits_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vlogits_raw));
    for (x_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 5) % 23)) - 11)) * 0.03125);
    for (logits_f32, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 29 + 13) % 41)) - 20)) * 0.125;
    for (vlogits_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 31 + 7) % 61)) - 30)) * 0.125);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const base_gpu: MoELayerArgs = .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .router = .{ .logits = .{ .binding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) }, .dtype = .f32 } },
        .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
        .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(table_raw.ptr) },
        .workspace = &ws_gpu,
        .workspace_buf = gpu_buf,
        .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_gpu_raw.ptr) },
        .m = m_wide,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .norm_topk_prob = true,
        .shared_expert = shared_ew,
        .shared_expert_dtype = .bf16,
        .shared_intermediate_size = intermediate_size,
        .shared_gate_weight = .{ .buffer = gpu_buf, .offset = arena.offsetOf(shared_gate_raw.ptr) },
        .shared_gate_dtype = .f16,
    };

    // Run Path 1: GPU bucketed hetero MoE + shared_expert + wide 512-row argmax & top-K logits.
    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeMoELayerHetero(&f, .{
            .base = base_gpu,
            .expert_schemes = .{ .buffer = gpu_buf, .offset = arena.offsetOf(schemes_raw.ptr) },
            .scheme_counts = scheme_counts,
        }, false);
        try pipes.encodeRouteTopK(&f, .f32, .{
            .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .workspace = &ws_cpu,
            .workspace_buf = gpu_buf,
            .m = m_wide,
            .num_experts = num_experts,
            .top_k = top_k,
            .norm_topk_prob = true,
        }, false);
        try pipes.encodeArgmaxRows(&f, .{
            .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(vlogits_raw.ptr) },
            .out_token_ids = .{ .buffer = gpu_buf, .offset = arena.offsetOf(argmax_raw.ptr) },
            .num_rows = m_wide,
            .vocab_size = vocab_wide,
            .logits_dtype = .f16,
        }, false);
        try pipes.encodeTopKLogitsRows(&f, .{
            .logits = .{ .buffer = gpu_buf, .offset = arena.offsetOf(vlogits_raw.ptr) },
            .out_candidates = .{ .buffer = gpu_buf, .offset = arena.offsetOf(topk_raw.ptr) },
            .num_rows = m_wide,
            .vocab_size = vocab_wide,
            .top_k = topk_k,
            .logits_dtype = .f16,
        }, false);
        const sig = try f.commit();
        try compute.wait(sig, 10_000);
    }

    // Run Path 2: CPU rebuildSchemeBucketsCpu at M = 512 (2048 assignments -> uses down_out staging!)
    // and verify identical buckets & identical MoE output against Path 1!
    ws_cpu.rebuildSchemeBucketsCpu(&expert_dtypes);
    for (0..NUM_SCHEME_BUCKETS) |s_idx| {
        const scheme: quant.WeightDType = @fromBackingInt(@intCast(s_idx));
        try std.testing.expectEqual(ws_gpu.bucketHeader(scheme).num_unique_experts, ws_cpu.bucketHeader(scheme).num_unique_experts);
        try std.testing.expectEqual(ws_gpu.bucketHeader(scheme).total_assignments, ws_cpu.bucketHeader(scheme).total_assignments);
    }

    var base_cpu = base_gpu;
    base_cpu.workspace = &ws_cpu;
    base_cpu.out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_cpu_raw.ptr) };
    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeGateUpDownHetero(&f, base_cpu, scheme_counts, true, false);
        try pipes.encodeReduceAndOptionalShared(&f, base_cpu, .q4_affine);
        const sig = try f.commit();
        try compute.wait(sig, 10_000);
    }

    const out_gpu_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_gpu_raw));
    const out_cpu_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_cpu_raw));
    for (out_gpu_f16, out_cpu_f16) |a, b| {
        const fa: f32 = @floatCast(a);
        const fb: f32 = @floatCast(b);
        try std.testing.expect(std.math.isFinite(fa));
        try std.testing.expectApproxEqAbs(fa, fb, 1e-3);
    }

    // Verify M = 512 argmaxRows & topKLogitsRows against CPU references.
    const argmax_gpu: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, argmax_raw));
    const argmax_ref = try gpa.alloc(u32, m_wide);
    defer gpa.free(argmax_ref);
    argmaxRowsRef(.f16, vlogits_raw, argmax_ref, m_wide, vocab_wide);
    try std.testing.expectEqualSlices(u32, argmax_ref, argmax_gpu);

    const topk_gpu: []const TopKCandidate = @alignCast(std.mem.bytesAsSlice(TopKCandidate, topk_raw));
    const topk_ref = try gpa.alloc(TopKCandidate, @as(usize, m_wide) * topk_k);
    defer gpa.free(topk_ref);
    topKLogitsRowsRef(.f16, vlogits_raw, topk_ref, m_wide, vocab_wide, topk_k);
    for (topk_ref, topk_gpu) |r, g| {
        try std.testing.expectEqual(r.token_id, g.token_id);
        try std.testing.expectApproxEqAbs(r.prob, g.prob, 1e-4);
    }
}

test "G20 GPU indirect MoE dispatch: static-grid guard, exact indirect args, runtime-empty buckets, and direct-vs-indirect parity" {
    // 1. M=1 static-grid guard truth table (ADR-0023 §3).
    try std.testing.expect(!shouldUseIndirectDispatch(1, 0));
    try std.testing.expect(!shouldUseIndirectDispatch(1, 1));
    try std.testing.expect(shouldUseIndirectDispatch(1, 2));
    try std.testing.expect(shouldUseIndirectDispatch(1, 5));
    try std.testing.expect(shouldUseIndirectDispatch(2, 1));
    try std.testing.expect(shouldUseIndirectDispatch(8, 1));
    try std.testing.expect(shouldUseIndirectDispatch(8, 3));

    try std.testing.expectEqual(@as(u32, 0), countActiveSchemeBuckets(.{ 0, 0, 0, 0, 0 }));
    try std.testing.expectEqual(@as(u32, 1), countActiveSchemeBuckets(.{ 0, 0, 16, 0, 0 }));
    try std.testing.expectEqual(@as(u32, 3), countActiveSchemeBuckets(.{ 2, 0, 4, 6, 0 }));
    try std.testing.expectEqual(@as(u32, 5), countActiveSchemeBuckets(.{ 1, 1, 1, 1, 1 }));

    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    const pipes = try MoePipelines.init(device, lib);
    defer pipes.close();

    // 2. Verify exact GPU-written IndirectDispatchArgs on homogeneous M=8 (correlated U < M*top_k)
    //    and direct vs indirect bit-for-bit output parity on q4_affine & f16.
    const m: u32 = 8;
    const num_experts: u32 = 16;
    const top_k: u32 = 4;
    const hidden_size: u32 = 192; // ceil(192/16) = 12
    const intermediate_size: u32 = 128; // ceil(128/8) = 16, ceil(128/16) = 8

    const ws_spec: WorkspaceSpec = .{
        .max_tokens = m,
        .num_experts = num_experts,
        .top_k = top_k,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
    };
    const ws_bytes = try MoeWorkspace.requiredBytes(ws_spec);
    const q4_bytes = try expertLayerBytes(.q4, num_experts, hidden_size, intermediate_size);
    const f16_bytes = try singleExpertBytesForDType(.f16, hidden_size, intermediate_size) * num_experts;
    const total_bytes = ws_bytes * 2 + @as(usize, @intCast(q4_bytes + f16_bytes)) + 256 * 1024;
    var arena = try makeTestArena(total_bytes);
    defer arena.deinit();

    var ws_ind = try MoeWorkspace.init(&arena, ws_spec);
    var ws_dir = try MoeWorkspace.init(&arena, ws_spec);

    const x_raw = try arena.allocAligned(m * hidden_size * @sizeOf(f16), 64);
    const logits_raw = try arena.allocAligned(m * num_experts * @sizeOf(f32), 64);
    const out_ind_raw = try arena.allocAligned(m * hidden_size * @sizeOf(f16), 64);
    const out_dir_raw = try arena.allocAligned(m * hidden_size * @sizeOf(f16), 64);
    const schemes_raw = try arena.allocAligned(num_experts, 64);
    const q4_table_raw = try arena.allocAligned(num_experts * @sizeOf(ExpertWeights), 64);
    const f16_table_raw = try arena.allocAligned(num_experts * @sizeOf(ExpertWeights), 64);
    const hetero_table_raw = try arena.allocAligned(num_experts * @sizeOf(ExpertWeights), 64);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    for (x_f16, 0..) |*v, i| {
        v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 5) % 19)) - 9)) * 0.0625);
    }

    // Skew logits heavily toward experts {0, 1, 2, 3, 4, 5} so:
    // - Homogeneous M=8 deduplicates to U = 6 < M * top_k = 32.
    // - Heterogeneous 5-bucket layout leaves `.bf16` and `.q2_affine` buckets runtime-EMPTY (0 active experts)!
    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
    for (0..m) |tok| {
        for (0..num_experts) |e| {
            logits_f32[tok * num_experts + e] = if (e < 6)
                5.0 + @as(f32, @floatFromInt(((tok * 3 + e * 5) % 7))) * 0.25
            else
                -8.0 - @as(f32, @floatFromInt(e)) * 0.1;
        }
    }

    // Populate q4_affine experts and f16 experts.
    const q4_table: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(ExpertWeights, q4_table_raw));
    const f16_table: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(ExpertWeights, f16_table_raw));
    const hetero_table: []ExpertWeights = @alignCast(std.mem.bytesAsSlice(ExpertWeights, hetero_table_raw));

    const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
    const gu_qw_bytes = try Q4.rowWeightBytes(hidden_size) * intermediate_size;
    const gu_sb_bytes = try Q4.groupsPerRow(hidden_size) * intermediate_size * @sizeOf(quant.ScaleBias);
    const dn_qw_bytes = try Q4.rowWeightBytes(intermediate_size) * hidden_size;
    const dn_sb_bytes = try Q4.groupsPerRow(intermediate_size) * hidden_size * @sizeOf(quant.ScaleBias);

    for (0..num_experts) |e| {
        const g_qw = try arena.allocAligned(gu_qw_bytes, 64);
        const g_sb = try arena.allocAligned(gu_sb_bytes, 64);
        const u_qw = try arena.allocAligned(gu_qw_bytes, 64);
        const u_sb = try arena.allocAligned(gu_sb_bytes, 64);
        const d_qw = try arena.allocAligned(dn_qw_bytes, 64);
        const d_sb = try arena.allocAligned(dn_sb_bytes, 64);
        for (g_qw, 0..) |*b, i| b.* = @truncate((e * 29 + i * 7 + 3) & 0xFF);
        for (u_qw, 0..) |*b, i| b.* = @truncate((e * 31 + i * 11 + 5) & 0xFF);
        for (d_qw, 0..) |*b, i| b.* = @truncate((e * 37 + i * 13 + 7) & 0xFF);
        for (std.mem.bytesAsSlice(quant.ScaleBias, g_sb)) |*sb| sb.* = .{ .scale = @floatCast(@as(f32, 0.015)), .bias = @floatCast(@as(f32, -0.10)) };
        for (std.mem.bytesAsSlice(quant.ScaleBias, u_sb)) |*sb| sb.* = .{ .scale = @floatCast(@as(f32, 0.015)), .bias = @floatCast(@as(f32, -0.10)) };
        for (std.mem.bytesAsSlice(quant.ScaleBias, d_sb)) |*sb| sb.* = .{ .scale = @floatCast(@as(f32, 0.015)), .bias = @floatCast(@as(f32, -0.10)) };
        q4_table[e] = .{
            .gate_qweight_offset = arena.offsetOf(g_qw.ptr),
            .gate_sb_offset = arena.offsetOf(g_sb.ptr),
            .up_qweight_offset = arena.offsetOf(u_qw.ptr),
            .up_sb_offset = arena.offsetOf(u_sb.ptr),
            .down_qweight_offset = arena.offsetOf(d_qw.ptr),
            .down_sb_offset = arena.offsetOf(d_sb.ptr),
        };

        const fg = try arena.allocAligned(intermediate_size * hidden_size * @sizeOf(f16), 64);
        const fu = try arena.allocAligned(intermediate_size * hidden_size * @sizeOf(f16), 64);
        const fd = try arena.allocAligned(hidden_size * intermediate_size * @sizeOf(f16), 64);
        for (@as([]f16, @alignCast(std.mem.bytesAsSlice(f16, fg))), 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((e * 17 + i) % 11)) - 5)) * 0.02);
        for (@as([]f16, @alignCast(std.mem.bytesAsSlice(f16, fu))), 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((e * 19 + i) % 11)) - 5)) * 0.02);
        for (@as([]f16, @alignCast(std.mem.bytesAsSlice(f16, fd))), 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((e * 23 + i) % 11)) - 5)) * 0.02);
        f16_table[e] = .{
            .gate_qweight_offset = arena.offsetOf(fg.ptr),
            .gate_sb_offset = 0,
            .up_qweight_offset = arena.offsetOf(fu.ptr),
            .up_sb_offset = 0,
            .down_qweight_offset = arena.offsetOf(fd.ptr),
            .down_sb_offset = 0,
        };
    }

    // Assign heterogeneous schemes: experts 0..2 = f16, 3..5 = q4_affine, 6..10 = bf16, 11..15 = q2_affine.
    // Because router only selects experts 0..5, `.bf16` and `.q2_affine` have 0 active experts at runtime
    // despite having non-zero static `scheme_counts`!
    var expert_dtypes: [num_experts]quant.WeightDType = undefined;
    var scheme_counts: [NUM_SCHEME_BUCKETS]u32 = @splat(0);
    for (0..num_experts) |e| {
        const dt: quant.WeightDType = if (e < 3) .f16 else if (e < 6) .q4_affine else if (e < 11) .bf16 else .q2_affine;
        expert_dtypes[e] = dt;
        schemes_raw[e] = @backingInt(dt);
        scheme_counts[@backingInt(dt)] += 1;
        hetero_table[e] = if (dt == .f16 or dt == .bf16) f16_table[e] else q4_table[e];
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const x_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) };
    const logits_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(logits_raw.ptr) };
    const out_ind_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_ind_raw.ptr) };
    const out_dir_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_dir_raw.ptr) };

    // Test A: Homogeneous q4_affine M=8 — indirect (default via encodeMoELayer) vs explicit direct.
    @memset(out_ind_raw, 0);
    @memset(out_dir_raw, 0);
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        // Indirect path via encodeMoELayer (M=8 > 1 -> indirect enabled).
        try pipes.encodeMoELayer(&frame, .q4, .{
            .x = x_bind,
            .router = .{ .logits = .{ .binding = logits_bind, .dtype = .f32 } },
            .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
            .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q4_table_raw.ptr) },
            .workspace = &ws_ind,
            .workspace_buf = gpu_buf,
            .out = out_ind_bind,
            .m = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
        }, false);
        // Direct path on ws_dir with explicit `use_indirect = false`.
        try pipes.encodeRouteTopK(&frame, .f32, .{
            .logits = logits_bind,
            .workspace = &ws_dir,
            .workspace_buf = gpu_buf,
            .m = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
        }, true);
        try pipes.encodeGateUpSwiglu(&frame, .q4, .{
            .x = x_bind,
            .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
            .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q4_table_raw.ptr) },
            .workspace = &ws_dir,
            .workspace_buf = gpu_buf,
            .m = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .use_indirect = false,
        }, true);
        try pipes.encodeDownProj(&frame, .q4, .{
            .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
            .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q4_table_raw.ptr) },
            .workspace = &ws_dir,
            .workspace_buf = gpu_buf,
            .m = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .use_indirect = false,
        }, true);
        try pipes.encodeReduce(&frame, .{
            .workspace = &ws_dir,
            .workspace_buf = gpu_buf,
            .out = out_dir_bind,
            .m = m,
            .hidden_size = hidden_size,
            .top_k = top_k,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    // Verify exact GPU-written IndirectDispatchArgs on homogeneous route_topk!
    const active_u = ws_ind.schedule_header.num_unique_experts;
    try std.testing.expectEqual(@as(u32, 6), active_u);
    try std.testing.expectEqual([3]u32{ 16, active_u, 1 }, ws_ind.scheduleIndirectGateUpQ().threadgroups_per_grid);
    try std.testing.expectEqual([3]u32{ 8, active_u, 1 }, ws_ind.scheduleIndirectGateUpF().threadgroups_per_grid);
    try std.testing.expectEqual([3]u32{ 12, active_u, 1 }, ws_ind.scheduleIndirectDown().threadgroups_per_grid);
    try std.testing.expectEqualSlices(u8, out_dir_raw, out_ind_raw);

    // Test B: Homogeneous f16 M=8 — indirect vs direct bit-for-bit parity.
    @memset(out_ind_raw, 0);
    @memset(out_dir_raw, 0);
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeMoELayerFloat(&frame, .f16, .{
            .x = x_bind,
            .router = .{ .logits = .{ .binding = logits_bind, .dtype = .f32 } },
            .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
            .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(f16_table_raw.ptr) },
            .workspace = &ws_ind,
            .workspace_buf = gpu_buf,
            .out = out_ind_bind,
            .m = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
        }, false);
        try pipes.encodeGateUpSwigluFloat(&frame, .f16, .{
            .x = x_bind,
            .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
            .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(f16_table_raw.ptr) },
            .workspace = &ws_dir,
            .workspace_buf = gpu_buf,
            .m = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .use_indirect = false,
        }, true);
        try pipes.encodeDownProjFloat(&frame, .f16, .{
            .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
            .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(f16_table_raw.ptr) },
            .workspace = &ws_dir,
            .workspace_buf = gpu_buf,
            .m = m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .use_indirect = false,
        }, true);
        try pipes.encodeReduce(&frame, .{
            .workspace = &ws_dir,
            .workspace_buf = gpu_buf,
            .out = out_dir_bind,
            .m = m,
            .hidden_size = hidden_size,
            .top_k = top_k,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }
    try std.testing.expectEqualSlices(u8, out_dir_raw, out_ind_raw);

    // Test C: Heterogeneous GSQ layer at BOTH M=1 (num_scheme_buckets=4 > 1 -> indirect ON!)
    //         and M=8 with runtime-empty `.bf16` and `.q2_affine` buckets (`active == 0`)!
    const test_ms = [_]u32{ 1, m };
    for (test_ms) |cur_m| {
        @memset(out_ind_raw, 0);
        @memset(out_dir_raw, 0);
        const base_ind: MoELayerArgs = .{
            .x = x_bind,
            .router = .{ .logits = .{ .binding = logits_bind, .dtype = .f32 } },
            .weights_base = .{ .buffer = gpu_buf, .offset = 0 },
            .expert_table = .{ .buffer = gpu_buf, .offset = arena.offsetOf(hetero_table_raw.ptr) },
            .workspace = &ws_ind,
            .workspace_buf = gpu_buf,
            .out = out_ind_bind,
            .m = cur_m,
            .num_experts = num_experts,
            .top_k = top_k,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
        };
        {
            var frame = try compute.beginFrame(5000);
            errdefer frame.abort();
            // Zero-sync GPU indirect hetero layer!
            try pipes.encodeMoELayerHetero(&frame, .{
                .base = base_ind,
                .expert_schemes = .{ .buffer = gpu_buf, .offset = arena.offsetOf(schemes_raw.ptr) },
                .scheme_counts = scheme_counts,
            }, false);
            // Reference routing for CPU bucket rebuild + host-exact direct dispatch.
            try pipes.encodeRouteTopK(&frame, .f32, .{
                .logits = logits_bind,
                .workspace = &ws_dir,
                .workspace_buf = gpu_buf,
                .m = cur_m,
                .num_experts = num_experts,
                .top_k = top_k,
                .hidden_size = hidden_size,
                .intermediate_size = intermediate_size,
            }, true);
            const sig = try frame.commit();
            try compute.wait(sig, 5000);
        }

        // Verify per-scheme indirect dispatch args match exact active counts (including 0-active buckets!).
        ws_dir.rebuildSchemeBucketsCpu(&expert_dtypes);
        for (0..NUM_SCHEME_BUCKETS) |s_idx| {
            const scheme: quant.WeightDType = @fromBackingInt(@intCast(s_idx));
            const exp_u = ws_ind.bucketHeader(scheme).num_unique_experts;
            try std.testing.expectEqual([3]u32{ 16, exp_u, 1 }, ws_ind.bucketIndirectGateUpQ(scheme).threadgroups_per_grid);
            try std.testing.expectEqual([3]u32{ 8, exp_u, 1 }, ws_ind.bucketIndirectGateUpF(scheme).threadgroups_per_grid);
            try std.testing.expectEqual([3]u32{ 12, exp_u, 1 }, ws_ind.bucketIndirectDown(scheme).threadgroups_per_grid);
            // CPU rebuildSchemeBucketsCpu must write identical indirect args.
            try std.testing.expectEqual(ws_ind.bucketIndirectGateUpQ(scheme).threadgroups_per_grid, ws_dir.bucketIndirectGateUpQ(scheme).threadgroups_per_grid);
            try std.testing.expectEqual(ws_ind.bucketIndirectGateUpF(scheme).threadgroups_per_grid, ws_dir.bucketIndirectGateUpF(scheme).threadgroups_per_grid);
            try std.testing.expectEqual(ws_ind.bucketIndirectDown(scheme).threadgroups_per_grid, ws_dir.bucketIndirectDown(scheme).threadgroups_per_grid);
        }
        // Confirm runtime-empty buckets really have 0 active experts!
        try std.testing.expectEqual(@as(u32, 0), ws_ind.bucketIndirectGateUpF(.bf16).threadgroups_per_grid[1]);
        try std.testing.expectEqual(@as(u32, 0), ws_ind.bucketIndirectGateUpQ(.q2_affine).threadgroups_per_grid[1]);

        var base_dir = base_ind;
        base_dir.workspace = &ws_dir;
        base_dir.out = out_dir_bind;
        {
            var frame = try compute.beginFrame(5000);
            errdefer frame.abort();
            try pipes.encodeGateUpDownHetero(&frame, base_dir, scheme_counts, true, false);
            try pipes.encodeReduce(&frame, .{
                .workspace = &ws_dir,
                .workspace_buf = gpu_buf,
                .out = out_dir_bind,
                .m = cur_m,
                .hidden_size = hidden_size,
                .top_k = top_k,
            }, true);
            const sig = try frame.commit();
            try compute.wait(sig, 5000);
        }

        const slice_bytes = @as(usize, cur_m) * hidden_size * @sizeOf(f16);
        try std.testing.expectEqualSlices(u8, out_dir_raw[0..slice_bytes], out_ind_raw[0..slice_bytes]);
    }
}
