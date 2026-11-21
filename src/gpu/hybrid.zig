//! M6 hybrid frontier model GPU operators (`R6`, `G8`, `G9`, `R5`, `G10`):
//! - **Gated DeltaNet (`R6`, `G8`)**: 16 KiB-aligned wired recurrent state arena
//!   `GdnStateArena` (committed `S_t` + `ADR-0019` loop-isolation scratch `S_tmp`),
//!   single-step decode, chunked prefill, `M = 1..64` tree verification with
//!   64-bit causal ancestor masks, fused RMSNorm + SiLU output gate, and GPU-side
//!   state commit (`ami_gdn_commit_state` / `ami_gdn_commit_tree`).
//! - **Qwen Sparse Attention (`G9`, `R5`)**: YaRN RoPE scaling, interleaved 3D
//!   MRoPE (`[11, 11, 10]` or general `[s0, s1, s2]`), hybrid `RGBA16Float`
//!   IOSurface `KvTile` rings (`512`-token full chunks and `128`-token circular
//!   sliding-window tiles), micro-block KV indexer with mandatory local recent
//!   block(s) and `ADR-0019` cross-loop index reuse, and sparse GQA decode /
//!   64-bit ancestor-masked tree verification.
//! - **4-Branch Gated Residual (`G10`)**: wired `[4, max_tokens, hidden]` stream
//!   (`f16` / `bf16` storage, `fp32` math) with fused init, read-gate, write-gate,
//!   and collapse kernels.
//!
//! Zero dynamic heap allocation on the hot path; every shared region is 16 KiB
//! aligned (`ADR-0002`, `ADR-0011`, `ADR-0015`, `ADR-0019`).

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const amf = @import("../format/amf.zig");
const quant = @import("../quant/spec.zig");
const attention = @import("attention.zig");

/// Maximum Gated DeltaNet key/value head dimension supported by `hybrid_ops.metal`.
pub const MAX_GDN_HEAD_DIM: u32 = 256;
/// Maximum QSA head dimension supported by `hybrid_ops.metal`.
pub const MAX_QSA_HEAD_DIM: u32 = 256;
/// Maximum speculative tree width (64-bit ancestor bitmask).
pub const MAX_TREE_NODES: u32 = attention.MAX_TREE_NODES;
/// Default full-attention IOSurface KvTile chunk length (tokens).
pub const QSA_FULL_CHUNK_TOKENS: u32 = attention.DEFAULT_CHUNK_TOKENS;
/// Circular sliding-window IOSurface KvTile length (tokens, R5).
pub const QSA_SWA_WINDOW_TOKENS: u32 = 128;
/// Default micro-block size in tokens (divides both 128 and 512).
pub const QSA_DEFAULT_BLOCK_TOKENS: u32 = 32;
/// Maximum selected micro-blocks per `(node, kv_head)` in `ami_qsa_index_blocks`.
pub const QSA_MAX_TOP_K_BLOCKS: u32 = 64;
/// Number of parallel residual branches in 4-Branch Gated Residual (`G10`).
pub const GR_NUM_BRANCHES: u32 = 4;

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
    InvalidRotaryDim,
    InvalidMropeSection,
    InvalidNodeCount,
    InvalidLayerIndex,
    UnsupportedTopKBlocks,
    MissingTreeInputs,
    MissingNormWeights,
    MissingSecondChunk,
};

// ===========================================================================
// Uniform Payloads (exact ABI match with `src/kernels/hybrid_ops.metal`)
// ===========================================================================

pub const GdnDecodeParams = extern struct {
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: f32,
    l2_eps: f32,
    rms_eps: f32,
    apply_l2_norm: u32,
    apply_out_norm: u32,
    apply_out_gate: u32,
    per_head_norm: u32,
    _pad0: u32 = 0,
};

pub const GdnPrefillParams = extern struct {
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: f32,
    l2_eps: f32,
    apply_l2_norm: u32,
    apply_out_gate: u32,
    chunk_size: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

pub const GdnTreeVerifyParams = extern struct {
    num_nodes: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: f32,
    l2_eps: f32,
    apply_l2_norm: u32,
    apply_out_gate: u32,
    scratch_leaf_node: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

pub const GdnOutNormGateParams = extern struct {
    num_tokens: u32,
    num_v_heads: u32,
    head_v_dim: u32,
    rms_eps: f32,
    apply_out_norm: u32,
    apply_out_gate: u32,
    per_head_norm: u32,
    _pad0: u32 = 0,
};

pub const GdnCommitStateParams = extern struct {
    total_f32_elems: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

pub const GdnCommitTreeParams = extern struct {
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    accepted_count: u32,
    use_node_buffer: u32,
    l2_eps: f32,
    apply_l2_norm: u32,
    inline_nodes: [MAX_TREE_NODES]u16 = @splat(0),
};

pub const QsaRopeKvParams = extern struct {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    pos_t: u32,
    pos_h: u32,
    pos_w: u32,
    mrope_s0: u32,
    mrope_s1: u32,
    mrope_s2: u32,
    chunk_tokens: u32,
    slot_in_chunk: u32,
    rms_eps: f32,
    rope_theta: f32,
    yarn_factor: f32,
    yarn_mscale: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    original_max_pos: u32,
    apply_qk_norm: u32,
};

pub const QsaTreeRopeParams = extern struct {
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    mrope_s0: u32,
    mrope_s1: u32,
    mrope_s2: u32,
    rms_eps: f32,
    rope_theta: f32,
    yarn_factor: f32,
    yarn_mscale: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    original_max_pos: u32,
    apply_qk_norm: u32,
};

pub const QsaKvTreeCommitParams = extern struct {
    num_kv_heads: u32,
    head_dim: u32,
    base_pos: u32,
    chunk_tokens: u32,
    ring_capacity: u32,
    accepted_count: u32,
    use_node_buffer: u32,
    has_second_chunk: u32,
    inline_nodes: [MAX_TREE_NODES]u16 = @splat(0),
};

pub const QsaBlockSummaryParams = extern struct {
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    window_size: u32,
    ring_capacity: u32,
    chunk_tokens: u32,
    block_tokens: u32,
    start_block: u32,
    num_blocks: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

pub const QsaIndexParams = extern struct {
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    window_size: u32,
    ring_capacity: u32,
    block_tokens: u32,
    top_k_blocks: u32,
    num_local_blocks: u32,
    scale: f32,
    _pad0: u32 = 0,
};

pub const QsaSparseAttnParams = extern struct {
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    window_size: u32,
    ring_capacity: u32,
    chunk_tokens: u32,
    block_tokens: u32,
    top_k_blocks: u32,
    has_tree: u32,
    scale: f32,
};

pub const GrParams = extern struct {
    m: u32,
    hidden_dim: u32,
    branch_token_stride: u32,
    mode: u32,
    w0: f32 = 0.0,
    w1: f32 = 0.0,
    w2: f32 = 0.0,
    w3: f32 = 0.0,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
    _pad3: u32 = 0,
};

pub const GrComputeGatesParams = extern struct {
    m: u32,
    hidden_dim: u32,
    branch_token_stride: u32,
    _pad0: u32 = 0,
};

pub const GdnComputeGatesParams = extern struct {
    m: u32,
    hidden_dim: u32,
    num_v_heads: u32,
    _pad0: u32 = 0,
};

pub const GdnConv1dParams = extern struct {
    m: u32,
    gdn_qk: u32,
    gdn_v: u32,
    conv_k: u32,
    is_tree: u32,
    write_state: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

pub const GdnConv1dCommitParams = extern struct {
    gdn_qk: u32,
    gdn_v: u32,
    conv_k: u32,
    accepted_count: u32,
    inline_nodes: [64]u16 = @splat(0),
};

pub const BlendF16Params = extern struct {
    count: u32,
    a: f32,
    b: f32,
    _pad0: u32 = 0,
};

comptime {
    if (@sizeOf(GdnDecodeParams) != 48) @compileError("GdnDecodeParams must be 48 bytes");
    if (@sizeOf(GdnPrefillParams) != 48) @compileError("GdnPrefillParams must be 48 bytes");
    if (@sizeOf(GdnTreeVerifyParams) != 48) @compileError("GdnTreeVerifyParams must be 48 bytes");
    if (@sizeOf(GdnOutNormGateParams) != 32) @compileError("GdnOutNormGateParams must be 32 bytes");
    if (@sizeOf(GdnCommitStateParams) != 16) @compileError("GdnCommitStateParams must be 16 bytes");
    if (@sizeOf(GdnCommitTreeParams) != 160) @compileError("GdnCommitTreeParams must be 160 bytes");
    if (@sizeOf(QsaRopeKvParams) != 80) @compileError("QsaRopeKvParams must be 80 bytes");
    if (@sizeOf(QsaTreeRopeParams) != 64) @compileError("QsaTreeRopeParams must be 64 bytes");
    if (@sizeOf(QsaKvTreeCommitParams) != 160) @compileError("QsaKvTreeCommitParams must be 160 bytes");
    if (@sizeOf(QsaBlockSummaryParams) != 48) @compileError("QsaBlockSummaryParams must be 48 bytes");
    if (@sizeOf(QsaIndexParams) != 48) @compileError("QsaIndexParams must be 48 bytes");
    if (@sizeOf(QsaSparseAttnParams) != 48) @compileError("QsaSparseAttnParams must be 48 bytes");
    if (@sizeOf(GrParams) != 48) @compileError("GrParams must be 48 bytes");
    if (@sizeOf(GrComputeGatesParams) != 16) @compileError("GrComputeGatesParams must be 16 bytes");
    if (@sizeOf(GdnComputeGatesParams) != 16) @compileError("GdnComputeGatesParams must be 16 bytes");
    if (@sizeOf(GdnConv1dParams) != 32) @compileError("GdnConv1dParams must be 32 bytes");
    if (@sizeOf(GdnConv1dCommitParams) != 144) @compileError("GdnConv1dCommitParams must be 144 bytes");
    if (@sizeOf(BlendF16Params) != 16) @compileError("BlendF16Params must be 16 bytes");
    if (@sizeOf(GdnCommitTreeParams) > metal.MAX_UNIFORM_BYTES) @compileError("GdnCommitTreeParams exceeds uniform stride");
    if (@sizeOf(QsaKvTreeCommitParams) > metal.MAX_UNIFORM_BYTES) @compileError("QsaKvTreeCommitParams exceeds uniform stride");
    if (@sizeOf(GdnConv1dCommitParams) > metal.MAX_UNIFORM_BYTES) @compileError("GdnConv1dCommitParams exceeds uniform stride");
}

// ===========================================================================
// Section A: R6 + G8 — Gated DeltaNet State Arena & Arguments
// ===========================================================================

pub const GdnStateSpec = struct {
    num_gdn_layers: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
};

/// Execution mode for GDN recurrent state updates (`ADR-0019`).
pub const GdnStateMode = enum {
    /// Standard single-pass (`max_loops = 1`): reads `S_t`, writes updated `S_t` in place.
    commit_in_place,
    /// `LoopSchedule` intermediate pass (`r < R - 1`): reads committed `S_t` and
    /// writes updated state into preallocated `S_tmp` without mutating `S_t`.
    loop_isolated_scratch,
};

/// Fixed-size wired recurrent state arena (`R6`, `ADR-0019`):
/// allocates committed `S_t` `[num_gdn_layers, num_v_heads, head_k_dim, head_v_dim]`
/// (`f32`) PLUS preallocated per-layer loop-isolation scratch `S_tmp` inside one
/// 16 KiB-aligned page block carved from `WiredArena`. Zero hot-path heap allocs.
pub const GdnStateArena = struct {
    spec: GdnStateSpec,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    base_offset: u64,
    layer_state_elems: usize,
    layer_stride_bytes: usize,
    scratch_base_offset: u64,

    pub fn validateSpec(spec: GdnStateSpec) ArenaError!void {
        if (spec.num_gdn_layers == 0 or spec.num_v_heads == 0 or
            spec.head_k_dim == 0 or spec.head_v_dim == 0)
        {
            return error.ZeroDimension;
        }
        if (spec.head_k_dim > MAX_GDN_HEAD_DIM or spec.head_v_dim > MAX_GDN_HEAD_DIM) {
            return error.InvalidConfig;
        }
    }

    pub fn layerElems(spec: GdnStateSpec) ArenaError!usize {
        try validateSpec(spec);
        const hv = std.math.mul(usize, spec.num_v_heads, spec.head_k_dim) catch return error.Overflow;
        return std.math.mul(usize, hv, spec.head_v_dim) catch return error.Overflow;
    }

    /// Exact page-aligned (`16 KiB`) arena bytes required for both committed `S_t`
    /// and loop-isolation scratch `S_tmp` across all `num_gdn_layers`.
    pub fn requiredBytes(spec: GdnStateSpec) ArenaError!usize {
        const elems = try layerElems(spec);
        const raw_layer_bytes = std.math.mul(usize, elems, @sizeOf(f32)) catch return error.Overflow;
        const stride = amf.align64(raw_layer_bytes);
        const half_bytes = std.math.mul(usize, stride, spec.num_gdn_layers) catch return error.Overflow;
        const total = std.math.mul(usize, half_bytes, 2) catch return error.Overflow;
        return page.alignUp(total) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: GdnStateSpec) ArenaError!GdnStateArena {
        const total = try requiredBytes(spec);
        const elems = try layerElems(spec);
        const stride = amf.align64(elems * @sizeOf(f32));
        const half_bytes = stride * spec.num_gdn_layers;

        const backing = try arena.allocPages(total);
        @memset(backing, 0);
        const base_off: u64 = @intCast(arena.offsetOf(backing.ptr));
        return .{
            .spec = spec,
            .backing = backing,
            .base_offset = base_off,
            .layer_state_elems = elems,
            .layer_stride_bytes = stride,
            .scratch_base_offset = base_off + @as(u64, @intCast(half_bytes)),
        };
    }

    pub fn bytes(self: *const GdnStateArena) []align(page.APPLE_PAGE_SIZE) u8 {
        return self.backing;
    }

    pub fn resetAll(self: *GdnStateArena) void {
        @memset(self.backing, 0);
    }

    pub fn resetLayer(self: *GdnStateArena, layer_idx: u32) ArenaError!void {
        const comm = try self.committedLayerSlice(layer_idx);
        const scr = try self.scratchLayerSlice(layer_idx);
        @memset(comm, 0.0);
        @memset(scr, 0.0);
    }

    pub fn committedOffset(self: *const GdnStateArena, layer_idx: u32) ArenaError!u64 {
        if (layer_idx >= self.spec.num_gdn_layers) return error.InvalidLayerIndex;
        return self.base_offset + @as(u64, layer_idx) * @as(u64, @intCast(self.layer_stride_bytes));
    }

    pub fn scratchOffset(self: *const GdnStateArena, layer_idx: u32) ArenaError!u64 {
        if (layer_idx >= self.spec.num_gdn_layers) return error.InvalidLayerIndex;
        return self.scratch_base_offset + @as(u64, layer_idx) * @as(u64, @intCast(self.layer_stride_bytes));
    }

    pub fn committedBinding(self: *const GdnStateArena, buf: metal.Buffer, layer_idx: u32) ArenaError!metal.BufferBinding {
        return .{ .buffer = buf, .offset = try self.committedOffset(layer_idx) };
    }

    pub fn scratchBinding(self: *const GdnStateArena, buf: metal.Buffer, layer_idx: u32) ArenaError!metal.BufferBinding {
        return .{ .buffer = buf, .offset = try self.scratchOffset(layer_idx) };
    }

    /// Returns `(state_in, state_out)` bindings for `mode`:
    /// - `.commit_in_place`: `(S_t, S_t)`
    /// - `.loop_isolated_scratch`: `(S_t, S_tmp)` (committed `S_t` stays untouched)
    pub fn stateBindings(
        self: *const GdnStateArena,
        buf: metal.Buffer,
        layer_idx: u32,
        mode: GdnStateMode,
    ) ArenaError!struct { in: metal.BufferBinding, out: metal.BufferBinding } {
        const comm = try self.committedBinding(buf, layer_idx);
        return switch (mode) {
            .commit_in_place => .{ .in = comm, .out = comm },
            .loop_isolated_scratch => .{ .in = comm, .out = try self.scratchBinding(buf, layer_idx) },
        };
    }

    pub fn committedLayerSlice(self: *const GdnStateArena, layer_idx: u32) ArenaError![]f32 {
        if (layer_idx >= self.spec.num_gdn_layers) return error.InvalidLayerIndex;
        const start = @as(usize, layer_idx) * self.layer_stride_bytes;
        const len_bytes = self.layer_state_elems * @sizeOf(f32);
        return @alignCast(std.mem.bytesAsSlice(f32, self.backing[start .. start + len_bytes]));
    }

    pub fn scratchLayerSlice(self: *const GdnStateArena, layer_idx: u32) ArenaError![]f32 {
        if (layer_idx >= self.spec.num_gdn_layers) return error.InvalidLayerIndex;
        const half_bytes = self.layer_stride_bytes * self.spec.num_gdn_layers;
        const start = half_bytes + @as(usize, layer_idx) * self.layer_stride_bytes;
        const len_bytes = self.layer_state_elems * @sizeOf(f32);
        return @alignCast(std.mem.bytesAsSlice(f32, self.backing[start .. start + len_bytes]));
    }
};

pub const GdnDecodeArgs = struct {
    /// `[num_k_heads, head_k_dim]` f16 query vector.
    q: metal.BufferBinding,
    /// `[num_k_heads, head_k_dim]` f16 key vector.
    k: metal.BufferBinding,
    /// `[num_v_heads, head_v_dim]` f16 value vector.
    v: metal.BufferBinding,
    /// `[num_v_heads]` f32 log-decay `g` (`exp(g) in (0, 1]`).
    g: metal.BufferBinding,
    /// `[num_v_heads]` f32 pre-sigmoid delta gate `beta`.
    beta: metal.BufferBinding,
    /// Optional `[num_v_heads, head_v_dim]` f16 SiLU output gate `z`.
    z: ?metal.BufferBinding = null,
    /// Optional `[head_v_dim]` (or `[num_v_heads, head_v_dim]` when `per_head_norm`) RMSNorm weights.
    norm_w: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .f16,
    per_head_norm: bool = false,
    /// Input recurrent state `S_{t-1}` `[num_v_heads, head_k_dim, head_v_dim]` f32.
    state_in: metal.BufferBinding,
    /// Output recurrent state (`S_t` for `.commit_in_place` or `S_tmp` for `.loop_isolated_scratch`).
    state_out: metal.BufferBinding,
    /// `[num_v_heads, head_v_dim]` f16 output.
    out: metal.BufferBinding,
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    /// Defaults to `1 / sqrt(head_k_dim)` when null.
    q_scale: ?f32 = null,
    l2_eps: f32 = 1e-6,
    rms_eps: f32 = 1e-6,
    apply_l2_norm: bool = true,
};

pub const GdnPrefillArgs = struct {
    /// `[seq_len, num_k_heads, head_k_dim]` f16.
    q: metal.BufferBinding,
    /// `[seq_len, num_k_heads, head_k_dim]` f16.
    k: metal.BufferBinding,
    /// `[seq_len, num_v_heads, head_v_dim]` f16.
    v: metal.BufferBinding,
    /// `[seq_len, num_v_heads]` f32 log-decay.
    g: metal.BufferBinding,
    /// `[seq_len, num_v_heads]` f32 pre-sigmoid delta gate.
    beta: metal.BufferBinding,
    /// Optional `[seq_len, num_v_heads, head_v_dim]` f16 SiLU output gate `z`.
    z: ?metal.BufferBinding = null,
    /// Optional RMSNorm weights (`f16` or `bf16`).
    norm_w: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .f16,
    per_head_norm: bool = false,
    state_in: metal.BufferBinding,
    state_out: metal.BufferBinding,
    /// `[seq_len, num_v_heads, head_v_dim]` f16.
    out: metal.BufferBinding,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: ?f32 = null,
    l2_eps: f32 = 1e-6,
    rms_eps: f32 = 1e-6,
    apply_l2_norm: bool = true,
    chunk_size: u32 = 64,
};

pub const GdnTreeVerifyArgs = struct {
    /// `[num_nodes, num_k_heads, head_k_dim]` f16.
    q: metal.BufferBinding,
    /// `[num_nodes, num_k_heads, head_k_dim]` f16.
    k: metal.BufferBinding,
    /// `[num_nodes, num_v_heads, head_v_dim]` f16.
    v: metal.BufferBinding,
    /// `[num_nodes, num_v_heads]` f32 log-decay.
    g: metal.BufferBinding,
    /// `[num_nodes, num_v_heads]` f32 pre-sigmoid delta gate.
    beta: metal.BufferBinding,
    /// Optional `[num_nodes, num_v_heads, head_v_dim]` f16 SiLU output gate.
    z: ?metal.BufferBinding = null,
    /// Optional RMSNorm weights (`f16` or `bf16`).
    norm_w: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .f16,
    per_head_norm: bool = false,
    /// `[num_nodes]u64` causal ancestor bitmasks.
    ancestor_masks: metal.BufferBinding,
    /// Committed initial state `S_{t-1}` (NEVER mutated by tree verify).
    state_in: metal.BufferBinding,
    /// Optional scratch state buffer (`S_tmp`) written only when `scratch_leaf_node != null`.
    state_out: ?metal.BufferBinding = null,
    /// Optional tree node index whose resulting state is written to `state_out`.
    scratch_leaf_node: ?u32 = null,
    /// `[num_nodes, num_v_heads, head_v_dim]` f16.
    out: metal.BufferBinding,
    num_nodes: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: ?f32 = null,
    l2_eps: f32 = 1e-6,
    rms_eps: f32 = 1e-6,
    apply_l2_norm: bool = true,
};

pub const GdnCommitTreeArgs = struct {
    k_tree: metal.BufferBinding,
    v_tree: metal.BufferBinding,
    g_tree: metal.BufferBinding,
    beta_tree: metal.BufferBinding,
    accepted_nodes: attention.AcceptedNodes,
    accepted_count: u32,
    state_in: metal.BufferBinding,
    state_out: metal.BufferBinding,
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    l2_eps: f32 = 1e-6,
    apply_l2_norm: bool = true,
};

// ===========================================================================
// Section B: G9 + R5 — QSA YaRN / Interleaved MRoPE / Hybrid KvTile Ring / Indexer
// ===========================================================================

/// YaRN RoPE scaling configuration (Peng et al. 2023 / Qwen3.8).
pub const YarnConfig = struct {
    factor: f32 = 1.0,
    /// Explicit amplitude scaling; defaults to `defaultMscale(factor)` when null.
    mscale: ?f32 = null,
    beta_fast: f32 = 32.0,
    beta_slow: f32 = 1.0,
    original_max_pos: u32 = 32768,

    pub fn defaultMscale(factor: f32) f32 {
        if (factor <= 1.0) return 1.0;
        return 0.1 * @log(factor) + 1.0;
    }

    pub fn effectiveMscale(self: YarnConfig) f32 {
        return self.mscale orelse defaultMscale(self.factor);
    }
};

/// Interleaved 3D MRoPE section `[s0, s1, s2]` summing to `rotary_dim / 2`.
pub const MropeSection = struct {
    section: [3]u32,

    /// Qwen3-VL / Qwen3.8 default `[11, 11, 10]` across `rotary_dim = 64` (`half = 32`).
    pub const QWEN38_DEFAULT: MropeSection = .{ .section = .{ 11, 11, 10 } };

    pub fn text1d(rotary_dim: u32) MropeSection {
        return .{ .section = .{ rotary_dim / 2, 0, 0 } };
    }

    pub fn validate(self: MropeSection, rotary_dim: u32) EncodeError!void {
        if (rotary_dim == 0 or (rotary_dim & 1) != 0) return error.InvalidRotaryDim;
        const sum: u64 = @as(u64, self.section[0]) + self.section[1] + self.section[2];
        if (sum != rotary_dim / 2) return error.InvalidMropeSection;
    }
};

/// 3D `(t, h, w)` token position for interleaved MRoPE (matches `uint[3]` in GPU buffers).
pub const Pos3D = extern struct {
    t: u32,
    h: u32,
    w: u32,

    pub fn text(pos: u32) Pos3D {
        return .{ .t = pos, .h = pos, .w = pos };
    }
};

/// Hybrid IOSurface `RGBA16Float` `KvTile` ring operating regime (`R5`).
pub const HybridKvRingMode = enum {
    /// Linear growing `512`-token full-attention chunks.
    full_chunk_512,
    /// Circular `128`-token sliding-window ring tiles.
    sliding_window_128,
};

pub const HybridKvRingSpec = struct {
    mode: HybridKvRingMode,
    num_kv_heads: u32,
    head_dim: u32,
    /// Number of chunk tiles in the ring:
    /// - `.full_chunk_512`: `ceil(max_seq_len / 512)` tiles (`ring_capacity = 0`, `window_size = 0`).
    /// - `.sliding_window_128`: number of `128`-token tiles (`1` tile -> `128`-token circular SWA).
    num_tiles: u32,
    /// Optional explicit sliding window override (defaults to `0` for full, `num_tiles * 128` for SWA).
    window_size_override: ?u32 = null,
};

/// Resident IOSurface `RGBA16Float` `KvTile` ring supporting both `512`-token
/// full-attention chunks and `128`-token circular sliding-window tiles (`R5`).
pub const HybridKvRing = struct {
    mode: HybridKvRingMode,
    num_kv_heads: u32,
    head_dim: u32,
    chunk_tokens: u32,
    window_size: u32,
    ring_capacity: u32,
    k_tiles: []metal.KvTile,
    v_tiles: []metal.KvTile,
    tile_table_off: u64,
    tile_table: []attention.KvTilePair,

    pub const InitError = ArenaError || metal.Error || std.mem.Allocator.Error;

    pub fn tableRequiredBytes(num_tiles: u32) ArenaError!usize {
        if (num_tiles == 0) return error.ZeroDimension;
        const raw = std.math.mul(usize, num_tiles, @sizeOf(attention.KvTilePair)) catch return error.Overflow;
        return amf.align64(raw);
    }

    pub fn init(
        allocator: std.mem.Allocator,
        device: metal.Device,
        arena: *arena_mod.WiredArena,
        spec: HybridKvRingSpec,
    ) InitError!HybridKvRing {
        if (spec.num_kv_heads == 0 or spec.head_dim == 0 or spec.num_tiles == 0) return error.ZeroDimension;
        if (spec.head_dim > MAX_QSA_HEAD_DIM or (spec.head_dim % 4) != 0) return error.InvalidConfig;

        const chunk_tokens: u32 = switch (spec.mode) {
            .full_chunk_512 => QSA_FULL_CHUNK_TOKENS,
            .sliding_window_128 => QSA_SWA_WINDOW_TOKENS,
        };
        const ring_capacity: u32 = switch (spec.mode) {
            .full_chunk_512 => 0,
            .sliding_window_128 => std.math.mul(u32, spec.num_tiles, chunk_tokens) catch return error.Overflow,
        };
        const window_size: u32 = switch (spec.mode) {
            .full_chunk_512 => spec.window_size_override orelse 0,
            .sliding_window_128 => blk: {
                const w = spec.window_size_override orelse ring_capacity;
                if (w == 0 or w > ring_capacity) return error.InvalidConfig;
                break :blk w;
            },
        };

        const rows = std.math.mul(u32, spec.num_kv_heads, chunk_tokens) catch return error.Overflow;
        const table_bytes = try tableRequiredBytes(spec.num_tiles);
        const table_raw = try arena.allocAligned(table_bytes, 64);
        const tile_table: []attention.KvTilePair = @alignCast(std.mem.bytesAsSlice(
            attention.KvTilePair,
            table_raw[0 .. @as(usize, spec.num_tiles) * @sizeOf(attention.KvTilePair)],
        ));

        const k_tiles = try allocator.alloc(metal.KvTile, spec.num_tiles);
        errdefer allocator.free(k_tiles);
        const v_tiles = try allocator.alloc(metal.KvTile, spec.num_tiles);
        errdefer allocator.free(v_tiles);

        var created: usize = 0;
        errdefer {
            for (0..created) |i| {
                v_tiles[i].close();
                k_tiles[i].close();
            }
        }
        for (0..spec.num_tiles) |i| {
            k_tiles[i] = try metal.KvTile.init(device, spec.head_dim, rows, .{});
            errdefer k_tiles[i].close();
            v_tiles[i] = try metal.KvTile.init(device, spec.head_dim, rows, .{});
            tile_table[i] = attention.KvTilePair.fromTiles(k_tiles[i], v_tiles[i], .rgba16f);
            created += 1;
        }

        return .{
            .mode = spec.mode,
            .num_kv_heads = spec.num_kv_heads,
            .head_dim = spec.head_dim,
            .chunk_tokens = chunk_tokens,
            .window_size = window_size,
            .ring_capacity = ring_capacity,
            .k_tiles = k_tiles,
            .v_tiles = v_tiles,
            .tile_table_off = @intCast(arena.offsetOf(table_raw.ptr)),
            .tile_table = tile_table,
        };
    }

    pub fn deinit(self: *HybridKvRing, allocator: std.mem.Allocator) void {
        for (self.k_tiles) |t| t.close();
        for (self.v_tiles) |t| t.close();
        allocator.free(self.k_tiles);
        allocator.free(self.v_tiles);
    }

    pub fn makeResident(self: *const HybridKvRing, compute: metal.Compute) metal.Error!void {
        try compute.makeTilesResident(self.k_tiles);
        try compute.makeTilesResident(self.v_tiles);
    }

    pub fn removeResident(self: *const HybridKvRing, compute: metal.Compute) metal.Error!void {
        try compute.removeTilesResident(self.v_tiles);
        try compute.removeTilesResident(self.k_tiles);
    }

    pub fn tileTableBinding(self: *const HybridKvRing, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.tile_table_off };
    }

    pub fn slotForPos(self: *const HybridKvRing, pos: u32) EncodeError!struct { tile_idx: u32, slot_in_chunk: u32 } {
        const phys = if (self.ring_capacity > 0) pos % self.ring_capacity else pos;
        const tile_idx = phys / self.chunk_tokens;
        if (tile_idx >= self.k_tiles.len) return error.InvalidConfig;
        return .{
            .tile_idx = tile_idx,
            .slot_in_chunk = phys % self.chunk_tokens,
        };
    }

    pub fn commitTilesForSpan(
        self: *const HybridKvRing,
        base_pos: u32,
        accepted_count: u32,
    ) EncodeError!struct {
        k_tile0: metal.KvTile,
        v_tile0: metal.KvTile,
        k_tile1: ?metal.KvTile,
        v_tile1: ?metal.KvTile,
    } {
        if (accepted_count == 0 or accepted_count > MAX_TREE_NODES) return error.InvalidNodeCount;
        const start_slot = try self.slotForPos(base_pos);
        const end_slot = try self.slotForPos(base_pos + accepted_count - 1);
        const crosses = (end_slot.tile_idx != start_slot.tile_idx);
        return .{
            .k_tile0 = self.k_tiles[start_slot.tile_idx],
            .v_tile0 = self.v_tiles[start_slot.tile_idx],
            .k_tile1 = if (crosses) self.k_tiles[end_slot.tile_idx] else null,
            .v_tile1 = if (crosses) self.v_tiles[end_slot.tile_idx] else null,
        };
    }
};

pub const QsaIndexBufferSpec = struct {
    max_nodes: u32,
    num_kv_heads: u32,
    top_k_blocks: u32,
};

/// Preallocated wired buffer storing QSA micro-block indexer outputs across
/// `ADR-0019` recurrent loop passes (`r = 0` populates, `r >= 1` reuses).
pub const QsaIndexBuffer = struct {
    spec: QsaIndexBufferSpec,
    selected_blocks_off: u64,
    selected_counts_off: u64,
    selected_blocks: []u32,
    selected_counts: []u32,

    pub fn validateSpec(spec: QsaIndexBufferSpec) ArenaError!void {
        if (spec.max_nodes == 0 or spec.num_kv_heads == 0 or spec.top_k_blocks == 0) {
            return error.ZeroDimension;
        }
        if (spec.max_nodes > MAX_TREE_NODES or spec.top_k_blocks > QSA_MAX_TOP_K_BLOCKS) {
            return error.InvalidConfig;
        }
    }

    pub fn requiredBytes(spec: QsaIndexBufferSpec) ArenaError!usize {
        try validateSpec(spec);
        const nh = std.math.mul(usize, spec.max_nodes, spec.num_kv_heads) catch return error.Overflow;
        const sb_elems = std.math.mul(usize, nh, spec.top_k_blocks) catch return error.Overflow;
        const sb_bytes = amf.align64(std.math.mul(usize, sb_elems, @sizeOf(u32)) catch return error.Overflow);
        const sc_bytes = amf.align64(std.math.mul(usize, nh, @sizeOf(u32)) catch return error.Overflow);
        return std.math.add(usize, sb_bytes, sc_bytes) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: QsaIndexBufferSpec) ArenaError!QsaIndexBuffer {
        try validateSpec(spec);
        const nh: usize = @as(usize, spec.max_nodes) * spec.num_kv_heads;
        const sb_bytes = nh * spec.top_k_blocks * @sizeOf(u32);
        const sc_bytes = nh * @sizeOf(u32);
        const sb_raw = try arena.allocAligned(sb_bytes, 64);
        const sc_raw = try arena.allocAligned(sc_bytes, 64);
        @memset(sb_raw, 0xFF);
        @memset(sc_raw, 0);
        return .{
            .spec = spec,
            .selected_blocks_off = @intCast(arena.offsetOf(sb_raw.ptr)),
            .selected_counts_off = @intCast(arena.offsetOf(sc_raw.ptr)),
            .selected_blocks = @alignCast(std.mem.bytesAsSlice(u32, sb_raw)),
            .selected_counts = @alignCast(std.mem.bytesAsSlice(u32, sc_raw)),
        };
    }

    pub fn blocksBinding(self: *const QsaIndexBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.selected_blocks_off };
    }

    pub fn countsBinding(self: *const QsaIndexBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.selected_counts_off };
    }
};

pub const QsaRopeKvWriteArgs = struct {
    q: metal.BufferBinding,
    k: metal.BufferBinding,
    v: metal.BufferBinding,
    q_norm: ?metal.BufferBinding = null,
    k_norm: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .f16,
    k_tile: metal.KvTile,
    v_tile: metal.KvTile,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    /// Rotary dimension (`<= head_dim`, even). Defaults to `head_dim` when 0.
    rotary_dim: u32 = 0,
    pos_3d: Pos3D,
    mrope: MropeSection,
    yarn: YarnConfig = .{},
    chunk_tokens: u32 = QSA_FULL_CHUNK_TOKENS,
    slot_in_chunk: u32,
    rms_eps: f32 = 1e-6,
    rope_theta: f32 = 1_000_000.0,
};

pub const QsaTreeRopeArgs = struct {
    q: metal.BufferBinding,
    k: metal.BufferBinding,
    v: metal.BufferBinding,
    q_norm: ?metal.BufferBinding = null,
    k_norm: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .f16,
    /// `[num_nodes]Pos3D` (`3 * num_nodes` u32s).
    positions_3d: metal.BufferBinding,
    k_tree_out: metal.BufferBinding,
    v_tree_out: metal.BufferBinding,
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32 = 0,
    mrope: MropeSection,
    yarn: YarnConfig = .{},
    rms_eps: f32 = 1e-6,
    rope_theta: f32 = 1_000_000.0,
};

pub const QsaKvTreeCommitArgs = struct {
    k_tree: metal.BufferBinding,
    v_tree: metal.BufferBinding,
    accepted_nodes: attention.AcceptedNodes,
    accepted_count: u32,
    k_tile0: metal.KvTile,
    v_tile0: metal.KvTile,
    k_tile1: ?metal.KvTile = null,
    v_tile1: ?metal.KvTile = null,
    num_kv_heads: u32,
    head_dim: u32,
    base_pos: u32,
    chunk_tokens: u32 = QSA_FULL_CHUNK_TOKENS,
    /// `0` for linear full-chunk tiles; `> 0` for circular SWA ring capacity (e.g., `128`).
    ring_capacity: u32 = 0,
};

pub const QsaBlockSummaryArgs = struct {
    kv_tiles: metal.BufferBinding,
    block_keys: metal.BufferBinding,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    window_size: u32 = 0,
    ring_capacity: u32 = 0,
    chunk_tokens: u32 = QSA_FULL_CHUNK_TOKENS,
    block_tokens: u32 = QSA_DEFAULT_BLOCK_TOKENS,
    start_block: u32,
    num_blocks: u32,
};

pub const QsaIndexBlocksArgs = struct {
    q: metal.BufferBinding,
    block_keys: metal.BufferBinding,
    index_buf: *const QsaIndexBuffer,
    index_metal_buf: metal.Buffer,
    num_nodes: u32 = 1,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    window_size: u32 = 0,
    ring_capacity: u32 = 0,
    block_tokens: u32 = QSA_DEFAULT_BLOCK_TOKENS,
    top_k_blocks: u32,
    /// Number of mandatory trailing local recent blocks (`>= 1`).
    num_local_blocks: u32 = 1,
    scale: ?f32 = null,
};

pub const QsaSparseAttnArgs = struct {
    q: metal.BufferBinding,
    /// Resident `KvTilePair` table (may be null only when `kv_len == 0`).
    kv_tiles: ?metal.BufferBinding = null,
    index_buf: *const QsaIndexBuffer,
    index_metal_buf: metal.Buffer,
    /// Optional speculative tree KV + 64-bit ancestor masks (all 3 required when `has_tree`).
    k_tree: ?metal.BufferBinding = null,
    v_tree: ?metal.BufferBinding = null,
    ancestor_masks: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    num_nodes: u32 = 1,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    window_size: u32 = 0,
    ring_capacity: u32 = 0,
    chunk_tokens: u32 = QSA_FULL_CHUNK_TOKENS,
    block_tokens: u32 = QSA_DEFAULT_BLOCK_TOKENS,
    top_k_blocks: u32,
    scale: ?f32 = null,
};

/// Combined QSA indexer + sparse attention dispatch supporting `ADR-0019`
/// cross-loop index reuse (`reuse_block_indices = false` on `r = 0`, `true` on `r >= 1`).
pub const QsaAttentionArgs = struct {
    q: metal.BufferBinding,
    kv_tiles: ?metal.BufferBinding = null,
    block_keys: metal.BufferBinding,
    index_buf: *const QsaIndexBuffer,
    index_metal_buf: metal.Buffer,
    k_tree: ?metal.BufferBinding = null,
    v_tree: ?metal.BufferBinding = null,
    ancestor_masks: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    num_nodes: u32 = 1,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    window_size: u32 = 0,
    ring_capacity: u32 = 0,
    chunk_tokens: u32 = QSA_FULL_CHUNK_TOKENS,
    block_tokens: u32 = QSA_DEFAULT_BLOCK_TOKENS,
    top_k_blocks: u32,
    num_local_blocks: u32 = 1,
    scale: ?f32 = null,
    /// When `true` (`r >= 1` in `LoopSchedule`, ADR-0019), skips `ami_qsa_index_blocks`
    /// and reuses the selected block indices already stored in `index_buf`.
    reuse_block_indices: bool = false,
};

// ===========================================================================
// Section C: G10 — 4-Branch Gated Residual Stream Arena & Arguments
// ===========================================================================

pub const GrStreamSpec = struct {
    max_tokens: u32,
    hidden_dim: u32,
    dtype: quant.FloatDType = .f16,
};

/// Wired `[4, max_tokens, hidden_dim]` 4-branch residual stream (`G10`),
/// carved as 16 KiB-aligned pages from `WiredArena` with zero hot-path allocs.
pub const GrStreamArena = struct {
    spec: GrStreamSpec,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    base_offset: u64,

    pub fn requiredBytes(spec: GrStreamSpec) ArenaError!usize {
        if (spec.max_tokens == 0 or spec.hidden_dim == 0) return error.ZeroDimension;
        const per_branch = std.math.mul(usize, spec.max_tokens, spec.hidden_dim) catch return error.Overflow;
        const total_elems = std.math.mul(usize, per_branch, GR_NUM_BRANCHES) catch return error.Overflow;
        const total_bytes = std.math.mul(usize, total_elems, 2) catch return error.Overflow;
        return page.alignUp(total_bytes) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: GrStreamSpec) ArenaError!GrStreamArena {
        const total = try requiredBytes(spec);
        const backing = try arena.allocPages(total);
        @memset(backing, 0);
        return .{
            .spec = spec,
            .backing = backing,
            .base_offset = @intCast(arena.offsetOf(backing.ptr)),
        };
    }

    pub fn binding(self: *const GrStreamArena, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.base_offset };
    }

    pub fn branchRawBytes(self: *const GrStreamArena, branch: u32) ArenaError![]u8 {
        if (branch >= GR_NUM_BRANCHES) return error.InvalidConfig;
        const stride = @as(usize, self.spec.max_tokens) * self.spec.hidden_dim * 2;
        const start = @as(usize, branch) * stride;
        return self.backing[start .. start + stride];
    }
};

pub const GrCollapseMode = union(enum) {
    /// Extract branch 0 only.
    stream0,
    /// Sum all 4 branches.
    sum,
    /// Mean of all 4 branches (`0.25 * sum`).
    mean,
    /// Weighted sum with shared `[4]f32` scalar weights across all tokens.
    weighted_shared: [GR_NUM_BRANCHES]f32,
    /// Weighted sum with per-token `[m, 4]f32` (`float4`) buffer weights.
    weighted_per_token: metal.BufferBinding,
};

pub const GrInitArgs = struct {
    x0: metal.BufferBinding,
    stream: metal.BufferBinding,
    m: u32,
    hidden_dim: u32,
    branch_token_stride: u32,
    stream_dtype: quant.FloatDType = .f16,
    act_dtype: quant.FloatDType = .f16,
    /// When `false` (default): `stream[0] = x0`, `stream[1..3] = 0`.
    /// When `true`: broadcasts `x0` to all 4 branches.
    broadcast_all: bool = false,
};

pub const GrReadGateArgs = struct {
    stream: metal.BufferBinding,
    /// `[m, 4]` `f32` (`float4`) per-token branch read gates.
    g_read: metal.BufferBinding,
    x_in: metal.BufferBinding,
    m: u32,
    hidden_dim: u32,
    branch_token_stride: u32,
    stream_dtype: quant.FloatDType = .f16,
    act_dtype: quant.FloatDType = .f16,
};

pub const GrWriteGateArgs = struct {
    stream: metal.BufferBinding,
    y_delta: metal.BufferBinding,
    /// `[m, 4]` `f32` (`float4`) per-token branch decay factors.
    decay: metal.BufferBinding,
    /// `[m, 4]` `f32` (`float4`) per-token branch write gates.
    g_write: metal.BufferBinding,
    m: u32,
    hidden_dim: u32,
    branch_token_stride: u32,
    stream_dtype: quant.FloatDType = .f16,
    act_dtype: quant.FloatDType = .f16,
};

pub const GrCollapseArgs = struct {
    stream: metal.BufferBinding,
    out: metal.BufferBinding,
    mode: GrCollapseMode = .sum,
    m: u32,
    hidden_dim: u32,
    branch_token_stride: u32,
    stream_dtype: quant.FloatDType = .f16,
    act_dtype: quant.FloatDType = .f16,
};

pub const GrComputeGatesArgs = struct {
    stream: metal.BufferBinding,
    w_read: metal.BufferBinding,
    w_write: metal.BufferBinding,
    w_decay: metal.BufferBinding,
    g_read_out: metal.BufferBinding,
    g_write_out: metal.BufferBinding,
    decay_out: metal.BufferBinding,
    m: u32,
    hidden_dim: u32,
    branch_token_stride: u32,
    weight_dtype: quant.FloatDType = .bf16,
};

pub const GdnComputeGatesArgs = struct {
    x: metal.BufferBinding,
    w_a: metal.BufferBinding,
    w_b: metal.BufferBinding,
    a_log: metal.BufferBinding,
    dt_bias: metal.BufferBinding,
    g_out: metal.BufferBinding,
    beta_out: metal.BufferBinding,
    m: u32,
    hidden_dim: u32,
    num_v_heads: u32,
    weight_dtype: quant.FloatDType = .bf16,
};

pub const GdnConv1dSiluArgs = struct {
    q_in: metal.BufferBinding,
    k_in: metal.BufferBinding,
    v_in: metal.BufferBinding,
    conv_w: metal.BufferBinding,
    conv_state_in: metal.BufferBinding,
    conv_state_out: metal.BufferBinding,
    ancestor_masks: ?metal.BufferBinding = null,
    q_out: metal.BufferBinding,
    k_out: metal.BufferBinding,
    v_out: metal.BufferBinding,
    m: u32,
    gdn_qk: u32,
    gdn_v: u32,
    conv_k: u32,
    is_tree: bool = false,
    write_state: bool = false,
};

pub const GdnConv1dCommitArgs = struct {
    q_raw: metal.BufferBinding,
    k_raw: metal.BufferBinding,
    v_raw: metal.BufferBinding,
    conv_state_in: metal.BufferBinding,
    conv_state_out: metal.BufferBinding,
    accepted_nodes: []const u16,
    gdn_qk: u32,
    gdn_v: u32,
    conv_k: u32,
};

pub const BlendF16Args = struct {
    dst: metal.BufferBinding,
    src: metal.BufferBinding,
    count: u32,
    a: f32,
    b: f32,
};

// ===========================================================================
// HybridPipelines: Startup-Compiled Metal 4 Compute Pipelines
// ===========================================================================

pub const HybridPipelines = struct {
    // GDN (R6 + G8)
    gdn_decode_d64_f16: metal.Pipeline,
    gdn_decode_d64_bf16: metal.Pipeline,
    gdn_decode_d128_f16: metal.Pipeline,
    gdn_decode_d128_bf16: metal.Pipeline,
    gdn_decode_dyn_f16: metal.Pipeline,
    gdn_decode_dyn_bf16: metal.Pipeline,
    gdn_prefill_d64: metal.Pipeline,
    gdn_prefill_d128: metal.Pipeline,
    gdn_prefill_dyn: metal.Pipeline,
    gdn_tree_verify_d64: metal.Pipeline,
    gdn_tree_verify_d128: metal.Pipeline,
    gdn_tree_verify_dyn: metal.Pipeline,
    gdn_out_norm_gate_f16: metal.Pipeline,
    gdn_out_norm_gate_bf16: metal.Pipeline,
    gdn_commit_state: metal.Pipeline,
    gdn_commit_tree: metal.Pipeline,

    // QSA (G9 + R5)
    qsa_rope_kv_write_f16: metal.Pipeline,
    qsa_rope_kv_write_bf16: metal.Pipeline,
    qsa_tree_rope_f16: metal.Pipeline,
    qsa_tree_rope_bf16: metal.Pipeline,
    qsa_kv_tree_commit: metal.Pipeline,
    qsa_update_block_summary: metal.Pipeline,
    qsa_index_blocks: metal.Pipeline,
    qsa_sparse_attn_d64: metal.Pipeline,
    qsa_sparse_attn_d128: metal.Pipeline,
    qsa_sparse_attn_dyn: metal.Pipeline,

    // 4-Branch GR (G10): indexed as `[stream_bf16][act_bf16]` (`0=f16, 1=bf16`)
    gr_init: [2][2]metal.Pipeline,
    gr_read: [2][2]metal.Pipeline,
    gr_write: [2][2]metal.Pipeline,
    gr_collapse: [2][2]metal.Pipeline,

    // Graph executor helpers
    gr_compute_gates_f16: metal.Pipeline,
    gr_compute_gates_bf16: metal.Pipeline,
    gdn_compute_gates_f16: metal.Pipeline,
    gdn_compute_gates_bf16: metal.Pipeline,
    gdn_conv1d_silu: metal.Pipeline,
    gdn_conv1d_commit: metal.Pipeline,
    blend_f16: metal.Pipeline,

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!HybridPipelines {
        const gdn_decode_d64_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer gdn_decode_d64_f16.close();

        const gdn_decode_d64_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer gdn_decode_d64_bf16.close();

        const gdn_decode_d128_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer gdn_decode_d128_f16.close();

        const gdn_decode_d128_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer gdn_decode_d128_bf16.close();

        const gdn_decode_dyn_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer gdn_decode_dyn_f16.close();

        const gdn_decode_dyn_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer gdn_decode_dyn_bf16.close();

        const gdn_prefill_d64 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_prefill", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
        });
        errdefer gdn_prefill_d64.close();

        const gdn_prefill_d128 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_prefill", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
        });
        errdefer gdn_prefill_d128.close();

        const gdn_prefill_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_prefill", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
        });
        errdefer gdn_prefill_dyn.close();

        const gdn_tree_verify_d64 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_tree_verify", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
        });
        errdefer gdn_tree_verify_d64.close();

        const gdn_tree_verify_d128 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_tree_verify", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
        });
        errdefer gdn_tree_verify_d128.close();

        const gdn_tree_verify_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_tree_verify", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
        });
        errdefer gdn_tree_verify_dyn.close();

        const gdn_out_norm_gate_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_out_norm_gate", &.{
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer gdn_out_norm_gate_f16.close();

        const gdn_out_norm_gate_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_out_norm_gate", &.{
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer gdn_out_norm_gate_bf16.close();

        const gdn_commit_state = try metal.Pipeline.init(device, library, "ami_gdn_commit_state");
        errdefer gdn_commit_state.close();

        const gdn_commit_tree = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_commit_tree", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
        });
        errdefer gdn_commit_tree.close();

        const qsa_rope_kv_write_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_qknorm_mrope_kv_write", &.{
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer qsa_rope_kv_write_f16.close();

        const qsa_rope_kv_write_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_qknorm_mrope_kv_write", &.{
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer qsa_rope_kv_write_bf16.close();

        const qsa_tree_rope_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_tree_qknorm_mrope", &.{
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer qsa_tree_rope_f16.close();

        const qsa_tree_rope_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_tree_qknorm_mrope", &.{
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer qsa_tree_rope_bf16.close();

        const qsa_kv_tree_commit = try metal.Pipeline.init(device, library, "ami_qsa_kv_tree_commit");
        errdefer qsa_kv_tree_commit.close();

        const qsa_update_block_summary = try metal.Pipeline.init(device, library, "ami_qsa_update_block_summary");
        errdefer qsa_update_block_summary.close();

        const qsa_index_blocks = try metal.Pipeline.init(device, library, "ami_qsa_index_blocks");
        errdefer qsa_index_blocks.close();

        const qsa_sparse_attn_d64 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_sparse_attn", &.{
            .{ .index = 2, .value = .{ .u32 = 64 } },
        });
        errdefer qsa_sparse_attn_d64.close();

        const qsa_sparse_attn_d128 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_sparse_attn", &.{
            .{ .index = 2, .value = .{ .u32 = 128 } },
        });
        errdefer qsa_sparse_attn_d128.close();

        const qsa_sparse_attn_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_sparse_attn", &.{
            .{ .index = 2, .value = .{ .u32 = 0 } },
        });
        errdefer qsa_sparse_attn_dyn.close();

        var gr_init: [2][2]metal.Pipeline = undefined;
        var gr_read: [2][2]metal.Pipeline = undefined;
        var gr_write: [2][2]metal.Pipeline = undefined;
        var gr_collapse: [2][2]metal.Pipeline = undefined;
        var gr_built: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < gr_built) : (idx += 1) {
                const kind = idx / 4;
                const pair = idx % 4;
                const sb = pair / 2;
                const ab = pair % 2;
                switch (kind) {
                    0 => gr_init[sb][ab].close(),
                    1 => gr_read[sb][ab].close(),
                    2 => gr_write[sb][ab].close(),
                    3 => gr_collapse[sb][ab].close(),
                    else => {},
                }
            }
        }
        const gr_names = [4][:0]const u8{
            "ami_gr_init_stream0",
            "ami_gr_read_gate",
            "ami_gr_write_gate",
            "ami_gr_collapse",
        };
        for (gr_names, 0..) |fn_name, kind| {
            for (0..2) |sb| {
                for (0..2) |ab| {
                    const p = try metal.Pipeline.initSpecialized(device, library, fn_name, &.{
                        .{ .index = 3, .value = .{ .bool = sb == 1 } },
                        .{ .index = 4, .value = .{ .bool = ab == 1 } },
                    });
                    switch (kind) {
                        0 => gr_init[sb][ab] = p,
                        1 => gr_read[sb][ab] = p,
                        2 => gr_write[sb][ab] = p,
                        3 => gr_collapse[sb][ab] = p,
                        else => unreachable,
                    }
                    gr_built += 1;
                }
            }
        }

        const gr_compute_gates_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_gr_compute_gates", &.{
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer gr_compute_gates_f16.close();
        const gr_compute_gates_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_gr_compute_gates", &.{
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer gr_compute_gates_bf16.close();

        const gdn_compute_gates_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_compute_gates", &.{
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer gdn_compute_gates_f16.close();
        const gdn_compute_gates_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_compute_gates", &.{
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer gdn_compute_gates_bf16.close();

        const gdn_conv1d_silu = try metal.Pipeline.init(device, library, "ami_gdn_conv1d_silu");
        errdefer gdn_conv1d_silu.close();
        const gdn_conv1d_commit = try metal.Pipeline.init(device, library, "ami_gdn_conv1d_commit");
        errdefer gdn_conv1d_commit.close();
        const blend_f16 = try metal.Pipeline.init(device, library, "ami_blend_f16");
        errdefer blend_f16.close();

        return .{
            .gdn_decode_d64_f16 = gdn_decode_d64_f16,
            .gdn_decode_d64_bf16 = gdn_decode_d64_bf16,
            .gdn_decode_d128_f16 = gdn_decode_d128_f16,
            .gdn_decode_d128_bf16 = gdn_decode_d128_bf16,
            .gdn_decode_dyn_f16 = gdn_decode_dyn_f16,
            .gdn_decode_dyn_bf16 = gdn_decode_dyn_bf16,
            .gdn_prefill_d64 = gdn_prefill_d64,
            .gdn_prefill_d128 = gdn_prefill_d128,
            .gdn_prefill_dyn = gdn_prefill_dyn,
            .gdn_tree_verify_d64 = gdn_tree_verify_d64,
            .gdn_tree_verify_d128 = gdn_tree_verify_d128,
            .gdn_tree_verify_dyn = gdn_tree_verify_dyn,
            .gdn_out_norm_gate_f16 = gdn_out_norm_gate_f16,
            .gdn_out_norm_gate_bf16 = gdn_out_norm_gate_bf16,
            .gdn_commit_state = gdn_commit_state,
            .gdn_commit_tree = gdn_commit_tree,
            .qsa_rope_kv_write_f16 = qsa_rope_kv_write_f16,
            .qsa_rope_kv_write_bf16 = qsa_rope_kv_write_bf16,
            .qsa_tree_rope_f16 = qsa_tree_rope_f16,
            .qsa_tree_rope_bf16 = qsa_tree_rope_bf16,
            .qsa_kv_tree_commit = qsa_kv_tree_commit,
            .qsa_update_block_summary = qsa_update_block_summary,
            .qsa_index_blocks = qsa_index_blocks,
            .qsa_sparse_attn_d64 = qsa_sparse_attn_d64,
            .qsa_sparse_attn_d128 = qsa_sparse_attn_d128,
            .qsa_sparse_attn_dyn = qsa_sparse_attn_dyn,
            .gr_init = gr_init,
            .gr_read = gr_read,
            .gr_write = gr_write,
            .gr_collapse = gr_collapse,
            .gr_compute_gates_f16 = gr_compute_gates_f16,
            .gr_compute_gates_bf16 = gr_compute_gates_bf16,
            .gdn_compute_gates_f16 = gdn_compute_gates_f16,
            .gdn_compute_gates_bf16 = gdn_compute_gates_bf16,
            .gdn_conv1d_silu = gdn_conv1d_silu,
            .gdn_conv1d_commit = gdn_conv1d_commit,
            .blend_f16 = blend_f16,
        };
    }

    pub fn close(self: HybridPipelines) void {
        self.blend_f16.close();
        self.gdn_conv1d_commit.close();
        self.gdn_conv1d_silu.close();
        self.gdn_compute_gates_bf16.close();
        self.gdn_compute_gates_f16.close();
        self.gr_compute_gates_bf16.close();
        self.gr_compute_gates_f16.close();
        for (0..2) |sb| {
            for (0..2) |ab| {
                self.gr_collapse[sb][ab].close();
                self.gr_write[sb][ab].close();
                self.gr_read[sb][ab].close();
                self.gr_init[sb][ab].close();
            }
        }
        self.qsa_sparse_attn_dyn.close();
        self.qsa_sparse_attn_d128.close();
        self.qsa_sparse_attn_d64.close();
        self.qsa_index_blocks.close();
        self.qsa_update_block_summary.close();
        self.qsa_kv_tree_commit.close();
        self.qsa_tree_rope_bf16.close();
        self.qsa_tree_rope_f16.close();
        self.qsa_rope_kv_write_bf16.close();
        self.qsa_rope_kv_write_f16.close();
        self.gdn_commit_tree.close();
        self.gdn_commit_state.close();
        self.gdn_out_norm_gate_bf16.close();
        self.gdn_out_norm_gate_f16.close();
        self.gdn_tree_verify_dyn.close();
        self.gdn_tree_verify_d128.close();
        self.gdn_tree_verify_d64.close();
        self.gdn_prefill_dyn.close();
        self.gdn_prefill_d128.close();
        self.gdn_prefill_d64.close();
        self.gdn_decode_dyn_bf16.close();
        self.gdn_decode_dyn_f16.close();
        self.gdn_decode_d128_bf16.close();
        self.gdn_decode_d128_f16.close();
        self.gdn_decode_d64_bf16.close();
        self.gdn_decode_d64_f16.close();
    }

    // -----------------------------------------------------------------------
    // GDN Encoders (`R6` + `G8`)
    // -----------------------------------------------------------------------

    fn validateGdnDims(num_k_heads: u32, num_v_heads: u32, head_k_dim: u32, head_v_dim: u32) EncodeError!void {
        if (num_k_heads == 0 or num_v_heads == 0 or head_k_dim == 0 or head_v_dim == 0) {
            return error.ZeroDimension;
        }
        if ((num_v_heads % num_k_heads) != 0) return error.InvalidConfig;
        if (head_k_dim > MAX_GDN_HEAD_DIM or head_v_dim > MAX_GDN_HEAD_DIM) return error.InvalidHeadDim;
    }

    /// Encodes single-step Gated DeltaNet decode (`M = 1`) with fused optional
    /// RMSNorm + SiLU output gate and explicit `state_in` -> `state_out` bindings.
    pub fn encodeGdnDecode(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GdnDecodeArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try validateGdnDims(args.num_k_heads, args.num_v_heads, args.head_k_dim, args.head_v_dim);
        const q_scale = args.q_scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_k_dim))));
        const params: GdnDecodeParams = .{
            .num_k_heads = args.num_k_heads,
            .num_v_heads = args.num_v_heads,
            .head_k_dim = args.head_k_dim,
            .head_v_dim = args.head_v_dim,
            .q_scale = q_scale,
            .l2_eps = args.l2_eps,
            .rms_eps = args.rms_eps,
            .apply_l2_norm = if (args.apply_l2_norm) 1 else 0,
            .apply_out_norm = if (args.norm_w != null) 1 else 0,
            .apply_out_gate = if (args.z != null) 1 else 0,
            .per_head_norm = if (args.per_head_norm) 1 else 0,
        };
        const pipe = switch (args.head_k_dim) {
            64 => if (args.norm_dtype == .bf16) self.gdn_decode_d64_bf16 else self.gdn_decode_d64_f16,
            128 => if (args.norm_dtype == .bf16) self.gdn_decode_d128_bf16 else self.gdn_decode_d128_f16,
            else => if (args.norm_dtype == .bf16) self.gdn_decode_dyn_bf16 else self.gdn_decode_dyn_f16,
        };
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.q,
                args.k,
                args.v,
                args.g,
                args.beta,
                args.z orelse args.v,
                args.norm_w orelse args.v,
                args.state_in,
                args.state_out,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.num_v_heads, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes chunked Gated DeltaNet prefill (`seq_len >= 1`) keeping state
    /// columns in FP32 registers across the full sequence, with optional
    /// RMSNorm + SiLU output gate.
    pub fn encodeGdnPrefill(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GdnPrefillArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.seq_len == 0) return error.ZeroDimension;
        try validateGdnDims(args.num_k_heads, args.num_v_heads, args.head_k_dim, args.head_v_dim);
        const q_scale = args.q_scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_k_dim))));
        const need_post_norm = (args.norm_w != null);
        const params: GdnPrefillParams = .{
            .seq_len = args.seq_len,
            .num_k_heads = args.num_k_heads,
            .num_v_heads = args.num_v_heads,
            .head_k_dim = args.head_k_dim,
            .head_v_dim = args.head_v_dim,
            .q_scale = q_scale,
            .l2_eps = args.l2_eps,
            .apply_l2_norm = if (args.apply_l2_norm) 1 else 0,
            .apply_out_gate = if (!need_post_norm and args.z != null) 1 else 0,
            .chunk_size = if (args.chunk_size > 0) args.chunk_size else 64,
        };
        const pipe = switch (args.head_k_dim) {
            64 => self.gdn_prefill_d64,
            128 => self.gdn_prefill_d128,
            else => self.gdn_prefill_dyn,
        };
        const groups_y: u64 = (@as(u64, args.head_v_dim) + 7) / 8;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.q,
                args.k,
                args.v,
                args.g,
                args.beta,
                args.z orelse args.v,
                args.state_in,
                args.state_out,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.num_v_heads, groups_y, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );

        if (need_post_norm) {
            const norm_params: GdnOutNormGateParams = .{
                .num_tokens = args.seq_len,
                .num_v_heads = args.num_v_heads,
                .head_v_dim = args.head_v_dim,
                .rms_eps = args.rms_eps,
                .apply_out_norm = 1,
                .apply_out_gate = if (args.z != null) 1 else 0,
                .per_head_norm = if (args.per_head_norm) 1 else 0,
            };
            const norm_pipe = if (args.norm_dtype == .bf16) self.gdn_out_norm_gate_bf16 else self.gdn_out_norm_gate_f16;
            try frame.encodeThreadgroups(
                norm_pipe,
                &.{
                    args.out,
                    args.z orelse args.out,
                    args.norm_w.?,
                },
                std.mem.asBytes(&norm_params),
                .{ args.num_v_heads, args.seq_len, 1 },
                .{ 32, 1, 1 },
                true,
            );
        }
    }

    /// Encodes `M = 1..64` speculative tree verification for Gated DeltaNet
    /// using 64-bit causal ancestor masks without mutating committed `state_in`.
    pub fn encodeGdnTreeVerify(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GdnTreeVerifyArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_nodes == 0 or args.num_nodes > MAX_TREE_NODES) return error.InvalidNodeCount;
        try validateGdnDims(args.num_k_heads, args.num_v_heads, args.head_k_dim, args.head_v_dim);
        if (args.scratch_leaf_node) |leaf| {
            if (leaf >= args.num_nodes or args.state_out == null) return error.InvalidConfig;
        }
        const q_scale = args.q_scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_k_dim))));
        const need_post_norm = (args.norm_w != null);
        const params: GdnTreeVerifyParams = .{
            .num_nodes = args.num_nodes,
            .num_k_heads = args.num_k_heads,
            .num_v_heads = args.num_v_heads,
            .head_k_dim = args.head_k_dim,
            .head_v_dim = args.head_v_dim,
            .q_scale = q_scale,
            .l2_eps = args.l2_eps,
            .apply_l2_norm = if (args.apply_l2_norm) 1 else 0,
            .apply_out_gate = if (!need_post_norm and args.z != null) 1 else 0,
            .scratch_leaf_node = args.scratch_leaf_node orelse 0xFFFF_FFFF,
        };
        const pipe = switch (args.head_k_dim) {
            64 => self.gdn_tree_verify_d64,
            128 => self.gdn_tree_verify_d128,
            else => self.gdn_tree_verify_dyn,
        };
        const groups_y: u64 = (@as(u64, args.head_v_dim) + 7) / 8;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.q,
                args.k,
                args.v,
                args.g,
                args.beta,
                args.z orelse args.v,
                args.ancestor_masks,
                args.state_in,
                args.state_out orelse args.state_in,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.num_v_heads, groups_y, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );

        if (need_post_norm) {
            const norm_params: GdnOutNormGateParams = .{
                .num_tokens = args.num_nodes,
                .num_v_heads = args.num_v_heads,
                .head_v_dim = args.head_v_dim,
                .rms_eps = args.rms_eps,
                .apply_out_norm = 1,
                .apply_out_gate = if (args.z != null) 1 else 0,
                .per_head_norm = if (args.per_head_norm) 1 else 0,
            };
            const norm_pipe = if (args.norm_dtype == .bf16) self.gdn_out_norm_gate_bf16 else self.gdn_out_norm_gate_f16;
            try frame.encodeThreadgroups(
                norm_pipe,
                &.{
                    args.out,
                    args.z orelse args.out,
                    args.norm_w.?,
                },
                std.mem.asBytes(&norm_params),
                .{ args.num_v_heads, args.num_nodes, 1 },
                .{ 32, 1, 1 },
                true,
            );
        }
    }

    /// Commits `S_tmp[layer_idx] -> S_t[layer_idx]` on GPU (`ami_gdn_commit_state`,
    /// ADR-0019 final pass `r == R - 1`). Zero CPU-GPU sync, zero heap allocs.
    pub fn encodeCommitScratchState(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        state_arena: *const GdnStateArena,
        state_buf: metal.Buffer,
        layer_idx: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const src = state_arena.scratchBinding(state_buf, layer_idx) catch return error.InvalidLayerIndex;
        const dst = state_arena.committedBinding(state_buf, layer_idx) catch return error.InvalidLayerIndex;
        const elems: u32 = @intCast(state_arena.layer_state_elems);
        const params: GdnCommitStateParams = .{ .total_f32_elems = elems };
        const vec_n = elems >> 2;
        const tail = elems & 3;
        const total_threads: u64 = @as(u64, vec_n) + tail;
        try frame.encodeThreads(
            self.gdn_commit_state,
            &.{ src, dst },
            std.mem.asBytes(&params),
            .{ total_threads, 1, 1 },
            .{ @min(total_threads, 256), 1, 1 },
            insert_barrier_before,
        );
    }

    /// Commits an accepted speculative tree path onto `state_in` -> `state_out`
    /// (`ami_gdn_commit_tree`) without storing per-node state copies.
    pub fn encodeCommitTreeState(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GdnCommitTreeArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.accepted_count == 0 or args.accepted_count > MAX_TREE_NODES) return error.InvalidNodeCount;
        try validateGdnDims(args.num_k_heads, args.num_v_heads, args.head_k_dim, args.head_v_dim);
        var params: GdnCommitTreeParams = .{
            .num_k_heads = args.num_k_heads,
            .num_v_heads = args.num_v_heads,
            .head_k_dim = args.head_k_dim,
            .head_v_dim = args.head_v_dim,
            .accepted_count = args.accepted_count,
            .use_node_buffer = 0,
            .l2_eps = args.l2_eps,
            .apply_l2_norm = if (args.apply_l2_norm) 1 else 0,
        };
        var node_buf: metal.BufferBinding = args.k_tree;
        switch (args.accepted_nodes) {
            .inline_u16 => |nodes| {
                if (nodes.len < args.accepted_count) return error.InvalidNodeCount;
                for (0..args.accepted_count) |i| params.inline_nodes[i] = nodes[i];
            },
            .inline_u32 => |nodes| {
                if (nodes.len < args.accepted_count) return error.InvalidNodeCount;
                for (0..args.accepted_count) |i| {
                    if (nodes[i] >= MAX_TREE_NODES) return error.InvalidNodeCount;
                    params.inline_nodes[i] = @intCast(nodes[i]);
                }
            },
            .buffer_u32 => |buf_binding| {
                params.use_node_buffer = 1;
                node_buf = buf_binding;
            },
        }
        const groups_y: u64 = (@as(u64, args.head_v_dim) + 7) / 8;
        try frame.encodeThreadgroups(
            self.gdn_commit_tree,
            &.{
                args.k_tree,
                args.v_tree,
                args.g_tree,
                args.beta_tree,
                node_buf,
                args.state_in,
                args.state_out,
            },
            std.mem.asBytes(&params),
            .{ args.num_v_heads, groups_y, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    // -----------------------------------------------------------------------
    // QSA Encoders (`G9` + `R5`)
    // -----------------------------------------------------------------------

    pub fn encodeQsaRopeKvWrite(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: QsaRopeKvWriteArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_q_heads == 0 or args.num_kv_heads == 0 or args.head_dim == 0 or args.chunk_tokens == 0) {
            return error.ZeroDimension;
        }
        if (args.head_dim > MAX_QSA_HEAD_DIM or (args.head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.slot_in_chunk >= args.chunk_tokens) return error.InvalidConfig;
        const rotary_dim = if (args.rotary_dim == 0) args.head_dim else args.rotary_dim;
        if (rotary_dim > args.head_dim) return error.InvalidRotaryDim;
        try args.mrope.validate(rotary_dim);

        const apply_norm = (args.q_norm != null or args.k_norm != null);
        if (apply_norm and (args.q_norm == null or args.k_norm == null)) return error.MissingNormWeights;

        const params: QsaRopeKvParams = .{
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .rotary_dim = rotary_dim,
            .pos_t = args.pos_3d.t,
            .pos_h = args.pos_3d.h,
            .pos_w = args.pos_3d.w,
            .mrope_s0 = args.mrope.section[0],
            .mrope_s1 = args.mrope.section[1],
            .mrope_s2 = args.mrope.section[2],
            .chunk_tokens = args.chunk_tokens,
            .slot_in_chunk = args.slot_in_chunk,
            .rms_eps = args.rms_eps,
            .rope_theta = args.rope_theta,
            .yarn_factor = args.yarn.factor,
            .yarn_mscale = args.yarn.effectiveMscale(),
            .yarn_beta_fast = args.yarn.beta_fast,
            .yarn_beta_slow = args.yarn.beta_slow,
            .original_max_pos = args.yarn.original_max_pos,
            .apply_qk_norm = if (apply_norm) 1 else 0,
        };
        const pipe = if (args.norm_dtype == .bf16) self.qsa_rope_kv_write_bf16 else self.qsa_rope_kv_write_f16;
        try frame.encodeThreadgroupsWithTextures(
            pipe,
            &.{
                args.q,
                args.k,
                args.v,
                args.q_norm orelse args.q,
                args.k_norm orelse args.k,
            },
            &.{
                .{ .tile = args.k_tile, .view = .rgba16f },
                .{ .tile = args.v_tile, .view = .rgba16f },
            },
            std.mem.asBytes(&params),
            .{ @as(u64, args.num_q_heads) + args.num_kv_heads, 1, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeQsaTreeRope(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: QsaTreeRopeArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_nodes == 0 or args.num_nodes > MAX_TREE_NODES) return error.InvalidNodeCount;
        if (args.num_q_heads == 0 or args.num_kv_heads == 0 or args.head_dim == 0) return error.ZeroDimension;
        if (args.head_dim > MAX_QSA_HEAD_DIM or (args.head_dim % 4) != 0) return error.InvalidHeadDim;
        const rotary_dim = if (args.rotary_dim == 0) args.head_dim else args.rotary_dim;
        if (rotary_dim > args.head_dim) return error.InvalidRotaryDim;
        try args.mrope.validate(rotary_dim);

        const apply_norm = (args.q_norm != null or args.k_norm != null);
        if (apply_norm and (args.q_norm == null or args.k_norm == null)) return error.MissingNormWeights;

        const params: QsaTreeRopeParams = .{
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .rotary_dim = rotary_dim,
            .mrope_s0 = args.mrope.section[0],
            .mrope_s1 = args.mrope.section[1],
            .mrope_s2 = args.mrope.section[2],
            .rms_eps = args.rms_eps,
            .rope_theta = args.rope_theta,
            .yarn_factor = args.yarn.factor,
            .yarn_mscale = args.yarn.effectiveMscale(),
            .yarn_beta_fast = args.yarn.beta_fast,
            .yarn_beta_slow = args.yarn.beta_slow,
            .original_max_pos = args.yarn.original_max_pos,
            .apply_qk_norm = if (apply_norm) 1 else 0,
        };
        const pipe = if (args.norm_dtype == .bf16) self.qsa_tree_rope_bf16 else self.qsa_tree_rope_f16;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.q,
                args.k,
                args.v,
                args.q_norm orelse args.q,
                args.k_norm orelse args.k,
                args.positions_3d,
                args.k_tree_out,
                args.v_tree_out,
            },
            std.mem.asBytes(&params),
            .{ @as(u64, args.num_q_heads) + args.num_kv_heads, args.num_nodes, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeQsaKvTreeCommit(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: QsaKvTreeCommitArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.accepted_count == 0 or args.accepted_count > MAX_TREE_NODES) return error.InvalidNodeCount;
        if (args.num_kv_heads == 0 or args.head_dim == 0 or args.chunk_tokens == 0) return error.ZeroDimension;
        if (args.head_dim > MAX_QSA_HEAD_DIM or (args.head_dim % 4) != 0) return error.InvalidHeadDim;

        const base_phys = if (args.ring_capacity > 0) args.base_pos % args.ring_capacity else args.base_pos;
        const end_pos = args.base_pos + args.accepted_count - 1;
        const end_phys = if (args.ring_capacity > 0) end_pos % args.ring_capacity else end_pos;
        const crosses = (end_phys / args.chunk_tokens) != (base_phys / args.chunk_tokens);
        if (crosses and (args.k_tile1 == null or args.v_tile1 == null)) return error.MissingSecondChunk;

        var params: QsaKvTreeCommitParams = .{
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .base_pos = args.base_pos,
            .chunk_tokens = args.chunk_tokens,
            .ring_capacity = args.ring_capacity,
            .accepted_count = args.accepted_count,
            .use_node_buffer = 0,
            .has_second_chunk = if (crosses) 1 else 0,
        };
        var node_buf: metal.BufferBinding = args.k_tree;
        switch (args.accepted_nodes) {
            .inline_u16 => |nodes| {
                if (nodes.len < args.accepted_count) return error.InvalidNodeCount;
                for (0..args.accepted_count) |i| params.inline_nodes[i] = nodes[i];
            },
            .inline_u32 => |nodes| {
                if (nodes.len < args.accepted_count) return error.InvalidNodeCount;
                for (0..args.accepted_count) |i| {
                    if (nodes[i] >= MAX_TREE_NODES) return error.InvalidNodeCount;
                    params.inline_nodes[i] = @intCast(nodes[i]);
                }
            },
            .buffer_u32 => |buf_binding| {
                params.use_node_buffer = 1;
                node_buf = buf_binding;
            },
        }
        try frame.encodeThreadgroupsWithTextures(
            self.qsa_kv_tree_commit,
            &.{ args.k_tree, args.v_tree, node_buf },
            &.{
                .{ .tile = args.k_tile0, .view = .rgba16f },
                .{ .tile = args.v_tile0, .view = .rgba16f },
                .{ .tile = args.k_tile1 orelse args.k_tile0, .view = .rgba16f },
                .{ .tile = args.v_tile1 orelse args.v_tile0, .view = .rgba16f },
            },
            std.mem.asBytes(&params),
            .{ args.num_kv_heads, args.accepted_count, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeQsaUpdateBlockSummary(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: QsaBlockSummaryArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_kv_heads == 0 or args.head_dim == 0 or args.chunk_tokens == 0 or
            args.block_tokens == 0 or args.num_blocks == 0)
        {
            return error.ZeroDimension;
        }
        if (args.head_dim > MAX_QSA_HEAD_DIM or (args.head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.ring_capacity > 0 and (args.ring_capacity % args.block_tokens) != 0) return error.InvalidConfig;
        const params: QsaBlockSummaryParams = .{
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .kv_len = args.kv_len,
            .window_size = args.window_size,
            .ring_capacity = args.ring_capacity,
            .chunk_tokens = args.chunk_tokens,
            .block_tokens = args.block_tokens,
            .start_block = args.start_block,
            .num_blocks = args.num_blocks,
        };
        try frame.encodeThreadgroups(
            self.qsa_update_block_summary,
            &.{ args.kv_tiles, args.block_keys },
            std.mem.asBytes(&params),
            .{ args.num_kv_heads, args.num_blocks, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeQsaIndexBlocks(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: QsaIndexBlocksArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_nodes == 0 or args.num_nodes > MAX_TREE_NODES) return error.InvalidNodeCount;
        if (args.num_q_heads == 0 or args.num_kv_heads == 0 or args.head_dim == 0 or args.block_tokens == 0) {
            return error.ZeroDimension;
        }
        if ((args.num_q_heads % args.num_kv_heads) != 0) return error.InvalidConfig;
        if (args.head_dim > MAX_QSA_HEAD_DIM or (args.head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.top_k_blocks == 0 or args.top_k_blocks > QSA_MAX_TOP_K_BLOCKS) return error.UnsupportedTopKBlocks;
        if (args.num_nodes > args.index_buf.spec.max_nodes or
            args.num_kv_heads > args.index_buf.spec.num_kv_heads or
            args.top_k_blocks > args.index_buf.spec.top_k_blocks)
        {
            return error.InvalidConfig;
        }
        if (args.ring_capacity > 0 and (args.ring_capacity % args.block_tokens) != 0) return error.InvalidConfig;

        const scale = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_dim))));
        const params: QsaIndexParams = .{
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .kv_len = args.kv_len,
            .window_size = args.window_size,
            .ring_capacity = args.ring_capacity,
            .block_tokens = args.block_tokens,
            .top_k_blocks = args.top_k_blocks,
            .num_local_blocks = if (args.num_local_blocks > 0) args.num_local_blocks else 1,
            .scale = scale,
        };
        try frame.encodeThreadgroups(
            self.qsa_index_blocks,
            &.{
                args.q,
                args.block_keys,
                args.index_buf.blocksBinding(args.index_metal_buf),
                args.index_buf.countsBinding(args.index_metal_buf),
            },
            std.mem.asBytes(&params),
            .{ args.num_kv_heads, args.num_nodes, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeQsaSparseAttn(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: QsaSparseAttnArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_nodes == 0 or args.num_nodes > MAX_TREE_NODES) return error.InvalidNodeCount;
        if (args.num_q_heads == 0 or args.num_kv_heads == 0 or args.head_dim == 0 or
            args.chunk_tokens == 0 or args.block_tokens == 0)
        {
            return error.ZeroDimension;
        }
        if ((args.num_q_heads % args.num_kv_heads) != 0) return error.InvalidConfig;
        if (args.head_dim > MAX_QSA_HEAD_DIM or (args.head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.top_k_blocks == 0 or args.top_k_blocks > QSA_MAX_TOP_K_BLOCKS) return error.UnsupportedTopKBlocks;
        if (args.kv_len > 0 and args.kv_tiles == null) return error.InvalidConfig;

        const has_any_tree = (args.k_tree != null or args.v_tree != null or args.ancestor_masks != null);
        if (has_any_tree and (args.k_tree == null or args.v_tree == null or args.ancestor_masks == null)) {
            return error.MissingTreeInputs;
        }

        const scale = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_dim))));
        const params: QsaSparseAttnParams = .{
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .kv_len = args.kv_len,
            .window_size = args.window_size,
            .ring_capacity = args.ring_capacity,
            .chunk_tokens = args.chunk_tokens,
            .block_tokens = args.block_tokens,
            .top_k_blocks = args.top_k_blocks,
            .has_tree = if (has_any_tree) 1 else 0,
            .scale = scale,
        };
        const gqa_ratio = args.num_q_heads / args.num_kv_heads;
        const q_groups_per_kv: u64 = (@as(u64, gqa_ratio) + 7) / 8;
        const pipe = switch (args.head_dim) {
            64 => self.qsa_sparse_attn_d64,
            128 => self.qsa_sparse_attn_d128,
            else => self.qsa_sparse_attn_dyn,
        };
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.q,
                args.kv_tiles orelse args.q,
                args.index_buf.blocksBinding(args.index_metal_buf),
                args.index_buf.countsBinding(args.index_metal_buf),
                args.k_tree orelse args.q,
                args.v_tree orelse args.q,
                args.ancestor_masks orelse args.q,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ @as(u64, args.num_kv_heads) * q_groups_per_kv, args.num_nodes, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Combined QSA indexer + sparse attention entry point supporting `ADR-0019`
    /// cross-loop index reuse:
    /// - `reuse_block_indices == false` (`r = 0`): runs `ami_qsa_index_blocks`
    ///   then `ami_qsa_sparse_attn`.
    /// - `reuse_block_indices == true` (`r >= 1`): skips `ami_qsa_index_blocks`
    ///   completely and runs `ami_qsa_sparse_attn` over the wired `index_buf`.
    pub fn encodeQsaAttention(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: QsaAttentionArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        var attn_barrier = insert_barrier_before;
        if (!args.reuse_block_indices) {
            try self.encodeQsaIndexBlocks(frame, .{
                .q = args.q,
                .block_keys = args.block_keys,
                .index_buf = args.index_buf,
                .index_metal_buf = args.index_metal_buf,
                .num_nodes = args.num_nodes,
                .num_q_heads = args.num_q_heads,
                .num_kv_heads = args.num_kv_heads,
                .head_dim = args.head_dim,
                .kv_len = args.kv_len,
                .window_size = args.window_size,
                .ring_capacity = args.ring_capacity,
                .block_tokens = args.block_tokens,
                .top_k_blocks = args.top_k_blocks,
                .num_local_blocks = args.num_local_blocks,
                .scale = args.scale,
            }, insert_barrier_before);
            attn_barrier = true;
        }
        try self.encodeQsaSparseAttn(frame, .{
            .q = args.q,
            .kv_tiles = args.kv_tiles,
            .index_buf = args.index_buf,
            .index_metal_buf = args.index_metal_buf,
            .k_tree = args.k_tree,
            .v_tree = args.v_tree,
            .ancestor_masks = args.ancestor_masks,
            .out = args.out,
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .kv_len = args.kv_len,
            .window_size = args.window_size,
            .ring_capacity = args.ring_capacity,
            .chunk_tokens = args.chunk_tokens,
            .block_tokens = args.block_tokens,
            .top_k_blocks = args.top_k_blocks,
            .scale = args.scale,
        }, attn_barrier);
    }

    // -----------------------------------------------------------------------
    // 4-Branch Gated Residual Encoders (`G10`)
    // -----------------------------------------------------------------------

    fn grIndex(dt: quant.FloatDType) usize {
        return if (dt == .bf16) 1 else 0;
    }

    pub fn encodeGrInitStream0(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GrInitArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_dim == 0 or args.branch_token_stride < args.m) return error.InvalidConfig;
        const params: GrParams = .{
            .m = args.m,
            .hidden_dim = args.hidden_dim,
            .branch_token_stride = args.branch_token_stride,
            .mode = if (args.broadcast_all) 1 else 0,
        };
        const total: u64 = @as(u64, args.m) * args.hidden_dim;
        const pipe = self.gr_init[grIndex(args.stream_dtype)][grIndex(args.act_dtype)];
        try frame.encodeThreads(
            pipe,
            &.{ args.x0, args.stream },
            std.mem.asBytes(&params),
            .{ total, 1, 1 },
            .{ @min(total, 256), 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeGrReadGate(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GrReadGateArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_dim == 0 or args.branch_token_stride < args.m) return error.InvalidConfig;
        const params: GrParams = .{
            .m = args.m,
            .hidden_dim = args.hidden_dim,
            .branch_token_stride = args.branch_token_stride,
            .mode = 0,
        };
        const total: u64 = @as(u64, args.m) * args.hidden_dim;
        const pipe = self.gr_read[grIndex(args.stream_dtype)][grIndex(args.act_dtype)];
        try frame.encodeThreads(
            pipe,
            &.{ args.stream, args.g_read, args.x_in },
            std.mem.asBytes(&params),
            .{ total, 1, 1 },
            .{ @min(total, 256), 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeGrWriteGate(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GrWriteGateArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_dim == 0 or args.branch_token_stride < args.m) return error.InvalidConfig;
        const params: GrParams = .{
            .m = args.m,
            .hidden_dim = args.hidden_dim,
            .branch_token_stride = args.branch_token_stride,
            .mode = 0,
        };
        const total: u64 = @as(u64, args.m) * args.hidden_dim;
        const pipe = self.gr_write[grIndex(args.stream_dtype)][grIndex(args.act_dtype)];
        try frame.encodeThreads(
            pipe,
            &.{ args.stream, args.y_delta, args.decay, args.g_write },
            std.mem.asBytes(&params),
            .{ total, 1, 1 },
            .{ @min(total, 256), 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeGrCollapse(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GrCollapseArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_dim == 0 or args.branch_token_stride < args.m) return error.InvalidConfig;
        var params: GrParams = .{
            .m = args.m,
            .hidden_dim = args.hidden_dim,
            .branch_token_stride = args.branch_token_stride,
            .mode = 0,
        };
        var weights_binding: metal.BufferBinding = args.stream;
        switch (args.mode) {
            .stream0 => params.mode = 0,
            .sum => params.mode = 1,
            .mean => params.mode = 2,
            .weighted_shared => |w| {
                params.mode = 3;
                params.w0 = w[0];
                params.w1 = w[1];
                params.w2 = w[2];
                params.w3 = w[3];
            },
            .weighted_per_token => |wb| {
                params.mode = 4;
                weights_binding = wb;
            },
        }
        const total: u64 = @as(u64, args.m) * args.hidden_dim;
        const pipe = self.gr_collapse[grIndex(args.stream_dtype)][grIndex(args.act_dtype)];
        try frame.encodeThreads(
            pipe,
            &.{ args.stream, weights_binding, args.out },
            std.mem.asBytes(&params),
            .{ total, 1, 1 },
            .{ @min(total, 256), 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeGrComputeGates(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GrComputeGatesArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_dim == 0 or args.branch_token_stride < args.m) return error.InvalidConfig;
        const params: GrComputeGatesParams = .{
            .m = args.m,
            .hidden_dim = args.hidden_dim,
            .branch_token_stride = args.branch_token_stride,
        };
        const pipe = if (args.weight_dtype == .bf16) self.gr_compute_gates_bf16 else self.gr_compute_gates_f16;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.stream, args.w_read, args.w_write, args.w_decay, args.g_read_out, args.g_write_out, args.decay_out },
            std.mem.asBytes(&params),
            .{ @as(u64, args.m) * GR_NUM_BRANCHES, 1, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeGdnComputeGates(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GdnComputeGatesArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_dim == 0 or args.num_v_heads == 0) return error.ZeroDimension;
        const params: GdnComputeGatesParams = .{
            .m = args.m,
            .hidden_dim = args.hidden_dim,
            .num_v_heads = args.num_v_heads,
        };
        const pipe = if (args.weight_dtype == .bf16) self.gdn_compute_gates_bf16 else self.gdn_compute_gates_f16;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.w_a, args.w_b, args.a_log, args.dt_bias, args.g_out, args.beta_out },
            std.mem.asBytes(&params),
            .{ @as(u64, args.m) * args.num_v_heads, 1, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeGdnConv1dSilu(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GdnConv1dSiluArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.gdn_qk == 0 or args.gdn_v == 0 or args.conv_k == 0 or args.conv_k > 8) {
            return error.InvalidConfig;
        }
        if (args.is_tree and (args.ancestor_masks == null or args.m > MAX_TREE_NODES)) {
            return error.InvalidConfig;
        }
        const conv_dim: u64 = 2 * @as(u64, args.gdn_qk) + args.gdn_v;
        const params: GdnConv1dParams = .{
            .m = args.m,
            .gdn_qk = args.gdn_qk,
            .gdn_v = args.gdn_v,
            .conv_k = args.conv_k,
            .is_tree = if (args.is_tree) 1 else 0,
            .write_state = if (args.write_state) 1 else 0,
        };
        const masks_bind = args.ancestor_masks orelse args.conv_state_in;
        try frame.encodeThreads(
            self.gdn_conv1d_silu,
            &.{
                args.q_in,
                args.k_in,
                args.v_in,
                args.conv_w,
                args.conv_state_in,
                args.conv_state_out,
                masks_bind,
                args.q_out,
                args.k_out,
                args.v_out,
            },
            std.mem.asBytes(&params),
            .{ conv_dim, 1, 1 },
            .{ @min(conv_dim, 256), 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeGdnConv1dCommit(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GdnConv1dCommitArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.gdn_qk == 0 or args.gdn_v == 0 or args.conv_k <= 1 or args.conv_k > 8) return error.InvalidConfig;
        if (args.accepted_nodes.len == 0 or args.accepted_nodes.len > MAX_TREE_NODES) return error.InvalidConfig;
        const conv_dim: u64 = 2 * @as(u64, args.gdn_qk) + args.gdn_v;
        var params: GdnConv1dCommitParams = .{
            .gdn_qk = args.gdn_qk,
            .gdn_v = args.gdn_v,
            .conv_k = args.conv_k,
            .accepted_count = @intCast(args.accepted_nodes.len),
        };
        for (args.accepted_nodes, 0..) |nid, i| {
            params.inline_nodes[i] = nid;
        }
        try frame.encodeThreads(
            self.gdn_conv1d_commit,
            &.{ args.q_raw, args.k_raw, args.v_raw, args.conv_state_in, args.conv_state_out },
            std.mem.asBytes(&params),
            .{ conv_dim, 1, 1 },
            .{ @min(conv_dim, 256), 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeBlendF16(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: BlendF16Args,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.count == 0) return error.ZeroDimension;
        const params: BlendF16Params = .{
            .count = args.count,
            .a = args.a,
            .b = args.b,
        };
        try frame.encodeThreads(
            self.blend_f16,
            &.{ args.dst, args.src },
            std.mem.asBytes(&params),
            .{ args.count, 1, 1 },
            .{ @min(@as(u64, args.count), 256), 1, 1 },
            insert_barrier_before,
        );
    }
};

// ===========================================================================
// CPU FP32 Reference Implementations (for exact step-by-step parity tests)
// ===========================================================================

fn siluF32(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

fn sigmoidF32(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub fn gdnStepCpuRef(
    q: []const f16,
    k: []const f16,
    v: []const f16,
    g: []const f32,
    beta: []const f32,
    z: ?[]const f16,
    norm_w: ?[]const u16,
    norm_dtype: quant.FloatDType,
    per_head_norm: bool,
    state_in: []const f32,
    state_out: []f32,
    out_f32: []f32,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    q_scale: f32,
    l2_eps: f32,
    rms_eps: f32,
    apply_l2_norm: bool,
) void {
    const v_per_k = num_v_heads / num_k_heads;
    var q_buf: [MAX_GDN_HEAD_DIM]f32 = undefined;
    var k_buf: [MAX_GDN_HEAD_DIM]f32 = undefined;
    var o_buf: [MAX_GDN_HEAD_DIM]f32 = undefined;

    for (0..num_v_heads) |vh| {
        const kh = vh / v_per_k;
        var q_sq: f32 = 0.0;
        var k_sq: f32 = 0.0;
        for (0..head_k_dim) |dk| {
            const qv: f32 = @floatCast(q[kh * head_k_dim + dk]);
            const kv: f32 = @floatCast(k[kh * head_k_dim + dk]);
            q_buf[dk] = qv;
            k_buf[dk] = kv;
            q_sq += qv * qv;
            k_sq += kv * kv;
        }
        const q_fac: f32 = if (apply_l2_norm) (q_scale / @sqrt(q_sq + l2_eps)) else q_scale;
        const k_fac: f32 = if (apply_l2_norm) (1.0 / @sqrt(k_sq + l2_eps)) else 1.0;
        for (0..head_k_dim) |dk| {
            q_buf[dk] *= q_fac;
            k_buf[dk] *= k_fac;
        }

        const alpha = @exp(g[vh]);
        const beta_sig = sigmoidF32(beta[vh]);
        const s_base = vh * head_k_dim * head_v_dim;
        const v_base = vh * head_v_dim;

        for (0..head_v_dim) |dv| {
            var pred_v: f32 = 0.0;
            for (0..head_k_dim) |dk| {
                const s_dec = state_in[s_base + dk * head_v_dim + dv] * alpha;
                state_out[s_base + dk * head_v_dim + dv] = s_dec;
                pred_v += s_dec * k_buf[dk];
            }
            const v_val: f32 = @floatCast(v[v_base + dv]);
            const delta = beta_sig * (v_val - pred_v);
            var o_acc: f32 = 0.0;
            for (0..head_k_dim) |dk| {
                const s_new = state_out[s_base + dk * head_v_dim + dv] + k_buf[dk] * delta;
                state_out[s_base + dk * head_v_dim + dv] = s_new;
                o_acc += s_new * q_buf[dk];
            }
            o_buf[dv] = o_acc;
        }

        var inv_rms: f32 = 1.0;
        if (norm_w != null) {
            var sq: f32 = 0.0;
            for (0..head_v_dim) |dv| sq += o_buf[dv] * o_buf[dv];
            inv_rms = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(head_v_dim)) + rms_eps);
        }
        const nw_base = if (per_head_norm) vh * head_v_dim else 0;
        for (0..head_v_dim) |dv| {
            var val = o_buf[dv];
            if (norm_w) |nw| {
                val = (val * inv_rms) * quant.floatBitsToF32(norm_dtype, nw[nw_base + dv]);
            }
            if (z) |zv| {
                val *= siluF32(@floatCast(zv[v_base + dv]));
            }
            out_f32[v_base + dv] = val;
        }
    }
}

pub fn mropeInterleavedAxisCpuRef(j: u32, section: [3]u32) u32 {
    const m1 = @min(section[0], @min(section[1], section[2]));
    if (j < m1 * 3) return j % 3;
    const rem = j - m1 * 3;
    const a0 = section[0] - m1;
    const a1 = section[1] - m1;
    const a2 = section[2] - m1;
    var ax_a: u32 = 0;
    var cnt_a: u32 = 0;
    var ax_b: u32 = 0;
    var cnt_b: u32 = 0;
    if (a0 == 0) {
        ax_a = 1;
        cnt_a = a1;
        ax_b = 2;
        cnt_b = a2;
    } else if (a1 == 0) {
        ax_a = 0;
        cnt_a = a0;
        ax_b = 2;
        cnt_b = a2;
    } else {
        ax_a = 0;
        cnt_a = a0;
        ax_b = 1;
        cnt_b = a1;
    }
    const m2 = @min(cnt_a, cnt_b);
    if (rem < m2 * 2) return if ((rem & 1) == 0) ax_a else ax_b;
    return if (cnt_a > cnt_b) ax_a else ax_b;
}

pub fn yarnInvFreqCpuRef(j: u32, rotary_dim: u32, rope_theta: f32, yarn: YarnConfig) f32 {
    const log_theta = @log(rope_theta);
    const inv_freq_extrap = @exp(-log_theta * (2.0 * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(rotary_dim))));
    if (yarn.factor <= 1.0 or yarn.original_max_pos == 0) return inv_freq_extrap;
    const two_pi: f32 = 6.2831853071795864769;
    const half_rot: f32 = @floatFromInt(rotary_dim / 2);
    const f_orig: f32 = @floatFromInt(yarn.original_max_pos);
    const raw_low = (@as(f32, @floatFromInt(rotary_dim)) * @log(f_orig / (yarn.beta_fast * two_pi))) / (2.0 * log_theta);
    const raw_high = (@as(f32, @floatFromInt(rotary_dim)) * @log(f_orig / (yarn.beta_slow * two_pi))) / (2.0 * log_theta);
    const low = std.math.clamp(@floor(raw_low), 0.0, half_rot - 1.0);
    var high = std.math.clamp(@ceil(raw_high), 0.0, half_rot - 1.0);
    if (high <= low) high = low + 0.001;
    const ramp = std.math.clamp((@as(f32, @floatFromInt(j)) - low) / (high - low), 0.0, 1.0);
    const extrap_w = 1.0 - ramp;
    return inv_freq_extrap * extrap_w + (inv_freq_extrap / yarn.factor) * ramp;
}

pub fn qsaQkNormMropeHeadCpuRef(
    src: []const f16,
    norm_w: ?[]const u16,
    norm_dtype: quant.FloatDType,
    out: []f32,
    head_dim: u32,
    rotary_dim: u32,
    pos_3d: Pos3D,
    mrope: MropeSection,
    yarn: YarnConfig,
    rms_eps: f32,
    rope_theta: f32,
) void {
    var sq: f32 = 0.0;
    for (0..head_dim) |d| {
        const v: f32 = @floatCast(src[d]);
        out[d] = v;
        sq += v * v;
    }
    if (norm_w) |nw| {
        const inv_rms = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(head_dim)) + rms_eps);
        for (0..head_dim) |d| {
            out[d] = (out[d] * inv_rms) * quant.floatBitsToF32(norm_dtype, nw[d]);
        }
    }
    const half_rot = rotary_dim / 2;
    const mscale = yarn.effectiveMscale();
    for (0..half_rot) |j| {
        const axis = mropeInterleavedAxisCpuRef(@intCast(j), mrope.section);
        const axis_pos = switch (axis) {
            0 => pos_3d.t,
            1 => pos_3d.h,
            else => pos_3d.w,
        };
        const inv_f = yarnInvFreqCpuRef(@intCast(j), rotary_dim, rope_theta, yarn);
        const angle = @as(f32, @floatFromInt(axis_pos)) * inv_f;
        const c = @cos(angle) * mscale;
        const s = @sin(angle) * mscale;
        const x0 = out[j];
        const x1 = out[j + half_rot];
        out[j] = x0 * c - x1 * s;
        out[j + half_rot] = x1 * c + x0 * s;
    }
}

pub fn qsaIndexBlocksCpuRef(
    q: []const f16,
    block_keys: []const f16,
    selected_blocks_out: []u32,
    selected_counts_out: []u32,
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    window_size: u32,
    ring_capacity: u32,
    block_tokens: u32,
    top_k_blocks: u32,
    num_local_blocks: u32,
    scale: f32,
) void {
    const gqa_ratio = num_q_heads / num_kv_heads;
    const k_cap = @min(top_k_blocks, QSA_MAX_TOP_K_BLOCKS);
    const ring_blocks: u32 = if (ring_capacity > 0) ring_capacity / block_tokens else 0;

    for (0..num_nodes) |node| {
        for (0..num_kv_heads) |kvh| {
            const cnt_idx = node * num_kv_heads + kvh;
            const sel_base = cnt_idx * top_k_blocks;
            for (0..top_k_blocks) |i| selected_blocks_out[sel_base + i] = 0xFFFF_FFFF;

            if (kv_len == 0 or k_cap == 0) {
                selected_counts_out[cnt_idx] = 0;
                continue;
            }
            const win_start: u32 = if (window_size > 0 and kv_len > window_size) kv_len - window_size else 0;
            const first_block = win_start / block_tokens;
            const last_block = (kv_len - 1) / block_tokens;
            const num_valid = last_block - first_block + 1;
            if (num_valid <= k_cap) {
                for (0..num_valid) |i| selected_blocks_out[sel_base + i] = first_block + @as(u32, @intCast(i));
                selected_counts_out[cnt_idx] = num_valid;
                continue;
            }

            const local_cnt = @min(@max(num_local_blocks, 1), k_cap);
            const local_start = last_block + 1 - local_cnt;
            const k_sel = k_cap - local_cnt;

            var chosen: [QSA_MAX_TOP_K_BLOCKS]u32 = @splat(0xFFFF_FFFF);
            for (0..k_sel) |k| {
                var best_s: f32 = -std.math.inf(f32);
                var best_b: u32 = 0xFFFF_FFFF;
                var b = first_block;
                while (b < local_start) : (b += 1) {
                    var already = false;
                    for (0..k) |prev| {
                        if (chosen[prev] == b) {
                            already = true;
                            break;
                        }
                    }
                    if (already) continue;

                    const phys_b = if (ring_blocks > 0) b % ring_blocks else b;
                    const bk = block_keys[(phys_b * num_kv_heads + kvh) * head_dim ..][0..head_dim];
                    var head_max: f32 = -std.math.inf(f32);
                    for (0..gqa_ratio) |g| {
                        const qh = kvh * gqa_ratio + g;
                        const qv = q[(node * num_q_heads + qh) * head_dim ..][0..head_dim];
                        var d_acc: f32 = 0.0;
                        for (0..head_dim) |d| {
                            d_acc += @as(f32, @floatCast(qv[d])) * @as(f32, @floatCast(bk[d]));
                        }
                        head_max = @max(head_max, d_acc * scale);
                    }
                    if (head_max > best_s or (head_max == best_s and b < best_b)) {
                        best_s = head_max;
                        best_b = b;
                    }
                }
                chosen[k] = best_b;
                selected_blocks_out[sel_base + k] = best_b;
            }
            for (0..local_cnt) |li| {
                selected_blocks_out[sel_base + k_sel + li] = local_start + @as(u32, @intCast(li));
            }
            selected_counts_out[cnt_idx] = k_cap;
        }
    }
}

pub fn qsaSparseAttnCpuRef(
    q: []const f16,
    /// Full logical KV history `[kv_len, num_kv_heads, head_dim]` f16 (indexed by logical token `t`).
    k_hist: []const f16,
    v_hist: []const f16,
    selected_blocks: []const u32,
    selected_counts: []const u32,
    k_tree: ?[]const f16,
    v_tree: ?[]const f16,
    ancestor_masks: ?[]const u64,
    out_f32: []f32,
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    window_size: u32,
    block_tokens: u32,
    top_k_blocks: u32,
    scale: f32,
) void {
    const gqa_ratio = num_q_heads / num_kv_heads;
    const win_start: u32 = if (window_size > 0 and kv_len > window_size) kv_len - window_size else 0;

    for (0..num_nodes) |node| {
        for (0..num_q_heads) |qh| {
            const kvh = qh / gqa_ratio;
            const q_row = q[(node * num_q_heads + qh) * head_dim ..][0..head_dim];
            const out_row = out_f32[(node * num_q_heads + qh) * head_dim ..][0..head_dim];
            @memset(out_row, 0.0);

            var m_val: f32 = -std.math.inf(f32);
            var l_val: f32 = 0.0;

            if (kv_len > 0) {
                const cnt_idx = node * num_kv_heads + kvh;
                const sel_cnt = @min(selected_counts[cnt_idx], top_k_blocks);
                const sel_base = cnt_idx * top_k_blocks;
                for (0..sel_cnt) |bi| {
                    const b = selected_blocks[sel_base + bi];
                    if (b == 0xFFFF_FFFF) continue;
                    const t0 = @max(b * block_tokens, win_start);
                    const t1 = @min((b + 1) * block_tokens, kv_len);
                    var t = t0;
                    while (t < t1) : (t += 1) {
                        const k_row = k_hist[(t * num_kv_heads + kvh) * head_dim ..][0..head_dim];
                        const v_row = v_hist[(t * num_kv_heads + kvh) * head_dim ..][0..head_dim];
                        var dot_acc: f32 = 0.0;
                        for (0..head_dim) |d| dot_acc += @as(f32, @floatCast(q_row[d])) * @as(f32, @floatCast(k_row[d]));
                        const s = dot_acc * scale;
                        const new_m = @max(m_val, s);
                        const corr = @exp(m_val - new_m);
                        const p_w = @exp(s - new_m);
                        l_val = l_val * corr + p_w;
                        for (0..head_dim) |d| out_row[d] = out_row[d] * corr + @as(f32, @floatCast(v_row[d])) * p_w;
                        m_val = new_m;
                    }
                }
            }

            if (k_tree != null and v_tree != null and ancestor_masks != null) {
                const mask = ancestor_masks.?[node];
                for (0..num_nodes) |j| {
                    if ((mask & (@as(u64, 1) << @intCast(j))) == 0) continue;
                    const k_row = k_tree.?[(j * num_kv_heads + kvh) * head_dim ..][0..head_dim];
                    const v_row = v_tree.?[(j * num_kv_heads + kvh) * head_dim ..][0..head_dim];
                    var dot_acc: f32 = 0.0;
                    for (0..head_dim) |d| dot_acc += @as(f32, @floatCast(q_row[d])) * @as(f32, @floatCast(k_row[d]));
                    const s = dot_acc * scale;
                    const new_m = @max(m_val, s);
                    const corr = @exp(m_val - new_m);
                    const p_w = @exp(s - new_m);
                    l_val = l_val * corr + p_w;
                    for (0..head_dim) |d| out_row[d] = out_row[d] * corr + @as(f32, @floatCast(v_row[d])) * p_w;
                    m_val = new_m;
                }
            }

            if (l_val > 0.0) {
                for (0..head_dim) |d| out_row[d] /= l_val;
            }
        }
    }
}

// ===========================================================================
// Tests: Alignment, GDN (Prefill/Decode/Tree/S_tmp Isolation), QSA (YaRN,
// Interleaved MRoPE, 128-tok SWA Ring, Sparse Top-K, r>=1 Index Reuse), and
// 4-Branch Gated Residual (f16 & bf16) CPU FP32 Parity
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

test "GdnStateArena and GrStreamArena enforce 16 KiB page alignment and zero-init" {
    const gdn_spec: GdnStateSpec = .{
        .num_gdn_layers = 3,
        .num_v_heads = 4,
        .head_k_dim = 64,
        .head_v_dim = 64,
    };
    const gr_spec: GrStreamSpec = .{
        .max_tokens = 16,
        .hidden_dim = 192,
        .dtype = .bf16,
    };
    const gdn_bytes = try GdnStateArena.requiredBytes(gdn_spec);
    const gr_bytes = try GrStreamArena.requiredBytes(gr_spec);
    try std.testing.expect(std.mem.isAligned(gdn_bytes, page.APPLE_PAGE_SIZE));
    try std.testing.expect(std.mem.isAligned(gr_bytes, page.APPLE_PAGE_SIZE));

    var arena = try makeTestArena(gdn_bytes + gr_bytes + page.APPLE_PAGE_SIZE);
    defer arena.deinit();

    var gdn_arena = try GdnStateArena.init(&arena, gdn_spec);
    try page.checkShareable(gdn_arena.bytes().ptr, gdn_arena.bytes().len);
    const l0_comm = try gdn_arena.committedLayerSlice(0);
    const l0_scr = try gdn_arena.scratchLayerSlice(0);
    try std.testing.expectEqual(@as(usize, 4 * 64 * 64), l0_comm.len);
    try std.testing.expectEqual(@as(usize, 4 * 64 * 64), l0_scr.len);
    try std.testing.expect(@intFromPtr(l0_comm.ptr) != @intFromPtr(l0_scr.ptr));
    for (l0_comm) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
    try std.testing.expectError(error.InvalidLayerIndex, gdn_arena.committedLayerSlice(3));

    const gr_arena = try GrStreamArena.init(&arena, gr_spec);
    try page.checkShareable(gr_arena.backing.ptr, gr_arena.backing.len);
    for (gr_arena.backing) |b| try std.testing.expectEqual(@as(u8, 0), b);
}

test "R6 + G8 GDN decode, chunked prefill, and ADR-0019 S_tmp loop isolation match CPU fp32 ref" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const seq_len: u32 = 19; // spans 3 chunks (8 + 8 + 3)
    const num_k_heads: u32 = 2;
    const num_v_heads: u32 = 4;
    const head_k_dim: u32 = 64;
    const head_v_dim: u32 = 64;
    const q_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_k_dim)));

    const state_spec: GdnStateSpec = .{
        .num_gdn_layers = 2,
        .num_v_heads = num_v_heads,
        .head_k_dim = head_k_dim,
        .head_v_dim = head_v_dim,
    };
    const state_bytes = try GdnStateArena.requiredBytes(state_spec);
    var arena = try makeTestArena(state_bytes + 512 * 1024);
    defer arena.deinit();

    var gdn_arena = try GdnStateArena.init(&arena, state_spec);
    const q_elems = @as(usize, seq_len) * num_k_heads * head_k_dim;
    const v_elems = @as(usize, seq_len) * num_v_heads * head_v_dim;
    const gb_elems = @as(usize, seq_len) * num_v_heads;

    const q_raw = try arena.allocAligned(q_elems * @sizeOf(f16), 64);
    const k_raw = try arena.allocAligned(q_elems * @sizeOf(f16), 64);
    const v_raw = try arena.allocAligned(v_elems * @sizeOf(f16), 64);
    const z_raw = try arena.allocAligned(v_elems * @sizeOf(f16), 64);
    const g_raw = try arena.allocAligned(gb_elems * @sizeOf(f32), 64);
    const beta_raw = try arena.allocAligned(gb_elems * @sizeOf(f32), 64);
    const nw_raw = try arena.allocAligned(@as(usize, head_v_dim) * @sizeOf(u16), 64);
    const out_prefill_raw = try arena.allocAligned(v_elems * @sizeOf(f16), 64);
    const out_decode_raw = try arena.allocAligned(@as(usize, num_v_heads) * head_v_dim * @sizeOf(f16), 64);
    @memset(out_prefill_raw, 0);
    @memset(out_decode_raw, 0);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
    const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw));
    const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw));
    const z_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, z_raw));
    const g_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, g_raw));
    const beta_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, beta_raw));
    const nw_bf16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, nw_raw));

    for (q_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 11, 0.08));
    for (k_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 29, 0.08));
    for (v_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 53, 0.09));
    for (z_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 71, 0.07));
    for (g_f32, 0..) |*x, i| x.* = -0.05 - @abs(synthF32(i + 97, 0.04)); // negative log-decay => exp(g) in (0, 1]
    for (beta_f32, 0..) |*x, i| x.* = synthF32(i + 131, 0.08);
    for (nw_bf16, 0..) |*x, i| x.* = quant.f32ToBf16(0.85 + @abs(synthF32(i + 157, 0.02)));

    // Seed non-zero initial committed state on layer 0 to verify non-zero S_{t-1} handling.
    const l0_comm = try gdn_arena.committedLayerSlice(0);
    for (l0_comm, 0..) |*s, i| s.* = synthF32(i + 301, 0.02);

    const ref_state_init = try gpa.dupe(f32, l0_comm);
    defer gpa.free(ref_state_init);
    const ref_state = try gpa.dupe(f32, l0_comm);
    defer gpa.free(ref_state);
    const ref_out = try gpa.alloc(f32, v_elems);
    defer gpa.free(ref_out);

    for (0..seq_len) |t| {
        gdnStepCpuRef(
            q_f16[t * num_k_heads * head_k_dim ..][0 .. num_k_heads * head_k_dim],
            k_f16[t * num_k_heads * head_k_dim ..][0 .. num_k_heads * head_k_dim],
            v_f16[t * num_v_heads * head_v_dim ..][0 .. num_v_heads * head_v_dim],
            g_f32[t * num_v_heads ..][0..num_v_heads],
            beta_f32[t * num_v_heads ..][0..num_v_heads],
            z_f16[t * num_v_heads * head_v_dim ..][0 .. num_v_heads * head_v_dim],
            nw_bf16,
            .bf16,
            false,
            ref_state,
            ref_state,
            ref_out[t * num_v_heads * head_v_dim ..][0 .. num_v_heads * head_v_dim],
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            q_scale,
            1e-6,
            1e-6,
            true,
        );
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    // 1) Run chunked prefill in `.loop_isolated_scratch` mode (`r < R - 1`).
    // Committed `S_t` MUST stay bitwise identical to `ref_state_init`, while `S_tmp` matches `ref_state`.
    const iso_binds = try gdn_arena.stateBindings(gpu_buf, 0, .loop_isolated_scratch);
    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeGdnPrefill(&f, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) },
            .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) },
            .g = .{ .buffer = gpu_buf, .offset = arena.offsetOf(g_raw.ptr) },
            .beta = .{ .buffer = gpu_buf, .offset = arena.offsetOf(beta_raw.ptr) },
            .z = .{ .buffer = gpu_buf, .offset = arena.offsetOf(z_raw.ptr) },
            .norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(nw_raw.ptr) },
            .norm_dtype = .bf16,
            .state_in = iso_binds.in,
            .state_out = iso_binds.out,
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_prefill_raw.ptr) },
            .seq_len = seq_len,
            .num_k_heads = num_k_heads,
            .num_v_heads = num_v_heads,
            .head_k_dim = head_k_dim,
            .head_v_dim = head_v_dim,
        }, false);
        const sig = try f.commit();
        try compute.wait(sig, 5000);
    }

    // Committed state untouched!
    for (ref_state_init, l0_comm) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }
    // Scratch state matches step-by-step CPU fp32 reference!
    const l0_scr = try gdn_arena.scratchLayerSlice(0);
    for (ref_state, l0_scr) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
    }
    // Prefill outputs match CPU fp32 reference!
    const out_prefill: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_prefill_raw));
    for (ref_out, out_prefill) |expected, actual_f16| {
        const actual: f32 = @floatCast(actual_f16);
        try std.testing.expectApproxEqAbs(expected, actual, @max(0.015, @abs(expected) * 0.015));
    }

    // 2) Commit scratch `S_tmp -> S_t` on final pass `r == R - 1` via `encodeCommitScratchState`.
    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeCommitScratchState(&f, &gdn_arena, gpu_buf, 0, false);
        const sig = try f.commit();
        try compute.wait(sig, 5000);
    }
    for (ref_state, l0_comm) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
    }

    // 3) Run single-step decode (`M = 1`) on layer 1 step-by-step matching CPU fp32 ref.
    const l1_comm = try gdn_arena.committedLayerSlice(1);
    @memcpy(l1_comm, ref_state_init);
    @memcpy(ref_state, ref_state_init);
    const in_place_l1 = try gdn_arena.stateBindings(gpu_buf, 1, .commit_in_place);
    const out_decode: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_decode_raw));

    for (0..5) |t| {
        var step_ref: [num_v_heads * head_v_dim]f32 = undefined;
        gdnStepCpuRef(
            q_f16[t * num_k_heads * head_k_dim ..][0 .. num_k_heads * head_k_dim],
            k_f16[t * num_k_heads * head_k_dim ..][0 .. num_k_heads * head_k_dim],
            v_f16[t * num_v_heads * head_v_dim ..][0 .. num_v_heads * head_v_dim],
            g_f32[t * num_v_heads ..][0..num_v_heads],
            beta_f32[t * num_v_heads ..][0..num_v_heads],
            z_f16[t * num_v_heads * head_v_dim ..][0 .. num_v_heads * head_v_dim],
            nw_bf16,
            .bf16,
            false,
            ref_state,
            ref_state,
            &step_ref,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            q_scale,
            1e-6,
            1e-6,
            true,
        );

        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeGdnDecode(&f, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) + t * num_k_heads * head_k_dim * @sizeOf(f16) },
            .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) + t * num_k_heads * head_k_dim * @sizeOf(f16) },
            .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) + t * num_v_heads * head_v_dim * @sizeOf(f16) },
            .g = .{ .buffer = gpu_buf, .offset = arena.offsetOf(g_raw.ptr) + t * num_v_heads * @sizeOf(f32) },
            .beta = .{ .buffer = gpu_buf, .offset = arena.offsetOf(beta_raw.ptr) + t * num_v_heads * @sizeOf(f32) },
            .z = .{ .buffer = gpu_buf, .offset = arena.offsetOf(z_raw.ptr) + t * num_v_heads * head_v_dim * @sizeOf(f16) },
            .norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(nw_raw.ptr) },
            .norm_dtype = .bf16,
            .state_in = in_place_l1.in,
            .state_out = in_place_l1.out,
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_decode_raw.ptr) },
            .num_k_heads = num_k_heads,
            .num_v_heads = num_v_heads,
            .head_k_dim = head_k_dim,
            .head_v_dim = head_v_dim,
        }, false);
        const sig = try f.commit();
        try compute.wait(sig, 5000);

        for (step_ref, out_decode) |expected, actual_f16| {
            const actual: f32 = @floatCast(actual_f16);
            try std.testing.expectApproxEqAbs(expected, actual, @max(0.01, @abs(expected) * 0.01));
        }
        for (ref_state, l1_comm) |expected, actual| {
            try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
        }
    }
}

test "G8 GDN M-node speculative tree verify + commitTree matches CPU fp32 ref without mutating S_t" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    // 7-node branching tree:
    //   0 -> 1 -> 2 -> 3
    //   0 -> 1 -> 4
    //   0 -> 5 -> 6
    const num_nodes: u32 = 7;
    const masks_init = [num_nodes]u64{
        (1 << 0),
        (1 << 0) | (1 << 1),
        (1 << 0) | (1 << 1) | (1 << 2),
        (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),
        (1 << 0) | (1 << 1) | (1 << 4),
        (1 << 0) | (1 << 5),
        (1 << 0) | (1 << 5) | (1 << 6),
    };
    const paths = [num_nodes][]const u16{
        &.{0},
        &.{ 0, 1 },
        &.{ 0, 1, 2 },
        &.{ 0, 1, 2, 3 },
        &.{ 0, 1, 4 },
        &.{ 0, 5 },
        &.{ 0, 5, 6 },
    };

    const num_k_heads: u32 = 2;
    const num_v_heads: u32 = 2;
    const head_k_dim: u32 = 128;
    const head_v_dim: u32 = 64;
    const q_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_k_dim)));

    const state_spec: GdnStateSpec = .{
        .num_gdn_layers = 1,
        .num_v_heads = num_v_heads,
        .head_k_dim = head_k_dim,
        .head_v_dim = head_v_dim,
    };
    const state_bytes = try GdnStateArena.requiredBytes(state_spec);
    var arena = try makeTestArena(state_bytes + 512 * 1024);
    defer arena.deinit();

    var gdn_arena = try GdnStateArena.init(&arena, state_spec);
    const l0_comm = try gdn_arena.committedLayerSlice(0);
    for (l0_comm, 0..) |*s, i| s.* = synthF32(i + 503, 0.015);
    const orig_state = try gpa.dupe(f32, l0_comm);
    defer gpa.free(orig_state);

    const q_raw = try arena.allocAligned(@as(usize, num_nodes) * num_k_heads * head_k_dim * @sizeOf(f16), 64);
    const k_raw = try arena.allocAligned(@as(usize, num_nodes) * num_k_heads * head_k_dim * @sizeOf(f16), 64);
    const v_raw = try arena.allocAligned(@as(usize, num_nodes) * num_v_heads * head_v_dim * @sizeOf(f16), 64);
    const z_raw = try arena.allocAligned(@as(usize, num_nodes) * num_v_heads * head_v_dim * @sizeOf(f16), 64);
    const g_raw = try arena.allocAligned(@as(usize, num_nodes) * num_v_heads * @sizeOf(f32), 64);
    const beta_raw = try arena.allocAligned(@as(usize, num_nodes) * num_v_heads * @sizeOf(f32), 64);
    const masks_raw = try arena.allocAligned(@as(usize, num_nodes) * @sizeOf(u64), 64);
    const out_raw = try arena.allocAligned(@as(usize, num_nodes) * num_v_heads * head_v_dim * @sizeOf(f16), 64);
    @memset(out_raw, 0);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
    const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw));
    const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw));
    const z_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, z_raw));
    const g_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, g_raw));
    const beta_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, beta_raw));
    const masks_u64: []u64 = @alignCast(std.mem.bytesAsSlice(u64, masks_raw));
    @memcpy(masks_u64, &masks_init);

    for (q_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 17, 0.07));
    for (k_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 41, 0.07));
    for (v_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 67, 0.08));
    for (z_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 89, 0.06));
    for (g_f32, 0..) |*x, i| x.* = -0.04 - @abs(synthF32(i + 109, 0.03));
    for (beta_f32, 0..) |*x, i| x.* = synthF32(i + 149, 0.09);

    // CPU reference along every root-to-node branch.
    const ref_out = try gpa.alloc(f32, @as(usize, num_nodes) * num_v_heads * head_v_dim);
    defer gpa.free(ref_out);
    const leaf_state_ref = try gpa.alloc(f32, l0_comm.len);
    defer gpa.free(leaf_state_ref);
    const tmp_state = try gpa.alloc(f32, l0_comm.len);
    defer gpa.free(tmp_state);

    const accepted_leaf: u32 = 6; // branch 0 -> 5 -> 6
    for (0..num_nodes) |node| {
        @memcpy(tmp_state, orig_state);
        for (paths[node]) |anc| {
            const a: usize = anc;
            gdnStepCpuRef(
                q_f16[a * num_k_heads * head_k_dim ..][0 .. num_k_heads * head_k_dim],
                k_f16[a * num_k_heads * head_k_dim ..][0 .. num_k_heads * head_k_dim],
                v_f16[a * num_v_heads * head_v_dim ..][0 .. num_v_heads * head_v_dim],
                g_f32[a * num_v_heads ..][0..num_v_heads],
                beta_f32[a * num_v_heads ..][0..num_v_heads],
                z_f16[a * num_v_heads * head_v_dim ..][0 .. num_v_heads * head_v_dim],
                null,
                .f16,
                false,
                tmp_state,
                tmp_state,
                ref_out[node * num_v_heads * head_v_dim ..][0 .. num_v_heads * head_v_dim],
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                q_scale,
                1e-6,
                1e-6,
                true,
            );
        }
        if (node == accepted_leaf) @memcpy(leaf_state_ref, tmp_state);
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const iso = try gdn_arena.stateBindings(gpu_buf, 0, .loop_isolated_scratch);
    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeGdnTreeVerify(&f, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) },
            .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) },
            .g = .{ .buffer = gpu_buf, .offset = arena.offsetOf(g_raw.ptr) },
            .beta = .{ .buffer = gpu_buf, .offset = arena.offsetOf(beta_raw.ptr) },
            .z = .{ .buffer = gpu_buf, .offset = arena.offsetOf(z_raw.ptr) },
            .ancestor_masks = .{ .buffer = gpu_buf, .offset = arena.offsetOf(masks_raw.ptr) },
            .state_in = iso.in,
            .state_out = iso.out,
            .scratch_leaf_node = accepted_leaf,
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .num_nodes = num_nodes,
            .num_k_heads = num_k_heads,
            .num_v_heads = num_v_heads,
            .head_k_dim = head_k_dim,
            .head_v_dim = head_v_dim,
        }, false);
        const sig = try f.commit();
        try compute.wait(sig, 5000);
    }

    // Committed state must remain untouched during tree verify!
    for (orig_state, l0_comm) |expected, actual| try std.testing.expectEqual(expected, actual);
    // Scratch state captured node 6's leaf state!
    const l0_scr = try gdn_arena.scratchLayerSlice(0);
    for (leaf_state_ref, l0_scr) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
    // All 7 tree node outputs match CPU fp32 reference!
    const out_gpu: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
    for (ref_out, out_gpu) |expected, actual_f16| {
        const actual: f32 = @floatCast(actual_f16);
        try std.testing.expectApproxEqAbs(expected, actual, @max(0.01, @abs(expected) * 0.01));
    }

    // Now test `encodeCommitTreeState` committing accepted branch [0, 5, 6] directly into committed S_t!
    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeCommitTreeState(&f, .{
            .k_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) },
            .v_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) },
            .g_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(g_raw.ptr) },
            .beta_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(beta_raw.ptr) },
            .accepted_nodes = .{ .inline_u16 = paths[accepted_leaf] },
            .accepted_count = @intCast(paths[accepted_leaf].len),
            .state_in = iso.in,
            .state_out = iso.in,
            .num_k_heads = num_k_heads,
            .num_v_heads = num_v_heads,
            .head_k_dim = head_k_dim,
            .head_v_dim = head_v_dim,
        }, false);
        const sig = try f.commit();
        try compute.wait(sig, 5000);
    }
    for (leaf_state_ref, l0_comm) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
}

test "G9 QSA YaRN + interleaved 3D MRoPE [11,11,10] (text & multimodal) matches CPU fp32 ref" {
    // Verify interleaved axis pattern for Qwen3.8 `[11, 11, 10]` (half_rot = 32):
    // j = 0..29 -> T, H, W repeating; j = 30 -> T; j = 31 -> H.
    var counts: [3]u32 = .{ 0, 0, 0 };
    for (0..32) |j| {
        const ax = mropeInterleavedAxisCpuRef(@intCast(j), MropeSection.QWEN38_DEFAULT.section);
        counts[ax] += 1;
        if (j < 30) try std.testing.expectEqual(@as(u32, @intCast(j % 3)), ax);
    }
    try std.testing.expectEqual([3]u32{ 11, 11, 10 }, counts);
    try std.testing.expectEqual(@as(u32, 0), mropeInterleavedAxisCpuRef(30, MropeSection.QWEN38_DEFAULT.section));
    try std.testing.expectEqual(@as(u32, 1), mropeInterleavedAxisCpuRef(31, MropeSection.QWEN38_DEFAULT.section));

    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const num_nodes: u32 = 4;
    const num_q_heads: u32 = 4;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 128;
    const rotary_dim: u32 = 64; // Partial YaRN + [11, 11, 10] MRoPE on first 64 dims, dims 64..128 pass through!
    const yarn: YarnConfig = .{
        .factor = 4.0,
        .mscale = null, // defaultMscale(4.0)
        .beta_fast = 32.0,
        .beta_slow = 1.0,
        .original_max_pos = 32768,
    };

    var arena = try makeTestArena(256 * 1024);
    defer arena.deinit();

    const q_raw = try arena.allocAligned(@as(usize, num_nodes) * num_q_heads * head_dim * @sizeOf(f16), 64);
    const k_raw = try arena.allocAligned(@as(usize, num_nodes) * num_kv_heads * head_dim * @sizeOf(f16), 64);
    const v_raw = try arena.allocAligned(@as(usize, num_nodes) * num_kv_heads * head_dim * @sizeOf(f16), 64);
    const qn_raw = try arena.allocAligned(@as(usize, head_dim) * @sizeOf(u16), 64);
    const kn_raw = try arena.allocAligned(@as(usize, head_dim) * @sizeOf(u16), 64);
    const pos_raw = try arena.allocAligned(@as(usize, num_nodes) * @sizeOf(Pos3D), 64);
    const k_out_raw = try arena.allocAligned(@as(usize, num_nodes) * num_kv_heads * head_dim * @sizeOf(f16), 64);
    const v_out_raw = try arena.allocAligned(@as(usize, num_nodes) * num_kv_heads * head_dim * @sizeOf(f16), 64);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
    const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw));
    const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw));
    const qn_bf16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, qn_raw));
    const kn_bf16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, kn_raw));
    const pos_3d: []Pos3D = @alignCast(std.mem.bytesAsSlice(Pos3D, pos_raw));

    for (q_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 13, 0.08));
    for (k_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 37, 0.08));
    for (v_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 61, 0.08));
    for (qn_bf16, 0..) |*x, i| x.* = quant.f32ToBf16(0.9 + @abs(synthF32(i + 83, 0.02)));
    for (kn_bf16, 0..) |*x, i| x.* = quant.f32ToBf16(0.8 + @abs(synthF32(i + 101, 0.02)));
    // Mix text tokens (t=h=w) and multimodal 3D positions (t, h, w).
    pos_3d[0] = Pos3D.text(42);
    pos_3d[1] = .{ .t = 120, .h = 7, .w = 19 };
    pos_3d[2] = .{ .t = 4096, .h = 15, .w = 3 };
    pos_3d[3] = Pos3D.text(50000);

    // Compute CPU fp32 reference before in-place Q update.
    var q_ref: [num_nodes * num_q_heads * head_dim]f32 = undefined;
    var k_ref: [num_nodes * num_kv_heads * head_dim]f32 = undefined;
    for (0..num_nodes) |n| {
        for (0..num_q_heads) |qh| {
            const off = (n * num_q_heads + qh) * head_dim;
            qsaQkNormMropeHeadCpuRef(
                q_f16[off..][0..head_dim],
                qn_bf16,
                .bf16,
                q_ref[off..][0..head_dim],
                head_dim,
                rotary_dim,
                pos_3d[n],
                MropeSection.QWEN38_DEFAULT,
                yarn,
                1e-6,
                1_000_000.0,
            );
        }
        for (0..num_kv_heads) |kh| {
            const off = (n * num_kv_heads + kh) * head_dim;
            qsaQkNormMropeHeadCpuRef(
                k_f16[off..][0..head_dim],
                kn_bf16,
                .bf16,
                k_ref[off..][0..head_dim],
                head_dim,
                rotary_dim,
                pos_3d[n],
                MropeSection.QWEN38_DEFAULT,
                yarn,
                1e-6,
                1_000_000.0,
            );
        }
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    var f = try compute.beginFrame(5000);
    errdefer f.abort();
    try pipes.encodeQsaTreeRope(&f, .{
        .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
        .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) },
        .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) },
        .q_norm = .{ .buffer = gpu_buf, .offset = arena.offsetOf(qn_raw.ptr) },
        .k_norm = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kn_raw.ptr) },
        .norm_dtype = .bf16,
        .positions_3d = .{ .buffer = gpu_buf, .offset = arena.offsetOf(pos_raw.ptr) },
        .k_tree_out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_out_raw.ptr) },
        .v_tree_out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_out_raw.ptr) },
        .num_nodes = num_nodes,
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .rotary_dim = rotary_dim,
        .mrope = MropeSection.QWEN38_DEFAULT,
        .yarn = yarn,
    }, false);
    const sig = try f.commit();
    try compute.wait(sig, 5000);

    const k_out: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, k_out_raw));
    const v_out: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, v_out_raw));
    for (q_ref, q_f16) |exp_v, act_h| {
        try std.testing.expectApproxEqAbs(exp_v, @as(f32, @floatCast(act_h)), @max(0.015, @abs(exp_v) * 0.015));
    }
    for (k_ref, k_out) |exp_v, act_h| {
        try std.testing.expectApproxEqAbs(exp_v, @as(f32, @floatCast(act_h)), @max(0.015, @abs(exp_v) * 0.015));
    }
    for (v_f16, v_out) |exp_h, act_h| try std.testing.expectEqual(exp_h, act_h);
}

test "G9 + R5 QSA 128-tok circular SWA tile wrap-around + sparse attention matches CPU fp32 ref" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const num_q_heads: u32 = 8;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 64;
    const seq_len: u32 = 175; // exceeds 128 -> wraps circularly around single 128-token SWA tile!
    const block_tokens: u32 = 32;
    const top_k_blocks: u32 = 3; // 2 top-scoring earlier blocks + 1 mandatory local recent block
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    var arena = try makeTestArena(512 * 1024);
    defer arena.deinit();

    var ring = try HybridKvRing.init(gpa, device, &arena, .{
        .mode = .sliding_window_128,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .num_tiles = 1,
    });
    defer ring.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 128), ring.chunk_tokens);
    try std.testing.expectEqual(@as(u32, 128), ring.window_size);
    try std.testing.expectEqual(@as(u32, 128), ring.ring_capacity);

    const idx_buf = try QsaIndexBuffer.init(&arena, .{
        .max_nodes = 4,
        .num_kv_heads = num_kv_heads,
        .top_k_blocks = top_k_blocks,
    });

    const ring_blocks: u32 = ring.ring_capacity / block_tokens; // 4
    const bk_raw = try arena.allocAligned(@as(usize, ring_blocks) * num_kv_heads * head_dim * @sizeOf(f16), 64);
    const q_raw = try arena.allocAligned(@as(usize, num_q_heads) * head_dim * @sizeOf(f16), 64);
    const k_step_raw = try arena.allocAligned(@as(usize, num_kv_heads) * head_dim * @sizeOf(f16), 64);
    const v_step_raw = try arena.allocAligned(@as(usize, num_kv_heads) * head_dim * @sizeOf(f16), 64);
    const out_raw = try arena.allocAligned(@as(usize, num_q_heads) * head_dim * @sizeOf(f16), 64);
    @memset(bk_raw, 0);
    @memset(out_raw, 0);

    // Populate logical history [0..seq_len) and write circularly into the 128-token IOSurface KvTile.
    const k_hist = try gpa.alloc(f16, @as(usize, seq_len) * num_kv_heads * head_dim);
    defer gpa.free(k_hist);
    const v_hist = try gpa.alloc(f16, @as(usize, seq_len) * num_kv_heads * head_dim);
    defer gpa.free(v_hist);
    for (k_hist, 0..) |*x, i| x.* = @floatCast(synthF32(i + 211, 0.08));
    for (v_hist, 0..) |*x, i| x.* = @floatCast(synthF32(i + 409, 0.08));

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
    for (q_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 613, 0.09));

    // Write rotated/raw K,V into the single 128-tok circular KvTile via IOSurface lock (tokens 0..173),
    // and write the final token 174 via GPU `encodeQsaRopeKvWrite` to test GPU circular write too!
    {
        var kg = try ring.k_tiles[0].lock(false);
        errdefer kg.release();
        var vg = try ring.v_tiles[0].lock(false);
        errdefer vg.release();
        const k_stride_f16 = kg.bytes_per_row / @sizeOf(f16);
        const v_stride_f16 = vg.bytes_per_row / @sizeOf(f16);
        const k_dst: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
        const v_dst: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vg.bytes));

        for (0..seq_len - 1) |t| {
            const slot = try ring.slotForPos(@intCast(t));
            for (0..num_kv_heads) |kvh| {
                const row = kvh * ring.chunk_tokens + slot.slot_in_chunk;
                const src_off = (t * num_kv_heads + kvh) * head_dim;
                @memcpy(k_dst[row * k_stride_f16 ..][0..head_dim], k_hist[src_off..][0..head_dim]);
                @memcpy(v_dst[row * v_stride_f16 ..][0..head_dim], v_hist[src_off..][0..head_dim]);
            }
        }
        try vg.unlock();
        try kg.unlock();
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try ring.makeResident(compute);

    // Write final token `pos = 174` (`174 % 128 = 46`) via `encodeQsaRopeKvWrite` with pos=0 (identity RoPE) so k_hist matches verbatim.
    const k_step_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_step_raw));
    const v_step_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_step_raw));
    const last_off = (@as(usize, seq_len - 1) * num_kv_heads) * head_dim;
    @memcpy(k_step_f16, k_hist[last_off..][0 .. num_kv_heads * head_dim]);
    @memcpy(v_step_f16, v_hist[last_off..][0 .. num_kv_heads * head_dim]);
    const last_slot = try ring.slotForPos(seq_len - 1);
    try std.testing.expectEqual(@as(u32, 46), last_slot.slot_in_chunk);

    // Copy q_f16 before `encodeQsaRopeKvWrite` (pos=0 keeps q unchanged when yarn.factor=1.0).
    const win_start = seq_len - ring.window_size; // 47
    const first_block = win_start / block_tokens; // 1
    const last_block = (seq_len - 1) / block_tokens; // 5
    const num_valid_blocks = last_block - first_block + 1; // 5 (blocks 1,2,3,4,5)

    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeQsaRopeKvWrite(&f, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_step_raw.ptr) },
            .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_step_raw.ptr) },
            .k_tile = ring.k_tiles[last_slot.tile_idx],
            .v_tile = ring.v_tiles[last_slot.tile_idx],
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .rotary_dim = head_dim,
            .pos_3d = Pos3D.text(0),
            .mrope = MropeSection.text1d(head_dim),
            .chunk_tokens = ring.chunk_tokens,
            .slot_in_chunk = last_slot.slot_in_chunk,
        }, false);
        try pipes.encodeQsaUpdateBlockSummary(&f, .{
            .kv_tiles = ring.tileTableBinding(gpu_buf),
            .block_keys = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bk_raw.ptr) },
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .kv_len = seq_len,
            .window_size = ring.window_size,
            .ring_capacity = ring.ring_capacity,
            .chunk_tokens = ring.chunk_tokens,
            .block_tokens = block_tokens,
            .start_block = first_block,
            .num_blocks = num_valid_blocks,
        }, true);
        try pipes.encodeQsaAttention(&f, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .kv_tiles = ring.tileTableBinding(gpu_buf),
            .block_keys = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bk_raw.ptr) },
            .index_buf = &idx_buf,
            .index_metal_buf = gpu_buf,
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .num_nodes = 1,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .kv_len = seq_len,
            .window_size = ring.window_size,
            .ring_capacity = ring.ring_capacity,
            .chunk_tokens = ring.chunk_tokens,
            .block_tokens = block_tokens,
            .top_k_blocks = top_k_blocks,
            .num_local_blocks = 1,
            .scale = scale,
            .reuse_block_indices = false,
        }, true);
        const sig = try f.commit();
        try compute.wait(sig, 5000);
    }

    // CPU reference using the exact GPU-updated block_keys and logical window [47..175).
    const bk_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, bk_raw));
    var ref_sel_blocks: [num_kv_heads * top_k_blocks]u32 = undefined;
    var ref_sel_counts: [num_kv_heads]u32 = undefined;
    qsaIndexBlocksCpuRef(
        q_f16,
        bk_f16,
        &ref_sel_blocks,
        &ref_sel_counts,
        1,
        num_q_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        ring.window_size,
        ring.ring_capacity,
        block_tokens,
        top_k_blocks,
        1,
        scale,
    );
    for (0..num_kv_heads) |kvh| {
        try std.testing.expectEqual(ref_sel_counts[kvh], idx_buf.selected_counts[kvh]);
        // Mandatory local recent block `last_block = 5` must be selected!
        try std.testing.expectEqual(last_block, idx_buf.selected_blocks[kvh * top_k_blocks + top_k_blocks - 1]);
        for (0..top_k_blocks) |i| {
            try std.testing.expectEqual(ref_sel_blocks[kvh * top_k_blocks + i], idx_buf.selected_blocks[kvh * top_k_blocks + i]);
        }
    }

    var ref_out: [num_q_heads * head_dim]f32 = undefined;
    qsaSparseAttnCpuRef(
        q_f16,
        k_hist,
        v_hist,
        &ref_sel_blocks,
        &ref_sel_counts,
        null,
        null,
        null,
        &ref_out,
        1,
        num_q_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        ring.window_size,
        block_tokens,
        top_k_blocks,
        scale,
    );
    const out_gpu: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
    for (ref_out, out_gpu) |exp_v, act_h| {
        try std.testing.expectApproxEqAbs(exp_v, @as(f32, @floatCast(act_h)), @max(0.01, @abs(exp_v) * 0.01));
    }
}

test "G9 QSA 512-tok multi-chunk sparse top-K + 64-bit tree verify + ADR-0019 r>=1 index reuse" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const num_nodes: u32 = 6;
    const num_q_heads: u32 = 8;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 128;
    const base_pos: u32 = 700; // spans two 512-token full chunks (blocks 0..21)
    const block_tokens: u32 = 32;
    const num_total_blocks: u32 = (base_pos + block_tokens - 1) / block_tokens; // 22
    const top_k_blocks: u32 = 6; // 5 top-scoring + 1 mandatory local block (21)
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const masks_init = [num_nodes]u64{
        (1 << 0),
        (1 << 0) | (1 << 1),
        (1 << 0) | (1 << 1) | (1 << 2),
        (1 << 0) | (1 << 3),
        (1 << 0) | (1 << 3) | (1 << 4),
        (1 << 0) | (1 << 3) | (1 << 5),
    };

    var arena = try makeTestArena(2 * 1024 * 1024);
    defer arena.deinit();

    var ring = try HybridKvRing.init(gpa, device, &arena, .{
        .mode = .full_chunk_512,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .num_tiles = 2,
    });
    defer ring.deinit(gpa);

    const idx_buf = try QsaIndexBuffer.init(&arena, .{
        .max_nodes = num_nodes,
        .num_kv_heads = num_kv_heads,
        .top_k_blocks = top_k_blocks,
    });

    const bk_raw = try arena.allocAligned(@as(usize, num_total_blocks) * num_kv_heads * head_dim * @sizeOf(f16), 64);
    const q_raw = try arena.allocAligned(@as(usize, num_nodes) * num_q_heads * head_dim * @sizeOf(f16), 64);
    const kt_raw = try arena.allocAligned(@as(usize, num_nodes) * num_kv_heads * head_dim * @sizeOf(f16), 64);
    const vt_raw = try arena.allocAligned(@as(usize, num_nodes) * num_kv_heads * head_dim * @sizeOf(f16), 64);
    const masks_raw = try arena.allocAligned(@as(usize, num_nodes) * @sizeOf(u64), 64);
    const out_r0_raw = try arena.allocAligned(@as(usize, num_nodes) * num_q_heads * head_dim * @sizeOf(f16), 64);
    const out_r1_raw = try arena.allocAligned(@as(usize, num_nodes) * num_q_heads * head_dim * @sizeOf(f16), 64);
    @memset(out_r0_raw, 0);
    @memset(out_r1_raw, 0);

    const k_hist = try gpa.alloc(f16, @as(usize, base_pos) * num_kv_heads * head_dim);
    defer gpa.free(k_hist);
    const v_hist = try gpa.alloc(f16, @as(usize, base_pos) * num_kv_heads * head_dim);
    defer gpa.free(v_hist);
    for (k_hist, 0..) |*x, i| x.* = @floatCast(synthF32(i + 701, 0.07));
    for (v_hist, 0..) |*x, i| x.* = @floatCast(synthF32(i + 907, 0.07));

    // Populate the two 512-token IOSurface RGBA16Float KvTiles.
    for (0..2) |chunk_idx| {
        var kg = try ring.k_tiles[chunk_idx].lock(false);
        errdefer kg.release();
        var vg = try ring.v_tiles[chunk_idx].lock(false);
        errdefer vg.release();
        const ks = kg.bytes_per_row / @sizeOf(f16);
        const vs = vg.bytes_per_row / @sizeOf(f16);
        const k_dst: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
        const v_dst: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vg.bytes));
        const t_begin = chunk_idx * 512;
        const t_end = @min(t_begin + 512, base_pos);
        for (t_begin..t_end) |t| {
            const slot = t - t_begin;
            for (0..num_kv_heads) |kvh| {
                const row = kvh * 512 + slot;
                const src = (t * num_kv_heads + kvh) * head_dim;
                @memcpy(k_dst[row * ks ..][0..head_dim], k_hist[src..][0..head_dim]);
                @memcpy(v_dst[row * vs ..][0..head_dim], v_hist[src..][0..head_dim]);
            }
        }
        try vg.unlock();
        try kg.unlock();
    }

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
    const kt_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kt_raw));
    const vt_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vt_raw));
    const masks_u64: []u64 = @alignCast(std.mem.bytesAsSlice(u64, masks_raw));
    for (q_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 1103, 0.08));
    for (kt_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 1301, 0.08));
    for (vt_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 1511, 0.08));
    @memcpy(masks_u64, &masks_init);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try ring.makeResident(compute);

    // Loop pass r = 0: compute summaries + run indexer (`reuse_block_indices = false`) + sparse tree attention.
    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeQsaUpdateBlockSummary(&f, .{
            .kv_tiles = ring.tileTableBinding(gpu_buf),
            .block_keys = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bk_raw.ptr) },
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .kv_len = base_pos,
            .chunk_tokens = ring.chunk_tokens,
            .block_tokens = block_tokens,
            .start_block = 0,
            .num_blocks = num_total_blocks,
        }, false);
        try pipes.encodeQsaAttention(&f, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .kv_tiles = ring.tileTableBinding(gpu_buf),
            .block_keys = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bk_raw.ptr) },
            .index_buf = &idx_buf,
            .index_metal_buf = gpu_buf,
            .k_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kt_raw.ptr) },
            .v_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(vt_raw.ptr) },
            .ancestor_masks = .{ .buffer = gpu_buf, .offset = arena.offsetOf(masks_raw.ptr) },
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_r0_raw.ptr) },
            .num_nodes = num_nodes,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .kv_len = base_pos,
            .chunk_tokens = ring.chunk_tokens,
            .block_tokens = block_tokens,
            .top_k_blocks = top_k_blocks,
            .num_local_blocks = 1,
            .scale = scale,
            .reuse_block_indices = false,
        }, true);
        const sig = try f.commit();
        try compute.wait(sig, 5000);
    }

    // Verify pass r = 0 matches CPU reference (indexer + sparse tree attention).
    const bk_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, bk_raw));
    var ref_sel_blocks: [num_nodes * num_kv_heads * top_k_blocks]u32 = undefined;
    var ref_sel_counts: [num_nodes * num_kv_heads]u32 = undefined;
    qsaIndexBlocksCpuRef(
        q_f16,
        bk_f16,
        &ref_sel_blocks,
        &ref_sel_counts,
        num_nodes,
        num_q_heads,
        num_kv_heads,
        head_dim,
        base_pos,
        0,
        0,
        block_tokens,
        top_k_blocks,
        1,
        scale,
    );
    try std.testing.expectEqualSlices(u32, &ref_sel_counts, idx_buf.selected_counts);
    try std.testing.expectEqualSlices(u32, &ref_sel_blocks, idx_buf.selected_blocks);
    // Verify mandatory local recent block (21) is present for every `(node, kvh)`.
    for (0..num_nodes * num_kv_heads) |nh| {
        try std.testing.expectEqual(@as(u32, num_total_blocks - 1), idx_buf.selected_blocks[nh * top_k_blocks + top_k_blocks - 1]);
    }

    const ref_out = try gpa.alloc(f32, @as(usize, num_nodes) * num_q_heads * head_dim);
    defer gpa.free(ref_out);
    qsaSparseAttnCpuRef(
        q_f16,
        k_hist,
        v_hist,
        &ref_sel_blocks,
        &ref_sel_counts,
        kt_f16,
        vt_f16,
        masks_u64,
        ref_out,
        num_nodes,
        num_q_heads,
        num_kv_heads,
        head_dim,
        base_pos,
        0,
        block_tokens,
        top_k_blocks,
        scale,
    );
    const out_r0: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_r0_raw));
    for (ref_out, out_r0) |exp_v, act_h| {
        try std.testing.expectApproxEqAbs(exp_v, @as(f32, @floatCast(act_h)), @max(0.01, @abs(exp_v) * 0.01));
    }

    // ADR-0019 Cross-loop index reuse test (r = 1):
    // Deliberately poison `block_keys` with huge negative values; with `reuse_block_indices = true`,
    // `ami_qsa_index_blocks` is NOT dispatched, so `idx_buf` stays unchanged and `out_r1 == out_r0` bit-for-bit!
    for (bk_f16) |*x| x.* = @floatCast(@as(f32, -1000.0));
    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeQsaAttention(&f, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .kv_tiles = ring.tileTableBinding(gpu_buf),
            .block_keys = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bk_raw.ptr) },
            .index_buf = &idx_buf,
            .index_metal_buf = gpu_buf,
            .k_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kt_raw.ptr) },
            .v_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(vt_raw.ptr) },
            .ancestor_masks = .{ .buffer = gpu_buf, .offset = arena.offsetOf(masks_raw.ptr) },
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_r1_raw.ptr) },
            .num_nodes = num_nodes,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .kv_len = base_pos,
            .chunk_tokens = ring.chunk_tokens,
            .block_tokens = block_tokens,
            .top_k_blocks = top_k_blocks,
            .num_local_blocks = 1,
            .scale = scale,
            .reuse_block_indices = true,
        }, false);
        const sig = try f.commit();
        try compute.wait(sig, 5000);
    }
    try std.testing.expectEqualSlices(u32, &ref_sel_blocks, idx_buf.selected_blocks);
    const out_r1: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_r1_raw));
    try std.testing.expectEqualSlices(f16, out_r0, out_r1);

    // Also test `encodeQsaKvTreeCommit` crossing a chunk boundary (commit 3 nodes starting at pos 510 -> 510, 511, 512).
    const commit_nodes = [_]u16{ 0, 3, 5 };
    const commit_tiles = try ring.commitTilesForSpan(510, 3);
    try std.testing.expect(commit_tiles.k_tile1 != null);
    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeQsaKvTreeCommit(&f, .{
            .k_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kt_raw.ptr) },
            .v_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(vt_raw.ptr) },
            .accepted_nodes = .{ .inline_u16 = &commit_nodes },
            .accepted_count = 3,
            .k_tile0 = commit_tiles.k_tile0,
            .v_tile0 = commit_tiles.v_tile0,
            .k_tile1 = commit_tiles.k_tile1,
            .v_tile1 = commit_tiles.v_tile1,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .base_pos = 510,
            .chunk_tokens = ring.chunk_tokens,
            .ring_capacity = ring.ring_capacity,
        }, false);
        const sig = try f.commit();
        try compute.wait(sig, 5000);
    }
    // Verify slot 511 of tile0 matches node 3 and slot 0 of tile1 matches node 5!
    {
        var kg0 = try ring.k_tiles[0].lock(true);
        errdefer kg0.release();
        var kg1 = try ring.k_tiles[1].lock(true);
        errdefer kg1.release();
        const s0 = kg0.bytes_per_row / @sizeOf(f16);
        const s1 = kg1.bytes_per_row / @sizeOf(f16);
        const k0: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, kg0.bytes));
        const k1: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, kg1.bytes));
        for (0..num_kv_heads) |kvh| {
            const exp_n3 = kt_f16[(3 * num_kv_heads + kvh) * head_dim ..][0..head_dim];
            const exp_n5 = kt_f16[(5 * num_kv_heads + kvh) * head_dim ..][0..head_dim];
            try std.testing.expectEqualSlices(f16, exp_n3, k0[(kvh * 512 + 511) * s0 ..][0..head_dim]);
            try std.testing.expectEqualSlices(f16, exp_n5, k1[(kvh * 512 + 0) * s1 ..][0..head_dim]);
        }
        try kg1.unlock();
        try kg0.unlock();
    }
}

fn verifyGrRoundtrip(stream_dtype: quant.FloatDType, act_dtype: quant.FloatDType, m: u32, max_tokens: u32, hidden_dim: u32) !void {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const gr_spec: GrStreamSpec = .{
        .max_tokens = max_tokens,
        .hidden_dim = hidden_dim,
        .dtype = stream_dtype,
    };
    const stream_bytes = try GrStreamArena.requiredBytes(gr_spec);
    var arena = try makeTestArena(stream_bytes + 256 * 1024);
    defer arena.deinit();

    const gr_stream = try GrStreamArena.init(&arena, gr_spec);
    const act_bytes: usize = @as(usize, m) * hidden_dim * 2;
    const gate_bytes: usize = @as(usize, m) * GR_NUM_BRANCHES * @sizeOf(f32);

    const x0_raw = try arena.allocAligned(act_bytes, 64);
    const x_in_raw = try arena.allocAligned(act_bytes, 64);
    const y_delta_raw = try arena.allocAligned(act_bytes, 64);
    const out_raw = try arena.allocAligned(act_bytes, 64);
    const g_read_raw = try arena.allocAligned(gate_bytes, 64);
    const decay_raw = try arena.allocAligned(gate_bytes, 64);
    const g_write_raw = try arena.allocAligned(gate_bytes, 64);
    const collapse_w_raw = try arena.allocAligned(gate_bytes, 64);

    const x0_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, x0_raw));
    const yd_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, y_delta_raw));
    const g_read: []f32 = @alignCast(std.mem.bytesAsSlice(f32, g_read_raw));
    const decay: []f32 = @alignCast(std.mem.bytesAsSlice(f32, decay_raw));
    const g_write: []f32 = @alignCast(std.mem.bytesAsSlice(f32, g_write_raw));
    const collapse_w: []f32 = @alignCast(std.mem.bytesAsSlice(f32, collapse_w_raw));

    for (x0_u16, 0..) |*v, i| v.* = quant.f32ToFloatBits(act_dtype, synthF32(i + 19, 0.08));
    for (yd_u16, 0..) |*v, i| v.* = quant.f32ToFloatBits(act_dtype, synthF32(i + 47, 0.07));
    for (g_read, 0..) |*v, i| v.* = 0.25 + synthF32(i + 73, 0.03);
    for (decay, 0..) |*v, i| v.* = 0.8 + synthF32(i + 103, 0.015);
    for (g_write, 0..) |*v, i| v.* = 0.3 + synthF32(i + 137, 0.02);
    for (collapse_w, 0..) |*v, i| v.* = 0.25 + synthF32(i + 163, 0.025);

    // CPU fp32 reference through: init_stream0 -> write_gate (layer 0) -> read_gate -> write_gate (layer 1) -> collapse.
    const ref_stream = try gpa.alloc(f32, GR_NUM_BRANCHES * @as(usize, m) * hidden_dim);
    defer gpa.free(ref_stream);
    @memset(ref_stream, 0.0);
    for (0..@as(usize, m) * hidden_dim) |i| {
        ref_stream[i] = quant.floatBitsToF32(stream_dtype, quant.f32ToFloatBits(stream_dtype, quant.floatBitsToF32(act_dtype, x0_u16[i])));
    }
    const ref_x_in = try gpa.alloc(f32, @as(usize, m) * hidden_dim);
    defer gpa.free(ref_x_in);
    const ref_out = try gpa.alloc(f32, @as(usize, m) * hidden_dim);
    defer gpa.free(ref_out);

    // Step 1: write_gate (populates all 4 branches from stream-0 + y_delta).
    for (0..m) |tok| {
        for (0..hidden_dim) |d| {
            const yd = quant.floatBitsToF32(act_dtype, yd_u16[tok * hidden_dim + d]);
            for (0..GR_NUM_BRANCHES) |b| {
                const idx = (b * m + tok) * hidden_dim + d;
                const v = decay[tok * 4 + b] * ref_stream[idx] + g_write[tok * 4 + b] * yd;
                ref_stream[idx] = quant.floatBitsToF32(stream_dtype, quant.f32ToFloatBits(stream_dtype, v));
            }
        }
    }
    // Step 2: read_gate.
    for (0..m) |tok| {
        for (0..hidden_dim) |d| {
            var acc: f32 = 0.0;
            for (0..GR_NUM_BRANCHES) |b| {
                acc += g_read[tok * 4 + b] * ref_stream[(b * m + tok) * hidden_dim + d];
            }
            ref_x_in[tok * hidden_dim + d] = acc;
        }
    }
    // Step 3: weighted_per_token collapse.
    for (0..m) |tok| {
        for (0..hidden_dim) |d| {
            var acc: f32 = 0.0;
            for (0..GR_NUM_BRANCHES) |b| {
                acc += collapse_w[tok * 4 + b] * ref_stream[(b * m + tok) * hidden_dim + d];
            }
            ref_out[tok * hidden_dim + d] = acc;
        }
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeGrInitStream0(&f, .{
            .x0 = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x0_raw.ptr) },
            .stream = gr_stream.binding(gpu_buf),
            .m = m,
            .hidden_dim = hidden_dim,
            .branch_token_stride = max_tokens,
            .stream_dtype = stream_dtype,
            .act_dtype = act_dtype,
        }, false);
        try pipes.encodeGrWriteGate(&f, .{
            .stream = gr_stream.binding(gpu_buf),
            .y_delta = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_delta_raw.ptr) },
            .decay = .{ .buffer = gpu_buf, .offset = arena.offsetOf(decay_raw.ptr) },
            .g_write = .{ .buffer = gpu_buf, .offset = arena.offsetOf(g_write_raw.ptr) },
            .m = m,
            .hidden_dim = hidden_dim,
            .branch_token_stride = max_tokens,
            .stream_dtype = stream_dtype,
            .act_dtype = act_dtype,
        }, true);
        try pipes.encodeGrReadGate(&f, .{
            .stream = gr_stream.binding(gpu_buf),
            .g_read = .{ .buffer = gpu_buf, .offset = arena.offsetOf(g_read_raw.ptr) },
            .x_in = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_in_raw.ptr) },
            .m = m,
            .hidden_dim = hidden_dim,
            .branch_token_stride = max_tokens,
            .stream_dtype = stream_dtype,
            .act_dtype = act_dtype,
        }, true);
        try pipes.encodeGrCollapse(&f, .{
            .stream = gr_stream.binding(gpu_buf),
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
            .mode = .{ .weighted_per_token = .{ .buffer = gpu_buf, .offset = arena.offsetOf(collapse_w_raw.ptr) } },
            .m = m,
            .hidden_dim = hidden_dim,
            .branch_token_stride = max_tokens,
            .stream_dtype = stream_dtype,
            .act_dtype = act_dtype,
        }, true);
        const sig = try f.commit();
        try compute.wait(sig, 5000);
    }

    const x_in_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, x_in_raw));
    const out_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, out_raw));
    const rel_tol: f32 = if (stream_dtype == .bf16 or act_dtype == .bf16) 0.02 else 0.005;
    for (ref_x_in, x_in_u16) |exp_v, act_bits| {
        const act_v = quant.floatBitsToF32(act_dtype, act_bits);
        try std.testing.expectApproxEqAbs(exp_v, act_v, @max(rel_tol, @abs(exp_v) * rel_tol));
    }
    for (ref_out, out_u16) |exp_v, act_bits| {
        const act_v = quant.floatBitsToF32(act_dtype, act_bits);
        try std.testing.expectApproxEqAbs(exp_v, act_v, @max(rel_tol, @abs(exp_v) * rel_tol));
    }
}

test "G10 4-branch Gated Residual (f16 and bf16 streams) matches CPU fp32 ref" {
    try verifyGrRoundtrip(.f16, .f16, 1, 8, 192);
    try verifyGrRoundtrip(.bf16, .f16, 5, 16, 257);
    try verifyGrRoundtrip(.bf16, .bf16, 9, 16, 128);
}

test "M6 hybrid edge cases: circular 128-tok SWA tree commit wrap, GrCollapseMode variants, and error checks" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    var arena = try makeTestArena(512 * 1024);
    defer arena.deinit();

    // 1) Circular 128-token SWA tree commit wrap-around across slot 126, 127 -> 0, 1.
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 64;
    var swa_ring = try HybridKvRing.init(gpa, device, &arena, .{
        .mode = .sliding_window_128,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .num_tiles = 1,
    });
    defer swa_ring.deinit(gpa);

    const kt_raw = try arena.allocAligned(@as(usize, 4) * num_kv_heads * head_dim * @sizeOf(f16), 64);
    const vt_raw = try arena.allocAligned(@as(usize, 4) * num_kv_heads * head_dim * @sizeOf(f16), 64);
    const kt_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kt_raw));
    const vt_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vt_raw));
    for (kt_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 2003, 0.1));
    for (vt_f16, 0..) |*x, i| x.* = @floatCast(synthF32(i + 2503, 0.1));

    // 2) GrCollapseMode (.stream0, .sum, .mean, .weighted_shared) with broadcast_all=true.
    const hidden_dim: u32 = 64;
    const gr_arena = try GrStreamArena.init(&arena, .{ .max_tokens = 2, .hidden_dim = hidden_dim, .dtype = .f16 });
    const x0_raw = try arena.allocAligned(@as(usize, 2) * hidden_dim * @sizeOf(f16), 64);
    const out0_raw = try arena.allocAligned(@as(usize, 2) * hidden_dim * @sizeOf(f16), 64);
    const out_sum_raw = try arena.allocAligned(@as(usize, 2) * hidden_dim * @sizeOf(f16), 64);
    const out_mean_raw = try arena.allocAligned(@as(usize, 2) * hidden_dim * @sizeOf(f16), 64);
    const out_w_raw = try arena.allocAligned(@as(usize, 2) * hidden_dim * @sizeOf(f16), 64);
    const x0_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x0_raw));
    for (x0_f16, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.25);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try swa_ring.makeResident(compute);

    const wrap_nodes = [_]u16{ 0, 1, 2, 3 };
    const commit_tiles = try swa_ring.commitTilesForSpan(126, 4);
    try std.testing.expect(commit_tiles.k_tile1 == null); // single 128-tok circular tile wraps in place

    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeQsaKvTreeCommit(&f, .{
            .k_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kt_raw.ptr) },
            .v_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(vt_raw.ptr) },
            .accepted_nodes = .{ .inline_u16 = &wrap_nodes },
            .accepted_count = 4,
            .k_tile0 = commit_tiles.k_tile0,
            .v_tile0 = commit_tiles.v_tile0,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .base_pos = 126,
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
        }, false);

        try pipes.encodeGrInitStream0(&f, .{
            .x0 = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x0_raw.ptr) },
            .stream = gr_arena.binding(gpu_buf),
            .m = 2,
            .hidden_dim = hidden_dim,
            .branch_token_stride = 2,
            .broadcast_all = true,
        }, false);
        try pipes.encodeGrCollapse(&f, .{
            .stream = gr_arena.binding(gpu_buf),
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out0_raw.ptr) },
            .mode = .stream0,
            .m = 2,
            .hidden_dim = hidden_dim,
            .branch_token_stride = 2,
        }, true);
        try pipes.encodeGrCollapse(&f, .{
            .stream = gr_arena.binding(gpu_buf),
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_sum_raw.ptr) },
            .mode = .sum,
            .m = 2,
            .hidden_dim = hidden_dim,
            .branch_token_stride = 2,
        }, false);
        try pipes.encodeGrCollapse(&f, .{
            .stream = gr_arena.binding(gpu_buf),
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_mean_raw.ptr) },
            .mode = .mean,
            .m = 2,
            .hidden_dim = hidden_dim,
            .branch_token_stride = 2,
        }, false);
        try pipes.encodeGrCollapse(&f, .{
            .stream = gr_arena.binding(gpu_buf),
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_w_raw.ptr) },
            .mode = .{ .weighted_shared = .{ 0.1, 0.2, 0.3, 0.4 } },
            .m = 2,
            .hidden_dim = hidden_dim,
            .branch_token_stride = 2,
        }, false);

        // Explicit validation error checks (zero silent failures).
        const b0: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = 0 };
        try std.testing.expectError(error.InvalidMropeSection, (MropeSection{ .section = .{ 10, 10, 10 } }).validate(64));
        try std.testing.expectError(error.InvalidNodeCount, pipes.encodeGdnTreeVerify(&f, .{
            .q = b0,
            .k = b0,
            .v = b0,
            .g = b0,
            .beta = b0,
            .ancestor_masks = b0,
            .state_in = b0,
            .out = b0,
            .num_nodes = 0,
            .num_k_heads = 2,
            .num_v_heads = 2,
            .head_k_dim = 64,
            .head_v_dim = 64,
        }, false));
        try std.testing.expectError(error.MissingNormWeights, pipes.encodeQsaRopeKvWrite(&f, .{
            .q = b0,
            .k = b0,
            .v = b0,
            .q_norm = b0,
            .k_norm = null,
            .k_tile = swa_ring.k_tiles[0],
            .v_tile = swa_ring.v_tiles[0],
            .num_q_heads = 4,
            .num_kv_heads = 2,
            .head_dim = 64,
            .pos_3d = Pos3D.text(0),
            .mrope = MropeSection.QWEN38_DEFAULT,
            .slot_in_chunk = 0,
        }, false));

        const sig = try f.commit();
        try compute.wait(sig, 5000);
    }

    // Verify circular wrap slots: pos 126->126, 127->127, 128->0, 129->1.
    {
        var kg = try swa_ring.k_tiles[0].lock(true);
        errdefer kg.release();
        const stride = kg.bytes_per_row / @sizeOf(f16);
        const k_mem: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
        const expected_slots = [4]u32{ 126, 127, 0, 1 };
        for (expected_slots, 0..) |slot, node| {
            for (0..num_kv_heads) |kvh| {
                const exp = kt_f16[(node * num_kv_heads + kvh) * head_dim ..][0..head_dim];
                const got = k_mem[(kvh * 128 + slot) * stride ..][0..head_dim];
                try std.testing.expectEqualSlices(f16, exp, got);
            }
        }
        try kg.unlock();
    }

    // Verify GR collapse modes against broadcasted `x0`.
    const out0: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out0_raw));
    const out_sum: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_sum_raw));
    const out_mean: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_mean_raw));
    const out_w: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_w_raw));
    for (x0_f16, 0..) |xh, i| {
        const xv: f32 = @floatCast(xh);
        try std.testing.expectApproxEqAbs(xv, @as(f32, @floatCast(out0[i])), 1e-3);
        try std.testing.expectApproxEqAbs(xv * 4.0, @as(f32, @floatCast(out_sum[i])), 2e-3);
        try std.testing.expectApproxEqAbs(xv, @as(f32, @floatCast(out_mean[i])), 1e-3);
        try std.testing.expectApproxEqAbs(xv, @as(f32, @floatCast(out_w[i])), 2e-3);
    }
}
