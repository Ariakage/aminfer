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

/// Maximum wide-chunk GPU prefill width (`G15`, ADR-0020).
pub const MAX_WIDE_PREFILL_TOKENS: u32 = attention.MAX_WIDE_PREFILL_TOKENS;
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
/// Maximum MLA compressed latent dimension (`kv_lora_rank`, multiples of 32; `1536` for `deepseek41`).
pub const MAX_MLA_LATENT_DIM: u32 = 2048;
/// Maximum MLA decoupled RoPE key dimension (`qk_rope_dim`, even).
pub const MAX_MLA_ROPE_DIM: u32 = 256;
/// Channels per `E4M3` FP8 scale in MLA `FP4` (`E2M1` + `E4M3`) latent KV (`R7`, `G11`).
pub const MLA_FP4_GROUP_SIZE: u32 = 16;
/// Maximum selected sparse blocks per query token in `ami_csa2_index_blocks`.
pub const CSA2_MAX_TOP_K_BLOCKS: u32 = 64;
/// Default fine micro-block length in tokens for CSA2 (`G11`).
pub const CSA2_DEFAULT_BLOCK_TOKENS: u32 = 32;
/// Default fine blocks per coarse super-block in CSA2 hierarchical indexing (`G11`).
pub const CSA2_DEFAULT_COARSE_GROUP_BLOCKS: u32 = 8;
/// Full `DeepSeek-V4.1-Flash-1017` (`deepseek41`) MLA latent rank (`768 B` E2M1 + `96 B` E4M3 = `864 B`).
pub const DEEPSEEK41_KV_LORA_RANK: u32 = 1536;
/// Full `DeepSeek-V4.1-Flash-1017` compressed `E4M3` RoPE tail (`26 B` -> `864 + 26 = 890 B/tok`).
pub const DEEPSEEK41_QK_ROPE_DIM: u32 = 26;
/// Exact `deepseek41` main FP4 KV footprint per token (`R7`, ADR-0015).
pub const DEEPSEEK41_BYTES_PER_TOKEN: u32 = 890;
/// Supported hyper-stream counts for Single-Pass Mega-mHC (`G10`).
pub const MIN_MHC_STREAMS: u32 = 2;
pub const DEFAULT_MHC_STREAMS: u32 = 4;
pub const MAX_MHC_STREAMS: u32 = 8;
/// Maximum batched decode slots for `ami_mla_csa2_attend` / `ami_csa2_index_blocks`.
pub const MAX_MLA_BATCH_DECODE_SLOTS: u32 = 16;

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
    write_state_only: u32 = 0,
    out_start_token: u32 = 0,
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
    num_retired_chunks: u32 = 0,
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
    causal_q_start_pos: u32 = 0xFFFF_FFFF,
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
    num_retired_chunks: u32 = 0,
    retired_plane_stride_bytes: u32 = 0,
    retired_chunk_stride_bytes: u32 = 0,
    causal_q_start_pos: u32 = 0xFFFF_FFFF,
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

pub const GdnConv1dQParams = extern struct {
    m: u32,
    gdn_qk: u32,
    gdn_v: u32,
    conv_k: u32,
    is_tree: u32 = 1,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

pub const QsaPrefillRopeParams = extern struct {
    num_tokens: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    start_pos: u32,
    chunk_tokens: u32,
    num_retired_chunks: u32,
    ring_capacity: u32,
    mrope_s0: u32,
    mrope_s1: u32,
    mrope_s2: u32,
    use_positions_3d: u32,
    apply_qk_norm: u32,
    kv_only: u32,
    write_to_tiles: u32,
    rms_eps: f32,
    rope_theta: f32,
    yarn_factor: f32,
    yarn_mscale: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    original_max_pos: u32,
    _pad0: u32 = 0,
};

pub const QsaPrefillKvCommitParams = extern struct {
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    chunk_tokens: u32,
    num_retired_chunks: u32,
    ring_capacity: u32,
    _pad0: u32 = 0,
};

pub const QsaSwaPrefillAttnParams = extern struct {
    num_q: u32,
    q_start_pos: u32,
    chunk_start_pos: u32,
    kv_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    window_size: u32,
    ring_capacity: u32,
    chunk_tokens: u32,
    scale: f32,
    num_retired_chunks: u32 = 0,
    retired_plane_stride_bytes: u32 = 0,
    retired_chunk_stride_bytes: u32 = 0,
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

pub const GdnBatchDecodeParams = extern struct {
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    slot_stride_f32: u32,
    q_scale: f32,
    l2_eps: f32,
    rms_eps: f32,
    apply_l2_norm: u32,
    apply_out_norm: u32,
    apply_out_gate: u32,
    per_head_norm: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

pub const GdnCloneStateParams = extern struct {
    num_gdn_layers: u32,
    state_f32_per_layer: u32,
    src_state_layer_stride_f32: u32,
    dst_state_layer_stride_f32: u32,
    conv_f16_per_layer: u32,
    src_conv_layer_stride_f16: u32,
    dst_conv_layer_stride_f16: u32,
    _pad0: u32 = 0,
};

pub const MlaFp4KvWriteParams = extern struct {
    num_tokens: u32,
    start_pos: u32,
    kv_lora_rank: u32,
    qk_rope_dim: u32,
    rope_tail_format: u32,
    max_tokens: u32,
    swa_ring_capacity: u32,
    flags: u32 = 0,
    e2m1_plane_offset: u64,
    scale_plane_offset: u64,
    rope_plane_offset: u64,
    swa_rope_plane_offset: u64 = 0,
};

pub const Csa2SummaryParams = extern struct {
    kv_len: u32,
    kv_lora_rank: u32,
    index_dim: u32,
    block_tokens: u32,
    coarse_group_blocks: u32,
    start_coarse_block: u32,
    num_coarse_blocks: u32,
    _pad0: u32 = 0,
    e2m1_plane_offset: u64,
    scale_plane_offset: u64,
    fine_plane_offset: u64,
    coarse_plane_offset: u64,
};

pub const Csa2IndexParams = extern struct {
    num_nodes: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    index_dim: u32,
    kv_len: u32,
    encoder_prefix_len: u32,
    causal_q_start_pos: u32,
    swa_window: u32,
    block_tokens: u32,
    coarse_group_blocks: u32,
    top_k_coarse: u32,
    top_k_blocks: u32,
    num_local_blocks: u32,
    has_batch_slots: u32,
    scale: f32,
    _pad0: u32 = 0,
    fine_batch_stride_bytes: u64 = 0,
    coarse_batch_stride_bytes: u64 = 0,
};

pub const MlaCsa2AttnParams = extern struct {
    num_nodes: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    qk_rope_dim: u32,
    rope_tail_format: u32,
    kv_len: u32,
    encoder_prefix_len: u32,
    causal_q_start_pos: u32,
    swa_window: u32,
    swa_ring_capacity: u32,
    block_tokens: u32,
    top_k_blocks: u32,
    has_tree: u32,
    has_batch_slots: u32,
    scale: f32,
    _pad0: u32 = 0,
    e2m1_plane_offset: u64,
    scale_plane_offset: u64,
    rope_plane_offset: u64,
    swa_rope_plane_offset: u64 = 0,
    kv_batch_stride_bytes: u64 = 0,
    swa_batch_stride_bytes: u64 = 0,
};

pub const MhcParams = extern struct {
    m: u32,
    num_streams: u32,
    hidden_size: u32,
    flags: u32,
    sinkhorn_iters: u32,
    rms_eps: f32,
    post_scale: f32,
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
    if (@sizeOf(QsaSparseAttnParams) != 64) @compileError("QsaSparseAttnParams must be 64 bytes");
    if (@sizeOf(GrParams) != 48) @compileError("GrParams must be 48 bytes");
    if (@sizeOf(GrComputeGatesParams) != 16) @compileError("GrComputeGatesParams must be 16 bytes");
    if (@sizeOf(GdnComputeGatesParams) != 16) @compileError("GdnComputeGatesParams must be 16 bytes");
    if (@sizeOf(GdnConv1dParams) != 32) @compileError("GdnConv1dParams must be 32 bytes");
    if (@sizeOf(GdnConv1dQParams) != 32) @compileError("GdnConv1dQParams must be 32 bytes");
    if (@sizeOf(QsaPrefillRopeParams) != 96) @compileError("QsaPrefillRopeParams must be 96 bytes");
    if (@sizeOf(QsaPrefillKvCommitParams) != 32) @compileError("QsaPrefillKvCommitParams must be 32 bytes");
    if (@sizeOf(QsaSwaPrefillAttnParams) != 64) @compileError("QsaSwaPrefillAttnParams must be 64 bytes");
    if (@sizeOf(GdnConv1dCommitParams) != 144) @compileError("GdnConv1dCommitParams must be 144 bytes");
    if (@sizeOf(BlendF16Params) != 16) @compileError("BlendF16Params must be 16 bytes");
    if (@sizeOf(GdnBatchDecodeParams) != 64) @compileError("GdnBatchDecodeParams must be 64 bytes");
    if (@sizeOf(GdnCloneStateParams) != 32) @compileError("GdnCloneStateParams must be 32 bytes");
    if (@sizeOf(MlaFp4KvWriteParams) != 64) @compileError("MlaFp4KvWriteParams must be 64 bytes");
    if (@sizeOf(Csa2SummaryParams) != 64) @compileError("Csa2SummaryParams must be 64 bytes");
    if (@sizeOf(Csa2IndexParams) != 80) @compileError("Csa2IndexParams must be 80 bytes");
    if (@sizeOf(MlaCsa2AttnParams) != 112) @compileError("MlaCsa2AttnParams must be 112 bytes");
    if (@sizeOf(MhcParams) != 32) @compileError("MhcParams must be 32 bytes");
    if (@sizeOf(GdnCommitTreeParams) > metal.MAX_UNIFORM_BYTES) @compileError("GdnCommitTreeParams exceeds uniform stride");
    if (@sizeOf(QsaKvTreeCommitParams) > metal.MAX_UNIFORM_BYTES) @compileError("QsaKvTreeCommitParams exceeds uniform stride");
    if (@sizeOf(GdnConv1dCommitParams) > metal.MAX_UNIFORM_BYTES) @compileError("GdnConv1dCommitParams exceeds uniform stride");
    if (@sizeOf(MlaCsa2AttnParams) > metal.MAX_UNIFORM_BYTES) @compileError("MlaCsa2AttnParams exceeds uniform stride");
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

pub const GdnSnapshotPoolError = ArenaError || error{
    PoolExhausted,
    InvalidSlot,
    DoubleFree,
};

pub const GdnSnapshotPoolSpec = struct {
    state: GdnStateSpec,
    /// Optional `conv1d` history elements (`(conv_k - 1) * (2 * gdn_qk + gdn_v)`)
    /// per GDN layer stored as `f16` alongside the recurrent state (`0` disables conv1d storage).
    conv_f16_per_layer: u32 = 0,
    max_snapshots: u32,
};

/// Preallocated `16 KiB` page-aligned `WiredArena` pool storing `max_snapshots`
/// `GDN` recurrent state + optional `conv1d` history slabs for `R8` shared
/// prefix checkpointing (`0` heap allocations at init and on the hot path, O(1)
/// free-list acquire/release).
pub const GdnSnapshotPool = struct {
    spec: GdnSnapshotPoolSpec,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    base_offset: u64,
    layer_state_elems: u32,
    state_layer_stride_bytes: usize,
    state_slab_bytes: usize,
    conv_layer_stride_bytes: usize,
    conv_slab_bytes: usize,
    conv_base_offset: u64,
    free_stack: []u32,
    in_use: []u8,
    free_top: u32,

    pub fn validateSpec(spec: GdnSnapshotPoolSpec) GdnSnapshotPoolError!void {
        try GdnStateArena.validateSpec(spec.state);
        if (spec.max_snapshots == 0) return error.ZeroDimension;
    }

    pub fn dataRequiredBytes(spec: GdnSnapshotPoolSpec) GdnSnapshotPoolError!usize {
        try validateSpec(spec);
        const state_elems = try GdnStateArena.layerElems(spec.state);
        const state_layer_stride = amf.align64(std.math.mul(usize, state_elems, @sizeOf(f32)) catch return error.Overflow);
        const state_slab = std.math.mul(usize, state_layer_stride, spec.state.num_gdn_layers) catch return error.Overflow;
        const all_states = std.math.mul(usize, state_slab, spec.max_snapshots) catch return error.Overflow;

        const conv_layer_stride = if (spec.conv_f16_per_layer > 0)
            amf.align64(std.math.mul(usize, spec.conv_f16_per_layer, @sizeOf(f16)) catch return error.Overflow)
        else
            0;
        const conv_slab = std.math.mul(usize, conv_layer_stride, spec.state.num_gdn_layers) catch return error.Overflow;
        const all_convs = std.math.mul(usize, conv_slab, spec.max_snapshots) catch return error.Overflow;

        const total = std.math.add(usize, all_states, all_convs) catch return error.Overflow;
        return page.alignUp(total) catch return error.Overflow;
    }

    pub fn requiredArenaBytes(spec: GdnSnapshotPoolSpec) GdnSnapshotPoolError!usize {
        const data_bytes = try dataRequiredBytes(spec);
        const stack_bytes = amf.align64(std.math.mul(usize, spec.max_snapshots, @sizeOf(u32)) catch return error.Overflow);
        const flags_bytes = amf.align64(spec.max_snapshots);
        const meta = std.math.add(usize, stack_bytes, flags_bytes) catch return error.Overflow;
        const total = std.math.add(usize, data_bytes, meta) catch return error.Overflow;
        return page.alignUp(total) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: GdnSnapshotPoolSpec) GdnSnapshotPoolError!GdnSnapshotPool {
        try validateSpec(spec);
        const state_elems_usize = try GdnStateArena.layerElems(spec.state);
        const state_elems = std.math.cast(u32, state_elems_usize) orelse return error.Overflow;
        const state_layer_stride = amf.align64(state_elems_usize * @sizeOf(f32));
        const state_slab = state_layer_stride * spec.state.num_gdn_layers;
        const all_states = state_slab * spec.max_snapshots;

        const conv_layer_stride = if (spec.conv_f16_per_layer > 0)
            amf.align64(@as(usize, spec.conv_f16_per_layer) * @sizeOf(f16))
        else
            0;
        const conv_slab = conv_layer_stride * spec.state.num_gdn_layers;
        const data_bytes = try dataRequiredBytes(spec);

        const backing = try arena.allocPages(data_bytes);
        @memset(backing, 0);
        const base_off: u64 = @intCast(arena.offsetOf(backing.ptr));

        const stack_raw = try arena.allocAligned(amf.align64(@as(usize, spec.max_snapshots) * @sizeOf(u32)), 64);
        const free_stack: []u32 = @alignCast(std.mem.bytesAsSlice(
            u32,
            stack_raw[0 .. @as(usize, spec.max_snapshots) * @sizeOf(u32)],
        ));
        const flags_raw = try arena.allocAligned(amf.align64(spec.max_snapshots), 64);
        const in_use = flags_raw[0..spec.max_snapshots];

        var pool: GdnSnapshotPool = .{
            .spec = spec,
            .backing = backing,
            .base_offset = base_off,
            .layer_state_elems = state_elems,
            .state_layer_stride_bytes = state_layer_stride,
            .state_slab_bytes = state_slab,
            .conv_layer_stride_bytes = conv_layer_stride,
            .conv_slab_bytes = conv_slab,
            .conv_base_offset = base_off + @as(u64, @intCast(all_states)),
            .free_stack = free_stack,
            .in_use = in_use,
            .free_top = spec.max_snapshots,
        };
        pool.reset();
        return pool;
    }

    pub fn reset(self: *GdnSnapshotPool) void {
        @memset(self.in_use, 0);
        const n = self.spec.max_snapshots;
        for (0..n) |i| {
            self.free_stack[i] = n - 1 - @as(u32, @intCast(i));
        }
        self.free_top = n;
    }

    pub fn bytes(self: *const GdnSnapshotPool) []align(page.APPLE_PAGE_SIZE) u8 {
        return self.backing;
    }

    pub fn acquireSlot(self: *GdnSnapshotPool) GdnSnapshotPoolError!u32 {
        if (self.free_top == 0) return error.PoolExhausted;
        self.free_top -= 1;
        const slot = self.free_stack[self.free_top];
        std.debug.assert(slot < self.spec.max_snapshots);
        std.debug.assert(self.in_use[slot] == 0);
        self.in_use[slot] = 1;
        return slot;
    }

    pub fn releaseSlot(self: *GdnSnapshotPool, slot_id: u32) GdnSnapshotPoolError!void {
        if (slot_id >= self.spec.max_snapshots) return error.InvalidSlot;
        if (self.in_use[slot_id] == 0) return error.DoubleFree;
        self.in_use[slot_id] = 0;
        self.free_stack[self.free_top] = slot_id;
        self.free_top += 1;
    }

    pub fn stateBinding(self: *const GdnSnapshotPool, buf: metal.Buffer, slot_id: u32) GdnSnapshotPoolError!metal.BufferBinding {
        if (slot_id >= self.spec.max_snapshots) return error.InvalidSlot;
        return .{
            .buffer = buf,
            .offset = self.base_offset + @as(u64, slot_id) * @as(u64, @intCast(self.state_slab_bytes)),
        };
    }

    pub fn convBinding(self: *const GdnSnapshotPool, buf: metal.Buffer, slot_id: u32) GdnSnapshotPoolError!?metal.BufferBinding {
        if (slot_id >= self.spec.max_snapshots) return error.InvalidSlot;
        if (self.spec.conv_f16_per_layer == 0) return null;
        return .{
            .buffer = buf,
            .offset = self.conv_base_offset + @as(u64, slot_id) * @as(u64, @intCast(self.conv_slab_bytes)),
        };
    }

    pub fn stateLayerStrideF32(self: *const GdnSnapshotPool) u32 {
        return @intCast(self.state_layer_stride_bytes / @sizeOf(f32));
    }

    pub fn convLayerStrideF16(self: *const GdnSnapshotPool) u32 {
        return @intCast(self.conv_layer_stride_bytes / @sizeOf(f16));
    }

    pub fn freeCount(self: *const GdnSnapshotPool) u32 {
        return self.free_top;
    }

    pub fn activeCount(self: *const GdnSnapshotPool) u32 {
        return self.spec.max_snapshots - self.free_top;
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
    /// When `true` (`P0-B1` non-final `L - 1`), updates recurrent `state_out`
    /// only and skips computing/writing `out` and post-norm.
    write_state_only: bool = false,
    /// Optional first token index emitted to `out` (e.g. `seq_len - 1` for
    /// `P0-B1` final-chunk `M = 1` tail narrowing). Defaults to `0` (all rows).
    out_start_token: u32 = 0,
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
    /// Optional asymmetric V head dimension (`0` defaults to `head_dim`, e.g.,
    /// `head_dim = 192, v_head_dim = 128` on `MiMo-V2.6` SWA & Global GQA).
    v_head_dim: u32 = 0,
    /// Number of chunk tiles in the ring:
    /// - `.full_chunk_512`: `ceil(max_seq_len / 512)` tiles (`ring_capacity = 0`, `window_size = 0`).
    /// - `.sliding_window_128`: number of `128`-token tiles (`1` tile -> `128`-token circular SWA).
    num_tiles: u32,
    /// Optional explicit sliding window override (defaults to `0` for full, `num_tiles * 128` for SWA).
    window_size_override: ?u32 = null,
};

/// Resident IOSurface `RGBA16Float` `KvTile` ring supporting both `512`-token
/// full-attention chunks and `128`-token circular sliding-window tiles (`R5`, `G26`),
/// including asymmetric `head_dim != v_head_dim` (`K = 192, V = 128`).
pub const HybridKvRing = struct {
    mode: HybridKvRingMode,
    num_kv_heads: u32,
    head_dim: u32,
    v_head_dim: u32,
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
        const eff_v_head_dim: u32 = if (spec.v_head_dim > 0) spec.v_head_dim else spec.head_dim;
        if (spec.num_kv_heads == 0 or spec.head_dim == 0 or eff_v_head_dim == 0 or spec.num_tiles == 0) return error.ZeroDimension;
        if (spec.head_dim > MAX_QSA_HEAD_DIM or (spec.head_dim % 4) != 0) return error.InvalidConfig;
        if (eff_v_head_dim > MAX_QSA_HEAD_DIM or (eff_v_head_dim % 4) != 0) return error.InvalidConfig;

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
            v_tiles[i] = try metal.KvTile.init(device, eff_v_head_dim, rows, .{});
            tile_table[i] = attention.KvTilePair.fromTiles(k_tiles[i], v_tiles[i], .rgba16f);
            created += 1;
        }

        return .{
            .mode = spec.mode,
            .num_kv_heads = spec.num_kv_heads,
            .head_dim = spec.head_dim,
            .v_head_dim = eff_v_head_dim,
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
        if (spec.max_nodes > MAX_WIDE_PREFILL_TOKENS or spec.top_k_blocks > QSA_MAX_TOP_K_BLOCKS) {
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

    /// Returns a zero-copy view covering `count` nodes starting at `start_node`,
    /// reusing the same underlying wired `selected_blocks` / `selected_counts` storage.
    pub fn nodeSlice(self: *const QsaIndexBuffer, start_node: u32, count: u32) ArenaError!QsaIndexBuffer {
        if (count == 0 or @as(u64, start_node) + count > self.spec.max_nodes) {
            return error.InvalidConfig;
        }
        const heads: usize = self.spec.num_kv_heads;
        const top_k: usize = self.spec.top_k_blocks;
        const cnt_start: usize = @as(usize, start_node) * heads;
        const cnt_len: usize = @as(usize, count) * heads;
        const blk_start: usize = cnt_start * top_k;
        const blk_len: usize = cnt_len * top_k;
        return .{
            .spec = .{
                .max_nodes = count,
                .num_kv_heads = self.spec.num_kv_heads,
                .top_k_blocks = self.spec.top_k_blocks,
            },
            .selected_blocks_off = self.selected_blocks_off + @as(u64, blk_start) * @sizeOf(u32),
            .selected_counts_off = self.selected_counts_off + @as(u64, cnt_start) * @sizeOf(u32),
            .selected_blocks = self.selected_blocks[blk_start .. blk_start + blk_len],
            .selected_counts = self.selected_counts[cnt_start .. cnt_start + cnt_len],
        };
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
    num_retired_chunks: u32 = 0,
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
    /// When set, runs causal prefill indexing where query `node` only considers
    /// blocks up to `min(kv_len, causal_q_start_pos + node + 1)` (skipping
    /// strictly upper-triangular blocks).
    causal_q_start_pos: ?u32 = null,
};

pub const QsaSparseAttnArgs = struct {
    q: metal.BufferBinding,
    /// Resident `KvTilePair` table (may be null only when `kv_len <= num_retired_chunks * chunk_tokens`).
    kv_tiles: ?metal.BufferBinding = null,
    index_buf: *const QsaIndexBuffer,
    index_metal_buf: metal.Buffer,
    /// Optional speculative tree KV + 64-bit ancestor masks (all 3 required when `has_tree`).
    k_tree: ?metal.BufferBinding = null,
    v_tree: ?metal.BufferBinding = null,
    ancestor_masks: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    /// Optional `R5` group-64 `q8` retired KV tier (`num_retired_chunks > 0` requires
    /// `retired_pool`, `retired_chunk_ids`, `ring_capacity == 0`, and `head_dim % 64 == 0`).
    retired_pool: ?metal.BufferBinding = null,
    retired_chunk_ids: ?metal.BufferBinding = null,
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
    num_retired_chunks: u32 = 0,
    retired_plane_stride_bytes: u32 = 0,
    retired_chunk_stride_bytes: u32 = 0,
    /// When set, enforces causal `kv_pos <= causal_q_start_pos + node` per query
    /// row (strictly skipping upper-triangular KV positions).
    causal_q_start_pos: ?u32 = null,
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
    retired_pool: ?metal.BufferBinding = null,
    retired_chunk_ids: ?metal.BufferBinding = null,
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
    num_retired_chunks: u32 = 0,
    retired_plane_stride_bytes: u32 = 0,
    retired_chunk_stride_bytes: u32 = 0,
    /// When `true` (`r >= 1` in `LoopSchedule`, ADR-0019), skips `ami_qsa_index_blocks`
    /// and reuses the selected block indices already stored in `index_buf`.
    reuse_block_indices: bool = false,
    /// Optional causal prefill starting position for wide-chunk QSA prefill (`G15`).
    causal_q_start_pos: ?u32 = null,
};

pub const QsaPrefillRopeArgs = struct {
    q: metal.BufferBinding,
    k: metal.BufferBinding,
    v: metal.BufferBinding,
    q_norm: ?metal.BufferBinding = null,
    k_norm: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .f16,
    /// Optional `[num_tokens]Pos3D` buffer; defaults to 1D `start_pos + tok` when null.
    positions_3d: ?metal.BufferBinding = null,
    /// Bindless `HybridKvRing` `KvTilePair` table (required when `write_to_tiles == true`).
    kv_tiles: ?metal.BufferBinding = null,
    num_tokens: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32 = 0,
    start_pos: u32,
    mrope: MropeSection,
    yarn: YarnConfig = .{},
    chunk_tokens: u32 = QSA_FULL_CHUNK_TOKENS,
    num_retired_chunks: u32 = 0,
    ring_capacity: u32 = 0,
    /// When `true` (`P0-B1` non-final `L - 1`), skips Q updates and writes K/V only.
    kv_only: bool = false,
    /// When `true`, writes K and V directly into `kv_tiles`; when `false` (circular
    /// SWA prefill), writes rotated K in-place to `k` so ring history stays intact
    /// until `encodeQsaPrefillKvRingCommit` runs after attention.
    write_to_tiles: bool = true,
    rms_eps: f32 = 1e-6,
    rope_theta: f32 = 1_000_000.0,
};

pub const QsaPrefillKvCommitArgs = struct {
    k: metal.BufferBinding,
    v: metal.BufferBinding,
    kv_tiles: metal.BufferBinding,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    chunk_tokens: u32 = QSA_FULL_CHUNK_TOKENS,
    num_retired_chunks: u32 = 0,
    ring_capacity: u32 = 0,
};

pub const QsaSwaPrefillAttnArgs = struct {
    q: metal.BufferBinding,
    kv_tiles: ?metal.BufferBinding = null,
    k_chunk: metal.BufferBinding,
    v_chunk: metal.BufferBinding,
    out: metal.BufferBinding,
    retired_pool: ?metal.BufferBinding = null,
    retired_chunk_ids: ?metal.BufferBinding = null,
    num_q: u32,
    q_start_pos: u32,
    chunk_start_pos: u32,
    kv_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    window_size: u32 = QSA_SWA_WINDOW_TOKENS,
    ring_capacity: u32 = QSA_SWA_WINDOW_TOKENS,
    chunk_tokens: u32 = QSA_SWA_WINDOW_TOKENS,
    scale: ?f32 = null,
    num_retired_chunks: u32 = 0,
    retired_plane_stride_bytes: u32 = 0,
    retired_chunk_stride_bytes: u32 = 0,
    /// Optional `G19` MMA override (`null` enforces strict `Q_len >= 32 && KV_len >= 64` guard).
    use_mma_override: ?bool = null,
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

pub const GdnConv1dQSiluArgs = struct {
    q_cur: metal.BufferBinding,
    q_hist_r0: metal.BufferBinding,
    conv_w: metal.BufferBinding,
    conv_state_in: metal.BufferBinding,
    ancestor_masks: ?metal.BufferBinding = null,
    q_out: metal.BufferBinding,
    m: u32,
    gdn_qk: u32,
    gdn_v: u32,
    conv_k: u32,
    is_tree: bool = true,
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

pub const GdnBatchDecodeArgs = struct {
    /// `[batch_size, num_k_heads, head_k_dim]` f16 queries.
    q: metal.BufferBinding,
    /// `[batch_size, num_k_heads, head_k_dim]` f16 keys.
    k: metal.BufferBinding,
    /// `[batch_size, num_v_heads, head_v_dim]` f16 values.
    v: metal.BufferBinding,
    /// `[batch_size, num_v_heads]` f32 log-decay gates.
    g: metal.BufferBinding,
    /// `[batch_size, num_v_heads]` f32 raw write gates (passed through sigmoid).
    beta: metal.BufferBinding,
    /// Optional `[batch_size, num_v_heads, head_v_dim]` f16 SiLU output gate.
    z: ?metal.BufferBinding = null,
    /// Optional `[head_v_dim]` (or `[num_v_heads, head_v_dim]`) RMSNorm weights.
    norm_w: ?metal.BufferBinding = null,
    /// Base of pooled per-slot `f32` recurrent states (read).
    state_in_pool: metal.BufferBinding,
    /// Base of pooled per-slot `f32` recurrent states (write; may alias `state_in_pool`).
    state_out_pool: metal.BufferBinding,
    /// `[batch_size, num_v_heads, head_v_dim]` f16 output.
    out: metal.BufferBinding,
    /// `[batch_size]` `u32` slot indices into `state_in_pool`.
    state_in_slots: metal.BufferBinding,
    /// `[batch_size]` `u32` slot indices into `state_out_pool` (defaults to `state_in_slots`).
    state_out_slots: ?metal.BufferBinding = null,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    /// Stride between consecutive slots in `state_{in,out}_pool` in `f32` elements
    /// (`>= num_v_heads * head_k_dim * head_v_dim`).
    slot_stride_f32: u32,
    q_scale: ?f32 = null,
    l2_eps: f32 = 1e-6,
    rms_eps: f32 = 1e-6,
    apply_l2_norm: bool = true,
    per_head_norm: bool = false,
    norm_dtype: quant.FloatDType = .f16,
};

pub const GdnCloneStateArgs = struct {
    /// Source multi-layer GDN recurrent state slab (`[num_gdn_layers, ...]` `f32`).
    src_state: metal.BufferBinding,
    /// Destination multi-layer GDN recurrent state slab (`[num_gdn_layers, ...]` `f32`).
    dst_state: metal.BufferBinding,
    /// Optional source multi-layer `conv1d` history slab (`f16`; required iff `conv_f16_per_layer > 0`).
    src_conv: ?metal.BufferBinding = null,
    /// Optional destination multi-layer `conv1d` history slab (`f16`; required iff `conv_f16_per_layer > 0`).
    dst_conv: ?metal.BufferBinding = null,
    num_gdn_layers: u32,
    state_f32_per_layer: u32,
    src_state_layer_stride_f32: u32,
    dst_state_layer_stride_f32: u32,
    conv_f16_per_layer: u32 = 0,
    src_conv_layer_stride_f16: u32 = 0,
    dst_conv_layer_stride_f16: u32 = 0,
};

// ===========================================================================
// Section E: M7 (R7 + G11) — MLA + CSA2 (Full / Reindex / Reuse), FP4 KV
// (`E2M1` + `E4M3` per-16-channel scale, 890 B/tok) & SWA Bounded Replay
// ===========================================================================

/// Static per-layer CSA2 KV & sparse-index sharing mode (`R7`, ADR-0015).
pub const Csa2Mode = enum(u8) {
    /// Owns its own physical FP4 latent KV slab & runs hierarchical indexer (`r = 0`).
    full = 0,
    /// Aliases a `shared_kv_owner` `Full` layer's FP4 KV slab (0 extra KV bytes),
    /// but runs its own hierarchical indexer over its own query projections (`r = 0`).
    reindex = 1,
    /// Aliases both a `shared_kv_owner` `Full` layer's FP4 KV slab (0 extra KV bytes)
    /// AND a `shared_index_owner` layer's `Csa2IndexBuffer` (0 indexer dispatches).
    reuse = 2,
};

/// Storage encoding for the decoupled MLA RoPE key tail (`qk_rope_dim`).
pub const MlaRopeTailFormat = enum(u8) {
    /// IEEE `f16` (`2 * qk_rope_dim` bytes/token).
    f16 = 0,
    /// OCP FP8 `E4M3FN` (`1 * qk_rope_dim` bytes/token; `26 B` in `deepseek41` full geometry).
    e4m3 = 1,
};

fn alignUp16(val: usize) ArenaError!usize {
    const added = std.math.add(usize, val, 15) catch return error.Overflow;
    return added & ~@as(usize, 15);
}

/// Decodes a signed 4-bit `E2M1` nibble (`1` sign, `2` exp, `1` mantissa) to `f32`.
/// Representable magnitudes: `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`.
pub fn e2m1NibbleToF32(nib: u4) f32 {
    const lut = [8]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };
    const mag = nib & 0x7;
    const v = lut[mag];
    return if ((nib & 0x8) != 0 and mag != 0) -v else v;
}

/// Encodes `f32` to signed 4-bit `E2M1` with round-to-nearest-even (bit-exact with GPU).
pub fn f32ToE2m1Nibble(x: f32) u4 {
    if (std.math.isNan(x)) return 0;
    const neg = x < 0.0;
    const a = @abs(x);
    const mag: u4 = if (a <= 0.25)
        0
    else if (a < 0.75)
        1
    else if (a <= 1.25)
        2
    else if (a < 1.75)
        3
    else if (a <= 2.50)
        4
    else if (a < 3.50)
        5
    else if (a <= 5.00)
        6
    else
        7;
    if (mag == 0) return 0;
    return if (neg) (mag | 0x8) else mag;
}

/// Decodes OCP FP8 `E4M3FN` (`1` sign, `4` exp bias 7, `3` mantissa, max `448.0`) to `f32`.
pub fn e4m3ToF32(b: u8) f32 {
    const s = (b >> 7) & 1;
    const e: u32 = (b >> 3) & 0xF;
    const m: u32 = b & 0x7;
    const val: f32 = if (e == 0)
        @as(f32, @floatFromInt(m)) * (1.0 / 512.0)
    else if (e == 15 and m == 7)
        0.0
    else
        std.math.ldexp(@as(f32, @floatFromInt(8 + m)), @as(i32, @intCast(e)) - 10);
    return if (s != 0) -val else val;
}

/// Encodes `f32` to OCP FP8 `E4M3FN` with round-to-nearest-even and saturation at `+-448.0`
/// (bit-exact with `mla_f32_to_e4m3` in `hybrid_ops.metal`).
pub fn f32ToE4m3(x: f32) u8 {
    if (std.math.isNan(x)) return 0;
    const bits: u32 = @bitCast(x);
    const sign_bit: u8 = @intCast((bits >> 31) & 1);
    const a = @abs(x);
    if (a == 0.0) return 0;
    if (a >= 448.0) return (sign_bit << 7) | 0x7E;
    if (a < (1.0 / 64.0)) {
        const scaled = a * 512.0;
        const q_floor: u32 = @intFromFloat(@floor(scaled));
        const frac = scaled - @as(f32, @floatFromInt(q_floor));
        var q = q_floor;
        if (frac > 0.5 or (frac == 0.5 and (q_floor & 1) != 0)) {
            q += 1;
        }
        if (q == 0) return 0;
        if (q >= 8) return (sign_bit << 7) | (1 << 3);
        return (sign_bit << 7) | @as(u8, @intCast(q));
    }
    const abs_bits: u32 = @bitCast(a);
    const f_exp: i32 = @as(i32, @intCast((abs_bits >> 23) & 0xFF)) - 127;
    const f_mant: u32 = abs_bits & 0x7FFFFF;
    var m3: u32 = f_mant >> 20;
    const rem: u32 = f_mant & 0xFFFFF;
    var e4: u32 = @intCast(f_exp + 7);
    if (rem > 0x80000 or (rem == 0x80000 and (m3 & 1) != 0)) {
        m3 += 1;
        if (m3 == 8) {
            m3 = 0;
            e4 += 1;
        }
    }
    if (e4 > 15 or (e4 == 15 and m3 >= 7)) {
        e4 = 15;
        m3 = 6;
    }
    return (sign_bit << 7) | @as(u8, @intCast(e4 << 3)) | @as(u8, @intCast(m3));
}

/// Host reference encoder for one MLA latent row (`kv_lora_rank` multiple of 32)
/// into packed `E2M1` nibbles (`kv_lora_rank / 2` B) + per-16 `E4M3` scales (`kv_lora_rank / 16` B).
pub fn encodeMlaFp4LatentRowCpu(src_f32: []const f32, dst_e2m1: []u8, dst_scales: []u8) void {
    std.debug.assert(src_f32.len > 0 and (src_f32.len % 32) == 0);
    const num_groups = src_f32.len / MLA_FP4_GROUP_SIZE;
    std.debug.assert(dst_e2m1.len == src_f32.len / 2);
    std.debug.assert(dst_scales.len == num_groups);

    for (0..num_groups) |g| {
        const ch0 = g * MLA_FP4_GROUP_SIZE;
        var amax: f32 = 0.0;
        for (0..MLA_FP4_GROUP_SIZE) |i| {
            amax = @max(amax, @abs(src_f32[ch0 + i]));
        }
        var s_byte: u8 = 0;
        var inv_s: f32 = 0.0;
        if (amax > 0.0) {
            s_byte = f32ToE4m3(amax * (1.0 / 6.0));
            if (s_byte == 0) s_byte = 1;
            inv_s = 1.0 / e4m3ToF32(s_byte);
        }
        dst_scales[g] = s_byte;

        const byte0 = g * 8;
        for (0..8) |b| {
            const q0: u8 = if (s_byte != 0) @as(u8, f32ToE2m1Nibble(src_f32[ch0 + b * 2] * inv_s)) else 0;
            const q1: u8 = if (s_byte != 0) @as(u8, f32ToE2m1Nibble(src_f32[ch0 + b * 2 + 1] * inv_s)) else 0;
            dst_e2m1[byte0 + b] = (q1 << 4) | (q0 & 0xF);
        }
    }
}

/// Host reference decoder for one FP4 (`E2M1` + per-16 `E4M3` scale) MLA latent row.
pub fn decodeMlaFp4LatentRowCpu(src_e2m1: []const u8, src_scales: []const u8, dst_f32: []f32) void {
    std.debug.assert(dst_f32.len > 0 and (dst_f32.len % 32) == 0);
    const num_groups = dst_f32.len / MLA_FP4_GROUP_SIZE;
    std.debug.assert(src_e2m1.len == dst_f32.len / 2);
    std.debug.assert(src_scales.len == num_groups);

    for (0..num_groups) |g| {
        const sc = e4m3ToF32(src_scales[g]);
        const ch0 = g * MLA_FP4_GROUP_SIZE;
        const byte0 = g * 8;
        for (0..8) |b| {
            const packed_byte = src_e2m1[byte0 + b];
            dst_f32[ch0 + b * 2] = e2m1NibbleToF32(@truncate(packed_byte & 0xF)) * sc;
            dst_f32[ch0 + b * 2 + 1] = e2m1NibbleToF32(@truncate((packed_byte >> 4) & 0xF)) * sc;
        }
    }
}

/// Host reference encoder for one decoupled MLA RoPE key row (`f16` or `E4M3`).
pub fn encodeMlaRopeRowCpu(src_f32: []const f32, format: MlaRopeTailFormat, dst_bytes: []u8) void {
    switch (format) {
        .f16 => {
            std.debug.assert(dst_bytes.len == src_f32.len * 2);
            const dst_f16: []align(1) f16 = std.mem.bytesAsSlice(f16, dst_bytes);
            for (src_f32, 0..) |v, i| dst_f16[i] = @floatCast(v);
        },
        .e4m3 => {
            std.debug.assert(dst_bytes.len == src_f32.len);
            for (src_f32, 0..) |v, i| dst_bytes[i] = f32ToE4m3(v);
        },
    }
}

/// Host reference decoder for one decoupled MLA RoPE key row (`f16` or `E4M3`).
pub fn decodeMlaRopeRowCpu(src_bytes: []const u8, format: MlaRopeTailFormat, dst_f32: []f32) void {
    switch (format) {
        .f16 => {
            std.debug.assert(src_bytes.len == dst_f32.len * 2);
            const src_f16: []align(1) const f16 = std.mem.bytesAsSlice(f16, src_bytes);
            for (src_f16, 0..) |v, i| dst_f32[i] = @floatCast(v);
        },
        .e4m3 => {
            std.debug.assert(src_bytes.len == dst_f32.len);
            for (src_bytes, 0..) |b, i| dst_f32[i] = e4m3ToF32(b);
        },
    }
}

/// Geometry descriptor for wired FP4 (`E2M1` + per-16 `E4M3`) MLA latent KV slabs.
pub const MlaFp4KvSpec = struct {
    /// Number of physical `Full` CSA2 layers owning KV slabs (`Reindex`/`Reuse` allocate 0).
    num_kv_layers: u32,
    /// Maximum tokens per physical KV layer slab.
    max_tokens: u32,
    /// Compressed MLA latent rank (multiple of 32, `<= MAX_MLA_LATENT_DIM`).
    kv_lora_rank: u32,
    /// Decoupled RoPE key tail dimension (even, `<= MAX_MLA_ROPE_DIM`).
    qk_rope_dim: u32,
    /// RoPE tail storage format (`f16` or compressed `e4m3`).
    rope_tail_format: MlaRopeTailFormat = .f16,

    /// Canonical `DeepSeek-V4.1-Flash-1017` (`deepseek41`) full geometry:
    /// `kv_lora_rank = 1536` (`768 B` E2M1 + `96 B` E4M3 = `864 B`) +
    /// `26 B` compressed `E4M3` RoPE tail = **exact `890 B/token`** (`R7`, ADR-0015).
    pub fn deepseek41Full(max_tokens: u32, num_kv_layers: u32) MlaFp4KvSpec {
        return .{
            .num_kv_layers = num_kv_layers,
            .max_tokens = max_tokens,
            .kv_lora_rank = DEEPSEEK41_KV_LORA_RANK,
            .qk_rope_dim = DEEPSEEK41_QK_ROPE_DIM,
            .rope_tail_format = .e4m3,
        };
    }

    pub fn validate(self: MlaFp4KvSpec) ArenaError!void {
        if (self.num_kv_layers == 0 or self.max_tokens == 0 or self.kv_lora_rank == 0) {
            return error.ZeroDimension;
        }
        if ((self.kv_lora_rank % 32) != 0 or self.kv_lora_rank > MAX_MLA_LATENT_DIM) {
            return error.InvalidConfig;
        }
        if ((self.qk_rope_dim % 2) != 0 or self.qk_rope_dim > MAX_MLA_ROPE_DIM) {
            return error.InvalidConfig;
        }
    }

    pub fn latentDataBytesPerToken(self: MlaFp4KvSpec) u32 {
        return self.kv_lora_rank / 2;
    }

    pub fn latentScaleBytesPerToken(self: MlaFp4KvSpec) u32 {
        return self.kv_lora_rank / MLA_FP4_GROUP_SIZE;
    }

    pub fn ropeTailBytesPerToken(self: MlaFp4KvSpec) u32 {
        return switch (self.rope_tail_format) {
            .f16 => self.qk_rope_dim * 2,
            .e4m3 => self.qk_rope_dim,
        };
    }

    /// Exact logical compressed KV footprint per token (`890 B/tok` for `deepseek41Full`).
    pub fn bytesPerToken(self: MlaFp4KvSpec) u32 {
        return self.latentDataBytesPerToken() + self.latentScaleBytesPerToken() + self.ropeTailBytesPerToken();
    }
};

/// 16 KiB-aligned wired `MTLBuffer` arena holding `num_kv_layers` physical `Full`
/// MLA FP4 (`E2M1` + per-16 `E4M3` scale) latent KV slabs (`R7`, `G11`).
///
/// Each physical layer occupies a `16 KiB`-aligned planar slab:
///   - Plane 0 (`e2m1_plane_offset = 0`): `[max_tokens, kv_lora_rank / 2]` `u8`
///   - Plane 1 (`scale_plane_offset`):    `[max_tokens, kv_lora_rank / 16]` `u8`
///   - Plane 2 (`rope_plane_offset`):     `[max_tokens, ropeTailBytesPerToken()]` `u8`
///
/// `Reindex` and `Reuse` layers allocate **0** bytes here and alias their
/// `shared_kv_owner` `Full` layer's slab via `sharedLayerBinding`.
pub const MlaFp4KvArena = struct {
    spec: MlaFp4KvSpec,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    base_offset: u64,
    layer_stride_bytes: u64,
    e2m1_plane_offset: u64,
    scale_plane_offset: u64,
    rope_plane_offset: u64,
    e2m1_plane_bytes: usize,
    scale_plane_bytes: usize,
    rope_plane_bytes: usize,

    const PlaneLayout = struct {
        e2m1_bytes: usize,
        scale_off: usize,
        scale_bytes: usize,
        rope_off: usize,
        rope_bytes: usize,
        layer_stride: usize,
    };

    fn computeLayout(spec: MlaFp4KvSpec) ArenaError!PlaneLayout {
        try spec.validate();
        const e2m1_bytes = std.math.mul(usize, spec.max_tokens, spec.latentDataBytesPerToken()) catch return error.Overflow;
        const scale_off = try alignUp16(e2m1_bytes);
        const scale_bytes = std.math.mul(usize, spec.max_tokens, spec.latentScaleBytesPerToken()) catch return error.Overflow;
        const rope_off = try alignUp16(std.math.add(usize, scale_off, scale_bytes) catch return error.Overflow);
        const rope_bytes = std.math.mul(usize, spec.max_tokens, spec.ropeTailBytesPerToken()) catch return error.Overflow;
        const raw_layer = std.math.add(usize, rope_off, rope_bytes) catch return error.Overflow;
        const layer_stride = page.alignUp(raw_layer) catch return error.Overflow;
        return .{
            .e2m1_bytes = e2m1_bytes,
            .scale_off = scale_off,
            .scale_bytes = scale_bytes,
            .rope_off = rope_off,
            .rope_bytes = rope_bytes,
            .layer_stride = layer_stride,
        };
    }

    pub fn requiredBytes(spec: MlaFp4KvSpec) ArenaError!usize {
        const l = try computeLayout(spec);
        return std.math.mul(usize, l.layer_stride, spec.num_kv_layers) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: MlaFp4KvSpec) ArenaError!MlaFp4KvArena {
        const l = try computeLayout(spec);
        const total = std.math.mul(usize, l.layer_stride, spec.num_kv_layers) catch return error.Overflow;
        const backing = try arena.allocPages(total);
        @memset(backing, 0);
        return .{
            .spec = spec,
            .backing = backing,
            .base_offset = @intCast(arena.offsetOf(backing.ptr)),
            .layer_stride_bytes = @intCast(l.layer_stride),
            .e2m1_plane_offset = 0,
            .scale_plane_offset = @intCast(l.scale_off),
            .rope_plane_offset = @intCast(l.rope_off),
            .e2m1_plane_bytes = l.e2m1_bytes,
            .scale_plane_bytes = l.scale_bytes,
            .rope_plane_bytes = l.rope_bytes,
        };
    }

    pub fn bytes(self: *const MlaFp4KvArena) []align(page.APPLE_PAGE_SIZE) u8 {
        return self.backing;
    }

    pub fn layerBinding(self: *const MlaFp4KvArena, kv_slot: u32, buf: metal.Buffer) ArenaError!metal.BufferBinding {
        if (kv_slot >= self.spec.num_kv_layers) return error.InvalidLayerIndex;
        return .{
            .buffer = buf,
            .offset = self.base_offset + @as(u64, kv_slot) * self.layer_stride_bytes,
        };
    }

    /// Resolves the physical KV slab binding for any CSA2 layer mode:
    /// `Full` binds its own physical slot, while `Reindex`/`Reuse` alias
    /// `shared_kv_owner` with zero extra KV bytes.
    pub fn sharedLayerBinding(
        self: *const MlaFp4KvArena,
        mode: Csa2Mode,
        shared_kv_owner: u32,
        buf: metal.Buffer,
    ) ArenaError!metal.BufferBinding {
        _ = mode;
        return self.layerBinding(shared_kv_owner, buf);
    }

    pub fn layerE2m1Slice(self: *const MlaFp4KvArena, kv_slot: u32) ArenaError![]u8 {
        if (kv_slot >= self.spec.num_kv_layers) return error.InvalidLayerIndex;
        const base = @as(usize, kv_slot) * @as(usize, @intCast(self.layer_stride_bytes));
        return self.backing[base .. base + self.e2m1_plane_bytes];
    }

    pub fn layerScaleSlice(self: *const MlaFp4KvArena, kv_slot: u32) ArenaError![]u8 {
        if (kv_slot >= self.spec.num_kv_layers) return error.InvalidLayerIndex;
        const base = @as(usize, kv_slot) * @as(usize, @intCast(self.layer_stride_bytes)) +
            @as(usize, @intCast(self.scale_plane_offset));
        return self.backing[base .. base + self.scale_plane_bytes];
    }

    pub fn layerRopeSlice(self: *const MlaFp4KvArena, kv_slot: u32) ArenaError![]u8 {
        if (kv_slot >= self.spec.num_kv_layers) return error.InvalidLayerIndex;
        const base = @as(usize, kv_slot) * @as(usize, @intCast(self.layer_stride_bytes)) +
            @as(usize, @intCast(self.rope_plane_offset));
        return self.backing[base .. base + self.rope_plane_bytes];
    }
};

/// Descriptor for circular `f16` local SWA bounded replay rings paired with `MlaFp4KvArena`.
pub const MlaSwaRingSpec = struct {
    num_kv_layers: u32,
    swa_window: u32,
    kv_lora_rank: u32,
    qk_rope_dim: u32,

    pub fn validate(self: MlaSwaRingSpec) ArenaError!void {
        if (self.num_kv_layers == 0 or self.swa_window == 0 or self.kv_lora_rank == 0) {
            return error.ZeroDimension;
        }
        if ((self.kv_lora_rank % 32) != 0 or self.kv_lora_rank > MAX_MLA_LATENT_DIM) {
            return error.InvalidConfig;
        }
        if ((self.qk_rope_dim % 2) != 0 or self.qk_rope_dim > MAX_MLA_ROPE_DIM) {
            return error.InvalidConfig;
        }
    }
};

/// 16 KiB-aligned wired circular `f16` SWA replay ring (`[num_kv_layers, swa_window, ...]`).
pub const MlaSwaRingArena = struct {
    spec: MlaSwaRingSpec,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    base_offset: u64,
    layer_stride_bytes: u64,
    rope_plane_offset: u64,
    latent_plane_bytes: usize,
    rope_plane_bytes: usize,

    fn computeLayout(spec: MlaSwaRingSpec) ArenaError!struct {
        lat_bytes: usize,
        rope_off: usize,
        rope_bytes: usize,
        layer_stride: usize,
    } {
        try spec.validate();
        const lat_elems = std.math.mul(usize, spec.swa_window, spec.kv_lora_rank) catch return error.Overflow;
        const lat_bytes = std.math.mul(usize, lat_elems, @sizeOf(f16)) catch return error.Overflow;
        const rope_off = try alignUp16(lat_bytes);
        const rope_elems = std.math.mul(usize, spec.swa_window, spec.qk_rope_dim) catch return error.Overflow;
        const rope_bytes = std.math.mul(usize, rope_elems, @sizeOf(f16)) catch return error.Overflow;
        const raw_layer = std.math.add(usize, rope_off, rope_bytes) catch return error.Overflow;
        const layer_stride = page.alignUp(raw_layer) catch return error.Overflow;
        return .{
            .lat_bytes = lat_bytes,
            .rope_off = rope_off,
            .rope_bytes = rope_bytes,
            .layer_stride = layer_stride,
        };
    }

    pub fn requiredBytes(spec: MlaSwaRingSpec) ArenaError!usize {
        const l = try computeLayout(spec);
        return std.math.mul(usize, l.layer_stride, spec.num_kv_layers) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: MlaSwaRingSpec) ArenaError!MlaSwaRingArena {
        const l = try computeLayout(spec);
        const total = std.math.mul(usize, l.layer_stride, spec.num_kv_layers) catch return error.Overflow;
        const backing = try arena.allocPages(total);
        @memset(backing, 0);
        return .{
            .spec = spec,
            .backing = backing,
            .base_offset = @intCast(arena.offsetOf(backing.ptr)),
            .layer_stride_bytes = @intCast(l.layer_stride),
            .rope_plane_offset = @intCast(l.rope_off),
            .latent_plane_bytes = l.lat_bytes,
            .rope_plane_bytes = l.rope_bytes,
        };
    }

    pub fn layerBinding(self: *const MlaSwaRingArena, kv_slot: u32, buf: metal.Buffer) ArenaError!metal.BufferBinding {
        if (kv_slot >= self.spec.num_kv_layers) return error.InvalidLayerIndex;
        return .{
            .buffer = buf,
            .offset = self.base_offset + @as(u64, kv_slot) * self.layer_stride_bytes,
        };
    }
};

/// Descriptor for two-stage CSA2 hierarchical block summaries (fine + coarse).
pub const Csa2SummarySpec = struct {
    num_kv_layers: u32,
    max_tokens: u32,
    index_dim: u32,
    block_tokens: u32 = CSA2_DEFAULT_BLOCK_TOKENS,
    coarse_group_blocks: u32 = CSA2_DEFAULT_COARSE_GROUP_BLOCKS,

    pub fn validate(self: Csa2SummarySpec) ArenaError!void {
        if (self.num_kv_layers == 0 or self.max_tokens == 0 or self.index_dim == 0 or
            self.block_tokens == 0 or self.coarse_group_blocks == 0)
        {
            return error.ZeroDimension;
        }
        if ((self.index_dim % MLA_FP4_GROUP_SIZE) != 0 or self.index_dim > MAX_MLA_LATENT_DIM) {
            return error.InvalidConfig;
        }
    }

    pub fn maxFineBlocks(self: Csa2SummarySpec) u32 {
        return (self.max_tokens + self.block_tokens - 1) / self.block_tokens;
    }

    pub fn maxCoarseBlocks(self: Csa2SummarySpec) u32 {
        const fine = self.maxFineBlocks();
        return (fine + self.coarse_group_blocks - 1) / self.coarse_group_blocks;
    }
};

/// 16 KiB-aligned wired arena holding CSA2 fine & coarse `f16` block summaries per physical `Full` KV layer.
pub const Csa2SummaryArena = struct {
    spec: Csa2SummarySpec,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    base_offset: u64,
    layer_stride_bytes: u64,
    fine_plane_offset: u64,
    coarse_plane_offset: u64,
    fine_plane_bytes: usize,
    coarse_plane_bytes: usize,

    fn computeLayout(spec: Csa2SummarySpec) ArenaError!struct {
        fine_bytes: usize,
        coarse_off: usize,
        coarse_bytes: usize,
        layer_stride: usize,
    } {
        try spec.validate();
        const fine_elems = std.math.mul(usize, spec.maxFineBlocks(), spec.index_dim) catch return error.Overflow;
        const fine_bytes = std.math.mul(usize, fine_elems, @sizeOf(f16)) catch return error.Overflow;
        const coarse_off = try alignUp16(fine_bytes);
        const coarse_elems = std.math.mul(usize, spec.maxCoarseBlocks(), spec.index_dim) catch return error.Overflow;
        const coarse_bytes = std.math.mul(usize, coarse_elems, @sizeOf(f16)) catch return error.Overflow;
        const raw_layer = std.math.add(usize, coarse_off, coarse_bytes) catch return error.Overflow;
        const layer_stride = page.alignUp(raw_layer) catch return error.Overflow;
        return .{
            .fine_bytes = fine_bytes,
            .coarse_off = coarse_off,
            .coarse_bytes = coarse_bytes,
            .layer_stride = layer_stride,
        };
    }

    pub fn requiredBytes(spec: Csa2SummarySpec) ArenaError!usize {
        const l = try computeLayout(spec);
        return std.math.mul(usize, l.layer_stride, spec.num_kv_layers) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: Csa2SummarySpec) ArenaError!Csa2SummaryArena {
        const l = try computeLayout(spec);
        const total = std.math.mul(usize, l.layer_stride, spec.num_kv_layers) catch return error.Overflow;
        const backing = try arena.allocPages(total);
        @memset(backing, 0);
        return .{
            .spec = spec,
            .backing = backing,
            .base_offset = @intCast(arena.offsetOf(backing.ptr)),
            .layer_stride_bytes = @intCast(l.layer_stride),
            .fine_plane_offset = 0,
            .coarse_plane_offset = @intCast(l.coarse_off),
            .fine_plane_bytes = l.fine_bytes,
            .coarse_plane_bytes = l.coarse_bytes,
        };
    }

    pub fn layerBinding(self: *const Csa2SummaryArena, kv_slot: u32, buf: metal.Buffer) ArenaError!metal.BufferBinding {
        if (kv_slot >= self.spec.num_kv_layers) return error.InvalidLayerIndex;
        return .{
            .buffer = buf,
            .offset = self.base_offset + @as(u64, kv_slot) * self.layer_stride_bytes,
        };
    }

    pub fn fineBinding(self: *const Csa2SummaryArena, kv_slot: u32, buf: metal.Buffer) ArenaError!metal.BufferBinding {
        if (kv_slot >= self.spec.num_kv_layers) return error.InvalidLayerIndex;
        return .{
            .buffer = buf,
            .offset = self.base_offset + @as(u64, kv_slot) * self.layer_stride_bytes + self.fine_plane_offset,
        };
    }

    pub fn coarseBinding(self: *const Csa2SummaryArena, kv_slot: u32, buf: metal.Buffer) ArenaError!metal.BufferBinding {
        if (kv_slot >= self.spec.num_kv_layers) return error.InvalidLayerIndex;
        return .{
            .buffer = buf,
            .offset = self.base_offset + @as(u64, kv_slot) * self.layer_stride_bytes + self.coarse_plane_offset,
        };
    }
};

/// Descriptor for `Csa2IndexBuffer` (shared across heads, reusable across `Reuse` layers and `r >= 1` loops).
pub const Csa2IndexBufferSpec = struct {
    max_nodes: u32 = MAX_WIDE_PREFILL_TOKENS,
    top_k_blocks: u32 = 16,
};

/// 16 KiB-aligned wired CSA2 sparse block index buffer (`selected_blocks` + `selected_counts`).
pub const Csa2IndexBuffer = struct {
    spec: Csa2IndexBufferSpec,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    blocks_offset: u64,
    counts_offset: u64,
    blocks: []u32,
    counts: []u32,

    pub fn validateSpec(spec: Csa2IndexBufferSpec) ArenaError!void {
        if (spec.max_nodes == 0 or spec.top_k_blocks == 0) return error.ZeroDimension;
        if (spec.max_nodes > MAX_WIDE_PREFILL_TOKENS or spec.top_k_blocks > CSA2_MAX_TOP_K_BLOCKS) {
            return error.InvalidConfig;
        }
    }

    pub fn requiredBytes(spec: Csa2IndexBufferSpec) ArenaError!usize {
        try validateSpec(spec);
        const blocks_elems = std.math.mul(usize, spec.max_nodes, spec.top_k_blocks) catch return error.Overflow;
        const blocks_bytes = std.math.mul(usize, blocks_elems, @sizeOf(u32)) catch return error.Overflow;
        const counts_off = try alignUp16(blocks_bytes);
        const counts_bytes = std.math.mul(usize, spec.max_nodes, @sizeOf(u32)) catch return error.Overflow;
        const raw = std.math.add(usize, counts_off, counts_bytes) catch return error.Overflow;
        return page.alignUp(raw) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: Csa2IndexBufferSpec) ArenaError!Csa2IndexBuffer {
        try validateSpec(spec);
        const blocks_elems = std.math.mul(usize, spec.max_nodes, spec.top_k_blocks) catch return error.Overflow;
        const blocks_bytes = std.math.mul(usize, blocks_elems, @sizeOf(u32)) catch return error.Overflow;
        const counts_off = try alignUp16(blocks_bytes);
        const counts_bytes = std.math.mul(usize, spec.max_nodes, @sizeOf(u32)) catch return error.Overflow;
        const total = try requiredBytes(spec);

        const backing = try arena.allocPages(total);
        @memset(backing, 0);
        const base_off: u64 = @intCast(arena.offsetOf(backing.ptr));
        return .{
            .spec = spec,
            .backing = backing,
            .blocks_offset = base_off,
            .counts_offset = base_off + @as(u64, @intCast(counts_off)),
            .blocks = @alignCast(std.mem.bytesAsSlice(u32, backing[0..blocks_bytes])),
            .counts = @alignCast(std.mem.bytesAsSlice(u32, backing[counts_off .. counts_off + counts_bytes])),
        };
    }

    pub fn blocksBinding(self: *const Csa2IndexBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.blocks_offset };
    }

    pub fn countsBinding(self: *const Csa2IndexBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.counts_offset };
    }
};

pub const MlaFp4KvWriteArgs = struct {
    /// `[num_tokens, kv_lora_rank]` `f16` latent vectors.
    latent_in: metal.BufferBinding,
    /// `[num_tokens, qk_rope_dim]` `f16` RoPE key vectors (optional if `qk_rope_dim == 0`).
    rope_in: ?metal.BufferBinding = null,
    /// Destination `MlaFp4KvArena` and owning physical `kv_slot`.
    kv_arena: *const MlaFp4KvArena,
    kv_metal_buf: metal.Buffer,
    kv_slot: u32,
    /// Optional circular `f16` SWA replay ring for exact recent-window replay.
    swa_ring: ?*const MlaSwaRingArena = null,
    swa_metal_buf: ?metal.Buffer = null,
    num_tokens: u32,
    /// Logical token position of token 0 (`0` for CED encoder->decoder global KV projection,
    /// or current `kv_len` for causal prefill/decode append).
    start_pos: u32,
    /// Set `true` when projecting encoder final hidden states into global decoder KV (CED).
    is_encoder_projection: bool = false,
};

pub const Csa2UpdateSummariesArgs = struct {
    kv_arena: *const MlaFp4KvArena,
    kv_metal_buf: metal.Buffer,
    summary_arena: *const Csa2SummaryArena,
    summary_metal_buf: metal.Buffer,
    kv_slot: u32,
    kv_len: u32,
    /// First coarse super-block to update (`0` for full rebuild; `(start_pos / (block_tokens * coarse_group_blocks))` for incremental append).
    start_coarse_block: u32 = 0,
    /// Optional explicit coarse block count; defaults to updating through the end of `kv_len`.
    num_coarse_blocks: ?u32 = null,
};

pub const Csa2IndexBlocksArgs = struct {
    /// `[num_nodes, num_q_heads, kv_lora_rank]` `f16` absorbed query vectors.
    q_nope: metal.BufferBinding,
    summary_arena: *const Csa2SummaryArena,
    summary_metal_buf: metal.Buffer,
    kv_slot: u32,
    index_buf: *const Csa2IndexBuffer,
    index_metal_buf: metal.Buffer,
    num_nodes: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    kv_len: u32,
    /// CED encoder prefix length `[0, encoder_prefix_len)` always visible in decoder mode (`0` for causal decoder-only).
    encoder_prefix_len: u32 = 0,
    /// Causal query start position (`null` = full non-causal visibility up to `kv_len`).
    causal_q_start_pos: ?u32 = null,
    /// SWA bounded window size `W` (`0` = disabled; `> 0` = exclude `[eff_kv_len - W, eff_kv_len)` from sparse budget).
    swa_window: u32 = 0,
    top_k_blocks: u32 = 16,
    /// Stage-1 coarse super-block shortlist size (`0` = auto `ceil(top_k_blocks / coarse_group_blocks) + 1`).
    top_k_coarse: u32 = 0,
    /// Mandatory local recent blocks when `swa_window == 0`.
    num_local_blocks: u32 = 1,
    /// Optional per-slot `kv_len[num_nodes]` (`u32`) for batched decode (`M = 1..16`).
    batch_kv_lens: ?metal.BufferBinding = null,
    scale: ?f32 = null,
};

pub const MlaCsa2AttendArgs = struct {
    /// `[num_nodes, num_q_heads, kv_lora_rank]` `f16` absorbed query vectors.
    q_nope: metal.BufferBinding,
    /// `[num_nodes, num_q_heads, qk_rope_dim]` `f16` RoPE query vectors (optional if `qk_rope_dim == 0`).
    q_pe: ?metal.BufferBinding = null,
    kv_arena: *const MlaFp4KvArena,
    kv_metal_buf: metal.Buffer,
    kv_slot: u32,
    /// Optional circular `f16` SWA replay ring for exact recent-window replay (`swa_window > 0`).
    swa_ring: ?*const MlaSwaRingArena = null,
    swa_metal_buf: ?metal.Buffer = null,
    index_buf: *const Csa2IndexBuffer,
    index_metal_buf: metal.Buffer,
    /// Optional speculative tree latent keys `[num_nodes, kv_lora_rank]` `f16`.
    kv_tree_latent: ?metal.BufferBinding = null,
    /// Optional speculative tree RoPE keys `[num_nodes, qk_rope_dim]` `f16`.
    kv_tree_rope: ?metal.BufferBinding = null,
    /// Optional speculative tree 64-bit ancestor masks `[num_nodes]` `u64`.
    ancestor_masks: ?metal.BufferBinding = null,
    /// Optional per-slot `kv_len[num_nodes]` (`u32`) for batched decode (`M = 1..16`).
    batch_kv_lens: ?metal.BufferBinding = null,
    /// `[num_nodes, num_q_heads, kv_lora_rank]` `f16` absorbed attention output.
    out: metal.BufferBinding,
    num_nodes: u32,
    num_q_heads: u32,
    kv_len: u32,
    encoder_prefix_len: u32 = 0,
    causal_q_start_pos: ?u32 = null,
    swa_window: u32 = 0,
    block_tokens: u32 = CSA2_DEFAULT_BLOCK_TOKENS,
    top_k_blocks: u32 = 16,
    scale: ?f32 = null,
    /// Optional `G19` MMA override (`null` enforces strict `Q_len >= 32 && KV_len >= 64` guard).
    use_mma_override: ?bool = null,
};

/// Dispatch telemetry returned by `encodeMlaCsa2Layer` verifying zero duplicate KV
/// writes on `Reindex`/`Reuse` and zero indexer dispatches on `Reuse` / `r >= 1`.
pub const MlaCsa2StepDispatches = struct {
    kv_writes: u32 = 0,
    summary_updates: u32 = 0,
    index_dispatches: u32 = 0,
    attn_dispatches: u32 = 0,
};

pub const MlaCsa2LayerArgs = struct {
    mode: Csa2Mode,
    /// Physical `MlaFp4KvArena` slot of this layer (`Full`) or its `shared_kv_owner` (`Reindex`/`Reuse`).
    shared_kv_owner: u32,
    /// Recurrent-depth loop step (`ADR-0019`): `0` writes KV & computes sparse indices;
    /// `r >= 1` reuses KV & `index_buf` with zero duplicate writes/index dispatches.
    loop_step: u32 = 0,
    /// Optional new KV tokens to write on `Full` layers when `loop_step == 0`
    /// (ignored with `0` KV writes on `Reindex`/`Reuse` and `loop_step >= 1`).
    write_latent_in: ?metal.BufferBinding = null,
    write_rope_in: ?metal.BufferBinding = null,
    write_start_pos: u32 = 0,
    write_num_tokens: u32 = 0,
    is_encoder_projection: bool = false,
    /// Attention & indexer inputs.
    q_nope: metal.BufferBinding,
    q_pe: ?metal.BufferBinding = null,
    kv_arena: *const MlaFp4KvArena,
    kv_metal_buf: metal.Buffer,
    swa_ring: ?*const MlaSwaRingArena = null,
    swa_metal_buf: ?metal.Buffer = null,
    summary_arena: ?*const Csa2SummaryArena = null,
    summary_metal_buf: ?metal.Buffer = null,
    index_buf: *const Csa2IndexBuffer,
    index_metal_buf: metal.Buffer,
    kv_tree_latent: ?metal.BufferBinding = null,
    kv_tree_rope: ?metal.BufferBinding = null,
    ancestor_masks: ?metal.BufferBinding = null,
    batch_kv_lens: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    num_nodes: u32,
    num_q_heads: u32,
    kv_len: u32,
    encoder_prefix_len: u32 = 0,
    causal_q_start_pos: ?u32 = null,
    swa_window: u32 = 0,
    block_tokens: u32 = CSA2_DEFAULT_BLOCK_TOKENS,
    top_k_blocks: u32 = 16,
    top_k_coarse: u32 = 0,
    num_local_blocks: u32 = 1,
    scale: ?f32 = null,
};

// ===========================================================================
// Section F: M7 (G10) — Single-Pass Mega-mHC (`mhc_single_pass`)
// ===========================================================================

/// Configuration for Single-Pass Mega-mHC (`G10`, `2..8` hyper-streams, default `4`).
pub const MhcConfig = struct {
    num_streams: u32 = DEFAULT_MHC_STREAMS,
    hidden_size: u32,
    stream_dtype: quant.FloatDType = .f16,
    act_dtype: quant.FloatDType = .f16,
    norm_dtype: quant.FloatDType = .f16,
    rms_eps: f32 = 1e-6,
    /// `0` uses `h_res` weights directly; `> 0` projects `exp(h_res)` onto the Birkhoff
    /// doubly-stochastic manifold via `sinkhorn_iters` Sinkhorn-Knopp iterations in-kernel.
    sinkhorn_iters: u32 = 0,
    /// When `true`, applies `sigmoid(h_pre)` / `post_scale * sigmoid(h_post)` in-kernel.
    activate_coeffs: bool = false,
    /// When `true`, broadcasts `[1, num_streams]` / `[1, num_streams, num_streams]` across all `M` tokens.
    shared_coeffs: bool = false,
    /// Apply `H_res` stream-to-stream mixing in-place during `encodeMegaMhc`.
    apply_h_res_pre: bool = false,
    /// Apply `H_res` stream-to-stream mixing in-place during `encodeMegaMhcPostMix`.
    apply_h_res_post: bool = true,
    /// Post-mixing scale multiplier (`1.0` default; `2.0` for `2 * sigmoid` DeepSeek mHC).
    post_scale: f32 = 1.0,

    pub fn validate(self: MhcConfig) EncodeError!void {
        if (self.hidden_size == 0) return error.ZeroDimension;
        if (self.num_streams < MIN_MHC_STREAMS or self.num_streams > MAX_MHC_STREAMS) {
            return error.InvalidConfig;
        }
        if (self.rms_eps < 0.0 or !std.math.isFinite(self.rms_eps)) return error.InvalidConfig;
        if (!std.math.isFinite(self.post_scale)) return error.InvalidConfig;
    }
};

pub const MhcStreamSpec = struct {
    max_tokens: u32,
    num_streams: u32 = DEFAULT_MHC_STREAMS,
    hidden_size: u32,
    dtype: quant.FloatDType = .f16,
};

/// 16 KiB-aligned wired `[max_tokens, num_streams, hidden_size]` Mega-mHC stream state buffer.
pub const MhcStreamArena = struct {
    spec: MhcStreamSpec,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    base_offset: u64,

    pub fn requiredBytes(spec: MhcStreamSpec) ArenaError!usize {
        if (spec.max_tokens == 0 or spec.hidden_size == 0) return error.ZeroDimension;
        if (spec.num_streams < MIN_MHC_STREAMS or spec.num_streams > MAX_MHC_STREAMS) {
            return error.InvalidConfig;
        }
        const per_tok = std.math.mul(usize, spec.num_streams, spec.hidden_size) catch return error.Overflow;
        const total_elems = std.math.mul(usize, per_tok, spec.max_tokens) catch return error.Overflow;
        const total_bytes = std.math.mul(usize, total_elems, 2) catch return error.Overflow;
        return page.alignUp(total_bytes) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: MhcStreamSpec) ArenaError!MhcStreamArena {
        const total = try requiredBytes(spec);
        const backing = try arena.allocPages(total);
        @memset(backing, 0);
        return .{
            .spec = spec,
            .backing = backing,
            .base_offset = @intCast(arena.offsetOf(backing.ptr)),
        };
    }

    pub fn binding(self: *const MhcStreamArena, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.base_offset };
    }
};

pub const MegaMhcArgs = struct {
    config: MhcConfig,
    /// `[m, num_streams, hidden_size]` (`stream_dtype`), read and optionally mixed in-place.
    streams: metal.BufferBinding,
    /// `[m, num_streams]` (or `[1, num_streams]` if `shared_coeffs`) `f32` pre-mix weights.
    h_pre: metal.BufferBinding,
    /// Optional `[m, num_streams, num_streams]` `f32` residual mixing matrix (required if `apply_h_res_pre`).
    h_res: ?metal.BufferBinding = null,
    /// Optional `[hidden_size]` (`norm_dtype`) RMSNorm weights fused into branch output.
    norm_w: ?metal.BufferBinding = null,
    /// `[m, hidden_size]` (`act_dtype`) mixed (and optionally RMS-normalized) sublayer input.
    branch_out: metal.BufferBinding,
    m: u32,
};

pub const MegaMhcPostMixArgs = struct {
    config: MhcConfig,
    /// `[m, num_streams, hidden_size]` (`stream_dtype`), updated in-place with zero extra intermediates.
    streams: metal.BufferBinding,
    /// `[m, hidden_size]` (`act_dtype`) Attention/FFN sublayer output.
    sublayer_out: metal.BufferBinding,
    /// `[m, num_streams]` (or `[1, num_streams]` if `shared_coeffs`) `f32` post-mix weights.
    h_post: metal.BufferBinding,
    /// Optional `[m, num_streams, num_streams]` `f32` residual mixing matrix (required if `apply_h_res_post`).
    h_res: ?metal.BufferBinding = null,
    m: u32,
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
    gdn_batch_decode_d64_f16: metal.Pipeline,
    gdn_batch_decode_d64_bf16: metal.Pipeline,
    gdn_batch_decode_d128_f16: metal.Pipeline,
    gdn_batch_decode_d128_bf16: metal.Pipeline,
    gdn_batch_decode_dyn_f16: metal.Pipeline,
    gdn_batch_decode_dyn_bf16: metal.Pipeline,
    gdn_clone_state: metal.Pipeline,
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
    // M6.8 (G15, ADR-0020): wide-chunk QSA & 128-token SWA ring prefill
    qsa_prefill_rope_f16: metal.Pipeline,
    qsa_prefill_rope_bf16: metal.Pipeline,
    qsa_prefill_kv_ring_commit: metal.Pipeline,
    qsa_swa_prefill_attn_d64: metal.Pipeline,
    qsa_swa_prefill_attn_d128: metal.Pipeline,
    qsa_swa_prefill_attn_dyn: metal.Pipeline,
    // M7.8 (G19, ADR-0023): hardware `simdgroup_half8x8` QSA prefill
    qsa_swa_prefill_attn_mma_d64: metal.Pipeline,
    qsa_swa_prefill_attn_mma_d128: metal.Pipeline,

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
    gdn_conv1d_q_silu: metal.Pipeline,
    gdn_conv1d_commit: metal.Pipeline,
    blend_f16: metal.Pipeline,

    // M7 (R7 + G11): MLA + CSA2 + FP4 KV + SWA bounded replay
    mla_fp4_kv_write: metal.Pipeline,
    csa2_update_summaries: metal.Pipeline,
    csa2_index_blocks: metal.Pipeline,
    mla_csa2_attend: metal.Pipeline,
    // M7.8 (G19, ADR-0023): hardware `simdgroup_half8x8` MLA prefill
    mla_csa2_attend_mma: metal.Pipeline,

    // M7 (G10): Single-Pass Mega-mHC (`[stream_bf16][act_bf16]`)
    mhc_single_pass: [2][2]metal.Pipeline,
    mhc_post_mix: [2][2]metal.Pipeline,

    // M9.1 (`G26`): Asymmetric K/V (`192/128`, `256/256`, dynamic <= 256) `HybridKvRing`
    // SWA (`128 + sink`), partial/3D-MRoPE + `(1+w)` QK-norm + `v_scale`, sigmoid MoE
    // routing (`e_score_correction_bias`), and `DFlash` tap ring capture.
    g26_qknorm_rope_kv_f16: metal.Pipeline,
    g26_qknorm_rope_kv_bf16: metal.Pipeline,
    g26_kv_commit: metal.Pipeline,
    g26_swa_attn_192_128: metal.Pipeline,
    g26_swa_attn_256_256: metal.Pipeline,
    g26_swa_attn_dyn: metal.Pipeline,
    g26_moe_route_sigmoid_f32: metal.Pipeline,
    g26_moe_route_sigmoid_f16: metal.Pipeline,
    g26_capture_tap_ring: metal.Pipeline,

    // M9.2 (`G27`): GLM-5.2/5.3 (`glm_moe_dsa`) & GLM-5.3-Flash (`glm5_next`)
    // Single-kernel 20-step `hc_mult = 4` mHC Sinkhorn, Kimi Delta Attention (KDA),
    // and `GlmMoeDsaIndexer` + `NoPE MLA` (`index_kpool = 4`) sparse attention.
    g27_mhc_pre_sinkhorn: [2][2]metal.Pipeline,
    g27_mhc_post_mix: [2][2]metal.Pipeline,
    kda_conv1d_silu: metal.Pipeline,
    kda_compute_gates: metal.Pipeline,
    kda_decode_d64_f16: metal.Pipeline,
    kda_decode_d64_bf16: metal.Pipeline,
    kda_decode_d128_f16: metal.Pipeline,
    kda_decode_d128_bf16: metal.Pipeline,
    kda_decode_dyn_f16: metal.Pipeline,
    kda_decode_dyn_bf16: metal.Pipeline,
    kda_prefill_d64: metal.Pipeline,
    kda_prefill_d128: metal.Pipeline,
    kda_prefill_dyn: metal.Pipeline,
    kda_tree_verify_d64: metal.Pipeline,
    kda_tree_verify_d128: metal.Pipeline,
    kda_tree_verify_dyn: metal.Pipeline,
    kda_out_norm_gate_f16: metal.Pipeline,
    kda_out_norm_gate_bf16: metal.Pipeline,
    kda_commit_tree: metal.Pipeline,
    g27_rope_interleave: metal.Pipeline,
    g27_layernorm: metal.Pipeline,
    g27_mla_kv_write: metal.Pipeline,
    g27_dsa_idx_kv_write: metal.Pipeline,
    g27_dsa_kpool_build: metal.Pipeline,
    g27_dsa_scores: metal.Pipeline,
    g27_dsa_select_topk: metal.Pipeline,
    g27_mla_sparse_attend: metal.Pipeline,

    // M9.3 (`G28`): Full-causal absorbed MLA + head-wise/element-wise sigmoid output gate (`bailing_hybrid`).
    g28_mla_full_attend: metal.Pipeline,
    g28_mla_out_gate: metal.Pipeline,

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

        const gdn_batch_decode_d64_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_batch_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer gdn_batch_decode_d64_f16.close();

        const gdn_batch_decode_d64_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_batch_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer gdn_batch_decode_d64_bf16.close();

        const gdn_batch_decode_d128_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_batch_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer gdn_batch_decode_d128_f16.close();

        const gdn_batch_decode_d128_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_batch_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer gdn_batch_decode_d128_bf16.close();

        const gdn_batch_decode_dyn_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_batch_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer gdn_batch_decode_dyn_f16.close();

        const gdn_batch_decode_dyn_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_gdn_batch_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer gdn_batch_decode_dyn_bf16.close();

        const gdn_clone_state = try metal.Pipeline.init(device, library, "ami_gdn_clone_state");
        errdefer gdn_clone_state.close();

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
        const gdn_conv1d_q_silu = try metal.Pipeline.init(device, library, "ami_gdn_conv1d_q_silu");
        errdefer gdn_conv1d_q_silu.close();
        const gdn_conv1d_commit = try metal.Pipeline.init(device, library, "ami_gdn_conv1d_commit");
        errdefer gdn_conv1d_commit.close();
        const blend_f16 = try metal.Pipeline.init(device, library, "ami_blend_f16");
        errdefer blend_f16.close();

        const qsa_prefill_rope_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_prefill_qknorm_mrope", &.{
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer qsa_prefill_rope_f16.close();

        const qsa_prefill_rope_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_prefill_qknorm_mrope", &.{
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer qsa_prefill_rope_bf16.close();

        const qsa_prefill_kv_ring_commit = try metal.Pipeline.init(device, library, "ami_qsa_prefill_kv_ring_commit");
        errdefer qsa_prefill_kv_ring_commit.close();

        const qsa_swa_prefill_attn_d64 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_swa_prefill_attn", &.{
            .{ .index = 2, .value = .{ .u32 = 64 } },
        });
        errdefer qsa_swa_prefill_attn_d64.close();

        const qsa_swa_prefill_attn_d128 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_swa_prefill_attn", &.{
            .{ .index = 2, .value = .{ .u32 = 128 } },
        });
        errdefer qsa_swa_prefill_attn_d128.close();

        const qsa_swa_prefill_attn_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_swa_prefill_attn", &.{
            .{ .index = 2, .value = .{ .u32 = 0 } },
        });
        errdefer qsa_swa_prefill_attn_dyn.close();

        const qsa_swa_prefill_attn_mma_d64 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_swa_prefill_attn_mma", &.{
            .{ .index = 2, .value = .{ .u32 = 64 } },
        });
        errdefer qsa_swa_prefill_attn_mma_d64.close();

        const qsa_swa_prefill_attn_mma_d128 = try metal.Pipeline.initSpecialized(device, library, "ami_qsa_swa_prefill_attn_mma", &.{
            .{ .index = 2, .value = .{ .u32 = 128 } },
        });
        errdefer qsa_swa_prefill_attn_mma_d128.close();

        const mla_fp4_kv_write = try metal.Pipeline.init(device, library, "ami_mla_fp4_kv_write");
        errdefer mla_fp4_kv_write.close();

        const csa2_update_summaries = try metal.Pipeline.init(device, library, "ami_csa2_update_summaries");
        errdefer csa2_update_summaries.close();

        const csa2_index_blocks = try metal.Pipeline.init(device, library, "ami_csa2_index_blocks");
        errdefer csa2_index_blocks.close();

        const mla_csa2_attend = try metal.Pipeline.init(device, library, "ami_mla_csa2_attend");
        errdefer mla_csa2_attend.close();

        const mla_csa2_attend_mma = try metal.Pipeline.init(device, library, "ami_mla_csa2_attend_mma");
        errdefer mla_csa2_attend_mma.close();

        var mhc_single_pass: [2][2]metal.Pipeline = undefined;
        var mhc_post_mix: [2][2]metal.Pipeline = undefined;
        var mhc_built: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < mhc_built) : (idx += 1) {
                const kind = idx / 4;
                const pair = idx % 4;
                const sb = pair / 2;
                const ab = pair % 2;
                switch (kind) {
                    0 => mhc_single_pass[sb][ab].close(),
                    1 => mhc_post_mix[sb][ab].close(),
                    else => {},
                }
            }
        }
        const mhc_names = [2][:0]const u8{
            "ami_mhc_single_pass",
            "ami_mhc_post_mix",
        };
        for (mhc_names, 0..) |fn_name, kind| {
            for (0..2) |sb| {
                for (0..2) |ab| {
                    const p = try metal.Pipeline.initSpecialized(device, library, fn_name, &.{
                        .{ .index = 3, .value = .{ .bool = sb == 1 } },
                        .{ .index = 4, .value = .{ .bool = ab == 1 } },
                    });
                    switch (kind) {
                        0 => mhc_single_pass[sb][ab] = p,
                        1 => mhc_post_mix[sb][ab] = p,
                        else => unreachable,
                    }
                    mhc_built += 1;
                }
            }
        }

        const g26_qknorm_rope_kv_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_g26_qknorm_rope_kv", &.{
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer g26_qknorm_rope_kv_f16.close();
        const g26_qknorm_rope_kv_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_g26_qknorm_rope_kv", &.{
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer g26_qknorm_rope_kv_bf16.close();
        const g26_kv_commit = try metal.Pipeline.init(device, library, "ami_g26_kv_commit");
        errdefer g26_kv_commit.close();
        const g26_swa_attn_192_128 = try metal.Pipeline.initSpecialized(device, library, "ami_g26_swa_attn", &.{
            .{ .index = 5, .value = .{ .u32 = 192 } },
            .{ .index = 6, .value = .{ .u32 = 128 } },
        });
        errdefer g26_swa_attn_192_128.close();
        const g26_swa_attn_256_256 = try metal.Pipeline.initSpecialized(device, library, "ami_g26_swa_attn", &.{
            .{ .index = 5, .value = .{ .u32 = 256 } },
            .{ .index = 6, .value = .{ .u32 = 256 } },
        });
        errdefer g26_swa_attn_256_256.close();
        const g26_swa_attn_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_g26_swa_attn", &.{
            .{ .index = 5, .value = .{ .u32 = 0 } },
            .{ .index = 6, .value = .{ .u32 = 0 } },
        });
        errdefer g26_swa_attn_dyn.close();
        const g26_moe_route_sigmoid_f32 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_route_topk_sigmoid", &.{
            .{ .index = 7, .value = .{ .bool = false } },
        });
        errdefer g26_moe_route_sigmoid_f32.close();
        const g26_moe_route_sigmoid_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_moe_route_topk_sigmoid", &.{
            .{ .index = 7, .value = .{ .bool = true } },
        });
        errdefer g26_moe_route_sigmoid_f16.close();
        const g26_capture_tap_ring = try metal.Pipeline.init(device, library, "ami_g26_capture_tap_ring");
        errdefer g26_capture_tap_ring.close();

        var g27_mhc_pre_sinkhorn: [2][2]metal.Pipeline = undefined;
        var g27_mhc_post_mix: [2][2]metal.Pipeline = undefined;
        var g27_mhc_built: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < g27_mhc_built) : (idx += 1) {
                const kind = idx / 4;
                const pair = idx % 4;
                const sb = pair / 2;
                const ab = pair % 2;
                if (kind == 0) g27_mhc_pre_sinkhorn[sb][ab].close() else g27_mhc_post_mix[sb][ab].close();
            }
        }
        const g27_mhc_names = [2][:0]const u8{ "ami_g27_mhc_pre_sinkhorn", "ami_g27_mhc_post_mix" };
        for (g27_mhc_names, 0..) |fn_name, kind| {
            for (0..2) |sb| {
                for (0..2) |ab| {
                    const p = try metal.Pipeline.initSpecialized(device, library, fn_name, &.{
                        .{ .index = 3, .value = .{ .bool = sb == 1 } },
                        .{ .index = 4, .value = .{ .bool = ab == 1 } },
                    });
                    if (kind == 0) g27_mhc_pre_sinkhorn[sb][ab] = p else g27_mhc_post_mix[sb][ab] = p;
                    g27_mhc_built += 1;
                }
            }
        }

        const kda_conv1d_silu = try metal.Pipeline.init(device, library, "ami_kda_conv1d_silu");
        errdefer kda_conv1d_silu.close();
        const kda_compute_gates = try metal.Pipeline.init(device, library, "ami_kda_compute_gates");
        errdefer kda_compute_gates.close();
        const kda_decode_d64_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer kda_decode_d64_f16.close();
        const kda_decode_d64_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer kda_decode_d64_bf16.close();
        const kda_decode_d128_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer kda_decode_d128_f16.close();
        const kda_decode_d128_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer kda_decode_d128_bf16.close();
        const kda_decode_dyn_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer kda_decode_dyn_f16.close();
        const kda_decode_dyn_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_decode", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer kda_decode_dyn_bf16.close();
        const kda_prefill_d64 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_prefill", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
        });
        errdefer kda_prefill_d64.close();
        const kda_prefill_d128 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_prefill", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
        });
        errdefer kda_prefill_d128.close();
        const kda_prefill_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_kda_prefill", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
        });
        errdefer kda_prefill_dyn.close();
        const kda_tree_verify_d64 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_tree_verify", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
        });
        errdefer kda_tree_verify_d64.close();
        const kda_tree_verify_d128 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_tree_verify", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
        });
        errdefer kda_tree_verify_d128.close();
        const kda_tree_verify_dyn = try metal.Pipeline.initSpecialized(device, library, "ami_kda_tree_verify", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
        });
        errdefer kda_tree_verify_dyn.close();
        const kda_out_norm_gate_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_out_norm_gate", &.{
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer kda_out_norm_gate_f16.close();
        const kda_out_norm_gate_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_kda_out_norm_gate", &.{
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer kda_out_norm_gate_bf16.close();
        const kda_commit_tree = try metal.Pipeline.initSpecialized(device, library, "ami_kda_commit_tree", &.{
            .{ .index = 0, .value = .{ .u32 = 0 } },
        });
        errdefer kda_commit_tree.close();
        const g27_rope_interleave = try metal.Pipeline.init(device, library, "ami_g27_rope_interleave");
        errdefer g27_rope_interleave.close();
        const g27_layernorm = try metal.Pipeline.init(device, library, "ami_g27_layernorm");
        errdefer g27_layernorm.close();
        const g27_mla_kv_write = try metal.Pipeline.init(device, library, "ami_g27_mla_kv_write");
        errdefer g27_mla_kv_write.close();
        const g27_dsa_idx_kv_write = try metal.Pipeline.init(device, library, "ami_g27_dsa_idx_kv_write");
        errdefer g27_dsa_idx_kv_write.close();
        const g27_dsa_kpool_build = try metal.Pipeline.init(device, library, "ami_g27_dsa_kpool_build");
        errdefer g27_dsa_kpool_build.close();
        const g27_dsa_scores = try metal.Pipeline.init(device, library, "ami_g27_dsa_scores");
        errdefer g27_dsa_scores.close();
        const g27_dsa_select_topk = try metal.Pipeline.init(device, library, "ami_g27_dsa_select_topk");
        errdefer g27_dsa_select_topk.close();
        const g27_mla_sparse_attend = try metal.Pipeline.init(device, library, "ami_g27_mla_sparse_attend");
        errdefer g27_mla_sparse_attend.close();
        const g28_mla_full_attend = try metal.Pipeline.init(device, library, "ami_g28_mla_full_attend");
        errdefer g28_mla_full_attend.close();
        const g28_mla_out_gate = try metal.Pipeline.init(device, library, "ami_g28_mla_out_gate");
        errdefer g28_mla_out_gate.close();

        return .{
            .gdn_decode_d64_f16 = gdn_decode_d64_f16,
            .gdn_decode_d64_bf16 = gdn_decode_d64_bf16,
            .gdn_decode_d128_f16 = gdn_decode_d128_f16,
            .gdn_decode_d128_bf16 = gdn_decode_d128_bf16,
            .gdn_decode_dyn_f16 = gdn_decode_dyn_f16,
            .gdn_decode_dyn_bf16 = gdn_decode_dyn_bf16,
            .gdn_batch_decode_d64_f16 = gdn_batch_decode_d64_f16,
            .gdn_batch_decode_d64_bf16 = gdn_batch_decode_d64_bf16,
            .gdn_batch_decode_d128_f16 = gdn_batch_decode_d128_f16,
            .gdn_batch_decode_d128_bf16 = gdn_batch_decode_d128_bf16,
            .gdn_batch_decode_dyn_f16 = gdn_batch_decode_dyn_f16,
            .gdn_batch_decode_dyn_bf16 = gdn_batch_decode_dyn_bf16,
            .gdn_clone_state = gdn_clone_state,
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
            .qsa_prefill_rope_f16 = qsa_prefill_rope_f16,
            .qsa_prefill_rope_bf16 = qsa_prefill_rope_bf16,
            .qsa_prefill_kv_ring_commit = qsa_prefill_kv_ring_commit,
            .qsa_swa_prefill_attn_d64 = qsa_swa_prefill_attn_d64,
            .qsa_swa_prefill_attn_d128 = qsa_swa_prefill_attn_d128,
            .qsa_swa_prefill_attn_dyn = qsa_swa_prefill_attn_dyn,
            .qsa_swa_prefill_attn_mma_d64 = qsa_swa_prefill_attn_mma_d64,
            .qsa_swa_prefill_attn_mma_d128 = qsa_swa_prefill_attn_mma_d128,
            .gr_init = gr_init,
            .gr_read = gr_read,
            .gr_write = gr_write,
            .gr_collapse = gr_collapse,
            .gr_compute_gates_f16 = gr_compute_gates_f16,
            .gr_compute_gates_bf16 = gr_compute_gates_bf16,
            .gdn_compute_gates_f16 = gdn_compute_gates_f16,
            .gdn_compute_gates_bf16 = gdn_compute_gates_bf16,
            .gdn_conv1d_silu = gdn_conv1d_silu,
            .gdn_conv1d_q_silu = gdn_conv1d_q_silu,
            .gdn_conv1d_commit = gdn_conv1d_commit,
            .blend_f16 = blend_f16,
            .mla_fp4_kv_write = mla_fp4_kv_write,
            .csa2_update_summaries = csa2_update_summaries,
            .csa2_index_blocks = csa2_index_blocks,
            .mla_csa2_attend = mla_csa2_attend,
            .mla_csa2_attend_mma = mla_csa2_attend_mma,
            .mhc_single_pass = mhc_single_pass,
            .mhc_post_mix = mhc_post_mix,
            .g26_qknorm_rope_kv_f16 = g26_qknorm_rope_kv_f16,
            .g26_qknorm_rope_kv_bf16 = g26_qknorm_rope_kv_bf16,
            .g26_kv_commit = g26_kv_commit,
            .g26_swa_attn_192_128 = g26_swa_attn_192_128,
            .g26_swa_attn_256_256 = g26_swa_attn_256_256,
            .g26_swa_attn_dyn = g26_swa_attn_dyn,
            .g26_moe_route_sigmoid_f32 = g26_moe_route_sigmoid_f32,
            .g26_moe_route_sigmoid_f16 = g26_moe_route_sigmoid_f16,
            .g26_capture_tap_ring = g26_capture_tap_ring,
            .g27_mhc_pre_sinkhorn = g27_mhc_pre_sinkhorn,
            .g27_mhc_post_mix = g27_mhc_post_mix,
            .kda_conv1d_silu = kda_conv1d_silu,
            .kda_compute_gates = kda_compute_gates,
            .kda_decode_d64_f16 = kda_decode_d64_f16,
            .kda_decode_d64_bf16 = kda_decode_d64_bf16,
            .kda_decode_d128_f16 = kda_decode_d128_f16,
            .kda_decode_d128_bf16 = kda_decode_d128_bf16,
            .kda_decode_dyn_f16 = kda_decode_dyn_f16,
            .kda_decode_dyn_bf16 = kda_decode_dyn_bf16,
            .kda_prefill_d64 = kda_prefill_d64,
            .kda_prefill_d128 = kda_prefill_d128,
            .kda_prefill_dyn = kda_prefill_dyn,
            .kda_tree_verify_d64 = kda_tree_verify_d64,
            .kda_tree_verify_d128 = kda_tree_verify_d128,
            .kda_tree_verify_dyn = kda_tree_verify_dyn,
            .kda_out_norm_gate_f16 = kda_out_norm_gate_f16,
            .kda_out_norm_gate_bf16 = kda_out_norm_gate_bf16,
            .kda_commit_tree = kda_commit_tree,
            .g27_rope_interleave = g27_rope_interleave,
            .g27_layernorm = g27_layernorm,
            .g27_mla_kv_write = g27_mla_kv_write,
            .g27_dsa_idx_kv_write = g27_dsa_idx_kv_write,
            .g27_dsa_kpool_build = g27_dsa_kpool_build,
            .g27_dsa_scores = g27_dsa_scores,
            .g27_dsa_select_topk = g27_dsa_select_topk,
            .g27_mla_sparse_attend = g27_mla_sparse_attend,
            .g28_mla_full_attend = g28_mla_full_attend,
            .g28_mla_out_gate = g28_mla_out_gate,
        };
    }

    pub fn close(self: HybridPipelines) void {
        self.g28_mla_out_gate.close();
        self.g28_mla_full_attend.close();
        self.g27_mla_sparse_attend.close();
        self.g27_dsa_select_topk.close();
        self.g27_dsa_scores.close();
        self.g27_dsa_kpool_build.close();
        self.g27_dsa_idx_kv_write.close();
        self.g27_mla_kv_write.close();
        self.g27_layernorm.close();
        self.g27_rope_interleave.close();
        self.kda_commit_tree.close();
        self.kda_out_norm_gate_bf16.close();
        self.kda_out_norm_gate_f16.close();
        self.kda_tree_verify_dyn.close();
        self.kda_tree_verify_d128.close();
        self.kda_tree_verify_d64.close();
        self.kda_prefill_dyn.close();
        self.kda_prefill_d128.close();
        self.kda_prefill_d64.close();
        self.kda_decode_dyn_bf16.close();
        self.kda_decode_dyn_f16.close();
        self.kda_decode_d128_bf16.close();
        self.kda_decode_d128_f16.close();
        self.kda_decode_d64_bf16.close();
        self.kda_decode_d64_f16.close();
        self.kda_compute_gates.close();
        self.kda_conv1d_silu.close();
        for (0..2) |sb| {
            for (0..2) |ab| {
                self.g27_mhc_post_mix[sb][ab].close();
                self.g27_mhc_pre_sinkhorn[sb][ab].close();
            }
        }
        self.g26_capture_tap_ring.close();
        self.g26_moe_route_sigmoid_f16.close();
        self.g26_moe_route_sigmoid_f32.close();
        self.g26_swa_attn_dyn.close();
        self.g26_swa_attn_256_256.close();
        self.g26_swa_attn_192_128.close();
        self.g26_kv_commit.close();
        self.g26_qknorm_rope_kv_bf16.close();
        self.g26_qknorm_rope_kv_f16.close();
        for (0..2) |sb| {
            for (0..2) |ab| {
                self.mhc_post_mix[sb][ab].close();
                self.mhc_single_pass[sb][ab].close();
            }
        }
        self.mla_csa2_attend_mma.close();
        self.mla_csa2_attend.close();
        self.csa2_index_blocks.close();
        self.csa2_update_summaries.close();
        self.mla_fp4_kv_write.close();
        self.qsa_swa_prefill_attn_mma_d128.close();
        self.qsa_swa_prefill_attn_mma_d64.close();
        self.qsa_swa_prefill_attn_dyn.close();
        self.qsa_swa_prefill_attn_d128.close();
        self.qsa_swa_prefill_attn_d64.close();
        self.qsa_prefill_kv_ring_commit.close();
        self.qsa_prefill_rope_bf16.close();
        self.qsa_prefill_rope_f16.close();
        self.blend_f16.close();
        self.gdn_conv1d_commit.close();
        self.gdn_conv1d_q_silu.close();
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
        self.gdn_clone_state.close();
        self.gdn_batch_decode_dyn_bf16.close();
        self.gdn_batch_decode_dyn_f16.close();
        self.gdn_batch_decode_d128_bf16.close();
        self.gdn_batch_decode_d128_f16.close();
        self.gdn_batch_decode_d64_bf16.close();
        self.gdn_batch_decode_d64_f16.close();
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

    /// Encodes multi-request batched Gated DeltaNet decode (`ami_gdn_batch_decode`, `C14`)
    /// executing `batch_size` (`1..16+`) independent slots in ONE GPU pass using
    /// per-slot `state_in_slots[b]` and `state_out_slots[b]`.
    pub fn encodeGdnBatchDecode(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GdnBatchDecodeArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.batch_size == 0) return error.ZeroDimension;
        try validateGdnDims(args.num_k_heads, args.num_v_heads, args.head_k_dim, args.head_v_dim);
        const min_stride = std.math.mul(u32, args.num_v_heads, args.head_k_dim * args.head_v_dim) catch return error.InvalidConfig;
        if (args.slot_stride_f32 < min_stride) return error.InvalidConfig;

        const q_scale = args.q_scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_k_dim))));
        const params: GdnBatchDecodeParams = .{
            .batch_size = args.batch_size,
            .num_k_heads = args.num_k_heads,
            .num_v_heads = args.num_v_heads,
            .head_k_dim = args.head_k_dim,
            .head_v_dim = args.head_v_dim,
            .slot_stride_f32 = args.slot_stride_f32,
            .q_scale = q_scale,
            .l2_eps = args.l2_eps,
            .rms_eps = args.rms_eps,
            .apply_l2_norm = if (args.apply_l2_norm) 1 else 0,
            .apply_out_norm = if (args.norm_w != null) 1 else 0,
            .apply_out_gate = if (args.z != null) 1 else 0,
            .per_head_norm = if (args.per_head_norm) 1 else 0,
        };
        const pipe = switch (args.head_k_dim) {
            64 => if (args.norm_dtype == .bf16) self.gdn_batch_decode_d64_bf16 else self.gdn_batch_decode_d64_f16,
            128 => if (args.norm_dtype == .bf16) self.gdn_batch_decode_d128_bf16 else self.gdn_batch_decode_d128_f16,
            else => if (args.norm_dtype == .bf16) self.gdn_batch_decode_dyn_bf16 else self.gdn_batch_decode_dyn_f16,
        };
        const out_slots = args.state_out_slots orelse args.state_in_slots;
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
                args.state_in_pool,
                args.state_out_pool,
                args.out,
                args.state_in_slots,
                out_slots,
            },
            std.mem.asBytes(&params),
            .{ args.num_v_heads, args.batch_size, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_gdn_clone_state` (`R8`), copying a full multi-layer `GDN`
    /// recurrent state `[num_gdn_layers, num_v_heads, head_k_dim, head_v_dim]`
    /// plus optional `conv1d` history between a slot's `GdnStateArena` slab and
    /// a `PrefixCache` / `GdnSnapshotPool` slab in ONE GPU dispatch (`0` CPU `memcpy`).
    pub fn encodeGdnCloneState(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GdnCloneStateArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_gdn_layers == 0 or args.state_f32_per_layer == 0) return error.ZeroDimension;
        if (args.src_state_layer_stride_f32 < args.state_f32_per_layer or
            args.dst_state_layer_stride_f32 < args.state_f32_per_layer)
        {
            return error.InvalidConfig;
        }
        if (args.conv_f16_per_layer > 0) {
            if (args.src_conv == null or args.dst_conv == null) return error.InvalidConfig;
            if (args.src_conv_layer_stride_f16 < args.conv_f16_per_layer or
                args.dst_conv_layer_stride_f16 < args.conv_f16_per_layer)
            {
                return error.InvalidConfig;
            }
        }
        const params: GdnCloneStateParams = .{
            .num_gdn_layers = args.num_gdn_layers,
            .state_f32_per_layer = args.state_f32_per_layer,
            .src_state_layer_stride_f32 = args.src_state_layer_stride_f32,
            .dst_state_layer_stride_f32 = args.dst_state_layer_stride_f32,
            .conv_f16_per_layer = args.conv_f16_per_layer,
            .src_conv_layer_stride_f16 = args.src_conv_layer_stride_f16,
            .dst_conv_layer_stride_f16 = args.dst_conv_layer_stride_f16,
        };
        const max_elems: u64 = @max(args.state_f32_per_layer, args.conv_f16_per_layer);
        const groups_x: u64 = (max_elems + 1023) / 1024;
        try frame.encodeThreadgroups(
            self.gdn_clone_state,
            &.{
                args.src_state,
                args.dst_state,
                args.src_conv orelse args.src_state,
                args.dst_conv orelse args.dst_state,
            },
            std.mem.asBytes(&params),
            .{ groups_x, args.num_gdn_layers, 1 },
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
        if (!args.write_state_only and args.out_start_token >= args.seq_len) return error.InvalidConfig;
        try validateGdnDims(args.num_k_heads, args.num_v_heads, args.head_k_dim, args.head_v_dim);
        const q_scale = args.q_scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_k_dim))));
        const need_post_norm = (!args.write_state_only) and (args.norm_w != null);
        const params: GdnPrefillParams = .{
            .seq_len = args.seq_len,
            .num_k_heads = args.num_k_heads,
            .num_v_heads = args.num_v_heads,
            .head_k_dim = args.head_k_dim,
            .head_v_dim = args.head_v_dim,
            .q_scale = q_scale,
            .l2_eps = args.l2_eps,
            .apply_l2_norm = if (args.apply_l2_norm) 1 else 0,
            .apply_out_gate = if (!args.write_state_only and !need_post_norm and args.z != null) 1 else 0,
            .chunk_size = if (args.chunk_size > 0) args.chunk_size else 64,
            .write_state_only = if (args.write_state_only) 1 else 0,
            .out_start_token = if (args.write_state_only) 0 else args.out_start_token,
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
            const emitted_tokens = args.seq_len - args.out_start_token;
            const z_row_bytes: u64 = @as(u64, args.num_v_heads) * args.head_v_dim * @sizeOf(f16);
            const z_bind: metal.BufferBinding = if (args.z) |zb|
                .{ .buffer = zb.buffer, .offset = zb.offset + @as(u64, args.out_start_token) * z_row_bytes }
            else
                args.out;
            const norm_params: GdnOutNormGateParams = .{
                .num_tokens = emitted_tokens,
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
                    z_bind,
                    args.norm_w.?,
                },
                std.mem.asBytes(&norm_params),
                .{ args.num_v_heads, emitted_tokens, 1 },
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
            .num_retired_chunks = args.num_retired_chunks,
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
        if (args.num_nodes == 0 or args.num_nodes > MAX_WIDE_PREFILL_TOKENS) return error.InvalidNodeCount;
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
            .causal_q_start_pos = args.causal_q_start_pos orelse 0xFFFF_FFFF,
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
        const has_any_tree = (args.k_tree != null or args.v_tree != null or args.ancestor_masks != null);
        const max_nodes = if (has_any_tree) MAX_TREE_NODES else MAX_WIDE_PREFILL_TOKENS;
        if (args.num_nodes == 0 or args.num_nodes > max_nodes) return error.InvalidNodeCount;
        if (args.num_q_heads == 0 or args.num_kv_heads == 0 or args.head_dim == 0 or
            args.chunk_tokens == 0 or args.block_tokens == 0)
        {
            return error.ZeroDimension;
        }
        if ((args.num_q_heads % args.num_kv_heads) != 0) return error.InvalidConfig;
        if (args.head_dim > MAX_QSA_HEAD_DIM or (args.head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.top_k_blocks == 0 or args.top_k_blocks > QSA_MAX_TOP_K_BLOCKS) return error.UnsupportedTopKBlocks;

        const retired_tokens = std.math.mul(u64, args.num_retired_chunks, args.chunk_tokens) catch return error.InvalidConfig;
        if (args.num_retired_chunks > 0) {
            if (args.ring_capacity != 0) return error.InvalidConfig;
            if ((args.head_dim % attention.KV_Q8_GROUP_SIZE) != 0) return error.InvalidHeadDim;
            if (args.retired_pool == null or args.retired_chunk_ids == null or
                args.retired_plane_stride_bytes == 0 or
                args.retired_chunk_stride_bytes < args.retired_plane_stride_bytes * 2)
            {
                return error.InvalidConfig;
            }
        }
        if (@as(u64, args.kv_len) > retired_tokens and args.kv_tiles == null) return error.InvalidConfig;

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
            .num_retired_chunks = args.num_retired_chunks,
            .retired_plane_stride_bytes = args.retired_plane_stride_bytes,
            .retired_chunk_stride_bytes = args.retired_chunk_stride_bytes,
            .causal_q_start_pos = args.causal_q_start_pos orelse 0xFFFF_FFFF,
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
                args.retired_pool orelse args.q,
                args.retired_chunk_ids orelse args.q,
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
                .causal_q_start_pos = args.causal_q_start_pos,
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
            .retired_pool = args.retired_pool,
            .retired_chunk_ids = args.retired_chunk_ids,
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
            .num_retired_chunks = args.num_retired_chunks,
            .retired_plane_stride_bytes = args.retired_plane_stride_bytes,
            .retired_chunk_stride_bytes = args.retired_chunk_stride_bytes,
            .causal_q_start_pos = args.causal_q_start_pos,
        }, attn_barrier);
    }

    /// Encodes `ami_qsa_prefill_qknorm_mrope` (`G15`, ADR-0020): applies QK-Norm
    /// + YaRN + interleaved 3D MRoPE across `num_tokens` (`1..1024+`) and either
    /// writes K/V directly into bindless `HybridKvRing` tiles (`write_to_tiles = true`)
    /// or rotates K in-place (`write_to_tiles = false`, preserving circular SWA
    /// ring history until `encodeQsaPrefillKvRingCommit` runs after attention).
    pub fn encodeQsaPrefillRope(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: QsaPrefillRopeArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_tokens == 0 or args.num_q_heads == 0 or args.num_kv_heads == 0 or
            args.head_dim == 0 or args.chunk_tokens == 0)
        {
            return error.ZeroDimension;
        }
        if (args.head_dim > MAX_QSA_HEAD_DIM or (args.head_dim % 4) != 0) return error.InvalidHeadDim;
        const rotary_dim = if (args.rotary_dim == 0) args.head_dim else args.rotary_dim;
        if (rotary_dim > args.head_dim) return error.InvalidRotaryDim;
        try args.mrope.validate(rotary_dim);

        const apply_norm = (args.q_norm != null or args.k_norm != null);
        if (apply_norm and (args.q_norm == null or args.k_norm == null)) return error.MissingNormWeights;
        if (args.write_to_tiles and args.kv_tiles == null) return error.InvalidConfig;
        if (args.ring_capacity == 0) {
            const retired_tokens = std.math.mul(u64, args.num_retired_chunks, args.chunk_tokens) catch return error.InvalidConfig;
            if (args.write_to_tiles and @as(u64, args.start_pos) < retired_tokens) return error.InvalidConfig;
        } else if ((args.ring_capacity % args.chunk_tokens) != 0) {
            return error.InvalidConfig;
        }

        const params: QsaPrefillRopeParams = .{
            .num_tokens = args.num_tokens,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .rotary_dim = rotary_dim,
            .start_pos = args.start_pos,
            .chunk_tokens = args.chunk_tokens,
            .num_retired_chunks = args.num_retired_chunks,
            .ring_capacity = args.ring_capacity,
            .mrope_s0 = args.mrope.section[0],
            .mrope_s1 = args.mrope.section[1],
            .mrope_s2 = args.mrope.section[2],
            .use_positions_3d = if (args.positions_3d != null) 1 else 0,
            .apply_qk_norm = if (apply_norm) 1 else 0,
            .kv_only = if (args.kv_only) 1 else 0,
            .write_to_tiles = if (args.write_to_tiles) 1 else 0,
            .rms_eps = args.rms_eps,
            .rope_theta = args.rope_theta,
            .yarn_factor = args.yarn.factor,
            .yarn_mscale = args.yarn.effectiveMscale(),
            .yarn_beta_fast = args.yarn.beta_fast,
            .yarn_beta_slow = args.yarn.beta_slow,
            .original_max_pos = args.yarn.original_max_pos,
        };
        const pipe = if (args.norm_dtype == .bf16) self.qsa_prefill_rope_bf16 else self.qsa_prefill_rope_f16;
        const total_heads: u64 = if (args.kv_only)
            args.num_kv_heads
        else
            @as(u64, args.num_q_heads) + args.num_kv_heads;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.q,
                args.k,
                args.v,
                args.q_norm orelse args.q,
                args.k_norm orelse args.k,
                args.positions_3d orelse args.q,
                args.kv_tiles orelse args.q,
            },
            std.mem.asBytes(&params),
            .{ total_heads, args.num_tokens, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_qsa_prefill_kv_ring_commit` (`G15`, ADR-0020): commits the
    /// trailing `min(num_tokens, ring_capacity)` rotated K and raw V rows into
    /// bindless `HybridKvRing` `RGBA16Float` tiles with zero slot write races.
    pub fn encodeQsaPrefillKvRingCommit(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: QsaPrefillKvCommitArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_tokens == 0 or args.num_kv_heads == 0 or args.head_dim == 0 or args.chunk_tokens == 0) {
            return error.ZeroDimension;
        }
        if (args.head_dim > MAX_QSA_HEAD_DIM or (args.head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.ring_capacity == 0) {
            const retired_tokens = std.math.mul(u64, args.num_retired_chunks, args.chunk_tokens) catch return error.InvalidConfig;
            if (@as(u64, args.start_pos) < retired_tokens) return error.InvalidConfig;
        } else if ((args.ring_capacity % args.chunk_tokens) != 0) {
            return error.InvalidConfig;
        }

        const commit_cnt: u32 = if (args.ring_capacity > 0) @min(args.num_tokens, args.ring_capacity) else args.num_tokens;
        const params: QsaPrefillKvCommitParams = .{
            .num_tokens = args.num_tokens,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .start_pos = args.start_pos,
            .chunk_tokens = args.chunk_tokens,
            .num_retired_chunks = args.num_retired_chunks,
            .ring_capacity = args.ring_capacity,
        };
        try frame.encodeThreadgroups(
            self.qsa_prefill_kv_ring_commit,
            &.{ args.k, args.v, args.kv_tiles },
            std.mem.asBytes(&params),
            .{ args.num_kv_heads, commit_cnt, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_qsa_swa_prefill_attn` (`G15`, ADR-0020): causal sliding-window
    /// prefill attention across `num_q` (`1..1024+`) queries reading unmodified
    /// `HybridKvRing` prefix (`t < chunk_start_pos`) plus in-flight chunk `k_chunk`/
    /// `v_chunk` (`t >= chunk_start_pos`), strictly skipping upper-triangular
    /// (`kv_pos > q_pos`) and out-of-window positions.
    /// Returns `true` when QSA prefill qualifies for the `G19` `simdgroup_half8x8`
    /// MMA tile kernel (`Q_len >= 32` and `KV_len >= 64` and `head_dim in {64, 128}`).
    pub fn shouldUseQsaMmaPrefill(args: QsaSwaPrefillAttnArgs) bool {
        if (args.head_dim != 64 and args.head_dim != 128) return false;
        if (args.use_mma_override) |ov| return ov;
        return attention.shouldUseG19MmaPrefill(args.num_q, args.kv_len);
    }

    /// Encodes `ami_qsa_swa_prefill_attn` / `ami_qsa_swa_prefill_attn_mma` (`G9` +
    /// `G15` / `G19`, ADR-0020, ADR-0023 §2): strictly gates hardware `simdgroup_half8x8`
    /// MMA tiles on `Q_len >= 32` and `KV_len >= 64` (never `M = 1` or narrow draft trees).
    pub fn encodeQsaSwaPrefillAttn(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: QsaSwaPrefillAttnArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_q == 0 or args.num_q_heads == 0 or args.num_kv_heads == 0 or
            args.head_dim == 0 or args.chunk_tokens == 0)
        {
            return error.ZeroDimension;
        }
        if ((args.num_q_heads % args.num_kv_heads) != 0) return error.InvalidConfig;
        if (args.head_dim > MAX_QSA_HEAD_DIM or (args.head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.ring_capacity > 0 and (args.ring_capacity % args.chunk_tokens) != 0) return error.InvalidConfig;
        if (args.num_retired_chunks > 0) {
            if (args.ring_capacity != 0) return error.InvalidConfig;
            if ((args.head_dim % attention.KV_Q8_GROUP_SIZE) != 0) return error.InvalidHeadDim;
            if (args.retired_pool == null or args.retired_chunk_ids == null or
                args.retired_plane_stride_bytes == 0 or
                args.retired_chunk_stride_bytes < args.retired_plane_stride_bytes * 2)
            {
                return error.InvalidConfig;
            }
        }
        if (args.chunk_start_pos > 0 and args.kv_tiles == null and args.num_retired_chunks == 0) {
            return error.InvalidConfig;
        }

        const scale = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_dim))));
        const params: QsaSwaPrefillAttnParams = .{
            .num_q = args.num_q,
            .q_start_pos = args.q_start_pos,
            .chunk_start_pos = args.chunk_start_pos,
            .kv_len = args.kv_len,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .window_size = args.window_size,
            .ring_capacity = args.ring_capacity,
            .chunk_tokens = args.chunk_tokens,
            .scale = scale,
            .num_retired_chunks = args.num_retired_chunks,
            .retired_plane_stride_bytes = args.retired_plane_stride_bytes,
            .retired_chunk_stride_bytes = args.retired_chunk_stride_bytes,
        };
        const bindings = [_]metal.BufferBinding{
            args.q,
            args.kv_tiles orelse args.q,
            args.k_chunk,
            args.v_chunk,
            args.out,
            args.retired_pool orelse args.q,
            args.retired_chunk_ids orelse args.q,
        };
        const gqa_ratio = args.num_q_heads / args.num_kv_heads;
        if (shouldUseQsaMmaPrefill(args)) {
            const pipe = switch (args.head_dim) {
                64 => self.qsa_swa_prefill_attn_mma_d64,
                128 => self.qsa_swa_prefill_attn_mma_d128,
                else => unreachable,
            };
            const total_q: u64 = @as(u64, args.num_q) * gqa_ratio;
            const q_tiles: u64 = (total_q + 31) / 32;
            try frame.encodeThreadgroups(
                pipe,
                &bindings,
                std.mem.asBytes(&params),
                .{ args.num_kv_heads, q_tiles, 1 },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
            return;
        }

        const q_groups_per_kv: u64 = (@as(u64, gqa_ratio) + 7) / 8;
        const pipe = switch (args.head_dim) {
            64 => self.qsa_swa_prefill_attn_d64,
            128 => self.qsa_swa_prefill_attn_d128,
            else => self.qsa_swa_prefill_attn_dyn,
        };
        try frame.encodeThreadgroups(
            pipe,
            &bindings,
            std.mem.asBytes(&params),
            .{ @as(u64, args.num_kv_heads) * q_groups_per_kv, args.num_q, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
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

    pub fn encodeGdnConv1dQSilu(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: GdnConv1dQSiluArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.gdn_qk == 0 or args.gdn_v == 0 or args.conv_k == 0 or args.conv_k > 8) {
            return error.InvalidConfig;
        }
        if (args.is_tree and (args.ancestor_masks == null or args.m > MAX_TREE_NODES)) {
            return error.InvalidConfig;
        }
        const params: GdnConv1dQParams = .{
            .m = args.m,
            .gdn_qk = args.gdn_qk,
            .gdn_v = args.gdn_v,
            .conv_k = args.conv_k,
            .is_tree = if (args.is_tree) 1 else 0,
        };
        try frame.encodeThreads(
            self.gdn_conv1d_q_silu,
            &.{
                args.q_cur,
                args.q_hist_r0,
                args.conv_w,
                args.conv_state_in,
                args.ancestor_masks orelse args.conv_state_in,
                args.q_out,
            },
            std.mem.asBytes(&params),
            .{ args.gdn_qk, 1, 1 },
            .{ @min(@as(u64, args.gdn_qk), 256), 1, 1 },
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

    // -----------------------------------------------------------------------
    // M7 (R7 + G11): MLA + CSA2 + FP4 KV + SWA Bounded Replay Encoders
    // -----------------------------------------------------------------------

    /// Quantizes `[num_tokens, kv_lora_rank]` latent rows to FP4 (`E2M1` +
    /// per-16 `E4M3` scale) and writes `[num_tokens, qk_rope_dim]` RoPE keys
    /// into a physical `Full` slot of `MlaFp4KvArena` (optionally mirroring
    /// exact `f16` rows into a circular `MlaSwaRingArena`).
    pub fn encodeMlaFp4KvWrite(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: MlaFp4KvWriteArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_tokens == 0 or args.num_tokens > MAX_WIDE_PREFILL_TOKENS) return error.InvalidNodeCount;
        const spec = args.kv_arena.spec;
        if (args.kv_slot >= spec.num_kv_layers) return error.InvalidLayerIndex;
        const end_pos = std.math.add(u32, args.start_pos, args.num_tokens) catch return error.InvalidConfig;
        if (end_pos > spec.max_tokens) return error.InvalidConfig;
        if (spec.qk_rope_dim > 0 and args.rope_in == null) return error.InvalidConfig;

        var swa_capacity: u32 = 0;
        var swa_rope_off: u64 = 0;
        var swa_binding: metal.BufferBinding = args.latent_in;
        if (args.swa_ring) |ring| {
            const swa_buf = args.swa_metal_buf orelse return error.InvalidConfig;
            if (ring.spec.kv_lora_rank != spec.kv_lora_rank or ring.spec.qk_rope_dim != spec.qk_rope_dim) {
                return error.InvalidConfig;
            }
            swa_binding = ring.layerBinding(args.kv_slot, swa_buf) catch return error.InvalidLayerIndex;
            swa_capacity = ring.spec.swa_window;
            swa_rope_off = ring.rope_plane_offset;
        }

        const kv_binding = args.kv_arena.layerBinding(args.kv_slot, args.kv_metal_buf) catch return error.InvalidLayerIndex;
        const params: MlaFp4KvWriteParams = .{
            .num_tokens = args.num_tokens,
            .start_pos = args.start_pos,
            .kv_lora_rank = spec.kv_lora_rank,
            .qk_rope_dim = spec.qk_rope_dim,
            .rope_tail_format = @backingInt(spec.rope_tail_format),
            .max_tokens = spec.max_tokens,
            .swa_ring_capacity = swa_capacity,
            .flags = if (args.is_encoder_projection) 1 else 0,
            .e2m1_plane_offset = args.kv_arena.e2m1_plane_offset,
            .scale_plane_offset = args.kv_arena.scale_plane_offset,
            .rope_plane_offset = args.kv_arena.rope_plane_offset,
            .swa_rope_plane_offset = swa_rope_off,
        };
        try frame.encodeThreadgroups(
            self.mla_fp4_kv_write,
            &.{
                args.latent_in,
                args.rope_in orelse args.latent_in,
                kv_binding,
                swa_binding,
            },
            std.mem.asBytes(&params),
            .{ args.num_tokens, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Stage-1 hierarchical summary updater: builds fine (`[num_fine, index_dim]`)
    /// and coarse (`[num_coarse, index_dim]`) `f16` block summaries directly from
    /// FP4 (`E2M1` + `E4M3`) latent KV slabs.
    pub fn encodeCsa2UpdateSummaries(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: Csa2UpdateSummariesArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.kv_len == 0) return error.ZeroDimension;
        const kv_spec = args.kv_arena.spec;
        const sum_spec = args.summary_arena.spec;
        if (args.kv_len > kv_spec.max_tokens or args.kv_len > sum_spec.max_tokens) return error.InvalidConfig;
        if (sum_spec.index_dim > kv_spec.kv_lora_rank) return error.InvalidConfig;
        if (args.kv_slot >= kv_spec.num_kv_layers or args.kv_slot >= sum_spec.num_kv_layers) {
            return error.InvalidLayerIndex;
        }

        const total_fine = (args.kv_len + sum_spec.block_tokens - 1) / sum_spec.block_tokens;
        const total_coarse = (total_fine + sum_spec.coarse_group_blocks - 1) / sum_spec.coarse_group_blocks;
        if (args.start_coarse_block >= total_coarse) return error.InvalidConfig;
        const max_rem = total_coarse - args.start_coarse_block;
        const num_coarse = args.num_coarse_blocks orelse max_rem;
        if (num_coarse == 0 or num_coarse > max_rem) return error.InvalidConfig;

        const kv_binding = args.kv_arena.layerBinding(args.kv_slot, args.kv_metal_buf) catch return error.InvalidLayerIndex;
        const sum_binding = args.summary_arena.layerBinding(args.kv_slot, args.summary_metal_buf) catch return error.InvalidLayerIndex;
        const params: Csa2SummaryParams = .{
            .kv_len = args.kv_len,
            .kv_lora_rank = kv_spec.kv_lora_rank,
            .index_dim = sum_spec.index_dim,
            .block_tokens = sum_spec.block_tokens,
            .coarse_group_blocks = sum_spec.coarse_group_blocks,
            .start_coarse_block = args.start_coarse_block,
            .num_coarse_blocks = num_coarse,
            .e2m1_plane_offset = args.kv_arena.e2m1_plane_offset,
            .scale_plane_offset = args.kv_arena.scale_plane_offset,
            .fine_plane_offset = args.summary_arena.fine_plane_offset,
            .coarse_plane_offset = args.summary_arena.coarse_plane_offset,
        };
        try frame.encodeThreadgroups(
            self.csa2_update_summaries,
            &.{ kv_binding, sum_binding },
            std.mem.asBytes(&params),
            .{ num_coarse, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Stage-2 hierarchical CSA2 sparse block indexer (`Full` and `Reindex` modes).
    pub fn encodeCsa2IndexBlocks(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: Csa2IndexBlocksArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const has_batch = args.batch_kv_lens != null;
        const max_nodes = if (has_batch) MAX_MLA_BATCH_DECODE_SLOTS else MAX_WIDE_PREFILL_TOKENS;
        if (args.num_nodes == 0 or args.num_nodes > max_nodes) return error.InvalidNodeCount;
        if (args.num_q_heads == 0 or args.kv_lora_rank == 0) return error.ZeroDimension;
        if ((args.kv_lora_rank % 32) != 0 or args.kv_lora_rank > MAX_MLA_LATENT_DIM) return error.InvalidHeadDim;
        if (args.top_k_blocks == 0 or args.top_k_blocks > CSA2_MAX_TOP_K_BLOCKS) return error.UnsupportedTopKBlocks;
        if (args.num_nodes > args.index_buf.spec.max_nodes or args.top_k_blocks > args.index_buf.spec.top_k_blocks) {
            return error.InvalidConfig;
        }
        const sum_spec = args.summary_arena.spec;
        if (sum_spec.index_dim > args.kv_lora_rank) return error.InvalidConfig;
        if (has_batch) {
            const end_slot = std.math.add(u32, args.kv_slot, args.num_nodes) catch return error.InvalidLayerIndex;
            if (end_slot > sum_spec.num_kv_layers) return error.InvalidLayerIndex;
        } else {
            if (args.kv_slot >= sum_spec.num_kv_layers) return error.InvalidLayerIndex;
        }

        const g_blocks = sum_spec.coarse_group_blocks;
        const auto_coarse = @min(CSA2_MAX_TOP_K_BLOCKS, ((args.top_k_blocks + g_blocks - 1) / g_blocks) + 1);
        const top_k_coarse = if (args.top_k_coarse > 0) @min(args.top_k_coarse, CSA2_MAX_TOP_K_BLOCKS) else auto_coarse;
        const scale = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(sum_spec.index_dim))));

        const fine_binding = args.summary_arena.fineBinding(args.kv_slot, args.summary_metal_buf) catch return error.InvalidLayerIndex;
        const coarse_binding = args.summary_arena.coarseBinding(args.kv_slot, args.summary_metal_buf) catch return error.InvalidLayerIndex;
        const params: Csa2IndexParams = .{
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .kv_lora_rank = args.kv_lora_rank,
            .index_dim = sum_spec.index_dim,
            .kv_len = args.kv_len,
            .encoder_prefix_len = args.encoder_prefix_len,
            .causal_q_start_pos = args.causal_q_start_pos orelse 0xFFFF_FFFF,
            .swa_window = args.swa_window,
            .block_tokens = sum_spec.block_tokens,
            .coarse_group_blocks = g_blocks,
            .top_k_coarse = top_k_coarse,
            .top_k_blocks = args.top_k_blocks,
            .num_local_blocks = if (args.num_local_blocks > 0) args.num_local_blocks else 1,
            .has_batch_slots = if (has_batch) 1 else 0,
            .scale = scale,
            .fine_batch_stride_bytes = if (has_batch) args.summary_arena.layer_stride_bytes else 0,
            .coarse_batch_stride_bytes = if (has_batch) args.summary_arena.layer_stride_bytes else 0,
        };
        try frame.encodeThreadgroups(
            self.csa2_index_blocks,
            &.{
                args.q_nope,
                fine_binding,
                coarse_binding,
                args.batch_kv_lens orelse args.q_nope,
                args.index_buf.blocksBinding(args.index_metal_buf),
                args.index_buf.countsBinding(args.index_metal_buf),
            },
            std.mem.asBytes(&params),
            .{ args.num_nodes, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Online-softmax absorbed MLA + CSA2 attention over selected FP4 (`E2M1` +
    /// `E4M3`) latent KV blocks + deduplicated SWA bounded window replay +
    /// optional 64-bit ancestor-masked speculative tree tokens.
    pub fn encodeMlaCsa2Attend(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: MlaCsa2AttendArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const has_any_tree = (args.kv_tree_latent != null or args.kv_tree_rope != null or args.ancestor_masks != null);
        const has_batch = args.batch_kv_lens != null;
        if (has_any_tree and has_batch) return error.InvalidConfig;
        const max_nodes: u32 = if (has_any_tree)
            MAX_TREE_NODES
        else if (has_batch)
            MAX_MLA_BATCH_DECODE_SLOTS
        else
            MAX_WIDE_PREFILL_TOKENS;
        if (args.num_nodes == 0 or args.num_nodes > max_nodes) return error.InvalidNodeCount;
        if (args.num_q_heads == 0 or args.block_tokens == 0) return error.ZeroDimension;
        if (args.top_k_blocks == 0 or args.top_k_blocks > CSA2_MAX_TOP_K_BLOCKS) return error.UnsupportedTopKBlocks;
        if (args.num_nodes > args.index_buf.spec.max_nodes or args.top_k_blocks > args.index_buf.spec.top_k_blocks) {
            return error.InvalidConfig;
        }
        const spec = args.kv_arena.spec;
        if (spec.qk_rope_dim > 0 and args.q_pe == null) return error.InvalidConfig;
        if (has_any_tree) {
            if (args.kv_tree_latent == null or args.ancestor_masks == null) return error.MissingTreeInputs;
            if (spec.qk_rope_dim > 0 and args.kv_tree_rope == null) return error.MissingTreeInputs;
        }
        if (has_batch) {
            const end_slot = std.math.add(u32, args.kv_slot, args.num_nodes) catch return error.InvalidLayerIndex;
            if (end_slot > spec.num_kv_layers) return error.InvalidLayerIndex;
        } else {
            if (args.kv_slot >= spec.num_kv_layers) return error.InvalidLayerIndex;
        }

        var swa_capacity: u32 = 0;
        var swa_rope_off: u64 = 0;
        var swa_batch_stride: u64 = 0;
        var swa_binding: metal.BufferBinding = args.q_nope;
        if (args.swa_ring) |ring| {
            const swa_buf = args.swa_metal_buf orelse return error.InvalidConfig;
            if (ring.spec.kv_lora_rank != spec.kv_lora_rank or ring.spec.qk_rope_dim != spec.qk_rope_dim) {
                return error.InvalidConfig;
            }
            if (has_batch) {
                const end_slot = std.math.add(u32, args.kv_slot, args.num_nodes) catch return error.InvalidLayerIndex;
                if (end_slot > ring.spec.num_kv_layers) return error.InvalidLayerIndex;
            }
            swa_binding = ring.layerBinding(args.kv_slot, swa_buf) catch return error.InvalidLayerIndex;
            swa_capacity = ring.spec.swa_window;
            swa_rope_off = ring.rope_plane_offset;
            swa_batch_stride = if (has_batch) ring.layer_stride_bytes else 0;
        }

        const kv_binding = args.kv_arena.layerBinding(args.kv_slot, args.kv_metal_buf) catch return error.InvalidLayerIndex;
        const scale = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(spec.kv_lora_rank + spec.qk_rope_dim))));
        const params: MlaCsa2AttnParams = .{
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .kv_lora_rank = spec.kv_lora_rank,
            .qk_rope_dim = spec.qk_rope_dim,
            .rope_tail_format = @backingInt(spec.rope_tail_format),
            .kv_len = args.kv_len,
            .encoder_prefix_len = args.encoder_prefix_len,
            .causal_q_start_pos = args.causal_q_start_pos orelse 0xFFFF_FFFF,
            .swa_window = args.swa_window,
            .swa_ring_capacity = swa_capacity,
            .block_tokens = args.block_tokens,
            .top_k_blocks = args.top_k_blocks,
            .has_tree = if (has_any_tree) 1 else 0,
            .has_batch_slots = if (has_batch) 1 else 0,
            .scale = scale,
            .e2m1_plane_offset = args.kv_arena.e2m1_plane_offset,
            .scale_plane_offset = args.kv_arena.scale_plane_offset,
            .rope_plane_offset = args.kv_arena.rope_plane_offset,
            .swa_rope_plane_offset = swa_rope_off,
            .kv_batch_stride_bytes = if (has_batch) args.kv_arena.layer_stride_bytes else 0,
            .swa_batch_stride_bytes = swa_batch_stride,
        };
        const bindings = [_]metal.BufferBinding{
            args.q_nope,
            args.q_pe orelse args.q_nope,
            kv_binding,
            swa_binding,
            args.index_buf.blocksBinding(args.index_metal_buf),
            args.index_buf.countsBinding(args.index_metal_buf),
            args.kv_tree_latent orelse args.q_nope,
            args.kv_tree_rope orelse args.q_nope,
            args.ancestor_masks orelse args.q_nope,
            args.batch_kv_lens orelse args.q_nope,
            args.out,
        };

        // G19 short-sequence + non-tree/non-batch guard: hardware simdgroup_half8x8
        // MMA activates strictly when Q_len >= 32 and KV_len >= 64 on contiguous prefill.
        const can_use_mma = !has_any_tree and !has_batch;
        const use_mma = can_use_mma and (args.use_mma_override orelse attention.shouldUseG19MmaPrefill(args.num_nodes, args.kv_len));
        if (use_mma) {
            const q_tiles: u64 = (@as(u64, args.num_nodes) + 31) / 32;
            try frame.encodeThreadgroups(
                self.mla_csa2_attend_mma,
                &bindings,
                std.mem.asBytes(&params),
                .{ args.num_q_heads, q_tiles, 1 },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
            return;
        }

        try frame.encodeThreadgroups(
            self.mla_csa2_attend,
            &bindings,
            std.mem.asBytes(&params),
            .{ args.num_q_heads, args.num_nodes, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Unified CSA2 (`Full` / `Reindex` / `Reuse`) + `LoopSchedule` (`r >= 1`
    /// index reuse) layer dispatcher:
    /// - `Full`: writes KV + updates summaries + runs hierarchical indexer only
    ///   when `loop_step == 0`, then attends.
    /// - `Reindex`: `0` KV writes (aliases `shared_kv_owner`), runs indexer only
    ///   when `loop_step == 0`, then attends.
    /// - `Reuse`: `0` KV writes and `0` indexer dispatches (aliases both
    ///   `shared_kv_owner` and owner `index_buf`), runs attend only.
    pub fn encodeMlaCsa2Layer(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: MlaCsa2LayerArgs,
        insert_barrier_before: bool,
    ) EncodeError!MlaCsa2StepDispatches {
        var counts: MlaCsa2StepDispatches = .{};
        var next_barrier = insert_barrier_before;

        // 1. FP4 KV write + summary update: `Full` layers only, `loop_step == 0` only.
        if (args.mode == .full and args.loop_step == 0) {
            if (args.write_latent_in) |lat_in| {
                if (args.write_num_tokens > 0) {
                    try self.encodeMlaFp4KvWrite(frame, .{
                        .latent_in = lat_in,
                        .rope_in = args.write_rope_in,
                        .kv_arena = args.kv_arena,
                        .kv_metal_buf = args.kv_metal_buf,
                        .kv_slot = args.shared_kv_owner,
                        .swa_ring = args.swa_ring,
                        .swa_metal_buf = args.swa_metal_buf,
                        .num_tokens = args.write_num_tokens,
                        .start_pos = args.write_start_pos,
                        .is_encoder_projection = args.is_encoder_projection,
                    }, next_barrier);
                    counts.kv_writes += 1;
                    next_barrier = true;

                    if (args.summary_arena) |sum_arena| {
                        const sum_buf = args.summary_metal_buf orelse return error.InvalidConfig;
                        const coarse_span = sum_arena.spec.block_tokens * sum_arena.spec.coarse_group_blocks;
                        const start_coarse = args.write_start_pos / coarse_span;
                        try self.encodeCsa2UpdateSummaries(frame, .{
                            .kv_arena = args.kv_arena,
                            .kv_metal_buf = args.kv_metal_buf,
                            .summary_arena = sum_arena,
                            .summary_metal_buf = sum_buf,
                            .kv_slot = args.shared_kv_owner,
                            .kv_len = args.kv_len,
                            .start_coarse_block = start_coarse,
                        }, next_barrier);
                        counts.summary_updates += 1;
                        next_barrier = true;
                    }
                }
            }
        }

        // 2. Hierarchical CSA2 indexing: `Full` and `Reindex` on `loop_step == 0` only
        //    (strictly skipped in `Reuse` mode and on `loop_step >= 1`).
        if (args.mode != .reuse and args.loop_step == 0) {
            const sum_arena = args.summary_arena orelse return error.InvalidConfig;
            const sum_buf = args.summary_metal_buf orelse return error.InvalidConfig;
            try self.encodeCsa2IndexBlocks(frame, .{
                .q_nope = args.q_nope,
                .summary_arena = sum_arena,
                .summary_metal_buf = sum_buf,
                .kv_slot = args.shared_kv_owner,
                .index_buf = args.index_buf,
                .index_metal_buf = args.index_metal_buf,
                .num_nodes = args.num_nodes,
                .num_q_heads = args.num_q_heads,
                .kv_lora_rank = args.kv_arena.spec.kv_lora_rank,
                .kv_len = args.kv_len,
                .encoder_prefix_len = args.encoder_prefix_len,
                .causal_q_start_pos = args.causal_q_start_pos,
                .swa_window = args.swa_window,
                .top_k_blocks = args.top_k_blocks,
                .top_k_coarse = args.top_k_coarse,
                .num_local_blocks = args.num_local_blocks,
                .batch_kv_lens = args.batch_kv_lens,
                .scale = args.scale,
            }, next_barrier);
            counts.index_dispatches += 1;
            next_barrier = true;
        }

        // 3. Online-softmax absorbed MLA + CSA2 + SWA bounded replay attention.
        try self.encodeMlaCsa2Attend(frame, .{
            .q_nope = args.q_nope,
            .q_pe = args.q_pe,
            .kv_arena = args.kv_arena,
            .kv_metal_buf = args.kv_metal_buf,
            .kv_slot = args.shared_kv_owner,
            .swa_ring = args.swa_ring,
            .swa_metal_buf = args.swa_metal_buf,
            .index_buf = args.index_buf,
            .index_metal_buf = args.index_metal_buf,
            .kv_tree_latent = args.kv_tree_latent,
            .kv_tree_rope = args.kv_tree_rope,
            .ancestor_masks = args.ancestor_masks,
            .batch_kv_lens = args.batch_kv_lens,
            .out = args.out,
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .kv_len = args.kv_len,
            .encoder_prefix_len = args.encoder_prefix_len,
            .causal_q_start_pos = args.causal_q_start_pos,
            .swa_window = args.swa_window,
            .block_tokens = args.block_tokens,
            .top_k_blocks = args.top_k_blocks,
            .scale = args.scale,
        }, next_barrier);
        counts.attn_dispatches += 1;
        return counts;
    }

    // -----------------------------------------------------------------------
    // M7 (G10): Single-Pass Mega-mHC Encoders (`mhc_single_pass`)
    // -----------------------------------------------------------------------

    /// Single-pass Mega-mHC pre-sublayer kernel (`ami_mhc_single_pass`): fuses
    /// `2..8` hyper-stream mixing, optional RMSNorm, and optional in-place
    /// Birkhoff-projected `H_res` stream mixing with zero extra global buffers.
    pub fn encodeMegaMhc(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: MegaMhcArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0) return error.ZeroDimension;
        try args.config.validate();
        if (args.config.apply_h_res_pre and args.h_res == null) return error.InvalidConfig;

        var flags: u32 = 0;
        if (args.norm_w != null) flags |= 1;
        if (args.config.norm_dtype == .bf16) flags |= 2;
        if (args.config.apply_h_res_pre) flags |= 4;
        if (args.config.activate_coeffs) flags |= 8;
        if (args.config.shared_coeffs) flags |= 16;

        const params: MhcParams = .{
            .m = args.m,
            .num_streams = args.config.num_streams,
            .hidden_size = args.config.hidden_size,
            .flags = flags,
            .sinkhorn_iters = args.config.sinkhorn_iters,
            .rms_eps = args.config.rms_eps,
            .post_scale = args.config.post_scale,
        };
        const pipe = self.mhc_single_pass[grIndex(args.config.stream_dtype)][grIndex(args.config.act_dtype)];
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.streams,
                args.h_pre,
                args.h_res orelse args.h_pre,
                args.norm_w orelse args.streams,
                args.branch_out,
            },
            std.mem.asBytes(&params),
            .{ args.m, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Single-pass Mega-mHC post-sublayer residual update (`ami_mhc_post_mix`):
    /// fuses optional Sinkhorn-Knopp Birkhoff `H_res` mixing + `H_post * y`
    /// back across all `num_streams` hyper-streams in-place (0 intermediates).
    pub fn encodeMegaMhcPostMix(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: MegaMhcPostMixArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0) return error.ZeroDimension;
        try args.config.validate();
        if (args.config.apply_h_res_post and args.h_res == null) return error.InvalidConfig;

        var flags: u32 = 0;
        if (args.config.apply_h_res_post) flags |= 4;
        if (args.config.activate_coeffs) flags |= 8;
        if (args.config.shared_coeffs) flags |= 16;

        const params: MhcParams = .{
            .m = args.m,
            .num_streams = args.config.num_streams,
            .hidden_size = args.config.hidden_size,
            .flags = flags,
            .sinkhorn_iters = args.config.sinkhorn_iters,
            .rms_eps = args.config.rms_eps,
            .post_scale = args.config.post_scale,
        };
        const pipe = self.mhc_post_mix[grIndex(args.config.stream_dtype)][grIndex(args.config.act_dtype)];
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.streams,
                args.sublayer_out,
                args.h_post,
                args.h_res orelse args.h_post,
            },
            std.mem.asBytes(&params),
            .{ args.m, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    // -----------------------------------------------------------------------
    // G26 Encoders: Asymmetric K/V (`192/128`, `256/256`, dynamic <= 256)
    // SWA (`128 + sink`), Partial/3D-MRoPE + `(1+w)` QK-Norm + `v_scale`,
    // Sigmoid + `e_score_correction_bias` MoE Routing, and DFlash Tap Capture
    // -----------------------------------------------------------------------

    /// Encodes `ami_g26_qknorm_rope_kv`: asymmetric `K/V <= 256` optional QK-Norm
    /// (supporting `qwen3_5` `(1 + w)` unit offset), partial NeoX/YaRN/3D-MRoPE,
    /// `attention_value_scale`, and direct bindless `HybridKvRing` tile write OR
    /// contiguous `k_out`/`v_out` staging. Zero heap allocation.
    pub fn encodeG26QkNormRopeKv(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G26QkNormRopeKvArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_tokens == 0 or args.num_q_heads == 0 or args.num_kv_heads == 0 or
            args.k_head_dim == 0 or args.v_head_dim == 0 or args.chunk_tokens == 0)
        {
            return error.ZeroDimension;
        }
        if (args.k_head_dim > MAX_QSA_HEAD_DIM or (args.k_head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.v_head_dim > MAX_QSA_HEAD_DIM or (args.v_head_dim % 4) != 0) return error.InvalidHeadDim;
        const rotary_dim: u32 = if (args.rotary_dim == 0) args.k_head_dim else args.rotary_dim;
        if (rotary_dim > args.k_head_dim or (rotary_dim % 2) != 0) return error.InvalidRotaryDim;
        const pos_mode: u32 = if (args.positions_3d != null)
            2
        else if (args.positions_1d != null)
            1
        else
            0;
        if (pos_mode == 2) {
            try args.mrope.validate(rotary_dim);
        }
        const apply_norm = (args.q_norm != null or args.k_norm != null);
        if (apply_norm and (args.q_norm == null or args.k_norm == null)) return error.MissingNormWeights;
        if (args.write_to_tiles and args.kv_tiles == null) return error.InvalidConfig;
        if (args.ring_capacity == 0) {
            const retired_tokens = std.math.mul(u64, args.num_retired_chunks, args.chunk_tokens) catch return error.InvalidConfig;
            if (args.write_to_tiles and pos_mode == 0 and @as(u64, args.start_pos) < retired_tokens) return error.InvalidConfig;
        } else if ((args.ring_capacity % args.chunk_tokens) != 0) {
            return error.InvalidConfig;
        }

        const params: G26QkNormRopeKvParams = .{
            .num_tokens = args.num_tokens,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .k_head_dim = args.k_head_dim,
            .v_head_dim = args.v_head_dim,
            .rotary_dim = rotary_dim,
            .start_pos = args.start_pos,
            .chunk_tokens = args.chunk_tokens,
            .num_retired_chunks = args.num_retired_chunks,
            .ring_capacity = args.ring_capacity,
            .tile_table_offset = args.tile_table_offset,
            .mrope_s0 = args.mrope.section[0],
            .mrope_s1 = args.mrope.section[1],
            .mrope_s2 = args.mrope.section[2],
            .pos_mode = pos_mode,
            .apply_qk_norm = if (apply_norm) 1 else 0,
            .qk_norm_unit_offset = if (args.qk_norm_unit_offset) 1 else 0,
            .kv_only = if (args.kv_only) 1 else 0,
            .write_to_tiles = if (args.write_to_tiles) 1 else 0,
            .rms_eps = args.rms_eps,
            .rope_theta = args.rope_theta,
            .v_scale = args.v_scale,
            .yarn_factor = args.yarn.factor,
            .yarn_mscale = args.yarn.effectiveMscale(),
            .yarn_beta_fast = args.yarn.beta_fast,
            .yarn_beta_slow = args.yarn.beta_slow,
            .original_max_pos = args.yarn.original_max_pos,
        };
        const pipe = if (args.norm_dtype == .bf16) self.g26_qknorm_rope_kv_bf16 else self.g26_qknorm_rope_kv_f16;
        const total_heads: u64 = if (args.kv_only)
            args.num_kv_heads
        else
            @as(u64, args.num_q_heads) + args.num_kv_heads;
        const pos_bind = args.positions_3d orelse args.positions_1d orelse args.q;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.q,
                args.k,
                args.v,
                args.q_norm orelse args.q,
                args.k_norm orelse args.q,
                pos_bind,
                args.k_out orelse args.k,
                args.v_out orelse args.v,
                args.kv_tiles orelse args.q,
            },
            std.mem.asBytes(&params),
            .{ total_heads, args.num_tokens, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_g26_kv_commit`: commits asymmetric `K/V <= 256` rows from
    /// either a prefill chunk (trailing `min(num_tokens, ring_capacity)`) or an
    /// accepted speculative tree path (`accepted_nodes`) into bindless
    /// `HybridKvRing` / global `KvTilePair` tiles. Zero heap allocation.
    pub fn encodeG26KvCommit(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G26KvCommitArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_kv_heads == 0 or args.k_head_dim == 0 or args.v_head_dim == 0 or args.chunk_tokens == 0) {
            return error.ZeroDimension;
        }
        if (args.k_head_dim > MAX_QSA_HEAD_DIM or (args.k_head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.v_head_dim > MAX_QSA_HEAD_DIM or (args.v_head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.ring_capacity > 0 and (args.ring_capacity % args.chunk_tokens) != 0) return error.InvalidConfig;

        var params: G26KvCommitParams = .{
            .num_kv_heads = args.num_kv_heads,
            .k_head_dim = args.k_head_dim,
            .v_head_dim = args.v_head_dim,
            .start_pos = args.start_pos,
            .num_tokens = 0,
            .chunk_tokens = args.chunk_tokens,
            .num_retired_chunks = args.num_retired_chunks,
            .ring_capacity = args.ring_capacity,
            .tile_table_offset = args.tile_table_offset,
            .use_node_list = 0,
            .use_node_buffer = 0,
        };
        var commit_cnt: u32 = 0;
        var nodes_bind = args.k_src;
        if (args.accepted_nodes) |nodes| {
            if (nodes.len == 0 or nodes.len > MAX_TREE_NODES) return error.InvalidNodeCount;
            params.use_node_list = 1;
            params.num_tokens = @intCast(nodes.len);
            for (nodes, 0..) |nid, i| params.inline_nodes[i] = nid;
            commit_cnt = @intCast(nodes.len);
        } else if (args.accepted_nodes_buf) |nb| {
            if (args.num_tokens == 0 or args.num_tokens > MAX_TREE_NODES) return error.InvalidNodeCount;
            params.use_node_list = 1;
            params.use_node_buffer = 1;
            params.num_tokens = args.num_tokens;
            nodes_bind = nb;
            commit_cnt = args.num_tokens;
        } else {
            if (args.num_tokens == 0) return error.ZeroDimension;
            params.num_tokens = args.num_tokens;
            commit_cnt = if (args.ring_capacity > 0) @min(args.num_tokens, args.ring_capacity) else args.num_tokens;
        }

        try frame.encodeThreadgroups(
            self.g26_kv_commit,
            &.{ args.k_src, args.v_src, nodes_bind, args.kv_tiles },
            std.mem.asBytes(&params),
            .{ args.num_kv_heads, commit_cnt, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_g26_swa_attn`: asymmetric `K/V <= 256` (`192/128`, `256/256`,
    /// or dynamic) `HybridKvRing` SWA attention across `M = 1` decode, chunked
    /// prefill (`is_prefill = true`), and `M = 1..64` speculative `verifyTree`
    /// (`ancestor_masks` + per-node `node_positions`) with exact FP32 online
    /// softmax `sink_bias`. Zero heap allocation.
    pub fn encodeG26SwaAttn(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G26SwaAttnArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_q == 0 or args.num_q_heads == 0 or args.num_kv_heads == 0 or
            args.k_head_dim == 0 or args.v_head_dim == 0 or args.chunk_tokens == 0)
        {
            return error.ZeroDimension;
        }
        if ((args.num_q_heads % args.num_kv_heads) != 0) return error.InvalidConfig;
        if (args.k_head_dim > MAX_QSA_HEAD_DIM or (args.k_head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.v_head_dim > MAX_QSA_HEAD_DIM or (args.v_head_dim % 4) != 0) return error.InvalidHeadDim;
        if (args.ring_capacity > 0 and (args.ring_capacity % args.chunk_tokens) != 0) return error.InvalidConfig;
        if (args.is_prefill) {
            if (args.k_extra == null or args.v_extra == null) return error.InvalidConfig;
            if (args.chunk_start_pos > 0 and args.kv_tiles == null) return error.InvalidConfig;
        } else {
            if (args.kv_len > 0 and args.kv_tiles == null) return error.InvalidConfig;
            const has_tree = (args.k_extra != null and args.v_extra != null and args.ancestor_masks != null);
            if (!has_tree and args.kv_len == 0 and args.sink_bias == null) return error.InvalidConfig;
        }

        const scale: f32 = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.k_head_dim))));
        const gqa_ratio = args.num_q_heads / args.num_kv_heads;
        const q_groups_per_kv = (gqa_ratio + 7) / 8;
        const has_tree = (!args.is_prefill and args.k_extra != null and args.v_extra != null and args.ancestor_masks != null);

        const params: G26SwaAttnParams = .{
            .num_q = args.num_q,
            .q_start_pos = args.q_start_pos,
            .chunk_start_pos = args.chunk_start_pos,
            .kv_len = args.kv_len,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .k_head_dim = args.k_head_dim,
            .v_head_dim = args.v_head_dim,
            .window_size = args.window_size,
            .ring_capacity = args.ring_capacity,
            .chunk_tokens = args.chunk_tokens,
            .tile_table_offset = args.tile_table_offset,
            .is_prefill = if (args.is_prefill) 1 else 0,
            .has_tree = if (has_tree) 1 else 0,
            .has_positions = if (!args.is_prefill and args.node_positions != null) 1 else 0,
            .has_sink = if (args.sink_bias != null) 1 else 0,
            .sink_dtype = @backingInt(args.sink_dtype),
            .scale = scale,
        };
        const pipe = if (args.k_head_dim == 192 and args.v_head_dim == 128)
            self.g26_swa_attn_192_128
        else if (args.k_head_dim == 256 and args.v_head_dim == 256)
            self.g26_swa_attn_256_256
        else
            self.g26_swa_attn_dyn;

        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.q,
                args.kv_tiles orelse args.q,
                args.k_extra orelse args.q,
                args.v_extra orelse args.q,
                args.ancestor_masks orelse args.q,
                args.node_positions orelse args.q,
                args.sink_bias orelse args.q,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ @as(u64, args.num_kv_heads) * q_groups_per_kv, args.num_q, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_moe_route_topk_sigmoid`: FP32 sigmoid + optional
    /// `e_score_correction_bias` + optional group top-k + unbiased weight renorm
    /// + `routed_scaling_factor`, emitting exact `MoeWorkspace` CSR schedule &
    /// G20 indirect dispatch arguments (uniform or G13 5-bucket heterogeneous).
    pub fn encodeG26MoeRouteSigmoid(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G26MoeRouteSigmoidArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_experts == 0 or args.top_k == 0) return error.ZeroDimension;
        if (args.num_experts > 1024 or args.top_k > 16 or args.top_k > args.num_experts) return error.InvalidConfig;
        if (args.is_bucketed and (args.expert_schemes == null or args.bucket_headers == null or
            args.bucket_unique_experts == null or args.bucket_expert_token_offsets == null))
        {
            return error.InvalidConfig;
        }
        const params: G26MoeRouteSigmoidParams = .{
            .m = args.m,
            .num_experts = args.num_experts,
            .top_k = args.top_k,
            .norm_topk_prob = if (args.norm_topk_prob) 1 else 0,
            .has_bias = if (args.bias != null) 1 else 0,
            .bias_dtype = @backingInt(args.bias_dtype),
            .n_group = @max(args.n_group, 1),
            .topk_group = @max(args.topk_group, 1),
            .is_bucketed = if (args.is_bucketed) 1 else 0,
            .bucket_unique_stride = args.bucket_unique_stride,
            .bucket_offsets_stride = args.bucket_offsets_stride,
            .gate_up_groups_x_q = args.gate_up_groups_x_q,
            .gate_up_groups_x_f = args.gate_up_groups_x_f,
            .down_groups_x = args.down_groups_x,
            .down_groups_x_q8 = args.down_groups_x_q8,
            .routed_scaling_factor = args.routed_scaling_factor,
            .group_score_topk = args.group_score_topk,
        };
        const pipe = if (args.logits_f16) self.g26_moe_route_sigmoid_f16 else self.g26_moe_route_sigmoid_f32;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.logits,
                args.bias orelse args.logits,
                args.sched_hdr,
                args.unique_experts,
                args.expert_token_offsets,
                args.token_indices,
                args.routing_weights,
                args.slot_k_indices,
                args.scratch_topk_experts,
                args.scratch_topk_weights,
                args.expert_schemes orelse args.logits,
                args.bucket_headers orelse args.sched_hdr,
                args.bucket_unique_experts orelse args.unique_experts,
                args.bucket_expert_token_offsets orelse args.expert_token_offsets,
            },
            std.mem.asBytes(&params),
            .{ 1, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_g26_capture_tap_ring`: copies layer hidden states (`mode = 0`)
    /// or accepted tree tap nodes (`mode = 1`) into a circular `[cap, num_taps, H]`
    /// `f16` `DFlash` context history ring with zero CPU-GPU syncs.
    pub fn encodeG26CaptureTapRing(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G26TapRingCaptureArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_rows == 0 or args.hidden_size == 0 or args.ring_cap == 0 or args.num_taps == 0) {
            return error.ZeroDimension;
        }
        var params: G26TapRingCaptureParams = .{
            .num_rows = args.num_rows,
            .hidden_size = args.hidden_size,
            .base_pos = args.base_pos,
            .ring_cap = args.ring_cap,
            .num_taps = args.num_taps,
            .tap_idx = args.tap_idx,
            .src_stride_rows = args.src_stride_rows,
            .mode = if (args.accepted_nodes != null) 1 else 0,
        };
        if (args.accepted_nodes) |nodes| {
            if (nodes.len != args.num_rows or nodes.len > MAX_TREE_NODES) return error.InvalidNodeCount;
            for (nodes, 0..) |nid, i| params.inline_nodes[i] = nid;
        }
        const grid_y: u64 = if (args.accepted_nodes != null) args.num_taps else 1;
        try frame.encodeThreadgroups(
            self.g26_capture_tap_ring,
            &.{ args.src, args.ring },
            std.mem.asBytes(&params),
            .{ args.num_rows, grid_y, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    // -----------------------------------------------------------------------
    // G27 Encoders: `GLM-5.2/5.3` (`glm_moe_dsa`) & `GLM-5.3-Flash` (`glm5_next`)
    //   - Single-kernel 20-step `hc_mult = 4` mHC Sinkhorn
    //   - Kimi Delta Attention (`KDA`) decode / prefill / tree_verify / commit
    //   - `GlmMoeDsaIndexer` + `NoPE MLA` (`index_kpool = 4`) sparse attention
    // -----------------------------------------------------------------------

    pub fn encodeG27MhcPreSinkhorn(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G27MhcPreSinkhornArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_size == 0) return error.ZeroDimension;
        const params: G27MhcSinkhornParams = .{
            .m = args.m,
            .hidden_size = args.hidden_size,
            .sinkhorn_iters = args.sinkhorn_iters,
            .flags = if (args.norm_w != null) 1 else 0,
            .fn_dtype = @backingInt(args.fn_dtype),
            .norm_dtype = @backingInt(args.norm_dtype),
            .rms_eps = args.rms_eps,
            .hc_eps = args.hc_eps,
        };
        const sb: usize = if (args.stream_dtype == .bf16) 1 else 0;
        const ab: usize = if (args.act_dtype == .bf16) 1 else 0;
        try frame.encodeThreadgroups(
            self.g27_mhc_pre_sinkhorn[sb][ab],
            &.{
                args.streams,
                args.fn_w,
                args.base_w,
                args.scale_w,
                args.norm_w orelse args.streams,
                args.branch_out,
                args.coeffs_out,
            },
            std.mem.asBytes(&params),
            .{ args.m, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeG27MhcPostMix(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G27MhcPostMixArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.hidden_size == 0) return error.ZeroDimension;
        const params: G27MhcSinkhornParams = .{
            .m = args.m,
            .hidden_size = args.hidden_size,
            .sinkhorn_iters = 0,
            .flags = 0,
            .fn_dtype = 0,
            .norm_dtype = 0,
            .rms_eps = 1e-5,
            .hc_eps = 1e-6,
        };
        const sb: usize = if (args.stream_dtype == .bf16) 1 else 0;
        const ab: usize = if (args.act_dtype == .bf16) 1 else 0;
        try frame.encodeThreadgroups(
            self.g27_mhc_post_mix[sb][ab],
            &.{ args.streams, args.sublayer_out, args.coeffs },
            std.mem.asBytes(&params),
            .{ args.m, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeKdaConv1dSilu(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: KdaConv1dSiluArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.qkv_dim == 0 or args.conv_k == 0) return error.ZeroDimension;
        if (args.conv_k > 8) return error.InvalidConfig;
        if (args.is_tree and args.m > MAX_TREE_NODES) return error.InvalidNodeCount;
        const params: KdaConv1dParams = .{
            .m = args.m,
            .qkv_dim = args.qkv_dim,
            .conv_k = args.conv_k,
            .write_state = if (args.write_state) 1 else 0,
            .is_tree = if (args.is_tree) 1 else 0,
            .w_dtype = @backingInt(args.w_dtype),
        };
        const conv_dim: u64 = 3 * @as(u64, args.qkv_dim);
        const groups_x: u64 = (conv_dim + 255) / 256;
        try frame.encodeThreadgroups(
            self.kda_conv1d_silu,
            &.{
                args.q_in,
                args.k_in,
                args.v_in,
                args.q_conv_w,
                args.k_conv_w,
                args.v_conv_w,
                args.conv_state_in,
                args.conv_state_out,
                args.ancestor_masks orelse args.q_in,
                args.q_out,
                args.k_out,
                args.v_out,
            },
            std.mem.asBytes(&params),
            .{ groups_x, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeKdaComputeGates(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: KdaComputeGatesArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.num_heads == 0 or args.head_dim == 0) return error.ZeroDimension;
        const params: KdaComputeGatesParams = .{
            .m = args.m,
            .num_heads = args.num_heads,
            .head_dim = args.head_dim,
            .lower_bound = args.lower_bound,
            .a_log_dtype = @backingInt(args.a_log_dtype),
            .dt_bias_dtype = @backingInt(args.dt_bias_dtype),
        };
        const total: u64 = @as(u64, args.m) * args.num_heads * args.head_dim;
        const groups_x: u64 = (total + 255) / 256;
        try frame.encodeThreadgroups(
            self.kda_compute_gates,
            &.{ args.f_raw, args.b_raw, args.a_log, args.dt_bias, args.g_out, args.beta_out },
            std.mem.asBytes(&params),
            .{ groups_x, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeKdaDecode(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: KdaDecodeArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try validateGdnDims(args.num_heads, args.num_heads, args.head_k_dim, args.head_v_dim);
        const q_scale = args.q_scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_k_dim))));
        const params: KdaDecodeParams = .{
            .num_heads = args.num_heads,
            .head_k_dim = args.head_k_dim,
            .head_v_dim = args.head_v_dim,
            .q_scale = q_scale,
            .l2_eps = args.l2_eps,
            .rms_eps = args.rms_eps,
            .apply_l2_norm = if (args.apply_l2_norm) 1 else 0,
            .apply_out_norm = if (args.norm_w != null) 1 else 0,
            .apply_out_gate = if (args.z != null) 1 else 0,
            .per_head_norm = if (args.per_head_norm) 1 else 0,
            .beta_apply_sigmoid = if (args.beta_apply_sigmoid) 1 else 0,
        };
        const pipe = switch (args.head_k_dim) {
            64 => if (args.norm_dtype == .bf16) self.kda_decode_d64_bf16 else self.kda_decode_d64_f16,
            128 => if (args.norm_dtype == .bf16) self.kda_decode_d128_bf16 else self.kda_decode_d128_f16,
            else => if (args.norm_dtype == .bf16) self.kda_decode_dyn_bf16 else self.kda_decode_dyn_f16,
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
            .{ args.num_heads, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeKdaPrefill(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: KdaPrefillArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.seq_len == 0) return error.ZeroDimension;
        try validateGdnDims(args.num_heads, args.num_heads, args.head_k_dim, args.head_v_dim);
        const q_scale = args.q_scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_k_dim))));
        const params: KdaPrefillParams = .{
            .seq_len = args.seq_len,
            .num_heads = args.num_heads,
            .head_k_dim = args.head_k_dim,
            .head_v_dim = args.head_v_dim,
            .q_scale = q_scale,
            .l2_eps = args.l2_eps,
            .apply_l2_norm = if (args.apply_l2_norm) 1 else 0,
            .beta_apply_sigmoid = if (args.beta_apply_sigmoid) 1 else 0,
            .write_state_only = if (args.write_state_only) 1 else 0,
            .out_start_token = args.out_start_token,
        };
        const pipe = switch (args.head_k_dim) {
            64 => self.kda_prefill_d64,
            128 => self.kda_prefill_d128,
            else => self.kda_prefill_dyn,
        };
        const v_groups: u64 = (@as(u64, args.head_v_dim) + 7) / 8;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.q, args.k, args.v, args.g, args.beta, args.state_in, args.state_out, args.out },
            std.mem.asBytes(&params),
            .{ args.num_heads, v_groups, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
        if (!args.write_state_only and (args.norm_w != null or args.z != null)) {
            const out_tokens = args.seq_len -| args.out_start_token;
            if (out_tokens > 0) {
                try self.encodeKdaOutNormGate(frame, .{
                    .out = args.out,
                    .z = args.z,
                    .norm_w = args.norm_w,
                    .norm_dtype = args.norm_dtype,
                    .num_tokens = out_tokens,
                    .num_heads = args.num_heads,
                    .head_v_dim = args.head_v_dim,
                    .rms_eps = args.rms_eps,
                    .per_head_norm = args.per_head_norm,
                }, true);
            }
        }
    }

    pub fn encodeKdaTreeVerify(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: KdaTreeVerifyArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_nodes == 0) return error.ZeroDimension;
        if (args.num_nodes > MAX_TREE_NODES) return error.InvalidNodeCount;
        try validateGdnDims(args.num_heads, args.num_heads, args.head_k_dim, args.head_v_dim);
        const q_scale = args.q_scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_k_dim))));
        const params: KdaTreeVerifyParams = .{
            .num_nodes = args.num_nodes,
            .num_heads = args.num_heads,
            .head_k_dim = args.head_k_dim,
            .head_v_dim = args.head_v_dim,
            .q_scale = q_scale,
            .l2_eps = args.l2_eps,
            .apply_l2_norm = if (args.apply_l2_norm) 1 else 0,
            .beta_apply_sigmoid = if (args.beta_apply_sigmoid) 1 else 0,
            .scratch_leaf_node = args.scratch_leaf_node orelse std.math.maxInt(u32),
        };
        const pipe = switch (args.head_k_dim) {
            64 => self.kda_tree_verify_d64,
            128 => self.kda_tree_verify_d128,
            else => self.kda_tree_verify_dyn,
        };
        const v_groups: u64 = (@as(u64, args.head_v_dim) + 7) / 8;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.q,
                args.k,
                args.v,
                args.g,
                args.beta,
                args.ancestor_masks,
                args.state_in,
                args.state_out orelse args.state_in,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.num_heads, v_groups, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
        if (args.norm_w != null or args.z != null) {
            try self.encodeKdaOutNormGate(frame, .{
                .out = args.out,
                .z = args.z,
                .norm_w = args.norm_w,
                .norm_dtype = args.norm_dtype,
                .num_tokens = args.num_nodes,
                .num_heads = args.num_heads,
                .head_v_dim = args.head_v_dim,
                .rms_eps = args.rms_eps,
                .per_head_norm = args.per_head_norm,
            }, true);
        }
    }

    pub fn encodeKdaOutNormGate(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: KdaOutNormGateArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_tokens == 0 or args.num_heads == 0 or args.head_v_dim == 0) return error.ZeroDimension;
        const params: GdnOutNormGateParams = .{
            .num_tokens = args.num_tokens,
            .num_v_heads = args.num_heads,
            .head_v_dim = args.head_v_dim,
            .rms_eps = args.rms_eps,
            .apply_out_norm = if (args.norm_w != null) 1 else 0,
            .apply_out_gate = if (args.z != null) 1 else 0,
            .per_head_norm = if (args.per_head_norm) 1 else 0,
        };
        const pipe = if (args.norm_dtype == .bf16) self.kda_out_norm_gate_bf16 else self.kda_out_norm_gate_f16;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.out, args.z orelse args.out, args.norm_w orelse args.out },
            std.mem.asBytes(&params),
            .{ args.num_heads, args.num_tokens, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeKdaCommitTree(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: KdaCommitTreeArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.accepted_nodes.len == 0) return;
        if (args.accepted_nodes.len > MAX_TREE_NODES) return error.InvalidNodeCount;
        try validateGdnDims(args.num_heads, args.num_heads, args.head_k_dim, args.head_v_dim);
        var params: KdaCommitTreeParams = .{
            .num_heads = args.num_heads,
            .head_k_dim = args.head_k_dim,
            .head_v_dim = args.head_v_dim,
            .accepted_count = @intCast(args.accepted_nodes.len),
            .use_node_buffer = 0,
            .l2_eps = args.l2_eps,
            .apply_l2_norm = if (args.apply_l2_norm) 1 else 0,
            .beta_apply_sigmoid = if (args.beta_apply_sigmoid) 1 else 0,
        };
        for (args.accepted_nodes, 0..) |nid, i| params.inline_nodes[i] = nid;
        const v_groups: u64 = (@as(u64, args.head_v_dim) + 7) / 8;
        try frame.encodeThreadgroups(
            self.kda_commit_tree,
            &.{ args.k_tree, args.v_tree, args.g_tree, args.beta_tree, args.k_tree, args.state_in, args.state_out },
            std.mem.asBytes(&params),
            .{ args.num_heads, v_groups, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeG27RopeInterleave(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G27RopeInterleaveArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_tokens == 0 or args.num_heads == 0 or args.rot_dim < 2) return error.ZeroDimension;
        if ((args.rot_dim % 2) != 0 or args.rot_dim > 256) return error.InvalidHeadDim;
        const params: G27RopeInterleaveParams = .{
            .num_tokens = args.num_tokens,
            .num_heads = args.num_heads,
            .head_stride = args.head_stride,
            .rot_dim = args.rot_dim,
            .rot_offset = args.rot_offset,
            .base_pos = args.base_pos,
            .use_pos_buf = if (args.positions != null) 1 else 0,
            .rope_theta = args.rope_theta,
            .attn_scaling = args.attn_scaling,
        };
        try frame.encodeThreadgroups(
            self.g27_rope_interleave,
            &.{ args.x, args.positions orelse args.x },
            std.mem.asBytes(&params),
            .{ args.num_heads, args.num_tokens, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeG27LayerNorm(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G27LayerNormArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_rows == 0 or args.dim == 0) return error.ZeroDimension;
        const params: G27LayerNormParams = .{
            .num_rows = args.num_rows,
            .dim = args.dim,
            .w_dtype = @backingInt(args.w_dtype),
            .b_dtype = @backingInt(args.b_dtype),
            .eps = args.eps,
            .has_bias = if (args.bias != null) 1 else 0,
        };
        try frame.encodeThreadgroups(
            self.g27_layernorm,
            &.{ args.x, args.weight, args.bias orelse args.weight, args.y },
            std.mem.asBytes(&params),
            .{ args.num_rows, 1, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeG27MlaKvWrite(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G27MlaKvWriteArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_tokens == 0 or args.kv_lora_rank == 0) return error.ZeroDimension;
        var params: G27MlaKvWriteParams = .{
            .num_tokens = args.num_tokens,
            .kv_lora_rank = args.kv_lora_rank,
            .qk_rope_head_dim = args.qk_rope_head_dim,
            .start_pos = args.start_pos,
            .is_tree_commit = if (args.accepted_nodes != null) 1 else 0,
        };
        if (args.accepted_nodes) |nodes| {
            if (nodes.len != args.num_tokens or nodes.len > MAX_TREE_NODES) return error.InvalidNodeCount;
            for (nodes, 0..) |nid, i| params.inline_nodes[i] = nid;
        }
        try frame.encodeThreadgroups(
            self.g27_mla_kv_write,
            &.{ args.latent_in, args.rope_in orelse args.latent_in, args.kv_cache },
            std.mem.asBytes(&params),
            .{ args.num_tokens, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeG27DsaIdxKvWrite(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G27DsaIdxKvWriteArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_tokens == 0 or args.index_head_dim == 0) return error.ZeroDimension;
        var params: G27DsaIdxKvWriteParams = .{
            .num_tokens = args.num_tokens,
            .index_head_dim = args.index_head_dim,
            .start_pos = args.start_pos,
            .has_gate = if (args.gate_in != null and args.gate_cache != null) 1 else 0,
            .is_tree_commit = if (args.accepted_nodes != null) 1 else 0,
        };
        if (args.accepted_nodes) |nodes| {
            if (nodes.len != args.num_tokens or nodes.len > MAX_TREE_NODES) return error.InvalidNodeCount;
            for (nodes, 0..) |nid, i| params.inline_nodes[i] = nid;
        }
        try frame.encodeThreadgroups(
            self.g27_dsa_idx_kv_write,
            &.{
                args.k_in,
                args.gate_in orelse args.k_in,
                args.k_cache,
                args.gate_cache orelse args.k_cache,
            },
            std.mem.asBytes(&params),
            .{ args.num_tokens, 1, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeG27DsaKpoolBuild(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G27DsaKpoolBuildArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.end_pool <= args.start_pool) return;
        if (args.index_head_dim == 0 or args.kpool == 0) return error.ZeroDimension;
        const params: G27DsaKpoolBuildParams = .{
            .start_pool = args.start_pool,
            .end_pool = args.end_pool,
            .index_head_dim = args.index_head_dim,
            .kpool = args.kpool,
            .ape_dtype = @backingInt(args.ape_dtype),
        };
        try frame.encodeThreadgroups(
            self.g27_dsa_kpool_build,
            &.{ args.k_cache, args.gate_cache, args.kpool_ape, args.pool_k_cache },
            std.mem.asBytes(&params),
            .{ args.end_pool - args.start_pool, 1, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeG27DsaScoresAndTopK(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G27DsaIndexArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_rows == 0 or args.index_n_heads == 0 or args.index_head_dim == 0 or args.max_out_k == 0) {
            return error.ZeroDimension;
        }
        if (args.is_tree and args.num_rows > MAX_TREE_NODES) return error.InvalidNodeCount;
        const kpool: u32 = @max(args.kpool, 1);
        const total_vis_upper: u32 = if (args.is_tree) (args.committed_kv_len + args.num_rows) else (args.causal_base_pos + args.num_rows);
        const num_candidates: u32 = if (kpool <= 1) total_vis_upper else (total_vis_upper / kpool);

        var select_barrier = insert_barrier_before;
        if (num_candidates > 0) {
            const score_params: G27DsaScoresParams = .{
                .num_rows = args.num_rows,
                .index_n_heads = args.index_n_heads,
                .index_head_dim = args.index_head_dim,
                .committed_kv_len = args.committed_kv_len,
                .num_candidates = num_candidates,
                .kpool = kpool,
                .causal_base_pos = args.causal_base_pos,
                .is_tree = if (args.is_tree) 1 else 0,
                .weights_f16 = if (args.weights_f16) 1 else 0,
                .ape_dtype = @backingInt(args.ape_dtype),
                .softmax_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(args.index_head_dim))),
                .head_weight_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(args.index_n_heads))),
            };
            const groups_x: u64 = (@as(u64, num_candidates) + 63) / 64;
            try frame.encodeThreadgroups(
                self.g27_dsa_scores,
                &.{
                    args.q_idx,
                    args.weights,
                    args.k_cand_cache,
                    args.k_raw_cache orelse args.k_cand_cache,
                    args.k_tree orelse args.k_cand_cache,
                    args.gate_cache orelse args.k_cand_cache,
                    args.gate_tree orelse args.k_cand_cache,
                    args.kpool_ape orelse args.k_cand_cache,
                    args.ancestor_masks orelse args.q_idx,
                    args.scores_scratch,
                },
                std.mem.asBytes(&score_params),
                .{ groups_x, args.num_rows, 1 },
                .{ 64, 1, 1 },
                insert_barrier_before,
            );
            select_barrier = true;
        }

        const sel_params: G27DsaSelectTopKParams = .{
            .num_rows = args.num_rows,
            .committed_kv_len = args.committed_kv_len,
            .num_candidates = num_candidates,
            .index_topk = args.index_topk,
            .max_out_k = args.max_out_k,
            .kpool = kpool,
            .always_select_tail = if (args.always_select_tail) 1 else 0,
            .causal_base_pos = args.causal_base_pos,
            .is_tree = if (args.is_tree) 1 else 0,
        };
        try frame.encodeThreadgroups(
            self.g27_dsa_select_topk,
            &.{
                args.scores_scratch,
                args.ancestor_masks orelse args.q_idx,
                args.out_indices,
                args.out_counts,
            },
            std.mem.asBytes(&sel_params),
            .{ args.num_rows, 1, 1 },
            .{ 256, 1, 1 },
            select_barrier,
        );
    }

    pub fn encodeG27MlaSparseAttn(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G27MlaSparseAttnArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_rows == 0 or args.num_q_heads == 0 or args.kv_lora_rank == 0 or args.max_out_k == 0) {
            return error.ZeroDimension;
        }
        const params: G27MlaSparseAttnParams = .{
            .num_rows = args.num_rows,
            .num_q_heads = args.num_q_heads,
            .kv_lora_rank = args.kv_lora_rank,
            .qk_rope_head_dim = args.qk_rope_head_dim,
            .committed_kv_len = args.committed_kv_len,
            .max_out_k = args.max_out_k,
            .scale = args.scale,
        };
        try frame.encodeThreadgroups(
            self.g27_mla_sparse_attend,
            &.{
                args.q_nope,
                args.q_pe orelse args.q_nope,
                args.kv_cache,
                args.kv_tree_latent orelse args.kv_cache,
                args.kv_tree_rope orelse args.kv_cache,
                args.selected_indices,
                args.selected_counts,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.num_q_heads, args.num_rows, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_g28_mla_full_attend`: full-causal absorbed MLA attention without
    /// sparse indexer (`Ling-3.1` `bailing_hybrid`), supporting decode, wide prefill,
    /// and `verifyTree` with `ancestor_masks`. Zero heap allocation.
    pub fn encodeG28MlaFullAttn(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G28MlaFullAttnArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_rows == 0 or args.num_q_heads == 0 or args.kv_lora_rank == 0) {
            return error.ZeroDimension;
        }
        if (args.is_tree and args.num_rows > MAX_TREE_NODES) return error.InvalidNodeCount;
        const params: G28MlaFullAttnParams = .{
            .num_rows = args.num_rows,
            .num_q_heads = args.num_q_heads,
            .kv_lora_rank = args.kv_lora_rank,
            .qk_rope_head_dim = args.qk_rope_head_dim,
            .committed_kv_len = args.committed_kv_len,
            .causal_base_pos = args.causal_base_pos,
            .is_tree = if (args.is_tree) 1 else 0,
            .scale = args.scale,
        };
        try frame.encodeThreadgroups(
            self.g28_mla_full_attend,
            &.{
                args.q_nope,
                args.q_pe orelse args.q_nope,
                args.kv_cache,
                args.kv_tree_latent orelse args.kv_cache,
                args.kv_tree_rope orelse args.kv_cache,
                args.ancestor_masks orelse args.q_nope,
                args.out,
            },
            std.mem.asBytes(&params),
            .{ args.num_q_heads, args.num_rows, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_g28_mla_out_gate`: in-place `attn_out *= sigmoid(float(gate))`
    /// (head-wise `[num_rows, num_heads]` or element-wise `[num_rows, num_heads * v_head_dim]`)
    /// applied to MLA output before `self_attn.o_proj`. Zero heap allocation.
    pub fn encodeG28MlaOutGate(
        self: HybridPipelines,
        frame: *metal.Compute.Frame,
        args: G28MlaOutGateArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.num_rows == 0 or args.num_heads == 0 or args.v_head_dim == 0) {
            return error.ZeroDimension;
        }
        const total: u64 = @as(u64, args.num_rows) * args.num_heads * args.v_head_dim;
        const params: G28MlaOutGateParams = .{
            .num_rows = args.num_rows,
            .num_heads = args.num_heads,
            .v_head_dim = args.v_head_dim,
            .head_wise = if (args.head_wise) 1 else 0,
        };
        const groups_x: u64 = (total + 255) / 256;
        try frame.encodeThreadgroups(
            self.g28_mla_out_gate,
            &.{ args.attn_out, args.gate },
            std.mem.asBytes(&params),
            .{ groups_x, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }
};

/// Element scalar dtype tag shared by `G26` sink bias and MoE router bias buffers.
pub const ScalarDType = enum(u32) {
    f16 = 0,
    bf16 = 1,
    f32 = 2,
};

// ===========================================================================
// M9.2 (`G27`) Parameter Payloads, Wired Arenas, and Argument Structs
// ===========================================================================

pub const G27MhcSinkhornParams = extern struct {
    m: u32,
    hidden_size: u32,
    sinkhorn_iters: u32,
    flags: u32,
    fn_dtype: u32,
    norm_dtype: u32,
    rms_eps: f32,
    hc_eps: f32,

    comptime {
        if (@sizeOf(G27MhcSinkhornParams) != 32) @compileError("G27MhcSinkhornParams must be 32 bytes");
    }
};

pub const G27MhcPreSinkhornArgs = struct {
    streams: metal.BufferBinding,
    fn_w: metal.BufferBinding,
    base_w: metal.BufferBinding,
    scale_w: metal.BufferBinding,
    norm_w: ?metal.BufferBinding = null,
    branch_out: metal.BufferBinding,
    coeffs_out: metal.BufferBinding,
    m: u32,
    hidden_size: u32,
    sinkhorn_iters: u32 = 20,
    stream_dtype: quant.FloatDType = .f16,
    act_dtype: quant.FloatDType = .f16,
    fn_dtype: ScalarDType = .bf16,
    norm_dtype: ScalarDType = .bf16,
    rms_eps: f32 = 1e-5,
    hc_eps: f32 = 1e-6,
};

pub const G27MhcPostMixArgs = struct {
    streams: metal.BufferBinding,
    sublayer_out: metal.BufferBinding,
    coeffs: metal.BufferBinding,
    m: u32,
    hidden_size: u32,
    stream_dtype: quant.FloatDType = .f16,
    act_dtype: quant.FloatDType = .f16,
};

pub const KdaConv1dParams = extern struct {
    m: u32,
    qkv_dim: u32,
    conv_k: u32,
    write_state: u32,
    is_tree: u32,
    w_dtype: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,

    comptime {
        if (@sizeOf(KdaConv1dParams) != 32) @compileError("KdaConv1dParams must be 32 bytes");
    }
};

pub const KdaComputeGatesParams = extern struct {
    m: u32,
    num_heads: u32,
    head_dim: u32,
    lower_bound: f32,
    a_log_dtype: u32,
    dt_bias_dtype: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,

    comptime {
        if (@sizeOf(KdaComputeGatesParams) != 32) @compileError("KdaComputeGatesParams must be 32 bytes");
    }
};

pub const KdaDecodeParams = extern struct {
    num_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: f32,
    l2_eps: f32,
    rms_eps: f32,
    apply_l2_norm: u32,
    apply_out_norm: u32,
    apply_out_gate: u32,
    per_head_norm: u32,
    beta_apply_sigmoid: u32,
    _pad0: u32 = 0,

    comptime {
        if (@sizeOf(KdaDecodeParams) != 48) @compileError("KdaDecodeParams must be 48 bytes");
    }
};

pub const KdaPrefillParams = extern struct {
    seq_len: u32,
    num_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: f32,
    l2_eps: f32,
    apply_l2_norm: u32,
    beta_apply_sigmoid: u32,
    write_state_only: u32,
    out_start_token: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,

    comptime {
        if (@sizeOf(KdaPrefillParams) != 48) @compileError("KdaPrefillParams must be 48 bytes");
    }
};

pub const KdaTreeVerifyParams = extern struct {
    num_nodes: u32,
    num_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: f32,
    l2_eps: f32,
    apply_l2_norm: u32,
    beta_apply_sigmoid: u32,
    scratch_leaf_node: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,

    comptime {
        if (@sizeOf(KdaTreeVerifyParams) != 48) @compileError("KdaTreeVerifyParams must be 48 bytes");
    }
};

pub const KdaCommitTreeParams = extern struct {
    num_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    accepted_count: u32,
    use_node_buffer: u32,
    l2_eps: f32,
    apply_l2_norm: u32,
    beta_apply_sigmoid: u32,
    inline_nodes: [64]u16 = @splat(0),

    comptime {
        if (@sizeOf(KdaCommitTreeParams) != 160) @compileError("KdaCommitTreeParams must be 160 bytes");
    }
};

pub const KdaConv1dSiluArgs = struct {
    q_in: metal.BufferBinding,
    k_in: metal.BufferBinding,
    v_in: metal.BufferBinding,
    q_conv_w: metal.BufferBinding,
    k_conv_w: metal.BufferBinding,
    v_conv_w: metal.BufferBinding,
    conv_state_in: metal.BufferBinding,
    conv_state_out: metal.BufferBinding,
    ancestor_masks: ?metal.BufferBinding = null,
    q_out: metal.BufferBinding,
    k_out: metal.BufferBinding,
    v_out: metal.BufferBinding,
    m: u32,
    qkv_dim: u32,
    conv_k: u32 = 4,
    write_state: bool = true,
    is_tree: bool = false,
    w_dtype: ScalarDType = .f32,
};

pub const KdaComputeGatesArgs = struct {
    f_raw: metal.BufferBinding,
    b_raw: metal.BufferBinding,
    a_log: metal.BufferBinding,
    dt_bias: metal.BufferBinding,
    g_out: metal.BufferBinding,
    beta_out: metal.BufferBinding,
    m: u32,
    num_heads: u32,
    head_dim: u32,
    lower_bound: f32 = -5.0,
    a_log_dtype: ScalarDType = .f32,
    dt_bias_dtype: ScalarDType = .f32,
};

pub const KdaDecodeArgs = struct {
    q: metal.BufferBinding,
    k: metal.BufferBinding,
    v: metal.BufferBinding,
    g: metal.BufferBinding,
    beta: metal.BufferBinding,
    z: ?metal.BufferBinding = null,
    norm_w: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .bf16,
    state_in: metal.BufferBinding,
    state_out: metal.BufferBinding,
    out: metal.BufferBinding,
    num_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: ?f32 = null,
    l2_eps: f32 = 1e-6,
    rms_eps: f32 = 1e-5,
    apply_l2_norm: bool = true,
    per_head_norm: bool = false,
    beta_apply_sigmoid: bool = false,
};

pub const KdaPrefillArgs = struct {
    q: metal.BufferBinding,
    k: metal.BufferBinding,
    v: metal.BufferBinding,
    g: metal.BufferBinding,
    beta: metal.BufferBinding,
    z: ?metal.BufferBinding = null,
    norm_w: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .bf16,
    state_in: metal.BufferBinding,
    state_out: metal.BufferBinding,
    out: metal.BufferBinding,
    seq_len: u32,
    num_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: ?f32 = null,
    l2_eps: f32 = 1e-6,
    rms_eps: f32 = 1e-5,
    apply_l2_norm: bool = true,
    per_head_norm: bool = false,
    beta_apply_sigmoid: bool = false,
    write_state_only: bool = false,
    out_start_token: u32 = 0,
};

pub const KdaTreeVerifyArgs = struct {
    q: metal.BufferBinding,
    k: metal.BufferBinding,
    v: metal.BufferBinding,
    g: metal.BufferBinding,
    beta: metal.BufferBinding,
    z: ?metal.BufferBinding = null,
    norm_w: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .bf16,
    ancestor_masks: metal.BufferBinding,
    state_in: metal.BufferBinding,
    state_out: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    num_nodes: u32,
    num_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: ?f32 = null,
    l2_eps: f32 = 1e-6,
    rms_eps: f32 = 1e-5,
    apply_l2_norm: bool = true,
    per_head_norm: bool = false,
    beta_apply_sigmoid: bool = false,
    scratch_leaf_node: ?u32 = null,
};

pub const KdaOutNormGateArgs = struct {
    out: metal.BufferBinding,
    z: ?metal.BufferBinding = null,
    norm_w: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .bf16,
    num_tokens: u32,
    num_heads: u32,
    head_v_dim: u32,
    rms_eps: f32 = 1e-5,
    per_head_norm: bool = false,
};

pub const KdaCommitTreeArgs = struct {
    k_tree: metal.BufferBinding,
    v_tree: metal.BufferBinding,
    g_tree: metal.BufferBinding,
    beta_tree: metal.BufferBinding,
    accepted_nodes: []const u16,
    state_in: metal.BufferBinding,
    state_out: metal.BufferBinding,
    num_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    l2_eps: f32 = 1e-6,
    apply_l2_norm: bool = true,
    beta_apply_sigmoid: bool = false,
};

pub const G27RopeInterleaveParams = extern struct {
    num_tokens: u32,
    num_heads: u32,
    head_stride: u32,
    rot_dim: u32,
    rot_offset: u32,
    base_pos: u32,
    use_pos_buf: u32,
    rope_theta: f32,
    attn_scaling: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,

    comptime {
        if (@sizeOf(G27RopeInterleaveParams) != 48) @compileError("G27RopeInterleaveParams must be 48 bytes");
    }
};

pub const G27RopeInterleaveArgs = struct {
    x: metal.BufferBinding,
    positions: ?metal.BufferBinding = null,
    num_tokens: u32,
    num_heads: u32,
    head_stride: u32,
    rot_dim: u32,
    rot_offset: u32 = 0,
    base_pos: u32 = 0,
    rope_theta: f32 = 8_000_000.0,
    attn_scaling: f32 = 1.0,
};

pub const G27LayerNormParams = extern struct {
    num_rows: u32,
    dim: u32,
    w_dtype: u32,
    b_dtype: u32,
    eps: f32,
    has_bias: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,

    comptime {
        if (@sizeOf(G27LayerNormParams) != 32) @compileError("G27LayerNormParams must be 32 bytes");
    }
};

pub const G27LayerNormArgs = struct {
    x: metal.BufferBinding,
    weight: metal.BufferBinding,
    bias: ?metal.BufferBinding = null,
    y: metal.BufferBinding,
    num_rows: u32,
    dim: u32,
    w_dtype: ScalarDType = .bf16,
    b_dtype: ScalarDType = .bf16,
    eps: f32 = 1e-6,
};

pub const G27MlaKvWriteParams = extern struct {
    num_tokens: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32,
    start_pos: u32,
    is_tree_commit: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
    inline_nodes: [64]u16 = @splat(0),

    comptime {
        if (@sizeOf(G27MlaKvWriteParams) != 160) @compileError("G27MlaKvWriteParams must be 160 bytes");
    }
};

pub const G27MlaKvWriteArgs = struct {
    latent_in: metal.BufferBinding,
    rope_in: ?metal.BufferBinding = null,
    kv_cache: metal.BufferBinding,
    num_tokens: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32,
    start_pos: u32,
    accepted_nodes: ?[]const u16 = null,
};

pub const G27DsaIdxKvWriteParams = extern struct {
    num_tokens: u32,
    index_head_dim: u32,
    start_pos: u32,
    has_gate: u32,
    is_tree_commit: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
    inline_nodes: [64]u16 = @splat(0),

    comptime {
        if (@sizeOf(G27DsaIdxKvWriteParams) != 160) @compileError("G27DsaIdxKvWriteParams must be 160 bytes");
    }
};

pub const G27DsaIdxKvWriteArgs = struct {
    k_in: metal.BufferBinding,
    gate_in: ?metal.BufferBinding = null,
    k_cache: metal.BufferBinding,
    gate_cache: ?metal.BufferBinding = null,
    num_tokens: u32,
    index_head_dim: u32,
    start_pos: u32,
    accepted_nodes: ?[]const u16 = null,
};

pub const G27DsaKpoolBuildParams = extern struct {
    start_pool: u32,
    end_pool: u32,
    index_head_dim: u32,
    kpool: u32,
    ape_dtype: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,

    comptime {
        if (@sizeOf(G27DsaKpoolBuildParams) != 32) @compileError("G27DsaKpoolBuildParams must be 32 bytes");
    }
};

pub const G27DsaKpoolBuildArgs = struct {
    k_cache: metal.BufferBinding,
    gate_cache: metal.BufferBinding,
    kpool_ape: metal.BufferBinding,
    pool_k_cache: metal.BufferBinding,
    start_pool: u32,
    end_pool: u32,
    index_head_dim: u32,
    kpool: u32 = 4,
    ape_dtype: ScalarDType = .bf16,
};

pub const G27DsaScoresParams = extern struct {
    num_rows: u32,
    index_n_heads: u32,
    index_head_dim: u32,
    committed_kv_len: u32,
    num_candidates: u32,
    kpool: u32,
    causal_base_pos: u32,
    is_tree: u32,
    weights_f16: u32,
    ape_dtype: u32,
    softmax_scale: f32,
    head_weight_scale: f32,

    comptime {
        if (@sizeOf(G27DsaScoresParams) != 48) @compileError("G27DsaScoresParams must be 48 bytes");
    }
};

pub const G27DsaSelectTopKParams = extern struct {
    num_rows: u32,
    committed_kv_len: u32,
    num_candidates: u32,
    index_topk: u32,
    max_out_k: u32,
    kpool: u32,
    always_select_tail: u32,
    causal_base_pos: u32,
    is_tree: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,

    comptime {
        if (@sizeOf(G27DsaSelectTopKParams) != 48) @compileError("G27DsaSelectTopKParams must be 48 bytes");
    }
};

pub const G27DsaIndexArgs = struct {
    q_idx: metal.BufferBinding,
    weights: metal.BufferBinding,
    k_cand_cache: metal.BufferBinding,
    k_raw_cache: ?metal.BufferBinding = null,
    k_tree: ?metal.BufferBinding = null,
    gate_cache: ?metal.BufferBinding = null,
    gate_tree: ?metal.BufferBinding = null,
    kpool_ape: ?metal.BufferBinding = null,
    ancestor_masks: ?metal.BufferBinding = null,
    scores_scratch: metal.BufferBinding,
    out_indices: metal.BufferBinding,
    out_counts: metal.BufferBinding,
    num_rows: u32,
    index_n_heads: u32,
    index_head_dim: u32,
    committed_kv_len: u32,
    index_topk: u32,
    max_out_k: u32,
    kpool: u32 = 1,
    always_select_tail: bool = false,
    causal_base_pos: u32 = 0,
    is_tree: bool = false,
    weights_f16: bool = true,
    ape_dtype: ScalarDType = .bf16,
};

pub const G27MlaSparseAttnParams = extern struct {
    num_rows: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32,
    committed_kv_len: u32,
    max_out_k: u32,
    scale: f32,
    _pad0: u32 = 0,

    comptime {
        if (@sizeOf(G27MlaSparseAttnParams) != 32) @compileError("G27MlaSparseAttnParams must be 32 bytes");
    }
};

pub const G27MlaSparseAttnArgs = struct {
    q_nope: metal.BufferBinding,
    q_pe: ?metal.BufferBinding = null,
    kv_cache: metal.BufferBinding,
    kv_tree_latent: ?metal.BufferBinding = null,
    kv_tree_rope: ?metal.BufferBinding = null,
    selected_indices: metal.BufferBinding,
    selected_counts: metal.BufferBinding,
    out: metal.BufferBinding,
    num_rows: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32 = 0,
    committed_kv_len: u32,
    max_out_k: u32,
    scale: f32,
};

pub const G28MlaFullAttnParams = extern struct {
    num_rows: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32,
    committed_kv_len: u32,
    causal_base_pos: u32,
    is_tree: u32,
    scale: f32,

    comptime {
        if (@sizeOf(G28MlaFullAttnParams) != 32) @compileError("G28MlaFullAttnParams must be 32 bytes");
    }
};

pub const G28MlaFullAttnArgs = struct {
    q_nope: metal.BufferBinding,
    q_pe: ?metal.BufferBinding = null,
    kv_cache: metal.BufferBinding,
    kv_tree_latent: ?metal.BufferBinding = null,
    kv_tree_rope: ?metal.BufferBinding = null,
    ancestor_masks: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    num_rows: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32 = 0,
    committed_kv_len: u32,
    causal_base_pos: u32,
    is_tree: bool = false,
    scale: f32,
};

pub const G28MlaOutGateParams = extern struct {
    num_rows: u32,
    num_heads: u32,
    v_head_dim: u32,
    head_wise: u32,

    comptime {
        if (@sizeOf(G28MlaOutGateParams) != 16) @compileError("G28MlaOutGateParams must be 16 bytes");
    }
};

pub const G28MlaOutGateArgs = struct {
    attn_out: metal.BufferBinding,
    gate: metal.BufferBinding,
    num_rows: u32,
    num_heads: u32,
    v_head_dim: u32,
    head_wise: bool = true,
};

/// Wired preallocated FP16 compressed latent (+ optional RoPE) KV arena for
/// `GLM-5.2/5.3` (`glm_moe_dsa`) and `GLM-5.3-Flash` (`glm5_next` NoPE MLA).
pub const G27MlaKvSpec = struct {
    num_layers: u32,
    max_seq_len: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32 = 0,

    pub fn validate(self: G27MlaKvSpec) ArenaError!void {
        if (self.num_layers == 0 or self.max_seq_len == 0 or self.kv_lora_rank == 0) return error.ZeroDimension;
        if (self.kv_lora_rank > MAX_MLA_LATENT_DIM) return error.InvalidConfig;
        if (self.qk_rope_head_dim > MAX_MLA_ROPE_DIM) return error.InvalidConfig;
    }

    pub fn bytesPerToken(self: G27MlaKvSpec) u32 {
        return (self.kv_lora_rank + self.qk_rope_head_dim) * 2;
    }
};

pub const G27MlaKvArena = struct {
    spec: G27MlaKvSpec,
    layer_stride_bytes: usize,
    base_offset: u64,
    backing: []align(page.APPLE_PAGE_SIZE) u8,

    pub fn requiredBytes(spec: G27MlaKvSpec) ArenaError!usize {
        try spec.validate();
        const row_bytes = @as(usize, spec.bytesPerToken());
        const per_layer_raw = std.math.mul(usize, spec.max_seq_len, row_bytes) catch return error.Overflow;
        const layer_stride = std.mem.alignForward(usize, per_layer_raw, 64);
        const total = std.math.mul(usize, spec.num_layers, layer_stride) catch return error.Overflow;
        return std.mem.alignForward(usize, @max(total, page.APPLE_PAGE_SIZE), page.APPLE_PAGE_SIZE);
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: G27MlaKvSpec) ArenaError!G27MlaKvArena {
        const total = try requiredBytes(spec);
        const row_bytes = @as(usize, spec.bytesPerToken());
        const per_layer_raw = std.math.mul(usize, spec.max_seq_len, row_bytes) catch return error.Overflow;
        const layer_stride = std.mem.alignForward(usize, per_layer_raw, 64);
        const slice = try arena.allocAligned(total, page.APPLE_PAGE_SIZE);
        @memset(slice, 0);
        return .{
            .spec = spec,
            .layer_stride_bytes = layer_stride,
            .base_offset = arena.offsetOf(slice.ptr),
            .backing = @alignCast(slice),
        };
    }

    pub fn layerBinding(self: *const G27MlaKvArena, buf: metal.Buffer, mla_slot: u32) ArenaError!metal.BufferBinding {
        if (mla_slot >= self.spec.num_layers) return error.InvalidLayerIndex;
        return .{
            .buffer = buf,
            .offset = self.base_offset + @as(u64, mla_slot) * @as(u64, self.layer_stride_bytes),
        };
    }
};

/// Wired preallocated FP16 DSA indexer K (+ optional k-pool gate & pooled K) arena
/// covering ONLY `"full"` indexer layers (`21/78` on GLM-5.2/5.3, `11/45` on GLM-5.3-Flash).
pub const G27DsaIdxKvSpec = struct {
    num_full_layers: u32,
    max_seq_len: u32,
    index_head_dim: u32,
    kpool: u32 = 1,

    pub fn validate(self: G27DsaIdxKvSpec) ArenaError!void {
        if (self.num_full_layers == 0 or self.max_seq_len == 0 or self.index_head_dim == 0) return error.ZeroDimension;
        if (self.kpool == 0 or self.kpool > 8) return error.InvalidConfig;
    }
};

pub const G27DsaIdxKvArena = struct {
    spec: G27DsaIdxKvSpec,
    k_layer_stride: usize,
    gate_offset: usize,
    pool_offset: usize,
    pool_layer_stride: usize,
    base_offset: u64,
    backing: []align(page.APPLE_PAGE_SIZE) u8,

    pub fn requiredBytes(spec: G27DsaIdxKvSpec) ArenaError!usize {
        try spec.validate();
        const k_row = @as(usize, spec.index_head_dim) * 2;
        const k_layer = std.mem.alignForward(usize, std.math.mul(usize, spec.max_seq_len, k_row) catch return error.Overflow, 64);
        const k_total = std.math.mul(usize, spec.num_full_layers, k_layer) catch return error.Overflow;
        var total = k_total;
        if (spec.kpool > 1) {
            total = std.math.add(usize, total, k_total) catch return error.Overflow; // gate_cache
            const max_pools: usize = @max(@as(usize, 1), @as(usize, spec.max_seq_len) / spec.kpool);
            const p_layer = std.mem.alignForward(usize, std.math.mul(usize, max_pools, k_row) catch return error.Overflow, 64);
            const p_total = std.math.mul(usize, spec.num_full_layers, p_layer) catch return error.Overflow;
            total = std.math.add(usize, total, p_total) catch return error.Overflow;
        }
        return std.mem.alignForward(usize, @max(total, page.APPLE_PAGE_SIZE), page.APPLE_PAGE_SIZE);
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: G27DsaIdxKvSpec) ArenaError!G27DsaIdxKvArena {
        const total = try requiredBytes(spec);
        const k_row = @as(usize, spec.index_head_dim) * 2;
        const k_layer = std.mem.alignForward(usize, std.math.mul(usize, spec.max_seq_len, k_row) catch return error.Overflow, 64);
        const k_total = std.math.mul(usize, spec.num_full_layers, k_layer) catch return error.Overflow;
        const gate_off = k_total;
        const pool_off = if (spec.kpool > 1) k_total * 2 else k_total;
        const max_pools: usize = @max(@as(usize, 1), @as(usize, spec.max_seq_len) / spec.kpool);
        const p_layer = if (spec.kpool > 1)
            std.mem.alignForward(usize, std.math.mul(usize, max_pools, k_row) catch return error.Overflow, 64)
        else
            0;
        const slice = try arena.allocAligned(total, page.APPLE_PAGE_SIZE);
        @memset(slice, 0);
        return .{
            .spec = spec,
            .k_layer_stride = k_layer,
            .gate_offset = gate_off,
            .pool_offset = pool_off,
            .pool_layer_stride = p_layer,
            .base_offset = arena.offsetOf(slice.ptr),
            .backing = @alignCast(slice),
        };
    }

    pub fn kBinding(self: *const G27DsaIdxKvArena, buf: metal.Buffer, full_slot: u32) ArenaError!metal.BufferBinding {
        if (full_slot >= self.spec.num_full_layers) return error.InvalidLayerIndex;
        return .{ .buffer = buf, .offset = self.base_offset + @as(u64, full_slot) * self.k_layer_stride };
    }

    pub fn gateBinding(self: *const G27DsaIdxKvArena, buf: metal.Buffer, full_slot: u32) ArenaError!metal.BufferBinding {
        if (full_slot >= self.spec.num_full_layers) return error.InvalidLayerIndex;
        return .{ .buffer = buf, .offset = self.base_offset + self.gate_offset + @as(u64, full_slot) * self.k_layer_stride };
    }

    pub fn poolBinding(self: *const G27DsaIdxKvArena, buf: metal.Buffer, full_slot: u32) ArenaError!metal.BufferBinding {
        if (full_slot >= self.spec.num_full_layers) return error.InvalidLayerIndex;
        return .{ .buffer = buf, .offset = self.base_offset + self.pool_offset + @as(u64, full_slot) * self.pool_layer_stride };
    }
};

/// Wired shared DSA Top-K index cache reused across `"shared"` layers (`index_topk_freq = 4`)
/// and exposed via `GraphModel.sharedMtpSparseIndices()` for zero-reindex `1L` `MTP` (`D15`).
pub const G27DsaIndexBufferSpec = struct {
    max_rows: u32,
    max_out_k: u32,
    max_candidates: u32,

    pub fn validate(self: G27DsaIndexBufferSpec) ArenaError!void {
        if (self.max_rows == 0 or self.max_out_k == 0 or self.max_candidates == 0) return error.ZeroDimension;
    }
};

pub const G27DsaIndexBuffer = struct {
    spec: G27DsaIndexBufferSpec,
    indices_off: u64,
    counts_off: u64,
    scores_off: u64,
    indices_host: []u32,
    counts_host: []u32,

    pub fn requiredBytes(spec: G27DsaIndexBufferSpec) ArenaError!usize {
        try spec.validate();
        const idx_b = std.mem.alignForward(usize, std.math.mul(usize, std.math.mul(usize, spec.max_rows, spec.max_out_k) catch return error.Overflow, 4) catch return error.Overflow, 64);
        const cnt_b = std.mem.alignForward(usize, std.math.mul(usize, spec.max_rows, 4) catch return error.Overflow, 64);
        const sc_b = std.mem.alignForward(usize, std.math.mul(usize, std.math.mul(usize, spec.max_rows, spec.max_candidates) catch return error.Overflow, 4) catch return error.Overflow, 64);
        return std.math.add(usize, std.math.add(usize, idx_b, cnt_b) catch return error.Overflow, sc_b) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: G27DsaIndexBufferSpec) ArenaError!G27DsaIndexBuffer {
        try spec.validate();
        const idx_elems = std.math.mul(usize, spec.max_rows, spec.max_out_k) catch return error.Overflow;
        const sc_elems = std.math.mul(usize, spec.max_rows, spec.max_candidates) catch return error.Overflow;
        const idx_raw = try arena.allocAligned(std.math.mul(usize, idx_elems, 4) catch return error.Overflow, 64);
        const cnt_raw = try arena.allocAligned(std.math.mul(usize, spec.max_rows, 4) catch return error.Overflow, 64);
        const sc_raw = try arena.allocAligned(std.math.mul(usize, sc_elems, 4) catch return error.Overflow, 64);
        @memset(idx_raw, 0xFF);
        @memset(cnt_raw, 0);
        @memset(sc_raw, 0);
        return .{
            .spec = spec,
            .indices_off = arena.offsetOf(idx_raw.ptr),
            .counts_off = arena.offsetOf(cnt_raw.ptr),
            .scores_off = arena.offsetOf(sc_raw.ptr),
            .indices_host = @alignCast(std.mem.bytesAsSlice(u32, idx_raw[0 .. idx_elems * 4])),
            .counts_host = @alignCast(std.mem.bytesAsSlice(u32, cnt_raw[0 .. @as(usize, spec.max_rows) * 4])),
        };
    }

    pub fn indicesBinding(self: *const G27DsaIndexBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.indices_off };
    }

    pub fn countsBinding(self: *const G27DsaIndexBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.counts_off };
    }

    pub fn scoresBinding(self: *const G27DsaIndexBuffer, buf: metal.Buffer) metal.BufferBinding {
        return .{ .buffer = buf, .offset = self.scores_off };
    }
};

pub const G26QkNormRopeKvParams = extern struct {
    num_tokens: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    rotary_dim: u32,
    start_pos: u32,
    chunk_tokens: u32,
    num_retired_chunks: u32,
    ring_capacity: u32,
    tile_table_offset: u32,
    mrope_s0: u32,
    mrope_s1: u32,
    mrope_s2: u32,
    pos_mode: u32,
    apply_qk_norm: u32,
    qk_norm_unit_offset: u32,
    kv_only: u32,
    write_to_tiles: u32,
    _pad0: u32 = 0,
    rms_eps: f32,
    rope_theta: f32,
    v_scale: f32,
    yarn_factor: f32,
    yarn_mscale: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    original_max_pos: u32,

    comptime {
        if (@sizeOf(G26QkNormRopeKvParams) != 112) @compileError("G26QkNormRopeKvParams must be 112 bytes");
    }
};

pub const G26QkNormRopeKvArgs = struct {
    q: metal.BufferBinding,
    k: metal.BufferBinding,
    v: metal.BufferBinding,
    q_norm: ?metal.BufferBinding = null,
    k_norm: ?metal.BufferBinding = null,
    norm_dtype: quant.FloatDType = .bf16,
    qk_norm_unit_offset: bool = false,
    positions_1d: ?metal.BufferBinding = null,
    positions_3d: ?metal.BufferBinding = null,
    k_out: ?metal.BufferBinding = null,
    v_out: ?metal.BufferBinding = null,
    kv_tiles: ?metal.BufferBinding = null,
    num_tokens: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    rotary_dim: u32 = 0,
    start_pos: u32 = 0,
    chunk_tokens: u32 = QSA_SWA_WINDOW_TOKENS,
    num_retired_chunks: u32 = 0,
    ring_capacity: u32 = 0,
    tile_table_offset: u32 = 0,
    mrope: MropeSection = MropeSection.QWEN38_DEFAULT,
    yarn: YarnConfig = .{},
    kv_only: bool = false,
    write_to_tiles: bool = false,
    rms_eps: f32 = 1e-6,
    rope_theta: f32 = 10_000.0,
    v_scale: f32 = 1.0,
};

pub const G26KvCommitParams = extern struct {
    num_kv_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    start_pos: u32,
    num_tokens: u32,
    chunk_tokens: u32,
    num_retired_chunks: u32,
    ring_capacity: u32,
    tile_table_offset: u32,
    use_node_list: u32,
    use_node_buffer: u32,
    _pad0: u32 = 0,
    inline_nodes: [MAX_TREE_NODES]u16 = @splat(0),

    comptime {
        if (@sizeOf(G26KvCommitParams) != 176) @compileError("G26KvCommitParams must be 176 bytes");
    }
};

pub const G26KvCommitArgs = struct {
    k_src: metal.BufferBinding,
    v_src: metal.BufferBinding,
    kv_tiles: metal.BufferBinding,
    num_kv_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    start_pos: u32,
    num_tokens: u32 = 0,
    accepted_nodes: ?[]const u16 = null,
    accepted_nodes_buf: ?metal.BufferBinding = null,
    chunk_tokens: u32 = QSA_SWA_WINDOW_TOKENS,
    num_retired_chunks: u32 = 0,
    ring_capacity: u32 = 0,
    tile_table_offset: u32 = 0,
};

pub const G26SwaAttnParams = extern struct {
    num_q: u32,
    q_start_pos: u32,
    chunk_start_pos: u32,
    kv_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    window_size: u32,
    ring_capacity: u32,
    chunk_tokens: u32,
    tile_table_offset: u32,
    is_prefill: u32,
    has_tree: u32,
    has_positions: u32,
    has_sink: u32,
    sink_dtype: u32,
    _pad0: u32 = 0,
    scale: f32,
    _pad1: u32 = 0,

    comptime {
        if (@sizeOf(G26SwaAttnParams) != 80) @compileError("G26SwaAttnParams must be 80 bytes");
    }
};

pub const G26SwaAttnArgs = struct {
    q: metal.BufferBinding,
    kv_tiles: ?metal.BufferBinding = null,
    k_extra: ?metal.BufferBinding = null,
    v_extra: ?metal.BufferBinding = null,
    ancestor_masks: ?metal.BufferBinding = null,
    node_positions: ?metal.BufferBinding = null,
    sink_bias: ?metal.BufferBinding = null,
    sink_dtype: ScalarDType = .bf16,
    out: metal.BufferBinding,
    num_q: u32,
    q_start_pos: u32 = 0,
    chunk_start_pos: u32 = 0,
    kv_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    window_size: u32 = QSA_SWA_WINDOW_TOKENS,
    ring_capacity: u32 = QSA_SWA_WINDOW_TOKENS,
    chunk_tokens: u32 = QSA_SWA_WINDOW_TOKENS,
    tile_table_offset: u32 = 0,
    is_prefill: bool = false,
    scale: ?f32 = null,
};

pub const G26MoeRouteSigmoidParams = extern struct {
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: u32,
    has_bias: u32,
    bias_dtype: u32,
    n_group: u32,
    topk_group: u32,
    is_bucketed: u32,
    bucket_unique_stride: u32,
    bucket_offsets_stride: u32,
    gate_up_groups_x_q: u32,
    gate_up_groups_x_f: u32,
    down_groups_x: u32,
    down_groups_x_q8: u32,
    routed_scaling_factor: f32,
    group_score_topk: u32 = 2,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,

    comptime {
        if (@sizeOf(G26MoeRouteSigmoidParams) != 80) @compileError("G26MoeRouteSigmoidParams must be 80 bytes");
    }
};

pub const G26MoeRouteSigmoidArgs = struct {
    logits: metal.BufferBinding,
    logits_f16: bool = false,
    bias: ?metal.BufferBinding = null,
    bias_dtype: ScalarDType = .f32,
    sched_hdr: metal.BufferBinding,
    unique_experts: metal.BufferBinding,
    expert_token_offsets: metal.BufferBinding,
    token_indices: metal.BufferBinding,
    routing_weights: metal.BufferBinding,
    slot_k_indices: metal.BufferBinding,
    scratch_topk_experts: metal.BufferBinding,
    scratch_topk_weights: metal.BufferBinding,
    expert_schemes: ?metal.BufferBinding = null,
    bucket_headers: ?metal.BufferBinding = null,
    bucket_unique_experts: ?metal.BufferBinding = null,
    bucket_expert_token_offsets: ?metal.BufferBinding = null,
    is_bucketed: bool = false,
    bucket_unique_stride: u32 = 0,
    bucket_offsets_stride: u32 = 0,
    m: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: bool = true,
    n_group: u32 = 1,
    topk_group: u32 = 1,
    group_score_topk: u32 = 2,
    routed_scaling_factor: f32 = 1.0,
    gate_up_groups_x_q: u32,
    gate_up_groups_x_f: u32,
    down_groups_x: u32,
    down_groups_x_q8: u32,
};

pub const G26TapRingCaptureParams = extern struct {
    num_rows: u32,
    hidden_size: u32,
    base_pos: u32,
    ring_cap: u32,
    num_taps: u32,
    tap_idx: u32,
    src_stride_rows: u32,
    mode: u32,
    inline_nodes: [MAX_TREE_NODES]u16 = @splat(0),

    comptime {
        if (@sizeOf(G26TapRingCaptureParams) != 160) @compileError("G26TapRingCaptureParams must be 160 bytes");
    }
};

pub const G26TapRingCaptureArgs = struct {
    src: metal.BufferBinding,
    ring: metal.BufferBinding,
    num_rows: u32,
    hidden_size: u32,
    base_pos: u32,
    ring_cap: u32,
    num_taps: u32,
    tap_idx: u32 = 0,
    src_stride_rows: u32 = 0,
    accepted_nodes: ?[]const u16 = null,
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

/// CPU `fp32` reference for Stage-1 hierarchical CSA2 block summaries (fine + coarse).
pub fn csa2UpdateSummariesCpuRef(
    e2m1_plane: []const u8,
    scale_plane: []const u8,
    kv_len: u32,
    kv_lora_rank: u32,
    index_dim: u32,
    block_tokens: u32,
    coarse_group_blocks: u32,
    fine_out: []f16,
    coarse_out: []f16,
) void {
    if (kv_len == 0) return;
    const total_fine = (kv_len + block_tokens - 1) / block_tokens;
    const g_blocks = @max(coarse_group_blocks, 1);
    const total_coarse = (total_fine + g_blocks - 1) / g_blocks;
    const e2m1_row_bytes: usize = kv_lora_rank / 2;
    const num_groups: usize = kv_lora_rank / MLA_FP4_GROUP_SIZE;

    for (0..total_coarse) |sb| {
        const first_fine = sb * g_blocks;
        const end_fine = @min(first_fine + g_blocks, total_fine);
        const valid_sub: f32 = @floatFromInt(end_fine - first_fine);
        for (0..index_dim) |d| {
            const byte_idx = d >> 1;
            const hi_nib = (d & 1) != 0;
            const grp_idx = d >> 4;
            var coarse_acc: f32 = 0.0;
            for (first_fine..end_fine) |fb| {
                const t0 = fb * block_tokens;
                const t1 = @min(t0 + block_tokens, kv_len);
                var fine_acc: f32 = 0.0;
                for (t0..t1) |t| {
                    const packed_b = e2m1_plane[t * e2m1_row_bytes + byte_idx];
                    const nib: u4 = @truncate(if (hi_nib) (packed_b >> 4) else (packed_b & 0xF));
                    const sc = e4m3ToF32(scale_plane[t * num_groups + grp_idx]);
                    fine_acc += e2m1NibbleToF32(nib) * sc;
                }
                const fine_mean = fine_acc / @as(f32, @floatFromInt(t1 - t0));
                fine_out[fb * index_dim + d] = @floatCast(fine_mean);
                coarse_acc += fine_mean;
            }
            coarse_out[sb * index_dim + d] = @floatCast(coarse_acc / valid_sub);
        }
    }
}

/// CPU `fp32` reference for Stage-2 hierarchical CSA2 sparse block selection.
pub fn csa2IndexBlocksCpuRef(
    q_nope: []const f16,
    fine_keys: []const f16,
    coarse_keys: []const f16,
    num_nodes: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    index_dim: u32,
    kv_len: u32,
    encoder_prefix_len: u32,
    causal_q_start_pos: ?u32,
    swa_window: u32,
    block_tokens: u32,
    coarse_group_blocks: u32,
    top_k_coarse: u32,
    top_k_blocks: u32,
    num_local_blocks: u32,
    scale: f32,
    selected_blocks_out: []u32,
    selected_counts_out: []u32,
) void {
    const k_cap = @min(top_k_blocks, CSA2_MAX_TOP_K_BLOCKS);
    const g_blocks = @max(coarse_group_blocks, 1);

    for (0..num_nodes) |node| {
        const sel_base = node * top_k_blocks;
        @memset(selected_blocks_out[sel_base .. sel_base + top_k_blocks], 0xFFFF_FFFF);
        const eff_kv_len: u32 = if (causal_q_start_pos) |q0|
            @min(kv_len, @max(encoder_prefix_len, q0 + @as(u32, @intCast(node)) + 1))
        else
            kv_len;
        if (eff_kv_len == 0 or k_cap == 0) {
            selected_counts_out[node] = 0;
            continue;
        }
        const swa_lo: u32 = if (swa_window > 0 and eff_kv_len > swa_window) eff_kv_len - swa_window else 0;
        if (swa_window > 0 and swa_lo == 0) {
            selected_counts_out[node] = 0;
            continue;
        }
        const num_valid = (eff_kv_len + block_tokens - 1) / block_tokens;
        const num_cand = if (swa_window > 0) ((swa_lo + block_tokens - 1) / block_tokens) else num_valid;
        if (num_cand <= k_cap) {
            for (0..num_cand) |i| selected_blocks_out[sel_base + i] = @intCast(i);
            selected_counts_out[node] = num_cand;
            continue;
        }

        const local_cnt: u32 = if (swa_window > 0) 0 else @min(@max(num_local_blocks, 1), k_cap);
        const search_end = num_cand - local_cnt;
        const k_sel = k_cap - local_cnt;
        const num_coarse = (search_end + g_blocks - 1) / g_blocks;
        const k_coarse = @min(top_k_coarse, CSA2_MAX_TOP_K_BLOCKS);
        const use_coarse = (g_blocks > 1 and k_coarse > 0 and num_coarse > k_coarse);

        var coarse_sel: [CSA2_MAX_TOP_K_BLOCKS]u32 = undefined;
        var coarse_cnt: u32 = 0;
        if (use_coarse and k_sel > 0) {
            var used_c: [1024]bool = @splat(false);
            std.debug.assert(num_coarse <= used_c.len);
            for (0..k_coarse) |_| {
                var best_s: f32 = -std.math.inf(f32);
                var best_b: u32 = 0xFFFF_FFFF;
                for (0..num_coarse) |sb| {
                    if (used_c[sb]) continue;
                    var best_h: f32 = -std.math.inf(f32);
                    for (0..num_q_heads) |h| {
                        const q_row = q_nope[(node * num_q_heads + h) * kv_lora_rank ..][0..index_dim];
                        const ck_row = coarse_keys[sb * index_dim ..][0..index_dim];
                        var dot_acc: f32 = 0.0;
                        for (0..index_dim) |d| dot_acc += @as(f32, @floatCast(q_row[d])) * @as(f32, @floatCast(ck_row[d]));
                        best_h = @max(best_h, dot_acc * scale);
                    }
                    const sb_u32: u32 = @intCast(sb);
                    if (best_h > best_s or (best_h == best_s and sb_u32 < best_b)) {
                        best_s = best_h;
                        best_b = sb_u32;
                    }
                }
                if (best_b == 0xFFFF_FFFF) break;
                used_c[best_b] = true;
                coarse_sel[coarse_cnt] = best_b;
                coarse_cnt += 1;
            }
        }

        var win_cnt: u32 = 0;
        if (k_sel > 0) {
            var cand_ids: [4096]u32 = undefined;
            var cand_scores: [4096]f32 = undefined;
            var n_cands: usize = 0;
            if (use_coarse) {
                for (0..coarse_cnt) |ci| {
                    for (0..g_blocks) |sub| {
                        const b = coarse_sel[ci] * g_blocks + @as(u32, @intCast(sub));
                        if (b >= search_end) continue;
                        cand_ids[n_cands] = b;
                        n_cands += 1;
                    }
                }
            } else {
                for (0..search_end) |b| {
                    cand_ids[n_cands] = @intCast(b);
                    n_cands += 1;
                }
            }
            for (0..n_cands) |i| {
                const b = cand_ids[i];
                var best_h: f32 = -std.math.inf(f32);
                for (0..num_q_heads) |h| {
                    const q_row = q_nope[(node * num_q_heads + h) * kv_lora_rank ..][0..index_dim];
                    const fk_row = fine_keys[@as(usize, b) * index_dim ..][0..index_dim];
                    var dot_acc: f32 = 0.0;
                    for (0..index_dim) |d| dot_acc += @as(f32, @floatCast(q_row[d])) * @as(f32, @floatCast(fk_row[d]));
                    best_h = @max(best_h, dot_acc * scale);
                }
                cand_scores[i] = best_h;
            }
            var used_f: [4096]bool = @splat(false);
            for (0..k_sel) |_| {
                var best_s: f32 = -std.math.inf(f32);
                var best_b: u32 = 0xFFFF_FFFF;
                var best_i: usize = 0;
                for (0..n_cands) |i| {
                    if (used_f[i]) continue;
                    const s = cand_scores[i];
                    const b = cand_ids[i];
                    if (s > best_s or (s == best_s and b < best_b)) {
                        best_s = s;
                        best_b = b;
                        best_i = i;
                    }
                }
                if (best_b == 0xFFFF_FFFF) break;
                used_f[best_i] = true;
                selected_blocks_out[sel_base + win_cnt] = best_b;
                win_cnt += 1;
            }
        }
        for (0..local_cnt) |i| {
            selected_blocks_out[sel_base + win_cnt + i] = search_end + @as(u32, @intCast(i));
        }
        selected_counts_out[node] = win_cnt + local_cnt;
    }
}

/// CPU `fp32` reference for absorbed MLA + CSA2 + SWA bounded window replay + optional tree attention.
pub fn mlaCsa2AttendCpuRef(
    q_nope: []const f16,
    q_pe: ?[]const f16,
    e2m1_plane: []const u8,
    scale_plane: []const u8,
    rope_plane: []const u8,
    rope_format: MlaRopeTailFormat,
    swa_lat_ring: ?[]const f16,
    swa_rope_ring: ?[]const f16,
    swa_ring_capacity: u32,
    selected_blocks: []const u32,
    selected_counts: []const u32,
    kv_tree_latent: ?[]const f16,
    kv_tree_rope: ?[]const f16,
    ancestor_masks: ?[]const u64,
    out_f32: []f32,
    num_nodes: u32,
    num_q_heads: u32,
    kv_lora_rank: u32,
    qk_rope_dim: u32,
    kv_len: u32,
    encoder_prefix_len: u32,
    causal_q_start_pos: ?u32,
    swa_window: u32,
    block_tokens: u32,
    top_k_blocks: u32,
    scale: f32,
) void {
    const e2m1_row_bytes: usize = kv_lora_rank / 2;
    const num_groups: usize = kv_lora_rank / MLA_FP4_GROUP_SIZE;
    const rope_row_bytes: usize = switch (rope_format) {
        .f16 => @as(usize, qk_rope_dim) * 2,
        .e4m3 => qk_rope_dim,
    };
    var kv_lat_tmp: [MAX_MLA_LATENT_DIM]f32 = undefined;
    var kv_rope_tmp: [MAX_MLA_ROPE_DIM]f32 = undefined;

    for (0..num_nodes) |node| {
        const eff_kv_len: u32 = if (causal_q_start_pos) |q0|
            @min(kv_len, @max(encoder_prefix_len, q0 + @as(u32, @intCast(node)) + 1))
        else
            kv_len;
        const tree_anc: u32 = if (ancestor_masks) |ms| @popCount(ms[node]) else 0;
        const depth_above_root: u32 = if (tree_anc > 1) tree_anc - 1 else 0;
        const eff_ring_pos: u32 = kv_len + depth_above_root;
        const ring_valid_lo: u32 = if (swa_ring_capacity > 0 and eff_ring_pos > swa_ring_capacity)
            eff_ring_pos - swa_ring_capacity
        else
            0;
        const swa_hi: u32 = eff_kv_len;
        const swa_lo: u32 = if (swa_window > 0)
            (if (eff_kv_len > swa_window) eff_kv_len - swa_window else 0)
        else
            eff_kv_len;
        const sel_cnt = @min(selected_counts[node], top_k_blocks);
        const sel_base = node * top_k_blocks;

        for (0..num_q_heads) |h| {
            const q_nope_row = q_nope[(node * num_q_heads + h) * kv_lora_rank ..][0..kv_lora_rank];
            const q_pe_row: ?[]const f16 = if (qk_rope_dim > 0)
                q_pe.?[(node * num_q_heads + h) * qk_rope_dim ..][0..qk_rope_dim]
            else
                null;
            const out_row = out_f32[(node * num_q_heads + h) * kv_lora_rank ..][0..kv_lora_rank];
            @memset(out_row, 0.0);

            var m_val: f32 = -std.math.inf(f32);
            var l_val: f32 = 0.0;

            // Pass 1: selected sparse FP4 blocks (deduplicated against SWA window)
            for (0..sel_cnt) |sb| {
                const block_id = selected_blocks[sel_base + sb];
                if (block_id == 0xFFFF_FFFF) continue;
                const t_start = block_id * block_tokens;
                if (t_start >= eff_kv_len) continue;
                const t_end = @min(t_start + block_tokens, eff_kv_len);
                for (t_start..t_end) |t| {
                    if (t >= swa_lo and t < swa_hi) continue;
                    decodeMlaFp4LatentRowCpu(
                        e2m1_plane[t * e2m1_row_bytes ..][0..e2m1_row_bytes],
                        scale_plane[t * num_groups ..][0..num_groups],
                        kv_lat_tmp[0..kv_lora_rank],
                    );
                    var dot_acc: f32 = 0.0;
                    for (0..kv_lora_rank) |d| dot_acc += @as(f32, @floatCast(q_nope_row[d])) * kv_lat_tmp[d];
                    if (qk_rope_dim > 0) {
                        decodeMlaRopeRowCpu(rope_plane[t * rope_row_bytes ..][0..rope_row_bytes], rope_format, kv_rope_tmp[0..qk_rope_dim]);
                        for (0..qk_rope_dim) |d| dot_acc += @as(f32, @floatCast(q_pe_row.?[d])) * kv_rope_tmp[d];
                    }
                    const s = dot_acc * scale;
                    const new_m = @max(m_val, s);
                    const corr: f32 = if (l_val > 0.0) @exp(m_val - new_m) else 0.0;
                    const p_w = @exp(s - new_m);
                    l_val = l_val * corr + p_w;
                    for (0..kv_lora_rank) |d| out_row[d] = out_row[d] * corr + kv_lat_tmp[d] * p_w;
                    m_val = new_m;
                }
            }

            // Pass 2: SWA bounded replay `[swa_lo, swa_hi)`
            for (swa_lo..swa_hi) |t| {
                const use_ring = (swa_ring_capacity > 0 and t >= ring_valid_lo and swa_lat_ring != null);
                var dot_acc: f32 = 0.0;
                if (use_ring) {
                    const slot = t % swa_ring_capacity;
                    const r_lat = swa_lat_ring.?[slot * kv_lora_rank ..][0..kv_lora_rank];
                    for (0..kv_lora_rank) |d| {
                        kv_lat_tmp[d] = @floatCast(r_lat[d]);
                        dot_acc += @as(f32, @floatCast(q_nope_row[d])) * kv_lat_tmp[d];
                    }
                    if (qk_rope_dim > 0) {
                        const r_rope = swa_rope_ring.?[slot * qk_rope_dim ..][0..qk_rope_dim];
                        for (0..qk_rope_dim) |d| {
                            dot_acc += @as(f32, @floatCast(q_pe_row.?[d])) * @as(f32, @floatCast(r_rope[d]));
                        }
                    }
                } else {
                    decodeMlaFp4LatentRowCpu(
                        e2m1_plane[t * e2m1_row_bytes ..][0..e2m1_row_bytes],
                        scale_plane[t * num_groups ..][0..num_groups],
                        kv_lat_tmp[0..kv_lora_rank],
                    );
                    for (0..kv_lora_rank) |d| dot_acc += @as(f32, @floatCast(q_nope_row[d])) * kv_lat_tmp[d];
                    if (qk_rope_dim > 0) {
                        decodeMlaRopeRowCpu(rope_plane[t * rope_row_bytes ..][0..rope_row_bytes], rope_format, kv_rope_tmp[0..qk_rope_dim]);
                        for (0..qk_rope_dim) |d| dot_acc += @as(f32, @floatCast(q_pe_row.?[d])) * kv_rope_tmp[d];
                    }
                }
                const s = dot_acc * scale;
                const new_m = @max(m_val, s);
                const corr: f32 = if (l_val > 0.0) @exp(m_val - new_m) else 0.0;
                const p_w = @exp(s - new_m);
                l_val = l_val * corr + p_w;
                for (0..kv_lora_rank) |d| out_row[d] = out_row[d] * corr + kv_lat_tmp[d] * p_w;
                m_val = new_m;
            }

            // Pass 3: Optional tree ancestors
            if (kv_tree_latent != null and ancestor_masks != null) {
                const mask = ancestor_masks.?[node];
                for (0..num_nodes) |j| {
                    if ((mask & (@as(u64, 1) << @intCast(j))) == 0) continue;
                    const t_lat = kv_tree_latent.?[j * kv_lora_rank ..][0..kv_lora_rank];
                    var dot_acc: f32 = 0.0;
                    for (0..kv_lora_rank) |d| {
                        kv_lat_tmp[d] = @floatCast(t_lat[d]);
                        dot_acc += @as(f32, @floatCast(q_nope_row[d])) * kv_lat_tmp[d];
                    }
                    if (qk_rope_dim > 0) {
                        const t_rope = kv_tree_rope.?[j * qk_rope_dim ..][0..qk_rope_dim];
                        for (0..qk_rope_dim) |d| {
                            dot_acc += @as(f32, @floatCast(q_pe_row.?[d])) * @as(f32, @floatCast(t_rope[d]));
                        }
                    }
                    const s = dot_acc * scale;
                    const new_m = @max(m_val, s);
                    const corr: f32 = if (l_val > 0.0) @exp(m_val - new_m) else 0.0;
                    const p_w = @exp(s - new_m);
                    l_val = l_val * corr + p_w;
                    for (0..kv_lora_rank) |d| out_row[d] = out_row[d] * corr + kv_lat_tmp[d] * p_w;
                    m_val = new_m;
                }
            }

            if (l_val > 0.0) {
                for (0..kv_lora_rank) |d| out_row[d] /= l_val;
            }
        }
    }
}

/// CPU `fp32` reference for Sinkhorn-Knopp Birkhoff doubly-stochastic projection of `H_res`.
pub fn mhcSinkhornCpuRef(raw: []const f32, num_streams: u32, iters: u32, out: []f32) void {
    const n: usize = num_streams;
    var max_v: f32 = -std.math.inf(f32);
    for (0..n * n) |idx| {
        out[idx] = raw[idx];
        max_v = @max(max_v, raw[idx]);
    }
    if (iters == 0) return;
    for (0..n * n) |idx| out[idx] = @exp(out[idx] - max_v);
    for (0..iters) |_| {
        for (0..n) |o| {
            var rsum: f32 = 0.0;
            for (0..n) |i| rsum += out[o * n + i];
            const inv_r = 1.0 / @max(rsum, 1e-12);
            for (0..n) |i| out[o * n + i] *= inv_r;
        }
        for (0..n) |i| {
            var csum: f32 = 0.0;
            for (0..n) |o| csum += out[o * n + i];
            const inv_c = 1.0 / @max(csum, 1e-12);
            for (0..n) |o| out[o * n + i] *= inv_c;
        }
    }
}

/// CPU `fp32` reference for `ami_mhc_single_pass` (pre-sublayer stream mix + optional RMSNorm + optional in-place `H_res`).
pub fn mhcSinglePassCpuRef(
    cfg: MhcConfig,
    streams_f32: []f32,
    h_pre: []const f32,
    h_res: ?[]const f32,
    norm_w_bits: ?[]const u16,
    branch_out_f32: []f32,
    m: u32,
) void {
    const n: usize = cfg.num_streams;
    const d_dim: usize = cfg.hidden_size;
    var hp: [MAX_MHC_STREAMS]f32 = undefined;
    var hr: [MAX_MHC_STREAMS * MAX_MHC_STREAMS]f32 = undefined;

    for (0..m) |tok| {
        const coeff_tok = if (cfg.shared_coeffs) 0 else tok;
        for (0..n) |i| {
            const raw = h_pre[coeff_tok * n + i];
            hp[i] = if (cfg.activate_coeffs) sigmoidF32(raw) else raw;
        }
        if (cfg.apply_h_res_pre) {
            mhcSinkhornCpuRef(h_res.?[coeff_tok * n * n ..][0 .. n * n], cfg.num_streams, cfg.sinkhorn_iters, hr[0 .. n * n]);
        }

        const tok_s = tok * n * d_dim;
        const tok_o = tok * d_dim;
        var sq_sum: f32 = 0.0;
        for (0..d_dim) |d| {
            var mix: f32 = 0.0;
            for (0..n) |i| mix += hp[i] * streams_f32[tok_s + i * d_dim + d];
            branch_out_f32[tok_o + d] = mix;
            sq_sum += mix * mix;
        }
        if (norm_w_bits) |nw| {
            const inv_rms = 1.0 / @sqrt(sq_sum / @as(f32, @floatFromInt(d_dim)) + cfg.rms_eps);
            for (0..d_dim) |d| {
                branch_out_f32[tok_o + d] = (branch_out_f32[tok_o + d] * inv_rms) * quant.floatBitsToF32(cfg.norm_dtype, nw[d]);
            }
        }
        if (cfg.apply_h_res_pre) {
            var s_old: [MAX_MHC_STREAMS]f32 = undefined;
            for (0..d_dim) |d| {
                for (0..n) |i| s_old[i] = streams_f32[tok_s + i * d_dim + d];
                for (0..n) |o| {
                    var acc: f32 = 0.0;
                    for (0..n) |i| acc += hr[o * n + i] * s_old[i];
                    streams_f32[tok_s + o * d_dim + d] = acc;
                }
            }
        }
    }
}

/// CPU `fp32` reference for `ami_mhc_post_mix` (fused in-place `H_res` stream mix + `H_post * sublayer_out`).
pub fn mhcPostMixCpuRef(
    cfg: MhcConfig,
    streams_f32: []f32,
    sublayer_out_f32: []const f32,
    h_post: []const f32,
    h_res: ?[]const f32,
    m: u32,
) void {
    const n: usize = cfg.num_streams;
    const d_dim: usize = cfg.hidden_size;
    var hp: [MAX_MHC_STREAMS]f32 = undefined;
    var hr: [MAX_MHC_STREAMS * MAX_MHC_STREAMS]f32 = undefined;

    for (0..m) |tok| {
        const coeff_tok = if (cfg.shared_coeffs) 0 else tok;
        for (0..n) |i| {
            const raw = h_post[coeff_tok * n + i];
            hp[i] = if (cfg.activate_coeffs) (cfg.post_scale * sigmoidF32(raw)) else (cfg.post_scale * raw);
        }
        if (cfg.apply_h_res_post) {
            mhcSinkhornCpuRef(h_res.?[coeff_tok * n * n ..][0 .. n * n], cfg.num_streams, cfg.sinkhorn_iters, hr[0 .. n * n]);
        }
        const tok_s = tok * n * d_dim;
        const tok_y = tok * d_dim;
        var s_old: [MAX_MHC_STREAMS]f32 = undefined;
        for (0..d_dim) |d| {
            const y_val = sublayer_out_f32[tok_y + d];
            for (0..n) |i| s_old[i] = streams_f32[tok_s + i * d_dim + d];
            for (0..n) |o| {
                var base_val: f32 = s_old[o];
                if (cfg.apply_h_res_post) {
                    base_val = 0.0;
                    for (0..n) |i| base_val += hr[o * n + i] * s_old[i];
                }
                streams_f32[tok_s + o * d_dim + d] = base_val + hp[o] * y_val;
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

fn makeM65HybridBudget(bytes: usize) budget_mod.Budget {
    return budget_mod.Budget.compute(.{
        .memory_bytes = bytes * 8,
        .user_wire_limit = bytes * 4,
        .global_user_wire_limit = bytes * 4,
        .gpu_working_set_bytes = bytes * 4,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
}

test "R5 QSA tiered q8 retired + active sparse attention parity vs pure fp16 active tiles (d=128 and d=64 with tree masks)" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const hybrid_pipes = try HybridPipelines.init(device, library);
    defer hybrid_pipes.close();
    const attn_pipes = try attention.AttentionPipelines.init(device, library);
    defer attn_pipes.close();

    const head_dims = [_]u32{ 128, 64 };
    for (head_dims) |head_dim| {
        const num_q_heads: u32 = 8;
        const num_kv_heads: u32 = 2;
        const chunk_tokens: u32 = QSA_FULL_CHUNK_TOKENS; // 512
        const total_chunks: u32 = 3;
        const num_retired: u32 = 2;
        const kv_len: u32 = 2 * chunk_tokens + 200; // 1,224 tokens (2 retired + 1 active partial)
        const block_tokens: u32 = QSA_DEFAULT_BLOCK_TOKENS; // 32
        const num_blocks: u32 = (kv_len + block_tokens - 1) / block_tokens;
        const top_k_blocks: u32 = 12;
        const num_nodes: u32 = 4;

        const pool_spec: attention.RetiredKvQ8PoolSpec = .{
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .chunk_tokens = chunk_tokens,
            .max_chunks = num_retired,
        };
        const idx_spec: QsaIndexBufferSpec = .{
            .max_nodes = num_nodes,
            .num_kv_heads = num_kv_heads,
            .top_k_blocks = top_k_blocks,
        };
        const block_keys_bytes = amf.align64(@as(usize, num_blocks) * num_kv_heads * head_dim * @sizeOf(f16));
        const arena_bytes: usize =
            (try attention.RetiredKvQ8Pool.requiredArenaBytes(pool_spec)) +
            (try HybridKvRing.tableRequiredBytes(total_chunks)) +
            (try QsaIndexBuffer.requiredBytes(idx_spec)) +
            block_keys_bytes +
            page.APPLE_PAGE_SIZE * 16;
        var arena = try arena_mod.WiredArena.init(arena_bytes, makeM65HybridBudget(arena_bytes));
        defer arena.deinit();

        var full_ring = try HybridKvRing.init(gpa, device, &arena, .{
            .mode = .full_chunk_512,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .num_tiles = total_chunks,
        });
        defer full_ring.deinit(gpa);

        var retired_pool = try attention.RetiredKvQ8Pool.init(&arena, pool_spec);
        const retired_ids_raw = try arena.allocAligned(amf.align64(@as(usize, num_retired) * @sizeOf(u32)), 64);
        const active_tail_tbl_raw = try arena.allocAligned(amf.align64(@sizeOf(attention.KvTilePair)), 64);
        const idx_buf = try QsaIndexBuffer.init(&arena, idx_spec);
        const block_keys_raw = try arena.allocAligned(block_keys_bytes, 64);
        const q_raw = try arena.allocAligned(amf.align64(@as(usize, num_nodes) * num_q_heads * head_dim * @sizeOf(f16)), 64);
        const kt_raw = try arena.allocAligned(amf.align64(@as(usize, num_nodes) * num_kv_heads * head_dim * @sizeOf(f16)), 64);
        const vt_raw = try arena.allocAligned(amf.align64(@as(usize, num_nodes) * num_kv_heads * head_dim * @sizeOf(f16)), 64);
        const masks_raw = try arena.allocAligned(amf.align64(@as(usize, num_nodes) * @sizeOf(u64)), 64);
        const out_fp16_raw = try arena.allocAligned(amf.align64(@as(usize, num_nodes) * num_q_heads * head_dim * @sizeOf(f16)), 64);
        const out_tiered_raw = try arena.allocAligned(amf.align64(@as(usize, num_nodes) * num_q_heads * head_dim * @sizeOf(f16)), 64);

        var rng = std.Random.DefaultPrng.init(0xA5A_0000 + head_dim);
        const rand = rng.random();
        for (0..total_chunks) |c| {
            var kg = try full_ring.k_tiles[c].lock(false);
            errdefer kg.release();
            var vg = try full_ring.v_tiles[c].lock(false);
            errdefer vg.release();
            const km: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
            const vm: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vg.bytes));
            for (km, vm) |*kx, *vx| {
                kx.* = @floatCast((rand.float(f32) - 0.5) * 0.9);
                vx.* = @floatCast((rand.float(f32) - 0.5) * 0.8);
            }
            try vg.unlock();
            try kg.unlock();
        }
        const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0 .. @as(usize, num_nodes) * num_q_heads * head_dim * @sizeOf(f16)]));
        const kt_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kt_raw[0 .. @as(usize, num_nodes) * num_kv_heads * head_dim * @sizeOf(f16)]));
        const vt_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vt_raw[0 .. @as(usize, num_nodes) * num_kv_heads * head_dim * @sizeOf(f16)]));
        const masks: []u64 = @alignCast(std.mem.bytesAsSlice(u64, masks_raw[0 .. @as(usize, num_nodes) * @sizeOf(u64)]));
        for (q_f16) |*x| x.* = @floatCast((rand.float(f32) - 0.5) * 0.9);
        for (kt_f16, vt_f16) |*kx, *vx| {
            kx.* = @floatCast((rand.float(f32) - 0.5) * 0.9);
            vx.* = @floatCast((rand.float(f32) - 0.5) * 0.8);
        }
        masks[0] = 0b0001;
        masks[1] = 0b0011;
        masks[2] = 0b0101;
        masks[3] = 0b1011;

        // Place chunk 2 (active) in `active_tail_tbl`, and retire chunks 0 & 1 into `retired_pool` (swapped slot order 1, 0).
        const active_tail_tbl: []attention.KvTilePair = @alignCast(std.mem.bytesAsSlice(
            attention.KvTilePair,
            active_tail_tbl_raw[0..@sizeOf(attention.KvTilePair)],
        ));
        active_tail_tbl[0] = attention.KvTilePair.fromTiles(full_ring.k_tiles[2], full_ring.v_tiles[2], .rgba16f);

        const s0 = try retired_pool.acquireSlot();
        const s1 = try retired_pool.acquireSlot();
        const retired_ids: []u32 = @alignCast(std.mem.bytesAsSlice(u32, retired_ids_raw[0 .. @as(usize, num_retired) * @sizeOf(u32)]));
        retired_ids[0] = s1;
        retired_ids[1] = s0;

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});
        try full_ring.makeResident(compute);

        {
            var f = try compute.beginFrame(5000);
            errdefer f.abort();
            // Compute block summaries while all 3 tiles are active.
            try hybrid_pipes.encodeQsaUpdateBlockSummary(&f, .{
                .kv_tiles = full_ring.tileTableBinding(gpu_buf),
                .block_keys = .{ .buffer = gpu_buf, .offset = arena.offsetOf(block_keys_raw.ptr) },
                .num_kv_heads = num_kv_heads,
                .head_dim = head_dim,
                .kv_len = kv_len,
                .chunk_tokens = chunk_tokens,
                .block_tokens = block_tokens,
                .start_block = 0,
                .num_blocks = num_blocks,
            }, false);
            // Retire chunks 0 and 1 into `retired_pool` on GPU.
            try attn_pipes.encodeKvRetireTileToQ8(&f, .{
                .k_tile = full_ring.k_tiles[0],
                .v_tile = full_ring.v_tiles[0],
                .pool = &retired_pool,
                .pool_buf = gpu_buf,
                .dst_slot = retired_ids[0],
            }, true);
            try attn_pipes.encodeKvRetireTileToQ8(&f, .{
                .k_tile = full_ring.k_tiles[1],
                .v_tile = full_ring.v_tiles[1],
                .pool = &retired_pool,
                .pool_buf = gpu_buf,
                .dst_slot = retired_ids[1],
            }, false);
            // Run pure fp16 QSA (r=0: indexes blocks + attends all 3 active tiles).
            try hybrid_pipes.encodeQsaAttention(&f, .{
                .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
                .kv_tiles = full_ring.tileTableBinding(gpu_buf),
                .block_keys = .{ .buffer = gpu_buf, .offset = arena.offsetOf(block_keys_raw.ptr) },
                .index_buf = &idx_buf,
                .index_metal_buf = gpu_buf,
                .k_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kt_raw.ptr) },
                .v_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(vt_raw.ptr) },
                .ancestor_masks = .{ .buffer = gpu_buf, .offset = arena.offsetOf(masks_raw.ptr) },
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_fp16_raw.ptr) },
                .num_nodes = num_nodes,
                .num_q_heads = num_q_heads,
                .num_kv_heads = num_kv_heads,
                .head_dim = head_dim,
                .kv_len = kv_len,
                .chunk_tokens = chunk_tokens,
                .block_tokens = block_tokens,
                .top_k_blocks = top_k_blocks,
                .reuse_block_indices = false,
            }, true);
            // Run tiered QSA (reuses exact same block indices, reads chunks 0..1 from q8 retired_pool + chunk 2 from active_tail_tbl!).
            try hybrid_pipes.encodeQsaAttention(&f, .{
                .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
                .kv_tiles = .{ .buffer = gpu_buf, .offset = arena.offsetOf(active_tail_tbl_raw.ptr) },
                .block_keys = .{ .buffer = gpu_buf, .offset = arena.offsetOf(block_keys_raw.ptr) },
                .index_buf = &idx_buf,
                .index_metal_buf = gpu_buf,
                .k_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kt_raw.ptr) },
                .v_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(vt_raw.ptr) },
                .ancestor_masks = .{ .buffer = gpu_buf, .offset = arena.offsetOf(masks_raw.ptr) },
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_tiered_raw.ptr) },
                .retired_pool = retired_pool.binding(gpu_buf),
                .retired_chunk_ids = .{ .buffer = gpu_buf, .offset = arena.offsetOf(retired_ids_raw.ptr) },
                .num_nodes = num_nodes,
                .num_q_heads = num_q_heads,
                .num_kv_heads = num_kv_heads,
                .head_dim = head_dim,
                .kv_len = kv_len,
                .chunk_tokens = chunk_tokens,
                .block_tokens = block_tokens,
                .top_k_blocks = top_k_blocks,
                .num_retired_chunks = num_retired,
                .retired_plane_stride_bytes = retired_pool.plane_stride_bytes,
                .retired_chunk_stride_bytes = retired_pool.chunk_stride_bytes,
                .reuse_block_indices = true,
            }, true);
            const sig = try f.commit();
            try compute.wait(sig, 5000);
        }

        const out_fp16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_fp16_raw[0 .. q_f16.len * @sizeOf(f16)]));
        const out_tiered: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_tiered_raw[0 .. q_f16.len * @sizeOf(f16)]));
        var max_err: f32 = 0.0;
        for (out_fp16, out_tiered) |e_h, g_h| {
            const err = @abs(@as(f32, @floatCast(e_h)) - @as(f32, @floatCast(g_h)));
            if (err > max_err) max_err = err;
        }
        try std.testing.expect(max_err < 3e-3);
    }
}

test "C14 multi-slot M_batch=6 ami_gdn_batch_decode bit-exact parity vs single-slot ami_gdn_decode (d=64 & d=128, f16 & bf16)" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const batch_size: u32 = 6;
    const pool_slots: u32 = 8; // larger than batch_size to test non-contiguous permuted slot IDs!
    const cases = [_]struct { dk: u32, dv: u32, norm_dt: quant.FloatDType }{
        .{ .dk = 64, .dv = 64, .norm_dt = .f16 },
        .{ .dk = 128, .dv = 128, .norm_dt = .bf16 },
    };

    for (cases) |tc| {
        const num_k_heads: u32 = 2;
        const num_v_heads: u32 = 4;
        const state_elems: usize = @as(usize, num_v_heads) * tc.dk * tc.dv;
        const slot_stride_bytes: usize = amf.align64(state_elems * @sizeOf(f32));
        const slot_stride_f32: u32 = @intCast(slot_stride_bytes / @sizeOf(f32));

        const arena_bytes: usize = slot_stride_bytes * pool_slots * 4 + page.APPLE_PAGE_SIZE * 16;
        var arena = try arena_mod.WiredArena.init(arena_bytes, makeM65HybridBudget(arena_bytes));
        defer arena.deinit();

        const state_in_raw = try arena.allocAligned(slot_stride_bytes * pool_slots, 64);
        const state_out_batch_raw = try arena.allocAligned(slot_stride_bytes * pool_slots, 64);
        const state_out_single_raw = try arena.allocAligned(slot_stride_bytes * pool_slots, 64);
        @memset(state_out_batch_raw, 0);
        @memset(state_out_single_raw, 0);

        const q_raw = try arena.allocAligned(amf.align64(@as(usize, batch_size) * num_k_heads * tc.dk * @sizeOf(f16)), 64);
        const k_raw = try arena.allocAligned(amf.align64(@as(usize, batch_size) * num_k_heads * tc.dk * @sizeOf(f16)), 64);
        const v_raw = try arena.allocAligned(amf.align64(@as(usize, batch_size) * num_v_heads * tc.dv * @sizeOf(f16)), 64);
        const g_raw = try arena.allocAligned(amf.align64(@as(usize, batch_size) * num_v_heads * @sizeOf(f32)), 64);
        const beta_raw = try arena.allocAligned(amf.align64(@as(usize, batch_size) * num_v_heads * @sizeOf(f32)), 64);
        const z_raw = try arena.allocAligned(amf.align64(@as(usize, batch_size) * num_v_heads * tc.dv * @sizeOf(f16)), 64);
        const norm_w_raw = try arena.allocAligned(amf.align64(@as(usize, num_v_heads) * tc.dv * 2), 64);
        const out_batch_raw = try arena.allocAligned(amf.align64(@as(usize, batch_size) * num_v_heads * tc.dv * @sizeOf(f16)), 64);
        const out_single_raw = try arena.allocAligned(amf.align64(@as(usize, batch_size) * num_v_heads * tc.dv * @sizeOf(f16)), 64);
        const in_slots_raw = try arena.allocAligned(amf.align64(@as(usize, batch_size) * @sizeOf(u32)), 64);
        const out_slots_raw = try arena.allocAligned(amf.align64(@as(usize, batch_size) * @sizeOf(u32)), 64);

        var rng = std.Random.DefaultPrng.init(0x6D0_0000 + tc.dk);
        const rand = rng.random();
        const s_in_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, state_in_raw));
        for (s_in_f32) |*x| x.* = (rand.float(f32) - 0.5) * 0.4;

        const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0 .. @as(usize, batch_size) * num_k_heads * tc.dk * @sizeOf(f16)]));
        const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw[0 .. @as(usize, batch_size) * num_k_heads * tc.dk * @sizeOf(f16)]));
        const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw[0 .. @as(usize, batch_size) * num_v_heads * tc.dv * @sizeOf(f16)]));
        const z_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, z_raw[0 .. @as(usize, batch_size) * num_v_heads * tc.dv * @sizeOf(f16)]));
        const g_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, g_raw[0 .. @as(usize, batch_size) * num_v_heads * @sizeOf(f32)]));
        const beta_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, beta_raw[0 .. @as(usize, batch_size) * num_v_heads * @sizeOf(f32)]));
        for (q_f16, k_f16) |*qp, *kp| {
            qp.* = @floatCast((rand.float(f32) - 0.5) * 1.2);
            kp.* = @floatCast((rand.float(f32) - 0.5) * 1.2);
        }
        for (v_f16, z_f16) |*vp, *zp| {
            vp.* = @floatCast((rand.float(f32) - 0.5) * 1.0);
            zp.* = @floatCast((rand.float(f32) - 0.5) * 1.5);
        }
        for (g_f32, beta_f32) |*gp, *bp| {
            gp.* = -0.05 - rand.float(f32) * 0.4;
            bp.* = (rand.float(f32) - 0.5) * 2.0;
        }
        const nw_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, norm_w_raw[0 .. @as(usize, num_v_heads) * tc.dv * 2]));
        for (nw_u16) |*w| {
            const val: f32 = 0.7 + rand.float(f32) * 0.6;
            w.* = switch (tc.norm_dt) {
                .f16 => @bitCast(@as(f16, @floatCast(val))),
                .bf16 => @truncate(@as(u32, @bitCast(val)) >> 16),
            };
        }

        // Permuted non-contiguous slot mappings (`in_slots != out_slots` -> also verifies scratch/slot isolation!).
        const in_slots_perm = [batch_size]u32{ 7, 2, 5, 0, 4, 1 };
        const out_slots_perm = [batch_size]u32{ 1, 6, 3, 7, 2, 4 };
        const in_slots: []u32 = @alignCast(std.mem.bytesAsSlice(u32, in_slots_raw[0 .. @as(usize, batch_size) * @sizeOf(u32)]));
        const out_slots: []u32 = @alignCast(std.mem.bytesAsSlice(u32, out_slots_raw[0 .. @as(usize, batch_size) * @sizeOf(u32)]));
        @memcpy(in_slots, &in_slots_perm);
        @memcpy(out_slots, &out_slots_perm);

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        {
            var f = try compute.beginFrame(5000);
            errdefer f.abort();
            try pipes.encodeGdnBatchDecode(&f, .{
                .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
                .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) },
                .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) },
                .g = .{ .buffer = gpu_buf, .offset = arena.offsetOf(g_raw.ptr) },
                .beta = .{ .buffer = gpu_buf, .offset = arena.offsetOf(beta_raw.ptr) },
                .z = .{ .buffer = gpu_buf, .offset = arena.offsetOf(z_raw.ptr) },
                .norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(norm_w_raw.ptr) },
                .state_in_pool = .{ .buffer = gpu_buf, .offset = arena.offsetOf(state_in_raw.ptr) },
                .state_out_pool = .{ .buffer = gpu_buf, .offset = arena.offsetOf(state_out_batch_raw.ptr) },
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_batch_raw.ptr) },
                .state_in_slots = .{ .buffer = gpu_buf, .offset = arena.offsetOf(in_slots_raw.ptr) },
                .state_out_slots = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_slots_raw.ptr) },
                .batch_size = batch_size,
                .num_k_heads = num_k_heads,
                .num_v_heads = num_v_heads,
                .head_k_dim = tc.dk,
                .head_v_dim = tc.dv,
                .slot_stride_f32 = slot_stride_f32,
                .per_head_norm = true,
                .norm_dtype = tc.norm_dt,
            }, false);

            const qk_bytes: u64 = @as(u64, num_k_heads) * tc.dk * @sizeOf(f16);
            const v_bytes: u64 = @as(u64, num_v_heads) * tc.dv * @sizeOf(f16);
            const gb_bytes: u64 = @as(u64, num_v_heads) * @sizeOf(f32);
            for (0..batch_size) |b| {
                try pipes.encodeGdnDecode(&f, .{
                    .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) + @as(u64, b) * qk_bytes },
                    .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) + @as(u64, b) * qk_bytes },
                    .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) + @as(u64, b) * v_bytes },
                    .g = .{ .buffer = gpu_buf, .offset = arena.offsetOf(g_raw.ptr) + @as(u64, b) * gb_bytes },
                    .beta = .{ .buffer = gpu_buf, .offset = arena.offsetOf(beta_raw.ptr) + @as(u64, b) * gb_bytes },
                    .z = .{ .buffer = gpu_buf, .offset = arena.offsetOf(z_raw.ptr) + @as(u64, b) * v_bytes },
                    .norm_w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(norm_w_raw.ptr) },
                    .state_in = .{ .buffer = gpu_buf, .offset = arena.offsetOf(state_in_raw.ptr) + @as(u64, in_slots_perm[b]) * slot_stride_bytes },
                    .state_out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(state_out_single_raw.ptr) + @as(u64, out_slots_perm[b]) * slot_stride_bytes },
                    .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_single_raw.ptr) + @as(u64, b) * v_bytes },
                    .num_k_heads = num_k_heads,
                    .num_v_heads = num_v_heads,
                    .head_k_dim = tc.dk,
                    .head_v_dim = tc.dv,
                    .per_head_norm = true,
                    .norm_dtype = tc.norm_dt,
                }, true);
            }
            const sig = try f.commit();
            try compute.wait(sig, 5000);
        }

        // Exact bit-for-bit equality for BOTH outputs AND updated recurrent states (including untouched slots 0 and 5)!
        try std.testing.expectEqualSlices(u8, out_single_raw, out_batch_raw);
        try std.testing.expectEqualSlices(u8, state_out_single_raw, state_out_batch_raw);
    }
}

test "R8 GdnSnapshotPool and ami_gdn_clone_state round-trip multi-layer state + conv1d with mutation isolation" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const state_spec: GdnStateSpec = .{
        .num_gdn_layers = 3,
        .num_v_heads = 4,
        .head_k_dim = 64,
        .head_v_dim = 64,
    };
    const conv_f16_per_layer: u32 = 3 * 512; // conv_k=4, conv_dim=512
    const conv_stride_bytes: usize = amf.align64(@as(usize, conv_f16_per_layer) * @sizeOf(f16));
    const conv_stride_f16: u32 = @intCast(conv_stride_bytes / @sizeOf(f16));

    const snap_spec: GdnSnapshotPoolSpec = .{
        .state = state_spec,
        .conv_f16_per_layer = conv_f16_per_layer,
        .max_snapshots = 2,
    };
    const arena_bytes: usize =
        (try GdnSnapshotPool.requiredArenaBytes(snap_spec)) +
        (try GdnStateArena.requiredBytes(state_spec)) * 2 +
        conv_stride_bytes * state_spec.num_gdn_layers * 2 +
        page.APPLE_PAGE_SIZE * 4;
    var arena = try arena_mod.WiredArena.init(arena_bytes, makeM65HybridBudget(arena_bytes));
    defer arena.deinit();

    var snap_pool = try GdnSnapshotPool.init(&arena, snap_spec);
    var src_arena = try GdnStateArena.init(&arena, state_spec);
    var dst_arena = try GdnStateArena.init(&arena, state_spec);
    const src_conv_raw = try arena.allocAligned(conv_stride_bytes * state_spec.num_gdn_layers, 64);
    const dst_conv_raw = try arena.allocAligned(conv_stride_bytes * state_spec.num_gdn_layers, 64);
    @memset(dst_conv_raw, 0);

    for (0..state_spec.num_gdn_layers) |l| {
        const s = try src_arena.committedLayerSlice(@intCast(l));
        for (s, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast((i + l * 41) % 37)) - 18)) * 0.125;
        const c_off = l * conv_stride_bytes;
        const c: []f16 = @alignCast(std.mem.bytesAsSlice(f16, src_conv_raw[c_off .. c_off + @as(usize, conv_f16_per_layer) * @sizeOf(f16)]));
        for (c, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i + l * 19) % 29)) - 14)) * 0.25);
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const snap0 = try snap_pool.acquireSlot();
    const snap1 = try snap_pool.acquireSlot();
    try std.testing.expectError(error.PoolExhausted, snap_pool.acquireSlot());

    // Save src -> snap0, then restore snap0 -> dst on GPU in one frame.
    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeGdnCloneState(&f, .{
            .src_state = try src_arena.committedBinding(gpu_buf, 0),
            .dst_state = try snap_pool.stateBinding(gpu_buf, snap0),
            .src_conv = .{ .buffer = gpu_buf, .offset = arena.offsetOf(src_conv_raw.ptr) },
            .dst_conv = try snap_pool.convBinding(gpu_buf, snap0),
            .num_gdn_layers = state_spec.num_gdn_layers,
            .state_f32_per_layer = snap_pool.layer_state_elems,
            .src_state_layer_stride_f32 = @intCast(src_arena.layer_stride_bytes / @sizeOf(f32)),
            .dst_state_layer_stride_f32 = snap_pool.stateLayerStrideF32(),
            .conv_f16_per_layer = conv_f16_per_layer,
            .src_conv_layer_stride_f16 = conv_stride_f16,
            .dst_conv_layer_stride_f16 = snap_pool.convLayerStrideF16(),
        }, false);
        try pipes.encodeGdnCloneState(&f, .{
            .src_state = try snap_pool.stateBinding(gpu_buf, snap0),
            .dst_state = try dst_arena.committedBinding(gpu_buf, 0),
            .src_conv = try snap_pool.convBinding(gpu_buf, snap0),
            .dst_conv = .{ .buffer = gpu_buf, .offset = arena.offsetOf(dst_conv_raw.ptr) },
            .num_gdn_layers = state_spec.num_gdn_layers,
            .state_f32_per_layer = snap_pool.layer_state_elems,
            .src_state_layer_stride_f32 = snap_pool.stateLayerStrideF32(),
            .dst_state_layer_stride_f32 = @intCast(dst_arena.layer_stride_bytes / @sizeOf(f32)),
            .conv_f16_per_layer = conv_f16_per_layer,
            .src_conv_layer_stride_f16 = snap_pool.convLayerStrideF16(),
            .dst_conv_layer_stride_f16 = conv_stride_f16,
        }, true);
        const sig = try f.commit();
        try compute.wait(sig, 5000);
    }

    for (0..state_spec.num_gdn_layers) |l| {
        try std.testing.expectEqualSlices(f32, try src_arena.committedLayerSlice(@intCast(l)), try dst_arena.committedLayerSlice(@intCast(l)));
    }
    try std.testing.expectEqualSlices(u8, src_conv_raw, dst_conv_raw);

    try snap_pool.releaseSlot(snap1);
    try std.testing.expectError(error.DoubleFree, snap_pool.releaseSlot(snap1));
    try snap_pool.releaseSlot(snap0);
    try std.testing.expectEqual(@as(u32, 0), snap_pool.activeCount());
}

test "G15: wide GDN prefill (M=128, M=256) matches CPU ref and supports write_state_only + P0-B1 tail" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    for ([_]u32{ 128, 256 }) |seq_len| {
        const num_k_heads: u32 = 2;
        const num_v_heads: u32 = 4;
        const head_k_dim: u32 = 64;
        const head_v_dim: u32 = 64;

        const qk_bytes = @as(usize, seq_len) * num_k_heads * head_k_dim * @sizeOf(f16);
        const v_bytes = @as(usize, seq_len) * num_v_heads * head_v_dim * @sizeOf(f16);
        const gb_bytes = @as(usize, seq_len) * num_v_heads * @sizeOf(f32);
        const state_spec: GdnStateSpec = .{
            .num_gdn_layers = 1,
            .num_v_heads = num_v_heads,
            .head_k_dim = head_k_dim,
            .head_v_dim = head_v_dim,
        };
        const state_bytes = try GdnStateArena.requiredBytes(state_spec);
        const arena_cap = try page.alignUp(2 * state_bytes + 2 * qk_bytes + 5 * v_bytes + 2 * gb_bytes + 8192) + page.APPLE_PAGE_SIZE;
        const budget = budget_mod.Budget.compute(.{
            .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
            .user_wire_limit = arena_cap,
            .global_user_wire_limit = arena_cap,
            .gpu_working_set_bytes = arena_cap,
            .configured_cap = null,
            .headroom_bytes = page.APPLE_PAGE_SIZE,
        });
        var arena = try arena_mod.WiredArena.init(arena_cap, budget);
        defer arena.deinit();

        var st_full = try GdnStateArena.init(&arena, state_spec);
        var st_only = try GdnStateArena.init(&arena, state_spec);
        const q_raw = try arena.allocAligned(qk_bytes, 64);
        const k_raw = try arena.allocAligned(qk_bytes, 64);
        const v_raw = try arena.allocAligned(v_bytes, 64);
        const z_raw = try arena.allocAligned(v_bytes, 64);
        const g_raw = try arena.allocAligned(gb_bytes, 64);
        const beta_raw = try arena.allocAligned(gb_bytes, 64);
        const out_full_raw = try arena.allocAligned(v_bytes, 64);
        const out_only_raw = try arena.allocAligned(v_bytes, 64);
        const out_tail_raw = try arena.allocAligned(@as(usize, num_v_heads) * head_v_dim * @sizeOf(f16), 64);

        const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
        const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw));
        const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw));
        const z_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, z_raw));
        const g_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, g_raw));
        const beta_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, beta_raw));
        const out_full: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_full_raw));
        const out_only: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_only_raw));
        const out_tail: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_tail_raw));

        for (q_f16, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 7) % 23)) - 11)) * 0.0625);
        for (k_f16, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 3) % 25)) - 12)) * 0.0625);
        for (v_f16, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 19 + 5) % 21)) - 10)) * 0.0625);
        for (z_f16, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 11 + 9) % 19)) - 9)) * 0.125);
        for (g_f32, 0..) |*x, i| x.* = -0.05 - 0.02 * @as(f32, @floatFromInt(i % 7));
        for (beta_f32, 0..) |*x, i| x.* = -0.5 + 0.2 * @as(f32, @floatFromInt(i % 6));
        @memset(out_full, @floatCast(0.0));
        @memset(out_only, @floatCast(-9.5)); // sentinel must remain untouched when write_state_only=true
        @memset(out_tail, @floatCast(0.0));

        // CPU fp32 reference across all seq_len steps.
        const state_elems = try GdnStateArena.layerElems(state_spec);
        const ref_state = try gpa.alloc(f32, state_elems);
        defer gpa.free(ref_state);
        @memset(ref_state, 0.0);
        const ref_out = try gpa.alloc(f32, @as(usize, seq_len) * num_v_heads * head_v_dim);
        defer gpa.free(ref_out);
        const q_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_k_dim)));
        for (0..seq_len) |t| {
            const q_row = q_f16[t * num_k_heads * head_k_dim ..][0 .. num_k_heads * head_k_dim];
            const k_row = k_f16[t * num_k_heads * head_k_dim ..][0 .. num_k_heads * head_k_dim];
            const v_row = v_f16[t * num_v_heads * head_v_dim ..][0 .. num_v_heads * head_v_dim];
            const z_row = z_f16[t * num_v_heads * head_v_dim ..][0 .. num_v_heads * head_v_dim];
            const g_row = g_f32[t * num_v_heads ..][0..num_v_heads];
            const b_row = beta_f32[t * num_v_heads ..][0..num_v_heads];
            const o_row = ref_out[t * num_v_heads * head_v_dim ..][0 .. num_v_heads * head_v_dim];
            gdnStepCpuRef(q_row, k_row, v_row, g_row, b_row, z_row, null, .f16, false, ref_state, ref_state, o_row, num_k_heads, num_v_heads, head_k_dim, head_v_dim, q_scale, 1e-6, 1e-6, true);
        }

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();
        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        const q_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) };
        const k_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) };
        const v_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) };
        const z_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(z_raw.ptr) };
        const g_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(g_raw.ptr) };
        const beta_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(beta_raw.ptr) };

        {
            var f = try compute.beginFrame(5000);
            errdefer f.abort();
            // 1) Full wide prefill
            try pipes.encodeGdnPrefill(&f, .{
                .q = q_b,
                .k = k_b,
                .v = v_b,
                .g = g_b,
                .beta = beta_b,
                .z = z_b,
                .state_in = try st_full.committedBinding(gpu_buf, 0),
                .state_out = try st_full.scratchBinding(gpu_buf, 0),
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_full_raw.ptr) },
                .seq_len = seq_len,
                .num_k_heads = num_k_heads,
                .num_v_heads = num_v_heads,
                .head_k_dim = head_k_dim,
                .head_v_dim = head_v_dim,
            }, false);
            // 2) State-only wide prefill (P0-B1 non-final chunk L-1)
            try pipes.encodeGdnPrefill(&f, .{
                .q = q_b,
                .k = k_b,
                .v = v_b,
                .g = g_b,
                .beta = beta_b,
                .z = z_b,
                .state_in = try st_only.committedBinding(gpu_buf, 0),
                .state_out = try st_only.scratchBinding(gpu_buf, 0),
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_only_raw.ptr) },
                .seq_len = seq_len,
                .num_k_heads = num_k_heads,
                .num_v_heads = num_v_heads,
                .head_k_dim = head_k_dim,
                .head_v_dim = head_v_dim,
                .write_state_only = true,
            }, false);
            // 3) Final-token-only tail wide prefill (P0-B1 final chunk L-1)
            try pipes.encodeGdnPrefill(&f, .{
                .q = q_b,
                .k = k_b,
                .v = v_b,
                .g = g_b,
                .beta = beta_b,
                .z = z_b,
                .state_in = try st_full.committedBinding(gpu_buf, 0),
                .state_out = try st_only.committedBinding(gpu_buf, 0),
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_tail_raw.ptr) },
                .seq_len = seq_len,
                .num_k_heads = num_k_heads,
                .num_v_heads = num_v_heads,
                .head_k_dim = head_k_dim,
                .head_v_dim = head_v_dim,
                .out_start_token = seq_len - 1,
            }, true);
            const sig = try f.commit();
            try compute.wait(sig, 10_000);
        }

        // Verify full output vs CPU fp32 ref.
        for (ref_out, 0..) |expected, idx| {
            const got: f32 = @floatCast(out_full[idx]);
            try std.testing.expect(std.math.isFinite(got));
            try std.testing.expectApproxEqAbs(expected, got, 5e-3);
        }
        // Verify write_state_only produced identical final state and NEVER touched out_only.
        const st_a = try st_full.scratchLayerSlice(0);
        const st_b = try st_only.scratchLayerSlice(0);
        for (ref_state, st_a, st_b) |r, a, b| {
            try std.testing.expectApproxEqAbs(r, a, 2e-3);
            try std.testing.expectEqual(a, b);
        }
        for (out_only) |v| try std.testing.expectEqual(@as(f16, @floatCast(-9.5)), v);
        // Verify P0-B1 tail output matches last row of full output bit-for-bit.
        const row_elems = @as(usize, num_v_heads) * head_v_dim;
        try std.testing.expectEqualSlices(f16, out_full[(@as(usize, seq_len - 1) * row_elems)..][0..row_elems], out_tail);
    }
}

test "G15: wide QSA sparse prefill + micro-block indexer (M=128, M=256) strictly skips upper-triangular positions" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const num_q_heads: u32 = 8;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 128;
    const block_tokens: u32 = 32;
    const top_k_blocks: u32 = 4;
    const num_local_blocks: u32 = 1;
    const seq_len: u32 = 256; // 8 micro-blocks > top_k_blocks = 4
    const num_blocks: u32 = seq_len / block_tokens;

    const q_bytes = @as(usize, seq_len) * num_q_heads * head_dim * @sizeOf(f16);
    const kv_bytes = @as(usize, seq_len) * num_kv_heads * head_dim * @sizeOf(f16);
    const bk_bytes = @as(usize, num_blocks) * num_kv_heads * head_dim * @sizeOf(f16);
    const idx_spec: QsaIndexBufferSpec = .{
        .max_nodes = seq_len,
        .num_kv_heads = num_kv_heads,
        .top_k_blocks = top_k_blocks,
    };
    const idx_bytes = try QsaIndexBuffer.requiredBytes(idx_spec);
    const table_bytes = try HybridKvRing.tableRequiredBytes(1);

    const arena_cap = try page.alignUp(2 * q_bytes + 2 * kv_bytes + bk_bytes + idx_bytes + table_bytes + 8192) + page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    var ring = try HybridKvRing.init(gpa, device, &arena, .{
        .mode = .full_chunk_512,
        .num_tiles = 1,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
    });
    defer ring.deinit(gpa);

    const index_buf = try QsaIndexBuffer.init(&arena, idx_spec);
    const q_raw = try arena.allocAligned(q_bytes, 64);
    const k_raw = try arena.allocAligned(kv_bytes, 64);
    const v_raw = try arena.allocAligned(kv_bytes, 64);
    const bk_raw = try arena.allocAligned(bk_bytes, 64);
    const out_raw = try arena.allocAligned(q_bytes, 64);
    @memset(out_raw, 0);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
    const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw));
    const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw));
    const bk_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, bk_raw));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));

    for (q_f16, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 5) % 25)) - 12)) * 0.0625);
    for (k_f16, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 19 + 11) % 27)) - 13)) * 0.0625);
    for (v_f16, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 23 + 3) % 23)) - 11)) * 0.0625);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try ring.makeResident(compute);

    const q_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) };
    const k_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) };
    const v_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) };
    const bk_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(bk_raw.ptr) };
    const out_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) };

    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeQsaPrefillRope(&f, .{
            .q = q_b,
            .k = k_b,
            .v = v_b,
            .kv_tiles = ring.tileTableBinding(gpu_buf),
            .num_tokens = seq_len,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .start_pos = 0,
            .mrope = MropeSection.text1d(head_dim),
            .chunk_tokens = ring.chunk_tokens,
            .write_to_tiles = true,
        }, false);
        try pipes.encodeQsaUpdateBlockSummary(&f, .{
            .kv_tiles = ring.tileTableBinding(gpu_buf),
            .block_keys = bk_b,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .kv_len = seq_len,
            .chunk_tokens = ring.chunk_tokens,
            .block_tokens = block_tokens,
            .start_block = 0,
            .num_blocks = num_blocks,
        }, true);
        try pipes.encodeQsaAttention(&f, .{
            .q = q_b,
            .kv_tiles = ring.tileTableBinding(gpu_buf),
            .block_keys = bk_b,
            .index_buf = &index_buf,
            .index_metal_buf = gpu_buf,
            .out = out_b,
            .num_nodes = seq_len,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .kv_len = seq_len,
            .chunk_tokens = ring.chunk_tokens,
            .block_tokens = block_tokens,
            .top_k_blocks = top_k_blocks,
            .num_local_blocks = num_local_blocks,
            .causal_q_start_pos = 0,
        }, true);
        const sig = try f.commit();
        try compute.wait(sig, 10_000);
    }

    // Lock the written K/V tile to construct CPU ground truth and verify strict causality
    // on sampled queries across all 8 micro-blocks (including M=128 and M=255).
    var kg = try ring.k_tiles[0].lock(true);
    errdefer kg.release();
    var vg = try ring.v_tiles[0].lock(true);
    errdefer vg.release();
    const stride = kg.bytes_per_row / @sizeOf(f16);
    const km: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
    const vm: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, vg.bytes));

    const k_hist = try gpa.alloc(f16, @as(usize, seq_len) * num_kv_heads * head_dim);
    defer gpa.free(k_hist);
    const v_hist = try gpa.alloc(f16, @as(usize, seq_len) * num_kv_heads * head_dim);
    defer gpa.free(v_hist);
    for (0..seq_len) |t| {
        for (0..num_kv_heads) |kvh| {
            const tile_off = (kvh * ring.chunk_tokens + t) * stride;
            const hist_off = (t * num_kv_heads + kvh) * head_dim;
            @memcpy(k_hist[hist_off .. hist_off + head_dim], km[tile_off .. tile_off + head_dim]);
            @memcpy(v_hist[hist_off .. hist_off + head_dim], vm[tile_off .. tile_off + head_dim]);
        }
    }
    try vg.unlock();
    try kg.unlock();

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const ref_blocks = try gpa.alloc(u32, @as(usize, num_kv_heads) * top_k_blocks);
    defer gpa.free(ref_blocks);
    const ref_counts = try gpa.alloc(u32, num_kv_heads);
    defer gpa.free(ref_counts);
    const ref_out = try gpa.alloc(f32, @as(usize, num_q_heads) * head_dim);
    defer gpa.free(ref_out);

    for ([_]u32{ 0, 17, 63, 64, 127, 128, 195, 255 }) |q_pos| {
        const eff_kv_len = q_pos + 1;
        const max_allowed_block = q_pos / block_tokens;
        // Every selected block for query q_pos MUST satisfy block <= q_pos / block_tokens!
        for (0..num_kv_heads) |kvh| {
            const cnt = index_buf.selected_counts[@as(usize, q_pos) * num_kv_heads + kvh];
            try std.testing.expect(cnt >= 1 and cnt <= top_k_blocks);
            const sb_off = (@as(usize, q_pos) * num_kv_heads + kvh) * top_k_blocks;
            for (0..cnt) |bi| {
                const blk = index_buf.selected_blocks[sb_off + bi];
                try std.testing.expect(blk <= max_allowed_block);
            }
        }

        const q_node = q_f16[@as(usize, q_pos) * num_q_heads * head_dim ..][0 .. num_q_heads * head_dim];
        qsaIndexBlocksCpuRef(
            q_node,
            bk_f16,
            ref_blocks,
            ref_counts,
            1,
            num_q_heads,
            num_kv_heads,
            head_dim,
            eff_kv_len,
            0,
            0,
            block_tokens,
            top_k_blocks,
            num_local_blocks,
            scale,
        );
        qsaSparseAttnCpuRef(
            q_node,
            k_hist,
            v_hist,
            ref_blocks,
            ref_counts,
            null,
            null,
            null,
            ref_out,
            1,
            num_q_heads,
            num_kv_heads,
            head_dim,
            eff_kv_len,
            0,
            block_tokens,
            top_k_blocks,
            scale,
        );

        const got_node = out_f16[@as(usize, q_pos) * num_q_heads * head_dim ..][0 .. num_q_heads * head_dim];
        for (ref_out, got_node) |expected, got_h| {
            const got: f32 = @floatCast(got_h);
            try std.testing.expect(std.math.isFinite(got));
            try std.testing.expectApproxEqAbs(expected, got, 5e-3);
        }
    }
}

test "G15: wide 128-token circular HybridKvRing SWA prefill (M=128, M=256) wraps cleanly with zero overwrite hazard" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const num_q_heads: u32 = 8;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 128;
    const window_size: u32 = QSA_SWA_WINDOW_TOKENS; // 128
    const prefix_len: u32 = 90;
    const wide_m: u32 = 256; // 2x ring capacity -> wraps ring twice!
    const total_len: u32 = prefix_len + wide_m;

    const q_bytes = @as(usize, total_len) * num_q_heads * head_dim * @sizeOf(f16);
    const kv_bytes = @as(usize, total_len) * num_kv_heads * head_dim * @sizeOf(f16);
    const table_bytes = try HybridKvRing.tableRequiredBytes(1);

    const arena_cap = try page.alignUp(2 * q_bytes + 2 * kv_bytes + table_bytes + 8192) + page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    var swa_ring = try HybridKvRing.init(gpa, device, &arena, .{
        .mode = .sliding_window_128,
        .num_tiles = 1,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
    });
    defer swa_ring.deinit(gpa);

    const q_raw = try arena.allocAligned(q_bytes, 64);
    const k_raw = try arena.allocAligned(kv_bytes, 64);
    const v_raw = try arena.allocAligned(kv_bytes, 64);
    const out_raw = try arena.allocAligned(q_bytes, 64);
    @memset(out_raw, 0);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
    const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw));
    const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));

    for (q_f16, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 7) % 25)) - 12)) * 0.0625);
    for (k_f16, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 29 + 3) % 27)) - 13)) * 0.0625);
    for (v_f16, 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 11) % 23)) - 11)) * 0.0625);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try swa_ring.makeResident(compute);

    const q_row_bytes: u64 = @as(u64, num_q_heads) * head_dim * @sizeOf(f16);
    const kv_row_bytes: u64 = @as(u64, num_kv_heads) * head_dim * @sizeOf(f16);

    // Step 1: Populate the 90-token prefix directly into the 128-token circular ring (state-only).
    {
        var f = try compute.beginFrame(5000);
        errdefer f.abort();
        try pipes.encodeQsaPrefillRope(&f, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) },
            .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) },
            .kv_tiles = swa_ring.tileTableBinding(gpu_buf),
            .num_tokens = prefix_len,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .start_pos = 0,
            .mrope = MropeSection.text1d(head_dim),
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
            .kv_only = true,
            .write_to_tiles = true,
        }, false);
        // Step 2: Rotate wide_m = 256 chunk Q & K in-place (leaving prefix ring untouched!),
        // run causal SWA prefill attention across ALL 256 tokens, then commit trailing 128 tokens!
        const q_wide_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) + @as(u64, prefix_len) * q_row_bytes };
        const k_wide_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) + @as(u64, prefix_len) * kv_row_bytes };
        const v_wide_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) + @as(u64, prefix_len) * kv_row_bytes };
        const out_wide_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) + @as(u64, prefix_len) * q_row_bytes };
        try pipes.encodeQsaPrefillRope(&f, .{
            .q = q_wide_b,
            .k = k_wide_b,
            .v = v_wide_b,
            .num_tokens = wide_m,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .start_pos = prefix_len,
            .mrope = MropeSection.text1d(head_dim),
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
            .write_to_tiles = false,
        }, true);
        try pipes.encodeQsaSwaPrefillAttn(&f, .{
            .q = q_wide_b,
            .kv_tiles = swa_ring.tileTableBinding(gpu_buf),
            .k_chunk = k_wide_b,
            .v_chunk = v_wide_b,
            .out = out_wide_b,
            .num_q = wide_m,
            .q_start_pos = prefix_len,
            .chunk_start_pos = prefix_len,
            .kv_len = total_len,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .window_size = window_size,
            .ring_capacity = swa_ring.ring_capacity,
            .chunk_tokens = swa_ring.chunk_tokens,
        }, true);
        try pipes.encodeQsaPrefillKvRingCommit(&f, .{
            .k = k_wide_b,
            .v = v_wide_b,
            .kv_tiles = swa_ring.tileTableBinding(gpu_buf),
            .num_tokens = wide_m,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .start_pos = prefix_len,
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
        }, true);
        const sig = try f.commit();
        try compute.wait(sig, 10_000);
    }

    // Build full rotated K history on CPU (prefix 0..89 via CPU RoPE + chunk 90..345 already rotated in k_f16).
    const k_all = try gpa.alloc(f16, @as(usize, total_len) * num_kv_heads * head_dim);
    defer gpa.free(k_all);
    @memcpy(k_all, k_f16);
    var rot_tmp: [128]f32 = undefined;
    const mrope_1d = MropeSection.text1d(head_dim);
    for (0..prefix_len) |pos| {
        for (0..num_kv_heads) |kvh| {
            const off = (pos * num_kv_heads + kvh) * head_dim;
            qsaQkNormMropeHeadCpuRef(
                k_f16[off .. off + head_dim],
                null,
                .f16,
                &rot_tmp,
                head_dim,
                head_dim,
                Pos3D.text(@intCast(pos)),
                mrope_1d,
                .{},
                1e-6,
                1_000_000.0,
            );
            for (0..head_dim) |d| k_all[off + d] = @floatCast(rot_tmp[d]);
        }
    }

    // 1) Verify causal 128-token SWA attention output across sampled queries spanning prefix+chunk+wrap!
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const gqa_ratio = num_q_heads / num_kv_heads;
    const scores = try gpa.alloc(f32, window_size);
    defer gpa.free(scores);
    for ([_]u32{ 0, 1, 37, 38, 127, 128, 200, wide_m - 1 }) |qi| {
        const q_pos = prefix_len + qi;
        const t_hi = q_pos + 1;
        const t_lo = if (t_hi > window_size) t_hi - window_size else 0;
        const win_len = t_hi - t_lo;
        for (0..num_q_heads) |qh| {
            const kvh = qh / gqa_ratio;
            const q_vec = q_f16[(@as(usize, q_pos) * num_q_heads + qh) * head_dim ..][0..head_dim];
            var max_s: f32 = -std.math.inf(f32);
            for (0..win_len) |wi| {
                const kv_pos = t_lo + @as(u32, @intCast(wi));
                const k_vec = k_all[(@as(usize, kv_pos) * num_kv_heads + kvh) * head_dim ..][0..head_dim];
                var dv: f32 = 0.0;
                for (0..head_dim) |d| dv += @as(f32, @floatCast(q_vec[d])) * @as(f32, @floatCast(k_vec[d]));
                const s = dv * scale;
                scores[wi] = s;
                if (s > max_s) max_s = s;
            }
            var sum_p: f32 = 0.0;
            for (scores[0..win_len]) |*s| {
                s.* = @exp(s.* - max_s);
                sum_p += s.*;
            }
            const got_row = out_f16[(@as(usize, q_pos) * num_q_heads + qh) * head_dim ..][0..head_dim];
            for (0..head_dim) |d| {
                var acc: f32 = 0.0;
                for (0..win_len) |wi| {
                    const kv_pos = t_lo + @as(u32, @intCast(wi));
                    acc += scores[wi] * @as(f32, @floatCast(v_f16[(@as(usize, kv_pos) * num_kv_heads + kvh) * head_dim + d]));
                }
                const expected = acc / sum_p;
                const got: f32 = @floatCast(got_row[d]);
                try std.testing.expect(std.math.isFinite(got));
                try std.testing.expectApproxEqAbs(expected, got, 5e-3);
            }
        }
    }

    // 2) Verify final circular ring contents hold exact trailing 128 tokens [total_len - 128 .. total_len)!
    var kg = try swa_ring.k_tiles[0].lock(true);
    errdefer kg.release();
    var vg = try swa_ring.v_tiles[0].lock(true);
    errdefer vg.release();
    const stride = kg.bytes_per_row / @sizeOf(f16);
    const km: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
    const vm: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, vg.bytes));
    for ((total_len - window_size)..total_len) |pos| {
        const slot = pos % window_size;
        for (0..num_kv_heads) |kvh| {
            const tile_off = (kvh * swa_ring.chunk_tokens + slot) * stride;
            const buf_off = (pos * num_kv_heads + kvh) * head_dim;
            try std.testing.expectEqualSlices(f16, k_all[buf_off .. buf_off + head_dim], km[tile_off .. tile_off + head_dim]);
            try std.testing.expectEqualSlices(f16, v_f16[buf_off .. buf_off + head_dim], vm[tile_off .. tile_off + head_dim]);
        }
    }
    try vg.unlock();
    try kg.unlock();
}

test "R7 + G11: MlaFp4KvSpec deepseek41 890 B/tok geometry, 16 KiB alignment, and FP4 E2M1+E4M3 GPU/CPU codec parity" {
    const gpa = std.testing.allocator;

    // 1. Verify exact `890 B/tok` deepseek41 full geometry (`768 B E2M1 + 96 B E4M3 + 26 B RoPE tail`).
    const ds41_spec = MlaFp4KvSpec.deepseek41Full(1024, 10);
    try ds41_spec.validate();
    try std.testing.expectEqual(@as(u32, 1536), ds41_spec.kv_lora_rank);
    try std.testing.expectEqual(@as(u32, 26), ds41_spec.qk_rope_dim);
    try std.testing.expectEqual(MlaRopeTailFormat.e4m3, ds41_spec.rope_tail_format);
    try std.testing.expectEqual(@as(u32, 768), ds41_spec.latentDataBytesPerToken());
    try std.testing.expectEqual(@as(u32, 96), ds41_spec.latentScaleBytesPerToken());
    try std.testing.expectEqual(@as(u32, 26), ds41_spec.ropeTailBytesPerToken());
    try std.testing.expectEqual(@as(u32, DEEPSEEK41_BYTES_PER_TOKEN), ds41_spec.bytesPerToken());
    try std.testing.expectEqual(@as(u32, 890), ds41_spec.bytesPerToken());

    // 2. Verify E2M1 table & E4M3 boundaries.
    const e2m1_vals = [8]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };
    for (e2m1_vals, 0..) |v, idx| {
        const nib: u4 = @intCast(idx);
        try std.testing.expectEqual(v, e2m1NibbleToF32(nib));
        try std.testing.expectEqual(nib, f32ToE2m1Nibble(v));
        if (idx > 0) {
            const neg_nib: u4 = nib | 0x8;
            try std.testing.expectEqual(-v, e2m1NibbleToF32(neg_nib));
            try std.testing.expectEqual(neg_nib, f32ToE2m1Nibble(-v));
        }
    }
    try std.testing.expectEqual(@as(f32, 0.0), e4m3ToF32(0x00));
    try std.testing.expectEqual(@as(f32, 1.0 / 512.0), e4m3ToF32(0x01));
    try std.testing.expectEqual(@as(f32, 1.0 / 64.0), e4m3ToF32(0x08));
    try std.testing.expectEqual(@as(f32, 1.0), e4m3ToF32(f32ToE4m3(1.0)));
    try std.testing.expectEqual(@as(f32, 1.5), e4m3ToF32(f32ToE4m3(1.5)));
    try std.testing.expectEqual(@as(f32, 448.0), e4m3ToF32(f32ToE4m3(448.0)));
    try std.testing.expectEqual(@as(f32, 448.0), e4m3ToF32(f32ToE4m3(1000.0)));
    try std.testing.expectEqual(@as(f32, -448.0), e4m3ToF32(f32ToE4m3(-900.0)));

    // 3. Verify 16 KiB alignment across all M7 arenas + GPU vs CPU bit-exact FP4 encode on deepseek41 (1536 + 26 E4M3) & synthetic (128 + 32 f16).
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const ds41_small = MlaFp4KvSpec.deepseek41Full(8, 2);
    const synth_spec: MlaFp4KvSpec = .{
        .num_kv_layers = 2,
        .max_tokens = 16,
        .kv_lora_rank = 128,
        .qk_rope_dim = 32,
        .rope_tail_format = .f16,
    };
    try std.testing.expectEqual(@as(u32, 128 / 2 + 128 / 16 + 32 * 2), synth_spec.bytesPerToken());

    const ds41_bytes = try MlaFp4KvArena.requiredBytes(ds41_small);
    const synth_bytes = try MlaFp4KvArena.requiredBytes(synth_spec);
    const swa_bytes = try MlaSwaRingArena.requiredBytes(.{
        .num_kv_layers = 2,
        .swa_window = 8,
        .kv_lora_rank = 128,
        .qk_rope_dim = 32,
    });
    const sum_bytes = try Csa2SummaryArena.requiredBytes(.{
        .num_kv_layers = 2,
        .max_tokens = 16,
        .index_dim = 64,
        .block_tokens = 4,
        .coarse_group_blocks = 2,
    });
    const idx_bytes = try Csa2IndexBuffer.requiredBytes(.{ .max_nodes = 16, .top_k_blocks = 4 });
    const mhc_bytes = try MhcStreamArena.requiredBytes(.{ .max_tokens = 8, .num_streams = 4, .hidden_size = 256 });

    const total_arena = ds41_bytes + synth_bytes + swa_bytes + sum_bytes + idx_bytes + mhc_bytes + 4 * page.APPLE_PAGE_SIZE;
    var arena = try makeTestArena(total_arena);
    defer arena.deinit();

    const ds41_arena = try MlaFp4KvArena.init(&arena, ds41_small);
    const synth_arena = try MlaFp4KvArena.init(&arena, synth_spec);
    const swa_ring = try MlaSwaRingArena.init(&arena, .{
        .num_kv_layers = 2,
        .swa_window = 8,
        .kv_lora_rank = 128,
        .qk_rope_dim = 32,
    });
    const sum_arena = try Csa2SummaryArena.init(&arena, .{
        .num_kv_layers = 2,
        .max_tokens = 16,
        .index_dim = 64,
        .block_tokens = 4,
        .coarse_group_blocks = 2,
    });
    const idx_buf = try Csa2IndexBuffer.init(&arena, .{ .max_nodes = 16, .top_k_blocks = 4 });
    const mhc_arena = try MhcStreamArena.init(&arena, .{ .max_tokens = 8, .num_streams = 4, .hidden_size = 256 });

    try page.checkShareable(ds41_arena.bytes().ptr, ds41_arena.bytes().len);
    try page.checkShareable(synth_arena.bytes().ptr, synth_arena.bytes().len);
    try page.checkShareable(swa_ring.backing.ptr, swa_ring.backing.len);
    try page.checkShareable(sum_arena.backing.ptr, sum_arena.backing.len);
    try page.checkShareable(idx_buf.backing.ptr, idx_buf.backing.len);
    try page.checkShareable(mhc_arena.backing.ptr, mhc_arena.backing.len);

    const num_toks: u32 = 4;
    const in_lat_bytes = try page.alignUp(@as(usize, num_toks) * 1536 * @sizeOf(f16));
    const in_rope_bytes = try page.alignUp(@as(usize, num_toks) * 32 * @sizeOf(f16));
    const in_lat_raw = try arena.allocPages(in_lat_bytes);
    const in_rope_raw = try arena.allocPages(in_rope_bytes);
    const in_lat_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, in_lat_raw));
    const in_rope_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, in_rope_raw));
    for (0..@as(usize, num_toks) * 1536) |i| in_lat_f16[i] = @floatCast(synthF32(i + 17, 0.17));
    for (0..@as(usize, num_toks) * 26) |i| in_rope_f16[i] = @floatCast(synthF32(i + 91, 0.21));

    const arena_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer arena_buf.close();
    try compute.makeResident(&.{arena_buf});

    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeMlaFp4KvWrite(&frame, .{
            .latent_in = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(in_lat_raw.ptr)) },
            .rope_in = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(in_rope_raw.ptr)) },
            .kv_arena = &ds41_arena,
            .kv_metal_buf = arena_buf,
            .kv_slot = 0,
            .num_tokens = num_toks,
            .start_pos = 0,
            .is_encoder_projection = true,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    // Compare GPU-written FP4 bytes bit-for-bit vs host reference AND verify reconstruction error bounds.
    const ref_e2m1 = try gpa.alloc(u8, 1536 / 2);
    defer gpa.free(ref_e2m1);
    const ref_scales = try gpa.alloc(u8, 1536 / 16);
    defer gpa.free(ref_scales);
    const ref_rope = try gpa.alloc(u8, 26);
    defer gpa.free(ref_rope);
    const src_f32 = try gpa.alloc(f32, 1536);
    defer gpa.free(src_f32);
    const dec_f32 = try gpa.alloc(f32, 1536);
    defer gpa.free(dec_f32);
    var src_rope_f32: [26]f32 = undefined;
    var dec_rope_f32: [26]f32 = undefined;

    const gpu_e2m1 = try ds41_arena.layerE2m1Slice(0);
    const gpu_scales = try ds41_arena.layerScaleSlice(0);
    const gpu_rope = try ds41_arena.layerRopeSlice(0);

    var sq_err: f64 = 0.0;
    var sq_sig: f64 = 0.0;
    var max_rel_rope_err: f32 = 0.0;

    for (0..num_toks) |t| {
        for (0..1536) |d| src_f32[d] = @floatCast(in_lat_f16[t * 1536 + d]);
        for (0..26) |d| src_rope_f32[d] = @floatCast(in_rope_f16[t * 26 + d]);
        encodeMlaFp4LatentRowCpu(src_f32, ref_e2m1, ref_scales);
        encodeMlaRopeRowCpu(&src_rope_f32, .e4m3, ref_rope);

        try std.testing.expectEqualSlices(u8, ref_e2m1, gpu_e2m1[t * 768 .. (t + 1) * 768]);
        try std.testing.expectEqualSlices(u8, ref_scales, gpu_scales[t * 96 .. (t + 1) * 96]);
        try std.testing.expectEqualSlices(u8, ref_rope, gpu_rope[t * 26 .. (t + 1) * 26]);

        decodeMlaFp4LatentRowCpu(ref_e2m1, ref_scales, dec_f32);
        for (0..1536) |d| {
            const diff: f64 = @as(f64, dec_f32[d]) - @as(f64, src_f32[d]);
            sq_err += diff * diff;
            sq_sig += @as(f64, src_f32[d]) * @as(f64, src_f32[d]);
        }
        decodeMlaRopeRowCpu(ref_rope, .e4m3, &dec_rope_f32);
        for (0..26) |d| {
            const denom = @max(@abs(src_rope_f32[d]), 0.1);
            max_rel_rope_err = @max(max_rel_rope_err, @abs(dec_rope_f32[d] - src_rope_f32[d]) / denom);
        }
    }
    const rel_rmse = @sqrt(sq_err / sq_sig);
    // FP4 E2M1 + per-16 E4M3 typically achieves ~7-10% relative RMSE; enforce strict < 14% bound.
    try std.testing.expect(rel_rmse < 0.14);
    // E4M3 has 3 mantissa bits (max relative step 1/16 = 6.25%); enforce < 8%.
    try std.testing.expect(max_rel_rope_err < 0.08);
}

test "R7 + G11: MLA + CSA2 Full/Reindex/Reuse, SWA bounded replay, causal prefill M=64, CED prefix, decode M=1,4 & tree verify vs CPU fp32 ref" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const kv_lora_rank: u32 = 128;
    const qk_rope_dim: u32 = 32;
    const index_dim: u32 = 64;
    const num_q_heads: u32 = 4;
    const block_tokens: u32 = 8;
    const coarse_group_blocks: u32 = 2;
    const top_k_blocks: u32 = 3;
    const top_k_coarse: u32 = 2;
    const swa_window: u32 = 16;
    const max_tokens: u32 = 128;
    const prefill_m: u32 = 64;

    // Only 1 physical KV slot is allocated for the 3 logical layers (`Full`, `Reindex`, `Reuse`),
    // plus 4 slots in a second arena for M=4 batched decode!
    const kv_spec: MlaFp4KvSpec = .{
        .num_kv_layers = 4,
        .max_tokens = max_tokens,
        .kv_lora_rank = kv_lora_rank,
        .qk_rope_dim = qk_rope_dim,
        .rope_tail_format = .f16,
    };
    const swa_spec: MlaSwaRingSpec = .{
        .num_kv_layers = 4,
        .swa_window = swa_window,
        .kv_lora_rank = kv_lora_rank,
        .qk_rope_dim = qk_rope_dim,
    };
    const sum_spec: Csa2SummarySpec = .{
        .num_kv_layers = 4,
        .max_tokens = max_tokens,
        .index_dim = index_dim,
        .block_tokens = block_tokens,
        .coarse_group_blocks = coarse_group_blocks,
    };
    const idx_spec: Csa2IndexBufferSpec = .{
        .max_nodes = prefill_m,
        .top_k_blocks = top_k_blocks,
    };

    const kv_bytes = try MlaFp4KvArena.requiredBytes(kv_spec);
    const swa_bytes = try MlaSwaRingArena.requiredBytes(swa_spec);
    const sum_bytes = try Csa2SummaryArena.requiredBytes(sum_spec);
    const idx_bytes = try Csa2IndexBuffer.requiredBytes(idx_spec);
    const io_lat_bytes = try page.alignUp(@as(usize, prefill_m) * kv_lora_rank * @sizeOf(f16));
    const io_rope_bytes = try page.alignUp(@as(usize, prefill_m) * qk_rope_dim * @sizeOf(f16));
    const q_nope_bytes = try page.alignUp(@as(usize, prefill_m) * num_q_heads * kv_lora_rank * @sizeOf(f16));
    const q_pe_bytes = try page.alignUp(@as(usize, prefill_m) * num_q_heads * qk_rope_dim * @sizeOf(f16));
    const out_bytes = q_nope_bytes;
    const misc_bytes = page.APPLE_PAGE_SIZE;

    var arena = try makeTestArena(
        kv_bytes + swa_bytes + sum_bytes + 2 * idx_bytes +
            io_lat_bytes + io_rope_bytes + 2 * q_nope_bytes + q_pe_bytes + 3 * out_bytes + 4 * misc_bytes,
    );
    defer arena.deinit();

    const kv_arena = try MlaFp4KvArena.init(&arena, kv_spec);
    const swa_ring = try MlaSwaRingArena.init(&arena, swa_spec);
    const sum_arena = try Csa2SummaryArena.init(&arena, sum_spec);
    const idx_full = try Csa2IndexBuffer.init(&arena, idx_spec);
    const idx_reindex = try Csa2IndexBuffer.init(&arena, idx_spec);

    const lat_raw = try arena.allocPages(io_lat_bytes);
    const rope_raw = try arena.allocPages(io_rope_bytes);
    const q_nope_0_raw = try arena.allocPages(q_nope_bytes);
    const q_nope_1_raw = try arena.allocPages(q_nope_bytes);
    const q_pe_raw = try arena.allocPages(q_pe_bytes);
    const out_full_raw = try arena.allocPages(out_bytes);
    const out_reindex_raw = try arena.allocPages(out_bytes);
    const out_reuse_raw = try arena.allocPages(out_bytes);
    const batch_lens_raw = try arena.allocPages(misc_bytes);
    const tree_lat_raw = try arena.allocPages(misc_bytes);
    const tree_rope_raw = try arena.allocPages(misc_bytes);
    const tree_masks_raw = try arena.allocPages(misc_bytes);

    const lat_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, lat_raw));
    const rope_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, rope_raw));
    const q_nope_0: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_nope_0_raw));
    const q_nope_1: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_nope_1_raw));
    const q_pe: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_pe_raw));
    const out_full: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_full_raw));
    const out_reindex: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_reindex_raw));
    const out_reuse: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_reuse_raw));
    const batch_lens: []u32 = @alignCast(std.mem.bytesAsSlice(u32, batch_lens_raw));
    const tree_lat_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, tree_lat_raw));
    const tree_rope_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, tree_rope_raw));
    const tree_masks: []u64 = @alignCast(std.mem.bytesAsSlice(u64, tree_masks_raw));

    for (0..@as(usize, prefill_m) * kv_lora_rank) |i| lat_f16[i] = @floatCast(synthF32(i + 11, 0.09));
    for (0..@as(usize, prefill_m) * qk_rope_dim) |i| rope_f16[i] = @floatCast(synthF32(i + 73, 0.11));
    for (0..@as(usize, prefill_m) * num_q_heads * kv_lora_rank) |i| {
        q_nope_0[i] = @floatCast(synthF32(i + 131, 0.08));
        q_nope_1[i] = @floatCast(synthF32(i + 509, 0.08));
    }
    for (0..@as(usize, prefill_m) * num_q_heads * qk_rope_dim) |i| q_pe[i] = @floatCast(synthF32(i + 257, 0.09));

    const arena_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer arena_buf.close();
    try compute.makeResident(&.{arena_buf});

    const lat_bind: metal.BufferBinding = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(lat_raw.ptr)) };
    const rope_bind: metal.BufferBinding = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(rope_raw.ptr)) };
    const q0_bind: metal.BufferBinding = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(q_nope_0_raw.ptr)) };
    const q1_bind: metal.BufferBinding = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(q_nope_1_raw.ptr)) };
    const qpe_bind: metal.BufferBinding = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(q_pe_raw.ptr)) };
    const ofull_bind: metal.BufferBinding = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(out_full_raw.ptr)) };
    const oreidx_bind: metal.BufferBinding = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(out_reindex_raw.ptr)) };
    const oreuse_bind: metal.BufferBinding = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(out_reuse_raw.ptr)) };
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_lora_rank + qk_rope_dim)));
    const idx_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(index_dim)));

    // --- Part A: Causal Prefill (M = 64) across Full -> Reindex -> Reuse + r=1 loop reuse ---
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        const d_full = try pipes.encodeMlaCsa2Layer(&frame, .{
            .mode = .full,
            .shared_kv_owner = 0,
            .loop_step = 0,
            .write_latent_in = lat_bind,
            .write_rope_in = rope_bind,
            .write_start_pos = 0,
            .write_num_tokens = prefill_m,
            .q_nope = q0_bind,
            .q_pe = qpe_bind,
            .kv_arena = &kv_arena,
            .kv_metal_buf = arena_buf,
            .swa_ring = &swa_ring,
            .swa_metal_buf = arena_buf,
            .summary_arena = &sum_arena,
            .summary_metal_buf = arena_buf,
            .index_buf = &idx_full,
            .index_metal_buf = arena_buf,
            .out = ofull_bind,
            .num_nodes = prefill_m,
            .num_q_heads = num_q_heads,
            .kv_len = prefill_m,
            .causal_q_start_pos = 0,
            .swa_window = swa_window,
            .block_tokens = block_tokens,
            .top_k_blocks = top_k_blocks,
            .top_k_coarse = top_k_coarse,
            .scale = attn_scale,
        }, false);
        try std.testing.expectEqual(@as(u32, 1), d_full.kv_writes);
        try std.testing.expectEqual(@as(u32, 1), d_full.summary_updates);
        try std.testing.expectEqual(@as(u32, 1), d_full.index_dispatches);
        try std.testing.expectEqual(@as(u32, 1), d_full.attn_dispatches);

        // Reindex: 0 KV writes (aliases slot 0), own indexer over q_nope_1.
        const d_reindex = try pipes.encodeMlaCsa2Layer(&frame, .{
            .mode = .reindex,
            .shared_kv_owner = 0,
            .loop_step = 0,
            .write_latent_in = lat_bind, // deliberately passed: Reindex MUST execute 0 KV writes!
            .write_rope_in = rope_bind,
            .write_start_pos = 0,
            .write_num_tokens = prefill_m,
            .q_nope = q1_bind,
            .q_pe = qpe_bind,
            .kv_arena = &kv_arena,
            .kv_metal_buf = arena_buf,
            .swa_ring = &swa_ring,
            .swa_metal_buf = arena_buf,
            .summary_arena = &sum_arena,
            .summary_metal_buf = arena_buf,
            .index_buf = &idx_reindex,
            .index_metal_buf = arena_buf,
            .out = oreidx_bind,
            .num_nodes = prefill_m,
            .num_q_heads = num_q_heads,
            .kv_len = prefill_m,
            .causal_q_start_pos = 0,
            .swa_window = swa_window,
            .block_tokens = block_tokens,
            .top_k_blocks = top_k_blocks,
            .top_k_coarse = top_k_coarse,
            .scale = attn_scale,
        }, true);
        try std.testing.expectEqual(@as(u32, 0), d_reindex.kv_writes);
        try std.testing.expectEqual(@as(u32, 0), d_reindex.summary_updates);
        try std.testing.expectEqual(@as(u32, 1), d_reindex.index_dispatches);
        try std.testing.expectEqual(@as(u32, 1), d_reindex.attn_dispatches);

        // Reuse: 0 KV writes AND 0 index dispatches (aliases slot 0 + idx_reindex).
        const d_reuse = try pipes.encodeMlaCsa2Layer(&frame, .{
            .mode = .reuse,
            .shared_kv_owner = 0,
            .loop_step = 0,
            .write_latent_in = lat_bind,
            .write_rope_in = rope_bind,
            .write_start_pos = 0,
            .write_num_tokens = prefill_m,
            .q_nope = q1_bind,
            .q_pe = qpe_bind,
            .kv_arena = &kv_arena,
            .kv_metal_buf = arena_buf,
            .swa_ring = &swa_ring,
            .swa_metal_buf = arena_buf,
            .index_buf = &idx_reindex,
            .index_metal_buf = arena_buf,
            .out = oreuse_bind,
            .num_nodes = prefill_m,
            .num_q_heads = num_q_heads,
            .kv_len = prefill_m,
            .causal_q_start_pos = 0,
            .swa_window = swa_window,
            .block_tokens = block_tokens,
            .top_k_blocks = top_k_blocks,
            .scale = attn_scale,
        }, true);
        try std.testing.expectEqual(@as(u32, 0), d_reuse.kv_writes);
        try std.testing.expectEqual(@as(u32, 0), d_reuse.summary_updates);
        try std.testing.expectEqual(@as(u32, 0), d_reuse.index_dispatches);
        try std.testing.expectEqual(@as(u32, 1), d_reuse.attn_dispatches);

        // LoopSchedule r = 1 pass on a Full layer MUST execute 0 KV writes and 0 index dispatches.
        const d_loop1 = try pipes.encodeMlaCsa2Layer(&frame, .{
            .mode = .full,
            .shared_kv_owner = 0,
            .loop_step = 1,
            .write_latent_in = lat_bind,
            .write_rope_in = rope_bind,
            .write_start_pos = 0,
            .write_num_tokens = prefill_m,
            .q_nope = q0_bind,
            .q_pe = qpe_bind,
            .kv_arena = &kv_arena,
            .kv_metal_buf = arena_buf,
            .swa_ring = &swa_ring,
            .swa_metal_buf = arena_buf,
            .index_buf = &idx_full,
            .index_metal_buf = arena_buf,
            .out = ofull_bind,
            .num_nodes = prefill_m,
            .num_q_heads = num_q_heads,
            .kv_len = prefill_m,
            .causal_q_start_pos = 0,
            .swa_window = swa_window,
            .block_tokens = block_tokens,
            .top_k_blocks = top_k_blocks,
            .scale = attn_scale,
        }, true);
        try std.testing.expectEqual(@as(u32, 0), d_loop1.kv_writes);
        try std.testing.expectEqual(@as(u32, 0), d_loop1.summary_updates);
        try std.testing.expectEqual(@as(u32, 0), d_loop1.index_dispatches);
        try std.testing.expectEqual(@as(u32, 1), d_loop1.attn_dispatches);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    // Verify strictly causal indexing (no upper-triangular blocks selected).
    for (0..prefill_m) |node| {
        const cnt = idx_full.counts[node];
        for (0..cnt) |k| {
            const b = idx_full.blocks[node * top_k_blocks + k];
            try std.testing.expect(b * block_tokens <= node);
        }
    }
    // Reuse output with same query & aliased index MUST be bit-identical to Reindex output.
    const total_out_elems: usize = @as(usize, prefill_m) * num_q_heads * kv_lora_rank;
    try std.testing.expectEqualSlices(f16, out_reindex[0..total_out_elems], out_reuse[0..total_out_elems]);

    // Verify CPU fp32 reference parity for Full & Reindex prefill (M = 64).
    const cpu_fine = try gpa.alloc(f16, sum_spec.maxFineBlocks() * index_dim);
    defer gpa.free(cpu_fine);
    const cpu_coarse = try gpa.alloc(f16, sum_spec.maxCoarseBlocks() * index_dim);
    defer gpa.free(cpu_coarse);
    const cpu_sel_blocks = try gpa.alloc(u32, @as(usize, prefill_m) * top_k_blocks);
    defer gpa.free(cpu_sel_blocks);
    const cpu_sel_counts = try gpa.alloc(u32, prefill_m);
    defer gpa.free(cpu_sel_counts);
    const cpu_out = try gpa.alloc(f32, total_out_elems);
    defer gpa.free(cpu_out);

    const e2m1_s0 = try kv_arena.layerE2m1Slice(0);
    const scale_s0 = try kv_arena.layerScaleSlice(0);
    const rope_s0 = try kv_arena.layerRopeSlice(0);
    const swa_lat_s0: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, swa_ring.backing[0..swa_ring.latent_plane_bytes]));
    const swa_rope_s0: []const f16 = @alignCast(std.mem.bytesAsSlice(
        f16,
        swa_ring.backing[@intCast(swa_ring.rope_plane_offset) .. @as(usize, @intCast(swa_ring.rope_plane_offset)) + swa_ring.rope_plane_bytes],
    ));

    csa2UpdateSummariesCpuRef(e2m1_s0, scale_s0, prefill_m, kv_lora_rank, index_dim, block_tokens, coarse_group_blocks, cpu_fine, cpu_coarse);
    csa2IndexBlocksCpuRef(
        q_nope_0,
        cpu_fine,
        cpu_coarse,
        prefill_m,
        num_q_heads,
        kv_lora_rank,
        index_dim,
        prefill_m,
        0,
        0,
        swa_window,
        block_tokens,
        coarse_group_blocks,
        top_k_coarse,
        top_k_blocks,
        1,
        attn_scale,
        cpu_sel_blocks,
        cpu_sel_counts,
    );
    try std.testing.expectEqualSlices(u32, cpu_sel_counts, idx_full.counts[0..prefill_m]);
    try std.testing.expectEqualSlices(u32, cpu_sel_blocks, idx_full.blocks[0 .. @as(usize, prefill_m) * top_k_blocks]);

    mlaCsa2AttendCpuRef(
        q_nope_0,
        q_pe,
        e2m1_s0,
        scale_s0,
        rope_s0,
        .f16,
        swa_lat_s0,
        swa_rope_s0,
        swa_window,
        cpu_sel_blocks,
        cpu_sel_counts,
        null,
        null,
        null,
        cpu_out,
        prefill_m,
        num_q_heads,
        kv_lora_rank,
        qk_rope_dim,
        prefill_m,
        0,
        0,
        swa_window,
        block_tokens,
        top_k_blocks,
        attn_scale,
    );
    var max_err_prefill: f32 = 0.0;
    for (0..total_out_elems) |i| {
        max_err_prefill = @max(max_err_prefill, @abs(@as(f32, @floatCast(out_full[i])) - cpu_out[i]));
    }
    try std.testing.expect(max_err_prefill < 2e-3);

    // --- Part B: Single-token decode (M = 1) append at pos 64 (kv_len = 65) + SWA bounded replay ---
    for (0..kv_lora_rank) |d| lat_f16[d] = @floatCast(synthF32(d + 901, 0.12));
    for (0..qk_rope_dim) |d| rope_f16[d] = @floatCast(synthF32(d + 977, 0.12));
    for (0..@as(usize, num_q_heads) * kv_lora_rank) |i| q_nope_0[i] = @floatCast(synthF32(i + 1021, 0.09));
    for (0..@as(usize, num_q_heads) * qk_rope_dim) |i| q_pe[i] = @floatCast(synthF32(i + 1103, 0.09));

    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        const d_dec1 = try pipes.encodeMlaCsa2Layer(&frame, .{
            .mode = .full,
            .shared_kv_owner = 0,
            .loop_step = 0,
            .write_latent_in = lat_bind,
            .write_rope_in = rope_bind,
            .write_start_pos = 64,
            .write_num_tokens = 1,
            .q_nope = q0_bind,
            .q_pe = qpe_bind,
            .kv_arena = &kv_arena,
            .kv_metal_buf = arena_buf,
            .swa_ring = &swa_ring,
            .swa_metal_buf = arena_buf,
            .summary_arena = &sum_arena,
            .summary_metal_buf = arena_buf,
            .index_buf = &idx_full,
            .index_metal_buf = arena_buf,
            .out = ofull_bind,
            .num_nodes = 1,
            .num_q_heads = num_q_heads,
            .kv_len = 65,
            .causal_q_start_pos = 64,
            .swa_window = swa_window,
            .block_tokens = block_tokens,
            .top_k_blocks = top_k_blocks,
            .top_k_coarse = top_k_coarse,
            .scale = attn_scale,
        }, false);
        try std.testing.expectEqual(@as(u32, 1), d_dec1.kv_writes);
        try std.testing.expectEqual(@as(u32, 1), d_dec1.summary_updates);
        try std.testing.expectEqual(@as(u32, 1), d_dec1.index_dispatches);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    csa2UpdateSummariesCpuRef(e2m1_s0, scale_s0, 65, kv_lora_rank, index_dim, block_tokens, coarse_group_blocks, cpu_fine, cpu_coarse);
    csa2IndexBlocksCpuRef(
        q_nope_0,
        cpu_fine,
        cpu_coarse,
        1,
        num_q_heads,
        kv_lora_rank,
        index_dim,
        65,
        0,
        64,
        swa_window,
        block_tokens,
        coarse_group_blocks,
        top_k_coarse,
        top_k_blocks,
        1,
        attn_scale,
        cpu_sel_blocks,
        cpu_sel_counts,
    );
    try std.testing.expectEqual(cpu_sel_counts[0], idx_full.counts[0]);
    try std.testing.expectEqualSlices(u32, cpu_sel_blocks[0..top_k_blocks], idx_full.blocks[0..top_k_blocks]);

    mlaCsa2AttendCpuRef(
        q_nope_0,
        q_pe,
        e2m1_s0,
        scale_s0,
        rope_s0,
        .f16,
        swa_lat_s0,
        swa_rope_s0,
        swa_window,
        cpu_sel_blocks,
        cpu_sel_counts,
        null,
        null,
        null,
        cpu_out,
        1,
        num_q_heads,
        kv_lora_rank,
        qk_rope_dim,
        65,
        0,
        64,
        swa_window,
        block_tokens,
        top_k_blocks,
        attn_scale,
    );
    var max_err_dec1: f32 = 0.0;
    for (0..@as(usize, num_q_heads) * kv_lora_rank) |i| {
        max_err_dec1 = @max(max_err_dec1, @abs(@as(f32, @floatCast(out_full[i])) - cpu_out[i]));
    }
    try std.testing.expect(max_err_dec1 < 2e-3);

    // --- Part C: Speculative tree verify (M = 4, u64 branching ancestor masks) + CED encoder prefix visibility ---
    const tree_m: u32 = 4;
    for (0..@as(usize, tree_m) * kv_lora_rank) |i| tree_lat_f16[i] = @floatCast(synthF32(i + 1501, 0.10));
    for (0..@as(usize, tree_m) * qk_rope_dim) |i| tree_rope_f16[i] = @floatCast(synthF32(i + 1701, 0.10));
    tree_masks[0] = 0b0001;
    tree_masks[1] = 0b0011;
    tree_masks[2] = 0b0111;
    tree_masks[3] = 0b1011;

    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeCsa2IndexBlocks(&frame, .{
            .q_nope = q0_bind,
            .summary_arena = &sum_arena,
            .summary_metal_buf = arena_buf,
            .kv_slot = 0,
            .index_buf = &idx_full,
            .index_metal_buf = arena_buf,
            .num_nodes = tree_m,
            .num_q_heads = num_q_heads,
            .kv_lora_rank = kv_lora_rank,
            .kv_len = 65,
            .encoder_prefix_len = 32,
            .causal_q_start_pos = null,
            .swa_window = swa_window,
            .top_k_blocks = top_k_blocks,
            .top_k_coarse = top_k_coarse,
            .scale = idx_scale,
        }, false);
        try pipes.encodeMlaCsa2Attend(&frame, .{
            .q_nope = q0_bind,
            .q_pe = qpe_bind,
            .kv_arena = &kv_arena,
            .kv_metal_buf = arena_buf,
            .kv_slot = 0,
            .swa_ring = &swa_ring,
            .swa_metal_buf = arena_buf,
            .index_buf = &idx_full,
            .index_metal_buf = arena_buf,
            .kv_tree_latent = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(tree_lat_raw.ptr)) },
            .kv_tree_rope = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(tree_rope_raw.ptr)) },
            .ancestor_masks = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(tree_masks_raw.ptr)) },
            .out = ofull_bind,
            .num_nodes = tree_m,
            .num_q_heads = num_q_heads,
            .kv_len = 65,
            .encoder_prefix_len = 32,
            .causal_q_start_pos = null,
            .swa_window = swa_window,
            .block_tokens = block_tokens,
            .top_k_blocks = top_k_blocks,
            .scale = attn_scale,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    csa2IndexBlocksCpuRef(
        q_nope_0,
        cpu_fine,
        cpu_coarse,
        tree_m,
        num_q_heads,
        kv_lora_rank,
        index_dim,
        65,
        32,
        null,
        swa_window,
        block_tokens,
        coarse_group_blocks,
        top_k_coarse,
        top_k_blocks,
        1,
        idx_scale,
        cpu_sel_blocks,
        cpu_sel_counts,
    );
    mlaCsa2AttendCpuRef(
        q_nope_0,
        q_pe,
        e2m1_s0,
        scale_s0,
        rope_s0,
        .f16,
        swa_lat_s0,
        swa_rope_s0,
        swa_window,
        cpu_sel_blocks,
        cpu_sel_counts,
        tree_lat_f16,
        tree_rope_f16,
        tree_masks,
        cpu_out,
        tree_m,
        num_q_heads,
        kv_lora_rank,
        qk_rope_dim,
        65,
        32,
        null,
        swa_window,
        block_tokens,
        top_k_blocks,
        attn_scale,
    );
    var max_err_tree: f32 = 0.0;
    for (0..@as(usize, tree_m) * num_q_heads * kv_lora_rank) |i| {
        max_err_tree = @max(max_err_tree, @abs(@as(f32, @floatCast(out_full[i])) - cpu_out[i]));
    }
    try std.testing.expect(max_err_tree < 2e-3);

    // --- Part D: Batched decode (M = 4 independent slots with distinct kv_lens) ---
    const batch_m: u32 = 4;
    batch_lens[0] = 19;
    batch_lens[1] = 37;
    batch_lens[2] = 52;
    batch_lens[3] = 64;

    // Populate slots 0..3 with distinct lengths & summaries.
    for (0..batch_m) |slot| {
        const slen = batch_lens[slot];
        for (0..@as(usize, slen) * kv_lora_rank) |i| lat_f16[i] = @floatCast(synthF32(i + slot * 313 + 7, 0.09));
        for (0..@as(usize, slen) * qk_rope_dim) |i| rope_f16[i] = @floatCast(synthF32(i + slot * 211 + 13, 0.09));
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeMlaFp4KvWrite(&frame, .{
            .latent_in = lat_bind,
            .rope_in = rope_bind,
            .kv_arena = &kv_arena,
            .kv_metal_buf = arena_buf,
            .kv_slot = @intCast(slot),
            .swa_ring = &swa_ring,
            .swa_metal_buf = arena_buf,
            .num_tokens = slen,
            .start_pos = 0,
        }, false);
        try pipes.encodeCsa2UpdateSummaries(&frame, .{
            .kv_arena = &kv_arena,
            .kv_metal_buf = arena_buf,
            .summary_arena = &sum_arena,
            .summary_metal_buf = arena_buf,
            .kv_slot = @intCast(slot),
            .kv_len = slen,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    const batch_lens_bind: metal.BufferBinding = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(batch_lens_raw.ptr)) };
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeCsa2IndexBlocks(&frame, .{
            .q_nope = q0_bind,
            .summary_arena = &sum_arena,
            .summary_metal_buf = arena_buf,
            .kv_slot = 0,
            .index_buf = &idx_full,
            .index_metal_buf = arena_buf,
            .num_nodes = batch_m,
            .num_q_heads = num_q_heads,
            .kv_lora_rank = kv_lora_rank,
            .kv_len = 0, // driven per-slot by batch_kv_lens
            .swa_window = swa_window,
            .top_k_blocks = top_k_blocks,
            .top_k_coarse = top_k_coarse,
            .batch_kv_lens = batch_lens_bind,
            .scale = idx_scale,
        }, false);
        try pipes.encodeMlaCsa2Attend(&frame, .{
            .q_nope = q0_bind,
            .q_pe = qpe_bind,
            .kv_arena = &kv_arena,
            .kv_metal_buf = arena_buf,
            .kv_slot = 0,
            .swa_ring = &swa_ring,
            .swa_metal_buf = arena_buf,
            .index_buf = &idx_full,
            .index_metal_buf = arena_buf,
            .batch_kv_lens = batch_lens_bind,
            .out = ofull_bind,
            .num_nodes = batch_m,
            .num_q_heads = num_q_heads,
            .kv_len = 0,
            .swa_window = swa_window,
            .block_tokens = block_tokens,
            .top_k_blocks = top_k_blocks,
            .scale = attn_scale,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    var max_err_batch: f32 = 0.0;
    const slot_out_elems = @as(usize, num_q_heads) * kv_lora_rank;
    for (0..batch_m) |slot| {
        const slen = batch_lens[slot];
        const e2m1_s = try kv_arena.layerE2m1Slice(@intCast(slot));
        const scale_s = try kv_arena.layerScaleSlice(@intCast(slot));
        const rope_s = try kv_arena.layerRopeSlice(@intCast(slot));
        const swa_off = slot * @as(usize, @intCast(swa_ring.layer_stride_bytes));
        const swa_lat_s: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, swa_ring.backing[swa_off .. swa_off + swa_ring.latent_plane_bytes]));
        const swa_rope_off = swa_off + @as(usize, @intCast(swa_ring.rope_plane_offset));
        const swa_rope_s: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, swa_ring.backing[swa_rope_off .. swa_rope_off + swa_ring.rope_plane_bytes]));

        csa2UpdateSummariesCpuRef(e2m1_s, scale_s, slen, kv_lora_rank, index_dim, block_tokens, coarse_group_blocks, cpu_fine, cpu_coarse);
        csa2IndexBlocksCpuRef(
            q_nope_0[slot * slot_out_elems ..][0..slot_out_elems],
            cpu_fine,
            cpu_coarse,
            1,
            num_q_heads,
            kv_lora_rank,
            index_dim,
            slen,
            0,
            null,
            swa_window,
            block_tokens,
            coarse_group_blocks,
            top_k_coarse,
            top_k_blocks,
            1,
            idx_scale,
            cpu_sel_blocks,
            cpu_sel_counts,
        );
        try std.testing.expectEqual(cpu_sel_counts[0], idx_full.counts[slot]);
        try std.testing.expectEqualSlices(u32, cpu_sel_blocks[0..top_k_blocks], idx_full.blocks[slot * top_k_blocks .. (slot + 1) * top_k_blocks]);

        mlaCsa2AttendCpuRef(
            q_nope_0[slot * slot_out_elems ..][0..slot_out_elems],
            q_pe[slot * num_q_heads * qk_rope_dim ..][0 .. num_q_heads * qk_rope_dim],
            e2m1_s,
            scale_s,
            rope_s,
            .f16,
            swa_lat_s,
            swa_rope_s,
            swa_window,
            cpu_sel_blocks,
            cpu_sel_counts,
            null,
            null,
            null,
            cpu_out[0..slot_out_elems],
            1,
            num_q_heads,
            kv_lora_rank,
            qk_rope_dim,
            slen,
            0,
            null,
            swa_window,
            block_tokens,
            top_k_blocks,
            attn_scale,
        );
        for (0..slot_out_elems) |i| {
            max_err_batch = @max(max_err_batch, @abs(@as(f32, @floatCast(out_full[slot * slot_out_elems + i])) - cpu_out[i]));
        }
    }
    try std.testing.expect(max_err_batch < 2e-3);
}

test "G10: Single-Pass Mega-mHC (4-stream f16 & 6-stream bf16, fused RMSNorm, Sinkhorn Birkhoff H_res & post-mix) vs CPU fp32 ref" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    // Verify Birkhoff doubly-stochastic manifold property on Sinkhorn-Knopp projection.
    const raw_mat = [_]f32{
        1.2,  -0.4, 0.3,  0.8,
        -0.5, 0.9,  1.1,  -0.2,
        0.4,  0.7,  -0.8, 1.3,
        -0.1, 0.2,  0.6,  0.5,
    };
    var ds_mat: [16]f32 = undefined;
    mhcSinkhornCpuRef(&raw_mat, 4, 16, &ds_mat);
    for (0..4) |o| {
        var rsum: f32 = 0.0;
        var csum: f32 = 0.0;
        for (0..4) |i| {
            try std.testing.expect(ds_mat[o * 4 + i] >= 0.0);
            rsum += ds_mat[o * 4 + i];
            csum += ds_mat[i * 4 + o];
        }
        try std.testing.expect(@abs(rsum - 1.0) < 1e-4);
        try std.testing.expect(@abs(csum - 1.0) < 1e-4);
    }

    const Case = struct {
        m: u32,
        num_streams: u32,
        hidden_size: u32,
        stream_dt: quant.FloatDType,
        act_dt: quant.FloatDType,
        norm_dt: quant.FloatDType,
        apply_h_res_pre: bool,
        apply_h_res_post: bool,
        sinkhorn_iters: u32,
        activate_coeffs: bool,
        post_scale: f32,
        tol: f32,
    };
    const cases = [_]Case{
        // Canonical DeepSeek-V4.1 4-stream f16 Mega-mHC: pre-mix + fused RMSNorm -> post-mix with Birkhoff H_res + 2*sigmoid H_post.
        .{
            .m = 8,
            .num_streams = 4,
            .hidden_size = 384,
            .stream_dt = .f16,
            .act_dt = .f16,
            .norm_dt = .f16,
            .apply_h_res_pre = false,
            .apply_h_res_post = true,
            .sinkhorn_iters = 12,
            .activate_coeffs = true,
            .post_scale = 2.0,
            .tol = 3e-3,
        },
        // Generalized 6-stream bf16 Mega-mHC with in-place pre-mix H_res and bf16 fused RMSNorm.
        .{
            .m = 4,
            .num_streams = 6,
            .hidden_size = 256,
            .stream_dt = .bf16,
            .act_dt = .bf16,
            .norm_dt = .bf16,
            .apply_h_res_pre = true,
            .apply_h_res_post = false,
            .sinkhorn_iters = 10,
            .activate_coeffs = false,
            .post_scale = 1.0,
            .tol = 2e-2,
        },
    };

    for (cases, 0..) |tc, ci| {
        const cfg: MhcConfig = .{
            .num_streams = tc.num_streams,
            .hidden_size = tc.hidden_size,
            .stream_dtype = tc.stream_dt,
            .act_dtype = tc.act_dt,
            .norm_dtype = tc.norm_dt,
            .rms_eps = 1e-6,
            .sinkhorn_iters = tc.sinkhorn_iters,
            .activate_coeffs = tc.activate_coeffs,
            .shared_coeffs = false,
            .apply_h_res_pre = tc.apply_h_res_pre,
            .apply_h_res_post = tc.apply_h_res_post,
            .post_scale = tc.post_scale,
        };
        const stream_spec: MhcStreamSpec = .{
            .max_tokens = tc.m,
            .num_streams = tc.num_streams,
            .hidden_size = tc.hidden_size,
            .dtype = tc.stream_dt,
        };
        const stream_bytes = try MhcStreamArena.requiredBytes(stream_spec);
        const act_bytes = try page.alignUp(@as(usize, tc.m) * tc.hidden_size * 2);
        const norm_bytes = try page.alignUp(@as(usize, tc.hidden_size) * 2);
        const coeff_bytes = page.APPLE_PAGE_SIZE;

        var arena = try makeTestArena(stream_bytes + 2 * act_bytes + norm_bytes + 3 * coeff_bytes);
        defer arena.deinit();

        const mhc_arena = try MhcStreamArena.init(&arena, stream_spec);
        const branch_out_raw = try arena.allocPages(act_bytes);
        const sublayer_raw = try arena.allocPages(act_bytes);
        const norm_w_raw = try arena.allocPages(norm_bytes);
        const h_pre_raw = try arena.allocPages(coeff_bytes);
        const h_post_raw = try arena.allocPages(coeff_bytes);
        const h_res_raw = try arena.allocPages(coeff_bytes);

        const stream_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, mhc_arena.backing));
        const branch_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, branch_out_raw));
        const sublayer_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, sublayer_raw));
        const norm_w_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, norm_w_raw));
        const h_pre: []f32 = @alignCast(std.mem.bytesAsSlice(f32, h_pre_raw));
        const h_post: []f32 = @alignCast(std.mem.bytesAsSlice(f32, h_post_raw));
        const h_res: []f32 = @alignCast(std.mem.bytesAsSlice(f32, h_res_raw));

        const total_stream_elems: usize = @as(usize, tc.m) * tc.num_streams * tc.hidden_size;
        const total_act_elems: usize = @as(usize, tc.m) * tc.hidden_size;

        const cpu_streams = try gpa.alloc(f32, total_stream_elems);
        defer gpa.free(cpu_streams);
        const cpu_branch = try gpa.alloc(f32, total_act_elems);
        defer gpa.free(cpu_branch);
        const cpu_sublayer = try gpa.alloc(f32, total_act_elems);
        defer gpa.free(cpu_sublayer);

        for (0..total_stream_elems) |i| {
            const bits = quant.f32ToFloatBits(tc.stream_dt, synthF32(i + ci * 997 + 19, 0.08));
            stream_bits[i] = bits;
            cpu_streams[i] = quant.floatBitsToF32(tc.stream_dt, bits);
        }
        for (0..total_act_elems) |i| {
            const bits = quant.f32ToFloatBits(tc.act_dt, synthF32(i + ci * 613 + 41, 0.07));
            sublayer_bits[i] = bits;
            cpu_sublayer[i] = quant.floatBitsToF32(tc.act_dt, bits);
        }
        for (0..tc.hidden_size) |d| {
            norm_w_bits[d] = quant.f32ToFloatBits(tc.norm_dt, 0.8 + synthF32(d + 5, 0.02));
        }
        for (0..@as(usize, tc.m) * tc.num_streams) |i| {
            h_pre[i] = if (tc.activate_coeffs) synthF32(i + 201, 0.15) else (0.25 + synthF32(i + 201, 0.01));
            h_post[i] = if (tc.activate_coeffs) synthF32(i + 307, 0.15) else (0.30 + synthF32(i + 307, 0.01));
        }
        for (0..@as(usize, tc.m) * tc.num_streams * tc.num_streams) |i| {
            h_res[i] = synthF32(i + 409, 0.12);
        }

        const arena_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer arena_buf.close();
        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        try compute.makeResident(&.{arena_buf});

        {
            var frame = try compute.beginFrame(5000);
            errdefer frame.abort();
            try pipes.encodeMegaMhc(&frame, .{
                .config = cfg,
                .streams = mhc_arena.binding(arena_buf),
                .h_pre = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(h_pre_raw.ptr)) },
                .h_res = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(h_res_raw.ptr)) },
                .norm_w = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(norm_w_raw.ptr)) },
                .branch_out = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(branch_out_raw.ptr)) },
                .m = tc.m,
            }, false);
            try pipes.encodeMegaMhcPostMix(&frame, .{
                .config = cfg,
                .streams = mhc_arena.binding(arena_buf),
                .sublayer_out = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(sublayer_raw.ptr)) },
                .h_post = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(h_post_raw.ptr)) },
                .h_res = .{ .buffer = arena_buf, .offset = @intCast(arena.offsetOf(h_res_raw.ptr)) },
                .m = tc.m,
            }, true);
            const sig = try frame.commit();
            try compute.wait(sig, 5000);
        }

        mhcSinglePassCpuRef(cfg, cpu_streams, h_pre, h_res, norm_w_bits[0..tc.hidden_size], cpu_branch, tc.m);
        // Quantize intermediate streams if apply_h_res_pre wrote back in stream_dt before post-mix read.
        if (tc.apply_h_res_pre) {
            for (0..total_stream_elems) |i| {
                cpu_streams[i] = quant.floatBitsToF32(tc.stream_dt, quant.f32ToFloatBits(tc.stream_dt, cpu_streams[i]));
            }
        }
        mhcPostMixCpuRef(cfg, cpu_streams, cpu_sublayer, h_post, h_res, tc.m);

        var max_branch_err: f32 = 0.0;
        for (0..total_act_elems) |i| {
            const got = quant.floatBitsToF32(tc.act_dt, branch_bits[i]);
            max_branch_err = @max(max_branch_err, @abs(got - cpu_branch[i]));
        }
        var max_stream_err: f32 = 0.0;
        for (0..total_stream_elems) |i| {
            const got = quant.floatBitsToF32(tc.stream_dt, stream_bits[i]);
            max_stream_err = @max(max_stream_err, @abs(got - cpu_streams[i]));
        }
        try std.testing.expect(max_branch_err < tc.tol);
        try std.testing.expect(max_stream_err < tc.tol);
    }
}

test "gpu.hybrid: G19 QSA and MLA simdgroup_half8x8 prefill parity vs scalar/vector path with short-sequence guard" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    // 1. QSA prefill MMA parity (D=64 & D=128) + short-sequence guard verification.
    try std.testing.expect(!HybridPipelines.shouldUseQsaMmaPrefill(.{
        .q = undefined,
        .k_chunk = undefined,
        .v_chunk = undefined,
        .out = undefined,
        .num_q = 16,
        .q_start_pos = 0,
        .chunk_start_pos = 0,
        .kv_len = 128,
        .num_q_heads = 8,
        .num_kv_heads = 2,
        .head_dim = 64,
    }));
    try std.testing.expect(HybridPipelines.shouldUseQsaMmaPrefill(.{
        .q = undefined,
        .k_chunk = undefined,
        .v_chunk = undefined,
        .out = undefined,
        .num_q = 32,
        .q_start_pos = 0,
        .chunk_start_pos = 0,
        .kv_len = 64,
        .num_q_heads = 8,
        .num_kv_heads = 2,
        .head_dim = 128,
    }));

    for ([_]u32{ 64, 128 }) |hd| {
        const num_q: u32 = 40;
        const kv_len: u32 = 72;
        const num_q_heads: u32 = 8;
        const num_kv_heads: u32 = 2;
        const q_elems: usize = @as(usize, num_q) * num_q_heads * hd;
        const kv_elems: usize = @as(usize, kv_len) * num_kv_heads * hd;
        var arena = try makeTestArena(try page.alignUp((3 * q_elems + 2 * kv_elems) * 2 + 4096));
        defer arena.deinit();

        const q_raw = try arena.allocAligned(q_elems * 2, 64);
        const k_raw = try arena.allocAligned(kv_elems * 2, 64);
        const v_raw = try arena.allocAligned(kv_elems * 2, 64);
        const o_vec_raw = try arena.allocAligned(q_elems * 2, 64);
        const o_mma_raw = try arena.allocAligned(q_elems * 2, 64);

        const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
        const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw));
        const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw));
        const o_vec: []f16 = @alignCast(std.mem.bytesAsSlice(f16, o_vec_raw));
        const o_mma: []f16 = @alignCast(std.mem.bytesAsSlice(f16, o_mma_raw));
        for (q_f16, 0..) |*v, i| v.* = @floatCast(synthF32(i + 13, 0.08));
        for (k_f16, 0..) |*v, i| v.* = @floatCast(synthF32(i + 37, 0.08));
        for (v_f16, 0..) |*v, i| v.* = @floatCast(synthF32(i + 61, 0.08));

        const buf = try metal.Buffer.wrap(device, arena.bytes());
        defer buf.close();
        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        try compute.makeResident(&.{buf});

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        const base_args: QsaSwaPrefillAttnArgs = .{
            .q = .{ .buffer = buf, .offset = arena.offsetOf(q_raw.ptr) },
            .k_chunk = .{ .buffer = buf, .offset = arena.offsetOf(k_raw.ptr) },
            .v_chunk = .{ .buffer = buf, .offset = arena.offsetOf(v_raw.ptr) },
            .out = .{ .buffer = buf, .offset = arena.offsetOf(o_vec_raw.ptr) },
            .num_q = num_q,
            .q_start_pos = kv_len - num_q,
            .chunk_start_pos = 0,
            .kv_len = kv_len,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = hd,
            .window_size = 48,
            .ring_capacity = 0,
            .chunk_tokens = 512,
            .use_mma_override = false,
        };
        try pipes.encodeQsaSwaPrefillAttn(&frame, base_args, false);
        var mma_args = base_args;
        mma_args.out = .{ .buffer = buf, .offset = arena.offsetOf(o_mma_raw.ptr) };
        mma_args.use_mma_override = true;
        try pipes.encodeQsaSwaPrefillAttn(&frame, mma_args, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        for (0..q_elems) |i| {
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(o_vec[i])), @as(f32, @floatCast(o_mma[i])), 3e-3);
        }
    }

    // 2. MLA + CSA2 prefill MMA parity (`num_nodes = 40 >= 32`, `kv_len = 80 >= 64`).
    const mla_nodes: u32 = 40;
    const mla_kv_len: u32 = 80;
    const mla_heads: u32 = 4;
    const mla_rank: u32 = 160;
    const mla_rope: u32 = 32;
    const swa_win: u32 = 24;
    const blk_tok: u32 = 16;
    const top_k: u32 = 3;

    const kv_spec: MlaFp4KvSpec = .{
        .num_kv_layers = 1,
        .max_tokens = 128,
        .kv_lora_rank = mla_rank,
        .qk_rope_dim = mla_rope,
        .rope_tail_format = .f16,
    };
    const swa_spec: MlaSwaRingSpec = .{
        .num_kv_layers = 1,
        .swa_window = swa_win,
        .kv_lora_rank = mla_rank,
        .qk_rope_dim = mla_rope,
    };
    const idx_spec: Csa2IndexBufferSpec = .{
        .max_nodes = mla_nodes,
        .top_k_blocks = top_k,
    };
    const q_nope_elems: usize = @as(usize, mla_nodes) * mla_heads * mla_rank;
    const q_pe_elems: usize = @as(usize, mla_nodes) * mla_heads * mla_rope;
    const lat_elems: usize = @as(usize, mla_kv_len) * mla_rank;
    const rope_elems: usize = @as(usize, mla_kv_len) * mla_rope;
    const arena_need = try MlaFp4KvArena.requiredBytes(kv_spec) +
        try MlaSwaRingArena.requiredBytes(swa_spec) +
        try Csa2IndexBuffer.requiredBytes(idx_spec) +
        try page.alignUp((3 * q_nope_elems + q_pe_elems + lat_elems + rope_elems) * 2 + 4096);
    var mla_arena = try makeTestArena(arena_need);
    defer mla_arena.deinit();

    const kv_arena = try MlaFp4KvArena.init(&mla_arena, kv_spec);
    const swa_ring = try MlaSwaRingArena.init(&mla_arena, swa_spec);
    const idx_buf = try Csa2IndexBuffer.init(&mla_arena, idx_spec);

    const q_nope_raw = try mla_arena.allocAligned(q_nope_elems * 2, 64);
    const q_pe_raw = try mla_arena.allocAligned(q_pe_elems * 2, 64);
    const lat_raw = try mla_arena.allocAligned(lat_elems * 2, 64);
    const rope_raw = try mla_arena.allocAligned(rope_elems * 2, 64);
    const out_vec_raw = try mla_arena.allocAligned(q_nope_elems * 2, 64);
    const out_mma_raw = try mla_arena.allocAligned(q_nope_elems * 2, 64);

    const q_nope_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_nope_raw));
    const q_pe_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_pe_raw));
    const lat_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, lat_raw));
    const rope_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, rope_raw));
    const out_vec_mla: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_vec_raw));
    const out_mma_mla: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_mma_raw));

    for (q_nope_f16, 0..) |*v, i| v.* = @floatCast(synthF32(i + 11, 0.07));
    for (q_pe_f16, 0..) |*v, i| v.* = @floatCast(synthF32(i + 23, 0.07));
    for (lat_f16, 0..) |*v, i| v.* = @floatCast(synthF32(i + 47, 0.08));
    for (rope_f16, 0..) |*v, i| v.* = @floatCast(synthF32(i + 71, 0.08));

    const sel_blocks = idx_buf.blocks;
    const sel_counts = idx_buf.counts;
    for (0..mla_nodes) |ni| {
        sel_counts[ni] = 2;
        sel_blocks[ni * top_k + 0] = 0;
        sel_blocks[ni * top_k + 1] = @intCast((ni % 2) + 1);
        sel_blocks[ni * top_k + 2] = 0xFFFF_FFFF;
    }

    const mla_buf = try metal.Buffer.wrap(device, mla_arena.bytes());
    defer mla_buf.close();
    var mla_compute = try metal.Compute.initWithFrames(device, 2);
    defer mla_compute.close();
    try mla_compute.makeResident(&.{mla_buf});

    {
        var frame = try mla_compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeMlaFp4KvWrite(&frame, .{
            .latent_in = .{ .buffer = mla_buf, .offset = mla_arena.offsetOf(lat_raw.ptr) },
            .rope_in = .{ .buffer = mla_buf, .offset = mla_arena.offsetOf(rope_raw.ptr) },
            .kv_arena = &kv_arena,
            .kv_metal_buf = mla_buf,
            .kv_slot = 0,
            .swa_ring = &swa_ring,
            .swa_metal_buf = mla_buf,
            .num_tokens = mla_kv_len,
            .start_pos = 0,
        }, false);

        const base_mla: MlaCsa2AttendArgs = .{
            .q_nope = .{ .buffer = mla_buf, .offset = mla_arena.offsetOf(q_nope_raw.ptr) },
            .q_pe = .{ .buffer = mla_buf, .offset = mla_arena.offsetOf(q_pe_raw.ptr) },
            .kv_arena = &kv_arena,
            .kv_metal_buf = mla_buf,
            .kv_slot = 0,
            .swa_ring = &swa_ring,
            .swa_metal_buf = mla_buf,
            .index_buf = &idx_buf,
            .index_metal_buf = mla_buf,
            .out = .{ .buffer = mla_buf, .offset = mla_arena.offsetOf(out_vec_raw.ptr) },
            .num_nodes = mla_nodes,
            .num_q_heads = mla_heads,
            .kv_len = mla_kv_len,
            .causal_q_start_pos = mla_kv_len - mla_nodes,
            .swa_window = swa_win,
            .block_tokens = blk_tok,
            .top_k_blocks = top_k,
            .use_mma_override = false,
        };
        try pipes.encodeMlaCsa2Attend(&frame, base_mla, true);
        var mma_mla = base_mla;
        mma_mla.out = .{ .buffer = mla_buf, .offset = mla_arena.offsetOf(out_mma_raw.ptr) };
        mma_mla.use_mma_override = true;
        try pipes.encodeMlaCsa2Attend(&frame, mma_mla, false);
        const sig = try frame.commit();
        try mla_compute.wait(sig, 5000);
    }

    _ = gpa;
    for (0..q_nope_elems) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(out_vec_mla[i])), @as(f32, @floatCast(out_mma_mla[i])), 4e-3);
    }
}

test "M9.1 G26: asymmetric 192/128 128+sink HybridKvRing SWA (prefill >128 wrap, decode, verifyTree+commitTreeKv) vs CPU fp32 ref" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const num_q_heads: u32 = 16;
    const num_kv_heads: u32 = 4;
    const k_head_dim: u32 = 192;
    const v_head_dim: u32 = 128;
    const rotary_dim: u32 = 64;
    const window_size: u32 = QSA_SWA_WINDOW_TOKENS; // 128
    const rope_theta: f32 = 10000.0;
    const v_scale: f32 = 0.707;
    const prefix_len: u32 = 90;
    const chunk2_len: u32 = 80; // 90 + 80 = 170 > 128 -> forces circular wraparound!
    const prefill_total: u32 = prefix_len + chunk2_len;
    const tree_m: u32 = 12;

    const q_row_elems: usize = @as(usize, num_q_heads) * k_head_dim;
    const k_row_elems: usize = @as(usize, num_kv_heads) * k_head_dim;
    const v_row_elems: usize = @as(usize, num_kv_heads) * v_head_dim;
    const o_row_elems: usize = @as(usize, num_q_heads) * v_head_dim;

    const max_rows: usize = @max(@as(usize, chunk2_len), @as(usize, prefix_len));
    const q_bytes = max_rows * q_row_elems * @sizeOf(f16);
    const k_bytes = max_rows * k_row_elems * @sizeOf(f16);
    const v_bytes = max_rows * v_row_elems * @sizeOf(f16);
    const o_bytes = max_rows * o_row_elems * @sizeOf(f16);
    const sink_bytes = @as(usize, num_q_heads) * @sizeOf(u16);
    const table_bytes = try HybridKvRing.tableRequiredBytes(1);

    const arena_cap = try page.alignUp(
        q_bytes + 2 * k_bytes + 2 * v_bytes + o_bytes + sink_bytes + table_bytes +
            @as(usize, tree_m) * (@sizeOf(u64) + @sizeOf(u32)) + 16384,
    ) + page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    var swa_ring = try HybridKvRing.init(gpa, device, &arena, .{
        .mode = .sliding_window_128,
        .num_tiles = 1,
        .num_kv_heads = num_kv_heads,
        .head_dim = k_head_dim,
        .v_head_dim = v_head_dim,
    });
    defer swa_ring.deinit(gpa);
    try std.testing.expectEqual(k_head_dim, swa_ring.head_dim);
    try std.testing.expectEqual(v_head_dim, swa_ring.v_head_dim);

    const q_raw = try arena.allocAligned(q_bytes, 64);
    const k_raw = try arena.allocAligned(k_bytes, 64);
    const v_raw = try arena.allocAligned(v_bytes, 64);
    const k_out_raw = try arena.allocAligned(k_bytes, 64);
    const v_out_raw = try arena.allocAligned(v_bytes, 64);
    const o_raw = try arena.allocAligned(o_bytes, 64);
    const sink_raw = try arena.allocAligned(sink_bytes, 64);
    const masks_raw = try arena.allocAligned(@as(usize, tree_m) * @sizeOf(u64), 64);
    const pos_raw = try arena.allocAligned(@as(usize, tree_m) * @sizeOf(u32), 64);

    const q_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0..q_bytes]));
    const k_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw[0..k_bytes]));
    const v_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw[0..v_bytes]));
    const o_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, o_raw[0..o_bytes]));
    const sink_bf16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, sink_raw[0..sink_bytes]));
    const masks_h: []u64 = @alignCast(std.mem.bytesAsSlice(u64, masks_raw[0 .. @as(usize, tree_m) * @sizeOf(u64)]));
    const pos_h: []u32 = @alignCast(std.mem.bytesAsSlice(u32, pos_raw[0 .. @as(usize, tree_m) * @sizeOf(u32)]));

    var sink_f32: [num_q_heads]f32 = undefined;
    for (0..num_q_heads) |h_idx| {
        const s_val: f32 = if ((h_idx % 2) == 0) (1.2 + 0.25 * @as(f32, @floatFromInt(h_idx))) else (-0.8 - 0.1 * @as(f32, @floatFromInt(h_idx)));
        const b16 = quant.f32ToBf16(s_val);
        sink_bf16[h_idx] = b16;
        sink_f32[h_idx] = quant.bf16ToF32(b16);
    }

    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{buf});
    try swa_ring.makeResident(compute);
    defer swa_ring.removeResident(compute) catch |err| {
        std.debug.panic("removeResident failed: {}", .{err});
    };

    const q_bind = metal.BufferBinding{ .buffer = buf, .offset = arena.offsetOf(q_raw.ptr) };
    const k_bind = metal.BufferBinding{ .buffer = buf, .offset = arena.offsetOf(k_raw.ptr) };
    const v_bind = metal.BufferBinding{ .buffer = buf, .offset = arena.offsetOf(v_raw.ptr) };
    const k_out_bind = metal.BufferBinding{ .buffer = buf, .offset = arena.offsetOf(k_out_raw.ptr) };
    const v_out_bind = metal.BufferBinding{ .buffer = buf, .offset = arena.offsetOf(v_out_raw.ptr) };
    const o_bind = metal.BufferBinding{ .buffer = buf, .offset = arena.offsetOf(o_raw.ptr) };
    const sink_bind = metal.BufferBinding{ .buffer = buf, .offset = arena.offsetOf(sink_raw.ptr) };
    const masks_bind = metal.BufferBinding{ .buffer = buf, .offset = arena.offsetOf(masks_raw.ptr) };
    const pos_bind = metal.BufferBinding{ .buffer = buf, .offset = arena.offsetOf(pos_raw.ptr) };
    const ring_tbl_bind = swa_ring.tileTableBinding(buf);

    // CPU FP32 reference KV history (up to 200 tokens) + partial NeoX RoPE helper.
    const max_hist: usize = 200;
    const ref_k = try gpa.alloc(f32, max_hist * k_row_elems);
    defer gpa.free(ref_k);
    const ref_v = try gpa.alloc(f32, max_hist * v_row_elems);
    defer gpa.free(ref_v);

    const applyPartialRopeF32 = struct {
        fn call(vec: []f32, pos: u32, r_dim: u32, theta: f32) void {
            const half: usize = r_dim / 2;
            const p_f: f32 = @floatFromInt(pos);
            for (0..half) |i| {
                const exp_arg = -(@as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(r_dim)));
                const freq = std.math.pow(f32, theta, exp_arg);
                const ang = p_f * freq;
                const c = @cos(ang);
                const s = @sin(ang);
                const x0 = vec[i];
                const x1 = vec[i + half];
                vec[i] = x0 * c - x1 * s;
                vec[i + half] = x1 * c + x0 * s;
            }
            for (vec) |*v| v.* = @floatCast(@as(f16, @floatCast(v.*)));
        }
    }.call;

    const evalSwaCpuRef = struct {
        fn call(
            q_rot: []const f32,
            k_hist: []const f32,
            v_hist: []const f32,
            k_extra: ?[]const f32,
            v_extra: ?[]const f32,
            q_pos: u32,
            prefix_kv_len: u32,
            extra_mask: u64,
            extra_count: u32,
            sinks: []const f32,
            out_row: []f32,
        ) void {
            const qk_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(k_head_dim)));
            const gqa_ratio = num_q_heads / num_kv_heads;
            const t_lo: u32 = if (q_pos + 1 > window_size) (q_pos + 1 - window_size) else 0;
            const t_hi: u32 = @min(prefix_kv_len, q_pos + 1);
            for (0..num_q_heads) |qh| {
                const kvh = qh / gqa_ratio;
                var m_val: f32 = sinks[qh];
                var l_val: f32 = 1.0;
                var acc: [v_head_dim]f32 = @splat(0.0);

                var t: u32 = t_lo;
                while (t < t_hi) : (t += 1) {
                    var dot: f32 = 0.0;
                    const ks = k_hist[t * k_row_elems + kvh * k_head_dim ..][0..k_head_dim];
                    const qs = q_rot[qh * k_head_dim ..][0..k_head_dim];
                    for (0..k_head_dim) |d| dot += qs[d] * ks[d];
                    const sc = dot * qk_scale;
                    const new_m = @max(m_val, sc);
                    const old_s = @exp(m_val - new_m);
                    const cur_s = @exp(sc - new_m);
                    l_val = l_val * old_s + cur_s;
                    const vs = v_hist[t * v_row_elems + kvh * v_head_dim ..][0..v_head_dim];
                    for (0..v_head_dim) |d| acc[d] = acc[d] * old_s + cur_s * vs[d];
                    m_val = new_m;
                }
                if (k_extra != null and v_extra != null) {
                    const ke = k_extra.?;
                    const ve = v_extra.?;
                    for (0..extra_count) |ei| {
                        if (((extra_mask >> @intCast(ei)) & 1) == 0) continue;
                        var dot: f32 = 0.0;
                        const ks = ke[ei * k_row_elems + kvh * k_head_dim ..][0..k_head_dim];
                        const qs = q_rot[qh * k_head_dim ..][0..k_head_dim];
                        for (0..k_head_dim) |d| dot += qs[d] * ks[d];
                        const sc = dot * qk_scale;
                        const new_m = @max(m_val, sc);
                        const old_s = @exp(m_val - new_m);
                        const cur_s = @exp(sc - new_m);
                        l_val = l_val * old_s + cur_s;
                        const vs = ve[ei * v_row_elems + kvh * v_head_dim ..][0..v_head_dim];
                        for (0..v_head_dim) |d| acc[d] = acc[d] * old_s + cur_s * vs[d];
                        m_val = new_m;
                    }
                }
                const inv_l: f32 = 1.0 / l_val;
                for (0..v_head_dim) |d| out_row[qh * v_head_dim + d] = acc[d] * inv_l;
            }
        }
    }.call;

    // --- Phase 1a: Prefill first 90 tokens into circular 128-token HybridKvRing ---
    for (0..prefix_len) |t| {
        for (0..q_row_elems) |i| q_h[t * q_row_elems + i] = @floatCast(synthF32(1000 + t * 17 + i, 0.25));
        for (0..k_row_elems) |i| k_h[t * k_row_elems + i] = @floatCast(synthF32(2000 + t * 23 + i, 0.25));
        for (0..v_row_elems) |i| v_h[t * v_row_elems + i] = @floatCast(synthF32(3000 + t * 31 + i, 0.25));
        for (0..num_kv_heads) |kvh| {
            const ks = ref_k[t * k_row_elems + kvh * k_head_dim ..][0..k_head_dim];
            for (0..k_head_dim) |d| ks[d] = @floatCast(k_h[t * k_row_elems + kvh * k_head_dim + d]);
            applyPartialRopeF32(ks, @intCast(t), rotary_dim, rope_theta);
            const vs = ref_v[t * v_row_elems + kvh * v_head_dim ..][0..v_head_dim];
            for (0..v_head_dim) |d| vs[d] = @floatCast(@as(f16, @floatCast(@as(f32, @floatCast(v_h[t * v_row_elems + kvh * v_head_dim + d])) * v_scale)));
        }
    }
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG26QkNormRopeKv(&frame, .{
            .q = q_bind,
            .k = k_bind,
            .v = v_bind,
            .k_out = k_out_bind,
            .v_out = v_out_bind,
            .num_tokens = prefix_len,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .rotary_dim = rotary_dim,
            .start_pos = 0,
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
            .kv_only = true,
            .write_to_tiles = false,
            .rope_theta = rope_theta,
            .v_scale = v_scale,
        }, false);
        try pipes.encodeG26KvCommit(&frame, .{
            .k_src = k_out_bind,
            .v_src = v_out_bind,
            .kv_tiles = ring_tbl_bind,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .num_tokens = prefix_len,
            .start_pos = 0,
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    // --- Phase 1b: Prefill next 80 tokens (pos 90..169 -> crosses 128 wrap!) ---
    const q_rot_chunk = try gpa.alloc(f32, chunk2_len * q_row_elems);
    defer gpa.free(q_rot_chunk);
    for (0..chunk2_len) |ci| {
        const t = prefix_len + @as(u32, @intCast(ci));
        for (0..q_row_elems) |i| q_h[ci * q_row_elems + i] = @floatCast(synthF32(5000 + t * 19 + i, 0.25));
        for (0..k_row_elems) |i| k_h[ci * k_row_elems + i] = @floatCast(synthF32(6000 + t * 29 + i, 0.25));
        for (0..v_row_elems) |i| v_h[ci * v_row_elems + i] = @floatCast(synthF32(7000 + t * 37 + i, 0.25));
        for (0..num_q_heads) |qh| {
            const qs = q_rot_chunk[ci * q_row_elems + qh * k_head_dim ..][0..k_head_dim];
            for (0..k_head_dim) |d| qs[d] = @floatCast(q_h[ci * q_row_elems + qh * k_head_dim + d]);
            applyPartialRopeF32(qs, t, rotary_dim, rope_theta);
        }
        for (0..num_kv_heads) |kvh| {
            const ks = ref_k[t * k_row_elems + kvh * k_head_dim ..][0..k_head_dim];
            for (0..k_head_dim) |d| ks[d] = @floatCast(k_h[ci * k_row_elems + kvh * k_head_dim + d]);
            applyPartialRopeF32(ks, t, rotary_dim, rope_theta);
            const vs = ref_v[t * v_row_elems + kvh * v_head_dim ..][0..v_head_dim];
            for (0..v_head_dim) |d| vs[d] = @floatCast(@as(f16, @floatCast(@as(f32, @floatCast(v_h[ci * v_row_elems + kvh * v_head_dim + d])) * v_scale)));
        }
    }
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG26QkNormRopeKv(&frame, .{
            .q = q_bind,
            .k = k_bind,
            .v = v_bind,
            .k_out = k_out_bind,
            .v_out = v_out_bind,
            .num_tokens = chunk2_len,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .rotary_dim = rotary_dim,
            .start_pos = prefix_len,
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
            .kv_only = false,
            .write_to_tiles = false,
            .rope_theta = rope_theta,
            .v_scale = v_scale,
        }, false);
        try pipes.encodeG26SwaAttn(&frame, .{
            .q = q_bind,
            .kv_tiles = ring_tbl_bind,
            .k_extra = k_out_bind,
            .v_extra = v_out_bind,
            .sink_bias = sink_bind,
            .sink_dtype = .bf16,
            .out = o_bind,
            .num_q = chunk2_len,
            .q_start_pos = prefix_len,
            .chunk_start_pos = prefix_len,
            .kv_len = prefill_total,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .window_size = window_size,
            .ring_capacity = swa_ring.ring_capacity,
            .chunk_tokens = swa_ring.chunk_tokens,
            .is_prefill = true,
        }, true);
        try pipes.encodeG26KvCommit(&frame, .{
            .k_src = k_out_bind,
            .v_src = v_out_bind,
            .kv_tiles = ring_tbl_bind,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .num_tokens = chunk2_len,
            .start_pos = prefix_len,
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    const ref_out = try gpa.alloc(f32, o_row_elems);
    defer gpa.free(ref_out);
    var max_err_prefill: f32 = 0.0;
    for (0..chunk2_len) |ci| {
        const q_pos = prefix_len + @as(u32, @intCast(ci));
        evalSwaCpuRef(
            q_rot_chunk[ci * q_row_elems ..][0..q_row_elems],
            ref_k,
            ref_v,
            null,
            null,
            q_pos,
            q_pos + 1,
            0,
            0,
            &sink_f32,
            ref_out,
        );
        for (0..o_row_elems) |i| {
            const diff = @abs(@as(f32, @floatCast(o_h[ci * o_row_elems + i])) - ref_out[i]);
            if (diff > max_err_prefill) max_err_prefill = diff;
        }
    }
    try std.testing.expect(max_err_prefill < 4e-3);

    // --- Phase 2: M = 1 Decode after circular wraparound (pos = 170) ---
    const dec_pos: u32 = prefill_total; // 170
    const q_rot_dec = try gpa.alloc(f32, q_row_elems);
    defer gpa.free(q_rot_dec);
    for (0..q_row_elems) |i| q_h[i] = @floatCast(synthF32(9000 + i, 0.25));
    for (0..k_row_elems) |i| k_h[i] = @floatCast(synthF32(9500 + i, 0.25));
    for (0..v_row_elems) |i| v_h[i] = @floatCast(synthF32(9800 + i, 0.25));
    for (0..num_q_heads) |qh| {
        const qs = q_rot_dec[qh * k_head_dim ..][0..k_head_dim];
        for (0..k_head_dim) |d| qs[d] = @floatCast(q_h[qh * k_head_dim + d]);
        applyPartialRopeF32(qs, dec_pos, rotary_dim, rope_theta);
    }
    for (0..num_kv_heads) |kvh| {
        const ks = ref_k[dec_pos * k_row_elems + kvh * k_head_dim ..][0..k_head_dim];
        for (0..k_head_dim) |d| ks[d] = @floatCast(k_h[kvh * k_head_dim + d]);
        applyPartialRopeF32(ks, dec_pos, rotary_dim, rope_theta);
        const vs = ref_v[dec_pos * v_row_elems + kvh * v_head_dim ..][0..v_head_dim];
        for (0..v_head_dim) |d| vs[d] = @floatCast(@as(f16, @floatCast(@as(f32, @floatCast(v_h[kvh * v_head_dim + d])) * v_scale)));
    }
    masks_h[0] = 1;
    pos_h[0] = dec_pos;
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG26QkNormRopeKv(&frame, .{
            .q = q_bind,
            .k = k_bind,
            .v = v_bind,
            .positions_1d = pos_bind,
            .k_out = k_out_bind,
            .v_out = v_out_bind,
            .num_tokens = 1,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .rotary_dim = rotary_dim,
            .start_pos = dec_pos,
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
            .kv_only = false,
            .write_to_tiles = false,
            .rope_theta = rope_theta,
            .v_scale = v_scale,
        }, false);
        try pipes.encodeG26SwaAttn(&frame, .{
            .q = q_bind,
            .kv_tiles = ring_tbl_bind,
            .k_extra = k_out_bind,
            .v_extra = v_out_bind,
            .ancestor_masks = masks_bind,
            .node_positions = pos_bind,
            .sink_bias = sink_bind,
            .sink_dtype = .bf16,
            .out = o_bind,
            .num_q = 1,
            .q_start_pos = dec_pos,
            .kv_len = dec_pos,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .window_size = window_size,
            .ring_capacity = swa_ring.ring_capacity,
            .chunk_tokens = swa_ring.chunk_tokens,
            .is_prefill = false,
        }, true);
        const acc0 = [_]u16{0};
        try pipes.encodeG26KvCommit(&frame, .{
            .k_src = k_out_bind,
            .v_src = v_out_bind,
            .kv_tiles = ring_tbl_bind,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .start_pos = dec_pos,
            .accepted_nodes = &acc0,
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }
    evalSwaCpuRef(q_rot_dec, ref_k, ref_v, null, null, dec_pos, dec_pos + 1, 0, 0, &sink_f32, ref_out);
    for (0..o_row_elems) |i| {
        try std.testing.expectApproxEqAbs(ref_out[i], @as(f32, @floatCast(o_h[i])), 4e-3);
    }

    // --- Phase 3: M = 12 branching verifyTree + commitTreeKv + follow-up decode ---
    const tree_base: u32 = dec_pos + 1; // 171
    // Branching speculative tree (depth 0..4):
    //   0(d0) -> 1(d1) -> 3(d2) -> 6(d3) -> 10(d4)
    //                  -> 4(d2) -> 7(d3)
    //         -> 2(d1) -> 5(d2) -> 8(d3) -> 11(d4)
    //                           -> 9(d3)
    const tree_depths = [tree_m]u32{ 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4 };
    const tree_masks = [tree_m]u64{
        0b0000_0000_0001, // 0
        0b0000_0000_0011, // 1: {0,1}
        0b0000_0000_0101, // 2: {0,2}
        0b0000_0000_1011, // 3: {0,1,3}
        0b0000_0001_0011, // 4: {0,1,4}
        0b0000_0010_0101, // 5: {0,2,5}
        0b0000_0100_1011, // 6: {0,1,3,6}
        0b0000_1001_0011, // 7: {0,1,4,7}
        0b0001_0010_0101, // 8: {0,2,5,8}
        0b0010_0010_0101, // 9: {0,2,5,9}
        0b0100_0100_1011, // 10: {0,1,3,6,10}
        0b1001_0010_0101, // 11: {0,2,5,8,11}
    };
    const tree_q_rot = try gpa.alloc(f32, tree_m * q_row_elems);
    defer gpa.free(tree_q_rot);
    const tree_k_rot = try gpa.alloc(f32, tree_m * k_row_elems);
    defer gpa.free(tree_k_rot);
    const tree_v_scaled = try gpa.alloc(f32, tree_m * v_row_elems);
    defer gpa.free(tree_v_scaled);
    for (0..tree_m) |ni| {
        const np = tree_base + tree_depths[ni];
        pos_h[ni] = np;
        masks_h[ni] = tree_masks[ni];
        for (0..q_row_elems) |i| q_h[ni * q_row_elems + i] = @floatCast(synthF32(12000 + ni * 41 + i, 0.25));
        for (0..k_row_elems) |i| k_h[ni * k_row_elems + i] = @floatCast(synthF32(13000 + ni * 43 + i, 0.25));
        for (0..v_row_elems) |i| v_h[ni * v_row_elems + i] = @floatCast(synthF32(14000 + ni * 47 + i, 0.25));
        for (0..num_q_heads) |qh| {
            const qs = tree_q_rot[ni * q_row_elems + qh * k_head_dim ..][0..k_head_dim];
            for (0..k_head_dim) |d| qs[d] = @floatCast(q_h[ni * q_row_elems + qh * k_head_dim + d]);
            applyPartialRopeF32(qs, np, rotary_dim, rope_theta);
        }
        for (0..num_kv_heads) |kvh| {
            const ks = tree_k_rot[ni * k_row_elems + kvh * k_head_dim ..][0..k_head_dim];
            for (0..k_head_dim) |d| ks[d] = @floatCast(k_h[ni * k_row_elems + kvh * k_head_dim + d]);
            applyPartialRopeF32(ks, np, rotary_dim, rope_theta);
            const vs = tree_v_scaled[ni * v_row_elems + kvh * v_head_dim ..][0..v_head_dim];
            for (0..v_head_dim) |d| vs[d] = @floatCast(@as(f16, @floatCast(@as(f32, @floatCast(v_h[ni * v_row_elems + kvh * v_head_dim + d])) * v_scale)));
        }
    }
    const accepted_branch = [_]u16{ 0, 2, 5, 8, 11 };
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG26QkNormRopeKv(&frame, .{
            .q = q_bind,
            .k = k_bind,
            .v = v_bind,
            .positions_1d = pos_bind,
            .k_out = k_out_bind,
            .v_out = v_out_bind,
            .num_tokens = tree_m,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .rotary_dim = rotary_dim,
            .start_pos = tree_base,
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
            .kv_only = false,
            .write_to_tiles = false,
            .rope_theta = rope_theta,
            .v_scale = v_scale,
        }, false);
        try pipes.encodeG26SwaAttn(&frame, .{
            .q = q_bind,
            .kv_tiles = ring_tbl_bind,
            .k_extra = k_out_bind,
            .v_extra = v_out_bind,
            .ancestor_masks = masks_bind,
            .node_positions = pos_bind,
            .sink_bias = sink_bind,
            .sink_dtype = .bf16,
            .out = o_bind,
            .num_q = tree_m,
            .q_start_pos = tree_base,
            .kv_len = tree_base,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .window_size = window_size,
            .ring_capacity = swa_ring.ring_capacity,
            .chunk_tokens = swa_ring.chunk_tokens,
            .is_prefill = false,
        }, true);
        try pipes.encodeG26KvCommit(&frame, .{
            .k_src = k_out_bind,
            .v_src = v_out_bind,
            .kv_tiles = ring_tbl_bind,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .start_pos = tree_base,
            .accepted_nodes = &accepted_branch,
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    var max_err_tree: f32 = 0.0;
    for (0..tree_m) |ni| {
        evalSwaCpuRef(
            tree_q_rot[ni * q_row_elems ..][0..q_row_elems],
            ref_k,
            ref_v,
            tree_k_rot,
            tree_v_scaled,
            pos_h[ni],
            tree_base,
            tree_masks[ni],
            tree_m,
            &sink_f32,
            ref_out,
        );
        for (0..o_row_elems) |i| {
            const diff = @abs(@as(f32, @floatCast(o_h[ni * o_row_elems + i])) - ref_out[i]);
            if (diff > max_err_tree) max_err_tree = diff;
        }
    }
    try std.testing.expect(max_err_tree < 4e-3);

    // Commit accepted branch in CPU ref, run one follow-up M=1 decode at pos = 176.
    for (accepted_branch, 0..) |nid, step| {
        const p = tree_base + @as(u32, @intCast(step));
        @memcpy(ref_k[p * k_row_elems ..][0..k_row_elems], tree_k_rot[@as(usize, nid) * k_row_elems ..][0..k_row_elems]);
        @memcpy(ref_v[p * v_row_elems ..][0..v_row_elems], tree_v_scaled[@as(usize, nid) * v_row_elems ..][0..v_row_elems]);
    }
    const post_pos: u32 = tree_base + @as(u32, accepted_branch.len); // 176
    for (0..q_row_elems) |i| q_h[i] = @floatCast(synthF32(19000 + i, 0.25));
    for (0..k_row_elems) |i| k_h[i] = @floatCast(synthF32(19500 + i, 0.25));
    for (0..v_row_elems) |i| v_h[i] = @floatCast(synthF32(19800 + i, 0.25));
    for (0..num_q_heads) |qh| {
        const qs = q_rot_dec[qh * k_head_dim ..][0..k_head_dim];
        for (0..k_head_dim) |d| qs[d] = @floatCast(q_h[qh * k_head_dim + d]);
        applyPartialRopeF32(qs, post_pos, rotary_dim, rope_theta);
    }
    for (0..num_kv_heads) |kvh| {
        const ks = ref_k[post_pos * k_row_elems + kvh * k_head_dim ..][0..k_head_dim];
        for (0..k_head_dim) |d| ks[d] = @floatCast(k_h[kvh * k_head_dim + d]);
        applyPartialRopeF32(ks, post_pos, rotary_dim, rope_theta);
        const vs = ref_v[post_pos * v_row_elems + kvh * v_head_dim ..][0..v_head_dim];
        for (0..v_head_dim) |d| vs[d] = @floatCast(@as(f16, @floatCast(@as(f32, @floatCast(v_h[kvh * v_head_dim + d])) * v_scale)));
    }
    masks_h[0] = 1;
    pos_h[0] = post_pos;
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG26QkNormRopeKv(&frame, .{
            .q = q_bind,
            .k = k_bind,
            .v = v_bind,
            .positions_1d = pos_bind,
            .k_out = k_out_bind,
            .v_out = v_out_bind,
            .num_tokens = 1,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .rotary_dim = rotary_dim,
            .start_pos = post_pos,
            .chunk_tokens = swa_ring.chunk_tokens,
            .ring_capacity = swa_ring.ring_capacity,
            .kv_only = false,
            .write_to_tiles = false,
            .rope_theta = rope_theta,
            .v_scale = v_scale,
        }, false);
        try pipes.encodeG26SwaAttn(&frame, .{
            .q = q_bind,
            .kv_tiles = ring_tbl_bind,
            .k_extra = k_out_bind,
            .v_extra = v_out_bind,
            .ancestor_masks = masks_bind,
            .node_positions = pos_bind,
            .sink_bias = sink_bind,
            .sink_dtype = .bf16,
            .out = o_bind,
            .num_q = 1,
            .q_start_pos = post_pos,
            .kv_len = post_pos,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .k_head_dim = k_head_dim,
            .v_head_dim = v_head_dim,
            .window_size = window_size,
            .ring_capacity = swa_ring.ring_capacity,
            .chunk_tokens = swa_ring.chunk_tokens,
            .is_prefill = false,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }
    evalSwaCpuRef(q_rot_dec, ref_k, ref_v, null, null, post_pos, post_pos + 1, 0, 0, &sink_f32, ref_out);
    for (0..o_row_elems) |i| {
        try std.testing.expectApproxEqAbs(ref_out[i], @as(f32, @floatCast(o_h[i])), 4e-3);
    }
}

test "M9.1 G26: ami_moe_route_topk_sigmoid (E=256 and E=384, top-8, e_score_correction_bias, scaling) vs CPU fp32 ref" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const cases = [_]struct { e: u32, m: u32, top_k: u32, scale: f32 }{
        .{ .e = 256, .m = 4, .top_k = 8, .scale = 1.0 },
        .{ .e = 384, .m = 2, .top_k = 8, .scale = 2.5 },
    };

    const arena_cap = try page.alignUp(256 * 1024);
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    const logits_raw = try arena.allocAligned(4 * 384 * @sizeOf(f32), 64);
    const bias_raw = try arena.allocAligned(384 * @sizeOf(f32), 64);
    const hdr_raw = try arena.allocAligned(64, 64);
    const uniq_raw = try arena.allocAligned(384 * @sizeOf(u32), 64);
    const offs_raw = try arena.allocAligned(385 * @sizeOf(u32), 64);
    const toks_raw = try arena.allocAligned(4 * 8 * @sizeOf(u32), 64);
    const wts_raw = try arena.allocAligned(4 * 8 * @sizeOf(f32), 64);
    const slots_raw = try arena.allocAligned(4 * 8 * @sizeOf(u32), 64);
    const sk_e_raw = try arena.allocAligned(4 * 8 * @sizeOf(u32), 64);
    const sk_w_raw = try arena.allocAligned(4 * 8 * @sizeOf(f32), 64);

    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw[0 .. 4 * 384 * 4]));
    const bias_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bias_raw[0 .. 384 * 4]));
    const hdr_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, hdr_raw[0..64]));
    const uniq_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, uniq_raw[0 .. 384 * 4]));
    const offs_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, offs_raw[0 .. 385 * 4]));
    const toks_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, toks_raw[0 .. 32 * 4]));
    const wts_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, wts_raw[0 .. 32 * 4]));
    const sk_e_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, sk_e_raw[0 .. 32 * 4]));
    const sk_w_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, sk_w_raw[0 .. 32 * 4]));

    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{buf});

    for (cases, 0..) |tc, ci| {
        for (0..tc.e) |ei| {
            bias_f32[ei] = synthF32(500 + ci * 1000 + ei, 0.35);
        }
        for (0..tc.m) |ti| {
            for (0..tc.e) |ei| {
                logits_f32[ti * tc.e + ei] = synthF32(2000 + ci * 5000 + ti * 401 + ei, 1.8);
            }
            // Cold-mask one expert with -inf to verify masking safety.
            logits_f32[ti * tc.e + (ti * 7 + 3) % tc.e] = -std.math.inf(f32);
        }

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG26MoeRouteSigmoid(&frame, .{
            .logits = .{ .buffer = buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .logits_f16 = false,
            .bias = .{ .buffer = buf, .offset = arena.offsetOf(bias_raw.ptr) },
            .bias_dtype = .f32,
            .sched_hdr = .{ .buffer = buf, .offset = arena.offsetOf(hdr_raw.ptr) },
            .unique_experts = .{ .buffer = buf, .offset = arena.offsetOf(uniq_raw.ptr) },
            .expert_token_offsets = .{ .buffer = buf, .offset = arena.offsetOf(offs_raw.ptr) },
            .token_indices = .{ .buffer = buf, .offset = arena.offsetOf(toks_raw.ptr) },
            .routing_weights = .{ .buffer = buf, .offset = arena.offsetOf(wts_raw.ptr) },
            .slot_k_indices = .{ .buffer = buf, .offset = arena.offsetOf(slots_raw.ptr) },
            .scratch_topk_experts = .{ .buffer = buf, .offset = arena.offsetOf(sk_e_raw.ptr) },
            .scratch_topk_weights = .{ .buffer = buf, .offset = arena.offsetOf(sk_w_raw.ptr) },
            .m = tc.m,
            .num_experts = tc.e,
            .top_k = tc.top_k,
            .norm_topk_prob = true,
            .routed_scaling_factor = tc.scale,
            .gate_up_groups_x_q = 8,
            .gate_up_groups_x_f = 4,
            .down_groups_x = 8,
            .down_groups_x_q8 = 16,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        // Compare against exact CPU FP32 MiMoV2MoEGate reference.
        for (0..tc.m) |ti| {
            var chosen_e: [8]u32 = undefined;
            var chosen_w: [8]f32 = undefined;
            var used: [384]bool = @splat(false);
            var sum_w: f32 = 0.0;
            for (0..tc.top_k) |ki| {
                var best_e: u32 = 0;
                var best_choice: f32 = -std.math.inf(f32);
                var best_unbiased: f32 = 0.0;
                for (0..tc.e) |ei| {
                    if (used[ei]) continue;
                    const lg = logits_f32[ti * tc.e + ei];
                    const sig_v: f32 = if (std.math.isNegativeInf(lg)) 0.0 else sigmoidF32(lg);
                    const ch: f32 = if (std.math.isNegativeInf(lg)) -std.math.inf(f32) else sig_v + bias_f32[ei];
                    if (ch > best_choice) {
                        best_choice = ch;
                        best_e = @intCast(ei);
                        best_unbiased = sig_v;
                    }
                }
                used[best_e] = true;
                chosen_e[ki] = best_e;
                chosen_w[ki] = best_unbiased;
                sum_w += best_unbiased;
            }
            for (0..tc.top_k) |ki| {
                chosen_w[ki] = (chosen_w[ki] / (sum_w + 1e-20)) * tc.scale;
                try std.testing.expectEqual(chosen_e[ki], sk_e_u32[ti * tc.top_k + ki]);
                try std.testing.expectApproxEqAbs(chosen_w[ki], sk_w_f32[ti * tc.top_k + ki], 1e-5);
            }
        }
        // Verify CSR schedule consistency (`num_unique_experts`, total assignments == m * top_k).
        const num_uniq = hdr_u32[0];
        try std.testing.expect(num_uniq >= tc.top_k and num_uniq <= tc.m * tc.top_k);
        try std.testing.expectEqual(tc.m * tc.top_k, offs_u32[num_uniq]);
        for (0..num_uniq) |ui| {
            const eid = uniq_u32[ui];
            const s0 = offs_u32[ui];
            const s1 = offs_u32[ui + 1];
            try std.testing.expect(s1 > s0);
            for (s0..s1) |si| {
                const tok = toks_u32[si];
                try std.testing.expect(tok < tc.m);
                var found = false;
                for (0..tc.top_k) |ki| {
                    if (sk_e_u32[tok * tc.top_k + ki] == eid) {
                        try std.testing.expectApproxEqAbs(sk_w_f32[tok * tc.top_k + ki], wts_f32[si], 1e-5);
                        found = true;
                        break;
                    }
                }
                try std.testing.expect(found);
            }
        }
    }
}

test "G27 single-kernel 20-step hc_mult=4 mHC Sinkhorn GPU == CPU parity and doubly-stochastic check" {
    const allocator = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();

    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    const pipes = try HybridPipelines.init(device, lib);
    defer pipes.close();

    const m: u32 = 3;
    const h: u32 = 64;
    const total_dim: usize = 4 * h;
    const sinkhorn_iters: u32 = 20;
    const rms_eps: f32 = 1e-5;
    const hc_eps: f32 = 1e-6;

    const arena_cap = try page.alignUp(2 * 1024 * 1024);
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    const streams_raw = try arena.allocAligned(@as(usize, m) * total_dim * 2, 64);
    const fn_raw = try arena.allocAligned(24 * total_dim * 2, 64);
    const base_raw = try arena.allocAligned(24 * 4, 64);
    const scale_raw = try arena.allocAligned(3 * 4, 64);
    const norm_raw = try arena.allocAligned(@as(usize, h) * 2, 64);
    const branch_raw = try arena.allocAligned(@as(usize, m) * h * 2, 64);
    const sub_raw = try arena.allocAligned(@as(usize, m) * h * 2, 64);
    const coeffs_raw = try arena.allocAligned(@as(usize, m) * 20 * 4, 64);

    const streams_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, streams_raw[0 .. @as(usize, m) * total_dim * 2]));
    const fn_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, fn_raw[0 .. 24 * total_dim * 2]));
    const base_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, base_raw[0 .. 24 * 4]));
    const scale_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, scale_raw[0 .. 3 * 4]));
    const norm_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, norm_raw[0 .. @as(usize, h) * 2]));
    const branch_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, branch_raw[0 .. @as(usize, m) * h * 2]));
    const sub_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, sub_raw[0 .. @as(usize, m) * h * 2]));
    const coeffs_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, coeffs_raw[0 .. @as(usize, m) * 20 * 4]));

    var prng = std.Random.DefaultPrng.init(0x9201_2026);
    const rng = prng.random();
    const streams_orig = try allocator.alloc(f32, @as(usize, m) * total_dim);
    defer allocator.free(streams_orig);
    for (0..streams_f16.len) |i| {
        const v: f16 = @floatCast((rng.float(f32) - 0.5) * 2.0);
        streams_f16[i] = v;
        streams_orig[i] = @floatCast(v);
    }
    for (0..fn_f16.len) |i| fn_f16[i] = @floatCast((rng.float(f32) - 0.5) * 0.25);
    for (0..24) |i| base_f32[i] = (rng.float(f32) - 0.5) * 0.6;
    scale_f32[0] = 0.35;
    scale_f32[1] = 0.45;
    scale_f32[2] = 0.55;
    for (0..h) |i| norm_f16[i] = @floatCast(0.8 + rng.float(f32) * 0.4);
    for (0..sub_f16.len) |i| sub_f16[i] = @floatCast((rng.float(f32) - 0.5) * 1.5);

    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{buf});
    defer compute.removeResident(&.{buf}) catch |err| {
        std.debug.panic("removeResident failed: {}", .{err});
    };

    const streams_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena.offsetOf(streams_raw.ptr) };
    const fn_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena.offsetOf(fn_raw.ptr) };
    const base_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena.offsetOf(base_raw.ptr) };
    const scale_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena.offsetOf(scale_raw.ptr) };
    const norm_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena.offsetOf(norm_raw.ptr) };
    const branch_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena.offsetOf(branch_raw.ptr) };
    const sub_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena.offsetOf(sub_raw.ptr) };
    const coeffs_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena.offsetOf(coeffs_raw.ptr) };

    var frame = try compute.beginFrame(5000);
    try pipes.encodeG27MhcPreSinkhorn(&frame, .{
        .streams = streams_b,
        .fn_w = fn_b,
        .base_w = base_b,
        .scale_w = scale_b,
        .norm_w = norm_b,
        .branch_out = branch_b,
        .coeffs_out = coeffs_b,
        .m = m,
        .hidden_size = h,
        .sinkhorn_iters = sinkhorn_iters,
        .fn_dtype = .f16,
        .norm_dtype = .f16,
        .rms_eps = rms_eps,
        .hc_eps = hc_eps,
    }, false);
    try pipes.encodeG27MhcPostMix(&frame, .{
        .streams = streams_b,
        .sublayer_out = sub_b,
        .coeffs = coeffs_b,
        .m = m,
        .hidden_size = h,
    }, true);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    // CPU FP32 reference matching `Glm5NextTextHyperConnection` verbatim.
    var dot_branch: f64 = 0.0;
    var na_branch: f64 = 0.0;
    var nb_branch: f64 = 0.0;
    var dot_streams: f64 = 0.0;
    var na_streams: f64 = 0.0;
    var nb_streams: f64 = 0.0;
    for (0..m) |tok| {
        const s_tok = streams_orig[tok * total_dim ..][0..total_dim];
        var sq: f32 = 0.0;
        for (s_tok) |v| sq += v * v;
        const inv_rms = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(total_dim)) + rms_eps);

        var raw: [24]f32 = undefined;
        for (0..24) |r| {
            var acc: f32 = 0.0;
            for (0..total_dim) |j| {
                acc += (s_tok[j] * inv_rms) * @as(f32, @floatCast(fn_f16[r * total_dim + j]));
            }
            raw[r] = acc;
        }
        var pre: [4]f32 = undefined;
        var post: [4]f32 = undefined;
        var comb: [4][4]f32 = undefined;
        for (0..4) |i| {
            pre[i] = sigmoidF32(raw[i] * scale_f32[0] + base_f32[i]) + hc_eps;
            post[i] = 2.0 * sigmoidF32(raw[4 + i] * scale_f32[1] + base_f32[4 + i]);
        }
        for (0..4) |r| {
            var row_z: [4]f32 = undefined;
            var mx: f32 = -std.math.inf(f32);
            for (0..4) |c| {
                row_z[c] = raw[8 + r * 4 + c] * scale_f32[2] + base_f32[8 + r * 4 + c];
                mx = @max(mx, row_z[c]);
            }
            var es: f32 = 0.0;
            for (0..4) |c| {
                row_z[c] = @exp(row_z[c] - mx);
                es += row_z[c];
            }
            for (0..4) |c| comb[r][c] = row_z[c] / es + hc_eps;
        }
        for (0..4) |c| {
            var cs: f32 = hc_eps;
            for (0..4) |r| cs += comb[r][c];
            for (0..4) |r| comb[r][c] /= cs;
        }
        for (1..sinkhorn_iters) |_| {
            for (0..4) |r| {
                var rs: f32 = hc_eps;
                for (0..4) |c| rs += comb[r][c];
                for (0..4) |c| comb[r][c] /= rs;
            }
            for (0..4) |c| {
                var cs: f32 = hc_eps;
                for (0..4) |r| cs += comb[r][c];
                for (0..4) |r| comb[r][c] /= cs;
            }
        }

        // Verify doubly-stochastic row/col sums == 1.0 on GPU output `coeffs_f32`.
        const gpu_c = coeffs_f32[tok * 20 ..][0..20];
        for (0..4) |r| {
            var rs: f32 = 0.0;
            var cs: f32 = 0.0;
            for (0..4) |c| {
                rs += gpu_c[4 + r * 4 + c];
                cs += gpu_c[4 + c * 4 + r];
                try std.testing.expectApproxEqAbs(comb[r][c], gpu_c[4 + r * 4 + c], 1e-4);
            }
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), rs, 1e-4);
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), cs, 1e-4);
        }

        // Check pre-mix + norm output.
        var mix_buf: [64]f32 = undefined;
        var mix_sq: f32 = 0.0;
        for (0..h) |d| {
            var mv: f32 = 0.0;
            for (0..4) |s| mv += pre[s] * s_tok[s * h + d];
            mix_buf[d] = mv;
            mix_sq += mv * mv;
        }
        const inv_sub = 1.0 / @sqrt(mix_sq / @as(f32, @floatFromInt(h)) + rms_eps);
        for (0..h) |d| {
            const ref = (mix_buf[d] * inv_sub) * @as(f32, @floatCast(norm_f16[d]));
            const got: f32 = @floatCast(branch_f16[tok * h + d]);
            dot_branch += @as(f64, ref) * @as(f64, got);
            na_branch += @as(f64, ref) * @as(f64, ref);
            nb_branch += @as(f64, got) * @as(f64, got);
        }

        // Check post-mix: `post[j] * sub_out[d] + sum_i comb[i][j] * streams[i][d]`.
        for (0..4) |j| {
            for (0..h) |d| {
                var acc: f32 = post[j] * @as(f32, @floatCast(sub_f16[tok * h + d]));
                for (0..4) |i| acc += comb[i][j] * s_tok[i * h + d];
                const got: f32 = @floatCast(streams_f16[tok * total_dim + j * h + d]);
                dot_streams += @as(f64, acc) * @as(f64, got);
                na_streams += @as(f64, acc) * @as(f64, acc);
                nb_streams += @as(f64, got) * @as(f64, got);
            }
        }
    }
    const cos_branch = dot_branch / @sqrt(na_branch * nb_branch);
    const cos_streams = dot_streams / @sqrt(na_streams * nb_streams);
    try std.testing.expect(cos_branch >= 0.99999);
    try std.testing.expect(cos_streams >= 0.99999);
}

test "G27 KDA conv1d + bounded/softplus forget gate + decode == prefill == verifyTree + commitTreeKv" {
    const allocator = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();

    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    const pipes = try HybridPipelines.init(device, lib);
    defer pipes.close();

    const seq_len: u32 = 6;
    const num_heads: u32 = 2;
    const head_dim: u32 = 64;
    const qkv_dim: u32 = num_heads * head_dim;
    const conv_k: u32 = 4;
    const hist: u32 = conv_k - 1;
    const lower_bound: f32 = -5.0;
    const rms_eps: f32 = 1e-5;
    const l2_eps: f32 = 1e-6;

    const arena_cap = try page.alignUp(4 * 1024 * 1024);
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    const q_raw = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 2, 64);
    const k_raw = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 2, 64);
    const v_raw = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 2, 64);
    const q_conv_out = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 2, 64);
    const k_conv_out = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 2, 64);
    const v_conv_out = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 2, 64);
    const f_raw = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 2, 64);
    const b_raw = try arena.allocAligned(@as(usize, seq_len) * num_heads * 2, 64);
    const z_raw = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 2, 64);
    const qw_raw = try arena.allocAligned(@as(usize, qkv_dim) * conv_k * 4, 64);
    const kw_raw = try arena.allocAligned(@as(usize, qkv_dim) * conv_k * 4, 64);
    const vw_raw = try arena.allocAligned(@as(usize, qkv_dim) * conv_k * 4, 64);
    const a_log_raw = try arena.allocAligned(@as(usize, num_heads) * 4, 64);
    const dt_raw = try arena.allocAligned(@as(usize, qkv_dim) * 4, 64);
    const norm_raw = try arena.allocAligned(@as(usize, head_dim) * 2, 64);
    const g_raw = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 4, 64);
    const beta_raw = try arena.allocAligned(@as(usize, seq_len) * num_heads * 4, 64);
    const conv_state_a = try arena.allocAligned(@as(usize, hist) * 3 * qkv_dim * 2, 64);
    const conv_state_b = try arena.allocAligned(@as(usize, hist) * 3 * qkv_dim * 2, 64);
    const state_a = try arena.allocAligned(@as(usize, num_heads) * head_dim * head_dim * 4, 64);
    const state_b = try arena.allocAligned(@as(usize, num_heads) * head_dim * head_dim * 4, 64);
    const state_c = try arena.allocAligned(@as(usize, num_heads) * head_dim * head_dim * 4, 64);
    const out_prefill = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 2, 64);
    const out_decode = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 2, 64);
    const out_tree = try arena.allocAligned(@as(usize, seq_len) * qkv_dim * 2, 64);
    const masks_raw = try arena.allocAligned(@as(usize, seq_len) * 8, 64);

    const q_in_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0 .. @as(usize, seq_len) * qkv_dim * 2]));
    const k_in_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw[0 .. @as(usize, seq_len) * qkv_dim * 2]));
    const v_in_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw[0 .. @as(usize, seq_len) * qkv_dim * 2]));
    const f_in_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, f_raw[0 .. @as(usize, seq_len) * qkv_dim * 2]));
    const b_in_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, b_raw[0 .. @as(usize, seq_len) * num_heads * 2]));
    const z_in_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, z_raw[0 .. @as(usize, seq_len) * qkv_dim * 2]));
    const qw_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, qw_raw[0 .. @as(usize, qkv_dim) * conv_k * 4]));
    const kw_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, kw_raw[0 .. @as(usize, qkv_dim) * conv_k * 4]));
    const vw_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, vw_raw[0 .. @as(usize, qkv_dim) * conv_k * 4]));
    const a_log_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, a_log_raw[0 .. @as(usize, num_heads) * 4]));
    const dt_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, dt_raw[0 .. @as(usize, qkv_dim) * 4]));
    const norm_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, norm_raw[0 .. @as(usize, head_dim) * 2]));
    const masks_u64: []u64 = @alignCast(std.mem.bytesAsSlice(u64, masks_raw[0 .. @as(usize, seq_len) * 8]));

    var prng = std.Random.DefaultPrng.init(0x9202_2026);
    const rng = prng.random();
    for (0..q_in_h.len) |i| {
        q_in_h[i] = @floatCast((rng.float(f32) - 0.5) * 1.2);
        k_in_h[i] = @floatCast((rng.float(f32) - 0.5) * 1.2);
        v_in_h[i] = @floatCast((rng.float(f32) - 0.5) * 1.2);
        f_in_h[i] = @floatCast((rng.float(f32) - 0.5) * 2.0);
        z_in_h[i] = @floatCast((rng.float(f32) - 0.5) * 2.0);
    }
    for (0..b_in_h.len) |i| b_in_h[i] = @floatCast((rng.float(f32) - 0.5) * 2.0);
    for (0..qw_f32.len) |i| {
        qw_f32[i] = (rng.float(f32) - 0.5) * 0.5;
        kw_f32[i] = (rng.float(f32) - 0.5) * 0.5;
        vw_f32[i] = (rng.float(f32) - 0.5) * 0.5;
    }
    for (0..num_heads) |i| a_log_f32[i] = -0.3 + @as(f32, @floatFromInt(i)) * 0.4;
    for (0..qkv_dim) |i| dt_f32[i] = (rng.float(f32) - 0.5) * 0.8;
    for (0..head_dim) |i| norm_h[i] = @floatCast(0.8 + rng.float(f32) * 0.4);
    for (0..seq_len) |i| masks_u64[i] = (@as(u64, 1) << @intCast(i + 1)) - 1;
    @memset(conv_state_a, 0);
    @memset(conv_state_b, 0);
    @memset(state_a, 0);
    @memset(state_b, 0);
    @memset(state_c, 0);

    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{buf});
    defer compute.removeResident(&.{buf}) catch |err| {
        std.debug.panic("removeResident failed: {}", .{err});
    };

    const bind = struct {
        fn of(b: metal.Buffer, a: *const arena_mod.WiredArena, ptr: [*]u8) metal.BufferBinding {
            return .{ .buffer = b, .offset = a.offsetOf(ptr) };
        }
    }.of;

    // 1) Prefill path (all 6 tokens in one pass).
    {
        var frame = try compute.beginFrame(5000);
        try pipes.encodeKdaConv1dSilu(&frame, .{
            .q_in = bind(buf, &arena, q_raw.ptr),
            .k_in = bind(buf, &arena, k_raw.ptr),
            .v_in = bind(buf, &arena, v_raw.ptr),
            .q_conv_w = bind(buf, &arena, qw_raw.ptr),
            .k_conv_w = bind(buf, &arena, kw_raw.ptr),
            .v_conv_w = bind(buf, &arena, vw_raw.ptr),
            .conv_state_in = bind(buf, &arena, conv_state_a.ptr),
            .conv_state_out = bind(buf, &arena, conv_state_a.ptr),
            .q_out = bind(buf, &arena, q_conv_out.ptr),
            .k_out = bind(buf, &arena, k_conv_out.ptr),
            .v_out = bind(buf, &arena, v_conv_out.ptr),
            .m = seq_len,
            .qkv_dim = qkv_dim,
            .conv_k = conv_k,
            .write_state = true,
            .is_tree = false,
            .w_dtype = .f32,
        }, false);
        try pipes.encodeKdaComputeGates(&frame, .{
            .f_raw = bind(buf, &arena, f_raw.ptr),
            .b_raw = bind(buf, &arena, b_raw.ptr),
            .a_log = bind(buf, &arena, a_log_raw.ptr),
            .dt_bias = bind(buf, &arena, dt_raw.ptr),
            .g_out = bind(buf, &arena, g_raw.ptr),
            .beta_out = bind(buf, &arena, beta_raw.ptr),
            .m = seq_len,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .lower_bound = lower_bound,
        }, false);
        try pipes.encodeKdaPrefill(&frame, .{
            .q = bind(buf, &arena, q_conv_out.ptr),
            .k = bind(buf, &arena, k_conv_out.ptr),
            .v = bind(buf, &arena, v_conv_out.ptr),
            .g = bind(buf, &arena, g_raw.ptr),
            .beta = bind(buf, &arena, beta_raw.ptr),
            .z = bind(buf, &arena, z_raw.ptr),
            .norm_w = bind(buf, &arena, norm_raw.ptr),
            .norm_dtype = .f16,
            .state_in = bind(buf, &arena, state_a.ptr),
            .state_out = bind(buf, &arena, state_a.ptr),
            .out = bind(buf, &arena, out_prefill.ptr),
            .seq_len = seq_len,
            .num_heads = num_heads,
            .head_k_dim = head_dim,
            .head_v_dim = head_dim,
            .l2_eps = l2_eps,
            .rms_eps = rms_eps,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    // 2) Sequential decode path (step-by-step `m = 1`).
    {
        var frame = try compute.beginFrame(5000);
        for (0..seq_len) |t| {
            const qkv_off = @as(u64, t) * qkv_dim * 2;
            const g_off = @as(u64, t) * qkv_dim * 4;
            const b_off = @as(u64, t) * num_heads * 4;
            try pipes.encodeKdaConv1dSilu(&frame, .{
                .q_in = .{ .buffer = buf, .offset = arena.offsetOf(q_raw.ptr) + qkv_off },
                .k_in = .{ .buffer = buf, .offset = arena.offsetOf(k_raw.ptr) + qkv_off },
                .v_in = .{ .buffer = buf, .offset = arena.offsetOf(v_raw.ptr) + qkv_off },
                .q_conv_w = bind(buf, &arena, qw_raw.ptr),
                .k_conv_w = bind(buf, &arena, kw_raw.ptr),
                .v_conv_w = bind(buf, &arena, vw_raw.ptr),
                .conv_state_in = bind(buf, &arena, conv_state_b.ptr),
                .conv_state_out = bind(buf, &arena, conv_state_b.ptr),
                .q_out = .{ .buffer = buf, .offset = arena.offsetOf(q_conv_out.ptr) + qkv_off },
                .k_out = .{ .buffer = buf, .offset = arena.offsetOf(k_conv_out.ptr) + qkv_off },
                .v_out = .{ .buffer = buf, .offset = arena.offsetOf(v_conv_out.ptr) + qkv_off },
                .m = 1,
                .qkv_dim = qkv_dim,
                .conv_k = conv_k,
                .write_state = true,
                .is_tree = false,
                .w_dtype = .f32,
            }, t > 0);
            try pipes.encodeKdaDecode(&frame, .{
                .q = .{ .buffer = buf, .offset = arena.offsetOf(q_conv_out.ptr) + qkv_off },
                .k = .{ .buffer = buf, .offset = arena.offsetOf(k_conv_out.ptr) + qkv_off },
                .v = .{ .buffer = buf, .offset = arena.offsetOf(v_conv_out.ptr) + qkv_off },
                .g = .{ .buffer = buf, .offset = arena.offsetOf(g_raw.ptr) + g_off },
                .beta = .{ .buffer = buf, .offset = arena.offsetOf(beta_raw.ptr) + b_off },
                .z = .{ .buffer = buf, .offset = arena.offsetOf(z_raw.ptr) + qkv_off },
                .norm_w = bind(buf, &arena, norm_raw.ptr),
                .norm_dtype = .f16,
                .state_in = bind(buf, &arena, state_b.ptr),
                .state_out = bind(buf, &arena, state_b.ptr),
                .out = .{ .buffer = buf, .offset = arena.offsetOf(out_decode.ptr) + qkv_off },
                .num_heads = num_heads,
                .head_k_dim = head_dim,
                .head_v_dim = head_dim,
                .l2_eps = l2_eps,
                .rms_eps = rms_eps,
            }, true);
        }
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    // 3) Tree-verify + commitTreeKv path over chain `[0,1,2,3,4,5]` from zero state.
    @memset(conv_state_b, 0);
    {
        var frame = try compute.beginFrame(5000);
        try pipes.encodeKdaConv1dSilu(&frame, .{
            .q_in = bind(buf, &arena, q_raw.ptr),
            .k_in = bind(buf, &arena, k_raw.ptr),
            .v_in = bind(buf, &arena, v_raw.ptr),
            .q_conv_w = bind(buf, &arena, qw_raw.ptr),
            .k_conv_w = bind(buf, &arena, kw_raw.ptr),
            .v_conv_w = bind(buf, &arena, vw_raw.ptr),
            .conv_state_in = bind(buf, &arena, conv_state_b.ptr),
            .conv_state_out = bind(buf, &arena, conv_state_b.ptr),
            .ancestor_masks = bind(buf, &arena, masks_raw.ptr),
            .q_out = bind(buf, &arena, q_conv_out.ptr),
            .k_out = bind(buf, &arena, k_conv_out.ptr),
            .v_out = bind(buf, &arena, v_conv_out.ptr),
            .m = seq_len,
            .qkv_dim = qkv_dim,
            .conv_k = conv_k,
            .write_state = false,
            .is_tree = true,
            .w_dtype = .f32,
        }, false);
        try pipes.encodeKdaTreeVerify(&frame, .{
            .q = bind(buf, &arena, q_conv_out.ptr),
            .k = bind(buf, &arena, k_conv_out.ptr),
            .v = bind(buf, &arena, v_conv_out.ptr),
            .g = bind(buf, &arena, g_raw.ptr),
            .beta = bind(buf, &arena, beta_raw.ptr),
            .z = bind(buf, &arena, z_raw.ptr),
            .norm_w = bind(buf, &arena, norm_raw.ptr),
            .norm_dtype = .f16,
            .ancestor_masks = bind(buf, &arena, masks_raw.ptr),
            .state_in = bind(buf, &arena, state_c.ptr),
            .out = bind(buf, &arena, out_tree.ptr),
            .num_nodes = seq_len,
            .num_heads = num_heads,
            .head_k_dim = head_dim,
            .head_v_dim = head_dim,
            .l2_eps = l2_eps,
            .rms_eps = rms_eps,
        }, true);
        const accepted = [_]u16{ 0, 1, 2, 3, 4, 5 };
        try pipes.encodeKdaCommitTree(&frame, .{
            .k_tree = bind(buf, &arena, k_conv_out.ptr),
            .v_tree = bind(buf, &arena, v_conv_out.ptr),
            .g_tree = bind(buf, &arena, g_raw.ptr),
            .beta_tree = bind(buf, &arena, beta_raw.ptr),
            .accepted_nodes = &accepted,
            .state_in = bind(buf, &arena, state_c.ptr),
            .state_out = bind(buf, &arena, state_c.ptr),
            .num_heads = num_heads,
            .head_k_dim = head_dim,
            .head_v_dim = head_dim,
            .l2_eps = l2_eps,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    // Verify decode == prefill == verifyTree outputs and final states match!
    const pre_h: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_prefill[0 .. @as(usize, seq_len) * qkv_dim * 2]));
    const dec_h: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_decode[0 .. @as(usize, seq_len) * qkv_dim * 2]));
    const tr_h: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_tree[0 .. @as(usize, seq_len) * qkv_dim * 2]));
    const sa_f32: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, state_a[0 .. @as(usize, num_heads) * head_dim * head_dim * 4]));
    const sb_f32: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, state_b[0 .. @as(usize, num_heads) * head_dim * head_dim * 4]));
    const sc_f32: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, state_c[0 .. @as(usize, num_heads) * head_dim * head_dim * 4]));

    for (0..pre_h.len) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(pre_h[i])), @as(f32, @floatCast(dec_h[i])), 2e-3);
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(pre_h[i])), @as(f32, @floatCast(tr_h[i])), 2e-3);
    }
    for (0..sa_f32.len) |i| {
        try std.testing.expectApproxEqAbs(sa_f32[i], sb_f32[i], 1e-4);
        try std.testing.expectApproxEqAbs(sa_f32[i], sc_f32[i], 1e-4);
    }

    // Verify against exact CPU FP32 KDA reference (`cos_sim >= 0.9999`).
    const cpu_state = try allocator.alloc(f32, @as(usize, num_heads) * head_dim * head_dim);
    defer allocator.free(cpu_state);
    @memset(cpu_state, 0.0);
    var dot_c: f64 = 0.0;
    var na_c: f64 = 0.0;
    var nb_c: f64 = 0.0;
    const q_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    for (0..seq_len) |t| {
        var qc: [qkv_dim]f32 = undefined;
        var kc: [qkv_dim]f32 = undefined;
        var vc: [qkv_dim]f32 = undefined;
        for (0..qkv_dim) |c| {
            var qa: f32 = 0.0;
            var ka: f32 = 0.0;
            var va: f32 = 0.0;
            for (0..conv_k) |tap| {
                const rel = @as(i32, @intCast(t)) - @as(i32, @intCast(hist - tap));
                if (rel >= 0) {
                    const rt: usize = @intCast(rel);
                    qa += @as(f32, @floatCast(q_in_h[rt * qkv_dim + c])) * qw_f32[c * conv_k + tap];
                    ka += @as(f32, @floatCast(k_in_h[rt * qkv_dim + c])) * kw_f32[c * conv_k + tap];
                    va += @as(f32, @floatCast(v_in_h[rt * qkv_dim + c])) * vw_f32[c * conv_k + tap];
                }
            }
            qc[c] = siluF32(qa);
            kc[c] = siluF32(ka);
            vc[c] = siluF32(va);
        }
        for (0..num_heads) |hi| {
            var qsq: f32 = 0.0;
            var ksq: f32 = 0.0;
            for (0..head_dim) |d| {
                const qv = qc[hi * head_dim + d];
                const kv = kc[hi * head_dim + d];
                qsq += qv * qv;
                ksq += kv * kv;
            }
            const qf = q_scale / @sqrt(qsq + l2_eps);
            const kf = 1.0 / @sqrt(ksq + l2_eps);
            const bv = sigmoidF32(@as(f32, @floatCast(b_in_h[t * num_heads + hi])));
            const decay_rate = @exp(a_log_f32[hi]);
            var alpha: [head_dim]f32 = undefined;
            for (0..head_dim) |d| {
                const gi = @as(f32, @floatCast(f_in_h[t * qkv_dim + hi * head_dim + d])) + dt_f32[hi * head_dim + d];
                alpha[d] = @exp(lower_bound * sigmoidF32(decay_rate * gi));
            }
            var o_head: [head_dim]f32 = undefined;
            for (0..head_dim) |dv| {
                var pred: f32 = 0.0;
                for (0..head_dim) |dk| {
                    const si = (hi * head_dim + dk) * head_dim + dv;
                    cpu_state[si] *= alpha[dk];
                    pred += cpu_state[si] * (kc[hi * head_dim + dk] * kf);
                }
                const delta = bv * (vc[hi * head_dim + dv] - pred);
                var ov: f32 = 0.0;
                for (0..head_dim) |dk| {
                    const si = (hi * head_dim + dk) * head_dim + dv;
                    cpu_state[si] += (kc[hi * head_dim + dk] * kf) * delta;
                    ov += cpu_state[si] * (qc[hi * head_dim + dk] * qf);
                }
                o_head[dv] = ov;
            }
            var osq: f32 = 0.0;
            for (0..head_dim) |dv| osq += o_head[dv] * o_head[dv];
            const inv_r = 1.0 / @sqrt(osq / @as(f32, @floatFromInt(head_dim)) + rms_eps);
            for (0..head_dim) |dv| {
                const ref = (o_head[dv] * inv_r) * @as(f32, @floatCast(norm_h[dv])) * sigmoidF32(@as(f32, @floatCast(z_in_h[t * qkv_dim + hi * head_dim + dv])));
                const got: f32 = @floatCast(pre_h[t * qkv_dim + hi * head_dim + dv]);
                dot_c += @as(f64, ref) * @as(f64, got);
                na_c += @as(f64, ref) * @as(f64, ref);
                nb_c += @as(f64, got) * @as(f64, got);
            }
        }
    }
    const cos_kda = dot_c / @sqrt(na_c * nb_c);
    try std.testing.expect(cos_kda >= 0.9999);
}

test "G27 GlmMoeDsaIndexer + radix Top-K + Sparse MLA AND NoPE-MLA index_kpool=4 + tail parity" {
    const device = try metal.Device.open();
    defer device.close();

    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    const pipes = try HybridPipelines.init(device, lib);
    defer pipes.close();

    const arena_cap = try page.alignUp(4 * 1024 * 1024);
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    // Case 1: GLM-5.2/5.3 (`kpool = 1`, `seq_len = 20`, `index_topk = 8` -> exercises 4-pass radix Top-K!)
    //         + interleaved RoPE + LayerNorm + Sparse MLA (`qk_rope = 16`, `kv_lora = 64`).
    // Case 2: GLM-5.3-Flash (`kpool = 4`, `always_select_tail = true`, `seq_len = 23` -> 5 pools + 3 tail,
    //         `index_topk = 12` -> top 3 pools = 12 tokens + 3 tail = 15 tokens!) + NoPE MLA (`qk_rope = 0`).
    const seq_len: u32 = 23;
    const idx_heads: u32 = 4;
    const idx_dim: u32 = 32;
    const n_q_heads: u32 = 2;
    const kv_rank: u32 = 64;
    const d_rope: u32 = 16;

    const q_idx_raw = try arena.allocAligned(@as(usize, seq_len) * idx_heads * idx_dim * 2, 64);
    const k_idx_raw = try arena.allocAligned(@as(usize, seq_len) * idx_dim * 2, 64);
    const k_normed_raw = try arena.allocAligned(@as(usize, seq_len) * idx_dim * 2, 64);
    const kn_w_raw = try arena.allocAligned(@as(usize, idx_dim) * 4, 64);
    const kn_b_raw = try arena.allocAligned(@as(usize, idx_dim) * 4, 64);
    const w_raw = try arena.allocAligned(@as(usize, seq_len) * idx_heads * 2, 64);
    const gate_raw = try arena.allocAligned(@as(usize, seq_len) * idx_dim * 2, 64);
    const ape_raw = try arena.allocAligned(4 * @as(usize, idx_dim) * 4, 64);
    const q_nope_raw = try arena.allocAligned(@as(usize, n_q_heads) * kv_rank * 2, 64);
    const q_pe_raw = try arena.allocAligned(@as(usize, n_q_heads) * d_rope * 2, 64);
    const lat_in_raw = try arena.allocAligned(@as(usize, seq_len) * kv_rank * 2, 64);
    const rope_in_raw = try arena.allocAligned(@as(usize, seq_len) * d_rope * 2, 64);
    const out_raw = try arena.allocAligned(@as(usize, n_q_heads) * kv_rank * 2, 64);

    const mla_kv = try G27MlaKvArena.init(&arena, .{
        .num_layers = 2,
        .max_seq_len = 32,
        .kv_lora_rank = kv_rank,
        .qk_rope_head_dim = d_rope,
    });
    const idx_kv = try G27DsaIdxKvArena.init(&arena, .{
        .num_full_layers = 1,
        .max_seq_len = 32,
        .index_head_dim = idx_dim,
        .kpool = 4,
    });
    const idx_buf = try G27DsaIndexBuffer.init(&arena, .{
        .max_rows = 4,
        .max_out_k = 16,
        .max_candidates = 32,
    });

    const q_idx_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_idx_raw[0 .. @as(usize, seq_len) * idx_heads * idx_dim * 2]));
    const k_idx_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_idx_raw[0 .. @as(usize, seq_len) * idx_dim * 2]));
    const kn_w: []f32 = @alignCast(std.mem.bytesAsSlice(f32, kn_w_raw[0 .. @as(usize, idx_dim) * 4]));
    const kn_b: []f32 = @alignCast(std.mem.bytesAsSlice(f32, kn_b_raw[0 .. @as(usize, idx_dim) * 4]));
    const w_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, w_raw[0 .. @as(usize, seq_len) * idx_heads * 2]));
    const gate_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate_raw[0 .. @as(usize, seq_len) * idx_dim * 2]));
    const ape_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, ape_raw[0 .. 4 * @as(usize, idx_dim) * 4]));
    const q_nope_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_nope_raw[0 .. @as(usize, n_q_heads) * kv_rank * 2]));
    const q_pe_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_pe_raw[0 .. @as(usize, n_q_heads) * d_rope * 2]));
    const lat_in_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, lat_in_raw[0 .. @as(usize, seq_len) * kv_rank * 2]));
    const rope_in_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, rope_in_raw[0 .. @as(usize, seq_len) * d_rope * 2]));
    const out_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw[0 .. @as(usize, n_q_heads) * kv_rank * 2]));

    var prng = std.Random.DefaultPrng.init(0x9203_2026);
    const rng = prng.random();
    for (q_idx_h) |*v| v.* = @floatCast((rng.float(f32) - 0.5) * 1.5);
    for (k_idx_h) |*v| v.* = @floatCast((rng.float(f32) - 0.5) * 1.5);
    for (kn_w) |*v| v.* = 0.8 + rng.float(f32) * 0.4;
    for (kn_b) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
    for (w_h) |*v| v.* = @floatCast(0.2 + rng.float(f32) * 1.0);
    for (gate_h) |*v| v.* = @floatCast((rng.float(f32) - 0.5) * 2.0);
    for (ape_f32) |*v| v.* = (rng.float(f32) - 0.5) * 0.5;
    for (q_nope_h) |*v| v.* = @floatCast((rng.float(f32) - 0.5) * 1.0);
    for (q_pe_h) |*v| v.* = @floatCast((rng.float(f32) - 0.5) * 1.0);
    for (lat_in_h) |*v| v.* = @floatCast((rng.float(f32) - 0.5) * 1.0);
    for (rope_in_h) |*v| v.* = @floatCast((rng.float(f32) - 0.5) * 1.0);

    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{buf});
    defer compute.removeResident(&.{buf}) catch |err| {
        std.debug.panic("removeResident failed: {}", .{err});
    };

    const bind = struct {
        fn of(b: metal.Buffer, a: *const arena_mod.WiredArena, ptr: [*]u8) metal.BufferBinding {
            return .{ .buffer = b, .offset = a.offsetOf(ptr) };
        }
    }.of;

    // Run LayerNorm + Interleaved RoPE + KV write + kpool=4 build + radix Top-K + Sparse MLA.
    const last_tok: u32 = seq_len - 1;
    {
        var frame = try compute.beginFrame(5000);
        try pipes.encodeG27LayerNorm(&frame, .{
            .x = bind(buf, &arena, k_idx_raw.ptr),
            .weight = bind(buf, &arena, kn_w_raw.ptr),
            .bias = bind(buf, &arena, kn_b_raw.ptr),
            .y = bind(buf, &arena, k_normed_raw.ptr),
            .num_rows = seq_len,
            .dim = idx_dim,
            .w_dtype = .f32,
            .b_dtype = .f32,
            .eps = 1e-6,
        }, false);
        try pipes.encodeG27RopeInterleave(&frame, .{
            .x = bind(buf, &arena, k_normed_raw.ptr),
            .num_tokens = seq_len,
            .num_heads = 1,
            .head_stride = idx_dim,
            .rot_dim = d_rope,
            .rot_offset = 0,
            .base_pos = 0,
            .rope_theta = 8_000_000.0,
        }, true);
        try pipes.encodeG27DsaIdxKvWrite(&frame, .{
            .k_in = bind(buf, &arena, k_normed_raw.ptr),
            .gate_in = bind(buf, &arena, gate_raw.ptr),
            .k_cache = try idx_kv.kBinding(buf, 0),
            .gate_cache = try idx_kv.gateBinding(buf, 0),
            .num_tokens = seq_len,
            .index_head_dim = idx_dim,
            .start_pos = 0,
        }, true);
        try pipes.encodeG27MlaKvWrite(&frame, .{
            .latent_in = bind(buf, &arena, lat_in_raw.ptr),
            .rope_in = bind(buf, &arena, rope_in_raw.ptr),
            .kv_cache = try mla_kv.layerBinding(buf, 0),
            .num_tokens = seq_len,
            .kv_lora_rank = kv_rank,
            .qk_rope_head_dim = d_rope,
            .start_pos = 0,
        }, false);
        // Test Case 1: token-level (`kpool = 1`, `index_topk = 8`) for query at `last_tok = 22`.
        try pipes.encodeG27DsaScoresAndTopK(&frame, .{
            .q_idx = .{ .buffer = buf, .offset = arena.offsetOf(q_idx_raw.ptr) + @as(u64, last_tok) * idx_heads * idx_dim * 2 },
            .weights = .{ .buffer = buf, .offset = arena.offsetOf(w_raw.ptr) + @as(u64, last_tok) * idx_heads * 2 },
            .k_cand_cache = try idx_kv.kBinding(buf, 0),
            .scores_scratch = idx_buf.scoresBinding(buf),
            .out_indices = idx_buf.indicesBinding(buf),
            .out_counts = idx_buf.countsBinding(buf),
            .num_rows = 1,
            .index_n_heads = idx_heads,
            .index_head_dim = idx_dim,
            .committed_kv_len = seq_len,
            .index_topk = 8,
            .max_out_k = 16,
            .kpool = 1,
            .causal_base_pos = last_tok,
        }, true);
        try pipes.encodeG27MlaSparseAttn(&frame, .{
            .q_nope = bind(buf, &arena, q_nope_raw.ptr),
            .q_pe = bind(buf, &arena, q_pe_raw.ptr),
            .kv_cache = try mla_kv.layerBinding(buf, 0),
            .selected_indices = idx_buf.indicesBinding(buf),
            .selected_counts = idx_buf.countsBinding(buf),
            .out = bind(buf, &arena, out_raw.ptr),
            .num_rows = 1,
            .num_q_heads = n_q_heads,
            .kv_lora_rank = kv_rank,
            .qk_rope_head_dim = d_rope,
            .committed_kv_len = seq_len,
            .max_out_k = 16,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_rank + d_rope))),
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    try std.testing.expectEqual(@as(u32, 8), idx_buf.counts_host[0]);
    // Verify sparse MLA output matches exact CPU softmax over those 8 selected tokens!
    const kv_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_rank + d_rope)));
    for (0..n_q_heads) |hd| {
        var scores_cpu: [8]f32 = undefined;
        var mx: f32 = -std.math.inf(f32);
        for (0..8) |si| {
            const tidx = idx_buf.indices_host[si];
            try std.testing.expect(tidx < seq_len);
            var dot_v: f32 = 0.0;
            for (0..kv_rank) |d| dot_v += @as(f32, @floatCast(q_nope_h[hd * kv_rank + d])) * @as(f32, @floatCast(lat_in_h[tidx * kv_rank + d]));
            for (0..d_rope) |r| dot_v += @as(f32, @floatCast(q_pe_h[hd * d_rope + r])) * @as(f32, @floatCast(rope_in_h[tidx * d_rope + r]));
            scores_cpu[si] = dot_v * kv_scale;
            mx = @max(mx, scores_cpu[si]);
        }
        var denom: f32 = 0.0;
        for (0..8) |si| {
            scores_cpu[si] = @exp(scores_cpu[si] - mx);
            denom += scores_cpu[si];
        }
        for (0..kv_rank) |d| {
            var acc: f32 = 0.0;
            for (0..8) |si| {
                const tidx = idx_buf.indices_host[si];
                acc += (scores_cpu[si] / denom) * @as(f32, @floatCast(lat_in_h[tidx * kv_rank + d]));
            }
            try std.testing.expectApproxEqAbs(acc, @as(f32, @floatCast(out_h[hd * kv_rank + d])), 2e-3);
        }
    }

    // Test Case 2: GLM-5.3-Flash `NoPE MLA` (`qk_rope = 0`, `kpool = 4`, `always_select_tail = true`, `index_topk = 12`):
    // `seq_len = 23` -> `5` complete pools `[0..19]` + `3` tail tokens `[20, 21, 22]`.
    // Selecting top `12 / 4 = 3` pools -> `12` tokens + `3` tail = `15` selected indices!
    {
        var frame = try compute.beginFrame(5000);
        try pipes.encodeG27DsaKpoolBuild(&frame, .{
            .k_cache = try idx_kv.kBinding(buf, 0),
            .gate_cache = try idx_kv.gateBinding(buf, 0),
            .kpool_ape = bind(buf, &arena, ape_raw.ptr),
            .pool_k_cache = try idx_kv.poolBinding(buf, 0),
            .start_pool = 0,
            .end_pool = seq_len / 4,
            .index_head_dim = idx_dim,
            .kpool = 4,
            .ape_dtype = .f32,
        }, false);
        try pipes.encodeG27DsaScoresAndTopK(&frame, .{
            .q_idx = .{ .buffer = buf, .offset = arena.offsetOf(q_idx_raw.ptr) + @as(u64, last_tok) * idx_heads * idx_dim * 2 },
            .weights = .{ .buffer = buf, .offset = arena.offsetOf(w_raw.ptr) + @as(u64, last_tok) * idx_heads * 2 },
            .k_cand_cache = try idx_kv.poolBinding(buf, 0),
            .k_raw_cache = try idx_kv.kBinding(buf, 0),
            .gate_cache = try idx_kv.gateBinding(buf, 0),
            .kpool_ape = bind(buf, &arena, ape_raw.ptr),
            .scores_scratch = idx_buf.scoresBinding(buf),
            .out_indices = idx_buf.indicesBinding(buf),
            .out_counts = idx_buf.countsBinding(buf),
            .num_rows = 1,
            .index_n_heads = idx_heads,
            .index_head_dim = idx_dim,
            .committed_kv_len = seq_len,
            .index_topk = 12,
            .max_out_k = 16,
            .kpool = 4,
            .always_select_tail = true,
            .causal_base_pos = last_tok,
            .ape_dtype = .f32,
        }, true);
        try pipes.encodeG27MlaKvWrite(&frame, .{
            .latent_in = bind(buf, &arena, lat_in_raw.ptr),
            .rope_in = null,
            .kv_cache = try mla_kv.layerBinding(buf, 1),
            .num_tokens = seq_len,
            .kv_lora_rank = kv_rank,
            .qk_rope_head_dim = 0,
            .start_pos = 0,
        }, false);
        try pipes.encodeG27MlaSparseAttn(&frame, .{
            .q_nope = bind(buf, &arena, q_nope_raw.ptr),
            .q_pe = null,
            .kv_cache = try mla_kv.layerBinding(buf, 1),
            .selected_indices = idx_buf.indicesBinding(buf),
            .selected_counts = idx_buf.countsBinding(buf),
            .out = bind(buf, &arena, out_raw.ptr),
            .num_rows = 1,
            .num_q_heads = n_q_heads,
            .kv_lora_rank = kv_rank,
            .qk_rope_head_dim = 0,
            .committed_kv_len = seq_len,
            .max_out_k = 16,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_rank))),
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    try std.testing.expectEqual(@as(u32, 15), idx_buf.counts_host[0]);
    // Verify tail tokens [20, 21, 22] are ALWAYS present at the end of selected indices!
    try std.testing.expectEqual(@as(u32, 20), idx_buf.indices_host[12]);
    try std.testing.expectEqual(@as(u32, 21), idx_buf.indices_host[13]);
    try std.testing.expectEqual(@as(u32, 22), idx_buf.indices_host[14]);
    try std.testing.expectEqual(@as(u32, 0xFFFF_FFFF), idx_buf.indices_host[15]);
    // Verify NoPE MLA sparse attention matches CPU reference over those 15 selected tokens!
    const nope_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_rank)));
    for (0..n_q_heads) |hd| {
        var scores_cpu: [15]f32 = undefined;
        var mx: f32 = -std.math.inf(f32);
        for (0..15) |si| {
            const tidx = idx_buf.indices_host[si];
            try std.testing.expect(tidx < seq_len);
            var dot_v: f32 = 0.0;
            for (0..kv_rank) |d| dot_v += @as(f32, @floatCast(q_nope_h[hd * kv_rank + d])) * @as(f32, @floatCast(lat_in_h[tidx * kv_rank + d]));
            scores_cpu[si] = dot_v * nope_scale;
            mx = @max(mx, scores_cpu[si]);
        }
        var denom: f32 = 0.0;
        for (0..15) |si| {
            scores_cpu[si] = @exp(scores_cpu[si] - mx);
            denom += scores_cpu[si];
        }
        for (0..kv_rank) |d| {
            var acc: f32 = 0.0;
            for (0..15) |si| {
                const tidx = idx_buf.indices_host[si];
                acc += (scores_cpu[si] / denom) * @as(f32, @floatCast(lat_in_h[tidx * kv_rank + d]));
            }
            try std.testing.expectApproxEqAbs(acc, @as(f32, @floatCast(out_h[hd * kv_rank + d])), 2e-3);
        }
    }
}

test "M9.3 G28: ami_moe_route_topk_sigmoid 512E group-limited top-2 sum vs max-1 (n_group=8, topk_group=4, scale=2.5) vs CPU fp32 ref" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const num_e: u32 = 512;
    const m: u32 = 3;
    const top_k: u32 = 8;
    const n_group: u32 = 8;
    const topk_group: u32 = 4;
    const scale: f32 = 2.5;

    const arena_cap = try page.alignUp(256 * 1024);
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    const logits_raw = try arena.allocAligned(m * num_e * @sizeOf(f32), 64);
    const bias_raw = try arena.allocAligned(num_e * @sizeOf(f32), 64);
    const hdr_raw = try arena.allocAligned(64, 64);
    const uniq_raw = try arena.allocAligned(num_e * @sizeOf(u32), 64);
    const offs_raw = try arena.allocAligned((num_e + 1) * @sizeOf(u32), 64);
    const toks_raw = try arena.allocAligned(m * top_k * @sizeOf(u32), 64);
    const wts_raw = try arena.allocAligned(m * top_k * @sizeOf(f32), 64);
    const slots_raw = try arena.allocAligned(m * top_k * @sizeOf(u32), 64);
    const sk_e_raw = try arena.allocAligned(m * top_k * @sizeOf(u32), 64);
    const sk_w_raw = try arena.allocAligned(m * top_k * @sizeOf(f32), 64);

    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw[0 .. m * num_e * 4]));
    const bias_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bias_raw[0 .. num_e * 4]));
    const hdr_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, hdr_raw[0..64]));
    const uniq_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, uniq_raw[0 .. num_e * 4]));
    const offs_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, offs_raw[0 .. (num_e + 1) * 4]));
    const toks_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, toks_raw[0 .. m * top_k * 4]));
    const wts_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, wts_raw[0 .. m * top_k * 4]));
    const sk_e_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, sk_e_raw[0 .. m * top_k * 4]));
    const sk_w_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, sk_w_raw[0 .. m * top_k * 4]));

    for (0..num_e) |ei| {
        bias_f32[ei] = synthF32(7100 + ei, 0.30);
    }
    for (0..m) |ti| {
        for (0..num_e) |ei| {
            logits_f32[ti * num_e + ei] = synthF32(9100 + ti * 521 + ei, 1.9);
        }
        // Engineer group 0 to have one very high spike + low 2nd best, and group 1 to have two strong experts
        // so `group_score_topk = 2` (BailingMoeV3Gate top-2 sum) and `group_score_topk = 1` (max-1) diverge!
        logits_f32[ti * num_e + 0] = 5.0;
        for (1..64) |j| logits_f32[ti * num_e + j] = -6.0;
        logits_f32[ti * num_e + 64] = 3.2;
        logits_f32[ti * num_e + 65] = 3.1;
        // Also test -inf cold-mask safety inside group 2.
        logits_f32[ti * num_e + 128 + ti] = -std.math.inf(f32);
    }

    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{buf});

    var mode2_e0: [top_k]u32 = undefined;
    var mode1_e0: [top_k]u32 = undefined;

    for ([_]u32{ 2, 1 }) |gs_topk| {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG26MoeRouteSigmoid(&frame, .{
            .logits = .{ .buffer = buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .logits_f16 = false,
            .bias = .{ .buffer = buf, .offset = arena.offsetOf(bias_raw.ptr) },
            .bias_dtype = .f32,
            .sched_hdr = .{ .buffer = buf, .offset = arena.offsetOf(hdr_raw.ptr) },
            .unique_experts = .{ .buffer = buf, .offset = arena.offsetOf(uniq_raw.ptr) },
            .expert_token_offsets = .{ .buffer = buf, .offset = arena.offsetOf(offs_raw.ptr) },
            .token_indices = .{ .buffer = buf, .offset = arena.offsetOf(toks_raw.ptr) },
            .routing_weights = .{ .buffer = buf, .offset = arena.offsetOf(wts_raw.ptr) },
            .slot_k_indices = .{ .buffer = buf, .offset = arena.offsetOf(slots_raw.ptr) },
            .scratch_topk_experts = .{ .buffer = buf, .offset = arena.offsetOf(sk_e_raw.ptr) },
            .scratch_topk_weights = .{ .buffer = buf, .offset = arena.offsetOf(sk_w_raw.ptr) },
            .m = m,
            .num_experts = num_e,
            .top_k = top_k,
            .n_group = n_group,
            .topk_group = topk_group,
            .group_score_topk = gs_topk,
            .norm_topk_prob = true,
            .routed_scaling_factor = scale,
            .gate_up_groups_x_q = 8,
            .gate_up_groups_x_f = 4,
            .down_groups_x = 8,
            .down_groups_x_q8 = 16,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        if (gs_topk == 2) {
            @memcpy(&mode2_e0, sk_e_u32[0..top_k]);
        } else {
            @memcpy(&mode1_e0, sk_e_u32[0..top_k]);
        }

        // Exact CPU FP32 BailingMoeV3Gate reference (`group_limited_topk`)
        const g_size = num_e / n_group;
        for (0..m) |ti| {
            var g_scores: [n_group]f32 = undefined;
            for (0..n_group) |g| {
                var m1: f32 = -std.math.inf(f32);
                var m2: f32 = -std.math.inf(f32);
                for (0..g_size) |j| {
                    const ei = g * g_size + j;
                    const lg = logits_f32[ti * num_e + ei];
                    const ch: f32 = if (std.math.isNegativeInf(lg)) -std.math.inf(f32) else sigmoidF32(lg) + bias_f32[ei];
                    if (ch > m1) {
                        m2 = m1;
                        m1 = ch;
                    } else if (ch > m2) {
                        m2 = ch;
                    }
                }
                g_scores[g] = if (gs_topk == 1) m1 else (if (std.math.isFinite(m2)) m1 + m2 else m1);
            }
            var g_allowed: [n_group]bool = @splat(false);
            for (0..topk_group) |_| {
                var bg: usize = 0;
                var bv: f32 = -std.math.inf(f32);
                for (0..n_group) |g| {
                    if (!g_allowed[g] and g_scores[g] > bv) {
                        bv = g_scores[g];
                        bg = g;
                    }
                }
                g_allowed[bg] = true;
            }

            var chosen_e: [top_k]u32 = undefined;
            var chosen_w: [top_k]f32 = undefined;
            var used: [num_e]bool = @splat(false);
            var sum_w: f32 = 0.0;
            for (0..top_k) |ki| {
                var best_e: u32 = 0;
                var best_ch: f32 = -std.math.inf(f32);
                var best_unbiased: f32 = 0.0;
                for (0..num_e) |ei| {
                    if (used[ei] or !g_allowed[ei / g_size]) continue;
                    const lg = logits_f32[ti * num_e + ei];
                    const sv: f32 = if (std.math.isNegativeInf(lg)) 0.0 else sigmoidF32(lg);
                    const ch: f32 = if (std.math.isNegativeInf(lg)) -std.math.inf(f32) else sv + bias_f32[ei];
                    if (ch > best_ch) {
                        best_ch = ch;
                        best_e = @intCast(ei);
                        best_unbiased = sv;
                    }
                }
                used[best_e] = true;
                chosen_e[ki] = best_e;
                chosen_w[ki] = best_unbiased;
                sum_w += best_unbiased;
            }
            for (0..top_k) |ki| {
                chosen_w[ki] = (chosen_w[ki] / (sum_w + 1e-20)) * scale;
                try std.testing.expectEqual(chosen_e[ki], sk_e_u32[ti * top_k + ki]);
                try std.testing.expectApproxEqAbs(chosen_w[ki], sk_w_f32[ti * top_k + ki], 1e-5);
            }
        }

        const num_u = hdr_u32[0];
        try std.testing.expectEqual(m * top_k, offs_u32[num_u]);
        for (0..num_u) |ui| {
            const eid = uniq_u32[ui];
            for (offs_u32[ui]..offs_u32[ui + 1]) |si| {
                const tok = toks_u32[si];
                var found = false;
                for (0..top_k) |ki| {
                    if (sk_e_u32[tok * top_k + ki] == eid) {
                        try std.testing.expectApproxEqAbs(sk_w_f32[tok * top_k + ki], wts_f32[si], 1e-5);
                        found = true;
                        break;
                    }
                }
                try std.testing.expect(found);
            }
        }
    }
    // Confirm top-2 sum vs max-1 actually exercise distinct group selection on this input.
    try std.testing.expect(!std.mem.eql(u32, &mode2_e0, &mode1_e0));
}

test "M9.3 G28: ami_g28_mla_full_attend + ami_g28_mla_out_gate (head_wise & elem_wise, decode/prefill/tree) vs CPU fp32 ref" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try HybridPipelines.init(device, library);
    defer pipes.close();

    const n_heads: u32 = 4;
    const kv_rank: u32 = 64;
    const d_rope: u32 = 32;
    const max_seq: u32 = 16;
    const m_prefill: u32 = 4;
    const m_tree: u32 = 4;

    const arena_cap = try page.alignUp(256 * 1024);
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    var mla_kv = try G27MlaKvArena.init(&arena, .{
        .num_layers = 2,
        .max_seq_len = max_seq,
        .kv_lora_rank = kv_rank,
        .qk_rope_head_dim = d_rope,
    });
    const lat_raw = try arena.allocAligned(max_seq * kv_rank * 2, 64);
    const rope_raw = try arena.allocAligned(max_seq * d_rope * 2, 64);
    const qn_raw = try arena.allocAligned(m_prefill * n_heads * kv_rank * 2, 64);
    const qp_raw = try arena.allocAligned(m_prefill * n_heads * d_rope * 2, 64);
    const tlat_raw = try arena.allocAligned(m_tree * kv_rank * 2, 64);
    const trope_raw = try arena.allocAligned(m_tree * d_rope * 2, 64);
    const masks_raw = try arena.allocAligned(m_tree * @sizeOf(u64), 64);
    const out_raw = try arena.allocAligned(m_prefill * n_heads * kv_rank * 2, 64);
    const gate_hw_raw = try arena.allocAligned(m_prefill * n_heads * 2, 64);
    const gate_ew_raw = try arena.allocAligned(m_prefill * n_heads * kv_rank * 2, 64);

    const lat_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, lat_raw[0 .. max_seq * kv_rank * 2]));
    const rope_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, rope_raw[0 .. max_seq * d_rope * 2]));
    const qn_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, qn_raw[0 .. m_prefill * n_heads * kv_rank * 2]));
    const qp_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, qp_raw[0 .. m_prefill * n_heads * d_rope * 2]));
    const tlat_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, tlat_raw[0 .. m_tree * kv_rank * 2]));
    const trope_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, trope_raw[0 .. m_tree * d_rope * 2]));
    const masks_u64: []u64 = @alignCast(std.mem.bytesAsSlice(u64, masks_raw[0 .. m_tree * 8]));
    const out_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw[0 .. m_prefill * n_heads * kv_rank * 2]));
    const gate_hw_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate_hw_raw[0 .. m_prefill * n_heads * 2]));
    const gate_ew_h: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate_ew_raw[0 .. m_prefill * n_heads * kv_rank * 2]));

    for (lat_h, 0..) |*v, i| v.* = @floatCast(synthF32(3100 + i, 0.35));
    for (rope_h, 0..) |*v, i| v.* = @floatCast(synthF32(3400 + i, 0.35));
    for (qn_h, 0..) |*v, i| v.* = @floatCast(synthF32(3700 + i, 0.35));
    for (qp_h, 0..) |*v, i| v.* = @floatCast(synthF32(4100 + i, 0.35));
    for (tlat_h, 0..) |*v, i| v.* = @floatCast(synthF32(4400 + i, 0.35));
    for (trope_h, 0..) |*v, i| v.* = @floatCast(synthF32(4700 + i, 0.35));
    for (gate_hw_h, 0..) |*v, i| v.* = @floatCast(synthF32(5100 + i, 1.4));
    for (gate_ew_h, 0..) |*v, i| v.* = @floatCast(synthF32(5400 + i, 1.4));
    masks_u64[0] = 0b0001;
    masks_u64[1] = 0b0011;
    masks_u64[2] = 0b0101;
    masks_u64[3] = 0b1011;

    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{buf});

    const bind = struct {
        fn at(b: metal.Buffer, a: *const arena_mod.WiredArena, ptr: [*]u8) metal.BufferBinding {
            return .{ .buffer = b, .offset = a.offsetOf(ptr) };
        }
    }.at;

    const attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_rank + d_rope)));

    // 1. Causal prefill (`m = 4`, positions `0..3`) + head_wise `g_proj` output gate
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG27MlaKvWrite(&frame, .{
            .latent_in = bind(buf, &arena, lat_raw.ptr),
            .rope_in = bind(buf, &arena, rope_raw.ptr),
            .kv_cache = try mla_kv.layerBinding(buf, 0),
            .num_tokens = 5,
            .kv_lora_rank = kv_rank,
            .qk_rope_head_dim = d_rope,
            .start_pos = 0,
        }, false);
        try pipes.encodeG28MlaFullAttn(&frame, .{
            .q_nope = bind(buf, &arena, qn_raw.ptr),
            .q_pe = bind(buf, &arena, qp_raw.ptr),
            .kv_cache = try mla_kv.layerBinding(buf, 0),
            .out = bind(buf, &arena, out_raw.ptr),
            .num_rows = m_prefill,
            .num_q_heads = n_heads,
            .kv_lora_rank = kv_rank,
            .qk_rope_head_dim = d_rope,
            .committed_kv_len = m_prefill,
            .causal_base_pos = 0,
            .is_tree = false,
            .scale = attn_scale,
        }, true);
        try pipes.encodeG28MlaOutGate(&frame, .{
            .attn_out = bind(buf, &arena, out_raw.ptr),
            .gate = bind(buf, &arena, gate_hw_raw.ptr),
            .num_rows = m_prefill,
            .num_heads = n_heads,
            .v_head_dim = kv_rank,
            .head_wise = true,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    var dot_ab: f64 = 0.0;
    var norm_a: f64 = 0.0;
    var norm_b: f64 = 0.0;
    for (0..m_prefill) |r| {
        for (0..n_heads) |hd| {
            var scores: [4]f32 = undefined;
            var mx: f32 = -std.math.inf(f32);
            for (0..r + 1) |k| {
                var dv: f32 = 0.0;
                for (0..kv_rank) |d| dv += @as(f32, @floatCast(qn_h[(r * n_heads + hd) * kv_rank + d])) * @as(f32, @floatCast(lat_h[k * kv_rank + d]));
                for (0..d_rope) |d| dv += @as(f32, @floatCast(qp_h[(r * n_heads + hd) * d_rope + d])) * @as(f32, @floatCast(rope_h[k * d_rope + d]));
                scores[k] = dv * attn_scale;
                mx = @max(mx, scores[k]);
            }
            var den: f32 = 0.0;
            for (0..r + 1) |k| {
                scores[k] = @exp(scores[k] - mx);
                den += scores[k];
            }
            const g_val = sigmoidF32(@as(f32, @floatCast(gate_hw_h[r * n_heads + hd])));
            for (0..kv_rank) |d| {
                var acc: f32 = 0.0;
                for (0..r + 1) |k| acc += (scores[k] / den) * @as(f32, @floatCast(lat_h[k * kv_rank + d]));
                const ref_v = acc * g_val;
                const got_v: f32 = @floatCast(out_h[(r * n_heads + hd) * kv_rank + d]);
                try std.testing.expectApproxEqAbs(ref_v, got_v, 3e-3);
                dot_ab += @as(f64, ref_v) * @as(f64, got_v);
                norm_a += @as(f64, ref_v) * @as(f64, ref_v);
                norm_b += @as(f64, got_v) * @as(f64, got_v);
            }
        }
    }
    try std.testing.expect(dot_ab / @sqrt(norm_a * norm_b) >= 0.9999);

    // 2. Branching speculative tree verify (`m = 4`, committed `5` + `ancestor_masks`) + element-wise gate
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG28MlaFullAttn(&frame, .{
            .q_nope = bind(buf, &arena, qn_raw.ptr),
            .q_pe = bind(buf, &arena, qp_raw.ptr),
            .kv_cache = try mla_kv.layerBinding(buf, 0),
            .kv_tree_latent = bind(buf, &arena, tlat_raw.ptr),
            .kv_tree_rope = bind(buf, &arena, trope_raw.ptr),
            .ancestor_masks = bind(buf, &arena, masks_raw.ptr),
            .out = bind(buf, &arena, out_raw.ptr),
            .num_rows = m_tree,
            .num_q_heads = n_heads,
            .kv_lora_rank = kv_rank,
            .qk_rope_head_dim = d_rope,
            .committed_kv_len = 5,
            .causal_base_pos = 5,
            .is_tree = true,
            .scale = attn_scale,
        }, false);
        try pipes.encodeG28MlaOutGate(&frame, .{
            .attn_out = bind(buf, &arena, out_raw.ptr),
            .gate = bind(buf, &arena, gate_ew_raw.ptr),
            .num_rows = m_tree,
            .num_heads = n_heads,
            .v_head_dim = kv_rank,
            .head_wise = false,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    dot_ab = 0.0;
    norm_a = 0.0;
    norm_b = 0.0;
    for (0..m_tree) |r| {
        const mask = masks_u64[r];
        for (0..n_heads) |hd| {
            var scores: [9]f32 = undefined;
            var active: [9]bool = @splat(false);
            var mx: f32 = -std.math.inf(f32);
            for (0..5) |k| {
                var dv: f32 = 0.0;
                for (0..kv_rank) |d| dv += @as(f32, @floatCast(qn_h[(r * n_heads + hd) * kv_rank + d])) * @as(f32, @floatCast(lat_h[k * kv_rank + d]));
                for (0..d_rope) |d| dv += @as(f32, @floatCast(qp_h[(r * n_heads + hd) * d_rope + d])) * @as(f32, @floatCast(rope_h[k * d_rope + d]));
                scores[k] = dv * attn_scale;
                active[k] = true;
                mx = @max(mx, scores[k]);
            }
            for (0..m_tree) |j| {
                if (((mask >> @intCast(j)) & 1) == 0) continue;
                var dv: f32 = 0.0;
                for (0..kv_rank) |d| dv += @as(f32, @floatCast(qn_h[(r * n_heads + hd) * kv_rank + d])) * @as(f32, @floatCast(tlat_h[j * kv_rank + d]));
                for (0..d_rope) |d| dv += @as(f32, @floatCast(qp_h[(r * n_heads + hd) * d_rope + d])) * @as(f32, @floatCast(trope_h[j * d_rope + d]));
                scores[5 + j] = dv * attn_scale;
                active[5 + j] = true;
                mx = @max(mx, scores[5 + j]);
            }
            var den: f32 = 0.0;
            for (0..9) |idx| {
                if (!active[idx]) continue;
                scores[idx] = @exp(scores[idx] - mx);
                den += scores[idx];
            }
            for (0..kv_rank) |d| {
                var acc: f32 = 0.0;
                for (0..5) |k| acc += (scores[k] / den) * @as(f32, @floatCast(lat_h[k * kv_rank + d]));
                for (0..m_tree) |j| {
                    if (!active[5 + j]) continue;
                    acc += (scores[5 + j] / den) * @as(f32, @floatCast(tlat_h[j * kv_rank + d]));
                }
                const g_idx = (r * n_heads + hd) * kv_rank + d;
                const ref_v = acc * sigmoidF32(@as(f32, @floatCast(gate_ew_h[g_idx])));
                const got_v: f32 = @floatCast(out_h[g_idx]);
                try std.testing.expectApproxEqAbs(ref_v, got_v, 3e-3);
                dot_ab += @as(f64, ref_v) * @as(f64, got_v);
                norm_a += @as(f64, ref_v) * @as(f64, ref_v);
                norm_b += @as(f64, got_v) * @as(f64, got_v);
            }
        }
    }
    try std.testing.expect(dot_ab / @sqrt(norm_a * norm_b) >= 0.9999);

    // 3. Single-token decode (`m = 1`, `d_rope = 0` NoPE guard)
    {
        const nope_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_rank)));
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG27MlaKvWrite(&frame, .{
            .latent_in = bind(buf, &arena, lat_raw.ptr),
            .rope_in = null,
            .kv_cache = try mla_kv.layerBinding(buf, 1),
            .num_tokens = 5,
            .kv_lora_rank = kv_rank,
            .qk_rope_head_dim = 0,
            .start_pos = 0,
        }, false);
        try pipes.encodeG28MlaFullAttn(&frame, .{
            .q_nope = bind(buf, &arena, qn_raw.ptr),
            .q_pe = null,
            .kv_cache = try mla_kv.layerBinding(buf, 1),
            .out = bind(buf, &arena, out_raw.ptr),
            .num_rows = 1,
            .num_q_heads = n_heads,
            .kv_lora_rank = kv_rank,
            .qk_rope_head_dim = 0,
            .committed_kv_len = 5,
            .causal_base_pos = 4,
            .is_tree = false,
            .scale = nope_scale,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        for (0..n_heads) |hd| {
            var scores: [5]f32 = undefined;
            var mx: f32 = -std.math.inf(f32);
            for (0..5) |k| {
                var dv: f32 = 0.0;
                for (0..kv_rank) |d| dv += @as(f32, @floatCast(qn_h[hd * kv_rank + d])) * @as(f32, @floatCast(lat_h[k * kv_rank + d]));
                scores[k] = dv * nope_scale;
                mx = @max(mx, scores[k]);
            }
            var den: f32 = 0.0;
            for (0..5) |k| {
                scores[k] = @exp(scores[k] - mx);
                den += scores[k];
            }
            for (0..kv_rank) |d| {
                var acc: f32 = 0.0;
                for (0..5) |k| acc += (scores[k] / den) * @as(f32, @floatCast(lat_h[k * kv_rank + d]));
                try std.testing.expectApproxEqAbs(acc, @as(f32, @floatCast(out_h[hd * kv_rank + d])), 3e-3);
            }
        }
    }
}
