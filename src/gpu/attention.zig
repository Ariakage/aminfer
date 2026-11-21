//! IOSurface KV-tile attention kernels, Spike S1 microbenchmark, RMSNorm,
//! QK-Norm + NeoX RoPE + KV-Write, and GQA decode attention (R3, R4, S1, G4).
//! RMSNorm and QK-Norm accept `f16` or `bf16` norm weights through the
//! `NORM_W_BF16` Metal function constant (G12, ADR-0016).

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const amf = @import("../format/amf.zig");
const quant = @import("../quant/spec.zig");
const dequant = @import("dequant.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const vmstat = @import("../platform/vmstat.zig");

/// Default number of tokens stored per KV cache chunk tile (ADR-0004).
pub const DEFAULT_CHUNK_TOKENS: u32 = 512;

/// Maximum speculative verification tree width (64-bit ancestor bitmask, ADR-0014).
pub const MAX_TREE_NODES: u32 = 64;

/// Maximum FlashDecoding split-KV partitions supported per dispatch.
pub const MAX_SPLITS: u32 = 64;

/// Apple M4 unified memory bus peak bandwidth in decimal GB/s.
pub const M4_BUS_BANDWIDTH_GBS: f64 = 120.0;

/// GPU-visible bindless handle pair for one KV chunk's K and V `KvTile`s.
/// Matches `struct KvTilePair` in `src/kernels/kv_attention.metal`.
pub const KvTilePair = extern struct {
    k_tex_id: u64,
    v_tex_id: u64,

    pub fn fromTiles(k: metal.KvTile, v: metal.KvTile, view: metal.TextureView) KvTilePair {
        return switch (view) {
            .rgba16f => .{ .k_tex_id = k.rgbaResourceId(), .v_tex_id = v.rgbaResourceId() },
            .r16f => .{ .k_tex_id = k.r16ResourceId(), .v_tex_id = v.r16ResourceId() },
        };
    }
};

/// GPU-visible pointer pair for Spike S1 aliasing `MTLBuffer` comparison.
/// Matches `struct KvBufPair` in `src/kernels/kv_attention.metal`.
pub const KvBufPair = extern struct {
    k_gpu_addr: u64,
    v_gpu_addr: u64,

    pub fn fromTiles(k: metal.KvTile, v: metal.KvTile) metal.Error!KvBufPair {
        const kb = k.aliasBuffer() orelse return error.InvalidArgument;
        const vb = v.aliasBuffer() orelse return error.InvalidArgument;
        return .{ .k_gpu_addr = kb.gpuAddress(), .v_gpu_addr = vb.gpuAddress() };
    }
};

/// Uniform payload matching `struct RmsNormParams` in `kv_attention.metal`.
pub const RmsNormParams = extern struct {
    num_rows: u32,
    hidden_dim: u32,
    eps: f32,
    _pad: u32 = 0,
};

/// Uniform payload matching `struct QkNormRopeKvParams` in `kv_attention.metal`.
pub const QkNormRopeKvParams = extern struct {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    pos: u32,
    chunk_tokens: u32,
    slot_in_chunk: u32,
    rms_eps: f32,
    rope_theta: f32,
    apply_qk_norm: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

/// Uniform payload matching `struct GqaDecodeParams` in `kv_attention.metal`.
pub const GqaDecodeParams = extern struct {
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    chunk_tokens: u32,
    num_splits: u32,
    q_groups_per_kv: u32,
    scale: f32,
    _pad: u32 = 0,
};

/// Uniform payload matching `struct SpikeS1Params` in `kv_attention.metal`.
pub const SpikeS1Params = extern struct {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    chunk_tokens: u32,
    num_chunks: u32,
    scale: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

/// Uniform payload matching `struct TreeQkNormRopeParams` in `kv_attention.metal`.
pub const TreeQkNormRopeParams = extern struct {
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rms_eps: f32,
    rope_theta: f32,
    apply_qk_norm: u32,
    _pad: u32 = 0,
};

/// Uniform payload matching `struct GqaTreeVerifyParams` in `kv_attention.metal`.
pub const GqaTreeVerifyParams = extern struct {
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    base_pos: u32,
    chunk_tokens: u32,
    num_splits: u32,
    q_groups_per_kv: u32,
    scale: f32,
};

/// Uniform payload matching `struct KvTreeCommitParams` in `kv_attention.metal`.
pub const KvTreeCommitParams = extern struct {
    num_kv_heads: u32,
    head_dim: u32,
    base_pos: u32,
    chunk_tokens: u32,
    accepted_count: u32,
    use_node_buffer: u32,
    has_second_chunk: u32,
    _pad: u32 = 0,
    inline_nodes: [MAX_TREE_NODES]u16 = @splat(0),
};

comptime {
    if (@sizeOf(KvTilePair) != 16) @compileError("KvTilePair must be 16 bytes");
    if (@sizeOf(KvBufPair) != 16) @compileError("KvBufPair must be 16 bytes");
    if (@sizeOf(RmsNormParams) != 16) @compileError("RmsNormParams must be 16 bytes");
    if (@sizeOf(QkNormRopeKvParams) != 48) @compileError("QkNormRopeKvParams must be 48 bytes");
    if (@sizeOf(GqaDecodeParams) != 32) @compileError("GqaDecodeParams must be 32 bytes");
    if (@sizeOf(SpikeS1Params) != 32) @compileError("SpikeS1Params must be 32 bytes");
    if (@sizeOf(TreeQkNormRopeParams) != 32) @compileError("TreeQkNormRopeParams must be 32 bytes");
    if (@sizeOf(GqaTreeVerifyParams) != 32) @compileError("GqaTreeVerifyParams must be 32 bytes");
    if (@sizeOf(KvTreeCommitParams) != 160) @compileError("KvTreeCommitParams must be 160 bytes");
    if (@sizeOf(KvTreeCommitParams) > metal.MAX_UNIFORM_BYTES) @compileError("KvTreeCommitParams exceeds uniform stride");
}

pub const RmsNormArgs = struct {
    x: metal.BufferBinding,
    /// `[hidden_dim]` norm weights stored as `w_dtype`.
    w: metal.BufferBinding,
    y: metal.BufferBinding,
    num_rows: u32 = 1,
    hidden_dim: u32,
    eps: f32 = 1e-6,
    w_dtype: quant.FloatDType = .f16,
};

pub const QkNormRopeKvArgs = struct {
    /// `[num_q_heads, head_dim]` fp16, updated in-place.
    q: metal.BufferBinding,
    /// `[num_kv_heads, head_dim]` fp16, read-only.
    k: metal.BufferBinding,
    /// `[num_kv_heads, head_dim]` fp16, read-only.
    v: metal.BufferBinding,
    /// Optional `[head_dim]` Qwen3 Q-norm weights in `norm_dtype` (null skips QK-norm).
    q_norm: ?metal.BufferBinding = null,
    /// Optional `[head_dim]` Qwen3 K-norm weights in `norm_dtype` (null skips QK-norm).
    k_norm: ?metal.BufferBinding = null,
    /// Current 512-token chunk's K tile (`RGBA16Float`).
    k_tile: metal.KvTile,
    /// Current 512-token chunk's V tile (`RGBA16Float`).
    v_tile: metal.KvTile,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    pos: u32,
    chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS,
    rms_eps: f32 = 1e-6,
    rope_theta: f32 = 1_000_000.0,
    /// Storage dtype of `q_norm` / `k_norm`.
    norm_dtype: quant.FloatDType = .f16,
};

pub const GqaDecodeArgs = struct {
    /// `[num_q_heads, head_dim]` fp16 query vector.
    q: metal.BufferBinding,
    /// `[num_chunks]` resident `KvTilePair` table (`RGBA16Float` resource IDs).
    kv_tiles: metal.BufferBinding,
    /// `[num_q_heads, head_dim]` fp16 output vector.
    out: metal.BufferBinding,
    /// Optional preallocated split-KV scratch buffer sized by `gqaScratchLayout`.
    /// When null or `seq_len <= 128`, executes single-pass directly into `out`.
    scratch: ?metal.BufferBinding = null,
    max_splits: u32 = MAX_SPLITS,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS,
    /// Explicit softmax scale; defaults to `1 / sqrt(head_dim)` when null.
    scale: ?f32 = null,
};

/// Batched speculative tree QK-Norm + NeoX RoPE arguments (G5).
/// Does not write committed IOSurface `KvTile`s; writes rotated K and V to
/// scratch `k_tree_out` / `v_tree_out` for tree verification and later commit.
pub const TreeQkNormRopeArgs = struct {
    /// `[num_nodes, num_q_heads * head_dim]` fp16, updated in-place.
    q: metal.BufferBinding,
    /// `[num_nodes, num_kv_heads * head_dim]` fp16, read-only.
    k: metal.BufferBinding,
    /// `[num_nodes, num_kv_heads * head_dim]` fp16, read-only.
    v: metal.BufferBinding,
    /// Optional `[head_dim]` Q-norm weights in `norm_dtype` (null skips QK-norm).
    q_norm: ?metal.BufferBinding = null,
    /// Optional `[head_dim]` K-norm weights in `norm_dtype` (null skips QK-norm).
    k_norm: ?metal.BufferBinding = null,
    /// `[num_nodes]u32` absolute token positions (`base_pos + depth[i]`).
    positions: metal.BufferBinding,
    /// `[num_nodes, num_kv_heads, head_dim]` fp16 rotated K output.
    k_tree_out: metal.BufferBinding,
    /// `[num_nodes, num_kv_heads, head_dim]` fp16 V output (may alias `v`).
    v_tree_out: metal.BufferBinding,
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rms_eps: f32 = 1e-6,
    rope_theta: f32 = 1_000_000.0,
    norm_dtype: quant.FloatDType = .f16,
};

/// Speculative GQA tree-attention verification arguments (G5).
pub const GqaTreeVerifyArgs = struct {
    /// `[num_nodes, num_q_heads, head_dim]` fp16 query vectors.
    q: metal.BufferBinding,
    /// `[num_chunks]` resident `KvTilePair` table (`RGBA16Float`). May be null
    /// only when `base_pos == 0`.
    kv_tiles: ?metal.BufferBinding = null,
    /// Number of already-committed prefix tokens `0..base_pos` (may be 0).
    base_pos: u32,
    /// `[num_nodes, num_kv_heads, head_dim]` fp16 tree-local rotated K.
    k_tree: metal.BufferBinding,
    /// `[num_nodes, num_kv_heads, head_dim]` fp16 tree-local V.
    v_tree: metal.BufferBinding,
    /// `[num_nodes]u64` causal ancestor bitmasks (`bit j == 1` iff `j` is in
    /// node `i`'s ancestor chain including `i`).
    ancestor_masks: metal.BufferBinding,
    /// `[num_nodes, num_q_heads, head_dim]` fp16 attention output.
    out: metal.BufferBinding,
    /// Optional preallocated split-KV scratch sized by `gqaTreeScratchLayout`.
    scratch: ?metal.BufferBinding = null,
    max_splits: u32 = MAX_SPLITS,
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS,
    /// Explicit softmax scale; defaults to `1 / sqrt(head_dim)` when null.
    scale: ?f32 = null,
};

/// Accepted tree node index list for `encodeKvTreeCommit`.
pub const AcceptedNodes = union(enum) {
    /// Inline `u16` node indices copied into the uniform ring (zero extra GPU buffer).
    inline_u16: []const u16,
    /// Inline `u32` node indices copied into the uniform ring (zero extra GPU buffer).
    inline_u32: []const u32,
    /// GPU-resident `[accepted_count]u32` buffer binding.
    buffer_u32: metal.BufferBinding,
};

/// Accepted-path KV commit arguments (`ami_kv_tree_commit`, G5).
pub const KvTreeCommitArgs = struct {
    /// `[M, num_kv_heads, head_dim]` fp16 tree-local rotated K (8-byte aligned).
    k_tree: metal.BufferBinding,
    /// `[M, num_kv_heads, head_dim]` fp16 tree-local V (8-byte aligned).
    v_tree: metal.BufferBinding,
    /// Node indices `[0..accepted_count)` along the accepted path.
    accepted_nodes: AcceptedNodes,
    /// Number of accepted nodes to commit (`1 <= accepted_count <= 64`).
    accepted_count: u32,
    /// Chunk tile for `base_pos / chunk_tokens`.
    k_tile0: metal.KvTile,
    v_tile0: metal.KvTile,
    /// Optional next chunk tile for `base_pos / chunk_tokens + 1`, required when
    /// `(base_pos + accepted_count - 1) / chunk_tokens > base_pos / chunk_tokens`.
    k_tile1: ?metal.KvTile = null,
    v_tile1: ?metal.KvTile = null,
    num_kv_heads: u32,
    head_dim: u32,
    base_pos: u32,
    chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS,
};

/// Sub-buffer offsets inside one preallocated FlashDecoding split-KV scratch region.
pub const GqaScratchLayout = struct {
    partial_m_offset: u64,
    partial_l_offset: u64,
    partial_o_offset: u64,
    total_bytes: usize,
};

/// Computes 64-byte-aligned scratch offsets for split-KV GQA decode attention.
pub fn gqaScratchLayout(num_q_heads: u32, head_dim: u32, max_splits: u32) GqaScratchLayout {
    const hs: usize = @as(usize, num_q_heads) * @as(usize, max_splits);
    const m_bytes = hs * @sizeOf(f32);
    const l_off = amf.align64(m_bytes);
    const l_bytes = hs * @sizeOf(f32);
    const o_off = amf.align64(l_off + l_bytes);
    const o_bytes = hs * @as(usize, head_dim) * @sizeOf(f32);
    return .{
        .partial_m_offset = 0,
        .partial_l_offset = l_off,
        .partial_o_offset = o_off,
        .total_bytes = amf.align64(o_off + o_bytes),
    };
}

/// Computes 64-byte-aligned scratch offsets for split-KV GQA tree verification
/// attention (`[max_nodes, num_q_heads, max_splits]` partials).
pub fn gqaTreeScratchLayout(max_nodes: u32, num_q_heads: u32, head_dim: u32, max_splits: u32) GqaScratchLayout {
    const mhs: usize = @as(usize, max_nodes) * @as(usize, num_q_heads) * @as(usize, max_splits);
    const m_bytes = mhs * @sizeOf(f32);
    const l_off = amf.align64(m_bytes);
    const l_bytes = mhs * @sizeOf(f32);
    const o_off = amf.align64(l_off + l_bytes);
    const o_bytes = mhs * @as(usize, head_dim) * @sizeOf(f32);
    return .{
        .partial_m_offset = 0,
        .partial_l_offset = l_off,
        .partial_o_offset = o_off,
        .total_bytes = amf.align64(o_off + o_bytes),
    };
}

/// Chooses the split-KV partition count for FlashDecoding on Apple Silicon.
pub fn chooseNumSplits(num_kv_heads: u32, seq_len: u32, chunk_tokens: u32, max_splits: u32) u32 {
    if (max_splits <= 1 or seq_len <= 128 or num_kv_heads == 0 or chunk_tokens == 0) return 1;
    const max_by_seq: u32 = @max(@as(u32, 1), seq_len / 64);
    const chunks: u32 = (seq_len + chunk_tokens - 1) / chunk_tokens;
    const target_tgs: u32 = if (seq_len >= 1024) 128 else 32;
    const target_splits: u32 = @max((target_tgs + num_kv_heads - 1) / num_kv_heads, chunks);
    return @min(max_splits, @min(max_by_seq, @max(target_splits, 1)));
}

/// Chooses the split-KV partition count for tree verification prefix KV.
pub fn chooseTreeNumSplits(
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    base_pos: u32,
    chunk_tokens: u32,
    max_splits: u32,
) u32 {
    if (max_splits <= 1 or base_pos <= 128 or num_nodes == 0 or num_kv_heads == 0 or chunk_tokens == 0) return 1;
    const gqa_ratio = num_q_heads / num_kv_heads;
    const q_groups_per_kv = (num_nodes * gqa_ratio + 7) / 8;
    const q_tgs: u32 = @max(@as(u32, 1), num_kv_heads * q_groups_per_kv);
    const max_by_seq: u32 = @max(@as(u32, 1), base_pos / 64);
    const chunks: u32 = (base_pos + chunk_tokens - 1) / chunk_tokens;
    const target_tgs: u32 = if (base_pos >= 1024) 128 else 64;
    const target_splits: u32 = @max((target_tgs + q_tgs - 1) / q_tgs, chunks);
    return @min(max_splits, @min(max_by_seq, @max(target_splits, 1)));
}

/// Startup-compiled Metal 4 compute pipelines for RMSNorm, QK-Norm + NeoX RoPE
/// + KV-tile write, and specialized GQA decode attention (`head_dim = 64, 128`).
pub const AttentionPipelines = struct {
    /// `ami_rmsnorm` specialized for `f16` norm weights.
    rmsnorm: metal.Pipeline,
    /// `ami_qknorm_rope_kv_write` specialized for `f16` norm weights.
    qknorm_rope_kv_write: metal.Pipeline,
    rmsnorm_bf16: metal.Pipeline,
    qknorm_rope_kv_write_bf16: metal.Pipeline,
    gqa_split_d64: metal.Pipeline,
    gqa_reduce_d64: metal.Pipeline,
    gqa_split_d128: metal.Pipeline,
    gqa_reduce_d128: metal.Pipeline,
    tree_qknorm_rope: metal.Pipeline,
    tree_qknorm_rope_bf16: metal.Pipeline,
    gqa_tree_split_d64_n16: metal.Pipeline,
    gqa_tree_split_d64_n32: metal.Pipeline,
    gqa_tree_split_d64_n64: metal.Pipeline,
    gqa_tree_split_d128_n16: metal.Pipeline,
    gqa_tree_split_d128_n32: metal.Pipeline,
    gqa_tree_split_d128_n64: metal.Pipeline,
    gqa_tree_reduce_d64: metal.Pipeline,
    gqa_tree_reduce_d128: metal.Pipeline,
    kv_tree_commit: metal.Pipeline,

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!AttentionPipelines {
        const rmsnorm = try metal.Pipeline.initSpecialized(device, library, "ami_rmsnorm", &.{
            .{ .index = 2, .value = .{ .bool = false } },
        });
        errdefer rmsnorm.close();

        const qknorm_rope_kv_write = try metal.Pipeline.initSpecialized(device, library, "ami_qknorm_rope_kv_write_norm", &.{
            .{ .index = 2, .value = .{ .bool = false } },
        });
        errdefer qknorm_rope_kv_write.close();

        const rmsnorm_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_rmsnorm", &.{
            .{ .index = 2, .value = .{ .bool = true } },
        });
        errdefer rmsnorm_bf16.close();

        const qknorm_rope_kv_write_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_qknorm_rope_kv_write_norm", &.{
            .{ .index = 2, .value = .{ .bool = true } },
        });
        errdefer qknorm_rope_kv_write_bf16.close();

        const gqa_split_d64 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_decode_attn_split", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
        });
        errdefer gqa_split_d64.close();

        const gqa_reduce_d64 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_decode_attn_reduce", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
        });
        errdefer gqa_reduce_d64.close();

        const gqa_split_d128 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_decode_attn_split", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
        });
        errdefer gqa_split_d128.close();

        const gqa_reduce_d128 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_decode_attn_reduce", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
        });
        errdefer gqa_reduce_d128.close();

        const tree_qknorm_rope = try metal.Pipeline.initSpecialized(device, library, "ami_tree_qknorm_rope", &.{
            .{ .index = 2, .value = .{ .bool = false } },
        });
        errdefer tree_qknorm_rope.close();

        const tree_qknorm_rope_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_tree_qknorm_rope", &.{
            .{ .index = 2, .value = .{ .bool = true } },
        });
        errdefer tree_qknorm_rope_bf16.close();

        const gqa_tree_split_d64_n16 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_tree_verify_attn_split", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .u32 = 16 } },
        });
        errdefer gqa_tree_split_d64_n16.close();

        const gqa_tree_split_d64_n32 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_tree_verify_attn_split", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .u32 = 32 } },
        });
        errdefer gqa_tree_split_d64_n32.close();

        const gqa_tree_split_d64_n64 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_tree_verify_attn_split", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .u32 = 64 } },
        });
        errdefer gqa_tree_split_d64_n64.close();

        const gqa_tree_split_d128_n16 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_tree_verify_attn_split", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
            .{ .index = 3, .value = .{ .u32 = 16 } },
        });
        errdefer gqa_tree_split_d128_n16.close();

        const gqa_tree_split_d128_n32 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_tree_verify_attn_split", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
            .{ .index = 3, .value = .{ .u32 = 32 } },
        });
        errdefer gqa_tree_split_d128_n32.close();

        const gqa_tree_split_d128_n64 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_tree_verify_attn_split", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
            .{ .index = 3, .value = .{ .u32 = 64 } },
        });
        errdefer gqa_tree_split_d128_n64.close();

        const gqa_tree_reduce_d64 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_tree_verify_attn_reduce", &.{
            .{ .index = 0, .value = .{ .u32 = 64 } },
        });
        errdefer gqa_tree_reduce_d64.close();

        const gqa_tree_reduce_d128 = try metal.Pipeline.initSpecialized(device, library, "ami_gqa_tree_verify_attn_reduce", &.{
            .{ .index = 0, .value = .{ .u32 = 128 } },
        });
        errdefer gqa_tree_reduce_d128.close();

        const kv_tree_commit = try metal.Pipeline.init(device, library, "ami_kv_tree_commit");
        errdefer kv_tree_commit.close();

        return .{
            .rmsnorm = rmsnorm,
            .qknorm_rope_kv_write = qknorm_rope_kv_write,
            .rmsnorm_bf16 = rmsnorm_bf16,
            .qknorm_rope_kv_write_bf16 = qknorm_rope_kv_write_bf16,
            .gqa_split_d64 = gqa_split_d64,
            .gqa_reduce_d64 = gqa_reduce_d64,
            .gqa_split_d128 = gqa_split_d128,
            .gqa_reduce_d128 = gqa_reduce_d128,
            .tree_qknorm_rope = tree_qknorm_rope,
            .tree_qknorm_rope_bf16 = tree_qknorm_rope_bf16,
            .gqa_tree_split_d64_n16 = gqa_tree_split_d64_n16,
            .gqa_tree_split_d64_n32 = gqa_tree_split_d64_n32,
            .gqa_tree_split_d64_n64 = gqa_tree_split_d64_n64,
            .gqa_tree_split_d128_n16 = gqa_tree_split_d128_n16,
            .gqa_tree_split_d128_n32 = gqa_tree_split_d128_n32,
            .gqa_tree_split_d128_n64 = gqa_tree_split_d128_n64,
            .gqa_tree_reduce_d64 = gqa_tree_reduce_d64,
            .gqa_tree_reduce_d128 = gqa_tree_reduce_d128,
            .kv_tree_commit = kv_tree_commit,
        };
    }

    pub fn close(self: AttentionPipelines) void {
        self.kv_tree_commit.close();
        self.gqa_tree_reduce_d128.close();
        self.gqa_tree_reduce_d64.close();
        self.gqa_tree_split_d128_n64.close();
        self.gqa_tree_split_d128_n32.close();
        self.gqa_tree_split_d128_n16.close();
        self.gqa_tree_split_d64_n64.close();
        self.gqa_tree_split_d64_n32.close();
        self.gqa_tree_split_d64_n16.close();
        self.tree_qknorm_rope_bf16.close();
        self.tree_qknorm_rope.close();
        self.gqa_reduce_d128.close();
        self.gqa_split_d128.close();
        self.gqa_reduce_d64.close();
        self.gqa_split_d64.close();
        self.qknorm_rope_kv_write_bf16.close();
        self.rmsnorm_bf16.close();
        self.qknorm_rope_kv_write.close();
        self.rmsnorm.close();
    }

    /// Encodes `ami_rmsnorm` specialized for `args.w_dtype` (zero heap
    /// allocation on the hot path).
    pub fn encodeRmsNorm(
        self: AttentionPipelines,
        frame: *metal.Compute.Frame,
        args: RmsNormArgs,
        insert_barrier_before: bool,
    ) metal.Error!void {
        if (args.num_rows == 0 or args.hidden_dim == 0 or args.eps < 0.0) {
            return error.InvalidArgument;
        }
        const params: RmsNormParams = .{
            .num_rows = args.num_rows,
            .hidden_dim = args.hidden_dim,
            .eps = args.eps,
        };
        const pipe = switch (args.w_dtype) {
            .f16 => self.rmsnorm,
            .bf16 => self.rmsnorm_bf16,
        };
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.w, args.y },
            std.mem.asBytes(&params),
            .{ args.num_rows, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_qknorm_rope_kv_write_norm` specialized for
    /// `args.norm_dtype`, writing `RGBA16Float` K/V texels into the IOSurface
    /// `KvTile`s (ADR-0004). Zero heap allocation on the hot path.
    pub fn encodeQkNormRopeKvWrite(
        self: AttentionPipelines,
        frame: *metal.Compute.Frame,
        args: QkNormRopeKvArgs,
        insert_barrier_before: bool,
    ) metal.Error!void {
        if (args.num_q_heads == 0 or args.num_kv_heads == 0 or args.chunk_tokens == 0 or
            args.head_dim == 0 or args.head_dim > 256 or (args.head_dim % 8) != 0 or
            args.rope_theta <= 0.0 or args.rms_eps < 0.0)
        {
            return error.InvalidArgument;
        }
        const has_qn = args.q_norm != null;
        const has_kn = args.k_norm != null;
        if (has_qn != has_kn) return error.InvalidArgument;

        const qn_binding = args.q_norm orelse args.q;
        const kn_binding = args.k_norm orelse args.q;
        const params: QkNormRopeKvParams = .{
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .pos = args.pos,
            .chunk_tokens = args.chunk_tokens,
            .slot_in_chunk = args.pos % args.chunk_tokens,
            .rms_eps = args.rms_eps,
            .rope_theta = args.rope_theta,
            .apply_qk_norm = if (has_qn) 1 else 0,
        };
        const total_heads: u64 = @as(u64, args.num_q_heads) + @as(u64, args.num_kv_heads);
        const pipe = switch (args.norm_dtype) {
            .f16 => self.qknorm_rope_kv_write,
            .bf16 => self.qknorm_rope_kv_write_bf16,
        };
        try frame.encodeThreadgroupsWithTextures(
            pipe,
            &.{ args.q, args.k, args.v, qn_binding, kn_binding },
            &.{
                .{ .tile = args.k_tile, .view = .rgba16f },
                .{ .tile = args.v_tile, .view = .rgba16f },
            },
            std.mem.asBytes(&params),
            .{ total_heads, 1, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_gqa_decode_attn_split` (and `ami_gqa_decode_attn_reduce`
    /// when split-KV is active). Zero heap allocation on the hot path.
    pub fn encodeGqaDecodeAttn(
        self: AttentionPipelines,
        frame: *metal.Compute.Frame,
        args: GqaDecodeArgs,
        insert_barrier_before: bool,
    ) metal.Error!void {
        if (args.num_q_heads == 0 or args.num_kv_heads == 0 or
            (args.num_q_heads % args.num_kv_heads) != 0 or
            args.seq_len == 0 or args.chunk_tokens == 0 or args.max_splits == 0)
        {
            return error.InvalidArgument;
        }
        const split_pipe, const reduce_pipe = switch (args.head_dim) {
            64 => .{ self.gqa_split_d64, self.gqa_reduce_d64 },
            128 => .{ self.gqa_split_d128, self.gqa_reduce_d128 },
            else => return error.InvalidArgument,
        };

        const gqa_ratio = args.num_q_heads / args.num_kv_heads;
        const q_groups_per_kv = (gqa_ratio + 7) / 8;
        const num_splits: u32 = if (args.scratch != null)
            chooseNumSplits(args.num_kv_heads, args.seq_len, args.chunk_tokens, @min(args.max_splits, MAX_SPLITS))
        else
            1;
        const scale: f32 = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_dim))));

        const params: GqaDecodeParams = .{
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .seq_len = args.seq_len,
            .chunk_tokens = args.chunk_tokens,
            .num_splits = num_splits,
            .q_groups_per_kv = q_groups_per_kv,
            .scale = scale,
        };

        const m_bind, const l_bind, const o_bind = if (args.scratch) |sc| blk: {
            const layout = gqaScratchLayout(args.num_q_heads, args.head_dim, num_splits);
            break :blk .{
                metal.BufferBinding{ .buffer = sc.buffer, .offset = sc.offset + layout.partial_m_offset },
                metal.BufferBinding{ .buffer = sc.buffer, .offset = sc.offset + layout.partial_l_offset },
                metal.BufferBinding{ .buffer = sc.buffer, .offset = sc.offset + layout.partial_o_offset },
            };
        } else .{ args.out, args.out, args.out };

        try frame.encodeThreadgroups(
            split_pipe,
            &.{ args.q, args.kv_tiles, args.out, m_bind, l_bind, o_bind },
            std.mem.asBytes(&params),
            .{ @as(u64, args.num_kv_heads) * q_groups_per_kv, num_splits, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );

        if (num_splits > 1) {
            try frame.encodeThreadgroups(
                reduce_pipe,
                &.{ m_bind, l_bind, o_bind, args.out },
                std.mem.asBytes(&params),
                .{ args.num_q_heads, 1, 1 },
                .{ 32, 1, 1 },
                true,
            );
        }
    }

    /// Encodes batched `ami_tree_qknorm_rope` for `1 <= num_nodes <= 64`.
    /// Updates `q` in-place and writes rotated K and V to `k_tree_out` /
    /// `v_tree_out` without touching committed IOSurface `KvTile`s (G5).
    /// Zero heap allocation on the hot path.
    pub fn encodeTreeQkNormRope(
        self: AttentionPipelines,
        frame: *metal.Compute.Frame,
        args: TreeQkNormRopeArgs,
        insert_barrier_before: bool,
    ) metal.Error!void {
        if (args.num_nodes == 0 or args.num_nodes > MAX_TREE_NODES or
            args.num_q_heads == 0 or args.num_kv_heads == 0 or
            args.head_dim == 0 or args.head_dim > 256 or (args.head_dim % 8) != 0 or
            args.rope_theta <= 0.0 or args.rms_eps < 0.0)
        {
            return error.InvalidArgument;
        }
        const has_qn = args.q_norm != null;
        const has_kn = args.k_norm != null;
        if (has_qn != has_kn) return error.InvalidArgument;

        const qn_binding = args.q_norm orelse args.q;
        const kn_binding = args.k_norm orelse args.q;
        const params: TreeQkNormRopeParams = .{
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .rms_eps = args.rms_eps,
            .rope_theta = args.rope_theta,
            .apply_qk_norm = if (has_qn) 1 else 0,
        };
        const total_heads: u64 = @as(u64, args.num_q_heads) + @as(u64, args.num_kv_heads);
        const pipe = switch (args.norm_dtype) {
            .f16 => self.tree_qknorm_rope,
            .bf16 => self.tree_qknorm_rope_bf16,
        };
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.q, args.k, args.v, qn_binding, kn_binding, args.positions, args.k_tree_out, args.v_tree_out },
            std.mem.asBytes(&params),
            .{ total_heads, args.num_nodes, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_gqa_tree_verify_attn_split` (and `ami_gqa_tree_verify_attn_reduce`
    /// when split-KV over committed prefix tiles is active), specialized on
    /// `ATTN_HEAD_DIM in {64, 128}` and `TREE_MAX_NODES in {16, 32, 64}` (G5, ADR-0014).
    /// Zero heap allocation on the hot path.
    pub fn encodeGqaTreeVerifyAttn(
        self: AttentionPipelines,
        frame: *metal.Compute.Frame,
        args: GqaTreeVerifyArgs,
        insert_barrier_before: bool,
    ) metal.Error!void {
        if (args.num_nodes == 0 or args.num_nodes > MAX_TREE_NODES or
            args.num_q_heads == 0 or args.num_kv_heads == 0 or
            (args.num_q_heads % args.num_kv_heads) != 0 or
            args.chunk_tokens == 0 or args.max_splits == 0)
        {
            return error.InvalidArgument;
        }
        if (args.base_pos > 0 and args.kv_tiles == null) {
            return error.InvalidArgument;
        }

        const split_pipe, const reduce_pipe = switch (args.head_dim) {
            64 => .{
                if (args.num_nodes <= 16)
                    self.gqa_tree_split_d64_n16
                else if (args.num_nodes <= 32)
                    self.gqa_tree_split_d64_n32
                else
                    self.gqa_tree_split_d64_n64,
                self.gqa_tree_reduce_d64,
            },
            128 => .{
                if (args.num_nodes <= 16)
                    self.gqa_tree_split_d128_n16
                else if (args.num_nodes <= 32)
                    self.gqa_tree_split_d128_n32
                else
                    self.gqa_tree_split_d128_n64,
                self.gqa_tree_reduce_d128,
            },
            else => return error.InvalidArgument,
        };

        const gqa_ratio = args.num_q_heads / args.num_kv_heads;
        const q_groups_per_kv = (args.num_nodes * gqa_ratio + 7) / 8;
        const num_splits: u32 = if (args.scratch != null and args.base_pos > 0)
            chooseTreeNumSplits(
                args.num_nodes,
                args.num_q_heads,
                args.num_kv_heads,
                args.base_pos,
                args.chunk_tokens,
                @min(args.max_splits, MAX_SPLITS),
            )
        else
            1;
        const scale: f32 = args.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(args.head_dim))));

        const params: GqaTreeVerifyParams = .{
            .num_nodes = args.num_nodes,
            .num_q_heads = args.num_q_heads,
            .num_kv_heads = args.num_kv_heads,
            .base_pos = args.base_pos,
            .chunk_tokens = args.chunk_tokens,
            .num_splits = num_splits,
            .q_groups_per_kv = q_groups_per_kv,
            .scale = scale,
        };

        const kv_tiles_bind = args.kv_tiles orelse args.q;
        const m_bind, const l_bind, const o_bind = if (args.scratch) |sc| blk: {
            const layout = gqaTreeScratchLayout(args.num_nodes, args.num_q_heads, args.head_dim, num_splits);
            break :blk .{
                metal.BufferBinding{ .buffer = sc.buffer, .offset = sc.offset + layout.partial_m_offset },
                metal.BufferBinding{ .buffer = sc.buffer, .offset = sc.offset + layout.partial_l_offset },
                metal.BufferBinding{ .buffer = sc.buffer, .offset = sc.offset + layout.partial_o_offset },
            };
        } else .{ args.out, args.out, args.out };

        try frame.encodeThreadgroups(
            split_pipe,
            &.{ args.q, kv_tiles_bind, args.k_tree, args.v_tree, args.ancestor_masks, args.out, m_bind, l_bind, o_bind },
            std.mem.asBytes(&params),
            .{ @as(u64, args.num_kv_heads) * q_groups_per_kv, num_splits, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );

        if (num_splits > 1) {
            try frame.encodeThreadgroups(
                reduce_pipe,
                &.{ m_bind, l_bind, o_bind, args.out },
                std.mem.asBytes(&params),
                .{ args.num_q_heads, args.num_nodes, 1 },
                .{ 32, 1, 1 },
                true,
            );
        }
    }

    /// Encodes `ami_kv_tree_commit`, copying the accepted path's rotated K and V
    /// rows from `k_tree`/`v_tree` into committed IOSurface `RGBA16Float`
    /// `KvTile`s at sequence positions `base_pos .. base_pos + accepted_count`,
    /// cleanly handling a `chunk_tokens` boundary across `tile0` and `tile1` in
    /// one GPU dispatch (G5). Zero heap allocation on the hot path.
    pub fn encodeKvTreeCommit(
        self: AttentionPipelines,
        frame: *metal.Compute.Frame,
        args: KvTreeCommitArgs,
        insert_barrier_before: bool,
    ) metal.Error!void {
        if (args.accepted_count == 0 or args.accepted_count > MAX_TREE_NODES or
            args.num_kv_heads == 0 or args.head_dim == 0 or (args.head_dim % 4) != 0 or
            args.chunk_tokens == 0 or
            (args.k_tree.offset % 8) != 0 or (args.v_tree.offset % 8) != 0)
        {
            return error.InvalidArgument;
        }
        const chunk0 = args.base_pos / args.chunk_tokens;
        const last_pos = args.base_pos + args.accepted_count - 1;
        const last_chunk = last_pos / args.chunk_tokens;
        if (last_chunk > chunk0 + 1) {
            return error.InvalidArgument;
        }
        const crosses_chunk = last_chunk > chunk0;
        if (crosses_chunk and (args.k_tile1 == null or args.v_tile1 == null)) {
            return error.InvalidArgument;
        }

        var params: KvTreeCommitParams = .{
            .num_kv_heads = args.num_kv_heads,
            .head_dim = args.head_dim,
            .base_pos = args.base_pos,
            .chunk_tokens = args.chunk_tokens,
            .accepted_count = args.accepted_count,
            .use_node_buffer = 0,
            .has_second_chunk = if (crosses_chunk) 1 else 0,
        };

        const nodes_bind: metal.BufferBinding = switch (args.accepted_nodes) {
            .inline_u16 => |sl| blk: {
                if (sl.len < args.accepted_count) return error.InvalidArgument;
                for (0..args.accepted_count) |i| params.inline_nodes[i] = sl[i];
                break :blk args.k_tree;
            },
            .inline_u32 => |sl| blk: {
                if (sl.len < args.accepted_count) return error.InvalidArgument;
                for (0..args.accepted_count) |i| {
                    params.inline_nodes[i] = std.math.cast(u16, sl[i]) orelse return error.InvalidArgument;
                }
                break :blk args.k_tree;
            },
            .buffer_u32 => |b| blk: {
                params.use_node_buffer = 1;
                break :blk b;
            },
        };

        const k_tile1 = args.k_tile1 orelse args.k_tile0;
        const v_tile1 = args.v_tile1 orelse args.v_tile0;

        try frame.encodeThreadgroupsWithTextures(
            self.kv_tree_commit,
            &.{ args.k_tree, args.v_tree, nodes_bind },
            &.{
                .{ .tile = args.k_tile0, .view = .rgba16f },
                .{ .tile = args.v_tile0, .view = .rgba16f },
                .{ .tile = k_tile1, .view = .rgba16f },
                .{ .tile = v_tile1, .view = .rgba16f },
            },
            std.mem.asBytes(&params),
            .{ args.num_kv_heads, args.accepted_count, 1 },
            .{ 32, 1, 1 },
            insert_barrier_before,
        );
    }
};

// ---------------------------------------------------------------------------
// Spike S1 Microbenchmark (R4 / ADR-0004)
// ---------------------------------------------------------------------------

pub const SpikeS1Layout = enum {
    token_major,
    channel_major,

    pub fn label(self: SpikeS1Layout) []const u8 {
        return switch (self) {
            .token_major => "token_major",
            .channel_major => "channel_major",
        };
    }
};

pub const SpikeS1Access = enum {
    rgba16f,
    r16f,
    alias_buffer,

    pub fn label(self: SpikeS1Access) []const u8 {
        return switch (self) {
            .rgba16f => "rgba16f",
            .r16f => "r16f",
            .alias_buffer => "alias_buf",
        };
    }

    fn modeCode(self: SpikeS1Access) u32 {
        return switch (self) {
            .rgba16f => 0,
            .r16f => 1,
            .alias_buffer => 2,
        };
    }
};

pub const SpikeS1Config = struct {
    num_q_heads: u32 = 32,
    num_kv_heads: u32 = 4,
    head_dim: u32 = 128,
    chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS,
    /// Chunks per context bank (32 chunks * 512 tokens = 16,384 context = 32 MiB K+V).
    num_chunks: u32 = 32,
    /// Rotated context banks to stream past M4 L2/SLC (2 banks = 64 MiB K+V).
    num_banks: u32 = 2,
    /// Extra GPU spin-up batches before the first timed case so M4 DVFS reaches peak clock.
    dvfs_warmup_batches: u32 = 8,
    warmup_batches: u32 = 4,
    timed_batches: u32 = 8,
    dispatches_per_batch: u32 = 8,
    timeout_ms: u64 = 30_000,
};

pub const SpikeS1CaseResult = struct {
    layout: SpikeS1Layout,
    access: SpikeS1Access,
    bytes_per_dispatch: u64,
    dispatches: u64,
    elapsed_ns: u64,

    pub fn latencyUs(self: SpikeS1CaseResult) f64 {
        if (self.dispatches == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.elapsed_ns)) / @as(f64, @floatFromInt(self.dispatches))) / 1000.0;
    }

    /// Effective KV bandwidth in decimal GB/s (`bytes_moved / elapsed_ns`).
    pub fn effectiveBandwidthGbs(self: SpikeS1CaseResult) f64 {
        if (self.elapsed_ns == 0) return 0.0;
        const total_bytes = @as(f64, @floatFromInt(self.bytes_per_dispatch)) * @as(f64, @floatFromInt(self.dispatches));
        return total_bytes / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    pub fn busUtilizationPct(self: SpikeS1CaseResult) f64 {
        return (self.effectiveBandwidthGbs() / M4_BUS_BANDWIDTH_GBS) * 100.0;
    }
};

pub const SpikeS1Report = struct {
    config: SpikeS1Config,
    tile_data_bytes: u64,
    tile_alloc_bytes: u64,
    tile_surface_alloc_size: u64,
    tile_bytes_per_row: u32,
    iosurface_mlock_supported: bool,
    results: [6]SpikeS1CaseResult,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn write(self: *const SpikeS1Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        const context_tokens: u64 = @as(u64, self.config.num_chunks) * self.config.chunk_tokens;
        try out.print(
            \\Spike S1 IOSurface KV-Tile Read Bandwidth
            \\s1.geometry                      q={d},kv={d},d={d},chunk={d},chunks={d} (ctx={d}),banks={d}
            \\s1.tile_bytes_per_row            {d}
            \\s1.tile_data_bytes               {d}
            \\s1.tile_alloc_bytes              {d}
            \\s1.tile_surface_alloc_size       {d}
            \\s1.iosurface_mlock_supported     {s}
            \\layout          access      bytes/disp   latency(us)   beta(GB/s)   bus(%)
            \\
        , .{
            self.config.num_q_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            self.config.chunk_tokens,
            self.config.num_chunks,
            context_tokens,
            self.config.num_banks,
            self.tile_bytes_per_row,
            self.tile_data_bytes,
            self.tile_alloc_bytes,
            self.tile_surface_alloc_size,
            if (self.iosurface_mlock_supported) "yes" else "no",
        });

        for (self.results) |res| {
            try out.print(
                \\{s: <15} {s: <11} {d: >10}   {d: >11.2}   {d: >10.2}   {d: >6.1}
                \\
            , .{
                res.layout.label(),
                res.access.label(),
                res.bytes_per_dispatch,
                res.latencyUs(),
                res.effectiveBandwidthGbs(),
                res.busUtilizationPct(),
            });
        }

        try out.print(
            \\s1.swapouts_delta                {d}
            \\s1.pageouts_delta                {d}
            \\
        , .{
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
        });
    }
};

pub const BenchError = error{
    InvalidConfig,
    NonFiniteOutput,
    OutOfMemory,
} || vmstat.Error || vmstat.SwapError || sysinfo.Error || metal.WrapError ||
    arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError;

fn fillTileDeterministic(tile: metal.KvTile, seed: u64) metal.Error!void {
    var guard = try tile.lock(false);
    errdefer guard.release();
    const table = [_]f16{
        @floatCast(0.03125),
        @floatCast(-0.0234375),
        @floatCast(0.046875),
        @floatCast(-0.015625),
        @floatCast(0.01953125),
        @floatCast(-0.0390625),
        @floatCast(0.0546875),
        @floatCast(-0.03125),
    };
    const vals = guard.f16Slice();
    const base_idx: usize = @intCast(seed & 7);
    for (vals, 0..) |*dst, i| {
        dst.* = table[(i + base_idx) & 7];
    }
    try guard.unlock();
}

/// Runs Spike S1 comparing M4 decode-attention KV read bandwidth across
/// `{token_major, channel_major}` x `{RGBA16Float, R16Float, alias MTLBuffer}`
/// and probes `mlock` support on IOSurface-backed `KvTile`s.
pub fn runSpikeS1(allocator: std.mem.Allocator, config: SpikeS1Config) BenchError!SpikeS1Report {
    if (config.head_dim != 128 or config.chunk_tokens != 512 or
        config.num_q_heads == 0 or config.num_kv_heads == 0 or
        (config.num_q_heads % config.num_kv_heads) != 0 or
        (config.num_q_heads / config.num_kv_heads) > 8 or
        config.num_chunks == 0 or config.num_banks == 0 or
        config.timed_batches == 0 or config.dispatches_per_batch == 0 or
        config.dispatches_per_batch > metal.UNIFORM_SLOTS_PER_FRAME)
    {
        return error.InvalidConfig;
    }

    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const dev_info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, dev_info, .{});

    const library = try metal.Library.openEmbedded(device);
    defer library.close();

    // Compile all 6 specialized Spike S1 pipelines (2 layouts x 3 access modes).
    const accesses = [_]SpikeS1Access{ .rgba16f, .r16f, .alias_buffer };
    const layouts = [_]SpikeS1Layout{ .token_major, .channel_major };

    var pipes: [2][3]metal.Pipeline = undefined;
    var pipes_init: usize = 0;
    defer {
        for (0..pipes_init) |idx| {
            pipes[idx / 3][idx % 3].close();
        }
    }
    for (layouts, 0..) |layout, li| {
        const fn_name: [:0]const u8 = switch (layout) {
            .token_major => "ami_spike_s1_token_major",
            .channel_major => "ami_spike_s1_channel_major",
        };
        for (accesses, 0..) |acc, ai| {
            pipes[li][ai] = try metal.Pipeline.initSpecialized(device, library, fn_name, &.{
                .{ .index = 1, .value = .{ .u32 = acc.modeCode() } },
            });
            pipes_init += 1;
        }
    }

    // Probe whether macOS permits mlock() over IOSurface-backed VA ranges.
    const iosurface_mlock_supported = blk: {
        const probe_tile = metal.KvTile.init(device, 128, 64, .{ .mlock = true }) catch |err| switch (err) {
            error.TileCreationFailed => break :blk false,
            else => return err,
        };
        defer probe_tile.close();
        break :blk (try probe_tile.info()).isMlocked();
    };

    const vm_before = try vmstat.Snapshot.capture();

    // Wired arena holds Q, bindless KvTilePair/KvBufPair tables for all banks, and partial O.
    const q_bytes = amf.align64(@as(usize, config.num_q_heads) * config.head_dim * @sizeOf(f16));
    const bank_table_bytes = amf.align64(@as(usize, config.num_chunks) * @sizeOf(KvTilePair));
    const rgba_tables_bytes = bank_table_bytes * config.num_banks;
    const r16_tables_bytes = bank_table_bytes * config.num_banks;
    const buf_tables_bytes = bank_table_bytes * config.num_banks;
    const out_elems: usize = @as(usize, config.num_q_heads) * config.num_chunks * config.head_dim;
    const out_bytes = amf.align64(out_elems * @sizeOf(f16));

    const arena_bytes = try page.alignUp(
        q_bytes + rgba_tables_bytes + r16_tables_bytes + buf_tables_bytes + out_bytes + 4 * page.APPLE_PAGE_SIZE,
    );
    var arena = try arena_mod.WiredArena.init(arena_bytes, wired_budget);
    defer arena.deinit();

    const q_raw = try arena.allocAligned(q_bytes, 64);
    const rgba_tbl_raw = try arena.allocAligned(rgba_tables_bytes, 64);
    const r16_tbl_raw = try arena.allocAligned(r16_tables_bytes, 64);
    const buf_tbl_raw = try arena.allocAligned(buf_tables_bytes, 64);
    const out_raw = try arena.allocAligned(out_bytes, 64);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
    for (q_f16, 0..) |*v, i| {
        const centered: i32 = @as(i32, @intCast((i * 13 + 5) % 15)) - 7;
        v.* = @floatCast(@as(f32, @floatFromInt(centered)) * 0.0625);
    }
    @memset(out_raw, 0);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const total_tiles_per_layout: usize = @as(usize, config.num_banks) * config.num_chunks * 2;
    const tiles = try allocator.alloc(metal.KvTile, total_tiles_per_layout);
    defer allocator.free(tiles);

    const kv_bytes_per_dispatch: u64 =
        @as(u64, config.num_chunks) * 2 * config.num_kv_heads * config.chunk_tokens * config.head_dim * @sizeOf(f16);
    const params: SpikeS1Params = .{
        .num_q_heads = config.num_q_heads,
        .num_kv_heads = config.num_kv_heads,
        .head_dim = config.head_dim,
        .chunk_tokens = config.chunk_tokens,
        .num_chunks = config.num_chunks,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.head_dim))),
    };

    const q_off: u64 = arena.offsetOf(q_raw.ptr);
    const rgba_tbl_base: u64 = arena.offsetOf(rgba_tbl_raw.ptr);
    const r16_tbl_base: u64 = arena.offsetOf(r16_tbl_raw.ptr);
    const buf_tbl_base: u64 = arena.offsetOf(buf_tbl_raw.ptr);
    const out_off: u64 = arena.offsetOf(out_raw.ptr);
    const out_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));

    var results: [6]SpikeS1CaseResult = undefined;
    var tile_data_bytes: u64 = 0;
    var tile_alloc_bytes: u64 = 0;
    var tile_surface_alloc_size: u64 = 0;
    var tile_bytes_per_row: u32 = 0;

    for (layouts, 0..) |layout, li| {
        const row_f16: u32 = switch (layout) {
            .token_major => config.head_dim,
            .channel_major => config.chunk_tokens,
        };
        const rows: u32 = switch (layout) {
            .token_major => config.num_kv_heads * config.chunk_tokens,
            .channel_major => config.num_kv_heads * config.head_dim,
        };

        var created: usize = 0;
        var resident = false;
        defer {
            if (resident) {
                compute.removeTilesResident(tiles[0..created]) catch |err| {
                    std.debug.panic("removeTilesResident failed during cleanup: {}", .{err});
                };
            }
            for (0..created) |i| tiles[i].close();
        }
        for (0..total_tiles_per_layout) |i| {
            tiles[i] = try metal.KvTile.init(device, row_f16, rows, .{ .alias_buffer = true });
            created += 1;
            try fillTileDeterministic(tiles[i], i + li * 17);
        }

        if (li == 0) {
            const meta = try tiles[0].info();
            tile_data_bytes = meta.dataBytes();
            tile_alloc_bytes = meta.allocBytes();
            tile_surface_alloc_size = meta.surfaceAllocSize();
            tile_bytes_per_row = meta.bytesPerRow();
        }

        try compute.makeTilesResident(tiles);
        resident = true;

        // Populate bindless chunk tables for each rotated bank.
        for (0..config.num_banks) |bank| {
            const tbl_off = bank * bank_table_bytes;
            const rgba_slice: []KvTilePair = @alignCast(std.mem.bytesAsSlice(
                KvTilePair,
                rgba_tbl_raw[tbl_off .. tbl_off + @as(usize, config.num_chunks) * @sizeOf(KvTilePair)],
            ));
            const r16_slice: []KvTilePair = @alignCast(std.mem.bytesAsSlice(
                KvTilePair,
                r16_tbl_raw[tbl_off .. tbl_off + @as(usize, config.num_chunks) * @sizeOf(KvTilePair)],
            ));
            const buf_slice: []KvBufPair = @alignCast(std.mem.bytesAsSlice(
                KvBufPair,
                buf_tbl_raw[tbl_off .. tbl_off + @as(usize, config.num_chunks) * @sizeOf(KvBufPair)],
            ));
            for (0..config.num_chunks) |c_idx| {
                const base_tile = (bank * config.num_chunks + c_idx) * 2;
                const k_tile = tiles[base_tile];
                const v_tile = tiles[base_tile + 1];
                rgba_slice[c_idx] = KvTilePair.fromTiles(k_tile, v_tile, .rgba16f);
                r16_slice[c_idx] = KvTilePair.fromTiles(k_tile, v_tile, .r16f);
                buf_slice[c_idx] = try KvBufPair.fromTiles(k_tile, v_tile);
            }
        }

        const tg_threads: u64 = switch (layout) {
            .token_major => 256,
            .channel_major => 128,
        };

        for (accesses, 0..) |acc, ai| {
            const pipe = pipes[li][ai];
            const tex_tbl_base = switch (acc) {
                .rgba16f, .alias_buffer => rgba_tbl_base,
                .r16f => r16_tbl_base,
            };

            var bank_cursor: usize = 0;
            const warmups = config.warmup_batches + if (li == 0 and ai == 0) config.dvfs_warmup_batches else 0;
            var warmup_sig: u64 = 0;
            for (0..warmups) |_| {
                var frame = try compute.beginFrame(config.timeout_ms);
                errdefer frame.abort();
                for (0..config.dispatches_per_batch) |d| {
                    const bank = bank_cursor % config.num_banks;
                    bank_cursor += 1;
                    const tbl_off: u64 = @as(u64, bank) * bank_table_bytes;
                    try frame.encodeThreadgroups(
                        pipe,
                        &.{
                            .{ .buffer = gpu_buf, .offset = q_off },
                            .{ .buffer = gpu_buf, .offset = tex_tbl_base + tbl_off },
                            .{ .buffer = gpu_buf, .offset = buf_tbl_base + tbl_off },
                            .{ .buffer = gpu_buf, .offset = out_off },
                        },
                        std.mem.asBytes(&params),
                        .{ config.num_kv_heads, config.num_chunks, 1 },
                        .{ tg_threads, 1, 1 },
                        d > 0,
                    );
                }
                warmup_sig = try frame.commit();
            }
            if (warmup_sig != 0) {
                try compute.wait(warmup_sig, config.timeout_ms);
            }

            var last_sig: u64 = 0;
            const t0 = try vmstat.nowNs();
            for (0..config.timed_batches) |_| {
                var frame = try compute.beginFrame(config.timeout_ms);
                errdefer frame.abort();
                for (0..config.dispatches_per_batch) |d| {
                    const bank = bank_cursor % config.num_banks;
                    bank_cursor += 1;
                    const tbl_off: u64 = @as(u64, bank) * bank_table_bytes;
                    try frame.encodeThreadgroups(
                        pipe,
                        &.{
                            .{ .buffer = gpu_buf, .offset = q_off },
                            .{ .buffer = gpu_buf, .offset = tex_tbl_base + tbl_off },
                            .{ .buffer = gpu_buf, .offset = buf_tbl_base + tbl_off },
                            .{ .buffer = gpu_buf, .offset = out_off },
                        },
                        std.mem.asBytes(&params),
                        .{ config.num_kv_heads, config.num_chunks, 1 },
                        .{ tg_threads, 1, 1 },
                        d > 0,
                    );
                }
                last_sig = try frame.commit();
            }
            try compute.wait(last_sig, config.timeout_ms);
            const t1 = try vmstat.nowNs();

            if (!std.math.isFinite(out_f16[0]) or !std.math.isFinite(out_f16[out_elems - 1])) {
                return error.NonFiniteOutput;
            }

            results[li * 3 + ai] = .{
                .layout = layout,
                .access = acc,
                .bytes_per_dispatch = kv_bytes_per_dispatch,
                .dispatches = @as(u64, config.timed_batches) * config.dispatches_per_batch,
                .elapsed_ns = t1 - t0,
            };
        }
    }

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .config = config,
        .tile_data_bytes = tile_data_bytes,
        .tile_alloc_bytes = tile_alloc_bytes,
        .tile_surface_alloc_size = tile_surface_alloc_size,
        .tile_bytes_per_row = tile_bytes_per_row,
        .iosurface_mlock_supported = iosurface_mlock_supported,
        .results = results,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

// ---------------------------------------------------------------------------
// Production G4 GQA KV-Tile Decode Attention Benchmark
// ---------------------------------------------------------------------------

pub const GqaBenchCase = struct {
    num_q_heads: u32 = 32,
    num_kv_heads: u32 = 4,
    head_dim: u32 = 128,
    seq_len: u32,
    chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS,
};

pub const DEFAULT_GQA_BENCH_CASES = [_]GqaBenchCase{
    .{ .seq_len = 512 },
    .{ .seq_len = 4096 },
    .{ .seq_len = 16384 },
};

pub const GqaBenchConfig = struct {
    cases: []const GqaBenchCase = &DEFAULT_GQA_BENCH_CASES,
    min_working_set_bytes: usize = 64 * 1024 * 1024,
    dvfs_warmup_batches: u32 = 8,
    warmup_batches: u32 = 4,
    timed_batches: u32 = 8,
    dispatches_per_batch: u32 = 8,
    timeout_ms: u64 = 30_000,
};

pub const GqaBenchCaseResult = struct {
    case: GqaBenchCase,
    num_chunks: u32,
    num_splits: u32,
    banks: usize,
    bytes_per_dispatch: u64,
    dispatches: u64,
    elapsed_ns: u64,

    pub fn latencyUs(self: GqaBenchCaseResult) f64 {
        if (self.dispatches == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.elapsed_ns)) / @as(f64, @floatFromInt(self.dispatches))) / 1000.0;
    }

    pub fn effectiveBandwidthGbs(self: GqaBenchCaseResult) f64 {
        if (self.elapsed_ns == 0) return 0.0;
        const total_bytes = @as(f64, @floatFromInt(self.bytes_per_dispatch)) * @as(f64, @floatFromInt(self.dispatches));
        return total_bytes / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    pub fn busUtilizationPct(self: GqaBenchCaseResult) f64 {
        return (self.effectiveBandwidthGbs() / M4_BUS_BANDWIDTH_GBS) * 100.0;
    }
};

pub const GqaBenchReport = struct {
    results: []GqaBenchCaseResult,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn deinit(self: *GqaBenchReport, allocator: std.mem.Allocator) void {
        allocator.free(self.results);
        self.results = &.{};
    }

    pub fn write(self: *const GqaBenchReport, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\G4 GQA Decode Attention over RGBA16Float IOSurface KV Tiles
            \\seq_len   chunks   splits   banks   bytes/disp   latency(us)   beta(GB/s)   bus(%)
            \\
        , .{});
        for (self.results) |res| {
            try out.print(
                \\{d: <9} {d: >6}   {d: >6}   {d: >5}   {d: >10}   {d: >11.2}   {d: >10.2}   {d: >6.1}
                \\
            , .{
                res.case.seq_len,
                res.num_chunks,
                res.num_splits,
                res.banks,
                res.bytes_per_dispatch,
                res.latencyUs(),
                res.effectiveBandwidthGbs(),
                res.busUtilizationPct(),
            });
        }
        try out.print(
            \\g4.swapouts_delta                {d}
            \\g4.pageouts_delta                {d}
            \\
        , .{
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
        });
    }
};

/// Benchmarks production `AttentionPipelines.encodeGqaDecodeAttn` across
/// context lengths (`seq_len = 512, 4096, 16384`) over rotated IOSurface KV tiles.
pub fn runGqaBench(allocator: std.mem.Allocator, config: GqaBenchConfig) BenchError!GqaBenchReport {
    if (config.timed_batches == 0 or config.dispatches_per_batch == 0 or
        config.dispatches_per_batch * 2 > metal.UNIFORM_SLOTS_PER_FRAME)
    {
        return error.InvalidConfig;
    }

    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const dev_info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, dev_info, .{});

    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try AttentionPipelines.init(device, library);
    defer pipes.close();

    const results = try allocator.alloc(GqaBenchCaseResult, config.cases.len);
    errdefer allocator.free(results);

    const vm_before = try vmstat.Snapshot.capture();

    for (config.cases, 0..) |case, case_idx| {
        if (case.seq_len == 0 or case.chunk_tokens == 0 or case.num_kv_heads == 0 or
            case.num_q_heads == 0 or (case.num_q_heads % case.num_kv_heads) != 0)
        {
            return error.InvalidConfig;
        }
        const num_chunks: u32 = (case.seq_len + case.chunk_tokens - 1) / case.chunk_tokens;
        const num_splits: u32 = chooseNumSplits(case.num_kv_heads, case.seq_len, case.chunk_tokens, MAX_SPLITS);
        const kv_bytes_per_ctx: u64 =
            @as(u64, case.seq_len) * 2 * case.num_kv_heads * case.head_dim * @sizeOf(f16);
        const tile_pair_bytes: usize =
            @as(usize, case.num_kv_heads) * case.chunk_tokens * case.head_dim * @sizeOf(f16) * 2;
        const bank_bytes: usize = tile_pair_bytes * num_chunks;
        const num_banks: usize = if (config.min_working_set_bytes == 0)
            1
        else
            @max(@as(usize, 1), (config.min_working_set_bytes + bank_bytes - 1) / bank_bytes);

        const q_bytes = amf.align64(@as(usize, case.num_q_heads) * case.head_dim * @sizeOf(f16));
        const out_bytes = q_bytes;
        const scratch_layout = gqaScratchLayout(case.num_q_heads, case.head_dim, num_splits);
        const bank_tbl_bytes = amf.align64(@as(usize, num_chunks) * @sizeOf(KvTilePair));
        const all_tbl_bytes = bank_tbl_bytes * num_banks;

        const arena_bytes = try page.alignUp(
            q_bytes + out_bytes + scratch_layout.total_bytes + all_tbl_bytes + 4 * page.APPLE_PAGE_SIZE,
        );
        var arena = try arena_mod.WiredArena.init(arena_bytes, wired_budget);
        defer arena.deinit();

        const q_raw = try arena.allocAligned(q_bytes, 64);
        const out_raw = try arena.allocAligned(out_bytes, 64);
        const scratch_raw = try arena.allocAligned(scratch_layout.total_bytes, 64);
        const tbl_raw = try arena.allocAligned(all_tbl_bytes, 64);

        const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
        for (q_f16, 0..) |*v, i| {
            const centered: i32 = @as(i32, @intCast((i * 11 + 3) % 15)) - 7;
            v.* = @floatCast(@as(f32, @floatFromInt(centered)) * 0.0625);
        }
        @memset(out_raw, 0);

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();

        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        const total_tiles: usize = num_banks * num_chunks * 2;
        const tiles = try allocator.alloc(metal.KvTile, total_tiles);
        defer allocator.free(tiles);

        var created: usize = 0;
        var resident = false;
        defer {
            if (resident) {
                compute.removeTilesResident(tiles[0..created]) catch |err| {
                    std.debug.panic("removeTilesResident failed during cleanup: {}", .{err});
                };
            }
            for (0..created) |i| tiles[i].close();
        }
        for (0..total_tiles) |i| {
            tiles[i] = try metal.KvTile.init(device, case.head_dim, case.num_kv_heads * case.chunk_tokens, .{});
            created += 1;
            try fillTileDeterministic(tiles[i], i + case_idx * 13);
        }

        try compute.makeTilesResident(tiles);
        resident = true;

        for (0..num_banks) |bank| {
            const off = bank * bank_tbl_bytes;
            const tbl_slice: []KvTilePair = @alignCast(std.mem.bytesAsSlice(
                KvTilePair,
                tbl_raw[off .. off + @as(usize, num_chunks) * @sizeOf(KvTilePair)],
            ));
            for (0..num_chunks) |c_idx| {
                const t_base = (bank * num_chunks + c_idx) * 2;
                tbl_slice[c_idx] = KvTilePair.fromTiles(tiles[t_base], tiles[t_base + 1], .rgba16f);
            }
        }

        const q_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) };
        const out_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) };
        const scratch_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = arena.offsetOf(scratch_raw.ptr) };
        const tbl_base: u64 = arena.offsetOf(tbl_raw.ptr);
        const out_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));

        var bank_cursor: usize = 0;
        const warmups = config.warmup_batches + if (case_idx == 0) config.dvfs_warmup_batches else 0;
        var warmup_sig: u64 = 0;
        for (0..warmups) |_| {
            var frame = try compute.beginFrame(config.timeout_ms);
            errdefer frame.abort();
            for (0..config.dispatches_per_batch) |d| {
                const bank = bank_cursor % num_banks;
                bank_cursor += 1;
                try pipes.encodeGqaDecodeAttn(&frame, .{
                    .q = q_bind,
                    .kv_tiles = .{ .buffer = gpu_buf, .offset = tbl_base + @as(u64, bank) * bank_tbl_bytes },
                    .out = out_bind,
                    .scratch = scratch_bind,
                    .max_splits = num_splits,
                    .num_q_heads = case.num_q_heads,
                    .num_kv_heads = case.num_kv_heads,
                    .head_dim = case.head_dim,
                    .seq_len = case.seq_len,
                    .chunk_tokens = case.chunk_tokens,
                }, d > 0);
            }
            warmup_sig = try frame.commit();
        }
        if (warmup_sig != 0) try compute.wait(warmup_sig, config.timeout_ms);

        var last_sig: u64 = 0;
        const t0 = try vmstat.nowNs();
        for (0..config.timed_batches) |_| {
            var frame = try compute.beginFrame(config.timeout_ms);
            errdefer frame.abort();
            for (0..config.dispatches_per_batch) |d| {
                const bank = bank_cursor % num_banks;
                bank_cursor += 1;
                try pipes.encodeGqaDecodeAttn(&frame, .{
                    .q = q_bind,
                    .kv_tiles = .{ .buffer = gpu_buf, .offset = tbl_base + @as(u64, bank) * bank_tbl_bytes },
                    .out = out_bind,
                    .scratch = scratch_bind,
                    .max_splits = num_splits,
                    .num_q_heads = case.num_q_heads,
                    .num_kv_heads = case.num_kv_heads,
                    .head_dim = case.head_dim,
                    .seq_len = case.seq_len,
                    .chunk_tokens = case.chunk_tokens,
                }, d > 0);
            }
            last_sig = try frame.commit();
        }
        try compute.wait(last_sig, config.timeout_ms);
        const t1 = try vmstat.nowNs();

        const out_count: usize = @as(usize, case.num_q_heads) * case.head_dim;
        if (!std.math.isFinite(out_f16[0]) or !std.math.isFinite(out_f16[out_count - 1])) {
            return error.NonFiniteOutput;
        }

        results[case_idx] = .{
            .case = case,
            .num_chunks = num_chunks,
            .num_splits = num_splits,
            .banks = num_banks,
            .bytes_per_dispatch = kv_bytes_per_ctx,
            .dispatches = @as(u64, config.timed_batches) * config.dispatches_per_batch,
            .elapsed_ns = t1 - t0,
        };
    }

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .results = results,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

// ---------------------------------------------------------------------------
// Spike S4: CoreML ANE Zero-Copy Handoff, D5 Draft-Placement Gate & Contention
// ---------------------------------------------------------------------------

pub const SpikeS4Placement = enum {
    gpu,
    ane,

    pub fn label(self: SpikeS4Placement) []const u8 {
        return switch (self) {
            .gpu => "gpu",
            .ane => "ane",
        };
    }
};

pub const SpikeS4Config = struct {
    /// Tiny handoff probe dims (`conv1x1 [handoff_c, 1, handoff_s]`).
    handoff_c: u32 = 64,
    handoff_s: u32 = 32,
    handoff_warmup_iters: u32 = 8,
    handoff_timed_iters: u32 = 40,

    /// Representative draft-head linear projection (`[draft_hidden, draft_hidden]` fp16).
    draft_hidden: u32 = 2048,
    draft_layers: u32 = 2,
    /// Batched draft frontier width (`conv1x1` spatial width, multiple of 32).
    draft_batch: u32 = 32,
    draft_warmup_iters: u32 = 6,
    draft_timed_iters: u32 = 24,
    gpu_dispatches_per_batch: u32 = 16,

    /// Concurrent ANE DRAM-streaming model (default 5 x 2048x2048 fp16 = 40 MiB > M4 SRAM cliff).
    stream_hidden: u32 = 2048,
    stream_layers: u32 = 5,
    stream_spatial: u32 = 32,

    /// Concurrent GPU GEMV contention benchmark dims (C9 4096x4096 q3/q4).
    gpu_gemv_n: u32 = 4096,
    gpu_gemv_k: u32 = 4096,
    min_working_set_bytes: usize = 64 * 1024 * 1024,
    contention_warmup_batches: u32 = 4,
    contention_timed_batches: u32 = 8,
    contention_dispatches_per_batch: u32 = 16,
    timeout_ms: u64 = 30_000,
};

pub const SpikeS4Error = BenchError || metal.ProtoEmitError || quant.ShapeError || std.Thread.SpawnError;

pub const SpikeS4Report = struct {
    config: SpikeS4Config,
    ane_core_count: u32,
    handoff_uses_ane: bool,
    handoff_zero_copy: bool,
    /// Full `GPU -> IOSurface -> ANE -> IOSurface -> GPU` round-trip latency (us).
    handoff_roundtrip_us: f64,
    /// Standalone tiny ANE invoke latency (us).
    ane_tiny_predict_us: f64,
    /// Two-commit GPU baseline without ANE (us).
    gpu_two_commit_baseline_us: f64,

    draft_m1_uses_ane: bool,
    draft_batch_uses_ane: bool,
    draft_m1_zero_copy: bool,
    draft_batch_zero_copy: bool,
    /// Standalone ANE `M = 1` draft-head step latency (us, `innerProduct`).
    ane_draft_m1_us: f64,
    /// In-command-buffer GPU `M = 1` `encodeDenseGemv` draft-head step latency (us).
    gpu_draft_m1_us: f64,
    /// Standalone ANE batched (`M = draft_batch`) draft-head step latency (us, `conv1x1`).
    ane_draft_batch_us: f64,
    /// In-command-buffer GPU batched (`M = draft_batch`) `encodeDenseGemm` latency (us).
    gpu_draft_batch_us: f64,
    /// D5 benchmark gate recommendation for serial EAGLE/MTP drafting (`M = 1`).
    recommended_m1_placement: SpikeS4Placement,
    /// D5 benchmark gate recommendation for batched drafting (`M = draft_batch`).
    recommended_batch_placement: SpikeS4Placement,

    stream_uses_ane: bool,
    stream_weight_bytes: u64,
    gpu_q3_solo_gbs: f64,
    gpu_q3_concurrent_gbs: f64,
    gpu_q4_solo_gbs: f64,
    gpu_q4_concurrent_gbs: f64,
    ane_concurrent_iters: u64,
    ane_concurrent_gbs: f64,

    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn q3ContentionPct(self: SpikeS4Report) f64 {
        if (self.gpu_q3_solo_gbs <= 0.0) return 0.0;
        return (1.0 - (self.gpu_q3_concurrent_gbs / self.gpu_q3_solo_gbs)) * 100.0;
    }

    pub fn q4ContentionPct(self: SpikeS4Report) f64 {
        if (self.gpu_q4_solo_gbs <= 0.0) return 0.0;
        return (1.0 - (self.gpu_q4_concurrent_gbs / self.gpu_q4_solo_gbs)) * 100.0;
    }

    pub fn write(self: *const SpikeS4Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\Spike S4 CoreML ANE Zero-Copy Handoff, D5 Gate & Bandwidth Contention
            \\s4.ane_cores                     {d}
            \\s4.handoff.uses_ane              {s}
            \\s4.handoff.zero_copy_backing     {s}
            \\s4.handoff.roundtrip_us          {d:.2}
            \\s4.handoff.ane_tiny_predict_us   {d:.2}
            \\s4.handoff.gpu_baseline_us       {d:.2}
            \\s4.draft.shape                   {d}L x {d}x{d} fp16 (batch={d})
            \\s4.draft.m1.ane_uses_ne          {s} (zero_copy={s})
            \\s4.draft.m1.ane_us               {d:.2} (with_handoff={d:.2})
            \\s4.draft.m1.gpu_in_cb_us         {d:.2}
            \\s4.draft.m1.d5_gate              {s}
            \\
        , .{
            self.ane_core_count,
            if (self.handoff_uses_ane) "yes" else "no",
            if (self.handoff_zero_copy) "yes" else "no",
            self.handoff_roundtrip_us,
            self.ane_tiny_predict_us,
            self.gpu_two_commit_baseline_us,
            self.config.draft_layers,
            self.config.draft_hidden,
            self.config.draft_hidden,
            self.config.draft_batch,
            if (self.draft_m1_uses_ane) "yes" else "no",
            if (self.draft_m1_zero_copy) "yes" else "no",
            self.ane_draft_m1_us,
            self.ane_draft_m1_us + self.handoff_roundtrip_us,
            self.gpu_draft_m1_us,
            self.recommended_m1_placement.label(),
        });
        try out.print(
            \\s4.draft.batch.ane_uses_ne       {s} (zero_copy={s})
            \\s4.draft.batch.ane_us            {d:.2} (with_handoff={d:.2})
            \\s4.draft.batch.gpu_in_cb_us      {d:.2}
            \\s4.draft.batch.d5_gate           {s}
            \\s4.contention.ane_stream_mib     {d:.1} (uses_ane={s}, iters={d}, ane_bw={d:.2} GB/s)
            \\s4.contention.gemv_q3            solo={d:.2} GB/s  concurrent={d:.2} GB/s  delta={d:.1}%
            \\s4.contention.gemv_q4            solo={d:.2} GB/s  concurrent={d:.2} GB/s  delta={d:.1}%
            \\s4.swapouts_delta                {d}
            \\s4.pageouts_delta                {d}
            \\
        , .{
            if (self.draft_batch_uses_ane) "yes" else "no",
            if (self.draft_batch_zero_copy) "yes" else "no",
            self.ane_draft_batch_us,
            self.ane_draft_batch_us + self.handoff_roundtrip_us,
            self.gpu_draft_batch_us,
            self.recommended_batch_placement.label(),
            @as(f64, @floatFromInt(self.stream_weight_bytes)) / (1024.0 * 1024.0),
            if (self.stream_uses_ane) "yes" else "no",
            self.ane_concurrent_iters,
            self.ane_concurrent_gbs,
            self.gpu_q3_solo_gbs,
            self.gpu_q3_concurrent_gbs,
            self.q3ContentionPct(),
            self.gpu_q4_solo_gbs,
            self.gpu_q4_concurrent_gbs,
            self.q4ContentionPct(),
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
        });
    }
};

fn fillDeterministicF16Weights(dst: []f16, seed: usize) void {
    const scale: f32 = 1.0 / 64.0;
    for (dst, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 13 + seed * 29 + 5) % 17)) - 8;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * scale);
    }
}

fn fillDeterministicQuantPool(
    comptime spec: quant.QuantSpec,
    qweight: []u8,
    scale_bias: []quant.ScaleBias,
    n: usize,
    k: usize,
) !void {
    const Ops = quant.QuantOps(spec);
    const groups = try Ops.groupsPerRow(k);
    const row_bytes = try Ops.rowWeightBytes(k);
    const q_mask: u8 = (@as(u8, 1) << @intCast(spec.bits.bitWidth())) - 1;
    var codes: [spec.group_size]u8 = undefined;
    for (0..n) |row| {
        for (0..groups) |g| {
            for (0..spec.group_size) |j| {
                codes[j] = @intCast((row * 7 + g * 5 + j * 3 + 1) & q_mask);
            }
            const g_off = row * row_bytes + g * Ops.groupWeightBytes();
            Ops.packGroup(&codes, qweight[g_off..][0..Ops.groupWeightBytes()]);
            scale_bias[row * groups + g] = .{
                .scale = @floatCast(0.03125),
                .bias = @floatCast(-0.0625),
            };
        }
    }
}

const AneStreamWorker = struct {
    model: metal.AneModel,
    plan: metal.AnePlan,
    started: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    completed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    failed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    fn run(self: *AneStreamWorker) void {
        while (!self.stop.load(.acquire)) {
            _ = self.model.predict(self.plan) catch {
                self.failed.store(true, .release);
                self.started.store(1, .release);
                return;
            };
            _ = self.completed.fetchAdd(1, .acq_rel);
            if (self.started.load(.monotonic) == 0) {
                self.started.store(1, .release);
            }
        }
    }
};

fn buildStackedAneModel(
    allocator: std.mem.Allocator,
    op: metal.CoreMlLinearOp,
    hidden: u32,
    num_layers: u32,
    batch_or_spatial: u32,
    units: metal.AneComputeUnits,
) SpikeS4Error!metal.AneModel {
    const w_elems = std.math.mul(usize, hidden, hidden) catch return error.InvalidConfig;
    const weights = try allocator.alloc(f16, w_elems);
    defer allocator.free(weights);
    fillDeterministicF16Weights(weights, hidden ^ num_layers);

    const layers = try allocator.alloc(metal.CoreMlLinearLayer, num_layers);
    defer allocator.free(layers);
    for (layers) |*l| {
        l.* = .{ .c_in = hidden, .c_out = hidden, .weights_f16 = weights };
    }

    const spec = try metal.emitCoreMlLinearModel(allocator, .{
        .op = op,
        .batch_or_spatial = batch_or_spatial,
        .layers = layers,
    });
    defer allocator.free(spec);

    return metal.AneModel.loadSpecData(spec, units);
}

fn runGemvContentionPass(
    compute: *metal.Compute,
    deq_pipes: dequant.DequantPipelines,
    bits: quant.QuantBits,
    gpu_buf: metal.Buffer,
    x_off: u64,
    y_off: u64,
    pool_base: u64,
    slot_stride: u64,
    sb_rel_off: u64,
    num_slots: usize,
    n: u32,
    k: u32,
    warmup_batches: u32,
    timed_batches: u32,
    dispatches_per_batch: u32,
    timeout_ms: u64,
) SpikeS4Error!f64 {
    const bytes_per_disp: u64 = switch (bits) {
        .q2 => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
            break :blk (try Ops.packedWeightBytes(n, k)) + (try Ops.scaleBiasBytes(n, k));
        },
        .q3 => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
            break :blk (try Ops.packedWeightBytes(n, k)) + (try Ops.scaleBiasBytes(n, k));
        },
        .q4 => blk: {
            const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
            break :blk (try Ops.packedWeightBytes(n, k)) + (try Ops.scaleBiasBytes(n, k));
        },
    };

    var cursor: usize = 0;
    var wsig: u64 = 0;
    for (0..warmup_batches) |_| {
        var frame = try compute.beginFrame(timeout_ms);
        errdefer frame.abort();
        for (0..dispatches_per_batch) |d| {
            const slot = cursor % num_slots;
            cursor += 1;
            const w_off = pool_base + @as(u64, slot) * slot_stride;
            try deq_pipes.encodeGemv(&frame, bits, .{
                .x = .{ .buffer = gpu_buf, .offset = x_off },
                .qweight = .{ .buffer = gpu_buf, .offset = w_off },
                .scale_bias = .{ .buffer = gpu_buf, .offset = w_off + sb_rel_off },
                .y = .{ .buffer = gpu_buf, .offset = y_off },
                .n = n,
                .k = k,
            }, d > 0);
        }
        wsig = try frame.commit();
    }
    if (wsig != 0) try compute.wait(wsig, timeout_ms);

    var lsig: u64 = 0;
    const t0 = try vmstat.nowNs();
    for (0..timed_batches) |_| {
        var frame = try compute.beginFrame(timeout_ms);
        errdefer frame.abort();
        for (0..dispatches_per_batch) |d| {
            const slot = cursor % num_slots;
            cursor += 1;
            const w_off = pool_base + @as(u64, slot) * slot_stride;
            try deq_pipes.encodeGemv(&frame, bits, .{
                .x = .{ .buffer = gpu_buf, .offset = x_off },
                .qweight = .{ .buffer = gpu_buf, .offset = w_off },
                .scale_bias = .{ .buffer = gpu_buf, .offset = w_off + sb_rel_off },
                .y = .{ .buffer = gpu_buf, .offset = y_off },
                .n = n,
                .k = k,
            }, d > 0);
        }
        lsig = try frame.commit();
    }
    try compute.wait(lsig, timeout_ms);
    const t1 = try vmstat.nowNs();
    const elapsed_ns = t1 - t0;
    if (elapsed_ns == 0) return 0.0;
    const total_bytes = @as(f64, @floatFromInt(bytes_per_disp)) *
        @as(f64, @floatFromInt(@as(u64, timed_batches) * dispatches_per_batch));
    return total_bytes / @as(f64, @floatFromInt(elapsed_ns));
}

/// Executes Spike S4 on real Apple Silicon hardware (ADR-0008, D5):
/// 1) Zero-copy `GPU -> IOSurface -> ANE -> IOSurface -> GPU` round-trip handoff latency,
/// 2) Standalone ANE draft-head step latency vs in-command-buffer GPU `encodeDenseGemv` /
///    `encodeDenseGemm` latency (`D5` benchmark gate decision), and
/// 3) Concurrent ANE streaming bandwidth contention on GPU `gemv_q3` and `gemv_q4`.
pub fn runSpikeS4(allocator: std.mem.Allocator, config: SpikeS4Config) SpikeS4Error!SpikeS4Report {
    if (config.handoff_c == 0 or config.handoff_s == 0 or (config.handoff_s % metal.KV_TILE_ROW_ALIGN_F16) != 0 or
        config.handoff_timed_iters == 0 or
        config.draft_hidden == 0 or (config.draft_hidden % metal.KV_TILE_ROW_ALIGN_F16) != 0 or
        config.draft_layers == 0 or config.draft_batch == 0 or
        (config.draft_batch % metal.KV_TILE_ROW_ALIGN_F16) != 0 or
        config.draft_timed_iters == 0 or config.gpu_dispatches_per_batch == 0 or
        config.stream_hidden == 0 or config.stream_layers == 0 or
        config.stream_spatial == 0 or (config.stream_spatial % metal.KV_TILE_ROW_ALIGN_F16) != 0 or
        config.gpu_gemv_n == 0 or config.gpu_gemv_k == 0 or (config.gpu_gemv_k % 64) != 0 or
        config.contention_timed_batches == 0 or config.contention_dispatches_per_batch == 0)
    {
        return error.InvalidConfig;
    }

    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const dev_info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, dev_info, .{});

    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const attn_pipes = try AttentionPipelines.init(device, library);
    defer attn_pipes.close();
    const deq_pipes = try dequant.DequantPipelines.init(device, library);
    defer deq_pipes.close();

    var compute = try metal.Compute.initWithFrames(device, 1);
    defer compute.close();

    const vm_before = try vmstat.Snapshot.capture();

    // -----------------------------------------------------------------------
    // Part 1: Zero-copy GPU -> IOSurface -> ANE -> IOSurface -> GPU handoff
    // -----------------------------------------------------------------------
    var handoff_uses_ane = false;
    var handoff_zero_copy = false;
    var handoff_roundtrip_us: f64 = 0.0;
    var ane_tiny_predict_us: f64 = 0.0;
    var gpu_two_commit_baseline_us: f64 = 0.0;

    {
        const hc = config.handoff_c;
        const hs = config.handoff_s;
        const handoff_model = try buildStackedAneModel(allocator, .conv1x1, hc, 1, hs, .cpu_and_neural_engine);
        defer handoff_model.close();
        handoff_uses_ane = (try handoff_model.info()).usesNeuralEngine();

        const in_tile = try metal.KvTile.init(device, hs, hc, .{ .alias_buffer = true });
        defer in_tile.close();
        const out_tile = try metal.KvTile.init(device, hs, hc, .{ .alias_buffer = true });
        defer out_tile.close();
        const in_alias = in_tile.aliasBuffer() orelse return error.TileCreationFailed;
        const out_alias = out_tile.aliasBuffer() orelse return error.TileCreationFailed;

        const in_tensor = try metal.AneTensor.fromKvTile(in_tile, &.{ hc, 1, hs });
        defer in_tensor.close();
        const out_tensor = try metal.AneTensor.fromKvTile(out_tile, &.{ hc, 1, hs });
        defer out_tensor.close();
        const plan = try metal.AnePlan.init(
            &.{.{ .name = "x", .tensor = in_tensor }},
            &.{.{ .name = "y", .tensor = out_tensor }},
        );
        defer plan.close();

        const vec_bytes = amf.align64(@as(usize, hc) * hs * @sizeOf(f16));
        const norm_bytes = amf.align64(@as(usize, hs) * @sizeOf(f16));
        const h_arena_cap = try page.alignUp(vec_bytes * 2 + norm_bytes + 2 * page.APPLE_PAGE_SIZE);
        var h_arena = try arena_mod.WiredArena.init(h_arena_cap, wired_budget);
        defer h_arena.deinit();

        const src_raw = try h_arena.allocAligned(vec_bytes, 64);
        const dst_raw = try h_arena.allocAligned(vec_bytes, 64);
        const nw_raw = try h_arena.allocAligned(norm_bytes, 64);
        const src_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, src_raw[0 .. @as(usize, hc) * hs * @sizeOf(f16)]));
        const dst_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, dst_raw[0 .. @as(usize, hc) * hs * @sizeOf(f16)]));
        const nw_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, nw_raw[0 .. @as(usize, hs) * @sizeOf(f16)]));
        for (src_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + 3) % 19)) - 9)) * 0.125);
        for (nw_f16) |*v| v.* = @floatCast(1.0);
        @memset(dst_raw, 0);

        const h_buf = try metal.Buffer.wrap(device, h_arena.bytes());
        defer h_buf.close();
        try compute.makeResident(&.{h_buf});
        defer compute.removeResident(&.{h_buf}) catch |err| {
            std.debug.panic("removeResident failed: {}", .{err});
        };
        try compute.makeTilesResident(&.{ in_tile, out_tile });
        defer compute.removeTilesResident(&.{ in_tile, out_tile }) catch |err| {
            std.debug.panic("removeTilesResident failed: {}", .{err});
        };

        const src_bind = metal.BufferBinding{ .buffer = h_buf, .offset = h_arena.offsetOf(src_raw.ptr) };
        const dst_bind = metal.BufferBinding{ .buffer = h_buf, .offset = h_arena.offsetOf(dst_raw.ptr) };
        const nw_bind = metal.BufferBinding{ .buffer = h_buf, .offset = h_arena.offsetOf(nw_raw.ptr) };
        const in_bind = metal.BufferBinding{ .buffer = in_alias, .offset = 0 };
        const out_bind = metal.BufferBinding{ .buffer = out_alias, .offset = 0 };

        // Warmup round-trip
        for (0..config.handoff_warmup_iters) |_| {
            var f1 = try compute.beginFrame(config.timeout_ms);
            errdefer f1.abort();
            try attn_pipes.encodeRmsNorm(&f1, .{ .x = src_bind, .w = nw_bind, .y = in_bind, .num_rows = hc, .hidden_dim = hs }, false);
            const s1 = try f1.commit();
            try compute.wait(s1, config.timeout_ms);

            const pinfo = try handoff_model.predict(plan);
            handoff_zero_copy = pinfo.allOutputsZeroCopy();

            var f2 = try compute.beginFrame(config.timeout_ms);
            errdefer f2.abort();
            try attn_pipes.encodeRmsNorm(&f2, .{ .x = out_bind, .w = nw_bind, .y = dst_bind, .num_rows = hc, .hidden_dim = hs }, false);
            const s2 = try f2.commit();
            try compute.wait(s2, config.timeout_ms);
        }

        if (!std.math.isFinite(dst_f16[0]) or !std.math.isFinite(dst_f16[dst_f16.len - 1])) {
            return error.NonFiniteOutput;
        }

        // Timed full GPU -> IOSurface -> ANE -> IOSurface -> GPU round-trip
        const rt0 = try vmstat.nowNs();
        for (0..config.handoff_timed_iters) |_| {
            var f1 = try compute.beginFrame(config.timeout_ms);
            errdefer f1.abort();
            try attn_pipes.encodeRmsNorm(&f1, .{ .x = src_bind, .w = nw_bind, .y = in_bind, .num_rows = hc, .hidden_dim = hs }, false);
            const s1 = try f1.commit();
            try compute.wait(s1, config.timeout_ms);

            _ = try handoff_model.predict(plan);

            var f2 = try compute.beginFrame(config.timeout_ms);
            errdefer f2.abort();
            try attn_pipes.encodeRmsNorm(&f2, .{ .x = out_bind, .w = nw_bind, .y = dst_bind, .num_rows = hc, .hidden_dim = hs }, false);
            const s2 = try f2.commit();
            try compute.wait(s2, config.timeout_ms);
        }
        const rt1 = try vmstat.nowNs();
        handoff_roundtrip_us = (@as(f64, @floatFromInt(rt1 - rt0)) / @as(f64, @floatFromInt(config.handoff_timed_iters))) / 1000.0;

        // Timed standalone tiny ANE predict
        const at0 = try vmstat.nowNs();
        for (0..config.handoff_timed_iters) |_| {
            _ = try handoff_model.predict(plan);
        }
        const at1 = try vmstat.nowNs();
        ane_tiny_predict_us = (@as(f64, @floatFromInt(at1 - at0)) / @as(f64, @floatFromInt(config.handoff_timed_iters))) / 1000.0;

        // Timed GPU two-commit baseline (without ANE)
        const gt0 = try vmstat.nowNs();
        for (0..config.handoff_timed_iters) |_| {
            var f1 = try compute.beginFrame(config.timeout_ms);
            errdefer f1.abort();
            try attn_pipes.encodeRmsNorm(&f1, .{ .x = src_bind, .w = nw_bind, .y = in_bind, .num_rows = hc, .hidden_dim = hs }, false);
            const s1 = try f1.commit();
            try compute.wait(s1, config.timeout_ms);

            var f2 = try compute.beginFrame(config.timeout_ms);
            errdefer f2.abort();
            try attn_pipes.encodeRmsNorm(&f2, .{ .x = in_bind, .w = nw_bind, .y = dst_bind, .num_rows = hc, .hidden_dim = hs }, false);
            const s2 = try f2.commit();
            try compute.wait(s2, config.timeout_ms);
        }
        const gt1 = try vmstat.nowNs();
        gpu_two_commit_baseline_us = (@as(f64, @floatFromInt(gt1 - gt0)) / @as(f64, @floatFromInt(config.handoff_timed_iters))) / 1000.0;
    }

    // -----------------------------------------------------------------------
    // Part 2: Standalone ANE draft-head step vs in-CB GPU dense GEMV / GEMM
    // -----------------------------------------------------------------------
    var draft_m1_uses_ane = false;
    var draft_batch_uses_ane = false;
    var draft_m1_zero_copy = false;
    var draft_batch_zero_copy = false;
    var ane_draft_m1_us: f64 = 0.0;
    var gpu_draft_m1_us: f64 = 0.0;
    var ane_draft_batch_us: f64 = 0.0;
    var gpu_draft_batch_us: f64 = 0.0;

    {
        const dh = config.draft_hidden;
        const dl = config.draft_layers;
        const db = config.draft_batch;

        // 2a. ANE M=1 innerProduct [1, dh] -> [1, dh]
        {
            const m1_model = try buildStackedAneModel(allocator, .inner_product, dh, dl, 1, .cpu_and_neural_engine);
            defer m1_model.close();
            draft_m1_uses_ane = (try m1_model.info()).usesNeuralEngine();

            const in_tile = try metal.KvTile.init(device, dh, 1, .{});
            defer in_tile.close();
            const out_tile = try metal.KvTile.init(device, dh, 1, .{});
            defer out_tile.close();
            try fillTileDeterministic(in_tile, 3);
            const in_t = try metal.AneTensor.fromKvTile(in_tile, &.{ 1, dh });
            defer in_t.close();
            const out_t = try metal.AneTensor.fromKvTile(out_tile, &.{ 1, dh });
            defer out_t.close();
            const plan = try metal.AnePlan.init(&.{.{ .name = "x", .tensor = in_t }}, &.{.{ .name = "y", .tensor = out_t }});
            defer plan.close();

            for (0..config.draft_warmup_iters) |_| {
                draft_m1_zero_copy = (try m1_model.predict(plan)).allOutputsZeroCopy();
            }
            const t0 = try vmstat.nowNs();
            for (0..config.draft_timed_iters) |_| {
                _ = try m1_model.predict(plan);
            }
            const t1 = try vmstat.nowNs();
            ane_draft_m1_us = (@as(f64, @floatFromInt(t1 - t0)) / @as(f64, @floatFromInt(config.draft_timed_iters))) / 1000.0;
        }

        // 2b. ANE batched conv1x1 [dh, 1, db] -> [dh, 1, db]
        {
            const batch_model = try buildStackedAneModel(allocator, .conv1x1, dh, dl, db, .cpu_and_neural_engine);
            defer batch_model.close();
            draft_batch_uses_ane = (try batch_model.info()).usesNeuralEngine();

            const in_tile = try metal.KvTile.init(device, db, dh, .{});
            defer in_tile.close();
            const out_tile = try metal.KvTile.init(device, db, dh, .{});
            defer out_tile.close();
            try fillTileDeterministic(in_tile, 5);
            const in_t = try metal.AneTensor.fromKvTile(in_tile, &.{ dh, 1, db });
            defer in_t.close();
            const out_t = try metal.AneTensor.fromKvTile(out_tile, &.{ dh, 1, db });
            defer out_t.close();
            const plan = try metal.AnePlan.init(&.{.{ .name = "x", .tensor = in_t }}, &.{.{ .name = "y", .tensor = out_t }});
            defer plan.close();

            for (0..config.draft_warmup_iters) |_| {
                draft_batch_zero_copy = (try batch_model.predict(plan)).allOutputsZeroCopy();
            }
            const t0 = try vmstat.nowNs();
            for (0..config.draft_timed_iters) |_| {
                _ = try batch_model.predict(plan);
            }
            const t1 = try vmstat.nowNs();
            ane_draft_batch_us = (@as(f64, @floatFromInt(t1 - t0)) / @as(f64, @floatFromInt(config.draft_timed_iters))) / 1000.0;
        }

        // 2c. In-command-buffer GPU dense GEMV (M=1) and dense GEMM (M=db), same `dl` layers x [dh, dh] fp16
        {
            const w_bytes_layer = try page.alignUp(@as(usize, dh) * dh * @sizeOf(f16));
            const act_bytes = amf.align64(@as(usize, db) * dh * @sizeOf(f16));
            const g_cap = try page.alignUp(w_bytes_layer * dl + act_bytes * 2 + 4 * page.APPLE_PAGE_SIZE);
            var g_arena = try arena_mod.WiredArena.init(g_cap, wired_budget);
            defer g_arena.deinit();

            const w_offs = try allocator.alloc(u64, dl);
            defer allocator.free(w_offs);
            for (0..dl) |li| {
                const wb = try g_arena.allocPages(w_bytes_layer);
                const wf16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wb[0 .. @as(usize, dh) * dh * @sizeOf(f16)]));
                fillDeterministicF16Weights(wf16, li + 7);
                w_offs[li] = g_arena.offsetOf(wb.ptr);
            }
            const a_raw = try g_arena.allocAligned(act_bytes, 64);
            const b_raw = try g_arena.allocAligned(act_bytes, 64);
            const a_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, a_raw));
            fillDeterministicF16Weights(a_f16, 11);
            @memset(b_raw, 0);

            const g_buf = try metal.Buffer.wrap(device, g_arena.bytes());
            defer g_buf.close();
            try compute.makeResident(&.{g_buf});
            defer compute.removeResident(&.{g_buf}) catch |err| {
                std.debug.panic("removeResident failed: {}", .{err});
            };

            const a_bind = metal.BufferBinding{ .buffer = g_buf, .offset = g_arena.offsetOf(a_raw.ptr) };
            const b_bind = metal.BufferBinding{ .buffer = g_buf, .offset = g_arena.offsetOf(b_raw.ptr) };

            const steps_per_frame = config.gpu_dispatches_per_batch;
            // GPU M=1 encodeDenseGemv
            {
                var wsig: u64 = 0;
                for (0..config.draft_warmup_iters) |_| {
                    var frame = try compute.beginFrame(config.timeout_ms);
                    errdefer frame.abort();
                    for (0..steps_per_frame) |step| {
                        for (0..dl) |li| {
                            const src = if ((li & 1) == 0) a_bind else b_bind;
                            const dst = if ((li & 1) == 0) b_bind else a_bind;
                            try deq_pipes.encodeDenseGemv(&frame, .f16, .{
                                .x = src,
                                .weight = .{ .buffer = g_buf, .offset = w_offs[li] },
                                .y = dst,
                                .n = dh,
                                .k = dh,
                            }, step > 0 or li > 0);
                        }
                    }
                    wsig = try frame.commit();
                }
                if (wsig != 0) try compute.wait(wsig, config.timeout_ms);

                var lsig: u64 = 0;
                const t0 = try vmstat.nowNs();
                for (0..config.draft_timed_iters) |_| {
                    var frame = try compute.beginFrame(config.timeout_ms);
                    errdefer frame.abort();
                    for (0..steps_per_frame) |step| {
                        for (0..dl) |li| {
                            const src = if ((li & 1) == 0) a_bind else b_bind;
                            const dst = if ((li & 1) == 0) b_bind else a_bind;
                            try deq_pipes.encodeDenseGemv(&frame, .f16, .{
                                .x = src,
                                .weight = .{ .buffer = g_buf, .offset = w_offs[li] },
                                .y = dst,
                                .n = dh,
                                .k = dh,
                            }, step > 0 or li > 0);
                        }
                    }
                    lsig = try frame.commit();
                }
                try compute.wait(lsig, config.timeout_ms);
                const t1 = try vmstat.nowNs();
                const total_steps = @as(u64, config.draft_timed_iters) * steps_per_frame;
                gpu_draft_m1_us = (@as(f64, @floatFromInt(t1 - t0)) / @as(f64, @floatFromInt(total_steps))) / 1000.0;
            }

            // GPU batched encodeDenseGemm (M = db)
            {
                var wsig: u64 = 0;
                for (0..config.draft_warmup_iters) |_| {
                    var frame = try compute.beginFrame(config.timeout_ms);
                    errdefer frame.abort();
                    for (0..steps_per_frame) |step| {
                        for (0..dl) |li| {
                            const src = if ((li & 1) == 0) a_bind else b_bind;
                            const dst = if ((li & 1) == 0) b_bind else a_bind;
                            try deq_pipes.encodeDenseGemm(&frame, .f16, .{
                                .x = src,
                                .weight = .{ .buffer = g_buf, .offset = w_offs[li] },
                                .y = dst,
                                .m = db,
                                .n = dh,
                                .k = dh,
                            }, step > 0 or li > 0);
                        }
                    }
                    wsig = try frame.commit();
                }
                if (wsig != 0) try compute.wait(wsig, config.timeout_ms);

                var lsig: u64 = 0;
                const t0 = try vmstat.nowNs();
                for (0..config.draft_timed_iters) |_| {
                    var frame = try compute.beginFrame(config.timeout_ms);
                    errdefer frame.abort();
                    for (0..steps_per_frame) |step| {
                        for (0..dl) |li| {
                            const src = if ((li & 1) == 0) a_bind else b_bind;
                            const dst = if ((li & 1) == 0) b_bind else a_bind;
                            try deq_pipes.encodeDenseGemm(&frame, .f16, .{
                                .x = src,
                                .weight = .{ .buffer = g_buf, .offset = w_offs[li] },
                                .y = dst,
                                .m = db,
                                .n = dh,
                                .k = dh,
                            }, step > 0 or li > 0);
                        }
                    }
                    lsig = try frame.commit();
                }
                try compute.wait(lsig, config.timeout_ms);
                const t1 = try vmstat.nowNs();
                const total_steps = @as(u64, config.draft_timed_iters) * steps_per_frame;
                gpu_draft_batch_us = (@as(f64, @floatFromInt(t1 - t0)) / @as(f64, @floatFromInt(total_steps))) / 1000.0;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Part 3: Concurrent ANE DRAM streaming contention on GPU gemv_q3 / gemv_q4
    // -----------------------------------------------------------------------
    var stream_uses_ane = false;
    const stream_weight_bytes: u64 =
        @as(u64, config.stream_layers) * config.stream_hidden * config.stream_hidden * @sizeOf(f16);
    var gpu_q3_solo_gbs: f64 = 0.0;
    var gpu_q3_concurrent_gbs: f64 = 0.0;
    var gpu_q4_solo_gbs: f64 = 0.0;
    var gpu_q4_concurrent_gbs: f64 = 0.0;
    var ane_concurrent_iters: u64 = 0;
    var ane_concurrent_gbs: f64 = 0.0;

    {
        const n = config.gpu_gemv_n;
        const k = config.gpu_gemv_k;
        const Q3 = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
        const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });

        const q3_w_bytes = try Q3.packedWeightBytes(n, k);
        const q3_sb_rel = amf.align64(q3_w_bytes);
        const q3_sb_bytes = try Q3.scaleBiasBytes(n, k);
        const q3_slot_bytes = try page.alignUp(q3_sb_rel + q3_sb_bytes);

        const q4_w_bytes = try Q4.packedWeightBytes(n, k);
        const q4_sb_rel = amf.align64(q4_w_bytes);
        const q4_sb_bytes = try Q4.scaleBiasBytes(n, k);
        const q4_slot_bytes = try page.alignUp(q4_sb_rel + q4_sb_bytes);

        const q3_slots: usize = if (config.min_working_set_bytes == 0)
            1
        else
            @max(@as(usize, 1), (config.min_working_set_bytes + q3_slot_bytes - 1) / q3_slot_bytes);
        const q4_slots: usize = if (config.min_working_set_bytes == 0)
            1
        else
            @max(@as(usize, 1), (config.min_working_set_bytes + q4_slot_bytes - 1) / q4_slot_bytes);

        const x_bytes = amf.align64(@as(usize, k) * @sizeOf(f16));
        const y_bytes = amf.align64(@as(usize, n) * @sizeOf(f16));
        const c_cap = try page.alignUp(
            q3_slots * q3_slot_bytes + q4_slots * q4_slot_bytes + x_bytes + y_bytes + 4 * page.APPLE_PAGE_SIZE,
        );
        var c_arena = try arena_mod.WiredArena.init(c_cap, wired_budget);
        defer c_arena.deinit();

        const q3_pool = try c_arena.allocPages(q3_slots * q3_slot_bytes);
        const q4_pool = try c_arena.allocPages(q4_slots * q4_slot_bytes);
        const x_raw = try c_arena.allocAligned(x_bytes, 64);
        const y_raw = try c_arena.allocAligned(y_bytes, 64);
        @memset(y_raw, 0);

        const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw[0 .. @as(usize, k) * @sizeOf(f16)]));
        fillDeterministicF16Weights(x_f16, 23);

        // Fill slot 0 once and replicate across rotating pool slots.
        try fillDeterministicQuantPool(
            .{ .bits = .q3, .group_size = 64 },
            q3_pool[0..q3_w_bytes],
            @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, q3_pool[q3_sb_rel .. q3_sb_rel + q3_sb_bytes])),
            n,
            k,
        );
        for (1..q3_slots) |s| {
            @memcpy(q3_pool[s * q3_slot_bytes ..][0..q3_slot_bytes], q3_pool[0..q3_slot_bytes]);
        }

        try fillDeterministicQuantPool(
            .{ .bits = .q4, .group_size = 64 },
            q4_pool[0..q4_w_bytes],
            @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, q4_pool[q4_sb_rel .. q4_sb_rel + q4_sb_bytes])),
            n,
            k,
        );
        for (1..q4_slots) |s| {
            @memcpy(q4_pool[s * q4_slot_bytes ..][0..q4_slot_bytes], q4_pool[0..q4_slot_bytes]);
        }

        const c_buf = try metal.Buffer.wrap(device, c_arena.bytes());
        defer c_buf.close();
        try compute.makeResident(&.{c_buf});
        defer compute.removeResident(&.{c_buf}) catch |err| {
            std.debug.panic("removeResident failed: {}", .{err});
        };

        const x_off = c_arena.offsetOf(x_raw.ptr);
        const y_off = c_arena.offsetOf(y_raw.ptr);
        const q3_base = c_arena.offsetOf(q3_pool.ptr);
        const q4_base = c_arena.offsetOf(q4_pool.ptr);

        // Extra DVFS spin-up before first timed GPU GEMV so M4 reaches peak clock.
        const q3_solo_warmups = config.contention_warmup_batches + if (config.min_working_set_bytes > 0) @as(u32, 12) else @as(u32, 0);
        // Solo GPU gemv_q3 & gemv_q4
        gpu_q3_solo_gbs = try runGemvContentionPass(
            &compute,
            deq_pipes,
            .q3,
            c_buf,
            x_off,
            y_off,
            q3_base,
            q3_slot_bytes,
            q3_sb_rel,
            q3_slots,
            n,
            k,
            q3_solo_warmups,
            config.contention_timed_batches,
            config.contention_dispatches_per_batch,
            config.timeout_ms,
        );
        gpu_q4_solo_gbs = try runGemvContentionPass(
            &compute,
            deq_pipes,
            .q4,
            c_buf,
            x_off,
            y_off,
            q4_base,
            q4_slot_bytes,
            q4_sb_rel,
            q4_slots,
            n,
            k,
            config.contention_warmup_batches,
            config.contention_timed_batches,
            config.contention_dispatches_per_batch,
            config.timeout_ms,
        );

        // Background ANE streaming model (conv1x1 stacked layers exceeding SRAM cliff)
        const stream_model = try buildStackedAneModel(
            allocator,
            .conv1x1,
            config.stream_hidden,
            config.stream_layers,
            config.stream_spatial,
            .cpu_and_neural_engine,
        );
        defer stream_model.close();
        stream_uses_ane = (try stream_model.info()).usesNeuralEngine();

        const s_in_tile = try metal.KvTile.init(device, config.stream_spatial, config.stream_hidden, .{});
        defer s_in_tile.close();
        const s_out_tile = try metal.KvTile.init(device, config.stream_spatial, config.stream_hidden, .{});
        defer s_out_tile.close();
        try fillTileDeterministic(s_in_tile, 31);
        const s_in_t = try metal.AneTensor.fromKvTile(s_in_tile, &.{ config.stream_hidden, 1, config.stream_spatial });
        defer s_in_t.close();
        const s_out_t = try metal.AneTensor.fromKvTile(s_out_tile, &.{ config.stream_hidden, 1, config.stream_spatial });
        defer s_out_t.close();
        const s_plan = try metal.AnePlan.init(&.{.{ .name = "x", .tensor = s_in_t }}, &.{.{ .name = "y", .tensor = s_out_t }});
        defer s_plan.close();

        var worker: AneStreamWorker = .{ .model = stream_model, .plan = s_plan };
        const bg_thread = try std.Thread.spawn(.{}, AneStreamWorker.run, .{&worker});
        errdefer {
            worker.stop.store(true, .release);
            bg_thread.join();
        }
        while (worker.started.load(.acquire) == 0) {
            std.atomic.spinLoopHint();
        }
        if (worker.failed.load(.acquire)) {
            worker.stop.store(true, .release);
            bg_thread.join();
            return error.AneFailed;
        }

        const iters_before = worker.completed.load(.acquire);
        const ct0 = try vmstat.nowNs();
        gpu_q3_concurrent_gbs = try runGemvContentionPass(
            &compute,
            deq_pipes,
            .q3,
            c_buf,
            x_off,
            y_off,
            q3_base,
            q3_slot_bytes,
            q3_sb_rel,
            q3_slots,
            n,
            k,
            config.contention_warmup_batches,
            config.contention_timed_batches,
            config.contention_dispatches_per_batch,
            config.timeout_ms,
        );
        gpu_q4_concurrent_gbs = try runGemvContentionPass(
            &compute,
            deq_pipes,
            .q4,
            c_buf,
            x_off,
            y_off,
            q4_base,
            q4_slot_bytes,
            q4_sb_rel,
            q4_slots,
            n,
            k,
            config.contention_warmup_batches,
            config.contention_timed_batches,
            config.contention_dispatches_per_batch,
            config.timeout_ms,
        );
        const ct1 = try vmstat.nowNs();
        const iters_after = worker.completed.load(.acquire);

        worker.stop.store(true, .release);
        bg_thread.join();
        if (worker.failed.load(.acquire)) return error.AneFailed;

        ane_concurrent_iters = iters_after -| iters_before;
        if (ct1 > ct0) {
            ane_concurrent_gbs = (@as(f64, @floatFromInt(ane_concurrent_iters)) * @as(f64, @floatFromInt(stream_weight_bytes))) /
                @as(f64, @floatFromInt(ct1 - ct0));
        }
    }

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    const recommended_m1: SpikeS4Placement = if (gpu_draft_m1_us <= ane_draft_m1_us + handoff_roundtrip_us) .gpu else .ane;
    const recommended_batch: SpikeS4Placement = if (gpu_draft_batch_us <= ane_draft_batch_us + handoff_roundtrip_us) .gpu else .ane;

    return .{
        .config = config,
        .ane_core_count = dev_info.aneCoreCount(),
        .handoff_uses_ane = handoff_uses_ane,
        .handoff_zero_copy = handoff_zero_copy,
        .handoff_roundtrip_us = handoff_roundtrip_us,
        .ane_tiny_predict_us = ane_tiny_predict_us,
        .gpu_two_commit_baseline_us = gpu_two_commit_baseline_us,
        .draft_m1_uses_ane = draft_m1_uses_ane,
        .draft_batch_uses_ane = draft_batch_uses_ane,
        .draft_m1_zero_copy = draft_m1_zero_copy,
        .draft_batch_zero_copy = draft_batch_zero_copy,
        .ane_draft_m1_us = ane_draft_m1_us,
        .gpu_draft_m1_us = gpu_draft_m1_us,
        .ane_draft_batch_us = ane_draft_batch_us,
        .gpu_draft_batch_us = gpu_draft_batch_us,
        .recommended_m1_placement = recommended_m1,
        .recommended_batch_placement = recommended_batch,
        .stream_uses_ane = stream_uses_ane,
        .stream_weight_bytes = stream_weight_bytes,
        .gpu_q3_solo_gbs = gpu_q3_solo_gbs,
        .gpu_q3_concurrent_gbs = gpu_q3_concurrent_gbs,
        .gpu_q4_solo_gbs = gpu_q4_solo_gbs,
        .gpu_q4_concurrent_gbs = gpu_q4_concurrent_gbs,
        .ane_concurrent_iters = ane_concurrent_iters,
        .ane_concurrent_gbs = ane_concurrent_gbs,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

// ---------------------------------------------------------------------------
// CPU fp32 Reference Helpers & Unit Tests
// ---------------------------------------------------------------------------

pub fn rmsNormCpuRef(x: []const f16, w: []const f16, out: []f32, num_rows: usize, hidden_dim: usize, eps: f32) void {
    for (0..num_rows) |r| {
        const x_row = x[r * hidden_dim .. (r + 1) * hidden_dim];
        const y_row = out[r * hidden_dim .. (r + 1) * hidden_dim];
        var sum_sq: f64 = 0.0;
        for (x_row) |v| {
            const fv: f64 = @floatCast(v);
            sum_sq += fv * fv;
        }
        const inv_rms: f32 = @floatCast(1.0 / @sqrt(sum_sq / @as(f64, @floatFromInt(hidden_dim)) + @as(f64, eps)));
        for (0..hidden_dim) |i| {
            const fv: f32 = @floatCast(x_row[i]);
            const fw: f32 = @floatCast(w[i]);
            y_row[i] = (fv * inv_rms) * fw;
        }
    }
}

/// CPU fp32 RMSNorm reference with `f16` or `bf16` norm weights (`w_bits`
/// holds raw 16-bit patterns of `w_dtype`).
pub fn rmsNormCpuRefDType(
    x: []const f16,
    w_dtype: quant.FloatDType,
    w_bits: []const u16,
    out: []f32,
    num_rows: usize,
    hidden_dim: usize,
    eps: f32,
) void {
    for (0..num_rows) |r| {
        const x_row = x[r * hidden_dim .. (r + 1) * hidden_dim];
        const y_row = out[r * hidden_dim .. (r + 1) * hidden_dim];
        var sum_sq: f64 = 0.0;
        for (x_row) |v| {
            const fv: f64 = @floatCast(v);
            sum_sq += fv * fv;
        }
        const inv_rms: f32 = @floatCast(1.0 / @sqrt(sum_sq / @as(f64, @floatFromInt(hidden_dim)) + @as(f64, eps)));
        for (0..hidden_dim) |i| {
            const fv: f32 = @floatCast(x_row[i]);
            y_row[i] = (fv * inv_rms) * quant.floatBitsToF32(w_dtype, w_bits[i]);
        }
    }
}

/// CPU fp32 QK-Norm + NeoX RoPE reference for one head with `f16` or `bf16`
/// norm weights (`norm_w_bits` holds raw 16-bit patterns of `norm_dtype`).
pub fn qkNormRopeHeadCpuRefDType(
    src: []const f16,
    norm_dtype: quant.FloatDType,
    norm_w_bits: ?[]const u16,
    out: []f32,
    pos: u32,
    rms_eps: f32,
    rope_theta: f32,
) void {
    const d = src.len;
    if (norm_w_bits) |nw| {
        var sum_sq: f64 = 0.0;
        for (src) |v| {
            const fv: f64 = @floatCast(v);
            sum_sq += fv * fv;
        }
        const inv_rms: f32 = @floatCast(1.0 / @sqrt(sum_sq / @as(f64, @floatFromInt(d)) + @as(f64, rms_eps)));
        for (0..d) |i| {
            const fv: f32 = @floatCast(src[i]);
            out[i] = (fv * inv_rms) * quant.floatBitsToF32(norm_dtype, nw[i]);
        }
    } else {
        for (0..d) |i| out[i] = @floatCast(src[i]);
    }
    ropeNeoxInPlace(out, pos, rope_theta);
}

fn ropeNeoxInPlace(out: []f32, pos: u32, rope_theta: f32) void {
    const d = out.len;
    const half_d = d / 2;
    const log_theta: f64 = @log(@as(f64, rope_theta));
    const fpos: f64 = @floatFromInt(pos);
    for (0..half_d) |j| {
        const inv_freq: f64 = @exp(-log_theta * (@as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(d))));
        const angle: f64 = fpos * inv_freq;
        const c: f64 = @cos(angle);
        const s: f64 = @sin(angle);
        const x0: f64 = out[j];
        const x1: f64 = out[j + half_d];
        out[j] = @floatCast(x0 * c - x1 * s);
        out[j + half_d] = @floatCast(x1 * c + x0 * s);
    }
}

pub fn qkNormRopeHeadCpuRef(
    src: []const f16,
    norm_w: ?[]const f16,
    out: []f32,
    pos: u32,
    rms_eps: f32,
    rope_theta: f32,
) void {
    const d = src.len;
    const half_d = d / 2;
    if (norm_w) |nw| {
        var sum_sq: f64 = 0.0;
        for (src) |v| {
            const fv: f64 = @floatCast(v);
            sum_sq += fv * fv;
        }
        const inv_rms: f32 = @floatCast(1.0 / @sqrt(sum_sq / @as(f64, @floatFromInt(d)) + @as(f64, rms_eps)));
        for (0..d) |i| {
            const fv: f32 = @floatCast(src[i]);
            const fw: f32 = @floatCast(nw[i]);
            out[i] = (fv * inv_rms) * fw;
        }
    } else {
        for (0..d) |i| {
            out[i] = @floatCast(src[i]);
        }
    }

    const log_theta: f64 = @log(@as(f64, rope_theta));
    const fpos: f64 = @floatFromInt(pos);
    for (0..half_d) |j| {
        const inv_freq: f64 = @exp(-log_theta * (@as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(d))));
        const angle: f64 = fpos * inv_freq;
        const c: f64 = @cos(angle);
        const s: f64 = @sin(angle);
        const x0: f64 = out[j];
        const x1: f64 = out[j + half_d];
        out[j] = @floatCast(x0 * c - x1 * s);
        out[j + half_d] = @floatCast(x1 * c + x0 * s);
    }
}

pub fn gqaDecodeCpuRef(
    allocator: std.mem.Allocator,
    q: []const f16,
    k_cache: []const f16, // [seq_len, num_kv_heads, head_dim]
    v_cache: []const f16, // [seq_len, num_kv_heads, head_dim]
    out: []f32,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) !void {
    const gqa_ratio = num_q_heads / num_kv_heads;
    const scores = try allocator.alloc(f64, seq_len);
    defer allocator.free(scores);

    for (0..num_q_heads) |qh| {
        const kvh = qh / gqa_ratio;
        const q_vec = q[qh * head_dim .. (qh + 1) * head_dim];

        var max_s: f64 = -std.math.inf(f64);
        for (0..seq_len) |t| {
            const kv_off = (t * num_kv_heads + kvh) * head_dim;
            var dot_acc: f64 = 0.0;
            for (0..head_dim) |d| {
                const fq: f64 = @floatCast(q_vec[d]);
                const fk: f64 = @floatCast(k_cache[kv_off + d]);
                dot_acc += fq * fk;
            }
            const s = dot_acc * @as(f64, scale);
            scores[t] = s;
            if (s > max_s) max_s = s;
        }

        var sum_exp: f64 = 0.0;
        for (0..seq_len) |t| {
            const p = @exp(scores[t] - max_s);
            scores[t] = p;
            sum_exp += p;
        }

        for (0..head_dim) |d| {
            var acc: f64 = 0.0;
            for (0..seq_len) |t| {
                const kv_off = (t * num_kv_heads + kvh) * head_dim;
                const fv: f64 = @floatCast(v_cache[kv_off + d]);
                acc += scores[t] * fv;
            }
            out[qh * head_dim + d] = @floatCast(acc / sum_exp);
        }
    }
}

/// CPU fp64/fp32 reference for GQA speculative tree-attention verification (G5).
/// Combines committed prefix KV `0..base_pos` (`k_prefix` / `v_prefix` of shape
/// `[base_pos, num_kv_heads, head_dim]`, may be empty when `base_pos == 0`) with
/// masked tree-local KV `0..num_nodes` where node `i` attends to tree node `j`
/// iff `(ancestor_masks[i] & (@as(u64, 1) << j)) != 0`.
pub fn gqaTreeVerifyCpuRef(
    allocator: std.mem.Allocator,
    q: []const f16, // [num_nodes, num_q_heads, head_dim]
    k_prefix: []const f16, // [base_pos, num_kv_heads, head_dim]
    v_prefix: []const f16, // [base_pos, num_kv_heads, head_dim]
    k_tree: []const f16, // [num_nodes, num_kv_heads, head_dim]
    v_tree: []const f16, // [num_nodes, num_kv_heads, head_dim]
    ancestor_masks: []const u64, // [num_nodes]
    out: []f32, // [num_nodes, num_q_heads, head_dim]
    num_nodes: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    base_pos: usize,
    scale: f32,
) !void {
    std.debug.assert(num_nodes > 0 and num_nodes <= MAX_TREE_NODES);
    const gqa_ratio = num_q_heads / num_kv_heads;
    const total_keys = base_pos + num_nodes;
    const scores = try allocator.alloc(f64, total_keys);
    defer allocator.free(scores);
    const valid = try allocator.alloc(bool, total_keys);
    defer allocator.free(valid);

    for (0..num_nodes) |node| {
        const mask = ancestor_masks[node];
        for (0..num_q_heads) |qh| {
            const kvh = qh / gqa_ratio;
            const q_vec = q[(node * num_q_heads + qh) * head_dim ..][0..head_dim];

            var max_s: f64 = -std.math.inf(f64);
            for (0..base_pos) |t| {
                const kv_off = (t * num_kv_heads + kvh) * head_dim;
                var dot_acc: f64 = 0.0;
                for (0..head_dim) |d| {
                    dot_acc += @as(f64, @floatCast(q_vec[d])) * @as(f64, @floatCast(k_prefix[kv_off + d]));
                }
                const s = dot_acc * @as(f64, scale);
                scores[t] = s;
                valid[t] = true;
                if (s > max_s) max_s = s;
            }
            for (0..num_nodes) |j| {
                const bit_set = ((mask >> @intCast(j)) & 1) != 0;
                valid[base_pos + j] = bit_set;
                if (bit_set) {
                    const kv_off = (j * num_kv_heads + kvh) * head_dim;
                    var dot_acc: f64 = 0.0;
                    for (0..head_dim) |d| {
                        dot_acc += @as(f64, @floatCast(q_vec[d])) * @as(f64, @floatCast(k_tree[kv_off + d]));
                    }
                    const s = dot_acc * @as(f64, scale);
                    scores[base_pos + j] = s;
                    if (s > max_s) max_s = s;
                } else {
                    scores[base_pos + j] = -std.math.inf(f64);
                }
            }

            var sum_exp: f64 = 0.0;
            for (0..total_keys) |idx| {
                if (valid[idx]) {
                    const p = @exp(scores[idx] - max_s);
                    scores[idx] = p;
                    sum_exp += p;
                } else {
                    scores[idx] = 0.0;
                }
            }

            const out_row = out[(node * num_q_heads + qh) * head_dim ..][0..head_dim];
            for (0..head_dim) |d| {
                var acc: f64 = 0.0;
                for (0..base_pos) |t| {
                    const kv_off = (t * num_kv_heads + kvh) * head_dim;
                    acc += scores[t] * @as(f64, @floatCast(v_prefix[kv_off + d]));
                }
                for (0..num_nodes) |j| {
                    if (valid[base_pos + j]) {
                        const kv_off = (j * num_kv_heads + kvh) * head_dim;
                        acc += scores[base_pos + j] * @as(f64, @floatCast(v_tree[kv_off + d]));
                    }
                }
                out_row[d] = if (sum_exp > 0.0) @floatCast(acc / sum_exp) else 0.0;
            }
        }
    }
}

fn testBudget(cap: usize) budget_mod.Budget {
    return budget_mod.Budget.compute(.{
        .memory_bytes = cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = cap,
        .global_user_wire_limit = cap,
        .gpu_working_set_bytes = cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
}

fn verifyRmsNormCase(num_rows: u32, hidden_dim: u32, in_place: bool) !void {
    return verifyRmsNormCaseDType(.f16, num_rows, hidden_dim, in_place);
}

fn verifyRmsNormCaseDType(w_dtype: quant.FloatDType, num_rows: u32, hidden_dim: u32, in_place: bool) !void {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try AttentionPipelines.init(device, library);
    defer pipes.close();

    const x_bytes = amf.align64(@as(usize, num_rows) * hidden_dim * @sizeOf(f16));
    const w_bytes = amf.align64(@as(usize, hidden_dim) * @sizeOf(f16));
    const cap = try page.alignUp(x_bytes * 2 + w_bytes + page.APPLE_PAGE_SIZE);
    var arena = try arena_mod.WiredArena.init(cap, testBudget(cap));
    defer arena.deinit();

    const x_raw = try arena.allocAligned(x_bytes, 64);
    const w_raw = try arena.allocAligned(w_bytes, 64);
    const y_raw = if (in_place) x_raw else try arena.allocAligned(x_bytes, 64);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw[0 .. @as(usize, num_rows) * hidden_dim * @sizeOf(f16)]));
    const w_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, w_raw[0 .. @as(usize, hidden_dim) * @sizeOf(u16)]));
    const y_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_raw[0 .. @as(usize, num_rows) * hidden_dim * @sizeOf(f16)]));

    for (x_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 19 + 7) % 41)) - 20;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.125);
    }
    for (w_bits, 0..) |*v, i| {
        // 1/3 multiples are inexact in both f16 and bf16, so the bf16 path
        // reads genuinely different bits than an f16 table would.
        v.* = quant.f32ToFloatBits(w_dtype, 0.5 + 0.05 * @as(f32, @floatFromInt(i % 17)) + @as(f32, @floatFromInt(i % 3)) / 3.0);
    }

    const ref = try gpa.alloc(f32, @as(usize, num_rows) * hidden_dim);
    defer gpa.free(ref);
    if (w_dtype == .f16) {
        const w_f16: []const f16 = @ptrCast(w_bits);
        rmsNormCpuRef(x_f16, w_f16, ref, num_rows, hidden_dim, 1e-6);
    } else {
        rmsNormCpuRefDType(x_f16, w_dtype, w_bits, ref, num_rows, hidden_dim, 1e-6);
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeRmsNorm(&frame, .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .w = .{ .buffer = gpu_buf, .offset = arena.offsetOf(w_raw.ptr) },
        .y = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_raw.ptr) },
        .num_rows = num_rows,
        .hidden_dim = hidden_dim,
        .eps = 1e-6,
        .w_dtype = w_dtype,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    for (ref, 0..) |expected, i| {
        const got: f32 = @floatCast(y_f16[i]);
        const tol: f32 = @max(0.005, @abs(expected) * 0.005);
        try std.testing.expectApproxEqAbs(expected, got, tol);
    }
}

test "ami_rmsnorm_f16 matches CPU fp32 reference on LLM, odd, batched and in-place shapes" {
    try verifyRmsNormCase(1, 64, false);
    try verifyRmsNormCase(1, 37, false);
    try verifyRmsNormCase(4, 192, false);
    try verifyRmsNormCase(1, 2048, true);
    try verifyRmsNormCase(1, 5120, false);
}

test "ami_rmsnorm with bf16 weights matches CPU fp32 reference on LLM, odd, batched and in-place shapes" {
    try verifyRmsNormCaseDType(.bf16, 1, 64, false);
    try verifyRmsNormCaseDType(.bf16, 1, 37, false);
    try verifyRmsNormCaseDType(.bf16, 4, 192, false);
    try verifyRmsNormCaseDType(.bf16, 1, 1024, true);
    try verifyRmsNormCaseDType(.bf16, 3, 5120, false);
}

fn verifyQkNormRopeKvWriteCase(
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    pos: u32,
    apply_qk_norm: bool,
    rope_theta: f32,
) !void {
    return verifyQkNormRopeKvWriteCaseDType(.f16, num_q_heads, num_kv_heads, head_dim, pos, apply_qk_norm, rope_theta);
}

fn verifyQkNormRopeKvWriteCaseDType(
    norm_dtype: quant.FloatDType,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    pos: u32,
    apply_qk_norm: bool,
    rope_theta: f32,
) !void {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try AttentionPipelines.init(device, library);
    defer pipes.close();

    const chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS;
    const slot_in_chunk: u32 = pos % chunk_tokens;

    const q_bytes = amf.align64(@as(usize, num_q_heads) * head_dim * @sizeOf(f16));
    const kv_bytes = amf.align64(@as(usize, num_kv_heads) * head_dim * @sizeOf(f16));
    const norm_bytes = amf.align64(@as(usize, head_dim) * @sizeOf(f16));
    const cap = try page.alignUp(q_bytes + kv_bytes * 2 + norm_bytes * 2 + page.APPLE_PAGE_SIZE);
    var arena = try arena_mod.WiredArena.init(cap, testBudget(cap));
    defer arena.deinit();

    const q_raw = try arena.allocAligned(q_bytes, 64);
    const k_raw = try arena.allocAligned(kv_bytes, 64);
    const v_raw = try arena.allocAligned(kv_bytes, 64);
    const qn_raw = try arena.allocAligned(norm_bytes, 64);
    const kn_raw = try arena.allocAligned(norm_bytes, 64);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0 .. @as(usize, num_q_heads) * head_dim * @sizeOf(f16)]));
    const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw[0 .. @as(usize, num_kv_heads) * head_dim * @sizeOf(f16)]));
    const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw[0 .. @as(usize, num_kv_heads) * head_dim * @sizeOf(f16)]));
    const qn_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, qn_raw[0 .. @as(usize, head_dim) * @sizeOf(u16)]));
    const kn_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, kn_raw[0 .. @as(usize, head_dim) * @sizeOf(u16)]));

    for (q_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 13 + 3) % 29)) - 14;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.125);
    }
    for (k_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 17 + 5) % 27)) - 13;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.125);
    }
    for (v_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 23 + 11) % 31)) - 15;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.0625);
    }
    for (qn_bits, 0..) |*v, i| v.* = quant.f32ToFloatBits(norm_dtype, 0.75 + 0.04 * @as(f32, @floatFromInt(i % 13)) + @as(f32, @floatFromInt(i % 3)) / 7.0);
    for (kn_bits, 0..) |*v, i| v.* = quant.f32ToFloatBits(norm_dtype, 0.60 + 0.05 * @as(f32, @floatFromInt(i % 11)) + @as(f32, @floatFromInt(i % 5)) / 9.0);

    const q_ref = try gpa.alloc(f32, q_f16.len);
    defer gpa.free(q_ref);
    const k_ref = try gpa.alloc(f32, k_f16.len);
    defer gpa.free(k_ref);

    for (0..num_q_heads) |h| {
        qkNormRopeHeadCpuRefDType(
            q_f16[h * head_dim .. (h + 1) * head_dim],
            norm_dtype,
            if (apply_qk_norm) qn_bits else null,
            q_ref[h * head_dim .. (h + 1) * head_dim],
            pos,
            1e-6,
            rope_theta,
        );
    }
    for (0..num_kv_heads) |h| {
        qkNormRopeHeadCpuRefDType(
            k_f16[h * head_dim .. (h + 1) * head_dim],
            norm_dtype,
            if (apply_qk_norm) kn_bits else null,
            k_ref[h * head_dim .. (h + 1) * head_dim],
            pos,
            1e-6,
            rope_theta,
        );
    }

    const k_tile = try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{});
    defer k_tile.close();
    const v_tile = try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{});
    defer v_tile.close();

    // Zero tiles before GPU write so any missed write stands out.
    inline for (.{ k_tile, v_tile }) |tile| {
        var g = try tile.lock(false);
        errdefer g.release();
        @memset(g.bytes, 0);
        try g.unlock();
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try compute.makeTilesResident(&.{ k_tile, v_tile });

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeQkNormRopeKvWrite(&frame, .{
        .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
        .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) },
        .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) },
        .q_norm = if (apply_qk_norm) .{ .buffer = gpu_buf, .offset = arena.offsetOf(qn_raw.ptr) } else null,
        .k_norm = if (apply_qk_norm) .{ .buffer = gpu_buf, .offset = arena.offsetOf(kn_raw.ptr) } else null,
        .k_tile = k_tile,
        .v_tile = v_tile,
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .pos = pos,
        .chunk_tokens = chunk_tokens,
        .rms_eps = 1e-6,
        .rope_theta = rope_theta,
        .norm_dtype = norm_dtype,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    for (q_ref, 0..) |expected, i| {
        const got: f32 = @floatCast(q_f16[i]);
        try std.testing.expectApproxEqAbs(expected, got, @max(0.01, @abs(expected) * 0.01));
    }

    // Inspect written K and V directly inside the IOSurface tiles via CPU lock.
    {
        var kg = try k_tile.lock(true);
        errdefer kg.release();
        const k_slice = kg.f16Slice();
        for (0..num_kv_heads) |kvh| {
            const row = kvh * chunk_tokens + slot_in_chunk;
            for (0..head_dim) |d| {
                const got: f32 = @floatCast(k_slice[row * head_dim + d]);
                const expected = k_ref[kvh * head_dim + d];
                try std.testing.expectApproxEqAbs(expected, got, @max(0.01, @abs(expected) * 0.01));
            }
        }
        try kg.unlock();
    }
    {
        var vg = try v_tile.lock(true);
        errdefer vg.release();
        const v_slice = vg.f16Slice();
        for (0..num_kv_heads) |kvh| {
            const row = kvh * chunk_tokens + slot_in_chunk;
            for (0..head_dim) |d| {
                try std.testing.expectEqual(v_f16[kvh * head_dim + d], v_slice[row * head_dim + d]);
            }
        }
        try vg.unlock();
    }
}

test "ami_qknorm_rope_kv_write matches CPU fp32 reference with/without QK-norm across head dims" {
    try verifyQkNormRopeKvWriteCase(32, 4, 128, 0, true, 1_000_000.0);
    try verifyQkNormRopeKvWriteCase(32, 4, 128, 519, true, 1_000_000.0);
    try verifyQkNormRopeKvWriteCase(40, 8, 128, 17, false, 10_000.0);
    try verifyQkNormRopeKvWriteCase(4, 2, 64, 511, true, 1_000_000.0);
}

test "ami_qknorm_rope_kv_write with bf16 QK-norm weights writes RGBA16Float KV tiles matching CPU fp32 reference" {
    try verifyQkNormRopeKvWriteCaseDType(.bf16, 16, 8, 128, 0, true, 1_000_000.0);
    try verifyQkNormRopeKvWriteCaseDType(.bf16, 32, 4, 128, 519, true, 1_000_000.0);
    try verifyQkNormRopeKvWriteCaseDType(.bf16, 4, 2, 64, 511, true, 1_000_000.0);
    try verifyQkNormRopeKvWriteCaseDType(.bf16, 40, 8, 128, 17, false, 10_000.0);
}

fn verifyGqaDecodeCase(
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    use_split_kv: bool,
) !void {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try AttentionPipelines.init(device, library);
    defer pipes.close();

    const chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS;
    const num_chunks: u32 = (seq_len + chunk_tokens - 1) / chunk_tokens;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const k_tiles = try gpa.alloc(metal.KvTile, num_chunks);
    defer gpa.free(k_tiles);
    const v_tiles = try gpa.alloc(metal.KvTile, num_chunks);
    defer gpa.free(v_tiles);

    var created_k: usize = 0;
    var created_v: usize = 0;
    defer {
        for (0..created_v) |i| v_tiles[i].close();
        for (0..created_k) |i| k_tiles[i].close();
    }
    for (0..num_chunks) |c_idx| {
        k_tiles[c_idx] = try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{});
        created_k += 1;
        v_tiles[c_idx] = try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{});
        created_v += 1;
    }

    // Flat reference K/V arrays [seq_len, num_kv_heads, head_dim].
    const kv_total: usize = @as(usize, seq_len) * num_kv_heads * head_dim;
    const k_ref_f16 = try gpa.alloc(f16, kv_total);
    defer gpa.free(k_ref_f16);
    const v_ref_f16 = try gpa.alloc(f16, kv_total);
    defer gpa.free(v_ref_f16);

    for (0..num_chunks) |c_idx| {
        var kg = try k_tiles[c_idx].lock(false);
        errdefer kg.release();
        var vg = try v_tiles[c_idx].lock(false);
        errdefer vg.release();
        @memset(kg.bytes, 0);
        @memset(vg.bytes, 0);

        const k_dst = kg.f16Slice();
        const v_dst = vg.f16Slice();
        const t_base = c_idx * chunk_tokens;
        const t_end = @min(t_base + chunk_tokens, seq_len);

        for (t_base..t_end) |t| {
            const slot = t - t_base;
            for (0..num_kv_heads) |kvh| {
                const row = kvh * chunk_tokens + slot;
                for (0..head_dim) |d| {
                    const kc: i32 = @as(i32, @intCast(((t * 31 + kvh * 17 + d * 7 + 5) % 29))) - 14;
                    const vc: i32 = @as(i32, @intCast(((t * 19 + kvh * 23 + d * 11 + 3) % 31))) - 15;
                    const kv: f16 = @floatCast(@as(f32, @floatFromInt(kc)) * 0.0625);
                    const vv: f16 = @floatCast(@as(f32, @floatFromInt(vc)) * 0.0625);
                    k_dst[row * head_dim + d] = kv;
                    v_dst[row * head_dim + d] = vv;
                    const flat_idx = (t * num_kv_heads + kvh) * head_dim + d;
                    k_ref_f16[flat_idx] = kv;
                    v_ref_f16[flat_idx] = vv;
                }
            }
        }
        try vg.unlock();
        try kg.unlock();
    }

    const q_bytes = amf.align64(@as(usize, num_q_heads) * head_dim * @sizeOf(f16));
    const out_bytes = q_bytes;
    const tbl_bytes = amf.align64(@as(usize, num_chunks) * @sizeOf(KvTilePair));
    const scratch_layout = gqaScratchLayout(num_q_heads, head_dim, MAX_SPLITS);
    const cap = try page.alignUp(q_bytes + out_bytes + tbl_bytes + scratch_layout.total_bytes + 4 * page.APPLE_PAGE_SIZE);
    var arena = try arena_mod.WiredArena.init(cap, testBudget(cap));
    defer arena.deinit();

    const q_raw = try arena.allocAligned(q_bytes, 64);
    const out_raw = try arena.allocAligned(out_bytes, 64);
    const tbl_raw = try arena.allocAligned(tbl_bytes, 64);
    const scratch_raw = try arena.allocAligned(scratch_layout.total_bytes, 64);
    @memset(out_raw, 0);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0 .. @as(usize, num_q_heads) * head_dim * @sizeOf(f16)]));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw[0 .. @as(usize, num_q_heads) * head_dim * @sizeOf(f16)]));
    for (q_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 13 + 9) % 25)) - 12;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.08);
    }

    const tbl_slice: []KvTilePair = @alignCast(std.mem.bytesAsSlice(
        KvTilePair,
        tbl_raw[0 .. @as(usize, num_chunks) * @sizeOf(KvTilePair)],
    ));
    for (0..num_chunks) |c_idx| {
        tbl_slice[c_idx] = KvTilePair.fromTiles(k_tiles[c_idx], v_tiles[c_idx], .rgba16f);
    }

    const out_ref = try gpa.alloc(f32, q_f16.len);
    defer gpa.free(out_ref);
    try gqaDecodeCpuRef(gpa, q_f16, k_ref_f16, v_ref_f16, out_ref, num_q_heads, num_kv_heads, head_dim, seq_len, scale);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try compute.makeTilesResident(k_tiles);
    try compute.makeTilesResident(v_tiles);

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeGqaDecodeAttn(&frame, .{
        .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
        .kv_tiles = .{ .buffer = gpu_buf, .offset = arena.offsetOf(tbl_raw.ptr) },
        .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
        .scratch = if (use_split_kv) .{ .buffer = gpu_buf, .offset = arena.offsetOf(scratch_raw.ptr) } else null,
        .max_splits = MAX_SPLITS,
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .seq_len = seq_len,
        .chunk_tokens = chunk_tokens,
        .scale = scale,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    for (out_ref, 0..) |expected, i| {
        const got: f32 = @floatCast(out_f16[i]);
        const tol: f32 = @max(0.005, @abs(expected) * 0.01);
        try std.testing.expectApproxEqAbs(expected, got, tol);
    }
}

test "ami_gqa_decode_attn matches CPU fp32 reference across GQA ratios, head_dim 64/128, and multi-chunk seq_len 520" {
    // Edge cases: seq_len = 1 and odd short seq_len = 37 (single-pass).
    try verifyGqaDecodeCase(4, 2, 64, 1, false);
    try verifyGqaDecodeCase(4, 2, 64, 37, false);
    try verifyGqaDecodeCase(32, 4, 128, 37, false);

    // Multi-chunk seq_len = 520 spanning 2 IOSurface KV tiles (single-pass & split-KV).
    try verifyGqaDecodeCase(32, 4, 128, 520, false);
    try verifyGqaDecodeCase(32, 4, 128, 520, true);
    try verifyGqaDecodeCase(40, 8, 128, 520, true);
    try verifyGqaDecodeCase(4, 2, 64, 520, true);

    // Full 4-chunk context spanning 4 IOSurface KV tiles (split-KV FlashDecoding).
    try verifyGqaDecodeCase(32, 4, 128, 1540, true);
}

test "end-to-end qknorm_rope_kv_write across 520 tokens feeds multi-chunk gqa_decode_attn" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try AttentionPipelines.init(device, library);
    defer pipes.close();

    const num_q_heads: u32 = 4;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 64;
    const chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS;
    const seq_len: u32 = 520;
    const num_chunks: u32 = 2;
    const rope_theta: f32 = 1_000_000.0;
    const eps: f32 = 1e-6;

    const k_tiles = [2]metal.KvTile{
        try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{}),
        try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{}),
    };
    defer for (k_tiles) |t| t.close();
    const v_tiles = [2]metal.KvTile{
        try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{}),
        try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{}),
    };
    defer for (v_tiles) |t| t.close();

    const q_bytes = amf.align64(@as(usize, num_q_heads) * head_dim * @sizeOf(f16));
    const kv_bytes = amf.align64(@as(usize, num_kv_heads) * head_dim * @sizeOf(f16));
    const norm_bytes = amf.align64(@as(usize, head_dim) * @sizeOf(f16));
    const tbl_bytes = amf.align64(@as(usize, num_chunks) * @sizeOf(KvTilePair));
    const scratch_layout = gqaScratchLayout(num_q_heads, head_dim, MAX_SPLITS);
    const cap = try page.alignUp(q_bytes * 2 + kv_bytes * 2 + norm_bytes * 2 + tbl_bytes + scratch_layout.total_bytes + 4 * page.APPLE_PAGE_SIZE);
    var arena = try arena_mod.WiredArena.init(cap, testBudget(cap));
    defer arena.deinit();

    const q_raw = try arena.allocAligned(q_bytes, 64);
    const out_raw = try arena.allocAligned(q_bytes, 64);
    const k_raw = try arena.allocAligned(kv_bytes, 64);
    const v_raw = try arena.allocAligned(kv_bytes, 64);
    const qn_raw = try arena.allocAligned(norm_bytes, 64);
    const kn_raw = try arena.allocAligned(norm_bytes, 64);
    const tbl_raw = try arena.allocAligned(tbl_bytes, 64);
    const scratch_raw = try arena.allocAligned(scratch_layout.total_bytes, 64);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0 .. @as(usize, num_q_heads) * head_dim * @sizeOf(f16)]));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw[0 .. @as(usize, num_q_heads) * head_dim * @sizeOf(f16)]));
    const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw[0 .. @as(usize, num_kv_heads) * head_dim * @sizeOf(f16)]));
    const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw[0 .. @as(usize, num_kv_heads) * head_dim * @sizeOf(f16)]));
    const qn_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, qn_raw[0 .. @as(usize, head_dim) * @sizeOf(f16)]));
    const kn_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kn_raw[0 .. @as(usize, head_dim) * @sizeOf(f16)]));
    for (qn_f16, 0..) |*v, i| v.* = @floatCast(0.8 + 0.02 * @as(f32, @floatFromInt(i % 9)));
    for (kn_f16, 0..) |*v, i| v.* = @floatCast(0.7 + 0.03 * @as(f32, @floatFromInt(i % 7)));

    const tbl_slice: []KvTilePair = @alignCast(std.mem.bytesAsSlice(
        KvTilePair,
        tbl_raw[0 .. @as(usize, num_chunks) * @sizeOf(KvTilePair)],
    ));
    for (0..num_chunks) |c_idx| {
        tbl_slice[c_idx] = KvTilePair.fromTiles(k_tiles[c_idx], v_tiles[c_idx], .rgba16f);
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try compute.makeTilesResident(&k_tiles);
    try compute.makeTilesResident(&v_tiles);

    const k_ref_all = try gpa.alloc(f16, @as(usize, seq_len) * num_kv_heads * head_dim);
    defer gpa.free(k_ref_all);
    const v_ref_all = try gpa.alloc(f16, @as(usize, seq_len) * num_kv_heads * head_dim);
    defer gpa.free(v_ref_all);
    const q_last_ref = try gpa.alloc(f16, q_f16.len);
    defer gpa.free(q_last_ref);
    var head_tmp: [head_dim]f32 = undefined;

    // Populate 0..511 via CPU into chunk 0, then write positions 511..519 via GPU shader
    // (spanning the chunk 0 -> chunk 1 boundary!) to keep the test fast while exercising
    // real shader writes into both tiles + multi-chunk attention read.
    {
        var kg = try k_tiles[0].lock(false);
        errdefer kg.release();
        var vg = try v_tiles[0].lock(false);
        errdefer vg.release();
        const k_dst = kg.f16Slice();
        const v_dst = vg.f16Slice();
        for (0..511) |pos| {
            for (0..num_kv_heads) |kvh| {
                const row = kvh * chunk_tokens + pos;
                for (0..head_dim) |d| {
                    const kc: i32 = @as(i32, @intCast((pos * 13 + kvh * 7 + d * 5) % 23)) - 11;
                    const vc: i32 = @as(i32, @intCast((pos * 17 + kvh * 11 + d * 3) % 25)) - 12;
                    const kv: f16 = @floatCast(@as(f32, @floatFromInt(kc)) * 0.0625);
                    const vv: f16 = @floatCast(@as(f32, @floatFromInt(vc)) * 0.0625);
                    k_dst[row * head_dim + d] = kv;
                    v_dst[row * head_dim + d] = vv;
                    k_ref_all[(pos * num_kv_heads + kvh) * head_dim + d] = kv;
                    v_ref_all[(pos * num_kv_heads + kvh) * head_dim + d] = vv;
                }
            }
        }
        try vg.unlock();
        try kg.unlock();
    }

    for (511..seq_len) |pos_usize| {
        const pos: u32 = @intCast(pos_usize);
        const chunk_idx = pos / chunk_tokens;
        for (q_f16, 0..) |*v, i| {
            const c: i32 = @as(i32, @intCast((pos_usize * 7 + i * 11) % 27)) - 13;
            v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.0625);
        }
        for (k_f16, 0..) |*v, i| {
            const c: i32 = @as(i32, @intCast((pos_usize * 13 + i * 17) % 29)) - 14;
            v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.0625);
        }
        for (v_f16, 0..) |*v, i| {
            const c: i32 = @as(i32, @intCast((pos_usize * 19 + i * 5) % 25)) - 12;
            v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.0625);
        }

        for (0..num_kv_heads) |kvh| {
            qkNormRopeHeadCpuRef(
                k_f16[kvh * head_dim .. (kvh + 1) * head_dim],
                kn_f16,
                &head_tmp,
                pos,
                eps,
                rope_theta,
            );
            for (0..head_dim) |d| {
                k_ref_all[(pos_usize * num_kv_heads + kvh) * head_dim + d] = @floatCast(head_tmp[d]);
                v_ref_all[(pos_usize * num_kv_heads + kvh) * head_dim + d] = v_f16[kvh * head_dim + d];
            }
        }
        if (pos == seq_len - 1) {
            for (0..num_q_heads) |qh| {
                qkNormRopeHeadCpuRef(
                    q_f16[qh * head_dim .. (qh + 1) * head_dim],
                    qn_f16,
                    &head_tmp,
                    pos,
                    eps,
                    rope_theta,
                );
                for (0..head_dim) |d| q_last_ref[qh * head_dim + d] = @floatCast(head_tmp[d]);
            }
        }

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeQkNormRopeKvWrite(&frame, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
            .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) },
            .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) },
            .q_norm = .{ .buffer = gpu_buf, .offset = arena.offsetOf(qn_raw.ptr) },
            .k_norm = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kn_raw.ptr) },
            .k_tile = k_tiles[chunk_idx],
            .v_tile = v_tiles[chunk_idx],
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .pos = pos,
            .chunk_tokens = chunk_tokens,
            .rms_eps = eps,
            .rope_theta = rope_theta,
        }, false);

        if (pos == seq_len - 1) {
            try pipes.encodeGqaDecodeAttn(&frame, .{
                .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
                .kv_tiles = .{ .buffer = gpu_buf, .offset = arena.offsetOf(tbl_raw.ptr) },
                .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
                .scratch = .{ .buffer = gpu_buf, .offset = arena.offsetOf(scratch_raw.ptr) },
                .num_q_heads = num_q_heads,
                .num_kv_heads = num_kv_heads,
                .head_dim = head_dim,
                .seq_len = seq_len,
                .chunk_tokens = chunk_tokens,
            }, true);
        }
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    const out_ref = try gpa.alloc(f32, q_f16.len);
    defer gpa.free(out_ref);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    try gqaDecodeCpuRef(gpa, q_last_ref, k_ref_all, v_ref_all, out_ref, num_q_heads, num_kv_heads, head_dim, seq_len, scale);

    for (out_ref, 0..) |expected, i| {
        const got: f32 = @floatCast(out_f16[i]);
        try std.testing.expectApproxEqAbs(expected, got, @max(0.01, @abs(expected) * 0.015));
    }
}

test "Spike S1 and G4 benchmark smoke tests exercise all 6 KV read paths and swap guard" {
    const gpa = std.testing.allocator;
    const s1 = try runSpikeS1(gpa, .{
        .num_chunks = 2,
        .num_banks = 1,
        .dvfs_warmup_batches = 0,
        .warmup_batches = 1,
        .timed_batches = 1,
        .dispatches_per_batch = 2,
        .timeout_ms = 5000,
    });
    try std.testing.expectEqual(@as(u64, 512 * 1024), s1.tile_data_bytes);
    try std.testing.expectEqual(@as(u64, 512 * 1024), s1.tile_alloc_bytes);
    try std.testing.expectEqual(@as(u32, 256), s1.tile_bytes_per_row);
    for (s1.results) |res| {
        try std.testing.expect(res.elapsed_ns > 0);
        try std.testing.expect(res.effectiveBandwidthGbs() > 0.0);
    }

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try s1.write(&aw.writer);
    const rendered = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, rendered, "token_major") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "channel_major") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "rgba16f") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "alias_buf") != null);

    const gqa_cases = [_]GqaBenchCase{.{ .seq_len = 520 }};
    var gqa_rep = try runGqaBench(gpa, .{
        .cases = &gqa_cases,
        .min_working_set_bytes = 0,
        .dvfs_warmup_batches = 0,
        .warmup_batches = 1,
        .timed_batches = 1,
        .dispatches_per_batch = 2,
        .timeout_ms = 5000,
    });
    defer gqa_rep.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), gqa_rep.results.len);
    try std.testing.expect(gqa_rep.results[0].effectiveBandwidthGbs() > 0.0);
}

fn buildBranchingTreeMasks(num_nodes: usize, masks_out: []u64, depths_out: []u32) void {
    // Deterministic branching tree (root 0, up to 3-way branching so masks are
    // genuinely non-sequential and exercise sparse ancestor chains up to M=64).
    masks_out[0] = 1;
    depths_out[0] = 0;
    for (1..num_nodes) |i| {
        const parent: usize = (i - 1) / 3;
        masks_out[i] = masks_out[parent] | (@as(u64, 1) << @intCast(i));
        depths_out[i] = depths_out[parent] + 1;
    }
}

fn verifyTreeQkNormRopeCase(
    norm_dtype: quant.FloatDType,
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    base_pos: u32,
    apply_qk_norm: bool,
    rope_theta: f32,
) !void {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try AttentionPipelines.init(device, library);
    defer pipes.close();

    const q_elems = @as(usize, num_nodes) * num_q_heads * head_dim;
    const kv_elems = @as(usize, num_nodes) * num_kv_heads * head_dim;
    const q_bytes = amf.align64(q_elems * @sizeOf(f16));
    const kv_bytes = amf.align64(kv_elems * @sizeOf(f16));
    const norm_bytes = amf.align64(@as(usize, head_dim) * @sizeOf(u16));
    const pos_bytes = amf.align64(@as(usize, num_nodes) * @sizeOf(u32));

    const cap = try page.alignUp(q_bytes + kv_bytes * 4 + norm_bytes * 2 + pos_bytes + 4 * page.APPLE_PAGE_SIZE);
    var arena = try arena_mod.WiredArena.init(cap, testBudget(cap));
    defer arena.deinit();

    const q_raw = try arena.allocAligned(q_bytes, 64);
    const k_raw = try arena.allocAligned(kv_bytes, 64);
    const v_raw = try arena.allocAligned(kv_bytes, 64);
    const k_out_raw = try arena.allocAligned(kv_bytes, 64);
    const v_out_raw = try arena.allocAligned(kv_bytes, 64);
    const qn_raw = try arena.allocAligned(norm_bytes, 64);
    const kn_raw = try arena.allocAligned(norm_bytes, 64);
    const pos_raw = try arena.allocAligned(pos_bytes, 64);
    @memset(k_out_raw, 0);
    @memset(v_out_raw, 0);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0 .. q_elems * @sizeOf(f16)]));
    const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw[0 .. kv_elems * @sizeOf(f16)]));
    const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw[0 .. kv_elems * @sizeOf(f16)]));
    const k_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_out_raw[0 .. kv_elems * @sizeOf(f16)]));
    const v_out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_out_raw[0 .. kv_elems * @sizeOf(f16)]));
    const qn_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, qn_raw[0 .. @as(usize, head_dim) * @sizeOf(u16)]));
    const kn_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, kn_raw[0 .. @as(usize, head_dim) * @sizeOf(u16)]));
    const positions: []u32 = @alignCast(std.mem.bytesAsSlice(u32, pos_raw[0 .. @as(usize, num_nodes) * @sizeOf(u32)]));

    var masks_tmp: [MAX_TREE_NODES]u64 = undefined;
    var depths_tmp: [MAX_TREE_NODES]u32 = undefined;
    buildBranchingTreeMasks(num_nodes, &masks_tmp, &depths_tmp);
    for (0..num_nodes) |i| positions[i] = base_pos + depths_tmp[i];

    for (q_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 3) % 29)) - 14)) * 0.125);
    for (k_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 5) % 27)) - 13)) * 0.125);
    for (v_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 23 + 11) % 31)) - 15)) * 0.0625);
    for (qn_bits, 0..) |*v, i| v.* = quant.f32ToFloatBits(norm_dtype, 0.75 + 0.04 * @as(f32, @floatFromInt(i % 13)));
    for (kn_bits, 0..) |*v, i| v.* = quant.f32ToFloatBits(norm_dtype, 0.60 + 0.05 * @as(f32, @floatFromInt(i % 11)));

    const q_ref = try gpa.alloc(f32, q_elems);
    defer gpa.free(q_ref);
    const k_ref = try gpa.alloc(f32, kv_elems);
    defer gpa.free(k_ref);

    for (0..num_nodes) |node| {
        for (0..num_q_heads) |h| {
            const off = (node * num_q_heads + h) * head_dim;
            qkNormRopeHeadCpuRefDType(
                q_f16[off .. off + head_dim],
                norm_dtype,
                if (apply_qk_norm) qn_bits else null,
                q_ref[off .. off + head_dim],
                positions[node],
                1e-6,
                rope_theta,
            );
        }
        for (0..num_kv_heads) |h| {
            const off = (node * num_kv_heads + h) * head_dim;
            qkNormRopeHeadCpuRefDType(
                k_f16[off .. off + head_dim],
                norm_dtype,
                if (apply_qk_norm) kn_bits else null,
                k_ref[off .. off + head_dim],
                positions[node],
                1e-6,
                rope_theta,
            );
        }
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeTreeQkNormRope(&frame, .{
        .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
        .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_raw.ptr) },
        .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_raw.ptr) },
        .q_norm = if (apply_qk_norm) .{ .buffer = gpu_buf, .offset = arena.offsetOf(qn_raw.ptr) } else null,
        .k_norm = if (apply_qk_norm) .{ .buffer = gpu_buf, .offset = arena.offsetOf(kn_raw.ptr) } else null,
        .positions = .{ .buffer = gpu_buf, .offset = arena.offsetOf(pos_raw.ptr) },
        .k_tree_out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(k_out_raw.ptr) },
        .v_tree_out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(v_out_raw.ptr) },
        .num_nodes = num_nodes,
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .rms_eps = 1e-6,
        .rope_theta = rope_theta,
        .norm_dtype = norm_dtype,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    for (q_ref, 0..) |expected, i| {
        const got: f32 = @floatCast(q_f16[i]);
        try std.testing.expectApproxEqAbs(expected, got, @max(0.01, @abs(expected) * 0.01));
    }
    for (k_ref, 0..) |expected, i| {
        const got: f32 = @floatCast(k_out_f16[i]);
        try std.testing.expectApproxEqAbs(expected, got, @max(0.01, @abs(expected) * 0.01));
    }
    for (v_f16, 0..) |expected, i| {
        try std.testing.expectEqual(expected, v_out_f16[i]);
    }
}

test "ami_tree_qknorm_rope matches CPU fp32 reference across M=1..64, f16/bf16 norms, and head dims" {
    try verifyTreeQkNormRopeCase(.f16, 1, 32, 4, 128, 0, true, 1_000_000.0);
    try verifyTreeQkNormRopeCase(.bf16, 16, 32, 4, 128, 510, true, 1_000_000.0);
    try verifyTreeQkNormRopeCase(.f16, 64, 4, 2, 64, 128, true, 10_000.0);
    try verifyTreeQkNormRopeCase(.bf16, 8, 40, 8, 128, 17, false, 1_000_000.0);
}

fn verifyGqaTreeCase(
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    base_pos: u32,
    use_split_kv: bool,
) !void {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try AttentionPipelines.init(device, library);
    defer pipes.close();

    const chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS;
    const num_chunks: u32 = if (base_pos == 0) 0 else (base_pos + chunk_tokens - 1) / chunk_tokens;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const k_tiles = try gpa.alloc(metal.KvTile, num_chunks);
    defer gpa.free(k_tiles);
    const v_tiles = try gpa.alloc(metal.KvTile, num_chunks);
    defer gpa.free(v_tiles);

    var created_k: usize = 0;
    var created_v: usize = 0;
    defer {
        for (0..created_v) |i| v_tiles[i].close();
        for (0..created_k) |i| k_tiles[i].close();
    }
    for (0..num_chunks) |c_idx| {
        k_tiles[c_idx] = try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{});
        created_k += 1;
        v_tiles[c_idx] = try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{});
        created_v += 1;
    }

    const prefix_kv_total = @as(usize, base_pos) * num_kv_heads * head_dim;
    const k_prefix_ref = try gpa.alloc(f16, prefix_kv_total);
    defer gpa.free(k_prefix_ref);
    const v_prefix_ref = try gpa.alloc(f16, prefix_kv_total);
    defer gpa.free(v_prefix_ref);

    for (0..num_chunks) |c_idx| {
        var kg = try k_tiles[c_idx].lock(false);
        errdefer kg.release();
        var vg = try v_tiles[c_idx].lock(false);
        errdefer vg.release();
        @memset(kg.bytes, 0);
        @memset(vg.bytes, 0);

        const k_dst = kg.f16Slice();
        const v_dst = vg.f16Slice();
        const t_base = c_idx * chunk_tokens;
        const t_end = @min(t_base + chunk_tokens, base_pos);

        for (t_base..t_end) |t| {
            const slot = t - t_base;
            for (0..num_kv_heads) |kvh| {
                const row = kvh * chunk_tokens + slot;
                for (0..head_dim) |d| {
                    const kc: i32 = @as(i32, @intCast((t * 31 + kvh * 17 + d * 7 + 5) % 29)) - 14;
                    const vc: i32 = @as(i32, @intCast((t * 19 + kvh * 23 + d * 11 + 3) % 31)) - 15;
                    const kv: f16 = @floatCast(@as(f32, @floatFromInt(kc)) * 0.0625);
                    const vv: f16 = @floatCast(@as(f32, @floatFromInt(vc)) * 0.0625);
                    k_dst[row * head_dim + d] = kv;
                    v_dst[row * head_dim + d] = vv;
                    const flat = (t * num_kv_heads + kvh) * head_dim + d;
                    k_prefix_ref[flat] = kv;
                    v_prefix_ref[flat] = vv;
                }
            }
        }
        try vg.unlock();
        try kg.unlock();
    }

    const q_elems = @as(usize, num_nodes) * num_q_heads * head_dim;
    const kv_tree_elems = @as(usize, num_nodes) * num_kv_heads * head_dim;
    const q_bytes = amf.align64(q_elems * @sizeOf(f16));
    const kv_tree_bytes = amf.align64(kv_tree_elems * @sizeOf(f16));
    const masks_bytes = amf.align64(@as(usize, num_nodes) * @sizeOf(u64));
    const tbl_bytes = amf.align64(@max(@as(usize, num_chunks), 1) * @sizeOf(KvTilePair));
    const scratch_layout = gqaTreeScratchLayout(num_nodes, num_q_heads, head_dim, MAX_SPLITS);

    const cap = try page.alignUp(
        q_bytes * 2 + kv_tree_bytes * 2 + masks_bytes + tbl_bytes + scratch_layout.total_bytes + 4 * page.APPLE_PAGE_SIZE,
    );
    var arena = try arena_mod.WiredArena.init(cap, testBudget(cap));
    defer arena.deinit();

    const q_raw = try arena.allocAligned(q_bytes, 64);
    const out_raw = try arena.allocAligned(q_bytes, 64);
    const kt_raw = try arena.allocAligned(kv_tree_bytes, 64);
    const vt_raw = try arena.allocAligned(kv_tree_bytes, 64);
    const masks_raw = try arena.allocAligned(masks_bytes, 64);
    const tbl_raw = try arena.allocAligned(tbl_bytes, 64);
    const scratch_raw = try arena.allocAligned(scratch_layout.total_bytes, 64);
    @memset(out_raw, 0);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0 .. q_elems * @sizeOf(f16)]));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw[0 .. q_elems * @sizeOf(f16)]));
    const kt_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kt_raw[0 .. kv_tree_elems * @sizeOf(f16)]));
    const vt_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vt_raw[0 .. kv_tree_elems * @sizeOf(f16)]));
    const masks: []u64 = @alignCast(std.mem.bytesAsSlice(u64, masks_raw[0 .. @as(usize, num_nodes) * @sizeOf(u64)]));

    for (q_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 9) % 25)) - 12)) * 0.08);
    for (kt_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 29 + 3) % 27)) - 13)) * 0.0625);
    for (vt_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 7) % 31)) - 15)) * 0.0625);

    var depths_tmp: [MAX_TREE_NODES]u32 = undefined;
    buildBranchingTreeMasks(num_nodes, masks, &depths_tmp);

    if (num_chunks > 0) {
        const tbl_slice: []KvTilePair = @alignCast(std.mem.bytesAsSlice(
            KvTilePair,
            tbl_raw[0 .. @as(usize, num_chunks) * @sizeOf(KvTilePair)],
        ));
        for (0..num_chunks) |c_idx| {
            tbl_slice[c_idx] = KvTilePair.fromTiles(k_tiles[c_idx], v_tiles[c_idx], .rgba16f);
        }
    }

    const out_ref = try gpa.alloc(f32, q_elems);
    defer gpa.free(out_ref);
    try gqaTreeVerifyCpuRef(
        gpa,
        q_f16,
        k_prefix_ref,
        v_prefix_ref,
        kt_f16,
        vt_f16,
        masks,
        out_ref,
        num_nodes,
        num_q_heads,
        num_kv_heads,
        head_dim,
        base_pos,
        scale,
    );

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    if (num_chunks > 0) {
        try compute.makeTilesResident(k_tiles);
        try compute.makeTilesResident(v_tiles);
    }

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeGqaTreeVerifyAttn(&frame, .{
        .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
        .kv_tiles = if (num_chunks > 0) .{ .buffer = gpu_buf, .offset = arena.offsetOf(tbl_raw.ptr) } else null,
        .base_pos = base_pos,
        .k_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kt_raw.ptr) },
        .v_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(vt_raw.ptr) },
        .ancestor_masks = .{ .buffer = gpu_buf, .offset = arena.offsetOf(masks_raw.ptr) },
        .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
        .scratch = if (use_split_kv) .{ .buffer = gpu_buf, .offset = arena.offsetOf(scratch_raw.ptr) } else null,
        .max_splits = MAX_SPLITS,
        .num_nodes = num_nodes,
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .chunk_tokens = chunk_tokens,
        .scale = scale,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    for (out_ref, 0..) |expected, i| {
        const got: f32 = @floatCast(out_f16[i]);
        const tol: f32 = @max(0.006, @abs(expected) * 0.012);
        try std.testing.expectApproxEqAbs(expected, got, tol);
    }
}

test "ami_gqa_tree_verify_attn matches CPU fp32 reference for M=1,8,16,32,64 at base_pos=0,128,520" {
    const m_values = [_]u32{ 1, 8, 16, 32, 64 };
    const base_positions = [_]u32{ 0, 128, 520 };

    // Qwen3-30B-A3B head_dim = 128 (gqa_ratio = 8) across all M and base_pos (including 2-chunk split-KV)
    for (m_values) |m| {
        for (base_positions) |bp| {
            try verifyGqaTreeCase(m, 32, 4, 128, bp, bp > 128);
        }
    }
    // Single-pass verification at base_pos = 520 (no scratch) + Qwen3-14B GQA=5
    try verifyGqaTreeCase(16, 32, 4, 128, 520, false);
    try verifyGqaTreeCase(16, 40, 8, 128, 520, true);

    // head_dim = 64 across all specialized TREE_MAX_NODES = {16, 32, 64} and base_pos = {0, 128, 520}
    try verifyGqaTreeCase(1, 4, 2, 64, 0, false);
    try verifyGqaTreeCase(8, 4, 2, 64, 128, false);
    try verifyGqaTreeCase(16, 4, 2, 64, 520, true);
    try verifyGqaTreeCase(32, 4, 2, 64, 520, true);
    try verifyGqaTreeCase(64, 4, 2, 64, 520, true);
}

test "ami_kv_tree_commit crosses 512-token IOSurface chunk boundary in one dispatch and feeds decode" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try AttentionPipelines.init(device, library);
    defer pipes.close();

    const num_q_heads: u32 = 8;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 128;
    const chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS;
    const base_pos: u32 = 510;
    const num_nodes: u32 = 16;
    const accepted = [_]u16{ 0, 2, 6, 9, 14 }; // 5 nodes -> positions 510, 511 (tile0), 512, 513, 514 (tile1)
    const accepted_u32 = [_]u32{ 0, 2, 6, 9, 14 };

    const k_tiles = [2]metal.KvTile{
        try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{}),
        try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{}),
    };
    defer for (k_tiles) |t| t.close();
    const v_tiles = [2]metal.KvTile{
        try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{}),
        try metal.KvTile.init(device, head_dim, num_kv_heads * chunk_tokens, .{}),
    };
    defer for (v_tiles) |t| t.close();

    const kv_tree_elems = @as(usize, num_nodes) * num_kv_heads * head_dim;
    const kv_tree_bytes = amf.align64(kv_tree_elems * @sizeOf(f16));
    const acc_bytes = amf.align64(accepted.len * @sizeOf(u32));
    const q_bytes = amf.align64(@as(usize, num_q_heads) * head_dim * @sizeOf(f16));
    const tbl_bytes = amf.align64(2 * @sizeOf(KvTilePair));
    const scratch_layout = gqaScratchLayout(num_q_heads, head_dim, MAX_SPLITS);

    const cap = try page.alignUp(kv_tree_bytes * 2 + acc_bytes + q_bytes * 2 + tbl_bytes + scratch_layout.total_bytes + 4 * page.APPLE_PAGE_SIZE);
    var arena = try arena_mod.WiredArena.init(cap, testBudget(cap));
    defer arena.deinit();

    const kt_raw = try arena.allocAligned(kv_tree_bytes, 64);
    const vt_raw = try arena.allocAligned(kv_tree_bytes, 64);
    const acc_raw = try arena.allocAligned(acc_bytes, 64);
    const q_raw = try arena.allocAligned(q_bytes, 64);
    const out_raw = try arena.allocAligned(q_bytes, 64);
    const tbl_raw = try arena.allocAligned(tbl_bytes, 64);
    const scratch_raw = try arena.allocAligned(scratch_layout.total_bytes, 64);

    const kt_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kt_raw[0 .. kv_tree_elems * @sizeOf(f16)]));
    const vt_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vt_raw[0 .. kv_tree_elems * @sizeOf(f16)]));
    const acc_buf: []u32 = @alignCast(std.mem.bytesAsSlice(u32, acc_raw[0 .. accepted.len * @sizeOf(u32)]));
    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0 .. @as(usize, num_q_heads) * head_dim * @sizeOf(f16)]));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw[0 .. @as(usize, num_q_heads) * head_dim * @sizeOf(f16)]));

    for (kt_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 19 + 5) % 31)) - 15)) * 0.0625);
    for (vt_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 23 + 3) % 29)) - 14)) * 0.0625);
    for (q_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 11 + 7) % 23)) - 11)) * 0.08);
    @memcpy(acc_buf, &accepted_u32);

    const tbl_slice: []KvTilePair = @alignCast(std.mem.bytesAsSlice(KvTilePair, tbl_raw[0 .. 2 * @sizeOf(KvTilePair)]));
    tbl_slice[0] = KvTilePair.fromTiles(k_tiles[0], v_tiles[0], .rgba16f);
    tbl_slice[1] = KvTilePair.fromTiles(k_tiles[1], v_tiles[1], .rgba16f);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try compute.makeTilesResident(&k_tiles);
    try compute.makeTilesResident(&v_tiles);

    // Reject crossing boundary when tile1 is omitted.
    {
        var bad_frame = try compute.beginFrame(5000);
        defer bad_frame.abort();
        try std.testing.expectError(error.InvalidArgument, pipes.encodeKvTreeCommit(&bad_frame, .{
            .k_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kt_raw.ptr) },
            .v_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(vt_raw.ptr) },
            .accepted_nodes = .{ .inline_u16 = &accepted },
            .accepted_count = accepted.len,
            .k_tile0 = k_tiles[0],
            .v_tile0 = v_tiles[0],
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .base_pos = base_pos,
            .chunk_tokens = chunk_tokens,
        }, false));
    }

    // Exercise all three AcceptedNodes variants (inline_u16, inline_u32, buffer_u32).
    const modes = [_]AcceptedNodes{
        .{ .inline_u16 = &accepted },
        .{ .inline_u32 = &accepted_u32 },
        .{ .buffer_u32 = .{ .buffer = gpu_buf, .offset = arena.offsetOf(acc_raw.ptr) } },
    };
    for (modes) |mode| {
        for (0..2) |ci| {
            var kg = try k_tiles[ci].lock(false);
            errdefer kg.release();
            var vg = try v_tiles[ci].lock(false);
            errdefer vg.release();
            @memset(kg.bytes, 0);
            @memset(vg.bytes, 0);
            try vg.unlock();
            try kg.unlock();
        }

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeKvTreeCommit(&frame, .{
            .k_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(kt_raw.ptr) },
            .v_tree = .{ .buffer = gpu_buf, .offset = arena.offsetOf(vt_raw.ptr) },
            .accepted_nodes = mode,
            .accepted_count = accepted.len,
            .k_tile0 = k_tiles[0],
            .v_tile0 = v_tiles[0],
            .k_tile1 = k_tiles[1],
            .v_tile1 = v_tiles[1],
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .base_pos = base_pos,
            .chunk_tokens = chunk_tokens,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        for (0..accepted.len) |step| {
            const pos = base_pos + @as(u32, @intCast(step));
            const tile_idx: usize = pos / chunk_tokens;
            const slot: usize = pos % chunk_tokens;
            const node_idx: usize = accepted[step];

            var kg = try k_tiles[tile_idx].lock(true);
            errdefer kg.release();
            var vg = try v_tiles[tile_idx].lock(true);
            errdefer vg.release();
            const k_sl = kg.f16Slice();
            const v_sl = vg.f16Slice();

            for (0..num_kv_heads) |kvh| {
                const row = kvh * chunk_tokens + slot;
                const src_off = (node_idx * num_kv_heads + kvh) * head_dim;
                for (0..head_dim) |d| {
                    try std.testing.expectEqual(kt_f16[src_off + d], k_sl[row * head_dim + d]);
                    try std.testing.expectEqual(vt_f16[src_off + d], v_sl[row * head_dim + d]);
                }
            }
            try vg.unlock();
            try kg.unlock();
        }
    }

    // Verify committed tiles (0..515) immediately feed production GQA decode attention.
    const seq_len: u32 = base_pos + accepted.len;
    const k_all_ref = try gpa.alloc(f16, @as(usize, seq_len) * num_kv_heads * head_dim);
    defer gpa.free(k_all_ref);
    const v_all_ref = try gpa.alloc(f16, @as(usize, seq_len) * num_kv_heads * head_dim);
    defer gpa.free(v_all_ref);
    @memset(k_all_ref, @floatCast(0.0));
    @memset(v_all_ref, @floatCast(0.0));
    for (0..accepted.len) |step| {
        const pos = base_pos + step;
        const node_idx: usize = accepted[step];
        for (0..num_kv_heads) |kvh| {
            const dst_off = (pos * num_kv_heads + kvh) * head_dim;
            const src_off = (node_idx * num_kv_heads + kvh) * head_dim;
            @memcpy(k_all_ref[dst_off .. dst_off + head_dim], kt_f16[src_off .. src_off + head_dim]);
            @memcpy(v_all_ref[dst_off .. dst_off + head_dim], vt_f16[src_off .. src_off + head_dim]);
        }
    }

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeGqaDecodeAttn(&frame, .{
        .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(q_raw.ptr) },
        .kv_tiles = .{ .buffer = gpu_buf, .offset = arena.offsetOf(tbl_raw.ptr) },
        .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) },
        .scratch = .{ .buffer = gpu_buf, .offset = arena.offsetOf(scratch_raw.ptr) },
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .seq_len = seq_len,
        .chunk_tokens = chunk_tokens,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    const out_ref = try gpa.alloc(f32, q_f16.len);
    defer gpa.free(out_ref);
    try gqaDecodeCpuRef(gpa, q_f16, k_all_ref, v_all_ref, out_ref, num_q_heads, num_kv_heads, head_dim, seq_len, 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))));
    for (out_ref, 0..) |expected, i| {
        const got: f32 = @floatCast(out_f16[i]);
        try std.testing.expectApproxEqAbs(expected, got, @max(0.006, @abs(expected) * 0.012));
    }
}

test "Spike S4 benchmark smoke test exercises GPU<->ANE handoff, D5 gate, and contention" {
    const gpa = std.testing.allocator;
    const s4 = try runSpikeS4(gpa, .{
        .handoff_c = 64,
        .handoff_s = 32,
        .handoff_warmup_iters = 2,
        .handoff_timed_iters = 4,
        .draft_hidden = 128,
        .draft_layers = 1,
        .draft_batch = 32,
        .draft_warmup_iters = 2,
        .draft_timed_iters = 4,
        .gpu_dispatches_per_batch = 2,
        .stream_hidden = 128,
        .stream_layers = 2,
        .stream_spatial = 32,
        .gpu_gemv_n = 256,
        .gpu_gemv_k = 256,
        .min_working_set_bytes = 0,
        .contention_warmup_batches = 1,
        .contention_timed_batches = 2,
        .contention_dispatches_per_batch = 2,
        .timeout_ms = 10_000,
    });
    try std.testing.expect(s4.handoff_roundtrip_us > 0.0);
    try std.testing.expect(s4.handoff_zero_copy);
    try std.testing.expect(s4.ane_draft_m1_us > 0.0);
    try std.testing.expect(s4.gpu_draft_m1_us > 0.0);
    try std.testing.expect(s4.ane_draft_batch_us > 0.0);
    try std.testing.expect(s4.gpu_draft_batch_us > 0.0);
    try std.testing.expect(s4.gpu_q3_solo_gbs > 0.0);
    try std.testing.expect(s4.gpu_q4_solo_gbs > 0.0);
    try std.testing.expect(s4.gpu_q3_concurrent_gbs > 0.0);
    try std.testing.expect(s4.gpu_q4_concurrent_gbs > 0.0);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try s4.write(&aw.writer);
    const rendered = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, rendered, "s4.handoff.roundtrip_us") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "s4.draft.m1.d5_gate") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "s4.contention.gemv_q3") != null);
}
