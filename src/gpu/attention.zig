//! IOSurface KV-tile attention kernels, Spike S1 microbenchmark, RMSNorm,
//! QK-Norm + NeoX RoPE + KV-Write, and GQA decode attention (R3, R4, S1, G4).

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const amf = @import("../format/amf.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const vmstat = @import("../platform/vmstat.zig");

/// Default number of tokens stored per KV cache chunk tile (ADR-0004).
pub const DEFAULT_CHUNK_TOKENS: u32 = 512;

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

comptime {
    if (@sizeOf(KvTilePair) != 16) @compileError("KvTilePair must be 16 bytes");
    if (@sizeOf(KvBufPair) != 16) @compileError("KvBufPair must be 16 bytes");
    if (@sizeOf(RmsNormParams) != 16) @compileError("RmsNormParams must be 16 bytes");
    if (@sizeOf(QkNormRopeKvParams) != 48) @compileError("QkNormRopeKvParams must be 48 bytes");
    if (@sizeOf(GqaDecodeParams) != 32) @compileError("GqaDecodeParams must be 32 bytes");
    if (@sizeOf(SpikeS1Params) != 32) @compileError("SpikeS1Params must be 32 bytes");
}

pub const RmsNormArgs = struct {
    x: metal.BufferBinding,
    w: metal.BufferBinding,
    y: metal.BufferBinding,
    num_rows: u32 = 1,
    hidden_dim: u32,
    eps: f32 = 1e-6,
};

pub const QkNormRopeKvArgs = struct {
    /// `[num_q_heads, head_dim]` fp16, updated in-place.
    q: metal.BufferBinding,
    /// `[num_kv_heads, head_dim]` fp16, read-only.
    k: metal.BufferBinding,
    /// `[num_kv_heads, head_dim]` fp16, read-only.
    v: metal.BufferBinding,
    /// Optional `[head_dim]` fp16 Qwen3 Q-norm weights (null skips QK-norm).
    q_norm: ?metal.BufferBinding = null,
    /// Optional `[head_dim]` fp16 Qwen3 K-norm weights (null skips QK-norm).
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

/// Chooses the split-KV partition count for FlashDecoding on Apple Silicon.
pub fn chooseNumSplits(num_kv_heads: u32, seq_len: u32, chunk_tokens: u32, max_splits: u32) u32 {
    if (max_splits <= 1 or seq_len <= 128 or num_kv_heads == 0 or chunk_tokens == 0) return 1;
    const max_by_seq: u32 = @max(@as(u32, 1), seq_len / 64);
    const chunks: u32 = (seq_len + chunk_tokens - 1) / chunk_tokens;
    const target_tgs: u32 = if (seq_len >= 1024) 128 else 32;
    const target_splits: u32 = @max((target_tgs + num_kv_heads - 1) / num_kv_heads, chunks);
    return @min(max_splits, @min(max_by_seq, @max(target_splits, 1)));
}

/// Startup-compiled Metal 4 compute pipelines for RMSNorm, QK-Norm + NeoX RoPE
/// + KV-tile write, and specialized GQA decode attention (`head_dim = 64, 128`).
pub const AttentionPipelines = struct {
    rmsnorm: metal.Pipeline,
    qknorm_rope_kv_write: metal.Pipeline,
    gqa_split_d64: metal.Pipeline,
    gqa_reduce_d64: metal.Pipeline,
    gqa_split_d128: metal.Pipeline,
    gqa_reduce_d128: metal.Pipeline,

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!AttentionPipelines {
        const rmsnorm = try metal.Pipeline.init(device, library, "ami_rmsnorm_f16");
        errdefer rmsnorm.close();

        const qknorm_rope_kv_write = try metal.Pipeline.init(device, library, "ami_qknorm_rope_kv_write");
        errdefer qknorm_rope_kv_write.close();

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

        return .{
            .rmsnorm = rmsnorm,
            .qknorm_rope_kv_write = qknorm_rope_kv_write,
            .gqa_split_d64 = gqa_split_d64,
            .gqa_reduce_d64 = gqa_reduce_d64,
            .gqa_split_d128 = gqa_split_d128,
            .gqa_reduce_d128 = gqa_reduce_d128,
        };
    }

    pub fn close(self: AttentionPipelines) void {
        self.gqa_reduce_d128.close();
        self.gqa_split_d128.close();
        self.gqa_reduce_d64.close();
        self.gqa_split_d64.close();
        self.qknorm_rope_kv_write.close();
        self.rmsnorm.close();
    }

    /// Encodes `ami_rmsnorm_f16` (zero heap allocation on the hot path).
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
        try frame.encodeThreadgroups(
            self.rmsnorm,
            &.{ args.x, args.w, args.y },
            std.mem.asBytes(&params),
            .{ args.num_rows, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes `ami_qknorm_rope_kv_write` (zero heap allocation on the hot path).
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
        try frame.encodeThreadgroupsWithTextures(
            self.qknorm_rope_kv_write,
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
    const w_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, w_raw[0 .. @as(usize, hidden_dim) * @sizeOf(f16)]));
    const y_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_raw[0 .. @as(usize, num_rows) * hidden_dim * @sizeOf(f16)]));

    for (x_f16, 0..) |*v, i| {
        const c: i32 = @as(i32, @intCast((i * 19 + 7) % 41)) - 20;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.125);
    }
    for (w_f16, 0..) |*v, i| {
        v.* = @floatCast(0.5 + 0.05 * @as(f32, @floatFromInt(i % 17)));
    }

    const ref = try gpa.alloc(f32, @as(usize, num_rows) * hidden_dim);
    defer gpa.free(ref);
    rmsNormCpuRef(x_f16, w_f16, ref, num_rows, hidden_dim, 1e-6);

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

fn verifyQkNormRopeKvWriteCase(
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
    const qn_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, qn_raw[0 .. @as(usize, head_dim) * @sizeOf(f16)]));
    const kn_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kn_raw[0 .. @as(usize, head_dim) * @sizeOf(f16)]));

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
    for (qn_f16, 0..) |*v, i| v.* = @floatCast(0.75 + 0.04 * @as(f32, @floatFromInt(i % 13)));
    for (kn_f16, 0..) |*v, i| v.* = @floatCast(0.60 + 0.05 * @as(f32, @floatFromInt(i % 11)));

    const q_ref = try gpa.alloc(f32, q_f16.len);
    defer gpa.free(q_ref);
    const k_ref = try gpa.alloc(f32, k_f16.len);
    defer gpa.free(k_ref);

    for (0..num_q_heads) |h| {
        qkNormRopeHeadCpuRef(
            q_f16[h * head_dim .. (h + 1) * head_dim],
            if (apply_qk_norm) qn_f16 else null,
            q_ref[h * head_dim .. (h + 1) * head_dim],
            pos,
            1e-6,
            rope_theta,
        );
    }
    for (0..num_kv_heads) |h| {
        qkNormRopeHeadCpuRef(
            k_f16[h * head_dim .. (h + 1) * head_dim],
            if (apply_qk_norm) kn_f16 else null,
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
