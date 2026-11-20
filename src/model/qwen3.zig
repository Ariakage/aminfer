//! End-to-end Qwen3-MoE and Qwen3-Dense non-speculative greedy decode engine
//! supporting native FP16, BF16, 3-bit/4-bit affine quantized, and
//! mixed-precision `.amf` checkpoints (W6, W13, ADR-0016).
//!
//! Loads `.amf` weights into one or more `Budget`-checked `WiredArena` shards
//! (supporting models exceeding `MTLDevice.maxBufferLength`), allocates
//! IOSurface-backed `RGBA16Float` `KvTile` chunks per layer, registers all
//! resources in Metal 4's `MTLResidencySet` at startup, and executes each
//! `stepDecode(token_id, pos)` with zero dynamic heap allocation inside a single
//! reusable Metal 4 command buffer.

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const arena_mod = @import("../mem/arena.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const dequant = @import("../gpu/dequant.zig");
const attention = @import("../gpu/attention.zig");
const moe = @import("../gpu/moe.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const vmstat = @import("../platform/vmstat.zig");

pub const ModelArch = enum {
    qwen3_moe,
    qwen3_dense,

    pub fn label(self: ModelArch) []const u8 {
        return switch (self) {
            .qwen3_moe => "qwen3_moe",
            .qwen3_dense => "qwen3",
        };
    }
};

pub const ModelConfig = struct {
    arch: ModelArch,
    /// Affine quantization width when configured in `config.json`; `null` for
    /// unquantized FP16 / BF16 checkpoints (ADR-0016).
    quant_bits: ?quant.QuantBits = null,
    /// Preferred unquantized 16-bit floating-point dtype parsed from
    /// `"torch_dtype"` (`"bfloat16"` or `"float16"`, default `.bf16`).
    default_float_dtype: quant.FloatDType = .bf16,
    group_size: u16 = 64,
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32 = 0,
    moe_intermediate_size: u32 = 0,
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    norm_topk_prob: bool = true,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1_000_000.0,
    max_position_embeddings: u32 = 32768,
    tie_word_embeddings: bool = false,
    bos_token_id: ?u32 = null,
    eos_token_id: ?u32 = null,

    pub const ConfigError = error{
        InvalidJson,
        UnsupportedArchitecture,
        InvalidConfig,
        OutOfMemory,
    };

    pub fn precisionLabel(self: ModelConfig) []const u8 {
        if (self.quant_bits) |qb| {
            return switch (qb) {
                .q3 => "q3",
                .q4 => "q4",
            };
        }
        return switch (self.default_float_dtype) {
            .f16 => "f16",
            .bf16 => "bf16",
        };
    }

    pub fn qDim(self: ModelConfig) u32 {
        return self.num_attention_heads * self.head_dim;
    }

    pub fn kvDim(self: ModelConfig) u32 {
        return self.num_key_value_heads * self.head_dim;
    }

    pub fn validate(self: ModelConfig) ConfigError!void {
        const align_req: u32 = if (self.quant_bits != null) blk: {
            if (self.group_size != 64) return error.InvalidConfig;
            break :blk self.group_size;
        } else 4;

        if (self.vocab_size == 0 or self.hidden_size == 0 or self.num_hidden_layers == 0 or
            self.num_attention_heads == 0 or self.num_key_value_heads == 0 or
            self.max_position_embeddings == 0)
        {
            return error.InvalidConfig;
        }
        if (self.head_dim != 64 and self.head_dim != 128) return error.InvalidConfig;
        if ((self.hidden_size % align_req) != 0) return error.InvalidConfig;
        if ((self.num_attention_heads % self.num_key_value_heads) != 0) return error.InvalidConfig;
        if ((self.qDim() % align_req) != 0) return error.InvalidConfig;
        if ((self.kvDim() % align_req) != 0) return error.InvalidConfig;
        if (self.rms_norm_eps <= 0.0 or self.rope_theta <= 0.0) return error.InvalidConfig;

        switch (self.arch) {
            .qwen3_moe => {
                if (self.num_experts == 0 or self.num_experts > moe.MAX_EXPERTS) return error.InvalidConfig;
                if (self.num_experts_per_tok == 0 or self.num_experts_per_tok > self.num_experts or
                    self.num_experts_per_tok > moe.MAX_TOP_K)
                {
                    return error.InvalidConfig;
                }
                if (self.moe_intermediate_size == 0 or (self.moe_intermediate_size % align_req) != 0) {
                    return error.InvalidConfig;
                }
            },
            .qwen3_dense => {
                if (self.intermediate_size == 0 or (self.intermediate_size % align_req) != 0) {
                    return error.InvalidConfig;
                }
            },
        }
    }

    /// Parses a HuggingFace / MLX Qwen3 `config.json` document at startup.
    pub fn fromJson(allocator: std.mem.Allocator, json_bytes: []const u8) ConfigError!ModelConfig {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidJson,
        };
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return error.InvalidJson,
        };

        const model_type = jsonReqString(obj, "model_type") orelse return error.InvalidJson;
        const arch: ModelArch = if (std.mem.eql(u8, model_type, "qwen3_moe"))
            .qwen3_moe
        else if (std.mem.eql(u8, model_type, "qwen3"))
            .qwen3_dense
        else
            return error.UnsupportedArchitecture;

        const vocab_size = jsonReqU32(obj, "vocab_size") orelse return error.InvalidJson;
        const hidden_size = jsonReqU32(obj, "hidden_size") orelse return error.InvalidJson;
        const num_layers = jsonReqU32(obj, "num_hidden_layers") orelse return error.InvalidJson;
        const num_q_heads = jsonReqU32(obj, "num_attention_heads") orelse return error.InvalidJson;
        const num_kv_heads = jsonOptU32(obj, "num_key_value_heads") orelse num_q_heads;
        const head_dim = jsonOptU32(obj, "head_dim") orelse blk: {
            if (num_q_heads == 0) return error.InvalidConfig;
            break :blk hidden_size / num_q_heads;
        };

        const default_float_dtype: quant.FloatDType = blk: {
            if (jsonReqString(obj, "torch_dtype")) |dt| {
                if (std.mem.eql(u8, dt, "float16")) break :blk .f16;
                if (std.mem.eql(u8, dt, "bfloat16")) break :blk .bf16;
            }
            break :blk .bf16;
        };

        var quant_bits: ?quant.QuantBits = null;
        var group_size: u16 = 64;
        const q_val: ?std.json.Value = blk: {
            if (obj.get("quantization")) |v| {
                if (v != .null) break :blk v;
            }
            if (obj.get("quantization_config")) |v| {
                if (v != .null) break :blk v;
            }
            break :blk null;
        };
        if (q_val) |qval| {
            if (qval != .object) return error.InvalidJson;
            if (jsonOptU32(qval.object, "bits")) |b| {
                quant_bits = switch (b) {
                    3 => .q3,
                    4 => .q4,
                    else => return error.InvalidConfig,
                };
            }
            if (jsonOptU32(qval.object, "group_size")) |gs| {
                if (gs > std.math.maxInt(u16)) return error.InvalidConfig;
                group_size = @intCast(gs);
            }
        }

        const cfg: ModelConfig = .{
            .arch = arch,
            .quant_bits = quant_bits,
            .default_float_dtype = default_float_dtype,
            .group_size = group_size,
            .vocab_size = vocab_size,
            .hidden_size = hidden_size,
            .num_hidden_layers = num_layers,
            .num_attention_heads = num_q_heads,
            .num_key_value_heads = num_kv_heads,
            .head_dim = head_dim,
            .intermediate_size = jsonOptU32(obj, "intermediate_size") orelse 0,
            .moe_intermediate_size = jsonOptU32(obj, "moe_intermediate_size") orelse 0,
            .num_experts = jsonOptU32(obj, "num_experts") orelse 0,
            .num_experts_per_tok = jsonOptU32(obj, "num_experts_per_tok") orelse 0,
            .norm_topk_prob = jsonOptBool(obj, "norm_topk_prob") orelse true,
            .rms_norm_eps = jsonOptF32(obj, "rms_norm_eps") orelse 1e-6,
            .rope_theta = jsonOptF32(obj, "rope_theta") orelse 1_000_000.0,
            .max_position_embeddings = jsonOptU32(obj, "max_position_embeddings") orelse 32768,
            .tie_word_embeddings = jsonOptBool(obj, "tie_word_embeddings") orelse false,
            .bos_token_id = jsonOptTokenId(obj, "bos_token_id"),
            .eos_token_id = jsonOptTokenId(obj, "eos_token_id"),
        };
        try cfg.validate();
        return cfg;
    }
};

fn jsonReqString(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .string => |s| s,
        else => null,
    };
}

fn jsonOptU32(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        else => null,
    };
}

fn jsonReqU32(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    return jsonOptU32(obj, key);
}

fn jsonOptTokenId(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        .array => |arr| if (arr.items.len > 0) switch (arr.items[0]) {
            .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
            else => null,
        } else null,
        else => null,
    };
}

fn jsonOptBool(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .bool => |b| b,
        else => null,
    };
}

fn jsonOptF32(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

pub const LoadOptions = struct {
    max_seq_len: u32 = 2048,
    chunk_tokens: u32 = attention.DEFAULT_CHUNK_TOKENS,
    /// Optional per-shard `WiredArena` ceiling (defaults to `MTLDevice.maxBufferLength`).
    max_shard_bytes: ?usize = null,
    /// Optional explicit wired budget override (defaults to `Budget.fromHost`).
    budget_override: ?budget_mod.Budget = null,
    verify_crcs: bool = true,
    timeout_ms: u64 = 30_000,
};

pub const LoadError = error{
    MissingTensor,
    UnexpectedTensorShape,
    UnexpectedTensorDType,
    TensorExceedsShardLimit,
    TooManyDispatchesForFrame,
    OutOfMemory,
} || ModelConfig.ConfigError || amf.AmfFile.OpenError || quant.ShapeError ||
    sysinfo.Error || metal.WrapError || budget_mod.PlanError ||
    arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError ||
    moe.WorkspaceError;

pub const StepError = error{
    TokenOutOfRange,
    PositionOutOfRange,
} || metal.Error || dequant.EncodeError || moe.EncodeError;

const LinearBinding = struct {
    weight: metal.BufferBinding,
    scale_bias: ?metal.BufferBinding,
    n: u32,
    k: u32,
    dtype: quant.WeightDType,
    alg_bytes: u64,
};

const NormBinding = struct {
    binding: metal.BufferBinding,
    dtype: quant.FloatDType,
};

const WeightShard = struct {
    arena: arena_mod.WiredArena,
    buffer: metal.Buffer,
};

const LayerWeights = struct {
    input_layernorm: NormBinding,
    q_proj: LinearBinding,
    k_proj: LinearBinding,
    v_proj: LinearBinding,
    o_proj: LinearBinding,
    q_norm: ?metal.BufferBinding,
    k_norm: ?metal.BufferBinding,
    qk_norm_dtype: quant.FloatDType,
    post_attention_layernorm: NormBinding,

    // MoE-specific (valid when config.arch == .qwen3_moe)
    router: moe.RouterSource,
    expert_dtype: quant.WeightDType,
    expert_weights_buf: metal.Buffer,
    expert_table: metal.BufferBinding,

    // Dense-specific (valid when config.arch == .qwen3_dense)
    dense_gate: LinearBinding,
    dense_up: LinearBinding,
    dense_down: LinearBinding,

    // Bindless KV chunk table for this layer (`[num_chunks]KvTilePair` in scratch buffer)
    kv_tiles_binding: metal.BufferBinding,
};

const AmfSource = union(enum) {
    bytes: struct {
        parsed: *const amf.Parsed,
        raw: []const u8,
    },
    file: *const amf.AmfFile,

    fn parsed(self: AmfSource) *const amf.Parsed {
        return switch (self) {
            .bytes => |b| b.parsed,
            .file => |f| &f.parsed,
        };
    }

    fn loadBlob(
        self: AmfSource,
        entry: *const amf.TensorEntry,
        arena: *arena_mod.WiredArena,
        verify_crc: bool,
    ) LoadError![]align(page.APPLE_PAGE_SIZE) u8 {
        switch (self) {
            .bytes => |b| {
                if (verify_crc) {
                    return b.parsed.loadBlobFromBytes(b.raw, entry, arena);
                }
                const start: usize = @intCast(entry.blob_offset);
                const len: usize = @intCast(entry.blob_bytes);
                if (start + len > b.raw.len) return error.BlobOutOfBounds;
                const dst = try arena.allocPages(len);
                @memcpy(dst, b.raw[start .. start + len]);
                return dst;
            },
            .file => |f| {
                if (verify_crc) {
                    return f.loadBlob(entry, arena);
                }
                const len: usize = @intCast(entry.blob_bytes);
                const dst = try arena.allocPages(len);
                var done: usize = 0;
                while (done < dst.len) {
                    const rc = std.c.pread(f.fd, dst.ptr + done, dst.len - done, @intCast(entry.blob_offset + done));
                    if (rc < 0) {
                        if (std.c.errno(rc) == .INTR) continue;
                        return error.ReadFailed;
                    }
                    if (rc == 0) return error.ShortRead;
                    done += @intCast(rc);
                }
                return dst;
            },
        }
    }
};

pub const Qwen3Model = struct {
    config: ModelConfig,
    max_seq_len: u32,
    chunk_tokens: u32,
    num_chunks: u32,
    timeout_ms: u64,
    wired_budget_bytes: u64,
    total_wired_bytes: u64,
    seq_len: u32,

    device: metal.Device,
    library: metal.Library,
    dequant_pipes: dequant.DequantPipelines,
    attn_pipes: attention.AttentionPipelines,
    moe_pipes: moe.MoePipelines,
    compute: metal.Compute,

    shards: []WeightShard,
    scratch_arena: arena_mod.WiredArena,
    scratch_buf: metal.Buffer,

    /// Flat `[num_layers * num_chunks * 2]` IOSurface KV tiles (K then V per chunk).
    kv_tiles: []metal.KvTile,
    layers: []LayerWeights,

    // Global weights (`lm_head` aliases `embed` directly when `tie_word_embeddings` is true)
    embed: LinearBinding,
    final_norm: NormBinding,
    lm_head: LinearBinding,

    // Preallocated hot-path activation slices & bindings inside `scratch_arena`/`scratch_buf`
    logits_host: []const f16,
    out_token_id_host: *u32,

    x_bind: metal.BufferBinding,
    x_norm_bind: metal.BufferBinding,
    q_bind: metal.BufferBinding,
    k_bind: metal.BufferBinding,
    v_bind: metal.BufferBinding,
    attn_out_bind: metal.BufferBinding,
    proj_out_bind: metal.BufferBinding,
    logits_bind: metal.BufferBinding,
    out_token_id_bind: metal.BufferBinding,
    gqa_scratch_bind: metal.BufferBinding,

    // Arch-specific FFN scratch
    moe_ws: ?moe.MoeWorkspace,
    dense_gate_bind: metal.BufferBinding,
    dense_up_bind: metal.BufferBinding,
    dense_act_bind: metal.BufferBinding,

    // Precomputed algorithmic traffic constants
    total_layers_static_bytes: u64,
    global_head_bytes: u64,

    pub fn loadFromBytes(
        allocator: std.mem.Allocator,
        amf_bytes: []const u8,
        config: ModelConfig,
        options: LoadOptions,
    ) LoadError!Qwen3Model {
        var parsed = try amf.Parsed.parseBytes(allocator, amf_bytes);
        defer parsed.deinit(allocator);
        return loadInternal(allocator, .{ .bytes = .{ .parsed = &parsed, .raw = amf_bytes } }, config, options);
    }

    pub fn loadFromFile(
        allocator: std.mem.Allocator,
        amf_path: [:0]const u8,
        config: ModelConfig,
        options: LoadOptions,
    ) LoadError!Qwen3Model {
        var file = try amf.AmfFile.open(allocator, amf_path);
        defer file.close(allocator);
        return loadInternal(allocator, .{ .file = &file }, config, options);
    }

    pub fn deinit(self: *Qwen3Model, allocator: std.mem.Allocator) void {
        self.compute.close();
        self.moe_pipes.close();
        self.attn_pipes.close();
        self.dequant_pipes.close();
        self.library.close();
        for (self.kv_tiles) |tile| tile.close();
        allocator.free(self.kv_tiles);
        self.scratch_buf.close();
        for (self.shards) |shard| shard.buffer.close();
        self.device.close();
        self.scratch_arena.deinit();
        for (self.shards) |*shard| shard.arena.deinit();
        allocator.free(self.shards);
        allocator.free(self.layers);
    }

    pub fn resetKvCache(self: *Qwen3Model) void {
        self.seq_len = 0;
    }

    pub fn shardCount(self: *const Qwen3Model) usize {
        return self.shards.len;
    }

    pub fn totalWiredBytes(self: *const Qwen3Model) u64 {
        return self.total_wired_bytes;
    }

    /// Mean algorithmic bytes transferred by one transformer layer at `seq_len`
    /// (dense attention weights + active FFN weights + norms + KV read/write).
    pub fn bytesPerLayerToken(self: *const Qwen3Model, seq_len: u32) u64 {
        const kv_read: u64 = 2 * @as(u64, self.config.num_key_value_heads) * seq_len * self.config.head_dim * @sizeOf(f16);
        const kv_write: u64 = 2 * @as(u64, self.config.num_key_value_heads) * self.config.head_dim * @sizeOf(f16);
        return (self.total_layers_static_bytes / self.config.num_hidden_layers) + kv_read + kv_write;
    }

    /// Algorithmic bytes transferred by global embedding row + final norm + `lm_head`.
    pub fn bytesGlobalHead(self: *const Qwen3Model) u64 {
        return self.global_head_bytes;
    }

    /// Algorithmic bytes transferred by one complete `stepDecode` at `seq_len`.
    pub fn bytesPerDecodeToken(self: *const Qwen3Model, seq_len: u32) u64 {
        const kv_per_layer: u64 =
            2 * @as(u64, self.config.num_key_value_heads) * (@as(u64, seq_len) + 1) * self.config.head_dim * @sizeOf(f16);
        return self.total_layers_static_bytes + @as(u64, self.config.num_hidden_layers) * kv_per_layer + self.global_head_bytes;
    }

    pub fn readLogitsF16(self: *const Qwen3Model) []const f16 {
        return self.logits_host;
    }

    /// Populates KV cache at `pos` for `token_id`, skipping final norm / `lm_head` / argmax.
    /// Zero dynamic heap allocation.
    pub fn prefillToken(self: *Qwen3Model, token_id: u32, pos: u32) StepError!void {
        _ = try self.runStep(token_id, pos, true);
    }

    /// Executes one complete non-speculative greedy decode step inside a single
    /// reusable Metal 4 command buffer with zero dynamic heap allocation and
    /// returns the argmax next token ID.
    pub fn stepDecode(self: *Qwen3Model, token_id: u32, pos: u32) StepError!u32 {
        return self.runStep(token_id, pos, false);
    }

    fn kvTileAt(self: *const Qwen3Model, layer_idx: usize, chunk_idx: usize) struct { k: metal.KvTile, v: metal.KvTile } {
        const base = (layer_idx * @as(usize, self.num_chunks) + chunk_idx) * 2;
        return .{ .k = self.kv_tiles[base], .v = self.kv_tiles[base + 1] };
    }

    fn encodeLinearGemv(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        lin: LinearBinding,
        x: metal.BufferBinding,
        y: metal.BufferBinding,
        insert_barrier_before: bool,
    ) dequant.EncodeError!void {
        switch (lin.dtype) {
            .f16 => try self.dequant_pipes.encodeDenseGemv(frame, .f16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
            .bf16 => try self.dequant_pipes.encodeDenseGemv(frame, .bf16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
            .q4_affine => try self.dequant_pipes.encodeGemv(frame, .q4, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
            .q3_affine => try self.dequant_pipes.encodeGemv(frame, .q3, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
        }
    }

    fn runStep(self: *Qwen3Model, token_id: u32, pos: u32, skip_lm_head: bool) StepError!u32 {
        if (token_id >= self.config.vocab_size) return error.TokenOutOfRange;
        if (pos >= self.max_seq_len) return error.PositionOutOfRange;

        const cur_seq_len: u32 = pos + 1;
        const chunk_idx: usize = @intCast(pos / self.chunk_tokens);

        // Encode embedding lookup + all transformer layers + optional head into
        // ONE reusable Metal 4 command buffer with zero CPU copies or heap allocs.
        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();

        try self.moe_pipes.encodeEmbedLookup(&frame, self.embed.dtype, .{
            .weight = self.embed.weight,
            .scale_bias = self.embed.scale_bias,
            .tokens = .{ .token_id = token_id },
            .out = self.x_bind,
            .m = 1,
            .vocab_size = self.config.vocab_size,
            .hidden_size = self.config.hidden_size,
        }, false);

        for (self.layers, 0..) |layer, l| {
            // input_layernorm: x -> x_norm (waits on embed lookup or prior layer)
            try self.attn_pipes.encodeRmsNorm(&frame, .{
                .x = self.x_bind,
                .w = layer.input_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = 1,
                .hidden_dim = self.config.hidden_size,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.input_layernorm.dtype,
            }, true);

            // q_proj, k_proj, v_proj read x_norm concurrently into disjoint outputs.
            try self.encodeLinearGemv(&frame, layer.q_proj, self.x_norm_bind, self.q_bind, true);
            try self.encodeLinearGemv(&frame, layer.k_proj, self.x_norm_bind, self.k_bind, false);
            try self.encodeLinearGemv(&frame, layer.v_proj, self.x_norm_bind, self.v_bind, false);

            // QK-Norm + NeoX RoPE + direct RGBA16Float IOSurface KV-tile write.
            const kv_pair = self.kvTileAt(l, chunk_idx);
            try self.attn_pipes.encodeQkNormRopeKvWrite(&frame, .{
                .q = self.q_bind,
                .k = self.k_bind,
                .v = self.v_bind,
                .q_norm = layer.q_norm,
                .k_norm = layer.k_norm,
                .k_tile = kv_pair.k,
                .v_tile = kv_pair.v,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .pos = pos,
                .chunk_tokens = self.chunk_tokens,
                .rms_eps = self.config.rms_norm_eps,
                .rope_theta = self.config.rope_theta,
                .norm_dtype = layer.qk_norm_dtype,
            }, true);

            // GQA decode attention over committed IOSurface KV tiles.
            try self.attn_pipes.encodeGqaDecodeAttn(&frame, .{
                .q = self.q_bind,
                .kv_tiles = layer.kv_tiles_binding,
                .out = self.attn_out_bind,
                .scratch = self.gqa_scratch_bind,
                .max_splits = attention.MAX_SPLITS,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .seq_len = cur_seq_len,
                .chunk_tokens = self.chunk_tokens,
            }, true);

            // o_proj + residual add into x.
            try self.encodeLinearGemv(&frame, layer.o_proj, self.attn_out_bind, self.proj_out_bind, true);
            try self.moe_pipes.encodeResidualAdd(&frame, .{
                .x = self.x_bind,
                .delta = self.proj_out_bind,
                .count = self.config.hidden_size,
            }, true);

            // post_attention_layernorm: x -> x_norm
            try self.attn_pipes.encodeRmsNorm(&frame, .{
                .x = self.x_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = 1,
                .hidden_dim = self.config.hidden_size,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, true);

            // FFN + residual add into x.
            switch (self.config.arch) {
                .qwen3_moe => {
                    const ws = &(self.moe_ws.?);
                    const moe_args: moe.MoELayerArgs = .{
                        .x = self.x_norm_bind,
                        .router = layer.router,
                        .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                        .expert_table = layer.expert_table,
                        .workspace = ws,
                        .workspace_buf = self.scratch_buf,
                        .residual = self.x_bind,
                        .out = self.x_bind,
                        .m = 1,
                        .num_experts = self.config.num_experts,
                        .top_k = self.config.num_experts_per_tok,
                        .hidden_size = self.config.hidden_size,
                        .intermediate_size = self.config.moe_intermediate_size,
                        .norm_topk_prob = self.config.norm_topk_prob,
                    };
                    switch (layer.expert_dtype) {
                        .q3_affine => try self.moe_pipes.encodeMoELayer(&frame, .q3, moe_args, true),
                        .q4_affine => try self.moe_pipes.encodeMoELayer(&frame, .q4, moe_args, true),
                        .f16 => try self.moe_pipes.encodeMoELayerFloat(&frame, .f16, moe_args, true),
                        .bf16 => try self.moe_pipes.encodeMoELayerFloat(&frame, .bf16, moe_args, true),
                    }
                },
                .qwen3_dense => {
                    try self.encodeLinearGemv(&frame, layer.dense_gate, self.x_norm_bind, self.dense_gate_bind, true);
                    try self.encodeLinearGemv(&frame, layer.dense_up, self.x_norm_bind, self.dense_up_bind, false);
                    try self.moe_pipes.encodeSiluMul(&frame, .{
                        .gate = self.dense_gate_bind,
                        .up = self.dense_up_bind,
                        .out = self.dense_act_bind,
                        .count = self.config.intermediate_size,
                    }, true);
                    try self.encodeLinearGemv(&frame, layer.dense_down, self.dense_act_bind, self.proj_out_bind, true);
                    try self.moe_pipes.encodeResidualAdd(&frame, .{
                        .x = self.x_bind,
                        .delta = self.proj_out_bind,
                        .count = self.config.hidden_size,
                    }, true);
                },
            }
        }

        if (!skip_lm_head) {
            try self.attn_pipes.encodeRmsNorm(&frame, .{
                .x = self.x_bind,
                .w = self.final_norm.binding,
                .y = self.x_norm_bind,
                .num_rows = 1,
                .hidden_dim = self.config.hidden_size,
                .eps = self.config.rms_norm_eps,
                .w_dtype = self.final_norm.dtype,
            }, true);
            try self.encodeLinearGemv(&frame, self.lm_head, self.x_norm_bind, self.logits_bind, true);
            try self.moe_pipes.encodeArgmax(&frame, .{
                .logits = self.logits_bind,
                .out_token_id = self.out_token_id_bind,
                .vocab_size = self.config.vocab_size,
            }, true);
        }

        const sig = try frame.commit();
        try self.compute.wait(sig, self.timeout_ms);
        self.seq_len = cur_seq_len;
        return if (skip_lm_head) 0 else self.out_token_id_host.*;
    }
};

fn findRequired(parsed: *const amf.Parsed, name: []const u8) LoadError!*const amf.TensorEntry {
    return parsed.find(name) orelse error.MissingTensor;
}

fn findLinear(parsed: *const amf.Parsed, base: []const u8) LoadError!*const amf.TensorEntry {
    if (parsed.find(base)) |e| return e;
    var alt_buf: [64]u8 = undefined;
    const with_weight = std.fmt.bufPrint(&alt_buf, "{s}.weight", .{base}) catch return error.MissingTensor;
    return parsed.find(with_weight) orelse error.MissingTensor;
}

fn validateNorm1D(entry: *const amf.TensorEntry, dim: u32) LoadError!quant.FloatDType {
    const dt = try entry.dataType();
    if (entry.rank != 1 or entry.shape[0] != dim) return error.UnexpectedTensorShape;
    return switch (dt) {
        .f16 => .f16,
        .bf16 => .bf16,
        else => error.UnexpectedTensorDType,
    };
}

fn validateLinear2D(entry: *const amf.TensorEntry, n: u32, k: u32) LoadError!quant.WeightDType {
    const dt = try entry.dataType();
    if (entry.rank != 2 or entry.shape[0] != n or entry.shape[1] != k) return error.UnexpectedTensorShape;
    return switch (dt) {
        .f16 => .f16,
        .bf16 => .bf16,
        .q4_affine => blk: {
            if ((k % 64) != 0) return error.UnalignedK;
            break :blk .q4_affine;
        },
        .q3_affine => blk: {
            if ((k % 64) != 0) return error.UnalignedK;
            break :blk .q3_affine;
        },
        .f32 => error.UnexpectedTensorDType,
    };
}

fn validateRouter2D(entry: *const amf.TensorEntry, num_experts: u32, hidden_size: u32) LoadError!amf.DType {
    const dt = try entry.dataType();
    if (entry.rank != 2 or entry.shape[0] != num_experts or entry.shape[1] != hidden_size) {
        return error.UnexpectedTensorShape;
    }
    switch (dt) {
        .f16, .bf16, .f32 => {},
        .q4_affine, .q3_affine => {
            if ((hidden_size % 64) != 0) return error.UnalignedK;
        },
    }
    return dt;
}

const LoadedBlob = struct {
    entry: *const amf.TensorEntry,
    slice: []align(page.APPLE_PAGE_SIZE) u8,
    offset: u64,
    buffer: metal.Buffer,

    fn asBinding(self: LoadedBlob) metal.BufferBinding {
        return .{ .buffer = self.buffer, .offset = self.offset };
    }

    fn asNormBinding(self: LoadedBlob, dtype: quant.FloatDType) NormBinding {
        return .{ .binding = self.asBinding(), .dtype = dtype };
    }

    fn asLinearBinding(self: LoadedBlob, dtype: quant.WeightDType) LinearBinding {
        const sb: ?metal.BufferBinding = switch (dtype) {
            .f16, .bf16 => null,
            .q4_affine, .q3_affine => .{ .buffer = self.buffer, .offset = self.offset + self.entry.params_offset },
        };
        return .{
            .weight = .{ .buffer = self.buffer, .offset = self.offset },
            .scale_bias = sb,
            .n = self.entry.shape[0],
            .k = self.entry.shape[1],
            .dtype = dtype,
            .alg_bytes = self.entry.data_bytes + self.entry.params_bytes,
        };
    }
};

fn loadBlobIntoShard(
    source: AmfSource,
    entry: *const amf.TensorEntry,
    shard: *WeightShard,
    verify_crc: bool,
) LoadError!LoadedBlob {
    const slice = try source.loadBlob(entry, &shard.arena, verify_crc);
    return .{
        .entry = entry,
        .slice = slice,
        .offset = shard.arena.offsetOf(slice.ptr),
        .buffer = shard.buffer,
    };
}

fn loadInternal(
    allocator: std.mem.Allocator,
    source: AmfSource,
    config: ModelConfig,
    options: LoadOptions,
) LoadError!Qwen3Model {
    try config.validate();
    if (options.max_seq_len == 0 or options.chunk_tokens == 0) return error.InvalidConfig;
    const max_seq_len = @min(options.max_seq_len, config.max_position_embeddings);
    const num_chunks: u32 = (max_seq_len + options.chunk_tokens - 1) / options.chunk_tokens;

    // Worst-case dispatches in one stepDecode command buffer: 1 embed + 15 per layer + 3 global.
    const max_dispatches: u64 = 1 + @as(u64, config.num_hidden_layers) * 15 + 3;
    if (max_dispatches > metal.UNIFORM_SLOTS_PER_FRAME) {
        return error.TooManyDispatchesForFrame;
    }

    const parsed = source.parsed();
    const num_layers: usize = config.num_hidden_layers;
    const h = config.hidden_size;
    const q_dim = config.qDim();
    const kv_dim = config.kvDim();

    // --- 1. Resolve & validate all required directory entries up front ---
    const embed_entry = try findLinear(parsed, "embed_tokens");
    const embed_wdt = try validateLinear2D(embed_entry, config.vocab_size, h);

    const final_norm_entry = try findRequired(parsed, "norm.weight");
    const final_norm_dt = try validateNorm1D(final_norm_entry, h);

    const lm_head_entry = if (config.tie_word_embeddings)
        embed_entry
    else
        try findLinear(parsed, "lm_head");
    const lm_head_wdt = try validateLinear2D(lm_head_entry, config.vocab_size, h);

    const group_count = 1 + num_layers;
    const group_bytes = try allocator.alloc(u64, group_count);
    defer allocator.free(group_bytes);
    @memset(group_bytes, 0);

    group_bytes[0] = embed_entry.blob_bytes + final_norm_entry.blob_bytes +
        if (config.tie_word_embeddings) @as(u64, 0) else lm_head_entry.blob_bytes;

    var name_buf: [64]u8 = undefined;
    for (0..num_layers) |l| {
        var bytes_l: u64 = 0;

        const in_ln = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable);
        _ = try validateNorm1D(in_ln, h);
        bytes_l += in_ln.blob_bytes;

        const q_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(q_e, q_dim, h);
        bytes_l += q_e.blob_bytes;

        const k_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(k_e, kv_dim, h);
        bytes_l += k_e.blob_bytes;

        const v_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(v_e, kv_dim, h);
        bytes_l += v_e.blob_bytes;

        const o_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(o_e, h, q_dim);
        bytes_l += o_e.blob_bytes;

        const qn_opt = parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable);
        const kn_opt = parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable);
        if ((qn_opt != null) != (kn_opt != null)) return error.MissingTensor;
        if (qn_opt) |qn| {
            const qn_dt = try validateNorm1D(qn, config.head_dim);
            const kn_dt = try validateNorm1D(kn_opt.?, config.head_dim);
            if (qn_dt != kn_dt) return error.UnexpectedTensorDType;
            bytes_l += qn.blob_bytes + kn_opt.?.blob_bytes;
        }

        const post_ln = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable);
        _ = try validateNorm1D(post_ln, h);
        bytes_l += post_ln.blob_bytes;

        switch (config.arch) {
            .qwen3_moe => {
                const router_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate", .{l}) catch unreachable);
                _ = try validateRouter2D(router_e, config.num_experts, h);
                bytes_l += router_e.blob_bytes;

                var layer_exp_dt: ?quant.WeightDType = null;
                for (0..config.num_experts) |e_idx| {
                    const gp = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e_idx }) catch unreachable);
                    const gp_dt = try validateLinear2D(gp, config.moe_intermediate_size, h);
                    if (layer_exp_dt) |expected| {
                        if (gp_dt != expected) return error.UnexpectedTensorDType;
                    } else {
                        layer_exp_dt = gp_dt;
                        if (gp_dt == .f16 or gp_dt == .bf16) {
                            if ((h % 4) != 0 or (config.moe_intermediate_size % 4) != 0) return error.UnalignedK;
                        }
                    }
                    bytes_l += gp.blob_bytes;

                    const up = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e_idx }) catch unreachable);
                    if ((try validateLinear2D(up, config.moe_intermediate_size, h)) != layer_exp_dt.?) return error.UnexpectedTensorDType;
                    bytes_l += up.blob_bytes;

                    const dp = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e_idx }) catch unreachable);
                    if ((try validateLinear2D(dp, h, config.moe_intermediate_size)) != layer_exp_dt.?) return error.UnexpectedTensorDType;
                    bytes_l += dp.blob_bytes;
                }
            },
            .qwen3_dense => {
                const gp = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable);
                _ = try validateLinear2D(gp, config.intermediate_size, h);
                bytes_l += gp.blob_bytes;

                const up = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}) catch unreachable);
                _ = try validateLinear2D(up, config.intermediate_size, h);
                bytes_l += up.blob_bytes;

                const dp = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}) catch unreachable);
                _ = try validateLinear2D(dp, h, config.intermediate_size);
                bytes_l += dp.blob_bytes;
            },
        }
        group_bytes[1 + l] = bytes_l;
    }

    // --- 2. Query Metal device, compute shard plan & enforce wired budget ---
    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    errdefer device.close();
    const dev_info = try device.info();
    const wired_budget = options.budget_override orelse budget_mod.Budget.fromHost(facts, dev_info, .{});

    const raw_shard_cap: u64 = if (options.max_shard_bytes) |cap|
        @min(@as(u64, cap), dev_info.maxBufferLength())
    else
        dev_info.maxBufferLength();
    const shard_cap: u64 = (raw_shard_cap / page.APPLE_PAGE_SIZE) * page.APPLE_PAGE_SIZE;
    if (shard_cap == 0) return error.TensorExceedsShardLimit;

    // Bin-pack atomic groups (global + per-layer) into `MTLBuffer`-sized shards.
    const group_shard = try allocator.alloc(usize, group_count);
    defer allocator.free(group_shard);
    var shard_sizes: std.ArrayList(u64) = .empty;
    defer shard_sizes.deinit(allocator);

    for (group_bytes, 0..) |gb, g_idx| {
        if (gb > shard_cap) return error.TensorExceedsShardLimit;
        if (shard_sizes.items.len == 0 or shard_sizes.items[shard_sizes.items.len - 1] + gb > shard_cap) {
            try shard_sizes.append(allocator, gb);
        } else {
            shard_sizes.items[shard_sizes.items.len - 1] += gb;
        }
        group_shard[g_idx] = shard_sizes.items.len - 1;
    }

    // Compute scratch arena footprint.
    const gqa_layout = attention.gqaScratchLayout(config.num_attention_heads, config.head_dim, attention.MAX_SPLITS);
    const kv_tbl_per_layer = amf.align64(@as(usize, num_chunks) * @sizeOf(attention.KvTilePair));
    const expert_tbl_per_layer: usize = if (config.arch == .qwen3_moe)
        amf.align64(@as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights))
    else
        0;
    const moe_ws_bytes: usize = if (config.arch == .qwen3_moe)
        try moe.MoeWorkspace.requiredBytes(.{
            .max_tokens = 1,
            .num_experts = config.num_experts,
            .top_k = config.num_experts_per_tok,
            .hidden_size = h,
            .intermediate_size = config.moe_intermediate_size,
        })
    else
        0;
    const dense_ffn_scratch: usize = if (config.arch == .qwen3_dense)
        3 * amf.align64(@as(usize, config.intermediate_size) * @sizeOf(f16))
    else
        0;

    const scratch_raw: usize =
        2 * amf.align64(@as(usize, h) * @sizeOf(f16)) + // x, x_norm
        2 * amf.align64(@as(usize, q_dim) * @sizeOf(f16)) + // q, attn_out
        2 * amf.align64(@as(usize, kv_dim) * @sizeOf(f16)) + // k, v
        amf.align64(@as(usize, h) * @sizeOf(f16)) + // proj_out
        amf.align64(@as(usize, config.vocab_size) * @sizeOf(f16)) + // logits
        64 + // out_token_id
        gqa_layout.total_bytes +
        num_layers * kv_tbl_per_layer +
        num_layers * expert_tbl_per_layer +
        moe_ws_bytes +
        dense_ffn_scratch +
        4 * page.APPLE_PAGE_SIZE;
    const scratch_arena_bytes = try page.alignUp(scratch_raw);

    // Compute total KV tile bytes (exact 16 KiB multiple per tile).
    const kv_tile_rows: u32 = config.num_key_value_heads * options.chunk_tokens;
    const kv_tile_alloc_bytes: u64 = try page.alignUp(@as(usize, kv_tile_rows) * config.head_dim * @sizeOf(f16));
    const total_kv_tiles: usize = num_layers * @as(usize, num_chunks) * 2;
    const total_kv_bytes: u64 = @as(u64, total_kv_tiles) * kv_tile_alloc_bytes;

    var total_weight_bytes: u64 = 0;
    for (shard_sizes.items) |sb| total_weight_bytes += sb;
    const total_wired_bytes: u64 = total_weight_bytes + scratch_arena_bytes + total_kv_bytes;

    // Enforce ADR-0012 wired memory budget BEFORE allocating a single page.
    try wired_budget.checkPlan(total_wired_bytes);

    // --- 3. Allocate wired weight shards and wrap zero-copy MTLBuffers ---
    const shards = try allocator.alloc(WeightShard, shard_sizes.items.len);
    var shards_inited: usize = 0;
    errdefer {
        for (0..shards_inited) |i| {
            shards[i].buffer.close();
            shards[i].arena.deinit();
        }
        allocator.free(shards);
    }
    for (shard_sizes.items, 0..) |sb, i| {
        var arena = try arena_mod.WiredArena.init(@intCast(sb), wired_budget);
        errdefer arena.deinit();
        const buf = try metal.Buffer.wrap(device, arena.bytes());
        shards[i] = .{ .arena = arena, .buffer = buf };
        shards_inited += 1;
    }

    // --- 4. Allocate scratch arena and zero-copy MTLBuffer ---
    var scratch_arena = try arena_mod.WiredArena.init(scratch_arena_bytes, wired_budget);
    errdefer scratch_arena.deinit();
    const scratch_buf = try metal.Buffer.wrap(device, scratch_arena.bytes());
    errdefer scratch_buf.close();

    // --- 5. Allocate IOSurface RGBA16Float KV tiles per layer & chunk ---
    const kv_tiles = try allocator.alloc(metal.KvTile, total_kv_tiles);
    var tiles_inited: usize = 0;
    errdefer {
        for (0..tiles_inited) |i| kv_tiles[i].close();
        allocator.free(kv_tiles);
    }
    for (0..total_kv_tiles) |i| {
        kv_tiles[i] = try metal.KvTile.init(device, config.head_dim, kv_tile_rows, .{});
        tiles_inited += 1;
    }

    // --- 6. Compile pipelines and create Metal 4 compute context + residency ---
    const library = try metal.Library.openEmbedded(device);
    errdefer library.close();
    const dequant_pipes = try dequant.DequantPipelines.init(device, library);
    errdefer dequant_pipes.close();
    const attn_pipes = try attention.AttentionPipelines.init(device, library);
    errdefer attn_pipes.close();
    const moe_pipes = try moe.MoePipelines.init(device, library);
    errdefer moe_pipes.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    errdefer compute.close();

    // Register every weight shard buffer, scratch buffer, and KV tile in MTLResidencySet.
    for (shards) |shard| try compute.makeResident(&.{shard.buffer});
    try compute.makeResident(&.{scratch_buf});
    try compute.makeTilesResident(kv_tiles);

    // --- 7. Carve hot-path scratch buffers & populate bindless KV tile tables ---
    const x_raw = try scratch_arena.allocAligned(@as(usize, h) * @sizeOf(f16), 64);
    const x_norm_raw = try scratch_arena.allocAligned(@as(usize, h) * @sizeOf(f16), 64);
    const q_raw = try scratch_arena.allocAligned(@as(usize, q_dim) * @sizeOf(f16), 64);
    const k_raw = try scratch_arena.allocAligned(@as(usize, kv_dim) * @sizeOf(f16), 64);
    const v_raw = try scratch_arena.allocAligned(@as(usize, kv_dim) * @sizeOf(f16), 64);
    const attn_out_raw = try scratch_arena.allocAligned(@as(usize, q_dim) * @sizeOf(f16), 64);
    const proj_out_raw = try scratch_arena.allocAligned(@as(usize, h) * @sizeOf(f16), 64);
    const logits_raw = try scratch_arena.allocAligned(@as(usize, config.vocab_size) * @sizeOf(f16), 64);
    const out_tok_raw = try scratch_arena.allocAligned(@sizeOf(u32), 64);
    const gqa_scratch_raw = try scratch_arena.allocAligned(gqa_layout.total_bytes, 64);
    @memset(out_tok_raw, 0);

    const moe_ws: ?moe.MoeWorkspace = if (config.arch == .qwen3_moe)
        try moe.MoeWorkspace.init(&scratch_arena, .{
            .max_tokens = 1,
            .num_experts = config.num_experts,
            .top_k = config.num_experts_per_tok,
            .hidden_size = h,
            .intermediate_size = config.moe_intermediate_size,
        })
    else
        null;

    var dense_gate_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    var dense_up_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    var dense_act_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    if (config.arch == .qwen3_dense) {
        const inter_bytes = @as(usize, config.intermediate_size) * @sizeOf(f16);
        const dg = try scratch_arena.allocAligned(inter_bytes, 64);
        const du = try scratch_arena.allocAligned(inter_bytes, 64);
        const da = try scratch_arena.allocAligned(inter_bytes, 64);
        dense_gate_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(dg.ptr) };
        dense_up_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(du.ptr) };
        dense_act_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(da.ptr) };
    }

    // --- 8. Load global weight blobs into shard 0 (zero-duplicate tied lm_head) ---
    const g_shard = &shards[group_shard[0]];
    const embed_blob = try loadBlobIntoShard(source, embed_entry, g_shard, options.verify_crcs);
    const final_norm_blob = try loadBlobIntoShard(source, final_norm_entry, g_shard, options.verify_crcs);
    const lm_head_blob = if (config.tie_word_embeddings)
        embed_blob
    else
        try loadBlobIntoShard(source, lm_head_entry, g_shard, options.verify_crcs);

    const embed_lin = embed_blob.asLinearBinding(embed_wdt);
    const lm_head_lin = lm_head_blob.asLinearBinding(lm_head_wdt);

    const embed_row_alg_bytes: u64 = switch (embed_wdt) {
        .f16, .bf16 => @as(u64, h) * 2,
        .q3_affine => (try quant.QuantOps(.{ .bits = .q3, .group_size = 64 }).rowWeightBytes(h)) +
            @as(u64, h / 64) * @sizeOf(quant.ScaleBias),
        .q4_affine => (try quant.QuantOps(.{ .bits = .q4, .group_size = 64 }).rowWeightBytes(h)) +
            @as(u64, h / 64) * @sizeOf(quant.ScaleBias),
    };
    const global_head_bytes: u64 = embed_row_alg_bytes + final_norm_entry.data_bytes + lm_head_lin.alg_bytes;

    // --- 9. Load per-layer weights into assigned shards & build device tables ---
    const layers = try allocator.alloc(LayerWeights, num_layers);
    errdefer allocator.free(layers);

    var total_layers_static_bytes: u64 = 0;
    const dummy_linear: LinearBinding = .{
        .weight = .{ .buffer = scratch_buf, .offset = 0 },
        .scale_bias = null,
        .n = 0,
        .k = 0,
        .dtype = .bf16,
        .alg_bytes = 0,
    };

    for (0..num_layers) |l| {
        const l_shard = &shards[group_shard[1 + l]];

        // Bindless KV chunk table for layer `l`.
        const kv_tbl_raw = try scratch_arena.allocAligned(@as(usize, num_chunks) * @sizeOf(attention.KvTilePair), 64);
        const kv_tbl: []attention.KvTilePair = @alignCast(std.mem.bytesAsSlice(attention.KvTilePair, kv_tbl_raw));
        for (0..num_chunks) |c_idx| {
            const tbase = (l * @as(usize, num_chunks) + c_idx) * 2;
            kv_tbl[c_idx] = attention.KvTilePair.fromTiles(kv_tiles[tbase], kv_tiles[tbase + 1], .rgba16f);
        }

        const in_ln_e = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable);
        const in_ln_dt = try validateNorm1D(in_ln_e, h);
        const in_ln_b = (try loadBlobIntoShard(source, in_ln_e, l_shard, options.verify_crcs)).asNormBinding(in_ln_dt);

        const q_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable);
        const q_b = (try loadBlobIntoShard(source, q_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(q_e, q_dim, h));

        const k_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable);
        const k_b = (try loadBlobIntoShard(source, k_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(k_e, kv_dim, h));

        const v_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable);
        const v_b = (try loadBlobIntoShard(source, v_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(v_e, kv_dim, h));

        const o_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable);
        const o_b = (try loadBlobIntoShard(source, o_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(o_e, h, q_dim));

        var q_norm_bind: ?metal.BufferBinding = null;
        var k_norm_bind: ?metal.BufferBinding = null;
        var qk_norm_dtype: quant.FloatDType = .f16;
        var qk_norm_bytes: u64 = 0;
        if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable)) |qn_e| {
            qk_norm_dtype = try validateNorm1D(qn_e, config.head_dim);
            q_norm_bind = (try loadBlobIntoShard(source, qn_e, l_shard, options.verify_crcs)).asBinding();
            qk_norm_bytes += qn_e.data_bytes;
        }
        if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable)) |kn_e| {
            k_norm_bind = (try loadBlobIntoShard(source, kn_e, l_shard, options.verify_crcs)).asBinding();
            qk_norm_bytes += kn_e.data_bytes;
        }

        const post_ln_e = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable);
        const post_ln_dt = try validateNorm1D(post_ln_e, h);
        const post_ln_b = (try loadBlobIntoShard(source, post_ln_e, l_shard, options.verify_crcs)).asNormBinding(post_ln_dt);

        var layer_w: LayerWeights = .{
            .input_layernorm = in_ln_b,
            .q_proj = q_b,
            .k_proj = k_b,
            .v_proj = v_b,
            .o_proj = o_b,
            .q_norm = q_norm_bind,
            .k_norm = k_norm_bind,
            .qk_norm_dtype = qk_norm_dtype,
            .post_attention_layernorm = post_ln_b,
            .router = .{ .weights = .{ .binding = .{ .buffer = scratch_buf, .offset = 0 }, .dtype = .f16 } },
            .expert_dtype = .q3_affine,
            .expert_weights_buf = l_shard.buffer,
            .expert_table = .{ .buffer = scratch_buf, .offset = 0 },
            .dense_gate = dummy_linear,
            .dense_up = dummy_linear,
            .dense_down = dummy_linear,
            .kv_tiles_binding = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(kv_tbl_raw.ptr) },
        };

        var ffn_active_bytes: u64 = 0;
        switch (config.arch) {
            .qwen3_moe => {
                const router_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate", .{l}) catch unreachable);
                const router_dt = try validateRouter2D(router_e, config.num_experts, h);
                const router_b = try loadBlobIntoShard(source, router_e, l_shard, options.verify_crcs);
                layer_w.router = switch (router_dt) {
                    .f16 => .{ .weights = .{ .binding = router_b.asBinding(), .dtype = .f16 } },
                    .bf16 => .{ .weights = .{ .binding = router_b.asBinding(), .dtype = .bf16 } },
                    .f32 => .{ .weights = .{ .binding = router_b.asBinding(), .dtype = .f32 } },
                    .q4_affine => .{ .quant_weights = .{
                        .qweight = router_b.asBinding(),
                        .scale_bias = .{ .buffer = router_b.buffer, .offset = router_b.offset + router_e.params_offset },
                        .bits = .q4,
                    } },
                    .q3_affine => .{ .quant_weights = .{
                        .qweight = router_b.asBinding(),
                        .scale_bias = .{ .buffer = router_b.buffer, .offset = router_b.offset + router_e.params_offset },
                        .bits = .q3,
                    } },
                };

                const exp_tbl_raw = try scratch_arena.allocAligned(@as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights), 64);
                const exp_tbl: []moe.ExpertWeights = @alignCast(std.mem.bytesAsSlice(moe.ExpertWeights, exp_tbl_raw));
                layer_w.expert_table = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(exp_tbl_raw.ptr) };

                var exp_dt: quant.WeightDType = .q3_affine;
                for (0..config.num_experts) |e_idx| {
                    const gp_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e_idx }) catch unreachable);
                    const gp_dt = try validateLinear2D(gp_e, config.moe_intermediate_size, h);
                    if (e_idx == 0) exp_dt = gp_dt;
                    const gp_b = try loadBlobIntoShard(source, gp_e, l_shard, options.verify_crcs);

                    const up_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e_idx }) catch unreachable);
                    const up_b = try loadBlobIntoShard(source, up_e, l_shard, options.verify_crcs);

                    const dp_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e_idx }) catch unreachable);
                    const dp_b = try loadBlobIntoShard(source, dp_e, l_shard, options.verify_crcs);

                    const is_float = (exp_dt == .f16 or exp_dt == .bf16);
                    exp_tbl[e_idx] = .{
                        .gate_qweight_offset = gp_b.offset,
                        .gate_sb_offset = if (is_float) 0 else gp_b.offset + gp_e.params_offset,
                        .up_qweight_offset = up_b.offset,
                        .up_sb_offset = if (is_float) 0 else up_b.offset + up_e.params_offset,
                        .down_qweight_offset = dp_b.offset,
                        .down_sb_offset = if (is_float) 0 else dp_b.offset + dp_e.params_offset,
                    };
                }
                layer_w.expert_dtype = exp_dt;

                const router_bytes: u64 = router_e.data_bytes + router_e.params_bytes;
                const experts_bytes: u64 = switch (exp_dt) {
                    .q3_affine => try moe.expertLayerBytes(.q3, config.num_experts_per_tok, h, config.moe_intermediate_size),
                    .q4_affine => try moe.expertLayerBytes(.q4, config.num_experts_per_tok, h, config.moe_intermediate_size),
                    .f16, .bf16 => try moe.expertLayerBytesFloat(config.num_experts_per_tok, h, config.moe_intermediate_size),
                };
                ffn_active_bytes = router_bytes + experts_bytes;
            },
            .qwen3_dense => {
                const gp_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable);
                const gp_b = (try loadBlobIntoShard(source, gp_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(gp_e, config.intermediate_size, h));

                const up_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}) catch unreachable);
                const up_b = (try loadBlobIntoShard(source, up_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(up_e, config.intermediate_size, h));

                const dp_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}) catch unreachable);
                const dp_b = (try loadBlobIntoShard(source, dp_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(dp_e, h, config.intermediate_size));

                layer_w.dense_gate = gp_b;
                layer_w.dense_up = up_b;
                layer_w.dense_down = dp_b;
                ffn_active_bytes = gp_b.alg_bytes + up_b.alg_bytes + dp_b.alg_bytes;
            },
        }

        total_layers_static_bytes +=
            in_ln_e.data_bytes + post_ln_e.data_bytes + qk_norm_bytes +
            q_b.alg_bytes + k_b.alg_bytes + v_b.alg_bytes + o_b.alg_bytes +
            ffn_active_bytes;
        layers[l] = layer_w;
    }

    return .{
        .config = config,
        .max_seq_len = max_seq_len,
        .chunk_tokens = options.chunk_tokens,
        .num_chunks = num_chunks,
        .timeout_ms = options.timeout_ms,
        .wired_budget_bytes = wired_budget.limit_bytes,
        .total_wired_bytes = total_wired_bytes,
        .seq_len = 0,
        .device = device,
        .library = library,
        .dequant_pipes = dequant_pipes,
        .attn_pipes = attn_pipes,
        .moe_pipes = moe_pipes,
        .compute = compute,
        .shards = shards,
        .scratch_arena = scratch_arena,
        .scratch_buf = scratch_buf,
        .kv_tiles = kv_tiles,
        .layers = layers,
        .embed = embed_lin,
        .final_norm = final_norm_blob.asNormBinding(final_norm_dt),
        .lm_head = lm_head_lin,
        .logits_host = @alignCast(std.mem.bytesAsSlice(f16, logits_raw)),
        .out_token_id_host = @ptrCast(@alignCast(out_tok_raw.ptr)),
        .x_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_raw.ptr) },
        .x_norm_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_norm_raw.ptr) },
        .q_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(q_raw.ptr) },
        .k_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(k_raw.ptr) },
        .v_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(v_raw.ptr) },
        .attn_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(attn_out_raw.ptr) },
        .proj_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(proj_out_raw.ptr) },
        .logits_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(logits_raw.ptr) },
        .out_token_id_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(out_tok_raw.ptr) },
        .gqa_scratch_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gqa_scratch_raw.ptr) },
        .moe_ws = moe_ws,
        .dense_gate_bind = dense_gate_bind,
        .dense_up_bind = dense_up_bind,
        .dense_act_bind = dense_act_bind,
        .total_layers_static_bytes = total_layers_static_bytes,
        .global_head_bytes = global_head_bytes,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "ModelConfig.fromJson parses MLX quantized and HF unquantized BF16/FP16 configs" {
    const gpa = std.testing.allocator;
    const moe_json =
        \\{
        \\  "model_type": "qwen3_moe",
        \\  "vocab_size": 151936,
        \\  "hidden_size": 2048,
        \\  "num_hidden_layers": 48,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 4,
        \\  "head_dim": 128,
        \\  "moe_intermediate_size": 768,
        \\  "num_experts": 128,
        \\  "num_experts_per_tok": 8,
        \\  "norm_topk_prob": true,
        \\  "rms_norm_eps": 1e-6,
        \\  "rope_theta": 1000000.0,
        \\  "max_position_embeddings": 32768,
        \\  "tie_word_embeddings": false,
        \\  "bos_token_id": 151643,
        \\  "eos_token_id": [151645, 151643],
        \\  "quantization": { "bits": 3, "group_size": 64 }
        \\}
    ;
    const moe_cfg = try ModelConfig.fromJson(gpa, moe_json);
    try std.testing.expectEqual(ModelArch.qwen3_moe, moe_cfg.arch);
    try std.testing.expectEqual(@as(?quant.QuantBits, .q3), moe_cfg.quant_bits);
    try std.testing.expectEqualStrings("q3", moe_cfg.precisionLabel());
    try std.testing.expectEqual(@as(u32, 4096), moe_cfg.qDim());
    try std.testing.expectEqual(@as(u32, 512), moe_cfg.kvDim());
    try std.testing.expectEqual(@as(u32, 128), moe_cfg.num_experts);
    try std.testing.expectEqual(@as(u32, 8), moe_cfg.num_experts_per_tok);
    try std.testing.expectEqual(@as(?u32, 151645), moe_cfg.eos_token_id);

    // MLX config using `quantization_config` only
    const dense_json =
        \\{
        \\  "model_type": "qwen3",
        \\  "vocab_size": 151936,
        \\  "hidden_size": 5120,
        \\  "num_hidden_layers": 40,
        \\  "num_attention_heads": 40,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 128,
        \\  "intermediate_size": 17408,
        \\  "rms_norm_eps": 1e-6,
        \\  "rope_theta": 1000000.0,
        \\  "tie_word_embeddings": true,
        \\  "eos_token_id": 151645,
        \\  "quantization_config": { "bits": 4, "group_size": 64 }
        \\}
    ;
    const dense_cfg = try ModelConfig.fromJson(gpa, dense_json);
    try std.testing.expectEqual(ModelArch.qwen3_dense, dense_cfg.arch);
    try std.testing.expectEqual(@as(?quant.QuantBits, .q4), dense_cfg.quant_bits);
    try std.testing.expectEqualStrings("q4", dense_cfg.precisionLabel());
    try std.testing.expectEqual(@as(u32, 17408), dense_cfg.intermediate_size);
    try std.testing.expect(dense_cfg.tie_word_embeddings);

    // Unquantized HuggingFace Qwen3-0.6B BF16 config (`qDim = 2048 != hidden_size = 1024`)
    const bf16_json =
        \\{
        \\  "model_type": "qwen3",
        \\  "vocab_size": 151936,
        \\  "hidden_size": 1024,
        \\  "num_hidden_layers": 28,
        \\  "num_attention_heads": 16,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 128,
        \\  "intermediate_size": 3072,
        \\  "rms_norm_eps": 1e-6,
        \\  "rope_theta": 1000000,
        \\  "max_position_embeddings": 40960,
        \\  "tie_word_embeddings": true,
        \\  "torch_dtype": "bfloat16",
        \\  "bos_token_id": 151643,
        \\  "eos_token_id": 151645
        \\}
    ;
    const bf16_cfg = try ModelConfig.fromJson(gpa, bf16_json);
    try std.testing.expectEqual(ModelArch.qwen3_dense, bf16_cfg.arch);
    try std.testing.expectEqual(@as(?quant.QuantBits, null), bf16_cfg.quant_bits);
    try std.testing.expectEqual(quant.FloatDType.bf16, bf16_cfg.default_float_dtype);
    try std.testing.expectEqualStrings("bf16", bf16_cfg.precisionLabel());
    try std.testing.expectEqual(@as(u32, 2048), bf16_cfg.qDim());
    try std.testing.expectEqual(@as(u32, 1024), bf16_cfg.hidden_size);
    try std.testing.expect(bf16_cfg.tie_word_embeddings);

    // Unquantized FP16 config relaxes % 64 == 0 to % 4 == 0
    const fp16_relaxed_json =
        \\{
        \\  "model_type": "qwen3",
        \\  "vocab_size": 512,
        \\  "hidden_size": 132,
        \\  "num_hidden_layers": 2,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 64,
        \\  "intermediate_size": 196,
        \\  "torch_dtype": "float16"
        \\}
    ;
    var fp16_cfg = try ModelConfig.fromJson(gpa, fp16_relaxed_json);
    try std.testing.expectEqual(@as(?quant.QuantBits, null), fp16_cfg.quant_bits);
    try std.testing.expectEqual(quant.FloatDType.f16, fp16_cfg.default_float_dtype);
    try std.testing.expectEqualStrings("f16", fp16_cfg.precisionLabel());
    // Same non-64-aligned geometry must be rejected when quantized:
    fp16_cfg.quant_bits = .q4;
    try std.testing.expectError(error.InvalidConfig, fp16_cfg.validate());

    try std.testing.expectError(error.UnsupportedArchitecture, ModelConfig.fromJson(gpa, "{\"model_type\":\"llama\"}"));
    try std.testing.expectError(error.InvalidJson, ModelConfig.fromJson(gpa, "{broken"));
}

const SyntheticRecord = struct {
    name: []const u8,
    kind: amf.TensorKind,
    dtype: amf.DType,
    quant_bits: u8,
    group_size: u16,
    layer: u16,
    expert: u16,
    shape: [4]u32,
    rank: u8,
    data: []u8,
    params: []u8,
};

const SynthDTypeSpec = struct {
    linear_dtype: quant.WeightDType,
    embed_dtype: quant.WeightDType,
    norm_dtype: quant.FloatDType = .f16,
    router_dtype: amf.DType = .f16,
    expert_dtype: ?quant.WeightDType = null,
    use_weight_suffix: bool = false,
};

fn nextRng(state: *u64) u64 {
    var x = state.*;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state.* = x;
    return x;
}

fn fillQuantTensor(
    allocator: std.mem.Allocator,
    bits: quant.QuantBits,
    n: u32,
    k: u32,
    scale_base: f32,
    seed: u64,
) !struct { data: []u8, params: []u8 } {
    var rng = seed ^ 0x9E37_79B9_7F4A_7C15;
    const w_bytes = switch (bits) {
        .q3 => try quant.QuantOps(.{ .bits = .q3, .group_size = 64 }).packedWeightBytes(n, k),
        .q4 => try quant.QuantOps(.{ .bits = .q4, .group_size = 64 }).packedWeightBytes(n, k),
    };
    const sb_bytes = switch (bits) {
        .q3 => try quant.QuantOps(.{ .bits = .q3, .group_size = 64 }).scaleBiasBytes(n, k),
        .q4 => try quant.QuantOps(.{ .bits = .q4, .group_size = 64 }).scaleBiasBytes(n, k),
    };
    const data = try allocator.alloc(u8, w_bytes);
    errdefer allocator.free(data);
    for (data) |*b| b.* = @truncate(nextRng(&rng));

    const params = try allocator.alloc(u8, sb_bytes);
    errdefer allocator.free(params);
    const sbs: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, params));
    const mid: f32 = if (bits == .q3) 3.5 else 7.5;
    for (sbs) |*sb| {
        const u = @as(f32, @floatFromInt(nextRng(&rng) & 0xFF)) / 255.0;
        const s = scale_base * (0.75 + 0.5 * u);
        const jitter = (@as(f32, @floatFromInt(nextRng(&rng) & 0xFF)) / 255.0 - 0.5) * 0.25 * s;
        sb.* = .{
            .scale = @floatCast(s),
            .bias = @floatCast(-mid * s + jitter),
        };
    }
    return .{ .data = data, .params = params };
}

fn fillFloat16Tensor(
    allocator: std.mem.Allocator,
    dtype: quant.FloatDType,
    elems: usize,
    amp: f32,
    centered_one: bool,
    seed: u64,
) ![]u8 {
    var rng = seed ^ 0xA076_1D64_78BD_642F;
    const raw = try allocator.alloc(u8, elems * @sizeOf(u16));
    const vals: []u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
    for (vals) |*v| {
        const u = @as(f32, @floatFromInt(nextRng(&rng) & 0xFFFF)) / 65535.0;
        const f = if (centered_one) (0.95 + 0.1 * u) else ((u * 2.0 - 1.0) * amp);
        v.* = quant.f32ToFloatBits(dtype, f);
    }
    return raw;
}

fn weightDTypeToAmf(wdt: quant.WeightDType) amf.DType {
    return switch (wdt) {
        .f16 => .f16,
        .bf16 => .bf16,
        .q4_affine => .q4_affine,
        .q3_affine => .q3_affine,
    };
}

fn floatDTypeToAmf(fdt: quant.FloatDType) amf.DType {
    return switch (fdt) {
        .f16 => .f16,
        .bf16 => .bf16,
    };
}

fn buildSyntheticAmfImageWithSpec(
    allocator: std.mem.Allocator,
    cfg: ModelConfig,
    spec: SynthDTypeSpec,
    seed: u64,
) ![]u8 {
    var recs: std.ArrayList(SyntheticRecord) = .empty;
    defer {
        for (recs.items) |r| {
            allocator.free(r.name);
            allocator.free(r.data);
            allocator.free(r.params);
        }
        recs.deinit(allocator);
    }

    const h = cfg.hidden_size;
    const q_dim = cfg.qDim();
    const kv_dim = cfg.kvDim();
    const exp_wdt = spec.expert_dtype orelse spec.linear_dtype;
    var t_idx: u64 = 1;

    const addWeight2D = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayList(SyntheticRecord),
            base_name: []const u8,
            use_suffix: bool,
            kind: amf.TensorKind,
            wdt: quant.WeightDType,
            layer: u16,
            expert: u16,
            n: u32,
            k: u32,
            amp_or_scale: f32,
            s: u64,
        ) !void {
            const is_float = (wdt == .f16 or wdt == .bf16);
            const name = if (use_suffix and is_float)
                try std.fmt.allocPrint(alloc, "{s}.weight", .{base_name})
            else
                try alloc.dupe(u8, base_name);
            errdefer alloc.free(name);

            switch (wdt) {
                .f16, .bf16 => {
                    const fdt: quant.FloatDType = if (wdt == .f16) .f16 else .bf16;
                    const data = try fillFloat16Tensor(alloc, fdt, @as(usize, n) * k, amp_or_scale, false, s);
                    errdefer alloc.free(data);
                    try list.append(alloc, .{
                        .name = name,
                        .kind = kind,
                        .dtype = floatDTypeToAmf(fdt),
                        .quant_bits = 0,
                        .group_size = 0,
                        .layer = layer,
                        .expert = expert,
                        .shape = .{ n, k, 0, 0 },
                        .rank = 2,
                        .data = data,
                        .params = try alloc.alloc(u8, 0),
                    });
                },
                .q3_affine, .q4_affine => {
                    const qb: quant.QuantBits = if (wdt == .q3_affine) .q3 else .q4;
                    const q = try fillQuantTensor(alloc, qb, n, k, amp_or_scale, s);
                    errdefer {
                        alloc.free(q.data);
                        alloc.free(q.params);
                    }
                    try list.append(alloc, .{
                        .name = name,
                        .kind = kind,
                        .dtype = weightDTypeToAmf(wdt),
                        .quant_bits = qb.bitWidth(),
                        .group_size = 64,
                        .layer = layer,
                        .expert = expert,
                        .shape = .{ n, k, 0, 0 },
                        .rank = 2,
                        .data = q.data,
                        .params = q.params,
                    });
                },
            }
        }
    }.call;

    const addNorm1D = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayList(SyntheticRecord),
            name: []u8,
            fdt: quant.FloatDType,
            layer: u16,
            dim: u32,
            s: u64,
        ) !void {
            errdefer alloc.free(name);
            const data = try fillFloat16Tensor(alloc, fdt, dim, 0.05, true, s);
            errdefer alloc.free(data);
            try list.append(alloc, .{
                .name = name,
                .kind = .norm,
                .dtype = floatDTypeToAmf(fdt),
                .quant_bits = 0,
                .group_size = 0,
                .layer = layer,
                .expert = amf.NO_EXPERT,
                .shape = .{ dim, 0, 0, 0 },
                .rank = 1,
                .data = data,
                .params = try alloc.alloc(u8, 0),
            });
        }
    }.call;

    const scaleFor = struct {
        fn call(wdt: quant.WeightDType, inv_sqrt: f32) f32 {
            return switch (wdt) {
                .q3_affine => 0.35 * inv_sqrt,
                .q4_affine => 0.17 * inv_sqrt,
                .f16, .bf16 => 1.0 * inv_sqrt,
            };
        }
    }.call;

    // embed_tokens
    const emb_scale: f32 = switch (spec.embed_dtype) {
        .q3_affine => 0.08,
        .q4_affine => 0.04,
        .f16, .bf16 => 0.25,
    };
    try addWeight2D(allocator, &recs, "embed_tokens", spec.use_weight_suffix, .embedding, spec.embed_dtype, amf.NO_LAYER, amf.NO_EXPERT, cfg.vocab_size, h, emb_scale, seed + t_idx);
    t_idx += 1;

    const inv_sqrt_h = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
    const inv_sqrt_q = 1.0 / @sqrt(@as(f32, @floatFromInt(q_dim)));

    var name_buf: [64]u8 = undefined;
    for (0..cfg.num_hidden_layers) |l_usize| {
        const l: u16 = @intCast(l_usize);
        try addNorm1D(allocator, &recs, try std.fmt.allocPrint(allocator, "layers.{d}.input_layernorm.weight", .{l}), spec.norm_dtype, l, h, seed + t_idx);
        t_idx += 1;
        try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, spec.linear_dtype, l, amf.NO_EXPERT, q_dim, h, scaleFor(spec.linear_dtype, inv_sqrt_h), seed + t_idx);
        t_idx += 1;
        try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, spec.linear_dtype, l, amf.NO_EXPERT, kv_dim, h, scaleFor(spec.linear_dtype, inv_sqrt_h), seed + t_idx);
        t_idx += 1;
        try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, spec.linear_dtype, l, amf.NO_EXPERT, kv_dim, h, scaleFor(spec.linear_dtype, inv_sqrt_h), seed + t_idx);
        t_idx += 1;
        try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, spec.linear_dtype, l, amf.NO_EXPERT, h, q_dim, scaleFor(spec.linear_dtype, inv_sqrt_q), seed + t_idx);
        t_idx += 1;
        try addNorm1D(allocator, &recs, try std.fmt.allocPrint(allocator, "layers.{d}.self_attn.q_norm.weight", .{l}), spec.norm_dtype, l, cfg.head_dim, seed + t_idx);
        t_idx += 1;
        try addNorm1D(allocator, &recs, try std.fmt.allocPrint(allocator, "layers.{d}.self_attn.k_norm.weight", .{l}), spec.norm_dtype, l, cfg.head_dim, seed + t_idx);
        t_idx += 1;
        try addNorm1D(allocator, &recs, try std.fmt.allocPrint(allocator, "layers.{d}.post_attention_layernorm.weight", .{l}), spec.norm_dtype, l, h, seed + t_idx);
        t_idx += 1;

        if (cfg.arch == .qwen3_moe) {
            const router_base = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate", .{l}) catch unreachable;
            switch (spec.router_dtype) {
                .f16 => try addWeight2D(allocator, &recs, router_base, true, .router, .f16, l, amf.NO_EXPERT, cfg.num_experts, h, 0.6 * inv_sqrt_h, seed + t_idx),
                .bf16 => try addWeight2D(allocator, &recs, router_base, true, .router, .bf16, l, amf.NO_EXPERT, cfg.num_experts, h, 0.6 * inv_sqrt_h, seed + t_idx),
                .q3_affine => try addWeight2D(allocator, &recs, router_base, false, .router, .q3_affine, l, amf.NO_EXPERT, cfg.num_experts, h, 0.25 * inv_sqrt_h, seed + t_idx),
                .q4_affine => try addWeight2D(allocator, &recs, router_base, false, .router, .q4_affine, l, amf.NO_EXPERT, cfg.num_experts, h, 0.12 * inv_sqrt_h, seed + t_idx),
                .f32 => return error.UnexpectedTensorDType,
            }
            t_idx += 1;
            const inv_sqrt_mi = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.moe_intermediate_size)));
            for (0..cfg.num_experts) |e_usize| {
                const e: u16 = @intCast(e_usize);
                try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }) catch unreachable, spec.use_weight_suffix, .hot_expert, exp_wdt, l, e, cfg.moe_intermediate_size, h, scaleFor(exp_wdt, inv_sqrt_h), seed + t_idx);
                t_idx += 1;
                try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }) catch unreachable, spec.use_weight_suffix, .hot_expert, exp_wdt, l, e, cfg.moe_intermediate_size, h, scaleFor(exp_wdt, inv_sqrt_h), seed + t_idx);
                t_idx += 1;
                try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }) catch unreachable, spec.use_weight_suffix, .hot_expert, exp_wdt, l, e, h, cfg.moe_intermediate_size, scaleFor(exp_wdt, inv_sqrt_mi), seed + t_idx);
                t_idx += 1;
            }
        } else {
            const inv_sqrt_di = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.intermediate_size)));
            try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, spec.linear_dtype, l, amf.NO_EXPERT, cfg.intermediate_size, h, scaleFor(spec.linear_dtype, inv_sqrt_h), seed + t_idx);
            t_idx += 1;
            try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, spec.linear_dtype, l, amf.NO_EXPERT, cfg.intermediate_size, h, scaleFor(spec.linear_dtype, inv_sqrt_h), seed + t_idx);
            t_idx += 1;
            try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, spec.linear_dtype, l, amf.NO_EXPERT, h, cfg.intermediate_size, scaleFor(spec.linear_dtype, inv_sqrt_di), seed + t_idx);
            t_idx += 1;
        }
    }

    try addNorm1D(allocator, &recs, try allocator.dupe(u8, "norm.weight"), spec.norm_dtype, amf.NO_LAYER, h, seed + t_idx);
    t_idx += 1;
    if (!cfg.tie_word_embeddings) {
        try addWeight2D(allocator, &recs, "lm_head", spec.use_weight_suffix, .lm_head, spec.linear_dtype, amf.NO_LAYER, amf.NO_EXPERT, cfg.vocab_size, h, 0.5 * scaleFor(spec.linear_dtype, inv_sqrt_h), seed + t_idx);
    }

    // Serialize all records into a valid 16 KiB-aligned .amf v0 image.
    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_raw_len: usize = recs.items.len * @sizeOf(amf.TensorEntry);
    const dir_bytes: usize = try page.alignUp(dir_raw_len);

    var total_size: usize = dir_offset + dir_bytes;
    for (recs.items) |r| {
        const p_off = if (r.params.len > 0) amf.align64(r.data.len) else 0;
        const content_len = if (r.params.len > 0) p_off + r.params.len else r.data.len;
        total_size += try page.alignUp(content_len);
    }

    const image = try allocator.alloc(u8, total_size);
    errdefer allocator.free(image);
    @memset(image, 0);

    var cursor: usize = dir_offset + dir_bytes;
    const entries = try allocator.alloc(amf.TensorEntry, recs.items.len);
    defer allocator.free(entries);

    for (recs.items, 0..) |r, i| {
        const p_off: usize = if (r.params.len > 0) amf.align64(r.data.len) else 0;
        const content_len: usize = if (r.params.len > 0) p_off + r.params.len else r.data.len;
        const blob_bytes: usize = try page.alignUp(content_len);

        @memcpy(image[cursor .. cursor + r.data.len], r.data);
        if (r.params.len > 0) {
            @memcpy(image[cursor + p_off .. cursor + p_off + r.params.len], r.params);
        }
        const crc = std.hash.Crc32.hash(image[cursor .. cursor + blob_bytes]);

        var e = amf.TensorEntry.zeroed(r.name, r.kind, r.dtype);
        e.quant_bits = r.quant_bits;
        e.rank = r.rank;
        e.group_size = r.group_size;
        e.layer = r.layer;
        e.expert = r.expert;
        e.shape = r.shape;
        e.blob_offset = cursor;
        e.blob_bytes = blob_bytes;
        e.data_bytes = r.data.len;
        e.params_offset = p_off;
        e.params_bytes = r.params.len;
        e.blob_crc32 = crc;
        entries[i] = e;
        cursor += blob_bytes;
    }

    const dir_bytes_slice = std.mem.sliceAsBytes(entries);
    @memcpy(image[dir_offset .. dir_offset + dir_bytes_slice.len], dir_bytes_slice);
    const dir_crc = std.hash.Crc32.hash(dir_bytes_slice);

    var hdr: amf.Header = .{
        .magic = amf.MAGIC,
        .version = amf.FORMAT_VERSION,
        .flags = 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = amf.CHECKSUM_CRC32_IEEE,
        .tensor_count = @intCast(entries.len),
        .reserved0 = 0,
        .directory_offset = dir_offset,
        .directory_bytes = dir_bytes,
        .directory_crc32 = dir_crc,
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    const mid = "qwen3-test";
    @memcpy(hdr.model_id[0..mid.len], mid);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return image;
}

fn buildSyntheticAmfImage(
    allocator: std.mem.Allocator,
    cfg: ModelConfig,
    embed_f16: bool,
    seed: u64,
) ![]u8 {
    const lin_wdt: quant.WeightDType = if (cfg.quant_bits) |qb|
        quant.WeightDType.fromQuant(qb)
    else
        quant.WeightDType.fromFloat(cfg.default_float_dtype);
    return buildSyntheticAmfImageWithSpec(allocator, cfg, .{
        .linear_dtype = lin_wdt,
        .embed_dtype = if (embed_f16) .f16 else lin_wdt,
        .norm_dtype = .f16,
        .router_dtype = .f16,
    }, seed);
}

/// Full CPU fp32 reference transformer executing the exact same `.amf` weights
/// step-for-step for FP16, BF16, 3-bit/4-bit, and mixed-precision Qwen3 models.
const CpuTransformerRef = struct {
    cfg: ModelConfig,
    bytes: []const u8,
    parsed: amf.Parsed,
    max_seq_len: u32,
    k_cache: []f16, // [num_layers, max_seq_len, kv_dim]
    v_cache: []f16, // [num_layers, max_seq_len, kv_dim]
    logits: []f32,

    fn init(allocator: std.mem.Allocator, bytes: []const u8, cfg: ModelConfig, max_seq_len: u32) !CpuTransformerRef {
        var parsed = try amf.Parsed.parseBytes(allocator, bytes);
        errdefer parsed.deinit(allocator);
        const kv_elems: usize = @as(usize, cfg.num_hidden_layers) * max_seq_len * cfg.kvDim();
        const k_cache = try allocator.alloc(f16, kv_elems);
        errdefer allocator.free(k_cache);
        const v_cache = try allocator.alloc(f16, kv_elems);
        errdefer allocator.free(v_cache);
        const logits = try allocator.alloc(f32, cfg.vocab_size);
        errdefer allocator.free(logits);
        @memset(k_cache, @as(f16, @floatCast(0.0)));
        @memset(v_cache, @as(f16, @floatCast(0.0)));
        @memset(logits, 0.0);
        return .{
            .cfg = cfg,
            .bytes = bytes,
            .parsed = parsed,
            .max_seq_len = max_seq_len,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .logits = logits,
        };
    }

    fn deinit(self: *CpuTransformerRef, allocator: std.mem.Allocator) void {
        allocator.free(self.logits);
        allocator.free(self.v_cache);
        allocator.free(self.k_cache);
        self.parsed.deinit(allocator);
    }

    fn normBits(self: *const CpuTransformerRef, name: []const u8) !struct { dtype: quant.FloatDType, bits: []const u16 } {
        const e = self.parsed.find(name) orelse return error.MissingTensor;
        const dt = try e.dataType();
        const fdt: quant.FloatDType = switch (dt) {
            .f16 => .f16,
            .bf16 => .bf16,
            else => return error.UnexpectedTensorDType,
        };
        const start: usize = @intCast(e.blob_offset);
        const len: usize = @intCast(e.data_bytes);
        return .{
            .dtype = fdt,
            .bits = @alignCast(std.mem.bytesAsSlice(u16, self.bytes[start .. start + len])),
        };
    }

    fn linearMatvec(self: *const CpuTransformerRef, base_name: []const u8, x: []const f16, y: []f32) !void {
        const e = try findLinear(&self.parsed, base_name);
        const blob_start: usize = @intCast(e.blob_offset);
        const raw = self.bytes[blob_start .. blob_start + @as(usize, @intCast(e.data_bytes))];
        const dt = try e.dataType();
        switch (dt) {
            .f16 => {
                const w_bits: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
                try quant.denseGemvCpuRef(.f16, x, w_bits, y, e.shape[0], e.shape[1]);
            },
            .bf16 => {
                const w_bits: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
                try quant.denseGemvCpuRef(.bf16, x, w_bits, y, e.shape[0], e.shape[1]);
            },
            .q3_affine, .q4_affine => {
                const sb_start = blob_start + @as(usize, @intCast(e.params_offset));
                const sbs: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    self.bytes[sb_start .. sb_start + @as(usize, @intCast(e.params_bytes))],
                ));
                switch (dt) {
                    .q3_affine => try quant.QuantOps(.{ .bits = .q3, .group_size = 64 }).matmulRef(x, raw, sbs, y, 1, e.shape[0], e.shape[1]),
                    .q4_affine => try quant.QuantOps(.{ .bits = .q4, .group_size = 64 }).matmulRef(x, raw, sbs, y, 1, e.shape[0], e.shape[1]),
                    else => unreachable,
                }
            },
            .f32 => {
                const wf: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, raw));
                const n: usize = e.shape[0];
                const k: usize = e.shape[1];
                for (0..n) |row| {
                    var acc: f32 = 0.0;
                    for (0..k) |j| acc += @as(f32, @floatCast(x[j])) * wf[row * k + j];
                    y[row] = acc;
                }
            },
        }
    }

    fn embedLookup(self: *const CpuTransformerRef, token_id: u32, out_f16: []f16) !void {
        const e = try findLinear(&self.parsed, "embed_tokens");
        const h: usize = self.cfg.hidden_size;
        const blob_start: usize = @intCast(e.blob_offset);
        const dt = try e.dataType();
        switch (dt) {
            .f16 => {
                const all: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, self.bytes[blob_start .. blob_start + @as(usize, @intCast(e.data_bytes))]));
                @memcpy(out_f16, all[@as(usize, token_id) * h ..][0..h]);
            },
            .bf16 => {
                const all: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, self.bytes[blob_start .. blob_start + @as(usize, @intCast(e.data_bytes))]));
                const row = all[@as(usize, token_id) * h ..][0..h];
                for (0..h) |j| {
                    const v = std.math.clamp(quant.bf16ToF32(row[j]), -65504.0, 65504.0);
                    out_f16[j] = @floatCast(v);
                }
            },
            .q3_affine => {
                const Q3 = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                const rb = try Q3.rowWeightBytes(h);
                const gs = try Q3.groupsPerRow(h);
                const qrow = self.bytes[blob_start + @as(usize, token_id) * rb ..][0..rb];
                const sb_start = blob_start + @as(usize, @intCast(e.params_offset));
                const sbs: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    self.bytes[sb_start .. sb_start + @as(usize, @intCast(e.params_bytes))],
                ));
                const sb_row = sbs[@as(usize, token_id) * gs ..][0..gs];
                var tmp: [64]f32 = undefined;
                for (0..gs) |g| {
                    Q3.dequantizeGroup(qrow[g * Q3.groupWeightBytes() ..][0..Q3.groupWeightBytes()], sb_row[g], &tmp);
                    for (0..64) |j| out_f16[g * 64 + j] = @floatCast(tmp[j]);
                }
            },
            .q4_affine => {
                const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                const rb = try Q4.rowWeightBytes(h);
                const gs = try Q4.groupsPerRow(h);
                const qrow = self.bytes[blob_start + @as(usize, token_id) * rb ..][0..rb];
                const sb_start = blob_start + @as(usize, @intCast(e.params_offset));
                const sbs: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    self.bytes[sb_start .. sb_start + @as(usize, @intCast(e.params_bytes))],
                ));
                const sb_row = sbs[@as(usize, token_id) * gs ..][0..gs];
                var tmp: [64]f32 = undefined;
                for (0..gs) |g| {
                    Q4.dequantizeGroup(qrow[g * Q4.groupWeightBytes() ..][0..Q4.groupWeightBytes()], sb_row[g], &tmp);
                    for (0..64) |j| out_f16[g * 64 + j] = @floatCast(tmp[j]);
                }
            },
            .f32 => return error.UnexpectedTensorDType,
        }
    }

    fn step(self: *CpuTransformerRef, allocator: std.mem.Allocator, token_id: u32, pos: u32) !u32 {
        const h: usize = self.cfg.hidden_size;
        const q_dim: usize = self.cfg.qDim();
        const kv_dim: usize = self.cfg.kvDim();
        const hd: usize = self.cfg.head_dim;

        const x = try allocator.alloc(f16, h);
        defer allocator.free(x);
        const x_norm_f32 = try allocator.alloc(f32, h);
        defer allocator.free(x_norm_f32);
        const x_norm = try allocator.alloc(f16, h);
        defer allocator.free(x_norm);
        const q_f32 = try allocator.alloc(f32, q_dim);
        defer allocator.free(q_f32);
        const q_f16 = try allocator.alloc(f16, q_dim);
        defer allocator.free(q_f16);
        const k_f32 = try allocator.alloc(f32, kv_dim);
        defer allocator.free(k_f32);
        const k_f16 = try allocator.alloc(f16, kv_dim);
        defer allocator.free(k_f16);
        const v_f32 = try allocator.alloc(f32, kv_dim);
        defer allocator.free(v_f32);
        const v_f16 = try allocator.alloc(f16, kv_dim);
        defer allocator.free(v_f16);
        const head_tmp = try allocator.alloc(f32, hd);
        defer allocator.free(head_tmp);
        const attn_out_f32 = try allocator.alloc(f32, q_dim);
        defer allocator.free(attn_out_f32);
        const attn_out_f16 = try allocator.alloc(f16, q_dim);
        defer allocator.free(attn_out_f16);
        const proj_f32 = try allocator.alloc(f32, h);
        defer allocator.free(proj_f32);

        try self.embedLookup(token_id, x);

        var name_buf: [64]u8 = undefined;
        for (0..self.cfg.num_hidden_layers) |l| {
            const in_ln = try self.normBits(try std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}));
            attention.rmsNormCpuRefDType(x, in_ln.dtype, in_ln.bits, x_norm_f32, 1, h, self.cfg.rms_norm_eps);
            for (0..h) |i| x_norm[i] = @floatCast(x_norm_f32[i]);

            try self.linearMatvec(try std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}), x_norm, q_f32);
            try self.linearMatvec(try std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}), x_norm, k_f32);
            try self.linearMatvec(try std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}), x_norm, v_f32);
            for (0..q_dim) |i| q_f16[i] = @floatCast(q_f32[i]);
            for (0..kv_dim) |i| {
                k_f16[i] = @floatCast(k_f32[i]);
                v_f16[i] = @floatCast(v_f32[i]);
            }

            const qn = try self.normBits(try std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}));
            const kn = try self.normBits(try std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}));
            for (0..self.cfg.num_attention_heads) |qh| {
                attention.qkNormRopeHeadCpuRefDType(q_f16[qh * hd ..][0..hd], qn.dtype, qn.bits, head_tmp, pos, self.cfg.rms_norm_eps, self.cfg.rope_theta);
                for (0..hd) |d| q_f16[qh * hd + d] = @floatCast(head_tmp[d]);
            }
            const layer_kv_base = l * @as(usize, self.max_seq_len) * kv_dim;
            const pos_kv_base = layer_kv_base + @as(usize, pos) * kv_dim;
            for (0..self.cfg.num_key_value_heads) |kvh| {
                attention.qkNormRopeHeadCpuRefDType(k_f16[kvh * hd ..][0..hd], kn.dtype, kn.bits, head_tmp, pos, self.cfg.rms_norm_eps, self.cfg.rope_theta);
                for (0..hd) |d| {
                    self.k_cache[pos_kv_base + kvh * hd + d] = @floatCast(head_tmp[d]);
                    self.v_cache[pos_kv_base + kvh * hd + d] = v_f16[kvh * hd + d];
                }
            }

            try attention.gqaDecodeCpuRef(
                allocator,
                q_f16,
                self.k_cache[layer_kv_base .. layer_kv_base + (@as(usize, pos) + 1) * kv_dim],
                self.v_cache[layer_kv_base .. layer_kv_base + (@as(usize, pos) + 1) * kv_dim],
                attn_out_f32,
                self.cfg.num_attention_heads,
                self.cfg.num_key_value_heads,
                self.cfg.head_dim,
                pos + 1,
                1.0 / @sqrt(@as(f32, @floatFromInt(self.cfg.head_dim))),
            );
            for (0..q_dim) |i| attn_out_f16[i] = @floatCast(attn_out_f32[i]);

            try self.linearMatvec(try std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}), attn_out_f16, proj_f32);
            for (0..h) |i| {
                const delta: f32 = @floatCast(@as(f16, @floatCast(proj_f32[i])));
                x[i] = @floatCast(@as(f32, @floatCast(x[i])) + delta);
            }

            const post_ln = try self.normBits(try std.fmt.bufPrint(&name_buf, "layers.{d}.post_attention_layernorm.weight", .{l}));
            attention.rmsNormCpuRefDType(x, post_ln.dtype, post_ln.bits, x_norm_f32, 1, h, self.cfg.rms_norm_eps);
            for (0..h) |i| x_norm[i] = @floatCast(x_norm_f32[i]);

            if (self.cfg.arch == .qwen3_moe) {
                const r_logits = try allocator.alloc(f32, self.cfg.num_experts);
                defer allocator.free(r_logits);
                try self.linearMatvec(try std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate", .{l}), x_norm, r_logits);
                var route = try moe.routeTopKRef(allocator, r_logits, 1, self.cfg.num_experts, self.cfg.num_experts_per_tok, self.cfg.norm_topk_prob);
                defer route.deinit(allocator);

                const mi: usize = self.cfg.moe_intermediate_size;
                const g_f32 = try allocator.alloc(f32, mi);
                defer allocator.free(g_f32);
                const u_f32 = try allocator.alloc(f32, mi);
                defer allocator.free(u_f32);
                const act_f16 = try allocator.alloc(f16, mi);
                defer allocator.free(act_f16);
                const sum_f32 = try allocator.alloc(f32, h);
                defer allocator.free(sum_f32);
                @memset(sum_f32, 0.0);

                for (0..self.cfg.num_experts_per_tok) |k_idx| {
                    const eid = route.topk_experts[k_idx];
                    const rw = route.topk_weights[k_idx];
                    try self.linearMatvec(try std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, eid }), x_norm, g_f32);
                    try self.linearMatvec(try std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, eid }), x_norm, u_f32);
                    for (0..mi) |i| act_f16[i] = @floatCast(moe.siluF32(g_f32[i]) * u_f32[i]);
                    try self.linearMatvec(try std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, eid }), act_f16, proj_f32);
                    for (0..h) |i| sum_f32[i] += rw * proj_f32[i];
                }
                for (0..h) |i| x[i] = @floatCast(@as(f32, @floatCast(x[i])) + sum_f32[i]);
            } else {
                const di: usize = self.cfg.intermediate_size;
                const g_f32 = try allocator.alloc(f32, di);
                defer allocator.free(g_f32);
                const u_f32 = try allocator.alloc(f32, di);
                defer allocator.free(u_f32);
                const act_f16 = try allocator.alloc(f16, di);
                defer allocator.free(act_f16);
                try self.linearMatvec(try std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}), x_norm, g_f32);
                try self.linearMatvec(try std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}), x_norm, u_f32);
                for (0..di) |i| {
                    const gf: f32 = @floatCast(@as(f16, @floatCast(g_f32[i])));
                    const uf: f32 = @floatCast(@as(f16, @floatCast(u_f32[i])));
                    act_f16[i] = @floatCast(moe.siluF32(gf) * uf);
                }
                try self.linearMatvec(try std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}), act_f16, proj_f32);
                for (0..h) |i| {
                    const delta: f32 = @floatCast(@as(f16, @floatCast(proj_f32[i])));
                    x[i] = @floatCast(@as(f32, @floatCast(x[i])) + delta);
                }
            }
        }

        const final_ln = try self.normBits("norm.weight");
        attention.rmsNormCpuRefDType(x, final_ln.dtype, final_ln.bits, x_norm_f32, 1, h, self.cfg.rms_norm_eps);
        for (0..h) |i| x_norm[i] = @floatCast(x_norm_f32[i]);

        const head_name = if (self.cfg.tie_word_embeddings) "embed_tokens" else "lm_head";
        try self.linearMatvec(head_name, x_norm, self.logits);

        const logits_f16 = try allocator.alloc(f16, self.cfg.vocab_size);
        defer allocator.free(logits_f16);
        for (0..self.cfg.vocab_size) |i| logits_f16[i] = @floatCast(self.logits[i]);
        return moe.argmaxRef(logits_f16);
    }
};

test "Qwen3-MoE 3-bit greedy multi-token decode from .amf matches CPU fp32 reference step-for-step" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .vocab_size = 256,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 8,
        .num_experts_per_tok = 2,
        .norm_topk_prob = true,
    };

    const image = try buildSyntheticAmfImage(gpa, cfg, false, 42);
    defer gpa.free(image);

    var cpu_ref = try CpuTransformerRef.init(gpa, image, cfg, 64);
    defer cpu_ref.deinit(gpa);

    // Force multi-shard allocation (`max_shard_bytes = 768 KiB`) to exercise >13.3 GiB sharding path!
    var model = try Qwen3Model.loadFromBytes(gpa, image, cfg, .{
        .max_seq_len = 64,
        .max_shard_bytes = 768 * 1024,
    });
    defer model.deinit(gpa);

    try std.testing.expect(model.shardCount() >= 2);
    try std.testing.expect(model.bytesPerDecodeToken(1) > 0);
    try std.testing.expect(model.bytesPerLayerToken(1) > 0);
    try std.testing.expect(model.bytesGlobalHead() > 0);

    var tok: u32 = 17;
    var generated: [6]u32 = undefined;
    for (0..6) |step_idx| {
        const pos: u32 = @intCast(step_idx);
        const expected_next = try cpu_ref.step(gpa, tok, pos);
        const gpu_next = try model.stepDecode(tok, pos);

        const gpu_logits = model.readLogitsF16();
        try std.testing.expectEqual(@as(usize, cfg.vocab_size), gpu_logits.len);
        for (0..cfg.vocab_size) |v| {
            const got: f32 = @floatCast(gpu_logits[v]);
            const exp: f32 = cpu_ref.logits[v];
            try std.testing.expectApproxEqAbs(exp, got, @max(0.05, @abs(exp) * 0.04));
        }
        try std.testing.expectEqual(expected_next, gpu_next);
        generated[step_idx] = gpu_next;
        tok = gpu_next;
    }

    // Verify resetKvCache + prefillToken produces identical deterministic replay.
    model.resetKvCache();
    try model.prefillToken(17, 0);
    const replay_tok1 = try model.stepDecode(generated[0], 1);
    try std.testing.expectEqual(generated[1], replay_tok1);

    // Steady-state zero-swap check over repeated decode steps.
    const vm0 = try vmstat.Snapshot.capture();
    _ = try model.stepDecode(generated[1], 2);
    const vm1 = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm0, vm1);
}

test "Qwen3-Dense 4-bit greedy multi-token decode (tied and untied f16 embed) matches CPU fp32 reference" {
    const gpa = std.testing.allocator;
    for ([_]struct { tied: bool, f16_emb: bool }{
        .{ .tied = false, .f16_emb = true },
        .{ .tied = true, .f16_emb = false },
    }) |mode| {
        const cfg: ModelConfig = .{
            .arch = .qwen3_dense,
            .quant_bits = .q4,
            .vocab_size = 256,
            .hidden_size = 128,
            .num_hidden_layers = 2,
            .num_attention_heads = 4,
            .num_key_value_heads = 2,
            .head_dim = 64,
            .intermediate_size = 256,
            .tie_word_embeddings = mode.tied,
        };

        const image = try buildSyntheticAmfImage(gpa, cfg, mode.f16_emb, 1337);
        defer gpa.free(image);

        var cpu_ref = try CpuTransformerRef.init(gpa, image, cfg, 64);
        defer cpu_ref.deinit(gpa);

        var model = try Qwen3Model.loadFromBytes(gpa, image, cfg, .{ .max_seq_len = 64 });
        defer model.deinit(gpa);

        var tok: u32 = 5;
        for (0..6) |step_idx| {
            const pos: u32 = @intCast(step_idx);
            const expected_next = try cpu_ref.step(gpa, tok, pos);
            const gpu_next = try model.stepDecode(tok, pos);

            const gpu_logits = model.readLogitsF16();
            for (0..cfg.vocab_size) |v| {
                const got: f32 = @floatCast(gpu_logits[v]);
                const exp: f32 = cpu_ref.logits[v];
                try std.testing.expectApproxEqAbs(exp, got, @max(0.05, @abs(exp) * 0.04));
            }
            try std.testing.expectEqual(expected_next, gpu_next);
            tok = gpu_next;
        }
    }
}

test "Qwen3Model enforces wired budget limit at startup without falling back to pageable memory" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = .q4,
        .vocab_size = 256,
        .hidden_size = 128,
        .num_hidden_layers = 1,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .intermediate_size = 256,
        .tie_word_embeddings = true,
    };
    const image = try buildSyntheticAmfImage(gpa, cfg, false, 7);
    defer gpa.free(image);

    const tiny_budget = budget_mod.Budget.compute(.{
        .memory_bytes = 64 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = 2 * page.APPLE_PAGE_SIZE,
        .global_user_wire_limit = 2 * page.APPLE_PAGE_SIZE,
        .gpu_working_set_bytes = 2 * page.APPLE_PAGE_SIZE,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    try std.testing.expectError(error.BudgetExceeded, Qwen3Model.loadFromBytes(gpa, image, cfg, .{
        .max_seq_len = 64,
        .budget_override = tiny_budget,
    }));
}

test "Qwen3Model loads from F_NOCACHE disk .amf and matches CPU reference across KV chunk boundaries" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .vocab_size = 256,
        .hidden_size = 128,
        .num_hidden_layers = 1,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .norm_topk_prob = true,
    };
    const image = try buildSyntheticAmfImage(gpa, cfg, false, 2026);
    defer gpa.free(image);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "qwen3_moe.amf", .data = image });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const raw_path = try std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], ".zig-cache/tmp/{s}/qwen3_moe.amf", .{tmp.sub_path});
    path_buf[raw_path.len] = 0;
    const rel_path: [:0]const u8 = path_buf[0..raw_path.len :0];

    var cpu_ref = try CpuTransformerRef.init(gpa, image, cfg, 24);
    defer cpu_ref.deinit(gpa);

    // Small chunk_tokens = 8 forces KV cache transitions across chunks 0 -> 1 -> 2.
    var model = try Qwen3Model.loadFromFile(gpa, rel_path, cfg, .{
        .max_seq_len = 24,
        .chunk_tokens = 8,
    });
    defer model.deinit(gpa);

    try std.testing.expectEqual(@as(u32, 3), model.num_chunks);
    var tok: u32 = 31;
    for (0..12) |step_idx| {
        const pos: u32 = @intCast(step_idx);
        const expected = try cpu_ref.step(gpa, tok, pos);
        const got = try model.stepDecode(tok, pos);
        try std.testing.expectEqual(expected, got);
        tok = got;
    }
}

test "Qwen3-Dense pure BF16 (tie_word_embeddings, qDim != hidden_size, .weight names) matches CPU fp32 reference" {
    const gpa = std.testing.allocator;
    // Mirrors Qwen3-0.6B layout: qDim = 4*64 = 256 != hidden_size = 128,
    // tie_word_embeddings = true, all tensors BF16 with `.weight` suffixes.
    const cfg: ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = null,
        .default_float_dtype = .bf16,
        .vocab_size = 256,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .intermediate_size = 384,
        .tie_word_embeddings = true,
    };
    const image = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .bf16,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
        .use_weight_suffix = true,
    }, 600);
    defer gpa.free(image);

    var cpu_ref = try CpuTransformerRef.init(gpa, image, cfg, 64);
    defer cpu_ref.deinit(gpa);

    var model = try Qwen3Model.loadFromBytes(gpa, image, cfg, .{ .max_seq_len = 64 });
    defer model.deinit(gpa);

    // Zero-duplicate tie_word_embeddings: lm_head reuses embed_tokens's exact wired offset & buffer.
    try std.testing.expectEqual(model.embed.weight.offset, model.lm_head.weight.offset);
    try std.testing.expectEqual(quant.WeightDType.bf16, model.embed.dtype);
    try std.testing.expectEqual(quant.WeightDType.bf16, model.lm_head.dtype);
    try std.testing.expectEqual(quant.FloatDType.bf16, model.final_norm.dtype);

    var tok: u32 = 19;
    for (0..6) |step_idx| {
        const pos: u32 = @intCast(step_idx);
        const expected = try cpu_ref.step(gpa, tok, pos);
        const got = try model.stepDecode(tok, pos);

        const gpu_logits = model.readLogitsF16();
        for (0..cfg.vocab_size) |v| {
            const g: f32 = @floatCast(gpu_logits[v]);
            const e: f32 = cpu_ref.logits[v];
            try std.testing.expectApproxEqAbs(e, g, @max(0.05, @abs(e) * 0.04));
        }
        try std.testing.expectEqual(expected, got);
        tok = got;
    }
}

test "Qwen3-Dense pure FP16 and Qwen3-MoE pure BF16 match CPU fp32 reference step-for-step" {
    const gpa = std.testing.allocator;

    // 1. Pure FP16 untied dense with relaxed (% 4 == 0, non-64) hidden & intermediate dims
    {
        const cfg: ModelConfig = .{
            .arch = .qwen3_dense,
            .quant_bits = null,
            .default_float_dtype = .f16,
            .vocab_size = 192,
            .hidden_size = 132,
            .num_hidden_layers = 2,
            .num_attention_heads = 4,
            .num_key_value_heads = 2,
            .head_dim = 64,
            .intermediate_size = 196,
            .tie_word_embeddings = false,
        };
        const image = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
            .linear_dtype = .f16,
            .embed_dtype = .f16,
            .norm_dtype = .f16,
            .use_weight_suffix = true,
        }, 1600);
        defer gpa.free(image);

        var cpu_ref = try CpuTransformerRef.init(gpa, image, cfg, 32);
        defer cpu_ref.deinit(gpa);

        var model = try Qwen3Model.loadFromBytes(gpa, image, cfg, .{ .max_seq_len = 32 });
        defer model.deinit(gpa);

        var tok: u32 = 11;
        for (0..5) |step_idx| {
            const pos: u32 = @intCast(step_idx);
            const expected = try cpu_ref.step(gpa, tok, pos);
            const got = try model.stepDecode(tok, pos);

            const gpu_logits = model.readLogitsF16();
            for (0..cfg.vocab_size) |v| {
                const g: f32 = @floatCast(gpu_logits[v]);
                const e: f32 = cpu_ref.logits[v];
                try std.testing.expectApproxEqAbs(e, g, @max(0.05, @abs(e) * 0.04));
            }
            try std.testing.expectEqual(expected, got);
            tok = got;
        }
    }

    // 2. Pure BF16 MoE (bf16 experts + bf16 router + bf16 norms)
    {
        const cfg: ModelConfig = .{
            .arch = .qwen3_moe,
            .quant_bits = null,
            .default_float_dtype = .bf16,
            .vocab_size = 192,
            .hidden_size = 128,
            .num_hidden_layers = 2,
            .num_attention_heads = 4,
            .num_key_value_heads = 2,
            .head_dim = 64,
            .moe_intermediate_size = 128,
            .num_experts = 8,
            .num_experts_per_tok = 2,
            .norm_topk_prob = true,
        };
        const image = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
            .linear_dtype = .bf16,
            .embed_dtype = .bf16,
            .norm_dtype = .bf16,
            .router_dtype = .bf16,
            .expert_dtype = .bf16,
            .use_weight_suffix = true,
        }, 2600);
        defer gpa.free(image);

        var cpu_ref = try CpuTransformerRef.init(gpa, image, cfg, 32);
        defer cpu_ref.deinit(gpa);

        var model = try Qwen3Model.loadFromBytes(gpa, image, cfg, .{ .max_seq_len = 32 });
        defer model.deinit(gpa);

        var tok: u32 = 23;
        for (0..5) |step_idx| {
            const pos: u32 = @intCast(step_idx);
            const expected = try cpu_ref.step(gpa, tok, pos);
            const got = try model.stepDecode(tok, pos);

            const gpu_logits = model.readLogitsF16();
            for (0..cfg.vocab_size) |v| {
                const g: f32 = @floatCast(gpu_logits[v]);
                const e: f32 = cpu_ref.logits[v];
                try std.testing.expectApproxEqAbs(e, g, @max(0.05, @abs(e) * 0.04));
            }
            try std.testing.expectEqual(expected, got);
            tok = got;
        }
    }
}

test "Qwen3-MoE mixed-precision (q3 experts + q3 router + q3 embed + bf16 norms) matches CPU fp32 reference" {
    const gpa = std.testing.allocator;
    // Mirrors real mlx-community/Qwen3-30B-A3B-3bit layout:
    // q3 attention/experts/router/embed/lm_head + bf16 norms.
    const cfg: ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .default_float_dtype = .bf16,
        .vocab_size = 256,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 8,
        .num_experts_per_tok = 2,
        .norm_topk_prob = true,
    };
    const image = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .q3_affine,
        .embed_dtype = .q3_affine,
        .norm_dtype = .bf16,
        .router_dtype = .q3_affine,
        .expert_dtype = .q3_affine,
        .use_weight_suffix = false,
    }, 3030);
    defer gpa.free(image);

    var cpu_ref = try CpuTransformerRef.init(gpa, image, cfg, 64);
    defer cpu_ref.deinit(gpa);

    var model = try Qwen3Model.loadFromBytes(gpa, image, cfg, .{ .max_seq_len = 64 });
    defer model.deinit(gpa);

    var tok: u32 = 41;
    for (0..6) |step_idx| {
        const pos: u32 = @intCast(step_idx);
        const expected = try cpu_ref.step(gpa, tok, pos);
        const got = try model.stepDecode(tok, pos);

        const gpu_logits = model.readLogitsF16();
        for (0..cfg.vocab_size) |v| {
            const g: f32 = @floatCast(gpu_logits[v]);
            const e: f32 = cpu_ref.logits[v];
            try std.testing.expectApproxEqAbs(e, g, @max(0.05, @abs(e) * 0.04));
        }
        try std.testing.expectEqual(expected, got);
        tok = got;
    }
}
