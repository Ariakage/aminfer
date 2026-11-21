//! L2 speculative draft heads (EAGLE-3, EAGLE-2, MTP) and benchmark-gated
//! ANE vs GPU placement (`D2`, `D3`, `D5`, ADR-0008, ADR-0013).
//!
//! Shares the target `Qwen3Model`'s Metal 4 device, pipelines, and command queue
//! so GPU-placed draft steps run in the same residency set and command path as
//! verification with zero dynamic heap allocation on the hot path. When
//! `Placement.ane` is active (or selected by `Placement.auto` via startup D5
//! gating), `layers[0].o_proj` executes on Apple's Neural Engine over zero-copy
//! IOSurface `KvTile` tensors (`AneTensor.fromKvTile` + `AnePlan` `outputBackings`).

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
const qwen3 = @import("../model/qwen3.zig");
const vmstat = @import("../platform/vmstat.zig");

/// Maximum top-K candidates extracted per draft node via `encodeTopKLogitsRows`.
pub const MAX_DRAFT_TOP_K: u32 = moe.MAX_TOP_K;

/// Maximum rows batched through the optional CoreML ANE `innerProduct` projection.
pub const ANE_MAX_BATCH_ROWS: u32 = 8;

/// L1/L2 speculative draft source selection (`ADR-0013`).
pub const DraftMode = enum {
    l1_only,
    eagle3,
    eagle2,
    mtp,
    small_draft,

    pub fn label(self: DraftMode) []const u8 {
        return switch (self) {
            .l1_only => "l1",
            .eagle3 => "eagle3",
            .eagle2 => "eagle2",
            .mtp => "mtp",
            .small_draft => "small",
        };
    }

    pub fn parse(str: []const u8) error{InvalidDraftMode}!DraftMode {
        if (std.mem.eql(u8, str, "l1") or std.mem.eql(u8, str, "l1_only")) return .l1_only;
        if (std.mem.eql(u8, str, "eagle3")) return .eagle3;
        if (std.mem.eql(u8, str, "eagle2")) return .eagle2;
        if (std.mem.eql(u8, str, "mtp")) return .mtp;
        if (std.mem.eql(u8, str, "small") or std.mem.eql(u8, str, "small_draft")) return .small_draft;
        return error.InvalidDraftMode;
    }
};

/// L2 draft head hardware placement policy (`D5`, ADR-0008).
pub const Placement = enum {
    auto,
    gpu,
    ane,

    pub fn label(self: Placement) []const u8 {
        return switch (self) {
            .auto => "auto",
            .gpu => "gpu",
            .ane => "ane",
        };
    }

    pub fn parse(str: []const u8) error{InvalidPlacement}!Placement {
        if (std.mem.eql(u8, str, "auto")) return .auto;
        if (std.mem.eql(u8, str, "gpu")) return .gpu;
        if (std.mem.eql(u8, str, "ane")) return .ane;
        return error.InvalidPlacement;
    }
};

/// Parsed L2 draft head configuration supporting real HuggingFace / vLLM
/// `Eagle3DraftModel` (`RedHatAI/Qwen3-30B-A3B-speculator.eagle3`) as well as
/// synthetic `eagle3`, `eagle2`, and `mtp` bundles from `aminfer-tools synth-draft`.
pub const DraftConfig = struct {
    format: DraftMode,
    vocab_size: u32,
    draft_vocab_size: u32,
    hidden_size: u32,
    target_hidden_size: u32,
    num_hidden_layers: u32 = 1,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 10_000.0,
    max_position_embeddings: u32 = 40960,
    norm_before_residual: bool = true,
    /// Optional explicit 0-indexed target transformer layer IDs for EAGLE-3 aux taps.
    aux_layer_ids: ?[3]u32 = null,
    share_embed_tokens: bool = false,
    share_lm_head: bool = false,

    pub const ConfigError = error{
        InvalidJson,
        UnsupportedFormat,
        InvalidConfig,
        OutOfMemory,
    };

    pub fn qDim(self: DraftConfig) u32 {
        return self.num_attention_heads * self.head_dim;
    }

    pub fn kvDim(self: DraftConfig) u32 {
        return self.num_key_value_heads * self.head_dim;
    }

    pub fn validate(self: DraftConfig) ConfigError!void {
        if (self.format == .l1_only or self.format == .small_draft) return error.UnsupportedFormat;
        if (self.vocab_size == 0 or self.draft_vocab_size == 0 or self.draft_vocab_size > self.vocab_size) {
            return error.InvalidConfig;
        }
        if (self.format != .eagle3 and self.draft_vocab_size != self.vocab_size) {
            return error.InvalidConfig;
        }
        if (self.hidden_size == 0 or self.target_hidden_size == 0 or self.num_hidden_layers == 0 or
            self.num_attention_heads == 0 or self.num_key_value_heads == 0 or
            self.intermediate_size == 0 or self.max_position_embeddings == 0)
        {
            return error.InvalidConfig;
        }
        if (self.head_dim != 64 and self.head_dim != 128) return error.InvalidConfig;
        if ((self.hidden_size % 4) != 0 or (self.target_hidden_size % 4) != 0 or (self.intermediate_size % 4) != 0) {
            return error.InvalidConfig;
        }
        if ((self.num_attention_heads % self.num_key_value_heads) != 0) return error.InvalidConfig;
        if (self.rms_norm_eps <= 0.0 or self.rope_theta <= 0.0) return error.InvalidConfig;
    }

    /// Parses an EAGLE-3, EAGLE-2, or MTP `config.json` document at startup.
    pub fn fromJson(allocator: std.mem.Allocator, json_bytes: []const u8) ConfigError!DraftConfig {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidJson,
        };
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return error.InvalidJson,
        };

        const format = detectFormat(obj) orelse return error.UnsupportedFormat;
        const layer_obj: std.json.ObjectMap = blk: {
            if (obj.get("transformer_layer_config")) |v| {
                if (v == .object) break :blk v.object;
            }
            break :blk obj;
        };

        const vocab_size = jsonOptU32(layer_obj, "vocab_size") orelse jsonOptU32(obj, "vocab_size") orelse return error.InvalidJson;
        const draft_vocab_size = jsonOptU32(obj, "draft_vocab_size") orelse vocab_size;
        const hidden_size = jsonOptU32(layer_obj, "hidden_size") orelse jsonOptU32(obj, "hidden_size") orelse return error.InvalidJson;
        const target_hidden_size = jsonOptU32(obj, "target_hidden_size") orelse hidden_size;
        const num_layers = jsonOptU32(layer_obj, "num_hidden_layers") orelse jsonOptU32(obj, "num_hidden_layers") orelse 1;
        const num_q_heads = jsonOptU32(layer_obj, "num_attention_heads") orelse jsonOptU32(obj, "num_attention_heads") orelse return error.InvalidJson;
        const num_kv_heads = jsonOptU32(layer_obj, "num_key_value_heads") orelse jsonOptU32(obj, "num_key_value_heads") orelse num_q_heads;
        const head_dim = jsonOptU32(layer_obj, "head_dim") orelse jsonOptU32(obj, "head_dim") orelse blk: {
            if (num_q_heads == 0) return error.InvalidConfig;
            break :blk hidden_size / num_q_heads;
        };
        const inter = jsonOptU32(layer_obj, "intermediate_size") orelse jsonOptU32(obj, "intermediate_size") orelse return error.InvalidJson;
        const rms_eps = jsonOptF32(layer_obj, "rms_norm_eps") orelse jsonOptF32(obj, "rms_norm_eps") orelse 1e-6;
        const default_theta: f32 = if (format == .eagle3) 10_000.0 else 1_000_000.0;
        const rope_theta = jsonOptF32(layer_obj, "rope_theta") orelse jsonOptF32(obj, "rope_theta") orelse default_theta;
        const max_pos = jsonOptU32(layer_obj, "max_position_embeddings") orelse jsonOptU32(obj, "max_position_embeddings") orelse 40960;
        const norm_before_res = jsonOptBool(obj, "norm_before_residual") orelse true;
        const share_embed = jsonOptBool(obj, "share_embed_tokens") orelse false;
        const share_lm = jsonOptBool(obj, "share_lm_head") orelse false;

        var aux_ids: ?[3]u32 = null;
        if (obj.get("eagle_aux_hidden_state_layer_ids")) |v| {
            if (v == .array and v.array.items.len == 3) {
                var ids: [3]u32 = undefined;
                for (v.array.items, 0..) |item, i| {
                    const raw_id: u32 = switch (item) {
                        .integer => |iv| if (iv >= 0 and iv <= std.math.maxInt(u32)) @intCast(iv) else return error.InvalidConfig,
                        else => return error.InvalidJson,
                    };
                    // HF / vLLM `hidden_states` indexing uses 1-based layer outputs (`layer_idx + 1`).
                    ids[i] = if (raw_id > 0) raw_id - 1 else 0;
                }
                aux_ids = ids;
            }
        }

        const cfg: DraftConfig = .{
            .format = format,
            .vocab_size = vocab_size,
            .draft_vocab_size = draft_vocab_size,
            .hidden_size = hidden_size,
            .target_hidden_size = target_hidden_size,
            .num_hidden_layers = num_layers,
            .num_attention_heads = num_q_heads,
            .num_key_value_heads = num_kv_heads,
            .head_dim = head_dim,
            .intermediate_size = inter,
            .rms_norm_eps = rms_eps,
            .rope_theta = rope_theta,
            .max_position_embeddings = max_pos,
            .norm_before_residual = norm_before_res,
            .aux_layer_ids = aux_ids,
            .share_embed_tokens = share_embed,
            .share_lm_head = share_lm,
        };
        try cfg.validate();
        return cfg;
    }
};

fn detectFormat(obj: std.json.ObjectMap) ?DraftMode {
    if (jsonOptString(obj, "speculators_model_type")) |s| {
        if (std.mem.eql(u8, s, "eagle3")) return .eagle3;
        if (std.mem.eql(u8, s, "eagle2") or std.mem.eql(u8, s, "eagle")) return .eagle2;
        if (std.mem.eql(u8, s, "mtp")) return .mtp;
    }
    if (jsonOptString(obj, "draft_format")) |s| {
        if (std.mem.eql(u8, s, "eagle3")) return .eagle3;
        if (std.mem.eql(u8, s, "eagle2")) return .eagle2;
        if (std.mem.eql(u8, s, "mtp")) return .mtp;
    }
    if (jsonOptString(obj, "model_type")) |s| {
        if (std.mem.eql(u8, s, "eagle3")) return .eagle3;
        if (std.mem.eql(u8, s, "eagle")) return .eagle2;
        if (std.mem.eql(u8, s, "mtp")) return .mtp;
    }
    if (obj.get("architectures")) |v| {
        if (v == .array and v.array.items.len > 0) {
            if (v.array.items[0] == .string) {
                const arch = v.array.items[0].string;
                if (std.mem.indexOf(u8, arch, "Eagle3") != null) return .eagle3;
                if (std.mem.indexOf(u8, arch, "Eagle") != null) return .eagle2;
                if (std.mem.indexOf(u8, arch, "Mtp") != null) return .mtp;
            }
        }
    }
    return null;
}

fn jsonOptString(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
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

pub const DraftLoadOptions = struct {
    placement: Placement = .auto,
    max_seq_len: u32 = 2048,
    chunk_tokens: u32 = attention.DEFAULT_CHUNK_TOKENS,
    max_tree_slots: u32 = attention.MAX_TREE_NODES,
    verify_crcs: bool = true,
    timeout_ms: u64 = 30_000,
    /// When true (default), `SmallDraftModel` uses its own independent Metal 4
    /// `Compute` command queue + residency set (distinct from `target.compute`)
    /// so `D8` PEARL can overlap round-`r+1` pre-drafting concurrently on
    /// hardware with `target.verifyTree(T_r)` (ADR-0008).
    own_compute_queue: bool = true,
};

pub const DraftLoadError = error{
    MissingTensor,
    UnexpectedTensorShape,
    UnexpectedTensorDType,
    InvalidD2tEntry,
    TargetIncompatible,
    BudgetExceeded,
    OutOfMemory,
} || DraftConfig.ConfigError || amf.AmfFile.OpenError || quant.ShapeError ||
    metal.WrapError || metal.ProtoEmitError || budget_mod.PlanError ||
    arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError ||
    vmstat.Error || moe.EncodeError;

pub const DraftStepError = error{
    TokenOutOfRange,
    PositionOutOfRange,
    SlotOutOfRange,
    BatchTooLarge,
} || metal.Error || dequant.EncodeError || moe.EncodeError;

const DraftLayerWeights = struct {
    input_layernorm: qwen3.NormBinding,
    hidden_norm: ?qwen3.NormBinding,
    q_proj: qwen3.LinearBinding,
    k_proj: qwen3.LinearBinding,
    v_proj: qwen3.LinearBinding,
    o_proj: qwen3.LinearBinding,
    q_norm: ?metal.BufferBinding,
    k_norm: ?metal.BufferBinding,
    qk_norm_dtype: quant.FloatDType,
    post_attention_layernorm: qwen3.NormBinding,
    gate_proj: qwen3.LinearBinding,
    up_proj: qwen3.LinearBinding,
    down_proj: qwen3.LinearBinding,
    kv_tiles_binding: metal.BufferBinding,
};

/// Zero-copy CoreML ANE linear projection executing `layers[0].o_proj` over
/// IOSurface-backed `KvTile`s (`alias_buffer = true` + `AneTensor` + `AnePlan`).
const AneProjection = struct {
    batch_rows: u32,
    c_in: u32,
    c_out: u32,
    model: metal.AneModel,
    in_tile: metal.KvTile,
    out_tile: metal.KvTile,
    in_tensor: metal.AneTensor,
    out_tensor: metal.AneTensor,
    plan: metal.AnePlan,
    uses_ane: bool,
    zero_copy_ok: bool,

    fn init(
        allocator: std.mem.Allocator,
        device: metal.Device,
        compute: *metal.Compute,
        c_in: u32,
        c_out: u32,
        batch_rows: u32,
        weights_f16: []const f16,
    ) DraftLoadError!AneProjection {
        const layers_spec = [_]metal.CoreMlLinearLayer{.{
            .c_in = c_in,
            .c_out = c_out,
            .weights_f16 = weights_f16,
        }};
        const spec_bytes = try metal.emitCoreMlLinearModel(allocator, .{
            .op = .inner_product,
            .batch_or_spatial = batch_rows,
            .specification_version = 7,
            .io_float16 = true,
            .input_name = "x",
            .output_name = "y",
            .layers = &layers_spec,
        });
        defer allocator.free(spec_bytes);

        const model = try metal.AneModel.loadSpecData(spec_bytes, .cpu_and_neural_engine);
        errdefer model.close();
        const minfo = try model.info();

        const in_tile = try metal.KvTile.init(device, c_in, batch_rows, .{ .alias_buffer = true });
        errdefer in_tile.close();
        const out_tile = try metal.KvTile.init(device, c_out, batch_rows, .{ .alias_buffer = true });
        errdefer out_tile.close();

        const in_tensor = try metal.AneTensor.fromKvTile(in_tile, &.{ batch_rows, c_in });
        errdefer in_tensor.close();
        const out_tensor = try metal.AneTensor.fromKvTile(out_tile, &.{ batch_rows, c_out });
        errdefer out_tensor.close();

        const plan = try metal.AnePlan.init(
            &.{.{ .name = "x", .tensor = in_tensor }},
            &.{.{ .name = "y", .tensor = out_tensor }},
        );
        errdefer plan.close();

        try compute.makeTilesResident(&.{ in_tile, out_tile });
        errdefer compute.removeTilesResident(&.{ in_tile, out_tile }) catch |err| {
            std.debug.panic("removeTilesResident failed during AneProjection errdefer: {}", .{err});
        };

        const pinfo = try model.predict(plan);
        return .{
            .batch_rows = batch_rows,
            .c_in = c_in,
            .c_out = c_out,
            .model = model,
            .in_tile = in_tile,
            .out_tile = out_tile,
            .in_tensor = in_tensor,
            .out_tensor = out_tensor,
            .plan = plan,
            .uses_ane = minfo.usesNeuralEngine(),
            .zero_copy_ok = pinfo.allOutputsZeroCopy(),
        };
    }

    fn close(self: *AneProjection, compute: *metal.Compute) void {
        compute.removeTilesResident(&.{ self.in_tile, self.out_tile }) catch |err| {
            std.debug.panic("removeTilesResident failed in AneProjection.close: {}", .{err});
        };
        self.plan.close();
        self.out_tensor.close();
        self.in_tensor.close();
        self.out_tile.close();
        self.in_tile.close();
        self.model.close();
    }

    fn inBinding(self: *const AneProjection) metal.BufferBinding {
        return .{ .buffer = self.in_tile.aliasBuffer().?, .offset = 0 };
    }

    fn outBinding(self: *const AneProjection) metal.BufferBinding {
        return .{ .buffer = self.out_tile.aliasBuffer().?, .offset = 0 };
    }
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
    ) DraftLoadError![]align(page.APPLE_PAGE_SIZE) u8 {
        switch (self) {
            .bytes => |b| {
                if (verify_crc) return b.parsed.loadBlobFromBytes(b.raw, entry, arena);
                const start: usize = @intCast(entry.blob_offset);
                const len: usize = @intCast(entry.blob_bytes);
                if (start + len > b.raw.len) return error.BlobOutOfBounds;
                const dst = try arena.allocPages(len);
                @memcpy(dst, b.raw[start .. start + len]);
                return dst;
            },
            .file => |f| {
                if (verify_crc) return f.loadBlob(entry, arena);
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

pub const DraftHead = struct {
    config: DraftConfig,
    requested_placement: Placement,
    effective_placement: Placement,
    placement_gate_ane_us: f64,
    placement_gate_gpu_us: f64,
    max_seq_len: u32,
    chunk_tokens: u32,
    num_chunks: u32,
    max_tree_slots: u32,
    seq_len: u32,
    timeout_ms: u64,
    total_wired_bytes: u64,

    weights_arena: arena_mod.WiredArena,
    weights_buf: metal.Buffer,
    scratch_arena: arena_mod.WiredArena,
    scratch_buf: metal.Buffer,

    kv_tiles: []metal.KvTile,
    layers: []DraftLayerWeights,

    /// Direct `u32` target token ID map `[draft_vocab_size]` for EAGLE-3 (empty for EAGLE-2/MTP).
    d2t_host: []const u32,

    embed: qwen3.LinearBinding,
    lm_head: qwen3.LinearBinding,
    fc: ?qwen3.LinearBinding,
    enorm: ?qwen3.NormBinding,
    hnorm: ?qwen3.NormBinding,
    eh_proj: ?qwen3.LinearBinding,
    final_norm: qwen3.NormBinding,

    // Optional zero-copy CoreML ANE projection for `layers[0].o_proj` (D5).
    ane_proj: ?AneProjection,

    // Preallocated scratch buffers & bindings (zero hot-path heap alloc)
    token_ids_host: []u32,
    parent_slots_host: []u32,
    positions_host: []u32,
    ancestor_masks_host: []u64,
    topk_host: []moe.TopKCandidate,
    hidden_pool_host: []f16,

    token_ids_bind: metal.BufferBinding,
    parent_slots_bind: metal.BufferBinding,
    positions_bind: metal.BufferBinding,
    ancestor_masks_bind: metal.BufferBinding,
    hidden_pool_bind: metal.BufferBinding, // [(1 + max_tree_slots), H]: [0]=root, [1+s]=tree slot s
    h_in_bind: metal.BufferBinding, // [max_tree_slots, H]
    emb_bind: metal.BufferBinding, // [max_tree_slots, H]
    e_norm_bind: metal.BufferBinding, // [max_tree_slots, H]
    h_norm_bind: metal.BufferBinding, // [max_tree_slots, H]
    x_cat_bind: metal.BufferBinding, // [max_tree_slots, 2*H]
    x_bind: metal.BufferBinding, // [max_tree_slots, H]
    x_norm_bind: metal.BufferBinding, // [max_tree_slots, H]
    q_pool_bind: metal.BufferBinding, // [max_tree_slots, q_dim]
    k_bind: metal.BufferBinding, // [max_tree_slots, kv_dim]
    v_bind: metal.BufferBinding, // [max_tree_slots, kv_dim]
    attn_pool_bind: metal.BufferBinding, // [max_tree_slots, q_dim]
    proj_out_bind: metal.BufferBinding, // [max_tree_slots, H]
    mlp_gate_bind: metal.BufferBinding, // [max_tree_slots, inter]
    mlp_up_bind: metal.BufferBinding, // [max_tree_slots, inter]
    mlp_act_bind: metal.BufferBinding, // [max_tree_slots, inter]
    logits_bind: metal.BufferBinding, // [max_tree_slots, draft_vocab_size]
    topk_bind: metal.BufferBinding, // [max_tree_slots, MAX_DRAFT_TOP_K]
    gqa_tree_scratch_bind: metal.BufferBinding,
    tree_k_base_off: u64,
    tree_v_base_off: u64,
    tree_kv_layer_stride: u64,

    // Cached root-step predictions computed by the most recent `syncAccepted`.
    root_top_k_count: u32,
    root_candidates: [MAX_DRAFT_TOP_K]moe.TopKCandidate,

    pub fn loadFromBytes(
        allocator: std.mem.Allocator,
        target: *qwen3.Qwen3Model,
        amf_bytes: []const u8,
        config: DraftConfig,
        options: DraftLoadOptions,
    ) DraftLoadError!DraftHead {
        var parsed = try amf.Parsed.parseBytes(allocator, amf_bytes);
        defer parsed.deinit(allocator);
        return loadInternal(allocator, target, .{ .bytes = .{ .parsed = &parsed, .raw = amf_bytes } }, config, options);
    }

    pub fn loadFromFile(
        allocator: std.mem.Allocator,
        target: *qwen3.Qwen3Model,
        amf_path: [:0]const u8,
        config: DraftConfig,
        options: DraftLoadOptions,
    ) DraftLoadError!DraftHead {
        var file = try amf.AmfFile.open(allocator, amf_path);
        defer file.close(allocator);
        return loadInternal(allocator, target, .{ .file = &file }, config, options);
    }

    pub fn deinit(self: *DraftHead, allocator: std.mem.Allocator, target: *qwen3.Qwen3Model) void {
        if (self.ane_proj) |*ap| {
            ap.close(&target.compute);
            self.ane_proj = null;
        }
        target.compute.removeTilesResident(self.kv_tiles) catch |err| {
            std.debug.panic("removeTilesResident failed in DraftHead.deinit: {}", .{err});
        };
        target.compute.removeResident(&.{ self.weights_buf, self.scratch_buf }) catch |err| {
            std.debug.panic("removeResident failed in DraftHead.deinit: {}", .{err});
        };
        for (self.kv_tiles) |tile| tile.close();
        allocator.free(self.kv_tiles);
        self.scratch_buf.close();
        self.weights_buf.close();
        self.scratch_arena.deinit();
        self.weights_arena.deinit();
        allocator.free(self.layers);
    }

    pub fn reset(self: *DraftHead) void {
        self.seq_len = 0;
        self.root_top_k_count = 0;
    }

    pub fn totalWiredBytes(self: *const DraftHead) u64 {
        return self.total_wired_bytes;
    }

    /// Returns the wired buffer binding for `draft_hidden_pool` (`[1 + max_tree_slots, H]`),
    /// where row `0` is root's draft hidden state and row `1 + slot` is tree slot `slot`.
    pub fn hiddenPoolBinding(self: *const DraftHead) metal.BufferBinding {
        return self.hidden_pool_bind;
    }

    /// Returns a buffer binding pointing at contiguous rows starting at `hidden_pool_row`.
    pub fn hiddenPoolRowBinding(self: *const DraftHead, hidden_pool_row: u32) metal.BufferBinding {
        const row_bytes = @as(u64, self.config.hidden_size) * @sizeOf(f16);
        return .{
            .buffer = self.scratch_buf,
            .offset = self.hidden_pool_bind.offset + @as(u64, hidden_pool_row) * row_bytes,
        };
    }

    /// Returns the cached top-K candidate tokens (already mapped to target vocab IDs)
    /// predicted for the root node by the most recent `syncAccepted`.
    pub fn rootCandidates(self: *const DraftHead) []const moe.TopKCandidate {
        return self.root_candidates[0..self.root_top_k_count];
    }

    fn kvTileAt(self: *const DraftHead, layer_idx: usize, chunk_idx: usize) struct { k: metal.KvTile, v: metal.KvTile } {
        const base = (layer_idx * @as(usize, self.num_chunks) + chunk_idx) * 2;
        return .{ .k = self.kv_tiles[base], .v = self.kv_tiles[base + 1] };
    }

    fn treeKBinding(self: *const DraftHead, layer_idx: usize, slot_offset: u32) metal.BufferBinding {
        const kv_row_bytes = @as(u64, self.config.kvDim()) * @sizeOf(f16);
        return .{
            .buffer = self.scratch_buf,
            .offset = self.tree_k_base_off + @as(u64, layer_idx) * self.tree_kv_layer_stride + @as(u64, slot_offset) * kv_row_bytes,
        };
    }

    fn treeVBinding(self: *const DraftHead, layer_idx: usize, slot_offset: u32) metal.BufferBinding {
        const kv_row_bytes = @as(u64, self.config.kvDim()) * @sizeOf(f16);
        return .{
            .buffer = self.scratch_buf,
            .offset = self.tree_v_base_off + @as(u64, layer_idx) * self.tree_kv_layer_stride + @as(u64, slot_offset) * kv_row_bytes,
        };
    }

    fn mapCandidateToTarget(self: *const DraftHead, cand: moe.TopKCandidate) moe.TopKCandidate {
        if (self.config.format == .eagle3 and self.d2t_host.len > 0) {
            const idx = @min(@as(usize, cand.token_id), self.d2t_host.len - 1);
            return .{
                .token_id = self.d2t_host[idx],
                .prob = cand.prob,
                .log_prob = cand.log_prob,
            };
        }
        return cand;
    }

    /// Synchronizes the draft head's committed IOSurface KV cache across `a`
    /// newly accepted `(token, target_feature)` pairs at `base_pos .. base_pos + a`:
    /// - For EAGLE-3: `target_features_bind` has shape `[a, 3 * H_tgt]` and is
    ///   projected via `fc.weight [H, 3 * H_tgt]`.
    /// - For EAGLE-2 / MTP: `target_features_bind` has shape `[a, H]`.
    ///
    /// Commits all `a` draft KV rows into `self.kv_tiles`, copies the last row's
    /// feedback hidden state (`h_prenorm[a - 1]`) into `hidden_pool[0]`, and when
    /// `top_k > 0` runs `final_norm` + `lm_head` + `encodeTopKLogitsRows` ONLY on
    /// the last row `a - 1` to populate `rootCandidates()`. Zero dynamic heap alloc.
    pub fn syncAccepted(
        self: *DraftHead,
        target: *qwen3.Qwen3Model,
        tokens: []const u32,
        target_features_bind: metal.BufferBinding,
        base_pos: u32,
        top_k: u32,
    ) DraftStepError!void {
        if (tokens.len == 0) return error.InvalidArgument;
        if (tokens.len > self.max_tree_slots) return error.BatchTooLarge;
        const m: u32 = @intCast(tokens.len);
        if (base_pos + m > self.max_seq_len) return error.PositionOutOfRange;
        if (top_k > MAX_DRAFT_TOP_K) return error.InvalidArgument;

        for (tokens, 0..) |tid, i| {
            if (tid >= self.config.vocab_size) return error.TokenOutOfRange;
            self.token_ids_host[i] = tid;
            self.positions_host[i] = base_pos + @as(u32, @intCast(i));
            self.ancestor_masks_host[i] = if (i == 63) ~@as(u64, 0) else (@as(u64, 1) << @as(u6, @intCast(i + 1))) - 1;
        }

        try self.runForwardInternal(
            target,
            m,
            0, // slot_start
            base_pos,
            true, // is_target_sync
            target_features_bind,
            true, // commit_kv
            top_k,
            true, // logits_last_row_only
        );

        if (top_k > 0) {
            const eff_k = @min(top_k, self.config.draft_vocab_size);
            for (0..eff_k) |k_idx| {
                self.root_candidates[k_idx] = self.mapCandidateToTarget(self.topk_host[k_idx]);
            }
            self.root_top_k_count = eff_k;
        } else {
            self.root_top_k_count = 0;
        }
        self.seq_len = base_pos + m;
    }

    /// Advances a batch of `m` draft tree nodes (`1 <= m <= max_tree_slots - slot_start`)
    /// into contiguous temporary tree KV slots `slot_start .. slot_start + m`
    /// without mutating committed DraftKV tiles (`self.seq_len` unchanged).
    ///
    /// - `parent_hidden_rows[i]` indexes `hidden_pool` (`0` = root, `1 + s` = tree slot `s`).
    /// - `slot_ancestor_masks[0 .. slot_start + m]` gives the tree-local KV bitmask
    ///   for every active draft tree slot (`0 .. slot_start + m - 1`).
    /// - Output feedback hidden states land in `hidden_pool[1 + slot_start .. 1 + slot_start + m]`.
    /// - If `top_k > 0`, mapped target-vocab candidates `[m * top_k]` are written to `out_candidates`.
    pub fn stepTreeBatch(
        self: *DraftHead,
        target: *qwen3.Qwen3Model,
        tokens: []const u32,
        parent_hidden_rows: []const u32,
        positions: []const u32,
        slot_ancestor_masks: []const u64,
        slot_start: u32,
        top_k: u32,
        out_candidates: ?[]moe.TopKCandidate,
    ) DraftStepError!void {
        if (tokens.len == 0 or tokens.len != parent_hidden_rows.len or tokens.len != positions.len) {
            return error.InvalidArgument;
        }
        const m: u32 = @intCast(tokens.len);
        const total_slots = slot_start + m;
        if (total_slots > self.max_tree_slots) return error.SlotOutOfRange;
        if (slot_ancestor_masks.len < total_slots) return error.InvalidArgument;
        if (top_k > MAX_DRAFT_TOP_K) return error.InvalidArgument;
        if (top_k > 0) {
            const dst = out_candidates orelse return error.InvalidArgument;
            if (dst.len < @as(usize, m) * top_k) return error.InvalidArgument;
        }

        for (0..m) |i| {
            if (tokens[i] >= self.config.vocab_size) return error.TokenOutOfRange;
            if (positions[i] >= self.max_seq_len) return error.PositionOutOfRange;
            if (parent_hidden_rows[i] > self.max_tree_slots) return error.SlotOutOfRange;
            self.token_ids_host[i] = tokens[i];
            self.parent_slots_host[i] = parent_hidden_rows[i];
            self.positions_host[i] = positions[i];
        }
        @memcpy(self.ancestor_masks_host[0..total_slots], slot_ancestor_masks[0..total_slots]);

        try self.runForwardInternal(
            target,
            m,
            slot_start,
            self.seq_len,
            false, // is_target_sync
            self.h_in_bind,
            false, // commit_kv
            top_k,
            false, // logits_last_row_only
        );

        if (top_k > 0) {
            const dst = out_candidates.?;
            const eff_k = @min(top_k, self.config.draft_vocab_size);
            for (0..m) |r| {
                for (0..eff_k) |k_idx| {
                    dst[r * top_k + k_idx] = self.mapCandidateToTarget(self.topk_host[r * eff_k + k_idx]);
                }
            }
        }
    }

    fn runForwardInternal(
        self: *DraftHead,
        target: *qwen3.Qwen3Model,
        m: u32,
        slot_start: u32,
        base_pos: u32,
        is_target_sync: bool,
        target_features_bind: metal.BufferBinding,
        commit_kv: bool,
        top_k: u32,
        logits_last_row_only: bool,
    ) DraftStepError!void {
        const h = self.config.hidden_size;
        const q_dim = self.config.qDim();
        const total_tree_nodes = slot_start + m;
        const q_slot_bind: metal.BufferBinding = .{
            .buffer = self.scratch_buf,
            .offset = self.q_pool_bind.offset + @as(u64, slot_start) * q_dim * @sizeOf(f16),
        };
        const attn_slot_bind: metal.BufferBinding = .{
            .buffer = self.scratch_buf,
            .offset = self.attn_pool_bind.offset + @as(u64, slot_start) * q_dim * @sizeOf(f16),
        };

        const use_ane_o_proj = (self.effective_placement == .ane and self.ane_proj != null and m <= self.ane_proj.?.batch_rows);

        var frame = try target.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();

        // 1. Prepare `h_in_bind [m, H]` (either from target verifier features or parent draft hidden pool).
        var h_in_active = self.h_in_bind;
        if (is_target_sync) {
            if (self.config.format == .eagle3) {
                try target.encodeLinearBatched(&frame, self.fc.?, target_features_bind, self.h_in_bind, m, false);
            } else {
                h_in_active = target_features_bind;
            }
        } else {
            try target.moe_pipes.encodeGatherRows(&frame, .{
                .src = self.hidden_pool_bind,
                .indices = self.parent_slots_bind,
                .dst = self.h_in_bind,
                .count = m,
                .row_elems = h,
            }, false);
        }

        // 2. Token embedding lookup `[m, H]`.
        try target.moe_pipes.encodeEmbedLookup(&frame, self.embed.dtype, .{
            .weight = self.embed.weight,
            .scale_bias = self.embed.scale_bias,
            .tokens = .{ .token_ids = self.token_ids_bind },
            .out = self.emb_bind,
            .m = m,
            .vocab_size = self.config.vocab_size,
            .hidden_size = h,
        }, false);

        // 3. Format-specific feature fusion before layer 0.
        switch (self.config.format) {
            .eagle3 => {
                const l0 = self.layers[0];
                try target.attn_pipes.encodeRmsNorm(&frame, .{
                    .x = self.emb_bind,
                    .w = l0.input_layernorm.binding,
                    .y = self.e_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = l0.input_layernorm.dtype,
                }, true);
                try target.attn_pipes.encodeRmsNorm(&frame, .{
                    .x = h_in_active,
                    .w = l0.hidden_norm.?.binding,
                    .y = self.h_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = l0.hidden_norm.?.dtype,
                }, false);

                // Concatenate `[e_norm, h_norm]` -> `x_cat [m, 2*H]`, and set `x_bind = residual`.
                try target.moe_pipes.encodeCopyStrided(&frame, .{
                    .src = self.e_norm_bind,
                    .dst = self.x_cat_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = 2 * h,
                    .dst_col_off = 0,
                }, true);
                try target.moe_pipes.encodeCopyStrided(&frame, .{
                    .src = self.h_norm_bind,
                    .dst = self.x_cat_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = 2 * h,
                    .dst_col_off = h,
                }, false);
                try target.moe_pipes.encodeCopyStrided(&frame, .{
                    .src = if (self.config.norm_before_residual) self.h_norm_bind else h_in_active,
                    .dst = self.x_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                    .dst_col_off = 0,
                }, false);
            },
            .eagle2 => {
                try target.moe_pipes.encodeCopyStrided(&frame, .{
                    .src = self.emb_bind,
                    .dst = self.x_cat_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = 2 * h,
                    .dst_col_off = 0,
                }, true);
                try target.moe_pipes.encodeCopyStrided(&frame, .{
                    .src = h_in_active,
                    .dst = self.x_cat_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = 2 * h,
                    .dst_col_off = h,
                }, false);
                try target.encodeLinearBatched(&frame, self.fc.?, self.x_cat_bind, self.x_bind, m, true);
            },
            .mtp => {
                try target.attn_pipes.encodeRmsNorm(&frame, .{
                    .x = self.emb_bind,
                    .w = self.enorm.?.binding,
                    .y = self.e_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = self.enorm.?.dtype,
                }, true);
                try target.attn_pipes.encodeRmsNorm(&frame, .{
                    .x = h_in_active,
                    .w = self.hnorm.?.binding,
                    .y = self.h_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = self.hnorm.?.dtype,
                }, false);
                try target.moe_pipes.encodeCopyStrided(&frame, .{
                    .src = self.e_norm_bind,
                    .dst = self.x_cat_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = 2 * h,
                    .dst_col_off = 0,
                }, true);
                try target.moe_pipes.encodeCopyStrided(&frame, .{
                    .src = self.h_norm_bind,
                    .dst = self.x_cat_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = 2 * h,
                    .dst_col_off = h,
                }, false);
                try target.encodeLinearBatched(&frame, self.eh_proj.?, self.x_cat_bind, self.x_bind, m, true);
            },
            .l1_only, .small_draft => unreachable,
        }

        // 4. Draft transformer layer(s).
        for (self.layers, 0..) |layer, l| {
            const qkv_in: metal.BufferBinding = if (self.config.format == .eagle3 and l == 0)
                self.x_cat_bind
            else blk: {
                try target.attn_pipes.encodeRmsNorm(&frame, .{
                    .x = self.x_bind,
                    .w = layer.input_layernorm.binding,
                    .y = self.x_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = layer.input_layernorm.dtype,
                }, true);
                break :blk self.x_norm_bind;
            };

            try target.encodeLinearBatched(&frame, layer.q_proj, qkv_in, q_slot_bind, m, true);
            try target.encodeLinearBatched(&frame, layer.k_proj, qkv_in, self.k_bind, m, false);
            try target.encodeLinearBatched(&frame, layer.v_proj, qkv_in, self.v_bind, m, false);

            try target.attn_pipes.encodeTreeQkNormRope(&frame, .{
                .q = q_slot_bind,
                .k = self.k_bind,
                .v = self.v_bind,
                .q_norm = layer.q_norm,
                .k_norm = layer.k_norm,
                .positions = self.positions_bind,
                .k_tree_out = self.treeKBinding(l, slot_start),
                .v_tree_out = self.treeVBinding(l, slot_start),
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .rms_eps = self.config.rms_norm_eps,
                .rope_theta = self.config.rope_theta,
                .norm_dtype = layer.qk_norm_dtype,
            }, true);

            try target.attn_pipes.encodeGqaTreeVerifyAttn(&frame, .{
                .q = self.q_pool_bind,
                .kv_tiles = if (base_pos > 0) layer.kv_tiles_binding else null,
                .base_pos = base_pos,
                .k_tree = self.treeKBinding(l, 0),
                .v_tree = self.treeVBinding(l, 0),
                .ancestor_masks = self.ancestor_masks_bind,
                .out = self.attn_pool_bind,
                .scratch = self.gqa_tree_scratch_bind,
                .max_splits = attention.MAX_SPLITS,
                .num_nodes = total_tree_nodes,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .chunk_tokens = self.chunk_tokens,
            }, true);

            // D5 Placement: either run `layers[0].o_proj` on ANE over zero-copy IOSurface
            // or stay in the same Metal 4 command buffer on GPU!
            if (l == 0 and use_ane_o_proj) {
                const ap = &(self.ane_proj.?);
                try target.moe_pipes.encodeCopyStrided(&frame, .{
                    .src = attn_slot_bind,
                    .dst = ap.inBinding(),
                    .num_rows = m,
                    .copy_cols = q_dim,
                    .src_stride = q_dim,
                    .dst_stride = q_dim,
                }, true);
                const mid_sig = try frame.commit();
                try target.compute.wait(mid_sig, self.timeout_ms);

                const pinfo = try ap.model.predict(ap.plan);
                ap.zero_copy_ok = pinfo.allOutputsZeroCopy();

                frame = try target.compute.beginFrame(self.timeout_ms);
                try target.moe_pipes.encodeResidualAdd(&frame, .{
                    .x = self.x_bind,
                    .delta = ap.outBinding(),
                    .count = m * h,
                }, false);
            } else {
                try target.encodeLinearBatched(&frame, layer.o_proj, attn_slot_bind, self.proj_out_bind, m, true);
                try target.moe_pipes.encodeResidualAdd(&frame, .{
                    .x = self.x_bind,
                    .delta = self.proj_out_bind,
                    .count = m * h,
                }, true);
            }

            try target.attn_pipes.encodeRmsNorm(&frame, .{
                .x = self.x_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, true);

            try target.encodeLinearBatched(&frame, layer.gate_proj, self.x_norm_bind, self.mlp_gate_bind, m, true);
            try target.encodeLinearBatched(&frame, layer.up_proj, self.x_norm_bind, self.mlp_up_bind, m, false);
            try target.moe_pipes.encodeSiluMul(&frame, .{
                .gate = self.mlp_gate_bind,
                .up = self.mlp_up_bind,
                .out = self.mlp_act_bind,
                .count = m * self.config.intermediate_size,
            }, true);
            try target.encodeLinearBatched(&frame, layer.down_proj, self.mlp_act_bind, self.proj_out_bind, m, true);
            try target.moe_pipes.encodeResidualAdd(&frame, .{
                .x = self.x_bind,
                .delta = self.proj_out_bind,
                .count = m * h,
            }, true);

            // If syncing accepted tokens, commit draft K/V rows into IOSurface KvTiles now.
            if (commit_kv) {
                var identity_nodes: [attention.MAX_TREE_NODES]u16 = undefined;
                for (0..m) |idx| identity_nodes[idx] = @intCast(idx);
                var offset: u32 = 0;
                while (offset < m) {
                    const cur_base = base_pos + offset;
                    const chunk0: u32 = cur_base / self.chunk_tokens;
                    const max_end: u32 = (chunk0 + 2) * self.chunk_tokens;
                    const take: u32 = @min(m - offset, max_end - cur_base);
                    const crosses = ((cur_base + take - 1) / self.chunk_tokens) > chunk0;
                    const t0 = self.kvTileAt(l, @intCast(chunk0));
                    const t1 = if (crosses) self.kvTileAt(l, @intCast(chunk0 + 1)) else null;
                    try target.attn_pipes.encodeKvTreeCommit(&frame, .{
                        .k_tree = self.treeKBinding(l, 0),
                        .v_tree = self.treeVBinding(l, 0),
                        .accepted_nodes = .{ .inline_u16 = identity_nodes[offset .. offset + take] },
                        .accepted_count = take,
                        .k_tile0 = t0.k,
                        .v_tile0 = t0.v,
                        .k_tile1 = if (t1) |p| p.k else null,
                        .v_tile1 = if (t1) |p| p.v else null,
                        .num_kv_heads = self.config.num_key_value_heads,
                        .head_dim = self.config.head_dim,
                        .base_pos = cur_base,
                        .chunk_tokens = self.chunk_tokens,
                    }, true);
                    offset += take;
                }
            }
        }

        // 5. Persist feedback hidden state `h_prenorm` into `hidden_pool`:
        //    - syncAccepted: row `m - 1` -> `hidden_pool[0]` (root state)
        //    - stepTreeBatch: rows `0..m` -> `hidden_pool[1 + slot_start .. 1 + slot_start + m]`
        const row_bytes = @as(u64, h) * @sizeOf(f16);
        if (commit_kv) {
            const last_src: metal.BufferBinding = .{
                .buffer = self.scratch_buf,
                .offset = self.x_bind.offset + @as(u64, m - 1) * row_bytes,
            };
            try target.moe_pipes.encodeCopyStrided(&frame, .{
                .src = last_src,
                .dst = self.hidden_pool_bind,
                .num_rows = 1,
                .copy_cols = h,
                .src_stride = h,
                .dst_stride = h,
            }, true);
        } else {
            const dst_pool: metal.BufferBinding = .{
                .buffer = self.scratch_buf,
                .offset = self.hidden_pool_bind.offset + @as(u64, 1 + slot_start) * row_bytes,
            };
            try target.moe_pipes.encodeCopyStrided(&frame, .{
                .src = self.x_bind,
                .dst = dst_pool,
                .num_rows = m,
                .copy_cols = h,
                .src_stride = h,
                .dst_stride = h,
            }, true);
        }

        // 6. Optional final norm + `lm_head` + single-dispatch FP32 softmax top-K.
        if (top_k > 0) {
            const eff_k = @min(top_k, self.config.draft_vocab_size);
            const logit_rows: u32 = if (logits_last_row_only) 1 else m;
            const norm_src: metal.BufferBinding = if (logits_last_row_only) .{
                .buffer = self.scratch_buf,
                .offset = self.x_bind.offset + @as(u64, m - 1) * row_bytes,
            } else self.x_bind;

            try target.attn_pipes.encodeRmsNorm(&frame, .{
                .x = norm_src,
                .w = self.final_norm.binding,
                .y = self.x_norm_bind,
                .num_rows = logit_rows,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = self.final_norm.dtype,
            }, true);
            try target.encodeLinearBatched(&frame, self.lm_head, self.x_norm_bind, self.logits_bind, logit_rows, true);
            try target.moe_pipes.encodeTopKLogitsRows(&frame, .{
                .logits = self.logits_bind,
                .out_candidates = self.topk_bind,
                .num_rows = logit_rows,
                .vocab_size = self.config.draft_vocab_size,
                .top_k = eff_k,
                .logits_dtype = .f16,
            }, true);
        }

        const sig = try frame.commit();
        try target.compute.wait(sig, self.timeout_ms);
    }
};

pub const SmallDraftLoadError = error{
    TargetIncompatible,
    BudgetExceeded,
    InvalidConfig,
    OutOfMemory,
} || qwen3.LoadError || DraftLoadError;

/// Standalone same-family small draft model (`D4`, ADR-0005, ADR-0008, ADR-0013),
/// e.g. `Qwen3-0.6B` BF16 pairing with `Qwen3-14B-4bit` or `Qwen3-30B-A3B`.
///
/// Shares the target tokenizer (`vocab_size == target.config.vocab_size`),
/// maintains its own committed IOSurface KV cache with zero target hidden-state
/// dependency, processes L1 `>= 4` seed chains in a single batched forward pass,
/// and expands dynamic EAGLE-2 confidence-ranked draft trees via
/// `Qwen3Model.stepDraftTreeBatch` (`encodeTopKLogitsRows`) with zero hot-path
/// heap allocation.
pub const SmallDraftModel = struct {
    model: qwen3.Qwen3Model,
    own_compute_queue: bool,
    requested_placement: Placement,
    effective_placement: Placement,
    placement_gate_ane_us: f64,
    placement_gate_gpu_us: f64,
    ane_proj: ?AneProjection,
    last_ane_zero_copy_ok: bool,
    root_top_k_count: u32,
    root_candidates: [MAX_DRAFT_TOP_K]moe.TopKCandidate,

    pub fn loadFromBytes(
        allocator: std.mem.Allocator,
        target: *const qwen3.Qwen3Model,
        amf_bytes: []const u8,
        config: qwen3.ModelConfig,
        options: DraftLoadOptions,
    ) SmallDraftLoadError!SmallDraftModel {
        const load_opts = try buildLoadOptions(target, config, options);
        var model = try qwen3.Qwen3Model.loadFromBytes(allocator, amf_bytes, config, load_opts);
        errdefer model.deinit(allocator);
        std.debug.assert(model.compute.handle != target.compute.handle);
        return finalizeInit(allocator, model, options);
    }

    pub fn loadFromFile(
        allocator: std.mem.Allocator,
        target: *const qwen3.Qwen3Model,
        amf_path: [:0]const u8,
        config: qwen3.ModelConfig,
        options: DraftLoadOptions,
    ) SmallDraftLoadError!SmallDraftModel {
        const load_opts = try buildLoadOptions(target, config, options);
        var model = try qwen3.Qwen3Model.loadFromFile(allocator, amf_path, config, load_opts);
        errdefer model.deinit(allocator);
        std.debug.assert(model.compute.handle != target.compute.handle);
        return finalizeInit(allocator, model, options);
    }

    fn buildLoadOptions(
        target: *const qwen3.Qwen3Model,
        config: qwen3.ModelConfig,
        options: DraftLoadOptions,
    ) SmallDraftLoadError!qwen3.LoadOptions {
        try config.validate();
        if (config.vocab_size != target.config.vocab_size) return error.TargetIncompatible;
        if (options.max_seq_len == 0 or options.chunk_tokens == 0 or
            options.max_tree_slots < 2 or options.max_tree_slots > attention.MAX_TREE_NODES)
        {
            return error.InvalidConfig;
        }
        if (target.total_wired_bytes >= target.wired_budget_bytes) return error.BudgetExceeded;
        const rem_limit = target.wired_budget_bytes - target.total_wired_bytes;
        const draft_budget = budget_mod.Budget.compute(.{
            .memory_bytes = target.wired_budget_bytes + 4 * page.APPLE_PAGE_SIZE,
            .user_wire_limit = rem_limit,
            .global_user_wire_limit = rem_limit,
            .gpu_working_set_bytes = rem_limit,
            .configured_cap = rem_limit,
            .headroom_bytes = 0,
        });
        return .{
            .max_seq_len = @min(options.max_seq_len + 1, config.max_position_embeddings),
            .chunk_tokens = options.chunk_tokens,
            .max_tree_nodes = options.max_tree_slots,
            .enable_feature_taps = false,
            .budget_override = draft_budget,
            .verify_crcs = options.verify_crcs,
            .timeout_ms = options.timeout_ms,
        };
    }

    fn finalizeInit(
        allocator: std.mem.Allocator,
        model_in: qwen3.Qwen3Model,
        options: DraftLoadOptions,
    ) SmallDraftLoadError!SmallDraftModel {
        var model = model_in;
        errdefer model.deinit(allocator);

        var effective_placement: Placement = .gpu;
        var ane_proj: ?AneProjection = null;
        errdefer if (ane_proj) |*ap| ap.close(&model.compute);
        var gate_ane_us: f64 = 0.0;
        var gate_gpu_us: f64 = 0.0;

        const q_dim = model.config.qDim();
        const h = model.config.hidden_size;
        const l0 = model.layers[0];
        const can_ane = (q_dim % metal.KV_TILE_ROW_ALIGN_F16 == 0) and
            (h % metal.KV_TILE_ROW_ALIGN_F16 == 0);
        if (options.placement == .ane and can_ane) {
            const wf16 = try model.extractLinearWeightsF16(allocator, l0.shard_idx, l0.o_proj);
            defer allocator.free(wf16);
            ane_proj = try AneProjection.init(allocator, model.device, &model.compute, q_dim, h, ANE_MAX_BATCH_ROWS, wf16);
            effective_placement = .ane;
        } else if (options.placement == .auto and can_ane) {
            const wf16 = try model.extractLinearWeightsF16(allocator, l0.shard_idx, l0.o_proj);
            defer allocator.free(wf16);
            var probe_ap = try AneProjection.init(allocator, model.device, &model.compute, q_dim, h, 1, wf16);
            const cal = try calibrateD5Placement(
                &model,
                &probe_ap,
                l0.o_proj,
                model.attn_out_bind,
                model.proj_out_bind,
                options.timeout_ms,
            );
            gate_ane_us = cal.ane_us;
            gate_gpu_us = cal.gpu_us;
            probe_ap.close(&model.compute);
            effective_placement = cal.winner;
            if (effective_placement == .ane) {
                ane_proj = try AneProjection.init(allocator, model.device, &model.compute, q_dim, h, ANE_MAX_BATCH_ROWS, wf16);
            }
        }

        const zc_ok = if (ane_proj) |ap| ap.zero_copy_ok else true;
        return .{
            .model = model,
            .own_compute_queue = options.own_compute_queue,
            .requested_placement = options.placement,
            .effective_placement = effective_placement,
            .placement_gate_ane_us = gate_ane_us,
            .placement_gate_gpu_us = gate_gpu_us,
            .ane_proj = ane_proj,
            .last_ane_zero_copy_ok = zc_ok,
            .root_top_k_count = 0,
            .root_candidates = undefined,
        };
    }

    pub fn deinit(self: *SmallDraftModel, allocator: std.mem.Allocator) void {
        if (self.ane_proj) |*ap| {
            ap.close(&self.model.compute);
            self.ane_proj = null;
        }
        self.model.deinit(allocator);
    }

    pub fn reset(self: *SmallDraftModel) void {
        self.model.resetKvCache();
        self.root_top_k_count = 0;
    }

    pub fn hasOwnComputeQueue(self: *const SmallDraftModel) bool {
        return self.own_compute_queue;
    }

    /// Rewinds committed KV length in O(1) without GPU work or allocation
    /// (`D8` PEARL rollback, ADR-0008). Causal KV rows at `< new_seq_len`
    /// remain intact and subsequent `syncAccepted` overwrites `>= new_seq_len`.
    pub fn rewindKvLen(self: *SmallDraftModel, new_seq_len: u32) error{InvalidArgument}!void {
        if (new_seq_len > self.model.seq_len or new_seq_len > self.model.max_seq_len) {
            return error.InvalidArgument;
        }
        self.model.seq_len = new_seq_len;
        self.root_top_k_count = 0;
    }

    /// Saves cached root candidates into a caller-owned fixed buffer (zero alloc).
    pub fn saveRootCandidates(self: *const SmallDraftModel, out: *[MAX_DRAFT_TOP_K]moe.TopKCandidate) u32 {
        const n = self.root_top_k_count;
        if (n > 0) @memcpy(out[0..n], self.root_candidates[0..n]);
        return n;
    }

    /// Restores cached root candidates from a saved snapshot (zero alloc).
    pub fn restoreRootCandidates(self: *SmallDraftModel, src: []const moe.TopKCandidate) void {
        const n = @min(src.len, MAX_DRAFT_TOP_K);
        if (n > 0) @memcpy(self.root_candidates[0..n], src[0..n]);
        self.root_top_k_count = @intCast(n);
    }

    pub fn totalWiredBytes(self: *const SmallDraftModel) u64 {
        return self.model.totalWiredBytes();
    }

    pub fn maxSeqLen(self: *const SmallDraftModel) u32 {
        return self.model.max_seq_len;
    }

    pub fn maxTreeSlots(self: *const SmallDraftModel) u32 {
        return self.model.max_tree_nodes;
    }

    pub fn seqLen(self: *const SmallDraftModel) u32 {
        return self.model.seq_len;
    }

    pub fn rootCandidates(self: *const SmallDraftModel) []const moe.TopKCandidate {
        return self.root_candidates[0..self.root_top_k_count];
    }

    fn aneHook(self: *SmallDraftModel) ?qwen3.AneOProjHook {
        if (self.effective_placement != .ane) return null;
        const ap = &(self.ane_proj orelse return null);
        return .{
            .model = ap.model,
            .plan = ap.plan,
            .in_bind = ap.inBinding(),
            .out_bind = ap.outBinding(),
            .max_batch_rows = ap.batch_rows,
            .c_in = ap.c_in,
            .c_out = ap.c_out,
            .last_zero_copy_ok = &self.last_ane_zero_copy_ok,
        };
    }

    /// Prefills `prompt_tokens` into the small draft model's own KV cache and
    /// commits `first_tok` at `pos = prompt_tokens.len` (`seq_len = prompt.len + 1`),
    /// caching `rootCandidates()` for position `prompt_tokens.len + 1`. Zero heap alloc.
    pub fn prefillAndSeedRoot(
        self: *SmallDraftModel,
        prompt_tokens: []const u32,
        first_tok: u32,
        top_k: u32,
    ) qwen3.StepError!void {
        if (prompt_tokens.len == 0) return error.InvalidArgument;
        if (prompt_tokens.len + 1 > self.model.max_seq_len) return error.PositionOutOfRange;
        self.reset();

        const chunk_cap: usize = self.model.max_tree_nodes;
        var pos: usize = 0;
        while (pos < prompt_tokens.len) {
            const take = @min(chunk_cap, prompt_tokens.len - pos);
            _ = try self.model.prefillChunk(prompt_tokens[pos .. pos + take], @intCast(pos), false);
            pos += take;
        }
        const first_slice = [_]u32{first_tok};
        try self.syncAccepted(&first_slice, @intCast(prompt_tokens.len), top_k);
    }

    /// Commits newly emitted tokens (`[d_1..d_k, bonus]`) into the small draft
    /// model's IOSurface KV cache at `base_pos .. base_pos + tokens.len` in ONE
    /// Metal 4 command buffer and refreshes `rootCandidates()` from the final token.
    /// Zero dynamic heap allocation.
    pub fn syncAccepted(
        self: *SmallDraftModel,
        tokens: []const u32,
        base_pos: u32,
        top_k: u32,
    ) qwen3.StepError!void {
        if (tokens.len == 0) return error.InvalidArgument;
        if (tokens.len > self.model.max_tree_nodes) return error.TreeTooLarge;
        const m: u32 = @intCast(tokens.len);
        if (base_pos + m > self.model.max_seq_len) return error.PositionOutOfRange;

        if (top_k == 0) {
            _ = try self.model.prefillChunk(tokens, base_pos, false);
            self.root_top_k_count = 0;
            return;
        }

        var positions: [attention.MAX_TREE_NODES]u32 = undefined;
        var masks: [attention.MAX_TREE_NODES]u64 = undefined;
        for (0..m) |i| {
            positions[i] = base_pos + @as(u32, @intCast(i));
            masks[i] = if (i == 63) ~@as(u64, 0) else (@as(u64, 1) << @as(u6, @intCast(i + 1))) - 1;
        }
        const eff_k: u32 = @min(top_k, @min(MAX_DRAFT_TOP_K, self.model.config.vocab_size));
        const hook_opt = self.aneHook();
        const hook_ptr: ?*const qwen3.AneOProjHook = if (hook_opt) |*hk| hk else null;
        const cands = try self.model.stepDraftTreeBatch(
            tokens,
            positions[0..m],
            0,
            masks[0..m],
            base_pos,
            eff_k,
            true, // logits_last_row_only
            true, // commit_kv
            hook_ptr,
        );
        @memcpy(self.root_candidates[0..eff_k], cands[0..eff_k]);
        self.root_top_k_count = eff_k;
    }

    /// Like `syncAccepted`, automatically splitting across `maxTreeSlots()` if
    /// `tokens.len > maxTreeSlots()` (e.g. pending PEARL root + full accepted chain).
    pub fn syncAcceptedChunked(
        self: *SmallDraftModel,
        tokens: []const u32,
        base_pos: u32,
        top_k: u32,
    ) qwen3.StepError!void {
        if (tokens.len == 0) return error.InvalidArgument;
        const cap: usize = self.model.max_tree_nodes;
        var off: usize = 0;
        while (off < tokens.len) {
            const take = @min(cap, tokens.len - off);
            const is_last = (off + take == tokens.len);
            try self.syncAccepted(
                tokens[off .. off + take],
                base_pos + @as(u32, @intCast(off)),
                if (is_last) top_k else 0,
            );
            off += take;
        }
    }

    /// Runs an entire L1 `>= 4` seed chain `[s_1..s_c]` in ONE batched forward pass
    /// (`ADR-0005`), placing its K/V into temporary tree slots `0..c` (without
    /// mutating committed KV) and writing the tail's top-K continuation candidates
    /// into `out_candidates[0..eff_k]`. Zero dynamic heap allocation.
    pub fn stepSeedChain(
        self: *SmallDraftModel,
        seed_tokens: []const u32,
        first_seed_pos: u32,
        slot_ancestor_masks: []u64,
        top_k: u32,
        out_candidates: []moe.TopKCandidate,
    ) qwen3.StepError!u32 {
        if (seed_tokens.len == 0) return error.InvalidArgument;
        if (seed_tokens.len > self.model.max_tree_nodes) return error.TreeTooLarge;
        const m: u32 = @intCast(seed_tokens.len);
        if (slot_ancestor_masks.len < m) return error.InvalidArgument;

        var positions: [attention.MAX_TREE_NODES]u32 = undefined;
        for (0..m) |i| {
            positions[i] = first_seed_pos + @as(u32, @intCast(i));
            slot_ancestor_masks[i] = if (i == 63) ~@as(u64, 0) else (@as(u64, 1) << @as(u6, @intCast(i + 1))) - 1;
        }
        const eff_k: u32 = @min(top_k, @min(MAX_DRAFT_TOP_K, self.model.config.vocab_size));
        if (out_candidates.len < eff_k) return error.InvalidArgument;

        const hook_opt = self.aneHook();
        const hook_ptr: ?*const qwen3.AneOProjHook = if (hook_opt) |*hk| hk else null;
        const cands = try self.model.stepDraftTreeBatch(
            seed_tokens,
            positions[0..m],
            0,
            slot_ancestor_masks[0..m],
            self.model.seq_len,
            eff_k,
            true, // logits_last_row_only (only seed tail needs next-token candidates)
            false, // commit_kv
            hook_ptr,
        );
        @memcpy(out_candidates[0..eff_k], cands[0..eff_k]);
        return eff_k;
    }

    /// Expands a batch of `m` draft frontier nodes placed into tree KV slots
    /// `slot_start .. slot_start + m` (without mutating committed KV) and writes
    /// `[m * eff_k]` top-K candidates into `out_candidates`. Zero dynamic heap alloc.
    pub fn stepTreeBatch(
        self: *SmallDraftModel,
        tokens: []const u32,
        positions: []const u32,
        slot_ancestor_masks: []const u64,
        slot_start: u32,
        top_k: u32,
        out_candidates: []moe.TopKCandidate,
    ) qwen3.StepError!u32 {
        const eff_k: u32 = @min(top_k, @min(MAX_DRAFT_TOP_K, self.model.config.vocab_size));
        if (out_candidates.len < tokens.len * @as(usize, eff_k)) return error.InvalidArgument;

        const hook_opt = self.aneHook();
        const hook_ptr: ?*const qwen3.AneOProjHook = if (hook_opt) |*hk| hk else null;
        const cands = try self.model.stepDraftTreeBatch(
            tokens,
            positions,
            slot_start,
            slot_ancestor_masks,
            self.model.seq_len,
            eff_k,
            false, // logits_last_row_only
            false, // commit_kv
            hook_ptr,
        );
        const total = tokens.len * @as(usize, eff_k);
        @memcpy(out_candidates[0..total], cands[0..total]);
        return eff_k;
    }
};

fn findRequired(parsed: *const amf.Parsed, name: []const u8) DraftLoadError!*const amf.TensorEntry {
    return parsed.find(name) orelse error.MissingTensor;
}

fn findLinearOpt(parsed: *const amf.Parsed, base: []const u8) ?*const amf.TensorEntry {
    if (parsed.find(base)) |e| return e;
    var alt_buf: [64]u8 = undefined;
    const with_weight = std.fmt.bufPrint(&alt_buf, "{s}.weight", .{base}) catch return null;
    return parsed.find(with_weight);
}

fn findLinearReq(parsed: *const amf.Parsed, base: []const u8) DraftLoadError!*const amf.TensorEntry {
    return findLinearOpt(parsed, base) orelse error.MissingTensor;
}

fn validateNorm1D(entry: *const amf.TensorEntry, dim: u32) DraftLoadError!quant.FloatDType {
    const dt = try entry.dataType();
    if (entry.rank != 1 or entry.shape[0] != dim) return error.UnexpectedTensorShape;
    return switch (dt) {
        .f16 => .f16,
        .bf16 => .bf16,
        else => error.UnexpectedTensorDType,
    };
}

fn validateLinear2D(entry: *const amf.TensorEntry, n: u32, k: u32) DraftLoadError!quant.WeightDType {
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
        .q2_affine => blk: {
            if ((k % 64) != 0) return error.UnalignedK;
            break :blk .q2_affine;
        },
        .f32, .u32 => error.UnexpectedTensorDType,
    };
}

const LoadedBlob = struct {
    entry: *const amf.TensorEntry,
    slice: []align(page.APPLE_PAGE_SIZE) u8,
    offset: u64,
    buffer: metal.Buffer,

    fn asBinding(self: LoadedBlob) metal.BufferBinding {
        return .{ .buffer = self.buffer, .offset = self.offset };
    }

    fn asNormBinding(self: LoadedBlob, dtype: quant.FloatDType) qwen3.NormBinding {
        return .{ .binding = self.asBinding(), .dtype = dtype };
    }

    fn asLinearBinding(self: LoadedBlob, dtype: quant.WeightDType) qwen3.LinearBinding {
        const sb: ?metal.BufferBinding = switch (dtype) {
            .f16, .bf16 => null,
            .q4_affine, .q3_affine, .q2_affine => .{ .buffer = self.buffer, .offset = self.offset + self.entry.params_offset },
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

fn loadBlobIntoArena(
    source: AmfSource,
    entry: *const amf.TensorEntry,
    arena: *arena_mod.WiredArena,
    buf: metal.Buffer,
    verify_crc: bool,
) DraftLoadError!LoadedBlob {
    const slice = try source.loadBlob(entry, arena, verify_crc);
    return .{
        .entry = entry,
        .slice = slice,
        .offset = arena.offsetOf(slice.ptr),
        .buffer = buf,
    };
}

fn convertWeightBlobToF16(
    allocator: std.mem.Allocator,
    blob: LoadedBlob,
    dtype: quant.WeightDType,
) DraftLoadError![]f16 {
    const elems = std.math.mul(usize, blob.entry.shape[0], blob.entry.shape[1]) catch return error.Overflow;
    const out = try allocator.alloc(f16, elems);
    errdefer allocator.free(out);
    switch (dtype) {
        .f16 => {
            const src: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, blob.slice[0 .. elems * 2]));
            @memcpy(out, src);
        },
        .bf16 => {
            const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, blob.slice[0 .. elems * 2]));
            for (src, 0..) |bits, i| {
                const v = std.math.clamp(quant.bf16ToF32(bits), -65504.0, 65504.0);
                out[i] = @floatCast(v);
            }
        },
        .q2_affine, .q3_affine, .q4_affine => return error.UnexpectedTensorDType,
    }
    return out;
}

/// Startup D5 placement micro-calibration (ADR-0008): compares zero-copy
/// `GPU -> IOSurface -> ANE -> IOSurface -> GPU` round-trip latency against
/// in-command-buffer GPU `encodeLinearGemv` on `layers[0].o_proj`.
fn calibrateD5Placement(
    target: *qwen3.Qwen3Model,
    ap: *AneProjection,
    o_proj: qwen3.LinearBinding,
    in_bind: metal.BufferBinding,
    out_bind: metal.BufferBinding,
    timeout_ms: u64,
) DraftLoadError!struct { ane_us: f64, gpu_us: f64, winner: Placement } {
    const warmups: u32 = 4;
    const iters: u32 = 12;

    // 1. Measure ANE + zero-copy IOSurface round-trip (`GPU copy -> ANE -> GPU residual`).
    for (0..warmups) |_| {
        var f1 = try target.compute.beginFrame(timeout_ms);
        errdefer f1.abort();
        try target.moe_pipes.encodeCopyStrided(&f1, .{
            .src = in_bind,
            .dst = ap.inBinding(),
            .num_rows = 1,
            .copy_cols = ap.c_in,
            .src_stride = ap.c_in,
            .dst_stride = ap.c_in,
        }, false);
        const s1 = try f1.commit();
        try target.compute.wait(s1, timeout_ms);
        _ = try ap.model.predict(ap.plan);
        var f2 = try target.compute.beginFrame(timeout_ms);
        errdefer f2.abort();
        try target.moe_pipes.encodeCopyStrided(&f2, .{
            .src = ap.outBinding(),
            .dst = out_bind,
            .num_rows = 1,
            .copy_cols = ap.c_out,
            .src_stride = ap.c_out,
            .dst_stride = ap.c_out,
        }, false);
        const s2 = try f2.commit();
        try target.compute.wait(s2, timeout_ms);
    }

    const at0 = try vmstat.nowNs();
    for (0..iters) |_| {
        var f1 = try target.compute.beginFrame(timeout_ms);
        errdefer f1.abort();
        try target.moe_pipes.encodeCopyStrided(&f1, .{
            .src = in_bind,
            .dst = ap.inBinding(),
            .num_rows = 1,
            .copy_cols = ap.c_in,
            .src_stride = ap.c_in,
            .dst_stride = ap.c_in,
        }, false);
        const s1 = try f1.commit();
        try target.compute.wait(s1, timeout_ms);
        _ = try ap.model.predict(ap.plan);
        var f2 = try target.compute.beginFrame(timeout_ms);
        errdefer f2.abort();
        try target.moe_pipes.encodeCopyStrided(&f2, .{
            .src = ap.outBinding(),
            .dst = out_bind,
            .num_rows = 1,
            .copy_cols = ap.c_out,
            .src_stride = ap.c_out,
            .dst_stride = ap.c_out,
        }, false);
        const s2 = try f2.commit();
        try target.compute.wait(s2, timeout_ms);
    }
    const at1 = try vmstat.nowNs();
    const ane_us = (@as(f64, @floatFromInt(at1 - at0)) / @as(f64, @floatFromInt(iters))) / 1000.0;

    // 2. Measure in-command-buffer GPU `encodeLinearGemv(o_proj)`.
    const dispatches_per_frame: u32 = 8;
    for (0..warmups) |_| {
        var frame = try target.compute.beginFrame(timeout_ms);
        errdefer frame.abort();
        for (0..dispatches_per_frame) |d| {
            try target.encodeLinearGemv(&frame, o_proj, in_bind, out_bind, d > 0);
        }
        const sig = try frame.commit();
        try target.compute.wait(sig, timeout_ms);
    }
    const gt0 = try vmstat.nowNs();
    for (0..iters) |_| {
        var frame = try target.compute.beginFrame(timeout_ms);
        errdefer frame.abort();
        for (0..dispatches_per_frame) |d| {
            try target.encodeLinearGemv(&frame, o_proj, in_bind, out_bind, d > 0);
        }
        const sig = try frame.commit();
        try target.compute.wait(sig, timeout_ms);
    }
    const gt1 = try vmstat.nowNs();
    const gpu_us = (@as(f64, @floatFromInt(gt1 - gt0)) / @as(f64, @floatFromInt(@as(u64, iters) * dispatches_per_frame))) / 1000.0;

    const winner: Placement = if (ane_us < gpu_us) .ane else .gpu;
    return .{ .ane_us = ane_us, .gpu_us = gpu_us, .winner = winner };
}

fn loadInternal(
    allocator: std.mem.Allocator,
    target: *qwen3.Qwen3Model,
    source: AmfSource,
    config: DraftConfig,
    options: DraftLoadOptions,
) DraftLoadError!DraftHead {
    try config.validate();
    if (options.max_seq_len == 0 or options.chunk_tokens == 0 or
        options.max_tree_slots == 0 or options.max_tree_slots > attention.MAX_TREE_NODES)
    {
        return error.InvalidConfig;
    }
    if (config.vocab_size != target.config.vocab_size or config.target_hidden_size != target.config.hidden_size) {
        return error.TargetIncompatible;
    }

    const parsed = source.parsed();
    const num_layers: usize = config.num_hidden_layers;
    const h = config.hidden_size;
    const h_tgt = config.target_hidden_size;
    const q_dim = config.qDim();
    const kv_dim = config.kvDim();
    const max_seq_len = @min(options.max_seq_len, config.max_position_embeddings);
    const num_chunks: u32 = (max_seq_len + options.chunk_tokens - 1) / options.chunk_tokens;

    // --- 1. Resolve & validate all required draft tensors & compute weight arena bytes ---
    var weights_bytes: u64 = 0;

    const d2t_entry: ?*const amf.TensorEntry = if (config.format == .eagle3) blk: {
        const e = try findRequired(parsed, "d2t");
        if ((try e.dataType()) != .u32) return error.UnexpectedTensorDType;
        if (e.rank != 1 or e.shape[0] != config.draft_vocab_size) return error.UnexpectedTensorShape;
        weights_bytes += e.blob_bytes;
        break :blk e;
    } else null;

    const embed_entry = findLinearOpt(parsed, "embed_tokens");
    if (embed_entry) |ee| {
        _ = try validateLinear2D(ee, config.vocab_size, h);
        weights_bytes += ee.blob_bytes;
    } else if (h != h_tgt) {
        return error.MissingTensor;
    }

    const lm_head_entry = findLinearOpt(parsed, "lm_head");
    if (lm_head_entry) |le| {
        _ = try validateLinear2D(le, config.draft_vocab_size, h);
        weights_bytes += le.blob_bytes;
    } else if (config.format == .eagle3 or h != h_tgt) {
        return error.MissingTensor;
    }

    const fc_entry: ?*const amf.TensorEntry = switch (config.format) {
        .eagle3 => blk: {
            const e = try findLinearReq(parsed, "fc");
            _ = try validateLinear2D(e, h, 3 * h_tgt);
            weights_bytes += e.blob_bytes;
            break :blk e;
        },
        .eagle2 => blk: {
            const e = try findLinearReq(parsed, "fc");
            _ = try validateLinear2D(e, h, 2 * h);
            weights_bytes += e.blob_bytes;
            break :blk e;
        },
        .mtp => null,
        .l1_only, .small_draft => unreachable,
    };

    const enorm_entry: ?*const amf.TensorEntry = if (config.format == .mtp) blk: {
        const e = try findRequired(parsed, "enorm.weight");
        _ = try validateNorm1D(e, h);
        weights_bytes += e.blob_bytes;
        break :blk e;
    } else null;
    const hnorm_entry: ?*const amf.TensorEntry = if (config.format == .mtp) blk: {
        const e = try findRequired(parsed, "hnorm.weight");
        _ = try validateNorm1D(e, h);
        weights_bytes += e.blob_bytes;
        break :blk e;
    } else null;
    const eh_proj_entry: ?*const amf.TensorEntry = if (config.format == .mtp) blk: {
        const e = try findLinearReq(parsed, "eh_proj");
        _ = try validateLinear2D(e, h, 2 * h);
        weights_bytes += e.blob_bytes;
        break :blk e;
    } else null;

    const final_norm_entry = if (config.format == .mtp)
        parsed.find("shared_head.norm.weight") orelse try findRequired(parsed, "norm.weight")
    else
        try findRequired(parsed, "norm.weight");
    const final_norm_dt = try validateNorm1D(final_norm_entry, h);
    weights_bytes += final_norm_entry.blob_bytes;

    var name_buf: [64]u8 = undefined;
    for (0..num_layers) |l| {
        const qkv_in: u32 = if (config.format == .eagle3 and l == 0) 2 * h else h;
        const in_ln = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable);
        _ = try validateNorm1D(in_ln, h);
        weights_bytes += in_ln.blob_bytes;

        if (config.format == .eagle3 and l == 0) {
            const hn = try findRequired(parsed, "layers.0.hidden_norm.weight");
            _ = try validateNorm1D(hn, h);
            weights_bytes += hn.blob_bytes;
        }
        const qe = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(qe, q_dim, qkv_in);
        weights_bytes += qe.blob_bytes;

        const ke = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(ke, kv_dim, qkv_in);
        weights_bytes += ke.blob_bytes;

        const ve = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(ve, kv_dim, qkv_in);
        weights_bytes += ve.blob_bytes;

        const oe = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(oe, h, q_dim);
        weights_bytes += oe.blob_bytes;

        const qn = parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable);
        const kn = parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable);
        if ((qn != null) != (kn != null)) return error.MissingTensor;
        if (qn) |qne| {
            const qdt = try validateNorm1D(qne, config.head_dim);
            const kdt = try validateNorm1D(kn.?, config.head_dim);
            if (qdt != kdt) return error.UnexpectedTensorDType;
            weights_bytes += qne.blob_bytes + kn.?.blob_bytes;
        }

        const pln = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable);
        _ = try validateNorm1D(pln, h);
        weights_bytes += pln.blob_bytes;

        const gp = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(gp, config.intermediate_size, h);
        weights_bytes += gp.blob_bytes;

        const up = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(up, config.intermediate_size, h);
        weights_bytes += up.blob_bytes;

        const dp = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(dp, h, config.intermediate_size);
        weights_bytes += dp.blob_bytes;
    }

    const weights_arena_bytes = try page.alignUp(@max(@as(usize, @intCast(weights_bytes)), page.APPLE_PAGE_SIZE));

    // --- 2. Compute draft scratch & KV tile footprint and enforce remaining wired budget ---
    const m_cap: usize = options.max_tree_slots;
    const kv_tbl_per_layer = amf.align64(@as(usize, num_chunks) * @sizeOf(attention.KvTilePair));
    const tree_kv_layer_stride = amf.align64(m_cap * @as(usize, kv_dim) * @sizeOf(f16));
    const gqa_tree_bytes = amf.align64(attention.gqaTreeScratchLayout(
        options.max_tree_slots,
        config.num_attention_heads,
        config.head_dim,
        attention.MAX_SPLITS,
    ).total_bytes);

    const scratch_raw: usize =
        amf.align64(m_cap * @sizeOf(u32)) * 3 + // token_ids, parent_slots, positions
        amf.align64(m_cap * @sizeOf(u64)) + // ancestor_masks
        amf.align64((1 + m_cap) * @as(usize, h) * @sizeOf(f16)) + // hidden_pool
        amf.align64(m_cap * @as(usize, h) * @sizeOf(f16)) * 7 + // h_in, emb, e_norm, h_norm, x, x_norm, proj_out
        amf.align64(m_cap * 2 * @as(usize, h) * @sizeOf(f16)) + // x_cat
        amf.align64(m_cap * @as(usize, q_dim) * @sizeOf(f16)) * 2 + // q_pool, attn_pool
        amf.align64(m_cap * @as(usize, kv_dim) * @sizeOf(f16)) * 2 + // k, v
        amf.align64(m_cap * @as(usize, config.intermediate_size) * @sizeOf(f16)) * 3 + // mlp gate/up/act
        amf.align64(m_cap * @as(usize, config.draft_vocab_size) * @sizeOf(f16)) + // logits
        amf.align64(m_cap * MAX_DRAFT_TOP_K * @sizeOf(moe.TopKCandidate)) + // topk
        gqa_tree_bytes +
        num_layers * kv_tbl_per_layer +
        2 * num_layers * tree_kv_layer_stride +
        8 * page.APPLE_PAGE_SIZE;
    const scratch_arena_bytes = try page.alignUp(scratch_raw);

    const kv_tile_rows: u32 = config.num_key_value_heads * options.chunk_tokens;
    const kv_tile_alloc_bytes: u64 = try page.alignUp(@as(usize, kv_tile_rows) * config.head_dim * @sizeOf(f16));
    const total_kv_tiles: usize = num_layers * @as(usize, num_chunks) * 2;
    const total_kv_bytes: u64 = @as(u64, total_kv_tiles) * kv_tile_alloc_bytes;
    const total_wired_bytes: u64 = @as(u64, weights_arena_bytes) + scratch_arena_bytes + total_kv_bytes;

    if (target.total_wired_bytes + total_wired_bytes > target.wired_budget_bytes) {
        return error.BudgetExceeded;
    }
    const rem_limit = target.wired_budget_bytes - target.total_wired_bytes;
    const draft_budget = budget_mod.Budget.compute(.{
        .memory_bytes = target.wired_budget_bytes + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = rem_limit,
        .global_user_wire_limit = rem_limit,
        .gpu_working_set_bytes = rem_limit,
        .configured_cap = rem_limit,
        .headroom_bytes = 0,
    });

    // --- 3. Allocate wired arenas, zero-copy MTLBuffers, and IOSurface KV tiles ---
    var weights_arena = try arena_mod.WiredArena.init(weights_arena_bytes, draft_budget);
    errdefer weights_arena.deinit();
    const weights_buf = try metal.Buffer.wrap(target.device, weights_arena.bytes());
    errdefer weights_buf.close();

    var scratch_arena = try arena_mod.WiredArena.init(scratch_arena_bytes, draft_budget);
    errdefer scratch_arena.deinit();
    const scratch_buf = try metal.Buffer.wrap(target.device, scratch_arena.bytes());
    errdefer scratch_buf.close();

    const kv_tiles = try allocator.alloc(metal.KvTile, total_kv_tiles);
    var tiles_inited: usize = 0;
    errdefer {
        for (0..tiles_inited) |i| kv_tiles[i].close();
        allocator.free(kv_tiles);
    }
    for (0..total_kv_tiles) |i| {
        kv_tiles[i] = try metal.KvTile.init(target.device, config.head_dim, kv_tile_rows, .{});
        tiles_inited += 1;
    }

    try target.compute.makeResident(&.{ weights_buf, scratch_buf });
    errdefer target.compute.removeResident(&.{ weights_buf, scratch_buf }) catch |err| {
        std.debug.panic("removeResident failed in DraftHead errdefer: {}", .{err});
    };
    try target.compute.makeTilesResident(kv_tiles);
    errdefer target.compute.removeTilesResident(kv_tiles) catch |err| {
        std.debug.panic("removeTilesResident failed in DraftHead errdefer: {}", .{err});
    };

    // --- 4. Carve scratch buffers ---
    const tok_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u32), 64);
    const par_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u32), 64);
    const pos_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u32), 64);
    const msk_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u64), 64);
    const hp_raw = try scratch_arena.allocAligned((1 + m_cap) * @as(usize, h) * @sizeOf(f16), 64);
    const hin_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const emb_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const en_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const hn_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const xcat_raw = try scratch_arena.allocAligned(m_cap * 2 * @as(usize, h) * @sizeOf(f16), 64);
    const x_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const xn_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const qp_raw = try scratch_arena.allocAligned(m_cap * @as(usize, q_dim) * @sizeOf(f16), 64);
    const k_raw = try scratch_arena.allocAligned(m_cap * @as(usize, kv_dim) * @sizeOf(f16), 64);
    const v_raw = try scratch_arena.allocAligned(m_cap * @as(usize, kv_dim) * @sizeOf(f16), 64);
    const ap_raw = try scratch_arena.allocAligned(m_cap * @as(usize, q_dim) * @sizeOf(f16), 64);
    const po_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const mg_raw = try scratch_arena.allocAligned(m_cap * @as(usize, config.intermediate_size) * @sizeOf(f16), 64);
    const mu_raw = try scratch_arena.allocAligned(m_cap * @as(usize, config.intermediate_size) * @sizeOf(f16), 64);
    const ma_raw = try scratch_arena.allocAligned(m_cap * @as(usize, config.intermediate_size) * @sizeOf(f16), 64);
    const log_raw = try scratch_arena.allocAligned(m_cap * @as(usize, config.draft_vocab_size) * @sizeOf(f16), 64);
    const tk_out_raw = try scratch_arena.allocAligned(m_cap * MAX_DRAFT_TOP_K * @sizeOf(moe.TopKCandidate), 64);
    const gqa_raw = try scratch_arena.allocAligned(gqa_tree_bytes, 64);
    const tree_k_raw = try scratch_arena.allocAligned(num_layers * tree_kv_layer_stride, 64);
    const tree_v_raw = try scratch_arena.allocAligned(num_layers * tree_kv_layer_stride, 64);
    @memset(tok_raw, 0);
    @memset(par_raw, 0);
    @memset(pos_raw, 0);
    @memset(msk_raw, 0);
    @memset(hp_raw, 0);
    @memset(tk_out_raw, 0);

    // --- 5. Load global draft weights ---
    var d2t_host: []const u32 = &.{};
    if (d2t_entry) |de| {
        const blob = try loadBlobIntoArena(source, de, &weights_arena, weights_buf, options.verify_crcs);
        const ids: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, blob.slice[0..@intCast(de.data_bytes)]));
        for (ids) |tid| {
            if (tid >= config.vocab_size) return error.InvalidD2tEntry;
        }
        d2t_host = ids;
    }

    const embed_lin: qwen3.LinearBinding = if (embed_entry) |ee|
        (try loadBlobIntoArena(source, ee, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(ee, config.vocab_size, h))
    else
        target.embed;

    const lm_head_lin: qwen3.LinearBinding = if (lm_head_entry) |le|
        (try loadBlobIntoArena(source, le, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(le, config.draft_vocab_size, h))
    else
        target.lm_head;

    const fc_lin: ?qwen3.LinearBinding = if (fc_entry) |fe| blk: {
        const expected_k: u32 = if (config.format == .eagle3) 3 * h_tgt else 2 * h;
        break :blk (try loadBlobIntoArena(source, fe, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(fe, h, expected_k));
    } else null;

    const enorm_b: ?qwen3.NormBinding = if (enorm_entry) |ee|
        (try loadBlobIntoArena(source, ee, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(ee, h))
    else
        null;
    const hnorm_b: ?qwen3.NormBinding = if (hnorm_entry) |he|
        (try loadBlobIntoArena(source, he, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(he, h))
    else
        null;
    const eh_proj_b: ?qwen3.LinearBinding = if (eh_proj_entry) |pe|
        (try loadBlobIntoArena(source, pe, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(pe, h, 2 * h))
    else
        null;

    const final_norm_b = (try loadBlobIntoArena(source, final_norm_entry, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(final_norm_dt);

    // --- 6. Load draft layer weights & build bindless KV tile tables ---
    const layers = try allocator.alloc(DraftLayerWeights, num_layers);
    errdefer allocator.free(layers);

    var l0_o_proj_blob: LoadedBlob = undefined;
    var l0_o_proj_dtype: quant.WeightDType = .bf16;

    for (0..num_layers) |l| {
        const qkv_in: u32 = if (config.format == .eagle3 and l == 0) 2 * h else h;
        const kv_tbl_raw = try scratch_arena.allocAligned(@as(usize, num_chunks) * @sizeOf(attention.KvTilePair), 64);
        const kv_tbl: []attention.KvTilePair = @alignCast(std.mem.bytesAsSlice(attention.KvTilePair, kv_tbl_raw));
        for (0..num_chunks) |c_idx| {
            const tbase = (l * @as(usize, num_chunks) + c_idx) * 2;
            kv_tbl[c_idx] = attention.KvTilePair.fromTiles(kv_tiles[tbase], kv_tiles[tbase + 1], .rgba16f);
        }

        const in_ln_e = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable);
        const in_ln_b = (try loadBlobIntoArena(source, in_ln_e, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(in_ln_e, h));

        const hid_ln_b: ?qwen3.NormBinding = if (config.format == .eagle3 and l == 0) blk: {
            const hn_e = try findRequired(parsed, "layers.0.hidden_norm.weight");
            break :blk (try loadBlobIntoArena(source, hn_e, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(hn_e, h));
        } else null;

        const q_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable);
        const q_b = (try loadBlobIntoArena(source, q_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(q_e, q_dim, qkv_in));

        const k_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable);
        const k_b = (try loadBlobIntoArena(source, k_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(k_e, kv_dim, qkv_in));

        const v_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable);
        const v_b = (try loadBlobIntoArena(source, v_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(v_e, kv_dim, qkv_in));

        const o_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable);
        const o_dt = try validateLinear2D(o_e, h, q_dim);
        const o_blob = try loadBlobIntoArena(source, o_e, &weights_arena, weights_buf, options.verify_crcs);
        const o_b = o_blob.asLinearBinding(o_dt);
        if (l == 0) {
            l0_o_proj_blob = o_blob;
            l0_o_proj_dtype = o_dt;
        }

        var q_norm_bind: ?metal.BufferBinding = null;
        var k_norm_bind: ?metal.BufferBinding = null;
        var qk_norm_dtype: quant.FloatDType = .f16;
        if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable)) |qn_e| {
            qk_norm_dtype = try validateNorm1D(qn_e, config.head_dim);
            q_norm_bind = (try loadBlobIntoArena(source, qn_e, &weights_arena, weights_buf, options.verify_crcs)).asBinding();
        }
        if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable)) |kn_e| {
            k_norm_bind = (try loadBlobIntoArena(source, kn_e, &weights_arena, weights_buf, options.verify_crcs)).asBinding();
        }

        const pln_e = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable);
        const pln_b = (try loadBlobIntoArena(source, pln_e, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(pln_e, h));

        const gp_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable);
        const gp_b = (try loadBlobIntoArena(source, gp_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(gp_e, config.intermediate_size, h));

        const up_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}) catch unreachable);
        const up_b = (try loadBlobIntoArena(source, up_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(up_e, config.intermediate_size, h));

        const dp_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}) catch unreachable);
        const dp_b = (try loadBlobIntoArena(source, dp_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(dp_e, h, config.intermediate_size));

        layers[l] = .{
            .input_layernorm = in_ln_b,
            .hidden_norm = hid_ln_b,
            .q_proj = q_b,
            .k_proj = k_b,
            .v_proj = v_b,
            .o_proj = o_b,
            .q_norm = q_norm_bind,
            .k_norm = k_norm_bind,
            .qk_norm_dtype = qk_norm_dtype,
            .post_attention_layernorm = pln_b,
            .gate_proj = gp_b,
            .up_proj = up_b,
            .down_proj = dp_b,
            .kv_tiles_binding = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(kv_tbl_raw.ptr) },
        };
    }

    // --- 7. D5 Benchmark-Gated Placement (`auto` vs explicit `gpu` / `ane`, ADR-0008) ---
    var effective_placement: Placement = .gpu;
    var ane_proj: ?AneProjection = null;
    errdefer if (ane_proj) |*ap| ap.close(&target.compute);
    var gate_ane_us: f64 = 0.0;
    var gate_gpu_us: f64 = 0.0;

    const can_ane = (l0_o_proj_dtype == .f16 or l0_o_proj_dtype == .bf16) and
        (q_dim % metal.KV_TILE_ROW_ALIGN_F16 == 0) and
        (h % metal.KV_TILE_ROW_ALIGN_F16 == 0);
    if (options.placement == .ane and can_ane) {
        const wf16 = try convertWeightBlobToF16(allocator, l0_o_proj_blob, l0_o_proj_dtype);
        defer allocator.free(wf16);
        ane_proj = try AneProjection.init(allocator, target.device, &target.compute, q_dim, h, ANE_MAX_BATCH_ROWS, wf16);
        effective_placement = .ane;
    } else if (options.placement == .auto and can_ane) {
        const wf16 = try convertWeightBlobToF16(allocator, l0_o_proj_blob, l0_o_proj_dtype);
        defer allocator.free(wf16);
        var probe_ap = try AneProjection.init(allocator, target.device, &target.compute, q_dim, h, 1, wf16);
        const cal = try calibrateD5Placement(
            target,
            &probe_ap,
            layers[0].o_proj,
            .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(ap_raw.ptr) },
            .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(po_raw.ptr) },
            options.timeout_ms,
        );
        gate_ane_us = cal.ane_us;
        gate_gpu_us = cal.gpu_us;
        probe_ap.close(&target.compute);
        effective_placement = cal.winner;
        if (effective_placement == .ane) {
            ane_proj = try AneProjection.init(allocator, target.device, &target.compute, q_dim, h, ANE_MAX_BATCH_ROWS, wf16);
        }
    }

    return .{
        .config = config,
        .requested_placement = options.placement,
        .effective_placement = effective_placement,
        .placement_gate_ane_us = gate_ane_us,
        .placement_gate_gpu_us = gate_gpu_us,
        .max_seq_len = max_seq_len,
        .chunk_tokens = options.chunk_tokens,
        .num_chunks = num_chunks,
        .max_tree_slots = options.max_tree_slots,
        .seq_len = 0,
        .timeout_ms = options.timeout_ms,
        .total_wired_bytes = total_wired_bytes,
        .weights_arena = weights_arena,
        .weights_buf = weights_buf,
        .scratch_arena = scratch_arena,
        .scratch_buf = scratch_buf,
        .kv_tiles = kv_tiles,
        .layers = layers,
        .d2t_host = d2t_host,
        .embed = embed_lin,
        .lm_head = lm_head_lin,
        .fc = fc_lin,
        .enorm = enorm_b,
        .hnorm = hnorm_b,
        .eh_proj = eh_proj_b,
        .final_norm = final_norm_b,
        .ane_proj = ane_proj,
        .token_ids_host = @alignCast(std.mem.bytesAsSlice(u32, tok_raw)),
        .parent_slots_host = @alignCast(std.mem.bytesAsSlice(u32, par_raw)),
        .positions_host = @alignCast(std.mem.bytesAsSlice(u32, pos_raw)),
        .ancestor_masks_host = @alignCast(std.mem.bytesAsSlice(u64, msk_raw)),
        .topk_host = @alignCast(std.mem.bytesAsSlice(moe.TopKCandidate, tk_out_raw)),
        .hidden_pool_host = @alignCast(std.mem.bytesAsSlice(f16, hp_raw)),
        .token_ids_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tok_raw.ptr) },
        .parent_slots_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(par_raw.ptr) },
        .positions_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(pos_raw.ptr) },
        .ancestor_masks_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(msk_raw.ptr) },
        .hidden_pool_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(hp_raw.ptr) },
        .h_in_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(hin_raw.ptr) },
        .emb_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(emb_raw.ptr) },
        .e_norm_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(en_raw.ptr) },
        .h_norm_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(hn_raw.ptr) },
        .x_cat_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(xcat_raw.ptr) },
        .x_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_raw.ptr) },
        .x_norm_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(xn_raw.ptr) },
        .q_pool_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(qp_raw.ptr) },
        .k_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(k_raw.ptr) },
        .v_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(v_raw.ptr) },
        .attn_pool_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(ap_raw.ptr) },
        .proj_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(po_raw.ptr) },
        .mlp_gate_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(mg_raw.ptr) },
        .mlp_up_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(mu_raw.ptr) },
        .mlp_act_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(ma_raw.ptr) },
        .logits_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(log_raw.ptr) },
        .topk_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tk_out_raw.ptr) },
        .gqa_tree_scratch_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gqa_raw.ptr) },
        .tree_k_base_off = scratch_arena.offsetOf(tree_k_raw.ptr),
        .tree_v_base_off = scratch_arena.offsetOf(tree_v_raw.ptr),
        .tree_kv_layer_stride = tree_kv_layer_stride,
        .root_top_k_count = 0,
        .root_candidates = undefined,
    };
}

// ---------------------------------------------------------------------------
// Synthetic draft `.amf` builder & tests
// ---------------------------------------------------------------------------

const SynthRec = struct {
    name: []const u8,
    kind: amf.TensorKind,
    dtype: amf.DType,
    shape: [4]u32,
    rank: u8,
    data: []u8,
};

fn nextRng(state: *u64) u64 {
    var x = state.*;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state.* = x;
    return x;
}

fn fillBf16(allocator: std.mem.Allocator, elems: usize, amp: f32, centered_one: bool, seed: u64) ![]u8 {
    var rng = seed ^ 0xA076_1D64_78BD_642F;
    const raw = try allocator.alloc(u8, elems * 2);
    const words: []u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
    for (words) |*w| {
        const u = @as(f32, @floatFromInt(nextRng(&rng) & 0xFFFF)) / 65535.0;
        const f = if (centered_one) (0.95 + 0.1 * u) else ((u * 2.0 - 1.0) * amp);
        w.* = quant.f32ToFloatBits(.bf16, f);
    }
    return raw;
}

/// Builds a deterministic in-memory 16 KiB-aligned `.amf` draft head image for
/// tests (`eagle3`, `eagle2`, or `mtp`).
pub fn buildSyntheticDraftAmfImage(allocator: std.mem.Allocator, cfg: DraftConfig, seed: u64) ![]u8 {
    var recs: std.ArrayList(SynthRec) = .empty;
    defer {
        for (recs.items) |r| {
            allocator.free(r.name);
            allocator.free(r.data);
        }
        recs.deinit(allocator);
    }

    const h = cfg.hidden_size;
    const h_tgt = cfg.target_hidden_size;
    const q_dim = cfg.qDim();
    const kv_dim = cfg.kvDim();
    const inter = cfg.intermediate_size;
    const inv_h = 0.5 / @sqrt(@as(f32, @floatFromInt(h)));
    const inv_q = 0.5 / @sqrt(@as(f32, @floatFromInt(q_dim)));
    const inv_i = 0.5 / @sqrt(@as(f32, @floatFromInt(inter)));
    var tidx: u64 = 1;

    const add2d = struct {
        fn call(alloc: std.mem.Allocator, list: *std.ArrayList(SynthRec), name: []const u8, kind: amf.TensorKind, n: u32, k: u32, amp: f32, s: u64) !void {
            const nm = try alloc.dupe(u8, name);
            errdefer alloc.free(nm);
            const data = try fillBf16(alloc, @as(usize, n) * k, amp, false, s);
            errdefer alloc.free(data);
            try list.append(alloc, .{ .name = nm, .kind = kind, .dtype = .bf16, .shape = .{ n, k, 0, 0 }, .rank = 2, .data = data });
        }
    }.call;

    const add1d = struct {
        fn call(alloc: std.mem.Allocator, list: *std.ArrayList(SynthRec), name: []const u8, dim: u32, s: u64) !void {
            const nm = try alloc.dupe(u8, name);
            errdefer alloc.free(nm);
            const data = try fillBf16(alloc, dim, 0.05, true, s);
            errdefer alloc.free(data);
            try list.append(alloc, .{ .name = nm, .kind = .norm, .dtype = .bf16, .shape = .{ dim, 0, 0, 0 }, .rank = 1, .data = data });
        }
    }.call;

    if (cfg.format == .eagle3) {
        const d2t_bytes = try allocator.alloc(u8, @as(usize, cfg.draft_vocab_size) * @sizeOf(u32));
        errdefer allocator.free(d2t_bytes);
        const ids: []u32 = @alignCast(std.mem.bytesAsSlice(u32, d2t_bytes));
        for (ids, 0..) |*id, i| id.* = @intCast((i * cfg.vocab_size) / cfg.draft_vocab_size);
        try recs.append(allocator, .{
            .name = try allocator.dupe(u8, "d2t"),
            .kind = .dense,
            .dtype = .u32,
            .shape = .{ cfg.draft_vocab_size, 0, 0, 0 },
            .rank = 1,
            .data = d2t_bytes,
        });
    }

    if (!cfg.share_embed_tokens) {
        try add2d(allocator, &recs, "embed_tokens.weight", .embedding, cfg.vocab_size, h, 0.2, seed + tidx);
        tidx += 1;
    }

    switch (cfg.format) {
        .eagle3 => {
            try add2d(allocator, &recs, "fc.weight", .dense, h, 3 * h_tgt, 0.5 / @sqrt(@as(f32, @floatFromInt(3 * h_tgt))), seed + tidx);
            tidx += 1;
        },
        .eagle2 => {
            try add2d(allocator, &recs, "fc.weight", .dense, h, 2 * h, 0.5 / @sqrt(@as(f32, @floatFromInt(2 * h))), seed + tidx);
            tidx += 1;
        },
        .mtp => {
            try add1d(allocator, &recs, "enorm.weight", h, seed + tidx);
            tidx += 1;
            try add1d(allocator, &recs, "hnorm.weight", h, seed + tidx);
            tidx += 1;
            try add2d(allocator, &recs, "eh_proj.weight", .dense, h, 2 * h, 0.5 / @sqrt(@as(f32, @floatFromInt(2 * h))), seed + tidx);
            tidx += 1;
        },
        .l1_only, .small_draft => unreachable,
    }

    var nbuf: [64]u8 = undefined;
    for (0..cfg.num_hidden_layers) |l| {
        const qkv_in: u32 = if (cfg.format == .eagle3 and l == 0) 2 * h else h;
        const inv_qkv = 0.5 / @sqrt(@as(f32, @floatFromInt(qkv_in)));
        try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, h, seed + tidx);
        tidx += 1;
        if (cfg.format == .eagle3 and l == 0) {
            try add1d(allocator, &recs, "layers.0.hidden_norm.weight", h, seed + tidx);
            tidx += 1;
        }
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj.weight", .{l}) catch unreachable, .dense, q_dim, qkv_in, inv_qkv, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj.weight", .{l}) catch unreachable, .dense, kv_dim, qkv_in, inv_qkv, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj.weight", .{l}) catch unreachable, .dense, kv_dim, qkv_in, inv_qkv, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj.weight", .{l}) catch unreachable, .dense, h, q_dim, inv_q, seed + tidx);
        tidx += 1;
        if (cfg.format == .mtp) {
            try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable, cfg.head_dim, seed + tidx);
            tidx += 1;
            try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable, cfg.head_dim, seed + tidx);
            tidx += 1;
        }
        try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, h, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj.weight", .{l}) catch unreachable, .dense, inter, h, inv_h, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj.weight", .{l}) catch unreachable, .dense, inter, h, inv_h, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj.weight", .{l}) catch unreachable, .dense, h, inter, inv_i, seed + tidx);
        tidx += 1;
    }

    const fn_name = if (cfg.format == .mtp) "shared_head.norm.weight" else "norm.weight";
    try add1d(allocator, &recs, fn_name, h, seed + tidx);
    tidx += 1;
    if (!cfg.share_lm_head) {
        try add2d(allocator, &recs, "lm_head.weight", .lm_head, cfg.draft_vocab_size, h, inv_h, seed + tidx);
    }

    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_raw_len: usize = recs.items.len * @sizeOf(amf.TensorEntry);
    const dir_bytes: usize = try page.alignUp(dir_raw_len);
    var total_size: usize = dir_offset + dir_bytes;
    for (recs.items) |r| total_size += try page.alignUp(r.data.len);

    const image = try allocator.alloc(u8, total_size);
    errdefer allocator.free(image);
    @memset(image, 0);

    const entries = try allocator.alloc(amf.TensorEntry, recs.items.len);
    defer allocator.free(entries);

    var cursor: usize = dir_offset + dir_bytes;
    for (recs.items, 0..) |r, i| {
        const blob_bytes = try page.alignUp(r.data.len);
        @memcpy(image[cursor .. cursor + r.data.len], r.data);
        const crc = std.hash.Crc32.hash(image[cursor .. cursor + blob_bytes]);
        var e = amf.TensorEntry.zeroed(r.name, r.kind, r.dtype);
        e.rank = r.rank;
        e.shape = r.shape;
        e.blob_offset = cursor;
        e.blob_bytes = blob_bytes;
        e.data_bytes = r.data.len;
        e.blob_crc32 = crc;
        entries[i] = e;
        cursor += blob_bytes;
    }

    const dir_slice = std.mem.sliceAsBytes(entries);
    @memcpy(image[dir_offset .. dir_offset + dir_slice.len], dir_slice);
    const dir_crc = std.hash.Crc32.hash(dir_slice);

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
    const mid = "draft-test";
    @memcpy(hdr.model_id[0..mid.len], mid);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return image;
}

test "DraftConfig.fromJson parses real RedHatAI EAGLE-3 plus EAGLE-2 and MTP configs" {
    const gpa = std.testing.allocator;
    const real_eagle3 =
        \\{
        \\  "architectures": ["Eagle3DraftModel"],
        \\  "draft_vocab_size": 64000,
        \\  "eagle_aux_hidden_state_layer_ids": null,
        \\  "norm_before_residual": true,
        \\  "speculators_model_type": "eagle3",
        \\  "target_hidden_size": null,
        \\  "transformer_layer_config": {
        \\    "head_dim": 128,
        \\    "hidden_size": 2048,
        \\    "intermediate_size": 6144,
        \\    "max_position_embeddings": 40960,
        \\    "model_type": "llama",
        \\    "num_attention_heads": 32,
        \\    "num_hidden_layers": 1,
        \\    "num_key_value_heads": 4,
        \\    "rms_norm_eps": 1e-06,
        \\    "rope_theta": 10000.0,
        \\    "vocab_size": 151936
        \\  }
        \\}
    ;
    const e3 = try DraftConfig.fromJson(gpa, real_eagle3);
    try std.testing.expectEqual(DraftMode.eagle3, e3.format);
    try std.testing.expectEqual(@as(u32, 151936), e3.vocab_size);
    try std.testing.expectEqual(@as(u32, 64000), e3.draft_vocab_size);
    try std.testing.expectEqual(@as(u32, 2048), e3.hidden_size);
    try std.testing.expectEqual(@as(u32, 2048), e3.target_hidden_size);
    try std.testing.expectEqual(@as(f32, 10_000.0), e3.rope_theta);
    try std.testing.expect(e3.norm_before_residual);
    try std.testing.expectEqual(@as(?[3]u32, null), e3.aux_layer_ids);

    const e2_json =
        \\{
        \\  "architectures": ["EagleLlamaForCausalLM"],
        \\  "model_type": "eagle",
        \\  "draft_format": "eagle2",
        \\  "vocab_size": 512,
        \\  "hidden_size": 128,
        \\  "num_hidden_layers": 1,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 64,
        \\  "intermediate_size": 256,
        \\  "rope_theta": 1000000.0
        \\}
    ;
    const e2 = try DraftConfig.fromJson(gpa, e2_json);
    try std.testing.expectEqual(DraftMode.eagle2, e2.format);
    try std.testing.expectEqual(@as(u32, 512), e2.draft_vocab_size);

    const mtp_json =
        \\{
        \\  "architectures": ["MtpDraftModel"],
        \\  "model_type": "mtp",
        \\  "draft_format": "mtp",
        \\  "vocab_size": 512,
        \\  "hidden_size": 128,
        \\  "num_hidden_layers": 1,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 64,
        \\  "intermediate_size": 256,
        \\  "share_embed_tokens": true,
        \\  "share_lm_head": true
        \\}
    ;
    const mtp = try DraftConfig.fromJson(gpa, mtp_json);
    try std.testing.expectEqual(DraftMode.mtp, mtp.format);
    try std.testing.expect(mtp.share_embed_tokens);
    try std.testing.expect(mtp.share_lm_head);
}

test "DraftHead executes EAGLE-3, EAGLE-2, and MTP on GPU, ANE zero-copy, and D5 auto gate" {
    const gpa = std.testing.allocator;
    const tgt_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = null,
        .default_float_dtype = .bf16,
        .vocab_size = 256,
        .hidden_size = 128,
        .num_hidden_layers = 4,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .intermediate_size = 256,
        .tie_word_embeddings = true,
    };

    // Build tiny synthetic target model directly via `qwen3` tests' compatible layout.
    // Use `bench.writeSyntheticQwen3AmfFile` or minimal inline builder: easiest via `synth`? No,
    // keep tests self-contained in Zig using a tiny temporary file via `amf`!
    const draft_cfgs = [_]struct { cfg: DraftConfig, place: Placement }{
        .{
            .cfg = .{
                .format = .eagle3,
                .vocab_size = 256,
                .draft_vocab_size = 128,
                .hidden_size = 128,
                .target_hidden_size = 128,
                .num_hidden_layers = 1,
                .num_attention_heads = 4,
                .num_key_value_heads = 2,
                .head_dim = 64,
                .intermediate_size = 256,
            },
            .place = .auto,
        },
        .{
            .cfg = .{
                .format = .eagle3,
                .vocab_size = 256,
                .draft_vocab_size = 128,
                .hidden_size = 128,
                .target_hidden_size = 128,
                .num_hidden_layers = 1,
                .num_attention_heads = 4,
                .num_key_value_heads = 2,
                .head_dim = 64,
                .intermediate_size = 256,
            },
            .place = .ane,
        },
        .{
            .cfg = .{
                .format = .eagle2,
                .vocab_size = 256,
                .draft_vocab_size = 256,
                .hidden_size = 128,
                .target_hidden_size = 128,
                .num_hidden_layers = 1,
                .num_attention_heads = 4,
                .num_key_value_heads = 2,
                .head_dim = 64,
                .intermediate_size = 256,
            },
            .place = .gpu,
        },
        .{
            .cfg = .{
                .format = .mtp,
                .vocab_size = 256,
                .draft_vocab_size = 256,
                .hidden_size = 128,
                .target_hidden_size = 128,
                .num_hidden_layers = 1,
                .num_attention_heads = 4,
                .num_key_value_heads = 2,
                .head_dim = 64,
                .intermediate_size = 256,
                .share_embed_tokens = true,
                .share_lm_head = true,
            },
            .place = .gpu,
        },
    };

    const tgt_img = try buildSyntheticTargetBf16ForTest(gpa, tgt_cfg, 901);
    defer gpa.free(tgt_img);

    var target = try qwen3.Qwen3Model.loadFromBytes(gpa, tgt_img, tgt_cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 16,
        .max_tree_nodes = 16,
        .enable_feature_taps = true,
    });
    defer target.deinit(gpa);

    const next_tok = try target.stepDecode(11, 0);
    _ = try target.stepDecode(next_tok, 1);

    for (draft_cfgs, 0..) |tc, idx| {
        const d_img = try buildSyntheticDraftAmfImage(gpa, tc.cfg, 100 + idx);
        defer gpa.free(d_img);

        var head = try DraftHead.loadFromBytes(gpa, &target, d_img, tc.cfg, .{
            .placement = tc.place,
            .max_seq_len = 64,
            .chunk_tokens = 16,
            .max_tree_slots = 16,
        });
        defer head.deinit(gpa, &target);

        if (tc.place == .ane) {
            try std.testing.expectEqual(Placement.ane, head.effective_placement);
            try std.testing.expect(head.ane_proj != null);
            try std.testing.expect(head.ane_proj.?.zero_copy_ok);
        } else if (tc.place == .auto) {
            try std.testing.expect(head.placement_gate_ane_us > 0.0);
            try std.testing.expect(head.placement_gate_gpu_us > 0.0);
        }

        const feat_bind = if (tc.cfg.format == .eagle3) target.aux_concat_bind else target.x_bind;
        const sync_toks = [_]u32{next_tok};
        try head.syncAccepted(&target, &sync_toks, feat_bind, 0, 4);
        try std.testing.expectEqual(@as(u32, 1), head.seq_len);
        const root_cands = head.rootCandidates();
        try std.testing.expectEqual(@as(usize, 4), root_cands.len);
        var prob_sum: f32 = 0.0;
        for (root_cands) |c| {
            try std.testing.expect(c.token_id < tgt_cfg.vocab_size);
            try std.testing.expect(c.prob > 0.0 and c.prob <= 1.0);
            prob_sum += c.prob;
        }
        try std.testing.expect(prob_sum <= 1.01);

        // Expand 2 frontier children from root hidden state (slot 0 -> tree slots 0 and 1).
        const step_toks = [_]u32{ root_cands[0].token_id, root_cands[1].token_id };
        const parents = [_]u32{ 0, 0 };
        const pos_arr = [_]u32{ 1, 1 };
        const masks = [_]u64{ 0b01, 0b10 };
        var child_cands: [2 * 3]moe.TopKCandidate = undefined;
        try head.stepTreeBatch(&target, &step_toks, &parents, &pos_arr, &masks, 0, 3, &child_cands);
        // Committed draft seq_len must not advance during speculative tree expansion!
        try std.testing.expectEqual(@as(u32, 1), head.seq_len);
        for (child_cands) |c| {
            try std.testing.expect(c.token_id < tgt_cfg.vocab_size);
            try std.testing.expect(std.math.isFinite(c.prob) and c.prob > 0.0);
        }
    }
}

test "D4 SmallDraftModel loads standalone Qwen3 draft, prefills, seeds L1 chain, and expands tree" {
    const gpa = std.testing.allocator;
    const tgt_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = null,
        .default_float_dtype = .bf16,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .intermediate_size = 256,
        .tie_word_embeddings = true,
    };
    const draft_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = null,
        .default_float_dtype = .bf16,
        .vocab_size = 128,
        .hidden_size = 64,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .tie_word_embeddings = true,
    };
    const tgt_bytes = try buildSyntheticTargetBf16ForTest(gpa, tgt_cfg, 8100);
    defer gpa.free(tgt_bytes);
    const draft_bytes = try buildSyntheticTargetBf16ForTest(gpa, draft_cfg, 8200);
    defer gpa.free(draft_bytes);

    var target = try qwen3.Qwen3Model.loadFromBytes(gpa, tgt_bytes, tgt_cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 16,
        .max_tree_nodes = 16,
    });
    defer target.deinit(gpa);

    var sd = try SmallDraftModel.loadFromBytes(gpa, &target, draft_bytes, draft_cfg, .{
        .placement = .auto,
        .max_seq_len = 64,
        .chunk_tokens = 16,
        .max_tree_slots = 16,
    });
    defer sd.deinit(gpa);

    try std.testing.expect(sd.totalWiredBytes() > 0);
    try std.testing.expect(sd.placement_gate_ane_us > 0.0 and sd.placement_gate_gpu_us > 0.0);

    // Prefill 4 prompt tokens + first_tok=11 -> seq_len = 5, rootCandidates populated.
    const prompt = [_]u32{ 3, 5, 7, 9 };
    try sd.prefillAndSeedRoot(&prompt, 11, 3);
    try std.testing.expectEqual(@as(u32, 5), sd.seqLen());
    const rc = sd.rootCandidates();
    try std.testing.expectEqual(@as(usize, 3), rc.len);
    for (rc) |c| {
        try std.testing.expect(c.token_id < draft_cfg.vocab_size);
        try std.testing.expect(std.math.isFinite(c.prob) and c.prob > 0.0);
    }

    // Batched L1 seed chain forward (3 tokens in ONE pass -> seq_len stays 5!).
    const seed = [_]u32{ 13, 15, 17 };
    var masks: [16]u64 = @splat(0);
    var seed_cands: [3]moe.TopKCandidate = undefined;
    const got_k = try sd.stepSeedChain(&seed, 5, &masks, 3, &seed_cands);
    try std.testing.expectEqual(@as(u32, 3), got_k);
    try std.testing.expectEqual(@as(u32, 5), sd.seqLen());

    // Expand 2 children from seed tail (slot 2) at slots 3..4!
    masks[3] = masks[2] | (@as(u64, 1) << 3);
    masks[4] = masks[2] | (@as(u64, 1) << 4);
    const next_toks = [_]u32{ seed_cands[0].token_id, seed_cands[1].token_id };
    const next_pos = [_]u32{ 8, 8 };
    var batch_cands: [2 * 3]moe.TopKCandidate = undefined;
    _ = try sd.stepTreeBatch(&next_toks, &next_pos, masks[0..5], 3, 3, &batch_cands);
    try std.testing.expectEqual(@as(u32, 5), sd.seqLen());
    for (batch_cands) |c| {
        try std.testing.expect(c.token_id < draft_cfg.vocab_size);
        try std.testing.expect(std.math.isFinite(c.prob) and c.prob > 0.0);
    }

    // Sync accepted tokens [13, 19] -> seq_len = 7!
    const acc = [_]u32{ 13, 19 };
    try sd.syncAccepted(&acc, 5, 3);
    try std.testing.expectEqual(@as(u32, 7), sd.seqLen());
    try std.testing.expectEqual(@as(usize, 3), sd.rootCandidates().len);

    // D8 PEARL verification: independent compute queue + O(1) KV rewind exactness!
    try std.testing.expect(sd.hasOwnComputeQueue());
    try std.testing.expect(sd.model.compute.handle != target.compute.handle);

    var sd_ctrl = try SmallDraftModel.loadFromBytes(gpa, &target, draft_bytes, draft_cfg, .{
        .placement = .gpu,
        .max_seq_len = 64,
        .chunk_tokens = 16,
        .max_tree_slots = 16,
        .own_compute_queue = true,
    });
    defer sd_ctrl.deinit(gpa);

    // 1. Control model: prefill [3,5,7,9]+11, directly sync [13, 19].
    try sd_ctrl.prefillAndSeedRoot(&prompt, 11, 3);
    try sd_ctrl.syncAccepted(&[_]u32{ 13, 19 }, 5, 3);

    // 2. PEARL model: prefill [3,5,7,9]+11, speculatively commit [13, 15, 17],
    //    rewind in O(1) to pos 6 (reusing causal KV of `13` at pos 5!), then sync only `[19]`.
    try sd.prefillAndSeedRoot(&prompt, 11, 3);
    try sd.syncAccepted(&[_]u32{ 13, 15, 17 }, 5, 3);
    try std.testing.expectEqual(@as(u32, 8), sd.seqLen());
    try sd.rewindKvLen(6);
    try std.testing.expectEqual(@as(u32, 6), sd.seqLen());
    try std.testing.expectEqual(@as(usize, 0), sd.rootCandidates().len);
    try sd.syncAccepted(&[_]u32{19}, 6, 3);
    try std.testing.expectEqual(@as(u32, 7), sd.seqLen());

    // Exact bit-for-bit candidate parity between O(1)-rewound PEARL KV and clean control!
    const got_cands = sd.rootCandidates();
    const exp_cands = sd_ctrl.rootCandidates();
    try std.testing.expectEqual(exp_cands.len, got_cands.len);
    for (exp_cands, got_cands) |e, g| {
        try std.testing.expectEqual(e.token_id, g.token_id);
        try std.testing.expectApproxEqAbs(e.prob, g.prob, 1e-4);
    }

    // Invalid forward rewind must fail explicitly.
    try std.testing.expectError(error.InvalidArgument, sd.rewindKvLen(8));
}

fn buildSyntheticTargetBf16ForTest(allocator: std.mem.Allocator, cfg: qwen3.ModelConfig, seed: u64) ![]u8 {
    var recs: std.ArrayList(SynthRec) = .empty;
    defer {
        for (recs.items) |r| {
            allocator.free(r.name);
            allocator.free(r.data);
        }
        recs.deinit(allocator);
    }
    const h = cfg.hidden_size;
    const q_dim = cfg.qDim();
    const kv_dim = cfg.kvDim();
    const inter = cfg.intermediate_size;
    const inv_h = 0.5 / @sqrt(@as(f32, @floatFromInt(h)));
    const inv_q = 0.5 / @sqrt(@as(f32, @floatFromInt(q_dim)));
    const inv_i = 0.5 / @sqrt(@as(f32, @floatFromInt(inter)));
    var tidx: u64 = 1;

    const add2d = struct {
        fn call(alloc: std.mem.Allocator, list: *std.ArrayList(SynthRec), name: []const u8, kind: amf.TensorKind, n: u32, k: u32, amp: f32, s: u64) !void {
            const nm = try alloc.dupe(u8, name);
            errdefer alloc.free(nm);
            const data = try fillBf16(alloc, @as(usize, n) * k, amp, false, s);
            errdefer alloc.free(data);
            try list.append(alloc, .{ .name = nm, .kind = kind, .dtype = .bf16, .shape = .{ n, k, 0, 0 }, .rank = 2, .data = data });
        }
    }.call;
    const add1d = struct {
        fn call(alloc: std.mem.Allocator, list: *std.ArrayList(SynthRec), name: []const u8, dim: u32, s: u64) !void {
            const nm = try alloc.dupe(u8, name);
            errdefer alloc.free(nm);
            const data = try fillBf16(alloc, dim, 0.05, true, s);
            errdefer alloc.free(data);
            try list.append(alloc, .{ .name = nm, .kind = .norm, .dtype = .bf16, .shape = .{ dim, 0, 0, 0 }, .rank = 1, .data = data });
        }
    }.call;

    try add2d(allocator, &recs, "embed_tokens.weight", .embedding, cfg.vocab_size, h, 0.2, seed + tidx);
    tidx += 1;
    var nbuf: [64]u8 = undefined;
    for (0..cfg.num_hidden_layers) |l| {
        try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, h, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj.weight", .{l}) catch unreachable, .dense, q_dim, h, inv_h, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj.weight", .{l}) catch unreachable, .dense, kv_dim, h, inv_h, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj.weight", .{l}) catch unreachable, .dense, kv_dim, h, inv_h, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj.weight", .{l}) catch unreachable, .dense, h, q_dim, inv_q, seed + tidx);
        tidx += 1;
        try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable, cfg.head_dim, seed + tidx);
        tidx += 1;
        try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable, cfg.head_dim, seed + tidx);
        tidx += 1;
        try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, h, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj.weight", .{l}) catch unreachable, .dense, inter, h, inv_h, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj.weight", .{l}) catch unreachable, .dense, inter, h, inv_h, seed + tidx);
        tidx += 1;
        try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj.weight", .{l}) catch unreachable, .dense, h, inter, inv_i, seed + tidx);
        tidx += 1;
    }
    try add1d(allocator, &recs, "norm.weight", h, seed + tidx);

    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_raw_len: usize = recs.items.len * @sizeOf(amf.TensorEntry);
    const dir_bytes: usize = try page.alignUp(dir_raw_len);
    var total_size: usize = dir_offset + dir_bytes;
    for (recs.items) |r| total_size += try page.alignUp(r.data.len);

    const image = try allocator.alloc(u8, total_size);
    errdefer allocator.free(image);
    @memset(image, 0);

    const entries = try allocator.alloc(amf.TensorEntry, recs.items.len);
    defer allocator.free(entries);

    var cursor: usize = dir_offset + dir_bytes;
    for (recs.items, 0..) |r, i| {
        const blob_bytes = try page.alignUp(r.data.len);
        @memcpy(image[cursor .. cursor + r.data.len], r.data);
        const crc = std.hash.Crc32.hash(image[cursor .. cursor + blob_bytes]);
        var e = amf.TensorEntry.zeroed(r.name, r.kind, r.dtype);
        e.rank = r.rank;
        e.shape = r.shape;
        e.blob_offset = cursor;
        e.blob_bytes = blob_bytes;
        e.data_bytes = r.data.len;
        e.blob_crc32 = crc;
        entries[i] = e;
        cursor += blob_bytes;
    }

    const dir_slice = std.mem.sliceAsBytes(entries);
    @memcpy(image[dir_offset .. dir_offset + dir_slice.len], dir_slice);
    const dir_crc = std.hash.Crc32.hash(dir_slice);

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
    const mid = "target-test";
    @memcpy(hdr.model_id[0..mid.len], mid);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return image;
}
