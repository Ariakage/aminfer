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
const tree_mod = @import("../spec/tree.zig");
const expert_mod = @import("../expert/source.zig");
const aio = @import("../storage/aio.zig");
const cache_mod = @import("../storage/cache.zig");
const wavefront = @import("../ane/wavefront.zig");
const predictor_mod = @import("../ane/predictor.zig");
const sme = @import("../cpu/sme.zig");
const accept_mod = @import("../spec/accept.zig");
const rng_mod = @import("../runtime/rng.zig");
const prefix_cache_mod = @import("../mem/prefix_cache.zig");
const batch_sched = @import("../runtime/batch_scheduler.zig");

/// Execution backend selection for D7 speculative MoE tree router prediction (`C13`, `ADR-0018`).
pub const SmeRouterMode = enum {
    auto,
    sme2,
    gpu,

    pub fn label(self: SmeRouterMode) []const u8 {
        return switch (self) {
            .auto => "auto",
            .sme2 => "sme2",
            .gpu => "gpu",
        };
    }

    pub fn parse(str: []const u8) error{InvalidSmeRouterMode}!SmeRouterMode {
        if (std.mem.eql(u8, str, "auto")) return .auto;
        if (std.mem.eql(u8, str, "sme2")) return .sme2;
        if (std.mem.eql(u8, str, "gpu")) return .gpu;
        return error.InvalidSmeRouterMode;
    }
};

/// Active backend chosen by startup `C13` router calibration.
pub const RouterPredBackend = enum {
    gpu,
    sme2,
    sme2_single_owner,

    pub fn label(self: RouterPredBackend) []const u8 {
        return switch (self) {
            .gpu => "gpu",
            .sme2 => "sme2",
            .sme2_single_owner => "sme2_single_owner",
        };
    }
};

pub const SmeRouterCalibration = struct {
    backend: RouterPredBackend = .gpu,
    sme2_us: f64 = 0.0,
    gpu_us: f64 = 0.0,
    bank_bytes: u64 = 0,
    sampled_layers: u32 = 0,
};

/// Prompt prefill execution mode (`ADR-0010`, `A1`, `G6`).
pub const PrefillMode = enum {
    auto,
    gpu,
    hybrid,

    pub fn label(self: PrefillMode) []const u8 {
        return switch (self) {
            .auto => "auto",
            .gpu => "gpu",
            .hybrid => "hybrid",
        };
    }

    pub fn parse(str: []const u8) error{InvalidPrefillMode}!PrefillMode {
        if (std.mem.eql(u8, str, "auto")) return .auto;
        if (std.mem.eql(u8, str, "gpu")) return .gpu;
        if (std.mem.eql(u8, str, "hybrid")) return .hybrid;
        return error.InvalidPrefillMode;
    }
};

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
                .q2 => "q2",
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
                    2 => .q2,
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

/// Maximum number of target layers sampled during fast GPU router expert prediction (D7).
pub const MAX_EXPERT_PRED_LAYERS: u32 = 8;

/// Returns the default 0-indexed transformer layer outputs tapped for EAGLE-3
/// (`[1, N/2 - 1, N - 4]` for `N >= 5`, matching vLLM/speculators `[2, N//2, N - 3]`
/// in HF 1-based `hidden_states` indexing, e.g. `[1, 23, 44]` for 48-layer Qwen3-30B-A3B).
pub fn defaultEagle3AuxLayerIds(num_layers: u32) [3]u32 {
    if (num_layers <= 1) return .{ 0, 0, 0 };
    if (num_layers == 2) return .{ 0, 0, 1 };
    if (num_layers == 3) return .{ 0, 1, 2 };
    if (num_layers == 4) return .{ 0, 1, 3 };
    return .{ 1, num_layers / 2 - 1, num_layers - 4 };
}

pub const LoadOptions = struct {
    max_seq_len: u32 = 2048,
    chunk_tokens: u32 = attention.DEFAULT_CHUNK_TOKENS,
    /// Maximum speculative tree nodes verified in one `verifyTree` pass
    /// (`1` = decode-only; up to `attention.MAX_TREE_NODES = 64`).
    max_tree_nodes: u32 = 1,
    /// Captures 3 EAGLE-3 auxiliary layer outputs (`[max_tree_nodes, 3*H]`) and
    /// top-layer unnormalized hidden states (`[max_tree_nodes, H]`) for L2 drafting.
    enable_feature_taps: bool = false,
    /// Optional explicit 0-indexed transformer layer IDs for EAGLE-3 aux feature taps.
    /// Defaults to `defaultEagle3AuxLayerIds(config.num_hidden_layers)`.
    aux_layer_ids: ?[3]u32 = null,
    /// Optional per-shard `WiredArena` ceiling (defaults to `MTLDevice.maxBufferLength`).
    max_shard_bytes: ?usize = null,
    /// Optional explicit wired budget override (defaults to `Budget.fromHost`).
    budget_override: ?budget_mod.Budget = null,
    verify_crcs: bool = true,
    timeout_ms: u64 = 30_000,

    // M4 Cold-Expert NVMe Streaming + ANE Cross-Layer Prediction (C3-C5, A2, W8)
    /// Number of cold experts per layer kept on NVMe instead of wired RAM (`0` = all hot,
    /// unless `.amf` entries are already tagged `TensorKind.cold_expert` or `heat_profile_json` specifies cold experts).
    cold_experts_per_layer: u32 = 0,
    /// Optional W8 `schema_version = 1` expert heat profile JSON bytes (`aminfer-tools expert-heat`).
    heat_profile_json: ?[]const u8 = null,
    /// Preallocated 16 KiB-aligned cold-expert slot count in `ExpertSlotPool`.
    cold_slot_count: u32 = 64,
    /// Async direct I/O backend (`auto`, `kqueue_aio`, `ecore_pread`).
    io_backend: aio.IoBackend = .auto,
    /// Enables ANE cross-layer MoE expert prediction driving async NVMe cold-expert prefetch (`A2`).
    enable_ane_predictor: bool = true,
    /// Optional offline W7 `.mlpackage`/`.mlmodelc` path for all-layers ANE expert predictor.
    ane_predictor_path: ?[:0]const u8 = null,

    // M4 Hybrid ANE/GPU Wavefront Prefill (A1, A3, G6, ADR-0010)
    /// Prompt prefill mode (`auto`, `gpu`, `hybrid`).
    prefill_mode: PrefillMode = .auto,
    /// Pre-compiles ANE attention blocks and 512-token GPU FFN chunk scratch at load time.
    enable_hybrid_prefill: bool = false,
    /// Static KV bucket strategy (`bucketed_full_ane` settled by Spike S6 / A3).
    hybrid_strategy: wavefront.StaticShapeStrategy = .bucketed_full_ane,
    /// Maximum static KV bucket length compiled for ANE hybrid prefill.
    hybrid_max_bucket: u32 = 4096,

    // M4.5 Bare-Metal ARMv9 SME2 Engine & L2-Resident Tree Router Prediction (W15, C11-C13, ADR-0018)
    /// Backend policy for D7 speculative MoE tree router prediction (`auto`, `sme2`, `gpu`).
    sme_router_mode: SmeRouterMode = .auto,

    // M6.5 R5 32K+ GPU Group-64 `q8` Retired KV Tier (ADR-0004, ADR-0019)
    /// When `> 0`, keep at most `N` active `RGBA16Float` IOSurface KV chunks per layer
    /// and compress older completed `chunk_tokens` (`512`-token) chunks on GPU into
    /// group-64 `q8` (`RetiredKvQ8Pool`, `-46.875%` retired KV RAM), recycling active
    /// IOSurface tiles back to the active window. `0` disables retirement (default).
    kv_retire_q8_after_chunks: u32 = 0,
};

pub const LoadError = error{
    MissingTensor,
    UnexpectedTensorShape,
    UnexpectedTensorDType,
    TensorExceedsShardLimit,
    TooManyDispatchesForFrame,
    ColdExpertsRequireFile,
    InvalidHeatProfile,
    HybridPrefillUnsupported,
    SmeCalibrationFailed,
    OutOfMemory,
} || ModelConfig.ConfigError || amf.AmfFile.OpenError || quant.ShapeError ||
    sysinfo.Error || metal.WrapError || budget_mod.PlanError ||
    arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError ||
    attention.RetiredKvQ8PoolError ||
    moe.WorkspaceError || aio.EcorePoolError || cache_mod.PoolError ||
    cache_mod.CacheError || expert_mod.NvmeExpert.InitError || page.ShareError ||
    predictor_mod.AneExpertPredictor.Error || wavefront.WavefrontPrefill.InitError ||
    wavefront.AneAttentionBucket.InitError || sme.Error || std.Thread.SpawnError;

pub const StepError = error{
    TokenOutOfRange,
    PositionOutOfRange,
    TreeNotEnabled,
    TreeTooLarge,
    HybridPrefillNotEnabled,
    WavefrontFailed,
} || metal.Error || dequant.EncodeError || moe.EncodeError ||
    attention.RetiredKvQ8PoolError ||
    cache_mod.CacheError || predictor_mod.AneExpertPredictor.Error ||
    sme.Error || accept_mod.VerifyError;

pub const VerifyTreeResult = struct {
    num_nodes: u32,
    /// `[num_nodes]` greedy argmax token IDs computed on GPU (`encodeArgmaxRows`).
    argmax_tokens: []const u32,
    /// Flat `[num_nodes * vocab_size]` `f16` target logits in wired shared RAM.
    logits_f16: []const f16,
    /// Sum of measured unique active experts across all MoE layers (`0` for dense).
    total_unique_experts: u64,
    /// Mean measured unique active experts per layer (`0.0` for dense).
    avg_unique_experts_per_layer: f64,
    /// Algorithmic bytes moved during this verification pass (using measured unique experts).
    actual_bytes_moved: u64,
};

/// Cold-expert streaming & ANE cross-layer prefetch statistics (`C3`-`C5`, `A2`).
pub const ColdExpertStats = struct {
    total_cold_experts: u32,
    slot_count: u32,
    io_backend: aio.IoBackend,
    ane_predictor_active: bool,
    ane_predictor_uses_ne: bool,
    ane_predictor_compile_ms: f64,
    cache_telemetry: cache_mod.Telemetry,
};

/// Configuration for `Qwen3Model.initHybridPrefill` (`A1`, `A3`, `G6`).
pub const HybridPrefillOptions = struct {
    strategy: wavefront.StaticShapeStrategy = .bucketed_full_ane,
    max_bucket: u32 = 4096,
};

/// Telemetry returned by `Qwen3Model.prefillHybridWavefront` (`A1`, `G6`).
pub const HybridPrefillResult = struct {
    tokens_prefilled: u32,
    argmax_token: u32,
    wavefront_stats: wavefront.WavefrontStats,
    all_outputs_zero_copy: bool,
    all_buckets_use_ane: bool,
};

/// Optional ANE override for `layers[0].o_proj` during `stepDraftTreeBatch` (`D4`/`D5`).
pub const AneOProjHook = struct {
    model: metal.AneModel,
    plan: metal.AnePlan,
    in_bind: metal.BufferBinding,
    out_bind: metal.BufferBinding,
    max_batch_rows: u32,
    c_in: u32,
    c_out: u32,
    last_zero_copy_ok: *bool,
};

pub const LinearBinding = struct {
    weight: metal.BufferBinding,
    scale_bias: ?metal.BufferBinding,
    n: u32,
    k: u32,
    dtype: quant.WeightDType,
    alg_bytes: u64,
    layout: amf.TensorLayout = .row_major,
    param_kind: amf.QuantParamKind = .affine_scale_bias,
    /// Zero-copy slice into the owning `WiredArena` shard when `layout == .sme2_panel64`
    /// (single-owner invariant, ADR-0018 Safeguard B.2).
    sme_packed: ?[]const u8 = null,
    /// Zero-copy CPU slices into the owning `WiredArena` shard for pure-CPU SME2 draft execution.
    weight_host: []const u8 = &.{},
    scale_bias_host: ?[]const u8 = null,
};

pub const NormBinding = struct {
    binding: metal.BufferBinding,
    dtype: quant.FloatDType,
    host_bytes: []const u8 = &.{},
};

const WeightShard = struct {
    arena: arena_mod.WiredArena,
    buffer: metal.Buffer,
};

const SmeExpertTriple = struct {
    gate: []const u8,
    up: []const u8,
    down: []const u8,
    dtype: quant.WeightDType = .q3_affine,
    param_kind: amf.QuantParamKind = .affine_scale_bias,
    is_cold: bool = false,
};

const LayerWeights = struct {
    shard_idx: u16,
    input_layernorm: NormBinding,
    q_proj: LinearBinding,
    k_proj: LinearBinding,
    v_proj: LinearBinding,
    o_proj: LinearBinding,
    q_norm: ?metal.BufferBinding,
    k_norm: ?metal.BufferBinding,
    q_norm_host: ?[]const u8 = null,
    k_norm_host: ?[]const u8 = null,
    qk_norm_dtype: quant.FloatDType,
    post_attention_layernorm: NormBinding,

    // MoE-specific (valid when config.arch == .qwen3_moe)
    router: moe.RouterSource,
    router_layout: amf.TensorLayout = .row_major,
    router_param_kind: amf.QuantParamKind = .affine_scale_bias,
    router_panel_dtype: ?sme.PanelDType = null,
    /// Zero-copy slice into the owning `WiredArena` shard when `router_layout == .sme2_panel64`.
    router_sme_packed: ?[]const u8 = null,
    /// Uniform expert dtype when `is_heterogeneous_moe == false`; expert 0's dtype otherwise.
    expert_dtype: quant.WeightDType,
    expert_layout: amf.TensorLayout = .row_major,
    expert_param_kind: amf.QuantParamKind = .affine_scale_bias,
    /// True iff experts inside this layer span > 1 `WeightDType` scheme (G13, ADR-0017).
    is_heterogeneous_moe: bool = false,
    /// Static expert count per `WeightDType` bucket in this layer (`sum == num_experts`).
    expert_scheme_counts: [moe.NUM_SCHEME_BUCKETS]u32 = @splat(0),
    /// Device-resident `[num_experts]u8` mapping `expert_id -> @backingInt(WeightDType)`.
    expert_schemes_bind: metal.BufferBinding,
    expert_weights_buf: metal.Buffer,
    expert_table: metal.BufferBinding,
    expert_table_host: []moe.ExpertWeights,
    /// Allocated only when `expert_layout == .sme2_panel64` (slices point zero-copy into shard arena).
    expert_sme_blobs: ?[]SmeExpertTriple = null,

    // Dense-specific (valid when config.arch == .qwen3_dense)
    dense_gate: LinearBinding,
    dense_up: LinearBinding,
    dense_down: LinearBinding,

    // Bindless KV chunk table for this layer (`[active_kv_chunks]KvTilePair` in scratch buffer)
    kv_tiles_binding: metal.BufferBinding,
    kv_tiles_host: []attention.KvTilePair,
    // R5 retired group-64 `q8` chunk IDs (`[max_retired_per_layer]u32`, empty when disabled)
    retired_ids_binding: metal.BufferBinding,
    retired_ids_host: []u32,
};

const LM_HEAD_MAX_TILES: usize = 64;

const LmHeadTileState = struct {
    max_val: f32 = -std.math.inf(f32),
    sum_exp: f32 = 0.0,
    count: u32 = 0,
    top_ids: [moe.MAX_TOP_K]u32 = @splat(0),
    top_vals: [moe.MAX_TOP_K]f32 = @splat(-std.math.inf(f32)),
};

/// Heap-pinned ARMv9 SME2 runtime state for single-owner `sme2_panel64` execution,
/// L2-resident speculative tree router prediction (`C13`), and non-temporal `ldnt1b`
/// cold-expert / off-GPU small-draft evaluation (`C12`, `D8`). Preallocated once at startup.
const SmeRuntime = struct {
    dispatcher: sme.SmeDispatcher,
    x_f32_scratch: []f32,
    x_act_u16_scratch: []align(sme.SVL_BYTES) u16,
    router_logits_f32: []align(sme.SVL_BYTES) f32,
    expert_up_f32: []align(sme.SVL_BYTES) f32,
    expert_inter_u16: []align(sme.SVL_BYTES) u16,
    expert_down_f32: []align(sme.SVL_BYTES) f32,
    expert_acc_f32: []align(sme.SVL_BYTES) f32,
    dense_up_f32: []align(sme.SVL_BYTES) f32,
    x_res_f32: []align(sme.SVL_BYTES) f32,
    rope_inv_freq: []f32,
    lm_tile_stats: []LmHeadTileState,

    // L2-resident D7 router predictor bank (`null` arena when using single-owner `sme2_panel64` routers).
    router_bank_arena: ?arena_mod.WiredArena,
    router_bank_packed: []const u8,
    router_bank_dtype: sme.PanelDType,
    router_bank_sampled: u32,
    router_bank_single_owner: bool,
    router_pred_backend: RouterPredBackend,
    router_pred_sme2_us: f64,
    router_pred_gpu_us: f64,

    fn deinit(self: *SmeRuntime, allocator: std.mem.Allocator) void {
        if (self.router_bank_arena) |*a| a.deinit();
        if (self.lm_tile_stats.len > 0) allocator.free(self.lm_tile_stats);
        if (self.rope_inv_freq.len > 0) allocator.free(self.rope_inv_freq);
        if (self.x_res_f32.len > 0) allocator.free(self.x_res_f32);
        if (self.dense_up_f32.len > 0) allocator.free(self.dense_up_f32);
        if (self.expert_acc_f32.len > 0) allocator.free(self.expert_acc_f32);
        if (self.expert_down_f32.len > 0) allocator.free(self.expert_down_f32);
        if (self.expert_inter_u16.len > 0) allocator.free(self.expert_inter_u16);
        if (self.expert_up_f32.len > 0) allocator.free(self.expert_up_f32);
        if (self.router_logits_f32.len > 0) allocator.free(self.router_logits_f32);
        if (self.x_act_u16_scratch.len > 0) allocator.free(self.x_act_u16_scratch);
        if (self.x_f32_scratch.len > 0) allocator.free(self.x_f32_scratch);
        self.dispatcher.deinit();
        allocator.destroy(self);
    }
};

/// Heap-pinned cold-expert NVMe streaming + ANE predictor state (allocated once
/// at startup so `ExpertCache`'s internal pointers to `pool` and `reader` stay
/// stable when `Qwen3Model` moves by value).
const ColdRuntime = struct {
    file: aio.DirectFile,
    pool: cache_mod.ExpertSlotPool,
    reader: aio.AsyncExpertReader,
    expert_cache: cache_mod.ExpertCache,
    ane_predictor: ?*predictor_mod.AneExpertPredictor,
    cold_masks_by_layer: []tree_mod.ExpertBitset,
    layer_has_cold: []bool,
    total_cold_experts: u32,
    pred_bitsets: [predictor_mod.MAX_LOOKAHEAD]tree_mod.ExpertBitset,
    pred_cands: [predictor_mod.MAX_LOOKAHEAD * predictor_mod.MAX_EXPERTS]predictor_mod.PrefetchCandidate,
    cold_bool_scratch: [moe.MAX_EXPERTS]bool,
    ane_req_seq: u64,
    ane_inflight: bool,

    fn deinit(self: *ColdRuntime, allocator: std.mem.Allocator) void {
        if (self.ane_predictor) |pred| {
            if (self.ane_inflight) {
                _ = pred.waitAsync(0, null, &self.pred_bitsets, &self.pred_cands) catch |err| {
                    std.debug.panic("ColdRuntime.deinit waitAsync failed: {}", .{err});
                };
                self.ane_inflight = false;
            }
            pred.deinit(allocator);
            self.ane_predictor = null;
        }
        self.expert_cache.deinit(allocator);
        self.reader.deinit(allocator);
        self.pool.deinit();
        self.file.close();
        allocator.free(self.layer_has_cold);
        allocator.free(self.cold_masks_by_layer);
        allocator.destroy(self);
    }
};

/// Heap-pinned hybrid ANE/GPU wavefront prefill runtime (`A1`, `A3`, `G6`, `ADR-0010`).
const HybridPrefillRuntime = struct {
    strategy: wavefront.StaticShapeStrategy,
    chunk_tokens: u32,
    num_layers: u32,
    num_chunks: u32,
    num_buckets: usize,
    buckets: [wavefront.MAX_STATIC_BUCKETS]u32,

    wavefront_sched: *wavefront.WavefrontPrefill,
    ane_buckets: []wavefront.AneAttentionBucket, // [num_layers * num_buckets]
    bound_plans: []wavefront.AneAttentionBucket.BoundPlan, // [num_layers * num_chunks]
    bound_bucket_idx: []usize, // [num_chunks]
    bound_uses_direct_past: []bool, // [num_chunks]

    // Double-buffered zero-copy IOSurface activation tiles (`slot = chunk & 1`)
    x_tiles: [2]metal.KvTile,
    x_norm_tiles: [2]metal.KvTile,
    attn_out_tiles: [2]metal.KvTile,
    mask_tiles: [2][wavefront.MAX_STATIC_BUCKETS]metal.KvTile,
    k_past_tiles: [2][wavefront.MAX_STATIC_BUCKETS]?metal.KvTile,
    v_past_tiles: [2][wavefront.MAX_STATIC_BUCKETS]?metal.KvTile,

    // Preallocated 512-token GPU FFN chunk arena (`G6`)
    ffn_arena: arena_mod.WiredArena,
    ffn_buf: metal.Buffer,
    chunk_token_ids_host: []u32,
    chunk_token_ids_bind: metal.BufferBinding,
    x_norm_chunk_bind: metal.BufferBinding,
    proj_out_chunk_bind: metal.BufferBinding,
    moe_ws_chunk: ?moe.MoeWorkspace,
    dense_gate_chunk_bind: metal.BufferBinding,
    dense_up_chunk_bind: metal.BufferBinding,
    dense_act_chunk_bind: metal.BufferBinding,
    total_wired_bytes: u64,

    fn deinit(self: *HybridPrefillRuntime, allocator: std.mem.Allocator) void {
        self.wavefront_sched.deinit(allocator);
        for (self.bound_plans) |bp| bp.close();
        allocator.free(self.bound_plans);
        allocator.free(self.bound_bucket_idx);
        allocator.free(self.bound_uses_direct_past);
        for (self.ane_buckets) |ab| ab.close();
        allocator.free(self.ane_buckets);
        for (0..2) |s| {
            for (0..self.num_buckets) |bi| {
                if (self.v_past_tiles[s][bi]) |t| t.close();
                if (self.k_past_tiles[s][bi]) |t| t.close();
                self.mask_tiles[s][bi].close();
            }
            self.attn_out_tiles[s].close();
            self.x_norm_tiles[s].close();
            self.x_tiles[s].close();
        }
        self.ffn_buf.close();
        self.ffn_arena.deinit();
        allocator.destroy(self);
    }
};

const AmfSource = union(enum) {
    bytes: struct {
        parsed: *const amf.Parsed,
        raw: []const u8,
    },
    file: struct {
        handle: *const amf.AmfFile,
        path: [:0]const u8,
    },

    fn parsed(self: AmfSource) *const amf.Parsed {
        return switch (self) {
            .bytes => |b| b.parsed,
            .file => |f| &f.handle.parsed,
        };
    }

    fn filePath(self: AmfSource) ?[:0]const u8 {
        return switch (self) {
            .bytes => null,
            .file => |f| f.path,
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
                    return f.handle.loadBlob(entry, arena);
                }
                const len: usize = @intCast(entry.blob_bytes);
                const dst = try arena.allocPages(len);
                var done: usize = 0;
                while (done < dst.len) {
                    const rc = std.c.pread(f.handle.fd, dst.ptr + done, dst.len - done, @intCast(entry.blob_offset + done));
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
    max_tree_nodes: u32,
    enable_feature_taps: bool,
    aux_layer_ids: [3]u32,
    timeout_ms: u64,
    wired_budget_bytes: u64,
    total_wired_bytes: u64,
    seq_len: u32,
    prefill_mode: PrefillMode,

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
    embed_shard_idx: u16,
    embed: LinearBinding,
    final_norm: NormBinding,
    lm_head: LinearBinding,

    // Preallocated hot-path activation slices & bindings inside `scratch_arena`/`scratch_buf`
    x_host: []f16,
    x_norm_host: []f16,
    q_host: []f16,
    k_host: []f16,
    v_host: []f16,
    attn_out_host: []f16,
    proj_out_host: []f16,
    logits_mut_host: []f16,
    logits_host: []const f16,
    out_token_ids_host: []u32,
    out_token_id_host: *u32,
    last_logits_row: usize,

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

    // Preallocated M3 tree-verification & feature-tap scratch (zero hot-path heap alloc)
    tree_token_ids_host: []u32,
    tree_positions_host: []u32,
    tree_ancestor_masks_host: []u64,
    tree_accepted_nodes_host: []u32,
    topk_host: []moe.TopKCandidate,
    tree_token_ids_bind: metal.BufferBinding,
    tree_positions_bind: metal.BufferBinding,
    tree_ancestor_masks_bind: metal.BufferBinding,
    tree_accepted_nodes_bind: metal.BufferBinding,
    topk_bind: metal.BufferBinding,
    tree_k_base_off: u64,
    tree_v_base_off: u64,
    tree_kv_layer_stride: u64,
    gqa_tree_scratch_bind: metal.BufferBinding,

    aux_concat_host: []f16,
    aux_concat_bind: metal.BufferBinding,
    accepted_aux_host: []f16,
    accepted_aux_bind: metal.BufferBinding,
    accepted_top_host: []f16,
    accepted_top_bind: metal.BufferBinding,

    // Arch-specific FFN scratch
    moe_ws: ?moe.MoeWorkspace,
    layer_sched_hdr_base_off: u64,
    layer_sched_hdr_raw: []u8,
    pred_topk_base_off: u64,
    pred_topk_layer_stride: u64,
    pred_topk_host: []u32,
    dense_gate_host: []f16,
    dense_up_host: []f16,
    dense_act_host: []f16,
    dense_gate_bind: metal.BufferBinding,
    dense_up_bind: metal.BufferBinding,
    dense_act_bind: metal.BufferBinding,

    // Precomputed algorithmic traffic constants
    total_layers_static_bytes: u64,
    total_layers_non_expert_bytes: u64,
    single_expert_bytes: u64,
    embed_row_alg_bytes: u64,
    global_head_bytes: u64,

    // M5 G13 Static Per-Expert Quantization Scheme & Exact Byte Tables (ADR-0017)
    is_heterogeneous_moe: bool,
    /// Flat `[num_hidden_layers * num_experts]` static per-expert `WeightDType` (empty for dense).
    expert_dtypes: []quant.WeightDType,
    /// Flat `[num_hidden_layers * num_experts]` exact algorithmic bytes per expert (empty for dense).
    expert_bytes: []u64,
    /// Flat `[num_hidden_layers * num_experts]` `u32` byte table for `D7` pruning (`PruneCostModel`).
    expert_bytes_u32: []u32,

    // M4 Cold-Expert NVMe Streaming + ANE Predictor & Hybrid Wavefront Prefill runtimes
    cold_rt: ?*ColdRuntime,
    hybrid_rt: ?*HybridPrefillRuntime,

    // M4.5 Single-Owner SME2 Weight Ownership & L2-Resident Tree Router Runtime (`W15`, `C11`-`C13`)
    has_sme2_weights: bool,
    has_pure_cpu_sme2_dense: bool,
    sme_router_mode: SmeRouterMode,
    sme_rt: ?*SmeRuntime,

    // M6.5 R5 Tiered Group-64 `q8` Retired KV Cache (`ADR-0004`, `ADR-0019`)
    kv_retire_after_chunks: u32,
    active_kv_chunks: u32,
    num_retired_chunks: u32,
    max_retired_per_layer: u32,
    kv_retire_events: u64,
    active_kv_wired_bytes: u64,
    retired_kv_wired_bytes: u64,
    retired_kv_pool: ?attention.RetiredKvQ8Pool,
    active_tile_phys_ring: []u32,

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
        return loadInternal(allocator, .{ .file = .{ .handle = &file, .path = amf_path } }, config, options);
    }

    pub fn deinit(self: *Qwen3Model, allocator: std.mem.Allocator) void {
        self.compute.close();
        if (self.sme_rt) |srt| {
            srt.deinit(allocator);
            self.sme_rt = null;
        }
        if (self.hybrid_rt) |hrt| {
            hrt.deinit(allocator);
            self.hybrid_rt = null;
        }
        if (self.cold_rt) |crt| {
            crt.deinit(allocator);
            self.cold_rt = null;
        }
        if (self.expert_bytes_u32.len > 0) allocator.free(self.expert_bytes_u32);
        if (self.expert_bytes.len > 0) allocator.free(self.expert_bytes);
        if (self.expert_dtypes.len > 0) allocator.free(self.expert_dtypes);
        for (self.layers) |layer| {
            if (layer.expert_sme_blobs) |esb| allocator.free(esb);
        }
        self.moe_pipes.close();
        self.attn_pipes.close();
        self.dequant_pipes.close();
        self.library.close();
        for (self.kv_tiles) |tile| tile.close();
        allocator.free(self.kv_tiles);
        allocator.free(self.active_tile_phys_ring);
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
        self.num_retired_chunks = 0;
        self.kv_retire_events = 0;
        if (self.retired_kv_pool) |*pool| {
            pool.reset();
        }
        for (0..self.active_kv_chunks) |i| {
            self.active_tile_phys_ring[i] = @intCast(i);
        }
        self.syncActiveKvTileTables();
    }

    pub fn kvRetireEnabled(self: *const Qwen3Model) bool {
        return self.retired_kv_pool != null;
    }

    pub fn retiredKvChunks(self: *const Qwen3Model) u32 {
        return self.num_retired_chunks;
    }

    pub fn kvRetireEvents(self: *const Qwen3Model) u64 {
        return self.kv_retire_events;
    }

    pub fn activeKvWiredBytes(self: *const Qwen3Model) u64 {
        return self.active_kv_wired_bytes;
    }

    pub fn retiredKvWiredBytes(self: *const Qwen3Model) u64 {
        return self.retired_kv_wired_bytes;
    }

    fn syncActiveKvTileTables(self: *Qwen3Model) void {
        const active_n: usize = self.active_kv_chunks;
        for (self.layers, 0..) |layer, l| {
            for (0..active_n) |slot| {
                const phys: usize = self.active_tile_phys_ring[slot];
                const tbase = (l * active_n + phys) * 2;
                layer.kv_tiles_host[slot] = attention.KvTilePair.fromTiles(
                    self.kv_tiles[tbase],
                    self.kv_tiles[tbase + 1],
                    .rgba16f,
                );
            }
        }
    }

    /// Compresses the oldest active `chunk_tokens` (`512`-token) KV chunk across
    /// all layers on GPU into `retired_kv_pool` (`group-64 q8`), rotates its
    /// `IOSurface` `KvTilePair` back to the tail of the active window, and
    /// updates every layer's bindless tile table (`0` dynamic heap allocs).
    fn retireOldestActiveChunk(self: *Qwen3Model) StepError!void {
        var pool = &(self.retired_kv_pool.?);
        std.debug.assert(self.num_retired_chunks < self.max_retired_per_layer);
        const retired_idx: usize = self.num_retired_chunks;
        const phys_oldest: usize = self.active_tile_phys_ring[0];
        const active_n: usize = self.active_kv_chunks;

        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();

        for (self.layers, 0..) |layer, l| {
            const dst_slot = try pool.acquireSlot();
            layer.retired_ids_host[retired_idx] = dst_slot;
            const tbase = (l * active_n + phys_oldest) * 2;
            try self.attn_pipes.encodeKvRetireTileToQ8(&frame, .{
                .k_tile = self.kv_tiles[tbase],
                .v_tile = self.kv_tiles[tbase + 1],
                .pool = pool,
                .pool_buf = self.scratch_buf,
                .dst_slot = dst_slot,
            }, false);
        }

        const sig = try frame.commit();
        try self.compute.wait(sig, self.timeout_ms);

        // Rotate the recycled physical IOSurface tile to the back of the active window.
        if (active_n > 1) {
            for (0..active_n - 1) |i| {
                self.active_tile_phys_ring[i] = self.active_tile_phys_ring[i + 1];
            }
            self.active_tile_phys_ring[active_n - 1] = @intCast(phys_oldest);
        }
        self.num_retired_chunks += 1;
        self.kv_retire_events += 1;
        self.syncActiveKvTileTables();
    }

    fn retireBeforeWritePos(self: *Qwen3Model, pos: u32) StepError!void {
        if (self.retired_kv_pool == null) return;
        const chunk_idx: u32 = pos / self.chunk_tokens;
        while (chunk_idx >= self.num_retired_chunks + self.kv_retire_after_chunks and
            self.num_retired_chunks < self.max_retired_per_layer and
            self.seq_len >= (self.num_retired_chunks + 1) * self.chunk_tokens)
        {
            try self.retireOldestActiveChunk();
        }
    }

    fn retireBeforeCommitRange(self: *Qwen3Model, base_pos: u32, count: u32) StepError!void {
        if (self.retired_kv_pool == null or count == 0) return;
        const last_chunk: u32 = (base_pos + count - 1) / self.chunk_tokens;
        while (last_chunk >= self.num_retired_chunks + self.active_kv_chunks and
            self.num_retired_chunks < self.max_retired_per_layer and
            base_pos >= (self.num_retired_chunks + 1) * self.chunk_tokens)
        {
            try self.retireOldestActiveChunk();
        }
    }

    fn retireExcessCompletedChunks(self: *Qwen3Model) StepError!void {
        if (self.retired_kv_pool == null or self.seq_len == 0) return;
        const total_live_chunks: u32 = (self.seq_len + self.chunk_tokens - 1) / self.chunk_tokens;
        while (total_live_chunks > self.num_retired_chunks + self.kv_retire_after_chunks and
            self.num_retired_chunks < self.max_retired_per_layer and
            self.seq_len >= (self.num_retired_chunks + 1) * self.chunk_tokens)
        {
            try self.retireOldestActiveChunk();
        }
    }

    pub fn shardCount(self: *const Qwen3Model) usize {
        return self.shards.len;
    }

    pub fn totalWiredBytes(self: *const Qwen3Model) u64 {
        return self.total_wired_bytes;
    }

    pub fn singleExpertBytes(self: *const Qwen3Model) u64 {
        return self.single_expert_bytes;
    }

    /// True iff any MoE layer in this checkpoint stores mixed per-expert
    /// quantization schemes (G13 / ADR-0017). False for dense and uniform MoE models.
    pub fn isHeterogeneousMoe(self: *const Qwen3Model) bool {
        return self.is_heterogeneous_moe;
    }

    /// Exact algorithmic weight + scale_bias bytes read for expert `(layer, expert)`
    /// (`gate_proj + up_proj + down_proj`), exposed for `SpecEngine` `D7` pruning.
    pub fn expertBytes(self: *const Qwen3Model, layer: usize, expert: usize) u64 {
        std.debug.assert(self.config.arch == .qwen3_moe);
        const idx = layer * @as(usize, self.config.num_experts) + expert;
        return self.expert_bytes[idx];
    }

    /// Static `WeightDType` scheme of expert `(layer, expert)` (`ADR-0017`).
    pub fn expertDType(self: *const Qwen3Model, layer: usize, expert: usize) quant.WeightDType {
        std.debug.assert(self.config.arch == .qwen3_moe);
        const idx = layer * @as(usize, self.config.num_experts) + expert;
        return self.expert_dtypes[idx];
    }

    /// Flat `[num_hidden_layers * num_experts]` `u32` expert byte table for `D7` pruning.
    pub fn expertBytesFlatU32(self: *const Qwen3Model) []const u32 {
        return self.expert_bytes_u32;
    }

    /// Flat `[num_hidden_layers * num_experts]` static per-expert `WeightDType` table.
    pub fn expertDTypesFlat(self: *const Qwen3Model) []const quant.WeightDType {
        return self.expert_dtypes;
    }

    pub fn totalLayersNonExpertBytes(self: *const Qwen3Model) u64 {
        return self.total_layers_non_expert_bytes;
    }

    /// Mean algorithmic bytes transferred by one transformer layer at `seq_len`
    /// (dense attention weights + active FFN weights + norms + KV read/write).
    pub fn bytesPerLayerToken(self: *const Qwen3Model, seq_len: u32) u64 {
        const retired_toks: u64 = @min(@as(u64, seq_len), @as(u64, self.num_retired_chunks) * @as(u64, self.chunk_tokens));
        const active_toks: u64 = @as(u64, seq_len) - retired_toks;
        const q8_row_bytes: u64 = (@as(u64, self.config.head_dim) / attention.KV_Q8_GROUP_SIZE) * attention.KV_Q8_GROUP_BYTES;
        const kv_read_active: u64 = 2 * @as(u64, self.config.num_key_value_heads) * active_toks * self.config.head_dim * @sizeOf(f16);
        const kv_read_retired: u64 = 2 * @as(u64, self.config.num_key_value_heads) * retired_toks * q8_row_bytes;
        const kv_write: u64 = 2 * @as(u64, self.config.num_key_value_heads) * self.config.head_dim * @sizeOf(f16);
        return (self.total_layers_static_bytes / self.config.num_hidden_layers) + kv_read_active + kv_read_retired + kv_write;
    }

    /// Algorithmic bytes transferred by global embedding row + final norm + `lm_head`.
    pub fn bytesGlobalHead(self: *const Qwen3Model) u64 {
        return self.global_head_bytes;
    }

    /// Algorithmic bytes transferred by one complete `stepDecode` at `seq_len`.
    pub fn bytesPerDecodeToken(self: *const Qwen3Model, seq_len: u32) u64 {
        const retired_toks: u64 = @min(@as(u64, seq_len), @as(u64, self.num_retired_chunks) * @as(u64, self.chunk_tokens));
        const active_toks: u64 = @as(u64, seq_len) - retired_toks;
        const q8_row_bytes: u64 = (@as(u64, self.config.head_dim) / attention.KV_Q8_GROUP_SIZE) * attention.KV_Q8_GROUP_BYTES;
        const kv_per_layer: u64 =
            2 * @as(u64, self.config.num_key_value_heads) * (active_toks + 1) * self.config.head_dim * @sizeOf(f16) +
            2 * @as(u64, self.config.num_key_value_heads) * retired_toks * q8_row_bytes;
        return self.total_layers_static_bytes + @as(u64, self.config.num_hidden_layers) * kv_per_layer + self.global_head_bytes;
    }

    pub fn readLogitsF16(self: *const Qwen3Model) []const f16 {
        return self.logits_host[0..self.config.vocab_size];
    }

    /// Returns the most recently computed token's logits row (`[vocab_size]f16`),
    /// tracking `last_logits_row` across single-token `stepDecode`, `prefillChunk`,
    /// `verifyTree`, and `prefillHybridWavefront`.
    pub fn readLastLogitsF16(self: *const Qwen3Model) []const f16 {
        const v: usize = self.config.vocab_size;
        return self.logits_host[self.last_logits_row * v .. (self.last_logits_row + 1) * v];
    }

    /// Samples a token from `readLastLogitsF16()` at `temperature > 0` using
    /// caller-preallocated `p_res` scratch (`len >= vocab_size`) and `rng` (`D11`).
    /// Zero dynamic heap allocation.
    pub fn sampleFromLastLogits(
        self: *const Qwen3Model,
        temperature: f32,
        rng: std.Random,
        p_res: []f32,
    ) accept_mod.VerifyError!u32 {
        try accept_mod.softmaxRow(f16, self.readLastLogitsF16(), temperature, p_res);
        return accept_mod.sampleCategorical(p_res[0..self.config.vocab_size], rng);
    }

    pub fn hasSme2Weights(self: *const Qwen3Model) bool {
        return self.has_sme2_weights;
    }

    /// True when every transformer layer projection is `sme2_panel64` on a dense
    /// model with tree drafting enabled, allowing `stepDraftTreeBatch` and
    /// `prefillChunk` to execute 100% off-GPU on ARMv9 SME2 (`ldnt1b`) with zero
    /// Metal GPU queue contention (`C12`, `C13`, `D8`).
    pub fn hasPureCpuSme2Draft(self: *const Qwen3Model) bool {
        return self.has_pure_cpu_sme2_dense;
    }

    pub fn routerPredCalibration(self: *const Qwen3Model) SmeRouterCalibration {
        const srt = self.sme_rt orelse return .{};
        return .{
            .backend = srt.router_pred_backend,
            .sme2_us = srt.router_pred_sme2_us,
            .gpu_us = srt.router_pred_gpu_us,
            .bank_bytes = if (srt.router_bank_arena) |*a| a.bytes().len else 0,
            .sampled_layers = srt.router_bank_sampled,
        };
    }

    /// Evaluates one `sme2_panel64` SwiGLU expert (`gate_proj`, `up_proj`, `down_proj`)
    /// on CPU SME2 with explicit `cache_hint` (`.non_temporal` emits `ldnt1*` to avoid
    /// evicting hot L2 routers/activations during one-shot cold-expert digestion, `C12`/`C13`)
    /// and fused `EpilogueOp.swiglu_gate_up` + optional `add_residual`. Zero heap allocations.
    pub fn evalSwiGluExpertSme2(
        dispatcher: *sme.SmeDispatcher,
        dtype: sme.PanelDType,
        param_kind: sme.QuantParamKind,
        cache_hint: sme.CacheHint,
        gate_packed: []const u8,
        up_packed: []const u8,
        down_packed: []const u8,
        x_u16: []const u16,
        up_f32_scratch: []f32,
        inter_u16_scratch: []u16,
        out_f32: []f32,
        residual_f32: ?[]const f32,
        m: usize,
        hidden_size: usize,
        inter_size: usize,
    ) sme.Error!void {
        const inter_out_dt: sme.OutputDType = if (dtype.activationIsBf16()) .bf16 else .f16;
        try dispatcher.gemmF32(.{
            .dtype = dtype,
            .param_kind = param_kind,
            .cache_hint = cache_hint,
            .epilogue = .none,
        }, x_u16, up_packed, null, up_f32_scratch, m, inter_size, hidden_size);

        try dispatcher.gemm(.{
            .dtype = dtype,
            .param_kind = param_kind,
            .cache_hint = cache_hint,
            .epilogue = .swiglu_gate_up,
            .out_dtype = inter_out_dt,
        }, x_u16, gate_packed, up_f32_scratch[0 .. m * inter_size], std.mem.sliceAsBytes(inter_u16_scratch[0 .. m * inter_size]), m, inter_size, hidden_size);

        try dispatcher.gemmF32(.{
            .dtype = dtype,
            .param_kind = param_kind,
            .cache_hint = cache_hint,
            .epilogue = if (residual_f32 != null) .add_residual else .none,
        }, inter_u16_scratch[0 .. m * inter_size], down_packed, residual_f32, out_f32, m, hidden_size, inter_size);
    }

    /// Returns the unnormalized top-layer hidden state `[num_nodes * hidden_size]`
    /// written by the most recent `stepDecode`, `prefillToken`, or `verifyTree`.
    pub fn readTopHiddenF16(self: *const Qwen3Model, num_nodes: u32) []const f16 {
        const count = @as(usize, @min(num_nodes, self.max_tree_nodes)) * self.config.hidden_size;
        return self.x_host[0..count];
    }

    /// Returns the concatenated 3-layer EAGLE-3 auxiliary features
    /// `[num_nodes * 3 * hidden_size]` when `enable_feature_taps == true`.
    pub fn readAuxConcatF16(self: *const Qwen3Model, num_nodes: u32) []const f16 {
        if (!self.enable_feature_taps) return &.{};
        const count = @as(usize, @min(num_nodes, self.max_tree_nodes)) * 3 * self.config.hidden_size;
        return self.aux_concat_host[0..count];
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
        std.debug.assert(chunk_idx >= self.num_retired_chunks);
        const active_idx: usize = chunk_idx - self.num_retired_chunks;
        std.debug.assert(active_idx < self.active_kv_chunks);
        const phys: usize = self.active_tile_phys_ring[active_idx];
        const base = (layer_idx * @as(usize, self.active_kv_chunks) + phys) * 2;
        return .{ .k = self.kv_tiles[base], .v = self.kv_tiles[base + 1] };
    }

    fn encodeLayerGqaDecode(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        l: usize,
        cur_seq_len: u32,
        insert_barrier_before: bool,
    ) metal.Error!void {
        const layer = self.layers[l];
        if (self.num_retired_chunks > 0) {
            const pool = self.retired_kv_pool.?;
            const retired_tokens: u64 = @as(u64, self.num_retired_chunks) * @as(u64, self.chunk_tokens);
            try self.attn_pipes.encodeGqaTieredDecodeAttn(frame, .{
                .q = self.q_bind,
                .kv_tiles = if (@as(u64, cur_seq_len) > retired_tokens) layer.kv_tiles_binding else null,
                .retired_pool = pool.binding(self.scratch_buf),
                .retired_chunk_ids = layer.retired_ids_binding,
                .out = self.attn_out_bind,
                .scratch = self.gqa_scratch_bind,
                .max_splits = attention.MAX_SPLITS,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .seq_len = cur_seq_len,
                .chunk_tokens = self.chunk_tokens,
                .num_retired_chunks = self.num_retired_chunks,
                .retired_plane_stride_bytes = pool.plane_stride_bytes,
                .retired_chunk_stride_bytes = pool.chunk_stride_bytes,
            }, insert_barrier_before);
        } else {
            try self.attn_pipes.encodeGqaDecodeAttn(frame, .{
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
            }, insert_barrier_before);
        }
    }

    fn encodeLayerGqaTreeVerify(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        l: usize,
        m: u32,
        base_pos: u32,
        k_tree: metal.BufferBinding,
        v_tree: metal.BufferBinding,
        insert_barrier_before: bool,
    ) metal.Error!void {
        const layer = self.layers[l];
        if (self.num_retired_chunks > 0 and base_pos > 0) {
            const pool = self.retired_kv_pool.?;
            const retired_tokens: u64 = @as(u64, self.num_retired_chunks) * @as(u64, self.chunk_tokens);
            try self.attn_pipes.encodeGqaTieredTreeVerifyAttn(frame, .{
                .q = self.q_bind,
                .kv_tiles = if (@as(u64, base_pos) > retired_tokens) layer.kv_tiles_binding else null,
                .retired_pool = pool.binding(self.scratch_buf),
                .retired_chunk_ids = layer.retired_ids_binding,
                .base_pos = base_pos,
                .k_tree = k_tree,
                .v_tree = v_tree,
                .ancestor_masks = self.tree_ancestor_masks_bind,
                .out = self.attn_out_bind,
                .scratch = self.gqa_tree_scratch_bind,
                .max_splits = attention.MAX_SPLITS,
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .chunk_tokens = self.chunk_tokens,
                .num_retired_chunks = self.num_retired_chunks,
                .retired_plane_stride_bytes = pool.plane_stride_bytes,
                .retired_chunk_stride_bytes = pool.chunk_stride_bytes,
            }, insert_barrier_before);
        } else {
            try self.attn_pipes.encodeGqaTreeVerifyAttn(frame, .{
                .q = self.q_bind,
                .kv_tiles = if (base_pos > 0) layer.kv_tiles_binding else null,
                .base_pos = base_pos,
                .k_tree = k_tree,
                .v_tree = v_tree,
                .ancestor_masks = self.tree_ancestor_masks_bind,
                .out = self.attn_out_bind,
                .scratch = self.gqa_tree_scratch_bind,
                .max_splits = attention.MAX_SPLITS,
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .chunk_tokens = self.chunk_tokens,
            }, insert_barrier_before);
        }
    }

    fn treeKBinding(self: *const Qwen3Model, layer_idx: usize) metal.BufferBinding {
        return .{ .buffer = self.scratch_buf, .offset = self.tree_k_base_off + @as(u64, layer_idx) * self.tree_kv_layer_stride };
    }

    fn treeVBinding(self: *const Qwen3Model, layer_idx: usize) metal.BufferBinding {
        return .{ .buffer = self.scratch_buf, .offset = self.tree_v_base_off + @as(u64, layer_idx) * self.tree_kv_layer_stride };
    }

    pub fn encodeLinearGemv(
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
            .q2_affine => try self.dequant_pipes.encodeGemv(frame, .q2, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
        }
    }

    /// Encodes a linear projection for `m >= 1` rows, using `encodeLinearGemv`
    /// for `m == 1` and weight-stationary bit-exact batched GEMV for `m > 1`.
    pub fn encodeLinearBatched(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        lin: LinearBinding,
        x: metal.BufferBinding,
        y: metal.BufferBinding,
        m: u32,
        insert_barrier_before: bool,
    ) dequant.EncodeError!void {
        if (m == 1) {
            return self.encodeLinearGemv(frame, lin, x, y, insert_barrier_before);
        }
        switch (lin.dtype) {
            .f16 => try self.dequant_pipes.encodeDenseGemvBatch(frame, .f16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
            .bf16 => try self.dequant_pipes.encodeDenseGemvBatch(frame, .bf16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
            .q4_affine => try self.dequant_pipes.encodeGemvBatch(frame, .q4, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
            .q3_affine => try self.dequant_pipes.encodeGemvBatch(frame, .q3, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
            .q2_affine => try self.dequant_pipes.encodeGemvBatch(frame, .q2, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
        }
    }

    pub fn hasColdExperts(self: *const Qwen3Model) bool {
        return self.cold_rt != null;
    }

    pub fn hasHybridPrefill(self: *const Qwen3Model) bool {
        return self.hybrid_rt != null;
    }

    pub fn coldStats(self: *const Qwen3Model) ?ColdExpertStats {
        const crt = self.cold_rt orelse return null;
        return .{
            .total_cold_experts = crt.total_cold_experts,
            .slot_count = crt.pool.numSlots(),
            .io_backend = crt.reader.activeBackend(),
            .ane_predictor_active = crt.ane_predictor != null,
            .ane_predictor_uses_ne = if (crt.ane_predictor) |p| p.usesNeuralEngine() else false,
            .ane_predictor_compile_ms = if (crt.ane_predictor) |p| p.totalCompileMs() else 0.0,
            .cache_telemetry = crt.expert_cache.telemetry(),
        };
    }

    pub fn resetColdTelemetry(self: *Qwen3Model) void {
        if (self.cold_rt) |crt| crt.expert_cache.resetTelemetry();
    }

    /// Populates `out_cold_masks[0..sampled]` with bitsets of currently non-resident
    /// cold experts at the exact `sampled` layers probed by `predictMoEExperts*` (`D7`).
    /// Returns the number of sampled layers populated (`0` when no cold experts exist).
    pub fn fillSampledColdMasks(self: *const Qwen3Model, out_cold_masks: []tree_mod.ExpertBitset) u32 {
        const crt = self.cold_rt orelse return 0;
        if (self.config.arch != .qwen3_moe) return 0;
        const n_layers = self.config.num_hidden_layers;
        const sampled: u32 = @min(n_layers, MAX_EXPERT_PRED_LAYERS);
        if (out_cold_masks.len < sampled) return 0;
        const n_exp = self.config.num_experts;
        for (0..sampled) |s| {
            const l_idx: u16 = @intCast(if (sampled == n_layers) s else (s * (n_layers - 1)) / (sampled - 1));
            var mask = tree_mod.ExpertBitset.empty();
            for (0..n_exp) |e| {
                if (crt.cold_masks_by_layer[l_idx].contains(@intCast(e)) and
                    !crt.expert_cache.isResident(l_idx, @intCast(e)))
                {
                    mask.set(@intCast(e));
                }
            }
            out_cold_masks[s] = mask;
        }
        return sampled;
    }

    /// Encodes a linear projection for `m >= 1` rows, dispatching to GEMV (`m == 1`),
    /// weight-stationary batched GEMV (`2 <= m < 16`), or tiled `simdgroup_float8x8`
    /// GEMM (`m >= 16`, e.g. 512-token hybrid prefill FFN chunks, `G6`).
    pub fn encodeLinearGemm(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        lin: LinearBinding,
        x: metal.BufferBinding,
        y: metal.BufferBinding,
        m: u32,
        insert_barrier_before: bool,
    ) dequant.EncodeError!void {
        if (m < 16) {
            return self.encodeLinearBatched(frame, lin, x, y, m, insert_barrier_before);
        }
        switch (lin.dtype) {
            .f16 => try self.dequant_pipes.encodeDenseGemm(frame, .f16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
            .bf16 => try self.dequant_pipes.encodeDenseGemm(frame, .bf16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
            .q4_affine => try self.dequant_pipes.encodeGemm(frame, .q4, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
            .q3_affine => try self.dequant_pipes.encodeGemm(frame, .q3, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
            .q2_affine => try self.dequant_pipes.encodeGemm(frame, .q2, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, insert_barrier_before),
        }
    }

    /// Dispatches one complete MoE FFN layer onto GPU: preserves the fast
    /// single-dispatch path untouched for uniform layers, and uses G13
    /// zero-sync scheme-bucketed dispatch for heterogeneous layers (`ADR-0017`).
    fn encodeMoELayerDispatch(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        layer: LayerWeights,
        moe_args: moe.MoELayerArgs,
        insert_barrier_before: bool,
    ) moe.EncodeError!void {
        if (layer.is_heterogeneous_moe) {
            try self.moe_pipes.encodeMoELayerHetero(frame, .{
                .base = moe_args,
                .expert_schemes = layer.expert_schemes_bind,
                .scheme_counts = layer.expert_scheme_counts,
            }, insert_barrier_before);
        } else {
            switch (layer.expert_dtype) {
                .q2_affine => try self.moe_pipes.encodeMoELayer(frame, .q2, moe_args, insert_barrier_before),
                .q3_affine => try self.moe_pipes.encodeMoELayer(frame, .q3, moe_args, insert_barrier_before),
                .q4_affine => try self.moe_pipes.encodeMoELayer(frame, .q4, moe_args, insert_barrier_before),
                .f16 => try self.moe_pipes.encodeMoELayerFloat(frame, .f16, moe_args, insert_barrier_before),
                .bf16 => try self.moe_pipes.encodeMoELayerFloat(frame, .bf16, moe_args, insert_barrier_before),
            }
        }
    }

    fn matchesAuxTap(self: *const Qwen3Model, layer_idx: usize) bool {
        if (!self.enable_feature_taps) return false;
        for (self.aux_layer_ids) |tap_l| {
            if (layer_idx == tap_l) return true;
        }
        return false;
    }

    fn encodeAuxTapIfMatched(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        layer_idx: usize,
        num_nodes: u32,
    ) moe.EncodeError!void {
        if (!self.enable_feature_taps) return;
        const h = self.config.hidden_size;
        var barrier = true;
        for (self.aux_layer_ids, 0..) |tap_l, slot| {
            if (layer_idx == tap_l) {
                try self.moe_pipes.encodeCopyStrided(frame, .{
                    .src = self.x_bind,
                    .dst = self.aux_concat_bind,
                    .num_rows = num_nodes,
                    .copy_cols = h,
                    .src_stride = h,
                    .src_col_off = 0,
                    .dst_stride = 3 * h,
                    .dst_col_off = @as(u32, @intCast(slot)) * h,
                }, barrier);
                barrier = false;
            }
        }
    }

    fn runStep(self: *Qwen3Model, token_id: u32, pos: u32, skip_lm_head: bool) StepError!u32 {
        if (token_id >= self.config.vocab_size) return error.TokenOutOfRange;
        if (pos >= self.max_seq_len) return error.PositionOutOfRange;
        try self.retireBeforeWritePos(pos);
        if (self.has_sme2_weights) {
            return self.runStepHeterogeneous(token_id, pos, skip_lm_head);
        }
        if (self.cold_rt != null) {
            return self.runStepCold(token_id, pos, skip_lm_head);
        }

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

            // GQA decode attention over committed IOSurface (and retired q8) KV tiles.
            try self.encodeLayerGqaDecode(&frame, l, cur_seq_len, true);

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
                    try self.encodeMoELayerDispatch(&frame, layer, moe_args, true);
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

            try self.encodeAuxTapIfMatched(&frame, l, 1);
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
        if (!skip_lm_head) self.last_logits_row = 0;
        return if (skip_lm_head) 0 else self.out_token_id_host.*;
    }

    /// Executes one batched `M`-node speculative tree verification forward pass
    /// inside a single reusable Metal 4 command buffer (G5), writing temporary
    /// rotated K/V rows into per-layer scratch buffers without mutating committed
    /// IOSurface `KvTile`s until `commitTreeKv` is called. Zero dynamic heap alloc.
    pub fn verifyTree(
        self: *Qwen3Model,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        compute_logits: bool,
    ) StepError!VerifyTreeResult {
        if (self.has_sme2_weights) {
            return self.verifyTreeHeterogeneous(token_ids, positions, ancestor_masks, base_pos, compute_logits);
        }
        if (self.cold_rt != null) {
            return self.verifyTreeCold(token_ids, positions, ancestor_masks, base_pos, compute_logits);
        }
        if (self.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (token_ids.len == 0 or token_ids.len != positions.len or token_ids.len != ancestor_masks.len) {
            return error.InvalidArgument;
        }
        if (token_ids.len > self.max_tree_nodes) return error.TreeTooLarge;
        if (base_pos > self.max_seq_len) return error.PositionOutOfRange;

        const m: u32 = @intCast(token_ids.len);
        for (0..m) |i| {
            if (token_ids[i] >= self.config.vocab_size) return error.TokenOutOfRange;
            if (positions[i] >= self.max_seq_len) return error.PositionOutOfRange;
        }
        @memcpy(self.tree_token_ids_host[0..m], token_ids);
        @memcpy(self.tree_positions_host[0..m], positions);
        @memcpy(self.tree_ancestor_masks_host[0..m], ancestor_masks);

        const h = self.config.hidden_size;

        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();

        try self.moe_pipes.encodeEmbedLookup(&frame, self.embed.dtype, .{
            .weight = self.embed.weight,
            .scale_bias = self.embed.scale_bias,
            .tokens = .{ .token_ids = self.tree_token_ids_bind },
            .out = self.x_bind,
            .m = m,
            .vocab_size = self.config.vocab_size,
            .hidden_size = h,
        }, false);

        for (self.layers, 0..) |layer, l| {
            try self.attn_pipes.encodeRmsNorm(&frame, .{
                .x = self.x_bind,
                .w = layer.input_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.input_layernorm.dtype,
            }, true);

            try self.encodeLinearBatched(&frame, layer.q_proj, self.x_norm_bind, self.q_bind, m, true);
            try self.encodeLinearBatched(&frame, layer.k_proj, self.x_norm_bind, self.k_bind, m, false);
            try self.encodeLinearBatched(&frame, layer.v_proj, self.x_norm_bind, self.v_bind, m, false);

            const k_tree_bind = self.treeKBinding(l);
            const v_tree_bind = self.treeVBinding(l);
            try self.attn_pipes.encodeTreeQkNormRope(&frame, .{
                .q = self.q_bind,
                .k = self.k_bind,
                .v = self.v_bind,
                .q_norm = layer.q_norm,
                .k_norm = layer.k_norm,
                .positions = self.tree_positions_bind,
                .k_tree_out = k_tree_bind,
                .v_tree_out = v_tree_bind,
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .rms_eps = self.config.rms_norm_eps,
                .rope_theta = self.config.rope_theta,
                .norm_dtype = layer.qk_norm_dtype,
            }, true);

            try self.encodeLayerGqaTreeVerify(&frame, l, m, base_pos, k_tree_bind, v_tree_bind, true);

            try self.encodeLinearBatched(&frame, layer.o_proj, self.attn_out_bind, self.proj_out_bind, m, true);
            try self.moe_pipes.encodeResidualAdd(&frame, .{
                .x = self.x_bind,
                .delta = self.proj_out_bind,
                .count = m * h,
            }, true);

            try self.attn_pipes.encodeRmsNorm(&frame, .{
                .x = self.x_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, true);

            switch (self.config.arch) {
                .qwen3_moe => {
                    // Use per-layer ScheduleHeader offset so all layers' unique active
                    // expert counts remain readable after the single command buffer completes.
                    var layer_ws = self.moe_ws.?;
                    layer_ws.schedule_header_off = self.layer_sched_hdr_base_off + @as(u64, l) * 64;
                    const moe_args: moe.MoELayerArgs = .{
                        .x = self.x_norm_bind,
                        .router = layer.router,
                        .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                        .expert_table = layer.expert_table,
                        .workspace = &layer_ws,
                        .workspace_buf = self.scratch_buf,
                        .residual = self.x_bind,
                        .out = self.x_bind,
                        .m = m,
                        .num_experts = self.config.num_experts,
                        .top_k = self.config.num_experts_per_tok,
                        .hidden_size = h,
                        .intermediate_size = self.config.moe_intermediate_size,
                        .norm_topk_prob = self.config.norm_topk_prob,
                    };
                    try self.encodeMoELayerDispatch(&frame, layer, moe_args, true);
                },
                .qwen3_dense => {
                    try self.encodeLinearBatched(&frame, layer.dense_gate, self.x_norm_bind, self.dense_gate_bind, m, true);
                    try self.encodeLinearBatched(&frame, layer.dense_up, self.x_norm_bind, self.dense_up_bind, m, false);
                    try self.moe_pipes.encodeSiluMul(&frame, .{
                        .gate = self.dense_gate_bind,
                        .up = self.dense_up_bind,
                        .out = self.dense_act_bind,
                        .count = m * self.config.intermediate_size,
                    }, true);
                    try self.encodeLinearBatched(&frame, layer.dense_down, self.dense_act_bind, self.proj_out_bind, m, true);
                    try self.moe_pipes.encodeResidualAdd(&frame, .{
                        .x = self.x_bind,
                        .delta = self.proj_out_bind,
                        .count = m * h,
                    }, true);
                },
            }

            try self.encodeAuxTapIfMatched(&frame, l, m);
        }

        if (compute_logits) {
            try self.attn_pipes.encodeRmsNorm(&frame, .{
                .x = self.x_bind,
                .w = self.final_norm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = self.final_norm.dtype,
            }, true);
            try self.encodeLinearBatched(&frame, self.lm_head, self.x_norm_bind, self.logits_bind, m, true);
            try self.moe_pipes.encodeArgmaxRows(&frame, .{
                .logits = self.logits_bind,
                .out_token_ids = self.out_token_id_bind,
                .num_rows = m,
                .vocab_size = self.config.vocab_size,
                .logits_dtype = .f16,
            }, true);
        }

        const sig = try frame.commit();
        try self.compute.wait(sig, self.timeout_ms);
        if (compute_logits) self.last_logits_row = @as(usize, m) - 1;

        var total_unique: u64 = 0;
        if (self.config.arch == .qwen3_moe) {
            for (0..self.layers.len) |l| {
                const hdr: *const moe.ScheduleHeader = @ptrCast(@alignCast(self.layer_sched_hdr_raw.ptr + l * 64));
                total_unique += hdr.num_unique_experts;
            }
        }
        const avg_unique: f64 = if (self.config.arch == .qwen3_moe)
            @as(f64, @floatFromInt(total_unique)) / @as(f64, @floatFromInt(self.layers.len))
        else
            0.0;

        const kv_per_layer: u64 =
            2 * @as(u64, self.config.num_key_value_heads) * (@as(u64, base_pos) + m) * self.config.head_dim * @sizeOf(f16);
        const kv_total: u64 = @as(u64, self.config.num_hidden_layers) * kv_per_layer;
        const head_bytes: u64 = if (compute_logits)
            @as(u64, m) * self.embed_row_alg_bytes + (self.global_head_bytes - self.embed_row_alg_bytes)
        else
            @as(u64, m) * self.embed_row_alg_bytes;
        const layer_weight_bytes: u64 = if (self.config.arch == .qwen3_moe)
            self.total_layers_non_expert_bytes + total_unique * self.single_expert_bytes
        else
            self.total_layers_static_bytes;

        return .{
            .num_nodes = m,
            .argmax_tokens = self.out_token_ids_host[0..m],
            .logits_f16 = self.logits_host[0 .. @as(usize, m) * self.config.vocab_size],
            .total_unique_experts = total_unique,
            .avg_unique_experts_per_layer = avg_unique,
            .actual_bytes_moved = layer_weight_bytes + kv_total + head_bytes,
        };
    }

    /// Commits ONLY the `accepted_nodes` KV rows from the most recent `verifyTree`
    /// pass into committed IOSurface `RGBA16Float` `KvTile`s at sequence positions
    /// `base_pos .. base_pos + accepted_nodes.len`, handling chunk boundaries cleanly,
    /// and gathers accepted target feature taps (`accepted_aux_bind`, `accepted_top_bind`)
    /// in the same Metal 4 command buffer. Zero dynamic heap allocation.
    pub fn commitTreeKv(
        self: *Qwen3Model,
        base_pos: u32,
        accepted_nodes: []const u16,
    ) StepError!void {
        if (self.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (accepted_nodes.len == 0 or accepted_nodes.len > self.max_tree_nodes) return error.InvalidArgument;
        const accepted_count: u32 = @intCast(accepted_nodes.len);
        if (base_pos + accepted_count > self.max_seq_len) return error.PositionOutOfRange;

        for (accepted_nodes, 0..) |node_idx, i| {
            if (node_idx >= self.max_tree_nodes) return error.InvalidArgument;
            self.tree_accepted_nodes_host[i] = node_idx;
        }
        try self.retireBeforeCommitRange(base_pos, accepted_count);

        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();

        // Commit accepted K/V rows per layer; loop handles even tiny test chunk sizes
        // where `accepted_count` might span >2 chunks (for 512-token chunks it runs once).
        var first_dispatch = true;
        var offset: u32 = 0;
        while (offset < accepted_count) {
            const cur_base = base_pos + offset;
            const chunk0: u32 = cur_base / self.chunk_tokens;
            const max_end_in_two_chunks: u32 = (chunk0 + 2) * self.chunk_tokens;
            const remain = accepted_count - offset;
            const take: u32 = @min(remain, max_end_in_two_chunks - cur_base);
            const crosses = ((cur_base + take - 1) / self.chunk_tokens) > chunk0;

            for (0..self.layers.len) |l| {
                const t0 = self.kvTileAt(l, @intCast(chunk0));
                const t1 = if (crosses) self.kvTileAt(l, @intCast(chunk0 + 1)) else null;
                try self.attn_pipes.encodeKvTreeCommit(&frame, .{
                    .k_tree = self.treeKBinding(l),
                    .v_tree = self.treeVBinding(l),
                    .accepted_nodes = .{ .inline_u16 = accepted_nodes[offset .. offset + take] },
                    .accepted_count = take,
                    .k_tile0 = t0.k,
                    .v_tile0 = t0.v,
                    .k_tile1 = if (t1) |p| p.k else null,
                    .v_tile1 = if (t1) |p| p.v else null,
                    .num_kv_heads = self.config.num_key_value_heads,
                    .head_dim = self.config.head_dim,
                    .base_pos = cur_base,
                    .chunk_tokens = self.chunk_tokens,
                }, !first_dispatch);
                first_dispatch = false;
            }
            offset += take;
        }

        if (self.enable_feature_taps) {
            const h = self.config.hidden_size;
            try self.moe_pipes.encodeGatherRows(&frame, .{
                .src = self.aux_concat_bind,
                .indices = self.tree_accepted_nodes_bind,
                .dst = self.accepted_aux_bind,
                .count = accepted_count,
                .row_elems = 3 * h,
            }, false);
            try self.moe_pipes.encodeGatherRows(&frame, .{
                .src = self.x_bind,
                .indices = self.tree_accepted_nodes_bind,
                .dst = self.accepted_top_bind,
                .count = accepted_count,
                .row_elems = h,
            }, false);
        }

        const sig = try frame.commit();
        try self.compute.wait(sig, self.timeout_ms);
        self.seq_len = base_pos + accepted_count;
        try self.retireExcessCompletedChunks();
    }

    /// Batched causal prompt prefill chunk (`1 <= tokens.len <= max_tree_nodes`)
    /// reusing `verifyTree` + `commitTreeKv` with zero heap allocation.
    pub fn prefillChunk(
        self: *Qwen3Model,
        tokens: []const u32,
        start_pos: u32,
        compute_logits: bool,
    ) StepError!u32 {
        if (self.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (tokens.len == 0 or tokens.len > self.max_tree_nodes) return error.TreeTooLarge;
        const m: usize = tokens.len;
        var positions: [attention.MAX_TREE_NODES]u32 = undefined;
        var masks: [attention.MAX_TREE_NODES]u64 = undefined;
        var accepted: [attention.MAX_TREE_NODES]u16 = undefined;
        for (0..m) |i| {
            positions[i] = start_pos + @as(u32, @intCast(i));
            masks[i] = if (i == 63) ~@as(u64, 0) else (@as(u64, 1) << @as(u6, @intCast(i + 1))) - 1;
            accepted[i] = @intCast(i);
        }
        if (self.has_pure_cpu_sme2_dense) {
            const cands = try self.stepDraftTreeBatchSme2(
                tokens,
                positions[0..m],
                0,
                masks[0..m],
                start_pos,
                if (compute_logits) 1 else 0,
                true,
                true,
            );
            return if (compute_logits) cands[0].token_id else 0;
        }
        const vr = try self.verifyTree(tokens, positions[0..m], masks[0..m], start_pos, compute_logits);
        try self.commitTreeKv(start_pos, accepted[0..m]);
        return if (compute_logits) vr.argmax_tokens[m - 1] else 0;
    }

    fn runPredictMoEExpertsFrame(
        self: *Qwen3Model,
        frame: *metal.Compute.Frame,
        hidden_bind: metal.BufferBinding,
        num_nodes: u32,
        sampled: u32,
        initial_barrier: bool,
    ) StepError!void {
        const n_layers = self.config.num_hidden_layers;
        const h = self.config.hidden_size;
        const top_k = self.config.num_experts_per_tok;
        const base_ws = self.moe_ws.?;

        for (0..sampled) |s| {
            const l_idx: usize = if (sampled == n_layers)
                s
            else
                (@as(usize, s) * (n_layers - 1)) / (sampled - 1);
            const layer = self.layers[l_idx];

            try self.attn_pipes.encodeRmsNorm(frame, .{
                .x = hidden_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = num_nodes,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, initial_barrier or s > 0);

            var ws = base_ws;
            ws.scratch_topk_experts_off = self.pred_topk_base_off + @as(u64, s) * self.pred_topk_layer_stride;
            const logits_bind = ws.routerLogitsBinding(self.scratch_buf);
            switch (layer.router) {
                .weights => |w| try self.moe_pipes.encodeRouterProj(frame, w.dtype, .{
                    .x = self.x_norm_bind,
                    .router_weights = w.binding,
                    .logits = logits_bind,
                    .m = num_nodes,
                    .num_experts = self.config.num_experts,
                    .hidden_size = h,
                }, true),
                .quant_weights => |q| try self.moe_pipes.encodeRouterProjQuant(frame, q.bits, .{
                    .x = self.x_norm_bind,
                    .qweight = q.qweight,
                    .scale_bias = q.scale_bias,
                    .logits = logits_bind,
                    .m = num_nodes,
                    .num_experts = self.config.num_experts,
                    .hidden_size = h,
                }, true),
                .logits => return error.InvalidArgument,
            }
            try self.moe_pipes.encodeRouteTopK(frame, .f32, .{
                .logits = logits_bind,
                .workspace = &ws,
                .workspace_buf = self.scratch_buf,
                .m = num_nodes,
                .num_experts = self.config.num_experts,
                .top_k = top_k,
                .norm_topk_prob = self.config.norm_topk_prob,
            }, true);
        }
    }

    fn unpackPredictedBitsets(
        self: *const Qwen3Model,
        num_nodes: u32,
        sampled: u32,
        out_bitsets: []tree_mod.ExpertBitset,
    ) void {
        const top_k = self.config.num_experts_per_tok;
        const stride_u32: usize = @intCast(self.pred_topk_layer_stride / @sizeOf(u32));
        for (0..num_nodes) |v| {
            for (0..sampled) |s| {
                var bs = tree_mod.ExpertBitset.empty();
                const row_base = s * stride_u32 + v * @as(usize, top_k);
                for (0..top_k) |k_idx| {
                    const eid = self.pred_topk_host[row_base + k_idx];
                    if (eid < tree_mod.ExpertBitset.MAX_EXPERTS) {
                        bs.set(eid);
                    }
                }
                out_bitsets[v * sampled + s] = bs;
            }
        }
    }

    /// Predicts per-node MoE expert bitsets across up to `MAX_EXPERT_PRED_LAYERS`
    /// evenly spaced target layers by applying the target model's own
    /// `post_attention_layernorm` + router gate to `hidden_bind [num_nodes, H]`
    /// inside a single tiny Metal 4 frame (D7 / ADR-0009).
    ///
    /// Writes `out_bitsets[v * sampled_layers + s]` and returns `sampled_layers`
    /// (`0` for dense models). Zero dynamic heap allocation.
    pub fn predictMoEExperts(
        self: *Qwen3Model,
        hidden_bind: metal.BufferBinding,
        num_nodes: u32,
        out_bitsets: []tree_mod.ExpertBitset,
    ) StepError!u32 {
        if (self.config.arch != .qwen3_moe or num_nodes == 0) return 0;
        if (self.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (num_nodes > self.max_tree_nodes) return error.TreeTooLarge;

        const sampled: u32 = @min(self.config.num_hidden_layers, MAX_EXPERT_PRED_LAYERS);
        if (out_bitsets.len < @as(usize, num_nodes) * sampled) return error.InvalidArgument;

        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();
        try self.runPredictMoEExpertsFrame(&frame, hidden_bind, num_nodes, sampled, false);
        const sig = try frame.commit();
        try self.compute.wait(sig, self.timeout_ms);

        self.unpackPredictedBitsets(num_nodes, sampled, out_bitsets);
        return sampled;
    }

    /// Embedding-based expert prediction for tree token IDs across up to
    /// `MAX_EXPERT_PRED_LAYERS` evenly spaced target layers (`D7`, `C13`).
    /// Dispatches to the L2-resident ARMv9 SME2 router bank (zero Metal 4
    /// command-buffer round-trip) when calibrated/enabled, otherwise runs in a
    /// single Metal 4 GPU frame. Zero dynamic heap allocation.
    pub fn predictMoEExpertsForTokens(
        self: *Qwen3Model,
        token_ids: []const u32,
        out_bitsets: []tree_mod.ExpertBitset,
    ) StepError!u32 {
        if (self.sme_rt) |srt| {
            if (srt.router_pred_backend != .gpu) {
                return self.predictMoEExpertsForTokensSme2(token_ids, out_bitsets);
            }
        }
        return self.predictMoEExpertsForTokensGpu(token_ids, out_bitsets);
    }

    /// Explicit GPU Metal 4 path for `predictMoEExpertsForTokens` (baseline & M1–M3 fallback).
    pub fn predictMoEExpertsForTokensGpu(
        self: *Qwen3Model,
        token_ids: []const u32,
        out_bitsets: []tree_mod.ExpertBitset,
    ) StepError!u32 {
        if (self.config.arch != .qwen3_moe or token_ids.len == 0) return 0;
        if (self.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (token_ids.len > self.max_tree_nodes) return error.TreeTooLarge;
        const m: u32 = @intCast(token_ids.len);
        const sampled: u32 = @min(self.config.num_hidden_layers, MAX_EXPERT_PRED_LAYERS);
        if (out_bitsets.len < @as(usize, m) * sampled) return error.InvalidArgument;
        for (token_ids) |tid| {
            if (tid >= self.config.vocab_size) return error.TokenOutOfRange;
        }
        @memcpy(self.tree_token_ids_host[0..m], token_ids);

        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();
        try self.moe_pipes.encodeEmbedLookup(&frame, self.embed.dtype, .{
            .weight = self.embed.weight,
            .scale_bias = self.embed.scale_bias,
            .tokens = .{ .token_ids = self.tree_token_ids_bind },
            .out = self.x_bind,
            .m = m,
            .vocab_size = self.config.vocab_size,
            .hidden_size = self.config.hidden_size,
        }, false);
        try self.runPredictMoEExpertsFrame(&frame, self.x_bind, m, sampled, true);
        const sig = try frame.commit();
        try self.compute.wait(sig, self.timeout_ms);

        self.unpackPredictedBitsets(m, sampled, out_bitsets);
        return sampled;
    }

    fn lookupEmbeddingsCpuF32(self: *const Qwen3Model, token_ids: []const u32, out_f32: []f32) StepError!void {
        const h: usize = self.config.hidden_size;
        const shard_bytes = self.shards[self.embed_shard_idx].arena.bytes();
        const w_off: usize = @intCast(self.embed.weight.offset);

        switch (self.embed.dtype) {
            .f16 => {
                for (token_ids, 0..) |tid, r| {
                    const row_off = w_off + @as(usize, tid) * h * @sizeOf(f16);
                    const src: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, shard_bytes[row_off .. row_off + h * 2]));
                    const dst = out_f32[r * h .. (r + 1) * h];
                    for (0..h) |c| dst[c] = @floatCast(src[c]);
                }
            },
            .bf16 => {
                for (token_ids, 0..) |tid, r| {
                    const row_off = w_off + @as(usize, tid) * h * @sizeOf(u16);
                    const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, shard_bytes[row_off .. row_off + h * 2]));
                    const dst = out_f32[r * h .. (r + 1) * h];
                    for (0..h) |c| dst[c] = quant.bf16ToF32(src[c]);
                }
            },
            .q4_affine => {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                const groups = h / 64;
                const row_bytes = groups * Ops.groupWeightBytes();
                const sb_off: usize = @intCast(self.embed.scale_bias.?.offset);
                for (token_ids, 0..) |tid, r| {
                    const row_w = w_off + @as(usize, tid) * row_bytes;
                    const row_sb = sb_off + @as(usize, tid) * groups * @sizeOf(quant.ScaleBias);
                    const sbs: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                        quant.ScaleBias,
                        shard_bytes[row_sb .. row_sb + groups * @sizeOf(quant.ScaleBias)],
                    ));
                    const dst = out_f32[r * h .. (r + 1) * h];
                    for (0..groups) |g| {
                        const goff = row_w + g * Ops.groupWeightBytes();
                        Ops.dequantizeGroup(shard_bytes[goff..][0..Ops.groupWeightBytes()], sbs[g], dst[g * 64 ..][0..64]);
                    }
                }
            },
            .q3_affine => {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                const groups = h / 64;
                const row_bytes = groups * Ops.groupWeightBytes();
                const sb_off: usize = @intCast(self.embed.scale_bias.?.offset);
                for (token_ids, 0..) |tid, r| {
                    const row_w = w_off + @as(usize, tid) * row_bytes;
                    const row_sb = sb_off + @as(usize, tid) * groups * @sizeOf(quant.ScaleBias);
                    const sbs: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                        quant.ScaleBias,
                        shard_bytes[row_sb .. row_sb + groups * @sizeOf(quant.ScaleBias)],
                    ));
                    const dst = out_f32[r * h .. (r + 1) * h];
                    for (0..groups) |g| {
                        const goff = row_w + g * Ops.groupWeightBytes();
                        Ops.dequantizeGroup(shard_bytes[goff..][0..Ops.groupWeightBytes()], sbs[g], dst[g * 64 ..][0..64]);
                    }
                }
            },
            .q2_affine => {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                const groups = h / 64;
                const row_bytes = groups * Ops.groupWeightBytes();
                const sb_off: usize = @intCast(self.embed.scale_bias.?.offset);
                for (token_ids, 0..) |tid, r| {
                    const row_w = w_off + @as(usize, tid) * row_bytes;
                    const row_sb = sb_off + @as(usize, tid) * groups * @sizeOf(quant.ScaleBias);
                    const sbs: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                        quant.ScaleBias,
                        shard_bytes[row_sb .. row_sb + groups * @sizeOf(quant.ScaleBias)],
                    ));
                    const dst = out_f32[r * h .. (r + 1) * h];
                    for (0..groups) |g| {
                        const goff = row_w + g * Ops.groupWeightBytes();
                        Ops.dequantizeGroup(shard_bytes[goff..][0..Ops.groupWeightBytes()], sbs[g], dst[g * 64 ..][0..64]);
                    }
                }
            },
        }
    }

    fn selectTopKBitset(logits: []const f32, top_k: u32) tree_mod.ExpertBitset {
        var best_idx: [moe.MAX_TOP_K]u32 = @splat(0);
        var best_val: [moe.MAX_TOP_K]f32 = @splat(-std.math.inf(f32));
        const k: usize = @intCast(@min(top_k, moe.MAX_TOP_K));
        for (logits, 0..) |v, ei| {
            if (v > best_val[k - 1]) {
                var pos: usize = k - 1;
                while (pos > 0 and v > best_val[pos - 1]) : (pos -= 1) {
                    best_val[pos] = best_val[pos - 1];
                    best_idx[pos] = best_idx[pos - 1];
                }
                best_val[pos] = v;
                best_idx[pos] = @intCast(ei);
            }
        }
        var bs = tree_mod.ExpertBitset.empty();
        for (0..k) |j| {
            if (best_idx[j] < tree_mod.ExpertBitset.MAX_EXPERTS) {
                bs.set(best_idx[j]);
            }
        }
        return bs;
    }

    /// L2-resident ARMv9 SME2 path for `predictMoEExpertsForTokens` (`C13`):
    /// reads token embeddings directly from unified wired RAM, normalizes in registers,
    /// and executes the sampled layers' router gates on P-core SME2 with ZERO Metal 4
    /// command-buffer round-trip and ZERO dynamic heap allocation.
    pub fn predictMoEExpertsForTokensSme2(
        self: *Qwen3Model,
        token_ids: []const u32,
        out_bitsets: []tree_mod.ExpertBitset,
    ) StepError!u32 {
        if (self.config.arch != .qwen3_moe or token_ids.len == 0) return 0;
        if (self.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (token_ids.len > self.max_tree_nodes) return error.TreeTooLarge;
        const srt = self.sme_rt orelse return self.predictMoEExpertsForTokensGpu(token_ids, out_bitsets);
        const m: usize = token_ids.len;
        const sampled: u32 = @min(self.config.num_hidden_layers, MAX_EXPERT_PRED_LAYERS);
        if (out_bitsets.len < m * sampled) return error.InvalidArgument;
        for (token_ids) |tid| {
            if (tid >= self.config.vocab_size) return error.TokenOutOfRange;
        }

        const h: usize = self.config.hidden_size;
        const n_exp: usize = self.config.num_experts;
        const top_k: u32 = self.config.num_experts_per_tok;
        const n_layers: usize = self.layers.len;

        const x_f32 = srt.x_f32_scratch[0 .. m * h];
        try self.lookupEmbeddingsCpuF32(token_ids, x_f32);

        var inv_rms_buf: [attention.MAX_TREE_NODES]f32 = undefined;
        const inv_h: f32 = 1.0 / @as(f32, @floatFromInt(h));
        for (0..m) |r| {
            const row = x_f32[r * h .. (r + 1) * h];
            var sum_sq: f32 = 0.0;
            for (row) |v| sum_sq += v * v;
            inv_rms_buf[r] = 1.0 / @sqrt(sum_sq * inv_h + self.config.rms_norm_eps);
        }

        if (!srt.router_bank_single_owner and srt.router_bank_packed.len > 0) {
            // Single stacked `[sampled * num_experts, H]` f16/bf16 `sme2_panel64` GEMM across
            // all sampled layers (post_attention_layernorm folded into weights).
            const x_act = srt.x_act_u16_scratch[0 .. m * h];
            const act_fdt: quant.FloatDType = if (srt.router_bank_dtype.activationIsBf16()) .bf16 else .f16;
            for (0..m) |r| {
                const row = x_f32[r * h .. (r + 1) * h];
                const dst = x_act[r * h .. (r + 1) * h];
                const scale = inv_rms_buf[r];
                for (0..h) |c| dst[c] = quant.f32ToFloatBits(act_fdt, row[c] * scale);
            }
            const total_n: usize = @as(usize, sampled) * n_exp;
            const logits_out = srt.router_logits_f32[0 .. m * total_n];
            try srt.dispatcher.gemmF32(.{
                .dtype = srt.router_bank_dtype,
                .param_kind = .affine_scale_bias,
                .cache_hint = .temporal,
            }, x_act, srt.router_bank_packed, null, logits_out, m, total_n, h);

            for (0..m) |r| {
                for (0..sampled) |s| {
                    const row_logits = logits_out[(r * sampled + s) * n_exp ..][0..n_exp];
                    out_bitsets[r * sampled + s] = selectTopKBitset(row_logits, top_k);
                }
            }
            return sampled;
        }

        // Single-owner `sme2_panel64` router path: read each sampled layer's `.amf` router
        // blob directly from `WiredArena` with ZERO duplicate RAM copies!
        const logits_out = srt.router_logits_f32[0 .. m * n_exp];
        const x_act = srt.x_act_u16_scratch[0 .. m * h];
        for (0..sampled) |s| {
            const l_idx: usize = if (sampled == n_layers) s else (s * (n_layers - 1)) / (sampled - 1);
            const layer = self.layers[l_idx];
            const pdt = layer.router_panel_dtype orelse return error.InvalidDType;
            const packed_w = layer.router_sme_packed orelse return error.InvalidArgument;
            const act_fdt: quant.FloatDType = if (pdt.activationIsBf16()) .bf16 else .f16;

            const shard_bytes = self.shards[layer.shard_idx].arena.bytes();
            const ln_off: usize = @intCast(layer.post_attention_layernorm.binding.offset);
            const ln_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, shard_bytes[ln_off .. ln_off + h * 2]));

            for (0..m) |r| {
                const row = x_f32[r * h .. (r + 1) * h];
                const dst = x_act[r * h .. (r + 1) * h];
                const scale = inv_rms_buf[r];
                for (0..h) |c| {
                    const w_ln = quant.floatBitsToF32(layer.post_attention_layernorm.dtype, ln_u16[c]);
                    dst[c] = quant.f32ToFloatBits(act_fdt, row[c] * scale * w_ln);
                }
            }

            try srt.dispatcher.gemmF32(.{
                .dtype = pdt,
                .param_kind = amfParamKindToSme(layer.router_param_kind),
                .cache_hint = .temporal,
            }, x_act, packed_w, null, logits_out, m, n_exp, h);

            for (0..m) |r| {
                out_bitsets[r * sampled + s] = selectTopKBitset(logits_out[r * n_exp ..][0..n_exp], top_k);
            }
        }
        return sampled;
    }

    // -----------------------------------------------------------------------
    // M4.5 Single-Owner Heterogeneous CPU SME2 + Metal GPU Execution (W15, C12-C13)
    // -----------------------------------------------------------------------

    fn runLinearSme2(
        self: *Qwen3Model,
        lin: LinearBinding,
        x_f16: []const f16,
        y_f16: []f16,
        m: u32,
        cache_hint: sme.CacheHint,
    ) StepError!void {
        const srt = self.sme_rt.?;
        const pdt = weightDTypeToPanelDType(lin.dtype);
        const packed_w = lin.sme_packed.?;
        const m_u: usize = m;
        const k_u: usize = lin.k;
        const n_u: usize = lin.n;
        const a_u16: []const u16 = if (pdt.activationIsBf16()) blk: {
            const dst = srt.x_act_u16_scratch[0 .. m_u * k_u];
            for (0..m_u * k_u) |i| {
                dst[i] = quant.f32ToBf16(@floatCast(x_f16[i]));
            }
            break :blk dst;
        } else @alignCast(std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(x_f16[0 .. m_u * k_u])));

        try srt.dispatcher.gemm(.{
            .dtype = pdt,
            .param_kind = amfParamKindToSme(lin.param_kind),
            .cache_hint = cache_hint,
            .epilogue = .none,
            .out_dtype = .f16,
        }, a_u16, packed_w, null, std.mem.sliceAsBytes(y_f16[0 .. m_u * n_u]), m_u, n_u, k_u);
    }

    fn runRouterSme2(
        self: *Qwen3Model,
        layer: LayerWeights,
        x_f16: []const f16,
        out_logits_f32: []f32,
        m: u32,
    ) StepError!void {
        const srt = self.sme_rt.?;
        const pdt = layer.router_panel_dtype.?;
        const packed_w = layer.router_sme_packed.?;
        const m_u: usize = m;
        const h: usize = self.config.hidden_size;
        const n_exp: usize = self.config.num_experts;
        const a_u16: []const u16 = if (pdt.activationIsBf16()) blk: {
            const dst = srt.x_act_u16_scratch[0 .. m_u * h];
            for (0..m_u * h) |i| {
                dst[i] = quant.f32ToBf16(@floatCast(x_f16[i]));
            }
            break :blk dst;
        } else @alignCast(std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(x_f16[0 .. m_u * h])));

        try srt.dispatcher.gemmF32(.{
            .dtype = pdt,
            .param_kind = amfParamKindToSme(layer.router_param_kind),
            .cache_hint = .temporal,
        }, a_u16, packed_w, null, out_logits_f32[0 .. m_u * n_exp], m_u, n_exp, h);
    }

    /// Executes `sme2_panel64` MoE experts directly on CPU SME2 over unified shared
    /// memory using `.temporal` (`ld1*`) for hot experts and `.non_temporal` (`ldnt1*`)
    /// for cold experts (`C12`/`C13`), adding the weighted SwiGLU sum into `self.x_host`.
    /// Returns the measured unique active expert count across the `m` tokens.
    fn runMoeExpertsSme2(
        self: *Qwen3Model,
        layer: LayerWeights,
        router_logits_f32: []const f32,
        x_norm_f16: []const f16,
        x_residual_io: []f16,
        m: u32,
    ) StepError!u32 {
        const srt = self.sme_rt.?;
        const experts = layer.expert_sme_blobs.?;
        const pdt = weightDTypeToPanelDType(layer.expert_dtype);
        const pkind = amfParamKindToSme(layer.expert_param_kind);
        const m_u: usize = m;
        const h: usize = self.config.hidden_size;
        const inter: usize = self.config.moe_intermediate_size;
        const n_exp: usize = self.config.num_experts;
        const top_k: usize = self.config.num_experts_per_tok;

        var used_mask = tree_mod.ExpertBitset.empty();
        var tok_top_e: [attention.MAX_TREE_NODES][moe.MAX_TOP_K]u32 = undefined;
        var tok_top_w: [attention.MAX_TREE_NODES][moe.MAX_TOP_K]f32 = undefined;

        for (0..m_u) |r| {
            const row = router_logits_f32[r * n_exp ..][0..n_exp];
            var max_v: f32 = -std.math.inf(f32);
            for (row) |v| {
                if (v > max_v) max_v = v;
            }
            var sum_exp: f64 = 0.0;
            var probs: [moe.MAX_EXPERTS]f32 = undefined;
            for (row, 0..) |v, e| {
                const p = @exp(@as(f64, v - max_v));
                probs[e] = @floatCast(p);
                sum_exp += p;
            }
            const inv_sum: f64 = if (sum_exp > 0.0) 1.0 / sum_exp else 0.0;
            for (0..n_exp) |e| probs[e] = @floatCast(@as(f64, probs[e]) * inv_sum);

            var best_e: [moe.MAX_TOP_K]u32 = @splat(0);
            var best_p: [moe.MAX_TOP_K]f32 = @splat(-std.math.inf(f32));
            for (0..n_exp) |e| {
                const p = probs[e];
                if (p > best_p[top_k - 1]) {
                    var pos: usize = top_k - 1;
                    while (pos > 0 and p > best_p[pos - 1]) : (pos -= 1) {
                        best_p[pos] = best_p[pos - 1];
                        best_e[pos] = best_e[pos - 1];
                    }
                    best_p[pos] = p;
                    best_e[pos] = @intCast(e);
                }
            }
            if (self.config.norm_topk_prob) {
                var w_sum: f64 = 0.0;
                for (0..top_k) |j| w_sum += @as(f64, best_p[j]);
                const inv_w: f64 = if (w_sum > 0.0) 1.0 / w_sum else 0.0;
                for (0..top_k) |j| best_p[j] = @floatCast(@as(f64, best_p[j]) * inv_w);
            }
            for (0..top_k) |j| {
                tok_top_e[r][j] = best_e[j];
                tok_top_w[r][j] = best_p[j];
                used_mask.set(best_e[j]);
            }
        }

        // Convert `x_norm_f16` once into the activation dtype expected by `pdt`,
        // and initialize FP32 residual accumulator from `x_residual_io`.
        const x_act_all = srt.x_act_u16_scratch[0 .. m_u * h];
        const act_fdt: quant.FloatDType = if (pdt.activationIsBf16()) .bf16 else .f16;
        for (0..m_u * h) |i| {
            x_act_all[i] = quant.f32ToFloatBits(act_fdt, @floatCast(x_norm_f16[i]));
            srt.expert_acc_f32[i] = @floatCast(x_residual_io[i]);
        }

        // Deduplicated per-expert evaluation: each unique active expert is streamed ONCE!
        var active_rows: [attention.MAX_TREE_NODES]usize = undefined;
        var active_weights: [attention.MAX_TREE_NODES]f32 = undefined;
        for (0..n_exp) |eid| {
            if (!used_mask.contains(@intCast(eid))) continue;
            var cnt: usize = 0;
            for (0..m_u) |r| {
                for (0..top_k) |j| {
                    if (tok_top_e[r][j] == eid) {
                        active_rows[cnt] = r;
                        active_weights[cnt] = tok_top_w[r][j];
                        cnt += 1;
                    }
                }
            }
            if (cnt == 0) continue;

            // Gather this expert's rows into contiguous `x_f32_scratch` (reinterpreted as u16)
            // or slice directly when `cnt == 1` and activation dtype matches (`M = 1` zero-copy fast path!).
            const e_triple = experts[eid];
            const e_pdt = if (layer.is_heterogeneous_moe) weightDTypeToPanelDType(e_triple.dtype) else pdt;
            const e_pkind = if (layer.is_heterogeneous_moe) amfParamKindToSme(e_triple.param_kind) else pkind;
            const e_act_matches = (e_pdt.activationIsBf16() == pdt.activationIsBf16());
            const hint: sme.CacheHint = if (e_triple.is_cold) .non_temporal else .temporal;
            if (cnt == 1 and e_act_matches) {
                const r = active_rows[0];
                try evalSwiGluExpertSme2(
                    &srt.dispatcher,
                    e_pdt,
                    e_pkind,
                    hint,
                    e_triple.gate,
                    e_triple.up,
                    e_triple.down,
                    x_act_all[r * h .. (r + 1) * h],
                    srt.expert_up_f32[0..inter],
                    srt.expert_inter_u16[0..inter],
                    srt.expert_down_f32[0..h],
                    null,
                    1,
                    h,
                    inter,
                );
                const w = active_weights[0];
                const acc_row = srt.expert_acc_f32[r * h .. (r + 1) * h];
                for (0..h) |c| acc_row[c] += w * srt.expert_down_f32[c];
            } else {
                // Pack `cnt` active token rows into `srt.x_f32_scratch` raw bytes (`cnt * h * 2 <= m * h * 4`).
                const x_gather: []u16 = @alignCast(std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(srt.x_f32_scratch))[0 .. cnt * h]);
                if (e_act_matches) {
                    for (0..cnt) |ci| {
                        const r = active_rows[ci];
                        @memcpy(x_gather[ci * h .. (ci + 1) * h], x_act_all[r * h .. (r + 1) * h]);
                    }
                } else {
                    const e_act_fdt: quant.FloatDType = if (e_pdt.activationIsBf16()) .bf16 else .f16;
                    for (0..cnt) |ci| {
                        const r = active_rows[ci];
                        const src_row = x_norm_f16[r * h .. (r + 1) * h];
                        const dst_row = x_gather[ci * h .. (ci + 1) * h];
                        for (0..h) |c| dst_row[c] = quant.f32ToFloatBits(e_act_fdt, @floatCast(src_row[c]));
                    }
                }
                try evalSwiGluExpertSme2(
                    &srt.dispatcher,
                    e_pdt,
                    e_pkind,
                    hint,
                    e_triple.gate,
                    e_triple.up,
                    e_triple.down,
                    x_gather,
                    srt.expert_up_f32[0 .. cnt * inter],
                    srt.expert_inter_u16[0 .. cnt * inter],
                    srt.expert_down_f32[0 .. cnt * h],
                    null,
                    cnt,
                    h,
                    inter,
                );
                for (0..cnt) |ci| {
                    const r = active_rows[ci];
                    const w = active_weights[ci];
                    const acc_row = srt.expert_acc_f32[r * h .. (r + 1) * h];
                    const dn_row = srt.expert_down_f32[ci * h .. (ci + 1) * h];
                    for (0..h) |c| acc_row[c] += w * dn_row[c];
                }
            }
        }

        for (0..m_u * h) |i| {
            x_residual_io[i] = @floatCast(srt.expert_acc_f32[i]);
        }
        return used_mask.count();
    }

    fn flushOptionalFrame(self: *Qwen3Model, frame_opt: *?metal.Compute.Frame) StepError!void {
        if (frame_opt.*) |*f| {
            const sig = try f.commit();
            frame_opt.* = null;
            try self.compute.wait(sig, self.timeout_ms);
        }
    }

    fn ensureOptionalFrame(self: *Qwen3Model, frame_opt: *?metal.Compute.Frame) StepError!*metal.Compute.Frame {
        if (frame_opt.* == null) {
            frame_opt.* = try self.compute.beginFrame(self.timeout_ms);
        }
        return &(frame_opt.*.?);
    }

    fn execLinearHetero(
        self: *Qwen3Model,
        frame_opt: *?metal.Compute.Frame,
        lin: LinearBinding,
        x_bind: metal.BufferBinding,
        x_host: []const f16,
        y_bind: metal.BufferBinding,
        y_host: []f16,
        m: u32,
        barrier_if_gpu: bool,
    ) StepError!bool {
        if (lin.layout == .sme2_panel64) {
            try self.flushOptionalFrame(frame_opt);
            try self.runLinearSme2(lin, x_host, y_host, m, .temporal);
            return false;
        }
        const had_frame = (frame_opt.* != null);
        const f = try self.ensureOptionalFrame(frame_opt);
        try self.encodeLinearBatched(f, lin, x_bind, y_bind, m, had_frame and barrier_if_gpu);
        return true;
    }

    fn runStepHeterogeneous(self: *Qwen3Model, token_id: u32, pos: u32, skip_lm_head: bool) StepError!u32 {
        if (token_id >= self.config.vocab_size) return error.TokenOutOfRange;
        if (pos >= self.max_seq_len) return error.PositionOutOfRange;

        const cur_seq_len: u32 = pos + 1;
        const chunk_idx: usize = @intCast(pos / self.chunk_tokens);
        const h: u32 = self.config.hidden_size;

        var frame_opt: ?metal.Compute.Frame = null;
        errdefer if (frame_opt) |*f| f.abort();

        {
            const f = try self.ensureOptionalFrame(&frame_opt);
            try self.moe_pipes.encodeEmbedLookup(f, self.embed.dtype, .{
                .weight = self.embed.weight,
                .scale_bias = self.embed.scale_bias,
                .tokens = .{ .token_id = token_id },
                .out = self.x_bind,
                .m = 1,
                .vocab_size = self.config.vocab_size,
                .hidden_size = h,
            }, false);
        }

        for (self.layers, 0..) |layer, l| {
            {
                const had = (frame_opt != null);
                const f = try self.ensureOptionalFrame(&frame_opt);
                try self.attn_pipes.encodeRmsNorm(f, .{
                    .x = self.x_bind,
                    .w = layer.input_layernorm.binding,
                    .y = self.x_norm_bind,
                    .num_rows = 1,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = layer.input_layernorm.dtype,
                }, had);
            }

            _ = try self.execLinearHetero(&frame_opt, layer.q_proj, self.x_norm_bind, self.x_norm_host, self.q_bind, self.q_host, 1, true);
            _ = try self.execLinearHetero(&frame_opt, layer.k_proj, self.x_norm_bind, self.x_norm_host, self.k_bind, self.k_host, 1, false);
            _ = try self.execLinearHetero(&frame_opt, layer.v_proj, self.x_norm_bind, self.x_norm_host, self.v_bind, self.v_host, 1, false);

            {
                const had = (frame_opt != null);
                const f = try self.ensureOptionalFrame(&frame_opt);
                const kv_pair = self.kvTileAt(l, chunk_idx);
                try self.attn_pipes.encodeQkNormRopeKvWrite(f, .{
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
                }, had);
                try self.encodeLayerGqaDecode(f, l, cur_seq_len, true);
            }

            _ = try self.execLinearHetero(&frame_opt, layer.o_proj, self.attn_out_bind, self.attn_out_host, self.proj_out_bind, self.proj_out_host, 1, true);
            {
                const had = (frame_opt != null);
                const f = try self.ensureOptionalFrame(&frame_opt);
                try self.moe_pipes.encodeResidualAdd(f, .{
                    .x = self.x_bind,
                    .delta = self.proj_out_bind,
                    .count = h,
                }, had);
                try self.attn_pipes.encodeRmsNorm(f, .{
                    .x = self.x_bind,
                    .w = layer.post_attention_layernorm.binding,
                    .y = self.x_norm_bind,
                    .num_rows = 1,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = layer.post_attention_layernorm.dtype,
                }, true);
            }

            switch (self.config.arch) {
                .qwen3_moe => {
                    const ws = &(self.moe_ws.?);
                    const router_src: moe.RouterSource = if (layer.router_layout == .sme2_panel64) blk: {
                        try self.flushOptionalFrame(&frame_opt);
                        try self.runRouterSme2(layer, self.x_norm_host, ws.router_logits, 1);
                        break :blk .{ .logits = .{ .binding = ws.routerLogitsBinding(self.scratch_buf), .dtype = .f32 } };
                    } else layer.router;

                    if (layer.expert_layout == .sme2_panel64) {
                        if (layer.router_layout == .row_major) {
                            const had = (frame_opt != null);
                            const f = try self.ensureOptionalFrame(&frame_opt);
                            try self.encodeLayerRouter(f, l, ws, self.scratch_buf, self.x_norm_bind, 1, had);
                            try self.flushOptionalFrame(&frame_opt);
                        }
                        _ = try self.runMoeExpertsSme2(layer, ws.router_logits, self.x_norm_host, self.x_host, 1);
                    } else {
                        const had = (frame_opt != null);
                        const f = try self.ensureOptionalFrame(&frame_opt);
                        const moe_args: moe.MoELayerArgs = .{
                            .x = self.x_norm_bind,
                            .router = router_src,
                            .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                            .expert_table = layer.expert_table,
                            .workspace = ws,
                            .workspace_buf = self.scratch_buf,
                            .residual = self.x_bind,
                            .out = self.x_bind,
                            .m = 1,
                            .num_experts = self.config.num_experts,
                            .top_k = self.config.num_experts_per_tok,
                            .hidden_size = h,
                            .intermediate_size = self.config.moe_intermediate_size,
                            .norm_topk_prob = self.config.norm_topk_prob,
                        };
                        try self.encodeMoELayerDispatch(f, layer, moe_args, had);
                    }
                },
                .qwen3_dense => {
                    _ = try self.execLinearHetero(&frame_opt, layer.dense_gate, self.x_norm_bind, self.x_norm_host, self.dense_gate_bind, self.dense_gate_host, 1, true);
                    _ = try self.execLinearHetero(&frame_opt, layer.dense_up, self.x_norm_bind, self.x_norm_host, self.dense_up_bind, self.dense_up_host, 1, false);
                    {
                        const had = (frame_opt != null);
                        const f = try self.ensureOptionalFrame(&frame_opt);
                        try self.moe_pipes.encodeSiluMul(f, .{
                            .gate = self.dense_gate_bind,
                            .up = self.dense_up_bind,
                            .out = self.dense_act_bind,
                            .count = self.config.intermediate_size,
                        }, had);
                    }
                    _ = try self.execLinearHetero(&frame_opt, layer.dense_down, self.dense_act_bind, self.dense_act_host, self.proj_out_bind, self.proj_out_host, 1, true);
                    {
                        const had = (frame_opt != null);
                        const f = try self.ensureOptionalFrame(&frame_opt);
                        try self.moe_pipes.encodeResidualAdd(f, .{
                            .x = self.x_bind,
                            .delta = self.proj_out_bind,
                            .count = h,
                        }, had);
                    }
                },
            }

            if (self.matchesAuxTap(l)) {
                const f = try self.ensureOptionalFrame(&frame_opt);
                try self.encodeAuxTapIfMatched(f, l, 1);
            }
        }

        if (!skip_lm_head) {
            {
                const had = (frame_opt != null);
                const f = try self.ensureOptionalFrame(&frame_opt);
                try self.attn_pipes.encodeRmsNorm(f, .{
                    .x = self.x_bind,
                    .w = self.final_norm.binding,
                    .y = self.x_norm_bind,
                    .num_rows = 1,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = self.final_norm.dtype,
                }, had);
            }
            _ = try self.execLinearHetero(&frame_opt, self.lm_head, self.x_norm_bind, self.x_norm_host, self.logits_bind, self.logits_mut_host, 1, true);
            {
                const had = (frame_opt != null);
                const f = try self.ensureOptionalFrame(&frame_opt);
                try self.moe_pipes.encodeArgmax(f, .{
                    .logits = self.logits_bind,
                    .out_token_id = self.out_token_id_bind,
                    .vocab_size = self.config.vocab_size,
                }, had);
            }
        }

        try self.flushOptionalFrame(&frame_opt);
        self.seq_len = cur_seq_len;
        if (!skip_lm_head) self.last_logits_row = 0;
        return if (skip_lm_head) 0 else self.out_token_id_host.*;
    }

    fn verifyTreeHeterogeneous(
        self: *Qwen3Model,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        compute_logits: bool,
    ) StepError!VerifyTreeResult {
        if (self.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (token_ids.len == 0 or token_ids.len != positions.len or token_ids.len != ancestor_masks.len) {
            return error.InvalidArgument;
        }
        if (token_ids.len > self.max_tree_nodes) return error.TreeTooLarge;
        if (base_pos > self.max_seq_len) return error.PositionOutOfRange;

        const m: u32 = @intCast(token_ids.len);
        for (0..m) |i| {
            if (token_ids[i] >= self.config.vocab_size) return error.TokenOutOfRange;
            if (positions[i] >= self.max_seq_len) return error.PositionOutOfRange;
        }
        @memcpy(self.tree_token_ids_host[0..m], token_ids);
        @memcpy(self.tree_positions_host[0..m], positions);
        @memcpy(self.tree_ancestor_masks_host[0..m], ancestor_masks);

        const h = self.config.hidden_size;
        var frame_opt: ?metal.Compute.Frame = null;
        errdefer if (frame_opt) |*f| f.abort();

        {
            const f = try self.ensureOptionalFrame(&frame_opt);
            try self.moe_pipes.encodeEmbedLookup(f, self.embed.dtype, .{
                .weight = self.embed.weight,
                .scale_bias = self.embed.scale_bias,
                .tokens = .{ .token_ids = self.tree_token_ids_bind },
                .out = self.x_bind,
                .m = m,
                .vocab_size = self.config.vocab_size,
                .hidden_size = h,
            }, false);
        }

        var total_unique: u64 = 0;
        for (self.layers, 0..) |layer, l| {
            {
                const had = (frame_opt != null);
                const f = try self.ensureOptionalFrame(&frame_opt);
                try self.attn_pipes.encodeRmsNorm(f, .{
                    .x = self.x_bind,
                    .w = layer.input_layernorm.binding,
                    .y = self.x_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = layer.input_layernorm.dtype,
                }, had);
            }

            _ = try self.execLinearHetero(&frame_opt, layer.q_proj, self.x_norm_bind, self.x_norm_host, self.q_bind, self.q_host, m, true);
            _ = try self.execLinearHetero(&frame_opt, layer.k_proj, self.x_norm_bind, self.x_norm_host, self.k_bind, self.k_host, m, false);
            _ = try self.execLinearHetero(&frame_opt, layer.v_proj, self.x_norm_bind, self.x_norm_host, self.v_bind, self.v_host, m, false);

            {
                const had = (frame_opt != null);
                const f = try self.ensureOptionalFrame(&frame_opt);
                const k_tree_bind = self.treeKBinding(l);
                const v_tree_bind = self.treeVBinding(l);
                try self.attn_pipes.encodeTreeQkNormRope(f, .{
                    .q = self.q_bind,
                    .k = self.k_bind,
                    .v = self.v_bind,
                    .q_norm = layer.q_norm,
                    .k_norm = layer.k_norm,
                    .positions = self.tree_positions_bind,
                    .k_tree_out = k_tree_bind,
                    .v_tree_out = v_tree_bind,
                    .num_nodes = m,
                    .num_q_heads = self.config.num_attention_heads,
                    .num_kv_heads = self.config.num_key_value_heads,
                    .head_dim = self.config.head_dim,
                    .rms_eps = self.config.rms_norm_eps,
                    .rope_theta = self.config.rope_theta,
                    .norm_dtype = layer.qk_norm_dtype,
                }, had);
                try self.encodeLayerGqaTreeVerify(f, l, m, base_pos, k_tree_bind, v_tree_bind, true);
            }

            _ = try self.execLinearHetero(&frame_opt, layer.o_proj, self.attn_out_bind, self.attn_out_host, self.proj_out_bind, self.proj_out_host, m, true);
            {
                const had = (frame_opt != null);
                const f = try self.ensureOptionalFrame(&frame_opt);
                try self.moe_pipes.encodeResidualAdd(f, .{
                    .x = self.x_bind,
                    .delta = self.proj_out_bind,
                    .count = m * h,
                }, had);
                try self.attn_pipes.encodeRmsNorm(f, .{
                    .x = self.x_bind,
                    .w = layer.post_attention_layernorm.binding,
                    .y = self.x_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = layer.post_attention_layernorm.dtype,
                }, true);
            }

            switch (self.config.arch) {
                .qwen3_moe => {
                    var layer_ws = self.moe_ws.?;
                    layer_ws.schedule_header_off = self.layer_sched_hdr_base_off + @as(u64, l) * 64;
                    const router_src: moe.RouterSource = if (layer.router_layout == .sme2_panel64) blk: {
                        try self.flushOptionalFrame(&frame_opt);
                        try self.runRouterSme2(layer, self.x_norm_host, layer_ws.router_logits, m);
                        break :blk .{ .logits = .{ .binding = layer_ws.routerLogitsBinding(self.scratch_buf), .dtype = .f32 } };
                    } else layer.router;

                    if (layer.expert_layout == .sme2_panel64) {
                        if (layer.router_layout == .row_major) {
                            const had = (frame_opt != null);
                            const f = try self.ensureOptionalFrame(&frame_opt);
                            try self.encodeLayerRouter(f, l, &layer_ws, self.scratch_buf, self.x_norm_bind, m, had);
                            try self.flushOptionalFrame(&frame_opt);
                        }
                        const uniq = try self.runMoeExpertsSme2(layer, layer_ws.router_logits, self.x_norm_host, self.x_host, m);
                        const hdr: *moe.ScheduleHeader = @ptrCast(@alignCast(self.layer_sched_hdr_raw.ptr + l * 64));
                        hdr.num_unique_experts = uniq;
                        total_unique += uniq;
                    } else {
                        const had = (frame_opt != null);
                        const f = try self.ensureOptionalFrame(&frame_opt);
                        const moe_args: moe.MoELayerArgs = .{
                            .x = self.x_norm_bind,
                            .router = router_src,
                            .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                            .expert_table = layer.expert_table,
                            .workspace = &layer_ws,
                            .workspace_buf = self.scratch_buf,
                            .residual = self.x_bind,
                            .out = self.x_bind,
                            .m = m,
                            .num_experts = self.config.num_experts,
                            .top_k = self.config.num_experts_per_tok,
                            .hidden_size = h,
                            .intermediate_size = self.config.moe_intermediate_size,
                            .norm_topk_prob = self.config.norm_topk_prob,
                        };
                        try self.encodeMoELayerDispatch(f, layer, moe_args, had);
                    }
                },
                .qwen3_dense => {
                    _ = try self.execLinearHetero(&frame_opt, layer.dense_gate, self.x_norm_bind, self.x_norm_host, self.dense_gate_bind, self.dense_gate_host, m, true);
                    _ = try self.execLinearHetero(&frame_opt, layer.dense_up, self.x_norm_bind, self.x_norm_host, self.dense_up_bind, self.dense_up_host, m, false);
                    {
                        const had = (frame_opt != null);
                        const f = try self.ensureOptionalFrame(&frame_opt);
                        try self.moe_pipes.encodeSiluMul(f, .{
                            .gate = self.dense_gate_bind,
                            .up = self.dense_up_bind,
                            .out = self.dense_act_bind,
                            .count = m * self.config.intermediate_size,
                        }, had);
                    }
                    _ = try self.execLinearHetero(&frame_opt, layer.dense_down, self.dense_act_bind, self.dense_act_host, self.proj_out_bind, self.proj_out_host, m, true);
                    {
                        const had = (frame_opt != null);
                        const f = try self.ensureOptionalFrame(&frame_opt);
                        try self.moe_pipes.encodeResidualAdd(f, .{
                            .x = self.x_bind,
                            .delta = self.proj_out_bind,
                            .count = m * h,
                        }, had);
                    }
                },
            }

            if (self.matchesAuxTap(l)) {
                const f = try self.ensureOptionalFrame(&frame_opt);
                try self.encodeAuxTapIfMatched(f, l, m);
            }
        }

        if (compute_logits) {
            {
                const had = (frame_opt != null);
                const f = try self.ensureOptionalFrame(&frame_opt);
                try self.attn_pipes.encodeRmsNorm(f, .{
                    .x = self.x_bind,
                    .w = self.final_norm.binding,
                    .y = self.x_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = self.final_norm.dtype,
                }, had);
            }
            _ = try self.execLinearHetero(&frame_opt, self.lm_head, self.x_norm_bind, self.x_norm_host, self.logits_bind, self.logits_mut_host, m, true);
            {
                const had = (frame_opt != null);
                const f = try self.ensureOptionalFrame(&frame_opt);
                try self.moe_pipes.encodeArgmaxRows(f, .{
                    .logits = self.logits_bind,
                    .out_token_ids = self.out_token_id_bind,
                    .num_rows = m,
                    .vocab_size = self.config.vocab_size,
                    .logits_dtype = .f16,
                }, had);
            }
        }

        try self.flushOptionalFrame(&frame_opt);
        if (compute_logits) self.last_logits_row = @as(usize, m) - 1;

        if (self.config.arch == .qwen3_moe) {
            total_unique = 0;
            for (0..self.layers.len) |l| {
                const hdr: *const moe.ScheduleHeader = @ptrCast(@alignCast(self.layer_sched_hdr_raw.ptr + l * 64));
                total_unique += hdr.num_unique_experts;
            }
        }
        const avg_unique: f64 = if (self.config.arch == .qwen3_moe)
            @as(f64, @floatFromInt(total_unique)) / @as(f64, @floatFromInt(self.layers.len))
        else
            0.0;

        const kv_per_layer: u64 =
            2 * @as(u64, self.config.num_key_value_heads) * (@as(u64, base_pos) + m) * self.config.head_dim * @sizeOf(f16);
        const kv_total: u64 = @as(u64, self.config.num_hidden_layers) * kv_per_layer;
        const head_bytes: u64 = if (compute_logits)
            @as(u64, m) * self.embed_row_alg_bytes + (self.global_head_bytes - self.embed_row_alg_bytes)
        else
            @as(u64, m) * self.embed_row_alg_bytes;
        const layer_weight_bytes: u64 = if (self.config.arch == .qwen3_moe)
            self.total_layers_non_expert_bytes + total_unique * self.single_expert_bytes
        else
            self.total_layers_static_bytes;

        return .{
            .num_nodes = m,
            .argmax_tokens = self.out_token_ids_host[0..m],
            .logits_f16 = self.logits_host[0 .. @as(usize, m) * self.config.vocab_size],
            .total_unique_experts = total_unique,
            .avg_unique_experts_per_layer = avg_unique,
            .actual_bytes_moved = layer_weight_bytes + kv_total + head_bytes,
        };
    }

    // -----------------------------------------------------------------------
    // M4 Cold-Expert NVMe Streaming + ANE Cross-Layer Prefetch (C3-C5, A2)
    // -----------------------------------------------------------------------

    fn encodeLayerAttentionDecode(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        l: usize,
        pos: u32,
        cur_seq_len: u32,
        chunk_idx: usize,
        barrier_before: bool,
    ) StepError!void {
        const layer = self.layers[l];
        const h = self.config.hidden_size;
        try self.attn_pipes.encodeRmsNorm(frame, .{
            .x = self.x_bind,
            .w = layer.input_layernorm.binding,
            .y = self.x_norm_bind,
            .num_rows = 1,
            .hidden_dim = h,
            .eps = self.config.rms_norm_eps,
            .w_dtype = layer.input_layernorm.dtype,
        }, barrier_before);
        try self.encodeLinearGemv(frame, layer.q_proj, self.x_norm_bind, self.q_bind, true);
        try self.encodeLinearGemv(frame, layer.k_proj, self.x_norm_bind, self.k_bind, false);
        try self.encodeLinearGemv(frame, layer.v_proj, self.x_norm_bind, self.v_bind, false);

        const kv_pair = self.kvTileAt(l, chunk_idx);
        try self.attn_pipes.encodeQkNormRopeKvWrite(frame, .{
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

        try self.encodeLayerGqaDecode(frame, l, cur_seq_len, true);

        try self.encodeLinearGemv(frame, layer.o_proj, self.attn_out_bind, self.proj_out_bind, true);
        try self.moe_pipes.encodeResidualAdd(frame, .{
            .x = self.x_bind,
            .delta = self.proj_out_bind,
            .count = h,
        }, true);
        try self.attn_pipes.encodeRmsNorm(frame, .{
            .x = self.x_bind,
            .w = layer.post_attention_layernorm.binding,
            .y = self.x_norm_bind,
            .num_rows = 1,
            .hidden_dim = h,
            .eps = self.config.rms_norm_eps,
            .w_dtype = layer.post_attention_layernorm.dtype,
        }, true);
    }

    fn encodeLayerAttentionTree(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        l: usize,
        m: u32,
        base_pos: u32,
        barrier_before: bool,
    ) StepError!void {
        const layer = self.layers[l];
        const h = self.config.hidden_size;
        try self.attn_pipes.encodeRmsNorm(frame, .{
            .x = self.x_bind,
            .w = layer.input_layernorm.binding,
            .y = self.x_norm_bind,
            .num_rows = m,
            .hidden_dim = h,
            .eps = self.config.rms_norm_eps,
            .w_dtype = layer.input_layernorm.dtype,
        }, barrier_before);

        try self.encodeLinearBatched(frame, layer.q_proj, self.x_norm_bind, self.q_bind, m, true);
        try self.encodeLinearBatched(frame, layer.k_proj, self.x_norm_bind, self.k_bind, m, false);
        try self.encodeLinearBatched(frame, layer.v_proj, self.x_norm_bind, self.v_bind, m, false);

        const k_tree_bind = self.treeKBinding(l);
        const v_tree_bind = self.treeVBinding(l);
        try self.attn_pipes.encodeTreeQkNormRope(frame, .{
            .q = self.q_bind,
            .k = self.k_bind,
            .v = self.v_bind,
            .q_norm = layer.q_norm,
            .k_norm = layer.k_norm,
            .positions = self.tree_positions_bind,
            .k_tree_out = k_tree_bind,
            .v_tree_out = v_tree_bind,
            .num_nodes = m,
            .num_q_heads = self.config.num_attention_heads,
            .num_kv_heads = self.config.num_key_value_heads,
            .head_dim = self.config.head_dim,
            .rms_eps = self.config.rms_norm_eps,
            .rope_theta = self.config.rope_theta,
            .norm_dtype = layer.qk_norm_dtype,
        }, true);

        try self.encodeLayerGqaTreeVerify(frame, l, m, base_pos, k_tree_bind, v_tree_bind, true);

        try self.encodeLinearBatched(frame, layer.o_proj, self.attn_out_bind, self.proj_out_bind, m, true);
        try self.moe_pipes.encodeResidualAdd(frame, .{
            .x = self.x_bind,
            .delta = self.proj_out_bind,
            .count = m * h,
        }, true);
        try self.attn_pipes.encodeRmsNorm(frame, .{
            .x = self.x_bind,
            .w = layer.post_attention_layernorm.binding,
            .y = self.x_norm_bind,
            .num_rows = m,
            .hidden_dim = h,
            .eps = self.config.rms_norm_eps,
            .w_dtype = layer.post_attention_layernorm.dtype,
        }, true);
    }

    fn encodeLayerRouter(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        l: usize,
        ws: *const moe.MoeWorkspace,
        ws_buf: metal.Buffer,
        x_norm: metal.BufferBinding,
        m: u32,
        barrier_before: bool,
    ) moe.EncodeError!void {
        const layer = self.layers[l];
        const logits_bind = ws.routerLogitsBinding(ws_buf);
        switch (layer.router) {
            .weights => |w| try self.moe_pipes.encodeRouterProj(frame, w.dtype, .{
                .x = x_norm,
                .router_weights = w.binding,
                .logits = logits_bind,
                .m = m,
                .num_experts = self.config.num_experts,
                .hidden_size = self.config.hidden_size,
            }, barrier_before),
            .quant_weights => |q| try self.moe_pipes.encodeRouterProjQuant(frame, q.bits, .{
                .x = x_norm,
                .qweight = q.qweight,
                .scale_bias = q.scale_bias,
                .logits = logits_bind,
                .m = m,
                .num_experts = self.config.num_experts,
                .hidden_size = self.config.hidden_size,
            }, barrier_before),
            .logits => return error.InvalidArgument,
        }
        if (layer.is_heterogeneous_moe) {
            try self.moe_pipes.encodeRouteTopKBucketed(frame, .f32, .{
                .logits = logits_bind,
                .expert_schemes = layer.expert_schemes_bind,
                .workspace = ws,
                .workspace_buf = ws_buf,
                .m = m,
                .num_experts = self.config.num_experts,
                .top_k = self.config.num_experts_per_tok,
                .norm_topk_prob = self.config.norm_topk_prob,
            }, true);
        } else {
            try self.moe_pipes.encodeRouteTopK(frame, .f32, .{
                .logits = logits_bind,
                .workspace = ws,
                .workspace_buf = ws_buf,
                .m = m,
                .num_experts = self.config.num_experts,
                .top_k = self.config.num_experts_per_tok,
                .norm_topk_prob = self.config.norm_topk_prob,
            }, true);
        }
    }

    fn encodeMoEGateUpDown(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        l: usize,
        weights_buf: metal.Buffer,
        ws: *const moe.MoeWorkspace,
        ws_buf: metal.Buffer,
        x_norm: metal.BufferBinding,
        m: u32,
        barrier_before: bool,
    ) moe.EncodeError!void {
        const layer = self.layers[l];
        if (layer.is_heterogeneous_moe) {
            const base: moe.MoELayerArgs = .{
                .x = x_norm,
                .router = layer.router,
                .weights_base = .{ .buffer = weights_buf, .offset = 0 },
                .expert_table = layer.expert_table,
                .workspace = ws,
                .workspace_buf = ws_buf,
                .residual = null,
                .out = x_norm,
                .m = m,
                .num_experts = self.config.num_experts,
                .top_k = self.config.num_experts_per_tok,
                .hidden_size = self.config.hidden_size,
                .intermediate_size = self.config.moe_intermediate_size,
                .norm_topk_prob = self.config.norm_topk_prob,
            };
            try self.moe_pipes.encodeGateUpDownHetero(frame, base, layer.expert_scheme_counts, true, barrier_before);
            return;
        }
        const gu_args: moe.GateUpSwigluArgs = .{
            .x = x_norm,
            .weights_base = .{ .buffer = weights_buf, .offset = 0 },
            .expert_table = layer.expert_table,
            .workspace = ws,
            .workspace_buf = ws_buf,
            .m = m,
            .num_experts = self.config.num_experts,
            .top_k = self.config.num_experts_per_tok,
            .hidden_size = self.config.hidden_size,
            .intermediate_size = self.config.moe_intermediate_size,
        };
        const dp_args: moe.DownProjArgs = .{
            .weights_base = .{ .buffer = weights_buf, .offset = 0 },
            .expert_table = layer.expert_table,
            .workspace = ws,
            .workspace_buf = ws_buf,
            .m = m,
            .num_experts = self.config.num_experts,
            .top_k = self.config.num_experts_per_tok,
            .hidden_size = self.config.hidden_size,
            .intermediate_size = self.config.moe_intermediate_size,
        };
        switch (layer.expert_dtype) {
            .q2_affine => {
                try self.moe_pipes.encodeGateUpSwiglu(frame, .q2, gu_args, barrier_before);
                try self.moe_pipes.encodeDownProj(frame, .q2, dp_args, true);
            },
            .q3_affine => {
                try self.moe_pipes.encodeGateUpSwiglu(frame, .q3, gu_args, barrier_before);
                try self.moe_pipes.encodeDownProj(frame, .q3, dp_args, true);
            },
            .q4_affine => {
                try self.moe_pipes.encodeGateUpSwiglu(frame, .q4, gu_args, barrier_before);
                try self.moe_pipes.encodeDownProj(frame, .q4, dp_args, true);
            },
            .f16 => {
                try self.moe_pipes.encodeGateUpSwigluFloat(frame, .f16, gu_args, barrier_before);
                try self.moe_pipes.encodeDownProjFloat(frame, .f16, dp_args, true);
            },
            .bf16 => {
                try self.moe_pipes.encodeGateUpSwigluFloat(frame, .bf16, gu_args, barrier_before);
                try self.moe_pipes.encodeDownProjFloat(frame, .bf16, dp_args, true);
            },
        }
    }

    fn drainAnePrefetch(self: *Qwen3Model, crt: *ColdRuntime, cur_layer: usize, m: u32) StepError!void {
        _ = self;
        if (!crt.ane_inflight) return;
        const pred = crt.ane_predictor orelse {
            crt.ane_inflight = false;
            return;
        };
        const top_m: u32 = if (m == 1)
            @min(@as(u32, 8), @max(@as(u32, 1), crt.pool.numSlots() / 4))
        else
            @min(m * 4, @max(@as(u32, 1), crt.pool.numSlots() / 2));
        const ares = try pred.waitAsync(top_m, crt.cold_masks_by_layer, &crt.pred_bitsets, &crt.pred_cands);
        crt.ane_inflight = false;
        for (crt.pred_cands[0..ares.result.prefetch_count]) |cand| {
            if (cand.layer <= cur_layer) continue;
            _ = crt.expert_cache.prefetch(cand.layer, cand.expert) catch |err| switch (err) {
                error.QueueFull, error.NoEvictableSlot => break,
                else => return err,
            };
        }
    }

    /// Resolves & pins all active experts in `ws` for `layer l`, running a
    /// pre-pass on `layer.expert_weights_buf` ONLY when the slot pool lives in
    /// a separate `MTLBuffer` (multi-shard cold edge case). In the normal
    /// single-shard layout (`pool_buf == layer.expert_weights_buf`), all hot
    /// and pinned cold experts run together in ONE zero-copy dispatch!
    fn prepareColdLayerExperts(
        self: *Qwen3Model,
        crt: *ColdRuntime,
        l: usize,
        ws: *moe.MoeWorkspace,
        ws_buf: metal.Buffer,
        x_norm: metal.BufferBinding,
        m: u32,
    ) StepError!struct { weights_buf: metal.Buffer, num_unique: u32 } {
        const layer = self.layers[l];
        const num_unique: u32 = ws.schedule_header.num_unique_experts;
        const active = ws.unique_experts[0..num_unique];

        const pool_buf = crt.pool.metal_buffer.?;
        var has_hot = false;
        var has_cold = false;
        for (active) |eid| {
            const pin = try crt.expert_cache.acquirePinned(@intCast(l), @intCast(eid));
            layer.expert_table_host[eid] = pin.weights;
            if (pin.is_hot) {
                has_hot = true;
            } else {
                has_cold = true;
            }
        }

        // Fast path: hot weights and cold slot pool live in the same `MTLBuffer`,
        // OR all active experts live in one buffer -> single zero-copy dispatch!
        if (pool_buf.handle == layer.expert_weights_buf.handle or !has_cold) {
            return .{ .weights_buf = layer.expert_weights_buf, .num_unique = num_unique };
        }
        if (!has_hot) {
            return .{ .weights_buf = pool_buf, .num_unique = num_unique };
        }

        // Multi-shard mixed hot + cold fallback: split CSR into hot pass (writes
        // hot `(tok, slot_k)` slots of `ws.down_out`) then leave cold-only CSR in
        // `ws` for the caller's main frame (`pool_buf`) + `encodeReduce`.
        var orig_u: [moe.MAX_EXPERTS]u32 = undefined;
        var orig_off: [moe.MAX_EXPERTS + 1]u32 = undefined;
        var orig_tok: [attention.MAX_TREE_NODES * moe.MAX_TOP_K]u32 = undefined;
        var orig_rw: [attention.MAX_TREE_NODES * moe.MAX_TOP_K]f32 = undefined;
        var orig_sk: [attention.MAX_TREE_NODES * moe.MAX_TOP_K]u32 = undefined;
        const total_a: usize = ws.schedule_header.total_assignments;
        @memcpy(orig_u[0..num_unique], ws.unique_experts[0..num_unique]);
        @memcpy(orig_off[0 .. num_unique + 1], ws.expert_token_offsets[0 .. num_unique + 1]);
        @memcpy(orig_tok[0..total_a], ws.token_indices[0..total_a]);
        @memcpy(orig_rw[0..total_a], ws.routing_weights[0..total_a]);
        @memcpy(orig_sk[0..total_a], ws.slot_k_indices[0..total_a]);

        // Pass 1: hot-only CSR -> `layer.expert_weights_buf`
        var hot_u: u32 = 0;
        var hot_a: u32 = 0;
        ws.expert_token_offsets[0] = 0;
        for (0..num_unique) |ui| {
            const eid = orig_u[ui];
            if (crt.cold_masks_by_layer[l].contains(eid)) continue;
            ws.unique_experts[hot_u] = eid;
            const s0 = orig_off[ui];
            const s1 = orig_off[ui + 1];
            for (s0..s1) |ai| {
                ws.token_indices[hot_a] = orig_tok[ai];
                ws.routing_weights[hot_a] = orig_rw[ai];
                ws.slot_k_indices[hot_a] = orig_sk[ai];
                hot_a += 1;
            }
            hot_u += 1;
            ws.expert_token_offsets[hot_u] = hot_a;
        }
        ws.schedule_header.num_unique_experts = hot_u;
        ws.schedule_header.total_assignments = hot_a;
        const n_exp: usize = self.config.num_experts;
        if (layer.is_heterogeneous_moe) {
            ws.rebuildSchemeBucketsCpu(self.expert_dtypes[l * n_exp .. (l + 1) * n_exp]);
        }

        var hot_frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer hot_frame.abort();
        try self.encodeMoEGateUpDown(&hot_frame, l, layer.expert_weights_buf, ws, ws_buf, x_norm, m, false);
        const hot_sig = try hot_frame.commit();
        try self.compute.wait(hot_sig, self.timeout_ms);

        // Pass 2: cold-only CSR left in `ws` for caller's frame on `pool_buf`
        var cold_u: u32 = 0;
        var cold_a: u32 = 0;
        ws.expert_token_offsets[0] = 0;
        for (0..num_unique) |ui| {
            const eid = orig_u[ui];
            if (!crt.cold_masks_by_layer[l].contains(eid)) continue;
            ws.unique_experts[cold_u] = eid;
            const s0 = orig_off[ui];
            const s1 = orig_off[ui + 1];
            for (s0..s1) |ai| {
                ws.token_indices[cold_a] = orig_tok[ai];
                ws.routing_weights[cold_a] = orig_rw[ai];
                ws.slot_k_indices[cold_a] = orig_sk[ai];
                cold_a += 1;
            }
            cold_u += 1;
            ws.expert_token_offsets[cold_u] = cold_a;
        }
        ws.schedule_header.num_unique_experts = cold_u;
        ws.schedule_header.total_assignments = cold_a;
        if (layer.is_heterogeneous_moe) {
            ws.rebuildSchemeBucketsCpu(self.expert_dtypes[l * n_exp .. (l + 1) * n_exp]);
        }
        return .{ .weights_buf = pool_buf, .num_unique = num_unique };
    }

    fn runStepCold(self: *Qwen3Model, token_id: u32, pos: u32, skip_lm_head: bool) StepError!u32 {
        if (token_id >= self.config.vocab_size) return error.TokenOutOfRange;
        if (pos >= self.max_seq_len) return error.PositionOutOfRange;

        const crt = self.cold_rt.?;
        const ws = &(self.moe_ws.?);
        const h = self.config.hidden_size;
        const cur_seq_len: u32 = pos + 1;
        const chunk_idx: usize = @intCast(pos / self.chunk_tokens);

        defer {
            if (crt.ane_inflight) {
                if (crt.ane_predictor) |pred| {
                    _ = pred.waitAsync(0, null, &crt.pred_bitsets, &crt.pred_cands) catch |err| {
                        std.debug.panic("runStepCold unwind waitAsync failed: {}", .{err});
                    };
                }
                crt.ane_inflight = false;
            }
            crt.expert_cache.releasePins();
        }
        var gpu_sig_pending: ?u64 = null;
        errdefer if (gpu_sig_pending) |sig| {
            self.compute.wait(sig, self.timeout_ms) catch |err| {
                std.debug.panic("runStepCold unwind GPU wait failed: {}", .{err});
            };
        };

        // Frame 0: embed + layer 0 attention + layer 0 router.
        {
            var frame = try self.compute.beginFrame(self.timeout_ms);
            errdefer frame.abort();
            try self.moe_pipes.encodeEmbedLookup(&frame, self.embed.dtype, .{
                .weight = self.embed.weight,
                .scale_bias = self.embed.scale_bias,
                .tokens = .{ .token_id = token_id },
                .out = self.x_bind,
                .m = 1,
                .vocab_size = self.config.vocab_size,
                .hidden_size = h,
            }, false);
            try self.encodeLayerAttentionDecode(&frame, 0, pos, cur_seq_len, chunk_idx, true);
            try self.encodeLayerRouter(&frame, 0, ws, self.scratch_buf, self.x_norm_bind, 1, true);
            const sig = try frame.commit();
            try self.compute.wait(sig, self.timeout_ms);
        }

        for (0..self.layers.len) |l| {
            // Submit async ANE lookahead prediction for `l + 1 .. l + lookahead`
            // from layer `l`'s post-attention residual `x_host[0..h]`.
            if (crt.ane_predictor) |pred| {
                if (l + 1 < self.layers.len) {
                    try pred.writeInputHiddenF16(self.x_host[0..h], 1);
                    const pushed = try pred.submitAsync(crt.ane_req_seq, @intCast(l), 1);
                    if (pushed) {
                        crt.ane_req_seq += 1;
                        crt.ane_inflight = true;
                    }
                }
            }

            // Pin active experts for layer `l` and update `expert_table_host` offsets.
            const prep = try self.prepareColdLayerExperts(crt, l, ws, self.scratch_buf, self.x_norm_bind, 1);

            // Frame `l + 1`: layer `l` MoE FFN + aux tap + layer `l + 1` attn/router (or head).
            var frame = try self.compute.beginFrame(self.timeout_ms);
            errdefer frame.abort();
            try self.encodeMoEGateUpDown(&frame, l, prep.weights_buf, ws, self.scratch_buf, self.x_norm_bind, 1, false);
            try self.moe_pipes.encodeReduce(&frame, .{
                .workspace = ws,
                .workspace_buf = self.scratch_buf,
                .residual = self.x_bind,
                .out = self.x_bind,
                .m = 1,
                .hidden_size = h,
                .top_k = self.config.num_experts_per_tok,
            }, true);
            try self.encodeAuxTapIfMatched(&frame, l, 1);

            if (l + 1 < self.layers.len) {
                try self.encodeLayerAttentionDecode(&frame, l + 1, pos, cur_seq_len, chunk_idx, true);
                try self.encodeLayerRouter(&frame, l + 1, ws, self.scratch_buf, self.x_norm_bind, 1, true);
            } else if (!skip_lm_head) {
                try self.attn_pipes.encodeRmsNorm(&frame, .{
                    .x = self.x_bind,
                    .w = self.final_norm.binding,
                    .y = self.x_norm_bind,
                    .num_rows = 1,
                    .hidden_dim = h,
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
            gpu_sig_pending = sig;

            // While GPU executes layer `l` FFN + layer `l + 1` attn/router, reap
            // ANE's cross-layer prediction and kick off async NVMe prefetches for
            // upcoming layers (`l + 1 .. l + lookahead`) into unpinned slots!
            try self.drainAnePrefetch(crt, l, 1);

            try self.compute.wait(sig, self.timeout_ms);
            gpu_sig_pending = null;
            crt.expert_cache.releasePins();
        }

        self.seq_len = cur_seq_len;
        if (!skip_lm_head) self.last_logits_row = 0;
        return if (skip_lm_head) 0 else self.out_token_id_host.*;
    }

    fn verifyTreeCold(
        self: *Qwen3Model,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        compute_logits: bool,
    ) StepError!VerifyTreeResult {
        if (self.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (token_ids.len == 0 or token_ids.len != positions.len or token_ids.len != ancestor_masks.len) {
            return error.InvalidArgument;
        }
        if (token_ids.len > self.max_tree_nodes) return error.TreeTooLarge;
        if (base_pos > self.max_seq_len) return error.PositionOutOfRange;

        const m: u32 = @intCast(token_ids.len);
        for (0..m) |i| {
            if (token_ids[i] >= self.config.vocab_size) return error.TokenOutOfRange;
            if (positions[i] >= self.max_seq_len) return error.PositionOutOfRange;
        }
        @memcpy(self.tree_token_ids_host[0..m], token_ids);
        @memcpy(self.tree_positions_host[0..m], positions);
        @memcpy(self.tree_ancestor_masks_host[0..m], ancestor_masks);

        const crt = self.cold_rt.?;
        const ws = &(self.moe_ws.?);
        const h = self.config.hidden_size;

        defer {
            if (crt.ane_inflight) {
                if (crt.ane_predictor) |pred| {
                    _ = pred.waitAsync(0, null, &crt.pred_bitsets, &crt.pred_cands) catch |err| {
                        std.debug.panic("verifyTreeCold unwind waitAsync failed: {}", .{err});
                    };
                }
                crt.ane_inflight = false;
            }
            crt.expert_cache.releasePins();
        }
        var gpu_sig_pending: ?u64 = null;
        errdefer if (gpu_sig_pending) |sig| {
            self.compute.wait(sig, self.timeout_ms) catch |err| {
                std.debug.panic("verifyTreeCold unwind GPU wait failed: {}", .{err});
            };
        };

        // Frame 0: embed + layer 0 tree attention + layer 0 router.
        {
            var frame = try self.compute.beginFrame(self.timeout_ms);
            errdefer frame.abort();
            try self.moe_pipes.encodeEmbedLookup(&frame, self.embed.dtype, .{
                .weight = self.embed.weight,
                .scale_bias = self.embed.scale_bias,
                .tokens = .{ .token_ids = self.tree_token_ids_bind },
                .out = self.x_bind,
                .m = m,
                .vocab_size = self.config.vocab_size,
                .hidden_size = h,
            }, false);
            try self.encodeLayerAttentionTree(&frame, 0, m, base_pos, true);
            try self.encodeLayerRouter(&frame, 0, ws, self.scratch_buf, self.x_norm_bind, m, true);
            const sig = try frame.commit();
            try self.compute.wait(sig, self.timeout_ms);
        }

        var total_unique: u64 = 0;
        for (0..self.layers.len) |l| {
            if (crt.ane_predictor) |pred| {
                if (l + 1 < self.layers.len) {
                    const pred_m: u32 = @min(m, pred.config.max_tokens);
                    try pred.writeInputHiddenF16(self.x_host[0 .. @as(usize, pred_m) * h], pred_m);
                    const pushed = try pred.submitAsync(crt.ane_req_seq, @intCast(l), pred_m);
                    if (pushed) {
                        crt.ane_req_seq += 1;
                        crt.ane_inflight = true;
                    }
                }
            }

            const prep = try self.prepareColdLayerExperts(crt, l, ws, self.scratch_buf, self.x_norm_bind, m);
            total_unique += prep.num_unique;

            var frame = try self.compute.beginFrame(self.timeout_ms);
            errdefer frame.abort();
            try self.encodeMoEGateUpDown(&frame, l, prep.weights_buf, ws, self.scratch_buf, self.x_norm_bind, m, false);
            try self.moe_pipes.encodeReduce(&frame, .{
                .workspace = ws,
                .workspace_buf = self.scratch_buf,
                .residual = self.x_bind,
                .out = self.x_bind,
                .m = m,
                .hidden_size = h,
                .top_k = self.config.num_experts_per_tok,
            }, true);
            try self.encodeAuxTapIfMatched(&frame, l, m);

            if (l + 1 < self.layers.len) {
                try self.encodeLayerAttentionTree(&frame, l + 1, m, base_pos, true);
                try self.encodeLayerRouter(&frame, l + 1, ws, self.scratch_buf, self.x_norm_bind, m, true);
            } else if (compute_logits) {
                try self.attn_pipes.encodeRmsNorm(&frame, .{
                    .x = self.x_bind,
                    .w = self.final_norm.binding,
                    .y = self.x_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = self.final_norm.dtype,
                }, true);
                try self.encodeLinearBatched(&frame, self.lm_head, self.x_norm_bind, self.logits_bind, m, true);
                try self.moe_pipes.encodeArgmaxRows(&frame, .{
                    .logits = self.logits_bind,
                    .out_token_ids = self.out_token_id_bind,
                    .num_rows = m,
                    .vocab_size = self.config.vocab_size,
                    .logits_dtype = .f16,
                }, true);
            }

            const sig = try frame.commit();
            gpu_sig_pending = sig;

            try self.drainAnePrefetch(crt, l, m);

            try self.compute.wait(sig, self.timeout_ms);
            gpu_sig_pending = null;
            crt.expert_cache.releasePins();
        }

        if (compute_logits) self.last_logits_row = @as(usize, m) - 1;
        const avg_unique: f64 = @as(f64, @floatFromInt(total_unique)) / @as(f64, @floatFromInt(self.layers.len));
        const kv_per_layer: u64 =
            2 * @as(u64, self.config.num_key_value_heads) * (@as(u64, base_pos) + m) * self.config.head_dim * @sizeOf(f16);
        const kv_total: u64 = @as(u64, self.config.num_hidden_layers) * kv_per_layer;
        const head_bytes: u64 = if (compute_logits)
            @as(u64, m) * self.embed_row_alg_bytes + (self.global_head_bytes - self.embed_row_alg_bytes)
        else
            @as(u64, m) * self.embed_row_alg_bytes;
        const layer_weight_bytes: u64 = self.total_layers_non_expert_bytes + total_unique * self.single_expert_bytes;

        return .{
            .num_nodes = m,
            .argmax_tokens = self.out_token_ids_host[0..m],
            .logits_f16 = self.logits_host[0 .. @as(usize, m) * self.config.vocab_size],
            .total_unique_experts = total_unique,
            .avg_unique_experts_per_layer = avg_unique,
            .actual_bytes_moved = layer_weight_bytes + kv_total + head_bytes,
        };
    }

    // -----------------------------------------------------------------------
    // D4 Small Draft Model Batched Tree Expansion & KV Sync
    // -----------------------------------------------------------------------

    /// Executes a batched draft-tree forward pass for `m = token_ids.len` new
    /// tokens placed at tree KV slots `slot_start .. slot_start + m`, attending
    /// over committed prefix KV `0..kv_base_pos` and tree KV `0..slot_start + m`
    /// via `slot_ancestor_masks`, and returning GPU-computed `[rows, top_k]`
    /// softmax top-K token candidates (`encodeTopKLogitsRows`).
    ///
    /// When `commit_kv == true` (`slot_start == 0`), also commits slots `0..m`
    /// straight into committed IOSurface `KvTile`s at `kv_base_pos .. +m` in
    /// the same Metal 4 command buffer and updates `self.seq_len = kv_base_pos + m`.
    /// Zero dynamic heap allocation.
    pub fn stepDraftTreeBatch(
        self: *Qwen3Model,
        token_ids: []const u32,
        positions: []const u32,
        slot_start: u32,
        slot_ancestor_masks: []const u64,
        kv_base_pos: u32,
        top_k: u32,
        logits_last_row_only: bool,
        commit_kv: bool,
        ane_o_proj_hook: ?*const AneOProjHook,
    ) StepError![]const moe.TopKCandidate {
        if (self.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (token_ids.len == 0 or token_ids.len != positions.len) return error.InvalidArgument;
        if (top_k == 0 or top_k > moe.MAX_TOP_K) return error.InvalidArgument;
        const m: u32 = @intCast(token_ids.len);
        const total_slots: u32 = slot_start + m;
        if (total_slots > self.max_tree_nodes) return error.TreeTooLarge;
        if (slot_ancestor_masks.len < total_slots) return error.InvalidArgument;
        if (commit_kv and (slot_start != 0 or kv_base_pos + m > self.max_seq_len)) return error.PositionOutOfRange;
        if (kv_base_pos > self.max_seq_len) return error.PositionOutOfRange;

        for (0..m) |i| {
            if (token_ids[i] >= self.config.vocab_size) return error.TokenOutOfRange;
            if (positions[i] >= self.max_seq_len) return error.PositionOutOfRange;
        }
        if (self.has_pure_cpu_sme2_dense and ane_o_proj_hook == null) {
            return self.stepDraftTreeBatchSme2(
                token_ids,
                positions,
                slot_start,
                slot_ancestor_masks,
                kv_base_pos,
                top_k,
                logits_last_row_only,
                commit_kv,
            );
        }
        @memcpy(self.tree_token_ids_host[0..m], token_ids);
        @memcpy(self.tree_positions_host[0..m], positions);
        @memcpy(self.tree_ancestor_masks_host[0..total_slots], slot_ancestor_masks[0..total_slots]);

        const h = self.config.hidden_size;
        const q_dim = self.config.qDim();
        const kv_dim = self.config.kvDim();
        const q_slot_off: u64 = @as(u64, slot_start) * q_dim * @sizeOf(f16);
        const kv_slot_off: u64 = @as(u64, slot_start) * kv_dim * @sizeOf(f16);
        const q_sub_bind: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = self.q_bind.offset + q_slot_off };
        const attn_out_sub_bind: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = self.attn_out_bind.offset + q_slot_off };

        if (commit_kv) {
            try self.retireBeforeCommitRange(kv_base_pos, m);
        }

        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();

        try self.moe_pipes.encodeEmbedLookup(&frame, self.embed.dtype, .{
            .weight = self.embed.weight,
            .scale_bias = self.embed.scale_bias,
            .tokens = .{ .token_ids = self.tree_token_ids_bind },
            .out = self.x_bind,
            .m = m,
            .vocab_size = self.config.vocab_size,
            .hidden_size = h,
        }, false);

        for (self.layers, 0..) |layer, l| {
            try self.attn_pipes.encodeRmsNorm(&frame, .{
                .x = self.x_bind,
                .w = layer.input_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.input_layernorm.dtype,
            }, true);

            try self.encodeLinearBatched(&frame, layer.q_proj, self.x_norm_bind, q_sub_bind, m, true);
            try self.encodeLinearBatched(&frame, layer.k_proj, self.x_norm_bind, self.k_bind, m, false);
            try self.encodeLinearBatched(&frame, layer.v_proj, self.x_norm_bind, self.v_bind, m, false);

            const k_tree_base = self.treeKBinding(l);
            const v_tree_base = self.treeVBinding(l);
            const k_tree_sub: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = k_tree_base.offset + kv_slot_off };
            const v_tree_sub: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = v_tree_base.offset + kv_slot_off };

            try self.attn_pipes.encodeTreeQkNormRope(&frame, .{
                .q = q_sub_bind,
                .k = self.k_bind,
                .v = self.v_bind,
                .q_norm = layer.q_norm,
                .k_norm = layer.k_norm,
                .positions = self.tree_positions_bind,
                .k_tree_out = k_tree_sub,
                .v_tree_out = v_tree_sub,
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .rms_eps = self.config.rms_norm_eps,
                .rope_theta = self.config.rope_theta,
                .norm_dtype = layer.qk_norm_dtype,
            }, true);

            try self.encodeLayerGqaTreeVerify(&frame, l, total_slots, kv_base_pos, k_tree_base, v_tree_base, true);

            // Optional ANE placement override for layer 0 `o_proj` (D5).
            if (ane_o_proj_hook) |hook| {
                if (l == 0 and m <= hook.max_batch_rows and q_dim == hook.c_in and h == hook.c_out) {
                    try self.moe_pipes.encodeCopyStrided(&frame, .{
                        .src = attn_out_sub_bind,
                        .dst = hook.in_bind,
                        .num_rows = m,
                        .copy_cols = q_dim,
                        .src_stride = q_dim,
                        .src_col_off = 0,
                        .dst_stride = q_dim,
                        .dst_col_off = 0,
                    }, true);
                    const sig0 = try frame.commit();
                    try self.compute.wait(sig0, self.timeout_ms);

                    const pinfo = try hook.model.predict(hook.plan);
                    hook.last_zero_copy_ok.* = pinfo.allOutputsZeroCopy();

                    frame = try self.compute.beginFrame(self.timeout_ms);
                    try self.moe_pipes.encodeResidualAdd(&frame, .{
                        .x = self.x_bind,
                        .delta = hook.out_bind,
                        .count = m * h,
                    }, false);
                } else {
                    try self.encodeLinearBatched(&frame, layer.o_proj, attn_out_sub_bind, self.proj_out_bind, m, true);
                    try self.moe_pipes.encodeResidualAdd(&frame, .{
                        .x = self.x_bind,
                        .delta = self.proj_out_bind,
                        .count = m * h,
                    }, true);
                }
            } else {
                try self.encodeLinearBatched(&frame, layer.o_proj, attn_out_sub_bind, self.proj_out_bind, m, true);
                try self.moe_pipes.encodeResidualAdd(&frame, .{
                    .x = self.x_bind,
                    .delta = self.proj_out_bind,
                    .count = m * h,
                }, true);
            }

            try self.attn_pipes.encodeRmsNorm(&frame, .{
                .x = self.x_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, true);

            switch (self.config.arch) {
                .qwen3_dense => {
                    try self.encodeLinearBatched(&frame, layer.dense_gate, self.x_norm_bind, self.dense_gate_bind, m, true);
                    try self.encodeLinearBatched(&frame, layer.dense_up, self.x_norm_bind, self.dense_up_bind, m, false);
                    try self.moe_pipes.encodeSiluMul(&frame, .{
                        .gate = self.dense_gate_bind,
                        .up = self.dense_up_bind,
                        .out = self.dense_act_bind,
                        .count = m * self.config.intermediate_size,
                    }, true);
                    try self.encodeLinearBatched(&frame, layer.dense_down, self.dense_act_bind, self.proj_out_bind, m, true);
                    try self.moe_pipes.encodeResidualAdd(&frame, .{
                        .x = self.x_bind,
                        .delta = self.proj_out_bind,
                        .count = m * h,
                    }, true);
                },
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
                        .m = m,
                        .num_experts = self.config.num_experts,
                        .top_k = self.config.num_experts_per_tok,
                        .hidden_size = h,
                        .intermediate_size = self.config.moe_intermediate_size,
                        .norm_topk_prob = self.config.norm_topk_prob,
                    };
                    try self.encodeMoELayerDispatch(&frame, layer, moe_args, true);
                },
            }
        }

        // Optionally commit accepted KV rows straight into IOSurface `KvTile`s
        // inside the same command buffer (eliminates a separate sync roundtrip).
        if (commit_kv) {
            var acc_nodes: [attention.MAX_TREE_NODES]u16 = undefined;
            for (0..m) |i| acc_nodes[i] = @intCast(i);
            var offset: u32 = 0;
            while (offset < m) {
                const cur_base = kv_base_pos + offset;
                const chunk0: u32 = cur_base / self.chunk_tokens;
                const max_end: u32 = (chunk0 + 2) * self.chunk_tokens;
                const take: u32 = @min(m - offset, max_end - cur_base);
                const crosses = ((cur_base + take - 1) / self.chunk_tokens) > chunk0;
                for (0..self.layers.len) |l| {
                    const t0 = self.kvTileAt(l, @intCast(chunk0));
                    const t1 = if (crosses) self.kvTileAt(l, @intCast(chunk0 + 1)) else null;
                    try self.attn_pipes.encodeKvTreeCommit(&frame, .{
                        .k_tree = self.treeKBinding(l),
                        .v_tree = self.treeVBinding(l),
                        .accepted_nodes = .{ .inline_u16 = acc_nodes[offset .. offset + take] },
                        .accepted_count = take,
                        .k_tile0 = t0.k,
                        .v_tile0 = t0.v,
                        .k_tile1 = if (t1) |p| p.k else null,
                        .v_tile1 = if (t1) |p| p.v else null,
                        .num_kv_heads = self.config.num_key_value_heads,
                        .head_dim = self.config.head_dim,
                        .base_pos = cur_base,
                        .chunk_tokens = self.chunk_tokens,
                    }, false);
                }
                offset += take;
            }
        }

        const out_rows: u32 = if (logits_last_row_only) 1 else m;
        if (logits_last_row_only) {
            const last_x_bind: metal.BufferBinding = .{
                .buffer = self.scratch_buf,
                .offset = self.x_bind.offset + @as(u64, m - 1) * h * @sizeOf(f16),
            };
            try self.attn_pipes.encodeRmsNorm(&frame, .{
                .x = last_x_bind,
                .w = self.final_norm.binding,
                .y = self.x_norm_bind,
                .num_rows = 1,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = self.final_norm.dtype,
            }, true);
            try self.encodeLinearGemv(&frame, self.lm_head, self.x_norm_bind, self.logits_bind, true);
        } else {
            try self.attn_pipes.encodeRmsNorm(&frame, .{
                .x = self.x_bind,
                .w = self.final_norm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = self.final_norm.dtype,
            }, true);
            try self.encodeLinearBatched(&frame, self.lm_head, self.x_norm_bind, self.logits_bind, m, true);
        }
        try self.moe_pipes.encodeTopKLogitsRows(&frame, .{
            .logits = self.logits_bind,
            .out_candidates = self.topk_bind,
            .num_rows = out_rows,
            .vocab_size = self.config.vocab_size,
            .top_k = top_k,
            .logits_dtype = .f16,
        }, true);

        const sig = try frame.commit();
        try self.compute.wait(sig, self.timeout_ms);
        if (commit_kv) {
            self.seq_len = kv_base_pos + m;
            try self.retireExcessCompletedChunks();
        }
        return self.topk_host[0 .. @as(usize, out_rows) * top_k];
    }

    // -----------------------------------------------------------------------
    // Pure-CPU ARMv9 SME2 Off-GPU Draft Tree Execution (C12, C13, D8 PEARL)
    // -----------------------------------------------------------------------

    fn runLinearSme2Ex(
        self: *Qwen3Model,
        lin: LinearBinding,
        x_f16: []const f16,
        aux_f32: ?[]const f32,
        out_bytes: []u8,
        out_dtype: sme.OutputDType,
        epilogue: sme.EpilogueOp,
        m: u32,
        cache_hint: sme.CacheHint,
    ) StepError!void {
        const srt = self.sme_rt.?;
        const pdt = weightDTypeToPanelDType(lin.dtype);
        const packed_w = lin.sme_packed.?;
        const m_u: usize = m;
        const k_u: usize = lin.k;
        const n_u: usize = lin.n;
        const a_u16: []const u16 = if (pdt.activationIsBf16()) blk: {
            const dst = srt.x_act_u16_scratch[0 .. m_u * k_u];
            for (0..m_u * k_u) |i| {
                dst[i] = quant.f32ToBf16(@floatCast(x_f16[i]));
            }
            break :blk dst;
        } else @alignCast(std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(x_f16[0 .. m_u * k_u])));

        try srt.dispatcher.gemm(.{
            .dtype = pdt,
            .param_kind = amfParamKindToSme(lin.param_kind),
            .cache_hint = cache_hint,
            .epilogue = epilogue,
            .out_dtype = out_dtype,
        }, a_u16, packed_w, aux_f32, out_bytes, m_u, n_u, k_u);
    }

    fn cpuEmbedLookup(self: *Qwen3Model, token_ids: []const u32) void {
        const h: usize = self.config.hidden_size;
        const w_bytes = self.embed.weight_host;
        switch (self.embed.dtype) {
            .f16 => {
                const src: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, w_bytes));
                for (token_ids, 0..) |tid, r| {
                    @memcpy(self.x_host[r * h ..][0..h], src[@as(usize, tid) * h ..][0..h]);
                }
            },
            .bf16 => {
                const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, w_bytes));
                for (token_ids, 0..) |tid, r| {
                    const row_src = src[@as(usize, tid) * h ..][0..h];
                    const row_dst = self.x_host[r * h ..][0..h];
                    for (0..h) |c| row_dst[c] = @floatCast(quant.bf16ToF32(row_src[c]));
                }
            },
            .q4_affine => {
                const Ops = quant.QuantOps(.{ .bits = .q4 });
                const groups = h / 64;
                const row_w_bytes = groups * Ops.groupWeightBytes();
                const sb_all: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, self.embed.scale_bias_host.?));
                var tmp: [64]f32 = undefined;
                for (token_ids, 0..) |tid, r| {
                    const row_w = w_bytes[@as(usize, tid) * row_w_bytes ..][0..row_w_bytes];
                    const row_sb = sb_all[@as(usize, tid) * groups ..][0..groups];
                    const row_dst = self.x_host[r * h ..][0..h];
                    for (0..groups) |g| {
                        Ops.dequantizeGroup(row_w[g * 32 ..][0..32], row_sb[g], &tmp);
                        for (0..64) |j| row_dst[g * 64 + j] = @floatCast(tmp[j]);
                    }
                }
            },
            .q3_affine => {
                const Ops = quant.QuantOps(.{ .bits = .q3 });
                const groups = h / 64;
                const row_w_bytes = groups * Ops.groupWeightBytes();
                const sb_all: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, self.embed.scale_bias_host.?));
                var tmp: [64]f32 = undefined;
                for (token_ids, 0..) |tid, r| {
                    const row_w = w_bytes[@as(usize, tid) * row_w_bytes ..][0..row_w_bytes];
                    const row_sb = sb_all[@as(usize, tid) * groups ..][0..groups];
                    const row_dst = self.x_host[r * h ..][0..h];
                    for (0..groups) |g| {
                        Ops.dequantizeGroup(row_w[g * 24 ..][0..24], row_sb[g], &tmp);
                        for (0..64) |j| row_dst[g * 64 + j] = @floatCast(tmp[j]);
                    }
                }
            },
            .q2_affine => {
                const Ops = quant.QuantOps(.{ .bits = .q2 });
                const groups = h / 64;
                const row_w_bytes = groups * Ops.groupWeightBytes();
                const sb_all: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, self.embed.scale_bias_host.?));
                var tmp: [64]f32 = undefined;
                for (token_ids, 0..) |tid, r| {
                    const row_w = w_bytes[@as(usize, tid) * row_w_bytes ..][0..row_w_bytes];
                    const row_sb = sb_all[@as(usize, tid) * groups ..][0..groups];
                    const row_dst = self.x_host[r * h ..][0..h];
                    for (0..groups) |g| {
                        Ops.dequantizeGroup(row_w[g * 16 ..][0..16], row_sb[g], &tmp);
                        for (0..64) |j| row_dst[g * 64 + j] = @floatCast(tmp[j]);
                    }
                }
            },
        }
    }

    fn cpuRmsNorm(
        x: []const f16,
        norm: NormBinding,
        y: []f16,
        num_rows: usize,
        hidden_dim: usize,
        eps: f32,
    ) void {
        const w_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, norm.host_bytes));
        const inv_dim: f32 = 1.0 / @as(f32, @floatFromInt(hidden_dim));
        for (0..num_rows) |r| {
            const x_row = x[r * hidden_dim ..][0..hidden_dim];
            const y_row = y[r * hidden_dim ..][0..hidden_dim];
            var ss: f32 = 0.0;
            var i: usize = 0;
            while (i + 8 <= hidden_dim) : (i += 8) {
                const v: @Vector(8, f32) = .{
                    @floatCast(x_row[i + 0]), @floatCast(x_row[i + 1]),
                    @floatCast(x_row[i + 2]), @floatCast(x_row[i + 3]),
                    @floatCast(x_row[i + 4]), @floatCast(x_row[i + 5]),
                    @floatCast(x_row[i + 6]), @floatCast(x_row[i + 7]),
                };
                ss += @reduce(.Add, v * v);
            }
            while (i < hidden_dim) : (i += 1) {
                const vf: f32 = @floatCast(x_row[i]);
                ss += vf * vf;
            }
            const inv_rms: f32 = 1.0 / @sqrt(ss * inv_dim + eps);
            switch (norm.dtype) {
                .f16 => {
                    for (0..hidden_dim) |c| {
                        const xf: f32 = @floatCast(x_row[c]);
                        const wf: f32 = @floatCast(@as(f16, @bitCast(w_u16[c])));
                        y_row[c] = @floatCast(xf * inv_rms * wf);
                    }
                },
                .bf16 => {
                    for (0..hidden_dim) |c| {
                        const xf: f32 = @floatCast(x_row[c]);
                        const wf: f32 = quant.bf16ToF32(w_u16[c]);
                        y_row[c] = @floatCast(xf * inv_rms * wf);
                    }
                },
            }
        }
    }

    fn cpuApplyHeadNormRope(
        vec_io: []f16,
        norm_opt: ?[]const u8,
        norm_dtype: quant.FloatDType,
        inv_freq: []const f32,
        pos: u32,
        rms_eps: f32,
        out_opt: ?[]f16,
    ) void {
        const d_dim = vec_io.len;
        const half_dim = d_dim / 2;
        var buf: [256]f32 = undefined;
        if (norm_opt) |nw_bytes| {
            const w_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, nw_bytes));
            var ss: f32 = 0.0;
            for (0..d_dim) |d| {
                const v: f32 = @floatCast(vec_io[d]);
                buf[d] = v;
                ss += v * v;
            }
            const inv_rms: f32 = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(d_dim)) + rms_eps);
            switch (norm_dtype) {
                .f16 => {
                    for (0..d_dim) |d| {
                        const wf: f32 = @floatCast(@as(f16, @bitCast(w_u16[d])));
                        buf[d] = buf[d] * inv_rms * wf;
                    }
                },
                .bf16 => {
                    for (0..d_dim) |d| {
                        buf[d] = buf[d] * inv_rms * quant.bf16ToF32(w_u16[d]);
                    }
                },
            }
        } else {
            for (0..d_dim) |d| buf[d] = @floatCast(vec_io[d]);
        }

        const fpos: f32 = @floatFromInt(pos);
        const dst = out_opt orelse vec_io;
        for (0..half_dim) |j| {
            const angle = fpos * inv_freq[j];
            const c = @cos(angle);
            const s = @sin(angle);
            const x0 = buf[j];
            const x1 = buf[j + half_dim];
            dst[j] = @floatCast(x0 * c - x1 * s);
            dst[j + half_dim] = @floatCast(x1 * c + x0 * s);
        }
    }

    const CpuGqaChunkView = struct {
        k_f16: []const f16,
        v_f16: []const f16,
        k_row_stride_f16: usize,
        v_row_stride_f16: usize,
    };

    const CpuGqaAttnCtx = struct {
        q_host: []const f16,
        attn_out_host: []f16,
        tree_k: []const f16,
        tree_v: []const f16,
        chunks: []const CpuGqaChunkView,
        ancestor_masks: []const u64,
        slot_start: u32,
        total_slots: u32,
        kv_base_pos: u32,
        chunk_tokens: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
    };

    fn cpuGqaAttnTask(ctx_ptr: *anyopaque, task_idx: u32) void {
        const ctx: *const CpuGqaAttnCtx = @ptrCast(@alignCast(ctx_ptr));
        const nqh: usize = ctx.num_q_heads;
        const hd: usize = ctx.head_dim;
        const q_dim: usize = nqh * hd;
        const kv_dim: usize = @as(usize, ctx.num_kv_heads) * hd;
        const q_groups: usize = ctx.num_q_heads / ctx.num_kv_heads;

        const r: usize = @as(usize, task_idx) / nqh;
        const qh: usize = @as(usize, task_idx) % nqh;
        const kvh: usize = qh / q_groups;
        const slot: usize = @as(usize, ctx.slot_start) + r;
        const mask: u64 = ctx.ancestor_masks[slot];

        const q_row = ctx.q_host[r * q_dim + qh * hd ..][0..hd];
        var q_f32: [256]f32 = undefined;
        var acc: [256]f32 = @splat(0.0);
        for (0..hd) |d| q_f32[d] = @floatCast(q_row[d]);

        var max_score: f32 = -std.math.inf(f32);
        var sum_exp: f32 = 0.0;
        const ct: usize = ctx.chunk_tokens;

        // 1. Committed IOSurface KV chunks (`0 .. kv_base_pos`).
        var rem_pos: usize = ctx.kv_base_pos;
        var c_idx: usize = 0;
        while (rem_pos > 0) : (c_idx += 1) {
            const take = @min(ct, rem_pos);
            const cv = ctx.chunks[c_idx];
            const k_head_base = kvh * ct * cv.k_row_stride_f16;
            const v_head_base = kvh * ct * cv.v_row_stride_f16;
            for (0..take) |s_in_c| {
                const k_row = cv.k_f16[k_head_base + s_in_c * cv.k_row_stride_f16 ..][0..hd];
                const v_row = cv.v_f16[v_head_base + s_in_c * cv.v_row_stride_f16 ..][0..hd];
                var dot: f32 = 0.0;
                var d: usize = 0;
                while (d + 8 <= hd) : (d += 8) {
                    const qv: @Vector(8, f32) = q_f32[d..][0..8].*;
                    const kv: @Vector(8, f32) = .{
                        @floatCast(k_row[d + 0]), @floatCast(k_row[d + 1]),
                        @floatCast(k_row[d + 2]), @floatCast(k_row[d + 3]),
                        @floatCast(k_row[d + 4]), @floatCast(k_row[d + 5]),
                        @floatCast(k_row[d + 6]), @floatCast(k_row[d + 7]),
                    };
                    dot += @reduce(.Add, qv * kv);
                }
                while (d < hd) : (d += 1) {
                    dot += q_f32[d] * @as(f32, @floatCast(k_row[d]));
                }
                const score = dot * ctx.scale;
                if (score > max_score) {
                    const rescale = if (max_score == -std.math.inf(f32)) 0.0 else @exp(max_score - score);
                    sum_exp = sum_exp * rescale + 1.0;
                    d = 0;
                    while (d + 8 <= hd) : (d += 8) {
                        const av: @Vector(8, f32) = acc[d..][0..8].*;
                        const vv: @Vector(8, f32) = .{
                            @floatCast(v_row[d + 0]), @floatCast(v_row[d + 1]),
                            @floatCast(v_row[d + 2]), @floatCast(v_row[d + 3]),
                            @floatCast(v_row[d + 4]), @floatCast(v_row[d + 5]),
                            @floatCast(v_row[d + 6]), @floatCast(v_row[d + 7]),
                        };
                        acc[d..][0..8].* = av * @as(@Vector(8, f32), @splat(rescale)) + vv;
                    }
                    while (d < hd) : (d += 1) {
                        acc[d] = acc[d] * rescale + @as(f32, @floatCast(v_row[d]));
                    }
                    max_score = score;
                } else {
                    const w = @exp(score - max_score);
                    sum_exp += w;
                    d = 0;
                    while (d + 8 <= hd) : (d += 8) {
                        const av: @Vector(8, f32) = acc[d..][0..8].*;
                        const vv: @Vector(8, f32) = .{
                            @floatCast(v_row[d + 0]), @floatCast(v_row[d + 1]),
                            @floatCast(v_row[d + 2]), @floatCast(v_row[d + 3]),
                            @floatCast(v_row[d + 4]), @floatCast(v_row[d + 5]),
                            @floatCast(v_row[d + 6]), @floatCast(v_row[d + 7]),
                        };
                        acc[d..][0..8].* = av + vv * @as(@Vector(8, f32), @splat(w));
                    }
                    while (d < hd) : (d += 1) {
                        acc[d] += w * @as(f32, @floatCast(v_row[d]));
                    }
                }
            }
            rem_pos -= take;
        }

        // 2. Temporary tree slots (`0 .. total_slots`, masked by `ancestor_masks[slot]`).
        for (0..ctx.total_slots) |s| {
            if ((mask & (@as(u64, 1) << @as(u6, @intCast(s)))) == 0) continue;
            const k_row = ctx.tree_k[s * kv_dim + kvh * hd ..][0..hd];
            const v_row = ctx.tree_v[s * kv_dim + kvh * hd ..][0..hd];
            var dot: f32 = 0.0;
            var d: usize = 0;
            while (d + 8 <= hd) : (d += 8) {
                const qv: @Vector(8, f32) = q_f32[d..][0..8].*;
                const kv: @Vector(8, f32) = .{
                    @floatCast(k_row[d + 0]), @floatCast(k_row[d + 1]),
                    @floatCast(k_row[d + 2]), @floatCast(k_row[d + 3]),
                    @floatCast(k_row[d + 4]), @floatCast(k_row[d + 5]),
                    @floatCast(k_row[d + 6]), @floatCast(k_row[d + 7]),
                };
                dot += @reduce(.Add, qv * kv);
            }
            while (d < hd) : (d += 1) {
                dot += q_f32[d] * @as(f32, @floatCast(k_row[d]));
            }
            const score = dot * ctx.scale;
            if (score > max_score) {
                const rescale = if (max_score == -std.math.inf(f32)) 0.0 else @exp(max_score - score);
                sum_exp = sum_exp * rescale + 1.0;
                d = 0;
                while (d + 8 <= hd) : (d += 8) {
                    const av: @Vector(8, f32) = acc[d..][0..8].*;
                    const vv: @Vector(8, f32) = .{
                        @floatCast(v_row[d + 0]), @floatCast(v_row[d + 1]),
                        @floatCast(v_row[d + 2]), @floatCast(v_row[d + 3]),
                        @floatCast(v_row[d + 4]), @floatCast(v_row[d + 5]),
                        @floatCast(v_row[d + 6]), @floatCast(v_row[d + 7]),
                    };
                    acc[d..][0..8].* = av * @as(@Vector(8, f32), @splat(rescale)) + vv;
                }
                while (d < hd) : (d += 1) {
                    acc[d] = acc[d] * rescale + @as(f32, @floatCast(v_row[d]));
                }
                max_score = score;
            } else {
                const w = @exp(score - max_score);
                sum_exp += w;
                d = 0;
                while (d + 8 <= hd) : (d += 8) {
                    const av: @Vector(8, f32) = acc[d..][0..8].*;
                    const vv: @Vector(8, f32) = .{
                        @floatCast(v_row[d + 0]), @floatCast(v_row[d + 1]),
                        @floatCast(v_row[d + 2]), @floatCast(v_row[d + 3]),
                        @floatCast(v_row[d + 4]), @floatCast(v_row[d + 5]),
                        @floatCast(v_row[d + 6]), @floatCast(v_row[d + 7]),
                    };
                    acc[d..][0..8].* = av + vv * @as(@Vector(8, f32), @splat(w));
                }
                while (d < hd) : (d += 1) {
                    acc[d] += w * @as(f32, @floatCast(v_row[d]));
                }
            }
        }

        const inv_sum: f32 = if (sum_exp > 0.0) 1.0 / sum_exp else 0.0;
        const out_row = ctx.attn_out_host[r * q_dim + qh * hd ..][0..hd];
        for (0..hd) |d| out_row[d] = @floatCast(acc[d] * inv_sum);
    }

    const CpuLmHeadCtx = struct {
        lm_head: LinearBinding,
        x_f32: []const f32,
        x_group_sums: []const f32,
        tile_stats: []LmHeadTileState,
        out_rows: usize,
        vocab_size: usize,
        hidden_dim: usize,
        top_k: usize,
        rows_per_tile: usize,
        num_tiles: usize,
    };

    inline fn insertTopKLocal(st: *LmHeadTileState, top_k: usize, tok_id: u32, score: f32) void {
        if (st.count < top_k) {
            var pos: usize = st.count;
            st.count += 1;
            while (pos > 0 and score > st.top_vals[pos - 1]) : (pos -= 1) {
                st.top_vals[pos] = st.top_vals[pos - 1];
                st.top_ids[pos] = st.top_ids[pos - 1];
            }
            st.top_vals[pos] = score;
            st.top_ids[pos] = tok_id;
        } else if (score > st.top_vals[top_k - 1]) {
            var pos: usize = top_k - 1;
            while (pos > 0 and score > st.top_vals[pos - 1]) : (pos -= 1) {
                st.top_vals[pos] = st.top_vals[pos - 1];
                st.top_ids[pos] = st.top_ids[pos - 1];
            }
            st.top_vals[pos] = score;
            st.top_ids[pos] = tok_id;
        }
    }

    inline fn updateLmState(st: *LmHeadTileState, top_k: usize, tok_id: u32, score: f32) void {
        if (score > st.max_val) {
            const diff = st.max_val - score;
            st.sum_exp = (if (diff > -20.0) st.sum_exp * @exp(diff) else 0.0) + 1.0;
            st.max_val = score;
            insertTopKLocal(st, top_k, tok_id, score);
        } else {
            const diff = score - st.max_val;
            if (diff > -18.0) {
                st.sum_exp += @exp(diff);
            }
            if (st.count < top_k or score > st.top_vals[top_k - 1]) {
                insertTopKLocal(st, top_k, tok_id, score);
            }
        }
    }

    fn cpuLmHeadTask(ctx_ptr: *anyopaque, tile_idx: u32) void {
        const ctx: *const CpuLmHeadCtx = @ptrCast(@alignCast(ctx_ptr));
        const v_start = @as(usize, tile_idx) * ctx.rows_per_tile;
        if (v_start >= ctx.vocab_size) return;
        const v_end = @min(ctx.vocab_size, v_start + ctx.rows_per_tile);
        const h = ctx.hidden_dim;
        const groups = h / 64;
        const out_rows = ctx.out_rows;
        const top_k = ctx.top_k;

        var local: [attention.MAX_TREE_NODES]LmHeadTileState = undefined;
        for (0..out_rows) |r| local[r] = .{};

        switch (ctx.lm_head.dtype) {
            .q4_affine => {
                const row_w_bytes = groups * 32;
                const w_bytes = ctx.lm_head.weight_host;
                const sb_all: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, ctx.lm_head.scale_bias_host.?));
                const mask_lo: @Vector(8, u8) = @splat(0x0F);
                const shift_hi: @Vector(8, u3) = @splat(4);
                if (out_rows == 1) {
                    const x0 = ctx.x_f32[0..h];
                    const gs0 = ctx.x_group_sums[0..groups];
                    var st0: LmHeadTileState = .{};
                    for (v_start..v_end) |v| {
                        const w_row = w_bytes[v * row_w_bytes ..][0..row_w_bytes];
                        const sb_row = sb_all[v * groups ..][0..groups];
                        var dot: f32 = 0.0;
                        for (0..groups) |g| {
                            const sb = sb_row[g];
                            const scale: f32 = @floatCast(sb.scale);
                            const bias: f32 = @floatCast(sb.bias);
                            const gw = w_row[g * 32 ..][0..32];
                            const xe_base = x0[g * 64 ..][0..32];
                            const xo_base = x0[g * 64 + 32 ..][0..32];
                            var gd: @Vector(8, f32) = @splat(0.0);
                            inline for (0..4) |u| {
                                const b8: @Vector(8, u8) = gw[u * 8 ..][0..8].*;
                                const lo: @Vector(8, f32) = @floatFromInt(b8 & mask_lo);
                                const hi: @Vector(8, f32) = @floatFromInt(b8 >> shift_hi);
                                const xe: @Vector(8, f32) = xe_base[u * 8 ..][0..8].*;
                                const xo: @Vector(8, f32) = xo_base[u * 8 ..][0..8].*;
                                gd += lo * xe + hi * xo;
                            }
                            dot += scale * @reduce(.Add, gd) + bias * gs0[g];
                        }
                        updateLmState(&st0, top_k, @intCast(v), dot);
                    }
                    ctx.tile_stats[@as(usize, tile_idx)] = st0;
                    return;
                }
                for (v_start..v_end) |v| {
                    const w_row = w_bytes[v * row_w_bytes ..][0..row_w_bytes];
                    const sb_row = sb_all[v * groups ..][0..groups];
                    var dots: [attention.MAX_TREE_NODES]f32 = @splat(0.0);
                    for (0..groups) |g| {
                        const sb = sb_row[g];
                        const scale: f32 = @floatCast(sb.scale);
                        const bias: f32 = @floatCast(sb.bias);
                        const gw = w_row[g * 32 ..][0..32];
                        var lo_arr: [4]@Vector(8, f32) = undefined;
                        var hi_arr: [4]@Vector(8, f32) = undefined;
                        inline for (0..4) |u| {
                            const b8: @Vector(8, u8) = gw[u * 8 ..][0..8].*;
                            lo_arr[u] = @floatFromInt(b8 & mask_lo);
                            hi_arr[u] = @floatFromInt(b8 >> shift_hi);
                        }
                        for (0..out_rows) |r| {
                            const xe_base = ctx.x_f32[r * h + g * 64 ..][0..32];
                            const xo_base = ctx.x_f32[r * h + g * 64 + 32 ..][0..32];
                            var gd: @Vector(8, f32) = @splat(0.0);
                            inline for (0..4) |u| {
                                const xe: @Vector(8, f32) = xe_base[u * 8 ..][0..8].*;
                                const xo: @Vector(8, f32) = xo_base[u * 8 ..][0..8].*;
                                gd += lo_arr[u] * xe + hi_arr[u] * xo;
                            }
                            dots[r] += scale * @reduce(.Add, gd) + bias * ctx.x_group_sums[r * groups + g];
                        }
                    }
                    for (0..out_rows) |r| {
                        updateLmState(&local[r], top_k, @intCast(v), dots[r]);
                    }
                }
            },
            .q3_affine => {
                const Ops = quant.QuantOps(.{ .bits = .q3 });
                const row_w_bytes = groups * 24;
                const w_bytes = ctx.lm_head.weight_host;
                const sb_all: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, ctx.lm_head.scale_bias_host.?));
                var w_f: [64]f32 = undefined;
                for (v_start..v_end) |v| {
                    const w_row = w_bytes[v * row_w_bytes ..][0..row_w_bytes];
                    const sb_row = sb_all[v * groups ..][0..groups];
                    var dots: [attention.MAX_TREE_NODES]f32 = @splat(0.0);
                    for (0..groups) |g| {
                        Ops.dequantizeGroup(w_row[g * 24 ..][0..24], sb_row[g], &w_f);
                        for (0..out_rows) |r| {
                            const x_g = ctx.x_f32[r * h + g * 64 ..][0..64];
                            var gd: @Vector(8, f32) = @splat(0.0);
                            inline for (0..8) |u| {
                                const wv: @Vector(8, f32) = w_f[u * 8 ..][0..8].*;
                                const xv: @Vector(8, f32) = x_g[u * 8 ..][0..8].*;
                                gd += wv * xv;
                            }
                            dots[r] += @reduce(.Add, gd);
                        }
                    }
                    for (0..out_rows) |r| {
                        updateLmState(&local[r], top_k, @intCast(v), dots[r]);
                    }
                }
            },
            .q2_affine => {
                const Ops = quant.QuantOps(.{ .bits = .q2 });
                const row_w_bytes = groups * 16;
                const w_bytes = ctx.lm_head.weight_host;
                const sb_all: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, ctx.lm_head.scale_bias_host.?));
                var w_f: [64]f32 = undefined;
                for (v_start..v_end) |v| {
                    const w_row = w_bytes[v * row_w_bytes ..][0..row_w_bytes];
                    const sb_row = sb_all[v * groups ..][0..groups];
                    var dots: [attention.MAX_TREE_NODES]f32 = @splat(0.0);
                    for (0..groups) |g| {
                        Ops.dequantizeGroup(w_row[g * 16 ..][0..16], sb_row[g], &w_f);
                        for (0..out_rows) |r| {
                            const x_g = ctx.x_f32[r * h + g * 64 ..][0..64];
                            var gd: @Vector(8, f32) = @splat(0.0);
                            inline for (0..8) |u| {
                                const wv: @Vector(8, f32) = w_f[u * 8 ..][0..8].*;
                                const xv: @Vector(8, f32) = x_g[u * 8 ..][0..8].*;
                                gd += wv * xv;
                            }
                            dots[r] += @reduce(.Add, gd);
                        }
                    }
                    for (0..out_rows) |r| {
                        updateLmState(&local[r], top_k, @intCast(v), dots[r]);
                    }
                }
            },
            .bf16, .f16 => {
                const w_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, ctx.lm_head.weight_host));
                const is_bf16 = (ctx.lm_head.dtype == .bf16);
                for (v_start..v_end) |v| {
                    const w_row = w_u16[v * h ..][0..h];
                    var dots: [attention.MAX_TREE_NODES]f32 = @splat(0.0);
                    var c: usize = 0;
                    while (c + 8 <= h) : (c += 8) {
                        const wv: @Vector(8, f32) = if (is_bf16) .{
                            quant.bf16ToF32(w_row[c + 0]), quant.bf16ToF32(w_row[c + 1]),
                            quant.bf16ToF32(w_row[c + 2]), quant.bf16ToF32(w_row[c + 3]),
                            quant.bf16ToF32(w_row[c + 4]), quant.bf16ToF32(w_row[c + 5]),
                            quant.bf16ToF32(w_row[c + 6]), quant.bf16ToF32(w_row[c + 7]),
                        } else .{
                            @floatCast(@as(f16, @bitCast(w_row[c + 0]))), @floatCast(@as(f16, @bitCast(w_row[c + 1]))),
                            @floatCast(@as(f16, @bitCast(w_row[c + 2]))), @floatCast(@as(f16, @bitCast(w_row[c + 3]))),
                            @floatCast(@as(f16, @bitCast(w_row[c + 4]))), @floatCast(@as(f16, @bitCast(w_row[c + 5]))),
                            @floatCast(@as(f16, @bitCast(w_row[c + 6]))), @floatCast(@as(f16, @bitCast(w_row[c + 7]))),
                        };
                        for (0..out_rows) |r| {
                            const xv: @Vector(8, f32) = ctx.x_f32[r * h + c ..][0..8].*;
                            dots[r] += @reduce(.Add, wv * xv);
                        }
                    }
                    while (c < h) : (c += 1) {
                        const wf: f32 = if (is_bf16) quant.bf16ToF32(w_row[c]) else @floatCast(@as(f16, @bitCast(w_row[c])));
                        for (0..out_rows) |r| dots[r] += wf * ctx.x_f32[r * h + c];
                    }
                    for (0..out_rows) |r| {
                        updateLmState(&local[r], top_k, @intCast(v), dots[r]);
                    }
                }
            },
        }

        for (0..out_rows) |r| {
            ctx.tile_stats[@as(usize, tile_idx) * out_rows + r] = local[r];
        }
    }

    fn stepDraftTreeBatchSme2(
        self: *Qwen3Model,
        token_ids: []const u32,
        positions: []const u32,
        slot_start: u32,
        slot_ancestor_masks: []const u64,
        kv_base_pos: u32,
        top_k: u32,
        logits_last_row_only: bool,
        commit_kv: bool,
    ) StepError![]const moe.TopKCandidate {
        const srt = self.sme_rt.?;
        const m: u32 = @intCast(token_ids.len);
        const m_u: usize = m;
        const total_slots: u32 = slot_start + m;
        const h: usize = self.config.hidden_size;
        const q_dim: usize = self.config.qDim();
        const kv_dim: usize = self.config.kvDim();
        const hd: usize = self.config.head_dim;
        const inter: usize = self.config.intermediate_size;
        const ct: usize = self.chunk_tokens;
        const active_chunks: usize = if (kv_base_pos > 0) (kv_base_pos + self.chunk_tokens - 1) / self.chunk_tokens else 0;
        if (active_chunks > 64) return error.PositionOutOfRange;

        self.cpuEmbedLookup(token_ids);

        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const scratch_bytes = self.scratch_arena.bytes();

        for (self.layers, 0..) |layer, l| {
            cpuRmsNorm(self.x_host, layer.input_layernorm, self.x_norm_host, m_u, h, self.config.rms_norm_eps);

            try self.runLinearSme2Ex(
                layer.q_proj,
                self.x_norm_host,
                null,
                std.mem.sliceAsBytes(self.q_host[0 .. m_u * q_dim]),
                .f16,
                .none,
                m,
                .non_temporal,
            );
            try self.runLinearSme2Ex(
                layer.k_proj,
                self.x_norm_host,
                null,
                std.mem.sliceAsBytes(self.k_host[0 .. m_u * kv_dim]),
                .f16,
                .none,
                m,
                .non_temporal,
            );
            try self.runLinearSme2Ex(
                layer.v_proj,
                self.x_norm_host,
                null,
                std.mem.sliceAsBytes(self.v_host[0 .. m_u * kv_dim]),
                .f16,
                .none,
                m,
                .non_temporal,
            );

            const tk_off: usize = @intCast(self.tree_k_base_off + @as(u64, l) * self.tree_kv_layer_stride);
            const tv_off: usize = @intCast(self.tree_v_base_off + @as(u64, l) * self.tree_kv_layer_stride);
            const tree_k: []f16 = @alignCast(std.mem.bytesAsSlice(f16, scratch_bytes[tk_off..][0 .. @as(usize, self.max_tree_nodes) * kv_dim * @sizeOf(f16)]));
            const tree_v: []f16 = @alignCast(std.mem.bytesAsSlice(f16, scratch_bytes[tv_off..][0 .. @as(usize, self.max_tree_nodes) * kv_dim * @sizeOf(f16)]));

            // QK-Norm + NeoX RoPE + write new tree_k / tree_v slots (`slot_start .. total_slots`).
            for (0..m_u) |r| {
                const pos = positions[r];
                const slot = @as(usize, slot_start) + r;
                for (0..self.config.num_attention_heads) |qh| {
                    cpuApplyHeadNormRope(
                        self.q_host[r * q_dim + qh * hd ..][0..hd],
                        layer.q_norm_host,
                        layer.qk_norm_dtype,
                        srt.rope_inv_freq,
                        pos,
                        self.config.rms_norm_eps,
                        null,
                    );
                }
                for (0..self.config.num_key_value_heads) |kvh| {
                    cpuApplyHeadNormRope(
                        self.k_host[r * kv_dim + kvh * hd ..][0..hd],
                        layer.k_norm_host,
                        layer.qk_norm_dtype,
                        srt.rope_inv_freq,
                        pos,
                        self.config.rms_norm_eps,
                        tree_k[slot * kv_dim + kvh * hd ..][0..hd],
                    );
                }
                @memcpy(tree_v[slot * kv_dim ..][0..kv_dim], self.v_host[r * kv_dim ..][0..kv_dim]);
            }

            // Lock active committed IOSurface KV chunks read-only and run parallel GQA tree attention.
            var k_guards: [64]metal.KvTile.LockGuard = undefined;
            var v_guards: [64]metal.KvTile.LockGuard = undefined;
            var chunk_views: [64]CpuGqaChunkView = undefined;
            var locked_chunks: usize = 0;
            errdefer {
                for (0..locked_chunks) |ci| {
                    v_guards[ci].release();
                    k_guards[ci].release();
                }
            }
            for (0..active_chunks) |ci| {
                const pair = self.kvTileAt(l, ci);
                k_guards[ci] = try pair.k.lock(true);
                errdefer k_guards[ci].release();
                v_guards[ci] = try pair.v.lock(true);
                chunk_views[ci] = .{
                    .k_f16 = k_guards[ci].f16Slice(),
                    .v_f16 = v_guards[ci].f16Slice(),
                    .k_row_stride_f16 = k_guards[ci].bytes_per_row / @sizeOf(f16),
                    .v_row_stride_f16 = v_guards[ci].bytes_per_row / @sizeOf(f16),
                };
                locked_chunks += 1;
            }

            var attn_ctx: CpuGqaAttnCtx = .{
                .q_host = self.q_host,
                .attn_out_host = self.attn_out_host,
                .tree_k = tree_k,
                .tree_v = tree_v,
                .chunks = chunk_views[0..active_chunks],
                .ancestor_masks = slot_ancestor_masks,
                .slot_start = slot_start,
                .total_slots = total_slots,
                .kv_base_pos = kv_base_pos,
                .chunk_tokens = self.chunk_tokens,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .scale = attn_scale,
            };
            const attn_tasks: u32 = m * self.config.num_attention_heads;
            srt.dispatcher.parallelFor(attn_tasks, null, @ptrCast(&attn_ctx), cpuGqaAttnTask);

            while (locked_chunks > 0) {
                locked_chunks -= 1;
                try v_guards[locked_chunks].unlock();
                try k_guards[locked_chunks].unlock();
            }

            // If committing KV (`slot_start == 0`), write `tree_k[0..m]` & `tree_v[0..m]` directly
            // into the zero-copy IOSurface `KvTile`s (`row = kvh * chunk_tokens + slot_in_chunk`).
            if (commit_kv) {
                var off: usize = 0;
                while (off < m_u) {
                    const pos0 = @as(usize, kv_base_pos) + off;
                    const ci = pos0 / ct;
                    const s_in_c = pos0 % ct;
                    const take = @min(m_u - off, ct - s_in_c);
                    const pair = self.kvTileAt(l, ci);
                    var kg = try pair.k.lock(false);
                    errdefer kg.release();
                    var vg = try pair.v.lock(false);
                    errdefer vg.release();
                    const k_dst = kg.f16Slice();
                    const v_dst = vg.f16Slice();
                    const k_rs = kg.bytes_per_row / @sizeOf(f16);
                    const v_rs = vg.bytes_per_row / @sizeOf(f16);
                    for (0..take) |t_i| {
                        const src_slot = off + t_i;
                        for (0..self.config.num_key_value_heads) |kvh| {
                            const row_idx = kvh * ct + s_in_c + t_i;
                            @memcpy(k_dst[row_idx * k_rs ..][0..hd], tree_k[src_slot * kv_dim + kvh * hd ..][0..hd]);
                            @memcpy(v_dst[row_idx * v_rs ..][0..hd], tree_v[src_slot * kv_dim + kvh * hd ..][0..hd]);
                        }
                    }
                    try vg.unlock();
                    try kg.unlock();
                    off += take;
                }
            }

            // O projection + fused residual add (`x += o_proj(attn_out)`).
            for (0..m_u * h) |i| srt.x_res_f32[i] = @floatCast(self.x_host[i]);
            try self.runLinearSme2Ex(
                layer.o_proj,
                self.attn_out_host,
                srt.x_res_f32[0 .. m_u * h],
                std.mem.sliceAsBytes(self.x_host[0 .. m_u * h]),
                .f16,
                .add_residual,
                m,
                .non_temporal,
            );

            // Post-attention RMSNorm + fused SwiGLU FFN (`up` -> `gate * silu` -> `down + residual`).
            cpuRmsNorm(self.x_host, layer.post_attention_layernorm, self.x_norm_host, m_u, h, self.config.rms_norm_eps);

            try self.runLinearSme2Ex(
                layer.dense_up,
                self.x_norm_host,
                null,
                std.mem.sliceAsBytes(srt.dense_up_f32[0 .. m_u * inter]),
                .f32,
                .none,
                m,
                .non_temporal,
            );
            try self.runLinearSme2Ex(
                layer.dense_gate,
                self.x_norm_host,
                srt.dense_up_f32[0 .. m_u * inter],
                std.mem.sliceAsBytes(self.dense_act_host[0 .. m_u * inter]),
                .f16,
                .swiglu_gate_up,
                m,
                .non_temporal,
            );
            for (0..m_u * h) |i| srt.x_res_f32[i] = @floatCast(self.x_host[i]);
            try self.runLinearSme2Ex(
                layer.dense_down,
                self.dense_act_host,
                srt.x_res_f32[0 .. m_u * h],
                std.mem.sliceAsBytes(self.x_host[0 .. m_u * h]),
                .f16,
                .add_residual,
                m,
                .non_temporal,
            );
        }

        if (commit_kv) {
            self.seq_len = kv_base_pos + m;
        }
        if (top_k == 0) {
            return self.topk_host[0..0];
        }

        const out_rows: usize = if (logits_last_row_only) 1 else m_u;
        const row_off: usize = if (logits_last_row_only) m_u - 1 else 0;
        cpuRmsNorm(
            self.x_host[row_off * h ..],
            self.final_norm,
            self.x_norm_host,
            out_rows,
            h,
            self.config.rms_norm_eps,
        );

        const vocab: usize = self.config.vocab_size;
        const eff_k: usize = @min(@as(usize, top_k), moe.MAX_TOP_K);

        if (self.lm_head.layout == .sme2_panel64) {
            const pdt = weightDTypeToPanelDType(self.lm_head.dtype);
            const packed_w = self.lm_head.sme_packed.?;
            const ps = sme.panelStrideBytes(pdt, h) catch return error.InvalidArgument;
            const a_u16: []const u16 = if (pdt.activationIsBf16()) blk: {
                const dst = srt.x_act_u16_scratch[0 .. out_rows * h];
                for (0..out_rows * h) |i| {
                    dst[i] = quant.f32ToBf16(@floatCast(self.x_norm_host[i]));
                }
                break :blk dst;
            } else @alignCast(std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(self.x_norm_host[0 .. out_rows * h])));

            const chunk_panels: usize = 64; // 1024 vocab rows per SME2 strip
            const chunk_rows_cap: usize = chunk_panels * sme.PANEL_COLS;
            const total_panels = sme.panelCount(vocab);
            var chunk_buf: [attention.MAX_TREE_NODES * 1024]f32 = undefined;

            var row_states: [attention.MAX_TREE_NODES]LmHeadTileState = undefined;
            for (0..out_rows) |r| row_states[r] = .{};

            var p_off: usize = 0;
            while (p_off < total_panels) : (p_off += chunk_panels) {
                const p_cnt = @min(chunk_panels, total_panels - p_off);
                const v_start = p_off * sme.PANEL_COLS;
                const v_cnt = @min(p_cnt * sme.PANEL_COLS, vocab - v_start);
                const w_slice = packed_w[p_off * ps .. (p_off + p_cnt) * ps];
                const out_slice = chunk_buf[0 .. out_rows * v_cnt];
                try srt.dispatcher.gemmF32(.{
                    .dtype = pdt,
                    .param_kind = amfParamKindToSme(self.lm_head.param_kind),
                    .cache_hint = .non_temporal,
                }, a_u16, w_slice, null, out_slice, out_rows, v_cnt, h);
                for (0..out_rows) |r| {
                    const logits_r = out_slice[r * v_cnt ..][0..v_cnt];
                    for (0..v_cnt) |j| {
                        updateLmState(&row_states[r], eff_k, @intCast(v_start + j), logits_r[j]);
                    }
                }
            }
            _ = chunk_rows_cap;

            for (0..out_rows) |r| {
                const st = &row_states[r];
                const inv_sum: f32 = if (st.sum_exp > 0.0) 1.0 / st.sum_exp else 0.0;
                const log_sum: f32 = if (st.sum_exp > 0.0) @log(st.sum_exp) else 0.0;
                for (0..eff_k) |ki| {
                    const logit = st.top_vals[ki];
                    const valid = (ki < st.count and st.sum_exp > 0.0);
                    const prob: f32 = if (valid) @exp(logit - st.max_val) * inv_sum else 0.0;
                    const log_prob: f32 = if (valid) (logit - st.max_val - log_sum) else -std.math.inf(f32);
                    self.topk_host[r * eff_k + ki] = .{
                        .token_id = if (ki < st.count) st.top_ids[ki] else 0,
                        .prob = prob,
                        .log_prob = log_prob,
                    };
                }
            }
            return self.topk_host[0 .. out_rows * eff_k];
        }

        const groups = h / 64;
        var x_group_sums: [attention.MAX_TREE_NODES * 128]f32 = undefined;
        const split_even_odd = (self.lm_head.dtype == .q4_affine);
        for (0..out_rows) |r| {
            const x_row = self.x_norm_host[r * h ..][0..h];
            const dst_row = srt.x_f32_scratch[r * h ..][0..h];
            for (0..groups) |g| {
                var gs: f32 = 0.0;
                if (split_even_odd) {
                    for (0..32) |j| {
                        const ve: f32 = @floatCast(x_row[g * 64 + 2 * j]);
                        const vo: f32 = @floatCast(x_row[g * 64 + 2 * j + 1]);
                        dst_row[g * 64 + j] = ve;
                        dst_row[g * 64 + 32 + j] = vo;
                        gs += ve + vo;
                    }
                } else {
                    for (0..64) |j| {
                        const vf: f32 = @floatCast(x_row[g * 64 + j]);
                        dst_row[g * 64 + j] = vf;
                        gs += vf;
                    }
                }
                x_group_sums[r * groups + g] = gs;
            }
        }

        const num_tiles: usize = @min(LM_HEAD_MAX_TILES, @max(@as(usize, 1), (vocab + 2047) / 2048));
        const rows_per_tile: usize = (vocab + num_tiles - 1) / num_tiles;

        var lm_ctx: CpuLmHeadCtx = .{
            .lm_head = self.lm_head,
            .x_f32 = srt.x_f32_scratch[0 .. out_rows * h],
            .x_group_sums = x_group_sums[0 .. out_rows * groups],
            .tile_stats = srt.lm_tile_stats[0 .. num_tiles * out_rows],
            .out_rows = out_rows,
            .vocab_size = vocab,
            .hidden_dim = h,
            .top_k = eff_k,
            .rows_per_tile = rows_per_tile,
            .num_tiles = num_tiles,
        };
        srt.dispatcher.parallelFor(@intCast(num_tiles), null, @ptrCast(&lm_ctx), cpuLmHeadTask);

        // Merge per-tile online softmax & local top-K into `self.topk_host[0 .. out_rows * eff_k]`.
        for (0..out_rows) |r| {
            var g_max: f32 = -std.math.inf(f32);
            for (0..num_tiles) |ti| {
                const st = &srt.lm_tile_stats[ti * out_rows + r];
                if (st.max_val > g_max) g_max = st.max_val;
            }
            var g_sum: f32 = 0.0;
            var merged: LmHeadTileState = .{};
            for (0..num_tiles) |ti| {
                const st = &srt.lm_tile_stats[ti * out_rows + r];
                if (st.max_val > -std.math.inf(f32)) {
                    g_sum += st.sum_exp * @exp(st.max_val - g_max);
                }
                for (0..st.count) |ki| {
                    insertTopKLocal(&merged, eff_k, st.top_ids[ki], st.top_vals[ki]);
                }
            }
            const inv_sum: f32 = if (g_sum > 0.0) 1.0 / g_sum else 0.0;
            const log_sum: f32 = if (g_sum > 0.0) @log(g_sum) else 0.0;
            for (0..eff_k) |ki| {
                const logit = merged.top_vals[ki];
                const valid = (ki < merged.count and g_sum > 0.0);
                const prob: f32 = if (valid) @exp(logit - g_max) * inv_sum else 0.0;
                const log_prob: f32 = if (valid) (logit - g_max - log_sum) else -std.math.inf(f32);
                self.topk_host[r * eff_k + ki] = .{
                    .token_id = if (ki < merged.count) merged.top_ids[ki] else 0,
                    .prob = prob,
                    .log_prob = log_prob,
                };
            }
        }
        return self.topk_host[0 .. out_rows * eff_k];
    }

    // -----------------------------------------------------------------------
    // G6 + A1 Hybrid ANE/GPU Wavefront Prefill (ADR-0010)
    // -----------------------------------------------------------------------

    /// GPU FFN half of hybrid wavefront prefill (`G6`): optional attention residual
    /// add (`x_in_out += attn_out`), `post_attention_layernorm`, and batched
    /// `m`-token Dense SwiGLU (`encodeLinearGemm`) or CSR-deduplicated MoE FFN
    /// + residual add directly over zero-copy IOSurface / `WiredArena` buffers.
    /// Zero dynamic heap allocation.
    pub fn runLayerFfnChunk(
        self: *const Qwen3Model,
        frame: *metal.Compute.Frame,
        layer_idx: usize,
        x_in_out: metal.BufferBinding,
        attn_out: ?metal.BufferBinding,
        m: u32,
        insert_barrier_before: bool,
    ) StepError!void {
        if (layer_idx >= self.layers.len or m == 0) return error.InvalidArgument;
        const layer = self.layers[layer_idx];
        const h = self.config.hidden_size;

        // Select preallocated 512-token chunk workspace when hybrid prefill is initialized,
        // otherwise reuse the standard tree/decode scratch workspace for `m <= max_tree_nodes`.
        const use_hybrid_ws = if (self.hybrid_rt) |hrt| m <= hrt.chunk_tokens else false;
        if (!use_hybrid_ws and m > self.max_tree_nodes) return error.TreeTooLarge;

        const x_norm_b = if (use_hybrid_ws) self.hybrid_rt.?.x_norm_chunk_bind else self.x_norm_bind;
        const proj_out_b = if (use_hybrid_ws) self.hybrid_rt.?.proj_out_chunk_bind else self.proj_out_bind;
        const ws_buf = if (use_hybrid_ws) self.hybrid_rt.?.ffn_buf else self.scratch_buf;

        var need_barrier = insert_barrier_before;
        if (attn_out) |ao| {
            try self.moe_pipes.encodeResidualAdd(frame, .{
                .x = x_in_out,
                .delta = ao,
                .count = m * h,
            }, need_barrier);
            need_barrier = true;
        }

        try self.attn_pipes.encodeRmsNorm(frame, .{
            .x = x_in_out,
            .w = layer.post_attention_layernorm.binding,
            .y = x_norm_b,
            .num_rows = m,
            .hidden_dim = h,
            .eps = self.config.rms_norm_eps,
            .w_dtype = layer.post_attention_layernorm.dtype,
        }, need_barrier);

        switch (self.config.arch) {
            .qwen3_dense => {
                const dg_b = if (use_hybrid_ws) self.hybrid_rt.?.dense_gate_chunk_bind else self.dense_gate_bind;
                const du_b = if (use_hybrid_ws) self.hybrid_rt.?.dense_up_chunk_bind else self.dense_up_bind;
                const da_b = if (use_hybrid_ws) self.hybrid_rt.?.dense_act_chunk_bind else self.dense_act_bind;
                try self.encodeLinearGemm(frame, layer.dense_gate, x_norm_b, dg_b, m, true);
                try self.encodeLinearGemm(frame, layer.dense_up, x_norm_b, du_b, m, false);
                try self.moe_pipes.encodeSiluMul(frame, .{
                    .gate = dg_b,
                    .up = du_b,
                    .out = da_b,
                    .count = m * self.config.intermediate_size,
                }, true);
                try self.encodeLinearGemm(frame, layer.dense_down, da_b, proj_out_b, m, true);
                try self.moe_pipes.encodeResidualAdd(frame, .{
                    .x = x_in_out,
                    .delta = proj_out_b,
                    .count = m * h,
                }, true);
            },
            .qwen3_moe => {
                const ws_ptr: *const moe.MoeWorkspace = if (use_hybrid_ws)
                    &(self.hybrid_rt.?.moe_ws_chunk.?)
                else
                    &(self.moe_ws.?);
                const moe_args: moe.MoELayerArgs = .{
                    .x = x_norm_b,
                    .router = layer.router,
                    .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                    .expert_table = layer.expert_table,
                    .workspace = ws_ptr,
                    .workspace_buf = ws_buf,
                    .residual = x_in_out,
                    .out = x_in_out,
                    .m = m,
                    .num_experts = self.config.num_experts,
                    .top_k = self.config.num_experts_per_tok,
                    .hidden_size = h,
                    .intermediate_size = self.config.moe_intermediate_size,
                    .norm_topk_prob = self.config.norm_topk_prob,
                };
                try self.encodeMoELayerDispatch(frame, layer, moe_args, true);
            },
        }
    }

    /// Initializes the reusable 2-chunk Hybrid ANE/GPU Wavefront Prefill runtime
    /// (`A1`, `A3`, `G6`, `ADR-0010`), compiling static power-of-two KV-bucket ANE
    /// attention graphs per layer and preallocating double-buffered IOSurface
    /// activation tiles + 512-token GPU FFN chunk scratch.
    pub fn initHybridPrefill(
        self: *Qwen3Model,
        allocator: std.mem.Allocator,
        opts: HybridPrefillOptions,
    ) LoadError!void {
        if (self.hybrid_rt != null) return;
        const c_tok = self.chunk_tokens;
        const h = self.config.hidden_size;
        const hd = self.config.head_dim;
        if ((c_tok % metal.KV_TILE_ROW_ALIGN_F16) != 0 or
            (h % metal.KV_TILE_ROW_ALIGN_F16) != 0 or
            (hd % metal.KV_TILE_ROW_ALIGN_F16) != 0)
        {
            return error.HybridPrefillUnsupported;
        }
        const geom: wavefront.AneAttentionGeometry = .{
            .hidden_size = h,
            .num_q_heads = self.config.num_attention_heads,
            .num_kv_heads = self.config.num_key_value_heads,
            .head_dim = hd,
            .chunk_tokens = c_tok,
        };
        if (!geom.validate()) return error.HybridPrefillUnsupported;

        // Build ascending power-of-two static KV buckets starting at `chunk_tokens`.
        const max_cap: u32 = @min(opts.max_bucket, self.num_chunks * c_tok);
        var buckets: [wavefront.MAX_STATIC_BUCKETS]u32 = @splat(0);
        var num_buckets: usize = 0;
        var b_len: u32 = c_tok;
        while (num_buckets < wavefront.MAX_STATIC_BUCKETS) {
            buckets[num_buckets] = b_len;
            num_buckets += 1;
            if (b_len >= max_cap) break;
            const next_b = std.math.mul(u32, b_len, 2) catch break;
            if (next_b > opts.max_bucket * 2) break;
            b_len = next_b;
        }
        if (num_buckets == 0) return error.HybridPrefillUnsupported;

        const num_layers: usize = self.layers.len;
        const num_chunks: usize = self.num_chunks;

        // Compute GPU FFN chunk arena footprint (preallocated once for `chunk_tokens`).
        const c_usize: usize = c_tok;
        const moe_ws_bytes: usize = if (self.config.arch == .qwen3_moe)
            try moe.MoeWorkspace.requiredBytes(.{
                .max_tokens = c_tok,
                .num_experts = self.config.num_experts,
                .top_k = self.config.num_experts_per_tok,
                .hidden_size = h,
                .intermediate_size = self.config.moe_intermediate_size,
            })
        else
            0;
        const dense_bytes: usize = if (self.config.arch == .qwen3_dense)
            3 * amf.align64(c_usize * @as(usize, self.config.intermediate_size) * @sizeOf(f16))
        else
            0;
        const ffn_raw: usize =
            amf.align64(c_usize * @sizeOf(u32)) + // chunk_token_ids
            2 * amf.align64(c_usize * @as(usize, h) * @sizeOf(f16)) + // x_norm_chunk, proj_out_chunk
            moe_ws_bytes +
            dense_bytes +
            4 * page.APPLE_PAGE_SIZE;
        const ffn_arena_bytes = try page.alignUp(ffn_raw);

        // Estimate IOSurface tile bytes and verify total wired budget.
        var iosurface_bytes: u64 = 0;
        iosurface_bytes += 6 * @as(u64, try page.alignUp(c_usize * @as(usize, h) * @sizeOf(f16)));
        for (0..num_buckets) |bi| {
            const bl: usize = buckets[bi];
            iosurface_bytes += 2 * @as(u64, try page.alignUp(c_usize * bl * @sizeOf(f16)));
            if (bl > c_tok) {
                const past_rows: usize = @as(usize, self.config.num_key_value_heads) * (bl - c_usize);
                iosurface_bytes += 4 * @as(u64, try page.alignUp(past_rows * @as(usize, hd) * @sizeOf(f16)));
            }
        }
        const extra_wired: u64 = ffn_arena_bytes + iosurface_bytes;
        const rem_limit = self.wired_budget_bytes -| self.total_wired_bytes;
        const rem_budget: budget_mod.Budget = .{
            .limit_bytes = rem_limit,
            .binding_limit = .configured_cap,
            .raw_limit_bytes = rem_limit,
        };
        try rem_budget.checkPlan(extra_wired);

        var ffn_arena = try arena_mod.WiredArena.init(ffn_arena_bytes, rem_budget);
        errdefer ffn_arena.deinit();
        const ffn_buf = try metal.Buffer.wrap(self.device, ffn_arena.bytes());
        errdefer ffn_buf.close();

        const tok_raw = try ffn_arena.allocAligned(c_usize * @sizeOf(u32), 64);
        const xn_raw = try ffn_arena.allocAligned(c_usize * @as(usize, h) * @sizeOf(f16), 64);
        const po_raw = try ffn_arena.allocAligned(c_usize * @as(usize, h) * @sizeOf(f16), 64);
        @memset(tok_raw, 0);
        const moe_ws_chunk: ?moe.MoeWorkspace = if (self.config.arch == .qwen3_moe)
            try moe.MoeWorkspace.init(&ffn_arena, .{
                .max_tokens = c_tok,
                .num_experts = self.config.num_experts,
                .top_k = self.config.num_experts_per_tok,
                .hidden_size = h,
                .intermediate_size = self.config.moe_intermediate_size,
            })
        else
            null;
        var dg_b: metal.BufferBinding = .{ .buffer = ffn_buf, .offset = 0 };
        var du_b: metal.BufferBinding = .{ .buffer = ffn_buf, .offset = 0 };
        var da_b: metal.BufferBinding = .{ .buffer = ffn_buf, .offset = 0 };
        if (self.config.arch == .qwen3_dense) {
            const inter_b = c_usize * @as(usize, self.config.intermediate_size) * @sizeOf(f16);
            const dg = try ffn_arena.allocAligned(inter_b, 64);
            const du = try ffn_arena.allocAligned(inter_b, 64);
            const da = try ffn_arena.allocAligned(inter_b, 64);
            dg_b = .{ .buffer = ffn_buf, .offset = ffn_arena.offsetOf(dg.ptr) };
            du_b = .{ .buffer = ffn_buf, .offset = ffn_arena.offsetOf(du.ptr) };
            da_b = .{ .buffer = ffn_buf, .offset = ffn_arena.offsetOf(da.ptr) };
        }

        // Allocate double-buffered zero-copy IOSurface tiles.
        var x_tiles: [2]metal.KvTile = undefined;
        var x_norm_tiles: [2]metal.KvTile = undefined;
        var attn_out_tiles: [2]metal.KvTile = undefined;
        var mask_tiles: [2][wavefront.MAX_STATIC_BUCKETS]metal.KvTile = undefined;
        var k_past_tiles: [2][wavefront.MAX_STATIC_BUCKETS]?metal.KvTile = @splat(@splat(null));
        var v_past_tiles: [2][wavefront.MAX_STATIC_BUCKETS]?metal.KvTile = @splat(@splat(null));
        var slots_inited: usize = 0;
        errdefer {
            for (0..slots_inited) |s| {
                for (0..num_buckets) |bi| {
                    if (v_past_tiles[s][bi]) |t| t.close();
                    if (k_past_tiles[s][bi]) |t| t.close();
                    mask_tiles[s][bi].close();
                }
                attn_out_tiles[s].close();
                x_norm_tiles[s].close();
                x_tiles[s].close();
            }
        }
        for (0..2) |s| {
            x_tiles[s] = try metal.KvTile.init(self.device, h, c_tok, .{ .alias_buffer = true });
            errdefer x_tiles[s].close();
            x_norm_tiles[s] = try metal.KvTile.init(self.device, h, c_tok, .{ .alias_buffer = true });
            errdefer x_norm_tiles[s].close();
            attn_out_tiles[s] = try metal.KvTile.init(self.device, h, c_tok, .{ .alias_buffer = true });
            errdefer attn_out_tiles[s].close();
            var b_done: usize = 0;
            errdefer {
                for (0..b_done) |bi| {
                    if (v_past_tiles[s][bi]) |t| t.close();
                    if (k_past_tiles[s][bi]) |t| t.close();
                    mask_tiles[s][bi].close();
                }
            }
            for (0..num_buckets) |bi| {
                const bl = buckets[bi];
                mask_tiles[s][bi] = try metal.KvTile.init(self.device, bl, c_tok, .{});
                errdefer mask_tiles[s][bi].close();
                if (bl > c_tok) {
                    const past_rows = self.config.num_key_value_heads * (bl - c_tok);
                    k_past_tiles[s][bi] = try metal.KvTile.init(self.device, hd, past_rows, .{});
                    errdefer if (k_past_tiles[s][bi]) |t| t.close();
                    v_past_tiles[s][bi] = try metal.KvTile.init(self.device, hd, past_rows, .{});
                }
                b_done += 1;
            }
            slots_inited += 1;
        }

        // Compile static-shape ANE attention buckets per layer from model weights.
        const ane_buckets = try allocator.alloc(wavefront.AneAttentionBucket, num_layers * num_buckets);
        var ab_done: usize = 0;
        errdefer {
            for (0..ab_done) |i| ane_buckets[i].close();
            allocator.free(ane_buckets);
        }
        for (0..num_layers) |l| {
            const layer = self.layers[l];
            const w_q = try self.extractLinearWeightsF16(allocator, layer.shard_idx, layer.q_proj);
            defer allocator.free(w_q);
            const w_k = try self.extractLinearWeightsF16(allocator, layer.shard_idx, layer.k_proj);
            defer allocator.free(w_k);
            const w_v = try self.extractLinearWeightsF16(allocator, layer.shard_idx, layer.v_proj);
            defer allocator.free(w_v);
            const w_o = try self.extractLinearWeightsF16(allocator, layer.shard_idx, layer.o_proj);
            defer allocator.free(w_o);

            for (0..num_buckets) |bi| {
                ane_buckets[l * num_buckets + bi] = try wavefront.AneAttentionBucket.initFromWeights(
                    allocator,
                    geom,
                    buckets[bi],
                    w_q,
                    w_k,
                    w_v,
                    w_o,
                    .cpu_and_neural_engine,
                );
                ab_done += 1;
            }
        }

        // Pre-bind zero-copy `AnePlan`s for every `(layer, chunk)` writing K/V
        // straight into committed IOSurface `self.kv_tiles`.
        const bound_bucket_idx = try allocator.alloc(usize, num_chunks);
        errdefer allocator.free(bound_bucket_idx);
        const bound_uses_direct_past = try allocator.alloc(bool, num_chunks);
        errdefer allocator.free(bound_uses_direct_past);
        const bound_plans = try allocator.alloc(wavefront.AneAttentionBucket.BoundPlan, num_layers * num_chunks);
        var bp_done: usize = 0;
        errdefer {
            for (0..bp_done) |i| bound_plans[i].close();
            allocator.free(bound_plans);
        }

        for (0..num_chunks) |c_idx| {
            const total_toks: u32 = @intCast((c_idx + 1) * c_usize);
            const bi = wavefront.selectBucketIndex(total_toks, buckets[0..num_buckets]) orelse (num_buckets - 1);
            bound_bucket_idx[c_idx] = bi;
            bound_uses_direct_past[c_idx] = (c_idx == 1 and buckets[bi] == 2 * c_tok);
        }

        for (0..num_layers) |l| {
            for (0..num_chunks) |c_idx| {
                const bi = bound_bucket_idx[c_idx];
                const slot: usize = c_idx & 1;
                const ab = ane_buckets[l * num_buckets + bi];
                const out_kv = self.kvTileAt(l, c_idx);
                const k_past: ?metal.KvTile = if (buckets[bi] == c_tok)
                    null
                else if (bound_uses_direct_past[c_idx])
                    self.kvTileAt(l, 0).k
                else
                    k_past_tiles[slot][bi];
                const v_past: ?metal.KvTile = if (buckets[bi] == c_tok)
                    null
                else if (bound_uses_direct_past[c_idx])
                    self.kvTileAt(l, 0).v
                else
                    v_past_tiles[slot][bi];

                bound_plans[l * num_chunks + c_idx] = try ab.bindTiles(
                    x_norm_tiles[slot],
                    k_past,
                    v_past,
                    mask_tiles[slot][bi],
                    attn_out_tiles[slot],
                    out_kv.k,
                    out_kv.v,
                );
                bp_done += 1;
            }
        }

        // Register FFN chunk buffer and activation alias tiles in Metal 4 residency set.
        try self.compute.makeResident(&.{ffn_buf});
        try self.compute.makeTilesResident(&.{
            x_tiles[0],        x_tiles[1],
            x_norm_tiles[0],   x_norm_tiles[1],
            attn_out_tiles[0], attn_out_tiles[1],
        });

        const wf_sched = try wavefront.WavefrontPrefill.init(allocator);
        errdefer wf_sched.deinit(allocator);

        const hrt = try allocator.create(HybridPrefillRuntime);
        hrt.* = .{
            .strategy = opts.strategy,
            .chunk_tokens = c_tok,
            .num_layers = @intCast(num_layers),
            .num_chunks = @intCast(num_chunks),
            .num_buckets = num_buckets,
            .buckets = buckets,
            .wavefront_sched = wf_sched,
            .ane_buckets = ane_buckets,
            .bound_plans = bound_plans,
            .bound_bucket_idx = bound_bucket_idx,
            .bound_uses_direct_past = bound_uses_direct_past,
            .x_tiles = x_tiles,
            .x_norm_tiles = x_norm_tiles,
            .attn_out_tiles = attn_out_tiles,
            .mask_tiles = mask_tiles,
            .k_past_tiles = k_past_tiles,
            .v_past_tiles = v_past_tiles,
            .ffn_arena = ffn_arena,
            .ffn_buf = ffn_buf,
            .chunk_token_ids_host = @alignCast(std.mem.bytesAsSlice(u32, tok_raw)),
            .chunk_token_ids_bind = .{ .buffer = ffn_buf, .offset = ffn_arena.offsetOf(tok_raw.ptr) },
            .x_norm_chunk_bind = .{ .buffer = ffn_buf, .offset = ffn_arena.offsetOf(xn_raw.ptr) },
            .proj_out_chunk_bind = .{ .buffer = ffn_buf, .offset = ffn_arena.offsetOf(po_raw.ptr) },
            .moe_ws_chunk = moe_ws_chunk,
            .dense_gate_chunk_bind = dg_b,
            .dense_up_chunk_bind = du_b,
            .dense_act_chunk_bind = da_b,
            .total_wired_bytes = extra_wired,
        };
        self.hybrid_rt = hrt;
        self.total_wired_bytes += extra_wired;
    }

    fn encodeEmbedChunkForHybrid(
        self: *Qwen3Model,
        hrt: *HybridPrefillRuntime,
        frame: *metal.Compute.Frame,
        tokens: []const u32,
        chunk_idx: u32,
        slot: u1,
        barrier_before: bool,
    ) StepError!void {
        const c_tok: usize = hrt.chunk_tokens;
        const start: usize = @as(usize, chunk_idx) * c_tok;
        const valid: usize = @min(c_tok, tokens.len -| start);
        for (0..c_tok) |i| {
            hrt.chunk_token_ids_host[i] = if (i < valid) tokens[start + i] else 0;
        }
        const x_b: metal.BufferBinding = .{ .buffer = hrt.x_tiles[slot].aliasBuffer().?, .offset = 0 };
        const xn_b: metal.BufferBinding = .{ .buffer = hrt.x_norm_tiles[slot].aliasBuffer().?, .offset = 0 };
        try self.moe_pipes.encodeEmbedLookup(frame, self.embed.dtype, .{
            .weight = self.embed.weight,
            .scale_bias = self.embed.scale_bias,
            .tokens = .{ .token_ids = hrt.chunk_token_ids_bind },
            .out = x_b,
            .m = hrt.chunk_tokens,
            .vocab_size = self.config.vocab_size,
            .hidden_size = self.config.hidden_size,
        }, barrier_before);
        try self.attn_pipes.encodeRmsNorm(frame, .{
            .x = x_b,
            .w = self.layers[0].input_layernorm.binding,
            .y = xn_b,
            .num_rows = hrt.chunk_tokens,
            .hidden_dim = self.config.hidden_size,
            .eps = self.config.rms_norm_eps,
            .w_dtype = self.layers[0].input_layernorm.dtype,
        }, true);
    }

    /// Executes 2-chunk overlapping Hybrid ANE/GPU Wavefront Prefill (`A1`, `G6`,
    /// `ADR-0010`) over `tokens`, overlapping ANE attention `A(l, c_next)` (writing
    /// K/V directly into IOSurface `KvTile`s) with GPU FFN/MoE `F(l, c_cur)`
    /// (`runLayerFfnChunk`). Zero dynamic heap allocation on the hot path.
    pub fn prefillHybridWavefront(
        self: *Qwen3Model,
        tokens: []const u32,
        compute_logits: bool,
    ) StepError!HybridPrefillResult {
        const hrt = self.hybrid_rt orelse return error.HybridPrefillNotEnabled;
        if (tokens.len == 0 or tokens.len > self.max_seq_len) return error.PositionOutOfRange;
        for (tokens) |tid| {
            if (tid >= self.config.vocab_size) return error.TokenOutOfRange;
        }

        const c_tok: u32 = hrt.chunk_tokens;
        const num_chunks: u32 = @intCast((tokens.len + c_tok - 1) / c_tok);
        if (num_chunks > hrt.num_chunks) return error.PositionOutOfRange;

        // Embed chunk 0 (and chunk 1 if >= 2 chunks) into slots 0 & 1 before wavefront starts.
        {
            var frame = try self.compute.beginFrame(self.timeout_ms);
            errdefer frame.abort();
            try self.encodeEmbedChunkForHybrid(hrt, &frame, tokens, 0, 0, false);
            const sig = try frame.commit();
            try self.compute.wait(sig, self.timeout_ms);
        }
        if (num_chunks >= 2) {
            var frame = try self.compute.beginFrame(self.timeout_ms);
            errdefer frame.abort();
            try self.encodeEmbedChunkForHybrid(hrt, &frame, tokens, 1, 1, false);
            const sig = try frame.commit();
            try self.compute.wait(sig, self.timeout_ms);
        }

        const Ctx = struct {
            model: *Qwen3Model,
            hrt: *HybridPrefillRuntime,
            tokens: []const u32,
            num_chunks: u32,
            compute_logits: bool,
            all_zero_copy: bool = true,
            all_use_ane: bool = true,

            fn copyPastKv(c: *@This(), layer_idx: usize, chunk_idx: u32, slot: u1, bi: usize) !void {
                const c_u: usize = c.hrt.chunk_tokens;
                const hd: usize = c.model.config.head_dim;
                const n_kv: usize = c.model.config.num_key_value_heads;
                const past_cap: usize = c.hrt.buckets[bi] - c_u;

                var kg = try c.hrt.k_past_tiles[slot][bi].?.lock(false);
                defer kg.release();
                var vg = try c.hrt.v_past_tiles[slot][bi].?.lock(false);
                defer vg.release();
                const k_dst = kg.f16Slice();
                const v_dst = vg.f16Slice();

                for (0..chunk_idx) |pc| {
                    const pair = c.model.kvTileAt(layer_idx, pc);
                    var sk = try pair.k.lock(true);
                    defer sk.release();
                    var sv = try pair.v.lock(true);
                    defer sv.release();
                    const ks = sk.f16Slice();
                    const vs = sv.f16Slice();
                    for (0..n_kv) |h_idx| {
                        const src_off = h_idx * c_u * hd;
                        const dst_off = (h_idx * past_cap + pc * c_u) * hd;
                        @memcpy(k_dst[dst_off .. dst_off + c_u * hd], ks[src_off .. src_off + c_u * hd]);
                        @memcpy(v_dst[dst_off .. dst_off + c_u * hd], vs[src_off .. src_off + c_u * hd]);
                    }
                }
            }

            fn aneStep(c: *@This(), task: wavefront.StepTask) anyerror!void {
                const bi = c.hrt.bound_bucket_idx[task.chunk];
                const bucket_len = c.hrt.buckets[bi];
                const valid_past: u32 = task.chunk * c.hrt.chunk_tokens;
                const valid_chunk: u32 = @intCast(@min(@as(usize, c.hrt.chunk_tokens), c.tokens.len - valid_past));

                {
                    var mg = try c.hrt.mask_tiles[task.slot][bi].lock(false);
                    errdefer mg.release();
                    wavefront.fillCausalBucketMask(mg.f16Slice(), c.hrt.chunk_tokens, bucket_len, valid_past, valid_chunk);
                    try mg.unlock();
                }

                if (task.chunk > 0 and !c.hrt.bound_uses_direct_past[task.chunk]) {
                    try c.copyPastKv(task.layer, task.chunk, task.slot, bi);
                }

                const ab = c.hrt.ane_buckets[@as(usize, task.layer) * c.hrt.num_buckets + bi];
                const bp = c.hrt.bound_plans[@as(usize, task.layer) * c.hrt.num_chunks + task.chunk];
                const pinfo = try ab.run(bp);
                if (!pinfo.allOutputsZeroCopy()) c.all_zero_copy = false;
                if (!ab.model_info.usesNeuralEngine()) c.all_use_ane = false;
            }

            fn gpuStep(c: *@This(), task: wavefront.StepTask) anyerror!void {
                const m = c.hrt.chunk_tokens;
                const h = c.model.config.hidden_size;
                const x_b: metal.BufferBinding = .{ .buffer = c.hrt.x_tiles[task.slot].aliasBuffer().?, .offset = 0 };
                const xn_b: metal.BufferBinding = .{ .buffer = c.hrt.x_norm_tiles[task.slot].aliasBuffer().?, .offset = 0 };
                const ao_b: metal.BufferBinding = .{ .buffer = c.hrt.attn_out_tiles[task.slot].aliasBuffer().?, .offset = 0 };

                var frame = try c.model.compute.beginFrame(c.model.timeout_ms);
                errdefer frame.abort();

                try c.model.runLayerFfnChunk(&frame, task.layer, x_b, ao_b, m, false);

                if (task.layer + 1 < c.hrt.num_layers) {
                    try c.model.attn_pipes.encodeRmsNorm(&frame, .{
                        .x = x_b,
                        .w = c.model.layers[task.layer + 1].input_layernorm.binding,
                        .y = xn_b,
                        .num_rows = m,
                        .hidden_dim = h,
                        .eps = c.model.config.rms_norm_eps,
                        .w_dtype = c.model.layers[task.layer + 1].input_layernorm.dtype,
                    }, true);
                } else {
                    if (c.compute_logits and task.chunk + 1 == c.num_chunks) {
                        const valid_past: usize = @as(usize, task.chunk) * m;
                        const last_row: u64 = (c.tokens.len - valid_past) - 1;
                        const last_x: metal.BufferBinding = .{
                            .buffer = x_b.buffer,
                            .offset = last_row * h * @sizeOf(f16),
                        };
                        try c.model.attn_pipes.encodeRmsNorm(&frame, .{
                            .x = last_x,
                            .w = c.model.final_norm.binding,
                            .y = c.model.x_norm_bind,
                            .num_rows = 1,
                            .hidden_dim = h,
                            .eps = c.model.config.rms_norm_eps,
                            .w_dtype = c.model.final_norm.dtype,
                        }, true);
                        try c.model.encodeLinearGemv(&frame, c.model.lm_head, c.model.x_norm_bind, c.model.logits_bind, true);
                        try c.model.moe_pipes.encodeArgmax(&frame, .{
                            .logits = c.model.logits_bind,
                            .out_token_id = c.model.out_token_id_bind,
                            .vocab_size = c.model.config.vocab_size,
                        }, true);
                    }
                    // Prepare chunk `task.chunk + 2` in `task.slot` (guaranteed free
                    // before `A(0, task.chunk + 2)` by the 2-chunk wavefront schedule).
                    if (task.chunk + 2 < c.num_chunks) {
                        try c.model.encodeEmbedChunkForHybrid(c.hrt, &frame, c.tokens, task.chunk + 2, task.slot, true);
                    }
                }

                const sig = try frame.commit();
                try c.model.compute.wait(sig, c.model.timeout_ms);
            }
        };

        var ctx: Ctx = .{
            .model = self,
            .hrt = hrt,
            .tokens = tokens,
            .num_chunks = num_chunks,
            .compute_logits = compute_logits,
        };
        const wf_stats = hrt.wavefront_sched.runWavefrontTyped(
            Ctx,
            num_chunks,
            hrt.num_layers,
            &ctx,
            Ctx.aneStep,
            Ctx.gpuStep,
        ) catch return error.WavefrontFailed;

        self.seq_len = @intCast(tokens.len);
        if (compute_logits) self.last_logits_row = 0;
        return .{
            .tokens_prefilled = @intCast(tokens.len),
            .argmax_token = if (compute_logits) self.out_token_id_host.* else 0,
            .wavefront_stats = wf_stats,
            .all_outputs_zero_copy = ctx.all_zero_copy,
            .all_buckets_use_ane = ctx.all_use_ane,
        };
    }

    /// Prefills `tokens[0..tokens.len]` using Hybrid ANE/GPU Wavefront Prefill
    /// when `hybrid_rt != null` and `tokens.len > chunk_tokens` (and `prefill_mode != .gpu`),
    /// otherwise using GPU causal chunked prefill (`prefillChunk` / `prefillToken`).
    /// Returns the greedy argmax next token when `compute_logits == true`, else `0`.
    pub fn prefillPrompt(
        self: *Qwen3Model,
        tokens: []const u32,
        compute_logits: bool,
    ) StepError!u32 {
        if (tokens.len == 0) return error.InvalidArgument;
        if (self.prefill_mode != .gpu and self.hybrid_rt != null and tokens.len > self.chunk_tokens) {
            const res = try self.prefillHybridWavefront(tokens, compute_logits);
            return res.argmax_token;
        }
        if (self.max_tree_nodes > 1) {
            var pos: usize = 0;
            var last_tok: u32 = 0;
            while (pos < tokens.len) {
                const take = @min(@as(usize, self.max_tree_nodes), tokens.len - pos);
                const is_last = (pos + take == tokens.len);
                last_tok = try self.prefillChunk(tokens[pos .. pos + take], @intCast(pos), compute_logits and is_last);
                pos += take;
            }
            return last_tok;
        }
        for (tokens[0 .. tokens.len - 1], 0..) |tid, i| {
            try self.prefillToken(tid, @intCast(i));
        }
        if (compute_logits) {
            return self.stepDecode(tokens[tokens.len - 1], @intCast(tokens.len - 1));
        }
        try self.prefillToken(tokens[tokens.len - 1], @intCast(tokens.len - 1));
        return 0;
    }

    // -----------------------------------------------------------------------
    // Weight Extraction Helpers (Startup Only) & Real-Checkpoint A2 Capture
    // -----------------------------------------------------------------------

    /// Extracts a row-major `[lin.n, lin.k]` fp16 copy of a linear weight from
    /// wired shard memory (dequantizing Q3/Q4 or converting BF16 -> F16 at startup).
    pub fn extractLinearWeightsF16(
        self: *const Qwen3Model,
        allocator: std.mem.Allocator,
        shard_idx: u16,
        lin: LinearBinding,
    ) LoadError![]f16 {
        const shard_bytes = self.shards[shard_idx].arena.bytes();
        const n: usize = lin.n;
        const k: usize = lin.k;
        const out = try allocator.alloc(f16, n * k);
        errdefer allocator.free(out);
        if (lin.layout == .sme2_panel64) {
            try unpackPanel64ToF16(
                weightDTypeToPanelDType(lin.dtype),
                amfParamKindToSme(lin.param_kind),
                lin.sme_packed.?,
                n,
                k,
                out,
            );
            return out;
        }
        const w_off: usize = @intCast(lin.weight.offset);

        switch (lin.dtype) {
            .f16 => {
                const src: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, shard_bytes[w_off .. w_off + n * k * 2]));
                @memcpy(out, src);
            },
            .bf16 => {
                const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, shard_bytes[w_off .. w_off + n * k * 2]));
                for (src, 0..) |bits, i| out[i] = @floatCast(quant.bf16ToF32(bits));
            },
            .q4_affine => {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                const groups = try Ops.groupsPerRow(k);
                const rb = try Ops.rowWeightBytes(k);
                const sb_off: usize = @intCast(lin.scale_bias.?.offset);
                const sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    shard_bytes[sb_off .. sb_off + n * groups * @sizeOf(quant.ScaleBias)],
                ));
                var gbuf: [64]f32 = undefined;
                for (0..n) |row| {
                    for (0..groups) |g| {
                        const goff = w_off + row * rb + g * Ops.groupWeightBytes();
                        Ops.dequantizeGroup(shard_bytes[goff..][0..Ops.groupWeightBytes()], sb[row * groups + g], &gbuf);
                        for (0..64) |j| out[row * k + g * 64 + j] = @floatCast(gbuf[j]);
                    }
                }
            },
            .q3_affine => {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                const groups = try Ops.groupsPerRow(k);
                const rb = try Ops.rowWeightBytes(k);
                const sb_off: usize = @intCast(lin.scale_bias.?.offset);
                const sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    shard_bytes[sb_off .. sb_off + n * groups * @sizeOf(quant.ScaleBias)],
                ));
                var gbuf: [64]f32 = undefined;
                for (0..n) |row| {
                    for (0..groups) |g| {
                        const goff = w_off + row * rb + g * Ops.groupWeightBytes();
                        Ops.dequantizeGroup(shard_bytes[goff..][0..Ops.groupWeightBytes()], sb[row * groups + g], &gbuf);
                        for (0..64) |j| out[row * k + g * 64 + j] = @floatCast(gbuf[j]);
                    }
                }
            },
            .q2_affine => {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                const groups = try Ops.groupsPerRow(k);
                const rb = try Ops.rowWeightBytes(k);
                const sb_off: usize = @intCast(lin.scale_bias.?.offset);
                const sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    shard_bytes[sb_off .. sb_off + n * groups * @sizeOf(quant.ScaleBias)],
                ));
                var gbuf: [64]f32 = undefined;
                for (0..n) |row| {
                    for (0..groups) |g| {
                        const goff = w_off + row * rb + g * Ops.groupWeightBytes();
                        Ops.dequantizeGroup(shard_bytes[goff..][0..Ops.groupWeightBytes()], sb[row * groups + g], &gbuf);
                        for (0..64) |j| out[row * k + g * 64 + j] = @floatCast(gbuf[j]);
                    }
                }
            },
        }
        return out;
    }

    /// Extracts all layers' MoE router gate weights (`[num_experts, hidden_size]`
    /// fp16 per layer), optionally folding each layer's `post_attention_layernorm`
    /// channel scales (`W_eff[e, c] = W_router[e, c] * ln[c]`) so ANE cross-layer
    /// prediction over unnormalized residual `x_l` preserves exact target channel scaling.
    pub fn extractRouterWeightsF16(
        self: *const Qwen3Model,
        allocator: std.mem.Allocator,
        fold_post_attn_norm: bool,
    ) LoadError![][]f16 {
        if (self.config.arch != .qwen3_moe) return error.InvalidConfig;
        const num_layers: usize = self.layers.len;
        const n_exp: usize = self.config.num_experts;
        const h: usize = self.config.hidden_size;

        const out = try allocator.alloc([]f16, num_layers);
        var built: usize = 0;
        errdefer {
            for (0..built) |i| allocator.free(out[i]);
            allocator.free(out);
        }

        const ln_f32 = try allocator.alloc(f32, h);
        defer allocator.free(ln_f32);

        for (0..num_layers) |l| {
            const layer = self.layers[l];
            const shard_bytes = self.shards[layer.shard_idx].arena.bytes();

            if (fold_post_attn_norm) {
                const ln_off: usize = @intCast(layer.post_attention_layernorm.binding.offset);
                const raw_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, shard_bytes[ln_off .. ln_off + h * 2]));
                for (0..h) |c| ln_f32[c] = quant.floatBitsToF32(layer.post_attention_layernorm.dtype, raw_u16[c]);
            } else {
                @memset(ln_f32, 1.0);
            }

            const w_f16 = try allocator.alloc(f16, n_exp * h);
            errdefer allocator.free(w_f16);
            if (layer.router_layout == .sme2_panel64) {
                try unpackPanel64ToF16(
                    layer.router_panel_dtype orelse return error.UnexpectedTensorDType,
                    amfParamKindToSme(layer.router_param_kind),
                    layer.router_sme_packed.?,
                    n_exp,
                    h,
                    w_f16,
                );
                if (fold_post_attn_norm) {
                    for (0..n_exp) |e| {
                        for (0..h) |c| {
                            w_f16[e * h + c] = @floatCast(@as(f32, @floatCast(w_f16[e * h + c])) * ln_f32[c]);
                        }
                    }
                }
            } else switch (layer.router) {
                .weights => |rw| {
                    const r_off: usize = @intCast(rw.binding.offset);
                    switch (rw.dtype) {
                        .f16 => {
                            const src: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, shard_bytes[r_off .. r_off + n_exp * h * 2]));
                            for (0..n_exp) |e| {
                                for (0..h) |c| w_f16[e * h + c] = @floatCast(@as(f32, @floatCast(src[e * h + c])) * ln_f32[c]);
                            }
                        },
                        .bf16 => {
                            const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, shard_bytes[r_off .. r_off + n_exp * h * 2]));
                            for (0..n_exp) |e| {
                                for (0..h) |c| w_f16[e * h + c] = @floatCast(quant.bf16ToF32(src[e * h + c]) * ln_f32[c]);
                            }
                        },
                        .f32 => {
                            const src: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, shard_bytes[r_off .. r_off + n_exp * h * 4]));
                            for (0..n_exp) |e| {
                                for (0..h) |c| w_f16[e * h + c] = @floatCast(src[e * h + c] * ln_f32[c]);
                            }
                        },
                    }
                },
                .quant_weights => |rq| {
                    const w_lin: LinearBinding = .{
                        .weight = rq.qweight,
                        .scale_bias = rq.scale_bias,
                        .n = self.config.num_experts,
                        .k = self.config.hidden_size,
                        .dtype = quant.WeightDType.fromQuant(rq.bits),
                        .alg_bytes = 0,
                    };
                    const raw_f16 = try self.extractLinearWeightsF16(allocator, layer.shard_idx, w_lin);
                    defer allocator.free(raw_f16);
                    for (0..n_exp) |e| {
                        for (0..h) |c| w_f16[e * h + c] = @floatCast(@as(f32, @floatCast(raw_f16[e * h + c])) * ln_f32[c]);
                    }
                },
                .logits => return error.InvalidConfig,
            }
            out[l] = w_f16;
            built += 1;
        }
        return out;
    }

    /// Builds an `AneExpertPredictor` (`A2`) for this MoE model either from an
    /// offline W7 `.mlpackage` (`path_opt`) or by compiling folded router weights
    /// in memory onto ANE.
    pub fn buildAnePredictor(
        self: *const Qwen3Model,
        allocator: std.mem.Allocator,
        lookahead_layers: u32,
        max_tokens: u32,
        enable_async_thread: bool,
        fold_post_attn_norm: bool,
        path_opt: ?[:0]const u8,
    ) LoadError!*predictor_mod.AneExpertPredictor {
        if (self.config.arch != .qwen3_moe or self.config.num_hidden_layers < 2) {
            return error.InvalidConfig;
        }
        const clamped_la: u32 = @min(lookahead_layers, self.config.num_hidden_layers - 1);
        const pcfg: predictor_mod.PredictorConfig = .{
            .num_layers = self.config.num_hidden_layers,
            .num_experts = self.config.num_experts,
            .hidden_size = self.config.hidden_size,
            .lookahead_layers = clamped_la,
            .max_tokens = max_tokens,
            .top_k = self.config.num_experts_per_tok,
            .compute_units = .cpu_and_neural_engine,
            .enable_async_thread = enable_async_thread,
        };
        if (path_opt) |p| {
            return predictor_mod.AneExpertPredictor.loadAllLayersFromPath(allocator, self.device, pcfg, p);
        }
        const routers = try self.extractRouterWeightsF16(allocator, fold_post_attn_norm);
        defer {
            for (routers) |r| allocator.free(r);
            allocator.free(routers);
        }
        const ro_slices = try allocator.alloc([]const f16, routers.len);
        defer allocator.free(ro_slices);
        for (routers, 0..) |r, i| ro_slices[i] = r;
        return predictor_mod.AneExpertPredictor.initFromRouterWeights(allocator, self.device, pcfg, ro_slices);
    }

    /// Runs one exact decode step while capturing every layer's post-attention
    /// hidden state (`out_hidden_per_layer[l * H .. +H]`) and exact GPU-routed
    /// top-K expert IDs (`out_topk_per_layer[l * top_k .. +top_k]`) for real-model
    /// `A2` cross-layer recall evaluation (`aminfer bench-a2`).
    pub fn stepDecodeCaptureLayerRouting(
        self: *Qwen3Model,
        token_id: u32,
        pos: u32,
        out_hidden_per_layer: []f16,
        out_topk_per_layer: []u32,
    ) StepError!u32 {
        if (self.config.arch != .qwen3_moe) return error.InvalidArgument;
        if (token_id >= self.config.vocab_size) return error.TokenOutOfRange;
        if (pos >= self.max_seq_len) return error.PositionOutOfRange;
        const h: usize = self.config.hidden_size;
        const top_k: usize = self.config.num_experts_per_tok;
        if (out_hidden_per_layer.len < self.layers.len * h or
            out_topk_per_layer.len < self.layers.len * top_k)
        {
            return error.InvalidArgument;
        }

        const ws = &(self.moe_ws.?);
        const cur_seq_len: u32 = pos + 1;
        const chunk_idx: usize = @intCast(pos / self.chunk_tokens);

        {
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
            try self.encodeLayerAttentionDecode(&frame, 0, pos, cur_seq_len, chunk_idx, true);
            try self.encodeLayerRouter(&frame, 0, ws, self.scratch_buf, self.x_norm_bind, 1, true);
            const sig = try frame.commit();
            try self.compute.wait(sig, self.timeout_ms);
        }

        for (0..self.layers.len) |l| {
            @memcpy(out_hidden_per_layer[l * h .. (l + 1) * h], self.x_host[0..h]);
            @memcpy(out_topk_per_layer[l * top_k .. (l + 1) * top_k], ws.scratch_topk_experts[0..top_k]);

            var frame = try self.compute.beginFrame(self.timeout_ms);
            errdefer frame.abort();
            try self.encodeMoEGateUpDown(&frame, l, self.layers[l].expert_weights_buf, ws, self.scratch_buf, self.x_norm_bind, 1, false);
            try self.moe_pipes.encodeReduce(&frame, .{
                .workspace = ws,
                .workspace_buf = self.scratch_buf,
                .residual = self.x_bind,
                .out = self.x_bind,
                .m = 1,
                .hidden_size = self.config.hidden_size,
                .top_k = self.config.num_experts_per_tok,
            }, true);
            if (l + 1 < self.layers.len) {
                try self.encodeLayerAttentionDecode(&frame, l + 1, pos, cur_seq_len, chunk_idx, true);
                try self.encodeLayerRouter(&frame, l + 1, ws, self.scratch_buf, self.x_norm_bind, 1, true);
            } else {
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
        }

        self.seq_len = cur_seq_len;
        return self.out_token_id_host.*;
    }

    /// Benchmarks GPU single-token router gate lookahead (`post_attention_layernorm`
    /// + quantized/fp16 router proj + top-K) for `delta` layers ahead (`l + 1 .. l + lookahead`)
    /// from `hidden_f16[0..H]` (`A2` GPU baseline).
    pub fn evalGpuRouterLookahead(
        self: *Qwen3Model,
        hidden_f16: []const f16,
        current_layer: u32,
        lookahead_layers: u32,
        out_logits_per_delta: [][]f32,
    ) StepError!u64 {
        if (self.config.arch != .qwen3_moe) return error.InvalidArgument;
        const h: usize = self.config.hidden_size;
        const n_exp: usize = self.config.num_experts;
        if (hidden_f16.len < h) return error.InvalidArgument;
        const pred_layers: u32 = @min(lookahead_layers, self.config.num_hidden_layers -| (current_layer + 1));
        if (pred_layers == 0 or out_logits_per_delta.len < pred_layers) return error.InvalidArgument;

        @memcpy(self.x_host[0..h], hidden_f16[0..h]);
        const ws = &(self.moe_ws.?);
        var total_ns: u64 = 0;

        for (0..pred_layers) |d| {
            const target_l: usize = @as(usize, current_layer) + 1 + d;
            const layer = self.layers[target_l];
            const t0 = vmstat.nowNs() catch 0;
            var frame = try self.compute.beginFrame(self.timeout_ms);
            errdefer frame.abort();
            try self.attn_pipes.encodeRmsNorm(&frame, .{
                .x = self.x_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = 1,
                .hidden_dim = self.config.hidden_size,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, false);
            try self.encodeLayerRouter(&frame, target_l, ws, self.scratch_buf, self.x_norm_bind, 1, true);
            const sig = try frame.commit();
            try self.compute.wait(sig, self.timeout_ms);
            const t1 = vmstat.nowNs() catch t0;
            total_ns += t1 -| t0;
            @memcpy(out_logits_per_delta[d][0..n_exp], ws.router_logits[0..n_exp]);
        }
        return total_ns;
    }

    fn extractRouterLayerFoldedF32(
        self: *const Qwen3Model,
        allocator: std.mem.Allocator,
        layer_idx: usize,
        out_f32: []f32,
    ) LoadError!void {
        const layer = self.layers[layer_idx];
        const shard_bytes = self.shards[layer.shard_idx].arena.bytes();
        const n_exp: usize = self.config.num_experts;
        const h: usize = self.config.hidden_size;

        const ln_f32 = try allocator.alloc(f32, h);
        defer allocator.free(ln_f32);
        const ln_off: usize = @intCast(layer.post_attention_layernorm.binding.offset);
        const raw_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, shard_bytes[ln_off .. ln_off + h * 2]));
        for (0..h) |c| ln_f32[c] = quant.floatBitsToF32(layer.post_attention_layernorm.dtype, raw_u16[c]);

        if (layer.router_layout == .sme2_panel64) {
            const tmp = try allocator.alloc(f16, n_exp * h);
            defer allocator.free(tmp);
            try unpackPanel64ToF16(
                layer.router_panel_dtype orelse return error.UnexpectedTensorDType,
                amfParamKindToSme(layer.router_param_kind),
                layer.router_sme_packed.?,
                n_exp,
                h,
                tmp,
            );
            for (0..n_exp) |e| {
                for (0..h) |c| out_f32[e * h + c] = @as(f32, @floatCast(tmp[e * h + c])) * ln_f32[c];
            }
            return;
        }

        switch (layer.router) {
            .weights => |rw| {
                const r_off: usize = @intCast(rw.binding.offset);
                switch (rw.dtype) {
                    .f16 => {
                        const src: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, shard_bytes[r_off .. r_off + n_exp * h * 2]));
                        for (0..n_exp) |e| {
                            for (0..h) |c| out_f32[e * h + c] = @as(f32, @floatCast(src[e * h + c])) * ln_f32[c];
                        }
                    },
                    .bf16 => {
                        const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, shard_bytes[r_off .. r_off + n_exp * h * 2]));
                        for (0..n_exp) |e| {
                            for (0..h) |c| out_f32[e * h + c] = quant.bf16ToF32(src[e * h + c]) * ln_f32[c];
                        }
                    },
                    .f32 => {
                        const src: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, shard_bytes[r_off .. r_off + n_exp * h * 4]));
                        for (0..n_exp) |e| {
                            for (0..h) |c| out_f32[e * h + c] = src[e * h + c] * ln_f32[c];
                        }
                    },
                }
            },
            .quant_weights => |rq| {
                const w_lin: LinearBinding = .{
                    .weight = rq.qweight,
                    .scale_bias = rq.scale_bias,
                    .n = self.config.num_experts,
                    .k = self.config.hidden_size,
                    .dtype = quant.WeightDType.fromQuant(rq.bits),
                    .alg_bytes = 0,
                };
                const tmp = try self.extractLinearWeightsF16(allocator, layer.shard_idx, w_lin);
                defer allocator.free(tmp);
                for (0..n_exp) |e| {
                    for (0..h) |c| out_f32[e * h + c] = @as(f32, @floatCast(tmp[e * h + c])) * ln_f32[c];
                }
            },
            .logits => return error.InvalidConfig,
        }
    }

    fn calibrateSmeRouter(self: *Qwen3Model, srt: *SmeRuntime, any_sampled_sme2: bool) StepError!void {
        const cal_m: usize = @min(@as(usize, self.max_tree_nodes), 8);
        var cal_tokens: [8]u32 = undefined;
        for (0..cal_m) |i| {
            cal_tokens[i] = @intCast((i * 13 + 1) % self.config.vocab_size);
        }
        var cal_bs: [8 * MAX_EXPERT_PRED_LAYERS]tree_mod.ExpertBitset = undefined;

        // Warmup + 3 timed reps on L2-resident SME2 router bank.
        _ = try self.predictMoEExpertsForTokensSme2(cal_tokens[0..cal_m], &cal_bs);
        var sme_times: [3]u64 = undefined;
        for (0..3) |rep| {
            const t0 = vmstat.nowNs() catch 0;
            _ = try self.predictMoEExpertsForTokensSme2(cal_tokens[0..cal_m], &cal_bs);
            const t1 = vmstat.nowNs() catch t0;
            sme_times[rep] = t1 -| t0;
        }
        std.mem.sort(u64, &sme_times, {}, comptime std.sort.asc(u64));
        srt.router_pred_sme2_us = @as(f64, @floatFromInt(sme_times[1])) / 1_000.0;

        if (srt.router_bank_single_owner) {
            srt.router_pred_gpu_us = 0.0;
            srt.router_pred_backend = .sme2_single_owner;
            return;
        }
        if (any_sampled_sme2) {
            srt.router_pred_gpu_us = 0.0;
            srt.router_pred_backend = .sme2;
            return;
        }

        // Measure GPU baseline for telemetry & `auto` startup gate (`C13`).
        _ = try self.predictMoEExpertsForTokensGpu(cal_tokens[0..cal_m], &cal_bs);
        var gpu_times: [3]u64 = undefined;
        for (0..3) |rep| {
            const t0 = vmstat.nowNs() catch 0;
            _ = try self.predictMoEExpertsForTokensGpu(cal_tokens[0..cal_m], &cal_bs);
            const t1 = vmstat.nowNs() catch t0;
            gpu_times[rep] = t1 -| t0;
        }
        std.mem.sort(u64, &gpu_times, {}, comptime std.sort.asc(u64));
        srt.router_pred_gpu_us = @as(f64, @floatFromInt(gpu_times[1])) / 1_000.0;

        srt.router_pred_backend = switch (self.sme_router_mode) {
            .sme2 => .sme2,
            .gpu => .gpu,
            .auto => if (srt.router_pred_sme2_us <= srt.router_pred_gpu_us) .sme2 else .gpu,
        };
    }

    fn initSmeRuntime(self: *Qwen3Model, allocator: std.mem.Allocator) LoadError!void {
        if (self.sme_rt != null) return;
        const caps = try sme.SmeCaps.probe();
        if (self.sme_router_mode == .sme2 and !caps.supportsSme2() and !self.has_sme2_weights) {
            return error.Sme2Unsupported;
        }

        const is_moe = (self.config.arch == .qwen3_moe);
        const sampled: u32 = if (is_moe) @min(self.config.num_hidden_layers, MAX_EXPERT_PRED_LAYERS) else 0;
        const h: usize = self.config.hidden_size;
        const can_sme_router_pred = is_moe and self.max_tree_nodes > 1 and (h % sme.GROUP_SIZE) == 0 and sampled > 0;

        var all_sampled_sme2 = can_sme_router_pred;
        var any_sampled_sme2 = false;
        if (can_sme_router_pred) {
            for (0..sampled) |s| {
                const l_idx: usize = if (sampled == self.layers.len)
                    s
                else
                    (s * (self.layers.len - 1)) / (sampled - 1);
                if (self.layers[l_idx].router_layout == .sme2_panel64) {
                    any_sampled_sme2 = true;
                } else {
                    all_sampled_sme2 = false;
                }
            }
        }

        const want_router_pred = can_sme_router_pred and
            (all_sampled_sme2 or (self.sme_router_mode != .gpu and caps.supportsSme2()));
        if (!self.has_sme2_weights and !want_router_pred) return;

        const q_dim: usize = self.config.qDim();
        const max_k: usize = @max(h, @max(q_dim, @max(self.config.intermediate_size, self.config.moe_intermediate_size)));
        const m_cap: usize = self.max_tree_nodes;

        const srt = try allocator.create(SmeRuntime);
        errdefer allocator.destroy(srt);

        var dispatcher = try sme.SmeDispatcher.init(allocator, .{
            .a_pack_u16_cap = 32 * @max(max_k, sme.GROUP_SIZE),
            .x_sums_f32_cap = @max(@as(usize, 64), 4 * (@max(max_k, sme.GROUP_SIZE) / sme.GROUP_SIZE)),
        });
        errdefer dispatcher.deinit();

        const x_f32_scratch = try allocator.alloc(f32, m_cap * h);
        errdefer allocator.free(x_f32_scratch);

        const x_act_u16_scratch = try allocator.alignedAlloc(u16, .fromByteUnits(sme.SVL_BYTES), m_cap * max_k);
        errdefer allocator.free(x_act_u16_scratch);

        const router_logits_f32: []align(sme.SVL_BYTES) f32 = if (is_moe)
            try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), m_cap * @max(@as(usize, 1), @as(usize, sampled)) * self.config.num_experts)
        else
            &.{};
        errdefer if (router_logits_f32.len > 0) allocator.free(router_logits_f32);

        var any_sme2_experts = false;
        for (self.layers) |layer| {
            if (layer.expert_layout == .sme2_panel64) {
                any_sme2_experts = true;
                break;
            }
        }
        const moe_inter: usize = self.config.moe_intermediate_size;
        const expert_up_f32: []align(sme.SVL_BYTES) f32 = if (any_sme2_experts)
            try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), m_cap * moe_inter)
        else
            &.{};
        errdefer if (expert_up_f32.len > 0) allocator.free(expert_up_f32);

        const expert_inter_u16: []align(sme.SVL_BYTES) u16 = if (any_sme2_experts)
            try allocator.alignedAlloc(u16, .fromByteUnits(sme.SVL_BYTES), m_cap * moe_inter)
        else
            &.{};
        errdefer if (expert_inter_u16.len > 0) allocator.free(expert_inter_u16);

        const expert_down_f32: []align(sme.SVL_BYTES) f32 = if (any_sme2_experts)
            try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), m_cap * h)
        else
            &.{};
        errdefer if (expert_down_f32.len > 0) allocator.free(expert_down_f32);

        const expert_acc_f32: []align(sme.SVL_BYTES) f32 = if (any_sme2_experts)
            try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), m_cap * h)
        else
            &.{};
        errdefer if (expert_acc_f32.len > 0) allocator.free(expert_acc_f32);

        var all_dense_sme2 = (self.config.arch == .qwen3_dense and
            self.max_tree_nodes > 1 and
            self.embed.layout == .row_major and
            self.layers.len > 0 and
            self.config.head_dim <= 256 and
            (self.config.head_dim % 2) == 0);
        if (all_dense_sme2) {
            for (self.layers) |layer| {
                if (layer.q_proj.layout != .sme2_panel64 or
                    layer.k_proj.layout != .sme2_panel64 or
                    layer.v_proj.layout != .sme2_panel64 or
                    layer.o_proj.layout != .sme2_panel64 or
                    layer.dense_gate.layout != .sme2_panel64 or
                    layer.dense_up.layout != .sme2_panel64 or
                    layer.dense_down.layout != .sme2_panel64)
                {
                    all_dense_sme2 = false;
                    break;
                }
            }
        }
        const dense_inter: usize = self.config.intermediate_size;
        const dense_up_f32: []align(sme.SVL_BYTES) f32 = if (all_dense_sme2)
            try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), m_cap * dense_inter)
        else
            &.{};
        errdefer if (dense_up_f32.len > 0) allocator.free(dense_up_f32);

        const x_res_f32: []align(sme.SVL_BYTES) f32 = if (all_dense_sme2)
            try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), m_cap * h)
        else
            &.{};
        errdefer if (x_res_f32.len > 0) allocator.free(x_res_f32);

        const rope_inv_freq: []f32 = if (all_dense_sme2) blk: {
            const half_dim: usize = self.config.head_dim / 2;
            const buf = try allocator.alloc(f32, half_dim);
            const log_theta: f32 = @log(self.config.rope_theta);
            const inv_d: f32 = 2.0 / @as(f32, @floatFromInt(self.config.head_dim));
            for (0..half_dim) |j| {
                buf[j] = @exp(-log_theta * (@as(f32, @floatFromInt(j)) * inv_d));
            }
            break :blk buf;
        } else &.{};
        errdefer if (rope_inv_freq.len > 0) allocator.free(rope_inv_freq);

        const lm_tile_stats: []LmHeadTileState = if (all_dense_sme2)
            try allocator.alloc(LmHeadTileState, LM_HEAD_MAX_TILES * m_cap)
        else
            &.{};
        errdefer if (lm_tile_stats.len > 0) allocator.free(lm_tile_stats);

        // Optional L2-resident folded router bank (`<= 8 MiB`) when `.amf` routers are
        // `row_major`. Single-owner `sme2_panel64` routers consume ZERO extra bytes.
        var bank_arena: ?arena_mod.WiredArena = null;
        errdefer if (bank_arena) |*a| a.deinit();
        var bank_packed: []const u8 = &.{};
        var bank_dtype: sme.PanelDType = .bf16;
        var bank_arena_added: u64 = 0;

        if (want_router_pred and !all_sampled_sme2) {
            const total_n: usize = @as(usize, sampled) * self.config.num_experts;
            const payload_bytes = try sme.packedBlobBytes(.bf16, total_n, h);
            const l2_bank_cap: usize = 8 * 1024 * 1024;
            if (payload_bytes <= l2_bank_cap) {
                const arena_bytes = try page.alignUp(payload_bytes);
                const rem_limit = self.wired_budget_bytes -| self.total_wired_bytes;
                if (arena_bytes <= rem_limit) {
                    const rem_budget: budget_mod.Budget = .{
                        .limit_bytes = rem_limit,
                        .binding_limit = .configured_cap,
                        .raw_limit_bytes = rem_limit,
                    };
                    var ba = try arena_mod.WiredArena.init(arena_bytes, rem_budget);
                    errdefer ba.deinit();

                    const folded_f32 = try allocator.alloc(f32, total_n * h);
                    defer allocator.free(folded_f32);
                    const layer_elems = @as(usize, self.config.num_experts) * h;
                    for (0..sampled) |s| {
                        const l_idx: usize = if (sampled == self.layers.len)
                            s
                        else
                            (s * (self.layers.len - 1)) / (sampled - 1);
                        try self.extractRouterLayerFoldedF32(allocator, l_idx, folded_f32[s * layer_elems ..][0..layer_elems]);
                    }

                    var max_abs: f32 = 0.0;
                    for (folded_f32) |v| {
                        const av = @abs(v);
                        if (av > max_abs) max_abs = av;
                    }
                    bank_dtype = if (caps.has_sme_f16f32 and max_abs <= 32768.0) .f16 else .bf16;

                    const dst = try ba.allocAligned(payload_bytes, sme.SVL_BYTES);
                    try sme.packDenseFromF32Panel64(bank_dtype, folded_f32, total_n, h, dst);
                    bank_packed = dst;
                    bank_arena = ba;
                    bank_arena_added = arena_bytes;
                }
            }
        }

        srt.* = .{
            .dispatcher = dispatcher,
            .x_f32_scratch = x_f32_scratch,
            .x_act_u16_scratch = x_act_u16_scratch,
            .router_logits_f32 = router_logits_f32,
            .expert_up_f32 = expert_up_f32,
            .expert_inter_u16 = expert_inter_u16,
            .expert_down_f32 = expert_down_f32,
            .expert_acc_f32 = expert_acc_f32,
            .dense_up_f32 = dense_up_f32,
            .x_res_f32 = x_res_f32,
            .rope_inv_freq = rope_inv_freq,
            .lm_tile_stats = lm_tile_stats,
            .router_bank_arena = bank_arena,
            .router_bank_packed = bank_packed,
            .router_bank_dtype = bank_dtype,
            .router_bank_sampled = if (want_router_pred) sampled else 0,
            .router_bank_single_owner = all_sampled_sme2,
            .router_pred_backend = if (all_sampled_sme2) .sme2_single_owner else .gpu,
            .router_pred_sme2_us = 0.0,
            .router_pred_gpu_us = 0.0,
        };
        // Ownership transferred into `srt`; clear local errdefer handle.
        bank_arena = null;
        self.sme_rt = srt;
        self.has_pure_cpu_sme2_dense = all_dense_sme2;
        self.total_wired_bytes += bank_arena_added;

        if (want_router_pred and (all_sampled_sme2 or srt.router_bank_packed.len > 0)) {
            self.calibrateSmeRouter(srt, any_sampled_sme2) catch return error.SmeCalibrationFailed;
            if (srt.router_pred_backend == .gpu) {
                if (srt.router_bank_arena) |*a| {
                    const reclaimed: u64 = a.bytes().len;
                    a.deinit();
                    srt.router_bank_arena = null;
                    srt.router_bank_packed = &.{};
                    self.total_wired_bytes -|= reclaimed;
                }
            }
        }
    }
};

fn weightDTypeToPanelDType(wdt: quant.WeightDType) sme.PanelDType {
    return switch (wdt) {
        .f16 => .f16,
        .bf16 => .bf16,
        .q4_affine => .q4_affine,
        .q3_affine => .q3_affine,
        .q2_affine => .q2_affine,
    };
}

fn amfDTypeToPanelDType(dt: amf.DType) LoadError!sme.PanelDType {
    return switch (dt) {
        .f16 => .f16,
        .bf16 => .bf16,
        .q4_affine => .q4_affine,
        .q3_affine => .q3_affine,
        .q2_affine => .q2_affine,
        .f32, .u32 => error.UnexpectedTensorDType,
    };
}

fn amfParamKindToSme(pk: amf.QuantParamKind) sme.QuantParamKind {
    return switch (pk) {
        .affine_scale_bias => .affine_scale_bias,
        .lut16_codebook => .lut16_codebook,
    };
}

/// Startup-only unpacker from `sme2_panel64` `[n, k]` into row-major `[]f16`
/// (used by optional ANE / hybrid prefill startup extractors).
fn unpackPanel64ToF16(
    dtype: sme.PanelDType,
    param_kind: sme.QuantParamKind,
    w_packed: []const u8,
    n: usize,
    k: usize,
    out: []f16,
) LoadError!void {
    const need = try sme.packedBlobBytes(dtype, n, k);
    if (w_packed.len < need or out.len < n * k) return error.UnexpectedTensorShape;
    const panels = sme.panelCount(n);
    const groups = k / sme.GROUP_SIZE;
    const g_bytes = sme.groupBytes(dtype);
    const ps = groups * g_bytes;

    for (0..panels) |p| {
        const col_base = p * sme.PANEL_COLS;
        const valid_cols = @min(sme.PANEL_COLS, n -| col_base);
        for (0..groups) |g| {
            const grp = w_packed[p * ps + g * g_bytes ..][0..g_bytes];
            const k_base = g * sme.GROUP_SIZE;
            switch (dtype) {
                .f16, .bf16 => {
                    const w_dt: quant.FloatDType = if (dtype == .bf16) .bf16 else .f16;
                    for (0..32) |kp| {
                        const c_off = kp * 64;
                        for (0..valid_cols) |c| {
                            const row = col_base + c;
                            const b0 = std.mem.readInt(u16, grp[c_off + c * 4 + 0 ..][0..2], .little);
                            const b1 = std.mem.readInt(u16, grp[c_off + c * 4 + 2 ..][0..2], .little);
                            out[row * k + k_base + 2 * kp] = @floatCast(quant.floatBitsToF32(w_dt, b0));
                            out[row * k + k_base + 2 * kp + 1] = @floatCast(quant.floatBitsToF32(w_dt, b1));
                        }
                    }
                },
                .q4_affine, .q3_affine, .q2_affine => {
                    var scales: [sme.PANEL_COLS]f32 = undefined;
                    var biases: [sme.PANEL_COLS]f32 = undefined;
                    var lut: [16]f32 = undefined;
                    for (0..16) |idx| {
                        const h_bits = std.mem.readInt(u16, grp[idx * 4 ..][0..2], .little);
                        const v: f32 = @floatCast(@as(f16, @bitCast(h_bits)));
                        scales[idx] = v;
                        lut[idx] = v;
                    }
                    for (0..sme.PANEL_COLS) |c| {
                        const b_bits = std.mem.readInt(u16, grp[64 + c * 4 ..][0..2], .little);
                        biases[c] = @floatCast(@as(f16, @bitCast(b_bits)));
                    }
                    const codes = grp[sme.QUANT_HEADER_BYTES..];
                    switch (dtype) {
                        .q4_affine => {
                            for (0..32) |kp| {
                                for (0..valid_cols) |c| {
                                    const row = col_base + c;
                                    const byte = codes[kp * 16 + c];
                                    const q0 = byte & 0x0F;
                                    const q1 = byte >> 4;
                                    const w0 = if (param_kind == .affine_scale_bias) @as(f32, @floatFromInt(q0)) * scales[c] + biases[c] else lut[q0] + biases[c];
                                    const w1 = if (param_kind == .affine_scale_bias) @as(f32, @floatFromInt(q1)) * scales[c] + biases[c] else lut[q1] + biases[c];
                                    out[row * k + k_base + 2 * kp] = @floatCast(w0);
                                    out[row * k + k_base + 2 * kp + 1] = @floatCast(w1);
                                }
                            }
                        },
                        .q3_affine => {
                            for (0..2) |half| {
                                const h_off = half * 192;
                                var quads: [4][64]u8 = undefined;
                                for (0..64) |i| {
                                    const b0 = codes[h_off + 0 + i];
                                    const b1 = codes[h_off + 64 + i];
                                    const b2 = codes[h_off + 128 + i];
                                    quads[0][i] = b0 & 0x77;
                                    quads[1][i] = b1 & 0x77;
                                    quads[2][i] = b2 & 0x77;
                                    quads[3][i] = ((b0 & 0x88) >> 3) | ((b1 & 0x88) >> 2) | ((b2 & 0x88) >> 1);
                                }
                                for (0..4) |q| {
                                    for (0..4) |kp_sub| {
                                        const kp = half * 16 + q * 4 + kp_sub;
                                        for (0..valid_cols) |c| {
                                            const row = col_base + c;
                                            const byte = quads[q][kp_sub * 16 + c];
                                            const q0 = byte & 0x07;
                                            const q1 = (byte >> 4) & 0x07;
                                            const w0 = if (param_kind == .affine_scale_bias) @as(f32, @floatFromInt(q0)) * scales[c] + biases[c] else lut[q0] + biases[c];
                                            const w1 = if (param_kind == .affine_scale_bias) @as(f32, @floatFromInt(q1)) * scales[c] + biases[c] else lut[q1] + biases[c];
                                            out[row * k + k_base + 2 * kp] = @floatCast(w0);
                                            out[row * k + k_base + 2 * kp + 1] = @floatCast(w1);
                                        }
                                    }
                                }
                            }
                        },
                        .q2_affine => {
                            for (0..32) |kp| {
                                for (0..8) |b| {
                                    const byte = codes[kp * 8 + b];
                                    const c0 = 2 * b;
                                    const c1 = c0 + 1;
                                    if (c0 < valid_cols) {
                                        const row0 = col_base + c0;
                                        const q00 = byte & 0x03;
                                        const q01 = (byte >> 2) & 0x03;
                                        const w00 = if (param_kind == .affine_scale_bias) @as(f32, @floatFromInt(q00)) * scales[c0] + biases[c0] else lut[q00] + biases[c0];
                                        const w01 = if (param_kind == .affine_scale_bias) @as(f32, @floatFromInt(q01)) * scales[c0] + biases[c0] else lut[q01] + biases[c0];
                                        out[row0 * k + k_base + 2 * kp] = @floatCast(w00);
                                        out[row0 * k + k_base + 2 * kp + 1] = @floatCast(w01);
                                    }
                                    if (c1 < valid_cols) {
                                        const row1 = col_base + c1;
                                        const q10 = (byte >> 4) & 0x03;
                                        const q11 = (byte >> 6) & 0x03;
                                        const w10 = if (param_kind == .affine_scale_bias) @as(f32, @floatFromInt(q10)) * scales[c1] + biases[c1] else lut[q10] + biases[c1];
                                        const w11 = if (param_kind == .affine_scale_bias) @as(f32, @floatFromInt(q11)) * scales[c1] + biases[c1] else lut[q11] + biases[c1];
                                        out[row1 * k + k_base + 2 * kp] = @floatCast(w10);
                                        out[row1 * k + k_base + 2 * kp + 1] = @floatCast(w11);
                                    }
                                }
                            }
                        },
                        else => unreachable,
                    }
                },
            }
        }
    }
}

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
    if ((try entry.tensorLayout()) != .row_major) return error.InvalidTensorLayout;
    return switch (dt) {
        .f16 => .f16,
        .bf16 => .bf16,
        else => error.UnexpectedTensorDType,
    };
}

fn validateLinear2D(entry: *const amf.TensorEntry, n: u32, k: u32) LoadError!quant.WeightDType {
    const dt = try entry.dataType();
    if (entry.rank != 2 or entry.shape[0] != n or entry.shape[1] != k) return error.UnexpectedTensorShape;
    const layout = try entry.tensorLayout();
    _ = try entry.quantParamKind();
    if (layout == .sme2_panel64 and (k % sme.GROUP_SIZE) != 0) return error.UnalignedK;
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

fn validateRouter2D(entry: *const amf.TensorEntry, num_experts: u32, hidden_size: u32) LoadError!amf.DType {
    const dt = try entry.dataType();
    if (entry.rank != 2 or entry.shape[0] != num_experts or entry.shape[1] != hidden_size) {
        return error.UnexpectedTensorShape;
    }
    const layout = try entry.tensorLayout();
    _ = try entry.quantParamKind();
    if (layout == .sme2_panel64) {
        if ((hidden_size % sme.GROUP_SIZE) != 0) return error.UnalignedK;
        if (dt == .f32 or dt == .u32) return error.UnexpectedTensorDType;
    }
    switch (dt) {
        .f16, .bf16, .f32 => {},
        .q4_affine, .q3_affine, .q2_affine => {
            if ((hidden_size % 64) != 0) return error.UnalignedK;
        },
        .u32 => return error.UnexpectedTensorDType,
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
        return .{
            .binding = self.asBinding(),
            .dtype = dtype,
            .host_bytes = self.slice[0..@intCast(self.entry.data_bytes)],
        };
    }

    fn asLinearBinding(self: LoadedBlob, dtype: quant.WeightDType) LoadError!LinearBinding {
        const layout = try self.entry.tensorLayout();
        const param_kind = try self.entry.quantParamKind();
        const sb: ?metal.BufferBinding = if (layout == .sme2_panel64)
            null
        else switch (dtype) {
            .f16, .bf16 => null,
            .q4_affine, .q3_affine, .q2_affine => .{ .buffer = self.buffer, .offset = self.offset + self.entry.params_offset },
        };
        const sb_host: ?[]const u8 = if (layout == .sme2_panel64)
            null
        else switch (dtype) {
            .f16, .bf16 => null,
            .q4_affine, .q3_affine, .q2_affine => self.slice[@intCast(self.entry.params_offset)..][0..@intCast(self.entry.params_bytes)],
        };
        return .{
            .weight = .{ .buffer = self.buffer, .offset = self.offset },
            .scale_bias = sb,
            .n = self.entry.shape[0],
            .k = self.entry.shape[1],
            .dtype = dtype,
            .alg_bytes = self.entry.data_bytes + self.entry.params_bytes,
            .layout = layout,
            .param_kind = param_kind,
            .sme_packed = if (layout == .sme2_panel64) self.slice[0..@intCast(self.entry.data_bytes)] else null,
            .weight_host = self.slice[0..@intCast(self.entry.data_bytes)],
            .scale_bias_host = sb_host,
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

/// Parses a W8 `schema_version = 1` expert heat profile JSON (`aminfer-tools expert-heat`)
/// into per-layer cold-expert bitsets (`cold_masks[l]`). When `override_cold_per_layer > 0`
/// and `ranked_hot_to_cold` is present, selects the coldest `override_cold_per_layer` experts
/// per layer; otherwise uses the profile's precomputed `cold_experts` array.
fn parseHeatProfileColdMask(
    allocator: std.mem.Allocator,
    json_bytes: []const u8,
    num_layers: usize,
    num_experts: u32,
    override_cold_per_layer: u32,
    cold_masks: []tree_mod.ExpertBitset,
) LoadError!void {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidHeatProfile,
    };
    defer parsed.deinit();

    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.InvalidHeatProfile,
    };
    const ver = jsonOptU32(root, "schema_version") orelse return error.InvalidHeatProfile;
    if (ver != 1) return error.InvalidHeatProfile;

    const layers_val = root.get("layers") orelse return error.InvalidHeatProfile;
    const layers_arr = switch (layers_val) {
        .array => |a| a,
        else => return error.InvalidHeatProfile,
    };

    for (layers_arr.items) |item| {
        const lobj = switch (item) {
            .object => |o| o,
            else => return error.InvalidHeatProfile,
        };
        const l_u32 = jsonOptU32(lobj, "layer") orelse return error.InvalidHeatProfile;
        if (l_u32 >= num_layers) continue;
        const l_idx: usize = @intCast(l_u32);

        if (override_cold_per_layer > 0) {
            if (lobj.get("ranked_hot_to_cold")) |rv| {
                const rarr = switch (rv) {
                    .array => |a| a,
                    else => return error.InvalidHeatProfile,
                };
                const cold_n: usize = @min(@as(usize, override_cold_per_layer), rarr.items.len);
                const start = rarr.items.len - cold_n;
                for (rarr.items[start..]) |ev| {
                    const eid: u32 = switch (ev) {
                        .integer => |i| if (i >= 0 and i < num_experts) @intCast(i) else return error.InvalidHeatProfile,
                        else => return error.InvalidHeatProfile,
                    };
                    cold_masks[l_idx].set(eid);
                }
                continue;
            }
        }
        if (lobj.get("cold_experts")) |cv| {
            const carr = switch (cv) {
                .array => |a| a,
                else => return error.InvalidHeatProfile,
            };
            for (carr.items) |ev| {
                const eid: u32 = switch (ev) {
                    .integer => |i| if (i >= 0 and i < num_experts) @intCast(i) else return error.InvalidHeatProfile,
                    else => return error.InvalidHeatProfile,
                };
                cold_masks[l_idx].set(eid);
            }
        }
    }
}

fn loadInternal(
    allocator: std.mem.Allocator,
    source: AmfSource,
    config: ModelConfig,
    options: LoadOptions,
) LoadError!Qwen3Model {
    var model = try loadCore(allocator, source, config, options);
    errdefer model.deinit(allocator);

    if (model.cold_rt) |crt| {
        if (options.enable_ane_predictor and config.num_hidden_layers >= 2) {
            const pred_max_tok: u32 = if (options.ane_predictor_path != null)
                32
            else if (model.max_tree_nodes > 1)
                @min(model.max_tree_nodes, 32)
            else
                1;
            const lookahead: u32 = @min(@as(u32, 3), config.num_hidden_layers - 1);
            crt.ane_predictor = try model.buildAnePredictor(
                allocator,
                lookahead,
                pred_max_tok,
                true,
                true,
                options.ane_predictor_path,
            );
        }
    }

    if (options.enable_hybrid_prefill or options.prefill_mode == .hybrid) {
        try model.initHybridPrefill(allocator, .{
            .strategy = options.hybrid_strategy,
            .max_bucket = options.hybrid_max_bucket,
        });
    }

    try model.initSmeRuntime(allocator);

    return model;
}

fn loadCore(
    allocator: std.mem.Allocator,
    source: AmfSource,
    config: ModelConfig,
    options: LoadOptions,
) LoadError!Qwen3Model {
    try config.validate();
    if (options.max_seq_len == 0 or options.chunk_tokens == 0) return error.InvalidConfig;
    if (options.max_tree_nodes == 0 or options.max_tree_nodes > attention.MAX_TREE_NODES) {
        return error.InvalidConfig;
    }
    const max_tree_nodes: u32 = options.max_tree_nodes;
    const aux_layer_ids: [3]u32 = options.aux_layer_ids orelse defaultEagle3AuxLayerIds(config.num_hidden_layers);
    if (options.enable_feature_taps) {
        for (aux_layer_ids) |tap_l| {
            if (tap_l >= config.num_hidden_layers) return error.InvalidConfig;
        }
    }

    const max_seq_len = @min(options.max_seq_len, config.max_position_embeddings);
    const num_chunks: u32 = (max_seq_len + options.chunk_tokens - 1) / options.chunk_tokens;

    // Worst-case dispatches in one stepDecode / verifyTree command buffer:
    // 1 embed + 15 per uniform layer (+ up to 2*(NUM_SCHEME_BUCKETS-1) for hetero MoE) + 3 aux taps + 3 global.
    const per_layer_max_dispatches: u64 = if (config.arch == .qwen3_moe)
        15 + 2 * (moe.NUM_SCHEME_BUCKETS - 1)
    else
        15;
    const max_dispatches: u64 = 1 + @as(u64, config.num_hidden_layers) * per_layer_max_dispatches + 6;
    if (max_dispatches > metal.UNIFORM_SLOTS_PER_FRAME) {
        return error.TooManyDispatchesForFrame;
    }

    const parsed = source.parsed();
    const num_layers: usize = config.num_hidden_layers;
    const h = config.hidden_size;
    const q_dim = config.qDim();
    const kv_dim = config.kvDim();

    // --- 0. Resolve MoE cold-expert placement (W8 heat profile / CLI count / AMF tags) ---
    var cold_masks = try allocator.alloc(tree_mod.ExpertBitset, num_layers);
    var cold_masks_owned = true;
    defer if (cold_masks_owned) allocator.free(cold_masks);
    for (cold_masks) |*m| m.* = tree_mod.ExpertBitset.empty();

    var layer_has_cold = try allocator.alloc(bool, num_layers);
    var layer_has_cold_owned = true;
    defer if (layer_has_cold_owned) allocator.free(layer_has_cold);
    @memset(layer_has_cold, false);

    if (config.arch == .qwen3_moe) {
        if (options.heat_profile_json) |hp_json| {
            try parseHeatProfileColdMask(
                allocator,
                hp_json,
                num_layers,
                config.num_experts,
                options.cold_experts_per_layer,
                cold_masks,
            );
        } else if (options.cold_experts_per_layer > 0) {
            const cold_n: u32 = @min(options.cold_experts_per_layer, config.num_experts);
            const first_cold: u32 = config.num_experts - cold_n;
            for (0..num_layers) |l| {
                for (first_cold..config.num_experts) |e_idx| {
                    cold_masks[l].set(@intCast(e_idx));
                }
            }
        }
    }

    // --- 1. Resolve & validate all required directory entries up front ---
    const embed_entry = try findLinear(parsed, "embed_tokens");
    const embed_wdt = try validateLinear2D(embed_entry, config.vocab_size, h);
    if ((try embed_entry.tensorLayout()) != .row_major) return error.InvalidTensorLayout;

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

    var total_cold_experts: u32 = 0;
    var max_cold_gate_bytes: usize = 0;
    var max_cold_up_bytes: usize = 0;
    var max_cold_down_bytes: usize = 0;

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

                var layer_exp_layout: ?amf.TensorLayout = null;
                for (0..config.num_experts) |e_idx| {
                    const gp = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e_idx }) catch unreachable);
                    const gp_dt = try validateLinear2D(gp, config.moe_intermediate_size, h);
                    const gp_layout = try gp.tensorLayout();
                    const gp_pk = try gp.quantParamKind();
                    if (gp_dt == .f16 or gp_dt == .bf16) {
                        if ((h % 4) != 0 or (config.moe_intermediate_size % 4) != 0) return error.UnalignedK;
                    }
                    if (layer_exp_layout) |exp_lay| {
                        if (gp_layout != exp_lay) return error.InvalidTensorLayout;
                    } else {
                        layer_exp_layout = gp_layout;
                    }

                    // ADR-0017 strict per-expert uniformity: gate/up/down within expert (l, e)
                    // MUST share one WeightDType, TensorLayout, and QuantParamKind.
                    const up = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e_idx }) catch unreachable);
                    if ((try validateLinear2D(up, config.moe_intermediate_size, h)) != gp_dt) return error.UnexpectedTensorDType;
                    if ((try up.tensorLayout()) != gp_layout) return error.InvalidTensorLayout;
                    if ((try up.quantParamKind()) != gp_pk) return error.InvalidQuantParamKind;

                    const dp = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e_idx }) catch unreachable);
                    if ((try validateLinear2D(dp, h, config.moe_intermediate_size)) != gp_dt) return error.UnexpectedTensorDType;
                    if ((try dp.tensorLayout()) != gp_layout) return error.InvalidTensorLayout;
                    if ((try dp.quantParamKind()) != gp_pk) return error.InvalidQuantParamKind;

                    if (layer_exp_layout.? == .sme2_panel64) {
                        // Single-owner SME2 experts live in wired shard RAM; cold-tagged ones
                        // stream via `ldnt1b` (`CacheHint.non_temporal`) in `runMoeExpertsSme2`.
                        bytes_l += gp.blob_bytes + up.blob_bytes + dp.blob_bytes;
                    } else {
                        // Also honor experts explicitly tagged `TensorKind.cold_expert` in `.amf`.
                        if ((try gp.tensorKind()) == .cold_expert or
                            (try up.tensorKind()) == .cold_expert or
                            (try dp.tensorKind()) == .cold_expert)
                        {
                            cold_masks[l].set(@intCast(e_idx));
                        }

                        if (cold_masks[l].contains(@intCast(e_idx))) {
                            layer_has_cold[l] = true;
                            total_cold_experts += 1;
                            max_cold_gate_bytes = @max(max_cold_gate_bytes, @as(usize, @intCast(gp.blob_bytes)));
                            max_cold_up_bytes = @max(max_cold_up_bytes, @as(usize, @intCast(up.blob_bytes)));
                            max_cold_down_bytes = @max(max_cold_down_bytes, @as(usize, @intCast(dp.blob_bytes)));
                        } else {
                            bytes_l += gp.blob_bytes + up.blob_bytes + dp.blob_bytes;
                        }
                    }
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

    const has_cold = total_cold_experts > 0;
    var cold_pool_spec: cache_mod.PoolSpec = .{
        .num_slots = 0,
        .gate_blob_bytes = 0,
        .up_blob_bytes = 0,
        .down_blob_bytes = 0,
    };
    var cold_pool_bytes: u64 = 0;
    if (has_cold) {
        if (source.filePath() == null) return error.ColdExpertsRequireFile;
        const min_slots: u32 = @max(config.num_experts_per_tok, @min(config.num_experts, max_tree_nodes * config.num_experts_per_tok));
        const eff_slots: u32 = @max(options.cold_slot_count, min_slots);
        cold_pool_spec = .{
            .num_slots = eff_slots,
            .gate_blob_bytes = max_cold_gate_bytes,
            .up_blob_bytes = max_cold_up_bytes,
            .down_blob_bytes = max_cold_down_bytes,
        };
        try cold_pool_spec.validate();
        cold_pool_bytes = try cold_pool_spec.totalBytes();
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
    // When cold experts are active and all hot weights + `ExpertSlotPool` fit in
    // ONE shard, place them together in shard 0 so hot experts and pinned cold
    // slots share one `MTLBuffer` and execute in a single zero-copy GPU dispatch!
    const group_shard = try allocator.alloc(usize, group_count);
    defer allocator.free(group_shard);
    var shard_sizes: std.ArrayList(u64) = .empty;
    defer shard_sizes.deinit(allocator);

    var all_groups_sum: u64 = 0;
    for (group_bytes) |gb| all_groups_sum += gb;
    const pool_in_shard0 = has_cold and (all_groups_sum + cold_pool_bytes <= shard_cap);

    if (pool_in_shard0) {
        try shard_sizes.append(allocator, all_groups_sum + cold_pool_bytes);
        @memset(group_shard, 0);
    } else {
        for (group_bytes, 0..) |gb, g_idx| {
            if (gb > shard_cap) return error.TensorExceedsShardLimit;
            if (shard_sizes.items.len == 0 or shard_sizes.items[shard_sizes.items.len - 1] + gb > shard_cap) {
                try shard_sizes.append(allocator, gb);
            } else {
                shard_sizes.items[shard_sizes.items.len - 1] += gb;
            }
            group_shard[g_idx] = shard_sizes.items.len - 1;
        }
    }

    // Compute scratch arena footprint (scaled by `max_tree_nodes` for zero hot-path alloc).
    const m_cap: usize = max_tree_nodes;
    const tree_enabled = max_tree_nodes > 1;
    const retire_after: u32 = options.kv_retire_q8_after_chunks;
    const use_retired_q8 = (retire_after > 0 and num_chunks > retire_after);
    if (use_retired_q8 and (config.head_dim % attention.KV_Q8_GROUP_SIZE) != 0) {
        return error.InvalidConfig;
    }
    const active_kv_chunks: u32 = if (use_retired_q8)
        @min(num_chunks, if (tree_enabled) @max(retire_after, 2) else retire_after)
    else
        num_chunks;
    const max_retired_per_layer: u32 = if (use_retired_q8) num_chunks - retire_after else 0;
    const retired_pool_spec: attention.RetiredKvQ8PoolSpec = .{
        .num_kv_heads = config.num_key_value_heads,
        .head_dim = config.head_dim,
        .chunk_tokens = options.chunk_tokens,
        .max_chunks = if (use_retired_q8) @as(u32, @intCast(num_layers)) * max_retired_per_layer else 1,
    };
    const retired_pool_bytes: usize = if (use_retired_q8)
        try attention.RetiredKvQ8Pool.requiredArenaBytes(retired_pool_spec)
    else
        0;
    const retired_ids_per_layer: usize = if (use_retired_q8)
        amf.align64(@as(usize, max_retired_per_layer) * @sizeOf(u32))
    else
        0;
    const gqa_layout = attention.gqaScratchLayout(config.num_attention_heads, config.head_dim, attention.MAX_SPLITS);
    const gqa_tree_bytes: usize = if (tree_enabled)
        amf.align64(attention.gqaTreeScratchLayout(max_tree_nodes, config.num_attention_heads, config.head_dim, attention.MAX_SPLITS).total_bytes)
    else
        0;
    const kv_tbl_per_layer = amf.align64(@as(usize, active_kv_chunks) * @sizeOf(attention.KvTilePair));
    const expert_tbl_per_layer: usize = if (config.arch == .qwen3_moe)
        amf.align64(@as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights))
    else
        0;
    const expert_schemes_per_layer: usize = if (config.arch == .qwen3_moe)
        amf.align64(@as(usize, config.num_experts))
    else
        0;
    const moe_ws_bytes: usize = if (config.arch == .qwen3_moe)
        try moe.MoeWorkspace.requiredBytes(.{
            .max_tokens = max_tree_nodes,
            .num_experts = config.num_experts,
            .top_k = config.num_experts_per_tok,
            .hidden_size = h,
            .intermediate_size = config.moe_intermediate_size,
        })
    else
        0;
    const layer_sched_hdr_bytes: usize = if (config.arch == .qwen3_moe and tree_enabled)
        num_layers * 64
    else
        0;
    const pred_sampled_layers: usize = if (config.arch == .qwen3_moe and tree_enabled)
        @min(num_layers, MAX_EXPERT_PRED_LAYERS)
    else
        0;
    const pred_topk_layer_stride: usize = if (pred_sampled_layers > 0)
        amf.align64(m_cap * @as(usize, config.num_experts_per_tok) * @sizeOf(u32))
    else
        0;
    const pred_topk_total_bytes: usize = pred_sampled_layers * pred_topk_layer_stride;
    const dense_ffn_scratch: usize = if (config.arch == .qwen3_dense)
        3 * amf.align64(m_cap * @as(usize, config.intermediate_size) * @sizeOf(f16))
    else
        0;
    const tree_kv_layer_stride: usize = if (tree_enabled)
        amf.align64(m_cap * @as(usize, kv_dim) * @sizeOf(f16))
    else
        0;
    const topk_scratch_bytes: usize = if (tree_enabled)
        amf.align64(m_cap * moe.MAX_TOP_K * @sizeOf(moe.TopKCandidate))
    else
        0;
    const tree_meta_bytes: usize = if (tree_enabled)
        amf.align64(m_cap * @sizeOf(u32)) * 3 + // token_ids, positions, accepted_nodes
            amf.align64(m_cap * @sizeOf(u64)) + // ancestor_masks
            topk_scratch_bytes +
            2 * num_layers * tree_kv_layer_stride // per-layer tree K & V
    else
        0;
    const feature_tap_bytes: usize = if (options.enable_feature_taps)
        2 * amf.align64(m_cap * 3 * @as(usize, h) * @sizeOf(f16)) + // aux_concat, accepted_aux
            amf.align64(m_cap * @as(usize, h) * @sizeOf(f16)) // accepted_top
    else
        0;

    const scratch_raw: usize =
        2 * amf.align64(m_cap * @as(usize, h) * @sizeOf(f16)) + // x, x_norm
        2 * amf.align64(m_cap * @as(usize, q_dim) * @sizeOf(f16)) + // q, attn_out
        2 * amf.align64(m_cap * @as(usize, kv_dim) * @sizeOf(f16)) + // k, v
        amf.align64(m_cap * @as(usize, h) * @sizeOf(f16)) + // proj_out
        amf.align64(m_cap * @as(usize, config.vocab_size) * @sizeOf(f16)) + // logits
        amf.align64(m_cap * @sizeOf(u32)) + // out_token_ids
        amf.align64(gqa_layout.total_bytes) +
        gqa_tree_bytes +
        tree_meta_bytes +
        feature_tap_bytes +
        num_layers * kv_tbl_per_layer +
        num_layers * retired_ids_per_layer +
        retired_pool_bytes +
        num_layers * expert_tbl_per_layer +
        num_layers * expert_schemes_per_layer +
        moe_ws_bytes +
        layer_sched_hdr_bytes +
        pred_topk_total_bytes +
        dense_ffn_scratch +
        8 * page.APPLE_PAGE_SIZE;
    const scratch_arena_bytes = try page.alignUp(scratch_raw);

    // Compute total KV tile bytes (exact 16 KiB multiple per tile).
    const kv_tile_rows: u32 = config.num_key_value_heads * options.chunk_tokens;
    const kv_tile_alloc_bytes: u64 = try page.alignUp(@as(usize, kv_tile_rows) * config.head_dim * @sizeOf(f16));
    const total_kv_tiles: usize = num_layers * @as(usize, active_kv_chunks) * 2;
    const total_kv_bytes: u64 = @as(u64, total_kv_tiles) * kv_tile_alloc_bytes;
    const retired_kv_wired_bytes: u64 = if (use_retired_q8)
        @as(u64, try attention.RetiredKvQ8Pool.dataRequiredBytes(retired_pool_spec))
    else
        0;

    var total_weight_bytes: u64 = 0;
    for (shard_sizes.items) |sb| total_weight_bytes += sb;
    const owned_pool_bytes: u64 = if (has_cold and !pool_in_shard0) cold_pool_bytes else 0;
    const total_wired_bytes: u64 = total_weight_bytes + owned_pool_bytes + scratch_arena_bytes + total_kv_bytes;

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

    // Allocate R5 retired `q8` KV pool inside `scratch_arena` when enabled (page-aligned first!).
    const retired_kv_pool: ?attention.RetiredKvQ8Pool = if (use_retired_q8)
        try attention.RetiredKvQ8Pool.init(&scratch_arena, retired_pool_spec)
    else
        null;

    // --- 5. Allocate IOSurface RGBA16Float KV tiles per layer & active chunk ---
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
    const active_tile_phys_ring = try allocator.alloc(u32, active_kv_chunks);
    errdefer allocator.free(active_tile_phys_ring);
    for (0..active_kv_chunks) |i| active_tile_phys_ring[i] = @intCast(i);

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

    // --- 6b. Initialize heap-pinned ColdRuntime (`C3`-`C5`) when cold experts exist ---
    var cold_rt: ?*ColdRuntime = null;
    errdefer if (cold_rt) |crt| crt.deinit(allocator);
    if (has_cold) {
        const crt = try allocator.create(ColdRuntime);
        errdefer allocator.destroy(crt);

        var dfile = try aio.DirectFile.openReadOnly(source.filePath().?);
        errdefer dfile.close();

        var pool = if (pool_in_shard0)
            try cache_mod.ExpertSlotPool.initFromArena(&shards[0].arena, shards[0].buffer, cold_pool_spec)
        else
            try cache_mod.ExpertSlotPool.initOwned(wired_budget, device, cold_pool_spec);
        errdefer pool.deinit();

        if (!pool_in_shard0) {
            try compute.makeResident(&.{pool.metal_buffer.?});
        }

        const reader_inflight: usize = @min(aio.MAX_INFLIGHT, @max(@as(usize, 32), @as(usize, cold_pool_spec.num_slots) * 3));
        var reader = try aio.AsyncExpertReader.init(allocator, .{
            .backend = options.io_backend,
            .max_inflight = reader_inflight,
            .worker_count = 4,
            .worker_qos = .utility,
            .io_policy = .important,
        });
        errdefer reader.deinit(allocator);

        crt.* = .{
            .file = dfile,
            .pool = pool,
            .reader = reader,
            .expert_cache = undefined,
            .ane_predictor = null,
            .cold_masks_by_layer = cold_masks,
            .layer_has_cold = layer_has_cold,
            .total_cold_experts = total_cold_experts,
            .pred_bitsets = @splat(tree_mod.ExpertBitset.empty()),
            .pred_cands = undefined,
            .cold_bool_scratch = @splat(false),
            .ane_req_seq = 1,
            .ane_inflight = false,
        };
        crt.expert_cache = try cache_mod.ExpertCache.init(allocator, &crt.pool, &crt.reader, .{
            .num_layers = @intCast(num_layers),
            .num_experts = @intCast(config.num_experts),
            .io_timeout_ms = options.timeout_ms,
        });
        cold_masks_owned = false;
        layer_has_cold_owned = false;
        cold_rt = crt;
    }

    // --- 7. Carve hot-path scratch buffers & populate bindless KV tile tables ---
    const x_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const x_norm_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const q_raw = try scratch_arena.allocAligned(m_cap * @as(usize, q_dim) * @sizeOf(f16), 64);
    const k_raw = try scratch_arena.allocAligned(m_cap * @as(usize, kv_dim) * @sizeOf(f16), 64);
    const v_raw = try scratch_arena.allocAligned(m_cap * @as(usize, kv_dim) * @sizeOf(f16), 64);
    const attn_out_raw = try scratch_arena.allocAligned(m_cap * @as(usize, q_dim) * @sizeOf(f16), 64);
    const proj_out_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const logits_raw = try scratch_arena.allocAligned(m_cap * @as(usize, config.vocab_size) * @sizeOf(f16), 64);
    const out_tok_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u32), 64);
    const gqa_scratch_raw = try scratch_arena.allocAligned(gqa_layout.total_bytes, 64);
    @memset(q_raw, 0);
    @memset(out_tok_raw, 0);

    var tree_token_ids_host: []u32 = &.{};
    var tree_positions_host: []u32 = &.{};
    var tree_ancestor_masks_host: []u64 = &.{};
    var tree_accepted_nodes_host: []u32 = &.{};
    var topk_host: []moe.TopKCandidate = &.{};
    var tree_token_ids_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    var tree_positions_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    var tree_ancestor_masks_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    var tree_accepted_nodes_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    var topk_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    var tree_k_base_off: u64 = 0;
    var tree_v_base_off: u64 = 0;
    var gqa_tree_scratch_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    if (tree_enabled) {
        const tt_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u32), 64);
        const tp_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u32), 64);
        const tm_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u64), 64);
        const ta_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u32), 64);
        const topk_raw = try scratch_arena.allocAligned(m_cap * moe.MAX_TOP_K * @sizeOf(moe.TopKCandidate), 64);
        const tk_raw = try scratch_arena.allocAligned(num_layers * tree_kv_layer_stride, 64);
        const tv_raw = try scratch_arena.allocAligned(num_layers * tree_kv_layer_stride, 64);
        const gt_raw = try scratch_arena.allocAligned(gqa_tree_bytes, 64);
        @memset(tt_raw, 0);
        @memset(tp_raw, 0);
        @memset(tm_raw, 0);
        @memset(ta_raw, 0);
        @memset(topk_raw, 0);
        tree_token_ids_host = @alignCast(std.mem.bytesAsSlice(u32, tt_raw));
        tree_positions_host = @alignCast(std.mem.bytesAsSlice(u32, tp_raw));
        tree_ancestor_masks_host = @alignCast(std.mem.bytesAsSlice(u64, tm_raw));
        tree_accepted_nodes_host = @alignCast(std.mem.bytesAsSlice(u32, ta_raw));
        topk_host = @alignCast(std.mem.bytesAsSlice(moe.TopKCandidate, topk_raw));
        tree_token_ids_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tt_raw.ptr) };
        tree_positions_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tp_raw.ptr) };
        tree_ancestor_masks_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tm_raw.ptr) };
        tree_accepted_nodes_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(ta_raw.ptr) };
        topk_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(topk_raw.ptr) };
        tree_k_base_off = scratch_arena.offsetOf(tk_raw.ptr);
        tree_v_base_off = scratch_arena.offsetOf(tv_raw.ptr);
        gqa_tree_scratch_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gt_raw.ptr) };
    }

    var aux_concat_host: []f16 = &.{};
    var aux_concat_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    var accepted_aux_host: []f16 = &.{};
    var accepted_aux_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    var accepted_top_host: []f16 = &.{};
    var accepted_top_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    if (options.enable_feature_taps) {
        const aux_bytes = m_cap * 3 * @as(usize, h) * @sizeOf(f16);
        const top_bytes = m_cap * @as(usize, h) * @sizeOf(f16);
        const ac_raw = try scratch_arena.allocAligned(aux_bytes, 64);
        const aa_raw = try scratch_arena.allocAligned(aux_bytes, 64);
        const at_raw = try scratch_arena.allocAligned(top_bytes, 64);
        @memset(ac_raw, 0);
        @memset(aa_raw, 0);
        @memset(at_raw, 0);
        aux_concat_host = @alignCast(std.mem.bytesAsSlice(f16, ac_raw));
        aux_concat_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(ac_raw.ptr) };
        accepted_aux_host = @alignCast(std.mem.bytesAsSlice(f16, aa_raw));
        accepted_aux_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(aa_raw.ptr) };
        accepted_top_host = @alignCast(std.mem.bytesAsSlice(f16, at_raw));
        accepted_top_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(at_raw.ptr) };
    }

    const moe_ws: ?moe.MoeWorkspace = if (config.arch == .qwen3_moe)
        try moe.MoeWorkspace.init(&scratch_arena, .{
            .max_tokens = max_tree_nodes,
            .num_experts = config.num_experts,
            .top_k = config.num_experts_per_tok,
            .hidden_size = h,
            .intermediate_size = config.moe_intermediate_size,
        })
    else
        null;

    var layer_sched_hdr_base_off: u64 = 0;
    var layer_sched_hdr_raw: []u8 = &.{};
    var pred_topk_base_off: u64 = 0;
    var pred_topk_host: []u32 = &.{};
    if (config.arch == .qwen3_moe and tree_enabled) {
        layer_sched_hdr_raw = try scratch_arena.allocAligned(layer_sched_hdr_bytes, 64);
        @memset(layer_sched_hdr_raw, 0);
        layer_sched_hdr_base_off = scratch_arena.offsetOf(layer_sched_hdr_raw.ptr);

        const pt_raw = try scratch_arena.allocAligned(pred_topk_total_bytes, 64);
        @memset(pt_raw, 0);
        pred_topk_base_off = scratch_arena.offsetOf(pt_raw.ptr);
        pred_topk_host = @alignCast(std.mem.bytesAsSlice(u32, pt_raw));
    }

    var dense_gate_host: []f16 = &.{};
    var dense_up_host: []f16 = &.{};
    var dense_act_host: []f16 = &.{};
    var dense_gate_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    var dense_up_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    var dense_act_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
    if (config.arch == .qwen3_dense) {
        const inter_bytes = m_cap * @as(usize, config.intermediate_size) * @sizeOf(f16);
        const dg = try scratch_arena.allocAligned(inter_bytes, 64);
        const du = try scratch_arena.allocAligned(inter_bytes, 64);
        const da = try scratch_arena.allocAligned(inter_bytes, 64);
        dense_gate_host = @alignCast(std.mem.bytesAsSlice(f16, dg));
        dense_up_host = @alignCast(std.mem.bytesAsSlice(f16, du));
        dense_act_host = @alignCast(std.mem.bytesAsSlice(f16, da));
        dense_gate_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(dg.ptr) };
        dense_up_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(du.ptr) };
        dense_act_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(da.ptr) };
    }

    // --- 8. Load global weight blobs into shard 0 (zero-duplicate tied lm_head) ---
    const embed_shard_idx: u16 = @intCast(group_shard[0]);
    const g_shard = &shards[embed_shard_idx];
    const embed_blob = try loadBlobIntoShard(source, embed_entry, g_shard, options.verify_crcs);
    const final_norm_blob = try loadBlobIntoShard(source, final_norm_entry, g_shard, options.verify_crcs);
    const lm_head_blob = if (config.tie_word_embeddings)
        embed_blob
    else
        try loadBlobIntoShard(source, lm_head_entry, g_shard, options.verify_crcs);

    const embed_lin = try embed_blob.asLinearBinding(embed_wdt);
    const lm_head_lin = try lm_head_blob.asLinearBinding(lm_head_wdt);
    var has_sme2_weights: bool = (lm_head_lin.layout == .sme2_panel64);

    const embed_row_alg_bytes: u64 = switch (embed_wdt) {
        .f16, .bf16 => @as(u64, h) * 2,
        .q2_affine => (try quant.QuantOps(.{ .bits = .q2, .group_size = 64 }).rowWeightBytes(h)) +
            @as(u64, h / 64) * @sizeOf(quant.ScaleBias),
        .q3_affine => (try quant.QuantOps(.{ .bits = .q3, .group_size = 64 }).rowWeightBytes(h)) +
            @as(u64, h / 64) * @sizeOf(quant.ScaleBias),
        .q4_affine => (try quant.QuantOps(.{ .bits = .q4, .group_size = 64 }).rowWeightBytes(h)) +
            @as(u64, h / 64) * @sizeOf(quant.ScaleBias),
    };
    const global_head_bytes: u64 = embed_row_alg_bytes + final_norm_entry.data_bytes + lm_head_lin.alg_bytes;

    // --- 9. Load per-layer weights into assigned shards & build device tables ---
    const total_expert_slots: usize = if (config.arch == .qwen3_moe)
        num_layers * @as(usize, config.num_experts)
    else
        0;
    const expert_dtypes: []quant.WeightDType = if (total_expert_slots > 0)
        try allocator.alloc(quant.WeightDType, total_expert_slots)
    else
        &.{};
    errdefer if (expert_dtypes.len > 0) allocator.free(expert_dtypes);
    const expert_bytes: []u64 = if (total_expert_slots > 0)
        try allocator.alloc(u64, total_expert_slots)
    else
        &.{};
    errdefer if (expert_bytes.len > 0) allocator.free(expert_bytes);
    const expert_bytes_u32: []u32 = if (total_expert_slots > 0)
        try allocator.alloc(u32, total_expert_slots)
    else
        &.{};
    errdefer if (expert_bytes_u32.len > 0) allocator.free(expert_bytes_u32);

    const layers = try allocator.alloc(LayerWeights, num_layers);
    var layers_built: usize = 0;
    errdefer {
        for (0..layers_built) |li| {
            if (layers[li].expert_sme_blobs) |esb| allocator.free(esb);
        }
        allocator.free(layers);
    }

    var total_layers_static_bytes: u64 = 0;
    var total_layers_non_expert_bytes: u64 = 0;
    var total_expert_bytes_sum: u64 = 0;
    var single_expert_bytes: u64 = 0;
    var model_is_heterogeneous_moe: bool = false;
    const dummy_linear: LinearBinding = .{
        .weight = .{ .buffer = scratch_buf, .offset = 0 },
        .scale_bias = null,
        .n = 0,
        .k = 0,
        .dtype = .bf16,
        .alg_bytes = 0,
    };

    for (0..num_layers) |l| {
        const shard_idx: u16 = @intCast(group_shard[1 + l]);
        const l_shard = &shards[shard_idx];

        // Bindless KV chunk table for layer `l`.
        const kv_tbl_raw = try scratch_arena.allocAligned(@as(usize, active_kv_chunks) * @sizeOf(attention.KvTilePair), 64);
        const kv_tbl: []attention.KvTilePair = @alignCast(std.mem.bytesAsSlice(attention.KvTilePair, kv_tbl_raw));
        for (0..active_kv_chunks) |c_idx| {
            const tbase = (l * @as(usize, active_kv_chunks) + c_idx) * 2;
            kv_tbl[c_idx] = attention.KvTilePair.fromTiles(kv_tiles[tbase], kv_tiles[tbase + 1], .rgba16f);
        }
        const retired_ids_bind: metal.BufferBinding, const retired_ids_host: []u32 = if (use_retired_q8) blk: {
            const r_raw = try scratch_arena.allocAligned(@as(usize, max_retired_per_layer) * @sizeOf(u32), 64);
            @memset(r_raw, 0);
            break :blk .{
                .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(r_raw.ptr) },
                @alignCast(std.mem.bytesAsSlice(u32, r_raw)),
            };
        } else .{ .{ .buffer = scratch_buf, .offset = 0 }, &.{} };

        const in_ln_e = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable);
        const in_ln_dt = try validateNorm1D(in_ln_e, h);
        const in_ln_b = (try loadBlobIntoShard(source, in_ln_e, l_shard, options.verify_crcs)).asNormBinding(in_ln_dt);

        const q_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable);
        const q_b = try (try loadBlobIntoShard(source, q_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(q_e, q_dim, h));

        const k_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable);
        const k_b = try (try loadBlobIntoShard(source, k_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(k_e, kv_dim, h));

        const v_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable);
        const v_b = try (try loadBlobIntoShard(source, v_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(v_e, kv_dim, h));

        const o_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable);
        const o_b = try (try loadBlobIntoShard(source, o_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(o_e, h, q_dim));

        if (q_b.layout == .sme2_panel64 or k_b.layout == .sme2_panel64 or v_b.layout == .sme2_panel64 or o_b.layout == .sme2_panel64) {
            has_sme2_weights = true;
        }

        var q_norm_bind: ?metal.BufferBinding = null;
        var k_norm_bind: ?metal.BufferBinding = null;
        var q_norm_host: ?[]const u8 = null;
        var k_norm_host: ?[]const u8 = null;
        var qk_norm_dtype: quant.FloatDType = .f16;
        var qk_norm_bytes: u64 = 0;
        if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable)) |qn_e| {
            qk_norm_dtype = try validateNorm1D(qn_e, config.head_dim);
            const qn_blob = try loadBlobIntoShard(source, qn_e, l_shard, options.verify_crcs);
            q_norm_bind = qn_blob.asBinding();
            q_norm_host = qn_blob.slice[0..@intCast(qn_e.data_bytes)];
            qk_norm_bytes += qn_e.data_bytes;
        }
        if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable)) |kn_e| {
            const kn_blob = try loadBlobIntoShard(source, kn_e, l_shard, options.verify_crcs);
            k_norm_bind = kn_blob.asBinding();
            k_norm_host = kn_blob.slice[0..@intCast(kn_e.data_bytes)];
            qk_norm_bytes += kn_e.data_bytes;
        }

        const post_ln_e = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable);
        const post_ln_dt = try validateNorm1D(post_ln_e, h);
        const post_ln_b = (try loadBlobIntoShard(source, post_ln_e, l_shard, options.verify_crcs)).asNormBinding(post_ln_dt);

        var layer_w: LayerWeights = .{
            .shard_idx = shard_idx,
            .input_layernorm = in_ln_b,
            .q_proj = q_b,
            .k_proj = k_b,
            .v_proj = v_b,
            .o_proj = o_b,
            .q_norm = q_norm_bind,
            .k_norm = k_norm_bind,
            .q_norm_host = q_norm_host,
            .k_norm_host = k_norm_host,
            .qk_norm_dtype = qk_norm_dtype,
            .post_attention_layernorm = post_ln_b,
            .router = .{ .weights = .{ .binding = .{ .buffer = scratch_buf, .offset = 0 }, .dtype = .f16 } },
            .expert_dtype = .q3_affine,
            .expert_schemes_bind = .{ .buffer = scratch_buf, .offset = 0 },
            .expert_weights_buf = l_shard.buffer,
            .expert_table = .{ .buffer = scratch_buf, .offset = 0 },
            .expert_table_host = &.{},
            .dense_gate = dummy_linear,
            .dense_up = dummy_linear,
            .dense_down = dummy_linear,
            .kv_tiles_binding = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(kv_tbl_raw.ptr) },
            .kv_tiles_host = kv_tbl,
            .retired_ids_binding = retired_ids_bind,
            .retired_ids_host = retired_ids_host,
        };
        errdefer if (layer_w.expert_sme_blobs) |esb| allocator.free(esb);

        const attn_static_bytes: u64 =
            in_ln_e.data_bytes + post_ln_e.data_bytes + qk_norm_bytes +
            q_b.alg_bytes + k_b.alg_bytes + v_b.alg_bytes + o_b.alg_bytes;
        var ffn_active_bytes: u64 = 0;
        var layer_non_expert_bytes: u64 = attn_static_bytes;
        switch (config.arch) {
            .qwen3_moe => {
                const router_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate", .{l}) catch unreachable);
                const router_dt = try validateRouter2D(router_e, config.num_experts, h);
                const router_layout = try router_e.tensorLayout();
                const router_pk = try router_e.quantParamKind();
                const router_b = try loadBlobIntoShard(source, router_e, l_shard, options.verify_crcs);
                layer_w.router_layout = router_layout;
                layer_w.router_param_kind = router_pk;
                if (router_layout == .sme2_panel64) {
                    has_sme2_weights = true;
                    layer_w.router_panel_dtype = try amfDTypeToPanelDType(router_dt);
                    layer_w.router_sme_packed = router_b.slice[0..@intCast(router_e.data_bytes)];
                    layer_w.router = .{ .logits = .{ .binding = .{ .buffer = scratch_buf, .offset = 0 }, .dtype = .f32 } };
                } else {
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
                        .q2_affine => .{ .quant_weights = .{
                            .qweight = router_b.asBinding(),
                            .scale_bias = .{ .buffer = router_b.buffer, .offset = router_b.offset + router_e.params_offset },
                            .bits = .q2,
                        } },
                        .u32 => unreachable,
                    };
                }

                const exp_tbl_raw = try scratch_arena.allocAligned(@as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights), 64);
                const exp_tbl: []moe.ExpertWeights = @alignCast(std.mem.bytesAsSlice(moe.ExpertWeights, exp_tbl_raw));
                layer_w.expert_table = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(exp_tbl_raw.ptr) };
                layer_w.expert_table_host = exp_tbl;

                const schemes_raw = try scratch_arena.allocAligned(amf.align64(config.num_experts), 64);
                @memset(schemes_raw, 0);
                layer_w.expert_schemes_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(schemes_raw.ptr) };

                var exp_dt: quant.WeightDType = .q3_affine;
                var exp_layout: amf.TensorLayout = .row_major;
                var exp_pk: amf.QuantParamKind = .affine_scale_bias;
                var layer_exp_bytes_sum: u64 = 0;
                const n_exp_usize: usize = config.num_experts;
                for (0..n_exp_usize) |e_idx| {
                    const gp_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e_idx }) catch unreachable);
                    const gp_dt = try validateLinear2D(gp_e, config.moe_intermediate_size, h);
                    const gp_pk = try gp_e.quantParamKind();
                    if (e_idx == 0) {
                        exp_dt = gp_dt;
                        exp_layout = try gp_e.tensorLayout();
                        exp_pk = gp_pk;
                        if (exp_layout == .sme2_panel64) {
                            has_sme2_weights = true;
                            layer_w.expert_sme_blobs = try allocator.alloc(SmeExpertTriple, config.num_experts);
                        }
                    }
                    const one_exp_e: u64 = try moe.singleExpertBytesForDType(gp_dt, h, config.moe_intermediate_size);
                    const flat_idx = l * n_exp_usize + e_idx;
                    expert_dtypes[flat_idx] = gp_dt;
                    expert_bytes[flat_idx] = one_exp_e;
                    expert_bytes_u32[flat_idx] = @intCast(@min(one_exp_e, std.math.maxInt(u32)));
                    layer_exp_bytes_sum += one_exp_e;
                    total_expert_bytes_sum += one_exp_e;
                    schemes_raw[e_idx] = @backingInt(gp_dt);
                    layer_w.expert_scheme_counts[@backingInt(gp_dt)] += 1;

                    const up_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e_idx }) catch unreachable);
                    const dp_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e_idx }) catch unreachable);

                    if (exp_layout == .sme2_panel64) {
                        const gp_b = try loadBlobIntoShard(source, gp_e, l_shard, options.verify_crcs);
                        const up_b = try loadBlobIntoShard(source, up_e, l_shard, options.verify_crcs);
                        const dp_b = try loadBlobIntoShard(source, dp_e, l_shard, options.verify_crcs);
                        const is_cold = ((try gp_e.tensorKind()) == .cold_expert or
                            (try up_e.tensorKind()) == .cold_expert or
                            (try dp_e.tensorKind()) == .cold_expert);
                        layer_w.expert_sme_blobs.?[e_idx] = .{
                            .gate = gp_b.slice[0..@intCast(gp_e.data_bytes)],
                            .up = up_b.slice[0..@intCast(up_e.data_bytes)],
                            .down = dp_b.slice[0..@intCast(dp_e.data_bytes)],
                            .dtype = gp_dt,
                            .param_kind = gp_pk,
                            .is_cold = is_cold,
                        };
                        exp_tbl[e_idx] = std.mem.zeroes(moe.ExpertWeights);
                        continue;
                    }

                    if (cold_rt) |crt| {
                        if (crt.cold_masks_by_layer[l].contains(@intCast(e_idx))) {
                            exp_tbl[e_idx] = std.mem.zeroes(moe.ExpertWeights);
                            const cold_triple = try cache_mod.ColdExpertTriple.fromAmfEntries(crt.file.fd, 0, gp_e, up_e, dp_e);
                            try crt.expert_cache.registerCold(@intCast(l), @intCast(e_idx), cold_triple);
                            continue;
                        }
                    }

                    const gp_b = try loadBlobIntoShard(source, gp_e, l_shard, options.verify_crcs);
                    const up_b = try loadBlobIntoShard(source, up_e, l_shard, options.verify_crcs);
                    const dp_b = try loadBlobIntoShard(source, dp_e, l_shard, options.verify_crcs);

                    const is_float = (gp_dt == .f16 or gp_dt == .bf16);
                    const ew: moe.ExpertWeights = .{
                        .gate_qweight_offset = gp_b.offset,
                        .gate_sb_offset = if (is_float) 0 else gp_b.offset + gp_e.params_offset,
                        .up_qweight_offset = up_b.offset,
                        .up_sb_offset = if (is_float) 0 else up_b.offset + up_e.params_offset,
                        .down_qweight_offset = dp_b.offset,
                        .down_sb_offset = if (is_float) 0 else dp_b.offset + dp_e.params_offset,
                    };
                    exp_tbl[e_idx] = ew;

                    if (cold_rt) |crt| {
                        try crt.expert_cache.registerHot(@intCast(l), @intCast(e_idx), .{
                            .gate = try expert_mod.WiredExpert.init(gp_b.slice),
                            .up = try expert_mod.WiredExpert.init(up_b.slice),
                            .down = try expert_mod.WiredExpert.init(dp_b.slice),
                            .weights = ew,
                            .metal_buffer = l_shard.buffer,
                        });
                    }
                }
                var active_schemes: u32 = 0;
                for (layer_w.expert_scheme_counts) |cnt| {
                    if (cnt > 0) active_schemes += 1;
                }
                layer_w.is_heterogeneous_moe = (active_schemes > 1);
                if (layer_w.is_heterogeneous_moe) model_is_heterogeneous_moe = true;
                layer_w.expert_dtype = exp_dt;
                layer_w.expert_layout = exp_layout;
                layer_w.expert_param_kind = exp_pk;

                const router_bytes: u64 = router_e.data_bytes + router_e.params_bytes;
                const mean_exp_l: u64 = layer_exp_bytes_sum / config.num_experts;
                const experts_bytes: u64 = mean_exp_l * config.num_experts_per_tok;
                ffn_active_bytes = router_bytes + experts_bytes;
                layer_non_expert_bytes += router_bytes;
            },
            .qwen3_dense => {
                const gp_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable);
                const gp_b = try (try loadBlobIntoShard(source, gp_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(gp_e, config.intermediate_size, h));

                const up_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}) catch unreachable);
                const up_b = try (try loadBlobIntoShard(source, up_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(up_e, config.intermediate_size, h));

                const dp_e = try findLinear(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}) catch unreachable);
                const dp_b = try (try loadBlobIntoShard(source, dp_e, l_shard, options.verify_crcs)).asLinearBinding(try validateLinear2D(dp_e, h, config.intermediate_size));

                if (gp_b.layout == .sme2_panel64 or up_b.layout == .sme2_panel64 or dp_b.layout == .sme2_panel64) {
                    has_sme2_weights = true;
                }

                layer_w.dense_gate = gp_b;
                layer_w.dense_up = up_b;
                layer_w.dense_down = dp_b;
                ffn_active_bytes = gp_b.alg_bytes + up_b.alg_bytes + dp_b.alg_bytes;
                layer_non_expert_bytes += ffn_active_bytes;
            },
        }

        total_layers_static_bytes += attn_static_bytes + ffn_active_bytes;
        total_layers_non_expert_bytes += layer_non_expert_bytes;
        layers[l] = layer_w;
        layers_built += 1;
    }

    if (total_expert_slots > 0) {
        single_expert_bytes = total_expert_bytes_sum / total_expert_slots;
    }

    const out_tok_ids_slice: []u32 = @alignCast(std.mem.bytesAsSlice(u32, out_tok_raw));
    const logits_mut_slice: []f16 = @alignCast(std.mem.bytesAsSlice(f16, logits_raw));

    return .{
        .config = config,
        .max_seq_len = max_seq_len,
        .chunk_tokens = options.chunk_tokens,
        .num_chunks = num_chunks,
        .max_tree_nodes = max_tree_nodes,
        .enable_feature_taps = options.enable_feature_taps,
        .aux_layer_ids = aux_layer_ids,
        .timeout_ms = options.timeout_ms,
        .wired_budget_bytes = wired_budget.limit_bytes,
        .total_wired_bytes = total_wired_bytes,
        .seq_len = 0,
        .prefill_mode = options.prefill_mode,
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
        .embed_shard_idx = embed_shard_idx,
        .embed = embed_lin,
        .final_norm = final_norm_blob.asNormBinding(final_norm_dt),
        .lm_head = lm_head_lin,
        .x_host = @alignCast(std.mem.bytesAsSlice(f16, x_raw)),
        .x_norm_host = @alignCast(std.mem.bytesAsSlice(f16, x_norm_raw)),
        .q_host = @alignCast(std.mem.bytesAsSlice(f16, q_raw)),
        .k_host = @alignCast(std.mem.bytesAsSlice(f16, k_raw)),
        .v_host = @alignCast(std.mem.bytesAsSlice(f16, v_raw)),
        .attn_out_host = @alignCast(std.mem.bytesAsSlice(f16, attn_out_raw)),
        .proj_out_host = @alignCast(std.mem.bytesAsSlice(f16, proj_out_raw)),
        .logits_mut_host = logits_mut_slice,
        .logits_host = logits_mut_slice,
        .out_token_ids_host = out_tok_ids_slice,
        .out_token_id_host = &out_tok_ids_slice[0],
        .last_logits_row = 0,
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
        .tree_token_ids_host = tree_token_ids_host,
        .tree_positions_host = tree_positions_host,
        .tree_ancestor_masks_host = tree_ancestor_masks_host,
        .tree_accepted_nodes_host = tree_accepted_nodes_host,
        .topk_host = topk_host,
        .tree_token_ids_bind = tree_token_ids_bind,
        .tree_positions_bind = tree_positions_bind,
        .tree_ancestor_masks_bind = tree_ancestor_masks_bind,
        .tree_accepted_nodes_bind = tree_accepted_nodes_bind,
        .topk_bind = topk_bind,
        .tree_k_base_off = tree_k_base_off,
        .tree_v_base_off = tree_v_base_off,
        .tree_kv_layer_stride = tree_kv_layer_stride,
        .gqa_tree_scratch_bind = gqa_tree_scratch_bind,
        .aux_concat_host = aux_concat_host,
        .aux_concat_bind = aux_concat_bind,
        .accepted_aux_host = accepted_aux_host,
        .accepted_aux_bind = accepted_aux_bind,
        .accepted_top_host = accepted_top_host,
        .accepted_top_bind = accepted_top_bind,
        .moe_ws = moe_ws,
        .layer_sched_hdr_base_off = layer_sched_hdr_base_off,
        .layer_sched_hdr_raw = layer_sched_hdr_raw,
        .pred_topk_base_off = pred_topk_base_off,
        .pred_topk_layer_stride = pred_topk_layer_stride,
        .pred_topk_host = pred_topk_host,
        .dense_gate_host = dense_gate_host,
        .dense_up_host = dense_up_host,
        .dense_act_host = dense_act_host,
        .dense_gate_bind = dense_gate_bind,
        .dense_up_bind = dense_up_bind,
        .dense_act_bind = dense_act_bind,
        .total_layers_static_bytes = total_layers_static_bytes,
        .total_layers_non_expert_bytes = total_layers_non_expert_bytes,
        .single_expert_bytes = single_expert_bytes,
        .embed_row_alg_bytes = embed_row_alg_bytes,
        .global_head_bytes = global_head_bytes,
        .is_heterogeneous_moe = model_is_heterogeneous_moe,
        .expert_dtypes = expert_dtypes,
        .expert_bytes = expert_bytes,
        .expert_bytes_u32 = expert_bytes_u32,
        .cold_rt = cold_rt,
        .hybrid_rt = null,
        .has_sme2_weights = has_sme2_weights,
        .has_pure_cpu_sme2_dense = false,
        .sme_router_mode = options.sme_router_mode,
        .sme_rt = null,
        .kv_retire_after_chunks = retire_after,
        .active_kv_chunks = active_kv_chunks,
        .num_retired_chunks = 0,
        .max_retired_per_layer = max_retired_per_layer,
        .kv_retire_events = 0,
        .active_kv_wired_bytes = total_kv_bytes,
        .retired_kv_wired_bytes = retired_kv_wired_bytes,
        .retired_kv_pool = retired_kv_pool,
        .active_tile_phys_ring = active_tile_phys_ring,
    };
}

// ---------------------------------------------------------------------------
// M6.5 C14 + R8 + R5: Multi-Slot Continuous Batching & PrefixCache Runtime
// ---------------------------------------------------------------------------

pub const BatchRuntimeConfig = struct {
    /// Maximum active concurrent request slots (`1..=batch_sched.MAX_BATCH_SLOTS`).
    max_slots: u32 = 8,
    /// Maximum cached `chunk_tokens` (`512`-token) full/tail blocks in `PrefixCache`
    /// (`0` disables `R8` shared prefix caching).
    prefix_cache_chunks: u32 = 0,
    /// Optional `R5` per-slot group-64 `q8` retirement threshold (`0` disables retirement).
    kv_retire_q8_after_chunks: u32 = 0,
    /// Optional explicit total physical `IOSurface` chunk groups across all slots + prefix cache
    /// (`0` auto-sizes within remaining wired budget).
    kv_pool_chunk_groups: u32 = 0,
    /// Optional explicit wired budget override (defaults to `Budget.fromHost`).
    budget_override: ?budget_mod.Budget = null,
};

pub const BatchRuntimeStats = struct {
    max_slots: u32,
    pool_chunk_groups: u32,
    pool_free_groups: u32,
    prefix_cache_capacity: u32,
    prefix_cache_live_entries: u32,
    prefix_cache_pinned_entries: u32,
    prefix_hits: u64,
    prefix_hit_tokens: u64,
    prefix_cow_copies: u64,
    kv_retire_events: u64,
    decode_batch_steps: u64,
    decode_batch_tokens: u64,
    total_wired_bytes: u64,
};

pub const BatchRuntimeError = error{
    InvalidConfig,
    TreeNotEnabled,
    BudgetExceeded,
    OutOfMemory,
} || sysinfo.Error || budget_mod.PlanError || arena_mod.WiredArena.InitError ||
    arena_mod.WiredArena.AllocError || metal.WrapError || metal.Error ||
    attention.RetiredKvQ8PoolError || prefix_cache_mod.Error;

const NONE_GROUP: u32 = std.math.maxInt(u32);

const BatchSlotState = struct {
    in_use: bool = false,
    seq_len: u32 = 0,
    matched_prefix_tokens: u32 = 0,
    prefix_cached_full_chunks: u32 = 0,
    last_cached_parent_id: u32 = prefix_cache_mod.NONE_ID,
    num_retired_chunks: u32 = 0,
    pinned_full_count: u32 = 0,
    /// `[max_chunks]` physical chunk group ID for chunk `c` (`NONE_GROUP` if unallocated/retired).
    chunk_group_ids: []u32 = &.{},
    /// `[max_chunks]` true iff chunk `c` is owned read-only by `PrefixCache` (pinned).
    chunk_is_shared: []bool = &.{},
    /// `[max_chunks]` `PrefixCache` entry IDs pinned by this slot.
    pinned_full_ids: []u32 = &.{},
    /// `[max_retired_per_slot]` true iff retired slot `r` is owned by `PrefixCache`.
    retired_is_shared: []bool = &.{},
    /// `[max_seq_len]` wired prompt token history for exact `PrefixCache` insertion (`0` alloc).
    prompt_tokens: []u32 = &.{},
};

/// Preallocated zero-allocation continuous-batching (`C14`), shared `PrefixCache` (`R8`),
/// and optional retired `q8` KV (`R5`) runtime attached to a loaded `Qwen3Model`.
///
/// Advances `B` (`1 <= B <= max_slots`) concurrent decode requests in ONE batched
/// `M = B` GPU/SME2 forward pass per iteration (`encodeBatchQkNormRopeKvWrite`,
/// `encodeBatchGqaDecodeAttn`, CSR-deduplicated MoE / batched Dense FFN, batched
/// `lm_head`, and isolated per-slot `D11` `RequestRng` sampling) with `0` dynamic
/// heap allocations on the hot path.
pub const Qwen3BatchRuntime = struct {
    model: *Qwen3Model,
    max_slots: u32,
    max_chunks: u32,
    active_per_slot: u32,
    max_retired_per_slot: u32,
    kv_retire_after_chunks: u32,
    pool_chunk_groups: u32,
    total_wired_bytes: u64,

    batch_arena: arena_mod.WiredArena,
    batch_buf: metal.Buffer,

    /// Flat `[pool_chunk_groups * num_layers * 2]` IOSurface `RGBA16Float` KV tiles.
    pool_tiles: []metal.KvTile,
    pool_free_stack: []u32,
    pool_free_top: u32,

    /// Per-layer `[max_slots * active_per_slot]` active `KvTilePair` table base offset.
    layer_kv_tiles_base_off: u64,
    layer_kv_tiles_stride_bytes: u64,
    layer_kv_tiles_host: []attention.KvTilePair,

    /// Per-layer `[max_slots * max_retired_per_slot]` retired `q8` ID table base offset.
    layer_retired_ids_base_off: u64,
    layer_retired_ids_stride_bytes: u64,
    layer_retired_ids_host: []u32,

    /// Per-layer `[max_slots]` `BatchKvWriteSlot` upload descriptors.
    layer_write_slots_base_off: u64,
    layer_write_slots_stride_bytes: u64,
    layer_write_slots_host: []attention.BatchKvWriteSlot,

    /// Per-layer `[max_slots]` `BatchGqaSlotDesc` upload descriptors.
    layer_gqa_descs_base_off: u64,
    layer_gqa_descs_stride_bytes: u64,
    layer_gqa_descs_host: []attention.BatchGqaSlotDesc,

    gqa_batch_scratch_bind: metal.BufferBinding,
    retired_pool: ?attention.RetiredKvQ8Pool,
    prefix_cache: ?prefix_cache_mod.PrefixCache,

    slots: []BatchSlotState,
    scratch_handles: []prefix_cache_mod.KvBlockHandle,
    scratch_evicted_handles: []prefix_cache_mod.KvBlockHandle,
    p_res: []f32,

    // Telemetry
    prefix_hits: u64 = 0,
    prefix_hit_tokens: u64 = 0,
    prefix_cow_copies: u64 = 0,
    kv_retire_events: u64 = 0,
    decode_batch_steps: u64 = 0,
    decode_batch_tokens: u64 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        model: *Qwen3Model,
        cfg: BatchRuntimeConfig,
    ) BatchRuntimeError!Qwen3BatchRuntime {
        if (cfg.max_slots == 0 or cfg.max_slots > batch_sched.MAX_BATCH_SLOTS) {
            return error.InvalidConfig;
        }
        const needed_tree: u32 = @max(cfg.max_slots, 2);
        if (model.max_tree_nodes < needed_tree) {
            return error.TreeNotEnabled;
        }

        const num_layers: usize = model.layers.len;
        const max_chunks: u32 = model.num_chunks;
        const retire_after: u32 = cfg.kv_retire_q8_after_chunks;
        const use_retired_q8 = (retire_after > 0 and max_chunks > retire_after);
        if (use_retired_q8 and (model.config.head_dim % attention.KV_Q8_GROUP_SIZE) != 0) {
            return error.InvalidConfig;
        }

        const active_per_slot: u32 = if (use_retired_q8)
            @min(max_chunks, @max(retire_after, 2))
        else
            max_chunks;
        const max_retired_per_slot: u32 = if (use_retired_q8) max_chunks - retire_after else 0;

        const facts = try sysinfo.Facts.read();
        const dev_info = try model.device.info();
        const wired_budget = cfg.budget_override orelse budget_mod.Budget.fromHost(facts, dev_info, .{});

        // Compute `batch_arena` size first so remaining wired budget determines safe KV pool capacity.
        const layer_kv_tiles_stride_bytes: usize = amf.align64(
            @as(usize, cfg.max_slots) * @as(usize, active_per_slot) * @sizeOf(attention.KvTilePair),
        );
        const layer_retired_ids_stride_bytes: usize = if (use_retired_q8)
            amf.align64(@as(usize, cfg.max_slots) * @as(usize, max_retired_per_slot) * @sizeOf(u32))
        else
            0;
        const layer_write_slots_stride_bytes: usize = amf.align64(
            @as(usize, cfg.max_slots) * @sizeOf(attention.BatchKvWriteSlot),
        );
        const layer_gqa_descs_stride_bytes: usize = amf.align64(
            @as(usize, cfg.max_slots) * @sizeOf(attention.BatchGqaSlotDesc),
        );
        const gqa_batch_layout = attention.gqaBatchScratchLayout(
            cfg.max_slots,
            model.config.num_attention_heads,
            model.config.head_dim,
            attention.MAX_SPLITS,
        );
        const retired_pool_chunks: u32 = if (use_retired_q8)
            @as(u32, @intCast(num_layers)) * (cfg.max_slots * max_retired_per_slot + cfg.prefix_cache_chunks)
        else
            1;
        const retired_pool_spec: attention.RetiredKvQ8PoolSpec = .{
            .num_kv_heads = model.config.num_key_value_heads,
            .head_dim = model.config.head_dim,
            .chunk_tokens = model.chunk_tokens,
            .max_chunks = retired_pool_chunks,
        };
        const retired_pool_bytes: usize = if (use_retired_q8)
            try attention.RetiredKvQ8Pool.requiredArenaBytes(retired_pool_spec)
        else
            0;
        const pc_spec: prefix_cache_mod.PrefixCacheSpec = .{
            .max_entries = if (cfg.prefix_cache_chunks > 0) cfg.prefix_cache_chunks else 1,
            .block_tokens = model.chunk_tokens,
            .num_kv_layers = @intCast(num_layers),
        };
        const pc_bytes: usize = if (cfg.prefix_cache_chunks > 0)
            try prefix_cache_mod.PrefixCache.requiredBytes(pc_spec)
        else
            0;
        const prompt_hist_bytes: usize = amf.align64(
            @as(usize, cfg.max_slots) * @as(usize, model.max_seq_len) * @sizeOf(u32),
        );

        const arena_raw: usize =
            retired_pool_bytes +
            pc_bytes +
            num_layers * layer_kv_tiles_stride_bytes +
            num_layers * layer_retired_ids_stride_bytes +
            num_layers * layer_write_slots_stride_bytes +
            num_layers * layer_gqa_descs_stride_bytes +
            amf.align64(gqa_batch_layout.total_bytes) +
            prompt_hist_bytes +
            8 * page.APPLE_PAGE_SIZE;
        const batch_arena_bytes = try page.alignUp(arena_raw);

        // Size shared IOSurface chunk group pool within remaining wired budget.
        const kv_tile_rows: u32 = model.config.num_key_value_heads * model.chunk_tokens;
        const kv_tile_alloc_bytes: u64 = try page.alignUp(
            @as(usize, kv_tile_rows) * model.config.head_dim * @sizeOf(f16),
        );
        const bytes_per_group: u64 = @as(u64, num_layers) * 2 * kv_tile_alloc_bytes;
        const desired_groups: u32 = if (cfg.kv_pool_chunk_groups > 0)
            cfg.kv_pool_chunk_groups
        else
            cfg.max_slots * active_per_slot + cfg.prefix_cache_chunks;

        const base_used: u64 = model.totalWiredBytes() + batch_arena_bytes;
        if (base_used + @as(u64, cfg.max_slots) * bytes_per_group > wired_budget.limit_bytes) {
            return error.BudgetExceeded;
        }
        const headroom_groups: u32 = @intCast(@min(
            @as(u64, desired_groups),
            (wired_budget.limit_bytes - base_used) / bytes_per_group,
        ));
        const pool_chunk_groups: u32 = if (cfg.kv_pool_chunk_groups > 0) blk: {
            if (cfg.kv_pool_chunk_groups > headroom_groups) return error.BudgetExceeded;
            break :blk cfg.kv_pool_chunk_groups;
        } else @max(cfg.max_slots, headroom_groups);

        const pool_kv_bytes: u64 = @as(u64, pool_chunk_groups) * bytes_per_group;
        const total_wired_bytes: u64 = batch_arena_bytes + pool_kv_bytes;
        try wired_budget.checkPlan(model.totalWiredBytes() + total_wired_bytes);

        // Allocate `batch_arena` + wrap `MTLBuffer` + register resident.
        var batch_arena = try arena_mod.WiredArena.init(batch_arena_bytes, wired_budget);
        errdefer batch_arena.deinit();
        const batch_buf = try metal.Buffer.wrap(model.device, batch_arena.bytes());
        errdefer batch_buf.close();

        const retired_pool: ?attention.RetiredKvQ8Pool = if (use_retired_q8)
            try attention.RetiredKvQ8Pool.init(&batch_arena, retired_pool_spec)
        else
            null;
        const prefix_cache: ?prefix_cache_mod.PrefixCache = if (cfg.prefix_cache_chunks > 0)
            try prefix_cache_mod.PrefixCache.init(&batch_arena, pc_spec)
        else
            null;

        const kv_tiles_raw = try batch_arena.allocAligned(num_layers * layer_kv_tiles_stride_bytes, 64);
        @memset(kv_tiles_raw, 0);
        const layer_kv_tiles_base_off: u64 = @intCast(batch_arena.offsetOf(kv_tiles_raw.ptr));
        const layer_kv_tiles_host: []attention.KvTilePair = @alignCast(std.mem.bytesAsSlice(attention.KvTilePair, kv_tiles_raw));

        var layer_retired_ids_base_off: u64 = 0;
        var layer_retired_ids_host: []u32 = &.{};
        if (use_retired_q8) {
            const ret_raw = try batch_arena.allocAligned(num_layers * layer_retired_ids_stride_bytes, 64);
            @memset(ret_raw, 0);
            layer_retired_ids_base_off = @intCast(batch_arena.offsetOf(ret_raw.ptr));
            layer_retired_ids_host = @alignCast(std.mem.bytesAsSlice(u32, ret_raw));
        }

        const ws_raw = try batch_arena.allocAligned(num_layers * layer_write_slots_stride_bytes, 64);
        @memset(ws_raw, 0);
        const layer_write_slots_base_off: u64 = @intCast(batch_arena.offsetOf(ws_raw.ptr));
        const layer_write_slots_host: []attention.BatchKvWriteSlot = @alignCast(
            std.mem.bytesAsSlice(attention.BatchKvWriteSlot, ws_raw),
        );

        const gd_raw = try batch_arena.allocAligned(num_layers * layer_gqa_descs_stride_bytes, 64);
        @memset(gd_raw, 0);
        const layer_gqa_descs_base_off: u64 = @intCast(batch_arena.offsetOf(gd_raw.ptr));
        const layer_gqa_descs_host: []attention.BatchGqaSlotDesc = @alignCast(
            std.mem.bytesAsSlice(attention.BatchGqaSlotDesc, gd_raw),
        );

        const gqa_sc_raw = try batch_arena.allocAligned(gqa_batch_layout.total_bytes, 64);
        const gqa_batch_scratch_bind: metal.BufferBinding = .{
            .buffer = batch_buf,
            .offset = @intCast(batch_arena.offsetOf(gqa_sc_raw.ptr)),
        };

        const prompt_hist_raw = try batch_arena.allocAligned(prompt_hist_bytes, 64);
        const prompt_hist_all: []u32 = @alignCast(std.mem.bytesAsSlice(u32, prompt_hist_raw));

        // Allocate shared IOSurface KV tile pool.
        const total_pool_tiles: usize = @as(usize, pool_chunk_groups) * num_layers * 2;
        const pool_tiles = try allocator.alloc(metal.KvTile, total_pool_tiles);
        var tiles_inited: usize = 0;
        errdefer {
            for (0..tiles_inited) |i| pool_tiles[i].close();
            allocator.free(pool_tiles);
        }
        for (0..total_pool_tiles) |i| {
            pool_tiles[i] = try metal.KvTile.init(model.device, model.config.head_dim, kv_tile_rows, .{});
            tiles_inited += 1;
        }

        try model.compute.makeResident(&.{batch_buf});
        try model.compute.makeTilesResident(pool_tiles);

        const pool_free_stack = try allocator.alloc(u32, pool_chunk_groups);
        errdefer allocator.free(pool_free_stack);
        for (0..pool_chunk_groups) |i| {
            pool_free_stack[i] = pool_chunk_groups - 1 - @as(u32, @intCast(i));
        }

        const slots = try allocator.alloc(BatchSlotState, cfg.max_slots);
        var slots_inited: usize = 0;
        errdefer {
            for (0..slots_inited) |si| {
                allocator.free(slots[si].chunk_group_ids);
                allocator.free(slots[si].chunk_is_shared);
                allocator.free(slots[si].pinned_full_ids);
                if (slots[si].retired_is_shared.len > 0) allocator.free(slots[si].retired_is_shared);
            }
            allocator.free(slots);
        }
        for (0..cfg.max_slots) |si| {
            const gids = try allocator.alloc(u32, max_chunks);
            errdefer allocator.free(gids);
            @memset(gids, NONE_GROUP);

            const shared = try allocator.alloc(bool, max_chunks);
            errdefer allocator.free(shared);
            @memset(shared, false);

            const pids = try allocator.alloc(u32, max_chunks);
            errdefer allocator.free(pids);
            @memset(pids, prefix_cache_mod.NONE_ID);

            const rshared: []bool = if (max_retired_per_slot > 0) blk: {
                const rs = try allocator.alloc(bool, max_retired_per_slot);
                @memset(rs, false);
                break :blk rs;
            } else &.{};

            const p_off = si * @as(usize, model.max_seq_len);
            slots[si] = .{
                .chunk_group_ids = gids,
                .chunk_is_shared = shared,
                .pinned_full_ids = pids,
                .retired_is_shared = rshared,
                .prompt_tokens = prompt_hist_all[p_off .. p_off + model.max_seq_len],
            };
            slots_inited += 1;
        }

        const scratch_handles = try allocator.alloc(prefix_cache_mod.KvBlockHandle, num_layers);
        errdefer allocator.free(scratch_handles);
        const scratch_evicted_handles = try allocator.alloc(prefix_cache_mod.KvBlockHandle, num_layers);
        errdefer allocator.free(scratch_evicted_handles);
        const p_res = try allocator.alloc(f32, model.config.vocab_size);
        errdefer allocator.free(p_res);

        return .{
            .model = model,
            .max_slots = cfg.max_slots,
            .max_chunks = max_chunks,
            .active_per_slot = active_per_slot,
            .max_retired_per_slot = max_retired_per_slot,
            .kv_retire_after_chunks = retire_after,
            .pool_chunk_groups = pool_chunk_groups,
            .total_wired_bytes = total_wired_bytes,
            .batch_arena = batch_arena,
            .batch_buf = batch_buf,
            .pool_tiles = pool_tiles,
            .pool_free_stack = pool_free_stack,
            .pool_free_top = pool_chunk_groups,
            .layer_kv_tiles_base_off = layer_kv_tiles_base_off,
            .layer_kv_tiles_stride_bytes = layer_kv_tiles_stride_bytes,
            .layer_kv_tiles_host = layer_kv_tiles_host,
            .layer_retired_ids_base_off = layer_retired_ids_base_off,
            .layer_retired_ids_stride_bytes = layer_retired_ids_stride_bytes,
            .layer_retired_ids_host = layer_retired_ids_host,
            .layer_write_slots_base_off = layer_write_slots_base_off,
            .layer_write_slots_stride_bytes = layer_write_slots_stride_bytes,
            .layer_write_slots_host = layer_write_slots_host,
            .layer_gqa_descs_base_off = layer_gqa_descs_base_off,
            .layer_gqa_descs_stride_bytes = layer_gqa_descs_stride_bytes,
            .layer_gqa_descs_host = layer_gqa_descs_host,
            .gqa_batch_scratch_bind = gqa_batch_scratch_bind,
            .retired_pool = retired_pool,
            .prefix_cache = prefix_cache,
            .slots = slots,
            .scratch_handles = scratch_handles,
            .scratch_evicted_handles = scratch_evicted_handles,
            .p_res = p_res,
        };
    }

    pub fn deinit(self: *Qwen3BatchRuntime, allocator: std.mem.Allocator) void {
        allocator.free(self.p_res);
        allocator.free(self.scratch_evicted_handles);
        allocator.free(self.scratch_handles);
        for (self.slots) |slot| {
            allocator.free(slot.chunk_group_ids);
            allocator.free(slot.chunk_is_shared);
            allocator.free(slot.pinned_full_ids);
            if (slot.retired_is_shared.len > 0) allocator.free(slot.retired_is_shared);
        }
        allocator.free(self.slots);
        allocator.free(self.pool_free_stack);
        for (self.pool_tiles) |tile| tile.close();
        allocator.free(self.pool_tiles);
        self.batch_buf.close();
        self.batch_arena.deinit();
    }

    pub fn totalWiredBytes(self: *const Qwen3BatchRuntime) u64 {
        return self.total_wired_bytes;
    }

    pub fn stats(self: *const Qwen3BatchRuntime) BatchRuntimeStats {
        const pc_cap: u32 = if (self.prefix_cache) |pc| pc.spec.max_entries else 0;
        const pc_live: u32 = if (self.prefix_cache) |pc| pc.liveEntries() else 0;
        const pc_pinned: u32 = if (self.prefix_cache) |pc| pc.pinnedEntries() else 0;
        return .{
            .max_slots = self.max_slots,
            .pool_chunk_groups = self.pool_chunk_groups,
            .pool_free_groups = self.pool_free_top,
            .prefix_cache_capacity = pc_cap,
            .prefix_cache_live_entries = pc_live,
            .prefix_cache_pinned_entries = pc_pinned,
            .prefix_hits = self.prefix_hits,
            .prefix_hit_tokens = self.prefix_hit_tokens,
            .prefix_cow_copies = self.prefix_cow_copies,
            .kv_retire_events = self.kv_retire_events,
            .decode_batch_steps = self.decode_batch_steps,
            .decode_batch_tokens = self.decode_batch_tokens,
            .total_wired_bytes = self.total_wired_bytes,
        };
    }

    pub fn batchBackend(self: *Qwen3BatchRuntime) batch_sched.BatchBackend {
        return .{
            .ctx = self,
            .lookupPrefixFn = if (self.prefix_cache != null) lookupPrefixCb else null,
            .prefillSlotFn = prefillSlotCb,
            .stepDecodeBatchFn = stepDecodeBatchCb,
            .releaseSlotFn = releaseSlotCb,
        };
    }

    fn poolTilePair(self: *const Qwen3BatchRuntime, group_id: u32, layer_idx: usize) struct { k: metal.KvTile, v: metal.KvTile } {
        const num_layers = self.model.layers.len;
        const base = (@as(usize, group_id) * num_layers + layer_idx) * 2;
        return .{ .k = self.pool_tiles[base], .v = self.pool_tiles[base + 1] };
    }

    fn layerKvTilesSlice(self: *const Qwen3BatchRuntime, layer_idx: usize) []attention.KvTilePair {
        const stride_elems = self.layer_kv_tiles_stride_bytes / @sizeOf(attention.KvTilePair);
        const start = layer_idx * stride_elems;
        return self.layer_kv_tiles_host[start .. start + @as(usize, self.max_slots) * self.active_per_slot];
    }

    fn slotKvTilesSlice(self: *const Qwen3BatchRuntime, layer_idx: usize, slot_id: u32) []attention.KvTilePair {
        const layer_slice = self.layerKvTilesSlice(layer_idx);
        const off = @as(usize, slot_id) * self.active_per_slot;
        return layer_slice[off .. off + self.active_per_slot];
    }

    fn slotKvTilesBinding(self: *const Qwen3BatchRuntime, layer_idx: usize, slot_id: u32) metal.BufferBinding {
        const off = self.layer_kv_tiles_base_off +
            @as(u64, layer_idx) * self.layer_kv_tiles_stride_bytes +
            @as(u64, slot_id) * @as(u64, self.active_per_slot) * @sizeOf(attention.KvTilePair);
        return .{ .buffer = self.batch_buf, .offset = off };
    }

    fn layerKvTilesBinding(self: *const Qwen3BatchRuntime, layer_idx: usize) metal.BufferBinding {
        return .{
            .buffer = self.batch_buf,
            .offset = self.layer_kv_tiles_base_off + @as(u64, layer_idx) * self.layer_kv_tiles_stride_bytes,
        };
    }

    fn slotRetiredIdsSlice(self: *const Qwen3BatchRuntime, layer_idx: usize, slot_id: u32) []u32 {
        if (self.max_retired_per_slot == 0) return &.{};
        const stride_elems = self.layer_retired_ids_stride_bytes / @sizeOf(u32);
        const start = layer_idx * stride_elems + @as(usize, slot_id) * self.max_retired_per_slot;
        return self.layer_retired_ids_host[start .. start + self.max_retired_per_slot];
    }

    fn slotRetiredIdsBinding(self: *const Qwen3BatchRuntime, layer_idx: usize, slot_id: u32) metal.BufferBinding {
        const off = self.layer_retired_ids_base_off +
            @as(u64, layer_idx) * self.layer_retired_ids_stride_bytes +
            @as(u64, slot_id) * @as(u64, self.max_retired_per_slot) * @sizeOf(u32);
        return .{ .buffer = self.batch_buf, .offset = off };
    }

    fn layerRetiredIdsBinding(self: *const Qwen3BatchRuntime, layer_idx: usize) metal.BufferBinding {
        return .{
            .buffer = self.batch_buf,
            .offset = self.layer_retired_ids_base_off + @as(u64, layer_idx) * self.layer_retired_ids_stride_bytes,
        };
    }

    fn layerWriteSlotsSlice(self: *const Qwen3BatchRuntime, layer_idx: usize) []attention.BatchKvWriteSlot {
        const stride_elems = self.layer_write_slots_stride_bytes / @sizeOf(attention.BatchKvWriteSlot);
        const start = layer_idx * stride_elems;
        return self.layer_write_slots_host[start .. start + self.max_slots];
    }

    fn layerWriteSlotsBinding(self: *const Qwen3BatchRuntime, layer_idx: usize) metal.BufferBinding {
        return .{
            .buffer = self.batch_buf,
            .offset = self.layer_write_slots_base_off + @as(u64, layer_idx) * self.layer_write_slots_stride_bytes,
        };
    }

    fn layerGqaDescsSlice(self: *const Qwen3BatchRuntime, layer_idx: usize) []attention.BatchGqaSlotDesc {
        const stride_elems = self.layer_gqa_descs_stride_bytes / @sizeOf(attention.BatchGqaSlotDesc);
        const start = layer_idx * stride_elems;
        return self.layer_gqa_descs_host[start .. start + self.max_slots];
    }

    fn layerGqaDescsBinding(self: *const Qwen3BatchRuntime, layer_idx: usize) metal.BufferBinding {
        return .{
            .buffer = self.batch_buf,
            .offset = self.layer_gqa_descs_base_off + @as(u64, layer_idx) * self.layer_gqa_descs_stride_bytes,
        };
    }

    fn releaseChunkGroup(self: *Qwen3BatchRuntime, group_id: u32) void {
        if (group_id == NONE_GROUP) return;
        std.debug.assert(group_id < self.pool_chunk_groups);
        std.debug.assert(self.pool_free_top < self.pool_chunk_groups);
        self.pool_free_stack[self.pool_free_top] = group_id;
        self.pool_free_top += 1;
    }

    fn reclaimEvictedHandles(self: *Qwen3BatchRuntime, handles: []const prefix_cache_mod.KvBlockHandle) void {
        if (handles.len == 0) return;
        if (handles[0].storage == .active_tile) {
            self.releaseChunkGroup(handles[0].pool_slot_id);
        } else if (self.retired_pool) |*rpool| {
            for (handles) |h| {
                if (h.storage == .retired_q8) {
                    rpool.releaseSlot(h.pool_slot_id) catch unreachable;
                }
            }
        }
    }

    fn acquireChunkGroup(self: *Qwen3BatchRuntime) ?u32 {
        if (self.pool_free_top == 0) {
            if (self.prefix_cache) |*pc| {
                while (self.pool_free_top == 0) {
                    _ = pc.evictLru(self.scratch_evicted_handles) catch break;
                    self.reclaimEvictedHandles(self.scratch_evicted_handles);
                }
            }
        }
        if (self.pool_free_top == 0) return null;
        self.pool_free_top -= 1;
        return self.pool_free_stack[self.pool_free_top];
    }

    fn syncSlotActiveTables(self: *Qwen3BatchRuntime, slot_id: u32) void {
        const slot = &self.slots[slot_id];
        const num_layers = self.model.layers.len;
        for (0..num_layers) |l| {
            const tbl = self.slotKvTilesSlice(l, slot_id);
            for (0..self.active_per_slot) |a_idx| {
                const c = slot.num_retired_chunks + @as(u32, @intCast(a_idx));
                if (c < self.max_chunks and slot.chunk_group_ids[c] != NONE_GROUP) {
                    const pair = self.poolTilePair(slot.chunk_group_ids[c], l);
                    tbl[a_idx] = attention.KvTilePair.fromTiles(pair.k, pair.v, .rgba16f);
                } else {
                    tbl[a_idx] = .{ .k_tex_id = 0, .v_tex_id = 0 };
                }
            }
        }
    }

    fn retireSlotOldestActiveChunk(self: *Qwen3BatchRuntime, slot_id: u32) bool {
        var rpool = &(self.retired_pool orelse return false);
        const slot = &self.slots[slot_id];
        const c = slot.num_retired_chunks;
        if (c >= self.max_retired_per_slot or c >= self.max_chunks) return false;
        // Do not retire a chunk that is shared read-only with `PrefixCache` or unallocated.
        if (slot.chunk_is_shared[c] or slot.chunk_group_ids[c] == NONE_GROUP) return false;
        const num_layers = self.model.layers.len;
        if (rpool.freeCount() < num_layers) return false;

        const src_group = slot.chunk_group_ids[c];
        var frame = self.model.compute.beginFrame(self.model.timeout_ms) catch return false;
        var acquired: usize = 0;
        for (0..num_layers) |l| {
            const dst_slot = rpool.acquireSlot() catch {
                frame.abort();
                for (0..acquired) |li| {
                    const rids = self.slotRetiredIdsSlice(li, slot_id);
                    rpool.releaseSlot(rids[c]) catch unreachable;
                }
                return false;
            };
            const rids = self.slotRetiredIdsSlice(l, slot_id);
            rids[c] = dst_slot;
            acquired += 1;
            const pair = self.poolTilePair(src_group, l);
            self.model.attn_pipes.encodeKvRetireTileToQ8(&frame, .{
                .k_tile = pair.k,
                .v_tile = pair.v,
                .pool = rpool,
                .pool_buf = self.batch_buf,
                .dst_slot = dst_slot,
            }, false) catch {
                frame.abort();
                for (0..acquired) |li| {
                    const rids_undo = self.slotRetiredIdsSlice(li, slot_id);
                    rpool.releaseSlot(rids_undo[c]) catch unreachable;
                }
                return false;
            };
        }
        const sig = frame.commit() catch return false;
        self.model.compute.wait(sig, self.model.timeout_ms) catch return false;

        slot.retired_is_shared[c] = false;
        slot.chunk_group_ids[c] = NONE_GROUP;
        self.releaseChunkGroup(src_group);
        slot.num_retired_chunks += 1;
        self.kv_retire_events += 1;
        self.syncSlotActiveTables(slot_id);
        return true;
    }

    fn ensureSlotWritableChunk(self: *Qwen3BatchRuntime, slot_id: u32, chunk_idx: u32) bool {
        if (chunk_idx >= self.max_chunks) return false;
        const slot = &self.slots[slot_id];
        if (chunk_idx < slot.num_retired_chunks) return false;

        // If retired q8 tier is enabled and this chunk exceeds active window, retire oldest private chunk.
        while (self.retired_pool != null and
            chunk_idx >= slot.num_retired_chunks + self.active_per_slot and
            slot.seq_len >= (slot.num_retired_chunks + 1) * self.model.chunk_tokens)
        {
            if (!self.retireSlotOldestActiveChunk(slot_id)) break;
        }
        if (chunk_idx >= slot.num_retired_chunks + self.active_per_slot) return false;

        if (slot.chunk_group_ids[chunk_idx] != NONE_GROUP) {
            return !slot.chunk_is_shared[chunk_idx];
        }
        const g = self.acquireChunkGroup() orelse return false;
        slot.chunk_group_ids[chunk_idx] = g;
        slot.chunk_is_shared[chunk_idx] = false;
        const a_idx: usize = chunk_idx - slot.num_retired_chunks;
        for (0..self.model.layers.len) |l| {
            const pair = self.poolTilePair(g, l);
            self.slotKvTilesSlice(l, slot_id)[a_idx] = attention.KvTilePair.fromTiles(pair.k, pair.v, .rgba16f);
        }
        return true;
    }

    fn maybeRetireSlotExcess(self: *Qwen3BatchRuntime, slot_id: u32) void {
        if (self.retired_pool == null) return;
        const slot = &self.slots[slot_id];
        if (slot.seq_len == 0) return;
        const total_live_chunks: u32 = (slot.seq_len + self.model.chunk_tokens - 1) / self.model.chunk_tokens;
        while (total_live_chunks > slot.num_retired_chunks + self.kv_retire_after_chunks and
            slot.seq_len >= (slot.num_retired_chunks + 1) * self.model.chunk_tokens)
        {
            if (!self.retireSlotOldestActiveChunk(slot_id)) break;
        }
    }

    fn copyTailGroupGpu(
        self: *Qwen3BatchRuntime,
        src_group: u32,
        dst_group: u32,
        valid_tokens: u32,
    ) bool {
        var frame = self.model.compute.beginFrame(self.model.timeout_ms) catch return false;
        for (0..self.model.layers.len) |l| {
            const src_pair = self.poolTilePair(src_group, l);
            const dst_pair = self.poolTilePair(dst_group, l);
            prefix_cache_mod.PrefixCache.encodeTailCowCopy(
                self.model.attn_pipes,
                &frame,
                src_pair.k,
                src_pair.v,
                dst_pair.k,
                dst_pair.v,
                self.model.config.num_key_value_heads,
                self.model.config.head_dim,
                self.model.chunk_tokens,
                valid_tokens,
                false,
            ) catch {
                frame.abort();
                return false;
            };
        }
        const sig = frame.commit() catch return false;
        self.model.compute.wait(sig, self.model.timeout_ms) catch return false;
        self.prefix_cow_copies += 1;
        return true;
    }

    fn resetSlotState(self: *Qwen3BatchRuntime, slot_id: u32) void {
        if (slot_id >= self.max_slots) return;
        const slot = &self.slots[slot_id];
        if (self.prefix_cache) |*pc| {
            for (slot.pinned_full_ids[0..slot.pinned_full_count]) |eid| {
                pc.unpin(eid) catch unreachable;
            }
        }
        slot.pinned_full_count = 0;

        for (0..self.max_chunks) |c| {
            const g = slot.chunk_group_ids[c];
            if (g != NONE_GROUP and !slot.chunk_is_shared[c]) {
                self.releaseChunkGroup(g);
            }
            slot.chunk_group_ids[c] = NONE_GROUP;
            slot.chunk_is_shared[c] = false;
        }
        if (self.retired_pool) |*rpool| {
            for (0..slot.num_retired_chunks) |rc| {
                if (!slot.retired_is_shared[rc]) {
                    for (0..self.model.layers.len) |l| {
                        const rids = self.slotRetiredIdsSlice(l, slot_id);
                        rpool.releaseSlot(rids[rc]) catch unreachable;
                    }
                }
                slot.retired_is_shared[rc] = false;
            }
        }
        slot.num_retired_chunks = 0;
        slot.seq_len = 0;
        slot.matched_prefix_tokens = 0;
        slot.prefix_cached_full_chunks = 0;
        slot.last_cached_parent_id = prefix_cache_mod.NONE_ID;
        slot.in_use = false;
        self.syncSlotActiveTables(slot_id);
    }

    fn lookupPrefixCb(ctx: *anyopaque, slot_id: u16, prompt_tokens: []const u32) batch_sched.PrefixLookupResult {
        const self: *Qwen3BatchRuntime = @ptrCast(@alignCast(ctx));
        return self.lookupPrefix(slot_id, prompt_tokens);
    }

    pub fn lookupPrefix(self: *Qwen3BatchRuntime, slot_id: u16, prompt_tokens: []const u32) batch_sched.PrefixLookupResult {
        if (slot_id >= self.max_slots) return .{};
        self.resetSlotState(slot_id);
        const slot = &self.slots[slot_id];
        slot.in_use = true;
        const handle: u64 = @as(u64, slot_id) + 1;

        var pc = &(self.prefix_cache orelse return .{ .matched_tokens = 0, .prefix_handle = handle });
        if (prompt_tokens.len <= 1 or prompt_tokens.len > self.model.max_seq_len) {
            return .{ .matched_tokens = 0, .prefix_handle = handle };
        }

        const cap_full: usize = @min(self.max_chunks, self.active_per_slot);
        var match = pc.matchAndPin(prompt_tokens, slot.pinned_full_ids[0..cap_full], .{
            .max_match_tokens = prompt_tokens.len - 1,
            .allow_partial_tail = true,
            .require_gdn_snapshot = false,
        }) catch return .{ .matched_tokens = 0, .prefix_handle = handle };

        if (match.matched_tokens == 0) {
            return .{ .matched_tokens = 0, .prefix_handle = handle };
        }

        // Bind matched full blocks (`0` copy!).
        for (0..match.full_block_count) |ci| {
            const eid = slot.pinned_full_ids[ci];
            const h0 = pc.entryHandle(eid, 0) catch {
                pc.releaseMatch(&match, slot.pinned_full_ids[0..match.full_block_count]) catch unreachable;
                self.resetSlotState(slot_id);
                slot.in_use = true;
                return .{ .matched_tokens = 0, .prefix_handle = handle };
            };
            if (h0.storage == .active_tile) {
                slot.chunk_group_ids[ci] = h0.pool_slot_id;
                slot.chunk_is_shared[ci] = true;
            } else {
                // Retired full prefix block (always precedes active blocks).
                slot.retired_is_shared[ci] = true;
                slot.num_retired_chunks += 1;
            }
        }
        slot.pinned_full_count = match.full_block_count;
        slot.prefix_cached_full_chunks = match.full_block_count;
        slot.last_cached_parent_id = if (match.full_block_count > 0)
            slot.pinned_full_ids[match.full_block_count - 1]
        else
            prefix_cache_mod.NONE_ID;

        for (0..self.model.layers.len) |l| {
            _ = pc.populateLayerKvTables(
                slot.pinned_full_ids[0..match.full_block_count],
                @intCast(l),
                self.slotRetiredIdsSlice(l, slot_id),
                self.slotKvTilesSlice(l, slot_id),
            ) catch {
                pc.releaseMatch(&match, slot.pinned_full_ids[0..match.full_block_count]) catch unreachable;
                self.resetSlotState(slot_id);
                slot.in_use = true;
                return .{ .matched_tokens = 0, .prefix_handle = handle };
            };
        }

        // Handle partial tail match via GPU CoW copy into a private writable chunk group.
        var final_matched = match.full_block_count * self.model.chunk_tokens;
        if (match.tail_tokens > 0 and match.tail_entry_id != prefix_cache_mod.NONE_ID) {
            const tail_c = match.full_block_count;
            const tail_src_handle = pc.entryHandle(match.tail_entry_id, 0) catch null;
            if (tail_src_handle != null and tail_src_handle.?.storage == .active_tile and
                tail_c < slot.num_retired_chunks + self.active_per_slot)
            {
                if (self.acquireChunkGroup()) |dst_g| {
                    if (self.copyTailGroupGpu(tail_src_handle.?.pool_slot_id, dst_g, match.tail_tokens)) {
                        slot.chunk_group_ids[tail_c] = dst_g;
                        slot.chunk_is_shared[tail_c] = false;
                        const a_idx: usize = tail_c - slot.num_retired_chunks;
                        for (0..self.model.layers.len) |l| {
                            const pair = self.poolTilePair(dst_g, l);
                            self.slotKvTilesSlice(l, slot_id)[a_idx] = attention.KvTilePair.fromTiles(pair.k, pair.v, .rgba16f);
                        }
                        final_matched += match.tail_tokens;
                    } else {
                        self.releaseChunkGroup(dst_g);
                    }
                }
            }
            pc.releaseTailPin(&match) catch unreachable;
        }

        @memcpy(slot.prompt_tokens[0..final_matched], prompt_tokens[0..final_matched]);
        slot.seq_len = final_matched;
        slot.matched_prefix_tokens = final_matched;
        if (final_matched > 0) {
            self.prefix_hits += 1;
            self.prefix_hit_tokens += final_matched;
        }
        return .{ .matched_tokens = final_matched, .prefix_handle = handle };
    }

    fn populatePrefixCacheAfterPrefill(self: *Qwen3BatchRuntime, slot_id: u32, is_final_chunk: bool) void {
        var pc = &(self.prefix_cache orelse return);
        const slot = &self.slots[slot_id];
        const bs: u32 = self.model.chunk_tokens;
        const num_layers = self.model.layers.len;

        // Promote every newly completed full `bs`-token chunk into `PrefixCache` (`0` copy!).
        while ((slot.prefix_cached_full_chunks + 1) * bs <= slot.seq_len) {
            const c = slot.prefix_cached_full_chunks;
            if (!slot.chunk_is_shared[c] and slot.chunk_group_ids[c] != NONE_GROUP) {
                const gid = slot.chunk_group_ids[c];
                for (0..num_layers) |l| {
                    const pair = self.poolTilePair(gid, l);
                    self.scratch_handles[l] = prefix_cache_mod.KvBlockHandle.activeFromTiles(pair.k, pair.v, gid);
                }
                const tok_start: usize = @as(usize, c) * bs;
                const ins = pc.insertFullBlock(
                    slot.last_cached_parent_id,
                    slot.prompt_tokens[tok_start .. tok_start + bs],
                    self.scratch_handles,
                    .{
                        .evict_if_full = true,
                        .evicted_handles_out = self.scratch_evicted_handles,
                        .pin_on_insert = true,
                    },
                ) catch break;

                if (ins.evicted != null) {
                    self.reclaimEvictedHandles(self.scratch_evicted_handles);
                }
                if (ins.already_existed) {
                    // Deduplicate: switch slot to existing shared entry and release duplicate chunk group.
                    const existing_h0 = pc.entryHandle(ins.entry_id, 0) catch break;
                    if (existing_h0.storage == .active_tile) {
                        self.releaseChunkGroup(gid);
                        slot.chunk_group_ids[c] = existing_h0.pool_slot_id;
                        const a_idx: usize = c - slot.num_retired_chunks;
                        for (0..num_layers) |l| {
                            const h_l = pc.entryHandle(ins.entry_id, @intCast(l)) catch break;
                            self.slotKvTilesSlice(l, slot_id)[a_idx] = h_l.toTilePair();
                        }
                    } else {
                        pc.unpin(ins.entry_id) catch unreachable;
                        break;
                    }
                }
                slot.chunk_is_shared[c] = true;
                slot.pinned_full_ids[slot.pinned_full_count] = ins.entry_id;
                slot.pinned_full_count += 1;
                slot.last_cached_parent_id = ins.entry_id;
            }
            slot.prefix_cached_full_chunks += 1;
        }

        // Snapshot partial tail (`1..bs-1`) via GPU CoW into `PrefixCache` when prompt finishes.
        if (is_final_chunk) {
            const tail_len = slot.seq_len % bs;
            const tail_c = slot.seq_len / bs;
            if (tail_len > 0 and tail_c < self.max_chunks and
                !slot.chunk_is_shared[tail_c] and slot.chunk_group_ids[tail_c] != NONE_GROUP)
            {
                if (self.acquireChunkGroup()) |snap_g| {
                    if (self.copyTailGroupGpu(slot.chunk_group_ids[tail_c], snap_g, tail_len)) {
                        for (0..num_layers) |l| {
                            const pair = self.poolTilePair(snap_g, l);
                            self.scratch_handles[l] = prefix_cache_mod.KvBlockHandle.activeFromTiles(pair.k, pair.v, snap_g);
                        }
                        const tok_start: usize = @as(usize, tail_c) * bs;
                        if (pc.insertTailBlock(
                            slot.last_cached_parent_id,
                            slot.prompt_tokens[tok_start .. tok_start + tail_len],
                            self.scratch_handles,
                            .{
                                .evict_if_full = true,
                                .evicted_handles_out = self.scratch_evicted_handles,
                                .pin_on_insert = false,
                            },
                        )) |ins| {
                            if (ins.evicted != null) {
                                self.reclaimEvictedHandles(self.scratch_evicted_handles);
                            }
                            if (ins.already_existed) {
                                self.releaseChunkGroup(snap_g);
                            }
                        } else |_| {
                            self.releaseChunkGroup(snap_g);
                        }
                    } else {
                        self.releaseChunkGroup(snap_g);
                    }
                }
            }
        }
    }

    fn runPrefillSubChunk(
        self: *Qwen3BatchRuntime,
        slot_id: u32,
        tokens: []const u32,
        base_pos: u32,
        compute_logits: bool,
    ) StepError!u32 {
        const m: u32 = @intCast(tokens.len);
        const model = self.model;
        const slot = &self.slots[slot_id];
        const h = model.config.hidden_size;

        var acc_nodes: [attention.MAX_TREE_NODES]u16 = undefined;
        for (0..m) |i| {
            model.tree_token_ids_host[i] = tokens[i];
            model.tree_positions_host[i] = base_pos + @as(u32, @intCast(i));
            model.tree_ancestor_masks_host[i] = if (i == 63) ~@as(u64, 0) else (@as(u64, 1) << @as(u6, @intCast(i + 1))) - 1;
            acc_nodes[i] = @intCast(i);
        }

        const first_c: u32 = base_pos / model.chunk_tokens;
        const last_c: u32 = (base_pos + m - 1) / model.chunk_tokens;
        var ci: u32 = first_c;
        while (ci <= last_c) : (ci += 1) {
            if (!self.ensureSlotWritableChunk(slot_id, ci)) return error.PositionOutOfRange;
        }

        var frame_opt: ?metal.Compute.Frame = null;
        errdefer if (frame_opt) |*f| f.abort();

        {
            const f = try model.ensureOptionalFrame(&frame_opt);
            try model.moe_pipes.encodeEmbedLookup(f, model.embed.dtype, .{
                .weight = model.embed.weight,
                .scale_bias = model.embed.scale_bias,
                .tokens = .{ .token_ids = model.tree_token_ids_bind },
                .out = model.x_bind,
                .m = m,
                .vocab_size = model.config.vocab_size,
                .hidden_size = h,
            }, false);
        }

        for (model.layers, 0..) |layer, l| {
            {
                const had = (frame_opt != null);
                const f = try model.ensureOptionalFrame(&frame_opt);
                try model.attn_pipes.encodeRmsNorm(f, .{
                    .x = model.x_bind,
                    .w = layer.input_layernorm.binding,
                    .y = model.x_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = model.config.rms_norm_eps,
                    .w_dtype = layer.input_layernorm.dtype,
                }, had);
            }

            if (model.has_sme2_weights) {
                _ = try model.execLinearHetero(&frame_opt, layer.q_proj, model.x_norm_bind, model.x_norm_host, model.q_bind, model.q_host, m, true);
                _ = try model.execLinearHetero(&frame_opt, layer.k_proj, model.x_norm_bind, model.x_norm_host, model.k_bind, model.k_host, m, false);
                _ = try model.execLinearHetero(&frame_opt, layer.v_proj, model.x_norm_bind, model.x_norm_host, model.v_bind, model.v_host, m, false);
            } else {
                const f = try model.ensureOptionalFrame(&frame_opt);
                try model.encodeLinearGemm(f, layer.q_proj, model.x_norm_bind, model.q_bind, m, true);
                try model.encodeLinearGemm(f, layer.k_proj, model.x_norm_bind, model.k_bind, m, false);
                try model.encodeLinearGemm(f, layer.v_proj, model.x_norm_bind, model.v_bind, m, false);
            }

            {
                const had = (frame_opt != null);
                const f = try model.ensureOptionalFrame(&frame_opt);
                const k_tree_bind = model.treeKBinding(l);
                const v_tree_bind = model.treeVBinding(l);
                try model.attn_pipes.encodeTreeQkNormRope(f, .{
                    .q = model.q_bind,
                    .k = model.k_bind,
                    .v = model.v_bind,
                    .q_norm = layer.q_norm,
                    .k_norm = layer.k_norm,
                    .positions = model.tree_positions_bind,
                    .k_tree_out = k_tree_bind,
                    .v_tree_out = v_tree_bind,
                    .num_nodes = m,
                    .num_q_heads = model.config.num_attention_heads,
                    .num_kv_heads = model.config.num_key_value_heads,
                    .head_dim = model.config.head_dim,
                    .rms_eps = model.config.rms_norm_eps,
                    .rope_theta = model.config.rope_theta,
                    .norm_dtype = layer.qk_norm_dtype,
                }, had);

                if (slot.num_retired_chunks > 0 and base_pos > 0) {
                    const rpool = self.retired_pool.?;
                    const ret_toks: u64 = @as(u64, slot.num_retired_chunks) * @as(u64, model.chunk_tokens);
                    try model.attn_pipes.encodeGqaTieredTreeVerifyAttn(f, .{
                        .q = model.q_bind,
                        .kv_tiles = if (@as(u64, base_pos) > ret_toks) self.slotKvTilesBinding(l, slot_id) else null,
                        .retired_pool = rpool.binding(self.batch_buf),
                        .retired_chunk_ids = self.slotRetiredIdsBinding(l, slot_id),
                        .base_pos = base_pos,
                        .k_tree = k_tree_bind,
                        .v_tree = v_tree_bind,
                        .ancestor_masks = model.tree_ancestor_masks_bind,
                        .out = model.attn_out_bind,
                        .scratch = model.gqa_tree_scratch_bind,
                        .max_splits = attention.MAX_SPLITS,
                        .num_nodes = m,
                        .num_q_heads = model.config.num_attention_heads,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .chunk_tokens = model.chunk_tokens,
                        .num_retired_chunks = slot.num_retired_chunks,
                        .retired_plane_stride_bytes = rpool.plane_stride_bytes,
                        .retired_chunk_stride_bytes = rpool.chunk_stride_bytes,
                    }, true);
                } else {
                    try model.attn_pipes.encodeGqaTreeVerifyAttn(f, .{
                        .q = model.q_bind,
                        .kv_tiles = if (base_pos > 0) self.slotKvTilesBinding(l, slot_id) else null,
                        .base_pos = base_pos,
                        .k_tree = k_tree_bind,
                        .v_tree = v_tree_bind,
                        .ancestor_masks = model.tree_ancestor_masks_bind,
                        .out = model.attn_out_bind,
                        .scratch = model.gqa_tree_scratch_bind,
                        .max_splits = attention.MAX_SPLITS,
                        .num_nodes = m,
                        .num_q_heads = model.config.num_attention_heads,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .chunk_tokens = model.chunk_tokens,
                    }, true);
                }
            }

            if (model.has_sme2_weights) {
                _ = try model.execLinearHetero(&frame_opt, layer.o_proj, model.attn_out_bind, model.attn_out_host, model.proj_out_bind, model.proj_out_host, m, true);
            } else {
                const f = try model.ensureOptionalFrame(&frame_opt);
                try model.encodeLinearGemm(f, layer.o_proj, model.attn_out_bind, model.proj_out_bind, m, true);
            }

            {
                const had = (frame_opt != null);
                const f = try model.ensureOptionalFrame(&frame_opt);
                try model.moe_pipes.encodeResidualAdd(f, .{
                    .x = model.x_bind,
                    .delta = model.proj_out_bind,
                    .count = m * h,
                }, had);
                try model.attn_pipes.encodeRmsNorm(f, .{
                    .x = model.x_bind,
                    .w = layer.post_attention_layernorm.binding,
                    .y = model.x_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = model.config.rms_norm_eps,
                    .w_dtype = layer.post_attention_layernorm.dtype,
                }, true);
            }

            switch (model.config.arch) {
                .qwen3_moe => {
                    const f = try model.ensureOptionalFrame(&frame_opt);
                    var layer_ws = model.moe_ws.?;
                    layer_ws.schedule_header_off = model.layer_sched_hdr_base_off + @as(u64, l) * 64;
                    const moe_args: moe.MoELayerArgs = .{
                        .x = model.x_norm_bind,
                        .router = layer.router,
                        .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                        .expert_table = layer.expert_table,
                        .workspace = &layer_ws,
                        .workspace_buf = model.scratch_buf,
                        .residual = model.x_bind,
                        .out = model.x_bind,
                        .m = m,
                        .num_experts = model.config.num_experts,
                        .top_k = model.config.num_experts_per_tok,
                        .hidden_size = h,
                        .intermediate_size = model.config.moe_intermediate_size,
                        .norm_topk_prob = model.config.norm_topk_prob,
                    };
                    try model.encodeMoELayerDispatch(f, layer, moe_args, true);
                },
                .qwen3_dense => {
                    if (model.has_sme2_weights) {
                        _ = try model.execLinearHetero(&frame_opt, layer.dense_gate, model.x_norm_bind, model.x_norm_host, model.dense_gate_bind, model.dense_gate_host, m, true);
                        _ = try model.execLinearHetero(&frame_opt, layer.dense_up, model.x_norm_bind, model.x_norm_host, model.dense_up_bind, model.dense_up_host, m, false);
                        {
                            const had = (frame_opt != null);
                            const f = try model.ensureOptionalFrame(&frame_opt);
                            try model.moe_pipes.encodeSiluMul(f, .{
                                .gate = model.dense_gate_bind,
                                .up = model.dense_up_bind,
                                .out = model.dense_act_bind,
                                .count = m * model.config.intermediate_size,
                            }, had);
                        }
                        _ = try model.execLinearHetero(&frame_opt, layer.dense_down, model.dense_act_bind, model.dense_act_host, model.proj_out_bind, model.proj_out_host, m, true);
                        {
                            const had = (frame_opt != null);
                            const f = try model.ensureOptionalFrame(&frame_opt);
                            try model.moe_pipes.encodeResidualAdd(f, .{
                                .x = model.x_bind,
                                .delta = model.proj_out_bind,
                                .count = m * h,
                            }, had);
                        }
                    } else {
                        const f = try model.ensureOptionalFrame(&frame_opt);
                        try model.encodeLinearGemm(f, layer.dense_gate, model.x_norm_bind, model.dense_gate_bind, m, true);
                        try model.encodeLinearGemm(f, layer.dense_up, model.x_norm_bind, model.dense_up_bind, m, false);
                        try model.moe_pipes.encodeSiluMul(f, .{
                            .gate = model.dense_gate_bind,
                            .up = model.dense_up_bind,
                            .out = model.dense_act_bind,
                            .count = m * model.config.intermediate_size,
                        }, true);
                        try model.encodeLinearGemm(f, layer.dense_down, model.dense_act_bind, model.proj_out_bind, m, true);
                        try model.moe_pipes.encodeResidualAdd(f, .{
                            .x = model.x_bind,
                            .delta = model.proj_out_bind,
                            .count = m * h,
                        }, true);
                    }
                },
            }
        }

        // Commit accepted causal K/V rows straight into this slot's private IOSurface tiles.
        {
            const had = (frame_opt != null);
            const f = try model.ensureOptionalFrame(&frame_opt);
            var first_commit = true;
            var offset: u32 = 0;
            while (offset < m) {
                const cur_base = base_pos + offset;
                const c0: u32 = cur_base / model.chunk_tokens;
                const max_end: u32 = (c0 + 2) * model.chunk_tokens;
                const take: u32 = @min(m - offset, max_end - cur_base);
                const crosses = ((cur_base + take - 1) / model.chunk_tokens) > c0;
                const g0 = slot.chunk_group_ids[c0];
                const g1: u32 = if (crosses) slot.chunk_group_ids[c0 + 1] else NONE_GROUP;
                for (0..model.layers.len) |l| {
                    const t0 = self.poolTilePair(g0, l);
                    const t1 = if (crosses) self.poolTilePair(g1, l) else null;
                    try model.attn_pipes.encodeKvTreeCommit(f, .{
                        .k_tree = model.treeKBinding(l),
                        .v_tree = model.treeVBinding(l),
                        .accepted_nodes = .{ .inline_u16 = acc_nodes[offset .. offset + take] },
                        .accepted_count = take,
                        .k_tile0 = t0.k,
                        .v_tile0 = t0.v,
                        .k_tile1 = if (t1) |p| p.k else null,
                        .v_tile1 = if (t1) |p| p.v else null,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .base_pos = cur_base,
                        .chunk_tokens = model.chunk_tokens,
                    }, if (first_commit) had else false);
                    first_commit = false;
                }
                offset += take;
            }
        }

        if (compute_logits) {
            const had = (frame_opt != null);
            const f = try model.ensureOptionalFrame(&frame_opt);
            const last_x_bind: metal.BufferBinding = .{
                .buffer = model.scratch_buf,
                .offset = model.x_bind.offset + @as(u64, m - 1) * h * @sizeOf(f16),
            };
            try model.attn_pipes.encodeRmsNorm(f, .{
                .x = last_x_bind,
                .w = model.final_norm.binding,
                .y = model.x_norm_bind,
                .num_rows = 1,
                .hidden_dim = h,
                .eps = model.config.rms_norm_eps,
                .w_dtype = model.final_norm.dtype,
            }, had);
            if (model.has_sme2_weights) {
                _ = try model.execLinearHetero(&frame_opt, model.lm_head, model.x_norm_bind, model.x_norm_host, model.logits_bind, model.logits_mut_host, 1, true);
                const f2 = try model.ensureOptionalFrame(&frame_opt);
                try model.moe_pipes.encodeArgmaxRows(f2, .{
                    .logits = model.logits_bind,
                    .out_token_ids = model.out_token_id_bind,
                    .num_rows = 1,
                    .vocab_size = model.config.vocab_size,
                    .logits_dtype = .f16,
                }, false);
            } else {
                try model.encodeLinearGemv(f, model.lm_head, model.x_norm_bind, model.logits_bind, true);
                try model.moe_pipes.encodeArgmaxRows(f, .{
                    .logits = model.logits_bind,
                    .out_token_ids = model.out_token_id_bind,
                    .num_rows = 1,
                    .vocab_size = model.config.vocab_size,
                    .logits_dtype = .f16,
                }, true);
            }
        }

        try model.flushOptionalFrame(&frame_opt);
        slot.seq_len = base_pos + m;
        if (compute_logits) model.last_logits_row = 0;
        return if (compute_logits) model.out_token_ids_host[0] else 0;
    }

    fn prefillSlotCb(ctx: *anyopaque, chunk: *const batch_sched.PrefillChunkView) batch_sched.BackendError!batch_sched.PrefillResult {
        const self: *Qwen3BatchRuntime = @ptrCast(@alignCast(ctx));
        return self.prefillSlot(chunk);
    }

    pub fn prefillSlot(self: *Qwen3BatchRuntime, chunk: *const batch_sched.PrefillChunkView) batch_sched.BackendError!batch_sched.PrefillResult {
        if (chunk.slot_id >= self.max_slots or chunk.tokens.len == 0) return error.InvalidState;
        if (chunk.start_pos + chunk.tokens.len > self.model.max_seq_len) return error.PrefillFailed;
        const slot = &self.slots[chunk.slot_id];
        if (!slot.in_use) {
            self.resetSlotState(chunk.slot_id);
            slot.in_use = true;
        }
        @memcpy(
            slot.prompt_tokens[chunk.start_pos .. chunk.start_pos + chunk.tokens.len],
            chunk.tokens,
        );

        const sub_cap: usize = self.model.max_tree_nodes;
        var off: usize = 0;
        var last_argmax: u32 = 0;
        while (off < chunk.tokens.len) {
            const take = @min(sub_cap, chunk.tokens.len - off);
            const sub_pos = chunk.start_pos + @as(u32, @intCast(off));
            const is_last = (off + take == chunk.tokens.len) and chunk.is_final_chunk;
            last_argmax = self.runPrefillSubChunk(
                chunk.slot_id,
                chunk.tokens[off .. off + take],
                sub_pos,
                is_last,
            ) catch return error.PrefillFailed;
            self.populatePrefixCacheAfterPrefill(chunk.slot_id, is_last);
            self.maybeRetireSlotExcess(chunk.slot_id);
            off += take;
        }

        if (chunk.is_final_chunk) {
            const first_tok: u32 = if (chunk.temperature > 0.0)
                self.model.sampleFromLastLogits(chunk.temperature, chunk.rng.random(), self.p_res) catch return error.PrefillFailed
            else
                last_argmax;
            return .{ .first_token = first_tok };
        }
        return .{};
    }

    fn stepDecodeBatchCb(
        ctx: *anyopaque,
        view: *const batch_sched.BatchStepView,
        out_next_tokens: []u32,
    ) batch_sched.BackendError!void {
        const self: *Qwen3BatchRuntime = @ptrCast(@alignCast(ctx));
        return self.stepDecodeBatch(view, out_next_tokens);
    }

    /// Advances all `B = view.batchSize()` (`1 <= B <= max_slots`) active decode
    /// slots in ONE batched `M = B` GPU/SME2 pass (`0` heap allocations).
    pub fn stepDecodeBatch(
        self: *Qwen3BatchRuntime,
        view: *const batch_sched.BatchStepView,
        out_next_tokens: []u32,
    ) batch_sched.BackendError!void {
        const b_usize = view.batchSize();
        if (b_usize == 0 or b_usize > self.max_slots or out_next_tokens.len < b_usize) {
            return error.InvalidState;
        }
        const b_u32: u32 = @intCast(b_usize);
        const model = self.model;
        const h = model.config.hidden_size;
        const v_size: usize = model.config.vocab_size;

        var max_seq: u32 = 0;
        for (0..b_usize) |bi| {
            const sid: u32 = view.slot_ids[bi];
            const tok = view.input_tokens[bi];
            const pos = view.seq_positions[bi];
            if (sid >= self.max_slots or tok >= model.config.vocab_size or pos >= model.max_seq_len) {
                return error.DecodeFailed;
            }
            const c = pos / model.chunk_tokens;
            if (!self.ensureSlotWritableChunk(sid, c)) return error.SlotUnavailable;
            model.tree_token_ids_host[bi] = tok;
            max_seq = @max(max_seq, pos + 1);
        }

        // Populate per-layer bindless KV write slots and tiered GQA slot descriptors.
        for (0..model.layers.len) |l| {
            const w_slots = self.layerWriteSlotsSlice(l);
            const g_descs = self.layerGqaDescsSlice(l);
            for (0..b_usize) |bi| {
                const sid: u32 = view.slot_ids[bi];
                const pos = view.seq_positions[bi];
                const slot = &self.slots[sid];
                const c = pos / model.chunk_tokens;
                const pair = self.poolTilePair(slot.chunk_group_ids[c], l);
                w_slots[bi] = attention.BatchKvWriteSlot.fromTiles(pair.k, pair.v, pos, model.chunk_tokens);
                g_descs[bi] = .{
                    .seq_len = pos + 1,
                    .tile_table_offset = sid * self.active_per_slot,
                    .retired_offset = sid * self.max_retired_per_slot,
                    .num_retired_chunks = slot.num_retired_chunks,
                };
            }
        }

        var frame_opt: ?metal.Compute.Frame = null;
        errdefer if (frame_opt) |*f| f.abort();

        {
            const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
            model.moe_pipes.encodeEmbedLookup(f, model.embed.dtype, .{
                .weight = model.embed.weight,
                .scale_bias = model.embed.scale_bias,
                .tokens = .{ .token_ids = model.tree_token_ids_bind },
                .out = model.x_bind,
                .m = b_u32,
                .vocab_size = model.config.vocab_size,
                .hidden_size = h,
            }, false) catch return error.DecodeFailed;
        }

        const ret_pool_bind: ?metal.BufferBinding = if (self.retired_pool) |rp| rp.binding(self.batch_buf) else null;
        const ret_plane_stride: u32 = if (self.retired_pool) |rp| rp.plane_stride_bytes else 0;
        const ret_chunk_stride: u32 = if (self.retired_pool) |rp| rp.chunk_stride_bytes else 0;

        for (model.layers, 0..) |layer, l| {
            {
                const had = (frame_opt != null);
                const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
                model.attn_pipes.encodeRmsNorm(f, .{
                    .x = model.x_bind,
                    .w = layer.input_layernorm.binding,
                    .y = model.x_norm_bind,
                    .num_rows = b_u32,
                    .hidden_dim = h,
                    .eps = model.config.rms_norm_eps,
                    .w_dtype = layer.input_layernorm.dtype,
                }, had) catch return error.DecodeFailed;
            }

            if (model.has_sme2_weights) {
                _ = model.execLinearHetero(&frame_opt, layer.q_proj, model.x_norm_bind, model.x_norm_host, model.q_bind, model.q_host, b_u32, true) catch return error.DecodeFailed;
                _ = model.execLinearHetero(&frame_opt, layer.k_proj, model.x_norm_bind, model.x_norm_host, model.k_bind, model.k_host, b_u32, false) catch return error.DecodeFailed;
                _ = model.execLinearHetero(&frame_opt, layer.v_proj, model.x_norm_bind, model.x_norm_host, model.v_bind, model.v_host, b_u32, false) catch return error.DecodeFailed;
            } else {
                const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
                model.encodeLinearGemm(f, layer.q_proj, model.x_norm_bind, model.q_bind, b_u32, true) catch return error.DecodeFailed;
                model.encodeLinearGemm(f, layer.k_proj, model.x_norm_bind, model.k_bind, b_u32, false) catch return error.DecodeFailed;
                model.encodeLinearGemm(f, layer.v_proj, model.x_norm_bind, model.v_bind, b_u32, false) catch return error.DecodeFailed;
            }

            {
                const had = (frame_opt != null);
                const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
                model.attn_pipes.encodeBatchQkNormRopeKvWrite(f, .{
                    .q = model.q_bind,
                    .k = model.k_bind,
                    .v = model.v_bind,
                    .q_norm = layer.q_norm,
                    .k_norm = layer.k_norm,
                    .slots = self.layerWriteSlotsBinding(l),
                    .batch_size = b_u32,
                    .num_q_heads = model.config.num_attention_heads,
                    .num_kv_heads = model.config.num_key_value_heads,
                    .head_dim = model.config.head_dim,
                    .chunk_tokens = model.chunk_tokens,
                    .rms_eps = model.config.rms_norm_eps,
                    .rope_theta = model.config.rope_theta,
                    .norm_dtype = layer.qk_norm_dtype,
                }, had) catch return error.DecodeFailed;

                model.attn_pipes.encodeBatchGqaDecodeAttn(f, .{
                    .q = model.q_bind,
                    .kv_tiles = self.layerKvTilesBinding(l),
                    .retired_pool = ret_pool_bind,
                    .retired_chunk_ids = if (ret_pool_bind != null) self.layerRetiredIdsBinding(l) else null,
                    .slots = self.layerGqaDescsBinding(l),
                    .out = model.attn_out_bind,
                    .scratch = self.gqa_batch_scratch_bind,
                    .max_splits = attention.MAX_SPLITS,
                    .batch_size = b_u32,
                    .num_q_heads = model.config.num_attention_heads,
                    .num_kv_heads = model.config.num_key_value_heads,
                    .head_dim = model.config.head_dim,
                    .max_seq_len = max_seq,
                    .chunk_tokens = model.chunk_tokens,
                    .retired_plane_stride_bytes = ret_plane_stride,
                    .retired_chunk_stride_bytes = ret_chunk_stride,
                }, true) catch return error.DecodeFailed;
            }

            if (model.has_sme2_weights) {
                _ = model.execLinearHetero(&frame_opt, layer.o_proj, model.attn_out_bind, model.attn_out_host, model.proj_out_bind, model.proj_out_host, b_u32, true) catch return error.DecodeFailed;
            } else {
                const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
                model.encodeLinearGemm(f, layer.o_proj, model.attn_out_bind, model.proj_out_bind, b_u32, true) catch return error.DecodeFailed;
            }

            {
                const had = (frame_opt != null);
                const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
                model.moe_pipes.encodeResidualAdd(f, .{
                    .x = model.x_bind,
                    .delta = model.proj_out_bind,
                    .count = b_u32 * h,
                }, had) catch return error.DecodeFailed;
                model.attn_pipes.encodeRmsNorm(f, .{
                    .x = model.x_bind,
                    .w = layer.post_attention_layernorm.binding,
                    .y = model.x_norm_bind,
                    .num_rows = b_u32,
                    .hidden_dim = h,
                    .eps = model.config.rms_norm_eps,
                    .w_dtype = layer.post_attention_layernorm.dtype,
                }, true) catch return error.DecodeFailed;
            }

            switch (model.config.arch) {
                .qwen3_moe => {
                    const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
                    var layer_ws = model.moe_ws.?;
                    layer_ws.schedule_header_off = model.layer_sched_hdr_base_off + @as(u64, l) * 64;
                    const moe_args: moe.MoELayerArgs = .{
                        .x = model.x_norm_bind,
                        .router = layer.router,
                        .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                        .expert_table = layer.expert_table,
                        .workspace = &layer_ws,
                        .workspace_buf = model.scratch_buf,
                        .residual = model.x_bind,
                        .out = model.x_bind,
                        .m = b_u32,
                        .num_experts = model.config.num_experts,
                        .top_k = model.config.num_experts_per_tok,
                        .hidden_size = h,
                        .intermediate_size = model.config.moe_intermediate_size,
                        .norm_topk_prob = model.config.norm_topk_prob,
                    };
                    model.encodeMoELayerDispatch(f, layer, moe_args, true) catch return error.DecodeFailed;
                },
                .qwen3_dense => {
                    if (model.has_sme2_weights) {
                        _ = model.execLinearHetero(&frame_opt, layer.dense_gate, model.x_norm_bind, model.x_norm_host, model.dense_gate_bind, model.dense_gate_host, b_u32, true) catch return error.DecodeFailed;
                        _ = model.execLinearHetero(&frame_opt, layer.dense_up, model.x_norm_bind, model.x_norm_host, model.dense_up_bind, model.dense_up_host, b_u32, false) catch return error.DecodeFailed;
                        {
                            const had = (frame_opt != null);
                            const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
                            model.moe_pipes.encodeSiluMul(f, .{
                                .gate = model.dense_gate_bind,
                                .up = model.dense_up_bind,
                                .out = model.dense_act_bind,
                                .count = b_u32 * model.config.intermediate_size,
                            }, had) catch return error.DecodeFailed;
                        }
                        _ = model.execLinearHetero(&frame_opt, layer.dense_down, model.dense_act_bind, model.dense_act_host, model.proj_out_bind, model.proj_out_host, b_u32, true) catch return error.DecodeFailed;
                        {
                            const had = (frame_opt != null);
                            const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
                            model.moe_pipes.encodeResidualAdd(f, .{
                                .x = model.x_bind,
                                .delta = model.proj_out_bind,
                                .count = b_u32 * h,
                            }, had) catch return error.DecodeFailed;
                        }
                    } else {
                        const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
                        model.encodeLinearGemm(f, layer.dense_gate, model.x_norm_bind, model.dense_gate_bind, b_u32, true) catch return error.DecodeFailed;
                        model.encodeLinearGemm(f, layer.dense_up, model.x_norm_bind, model.dense_up_bind, b_u32, false) catch return error.DecodeFailed;
                        model.moe_pipes.encodeSiluMul(f, .{
                            .gate = model.dense_gate_bind,
                            .up = model.dense_up_bind,
                            .out = model.dense_act_bind,
                            .count = b_u32 * model.config.intermediate_size,
                        }, true) catch return error.DecodeFailed;
                        model.encodeLinearGemm(f, layer.dense_down, model.dense_act_bind, model.proj_out_bind, b_u32, true) catch return error.DecodeFailed;
                        model.moe_pipes.encodeResidualAdd(f, .{
                            .x = model.x_bind,
                            .delta = model.proj_out_bind,
                            .count = b_u32 * h,
                        }, true) catch return error.DecodeFailed;
                    }
                },
            }
        }

        // Batched `M = B` final norm + `lm_head` + `argmax`.
        {
            const had = (frame_opt != null);
            const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
            model.attn_pipes.encodeRmsNorm(f, .{
                .x = model.x_bind,
                .w = model.final_norm.binding,
                .y = model.x_norm_bind,
                .num_rows = b_u32,
                .hidden_dim = h,
                .eps = model.config.rms_norm_eps,
                .w_dtype = model.final_norm.dtype,
            }, had) catch return error.DecodeFailed;
        }
        if (model.has_sme2_weights) {
            _ = model.execLinearHetero(&frame_opt, model.lm_head, model.x_norm_bind, model.x_norm_host, model.logits_bind, model.logits_mut_host, b_u32, true) catch return error.DecodeFailed;
            const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
            model.moe_pipes.encodeArgmaxRows(f, .{
                .logits = model.logits_bind,
                .out_token_ids = model.out_token_id_bind,
                .num_rows = b_u32,
                .vocab_size = model.config.vocab_size,
                .logits_dtype = .f16,
            }, false) catch return error.DecodeFailed;
        } else {
            const f = model.ensureOptionalFrame(&frame_opt) catch return error.DecodeFailed;
            model.encodeLinearGemm(f, model.lm_head, model.x_norm_bind, model.logits_bind, b_u32, true) catch return error.DecodeFailed;
            model.moe_pipes.encodeArgmaxRows(f, .{
                .logits = model.logits_bind,
                .out_token_ids = model.out_token_id_bind,
                .num_rows = b_u32,
                .vocab_size = model.config.vocab_size,
                .logits_dtype = .f16,
            }, true) catch return error.DecodeFailed;
        }

        model.flushOptionalFrame(&frame_opt) catch return error.DecodeFailed;

        // Per-slot `D11` isolated sampling & seq_len update.
        for (0..b_usize) |bi| {
            const sid: u32 = view.slot_ids[bi];
            const temp = view.temperatures[bi];
            if (temp > 0.0) {
                const row = model.logits_host[bi * v_size .. (bi + 1) * v_size];
                accept_mod.softmaxRow(f16, row, temp, self.p_res) catch return error.DecodeFailed;
                out_next_tokens[bi] = accept_mod.sampleCategorical(self.p_res[0..v_size], view.rngs[bi].random());
            } else {
                out_next_tokens[bi] = model.out_token_ids_host[bi];
            }
            self.slots[sid].seq_len = view.seq_positions[bi] + 1;
            self.maybeRetireSlotExcess(sid);
        }

        self.decode_batch_steps += 1;
        self.decode_batch_tokens += b_usize;
    }

    fn releaseSlotCb(ctx: *anyopaque, slot_id: u16, prefix_handle: u64) void {
        _ = prefix_handle;
        const self: *Qwen3BatchRuntime = @ptrCast(@alignCast(ctx));
        self.resetSlotState(slot_id);
    }
};

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

pub const SynthDTypeSpec = struct {
    linear_dtype: quant.WeightDType,
    embed_dtype: quant.WeightDType,
    norm_dtype: quant.FloatDType = .f16,
    router_dtype: amf.DType = .f16,
    expert_dtype: ?quant.WeightDType = null,
    /// Optional per-expert scheme cycle (`expert_dtypes_cycle[e % len]`) for G13 tests.
    expert_dtypes_cycle: ?[]const quant.WeightDType = null,
    /// Optional per-layer linear scheme cycle (`layer_linear_dtypes_cycle[l % len]`) for mixed-layer dense tests.
    layer_linear_dtypes_cycle: ?[]const quant.WeightDType = null,
    /// When true, intentionally mismatches layer 0 expert 0 `up_proj` dtype to test strict per-expert rejection.
    mismatch_expert0_up: bool = false,
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
        .q2 => try quant.QuantOps(.{ .bits = .q2, .group_size = 64 }).packedWeightBytes(n, k),
        .q3 => try quant.QuantOps(.{ .bits = .q3, .group_size = 64 }).packedWeightBytes(n, k),
        .q4 => try quant.QuantOps(.{ .bits = .q4, .group_size = 64 }).packedWeightBytes(n, k),
    };
    const sb_bytes = switch (bits) {
        .q2 => try quant.QuantOps(.{ .bits = .q2, .group_size = 64 }).scaleBiasBytes(n, k),
        .q3 => try quant.QuantOps(.{ .bits = .q3, .group_size = 64 }).scaleBiasBytes(n, k),
        .q4 => try quant.QuantOps(.{ .bits = .q4, .group_size = 64 }).scaleBiasBytes(n, k),
    };
    const data = try allocator.alloc(u8, w_bytes);
    errdefer allocator.free(data);
    for (data) |*b| b.* = @truncate(nextRng(&rng));

    const params = try allocator.alloc(u8, sb_bytes);
    errdefer allocator.free(params);
    const sbs: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, params));
    const mid: f32 = switch (bits) {
        .q2 => 1.5,
        .q3 => 3.5,
        .q4 => 7.5,
    };
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
        .q2_affine => .q2_affine,
    };
}

fn floatDTypeToAmf(fdt: quant.FloatDType) amf.DType {
    return switch (fdt) {
        .f16 => .f16,
        .bf16 => .bf16,
    };
}

pub fn buildSyntheticAmfImageWithSpec(
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
                .q2_affine, .q3_affine, .q4_affine => {
                    const qb: quant.QuantBits = switch (wdt) {
                        .q2_affine => .q2,
                        .q3_affine => .q3,
                        .q4_affine => .q4,
                        else => unreachable,
                    };
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
                .q2_affine => 0.65 * inv_sqrt,
                .q3_affine => 0.35 * inv_sqrt,
                .q4_affine => 0.17 * inv_sqrt,
                .f16, .bf16 => 1.0 * inv_sqrt,
            };
        }
    }.call;

    // embed_tokens
    const emb_scale: f32 = switch (spec.embed_dtype) {
        .q2_affine => 0.15,
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
        const layer_lin_dt: quant.WeightDType = if (spec.layer_linear_dtypes_cycle) |cyc|
            cyc[l_usize % cyc.len]
        else
            spec.linear_dtype;
        try addNorm1D(allocator, &recs, try std.fmt.allocPrint(allocator, "layers.{d}.input_layernorm.weight", .{l}), spec.norm_dtype, l, h, seed + t_idx);
        t_idx += 1;
        try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, layer_lin_dt, l, amf.NO_EXPERT, q_dim, h, scaleFor(layer_lin_dt, inv_sqrt_h), seed + t_idx);
        t_idx += 1;
        try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, layer_lin_dt, l, amf.NO_EXPERT, kv_dim, h, scaleFor(layer_lin_dt, inv_sqrt_h), seed + t_idx);
        t_idx += 1;
        try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, layer_lin_dt, l, amf.NO_EXPERT, kv_dim, h, scaleFor(layer_lin_dt, inv_sqrt_h), seed + t_idx);
        t_idx += 1;
        try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, layer_lin_dt, l, amf.NO_EXPERT, h, q_dim, scaleFor(layer_lin_dt, inv_sqrt_q), seed + t_idx);
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
                .q2_affine => try addWeight2D(allocator, &recs, router_base, false, .router, .q2_affine, l, amf.NO_EXPERT, cfg.num_experts, h, 0.45 * inv_sqrt_h, seed + t_idx),
                .q3_affine => try addWeight2D(allocator, &recs, router_base, false, .router, .q3_affine, l, amf.NO_EXPERT, cfg.num_experts, h, 0.25 * inv_sqrt_h, seed + t_idx),
                .q4_affine => try addWeight2D(allocator, &recs, router_base, false, .router, .q4_affine, l, amf.NO_EXPERT, cfg.num_experts, h, 0.12 * inv_sqrt_h, seed + t_idx),
                .f32, .u32 => return error.UnexpectedTensorDType,
            }
            t_idx += 1;
            const inv_sqrt_mi = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.moe_intermediate_size)));
            for (0..cfg.num_experts) |e_usize| {
                const e: u16 = @intCast(e_usize);
                const e_dt: quant.WeightDType = if (spec.expert_dtypes_cycle) |cyc|
                    cyc[(l_usize + e_usize) % cyc.len]
                else
                    exp_wdt;
                const up_dt: quant.WeightDType = if (spec.mismatch_expert0_up and l == 0 and e == 0)
                    (if (e_dt == .q2_affine) .q4_affine else .q2_affine)
                else
                    e_dt;
                try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }) catch unreachable, spec.use_weight_suffix, .hot_expert, e_dt, l, e, cfg.moe_intermediate_size, h, scaleFor(e_dt, inv_sqrt_h), seed + t_idx);
                t_idx += 1;
                try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }) catch unreachable, spec.use_weight_suffix, .hot_expert, up_dt, l, e, cfg.moe_intermediate_size, h, scaleFor(up_dt, inv_sqrt_h), seed + t_idx);
                t_idx += 1;
                try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }) catch unreachable, spec.use_weight_suffix, .hot_expert, e_dt, l, e, h, cfg.moe_intermediate_size, scaleFor(e_dt, inv_sqrt_mi), seed + t_idx);
                t_idx += 1;
            }
        } else {
            const inv_sqrt_di = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.intermediate_size)));
            try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, layer_lin_dt, l, amf.NO_EXPERT, cfg.intermediate_size, h, scaleFor(layer_lin_dt, inv_sqrt_h), seed + t_idx);
            t_idx += 1;
            try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, layer_lin_dt, l, amf.NO_EXPERT, cfg.intermediate_size, h, scaleFor(layer_lin_dt, inv_sqrt_h), seed + t_idx);
            t_idx += 1;
            try addWeight2D(allocator, &recs, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}) catch unreachable, spec.use_weight_suffix, .dense, layer_lin_dt, l, amf.NO_EXPERT, h, cfg.intermediate_size, scaleFor(layer_lin_dt, inv_sqrt_di), seed + t_idx);
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
            .q2_affine, .q3_affine, .q4_affine => {
                const sb_start = blob_start + @as(usize, @intCast(e.params_offset));
                const sbs: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    self.bytes[sb_start .. sb_start + @as(usize, @intCast(e.params_bytes))],
                ));
                switch (dt) {
                    .q2_affine => try quant.QuantOps(.{ .bits = .q2, .group_size = 64 }).matmulRef(x, raw, sbs, y, 1, e.shape[0], e.shape[1]),
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
            .u32 => return error.UnexpectedTensorDType,
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
            .q2_affine => {
                const Q2 = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                const rb = try Q2.rowWeightBytes(h);
                const gs = try Q2.groupsPerRow(h);
                const qrow = self.bytes[blob_start + @as(usize, token_id) * rb ..][0..rb];
                const sb_start = blob_start + @as(usize, @intCast(e.params_offset));
                const sbs: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    self.bytes[sb_start .. sb_start + @as(usize, @intCast(e.params_bytes))],
                ));
                const sb_row = sbs[@as(usize, token_id) * gs ..][0..gs];
                var tmp: [64]f32 = undefined;
                for (0..gs) |g| {
                    Q2.dequantizeGroup(qrow[g * Q2.groupWeightBytes() ..][0..Q2.groupWeightBytes()], sb_row[g], &tmp);
                    for (0..64) |j| out_f16[g * 64 + j] = @floatCast(tmp[j]);
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
            .f32, .u32 => return error.UnexpectedTensorDType,
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

test "verifyTree + commitTreeKv + feature taps match stepDecode bit-for-bit across chunk boundaries on MoE and Dense BF16" {
    const gpa = std.testing.allocator;
    const cases = [_]struct { cfg: ModelConfig, spec: SynthDTypeSpec }{
        .{
            .cfg = .{
                .arch = .qwen3_moe,
                .quant_bits = .q3,
                .default_float_dtype = .bf16,
                .vocab_size = 256,
                .hidden_size = 128,
                .num_hidden_layers = 5,
                .num_attention_heads = 4,
                .num_key_value_heads = 2,
                .head_dim = 64,
                .moe_intermediate_size = 128,
                .num_experts = 8,
                .num_experts_per_tok = 2,
                .norm_topk_prob = true,
            },
            .spec = .{
                .linear_dtype = .q3_affine,
                .embed_dtype = .q3_affine,
                .norm_dtype = .bf16,
                .router_dtype = .q3_affine,
                .expert_dtype = .q3_affine,
            },
        },
        .{
            .cfg = .{
                .arch = .qwen3_dense,
                .quant_bits = null,
                .default_float_dtype = .bf16,
                .vocab_size = 256,
                .hidden_size = 128,
                .num_hidden_layers = 3,
                .num_attention_heads = 4,
                .num_key_value_heads = 2,
                .head_dim = 64,
                .intermediate_size = 256,
                .tie_word_embeddings = true,
            },
            .spec = .{
                .linear_dtype = .bf16,
                .embed_dtype = .bf16,
                .norm_dtype = .bf16,
                .use_weight_suffix = true,
            },
        },
    };

    for (cases, 0..) |tc, ci| {
        const image = try buildSyntheticAmfImageWithSpec(gpa, tc.cfg, tc.spec, 777 + ci);
        defer gpa.free(image);

        // Reference sequential greedy sequence of 14 tokens across chunk_tokens = 8 boundary.
        var ref_model = try Qwen3Model.loadFromBytes(gpa, image, tc.cfg, .{
            .max_seq_len = 32,
            .chunk_tokens = 8,
            .enable_feature_taps = true,
        });
        defer ref_model.deinit(gpa);

        var seq: [15]u32 = undefined;
        seq[0] = 19;
        var pos: u32 = 0;
        for (0..14) |s| {
            seq[s + 1] = try ref_model.stepDecode(seq[s], pos);
            pos += 1;
        }
        // Capture single-token feature taps at pos = 5 for cross-checking against verifyTree.
        ref_model.resetKvCache();
        for (0..6) |s| {
            _ = try ref_model.stepDecode(seq[s], @intCast(s));
        }
        const ref_aux_pos5 = try gpa.dupe(f16, ref_model.readAuxConcatF16(1));
        defer gpa.free(ref_aux_pos5);
        const ref_top_pos5 = try gpa.dupe(f16, ref_model.readTopHiddenF16(1));
        defer gpa.free(ref_top_pos5);

        // Speculative tree model with chunk_tokens = 8 (forces chunk 0 -> 1 boundary commit!).
        var tree_model = try Qwen3Model.loadFromBytes(gpa, image, tc.cfg, .{
            .max_seq_len = 32,
            .chunk_tokens = 8,
            .max_tree_nodes = 16,
            .enable_feature_taps = true,
        });
        defer tree_model.deinit(gpa);

        // Prefill 0..5 via prefillChunk (5 tokens) so base_pos = 5 -> commit of 5 tokens crosses 8!
        _ = try tree_model.prefillChunk(seq[0..5], 0, false);
        try std.testing.expectEqual(@as(u32, 5), tree_model.seq_len);

        // Build a branching tree at base_pos = 5 where:
        //   node 0: seq[5] (pos 5, root)
        //   node 1: wrong sibling of root (pos 6, mask 0b000011)
        //   node 2: seq[6] (pos 6, mask 0b000101)  -- correct child!
        //   node 3: seq[7] (pos 7, mask 0b001101)  -- correct child!
        //   node 4: wrong child of node 2 (pos 7, mask 0b010101)
        //   node 5: seq[8] (pos 8, mask 0b101101)  -- crosses chunk 0->1!
        //   node 6: seq[9] (pos 9, mask 0b1101101) -- chunk 1!
        const wrong_tok: u32 = (seq[6] + 1) % tc.cfg.vocab_size;
        const wrong_tok2: u32 = (seq[7] + 1) % tc.cfg.vocab_size;
        const tokens = [_]u32{ seq[5], wrong_tok, seq[6], seq[7], wrong_tok2, seq[8], seq[9] };
        const positions = [_]u32{ 5, 6, 6, 7, 7, 8, 9 };
        const masks = [_]u64{
            0b0000001,
            0b0000011,
            0b0000101,
            0b0001101,
            0b0010101,
            0b0101101,
            0b1101101,
        };

        const vr = try tree_model.verifyTree(&tokens, &positions, &masks, 5, true);
        try std.testing.expectEqual(@as(u32, 7), vr.num_nodes);
        try std.testing.expectEqual(seq[6], vr.argmax_tokens[0]);
        try std.testing.expectEqual(seq[7], vr.argmax_tokens[2]);
        try std.testing.expectEqual(seq[8], vr.argmax_tokens[3]);
        try std.testing.expectEqual(seq[9], vr.argmax_tokens[5]);
        try std.testing.expectEqual(seq[10], vr.argmax_tokens[6]);
        if (tc.cfg.arch == .qwen3_moe) {
            try std.testing.expect(vr.total_unique_experts >= tc.cfg.num_hidden_layers * tc.cfg.num_experts_per_tok);
            try std.testing.expect(vr.avg_unique_experts_per_layer >= @as(f64, @floatFromInt(tc.cfg.num_experts_per_tok)));
        }
        try std.testing.expect(vr.actual_bytes_moved > 0);

        // Verify node 0's tapped features match single-token stepDecode at pos 5 closely.
        const tree_aux0 = tree_model.readAuxConcatF16(1);
        const tree_top0 = tree_model.readTopHiddenF16(1);
        for (ref_aux_pos5, tree_aux0) |a, b| {
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(a)), @as(f32, @floatCast(b)), 2e-2);
        }
        for (ref_top_pos5, tree_top0) |a, b| {
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(a)), @as(f32, @floatCast(b)), 2e-2);
        }

        // Commit accepted path [0, 2, 3, 5, 6] across chunk boundary 5..9 -> seq_len = 10.
        const accepted = [_]u16{ 0, 2, 3, 5, 6 };
        try tree_model.commitTreeKv(5, &accepted);
        try std.testing.expectEqual(@as(u32, 10), tree_model.seq_len);

        // Verify accepted_aux_host[0] and accepted_top_host[0] gathered cleanly.
        const h: usize = tc.cfg.hidden_size;
        try std.testing.expectEqualSlices(f16, tree_aux0[0 .. 3 * h], tree_model.accepted_aux_host[0 .. 3 * h]);
        try std.testing.expectEqualSlices(f16, tree_top0[0..h], tree_model.accepted_top_host[0..h]);

        // Subsequent stepDecode steps 10..14 must match sequential reference bit-for-bit!
        var cur = seq[10];
        for (10..14) |s| {
            const nxt = try tree_model.stepDecode(cur, @intCast(s));
            try std.testing.expectEqual(seq[s + 1], nxt);
            cur = nxt;
        }

        // Exercise router-based MoE expert prediction helper (D7).
        if (tc.cfg.arch == .qwen3_moe) {
            var bitsets: [7 * MAX_EXPERT_PRED_LAYERS]tree_mod.ExpertBitset = undefined;
            const sampled = try tree_model.predictMoEExpertsForTokens(&tokens, &bitsets);
            try std.testing.expectEqual(tc.cfg.num_hidden_layers, sampled);
            for (0..tokens.len * sampled) |bi| {
                try std.testing.expectEqual(tc.cfg.num_experts_per_tok, bitsets[bi].count());
            }
        }
    }
}

test "M4 W8 heat profile JSON parser selects per-layer cold experts" {
    const gpa = std.testing.allocator;
    const sample_json =
        \\{
        \\  "schema_version": 1,
        \\  "num_layers": 2,
        \\  "num_experts": 8,
        \\  "layers": [
        \\    { "layer": 0, "ranked_hot_to_cold": [5, 1, 2, 0, 7, 3, 6, 4], "cold_experts": [3, 6, 4] },
        \\    { "layer": 1, "ranked_hot_to_cold": [0, 2, 4, 6, 1, 3, 5, 7], "cold_experts": [1, 3, 5, 7] }
        \\  ]
        \\}
    ;
    var masks = [_]tree_mod.ExpertBitset{ tree_mod.ExpertBitset.empty(), tree_mod.ExpertBitset.empty() };
    try parseHeatProfileColdMask(gpa, sample_json, 2, 8, 0, &masks);
    try std.testing.expectEqual(@as(u32, 3), masks[0].count());
    try std.testing.expect(masks[0].contains(3) and masks[0].contains(6) and masks[0].contains(4));
    try std.testing.expectEqual(@as(u32, 4), masks[1].count());
    try std.testing.expect(masks[1].contains(1) and masks[1].contains(7));

    // Override count uses tail of `ranked_hot_to_cold`
    var masks2 = [_]tree_mod.ExpertBitset{ tree_mod.ExpertBitset.empty(), tree_mod.ExpertBitset.empty() };
    try parseHeatProfileColdMask(gpa, sample_json, 2, 8, 2, &masks2);
    try std.testing.expectEqual(@as(u32, 2), masks2[0].count());
    try std.testing.expect(masks2[0].contains(6) and masks2[0].contains(4));
    try std.testing.expectEqual(@as(u32, 2), masks2[1].count());
    try std.testing.expect(masks2[1].contains(5) and masks2[1].contains(7));

    try std.testing.expectError(error.InvalidHeatProfile, parseHeatProfileColdMask(gpa, "{\"schema_version\":2}", 2, 8, 0, &masks2));
}

test "M4 cold-expert NVMe streaming decode and verifyTree match all-hot bit-for-bit with ANE predictor" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .default_float_dtype = .bf16,
        .vocab_size = 256,
        .hidden_size = 128,
        .num_hidden_layers = 3,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 16,
        .num_experts_per_tok = 2,
        .norm_topk_prob = true,
    };
    const image = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .q3_affine,
        .embed_dtype = .q3_affine,
        .norm_dtype = .bf16,
        .router_dtype = .q3_affine,
        .expert_dtype = .q3_affine,
    }, 4040);
    defer gpa.free(image);

    // Write temporary `.amf` under `/Volumes/Akari/.tmp` (never `/tmp` or repo!).
    const tmp_path: [:0]const u8 = "/Volumes/Akari/.tmp/aminfer-test-m4-cold.amf";
    {
        const fd = std.c.open(tmp_path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.close(fd);
        var written: usize = 0;
        while (written < image.len) {
            const rc = std.c.write(fd, image.ptr + written, image.len - written);
            try std.testing.expect(rc > 0);
            written += @intCast(rc);
        }
    }
    defer _ = std.c.unlink(tmp_path.ptr);

    var hot_model = try Qwen3Model.loadFromFile(gpa, tmp_path, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 32,
        .max_tree_nodes = 8,
    });
    defer hot_model.deinit(gpa);

    var hot_seq: [9]u32 = undefined;
    hot_seq[0] = 17;
    for (0..8) |s| {
        hot_seq[s + 1] = try hot_model.stepDecode(hot_seq[s], @intCast(s));
    }

    const vm0 = try vmstat.Snapshot.capture();
    const backends = [_]aio.IoBackend{ .ecore_pread, .kqueue_aio };
    for (backends) |be| {
        var cold_model = try Qwen3Model.loadFromFile(gpa, tmp_path, cfg, .{
            .max_seq_len = 64,
            .chunk_tokens = 32,
            .max_tree_nodes = 8,
            .cold_experts_per_layer = 8,
            .cold_slot_count = 16,
            .io_backend = be,
            .enable_ane_predictor = true,
        });
        defer cold_model.deinit(gpa);

        try std.testing.expect(cold_model.hasColdExperts());
        try std.testing.expect(cold_model.totalWiredBytes() < hot_model.totalWiredBytes());

        const st0 = cold_model.coldStats().?;
        try std.testing.expectEqual(@as(u32, 24), st0.total_cold_experts);
        try std.testing.expectEqual(be, st0.io_backend);
        try std.testing.expect(st0.ane_predictor_active);
        try std.testing.expect(st0.ane_predictor_uses_ne);

        // 100% bit-for-bit decode parity vs all-hot baseline!
        var cur: u32 = hot_seq[0];
        for (0..8) |s| {
            const got = try cold_model.stepDecode(cur, @intCast(s));
            try std.testing.expectEqual(hot_seq[s + 1], got);
            cur = got;
        }

        // 100% bit-for-bit verifyTree parity on cold-expert model!
        cold_model.resetKvCache();
        for (0..3) |s| try cold_model.prefillToken(hot_seq[s], @intCast(s));
        const tree_toks = [_]u32{ hot_seq[3], hot_seq[4], hot_seq[5] };
        const tree_pos = [_]u32{ 3, 4, 5 };
        const tree_masks = [_]u64{ 0b001, 0b011, 0b111 };
        const vr = try cold_model.verifyTree(&tree_toks, &tree_pos, &tree_masks, 3, true);
        try std.testing.expectEqual(hot_seq[4], vr.argmax_tokens[0]);
        try std.testing.expectEqual(hot_seq[5], vr.argmax_tokens[1]);
        try std.testing.expectEqual(hot_seq[6], vr.argmax_tokens[2]);

        const st1 = cold_model.coldStats().?;
        try std.testing.expect(st1.cache_telemetry.hot_hits > 0);
        try std.testing.expect(st1.cache_telemetry.totalColdAcquires() > 0);
        try std.testing.expect(st1.cache_telemetry.bytes_read > 0);

        // Verify D7 cold-mask export reports sampled layers.
        var cm: [MAX_EXPERT_PRED_LAYERS]tree_mod.ExpertBitset = undefined;
        try std.testing.expectEqual(cfg.num_hidden_layers, cold_model.fillSampledColdMasks(&cm));
    }
    const vm1 = try vmstat.Snapshot.capture();
    try std.testing.expectEqual(@as(u64, 0), vmstat.Snapshot.swapoutDelta(vm0, vm1));
}

test "M4 G6 runLayerFfnChunk and A1 prefillHybridWavefront run zero-copy 2-chunk ANE/GPU prefill" {
    const gpa = std.testing.allocator;
    const cases = [_]struct { cfg: ModelConfig, spec: SynthDTypeSpec }{
        .{
            .cfg = .{
                .arch = .qwen3_dense,
                .quant_bits = null,
                .default_float_dtype = .bf16,
                .vocab_size = 192,
                .hidden_size = 128,
                .num_hidden_layers = 2,
                .num_attention_heads = 4,
                .num_key_value_heads = 2,
                .head_dim = 64,
                .intermediate_size = 256,
                .tie_word_embeddings = true,
            },
            .spec = .{
                .linear_dtype = .bf16,
                .embed_dtype = .bf16,
                .norm_dtype = .bf16,
                .use_weight_suffix = true,
            },
        },
        .{
            .cfg = .{
                .arch = .qwen3_moe,
                .quant_bits = .q3,
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
            },
            .spec = .{
                .linear_dtype = .q3_affine,
                .embed_dtype = .q3_affine,
                .norm_dtype = .bf16,
                .router_dtype = .q3_affine,
                .expert_dtype = .q3_affine,
            },
        },
    };

    for (cases, 0..) |tc, ci| {
        const image = try buildSyntheticAmfImageWithSpec(gpa, tc.cfg, tc.spec, 5050 + ci);
        defer gpa.free(image);

        var model = try Qwen3Model.loadFromBytes(gpa, image, tc.cfg, .{
            .max_seq_len = 192,
            .chunk_tokens = 64,
            .max_tree_nodes = 16,
            .prefill_mode = .hybrid,
            .enable_hybrid_prefill = true,
            .hybrid_max_bucket = 128,
        });
        defer model.deinit(gpa);

        try std.testing.expect(model.hasHybridPrefill());

        var prompt: [128]u32 = undefined;
        for (&prompt, 0..) |*p, i| p.* = @intCast((i * 7 + 5) % tc.cfg.vocab_size);

        const res = try model.prefillHybridWavefront(&prompt, true);
        try std.testing.expectEqual(@as(u32, 128), res.tokens_prefilled);
        try std.testing.expectEqual(@as(u32, 128), model.seq_len);
        try std.testing.expect(res.argmax_token < tc.cfg.vocab_size);
        try std.testing.expectEqual(@as(u32, 3), res.wavefront_stats.overlapped_steps);
        try std.testing.expect(res.all_outputs_zero_copy);
        try std.testing.expect(res.all_buckets_use_ane);

        // Decode seamlessly right after hybrid wavefront prefill over the ANE-written KV tiles!
        const next_tok = try model.stepDecode(res.argmax_token, 128);
        try std.testing.expect(next_tok < tc.cfg.vocab_size);
        for (model.readLogitsF16()) |lv| {
            try std.testing.expect(std.math.isFinite(@as(f32, @floatCast(lv))));
        }
    }
}

pub const RepackTargets = enum { routers, all_linear };

pub fn repackAmfBytesToSme2(
    allocator: std.mem.Allocator,
    src_bytes: []const u8,
    targets: RepackTargets,
    mark_cold_expert_from: ?u16,
) ![]u8 {
    var parsed = try amf.Parsed.parseBytes(allocator, src_bytes);
    defer parsed.deinit(allocator);

    const n_entries = parsed.entries.len;
    const new_entries = try allocator.alloc(amf.TensorEntry, n_entries);
    defer allocator.free(new_entries);
    const payloads = try allocator.alloc([]u8, n_entries);
    var built: usize = 0;
    defer {
        for (0..built) |i| allocator.free(payloads[i]);
        allocator.free(payloads);
    }

    for (parsed.entries, 0..) |e, i| {
        var ne = e;
        const kind = try e.tensorKind();
        const dt = try e.dataType();
        const should_pack: bool = switch (targets) {
            .routers => kind == .router,
            .all_linear => switch (kind) {
                .router, .dense, .hot_expert, .cold_expert, .lm_head, .shared_expert, .loop_adapter => true,
                .embedding, .norm, .sparse_table => false,
            },
        };
        if (mark_cold_expert_from) |cold_from| {
            if (kind == .hot_expert and e.expert != amf.NO_EXPERT and e.expert >= cold_from) {
                ne.kind = @backingInt(amf.TensorKind.cold_expert);
            }
        }

        const src_off: usize = @intCast(e.blob_offset);
        const src_len: usize = @intCast(e.blob_bytes);
        const src_blob = src_bytes[src_off .. src_off + src_len];

        if (should_pack and e.rank == 2 and dt != .f32 and dt != .u32) {
            const n: usize = e.shape[0];
            const k: usize = e.shape[1];
            const pdt = try amfDTypeToPanelDType(dt);
            const packed_len = try sme.packedBlobBytes(pdt, n, k);
            const aligned_len = try page.alignUp(packed_len);
            const dst = try allocator.alloc(u8, aligned_len);
            @memset(dst, 0);
            switch (dt) {
                .f16, .bf16 => {
                    const src_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, src_blob[0 .. n * k * 2]));
                    try sme.packDensePanel64(pdt, src_u16, n, k, dst[0..packed_len]);
                },
                .q4_affine => {
                    const sb_off: usize = @intCast(e.params_offset);
                    const sb_len: usize = @intCast(e.params_bytes);
                    const sbs: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, src_blob[sb_off .. sb_off + sb_len]));
                    try sme.packMlxQuantPanel64(.q4, src_blob[0..@intCast(e.data_bytes)], sbs, n, k, dst[0..packed_len]);
                },
                .q3_affine => {
                    const sb_off: usize = @intCast(e.params_offset);
                    const sb_len: usize = @intCast(e.params_bytes);
                    const sbs: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, src_blob[sb_off .. sb_off + sb_len]));
                    try sme.packMlxQuantPanel64(.q3, src_blob[0..@intCast(e.data_bytes)], sbs, n, k, dst[0..packed_len]);
                },
                .q2_affine => {
                    const sb_off: usize = @intCast(e.params_offset);
                    const sb_len: usize = @intCast(e.params_bytes);
                    const sbs: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, src_blob[sb_off .. sb_off + sb_len]));
                    try sme.packMlxQuantPanel64(.q2, src_blob[0..@intCast(e.data_bytes)], sbs, n, k, dst[0..packed_len]);
                },
                else => unreachable,
            }
            ne.setLayoutAndParamKind(.sme2_panel64, .affine_scale_bias);
            ne.data_bytes = packed_len;
            ne.params_offset = 0;
            ne.params_bytes = 0;
            ne.blob_bytes = aligned_len;
            ne.blob_crc32 = std.hash.Crc32.hash(dst);
            payloads[i] = dst;
        } else {
            const dst = try allocator.dupe(u8, src_blob);
            ne.blob_crc32 = std.hash.Crc32.hash(dst);
            payloads[i] = dst;
        }
        new_entries[i] = ne;
        built += 1;
    }

    const dir_raw_bytes: u64 = @as(u64, n_entries) * @sizeOf(amf.TensorEntry);
    const dir_page_bytes: u64 = try page.alignUp(@intCast(dir_raw_bytes));
    const dir_off: u64 = page.APPLE_PAGE_SIZE;
    const blobs_start: u64 = dir_off + dir_page_bytes;

    var cur_off: u64 = blobs_start;
    for (new_entries, 0..) |*ne, i| {
        ne.blob_offset = cur_off;
        ne.blob_bytes = payloads[i].len;
        cur_off += payloads[i].len;
    }

    const out = try allocator.alloc(u8, @intCast(cur_off));
    @memset(out, 0);
    for (new_entries, 0..) |ne, i| {
        const off: usize = @intCast(ne.blob_offset);
        @memcpy(out[off .. off + payloads[i].len], payloads[i]);
    }
    const dir_slice = std.mem.sliceAsBytes(new_entries);
    @memcpy(out[@intCast(dir_off)..@intCast(dir_off + dir_raw_bytes)], dir_slice);

    var hdr = parsed.header;
    hdr.directory_offset = dir_off;
    hdr.directory_bytes = dir_page_bytes;
    hdr.directory_crc32 = std.hash.Crc32.hash(dir_slice);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(out[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return out;
}

test "M4.5 W15 single-owner sme2_panel64 routers and all-linear match row_major GPU decode and verifyTree" {
    const gpa = std.testing.allocator;

    // 1. MoE q3_affine with single-owner `sme2_panel64` routers-only (`--targets routers`)
    {
        const cfg: ModelConfig = .{
            .arch = .qwen3_moe,
            .quant_bits = .q3,
            .default_float_dtype = .bf16,
            .vocab_size = 128,
            .hidden_size = 128,
            .num_hidden_layers = 2,
            .num_attention_heads = 2,
            .num_key_value_heads = 1,
            .head_dim = 64,
            .moe_intermediate_size = 128,
            .num_experts = 16,
            .num_experts_per_tok = 2,
            .norm_topk_prob = true,
        };
        const row_img = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
            .linear_dtype = .q3_affine,
            .embed_dtype = .q3_affine,
            .norm_dtype = .bf16,
            .router_dtype = .q3_affine,
            .expert_dtype = .q3_affine,
        }, 0x4501);
        defer gpa.free(row_img);
        const sme_img = try repackAmfBytesToSme2(gpa, row_img, .routers, null);
        defer gpa.free(sme_img);

        var gpu_model = try Qwen3Model.loadFromBytes(gpa, row_img, cfg, .{
            .max_seq_len = 32,
            .chunk_tokens = 16,
            .max_tree_nodes = 8,
            .sme_router_mode = .gpu,
        });
        defer gpu_model.deinit(gpa);
        var sme_model = try Qwen3Model.loadFromBytes(gpa, sme_img, cfg, .{
            .max_seq_len = 32,
            .chunk_tokens = 16,
            .max_tree_nodes = 8,
            .sme_router_mode = .auto,
        });
        defer sme_model.deinit(gpa);

        try std.testing.expect(!gpu_model.hasSme2Weights());
        try std.testing.expect(sme_model.hasSme2Weights());
        const cal = sme_model.routerPredCalibration();
        try std.testing.expectEqual(RouterPredBackend.sme2_single_owner, cal.backend);
        try std.testing.expectEqual(@as(u64, 0), cal.bank_bytes); // single-owner: ZERO duplicate RAM!

        var cur_g: u32 = 11;
        var cur_s: u32 = 11;
        for (0..5) |step| {
            cur_g = try gpu_model.stepDecode(cur_g, @intCast(step));
            cur_s = try sme_model.stepDecode(cur_s, @intCast(step));
            try std.testing.expectEqual(cur_g, cur_s);
        }

        // Verify tree parity as well.
        gpu_model.resetKvCache();
        sme_model.resetKvCache();
        try gpu_model.prefillToken(11, 0);
        try sme_model.prefillToken(11, 0);
        const toks = [_]u32{ 5, 19, 37, 73 };
        const pos = [_]u32{ 1, 2, 2, 3 };
        const masks = [_]u64{ 0b0001, 0b0011, 0b0101, 0b1011 };
        const vr_g = try gpu_model.verifyTree(&toks, &pos, &masks, 1, true);
        const vr_s = try sme_model.verifyTree(&toks, &pos, &masks, 1, true);
        try std.testing.expectEqualSlices(u32, vr_g.argmax_tokens, vr_s.argmax_tokens);
    }

    // 2. MoE q4_affine all-linear with cold-tagged `ldnt1b` experts (`--targets all-linear`)
    {
        const cfg: ModelConfig = .{
            .arch = .qwen3_moe,
            .quant_bits = .q4,
            .default_float_dtype = .f16,
            .vocab_size = 96,
            .hidden_size = 128,
            .num_hidden_layers = 2,
            .num_attention_heads = 2,
            .num_key_value_heads = 1,
            .head_dim = 64,
            .moe_intermediate_size = 128,
            .num_experts = 8,
            .num_experts_per_tok = 2,
            .norm_topk_prob = true,
        };
        const row_img = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
            .linear_dtype = .q4_affine,
            .embed_dtype = .f16,
            .norm_dtype = .f16,
            .router_dtype = .f16,
            .expert_dtype = .q4_affine,
        }, 0x4502);
        defer gpa.free(row_img);
        const sme_img = try repackAmfBytesToSme2(gpa, row_img, .all_linear, 4);
        defer gpa.free(sme_img);

        var gpu_model = try Qwen3Model.loadFromBytes(gpa, row_img, cfg, .{
            .max_seq_len = 32,
            .chunk_tokens = 16,
            .max_tree_nodes = 8,
            .sme_router_mode = .gpu,
        });
        defer gpu_model.deinit(gpa);
        var sme_model = try Qwen3Model.loadFromBytes(gpa, sme_img, cfg, .{
            .max_seq_len = 32,
            .chunk_tokens = 16,
            .max_tree_nodes = 8,
            .sme_router_mode = .auto,
        });
        defer sme_model.deinit(gpa);

        try std.testing.expect(sme_model.hasSme2Weights());
        var cur: u32 = 7;
        for (0..4) |step| {
            const g_tok = try gpu_model.stepDecode(cur, @intCast(step));
            const s_tok = try sme_model.stepDecode(cur, @intCast(step));
            try std.testing.expectEqual(g_tok, s_tok);
            var max_diff: f32 = 0.0;
            for (gpu_model.readLogitsF16(), sme_model.readLogitsF16()) |a, b| {
                const d = @abs(@as(f32, @floatCast(a)) - @as(f32, @floatCast(b)));
                if (d > max_diff) max_diff = d;
            }
            try std.testing.expect(max_diff < 0.05);
            cur = g_tok;
        }
    }

    // 3. Dense bf16 all-linear (`--targets all-linear`)
    {
        const cfg: ModelConfig = .{
            .arch = .qwen3_dense,
            .quant_bits = null,
            .default_float_dtype = .bf16,
            .vocab_size = 96,
            .hidden_size = 128,
            .num_hidden_layers = 2,
            .num_attention_heads = 2,
            .num_key_value_heads = 1,
            .head_dim = 64,
            .intermediate_size = 192,
            .tie_word_embeddings = false,
        };
        const row_img = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
            .linear_dtype = .bf16,
            .embed_dtype = .bf16,
            .norm_dtype = .bf16,
            .use_weight_suffix = true,
        }, 0x4503);
        defer gpa.free(row_img);
        const sme_img = try repackAmfBytesToSme2(gpa, row_img, .all_linear, null);
        defer gpa.free(sme_img);

        var gpu_model = try Qwen3Model.loadFromBytes(gpa, row_img, cfg, .{
            .max_seq_len = 32,
            .chunk_tokens = 16,
            .max_tree_nodes = 8,
        });
        defer gpu_model.deinit(gpa);
        var sme_model = try Qwen3Model.loadFromBytes(gpa, sme_img, cfg, .{
            .max_seq_len = 32,
            .chunk_tokens = 16,
            .max_tree_nodes = 8,
        });
        defer sme_model.deinit(gpa);

        const toks = [_]u32{ 3, 9, 15, 21 };
        const pos = [_]u32{ 0, 1, 1, 2 };
        const masks = [_]u64{ 0b0001, 0b0011, 0b0101, 0b1011 };
        const vr_g = try gpu_model.verifyTree(&toks, &pos, &masks, 0, true);
        const vr_s = try sme_model.verifyTree(&toks, &pos, &masks, 0, true);
        try std.testing.expectEqualSlices(u32, vr_g.argmax_tokens, vr_s.argmax_tokens);
    }
}

test "M4.5 C13 L2-resident SME2 router prediction matches GPU bitsets and calibrates automatically" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .default_float_dtype = .bf16,
        .vocab_size = 128,
        .hidden_size = 256,
        .num_hidden_layers = 8,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 32,
        .num_experts_per_tok = 4,
        .norm_topk_prob = true,
    };
    const image = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .q3_affine,
        .embed_dtype = .q3_affine,
        .norm_dtype = .bf16,
        .router_dtype = .bf16,
        .expert_dtype = .q3_affine,
    }, 0x4513);
    defer gpa.free(image);

    var model = try Qwen3Model.loadFromBytes(gpa, image, cfg, .{
        .max_seq_len = 32,
        .chunk_tokens = 16,
        .max_tree_nodes = 16,
        .sme_router_mode = .auto,
    });
    defer model.deinit(gpa);

    const caps = try sme.SmeCaps.probe();
    const cal = model.routerPredCalibration();
    if (caps.supportsSme2()) {
        try std.testing.expectEqual(RouterPredBackend.sme2, cal.backend);
        try std.testing.expect(cal.sme2_us > 0.0 and cal.gpu_us > 0.0);
        try std.testing.expect(cal.sme2_us < cal.gpu_us);
        try std.testing.expect(cal.bank_bytes > 0);
        try std.testing.expectEqual(@as(u32, 8), cal.sampled_layers);
    }

    const toks = [_]u32{ 1, 7, 19, 43, 61, 89, 103, 121 };
    var gpu_bs: [8 * MAX_EXPERT_PRED_LAYERS]tree_mod.ExpertBitset = undefined;
    var sme_bs: [8 * MAX_EXPERT_PRED_LAYERS]tree_mod.ExpertBitset = undefined;
    const s_g = try model.predictMoEExpertsForTokensGpu(&toks, &gpu_bs);
    const s_s = try model.predictMoEExpertsForTokensSme2(&toks, &sme_bs);
    try std.testing.expectEqual(s_g, s_s);
    try std.testing.expectEqual(@as(u32, 8), s_s);

    var overlap: usize = 0;
    const total = toks.len * @as(usize, s_s) * cfg.num_experts_per_tok;
    for (0..toks.len * s_s) |i| {
        try std.testing.expectEqual(cfg.num_experts_per_tok, sme_bs[i].count());
        overlap += gpu_bs[i].intersectWith(sme_bs[i]).count();
    }
    // Folded f16 L2 router bank agrees >98% with GPU top-K expert sets.
    try std.testing.expect(@as(f64, @floatFromInt(overlap)) / @as(f64, @floatFromInt(total)) >= 0.98);
}

test "M4.5 C12 evalSwiGluExpertSme2 non-temporal ldnt1b matches FP32 reference across bf16, q4 and q3" {
    const gpa = std.testing.allocator;
    var disp = try sme.SmeDispatcher.init(gpa, .{});
    defer disp.deinit();

    const m: usize = 6;
    const h: usize = 128;
    const inter: usize = 64;
    const dtypes = [_]sme.PanelDType{ .bf16, .q4_affine, .q3_affine, .q2_affine };

    var rng_st: u64 = 0x51C1_0045;
    for (dtypes) |dt| {
        const act_fdt: quant.FloatDType = if (dt.activationIsBf16()) .bf16 else .f16;
        const x_u16 = try gpa.alignedAlloc(u16, .fromByteUnits(sme.SVL_BYTES), m * h);
        defer gpa.free(x_u16);
        const x_f32 = try gpa.alloc(f32, m * h);
        defer gpa.free(x_f32);
        const resid_f32 = try gpa.alloc(f32, m * h);
        defer gpa.free(resid_f32);
        for (0..m * h) |i| {
            const v = (@as(f32, @floatFromInt(nextRng(&rng_st) & 0xFFFF)) / 32768.0 - 1.0) * 0.5;
            x_u16[i] = quant.f32ToFloatBits(act_fdt, v);
            x_f32[i] = quant.floatBitsToF32(act_fdt, x_u16[i]);
            resid_f32[i] = (@as(f32, @floatFromInt(nextRng(&rng_st) & 0xFFFF)) / 32768.0 - 1.0) * 0.25;
        }

        const gate_bytes = try sme.packedBlobBytes(dt, inter, h);
        const down_bytes = try sme.packedBlobBytes(dt, h, inter);
        const gate_p = try gpa.alignedAlloc(u8, .fromByteUnits(sme.SVL_BYTES), gate_bytes);
        defer gpa.free(gate_p);
        const up_p = try gpa.alignedAlloc(u8, .fromByteUnits(sme.SVL_BYTES), gate_bytes);
        defer gpa.free(up_p);
        const down_p = try gpa.alignedAlloc(u8, .fromByteUnits(sme.SVL_BYTES), down_bytes);
        defer gpa.free(down_p);

        const gate_ref = try gpa.alloc(f32, inter * h);
        defer gpa.free(gate_ref);
        const up_ref = try gpa.alloc(f32, inter * h);
        defer gpa.free(up_ref);
        const down_ref = try gpa.alloc(f32, h * inter);
        defer gpa.free(down_ref);

        if (!dt.isQuantized()) {
            const g_u16 = try gpa.alloc(u16, inter * h);
            defer gpa.free(g_u16);
            const u_u16 = try gpa.alloc(u16, inter * h);
            defer gpa.free(u_u16);
            const d_u16 = try gpa.alloc(u16, h * inter);
            defer gpa.free(d_u16);
            for (0..inter * h) |i| {
                const gv = (@as(f32, @floatFromInt(nextRng(&rng_st) & 0xFFFF)) / 32768.0 - 1.0) * 0.2;
                const uv = (@as(f32, @floatFromInt(nextRng(&rng_st) & 0xFFFF)) / 32768.0 - 1.0) * 0.2;
                g_u16[i] = quant.f32ToFloatBits(act_fdt, gv);
                u_u16[i] = quant.f32ToFloatBits(act_fdt, uv);
                gate_ref[i] = quant.floatBitsToF32(act_fdt, g_u16[i]);
                up_ref[i] = quant.floatBitsToF32(act_fdt, u_u16[i]);
            }
            for (0..h * inter) |i| {
                const dv = (@as(f32, @floatFromInt(nextRng(&rng_st) & 0xFFFF)) / 32768.0 - 1.0) * 0.2;
                d_u16[i] = quant.f32ToFloatBits(act_fdt, dv);
                down_ref[i] = quant.floatBitsToF32(act_fdt, d_u16[i]);
            }
            try sme.packDensePanel64(dt, g_u16, inter, h, gate_p);
            try sme.packDensePanel64(dt, u_u16, inter, h, up_p);
            try sme.packDensePanel64(dt, d_u16, h, inter, down_p);
        } else {
            const mask: u8 = switch (dt) {
                .q4_affine => 0x0F,
                .q3_affine => 0x07,
                .q2_affine => 0x03,
                else => unreachable,
            };
            const g_codes = try gpa.alloc(u8, inter * h);
            defer gpa.free(g_codes);
            const u_codes = try gpa.alloc(u8, inter * h);
            defer gpa.free(u_codes);
            const d_codes = try gpa.alloc(u8, h * inter);
            defer gpa.free(d_codes);
            const g_sb = try gpa.alloc(quant.ScaleBias, inter * (h / 64));
            defer gpa.free(g_sb);
            const u_sb = try gpa.alloc(quant.ScaleBias, inter * (h / 64));
            defer gpa.free(u_sb);
            const d_sb = try gpa.alloc(quant.ScaleBias, h * (inter / 64));
            defer gpa.free(d_sb);
            for (g_sb) |*sb| sb.* = .{ .scale = 0.04, .bias = -0.15 };
            for (u_sb) |*sb| sb.* = .{ .scale = 0.04, .bias = -0.15 };
            for (d_sb) |*sb| sb.* = .{ .scale = 0.04, .bias = -0.15 };
            for (0..inter) |r| {
                for (0..h) |c| {
                    g_codes[r * h + c] = @as(u8, @truncate(nextRng(&rng_st))) & mask;
                    u_codes[r * h + c] = @as(u8, @truncate(nextRng(&rng_st))) & mask;
                    const gsb = g_sb[r * (h / 64) + c / 64];
                    const usb = u_sb[r * (h / 64) + c / 64];
                    gate_ref[r * h + c] = @as(f32, @floatFromInt(g_codes[r * h + c])) * @as(f32, @floatCast(gsb.scale)) + @as(f32, @floatCast(gsb.bias));
                    up_ref[r * h + c] = @as(f32, @floatFromInt(u_codes[r * h + c])) * @as(f32, @floatCast(usb.scale)) + @as(f32, @floatCast(usb.bias));
                }
            }
            for (0..h) |r| {
                for (0..inter) |c| {
                    d_codes[r * inter + c] = @as(u8, @truncate(nextRng(&rng_st))) & mask;
                    const dsb = d_sb[r * (inter / 64) + c / 64];
                    down_ref[r * inter + c] = @as(f32, @floatFromInt(d_codes[r * inter + c])) * @as(f32, @floatCast(dsb.scale)) + @as(f32, @floatCast(dsb.bias));
                }
            }
            try sme.packQuantAffinePanel64(dt, g_codes, g_sb, inter, h, gate_p);
            try sme.packQuantAffinePanel64(dt, u_codes, u_sb, inter, h, up_p);
            try sme.packQuantAffinePanel64(dt, d_codes, d_sb, h, inter, down_p);
        }

        const up_scratch = try gpa.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), m * inter);
        defer gpa.free(up_scratch);
        const inter_u16 = try gpa.alignedAlloc(u16, .fromByteUnits(sme.SVL_BYTES), m * inter);
        defer gpa.free(inter_u16);
        const out_f32 = try gpa.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), m * h);
        defer gpa.free(out_f32);

        try Qwen3Model.evalSwiGluExpertSme2(
            &disp,
            dt,
            .affine_scale_bias,
            .non_temporal,
            gate_p,
            up_p,
            down_p,
            x_u16,
            up_scratch,
            inter_u16,
            out_f32,
            resid_f32,
            m,
            h,
            inter,
        );

        const ref_up = try gpa.alloc(f32, m * inter);
        defer gpa.free(ref_up);
        const ref_act = try gpa.alloc(f32, m * inter);
        defer gpa.free(ref_act);
        const ref_out = try gpa.alloc(f32, m * h);
        defer gpa.free(ref_out);
        sme.matmulRowMajorRefF32(x_f32, up_ref, null, .none, ref_up, m, inter, h);
        sme.matmulRowMajorRefF32(x_f32, gate_ref, ref_up, .swiglu_gate_up, ref_act, m, inter, h);
        for (ref_act) |*v| v.* = quant.floatBitsToF32(act_fdt, quant.f32ToFloatBits(act_fdt, v.*));
        sme.matmulRowMajorRefF32(ref_act, down_ref, resid_f32, .add_residual, ref_out, m, h, inter);

        for (out_f32, ref_out) |got, exp| {
            try std.testing.expectApproxEqAbs(exp, got, 1e-2);
        }
    }
}

test "M4.5 D11 sampleFromLastLogits uses Secure Enclave TRNG and replays bit-exact with explicit seed" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
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
    const img = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .bf16,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
    }, 0xD110);
    defer gpa.free(img);

    var model = try Qwen3Model.loadFromBytes(gpa, img, cfg, .{ .max_seq_len = 16, .chunk_tokens = 16 });
    defer model.deinit(gpa);

    _ = try model.stepDecode(5, 0);
    const p_res = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(p_res);

    var trng_a = try rng_mod.RequestRng.init(null);
    const trng_b = try rng_mod.RequestRng.init(null);
    try std.testing.expectEqual(rng_mod.SeedSource.hardware_trng, trng_a.seedSource());
    try std.testing.expect(trng_a.seed_used != trng_b.seed_used);

    var seq_a: [8]u32 = undefined;
    for (&seq_a) |*t| t.* = try model.sampleFromLastLogits(0.8, trng_a.random(), p_res);

    var replay = try rng_mod.RequestRng.init(trng_a.seed_used);
    try std.testing.expectEqual(rng_mod.SeedSource.explicit, replay.seedSource());
    var seq_replay: [8]u32 = undefined;
    for (&seq_replay) |*t| t.* = try model.sampleFromLastLogits(0.8, replay.random(), p_res);
    try std.testing.expectEqualSlices(u32, &seq_a, &seq_replay);
}

test "M5 G13 Qwen3Model heterogeneous per-expert MoE ({q2,q3,q4,f16,bf16}) matches CPU fp32 ref on decode and verifyTree" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q2,
        .group_size = 64,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 10,
        .num_experts_per_tok = 4,
        .norm_topk_prob = true,
        .tie_word_embeddings = false,
    };
    const all5 = [_]quant.WeightDType{ .q2_affine, .q3_affine, .q4_affine, .f16, .bf16 };
    const img = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .q4_affine,
        .embed_dtype = .q4_affine,
        .norm_dtype = .f16,
        .router_dtype = .q2_affine,
        .expert_dtypes_cycle = &all5,
    }, 0x5150_A115);
    defer gpa.free(img);

    var model = try Qwen3Model.loadFromBytes(gpa, img, cfg, .{
        .max_seq_len = 32,
        .chunk_tokens = 16,
        .max_tree_nodes = 8,
    });
    defer model.deinit(gpa);

    // Static metadata verification for SpecEngine D7 pruning.
    try std.testing.expect(model.isHeterogeneousMoe());
    try std.testing.expectEqual(@as(usize, 20), model.expertBytesFlatU32().len);
    for (0..cfg.num_hidden_layers) |l| {
        for (0..cfg.num_experts) |e| {
            const exp_dt = all5[(l + e) % all5.len];
            try std.testing.expectEqual(exp_dt, model.expertDType(l, e));
            const exp_b = try moe.singleExpertBytesForDType(exp_dt, cfg.hidden_size, cfg.moe_intermediate_size);
            try std.testing.expectEqual(exp_b, model.expertBytes(l, e));
            try std.testing.expectEqual(@as(u32, @intCast(exp_b)), model.expertBytesFlatU32()[l * cfg.num_experts + e]);
        }
    }

    var cpu = try CpuTransformerRef.init(gpa, img, cfg, 32);
    defer cpu.deinit(gpa);

    // 1. Multi-step decode parity vs CPU fp32 reference.
    const prompt = [_]u32{ 3, 17, 42, 9 };
    for (prompt, 0..) |tok, pos| {
        const gpu_next = try model.stepDecode(tok, @intCast(pos));
        const cpu_next = try cpu.step(gpa, tok, @intCast(pos));
        try std.testing.expectEqual(cpu_next, gpu_next);
        const gpu_logits = model.readLogitsF16();
        for (cpu.logits, 0..) |exp, i| {
            const got: f32 = @floatCast(gpu_logits[i]);
            try std.testing.expectApproxEqAbs(exp, got, @max(0.06, @abs(exp) * 0.04));
        }
    }

    // 2. Speculative tree verifyTree parity (4-node chain) vs sequential CPU ref.
    const tree_toks = [_]u32{ 11, 25, 61, 88 };
    const tree_pos = [_]u32{ 4, 5, 6, 7 };
    const tree_masks = [_]u64{ 0b0001, 0b0011, 0b0111, 0b1111 };
    const vr = try model.verifyTree(&tree_toks, &tree_pos, &tree_masks, 4, true);
    for (tree_toks, 0..) |tok, i| {
        const cpu_next = try cpu.step(gpa, tok, tree_pos[i]);
        try std.testing.expectEqual(cpu_next, vr.argmax_tokens[i]);
    }
}

test "M5 ADR-0017 rejects intra-expert dtype mismatch and preserves uniform MoE fast path" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q2,
        .group_size = 64,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .norm_topk_prob = true,
        .tie_word_embeddings = true,
    };

    // 1. Mismatched up_proj inside expert 0 MUST fail with error.UnexpectedTensorDType.
    const bad_img = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .q3_affine,
        .embed_dtype = .q4_affine,
        .expert_dtype = .q2_affine,
        .mismatch_expert0_up = true,
    }, 0xBAD_E001);
    defer gpa.free(bad_img);
    try std.testing.expectError(
        error.UnexpectedTensorDType,
        Qwen3Model.loadFromBytes(gpa, bad_img, cfg, .{ .max_seq_len = 16, .chunk_tokens = 16 }),
    );

    // 2. Uniform q2_affine MoE keeps `isHeterogeneousMoe() == false` (untouched fast path) and matches CPU ref.
    const uni_img = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .q2_affine,
        .embed_dtype = .q2_affine,
        .router_dtype = .q2_affine,
        .expert_dtype = .q2_affine,
    }, 0x0002_0002);
    defer gpa.free(uni_img);

    var uni_model = try Qwen3Model.loadFromBytes(gpa, uni_img, cfg, .{ .max_seq_len = 16, .chunk_tokens = 16 });
    defer uni_model.deinit(gpa);
    try std.testing.expect(!uni_model.isHeterogeneousMoe());

    var uni_cpu = try CpuTransformerRef.init(gpa, uni_img, cfg, 16);
    defer uni_cpu.deinit(gpa);
    for ([_]u32{ 5, 29, 77 }, 0..) |tok, pos| {
        const g = try uni_model.stepDecode(tok, @intCast(pos));
        const c = try uni_cpu.step(gpa, tok, @intCast(pos));
        try std.testing.expectEqual(c, g);
    }
}

test "M5 G13 mixed per-layer dense Qwen3 ({q2,q3,q4,bf16}) matches CPU fp32 ref" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = .q2,
        .group_size = 64,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 4,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 192,
        .tie_word_embeddings = false,
    };
    const layer_dts = [_]quant.WeightDType{ .bf16, .q4_affine, .q3_affine, .q2_affine };
    const img = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .q4_affine,
        .embed_dtype = .q4_affine,
        .norm_dtype = .bf16,
        .layer_linear_dtypes_cycle = &layer_dts,
    }, 0xDE11_5E05);
    defer gpa.free(img);

    var model = try Qwen3Model.loadFromBytes(gpa, img, cfg, .{
        .max_seq_len = 16,
        .chunk_tokens = 16,
        .max_tree_nodes = 4,
    });
    defer model.deinit(gpa);
    try std.testing.expect(!model.isHeterogeneousMoe());

    var cpu = try CpuTransformerRef.init(gpa, img, cfg, 16);
    defer cpu.deinit(gpa);
    for ([_]u32{ 2, 19, 63 }, 0..) |tok, pos| {
        const g = try model.stepDecode(tok, @intCast(pos));
        const c = try cpu.step(gpa, tok, @intCast(pos));
        try std.testing.expectEqual(c, g);
        const gpu_logits = model.readLogitsF16();
        for (cpu.logits, 0..) |exp, i| {
            const got: f32 = @floatCast(gpu_logits[i]);
            try std.testing.expectApproxEqAbs(exp, got, @max(0.06, @abs(exp) * 0.04));
        }
    }
}

test "M5 pure-CPU SME2 stepDraftTreeBatch and prefillChunk match GPU row_major parity" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = .q4,
        .group_size = 64,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 192,
        .tie_word_embeddings = true,
    };
    const layer_dts = [_]quant.WeightDType{ .q4_affine, .q3_affine };
    const row_img = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .q4_affine,
        .embed_dtype = .q4_affine,
        .norm_dtype = .bf16,
        .layer_linear_dtypes_cycle = &layer_dts,
    }, 0x5EE2_D801);
    defer gpa.free(row_img);

    const sme_img = try repackAmfBytesToSme2(gpa, row_img, .all_linear, null);
    defer gpa.free(sme_img);

    var gpu_m = try Qwen3Model.loadFromBytes(gpa, row_img, cfg, .{
        .max_seq_len = 32,
        .chunk_tokens = 8,
        .max_tree_nodes = 16,
    });
    defer gpu_m.deinit(gpa);
    try std.testing.expect(!gpu_m.hasPureCpuSme2Draft());

    var sme_m = try Qwen3Model.loadFromBytes(gpa, sme_img, cfg, .{
        .max_seq_len = 32,
        .chunk_tokens = 8,
        .max_tree_nodes = 16,
    });
    defer sme_m.deinit(gpa);
    try std.testing.expect(sme_m.hasPureCpuSme2Draft());

    // 1. Prefill chunk (across chunk boundary `chunk_tokens = 8`) must match top-1 token.
    const prompt = [_]u32{ 3, 17, 42, 9, 61, 88, 12, 5, 77 };
    const gpu_p0 = try gpu_m.prefillChunk(prompt[0..6], 0, false);
    const sme_p0 = try sme_m.prefillChunk(prompt[0..6], 0, false);
    try std.testing.expectEqual(gpu_p0, sme_p0);
    const gpu_p1 = try gpu_m.prefillChunk(prompt[6..9], 6, true);
    const sme_p1 = try sme_m.prefillChunk(prompt[6..9], 6, true);
    try std.testing.expectEqual(gpu_p1, sme_p1);
    try std.testing.expectEqual(gpu_m.seq_len, sme_m.seq_len);

    // 2. Tree-draft batch (3 nodes across 2 branches, `logits_last_row_only = false`, `commit_kv = false`).
    const toks = [_]u32{ gpu_p1, 21, 34 };
    const pos = [_]u32{ 9, 10, 10 };
    const masks = [_]u64{ 0b001, 0b011, 0b101 };
    const gpu_cands = try gpu_m.stepDraftTreeBatch(&toks, &pos, 0, &masks, 9, 3, false, false, null);
    const sme_cands = try sme_m.stepDraftTreeBatch(&toks, &pos, 0, &masks, 9, 3, false, false, null);
    try std.testing.expectEqual(gpu_cands.len, sme_cands.len);
    for (0..toks.len) |r| {
        try std.testing.expectEqual(gpu_cands[r * 3].token_id, sme_cands[r * 3].token_id);
        try std.testing.expectApproxEqAbs(gpu_cands[r * 3].prob, sme_cands[r * 3].prob, 0.03);
    }
}

test "M6.5 R5 Qwen3Model tiered q8 retired KV recycles IOSurface tiles and matches fp16 decode/tree-verify" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = null,
        .group_size = 64,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 192,
        .tie_word_embeddings = true,
    };
    const img = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .f16,
        .embed_dtype = .f16,
        .norm_dtype = .f16,
    }, 0xA65_0001);
    defer gpa.free(img);

    // Reference: pure fp16 IOSurface across 4 chunks (chunk_tokens = 8, max_seq_len = 32).
    var fp16_m = try Qwen3Model.loadFromBytes(gpa, img, cfg, .{
        .max_seq_len = 32,
        .chunk_tokens = 8,
        .max_tree_nodes = 8,
        .kv_retire_q8_after_chunks = 0,
    });
    defer fp16_m.deinit(gpa);

    // Tiered R5: keep at most 1 active chunk, retire completed chunks to group-64 q8.
    var tiered_m = try Qwen3Model.loadFromBytes(gpa, img, cfg, .{
        .max_seq_len = 32,
        .chunk_tokens = 8,
        .max_tree_nodes = 8,
        .kv_retire_q8_after_chunks = 1,
    });
    defer tiered_m.deinit(gpa);

    try std.testing.expect(tiered_m.kvRetireEnabled());
    try std.testing.expect(tiered_m.activeKvWiredBytes() < fp16_m.activeKvWiredBytes());
    try std.testing.expect(tiered_m.retiredKvWiredBytes() > 0);

    // Prefill 18 tokens (crosses 2 full 8-token chunks -> 2 chunks retired!).
    var prompt: [18]u32 = undefined;
    for (&prompt, 0..) |*p, i| p.* = @intCast((i * 11 + 5) % cfg.vocab_size);

    const fp16_first = try fp16_m.prefillPrompt(&prompt, true);
    const tiered_first = try tiered_m.prefillPrompt(&prompt, true);
    try std.testing.expectEqual(@as(u32, 2), tiered_m.retiredKvChunks());
    try std.testing.expectEqual(@as(u64, 2), tiered_m.kvRetireEvents());
    try std.testing.expectEqual(fp16_first, tiered_first);

    // Decode 8 more tokens crossing chunk 2 -> chunk 3 retirement with zero steady-state heap allocs.
    var tok_f = fp16_first;
    var tok_t = tiered_first;
    // Warm up 2 decode steps so both Metal 4 command-buffer ring frames cache their encoder state.
    for (0..2) |w| {
        const pos: u32 = @intCast(prompt.len + w);
        tok_f = try fp16_m.stepDecode(tok_f, pos);
        tok_t = try tiered_m.stepDecode(tok_t, pos);
        try std.testing.expectEqual(tok_f, tok_t);
    }
    const snap_before = try vmstat.Snapshot.capture();
    for (2..8) |step| {
        const pos: u32 = @intCast(prompt.len + step);
        tok_f = try fp16_m.stepDecode(tok_f, pos);
        tok_t = try tiered_m.stepDecode(tok_t, pos);
        try std.testing.expectEqual(tok_f, tok_t);
        const lf = fp16_m.readLastLogitsF16();
        const lt = tiered_m.readLastLogitsF16();
        var max_diff: f32 = 0.0;
        for (lf, lt) |a, b| {
            max_diff = @max(max_diff, @abs(@as(f32, @floatCast(a)) - @as(f32, @floatCast(b))));
        }
        try std.testing.expect(max_diff < 0.05);
    }
    const snap_after = try vmstat.Snapshot.capture();
    try std.testing.expectEqual(@as(u32, 3), tiered_m.retiredKvChunks());
    try vmstat.checkNoSwap(snap_before, snap_after);

    // Verify speculative tree verification over 3 retired chunks + active chunk!
    const tree_toks = [_]u32{ tok_t, 7, 19, 31 };
    const tree_pos = [_]u32{ 26, 27, 27, 28 };
    const tree_masks = [_]u64{ 0b0001, 0b0011, 0b0101, 0b1011 };
    const vr_f = try fp16_m.verifyTree(&tree_toks, &tree_pos, &tree_masks, 26, true);
    const vr_t = try tiered_m.verifyTree(&tree_toks, &tree_pos, &tree_masks, 26, true);
    for (0..tree_toks.len) |i| {
        try std.testing.expectEqual(vr_f.argmax_tokens[i], vr_t.argmax_tokens[i]);
    }

    // Reset must restore clean retired pool and allow fresh reuse.
    tiered_m.resetKvCache();
    try std.testing.expectEqual(@as(u32, 0), tiered_m.retiredKvChunks());
    const reuse_tok = try tiered_m.prefillPrompt(prompt[0..10], true);
    try std.testing.expectEqual(@as(u32, 1), tiered_m.retiredKvChunks());
    try std.testing.expect(reuse_tok < cfg.vocab_size);
}

test "M6.5 C14+R8 Qwen3BatchRuntime continuous batching + PrefixCache CoW + D11 determinism" {
    const gpa = std.testing.allocator;
    const cfg: ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q4,
        .group_size = 64,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .norm_topk_prob = true,
        .tie_word_embeddings = false,
    };
    const img = try buildSyntheticAmfImageWithSpec(gpa, cfg, .{
        .linear_dtype = .q4_affine,
        .embed_dtype = .f16,
        .norm_dtype = .f16,
    }, 0xC14_8801);
    defer gpa.free(img);

    var serial_m = try Qwen3Model.loadFromBytes(gpa, img, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 8,
        .max_tree_nodes = 16,
    });
    defer serial_m.deinit(gpa);

    var batch_m = try Qwen3Model.loadFromBytes(gpa, img, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 8,
        .max_tree_nodes = 16,
    });
    defer batch_m.deinit(gpa);

    var rt = try Qwen3BatchRuntime.init(gpa, &batch_m, .{
        .max_slots = 4,
        .prefix_cache_chunks = 16,
    });
    defer rt.deinit(gpa);

    // 4 prompts sharing a 12-token prefix (1 full 8-tok chunk + 4-tok CoW tail) + distinct suffixes.
    const shared_prefix = [_]u32{ 3, 17, 42, 9, 61, 88, 12, 5, 77, 23, 54, 11 };
    const p0 = shared_prefix ++ [_]u32{ 1, 2, 3 };
    const p1 = shared_prefix ++ [_]u32{ 4, 5, 6 };
    const p2 = shared_prefix ++ [_]u32{ 7, 8, 9, 10 };
    const p3 = shared_prefix ++ [_]u32{ 4, 5, 6 }; // identical to p1, tested with explicit D11 seed!
    const prompts = [_][]const u32{ &p0, &p1, &p2, &p3 };
    const gen_len: usize = 6;

    // Reference serial greedy outputs for p0, p1, p2 and seeded T=0.8 output for p3.
    var ref_out: [4][gen_len]u32 = undefined;
    const p_res = try gpa.alloc(f32, cfg.vocab_size);
    defer gpa.free(p_res);
    for (prompts, 0..) |pr, idx| {
        serial_m.resetKvCache();
        var rng_s = rng_mod.RequestRng.initDeterministic(0xD11_BEEF);
        var tok = try serial_m.prefillPrompt(pr, true);
        if (idx == 3) {
            tok = try serial_m.sampleFromLastLogits(0.8, rng_s.random(), p_res);
        }
        ref_out[idx][0] = tok;
        for (1..gen_len) |step| {
            tok = try serial_m.stepDecode(tok, @intCast(pr.len + step - 1));
            if (idx == 3) {
                tok = try serial_m.sampleFromLastLogits(0.8, rng_s.random(), p_res);
            }
            ref_out[idx][step] = tok;
        }
    }

    // Run continuous batching: warm prefix with p0 first, then submit p1, p2, p3 concurrently!
    var sched = try batch_sched.BatchScheduler.init(gpa, .{
        .max_batch_size = 4,
        .max_prefill_chunk_tokens = 32,
        .max_prompt_tokens = 64,
    });
    defer sched.deinit(gpa);

    const Collector = struct {
        tokens: [4][gen_len]u32 = @splat(@splat(0)),
        counts: [4]usize = @splat(0),
        prefix_hits: [4]u32 = @splat(0),
        finished: [4]bool = @splat(false),

        fn onPrefillDone(ctx: *anyopaque, _: u16, tag: u64, _: u32, hit: u32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.prefix_hits[@intCast(tag)] = hit;
            return true;
        }
        fn onToken(ctx: *anyopaque, _: u16, tag: u64, tid: u32, _: u32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const i: usize = @intCast(tag);
            if (self.counts[i] < gen_len) {
                self.tokens[i][self.counts[i]] = tid;
                self.counts[i] += 1;
            }
            return true;
        }
        fn onFinish(ctx: *anyopaque, _: u16, tag: u64, _: batch_sched.FinishReason, _: u32, _: u32) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.finished[@intCast(tag)] = true;
        }
    };
    var col: Collector = .{};
    const obs: batch_sched.StepObserver = .{
        .ctx = &col,
        .onPrefillDoneFn = Collector.onPrefillDone,
        .onTokenFn = Collector.onToken,
        .onFinishFn = Collector.onFinish,
    };
    const backend = rt.batchBackend();

    // Cold request 0 populates PrefixCache (block 0 + partial tail).
    _ = try sched.enqueue(prompts[0], .{ .request_id = 1, .user_tag = 0, .max_tokens = gen_len, .ignore_eos = true });
    while (sched.activeSlotCount() > 0) {
        _ = sched.step(backend, obs);
    }
    try std.testing.expectEqual(@as(u32, 0), col.prefix_hits[0]);
    try std.testing.expectEqualSlices(u32, &ref_out[0], &col.tokens[0]);

    // Warm concurrent batch of 3 requests (p1 greedy, p2 greedy, p3 T=0.8 seed=0xD11_BEEF).
    _ = try sched.enqueue(prompts[1], .{ .request_id = 2, .user_tag = 1, .max_tokens = gen_len, .ignore_eos = true });
    _ = try sched.enqueue(prompts[2], .{ .request_id = 3, .user_tag = 2, .max_tokens = gen_len, .ignore_eos = true });
    _ = try sched.enqueue(prompts[3], .{
        .request_id = 4,
        .user_tag = 3,
        .max_tokens = gen_len,
        .temperature = 0.8,
        .explicit_seed = 0xD11_BEEF,
        .ignore_eos = true,
    });

    // Warm first batch step, then verify zero steady-state heap allocations over remaining decode steps.
    _ = sched.step(backend, obs);
    const snap_before = try vmstat.Snapshot.capture();
    while (sched.activeSlotCount() > 0) {
        _ = sched.step(backend, obs);
    }
    const snap_after = try vmstat.Snapshot.capture();

    try vmstat.checkNoSwap(snap_before, snap_after);
    try std.testing.expect(sched.telemetry().max_observed_batch_size >= 3);
    // All 3 warm requests must hit >= 12 shared prefix tokens (8 zero-copy + 4 CoW tail);
    // p1 and p3 share all 15 prompt tokens, so whichever prefills second hits 14 tokens!
    try std.testing.expectEqual(@as(u32, 12), col.prefix_hits[2]);
    const p1_p3_match = (col.prefix_hits[1] == 12 and col.prefix_hits[3] == 14) or
        (col.prefix_hits[1] == 14 and col.prefix_hits[3] == 12);
    try std.testing.expect(p1_p3_match);

    // Bit-exact parity for greedy slots AND D11 seeded T=0.8 slot under concurrent batching!
    try std.testing.expectEqualSlices(u32, &ref_out[1], &col.tokens[1]);
    try std.testing.expectEqualSlices(u32, &ref_out[2], &col.tokens[2]);
    try std.testing.expectEqualSlices(u32, &ref_out[3], &col.tokens[3]);

    // Every slot released cleanly: 0 pinned PrefixCache entries remain!
    const st = rt.stats();
    try std.testing.expectEqual(@as(u32, 0), st.prefix_cache_pinned_entries);
    try std.testing.expect(st.prefix_hits >= 3);
    try std.testing.expect(st.prefix_cow_copies >= 3);
}
