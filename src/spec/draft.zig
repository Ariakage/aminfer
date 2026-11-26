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
const quant_formats = @import("../gpu/quant_formats.zig");
const attention = @import("../gpu/attention.zig");
const moe = @import("../gpu/moe.zig");
const qwen3 = @import("../model/qwen3.zig");
const graph = @import("../model/graph.zig");
const sme = @import("../cpu/sme.zig");
const vmstat = @import("../platform/vmstat.zig");

/// Maximum top-K candidates extracted per draft node via `encodeTopKLogitsRows`.
pub const MAX_DRAFT_TOP_K: u32 = moe.MAX_TOP_K;

/// Maximum independent chained native MTP predictor stages (`D15`, MiMo-V2.6 `3L` MTP).
pub const MAX_MTP_STAGES: usize = 8;

/// Maximum rows batched through the optional CoreML ANE `innerProduct` projection.
pub const ANE_MAX_BATCH_ROWS: u32 = 8;

/// L1/L2 speculative draft source selection (`ADR-0013`, `ADR-0015`, `ADR-0019`).
pub const DraftMode = enum {
    l1_only,
    eagle3,
    eagle2,
    mtp,
    small_draft,
    self_recurrent,
    dspark,
    dflare,
    dfly2,

    pub fn label(self: DraftMode) []const u8 {
        return switch (self) {
            .l1_only => "l1",
            .eagle3 => "eagle3",
            .eagle2 => "eagle2",
            .mtp => "mtp",
            .small_draft => "small",
            .self_recurrent => "self-recurrent",
            .dspark => "dspark",
            .dflare => "dflare",
            .dfly2 => "dfly2",
        };
    }

    pub fn parse(str: []const u8) error{InvalidDraftMode}!DraftMode {
        if (std.mem.eql(u8, str, "l1") or std.mem.eql(u8, str, "l1_only")) return .l1_only;
        if (std.mem.eql(u8, str, "eagle3")) return .eagle3;
        if (std.mem.eql(u8, str, "eagle2")) return .eagle2;
        if (std.mem.eql(u8, str, "mtp")) return .mtp;
        if (std.mem.eql(u8, str, "small") or std.mem.eql(u8, str, "small_draft")) return .small_draft;
        if (std.mem.eql(u8, str, "self-recurrent") or std.mem.eql(u8, str, "self_recurrent")) return .self_recurrent;
        if (std.mem.eql(u8, str, "dspark")) return .dspark;
        if (std.mem.eql(u8, str, "dflare") or std.mem.eql(u8, str, "dflash")) return .dflare;
        if (std.mem.eql(u8, str, "dfly2") or std.mem.eql(u8, str, "dfly")) return .dfly2;
        return error.InvalidDraftMode;
    }
};

/// L2 draft head hardware placement policy (`D5`, `D13`, ADR-0008, ADR-0019).
pub const Placement = enum {
    auto,
    gpu,
    ane,
    cpu_sme2,

    pub fn label(self: Placement) []const u8 {
        return switch (self) {
            .auto => "auto",
            .gpu => "gpu",
            .ane => "ane",
            .cpu_sme2 => "sme2",
        };
    }

    pub fn parse(str: []const u8) error{InvalidPlacement}!Placement {
        if (std.mem.eql(u8, str, "auto")) return .auto;
        if (std.mem.eql(u8, str, "gpu")) return .gpu;
        if (std.mem.eql(u8, str, "ane")) return .ane;
        if (std.mem.eql(u8, str, "sme2") or std.mem.eql(u8, str, "cpu_sme2")) return .cpu_sme2;
        return error.InvalidPlacement;
    }
};

/// Native MTP layer architecture family (`D15`).
pub const MtpArch = enum {
    standard,
    glm_moe_dsa,
    glm5_next,

    pub fn parse(s: []const u8) ?MtpArch {
        if (std.mem.eql(u8, s, "standard")) return .standard;
        if (std.mem.eql(u8, s, "glm_moe_dsa")) return .glm_moe_dsa;
        if (std.mem.eql(u8, s, "glm5_next")) return .glm5_next;
        return null;
    }
};

/// Shared sparse KV indices handed from a DSA target backbone (`GraphModel.sharedMtpSparseIndices`)
/// or an explicit caller/test into `DraftHead` for zero-reindex `1L` `MTP` (`index_share_for_mtp_iteration = true`).
pub const SharedSparseIndices = struct {
    gpu_buf: ?metal.BufferBinding = null,
    host_indices: []const u32 = &.{},
    k_selected: u32 = 0,
    max_k: u32 = 0,
    num_rows: u32 = 1,
    valid: bool = false,
};

/// Parsed L2 draft head configuration supporting real HuggingFace / vLLM
/// `Eagle3DraftModel` (`RedHatAI/Qwen3-30B-A3B-speculator.eagle3`), `GLM-5.2/5.3`
/// (`glm_moe_dsa`) and `GLM-5.3-Flash` (`glm5_next`) `1L` `MLA`/`NoPE-MLA` `MTP`,
/// as well as synthetic `eagle3`, `eagle2`, and `mtp` bundles from `aminfer-tools synth-draft`.
pub const DraftConfig = struct {
    format: DraftMode,
    mtp_arch: MtpArch = .standard,
    vocab_size: u32,
    draft_vocab_size: u32,
    hidden_size: u32,
    target_hidden_size: u32,
    num_hidden_layers: u32 = 1,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    /// Optional asymmetric value head dim (`0` => equals `head_dim`; `128` with `head_dim = 192` on MiMo-V2.6 MTP; `256` on GLM-5.x MLA MTP).
    v_head_dim: u32 = 0,
    /// Optional partial NeoX RoPE dim (`0` => equals `head_dim` for GQA; `64` on MiMo-V2.6 MTP).
    rope_dim: u32 = 0,
    /// Optional MLA low-rank projections (`GLM-5.2/5.3` `q_lora=2048, kv_lora=512, qk_nope=192, qk_rope=64`;
    /// `GLM-5.3-Flash` NoPE-MLA `q_lora=1536, kv_lora=512, qk_nope=256, qk_rope=0`).
    q_lora_rank: u32 = 0,
    kv_lora_rank: u32 = 0,
    qk_nope_head_dim: u32 = 0,
    qk_rope_head_dim: u32 = 0,
    /// DeepSeek/GLM interleaved RoPE pairs `(2*i, 2*i+1)` (`true` on `GLM-5.2/5.3`).
    rope_interleave: bool = false,
    /// Zero-reindex MTP fast path: reuse target backbone's shared sparse `topk` indices without running an MTP DSA indexer.
    index_share_for_mtp_iteration: bool = false,
    /// Sparse DSA `index_topk` cap (`2048` on `GLM-5.2/5.3` / `GLM-5.3-Flash`).
    index_topk: u32 = 0,
    /// Optional MoE inside MTP layer (`256` on `GLM-5.2/5.3`, `288` on `GLM-5.3-Flash`; `0` => dense MLP).
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    routed_scaling_factor: f32 = 1.0,
    norm_topk_prob: bool = true,
    moe_scoring_sigmoid: bool = false,
    /// Optional SwiGLU activation clamp (`10.0` on `GLM-5.3-Flash` `glm5_next`; `0.0` disables clamping).
    swiglu_limit: f32 = 0.0,
    /// Optional causal sliding window (`0` => full causal; `128` on MiMo-V2.6 MTP).
    sliding_window: u32 = 0,
    /// Optional attention value scale (`1.0` default; `0.707` Flash / `0.612` Pro on MiMo-V2.6 MTP).
    attention_value_scale: f32 = 1.0,
    /// Whether SWA layers include per-head `attention_sink_bias` softmax column (`MiMo-V2.6` MTP).
    add_swa_attention_sink_bias: bool = false,
    /// Chained multi-stage native MTP (`D15`, `num_nextn_predict_layers = 3`): each stage `k = 0..num_hidden_layers-1`
    /// has its own `enorm`/`hnorm`/`eh_proj`/transformer/KV/final_layernorm predicting depth `k + 1`.
    mtp_chained: bool = false,
    /// When `mtp_chained == true` and requested draft depth exceeds `num_hidden_layers`,
    /// cycle the last MTP stage (`false` = cleanly stop at `num_hidden_layers`).
    mtp_cycle_last_stage: bool = false,
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

    pub fn isMlaMtp(self: DraftConfig) bool {
        return self.format == .mtp and self.kv_lora_rank > 0;
    }

    pub fn isMoeMtp(self: DraftConfig) bool {
        return self.format == .mtp and self.num_experts > 0;
    }

    pub fn effectiveQkHeadDim(self: DraftConfig) u32 {
        return if (self.isMlaMtp()) (self.qk_nope_head_dim + self.qk_rope_head_dim) else self.head_dim;
    }

    pub fn effectiveVHeadDim(self: DraftConfig) u32 {
        return if (self.v_head_dim > 0) self.v_head_dim else self.effectiveQkHeadDim();
    }

    pub fn effectiveRopeDim(self: DraftConfig) u32 {
        if (self.isMlaMtp()) return self.qk_rope_head_dim;
        return if (self.rope_dim > 0) self.rope_dim else self.head_dim;
    }

    pub fn effectiveKvHeads(self: DraftConfig) u32 {
        return if (self.isMlaMtp()) self.num_attention_heads else self.num_key_value_heads;
    }

    pub fn qDim(self: DraftConfig) u32 {
        return self.num_attention_heads * self.effectiveQkHeadDim();
    }

    pub fn kvDim(self: DraftConfig) u32 {
        return self.effectiveKvHeads() * self.effectiveQkHeadDim();
    }

    pub fn vDim(self: DraftConfig) u32 {
        return self.effectiveKvHeads() * self.effectiveVHeadDim();
    }

    pub fn oInDim(self: DraftConfig) u32 {
        return self.num_attention_heads * self.effectiveVHeadDim();
    }

    pub fn effectiveMlpInter(self: DraftConfig) u32 {
        if (self.isMoeMtp()) {
            return if (self.shared_expert_intermediate_size > 0) self.shared_expert_intermediate_size else self.moe_intermediate_size;
        }
        return self.intermediate_size;
    }

    pub fn usesCustomMtpAttn(self: DraftConfig) bool {
        return self.isMlaMtp() or
            self.isMoeMtp() or
            self.index_share_for_mtp_iteration or
            self.swiglu_limit > 0.0 or
            self.effectiveVHeadDim() != self.effectiveQkHeadDim() or
            self.effectiveRopeDim() != self.effectiveQkHeadDim() or
            self.effectiveQkHeadDim() > 128 or
            self.sliding_window > 0 or
            self.attention_value_scale != 1.0 or
            self.add_swa_attention_sink_bias;
    }

    pub fn validate(self: DraftConfig) ConfigError!void {
        if (self.format == .l1_only or self.format == .small_draft or self.format == .self_recurrent or self.format == .dspark or self.format == .dflare or self.format == .dfly2) return error.UnsupportedFormat;
        if (self.vocab_size == 0 or self.draft_vocab_size == 0 or self.draft_vocab_size > self.vocab_size) {
            return error.InvalidConfig;
        }
        if (self.format != .eagle3 and self.draft_vocab_size != self.vocab_size) {
            return error.InvalidConfig;
        }
        const eff_inter = self.effectiveMlpInter();
        if (self.hidden_size == 0 or self.target_hidden_size == 0 or self.num_hidden_layers == 0 or
            self.num_attention_heads == 0 or self.num_key_value_heads == 0 or
            eff_inter == 0 or self.max_position_embeddings == 0)
        {
            return error.InvalidConfig;
        }
        if (self.mtp_chained) {
            if (self.format != .mtp or self.num_hidden_layers > MAX_MTP_STAGES) {
                return error.InvalidConfig;
            }
        }
        const qkd = self.effectiveQkHeadDim();
        const vhd = self.effectiveVHeadDim();
        const rdim = self.effectiveRopeDim();
        if (self.isMlaMtp()) {
            if (self.q_lora_rank == 0 or (self.q_lora_rank % 4) != 0) return error.InvalidConfig;
            if (self.kv_lora_rank == 0 or (self.kv_lora_rank % 4) != 0) return error.InvalidConfig;
            if (self.qk_nope_head_dim == 0 or (self.qk_nope_head_dim % 4) != 0) return error.InvalidConfig;
            if (self.qk_rope_head_dim > 128 or (self.qk_rope_head_dim % 2) != 0) return error.InvalidConfig;
            if (qkd == 0 or qkd > 256 or (qkd % 4) != 0) return error.InvalidConfig;
            if (vhd == 0 or vhd > 256 or (vhd % 4) != 0) return error.InvalidConfig;
        } else if (self.usesCustomMtpAttn() or self.mtp_chained) {
            if (self.head_dim == 0 or self.head_dim > 256 or (self.head_dim % 4) != 0) return error.InvalidConfig;
            if (vhd == 0 or vhd > 256 or (vhd % 4) != 0) return error.InvalidConfig;
            if (rdim == 0 or rdim > self.head_dim or (rdim % 2) != 0) return error.InvalidConfig;
        } else {
            if (self.head_dim != 64 and self.head_dim != 128) return error.InvalidConfig;
        }
        if (self.isMoeMtp()) {
            if (self.num_experts > 512) return error.InvalidConfig;
            if (self.num_experts_per_tok == 0 or self.num_experts_per_tok > 16 or self.num_experts_per_tok > self.num_experts) {
                return error.InvalidConfig;
            }
            if (self.moe_intermediate_size == 0 or (self.moe_intermediate_size % 4) != 0) return error.InvalidConfig;
            if (self.shared_expert_intermediate_size > 0 and (self.shared_expert_intermediate_size % 4) != 0) return error.InvalidConfig;
            if (!std.math.isFinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0.0) return error.InvalidConfig;
        }
        if ((self.hidden_size % 4) != 0 or (self.target_hidden_size % 4) != 0 or (eff_inter % 4) != 0) {
            return error.InvalidConfig;
        }
        if ((self.num_attention_heads % self.effectiveKvHeads()) != 0) return error.InvalidConfig;
        if (self.rms_norm_eps <= 0.0) return error.InvalidConfig;
        if (!self.isMlaMtp() or self.qk_rope_head_dim > 0) {
            if (self.rope_theta <= 0.0) return error.InvalidConfig;
        }
        if (!std.math.isFinite(self.attention_value_scale) or self.attention_value_scale <= 0.0) return error.InvalidConfig;
        if (!std.math.isFinite(self.swiglu_limit) or self.swiglu_limit < 0.0) return error.InvalidConfig;
    }

    /// Parses an EAGLE-3, EAGLE-2, or MTP (1L or chained 3L) `config.json` document at startup.
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

        const mtp_arch: MtpArch = blk: {
            if (jsonOptString(obj, "mtp_arch")) |s| {
                if (MtpArch.parse(s)) |a| break :blk a;
            }
            if (jsonOptString(obj, "model_type")) |s| {
                if (MtpArch.parse(s)) |a| break :blk a;
            }
            break :blk .standard;
        };

        const vocab_size = jsonOptU32(layer_obj, "vocab_size") orelse jsonOptU32(obj, "vocab_size") orelse return error.InvalidJson;
        const draft_vocab_size = jsonOptU32(obj, "draft_vocab_size") orelse vocab_size;
        const hidden_size = jsonOptU32(layer_obj, "hidden_size") orelse jsonOptU32(obj, "hidden_size") orelse return error.InvalidJson;
        const target_hidden_size = jsonOptU32(obj, "target_hidden_size") orelse hidden_size;
        const num_layers = jsonOptU32(layer_obj, "num_hidden_layers") orelse
            jsonOptU32(obj, "num_hidden_layers") orelse
            jsonOptU32(obj, "num_nextn_predict_layers") orelse 1;
        const num_q_heads = jsonOptU32(layer_obj, "num_attention_heads") orelse
            jsonOptU32(obj, "num_attention_heads") orelse
            jsonOptU32(obj, "swa_num_attention_heads") orelse return error.InvalidJson;
        const num_kv_heads = jsonOptU32(layer_obj, "num_key_value_heads") orelse
            jsonOptU32(obj, "num_key_value_heads") orelse
            jsonOptU32(obj, "swa_num_key_value_heads") orelse num_q_heads;

        const q_lora_rank = jsonOptU32(layer_obj, "q_lora_rank") orelse jsonOptU32(obj, "q_lora_rank") orelse 0;
        const kv_lora_rank = jsonOptU32(layer_obj, "kv_lora_rank") orelse jsonOptU32(obj, "kv_lora_rank") orelse 0;
        const qk_nope_head_dim = jsonOptU32(layer_obj, "qk_nope_head_dim") orelse jsonOptU32(obj, "qk_nope_head_dim") orelse 0;
        const qk_rope_head_dim = jsonOptU32(layer_obj, "qk_rope_head_dim") orelse jsonOptU32(obj, "qk_rope_head_dim") orelse 0;
        const is_mla = (format == .mtp and kv_lora_rank > 0);

        const head_dim = jsonOptU32(layer_obj, "head_dim") orelse
            jsonOptU32(obj, "head_dim") orelse
            jsonOptU32(obj, "swa_head_dim") orelse blk: {
            if (is_mla and (qk_nope_head_dim + qk_rope_head_dim) > 0) {
                break :blk qk_nope_head_dim + qk_rope_head_dim;
            }
            if (num_q_heads == 0) return error.InvalidConfig;
            break :blk hidden_size / num_q_heads;
        };
        const v_head_dim = jsonOptU32(layer_obj, "v_head_dim") orelse
            jsonOptU32(obj, "v_head_dim") orelse
            jsonOptU32(obj, "swa_v_head_dim") orelse head_dim;
        const explicit_rope_dim = jsonOptU32(layer_obj, "rope_dim") orelse
            jsonOptU32(obj, "rope_dim") orelse
            jsonOptU32(layer_obj, "rotary_dim") orelse
            jsonOptU32(obj, "rotary_dim");
        const partial_rotary_factor = jsonOptF32(layer_obj, "partial_rotary_factor") orelse
            jsonOptF32(obj, "partial_rotary_factor") orelse 1.0;
        const rope_dim: u32 = if (is_mla)
            qk_rope_head_dim
        else if (explicit_rope_dim) |rd|
            rd
        else if (partial_rotary_factor > 0.0 and partial_rotary_factor < 1.0) blk: {
            const raw: u32 = @intFromFloat(@floor(@as(f32, @floatFromInt(head_dim)) * partial_rotary_factor));
            break :blk raw & ~@as(u32, 1);
        } else head_dim;

        const rope_interleave = jsonOptBool(layer_obj, "rope_interleave") orelse
            jsonOptBool(obj, "rope_interleave") orelse (mtp_arch == .glm_moe_dsa);
        const index_share_mtp = jsonOptBool(layer_obj, "index_share_for_mtp_iteration") orelse
            jsonOptBool(obj, "index_share_for_mtp_iteration") orelse false;
        const index_topk = jsonOptU32(layer_obj, "index_topk") orelse jsonOptU32(obj, "index_topk") orelse 0;

        const num_experts = jsonOptU32(layer_obj, "num_experts") orelse
            jsonOptU32(obj, "num_experts") orelse
            jsonOptU32(layer_obj, "n_routed_experts") orelse
            jsonOptU32(obj, "n_routed_experts") orelse 0;
        const num_experts_per_tok = jsonOptU32(layer_obj, "num_experts_per_tok") orelse
            jsonOptU32(obj, "num_experts_per_tok") orelse 0;
        const moe_inter = jsonOptU32(layer_obj, "moe_intermediate_size") orelse
            jsonOptU32(obj, "moe_intermediate_size") orelse 0;
        const n_shared = jsonOptU32(layer_obj, "n_shared_experts") orelse
            jsonOptU32(obj, "n_shared_experts") orelse 0;
        const shared_inter = jsonOptU32(layer_obj, "shared_expert_intermediate_size") orelse
            jsonOptU32(obj, "shared_expert_intermediate_size") orelse
            (if (n_shared > 0 and moe_inter > 0) n_shared * moe_inter else 0);
        const routed_scaling = jsonOptF32(layer_obj, "routed_scaling_factor") orelse
            jsonOptF32(obj, "routed_scaling_factor") orelse
            (if (mtp_arch == .glm_moe_dsa) @as(f32, 2.5) else @as(f32, 1.0));
        const norm_topk_prob = jsonOptBool(layer_obj, "norm_topk_prob") orelse
            jsonOptBool(obj, "norm_topk_prob") orelse true;
        const moe_sigmoid: bool = blk: {
            if (jsonOptString(layer_obj, "scoring_func") orelse jsonOptString(obj, "scoring_func")) |sf| {
                if (std.mem.eql(u8, sf, "sigmoid")) break :blk true;
                if (std.mem.eql(u8, sf, "softmax")) break :blk false;
            }
            if (jsonOptBool(layer_obj, "moe_scoring_sigmoid") orelse jsonOptBool(obj, "moe_scoring_sigmoid")) |b| break :blk b;
            break :blk (mtp_arch == .glm_moe_dsa or mtp_arch == .glm5_next);
        };
        const swiglu_limit = jsonOptF32(layer_obj, "swiglu_limit") orelse
            jsonOptF32(obj, "swiglu_limit") orelse
            (if (mtp_arch == .glm5_next) @as(f32, 10.0) else @as(f32, 0.0));

        const sliding_window = jsonOptU32(layer_obj, "sliding_window") orelse
            jsonOptU32(obj, "sliding_window") orelse
            jsonOptU32(obj, "sliding_window_size") orelse 0;
        const v_scale = jsonOptF32(layer_obj, "attention_value_scale") orelse
            jsonOptF32(obj, "attention_value_scale") orelse 1.0;
        const add_swa_sink = jsonOptBool(layer_obj, "add_swa_attention_sink_bias") orelse
            jsonOptBool(obj, "add_swa_attention_sink_bias") orelse false;
        const mtp_chained = jsonOptBool(obj, "mtp_chained") orelse false;
        const mtp_cycle_last = jsonOptBool(obj, "mtp_cycle_last_stage") orelse false;
        const inter = jsonOptU32(layer_obj, "intermediate_size") orelse
            jsonOptU32(obj, "intermediate_size") orelse
            (if (moe_inter > 0) moe_inter else return error.InvalidJson);
        const rms_eps = jsonOptF32(layer_obj, "rms_norm_eps") orelse
            jsonOptF32(obj, "rms_norm_eps") orelse
            jsonOptF32(obj, "layernorm_epsilon") orelse
            (if (mtp_arch == .glm_moe_dsa or mtp_arch == .glm5_next) @as(f32, 1e-5) else @as(f32, 1e-6));
        const default_theta: f32 = if (mtp_arch == .glm_moe_dsa)
            8_000_000.0
        else if (format == .eagle3 or mtp_chained or mtp_arch == .glm5_next)
            10_000.0
        else
            1_000_000.0;
        const rope_theta = jsonOptF32(layer_obj, "rope_theta") orelse
            jsonOptF32(obj, "rope_theta") orelse
            jsonOptF32(obj, "swa_rope_theta") orelse default_theta;
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
            .mtp_arch = mtp_arch,
            .vocab_size = vocab_size,
            .draft_vocab_size = draft_vocab_size,
            .hidden_size = hidden_size,
            .target_hidden_size = target_hidden_size,
            .num_hidden_layers = num_layers,
            .num_attention_heads = num_q_heads,
            .num_key_value_heads = num_kv_heads,
            .head_dim = head_dim,
            .v_head_dim = v_head_dim,
            .rope_dim = rope_dim,
            .q_lora_rank = q_lora_rank,
            .kv_lora_rank = kv_lora_rank,
            .qk_nope_head_dim = qk_nope_head_dim,
            .qk_rope_head_dim = qk_rope_head_dim,
            .rope_interleave = rope_interleave,
            .index_share_for_mtp_iteration = index_share_mtp,
            .index_topk = index_topk,
            .num_experts = num_experts,
            .num_experts_per_tok = num_experts_per_tok,
            .moe_intermediate_size = moe_inter,
            .shared_expert_intermediate_size = shared_inter,
            .routed_scaling_factor = routed_scaling,
            .norm_topk_prob = norm_topk_prob,
            .moe_scoring_sigmoid = moe_sigmoid,
            .swiglu_limit = swiglu_limit,
            .sliding_window = sliding_window,
            .attention_value_scale = v_scale,
            .add_swa_attention_sink_bias = add_swa_sink,
            .mtp_chained = mtp_chained,
            .mtp_cycle_last_stage = mtp_cycle_last,
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
    if (jsonOptString(obj, "format")) |s| {
        if (std.mem.eql(u8, s, "eagle3")) return .eagle3;
        if (std.mem.eql(u8, s, "eagle2") or std.mem.eql(u8, s, "eagle")) return .eagle2;
        if (std.mem.eql(u8, s, "mtp")) return .mtp;
    }
    if (jsonOptString(obj, "mtp_arch") != null) return .mtp;
    if (jsonOptBool(obj, "mtp_chained") orelse false) return .mtp;
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
    quant_formats.InitError || quant_formats.ShapeError ||
    metal.WrapError || metal.ProtoEmitError || budget_mod.PlanError ||
    arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError ||
    vmstat.Error || moe.EncodeError || sme.WiredScratchError || std.Thread.SpawnError;

pub const DraftStepError = error{
    TokenOutOfRange,
    PositionOutOfRange,
    SlotOutOfRange,
    BatchTooLarge,
} || metal.Error || dequant.EncodeError || moe.EncodeError || sme.Error;

const MtpTreeQkNormRopeParams = extern struct {
    num_nodes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    v_head_dim: u32,
    rope_dim: u32,
    has_q_norm: u32,
    has_k_norm: u32,
    norm_dtype: u32,
    rms_eps: f32,
    rope_theta: f32,
    v_scale: f32,
};

const MtpKvCommitParams = extern struct {
    num_nodes: u32,
    num_kv_heads: u32,
    head_dim: u32,
    v_head_dim: u32,
    base_pos: u32,
    kv_cap: u32,
    layer_k_elem_off: u32,
    layer_v_elem_off: u32,
};

const MtpGqaTreeAttnParams = extern struct {
    num_nodes: u32,
    slot_start: u32,
    q_count: u32,
    base_pos: u32,
    kv_cap: u32,
    sliding_window: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    v_head_dim: u32,
    layer_k_elem_off: u32,
    layer_v_elem_off: u32,
    has_sink_bias: u32,
    sink_dtype: u32,
    scale: f32,
    has_sparse_indices: u32 = 0,
    sparse_k: u32 = 0,
    sparse_row_stride: u32 = 0,
    sparse_num_rows: u32 = 0,
    _pad0: u32 = 0,
};

const MtpMlaPrepParams = extern struct {
    num_nodes: u32,
    num_heads: u32,
    kv_lora_rank: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
    rope_interleave: u32,
    _pad0: u32 = 0,
    rope_theta: f32,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
    _pad3: u32 = 0,
};

const MtpSwigluClampParams = extern struct {
    count: u32,
    limit: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

const MtpMoeRouteParams = extern struct {
    m: u32,
    num_experts: u32,
    top_k: u32,
    use_sigmoid: u32,
    has_bias: u32,
    bias_dtype: u32,
    norm_topk_prob: u32,
    routed_scaling_factor: f32,
};

const MtpExpertMeta = extern struct {
    gate_off: u64,
    up_off: u64,
    down_off: u64,
    gate_sb_off: u64,
    up_sb_off: u64,
    down_sb_off: u64,
};

const MtpMoeExpertParams = extern struct {
    m: u32,
    hidden_size: u32,
    moe_inter: u32,
    top_k: u32,
    weight_dtype: u32,
    group_size: u32,
    swiglu_limit: f32,
    _pad0: u32 = 0,
};

comptime {
    if (@sizeOf(MtpTreeQkNormRopeParams) != 48) @compileError("MtpTreeQkNormRopeParams size mismatch");
    if (@sizeOf(MtpKvCommitParams) != 32) @compileError("MtpKvCommitParams size mismatch");
    if (@sizeOf(MtpGqaTreeAttnParams) != 80) @compileError("MtpGqaTreeAttnParams size mismatch");
    if (@sizeOf(MtpMlaPrepParams) != 48) @compileError("MtpMlaPrepParams size mismatch");
    if (@sizeOf(MtpSwigluClampParams) != 16) @compileError("MtpSwigluClampParams size mismatch");
    if (@sizeOf(MtpMoeRouteParams) != 32) @compileError("MtpMoeRouteParams size mismatch");
    if (@sizeOf(MtpExpertMeta) != 48) @compileError("MtpExpertMeta size mismatch");
    if (@sizeOf(MtpMoeExpertParams) != 32) @compileError("MtpMoeExpertParams size mismatch");
}

const MtpAttnPipelines = struct {
    qk_norm_rope_vscale: metal.Pipeline,
    kv_ring_commit: metal.Pipeline,
    gqa_tree_verify_attn: metal.Pipeline,
    mla_prep_rope_kv: metal.Pipeline,
    swiglu_clamped_mul: metal.Pipeline,
    moe_route_topk_sigmoid: metal.Pipeline,
    moe_gate_up_swiglu: metal.Pipeline,
    moe_down_reduce: metal.Pipeline,

    fn init(device: metal.Device, library: metal.Library) metal.Error!MtpAttnPipelines {
        const p0 = try metal.Pipeline.init(device, library, "mtp_tree_qk_norm_rope_vscale");
        errdefer p0.close();
        const p1 = try metal.Pipeline.init(device, library, "mtp_kv_ring_commit");
        errdefer p1.close();
        const p2 = try metal.Pipeline.init(device, library, "mtp_gqa_tree_verify_attn");
        errdefer p2.close();
        const p3 = try metal.Pipeline.init(device, library, "mtp_mla_prep_rope_kv");
        errdefer p3.close();
        const p4 = try metal.Pipeline.init(device, library, "mtp_swiglu_clamped_mul");
        errdefer p4.close();
        const p5 = try metal.Pipeline.init(device, library, "mtp_moe_route_topk_sigmoid");
        errdefer p5.close();
        const p6 = try metal.Pipeline.init(device, library, "mtp_moe_gate_up_swiglu");
        errdefer p6.close();
        const p7 = try metal.Pipeline.init(device, library, "mtp_moe_down_reduce");
        errdefer p7.close();
        return .{
            .qk_norm_rope_vscale = p0,
            .kv_ring_commit = p1,
            .gqa_tree_verify_attn = p2,
            .mla_prep_rope_kv = p3,
            .swiglu_clamped_mul = p4,
            .moe_route_topk_sigmoid = p5,
            .moe_gate_up_swiglu = p6,
            .moe_down_reduce = p7,
        };
    }

    fn close(self: MtpAttnPipelines) void {
        self.moe_down_reduce.close();
        self.moe_gate_up_swiglu.close();
        self.moe_route_topk_sigmoid.close();
        self.swiglu_clamped_mul.close();
        self.mla_prep_rope_kv.close();
        self.gqa_tree_verify_attn.close();
        self.kv_ring_commit.close();
        self.qk_norm_rope_vscale.close();
    }
};

const MtpMlaLayerWeights = struct {
    q_a_proj: qwen3.LinearBinding,
    q_a_layernorm: qwen3.NormBinding,
    q_b_proj: qwen3.LinearBinding,
    kv_a_proj_with_mqa: qwen3.LinearBinding,
    kv_a_layernorm: qwen3.NormBinding,
    kv_b_proj: qwen3.LinearBinding,
};

const MtpMoeLayerWeights = struct {
    router: qwen3.LinearBinding,
    router_bias: ?metal.BufferBinding = null,
    router_bias_dtype: u32 = 2,
    expert_meta_bind: metal.BufferBinding,
    expert_weight_dtype: u32,
    expert_group_size: u32,
    shared_gate_proj: ?qwen3.LinearBinding = null,
    shared_up_proj: ?qwen3.LinearBinding = null,
    shared_down_proj: ?qwen3.LinearBinding = null,
    shared_inter: u32 = 0,
};

const DraftLayerWeights = struct {
    input_layernorm: qwen3.NormBinding,
    hidden_norm: ?qwen3.NormBinding,
    /// Optional per-stage chained MTP fusion weights (`D15`, `layers.{k}.{enorm,hnorm,eh_proj,final_layernorm}`).
    stage_enorm: ?qwen3.NormBinding = null,
    stage_hnorm: ?qwen3.NormBinding = null,
    stage_eh_proj: ?qwen3.LinearBinding = null,
    stage_final_norm: ?qwen3.NormBinding = null,
    q_proj: qwen3.LinearBinding,
    k_proj: qwen3.LinearBinding,
    v_proj: qwen3.LinearBinding,
    o_proj: qwen3.LinearBinding,
    /// Optional `GLM-5.2/5.3` MLA / `GLM-5.3-Flash` NoPE-MLA projections for `1L` `MTP` (`D15`).
    mla: ?MtpMlaLayerWeights = null,
    q_norm: ?metal.BufferBinding,
    k_norm: ?metal.BufferBinding,
    qk_norm_dtype: quant.FloatDType,
    /// Optional per-head `attention_sink_bias` (`[num_attention_heads]`, `0=f16, 1=bf16, 2=f32`).
    sink_bias: ?metal.BufferBinding = null,
    sink_bias_dtype: u32 = 1,
    post_attention_layernorm: qwen3.NormBinding,
    gate_proj: qwen3.LinearBinding,
    up_proj: qwen3.LinearBinding,
    down_proj: qwen3.LinearBinding,
    /// Optional `GLM-5.2/5.3` (`256E`) / `GLM-5.3-Flash` (`288E`) routed MoE + shared_experts inside `1L` `MTP`.
    moe: ?MtpMoeLayerWeights = null,
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

/// Heap-pinned L2-resident (`<= 16 MiB`) P-core SME2 runner for compact
/// `DraftHead` projections (`D13`, ADR-0008 / ADR-0019): `EAGLE-3` `fc`,
/// `4B MTP` dual-norm `eh_proj`, and `EAGLE-2` `fc` (`0` Metal CBs, `0` allocs).
const SmeProjectionKind = enum {
    eagle3_fc,
    eagle2_fc,
    mtp_eh_proj,
};

const SmeProjectionRuntime = struct {
    weights_arena: ?arena_mod.WiredArena,
    proj: sme.SmeDraftProjection,
    dispatcher: sme.SmeDispatcher,
    cal_out_f16: []f16,
    weight_bytes: usize,
    kind: SmeProjectionKind,

    fn deinit(self: *SmeProjectionRuntime, allocator: std.mem.Allocator) void {
        allocator.free(self.cal_out_f16);
        self.dispatcher.deinit();
        self.proj.deinit();
        if (self.weights_arena) |*wa| wa.deinit();
    }

    fn totalWiredBytes(self: *const SmeProjectionRuntime) u64 {
        const w_b: u64 = if (self.weights_arena) |*wa| wa.bytes().len else 0;
        return w_b + self.proj.totalWiredBytes();
    }
};

pub const DraftHead = struct {
    config: DraftConfig,
    requested_placement: Placement,
    effective_placement: Placement,
    placement_gate_ane_us: f64,
    placement_gate_gpu_us: f64,
    placement_gate_sme2_us: f64,
    sme2_proj_bytes: usize,
    sme2_eval_count: u64,
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
    // Optional L2-resident P-core SME2 projection for `fc` / stage-0 `eh_proj` (D13).
    sme_proj: ?*SmeProjectionRuntime,
    // Optional per-stage L2-resident SME2 `eh_proj` runtimes for chained MTP stages `k >= 1`.
    extra_stage_sme_projs: [MAX_MTP_STAGES]?*SmeProjectionRuntime = @splat(null),

    // D15 custom MTP attention runtime (asymmetric head_dim/v_head_dim, partial RoPE, SWA, v_scale, sink_bias,
    // plus GLM-5.2/5.3 MLA, GLM-5.3-Flash NoPE-MLA, MoE FFN, and zero-reindex sparse index sharing).
    use_custom_attn: bool = false,
    kv_ring_cap: u32 = 0,
    mtp_pipes: ?MtpAttnPipelines = null,
    kv_ring_k_bind: metal.BufferBinding = undefined,
    kv_ring_v_bind: metal.BufferBinding = undefined,
    stage_h_prev_bind: metal.BufferBinding = undefined,
    stage_tree_slots_mask: [MAX_MTP_STAGES]u64 = @splat(0),
    mla_q_a_bind: metal.BufferBinding = undefined,
    mla_q_a_norm_bind: metal.BufferBinding = undefined,
    mla_kv_a_bind: metal.BufferBinding = undefined,
    mla_kv_a_pass_bind: metal.BufferBinding = undefined,
    mla_kv_a_norm_bind: metal.BufferBinding = undefined,
    mla_kv_b_bind: metal.BufferBinding = undefined,
    moe_router_logits_bind: metal.BufferBinding = undefined,
    moe_expert_ids_bind: metal.BufferBinding = undefined,
    moe_expert_weights_bind: metal.BufferBinding = undefined,
    moe_act_bind: metal.BufferBinding = undefined,
    max_sparse_indices: u32 = 0,
    shared_sparse_indices_host: []u32 = &.{},
    shared_sparse_indices_bind: metal.BufferBinding = undefined,
    active_sparse_bind: ?metal.BufferBinding = null,
    active_sparse_count: u32 = 0,
    active_sparse_stride: u32 = 0,
    active_sparse_rows: u32 = 0,
    /// Quantitative zero-reindex metrics (`index_share_for_mtp_iteration = true`, `0` indexer dispatches).
    zero_reindex_hits: u64 = 0,
    indexer_dispatches: u64 = 0,

    // Preallocated scratch buffers & bindings (zero hot-path heap alloc)
    token_ids_host: []u32,
    parent_slots_host: []u32,
    positions_host: []u32,
    positions_pool_host: []u32,
    ancestor_masks_host: []u64,
    topk_host: []moe.TopKCandidate,
    hidden_pool_host: []f16,
    h_in_host: []f16,
    emb_host: []f16,
    x_host: []f16,

    token_ids_bind: metal.BufferBinding,
    parent_slots_bind: metal.BufferBinding,
    positions_bind: metal.BufferBinding,
    positions_pool_bind: metal.BufferBinding,
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
    v_bind: metal.BufferBinding, // [max_tree_slots, v_dim]
    attn_pool_bind: metal.BufferBinding, // [max_tree_slots, o_in_dim]
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
    tree_v_layer_stride: u64,

    // Cached root-step predictions computed by the most recent `syncAccepted`.
    root_top_k_count: u32,
    root_candidates: [MAX_DRAFT_TOP_K]moe.TopKCandidate,

    // M8.1 (`G23`/`G24`): optional multi-format pipelines owned by `DraftHead` when
    // the draft `.amf` uses non-legacy `W21`/`W22` quantized tensors.
    qf_pipes: ?quant_formats.QuantFormatPipelines = null,
    embed_dequant_f32: ?[]f32 = null,

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

    pub fn loadFromBytesForGraph(
        allocator: std.mem.Allocator,
        target: *graph.GraphModel,
        amf_bytes: []const u8,
        config: DraftConfig,
        options: DraftLoadOptions,
    ) DraftLoadError!DraftHead {
        var parsed = try amf.Parsed.parseBytes(allocator, amf_bytes);
        defer parsed.deinit(allocator);
        return loadInternal(allocator, target, .{ .bytes = .{ .parsed = &parsed, .raw = amf_bytes } }, config, options);
    }

    pub fn loadFromFileForGraph(
        allocator: std.mem.Allocator,
        target: *graph.GraphModel,
        amf_path: [:0]const u8,
        config: DraftConfig,
        options: DraftLoadOptions,
    ) DraftLoadError!DraftHead {
        var file = try amf.AmfFile.open(allocator, amf_path);
        defer file.close(allocator);
        return loadInternal(allocator, target, .{ .file = &file }, config, options);
    }

    pub fn deinit(self: *DraftHead, allocator: std.mem.Allocator, target: *qwen3.Qwen3Model) void {
        self.deinitImpl(allocator, target);
    }

    pub fn deinitForGraph(self: *DraftHead, allocator: std.mem.Allocator, target: *graph.GraphModel) void {
        self.deinitImpl(allocator, target);
    }

    fn deinitImpl(self: *DraftHead, allocator: std.mem.Allocator, target: anytype) void {
        if (self.sme_proj) |sp| {
            sp.deinit(allocator);
            allocator.destroy(sp);
            self.sme_proj = null;
        }
        for (&self.extra_stage_sme_projs) |*slot| {
            if (slot.*) |sp| {
                sp.deinit(allocator);
                allocator.destroy(sp);
                slot.* = null;
            }
        }
        if (self.mtp_pipes) |mp| {
            mp.close();
            self.mtp_pipes = null;
        }
        if (self.ane_proj) |*ap| {
            ap.close(&target.compute);
            self.ane_proj = null;
        }
        if (self.qf_pipes) |*qfp| {
            target.compute.removeResident(&.{qfp.codebook_buf}) catch |err| {
                std.debug.panic("removeResident(qf_pipes) failed in DraftHead.deinit: {}", .{err});
            };
            qfp.close();
            self.qf_pipes = null;
        }
        if (self.embed_dequant_f32) |ed| {
            allocator.free(ed);
            self.embed_dequant_f32 = null;
        }
        if (self.kv_tiles.len > 0) {
            target.compute.removeTilesResident(self.kv_tiles) catch |err| {
                std.debug.panic("removeTilesResident failed in DraftHead.deinit: {}", .{err});
            };
            for (self.kv_tiles) |tile| tile.close();
            allocator.free(self.kv_tiles);
        }
        target.compute.removeResident(&.{ self.weights_buf, self.scratch_buf }) catch |err| {
            std.debug.panic("removeResident failed in DraftHead.deinit: {}", .{err});
        };
        self.scratch_buf.close();
        self.weights_buf.close();
        self.scratch_arena.deinit();
        self.weights_arena.deinit();
        allocator.free(self.layers);
    }

    pub fn reset(self: *DraftHead) void {
        self.seq_len = 0;
        self.root_top_k_count = 0;
        self.stage_tree_slots_mask = @splat(0);
        self.zero_reindex_hits = 0;
        self.indexer_dispatches = 0;
        self.clearSharedSparseIndices();
    }

    /// Binds shared sparse KV positions from a DSA target backbone (or standalone test) for the
    /// `index_share_for_mtp_iteration = true` zero-reindex fast path (`0` MTP indexer ops, `0` allocs).
    pub fn setSharedSparseIndices(self: *DraftHead, indices: SharedSparseIndices) void {
        if (!indices.valid or indices.k_selected == 0) {
            self.clearSharedSparseIndices();
            return;
        }
        const rows: u32 = if (indices.num_rows > 0) indices.num_rows else 1;
        const stride: u32 = if (indices.max_k > 0) indices.max_k else indices.k_selected;
        if (indices.gpu_buf) |gb| {
            self.active_sparse_bind = gb;
            self.active_sparse_count = indices.k_selected;
            self.active_sparse_stride = stride;
            self.active_sparse_rows = rows;
            return;
        }
        if (indices.host_indices.len > 0 and self.shared_sparse_indices_host.len > 0) {
            const take: usize = @min(@as(usize, indices.k_selected), @min(indices.host_indices.len, self.shared_sparse_indices_host.len));
            @memcpy(self.shared_sparse_indices_host[0..take], indices.host_indices[0..take]);
            self.active_sparse_bind = self.shared_sparse_indices_bind;
            self.active_sparse_count = @intCast(take);
            self.active_sparse_stride = @intCast(take);
            self.active_sparse_rows = 1;
            return;
        }
        self.clearSharedSparseIndices();
    }

    pub fn clearSharedSparseIndices(self: *DraftHead) void {
        self.active_sparse_bind = null;
        self.active_sparse_count = 0;
        self.active_sparse_stride = 0;
        self.active_sparse_rows = 0;
    }

    fn pullSharedSparseIndicesFromTarget(self: *DraftHead, target: anytype) void {
        if (!self.config.index_share_for_mtp_iteration) return;
        const TargetPtr = @TypeOf(target);
        const TargetT = @typeInfo(TargetPtr).pointer.child;
        if (comptime @hasDecl(TargetT, "sharedMtpSparseIndices")) {
            const info = target.sharedMtpSparseIndices();
            if (info.valid and info.k_selected > 0) {
                self.setSharedSparseIndices(.{
                    .gpu_buf = info.gpu_buf,
                    .host_indices = info.host_indices,
                    .k_selected = info.k_selected,
                    .max_k = info.max_k,
                    .num_rows = info.num_rows,
                    .valid = true,
                });
            }
        }
    }

    pub inline fn isChainedMtp(self: *const DraftHead) bool {
        return self.config.format == .mtp and self.config.mtp_chained;
    }

    pub inline fn numMtpStages(self: *const DraftHead) u32 {
        return if (self.isChainedMtp()) self.config.num_hidden_layers else 1;
    }

    pub fn totalWiredBytes(self: *const DraftHead) u64 {
        return self.total_wired_bytes;
    }

    fn encodeDraftLinearBatched(
        self: *const DraftHead,
        target: anytype,
        frame: *metal.Compute.Frame,
        lin: qwen3.LinearBinding,
        x: metal.BufferBinding,
        y: metal.BufferBinding,
        m: u32,
        barrier_before: bool,
    ) dequant.EncodeError!void {
        const amf_dt = lin.effectiveAmfDType();
        if (!dequant.usesQuantFormats(amf_dt, lin.group_size)) {
            return target.encodeLinearBatched(frame, lin, x, y, m, barrier_before);
        }
        const qf_ptr: ?*const quant_formats.QuantFormatPipelines = if (self.qf_pipes) |*qfp|
            qfp
        else if (target.qf_pipes) |*tqfp|
            tqfp
        else
            null;
        return target.dequant_pipes.encodeLinearUnified(frame, .{
            .amf_dtype = amf_dt,
            .group_size = lin.group_size,
            .tensor_scale_per_row = lin.tensor_scale_per_row,
            .weight = lin.weight,
            .params = lin.scale_bias,
            .x = x,
            .y = y,
            .m = m,
            .n = lin.n,
            .k = lin.k,
            .mode = if (m == 1) .gemv else .batched,
            .qf_pipes = qf_ptr,
            .act_quant_policy = target.act_quant_policy,
            .act_scratch = if (target.act_quant_buf) |ab| .{ .buffer = ab, .offset = 0 } else null,
            .act_scratch_bytes = target.act_quant_bytes,
        }, barrier_before);
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
        const v_row_bytes = @as(u64, self.config.vDim()) * @sizeOf(f16);
        return .{
            .buffer = self.scratch_buf,
            .offset = self.tree_v_base_off + @as(u64, layer_idx) * self.tree_v_layer_stride + @as(u64, slot_offset) * v_row_bytes,
        };
    }

    fn stagePrevHBinding(self: *const DraftHead, stage_idx: usize) metal.BufferBinding {
        const row_bytes = @as(u64, self.config.hidden_size) * @sizeOf(f16);
        return .{
            .buffer = self.scratch_buf,
            .offset = self.stage_h_prev_bind.offset + @as(u64, stage_idx) * row_bytes,
        };
    }

    fn stageSmeProj(self: *const DraftHead, stage_idx: usize) ?*SmeProjectionRuntime {
        if (stage_idx == 0) return self.sme_proj;
        if (stage_idx < MAX_MTP_STAGES) return self.extra_stage_sme_projs[stage_idx];
        return null;
    }

    fn stageEnorm(self: *const DraftHead, stage_idx: usize) qwen3.NormBinding {
        return self.layers[stage_idx].stage_enorm orelse self.enorm.?;
    }

    fn stageHnorm(self: *const DraftHead, stage_idx: usize) qwen3.NormBinding {
        return self.layers[stage_idx].stage_hnorm orelse self.hnorm.?;
    }

    fn stageEhProj(self: *const DraftHead, stage_idx: usize) qwen3.LinearBinding {
        return self.layers[stage_idx].stage_eh_proj orelse self.eh_proj.?;
    }

    fn stageFinalNorm(self: *const DraftHead, stage_idx: usize) qwen3.NormBinding {
        return self.layers[stage_idx].stage_final_norm orelse self.final_norm;
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
        return self.syncAcceptedImpl(target, tokens, target_features_bind, base_pos, top_k);
    }

    pub fn syncAcceptedForGraph(
        self: *DraftHead,
        target: *graph.GraphModel,
        tokens: []const u32,
        target_features_bind: metal.BufferBinding,
        base_pos: u32,
        top_k: u32,
    ) DraftStepError!void {
        return self.syncAcceptedImpl(target, tokens, target_features_bind, base_pos, top_k);
    }

    fn syncAcceptedImpl(
        self: *DraftHead,
        target: anytype,
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
            self.positions_pool_host[i] = base_pos + @as(u32, @intCast(i));
            self.ancestor_masks_host[i] = if (i == 63) ~@as(u64, 0) else (@as(u64, 1) << @as(u6, @intCast(i + 1))) - 1;
        }
        self.stage_tree_slots_mask = @splat(0);
        self.pullSharedSparseIndicesFromTarget(target);

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
            0, // active_stage_idx
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
        const default_stage: usize = if (self.isChainedMtp()) @min(@as(usize, 1), self.layers.len - 1) else 0;
        return self.stepTreeBatchImpl(target, tokens, parent_hidden_rows, positions, slot_ancestor_masks, slot_start, top_k, out_candidates, default_stage);
    }

    pub fn stepTreeBatchForGraph(
        self: *DraftHead,
        target: *graph.GraphModel,
        tokens: []const u32,
        parent_hidden_rows: []const u32,
        positions: []const u32,
        slot_ancestor_masks: []const u64,
        slot_start: u32,
        top_k: u32,
        out_candidates: ?[]moe.TopKCandidate,
    ) DraftStepError!void {
        const default_stage: usize = if (self.isChainedMtp()) @min(@as(usize, 1), self.layers.len - 1) else 0;
        return self.stepTreeBatchImpl(target, tokens, parent_hidden_rows, positions, slot_ancestor_masks, slot_start, top_k, out_candidates, default_stage);
    }

    /// Chained `D15` native MTP variant selecting explicit predictor stage `stage_idx`
    /// (`k = 0..num_hidden_layers-1`: stage `k = 1` predicts depth-2 from `h_0` + depth-1 token;
    /// stage `k = 2` predicts depth-3 from `h_1` + depth-2 token).
    pub fn stepTreeBatchForStage(
        self: *DraftHead,
        target: *qwen3.Qwen3Model,
        stage_idx: u32,
        tokens: []const u32,
        parent_hidden_rows: []const u32,
        positions: []const u32,
        slot_ancestor_masks: []const u64,
        slot_start: u32,
        top_k: u32,
        out_candidates: ?[]moe.TopKCandidate,
    ) DraftStepError!void {
        return self.stepTreeBatchImpl(target, tokens, parent_hidden_rows, positions, slot_ancestor_masks, slot_start, top_k, out_candidates, @intCast(stage_idx));
    }

    pub fn stepTreeBatchForStageForGraph(
        self: *DraftHead,
        target: *graph.GraphModel,
        stage_idx: u32,
        tokens: []const u32,
        parent_hidden_rows: []const u32,
        positions: []const u32,
        slot_ancestor_masks: []const u64,
        slot_start: u32,
        top_k: u32,
        out_candidates: ?[]moe.TopKCandidate,
    ) DraftStepError!void {
        return self.stepTreeBatchImpl(target, tokens, parent_hidden_rows, positions, slot_ancestor_masks, slot_start, top_k, out_candidates, @intCast(stage_idx));
    }

    fn stepTreeBatchImpl(
        self: *DraftHead,
        target: anytype,
        tokens: []const u32,
        parent_hidden_rows: []const u32,
        positions: []const u32,
        slot_ancestor_masks: []const u64,
        slot_start: u32,
        top_k: u32,
        out_candidates: ?[]moe.TopKCandidate,
        stage_idx_raw: usize,
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

        const eff_stage: usize = if (self.isChainedMtp())
            @min(stage_idx_raw, self.layers.len - 1)
        else
            0;

        var batch_bits: u64 = 0;
        for (0..m) |i| {
            if (tokens[i] >= self.config.vocab_size) return error.TokenOutOfRange;
            if (positions[i] >= self.max_seq_len) return error.PositionOutOfRange;
            if (parent_hidden_rows[i] > self.max_tree_slots) return error.SlotOutOfRange;
            self.token_ids_host[i] = tokens[i];
            self.parent_slots_host[i] = parent_hidden_rows[i];
            self.positions_host[i] = positions[i];
            self.positions_pool_host[slot_start + @as(u32, @intCast(i))] = positions[i];
            batch_bits |= (@as(u64, 1) << @as(u6, @intCast(slot_start + @as(u32, @intCast(i)))));
        }
        if (self.isChainedMtp()) {
            const allowed_mask = self.stage_tree_slots_mask[eff_stage] | batch_bits;
            for (0..total_slots) |s| {
                const self_bit: u64 = @as(u64, 1) << @as(u6, @intCast(s));
                self.ancestor_masks_host[s] = (slot_ancestor_masks[s] & allowed_mask) | self_bit;
            }
        } else {
            @memcpy(self.ancestor_masks_host[0..total_slots], slot_ancestor_masks[0..total_slots]);
        }
        self.pullSharedSparseIndicesFromTarget(target);

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
            eff_stage,
        );
        if (self.isChainedMtp()) {
            self.stage_tree_slots_mask[eff_stage] |= batch_bits;
        }

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

    fn encodeMtpStageFusion(
        self: *const DraftHead,
        target: anytype,
        frame: *metal.Compute.Frame,
        stage_idx: usize,
        h_in_active: metal.BufferBinding,
        m: u32,
        first_barrier: bool,
    ) DraftStepError!void {
        const h = self.config.hidden_size;
        const enorm_b = self.stageEnorm(stage_idx);
        const hnorm_b = self.stageHnorm(stage_idx);
        const eh_proj_b = self.stageEhProj(stage_idx);
        try target.attn_pipes.encodeRmsNorm(frame, .{
            .x = self.emb_bind,
            .w = enorm_b.binding,
            .y = self.e_norm_bind,
            .num_rows = m,
            .hidden_dim = h,
            .eps = self.config.rms_norm_eps,
            .w_dtype = enorm_b.dtype,
        }, first_barrier);
        try target.attn_pipes.encodeRmsNorm(frame, .{
            .x = h_in_active,
            .w = hnorm_b.binding,
            .y = self.h_norm_bind,
            .num_rows = m,
            .hidden_dim = h,
            .eps = self.config.rms_norm_eps,
            .w_dtype = hnorm_b.dtype,
        }, false);
        try target.moe_pipes.encodeCopyStrided(frame, .{
            .src = self.e_norm_bind,
            .dst = self.x_cat_bind,
            .num_rows = m,
            .copy_cols = h,
            .src_stride = h,
            .dst_stride = 2 * h,
            .dst_col_off = 0,
        }, true);
        try target.moe_pipes.encodeCopyStrided(frame, .{
            .src = self.h_norm_bind,
            .dst = self.x_cat_bind,
            .num_rows = m,
            .copy_cols = h,
            .src_stride = h,
            .dst_stride = 2 * h,
            .dst_col_off = h,
        }, false);
        try self.encodeDraftLinearBatched(target, frame, eh_proj_b, self.x_cat_bind, self.x_bind, m, true);
    }

    fn encodeDraftLayerBody(
        self: *DraftHead,
        target: anytype,
        frame: *metal.Compute.Frame,
        l: usize,
        m: u32,
        slot_start: u32,
        base_pos: u32,
        total_tree_nodes: u32,
        commit_kv: bool,
        use_ane_o_proj: bool,
    ) DraftStepError!void {
        const layer = self.layers[l];
        const h = self.config.hidden_size;
        const q_dim = self.config.qDim();
        const o_in_dim = self.config.oInDim();
        const qk_hd = self.config.effectiveQkHeadDim();
        const kv_heads = self.config.effectiveKvHeads();
        const q_slot_bind: metal.BufferBinding = .{
            .buffer = self.scratch_buf,
            .offset = self.q_pool_bind.offset + @as(u64, slot_start) * q_dim * @sizeOf(f16),
        };
        const attn_slot_bind: metal.BufferBinding = .{
            .buffer = self.scratch_buf,
            .offset = self.attn_pool_bind.offset + @as(u64, slot_start) * o_in_dim * @sizeOf(f16),
        };

        const qkv_in: metal.BufferBinding = if (self.config.format == .eagle3 and l == 0)
            self.x_cat_bind
        else blk: {
            try target.attn_pipes.encodeRmsNorm(frame, .{
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

        const seq_b = target.act_quant_policy.isEnabled();
        if (layer.mla) |mla| {
            // D15 `GLM-5.2/5.3` MLA (`qk_nope=192, qk_rope=64, rope_interleave=true`) and
            // `GLM-5.3-Flash` NoPE-MLA (`qk_nope=256, qk_rope=0`) inside 1L MTP.
            const mp = self.mtp_pipes.?;
            const dummy_b = self.x_bind;
            const kv_lora = self.config.kv_lora_rank;
            const qk_rope = self.config.qk_rope_head_dim;

            try self.encodeDraftLinearBatched(target, frame, mla.q_a_proj, qkv_in, self.mla_q_a_bind, m, true);
            try self.encodeDraftLinearBatched(target, frame, mla.kv_a_proj_with_mqa, qkv_in, self.mla_kv_a_bind, m, seq_b);

            try target.attn_pipes.encodeRmsNorm(frame, .{
                .x = self.mla_q_a_bind,
                .w = mla.q_a_layernorm.binding,
                .y = self.mla_q_a_norm_bind,
                .num_rows = m,
                .hidden_dim = self.config.q_lora_rank,
                .eps = self.config.rms_norm_eps,
                .w_dtype = mla.q_a_layernorm.dtype,
            }, true);

            const kv_norm_in: metal.BufferBinding = if (qk_rope > 0) blk: {
                try target.moe_pipes.encodeCopyStrided(frame, .{
                    .src = self.mla_kv_a_bind,
                    .dst = self.mla_kv_a_pass_bind,
                    .num_rows = m,
                    .copy_cols = kv_lora,
                    .src_stride = kv_lora + qk_rope,
                    .dst_stride = kv_lora,
                }, false);
                break :blk self.mla_kv_a_pass_bind;
            } else self.mla_kv_a_bind;

            try target.attn_pipes.encodeRmsNorm(frame, .{
                .x = kv_norm_in,
                .w = mla.kv_a_layernorm.binding,
                .y = self.mla_kv_a_norm_bind,
                .num_rows = m,
                .hidden_dim = kv_lora,
                .eps = self.config.rms_norm_eps,
                .w_dtype = mla.kv_a_layernorm.dtype,
            }, qk_rope > 0);

            try self.encodeDraftLinearBatched(target, frame, mla.q_b_proj, self.mla_q_a_norm_bind, q_slot_bind, m, true);
            try self.encodeDraftLinearBatched(target, frame, mla.kv_b_proj, self.mla_kv_a_norm_bind, self.mla_kv_b_bind, m, seq_b);

            const prep_p: MtpMlaPrepParams = .{
                .num_nodes = m,
                .num_heads = self.config.num_attention_heads,
                .kv_lora_rank = kv_lora,
                .qk_nope_head_dim = self.config.qk_nope_head_dim,
                .qk_rope_head_dim = qk_rope,
                .v_head_dim = self.config.effectiveVHeadDim(),
                .rope_interleave = if (self.config.rope_interleave) 1 else 0,
                .rope_theta = if (qk_rope > 0) self.config.rope_theta else 0.0,
            };
            try frame.encodeThreadgroups(
                mp.mla_prep_rope_kv,
                &.{
                    q_slot_bind,
                    self.mla_kv_a_bind,
                    self.mla_kv_b_bind,
                    self.positions_bind,
                    self.treeKBinding(l, slot_start),
                    self.treeVBinding(l, slot_start),
                },
                std.mem.asBytes(&prep_p),
                .{ self.config.num_attention_heads, m, 1 },
                .{ 128, 1, 1 },
                true,
            );

            // Zero-reindex fast path (`index_share_for_mtp_iteration = true`): skip any MTP DSA indexer
            // (`0` indexer ops) and attend strictly over shared sparse KV positions + tree-local KV positions.
            const use_sparse = self.config.index_share_for_mtp_iteration and
                self.active_sparse_bind != null and
                self.active_sparse_count > 0;
            if (use_sparse) {
                self.zero_reindex_hits += 1;
            }
            const kv_k_layer_elems: u32 = @intCast(@as(u64, l) * self.kv_ring_cap * self.config.kvDim());
            const kv_v_layer_elems: u32 = @intCast(@as(u64, l) * self.kv_ring_cap * self.config.vDim());
            const gqa_p: MtpGqaTreeAttnParams = .{
                .num_nodes = total_tree_nodes,
                .slot_start = slot_start,
                .q_count = m,
                .base_pos = base_pos,
                .kv_cap = self.kv_ring_cap,
                .sliding_window = self.config.sliding_window,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = kv_heads,
                .head_dim = qk_hd,
                .v_head_dim = self.config.effectiveVHeadDim(),
                .layer_k_elem_off = kv_k_layer_elems,
                .layer_v_elem_off = kv_v_layer_elems,
                .has_sink_bias = if (layer.sink_bias != null) 1 else 0,
                .sink_dtype = layer.sink_bias_dtype,
                .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(qk_hd))),
                .has_sparse_indices = if (use_sparse) 1 else 0,
                .sparse_k = if (use_sparse) self.active_sparse_count else 0,
                .sparse_row_stride = if (use_sparse) self.active_sparse_stride else 0,
                .sparse_num_rows = if (use_sparse) self.active_sparse_rows else 0,
            };
            try frame.encodeThreadgroups(
                mp.gqa_tree_verify_attn,
                &.{
                    q_slot_bind,
                    self.kv_ring_k_bind,
                    self.kv_ring_v_bind,
                    self.treeKBinding(l, 0),
                    self.treeVBinding(l, 0),
                    self.positions_pool_bind,
                    self.ancestor_masks_bind,
                    layer.sink_bias orelse dummy_b,
                    attn_slot_bind,
                    self.active_sparse_bind orelse self.shared_sparse_indices_bind,
                },
                std.mem.asBytes(&gqa_p),
                .{ self.config.num_attention_heads, m, 1 },
                .{ 128, 1, 1 },
                true,
            );
        } else {
            try self.encodeDraftLinearBatched(target, frame, layer.q_proj, qkv_in, q_slot_bind, m, true);
            try self.encodeDraftLinearBatched(target, frame, layer.k_proj, qkv_in, self.k_bind, m, seq_b);
            try self.encodeDraftLinearBatched(target, frame, layer.v_proj, qkv_in, self.v_bind, m, seq_b);

            if (self.use_custom_attn) {
                const mp = self.mtp_pipes.?;
                const dummy_b = self.x_bind;
                const norm_dt_u32: u32 = switch (layer.qk_norm_dtype) {
                    .f16 => 0,
                    .bf16 => 1,
                };
                const qkp: MtpTreeQkNormRopeParams = .{
                    .num_nodes = m,
                    .num_q_heads = self.config.num_attention_heads,
                    .num_kv_heads = self.config.num_key_value_heads,
                    .head_dim = self.config.head_dim,
                    .v_head_dim = self.config.effectiveVHeadDim(),
                    .rope_dim = self.config.effectiveRopeDim(),
                    .has_q_norm = if (layer.q_norm != null) 1 else 0,
                    .has_k_norm = if (layer.k_norm != null) 1 else 0,
                    .norm_dtype = norm_dt_u32,
                    .rms_eps = self.config.rms_norm_eps,
                    .rope_theta = self.config.rope_theta,
                    .v_scale = self.config.attention_value_scale,
                };
                try frame.encodeThreadgroups(
                    mp.qk_norm_rope_vscale,
                    &.{
                        q_slot_bind,
                        self.k_bind,
                        self.v_bind,
                        layer.q_norm orelse dummy_b,
                        layer.k_norm orelse dummy_b,
                        self.positions_bind,
                        self.treeKBinding(l, slot_start),
                        self.treeVBinding(l, slot_start),
                    },
                    std.mem.asBytes(&qkp),
                    .{ self.config.num_attention_heads + self.config.num_key_value_heads, m, 1 },
                    .{ 128, 1, 1 },
                    true,
                );

                const use_sparse = self.config.index_share_for_mtp_iteration and
                    self.active_sparse_bind != null and
                    self.active_sparse_count > 0;
                if (use_sparse) {
                    self.zero_reindex_hits += 1;
                }
                const kv_k_layer_elems: u32 = @intCast(@as(u64, l) * self.kv_ring_cap * self.config.kvDim());
                const kv_v_layer_elems: u32 = @intCast(@as(u64, l) * self.kv_ring_cap * self.config.vDim());
                const gqa_p: MtpGqaTreeAttnParams = .{
                    .num_nodes = total_tree_nodes,
                    .slot_start = slot_start,
                    .q_count = m,
                    .base_pos = base_pos,
                    .kv_cap = self.kv_ring_cap,
                    .sliding_window = self.config.sliding_window,
                    .num_q_heads = self.config.num_attention_heads,
                    .num_kv_heads = self.config.num_key_value_heads,
                    .head_dim = self.config.head_dim,
                    .v_head_dim = self.config.effectiveVHeadDim(),
                    .layer_k_elem_off = kv_k_layer_elems,
                    .layer_v_elem_off = kv_v_layer_elems,
                    .has_sink_bias = if (layer.sink_bias != null) 1 else 0,
                    .sink_dtype = layer.sink_bias_dtype,
                    .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.config.head_dim))),
                    .has_sparse_indices = if (use_sparse) 1 else 0,
                    .sparse_k = if (use_sparse) self.active_sparse_count else 0,
                    .sparse_row_stride = if (use_sparse) self.active_sparse_stride else 0,
                    .sparse_num_rows = if (use_sparse) self.active_sparse_rows else 0,
                };
                try frame.encodeThreadgroups(
                    mp.gqa_tree_verify_attn,
                    &.{
                        q_slot_bind,
                        self.kv_ring_k_bind,
                        self.kv_ring_v_bind,
                        self.treeKBinding(l, 0),
                        self.treeVBinding(l, 0),
                        self.positions_pool_bind,
                        self.ancestor_masks_bind,
                        layer.sink_bias orelse dummy_b,
                        attn_slot_bind,
                        self.active_sparse_bind orelse self.shared_sparse_indices_bind,
                    },
                    std.mem.asBytes(&gqa_p),
                    .{ self.config.num_attention_heads, m, 1 },
                    .{ 128, 1, 1 },
                    true,
                );
            } else {
                try target.attn_pipes.encodeTreeQkNormRope(frame, .{
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

                try target.attn_pipes.encodeGqaTreeVerifyAttn(frame, .{
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
            }
        }

        // D5 Placement: either run `layers[0].o_proj` on ANE over zero-copy IOSurface
        // or stay in the same Metal 4 command buffer on GPU!
        if (l == 0 and use_ane_o_proj) {
            const ap = &(self.ane_proj.?);
            try target.moe_pipes.encodeCopyStrided(frame, .{
                .src = attn_slot_bind,
                .dst = ap.inBinding(),
                .num_rows = m,
                .copy_cols = o_in_dim,
                .src_stride = o_in_dim,
                .dst_stride = o_in_dim,
            }, true);
            const mid_sig = try frame.commit();
            try target.compute.wait(mid_sig, self.timeout_ms);

            const pinfo = try ap.model.predict(ap.plan);
            ap.zero_copy_ok = pinfo.allOutputsZeroCopy();

            frame.* = try target.compute.beginFrame(self.timeout_ms);
            try target.moe_pipes.encodeResidualAdd(frame, .{
                .x = self.x_bind,
                .delta = ap.outBinding(),
                .count = m * h,
            }, false);
        } else {
            try self.encodeDraftLinearBatched(target, frame, layer.o_proj, attn_slot_bind, self.proj_out_bind, m, true);
            try target.moe_pipes.encodeResidualAdd(frame, .{
                .x = self.x_bind,
                .delta = self.proj_out_bind,
                .count = m * h,
            }, true);
        }

        try target.attn_pipes.encodeRmsNorm(frame, .{
            .x = self.x_bind,
            .w = layer.post_attention_layernorm.binding,
            .y = self.x_norm_bind,
            .num_rows = m,
            .hidden_dim = h,
            .eps = self.config.rms_norm_eps,
            .w_dtype = layer.post_attention_layernorm.dtype,
        }, true);

        if (layer.moe) |moe_l| {
            const mp = self.mtp_pipes.?;
            const dummy_b = self.x_bind;
            // 1. Optional shared expert path (`GLM-5.2/5.3` & `GLM-5.3-Flash` `1` shared expert).
            if (moe_l.shared_gate_proj != null and moe_l.shared_up_proj != null and moe_l.shared_down_proj != null) {
                try self.encodeDraftLinearBatched(target, frame, moe_l.shared_gate_proj.?, self.x_norm_bind, self.mlp_gate_bind, m, true);
                try self.encodeDraftLinearBatched(target, frame, moe_l.shared_up_proj.?, self.x_norm_bind, self.mlp_up_bind, m, seq_b);
                const cnt = m * moe_l.shared_inter;
                if (self.config.swiglu_limit > 0.0) {
                    const cp: MtpSwigluClampParams = .{ .count = cnt, .limit = self.config.swiglu_limit };
                    try frame.encodeThreads(
                        mp.swiglu_clamped_mul,
                        &.{ self.mlp_gate_bind, self.mlp_up_bind, self.mlp_act_bind },
                        std.mem.asBytes(&cp),
                        .{ cnt, 1, 1 },
                        .{ @min(cnt, 256), 1, 1 },
                        true,
                    );
                } else {
                    try target.moe_pipes.encodeSiluMul(frame, .{
                        .gate = self.mlp_gate_bind,
                        .up = self.mlp_up_bind,
                        .out = self.mlp_act_bind,
                        .count = cnt,
                    }, true);
                }
                try self.encodeDraftLinearBatched(target, frame, moe_l.shared_down_proj.?, self.mlp_act_bind, self.proj_out_bind, m, true);
                try target.moe_pipes.encodeResidualAdd(frame, .{
                    .x = self.x_bind,
                    .delta = self.proj_out_bind,
                    .count = m * h,
                }, true);
            }
            // 2. Routed MoE (`sigmoid + e_score_correction_bias` + top-K + optional `swiglu_limit` + residual add).
            try self.encodeDraftLinearBatched(target, frame, moe_l.router, self.x_norm_bind, self.moe_router_logits_bind, m, true);
            const rp: MtpMoeRouteParams = .{
                .m = m,
                .num_experts = self.config.num_experts,
                .top_k = self.config.num_experts_per_tok,
                .use_sigmoid = if (self.config.moe_scoring_sigmoid) 1 else 0,
                .has_bias = if (moe_l.router_bias != null) 1 else 0,
                .bias_dtype = moe_l.router_bias_dtype,
                .norm_topk_prob = if (self.config.norm_topk_prob) 1 else 0,
                .routed_scaling_factor = self.config.routed_scaling_factor,
            };
            try frame.encodeThreadgroups(
                mp.moe_route_topk_sigmoid,
                &.{
                    self.moe_router_logits_bind,
                    moe_l.router_bias orelse dummy_b,
                    self.moe_expert_ids_bind,
                    self.moe_expert_weights_bind,
                },
                std.mem.asBytes(&rp),
                .{ m, 1, 1 },
                .{ 128, 1, 1 },
                true,
            );
            const ep: MtpMoeExpertParams = .{
                .m = m,
                .hidden_size = h,
                .moe_inter = self.config.moe_intermediate_size,
                .top_k = self.config.num_experts_per_tok,
                .weight_dtype = moe_l.expert_weight_dtype,
                .group_size = moe_l.expert_group_size,
                .swiglu_limit = self.config.swiglu_limit,
            };
            const weights_root: metal.BufferBinding = .{ .buffer = self.weights_buf, .offset = 0 };
            try frame.encodeThreadgroups(
                mp.moe_gate_up_swiglu,
                &.{
                    self.x_norm_bind,
                    weights_root,
                    moe_l.expert_meta_bind,
                    self.moe_expert_ids_bind,
                    self.moe_act_bind,
                },
                std.mem.asBytes(&ep),
                .{ (self.config.moe_intermediate_size + 3) / 4, self.config.num_experts_per_tok, m },
                .{ 128, 1, 1 },
                true,
            );
            try frame.encodeThreadgroups(
                mp.moe_down_reduce,
                &.{
                    self.moe_act_bind,
                    weights_root,
                    moe_l.expert_meta_bind,
                    self.moe_expert_ids_bind,
                    self.moe_expert_weights_bind,
                    self.x_bind,
                },
                std.mem.asBytes(&ep),
                .{ (h + 3) / 4, m, 1 },
                .{ 128, 1, 1 },
                true,
            );
        } else {
            try self.encodeDraftLinearBatched(target, frame, layer.gate_proj, self.x_norm_bind, self.mlp_gate_bind, m, true);
            try self.encodeDraftLinearBatched(target, frame, layer.up_proj, self.x_norm_bind, self.mlp_up_bind, m, seq_b);
            const cnt = m * self.config.intermediate_size;
            if (self.config.swiglu_limit > 0.0 and self.mtp_pipes != null) {
                const mp = self.mtp_pipes.?;
                const cp: MtpSwigluClampParams = .{ .count = cnt, .limit = self.config.swiglu_limit };
                try frame.encodeThreads(
                    mp.swiglu_clamped_mul,
                    &.{ self.mlp_gate_bind, self.mlp_up_bind, self.mlp_act_bind },
                    std.mem.asBytes(&cp),
                    .{ cnt, 1, 1 },
                    .{ @min(cnt, 256), 1, 1 },
                    true,
                );
            } else {
                try target.moe_pipes.encodeSiluMul(frame, .{
                    .gate = self.mlp_gate_bind,
                    .up = self.mlp_up_bind,
                    .out = self.mlp_act_bind,
                    .count = cnt,
                }, true);
            }
            try self.encodeDraftLinearBatched(target, frame, layer.down_proj, self.mlp_act_bind, self.proj_out_bind, m, true);
            try target.moe_pipes.encodeResidualAdd(frame, .{
                .x = self.x_bind,
                .delta = self.proj_out_bind,
                .count = m * h,
            }, true);
        }

        // If syncing accepted tokens, commit draft K/V rows into KV ring or IOSurface KvTiles now.
        if (commit_kv) {
            if (self.use_custom_attn) {
                const mp = self.mtp_pipes.?;
                const kv_k_layer_elems: u32 = @intCast(@as(u64, l) * self.kv_ring_cap * self.config.kvDim());
                const kv_v_layer_elems: u32 = @intCast(@as(u64, l) * self.kv_ring_cap * self.config.vDim());
                const cp: MtpKvCommitParams = .{
                    .num_nodes = m,
                    .num_kv_heads = kv_heads,
                    .head_dim = qk_hd,
                    .v_head_dim = self.config.effectiveVHeadDim(),
                    .base_pos = base_pos,
                    .kv_cap = self.kv_ring_cap,
                    .layer_k_elem_off = kv_k_layer_elems,
                    .layer_v_elem_off = kv_v_layer_elems,
                };
                try frame.encodeThreadgroups(
                    mp.kv_ring_commit,
                    &.{
                        self.treeKBinding(l, 0),
                        self.treeVBinding(l, 0),
                        self.kv_ring_k_bind,
                        self.kv_ring_v_bind,
                    },
                    std.mem.asBytes(&cp),
                    .{ kv_heads, m, 1 },
                    .{ 128, 1, 1 },
                    true,
                );
            } else {
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
                    try target.attn_pipes.encodeKvTreeCommit(frame, .{
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
    }

    fn runForwardInternal(
        self: *DraftHead,
        target: anytype,
        m: u32,
        slot_start: u32,
        base_pos: u32,
        is_target_sync: bool,
        target_features_bind: metal.BufferBinding,
        commit_kv: bool,
        top_k: u32,
        logits_last_row_only: bool,
        active_stage_idx: usize,
    ) DraftStepError!void {
        const h = self.config.hidden_size;
        const total_tree_nodes = slot_start + m;
        const chained_mtp = self.isChainedMtp();
        const sme_stage: usize = if (chained_mtp and !is_target_sync) active_stage_idx else 0;
        const use_ane_o_proj = (self.effective_placement == .ane and self.ane_proj != null and m <= self.ane_proj.?.batch_rows);

        // D13 Placement (`Placement.cpu_sme2`): evaluate compact L2-resident (`<= 16 MiB`)
        // `EAGLE-3` `fc` or `MTP` dual-norm `eh_proj` / `EAGLE-2` `fc` on P-core SME2
        // BEFORE beginning the Metal 4 command buffer (`0` Metal CBs, `0` heap allocs)!
        var sme_did_fc = false;
        var sme_did_fusion = false;
        if (self.effective_placement == .cpu_sme2) {
            if (self.stageSmeProj(sme_stage)) |sp| {
                if (m <= sp.proj.max_m) {
                    switch (sp.kind) {
                        .eagle3_fc => {
                            if (is_target_sync) {
                                const k_fc: usize = sp.proj.weight.k;
                                const mk = @as(usize, m) * k_fc;
                                const src_aux = target.accepted_aux_host[0..mk];
                                const out_bytes = std.mem.sliceAsBytes(self.h_in_host[0 .. @as(usize, m) * h]);
                                if (sp.proj.weight.dtype.activationIsBf16()) {
                                    for (0..mk) |i| {
                                        sp.proj.x_act_u16[i] = quant.f32ToFloatBits(.bf16, @floatCast(src_aux[i]));
                                    }
                                    try sp.proj.evalU16(&sp.dispatcher, sp.proj.x_act_u16[0..mk], null, out_bytes, m, .{ .out_dtype = .f16 });
                                } else {
                                    const src_u16: []const u16 = @ptrCast(src_aux);
                                    try sp.proj.evalU16(&sp.dispatcher, src_u16, null, out_bytes, m, .{ .out_dtype = .f16 });
                                }
                                sme_did_fc = true;
                                self.sme2_eval_count += 1;
                            }
                        },
                        .mtp_eh_proj => {
                            if (self.embed.weight_host.len > 0) {
                                cpuEmbedLookupRowsWithScratch(self.embed, self.token_ids_host[0..m], h, self.emb_host[0 .. @as(usize, m) * h], self.embed_dequant_f32);
                                const h_src: []const f16 = if (is_target_sync)
                                    target.accepted_top_host[0 .. @as(usize, m) * h]
                                else blk: {
                                    for (0..m) |r| {
                                        const prow = self.parent_slots_host[r];
                                        @memcpy(
                                            self.h_in_host[r * h ..][0..h],
                                            self.hidden_pool_host[@as(usize, prow) * h ..][0..h],
                                        );
                                    }
                                    break :blk self.h_in_host[0 .. @as(usize, m) * h];
                                };
                                const en_w: ?sme.NormWeightSlice = normToSmeSlice(self.stageEnorm(sme_stage), h);
                                const hn_w: ?sme.NormWeightSlice = normToSmeSlice(self.stageHnorm(sme_stage), h);
                                try sp.proj.evalDualNormConcatLinear(
                                    &sp.dispatcher,
                                    .{ .f16 = self.emb_host[0 .. @as(usize, m) * h] },
                                    en_w,
                                    .{ .f16 = h_src },
                                    hn_w,
                                    self.config.rms_norm_eps,
                                    null,
                                    std.mem.sliceAsBytes(self.x_host[0 .. @as(usize, m) * h]),
                                    m,
                                    .{ .out_dtype = .f16 },
                                );
                                sme_did_fusion = true;
                                self.sme2_eval_count += 1;
                            }
                        },
                        .eagle2_fc => {
                            if (self.embed.weight_host.len > 0) {
                                cpuEmbedLookupRowsWithScratch(self.embed, self.token_ids_host[0..m], h, self.emb_host[0 .. @as(usize, m) * h], self.embed_dequant_f32);
                                const h_src: []const f16 = if (is_target_sync)
                                    target.accepted_top_host[0 .. @as(usize, m) * h]
                                else blk: {
                                    for (0..m) |r| {
                                        const prow = self.parent_slots_host[r];
                                        @memcpy(
                                            self.h_in_host[r * h ..][0..h],
                                            self.hidden_pool_host[@as(usize, prow) * h ..][0..h],
                                        );
                                    }
                                    break :blk self.h_in_host[0 .. @as(usize, m) * h];
                                };
                                const act_dt: quant.FloatDType = if (sp.proj.weight.dtype.activationIsBf16()) .bf16 else .f16;
                                const k_fc = 2 * @as(usize, h);
                                for (0..m) |r| {
                                    const dst_row = sp.proj.x_act_u16[r * k_fc ..][0..k_fc];
                                    for (0..h) |d| {
                                        dst_row[d] = quant.f32ToFloatBits(act_dt, @floatCast(self.emb_host[r * h + d]));
                                        dst_row[h + d] = quant.f32ToFloatBits(act_dt, @floatCast(h_src[r * h + d]));
                                    }
                                }
                                try sp.proj.evalU16(
                                    &sp.dispatcher,
                                    sp.proj.x_act_u16[0 .. @as(usize, m) * k_fc],
                                    null,
                                    std.mem.sliceAsBytes(self.x_host[0 .. @as(usize, m) * h]),
                                    m,
                                    .{ .out_dtype = .f16 },
                                );
                                sme_did_fusion = true;
                                self.sme2_eval_count += 1;
                            }
                        },
                    }
                }
            }
        }

        var frame = try target.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();

        if (!sme_did_fusion) {
            // 1. Prepare `h_in_bind [m, H]` (either from target verifier features or parent draft hidden pool).
            var h_in_active = self.h_in_bind;
            if (is_target_sync) {
                if (self.config.format == .eagle3) {
                    if (!sme_did_fc) {
                        try self.encodeDraftLinearBatched(target, &frame, self.fc.?, target_features_bind, self.h_in_bind, m, false);
                    }
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
            const emb_amf = self.embed.effectiveAmfDType();
            if (dequant.usesQuantFormats(emb_amf, self.embed.group_size)) {
                cpuEmbedLookupRowsWithScratch(self.embed, self.token_ids_host[0..m], h, self.emb_host, self.embed_dequant_f32);
            } else {
                try target.moe_pipes.encodeEmbedLookup(&frame, self.embed.dtype, .{
                    .weight = self.embed.weight,
                    .scale_bias = self.embed.scale_bias,
                    .tokens = .{ .token_ids = self.token_ids_bind },
                    .out = self.emb_bind,
                    .m = m,
                    .vocab_size = self.config.vocab_size,
                    .hidden_size = h,
                }, false);
            }

            // 3. Format-specific feature fusion before active layer/stage.
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
                    try self.encodeDraftLinearBatched(target, &frame, self.fc.?, self.x_cat_bind, self.x_bind, m, true);
                },
                .mtp => {
                    try self.encodeMtpStageFusion(target, &frame, sme_stage, h_in_active, m, true);
                },
                .l1_only, .small_draft, .self_recurrent, .dspark, .dflare, .dfly2 => unreachable,
            }
        }

        const row_bytes = @as(u64, h) * @sizeOf(f16);
        if (chained_mtp) {
            if (is_target_sync) {
                // Stage 0 predicts depth-1 (`root_candidates`) and updates stage-0 KV along accepted prefix.
                try self.encodeDraftLayerBody(target, &frame, 0, m, slot_start, base_pos, total_tree_nodes, commit_kv, use_ane_o_proj);

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

                if (top_k > 0) {
                    const eff_k = @min(top_k, self.config.draft_vocab_size);
                    const fn0 = self.stageFinalNorm(0);
                    try target.attn_pipes.encodeRmsNorm(&frame, .{
                        .x = last_src,
                        .w = fn0.binding,
                        .y = self.x_norm_bind,
                        .num_rows = 1,
                        .hidden_dim = h,
                        .eps = self.config.rms_norm_eps,
                        .w_dtype = fn0.dtype,
                    }, true);
                    try self.encodeDraftLinearBatched(target, &frame, self.lm_head, self.x_norm_bind, self.logits_bind, 1, true);
                    try target.moe_pipes.encodeTopKLogitsRows(&frame, .{
                        .logits = self.logits_bind,
                        .out_candidates = self.topk_bind,
                        .num_rows = 1,
                        .vocab_size = self.config.draft_vocab_size,
                        .top_k = eff_k,
                        .logits_dtype = .f16,
                    }, true);
                }

                // Advance stages `k = 1 .. num_stages - 1` along the accepted prefix in the SAME
                // Metal 4 command buffer, chaining `(h_{k-1}[i-1], tok[i])` into stage `k`'s KV cache.
                for (1..self.layers.len) |stage_k| {
                    const prev_h_slot = self.stagePrevHBinding(stage_k - 1);
                    const row0_src = if (base_pos > 0) prev_h_slot else self.x_bind;
                    try target.moe_pipes.encodeCopyStrided(&frame, .{
                        .src = row0_src,
                        .dst = self.h_in_bind,
                        .num_rows = 1,
                        .copy_cols = h,
                        .src_stride = h,
                        .dst_stride = h,
                    }, true);
                    if (m > 1) {
                        const h_in_row1: metal.BufferBinding = .{
                            .buffer = self.scratch_buf,
                            .offset = self.h_in_bind.offset + row_bytes,
                        };
                        try target.moe_pipes.encodeCopyStrided(&frame, .{
                            .src = self.x_bind,
                            .dst = h_in_row1,
                            .num_rows = m - 1,
                            .copy_cols = h,
                            .src_stride = h,
                            .dst_stride = h,
                        }, false);
                    }
                    try target.moe_pipes.encodeCopyStrided(&frame, .{
                        .src = last_src,
                        .dst = prev_h_slot,
                        .num_rows = 1,
                        .copy_cols = h,
                        .src_stride = h,
                        .dst_stride = h,
                    }, true);

                    try self.encodeMtpStageFusion(target, &frame, stage_k, self.h_in_bind, m, true);
                    try self.encodeDraftLayerBody(target, &frame, stage_k, m, slot_start, base_pos, total_tree_nodes, commit_kv, false);
                }
            } else {
                // Speculative tree/chain step for explicit predictor stage `active_stage_idx`.
                try self.encodeDraftLayerBody(target, &frame, active_stage_idx, m, slot_start, base_pos, total_tree_nodes, false, active_stage_idx == 0 and use_ane_o_proj);

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

                if (top_k > 0) {
                    const eff_k = @min(top_k, self.config.draft_vocab_size);
                    const fn_k = self.stageFinalNorm(active_stage_idx);
                    try target.attn_pipes.encodeRmsNorm(&frame, .{
                        .x = self.x_bind,
                        .w = fn_k.binding,
                        .y = self.x_norm_bind,
                        .num_rows = m,
                        .hidden_dim = h,
                        .eps = self.config.rms_norm_eps,
                        .w_dtype = fn_k.dtype,
                    }, true);
                    try self.encodeDraftLinearBatched(target, &frame, self.lm_head, self.x_norm_bind, self.logits_bind, m, true);
                    try target.moe_pipes.encodeTopKLogitsRows(&frame, .{
                        .logits = self.logits_bind,
                        .out_candidates = self.topk_bind,
                        .num_rows = m,
                        .vocab_size = self.config.draft_vocab_size,
                        .top_k = eff_k,
                        .logits_dtype = .f16,
                    }, true);
                }
            }
        } else {
            // Legacy stacked draft transformer layers (EAGLE-3, EAGLE-2, 1L MTP).
            for (0..self.layers.len) |l| {
                try self.encodeDraftLayerBody(target, &frame, l, m, slot_start, base_pos, total_tree_nodes, commit_kv, use_ane_o_proj);
            }

            // 5. Persist feedback hidden state `h_prenorm` into `hidden_pool`:
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
                try self.encodeDraftLinearBatched(target, &frame, self.lm_head, self.x_norm_bind, self.logits_bind, logit_rows, true);
                try target.moe_pipes.encodeTopKLogitsRows(&frame, .{
                    .logits = self.logits_bind,
                    .out_candidates = self.topk_bind,
                    .num_rows = logit_rows,
                    .vocab_size = self.config.draft_vocab_size,
                    .top_k = eff_k,
                    .logits_dtype = .f16,
                }, true);
            }
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
        const dev_info = try model.device.info();
        const can_ane = (dev_info.aneCoreCount() > 0) and
            (q_dim % metal.KV_TILE_ROW_ALIGN_F16 == 0) and
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

fn validateSink1D(entry: *const amf.TensorEntry, dim: u32) DraftLoadError!u32 {
    const dt = try entry.dataType();
    if (entry.rank != 1 or entry.shape[0] != dim) return error.UnexpectedTensorShape;
    return switch (dt) {
        .f16 => 0,
        .bf16 => 1,
        .f32 => 2,
        else => error.UnexpectedTensorDType,
    };
}

fn findLayerSinkOpt(parsed: *const amf.Parsed, l: usize, buf: *[64]u8) ?*const amf.TensorEntry {
    if (parsed.find(std.fmt.bufPrint(buf, "layers.{d}.self_attn.sink_bias", .{l}) catch return null)) |e| return e;
    if (parsed.find(std.fmt.bufPrint(buf, "layers.{d}.self_attn.attention_sink_bias", .{l}) catch return null)) |e| return e;
    if (parsed.find(std.fmt.bufPrint(buf, "layers.{d}.self_attn.sinks", .{l}) catch return null)) |e| return e;
    return null;
}

fn findLayerPostNormReq(parsed: *const amf.Parsed, l: usize, buf: *[64]u8) DraftLoadError!*const amf.TensorEntry {
    if (parsed.find(std.fmt.bufPrint(buf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable)) |e| return e;
    if (parsed.find(std.fmt.bufPrint(buf, "layers.{d}.pre_mlp_layernorm.weight", .{l}) catch unreachable)) |e| return e;
    return error.MissingTensor;
}

fn validateLinear2D(entry: *const amf.TensorEntry, n: u32, k: u32) DraftLoadError!quant.WeightDType {
    const dt = try entry.dataType();
    if (entry.rank != 2 or entry.shape[0] != n or entry.shape[1] != k) return error.UnexpectedTensorShape;
    const layout = try entry.tensorLayout();
    if (layout == .sme2_panel64) {
        if ((k % sme.GROUP_SIZE) != 0) return error.UnalignedK;
        _ = sme.panelDTypeFromAmf(dt) catch return error.UnexpectedTensorDType;
        return dequant.legacyWeightDTypeForAmf(dt);
    }
    try dequant.validateLinearAmfShape(dt, n, k, entry.group_size);
    return dequant.legacyWeightDTypeForAmf(dt);
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
        return .{
            .binding = self.asBinding(),
            .dtype = dtype,
            .host_bytes = self.slice[0..@intCast(self.entry.data_bytes)],
        };
    }

    fn asLinearBinding(self: LoadedBlob, dtype: quant.WeightDType) qwen3.LinearBinding {
        const amf_dt = self.entry.dataType() catch dequant.weightDTypeToAmf(dtype);
        const layout = self.entry.tensorLayout() catch .row_major;
        const param_kind = self.entry.quantParamKind() catch .affine_scale_bias;
        const has_params = (layout != .sme2_panel64 and self.entry.params_bytes > 0);
        const sb: ?metal.BufferBinding = if (has_params)
            .{ .buffer = self.buffer, .offset = self.offset + self.entry.params_offset }
        else
            null;
        const sb_host: ?[]const u8 = if (has_params)
            self.slice[@intCast(self.entry.params_offset)..][0..@intCast(self.entry.params_bytes)]
        else
            null;
        return .{
            .weight = .{ .buffer = self.buffer, .offset = self.offset },
            .scale_bias = sb,
            .n = self.entry.shape[0],
            .k = self.entry.shape[1],
            .dtype = dtype,
            .amf_dtype = amf_dt,
            .group_size = self.entry.group_size,
            .tensor_scale_per_row = dequant.isTensorScalePerRow(amf_dt, self.entry.shape[0], self.entry.params_bytes),
            .alg_bytes = self.entry.data_bytes + self.entry.params_bytes,
            .layout = layout,
            .param_kind = param_kind,
            .sme_packed = if (layout == .sme2_panel64) self.slice[0..@intCast(self.entry.data_bytes)] else null,
            .weight_host = self.slice[0..@intCast(self.entry.data_bytes)],
            .scale_bias_host = sb_host,
        };
    }
};

fn normToSmeSlice(nb: qwen3.NormBinding, h: u32) sme.NormWeightSlice {
    const u16_slice: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, nb.host_bytes[0 .. @as(usize, h) * 2]));
    return switch (nb.dtype) {
        .f16 => .{ .f16 = @ptrCast(u16_slice) },
        .bf16 => .{ .bf16 = u16_slice },
    };
}

fn cpuEmbedLookupRowsWithScratch(
    embed: qwen3.LinearBinding,
    token_ids: []const u32,
    h: u32,
    out: []f16,
    scratch_f32: ?[]f32,
) void {
    const emb_amf = embed.effectiveAmfDType();
    if (dequant.usesQuantFormats(emb_amf, embed.group_size)) {
        const qf_dt = quant_formats.DType.fromAmf(emb_amf) orelse return;
        const dim: usize = h;
        const p_bytes: []const u8 = embed.scale_bias_host orelse &.{};
        if (scratch_f32) |tmp| {
            if (tmp.len >= dim) {
                for (token_ids, 0..) |tid, r| {
                    quant_formats.dequantRowCpuRef(
                        qf_dt,
                        embed.group_size,
                        embed.weight_host,
                        p_bytes,
                        tid,
                        embed.n,
                        dim,
                        tmp,
                    ) catch return;
                    const dst = out[r * dim ..][0..dim];
                    for (0..dim) |d| dst[d] = @floatCast(tmp[d]);
                }
                return;
            }
        }
        return;
    }
    cpuEmbedLookupRows(embed, token_ids, h, out);
}

/// Zero-allocation CPU embedding lookup for `m <= 64` rows when `Placement.cpu_sme2`
/// executes `MTP` dual-norm `eh_proj` or `EAGLE-2` `fc` before opening the Metal frame.
fn cpuEmbedLookupRows(embed: qwen3.LinearBinding, token_ids: []const u32, h: u32, out: []f16) void {
    const dim: usize = h;
    const raw = embed.weight_host;
    switch (embed.dtype) {
        .f16 => {
            const src: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, raw));
            for (token_ids, 0..) |tid, r| {
                @memcpy(out[r * dim ..][0..dim], src[@as(usize, tid) * dim ..][0..dim]);
            }
        },
        .bf16 => {
            const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
            for (token_ids, 0..) |tid, r| {
                const row_s = src[@as(usize, tid) * dim ..][0..dim];
                const row_d = out[r * dim ..][0..dim];
                for (0..dim) |d| {
                    row_d[d] = @floatCast(quant.bf16ToF32(row_s[d]));
                }
            }
        },
        .q4_affine, .q3_affine, .q2_affine => {
            const sb_raw = embed.scale_bias_host orelse return;
            const groups_per_row: usize = dim / 64;
            const group_bytes: usize = switch (embed.dtype) {
                .q4_affine => 32,
                .q3_affine => 24,
                .q2_affine => 16,
                else => unreachable,
            };
            const row_q_bytes = groups_per_row * group_bytes;
            const row_sb_bytes = groups_per_row * 4;
            for (token_ids, 0..) |tid, r| {
                const q_row = raw[@as(usize, tid) * row_q_bytes ..][0..row_q_bytes];
                const sb_row = sb_raw[@as(usize, tid) * row_sb_bytes ..][0..row_sb_bytes];
                const dst = out[r * dim ..][0..dim];
                for (0..groups_per_row) |g| {
                    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, sb_row[g * 4 ..][0..2], .little))));
                    const bias: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, sb_row[g * 4 + 2 ..][0..2], .little))));
                    const gq = q_row[g * group_bytes ..][0..group_bytes];
                    switch (embed.dtype) {
                        .q4_affine => {
                            for (0..32) |bi| {
                                const byte = gq[bi];
                                dst[g * 64 + bi * 2] = @floatCast(@as(f32, @floatFromInt(byte & 0x0F)) * scale + bias);
                                dst[g * 64 + bi * 2 + 1] = @floatCast(@as(f32, @floatFromInt(byte >> 4)) * scale + bias);
                            }
                        },
                        .q3_affine => {
                            for (0..8) |blk| {
                                const b = gq[blk * 3 ..][0..3];
                                const w: u32 = @as(u32, b[0]) | (@as(u32, b[1]) << 8) | (@as(u32, b[2]) << 16);
                                for (0..8) |lane| {
                                    const code: u32 = (w >> @intCast(lane * 3)) & 0x7;
                                    dst[g * 64 + blk * 8 + lane] = @floatCast(@as(f32, @floatFromInt(code)) * scale + bias);
                                }
                            }
                        },
                        .q2_affine => {
                            for (0..16) |bi| {
                                const byte = gq[bi];
                                for (0..4) |lane| {
                                    const code: u32 = (byte >> @intCast(lane * 2)) & 0x3;
                                    dst[g * 64 + bi * 4 + lane] = @floatCast(@as(f32, @floatFromInt(code)) * scale + bias);
                                }
                            }
                        },
                        else => unreachable,
                    }
                }
            }
        },
    }
}

/// Packs a compact L2-resident (`<= host P-cluster L2`) `f16`/`bf16` projection
/// (`fc` or `eh_proj`) into `sme2_panel64` once at startup and builds
/// `SmeProjectionRuntime` (`D13`, `C16`). Returns `null` on `M1–M3` (`!has_sme2`)
/// or when the packed projection exceeds the host P-cluster L2 budget.
fn tryInitSmeProjectionRuntime(
    allocator: std.mem.Allocator,
    blob: LoadedBlob,
    dtype: quant.WeightDType,
    kind: SmeProjectionKind,
    max_m: u32,
    draft_budget: budget_mod.Budget,
) DraftLoadError!?*SmeProjectionRuntime {
    const caps = try sme.SmeCaps.probe();
    if (!caps.supportsSme2()) return null;

    const n: usize = blob.entry.shape[0];
    const k: usize = blob.entry.shape[1];
    if (k == 0 or (k % sme.GROUP_SIZE) != 0) return null;
    const layout = blob.entry.tensorLayout() catch return null;
    const l2_limit = caps.draftL2BudgetBytes();

    var weight: sme.SmeLinearWeight = undefined;
    var packed_bytes: usize = 0;
    var w_arena: arena_mod.WiredArena = undefined;

    if (layout == .sme2_panel64) {
        const sw = sme.SmeLinearWeight.fromAmfEntry(blob.entry, blob.slice, .temporal) catch return null;
        packed_bytes = sw.byteSize();
        if (packed_bytes > l2_limit) return null;
        const aligned_w = try page.alignUp(packed_bytes);
        draft_budget.checkPlan(aligned_w) catch return null;
        w_arena = arena_mod.WiredArena.init(aligned_w, draft_budget) catch return null;
        errdefer w_arena.deinit();
        const w_raw = try w_arena.allocPages(packed_bytes);
        @memcpy(w_raw[0..packed_bytes], sw.w_packed);
        weight = .{
            .w_packed = w_raw[0..packed_bytes],
            .n = sw.n,
            .k = sw.k,
            .dtype = sw.dtype,
            .param_kind = sw.param_kind,
            .cache_hint = .temporal,
        };
    } else {
        const raw_amf_dt = blob.entry.dataType() catch return null;
        if (dequant.usesQuantFormats(raw_amf_dt, blob.entry.group_size)) return null;
        switch (dtype) {
            .f16, .bf16 => {
                const panel_dt: sme.PanelDType = if (dtype == .bf16) .bf16 else .f16;
                packed_bytes = sme.packedBlobBytes(panel_dt, n, k) catch return null;
                if (packed_bytes > l2_limit) return null;
                const aligned_w = try page.alignUp(packed_bytes);
                draft_budget.checkPlan(aligned_w) catch return null;
                w_arena = arena_mod.WiredArena.init(aligned_w, draft_budget) catch return null;
                errdefer w_arena.deinit();
                const w_raw = try w_arena.allocPages(packed_bytes);
                const src_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, blob.slice[0 .. n * k * 2]));
                try sme.packDensePanel64(panel_dt, src_u16, n, k, w_raw[0..packed_bytes]);
                weight = .{
                    .w_packed = w_raw[0..packed_bytes],
                    .n = n,
                    .k = k,
                    .dtype = panel_dt,
                    .param_kind = .affine_scale_bias,
                    .cache_hint = .temporal,
                };
            },
            .q2_affine => {
                const OpsQ2 = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                packed_bytes = sme.packedBlobBytes(.q2_affine, n, k) catch return null;
                if (packed_bytes > l2_limit) return null;
                const groups = OpsQ2.groupsPerRow(k) catch return null;
                const row_bytes = OpsQ2.rowWeightBytes(k) catch return null;
                const need_w = n * row_bytes;
                const need_sb = n * groups * @sizeOf(quant.ScaleBias);
                const p_off: usize = @intCast(blob.entry.params_offset);
                if (blob.slice.len < need_w or blob.slice.len < p_off + need_sb) return null;

                const aligned_w = try page.alignUp(packed_bytes);
                draft_budget.checkPlan(aligned_w) catch return null;
                w_arena = arena_mod.WiredArena.init(aligned_w, draft_budget) catch return null;
                errdefer w_arena.deinit();
                const w_raw = try w_arena.allocPages(packed_bytes);

                const sb: []const quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
                    quant.ScaleBias,
                    blob.slice[p_off .. p_off + need_sb],
                ));
                try sme.packMlxQuantPanel64(.q2, blob.slice[0..need_w], sb, n, k, w_raw[0..packed_bytes]);
                weight = .{
                    .w_packed = w_raw[0..packed_bytes],
                    .n = n,
                    .k = k,
                    .dtype = .q2_affine,
                    .param_kind = .affine_scale_bias,
                    .cache_hint = .temporal,
                };
            },
            else => return null,
        }
    }
    errdefer w_arena.deinit();
    var proj = try sme.SmeDraftProjection.init(weight, .{
        .max_m = @min(@as(usize, max_m), sme.MAX_STEP_LORA_M),
        .l2_limit_bytes = l2_limit,
        .budget_override = draft_budget,
    });
    errdefer proj.deinit();

    var disp = try sme.SmeDispatcher.init(allocator, .{ .caps_override = caps });
    errdefer disp.deinit();

    const cal_m: usize = @min(@as(usize, max_m), 4);
    const cal_out_f16 = try allocator.alloc(f16, cal_m * n);
    errdefer allocator.free(cal_out_f16);
    @memset(cal_out_f16, 0);

    const rt = try allocator.create(SmeProjectionRuntime);
    rt.* = .{
        .weights_arena = w_arena,
        .proj = proj,
        .dispatcher = disp,
        .cal_out_f16 = cal_out_f16,
        .weight_bytes = packed_bytes,
        .kind = kind,
    };
    return rt;
}

/// D13 startup benchmark gate (`ADR-0008`, `ADR-0019`): compares P-core SME2
/// `SmeDraftProjection` against a Metal 4 GPU projection callback using
/// `sme.calibrateSmeVsGpuFn` (median-of-N, zero hot-path allocations).
fn calibrateD13SmeVsGpu(
    target: anytype,
    sp: *SmeProjectionRuntime,
    gpu_lin: qwen3.LinearBinding,
    in_bind: metal.BufferBinding,
    out_bind: metal.BufferBinding,
    cal_m: u32,
    timeout_ms: u64,
) DraftLoadError!sme.SmeVsGpuCalibration {
    const TargetPtr = @TypeOf(target);
    const GpuCtx = struct {
        tgt: TargetPtr,
        lin: qwen3.LinearBinding,
        in_b: metal.BufferBinding,
        out_b: metal.BufferBinding,
        m: u32,
        timeout: u64,
    };
    var ctx: GpuCtx = .{
        .tgt = target,
        .lin = gpu_lin,
        .in_b = in_bind,
        .out_b = out_bind,
        .m = cal_m,
        .timeout = timeout_ms,
    };
    const stepFn = struct {
        fn run(raw_ctx: *anyopaque) anyerror!void {
            const c: *GpuCtx = @ptrCast(@alignCast(raw_ctx));
            var frame = try c.tgt.compute.beginFrame(c.timeout);
            errdefer frame.abort();
            try c.tgt.encodeLinearBatched(&frame, c.lin, c.in_b, c.out_b, c.m, false);
            const sig = try frame.commit();
            try c.tgt.compute.wait(sig, c.timeout);
        }
    }.run;

    const out_bytes = std.mem.sliceAsBytes(sp.cal_out_f16[0 .. @as(usize, cal_m) * sp.proj.weight.n]);
    return sme.calibrateSmeVsGpuFn(
        &sp.dispatcher,
        &sp.proj,
        cal_m,
        out_bytes,
        &ctx,
        &stepFn,
        .{
            .warmups = 4,
            .iters = 12,
            .out_dtype = .f16,
        },
    ) catch |err| switch (err) {
        error.InvalidArgument => return error.InvalidConfig,
        error.BufferTooSmall => return error.InvalidConfig,
        error.Timeout => return error.Timeout,
        else => return error.ComputeFailed,
    };
}

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

/// Pure per-chip `D5` ANE-vs-GPU winner selector (`ADR-0008`, `C16`).
/// Prefers `.ane` strictly when both timings are finite/positive and `ane_us < gpu_us`;
/// otherwise stays on `.gpu` with zero chip-generation hardcoding.
pub fn selectD5Winner(ane_us: f64, gpu_us: f64) Placement {
    if (std.math.isFinite(ane_us) and ane_us > 0.0 and
        std.math.isFinite(gpu_us) and gpu_us > 0.0 and
        ane_us < gpu_us)
    {
        return .ane;
    }
    return .gpu;
}

/// Pure multi-generation `D5` + `D13` auto-placement selector (`ADR-0008`, `ADR-0019`, `C16`).
/// Handles `M1–M3` (`!has_sme2`), `M4` (`has_sme2`, `!has_nax`), and `M5+`
/// (`has_sme2`, `has_nax`) purely from host capabilities and measured latencies.
pub fn selectAutoPlacement(
    can_ane: bool,
    d5_ane_us: f64,
    d5_gpu_us: f64,
    has_sme2: bool,
    sme_prefer: bool,
    sme_us: f64,
) Placement {
    const d5_winner: Placement = if (can_ane) selectD5Winner(d5_ane_us, d5_gpu_us) else .gpu;
    const best_d5_us: f64 = if (can_ane) @min(d5_ane_us, d5_gpu_us) else std.math.inf(f64);
    if (has_sme2 and sme_prefer and std.math.isFinite(sme_us) and sme_us > 0.0 and
        (!can_ane or sme_us <= best_d5_us))
    {
        return .cpu_sme2;
    }
    return d5_winner;
}

/// Startup D5 placement micro-calibration (ADR-0008): compares zero-copy
/// `GPU -> IOSurface -> ANE -> IOSurface -> GPU` round-trip latency against
/// in-command-buffer GPU `encodeLinearGemv` on `layers[0].o_proj`.
fn calibrateD5Placement(
    target: anytype,
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

    const winner: Placement = selectD5Winner(ane_us, gpu_us);
    return .{ .ane_us = ane_us, .gpu_us = gpu_us, .winner = winner };
}

fn loadInternal(
    allocator: std.mem.Allocator,
    target: anytype,
    source: AmfSource,
    config_in: DraftConfig,
    options: DraftLoadOptions,
) DraftLoadError!DraftHead {
    const parsed = source.parsed();
    var config = config_in;
    if (config.format == .mtp and !config.mtp_chained) {
        if (findLinearOpt(parsed, "eh_proj") == null and findLinearOpt(parsed, "layers.0.eh_proj") != null) {
            config.mtp_chained = true;
        }
    }
    try config.validate();
    if (options.max_seq_len == 0 or options.chunk_tokens == 0 or
        options.max_tree_slots == 0 or options.max_tree_slots > attention.MAX_TREE_NODES)
    {
        return error.InvalidConfig;
    }
    if (config.vocab_size != target.config.vocab_size or config.target_hidden_size != target.config.hidden_size) {
        return error.TargetIncompatible;
    }

    const num_layers: usize = config.num_hidden_layers;
    const h = config.hidden_size;
    const h_tgt = config.target_hidden_size;
    const q_dim = config.qDim();
    const kv_dim = config.kvDim();
    const v_dim = config.vDim();
    const o_in_dim = config.oInDim();
    const is_mla = config.isMlaMtp();
    const is_moe = config.isMoeMtp();
    const qk_hd = config.effectiveQkHeadDim();
    const kv_heads = config.effectiveKvHeads();
    const eff_mlp_inter = config.effectiveMlpInter();
    const chained_mtp = (config.format == .mtp and config.mtp_chained);
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
        .l1_only, .small_draft, .self_recurrent, .dspark, .dflare, .dfly2 => unreachable,
    };

    const enorm_entry: ?*const amf.TensorEntry = if (config.format == .mtp and !chained_mtp) blk: {
        const e = parsed.find("enorm.weight") orelse parsed.find("layers.0.enorm.weight") orelse return error.MissingTensor;
        _ = try validateNorm1D(e, h);
        weights_bytes += e.blob_bytes;
        break :blk e;
    } else null;
    const hnorm_entry: ?*const amf.TensorEntry = if (config.format == .mtp and !chained_mtp) blk: {
        const e = parsed.find("hnorm.weight") orelse parsed.find("layers.0.hnorm.weight") orelse return error.MissingTensor;
        _ = try validateNorm1D(e, h);
        weights_bytes += e.blob_bytes;
        break :blk e;
    } else null;
    const eh_proj_entry: ?*const amf.TensorEntry = if (config.format == .mtp and !chained_mtp) blk: {
        const e = findLinearOpt(parsed, "eh_proj") orelse findLinearOpt(parsed, "layers.0.eh_proj") orelse return error.MissingTensor;
        _ = try validateLinear2D(e, h, 2 * h);
        weights_bytes += e.blob_bytes;
        break :blk e;
    } else null;

    const root_final_norm_entry: ?*const amf.TensorEntry = if (config.format == .mtp)
        parsed.find("shared_head.norm.weight") orelse parsed.find("layers.0.final_layernorm.weight") orelse parsed.find("norm.weight")
    else
        try findRequired(parsed, "norm.weight");
    if (root_final_norm_entry) |fne| {
        _ = try validateNorm1D(fne, h);
        weights_bytes += fne.blob_bytes;
    } else if (!chained_mtp) {
        return error.MissingTensor;
    }

    var name_buf: [64]u8 = undefined;
    var any_layer_has_sink_bias = false;
    var max_shared_inter: u32 = eff_mlp_inter;
    for (0..num_layers) |l| {
        if (chained_mtp) {
            const sen = parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.enorm.weight", .{l}) catch unreachable) orelse
                if (l == 0) try findRequired(parsed, "enorm.weight") else return error.MissingTensor;
            _ = try validateNorm1D(sen, h);
            weights_bytes += sen.blob_bytes;

            const shn = parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.hnorm.weight", .{l}) catch unreachable) orelse
                if (l == 0) try findRequired(parsed, "hnorm.weight") else return error.MissingTensor;
            _ = try validateNorm1D(shn, h);
            weights_bytes += shn.blob_bytes;

            const seh = findLinearOpt(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.eh_proj", .{l}) catch unreachable) orelse
                if (l == 0) try findLinearReq(parsed, "eh_proj") else return error.MissingTensor;
            _ = try validateLinear2D(seh, h, 2 * h);
            weights_bytes += seh.blob_bytes;

            if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.final_layernorm.weight", .{l}) catch unreachable)) |sfn| {
                _ = try validateNorm1D(sfn, h);
                weights_bytes += sfn.blob_bytes;
            } else if (root_final_norm_entry == null) {
                return error.MissingTensor;
            }
        }

        const qkv_in: u32 = if (config.format == .eagle3 and l == 0) 2 * h else h;
        const in_ln = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable);
        _ = try validateNorm1D(in_ln, h);
        weights_bytes += in_ln.blob_bytes;

        if (config.format == .eagle3 and l == 0) {
            const hn = try findRequired(parsed, "layers.0.hidden_norm.weight");
            _ = try validateNorm1D(hn, h);
            weights_bytes += hn.blob_bytes;
        }
        if (is_mla) {
            const qae = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_a_proj", .{l}) catch unreachable);
            _ = try validateLinear2D(qae, config.q_lora_rank, h);
            weights_bytes += qae.blob_bytes;

            const qan = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_a_layernorm.weight", .{l}) catch unreachable);
            _ = try validateNorm1D(qan, config.q_lora_rank);
            weights_bytes += qan.blob_bytes;

            const qbe = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_b_proj", .{l}) catch unreachable);
            _ = try validateLinear2D(qbe, q_dim, config.q_lora_rank);
            weights_bytes += qbe.blob_bytes;

            const kvae = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.kv_a_proj_with_mqa", .{l}) catch unreachable);
            _ = try validateLinear2D(kvae, config.kv_lora_rank + config.qk_rope_head_dim, h);
            weights_bytes += kvae.blob_bytes;

            const kvan = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.kv_a_layernorm.weight", .{l}) catch unreachable);
            _ = try validateNorm1D(kvan, config.kv_lora_rank);
            weights_bytes += kvan.blob_bytes;

            const kvb_n = config.num_attention_heads * (config.qk_nope_head_dim + config.effectiveVHeadDim());
            const kvbe = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.kv_b_proj", .{l}) catch unreachable);
            _ = try validateLinear2D(kvbe, kvb_n, config.kv_lora_rank);
            weights_bytes += kvbe.blob_bytes;
        } else {
            const qe = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable);
            _ = try validateLinear2D(qe, q_dim, qkv_in);
            weights_bytes += qe.blob_bytes;

            const ke = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable);
            _ = try validateLinear2D(ke, kv_dim, qkv_in);
            weights_bytes += ke.blob_bytes;

            const ve = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable);
            _ = try validateLinear2D(ve, v_dim, qkv_in);
            weights_bytes += ve.blob_bytes;
        }

        const oe = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable);
        _ = try validateLinear2D(oe, h, o_in_dim);
        weights_bytes += oe.blob_bytes;

        if (!is_mla) {
            const qn = parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable);
            const kn = parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable);
            if ((qn != null) != (kn != null)) return error.MissingTensor;
            if (qn) |qne| {
                const qdt = try validateNorm1D(qne, config.head_dim);
                const kdt = try validateNorm1D(kn.?, config.head_dim);
                if (qdt != kdt) return error.UnexpectedTensorDType;
                weights_bytes += qne.blob_bytes + kn.?.blob_bytes;
            }
        }

        if (findLayerSinkOpt(parsed, l, &name_buf)) |se| {
            _ = try validateSink1D(se, config.num_attention_heads);
            weights_bytes += se.blob_bytes;
            any_layer_has_sink_bias = true;
        }

        const pln = try findLayerPostNormReq(parsed, l, &name_buf);
        _ = try validateNorm1D(pln, h);
        weights_bytes += pln.blob_bytes;

        if (is_moe) {
            const rt = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate", .{l}) catch unreachable);
            _ = try validateLinear2D(rt, config.num_experts, h);
            weights_bytes += rt.blob_bytes;
            if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate.bias", .{l}) catch unreachable)) |rb| {
                _ = try validateSink1D(rb, config.num_experts);
                weights_bytes += rb.blob_bytes;
            }
            if (findLinearOpt(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_experts.gate_proj", .{l}) catch unreachable)) |sgp| {
                const s_inter = sgp.shape[0];
                if (s_inter == 0 or (s_inter % 4) != 0) return error.UnexpectedTensorShape;
                _ = try validateLinear2D(sgp, s_inter, h);
                const sup = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_experts.up_proj", .{l}) catch unreachable);
                _ = try validateLinear2D(sup, s_inter, h);
                const sdp = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_experts.down_proj", .{l}) catch unreachable);
                _ = try validateLinear2D(sdp, h, s_inter);
                weights_bytes += sgp.blob_bytes + sup.blob_bytes + sdp.blob_bytes;
                if (s_inter > max_shared_inter) max_shared_inter = s_inter;
            }
            weights_bytes += amf.align64(@as(usize, config.num_experts) * @sizeOf(MtpExpertMeta)) + 64;
            for (0..config.num_experts) |e_idx| {
                const eg = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e_idx }) catch unreachable);
                _ = try validateLinear2D(eg, config.moe_intermediate_size, h);
                const eu = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e_idx }) catch unreachable);
                _ = try validateLinear2D(eu, config.moe_intermediate_size, h);
                const ed = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e_idx }) catch unreachable);
                _ = try validateLinear2D(ed, h, config.moe_intermediate_size);
                weights_bytes += eg.blob_bytes + eu.blob_bytes + ed.blob_bytes + 192;
            }
        } else {
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
    }

    const use_custom_attn = config.usesCustomMtpAttn() or any_layer_has_sink_bias;
    const weights_arena_bytes = try page.alignUp(@max(@as(usize, @intCast(weights_bytes)) + 4 * page.APPLE_PAGE_SIZE, page.APPLE_PAGE_SIZE));

    // --- 2. Compute draft scratch & KV cache footprint and enforce remaining wired budget ---
    const m_cap: usize = options.max_tree_slots;
    const kv_tbl_per_layer = if (use_custom_attn) 0 else amf.align64(@as(usize, num_chunks) * @sizeOf(attention.KvTilePair));
    const tree_kv_layer_stride = amf.align64(m_cap * @as(usize, kv_dim) * @sizeOf(f16));
    const tree_v_layer_stride = amf.align64(m_cap * @as(usize, v_dim) * @sizeOf(f16));
    const kv_ring_cap: u32 = if (use_custom_attn)
        (if (config.sliding_window > 0) config.sliding_window else max_seq_len)
    else
        0;
    const kv_ring_k_bytes: usize = if (use_custom_attn)
        amf.align64(num_layers * @as(usize, kv_ring_cap) * @as(usize, kv_dim) * @sizeOf(f16))
    else
        64;
    const kv_ring_v_bytes: usize = if (use_custom_attn)
        amf.align64(num_layers * @as(usize, kv_ring_cap) * @as(usize, v_dim) * @sizeOf(f16))
    else
        64;
    const gqa_tree_bytes: usize = if (use_custom_attn)
        64
    else
        amf.align64(attention.gqaTreeScratchLayout(
            options.max_tree_slots,
            config.num_attention_heads,
            config.head_dim,
            attention.MAX_SPLITS,
        ).total_bytes);

    const mla_q_a_bytes: usize = if (is_mla) amf.align64(m_cap * @as(usize, config.q_lora_rank) * @sizeOf(f16)) else 64;
    const mla_kv_a_bytes: usize = if (is_mla) amf.align64(m_cap * @as(usize, config.kv_lora_rank + config.qk_rope_head_dim) * @sizeOf(f16)) else 64;
    const mla_kv_pass_bytes: usize = if (is_mla) amf.align64(m_cap * @as(usize, config.kv_lora_rank) * @sizeOf(f16)) else 64;
    const mla_kv_b_bytes: usize = if (is_mla) amf.align64(m_cap * @as(usize, config.num_attention_heads) * @as(usize, config.qk_nope_head_dim + config.effectiveVHeadDim()) * @sizeOf(f16)) else 64;
    const moe_router_bytes: usize = if (is_moe) amf.align64(m_cap * @as(usize, config.num_experts) * @sizeOf(f16)) else 64;
    const moe_ids_bytes: usize = if (is_moe) amf.align64(m_cap * @as(usize, config.num_experts_per_tok) * @sizeOf(u32)) else 64;
    const moe_w_bytes: usize = if (is_moe) amf.align64(m_cap * @as(usize, config.num_experts_per_tok) * @sizeOf(f32)) else 64;
    const moe_act_bytes: usize = if (is_moe) amf.align64(m_cap * @as(usize, config.num_experts_per_tok) * @as(usize, config.moe_intermediate_size) * @sizeOf(f16)) else 64;
    const max_sparse_indices: u32 = if (use_custom_attn) @max(if (config.index_topk > 0) @min(config.index_topk, max_seq_len) else @min(max_seq_len, 2048), 16) else 16;
    const sparse_idx_bytes: usize = amf.align64(@as(usize, max_sparse_indices) * @sizeOf(u32));
    const mlp_scratch_inter: usize = @max(@as(usize, max_shared_inter), @as(usize, eff_mlp_inter));

    const scratch_raw: usize =
        amf.align64(m_cap * @sizeOf(u32)) * 4 + // token_ids, parent_slots, positions, positions_pool
        amf.align64(m_cap * @sizeOf(u64)) + // ancestor_masks
        amf.align64((1 + m_cap) * @as(usize, h) * @sizeOf(f16)) + // hidden_pool
        amf.align64(MAX_MTP_STAGES * @as(usize, h) * @sizeOf(f16)) + // stage_h_prev
        amf.align64(m_cap * @as(usize, h) * @sizeOf(f16)) * 7 + // h_in, emb, e_norm, h_norm, x, x_norm, proj_out
        amf.align64(m_cap * 2 * @as(usize, h) * @sizeOf(f16)) + // x_cat
        amf.align64(m_cap * @as(usize, q_dim) * @sizeOf(f16)) + // q_pool
        amf.align64(m_cap * @as(usize, o_in_dim) * @sizeOf(f16)) + // attn_pool
        amf.align64(m_cap * @as(usize, kv_dim) * @sizeOf(f16)) + // k
        amf.align64(m_cap * @as(usize, v_dim) * @sizeOf(f16)) + // v
        amf.align64(m_cap * mlp_scratch_inter * @sizeOf(f16)) * 3 + // mlp gate/up/act
        amf.align64(m_cap * @as(usize, config.draft_vocab_size) * @sizeOf(f16)) + // logits
        amf.align64(m_cap * MAX_DRAFT_TOP_K * @sizeOf(moe.TopKCandidate)) + // topk
        mla_q_a_bytes * 2 +
        mla_kv_a_bytes +
        mla_kv_pass_bytes * 2 +
        mla_kv_b_bytes +
        moe_router_bytes +
        moe_ids_bytes +
        moe_w_bytes +
        moe_act_bytes +
        sparse_idx_bytes +
        gqa_tree_bytes +
        kv_ring_k_bytes +
        kv_ring_v_bytes +
        num_layers * kv_tbl_per_layer +
        num_layers * tree_kv_layer_stride +
        num_layers * tree_v_layer_stride +
        8 * page.APPLE_PAGE_SIZE;
    const scratch_arena_bytes = try page.alignUp(scratch_raw);

    const kv_tile_rows: u32 = kv_heads * options.chunk_tokens;
    const kv_tile_alloc_bytes: u64 = if (use_custom_attn) 0 else try page.alignUp(@as(usize, kv_tile_rows) * qk_hd * @sizeOf(f16));
    const total_kv_tiles: usize = if (use_custom_attn) 0 else num_layers * @as(usize, num_chunks) * 2;
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

    const kv_tiles: []metal.KvTile = if (total_kv_tiles > 0) try allocator.alloc(metal.KvTile, total_kv_tiles) else &.{};
    var tiles_inited: usize = 0;
    errdefer {
        for (0..tiles_inited) |i| kv_tiles[i].close();
        if (kv_tiles.len > 0) allocator.free(kv_tiles);
    }
    for (0..total_kv_tiles) |i| {
        kv_tiles[i] = try metal.KvTile.init(target.device, config.head_dim, kv_tile_rows, .{});
        tiles_inited += 1;
    }

    var mtp_pipes: ?MtpAttnPipelines = null;
    errdefer if (mtp_pipes) |mp| mp.close();
    if (use_custom_attn) {
        mtp_pipes = try MtpAttnPipelines.init(target.device, target.library);
    }

    try target.compute.makeResident(&.{ weights_buf, scratch_buf });
    errdefer target.compute.removeResident(&.{ weights_buf, scratch_buf }) catch |err| {
        std.debug.panic("removeResident failed in DraftHead errdefer: {}", .{err});
    };
    if (kv_tiles.len > 0) {
        try target.compute.makeTilesResident(kv_tiles);
    }
    errdefer if (kv_tiles.len > 0) {
        target.compute.removeTilesResident(kv_tiles) catch |err| {
            std.debug.panic("removeTilesResident failed in DraftHead errdefer: {}", .{err});
        };
    };

    // M8.1 (`G23`): compile optional `QuantFormatPipelines` strictly for any non-legacy
    // quantized tensors inside the draft `.amf` (`0` cost when draft uses legacy dtypes).
    var draft_qf_specs: [quant_formats.ALL_FORMAT_SPECS.len]quant_formats.FormatSpec = undefined;
    var draft_qf_len: usize = 0;
    var draft_embed_needs_qf = false;
    for (parsed.entries) |*e| {
        if (amf.isMetaEntry(e)) continue;
        const e_layout = try e.tensorLayout();
        if (e_layout != .row_major) continue;
        const e_dt = try e.dataType();
        if (!dequant.usesQuantFormats(e_dt, e.group_size)) continue;
        if ((try e.tensorKind()) == .embedding) draft_embed_needs_qf = true;
        const qf_dt = quant_formats.DType.fromAmf(e_dt) orelse continue;
        const spec = quant_formats.FormatSpec.normalized(qf_dt, e.group_size);
        var exists = false;
        for (draft_qf_specs[0..draft_qf_len]) |s| {
            if (s.dtype == spec.dtype and s.group_size == spec.group_size) {
                exists = true;
                break;
            }
        }
        if (!exists and draft_qf_len < draft_qf_specs.len) {
            draft_qf_specs[draft_qf_len] = spec;
            draft_qf_len += 1;
        }
    }
    var draft_qf_pipes: ?quant_formats.QuantFormatPipelines = null;
    errdefer if (draft_qf_pipes) |*qfp| {
        target.compute.removeResident(&.{qfp.codebook_buf}) catch |err| {
            std.debug.panic("removeResident(draft_qf_pipes) failed: {}", .{err});
        };
        qfp.close();
    };
    if (draft_qf_len > 0) {
        draft_qf_pipes = try quant_formats.QuantFormatPipelines.initWithOptions(target.device, target.library, .{
            .specs = draft_qf_specs[0..draft_qf_len],
            .include_moe = false,
            .include_act_quant = false,
        });
        try draft_qf_pipes.?.makeResident(target.compute);
    }
    if ((config.share_embed_tokens or embed_entry == null) and
        dequant.usesQuantFormats(target.embed.effectiveAmfDType(), target.embed.group_size))
    {
        draft_embed_needs_qf = true;
    }
    var draft_embed_dequant_f32: ?[]f32 = null;
    errdefer if (draft_embed_dequant_f32) |ed| allocator.free(ed);
    if (draft_embed_needs_qf) {
        draft_embed_dequant_f32 = try allocator.alloc(f32, h);
    }

    // --- 4. Carve scratch buffers ---
    const tok_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u32), 64);
    const par_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u32), 64);
    const pos_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u32), 64);
    const pos_pool_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u32), 64);
    const msk_raw = try scratch_arena.allocAligned(m_cap * @sizeOf(u64), 64);
    const hp_raw = try scratch_arena.allocAligned((1 + m_cap) * @as(usize, h) * @sizeOf(f16), 64);
    const stage_prev_raw = try scratch_arena.allocAligned(MAX_MTP_STAGES * @as(usize, h) * @sizeOf(f16), 64);
    const hin_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const emb_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const en_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const hn_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const xcat_raw = try scratch_arena.allocAligned(m_cap * 2 * @as(usize, h) * @sizeOf(f16), 64);
    const x_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const xn_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const qp_raw = try scratch_arena.allocAligned(m_cap * @as(usize, q_dim) * @sizeOf(f16), 64);
    const k_raw = try scratch_arena.allocAligned(m_cap * @as(usize, kv_dim) * @sizeOf(f16), 64);
    const v_raw = try scratch_arena.allocAligned(m_cap * @as(usize, v_dim) * @sizeOf(f16), 64);
    const ap_raw = try scratch_arena.allocAligned(m_cap * @as(usize, o_in_dim) * @sizeOf(f16), 64);
    const po_raw = try scratch_arena.allocAligned(m_cap * @as(usize, h) * @sizeOf(f16), 64);
    const mg_raw = try scratch_arena.allocAligned(m_cap * mlp_scratch_inter * @sizeOf(f16), 64);
    const mu_raw = try scratch_arena.allocAligned(m_cap * mlp_scratch_inter * @sizeOf(f16), 64);
    const ma_raw = try scratch_arena.allocAligned(m_cap * mlp_scratch_inter * @sizeOf(f16), 64);
    const log_raw = try scratch_arena.allocAligned(m_cap * @as(usize, config.draft_vocab_size) * @sizeOf(f16), 64);
    const tk_out_raw = try scratch_arena.allocAligned(m_cap * MAX_DRAFT_TOP_K * @sizeOf(moe.TopKCandidate), 64);
    const mla_qa_raw = try scratch_arena.allocAligned(mla_q_a_bytes, 64);
    const mla_qan_raw = try scratch_arena.allocAligned(mla_q_a_bytes, 64);
    const mla_kva_raw = try scratch_arena.allocAligned(mla_kv_a_bytes, 64);
    const mla_kvp_raw = try scratch_arena.allocAligned(mla_kv_pass_bytes, 64);
    const mla_kvn_raw = try scratch_arena.allocAligned(mla_kv_pass_bytes, 64);
    const mla_kvb_raw = try scratch_arena.allocAligned(mla_kv_b_bytes, 64);
    const moe_rt_raw = try scratch_arena.allocAligned(moe_router_bytes, 64);
    const moe_id_raw = try scratch_arena.allocAligned(moe_ids_bytes, 64);
    const moe_w_raw = try scratch_arena.allocAligned(moe_w_bytes, 64);
    const moe_act_raw = try scratch_arena.allocAligned(moe_act_bytes, 64);
    const sp_idx_raw = try scratch_arena.allocAligned(sparse_idx_bytes, 64);
    const gqa_raw = try scratch_arena.allocAligned(gqa_tree_bytes, 64);
    const kv_ring_k_raw = try scratch_arena.allocAligned(kv_ring_k_bytes, 64);
    const kv_ring_v_raw = try scratch_arena.allocAligned(kv_ring_v_bytes, 64);
    const tree_k_raw = try scratch_arena.allocAligned(num_layers * tree_kv_layer_stride, 64);
    const tree_v_raw = try scratch_arena.allocAligned(num_layers * tree_v_layer_stride, 64);
    @memset(tok_raw, 0);
    @memset(par_raw, 0);
    @memset(pos_raw, 0);
    @memset(pos_pool_raw, 0);
    @memset(msk_raw, 0);
    @memset(hp_raw, 0);
    @memset(stage_prev_raw, 0);
    @memset(tk_out_raw, 0);
    @memset(sp_idx_raw, 0);
    if (use_custom_attn) {
        @memset(kv_ring_k_raw, 0);
        @memset(kv_ring_v_raw, 0);
    }

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

    var fc_blob_opt: ?LoadedBlob = null;
    var fc_dt_opt: ?quant.WeightDType = null;
    const fc_lin: ?qwen3.LinearBinding = if (fc_entry) |fe| blk: {
        const expected_k: u32 = if (config.format == .eagle3) 3 * h_tgt else 2 * h;
        const fdt = try validateLinear2D(fe, h, expected_k);
        const fblob = try loadBlobIntoArena(source, fe, &weights_arena, weights_buf, options.verify_crcs);
        fc_blob_opt = fblob;
        fc_dt_opt = fdt;
        break :blk fblob.asLinearBinding(fdt);
    } else null;

    var enorm_b: ?qwen3.NormBinding = if (enorm_entry) |ee|
        (try loadBlobIntoArena(source, ee, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(ee, h))
    else
        null;
    var hnorm_b: ?qwen3.NormBinding = if (hnorm_entry) |he|
        (try loadBlobIntoArena(source, he, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(he, h))
    else
        null;
    var eh_blob_opt: ?LoadedBlob = null;
    var eh_dt_opt: ?quant.WeightDType = null;
    var eh_proj_b: ?qwen3.LinearBinding = if (eh_proj_entry) |pe| blk: {
        const edt = try validateLinear2D(pe, h, 2 * h);
        const eblob = try loadBlobIntoArena(source, pe, &weights_arena, weights_buf, options.verify_crcs);
        eh_blob_opt = eblob;
        eh_dt_opt = edt;
        break :blk eblob.asLinearBinding(edt);
    } else null;

    var final_norm_opt: ?qwen3.NormBinding = if (root_final_norm_entry) |fne|
        (try loadBlobIntoArena(source, fne, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(fne, h))
    else
        null;

    // --- 6. Load draft layer weights & build bindless KV tile tables ---
    const layers = try allocator.alloc(DraftLayerWeights, num_layers);
    errdefer allocator.free(layers);

    var stage_eh_blobs: [MAX_MTP_STAGES]?LoadedBlob = @splat(null);
    var stage_eh_dts: [MAX_MTP_STAGES]?quant.WeightDType = @splat(null);

    var l0_o_proj_blob: LoadedBlob = undefined;
    var l0_o_proj_dtype: quant.WeightDType = .bf16;

    for (0..num_layers) |l| {
        const qkv_in: u32 = if (config.format == .eagle3 and l == 0) 2 * h else h;
        const kv_tbl_bind: metal.BufferBinding = if (!use_custom_attn) blk: {
            const kv_tbl_raw = try scratch_arena.allocAligned(@as(usize, num_chunks) * @sizeOf(attention.KvTilePair), 64);
            const kv_tbl: []attention.KvTilePair = @alignCast(std.mem.bytesAsSlice(attention.KvTilePair, kv_tbl_raw));
            for (0..num_chunks) |c_idx| {
                const tbase = (l * @as(usize, num_chunks) + c_idx) * 2;
                kv_tbl[c_idx] = attention.KvTilePair.fromTiles(kv_tiles[tbase], kv_tiles[tbase + 1], .rgba16f);
            }
            break :blk .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(kv_tbl_raw.ptr) };
        } else .{ .buffer = scratch_buf, .offset = 0 };

        var stage_enorm_b: ?qwen3.NormBinding = null;
        var stage_hnorm_b: ?qwen3.NormBinding = null;
        var stage_eh_proj_b: ?qwen3.LinearBinding = null;
        var stage_final_norm_b: ?qwen3.NormBinding = null;
        if (chained_mtp) {
            const sen_e = parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.enorm.weight", .{l}) catch unreachable) orelse
                try findRequired(parsed, "enorm.weight");
            stage_enorm_b = (try loadBlobIntoArena(source, sen_e, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(sen_e, h));

            const shn_e = parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.hnorm.weight", .{l}) catch unreachable) orelse
                try findRequired(parsed, "hnorm.weight");
            stage_hnorm_b = (try loadBlobIntoArena(source, shn_e, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(shn_e, h));

            const seh_e = findLinearOpt(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.eh_proj", .{l}) catch unreachable) orelse
                try findLinearReq(parsed, "eh_proj");
            const seh_dt = try validateLinear2D(seh_e, h, 2 * h);
            const seh_blob = try loadBlobIntoArena(source, seh_e, &weights_arena, weights_buf, options.verify_crcs);
            stage_eh_proj_b = seh_blob.asLinearBinding(seh_dt);
            if (l < MAX_MTP_STAGES) {
                stage_eh_blobs[l] = seh_blob;
                stage_eh_dts[l] = seh_dt;
            }
            if (l == 0) {
                enorm_b = stage_enorm_b;
                hnorm_b = stage_hnorm_b;
                eh_proj_b = stage_eh_proj_b;
                eh_blob_opt = seh_blob;
                eh_dt_opt = seh_dt;
            }
            if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.final_layernorm.weight", .{l}) catch unreachable)) |sfn_e| {
                stage_final_norm_b = (try loadBlobIntoArena(source, sfn_e, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(sfn_e, h));
                if (final_norm_opt == null) final_norm_opt = stage_final_norm_b;
            }
        }

        const in_ln_e = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable);
        const in_ln_b = (try loadBlobIntoArena(source, in_ln_e, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(in_ln_e, h));

        const hid_ln_b: ?qwen3.NormBinding = if (config.format == .eagle3 and l == 0) blk: {
            const hn_e = try findRequired(parsed, "layers.0.hidden_norm.weight");
            break :blk (try loadBlobIntoArena(source, hn_e, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(hn_e, h));
        } else null;

        var q_b: qwen3.LinearBinding = undefined;
        var k_b: qwen3.LinearBinding = undefined;
        var v_b: qwen3.LinearBinding = undefined;
        var mla_w: ?MtpMlaLayerWeights = null;
        if (is_mla) {
            const qa_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_a_proj", .{l}) catch unreachable);
            const qa_b = (try loadBlobIntoArena(source, qa_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(qa_e, config.q_lora_rank, h));

            const qan_e = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_a_layernorm.weight", .{l}) catch unreachable);
            const qan_b = (try loadBlobIntoArena(source, qan_e, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(qan_e, config.q_lora_rank));

            const qb_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_b_proj", .{l}) catch unreachable);
            const qb_b = (try loadBlobIntoArena(source, qb_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(qb_e, q_dim, config.q_lora_rank));

            const kva_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.kv_a_proj_with_mqa", .{l}) catch unreachable);
            const kva_b = (try loadBlobIntoArena(source, kva_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(kva_e, config.kv_lora_rank + config.qk_rope_head_dim, h));

            const kvan_e = try findRequired(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.kv_a_layernorm.weight", .{l}) catch unreachable);
            const kvan_b = (try loadBlobIntoArena(source, kvan_e, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(kvan_e, config.kv_lora_rank));

            const kvb_n = config.num_attention_heads * (config.qk_nope_head_dim + config.effectiveVHeadDim());
            const kvb_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.kv_b_proj", .{l}) catch unreachable);
            const kvb_b = (try loadBlobIntoArena(source, kvb_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(kvb_e, kvb_n, config.kv_lora_rank));

            mla_w = .{
                .q_a_proj = qa_b,
                .q_a_layernorm = qan_b,
                .q_b_proj = qb_b,
                .kv_a_proj_with_mqa = kva_b,
                .kv_a_layernorm = kvan_b,
                .kv_b_proj = kvb_b,
            };
            q_b = qa_b;
            k_b = kva_b;
            v_b = kvb_b;
        } else {
            const q_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable);
            q_b = (try loadBlobIntoArena(source, q_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(q_e, q_dim, qkv_in));

            const k_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable);
            k_b = (try loadBlobIntoArena(source, k_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(k_e, kv_dim, qkv_in));

            const v_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable);
            v_b = (try loadBlobIntoArena(source, v_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(v_e, v_dim, qkv_in));
        }

        const o_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable);
        const o_dt = try validateLinear2D(o_e, h, o_in_dim);
        const o_blob = try loadBlobIntoArena(source, o_e, &weights_arena, weights_buf, options.verify_crcs);
        const o_b = o_blob.asLinearBinding(o_dt);
        if (l == 0) {
            l0_o_proj_blob = o_blob;
            l0_o_proj_dtype = o_dt;
        }

        var q_norm_bind: ?metal.BufferBinding = null;
        var k_norm_bind: ?metal.BufferBinding = null;
        var qk_norm_dtype: quant.FloatDType = .f16;
        if (!is_mla) {
            if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable)) |qn_e| {
                qk_norm_dtype = try validateNorm1D(qn_e, config.head_dim);
                q_norm_bind = (try loadBlobIntoArena(source, qn_e, &weights_arena, weights_buf, options.verify_crcs)).asBinding();
            }
            if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable)) |kn_e| {
                k_norm_bind = (try loadBlobIntoArena(source, kn_e, &weights_arena, weights_buf, options.verify_crcs)).asBinding();
            }
        }

        var sink_bias_bind: ?metal.BufferBinding = null;
        var sink_bias_dtype: u32 = 1;
        if (findLayerSinkOpt(parsed, l, &name_buf)) |se| {
            sink_bias_dtype = try validateSink1D(se, config.num_attention_heads);
            sink_bias_bind = (try loadBlobIntoArena(source, se, &weights_arena, weights_buf, options.verify_crcs)).asBinding();
        }

        const pln_e = try findLayerPostNormReq(parsed, l, &name_buf);
        const pln_b = (try loadBlobIntoArena(source, pln_e, &weights_arena, weights_buf, options.verify_crcs)).asNormBinding(try validateNorm1D(pln_e, h));

        var gp_b: qwen3.LinearBinding = undefined;
        var up_b: qwen3.LinearBinding = undefined;
        var dp_b: qwen3.LinearBinding = undefined;
        var moe_w: ?MtpMoeLayerWeights = null;
        if (is_moe) {
            const rt_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate", .{l}) catch unreachable);
            const rt_b = (try loadBlobIntoArena(source, rt_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(rt_e, config.num_experts, h));

            var rb_bind: ?metal.BufferBinding = null;
            var rb_dtype: u32 = 2;
            if (parsed.find(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate.bias", .{l}) catch unreachable)) |rb_e| {
                rb_dtype = try validateSink1D(rb_e, config.num_experts);
                rb_bind = (try loadBlobIntoArena(source, rb_e, &weights_arena, weights_buf, options.verify_crcs)).asBinding();
            }

            var sg_b: ?qwen3.LinearBinding = null;
            var su_b: ?qwen3.LinearBinding = null;
            var sd_b: ?qwen3.LinearBinding = null;
            var s_inter: u32 = 0;
            if (findLinearOpt(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_experts.gate_proj", .{l}) catch unreachable)) |sg_e| {
                s_inter = sg_e.shape[0];
                sg_b = (try loadBlobIntoArena(source, sg_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(sg_e, s_inter, h));
                const su_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_experts.up_proj", .{l}) catch unreachable);
                su_b = (try loadBlobIntoArena(source, su_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(su_e, s_inter, h));
                const sd_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_experts.down_proj", .{l}) catch unreachable);
                sd_b = (try loadBlobIntoArena(source, sd_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(sd_e, h, s_inter));
            }

            const meta_raw = try weights_arena.allocAligned(@as(usize, config.num_experts) * @sizeOf(MtpExpertMeta), 64);
            const meta_slice: []MtpExpertMeta = @alignCast(std.mem.bytesAsSlice(MtpExpertMeta, meta_raw));
            var exp_w_dt: u32 = 0;
            var exp_gs: u32 = 64;
            for (0..config.num_experts) |e_idx| {
                const eg_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e_idx }) catch unreachable);
                const eg_dt = try validateLinear2D(eg_e, config.moe_intermediate_size, h);
                const eg_b = (try loadBlobIntoArena(source, eg_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(eg_dt);

                const eu_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e_idx }) catch unreachable);
                const eu_dt = try validateLinear2D(eu_e, config.moe_intermediate_size, h);
                const eu_b = (try loadBlobIntoArena(source, eu_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(eu_dt);

                const ed_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e_idx }) catch unreachable);
                const ed_dt = try validateLinear2D(ed_e, h, config.moe_intermediate_size);
                const ed_b = (try loadBlobIntoArena(source, ed_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(ed_dt);

                const dt_u32: u32 = switch (eg_dt) {
                    .f16 => 0,
                    .bf16 => 1,
                    .q4_affine => 2,
                    .q3_affine => 3,
                    .q2_affine => 4,
                };
                if (e_idx == 0) {
                    exp_w_dt = dt_u32;
                    exp_gs = if (eg_e.group_size > 0) eg_e.group_size else 64;
                } else if (dt_u32 != exp_w_dt) {
                    return error.UnexpectedTensorDType;
                }
                if (eu_dt != eg_dt or ed_dt != eg_dt) return error.UnexpectedTensorDType;

                meta_slice[e_idx] = .{
                    .gate_off = eg_b.weight.offset,
                    .up_off = eu_b.weight.offset,
                    .down_off = ed_b.weight.offset,
                    .gate_sb_off = if (eg_b.scale_bias) |sb| sb.offset else 0,
                    .up_sb_off = if (eu_b.scale_bias) |sb| sb.offset else 0,
                    .down_sb_off = if (ed_b.scale_bias) |sb| sb.offset else 0,
                };
            }

            moe_w = .{
                .router = rt_b,
                .router_bias = rb_bind,
                .router_bias_dtype = rb_dtype,
                .expert_meta_bind = .{ .buffer = weights_buf, .offset = weights_arena.offsetOf(meta_raw.ptr) },
                .expert_weight_dtype = exp_w_dt,
                .expert_group_size = exp_gs,
                .shared_gate_proj = sg_b,
                .shared_up_proj = su_b,
                .shared_down_proj = sd_b,
                .shared_inter = s_inter,
            };
            gp_b = rt_b;
            up_b = rt_b;
            dp_b = rt_b;
        } else {
            const gp_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable);
            gp_b = (try loadBlobIntoArena(source, gp_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(gp_e, config.intermediate_size, h));

            const up_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}) catch unreachable);
            up_b = (try loadBlobIntoArena(source, up_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(up_e, config.intermediate_size, h));

            const dp_e = try findLinearReq(parsed, std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}) catch unreachable);
            dp_b = (try loadBlobIntoArena(source, dp_e, &weights_arena, weights_buf, options.verify_crcs)).asLinearBinding(try validateLinear2D(dp_e, h, config.intermediate_size));
        }

        layers[l] = .{
            .input_layernorm = in_ln_b,
            .hidden_norm = hid_ln_b,
            .stage_enorm = stage_enorm_b,
            .stage_hnorm = stage_hnorm_b,
            .stage_eh_proj = stage_eh_proj_b,
            .stage_final_norm = stage_final_norm_b,
            .q_proj = q_b,
            .k_proj = k_b,
            .v_proj = v_b,
            .o_proj = o_b,
            .mla = mla_w,
            .q_norm = q_norm_bind,
            .k_norm = k_norm_bind,
            .qk_norm_dtype = qk_norm_dtype,
            .sink_bias = sink_bias_bind,
            .sink_bias_dtype = sink_bias_dtype,
            .post_attention_layernorm = pln_b,
            .gate_proj = gp_b,
            .up_proj = up_b,
            .down_proj = dp_b,
            .moe = moe_w,
            .kv_tiles_binding = kv_tbl_bind,
        };
    }

    const final_norm_b = final_norm_opt orelse return error.MissingTensor;

    // --- 7. D5 + D13 Benchmark-Gated Placement (`auto`, `gpu`, `ane`, `sme2`, ADR-0008, ADR-0019) ---
    var effective_placement: Placement = .gpu;
    var ane_proj: ?AneProjection = null;
    errdefer if (ane_proj) |*ap| ap.close(&target.compute);
    var sme_proj: ?*SmeProjectionRuntime = null;
    errdefer if (sme_proj) |sp| {
        sp.deinit(allocator);
        allocator.destroy(sp);
    };
    var extra_stage_sme_projs: [MAX_MTP_STAGES]?*SmeProjectionRuntime = @splat(null);
    errdefer for (&extra_stage_sme_projs) |*slot| {
        if (slot.*) |sp| {
            sp.deinit(allocator);
            allocator.destroy(sp);
            slot.* = null;
        }
    };
    var gate_ane_us: f64 = 0.0;
    var gate_gpu_us: f64 = 0.0;
    var gate_sme2_us: f64 = 0.0;
    var sme2_proj_bytes: usize = 0;
    var wired_total_out: u64 = total_wired_bytes;

    const dev_info = try target.device.info();
    const can_ane = (dev_info.aneCoreCount() > 0) and
        (l0_o_proj_dtype == .f16 or l0_o_proj_dtype == .bf16) and
        (o_in_dim % metal.KV_TILE_ROW_ALIGN_F16 == 0) and
        (h % metal.KV_TILE_ROW_ALIGN_F16 == 0);

    // Prepare candidate D13 L2-resident (`<= 16 MiB`) SME2 projection for `fc` or `eh_proj`.
    const want_sme_eval = (options.placement == .cpu_sme2 or options.placement == .auto);
    var sme_gpu_lin_opt: ?qwen3.LinearBinding = null;
    var sme_gpu_in_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(xcat_raw.ptr) };
    var sme_gpu_out_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_raw.ptr) };
    if (want_sme_eval) {
        switch (config.format) {
            .eagle3 => {
                if (fc_blob_opt != null and fc_dt_opt != null and fc_lin != null) {
                    sme_proj = try tryInitSmeProjectionRuntime(allocator, fc_blob_opt.?, fc_dt_opt.?, .eagle3_fc, options.max_tree_slots, draft_budget);
                    sme_gpu_lin_opt = fc_lin;
                    sme_gpu_in_bind = target.accepted_aux_bind;
                    sme_gpu_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(hin_raw.ptr) };
                }
            },
            .mtp => {
                if (eh_blob_opt != null and eh_dt_opt != null and eh_proj_b != null) {
                    sme_proj = try tryInitSmeProjectionRuntime(allocator, eh_blob_opt.?, eh_dt_opt.?, .mtp_eh_proj, options.max_tree_slots, draft_budget);
                    sme_gpu_lin_opt = eh_proj_b;
                    sme_gpu_in_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(xcat_raw.ptr) };
                    sme_gpu_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_raw.ptr) };
                }
            },
            .eagle2 => {
                if (fc_blob_opt != null and fc_dt_opt != null and fc_lin != null) {
                    sme_proj = try tryInitSmeProjectionRuntime(allocator, fc_blob_opt.?, fc_dt_opt.?, .eagle2_fc, options.max_tree_slots, draft_budget);
                    sme_gpu_lin_opt = fc_lin;
                    sme_gpu_in_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(xcat_raw.ptr) };
                    sme_gpu_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_raw.ptr) };
                }
            },
            .l1_only, .small_draft, .self_recurrent, .dspark, .dflare, .dfly2 => {},
        }
    }

    if (options.placement == .cpu_sme2) {
        if (sme_proj) |sp| {
            const cal_m: u32 = @min(options.max_tree_slots, 4);
            const sme_cal = try calibrateD13SmeVsGpu(
                target,
                sp,
                sme_gpu_lin_opt.?,
                sme_gpu_in_bind,
                sme_gpu_out_bind,
                cal_m,
                options.timeout_ms,
            );
            gate_sme2_us = sme_cal.sme_us;
            gate_gpu_us = sme_cal.gpu_us;
            sme2_proj_bytes = sme_cal.weight_bytes;
            wired_total_out += sp.totalWiredBytes();
            effective_placement = .cpu_sme2;
        }
    } else if (options.placement == .ane and can_ane) {
        const wf16 = try convertWeightBlobToF16(allocator, l0_o_proj_blob, l0_o_proj_dtype);
        defer allocator.free(wf16);
        ane_proj = try AneProjection.init(allocator, target.device, &target.compute, o_in_dim, h, ANE_MAX_BATCH_ROWS, wf16);
        effective_placement = .ane;
    } else if (options.placement == .auto) {
        var best_us: f64 = std.math.inf(f64);
        if (can_ane) {
            const wf16 = try convertWeightBlobToF16(allocator, l0_o_proj_blob, l0_o_proj_dtype);
            defer allocator.free(wf16);
            var probe_ap = try AneProjection.init(allocator, target.device, &target.compute, o_in_dim, h, 1, wf16);
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
            best_us = @min(cal.ane_us, cal.gpu_us);
            if (effective_placement == .ane) {
                ane_proj = try AneProjection.init(allocator, target.device, &target.compute, o_in_dim, h, ANE_MAX_BATCH_ROWS, wf16);
            }
        }
        if (sme_proj) |sp| {
            const cal_m: u32 = @min(options.max_tree_slots, 4);
            const sme_cal = try calibrateD13SmeVsGpu(
                target,
                sp,
                sme_gpu_lin_opt.?,
                sme_gpu_in_bind,
                sme_gpu_out_bind,
                cal_m,
                options.timeout_ms,
            );
            gate_sme2_us = sme_cal.sme_us;
            if (gate_gpu_us == 0.0) gate_gpu_us = sme_cal.gpu_us;
            sme2_proj_bytes = sme_cal.weight_bytes;
            if (selectAutoPlacement(
                can_ane,
                gate_ane_us,
                gate_gpu_us,
                sp.dispatcher.caps.supportsSme2(),
                sme_cal.prefer_sme2,
                sme_cal.sme_us,
            ) == .cpu_sme2 and (!can_ane or sme_cal.sme_us <= best_us)) {
                if (ane_proj) |*ap| {
                    ap.close(&target.compute);
                    ane_proj = null;
                }
                wired_total_out += sp.totalWiredBytes();
                effective_placement = .cpu_sme2;
            } else {
                sp.deinit(allocator);
                allocator.destroy(sp);
                sme_proj = null;
            }
        }
    } else if (sme_proj) |sp| {
        sp.deinit(allocator);
        allocator.destroy(sp);
        sme_proj = null;
    }

    // For chained MTP under D13 SME2, opportunistically keep stage `k >= 1` `eh_proj`
    // L2-resident as long as cumulative SME2 draft projection weights fit the 16 MiB L2 cap.
    if (chained_mtp and effective_placement == .cpu_sme2 and sme_proj != null) {
        var cum_sme_bytes: usize = sme2_proj_bytes;
        for (1..@min(num_layers, MAX_MTP_STAGES)) |l| {
            if (stage_eh_blobs[l] != null and stage_eh_dts[l] != null) {
                const candidate = try tryInitSmeProjectionRuntime(allocator, stage_eh_blobs[l].?, stage_eh_dts[l].?, .mtp_eh_proj, options.max_tree_slots, draft_budget);
                if (candidate) |csp| {
                    if (cum_sme_bytes + csp.weight_bytes <= sme.MAX_L2_DRAFT_PROJ_BYTES and
                        target.total_wired_bytes + wired_total_out + csp.totalWiredBytes() <= target.wired_budget_bytes)
                    {
                        extra_stage_sme_projs[l] = csp;
                        cum_sme_bytes += csp.weight_bytes;
                        wired_total_out += csp.totalWiredBytes();
                    } else {
                        csp.deinit(allocator);
                        allocator.destroy(csp);
                    }
                }
            }
        }
        sme2_proj_bytes = cum_sme_bytes;
    }

    return .{
        .config = config,
        .requested_placement = options.placement,
        .effective_placement = effective_placement,
        .placement_gate_ane_us = gate_ane_us,
        .placement_gate_gpu_us = gate_gpu_us,
        .placement_gate_sme2_us = gate_sme2_us,
        .sme2_proj_bytes = sme2_proj_bytes,
        .sme2_eval_count = 0,
        .max_seq_len = max_seq_len,
        .chunk_tokens = options.chunk_tokens,
        .num_chunks = num_chunks,
        .max_tree_slots = options.max_tree_slots,
        .seq_len = 0,
        .timeout_ms = options.timeout_ms,
        .total_wired_bytes = wired_total_out,
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
        .sme_proj = sme_proj,
        .extra_stage_sme_projs = extra_stage_sme_projs,
        .use_custom_attn = use_custom_attn,
        .kv_ring_cap = kv_ring_cap,
        .mtp_pipes = mtp_pipes,
        .kv_ring_k_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(kv_ring_k_raw.ptr) },
        .kv_ring_v_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(kv_ring_v_raw.ptr) },
        .stage_h_prev_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(stage_prev_raw.ptr) },
        .stage_tree_slots_mask = @splat(0),
        .mla_q_a_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(mla_qa_raw.ptr) },
        .mla_q_a_norm_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(mla_qan_raw.ptr) },
        .mla_kv_a_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(mla_kva_raw.ptr) },
        .mla_kv_a_pass_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(mla_kvp_raw.ptr) },
        .mla_kv_a_norm_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(mla_kvn_raw.ptr) },
        .mla_kv_b_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(mla_kvb_raw.ptr) },
        .moe_router_logits_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(moe_rt_raw.ptr) },
        .moe_expert_ids_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(moe_id_raw.ptr) },
        .moe_expert_weights_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(moe_w_raw.ptr) },
        .moe_act_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(moe_act_raw.ptr) },
        .max_sparse_indices = max_sparse_indices,
        .shared_sparse_indices_host = @alignCast(std.mem.bytesAsSlice(u32, sp_idx_raw)),
        .shared_sparse_indices_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(sp_idx_raw.ptr) },
        .active_sparse_bind = null,
        .active_sparse_count = 0,
        .active_sparse_stride = 0,
        .active_sparse_rows = 0,
        .zero_reindex_hits = 0,
        .indexer_dispatches = 0,
        .token_ids_host = @alignCast(std.mem.bytesAsSlice(u32, tok_raw)),
        .parent_slots_host = @alignCast(std.mem.bytesAsSlice(u32, par_raw)),
        .positions_host = @alignCast(std.mem.bytesAsSlice(u32, pos_raw)),
        .positions_pool_host = @alignCast(std.mem.bytesAsSlice(u32, pos_pool_raw)),
        .ancestor_masks_host = @alignCast(std.mem.bytesAsSlice(u64, msk_raw)),
        .topk_host = @alignCast(std.mem.bytesAsSlice(moe.TopKCandidate, tk_out_raw)),
        .hidden_pool_host = @alignCast(std.mem.bytesAsSlice(f16, hp_raw)),
        .h_in_host = @alignCast(std.mem.bytesAsSlice(f16, hin_raw)),
        .emb_host = @alignCast(std.mem.bytesAsSlice(f16, emb_raw)),
        .x_host = @alignCast(std.mem.bytesAsSlice(f16, x_raw)),
        .token_ids_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tok_raw.ptr) },
        .parent_slots_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(par_raw.ptr) },
        .positions_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(pos_raw.ptr) },
        .positions_pool_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(pos_pool_raw.ptr) },
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
        .tree_v_layer_stride = tree_v_layer_stride,
        .root_top_k_count = 0,
        .root_candidates = undefined,
        .qf_pipes = draft_qf_pipes,
        .embed_dequant_f32 = draft_embed_dequant_f32,
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
    for (0..elems) |i| {
        const u = @as(f32, @floatFromInt(nextRng(&rng) & 0xFFFF)) / 65535.0;
        const f = if (centered_one) (0.95 + 0.1 * u) else ((u * 2.0 - 1.0) * amp);
        std.mem.writeInt(u16, raw[i * 2 ..][0..2], quant.f32ToFloatBits(.bf16, f), .little);
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
    const v_dim = cfg.vDim();
    const o_in_dim = cfg.oInDim();
    const inter = cfg.intermediate_size;
    const inv_h = 0.5 / @sqrt(@as(f32, @floatFromInt(h)));
    const inv_o = 0.5 / @sqrt(@as(f32, @floatFromInt(o_in_dim)));
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
        for (0..cfg.draft_vocab_size) |i| {
            const id: u32 = @intCast((i * cfg.vocab_size) / cfg.draft_vocab_size);
            std.mem.writeInt(u32, d2t_bytes[i * 4 ..][0..4], id, .little);
        }
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
            if (!cfg.mtp_chained) {
                try add1d(allocator, &recs, "enorm.weight", h, seed + tidx);
                tidx += 1;
                try add1d(allocator, &recs, "hnorm.weight", h, seed + tidx);
                tidx += 1;
                try add2d(allocator, &recs, "eh_proj.weight", .dense, h, 2 * h, 0.5 / @sqrt(@as(f32, @floatFromInt(2 * h))), seed + tidx);
                tidx += 1;
            }
        },
        .l1_only, .small_draft, .self_recurrent, .dspark, .dflare, .dfly2 => unreachable,
    }

    var nbuf: [64]u8 = undefined;
    for (0..cfg.num_hidden_layers) |l| {
        if (cfg.format == .mtp and cfg.mtp_chained) {
            try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.enorm.weight", .{l}) catch unreachable, h, seed + tidx);
            tidx += 1;
            try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.hnorm.weight", .{l}) catch unreachable, h, seed + tidx);
            tidx += 1;
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.eh_proj.weight", .{l}) catch unreachable, .dense, h, 2 * h, 0.5 / @sqrt(@as(f32, @floatFromInt(2 * h))), seed + tidx);
            tidx += 1;
            try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.final_layernorm.weight", .{l}) catch unreachable, h, seed + tidx);
            tidx += 1;
        }
        const qkv_in: u32 = if (cfg.format == .eagle3 and l == 0) 2 * h else h;
        const inv_qkv = 0.5 / @sqrt(@as(f32, @floatFromInt(qkv_in)));
        try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, h, seed + tidx);
        tidx += 1;
        if (cfg.format == .eagle3 and l == 0) {
            try add1d(allocator, &recs, "layers.0.hidden_norm.weight", h, seed + tidx);
            tidx += 1;
        }
        if (cfg.isMlaMtp()) {
            const q_lora = cfg.q_lora_rank;
            const kv_lora = cfg.kv_lora_rank;
            const inv_ql = 0.5 / @sqrt(@as(f32, @floatFromInt(q_lora)));
            const inv_kvl = 0.5 / @sqrt(@as(f32, @floatFromInt(kv_lora)));
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_a_proj.weight", .{l}) catch unreachable, .dense, q_lora, h, inv_h, seed + tidx);
            tidx += 1;
            try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_a_layernorm.weight", .{l}) catch unreachable, q_lora, seed + tidx);
            tidx += 1;
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_b_proj.weight", .{l}) catch unreachable, .dense, q_dim, q_lora, inv_ql, seed + tidx);
            tidx += 1;
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{l}) catch unreachable, .dense, kv_lora + cfg.qk_rope_head_dim, h, inv_h, seed + tidx);
            tidx += 1;
            try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.kv_a_layernorm.weight", .{l}) catch unreachable, kv_lora, seed + tidx);
            tidx += 1;
            const kvb_n = cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.effectiveVHeadDim());
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.kv_b_proj.weight", .{l}) catch unreachable, .dense, kvb_n, kv_lora, inv_kvl, seed + tidx);
            tidx += 1;
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj.weight", .{l}) catch unreachable, .dense, h, o_in_dim, inv_o, seed + tidx);
            tidx += 1;
        } else {
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj.weight", .{l}) catch unreachable, .dense, q_dim, qkv_in, inv_qkv, seed + tidx);
            tidx += 1;
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj.weight", .{l}) catch unreachable, .dense, kv_dim, qkv_in, inv_qkv, seed + tidx);
            tidx += 1;
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj.weight", .{l}) catch unreachable, .dense, v_dim, qkv_in, inv_qkv, seed + tidx);
            tidx += 1;
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj.weight", .{l}) catch unreachable, .dense, h, o_in_dim, inv_o, seed + tidx);
            tidx += 1;
            if (cfg.format == .mtp and !cfg.mtp_chained) {
                try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable, cfg.head_dim, seed + tidx);
                tidx += 1;
                try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable, cfg.head_dim, seed + tidx);
                tidx += 1;
            }
        }
        if (cfg.add_swa_attention_sink_bias) {
            const nm = try allocator.dupe(u8, std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.sink_bias", .{l}) catch unreachable);
            errdefer allocator.free(nm);
            const sdata = try fillBf16(allocator, cfg.num_attention_heads, 1.5, false, seed + tidx);
            errdefer allocator.free(sdata);
            try recs.append(allocator, .{ .name = nm, .kind = .norm, .dtype = .bf16, .shape = .{ cfg.num_attention_heads, 0, 0, 0 }, .rank = 1, .data = sdata });
            tidx += 1;
        }
        try add1d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, h, seed + tidx);
        tidx += 1;
        if (cfg.isMoeMtp()) {
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate.weight", .{l}) catch unreachable, .dense, cfg.num_experts, h, inv_h, seed + tidx);
            tidx += 1;
            {
                const bnm = try allocator.dupe(u8, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate.bias", .{l}) catch unreachable);
                errdefer allocator.free(bnm);
                const bdata = try fillBf16(allocator, cfg.num_experts, 0.2, false, seed + tidx);
                errdefer allocator.free(bdata);
                try recs.append(allocator, .{ .name = bnm, .kind = .norm, .dtype = .bf16, .shape = .{ cfg.num_experts, 0, 0, 0 }, .rank = 1, .data = bdata });
                tidx += 1;
            }
            if (cfg.shared_expert_intermediate_size > 0) {
                const si = cfg.shared_expert_intermediate_size;
                const inv_si = 0.5 / @sqrt(@as(f32, @floatFromInt(si)));
                try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.shared_experts.gate_proj.weight", .{l}) catch unreachable, .dense, si, h, inv_h, seed + tidx);
                tidx += 1;
                try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.shared_experts.up_proj.weight", .{l}) catch unreachable, .dense, si, h, inv_h, seed + tidx);
                tidx += 1;
                try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.shared_experts.down_proj.weight", .{l}) catch unreachable, .dense, h, si, inv_si, seed + tidx);
                tidx += 1;
            }
            const mi = cfg.moe_intermediate_size;
            const inv_mi = 0.5 / @sqrt(@as(f32, @floatFromInt(mi)));
            for (0..cfg.num_experts) |e_idx| {
                try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.gate_proj.weight", .{ l, e_idx }) catch unreachable, .dense, mi, h, inv_h, seed + tidx);
                tidx += 1;
                try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.up_proj.weight", .{ l, e_idx }) catch unreachable, .dense, mi, h, inv_h, seed + tidx);
                tidx += 1;
                try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.down_proj.weight", .{ l, e_idx }) catch unreachable, .dense, h, mi, inv_mi, seed + tidx);
                tidx += 1;
            }
        } else {
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj.weight", .{l}) catch unreachable, .dense, inter, h, inv_h, seed + tidx);
            tidx += 1;
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj.weight", .{l}) catch unreachable, .dense, inter, h, inv_h, seed + tidx);
            tidx += 1;
            try add2d(allocator, &recs, std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj.weight", .{l}) catch unreachable, .dense, h, inter, inv_i, seed + tidx);
            tidx += 1;
        }
    }

    if (!(cfg.format == .mtp and cfg.mtp_chained)) {
        const fn_name = if (cfg.format == .mtp) "shared_head.norm.weight" else "norm.weight";
        try add1d(allocator, &recs, fn_name, h, seed + tidx);
        tidx += 1;
    }
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

pub fn buildSyntheticTargetBf16ForTest(allocator: std.mem.Allocator, cfg: qwen3.ModelConfig, seed: u64) ![]u8 {
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

test "D13: Placement.cpu_sme2 executes EAGLE-3 fc & MTP eh_proj on P-core SME2 with parity vs GPU" {
    const allocator = std.testing.allocator;
    const tgt_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .vocab_size = 128,
        .hidden_size = 64,
        .num_hidden_layers = 4,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .max_position_embeddings = 256,
        .rope_theta = 10_000.0,
        .rms_norm_eps = 1e-6,
        .tie_word_embeddings = true,
    };
    const tgt_img = try buildSyntheticTargetBf16ForTest(allocator, tgt_cfg, 12345);
    defer allocator.free(tgt_img);

    var target = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_img, tgt_cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .enable_feature_taps = true,
    });
    defer target.deinit(allocator);

    const prompt = [_]u32{ 5, 19, 41 };
    _ = try target.prefillChunk(&prompt, 0, true);

    // 1. Test EAGLE-3 `fc` on `.gpu` vs `.cpu_sme2`!
    const e3_cfg: DraftConfig = .{
        .format = .eagle3,
        .vocab_size = 128,
        .draft_vocab_size = 64,
        .hidden_size = 64,
        .target_hidden_size = 64,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .rope_theta = 10_000.0,
    };
    const e3_img = try buildSyntheticDraftAmfImage(allocator, e3_cfg, 54321);
    defer allocator.free(e3_img);

    var e3_gpu = try DraftHead.loadFromBytes(allocator, &target, e3_img, e3_cfg, .{
        .placement = .gpu,
        .max_seq_len = 128,
        .max_tree_slots = 16,
    });
    defer e3_gpu.deinit(allocator, &target);

    var e3_sme = try DraftHead.loadFromBytes(allocator, &target, e3_img, e3_cfg, .{
        .placement = .cpu_sme2,
        .max_seq_len = 128,
        .max_tree_slots = 16,
    });
    defer e3_sme.deinit(allocator, &target);

    try std.testing.expectEqual(Placement.cpu_sme2, e3_sme.effective_placement);
    try std.testing.expect(e3_sme.placement_gate_sme2_us > 0.0);
    try std.testing.expect(e3_sme.placement_gate_gpu_us > 0.0);
    try std.testing.expect(e3_sme.sme2_proj_bytes > 0);
    try std.testing.expect(e3_sme.sme2_proj_bytes <= sme.MAX_L2_DRAFT_PROJ_BYTES);

    try e3_gpu.syncAccepted(&target, &prompt, target.accepted_aux_bind, 0, 4);
    try e3_sme.syncAccepted(&target, &prompt, target.accepted_aux_bind, 0, 4);
    try std.testing.expectEqual(@as(u64, 1), e3_sme.sme2_eval_count);
    try std.testing.expectEqual(e3_gpu.rootCandidates()[0].token_id, e3_sme.rootCandidates()[0].token_id);

    // 2. Test MTP dual-norm `eh_proj` on `.gpu` vs `.cpu_sme2`!
    const mtp_cfg: DraftConfig = .{
        .format = .mtp,
        .vocab_size = 128,
        .draft_vocab_size = 128,
        .hidden_size = 64,
        .target_hidden_size = 64,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .rope_theta = 10_000.0,
        .share_embed_tokens = true,
        .share_lm_head = true,
    };
    const mtp_img = try buildSyntheticDraftAmfImage(allocator, mtp_cfg, 67890);
    defer allocator.free(mtp_img);

    var mtp_gpu = try DraftHead.loadFromBytes(allocator, &target, mtp_img, mtp_cfg, .{
        .placement = .gpu,
        .max_seq_len = 128,
        .max_tree_slots = 16,
    });
    defer mtp_gpu.deinit(allocator, &target);

    var mtp_sme = try DraftHead.loadFromBytes(allocator, &target, mtp_img, mtp_cfg, .{
        .placement = .cpu_sme2,
        .max_seq_len = 128,
        .max_tree_slots = 16,
    });
    defer mtp_sme.deinit(allocator, &target);

    try std.testing.expectEqual(Placement.cpu_sme2, mtp_sme.effective_placement);
    try mtp_gpu.syncAccepted(&target, &prompt, target.accepted_top_bind, 0, 4);
    try mtp_sme.syncAccepted(&target, &prompt, target.accepted_top_bind, 0, 4);
    try std.testing.expectEqual(@as(u64, 1), mtp_sme.sme2_eval_count);
    try std.testing.expectEqual(mtp_gpu.rootCandidates()[0].token_id, mtp_sme.rootCandidates()[0].token_id);
}

test "C16: D5 and D13 placement selectors are generation-agnostic across M1-M3, M4, and M5+" {
    // D5 ANE-vs-GPU winner strictly follows measured latency (and guards non-finite/zero inputs).
    try std.testing.expectEqual(Placement.ane, selectD5Winner(18.0, 24.0));
    try std.testing.expectEqual(Placement.gpu, selectD5Winner(24.0, 24.0));
    try std.testing.expectEqual(Placement.gpu, selectD5Winner(30.0, 12.0));
    try std.testing.expectEqual(Placement.gpu, selectD5Winner(0.0, 12.0));
    try std.testing.expectEqual(Placement.gpu, selectD5Winner(std.math.nan(f64), 12.0));

    // M1-M3 (!has_sme2): never selects .cpu_sme2 even if sme_prefer were passed.
    try std.testing.expectEqual(Placement.ane, selectAutoPlacement(true, 18.0, 25.0, false, true, 5.0));
    try std.testing.expectEqual(Placement.gpu, selectAutoPlacement(true, 35.0, 20.0, false, true, 5.0));
    try std.testing.expectEqual(Placement.gpu, selectAutoPlacement(false, 0.0, 0.0, false, true, 5.0));

    // M4 (has_sme2, !has_nax): prefers .cpu_sme2 when L2-resident SME2 beats both GPU and ANE.
    try std.testing.expectEqual(Placement.cpu_sme2, selectAutoPlacement(true, 30.0, 22.0, true, true, 12.0));
    try std.testing.expectEqual(Placement.ane, selectAutoPlacement(true, 10.0, 22.0, true, true, 12.0));
    try std.testing.expectEqual(Placement.cpu_sme2, selectAutoPlacement(false, 0.0, 0.0, true, true, 12.0));

    // M5+ (has_sme2, has_nax): when faster GPU beats both ANE and SME2, routes cleanly to .gpu.
    try std.testing.expectEqual(Placement.gpu, selectAutoPlacement(true, 28.0, 9.0, true, false, 12.0));
    try std.testing.expectEqual(Placement.cpu_sme2, selectAutoPlacement(true, 28.0, 14.0, true, true, 11.0));
}

test "M9.1 D15 chained 3L native MTP (K=192, V=128, rope_dim=64, SWA, v_scale=0.707, sink_bias) multi-stage and D13 SME2 parity" {
    const allocator = std.testing.allocator;

    // 1. Verify DraftConfig.fromJson parses MiMo-V2.6 chained 3L MTP schema accurately.
    const mimo_mtp_json =
        \\{
        \\  "format": "mtp",
        \\  "mtp_chained": true,
        \\  "vocab_size": 151936,
        \\  "hidden_size": 4096,
        \\  "intermediate_size": 16384,
        \\  "num_nextn_predict_layers": 3,
        \\  "num_attention_heads": 64,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 192,
        \\  "v_head_dim": 128,
        \\  "partial_rotary_factor": 0.3333333333333333,
        \\  "rope_theta": 10000.0,
        \\  "sliding_window_size": 128,
        \\  "attention_value_scale": 0.70710678,
        \\  "add_swa_attention_sink_bias": true,
        \\  "rms_norm_eps": 1e-6
        \\}
    ;
    const parsed_cfg = try DraftConfig.fromJson(allocator, mimo_mtp_json);
    try std.testing.expectEqual(DraftMode.mtp, parsed_cfg.format);
    try std.testing.expect(parsed_cfg.mtp_chained);
    try std.testing.expectEqual(@as(u32, 3), parsed_cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 192), parsed_cfg.head_dim);
    try std.testing.expectEqual(@as(u32, 128), parsed_cfg.effectiveVHeadDim());
    try std.testing.expectEqual(@as(u32, 64), parsed_cfg.effectiveRopeDim());
    try std.testing.expectEqual(@as(u32, 128), parsed_cfg.sliding_window);
    try std.testing.expect(parsed_cfg.add_swa_attention_sink_bias);
    try std.testing.expect(parsed_cfg.usesCustomMtpAttn());

    // 2. Build synthetic target + chained 3L native MTP with asymmetric K=192/V=128, partial RoPE=64,
    //    sliding_window=8, attention_value_scale=0.707, and sink_bias.
    const tgt_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 256,
        .rope_theta = 10_000.0,
        .rms_norm_eps = 1e-6,
        .tie_word_embeddings = true,
    };
    const tgt_img = try buildSyntheticTargetBf16ForTest(allocator, tgt_cfg, 91015);
    defer allocator.free(tgt_img);

    var target = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_img, tgt_cfg, .{
        .max_seq_len = 64,
        .max_tree_nodes = 16,
        .enable_feature_taps = true,
    });
    defer target.deinit(allocator);

    const mtp3_cfg: DraftConfig = .{
        .format = .mtp,
        .mtp_chained = true,
        .vocab_size = 128,
        .draft_vocab_size = 128,
        .hidden_size = 128,
        .target_hidden_size = 128,
        .num_hidden_layers = 3,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 192,
        .v_head_dim = 128,
        .rope_dim = 64,
        .sliding_window = 8,
        .attention_value_scale = 0.70710678,
        .add_swa_attention_sink_bias = true,
        .intermediate_size = 256,
        .rope_theta = 10_000.0,
        .share_embed_tokens = true,
        .share_lm_head = true,
    };
    const mtp3_img = try buildSyntheticDraftAmfImage(allocator, mtp3_cfg, 91016);
    defer allocator.free(mtp3_img);

    var dh_gpu = try DraftHead.loadFromBytes(allocator, &target, mtp3_img, mtp3_cfg, .{
        .placement = .gpu,
        .max_seq_len = 64,
        .max_tree_slots = 16,
    });
    defer dh_gpu.deinit(allocator, &target);

    var dh_sme = try DraftHead.loadFromBytes(allocator, &target, mtp3_img, mtp3_cfg, .{
        .placement = .cpu_sme2,
        .max_seq_len = 64,
        .max_tree_slots = 16,
    });
    defer dh_sme.deinit(allocator, &target);

    try std.testing.expect(dh_gpu.isChainedMtp());
    try std.testing.expectEqual(@as(usize, 3), dh_gpu.numMtpStages());
    try std.testing.expect(dh_gpu.use_custom_attn);
    try std.testing.expectEqual(Placement.cpu_sme2, dh_sme.effective_placement);
    try std.testing.expect(dh_sme.sme2_proj_bytes <= sme.MAX_L2_DRAFT_PROJ_BYTES);
    try std.testing.expect(dh_sme.extra_stage_sme_projs[1] != null);

    // 3. Prefill 10 tokens (exceeding sliding_window=8 to test SWA ring wraparound), sync all 3 stages!
    const prompt = [_]u32{ 3, 11, 27, 45, 62, 7, 19, 33, 51, 88 };
    _ = try target.prefillChunk(&prompt, 0, true);
    try dh_gpu.syncAccepted(&target, &prompt, target.accepted_top_bind, 0, 4);
    try dh_sme.syncAccepted(&target, &prompt, target.accepted_top_bind, 0, 4);

    try std.testing.expectEqual(@as(u32, prompt.len), dh_gpu.seq_len);
    try std.testing.expectEqual(@as(u32, 4), dh_gpu.root_top_k_count);
    try std.testing.expectEqual(dh_gpu.rootCandidates()[0].token_id, dh_sme.rootCandidates()[0].token_id);
    try std.testing.expect(std.math.isFinite(dh_gpu.rootCandidates()[0].prob) and dh_gpu.rootCandidates()[0].prob > 0.0);

    // 4. Chain Stage 1 (depth 1 -> depth 2) and Stage 2 (depth 2 -> depth 3) on both GPU & SME2!
    const d1_toks = [_]u32{ dh_gpu.rootCandidates()[0].token_id, dh_gpu.rootCandidates()[1].token_id };
    const d1_parents = [_]u32{ 0, 0 };
    const d1_pos = [_]u32{ prompt.len, prompt.len };
    var masks = [_]u64{ 0b0001, 0b0010, 0b0101, 0b1010 };
    var cands_gpu_s1: [2 * 4]moe.TopKCandidate = undefined;
    var cands_sme_s1: [2 * 4]moe.TopKCandidate = undefined;
    try dh_gpu.stepTreeBatchForStage(&target, 1, &d1_toks, &d1_parents, &d1_pos, masks[0..2], 0, 4, &cands_gpu_s1);
    try dh_sme.stepTreeBatchForStage(&target, 1, &d1_toks, &d1_parents, &d1_pos, masks[0..2], 0, 4, &cands_sme_s1);
    try std.testing.expectEqual(cands_gpu_s1[0].token_id, cands_sme_s1[0].token_id);
    try std.testing.expectEqual(cands_gpu_s1[4].token_id, cands_sme_s1[4].token_id);
    try std.testing.expect(std.math.isFinite(cands_gpu_s1[0].prob) and cands_gpu_s1[0].prob > 0.0);

    const d2_toks = [_]u32{ cands_gpu_s1[0].token_id, cands_gpu_s1[4].token_id };
    const d2_parents = [_]u32{ 1, 2 }; // hidden_pool[1 + slot_idx] from Stage 1
    const d2_pos = [_]u32{ prompt.len + 1, prompt.len + 1 };
    var cands_gpu_s2: [2 * 4]moe.TopKCandidate = undefined;
    var cands_sme_s2: [2 * 4]moe.TopKCandidate = undefined;
    try dh_gpu.stepTreeBatchForStage(&target, 2, &d2_toks, &d2_parents, &d2_pos, masks[0..4], 2, 4, &cands_gpu_s2);
    try dh_sme.stepTreeBatchForStage(&target, 2, &d2_toks, &d2_parents, &d2_pos, masks[0..4], 2, 4, &cands_sme_s2);
    try std.testing.expectEqual(cands_gpu_s2[0].token_id, cands_sme_s2[0].token_id);
    try std.testing.expectEqual(cands_gpu_s2[4].token_id, cands_sme_s2[4].token_id);
    try std.testing.expect(dh_sme.sme2_eval_count >= 3);

    // 5. Commit a 1-token accepted follow-up (exercises stage_h_prev_bind across rounds!) and reset.
    const next_tok = [_]u32{d1_toks[0]};
    _ = try target.stepDecode(next_tok[0], prompt.len);
    try dh_gpu.syncAccepted(&target, &next_tok, target.accepted_top_bind, prompt.len, 4);
    try dh_sme.syncAccepted(&target, &next_tok, target.accepted_top_bind, prompt.len, 4);
    try std.testing.expectEqual(@as(u32, prompt.len + 1), dh_gpu.seq_len);
    try std.testing.expectEqual(dh_gpu.rootCandidates()[0].token_id, dh_sme.rootCandidates()[0].token_id);

    dh_gpu.reset();
    try std.testing.expectEqual(@as(u32, 0), dh_gpu.seq_len);
}

test "DraftHead D15 GLM-5.2/5.3 MLA and GLM-5.3-Flash NoPE-MLA zero-reindex 1L MTP + L2 SME2 parity" {
    const allocator = std.testing.allocator;

    // 1. Verify exact real-HF JSON config parsing for both `glm_moe_dsa` (`753B`) and `glm5_next` (`320B`).
    const glm53_mtp_json =
        \\{
        \\  "format": "mtp",
        \\  "mtp_arch": "glm_moe_dsa",
        \\  "vocab_size": 154880,
        \\  "hidden_size": 6144,
        \\  "num_hidden_layers": 1,
        \\  "num_attention_heads": 64,
        \\  "num_key_value_heads": 64,
        \\  "q_lora_rank": 2048,
        \\  "kv_lora_rank": 512,
        \\  "qk_nope_head_dim": 192,
        \\  "qk_rope_head_dim": 64,
        \\  "v_head_dim": 256,
        \\  "rope_interleave": true,
        \\  "rope_theta": 8000000.0,
        \\  "rms_norm_eps": 1e-5,
        \\  "index_share_for_mtp_iteration": true,
        \\  "index_topk": 2048,
        \\  "num_experts": 256,
        \\  "num_experts_per_tok": 8,
        \\  "moe_intermediate_size": 2048,
        \\  "n_shared_experts": 1,
        \\  "routed_scaling_factor": 2.5,
        \\  "norm_topk_prob": true,
        \\  "scoring_func": "sigmoid",
        \\  "share_embed_tokens": true,
        \\  "share_lm_head": true
        \\}
    ;
    const cfg_dsa = try DraftConfig.fromJson(allocator, glm53_mtp_json);
    try std.testing.expectEqual(DraftMode.mtp, cfg_dsa.format);
    try std.testing.expectEqual(MtpArch.glm_moe_dsa, cfg_dsa.mtp_arch);
    try std.testing.expect(cfg_dsa.isMlaMtp());
    try std.testing.expect(cfg_dsa.isMoeMtp());
    try std.testing.expect(cfg_dsa.rope_interleave);
    try std.testing.expect(cfg_dsa.index_share_for_mtp_iteration);
    try std.testing.expectEqual(@as(u32, 256), cfg_dsa.effectiveQkHeadDim());
    try std.testing.expectEqual(@as(u32, 64), cfg_dsa.effectiveRopeDim());
    try std.testing.expectEqual(@as(u32, 256), cfg_dsa.num_experts);
    try std.testing.expectEqual(@as(f32, 2.5), cfg_dsa.routed_scaling_factor);
    try std.testing.expect(cfg_dsa.moe_scoring_sigmoid);

    const glm53_flash_mtp_json =
        \\{
        \\  "format": "mtp",
        \\  "mtp_arch": "glm5_next",
        \\  "vocab_size": 154880,
        \\  "hidden_size": 4096,
        \\  "num_hidden_layers": 1,
        \\  "num_attention_heads": 64,
        \\  "num_key_value_heads": 64,
        \\  "q_lora_rank": 1536,
        \\  "kv_lora_rank": 512,
        \\  "qk_nope_head_dim": 256,
        \\  "qk_rope_head_dim": 0,
        \\  "v_head_dim": 256,
        \\  "rms_norm_eps": 1e-5,
        \\  "index_share_for_mtp_iteration": true,
        \\  "index_topk": 2048,
        \\  "num_experts": 288,
        \\  "num_experts_per_tok": 8,
        \\  "moe_intermediate_size": 2048,
        \\  "n_shared_experts": 1,
        \\  "swiglu_limit": 10.0,
        \\  "scoring_func": "sigmoid",
        \\  "share_embed_tokens": true,
        \\  "share_lm_head": true
        \\}
    ;
    const cfg_flash = try DraftConfig.fromJson(allocator, glm53_flash_mtp_json);
    try std.testing.expectEqual(MtpArch.glm5_next, cfg_flash.mtp_arch);
    try std.testing.expect(cfg_flash.isMlaMtp());
    try std.testing.expectEqual(@as(u32, 0), cfg_flash.effectiveRopeDim());
    try std.testing.expectEqual(@as(u32, 256), cfg_flash.effectiveQkHeadDim());
    try std.testing.expectEqual(@as(u32, 288), cfg_flash.num_experts);
    try std.testing.expectEqual(@as(f32, 10.0), cfg_flash.swiglu_limit);

    // 2. Build compact target and exercise BOTH:
    //    (a) GLM-5.2/5.3 interleaved-RoPE MLA + MoE 1L MTP (`rope_interleave=true, qk_rope=32`)
    //    (b) GLM-5.3-Flash NoPE-MLA + MoE + `swiglu_limit=10.0` 1L MTP (`qk_rope=0`)
    const tgt_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 256,
        .rope_theta = 10_000.0,
        .rms_norm_eps = 1e-5,
        .tie_word_embeddings = true,
    };
    const tgt_img = try buildSyntheticTargetBf16ForTest(allocator, tgt_cfg, 92015);
    defer allocator.free(tgt_img);

    const cases = [_]struct {
        arch: MtpArch,
        qk_nope: u32,
        qk_rope: u32,
        interleave: bool,
        swiglu_lim: f32,
        scale_f: f32,
        seed: u64,
    }{
        .{ .arch = .glm_moe_dsa, .qk_nope = 96, .qk_rope = 32, .interleave = true, .swiglu_lim = 0.0, .scale_f = 2.5, .seed = 92101 },
        .{ .arch = .glm5_next, .qk_nope = 128, .qk_rope = 0, .interleave = false, .swiglu_lim = 10.0, .scale_f = 1.0, .seed = 92202 },
    };

    for (cases) |tc| {
        var target = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_img, tgt_cfg, .{
            .max_seq_len = 64,
            .max_tree_nodes = 16,
            .enable_feature_taps = true,
        });
        defer target.deinit(allocator);

        const mtp_cfg: DraftConfig = .{
            .format = .mtp,
            .mtp_arch = tc.arch,
            .vocab_size = 128,
            .draft_vocab_size = 128,
            .hidden_size = 128,
            .target_hidden_size = 128,
            .num_hidden_layers = 1,
            .num_attention_heads = 2,
            .num_key_value_heads = 2,
            .head_dim = tc.qk_nope + tc.qk_rope,
            .v_head_dim = 128,
            .q_lora_rank = 64,
            .kv_lora_rank = 32,
            .qk_nope_head_dim = tc.qk_nope,
            .qk_rope_head_dim = tc.qk_rope,
            .rope_interleave = tc.interleave,
            .index_share_for_mtp_iteration = true,
            .index_topk = 8,
            .num_experts = 8,
            .num_experts_per_tok = 2,
            .moe_intermediate_size = 64,
            .shared_expert_intermediate_size = 64,
            .routed_scaling_factor = tc.scale_f,
            .norm_topk_prob = true,
            .moe_scoring_sigmoid = true,
            .swiglu_limit = tc.swiglu_lim,
            .intermediate_size = 64,
            .rms_norm_eps = 1e-5,
            .rope_theta = 8_000_000.0,
            .share_embed_tokens = true,
            .share_lm_head = true,
        };
        const mtp_img = try buildSyntheticDraftAmfImage(allocator, mtp_cfg, tc.seed);
        defer allocator.free(mtp_img);

        var dh_gpu = try DraftHead.loadFromBytes(allocator, &target, mtp_img, mtp_cfg, .{
            .placement = .gpu,
            .max_seq_len = 64,
            .max_tree_slots = 16,
        });
        defer dh_gpu.deinit(allocator, &target);

        var dh_sme = try DraftHead.loadFromBytes(allocator, &target, mtp_img, mtp_cfg, .{
            .placement = .cpu_sme2,
            .max_seq_len = 64,
            .max_tree_slots = 16,
        });
        defer dh_sme.deinit(allocator, &target);

        try std.testing.expect(dh_gpu.use_custom_attn);
        try std.testing.expectEqual(Placement.cpu_sme2, dh_sme.effective_placement);
        try std.testing.expect(dh_sme.sme2_proj_bytes <= sme.MAX_L2_DRAFT_PROJ_BYTES);

        // Prefill 10 tokens and sync committed MLA KV ring.
        const prompt = [_]u32{ 4, 12, 23, 37, 55, 61, 72, 83, 94, 105 };
        _ = try target.prefillChunk(&prompt, 0, true);

        const shared_idx = [_]u32{ 0, 2, 5, 9 };
        const sp_info: SharedSparseIndices = .{
            .host_indices = &shared_idx,
            .k_selected = shared_idx.len,
            .max_k = shared_idx.len,
            .num_rows = 1,
            .valid = true,
        };
        dh_gpu.setSharedSparseIndices(sp_info);
        dh_sme.setSharedSparseIndices(sp_info);

        try dh_gpu.syncAccepted(&target, &prompt, target.accepted_top_bind, 0, 4);
        try dh_sme.syncAccepted(&target, &prompt, target.accepted_top_bind, 0, 4);

        try std.testing.expectEqual(@as(u64, 1), dh_gpu.zero_reindex_hits);
        try std.testing.expectEqual(@as(u64, 0), dh_gpu.indexer_dispatches);
        try std.testing.expectEqual(@as(u64, 1), dh_sme.zero_reindex_hits);
        try std.testing.expectEqual(@as(u64, 0), dh_sme.indexer_dispatches);
        try std.testing.expectEqual(dh_gpu.rootCandidates()[0].token_id, dh_sme.rootCandidates()[0].token_id);
        try std.testing.expect(std.math.isFinite(dh_gpu.rootCandidates()[0].prob) and dh_gpu.rootCandidates()[0].prob > 0.0);

        // Speculative tree step over shared sparse positions + newly drafted tree KV slots!
        const tree_toks = [_]u32{ dh_gpu.rootCandidates()[0].token_id, dh_gpu.rootCandidates()[1].token_id };
        const tree_parents = [_]u32{ 0, 0 };
        const tree_pos = [_]u32{ prompt.len, prompt.len };
        const masks = [_]u64{ 0b01, 0b10 };
        var cands_gpu: [2 * 4]moe.TopKCandidate = undefined;
        var cands_sme: [2 * 4]moe.TopKCandidate = undefined;
        try dh_gpu.stepTreeBatch(&target, &tree_toks, &tree_parents, &tree_pos, &masks, 0, 4, &cands_gpu);
        try dh_sme.stepTreeBatch(&target, &tree_toks, &tree_parents, &tree_pos, &masks, 0, 4, &cands_sme);

        try std.testing.expectEqual(@as(u64, 2), dh_gpu.zero_reindex_hits);
        try std.testing.expectEqual(@as(u64, 0), dh_gpu.indexer_dispatches);
        try std.testing.expectEqual(@as(u64, 2), dh_sme.zero_reindex_hits);
        try std.testing.expectEqual(@as(u64, 0), dh_sme.indexer_dispatches);
        try std.testing.expect(dh_sme.sme2_eval_count >= 2);
        try std.testing.expectEqual(cands_gpu[0].token_id, cands_sme[0].token_id);
        try std.testing.expectEqual(cands_gpu[4].token_id, cands_sme[4].token_id);
        try std.testing.expect(std.math.isFinite(cands_gpu[0].prob) and cands_gpu[0].prob > 0.0);

        dh_gpu.reset();
        try std.testing.expectEqual(@as(u64, 0), dh_gpu.zero_reindex_hits);
        try std.testing.expectEqual(@as(u64, 0), dh_gpu.indexer_dispatches);
    }
}
