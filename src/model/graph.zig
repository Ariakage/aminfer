//! Composable Model Graph & Recurrent-Depth Executor (`W9`, `W10`, `W16`, ADR-0015 & ADR-0019).
//!
//! Executes frontier hybrid-sparse architectures (`Qwen3.8-Flash-Next` / `qwen4_exp`),
//! native looped transformers (`LoopLM`: `Prelude -> Core x R -> Coda` with `inject_x_init`
//! and opt-in `sigmoid_linear` early exit), and adapter-extended looped models
//! (`gated_loop_adapter` with L2-resident ARMv9 SME2 `SmeStepLoraAdapter` and `alpha_0 == 0.0`
//! bit-exact identity) over one unified 16 KiB-aligned wired memory substrate with zero
//! hot-path heap allocations.
//!
//! Enforces all four mandatory `ADR-0019` hybrid-sparse loop isolations on `r < R - 1`:
//!   1. `GDN` (`R6`/`G8`): evaluates `r < R - 1` into wired `S_tmp` without mutating
//!      committed `S_t`; commits `S_tmp -> S_t` once at `r = R - 1` (or `commitTree`).
//!   2. `PLE` (`C10`): calls `PleStore.lookupForLoopStep` so `r = 0` populates wired
//!      `LoopPleScratch` and `r >= 1` performs `0` extra NVMe reads (`nvme_reads_delta == 0`).
//!   3. `QSA` (`G9`): runs `ami_qsa_index_blocks` at `r = 0` and passes
//!      `reuse_block_indices = true` for `r >= 1`.
//!   4. `MoE` (`G7`, `C5`): deduplicates active experts across `M = 1..64` nodes via CSR
//!      gather (`E <= 1024`, `top_k <= 16`, optional `shared_expert`) and restricts `r >= 1`
//!      routing to wired/resident `ExpertSlotPool` slots (`0` cold NVMe stalls on `r >= 1`).

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const expert = @import("../expert/source.zig");
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const arena_mod = @import("../mem/arena.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const dequant = @import("../gpu/dequant.zig");
const attention = @import("../gpu/attention.zig");
const moe = @import("../gpu/moe.zig");
const hybrid = @import("../gpu/hybrid.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const aio = @import("../storage/aio.zig");
const cache_mod = @import("../storage/cache.zig");
const ple_mod = @import("../storage/ple.zig");
const sme = @import("../cpu/sme.zig");
const accept_mod = @import("../spec/accept.zig");
const qwen3 = @import("qwen3.zig");

/// Maximum wired bytes per GPU weight shard (matches `qwen3.zig` to stay below
/// macOS single-`MTLBuffer` maxBufferLength on 24 GiB–512 GiB Apple Silicon).
const MAX_SHARD_BYTES: usize = 12 * 1024 * 1024 * 1024;
const MAX_ADAPTER_PROJS_PER_LAYER: usize = 7;

// ===========================================================================
// Section 1: Composable Operator Registry (`W9`, ADR-0015 & ADR-0019)
// ===========================================================================

pub const BackboneKind = enum {
    decoder_only,
    /// Reserved for M7 Tier 3 (`DeepSeek-V4.1-Flash-1017`).
    causal_enc_dec,

    pub fn label(self: BackboneKind) []const u8 {
        return switch (self) {
            .decoder_only => "decoder_only",
            .causal_enc_dec => "causal_enc_dec",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!BackboneKind {
        if (std.mem.eql(u8, s, "decoder_only")) return .decoder_only;
        if (std.mem.eql(u8, s, "causal_enc_dec")) return .causal_enc_dec;
        return error.InvalidOperatorAxis;
    }
};

pub const AttentionKind = enum {
    gqa_full,
    gated_deltanet,
    qsa_sparse,
    /// Reserved for M7 Tier 3 (`CSA2`).
    mla_csa2,

    pub fn label(self: AttentionKind) []const u8 {
        return switch (self) {
            .gqa_full => "gqa_full",
            .gated_deltanet => "gated_deltanet",
            .qsa_sparse => "qsa_sparse",
            .mla_csa2 => "mla_csa2",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!AttentionKind {
        if (std.mem.eql(u8, s, "gqa_full") or std.mem.eql(u8, s, "full_attention")) return .gqa_full;
        if (std.mem.eql(u8, s, "gated_deltanet") or std.mem.eql(u8, s, "linear_attention")) return .gated_deltanet;
        if (std.mem.eql(u8, s, "qsa_sparse") or std.mem.eql(u8, s, "qsa")) return .qsa_sparse;
        if (std.mem.eql(u8, s, "mla_csa2") or std.mem.eql(u8, s, "csa2")) return .mla_csa2;
        return error.InvalidOperatorAxis;
    }
};

pub const FfnKind = enum {
    dense_swiglu,
    moe_swiglu,

    pub fn label(self: FfnKind) []const u8 {
        return switch (self) {
            .dense_swiglu => "dense_swiglu",
            .moe_swiglu => "moe_swiglu",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!FfnKind {
        if (std.mem.eql(u8, s, "dense_swiglu")) return .dense_swiglu;
        if (std.mem.eql(u8, s, "moe_swiglu")) return .moe_swiglu;
        return error.InvalidOperatorAxis;
    }
};

pub const ResidualKind = enum {
    single_stream,
    gated_4branch,
    /// Reserved for M7 Tier 3 (`Single-Pass mHC`).
    mhc_single_pass,

    pub fn label(self: ResidualKind) []const u8 {
        return switch (self) {
            .single_stream => "single_stream",
            .gated_4branch => "gated_4branch",
            .mhc_single_pass => "mhc_single_pass",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!ResidualKind {
        if (std.mem.eql(u8, s, "single_stream")) return .single_stream;
        if (std.mem.eql(u8, s, "gated_4branch")) return .gated_4branch;
        if (std.mem.eql(u8, s, "mhc_single_pass")) return .mhc_single_pass;
        return error.InvalidOperatorAxis;
    }
};

pub const MemoryLookupKind = enum {
    none,
    ple_ngram,
    /// Reserved for M7 Tier 3 (`196B Engram`).
    engram,

    pub fn label(self: MemoryLookupKind) []const u8 {
        return switch (self) {
            .none => "none",
            .ple_ngram => "ple_ngram",
            .engram => "engram",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!MemoryLookupKind {
        if (std.mem.eql(u8, s, "none")) return .none;
        if (std.mem.eql(u8, s, "ple_ngram")) return .ple_ngram;
        if (std.mem.eql(u8, s, "engram")) return .engram;
        return error.InvalidOperatorAxis;
    }
};

pub const LoopMode = enum {
    single_pass,
    native_recurrent,
    gated_loop_adapter,

    pub fn label(self: LoopMode) []const u8 {
        return switch (self) {
            .single_pass => "single_pass",
            .native_recurrent => "native_recurrent",
            .gated_loop_adapter => "gated_loop_adapter",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!LoopMode {
        if (std.mem.eql(u8, s, "single_pass")) return .single_pass;
        if (std.mem.eql(u8, s, "native_recurrent")) return .native_recurrent;
        if (std.mem.eql(u8, s, "gated_loop_adapter")) return .gated_loop_adapter;
        return error.InvalidOperatorAxis;
    }
};

pub const LoopAdapterKind = enum {
    none,
    sme2_step_lora,

    pub fn label(self: LoopAdapterKind) []const u8 {
        return switch (self) {
            .none => "none",
            .sme2_step_lora => "sme2_step_lora",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!LoopAdapterKind {
        if (std.mem.eql(u8, s, "none")) return .none;
        if (std.mem.eql(u8, s, "sme2_step_lora")) return .sme2_step_lora;
        return error.InvalidOperatorAxis;
    }
};

pub const EarlyExitGateKind = enum {
    none,
    sigmoid_linear,

    pub fn label(self: EarlyExitGateKind) []const u8 {
        return switch (self) {
            .none => "none",
            .sigmoid_linear => "sigmoid_linear",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!EarlyExitGateKind {
        if (std.mem.eql(u8, s, "none")) return .none;
        if (std.mem.eql(u8, s, "sigmoid_linear")) return .sigmoid_linear;
        return error.InvalidOperatorAxis;
    }
};

pub const AdapterProjKind = enum(u8) {
    q_proj = 0,
    k_proj = 1,
    v_proj = 2,
    o_proj = 3,
    gate_proj = 4,
    up_proj = 5,
    down_proj = 6,

    pub fn parse(s: []const u8) ?AdapterProjKind {
        if (std.mem.eql(u8, s, "q_proj")) return .q_proj;
        if (std.mem.eql(u8, s, "k_proj")) return .k_proj;
        if (std.mem.eql(u8, s, "v_proj")) return .v_proj;
        if (std.mem.eql(u8, s, "o_proj")) return .o_proj;
        if (std.mem.eql(u8, s, "gate_proj")) return .gate_proj;
        if (std.mem.eql(u8, s, "up_proj")) return .up_proj;
        if (std.mem.eql(u8, s, "down_proj")) return .down_proj;
        return null;
    }
};

/// Canonical recurrent-depth descriptor (`ADR-0019`, `W9`/`W16`).
pub const LoopSchedule = struct {
    mode: LoopMode = .single_pass,
    start_layer: u32 = 0,
    end_layer: u32 = 0,
    max_loops: u32 = 1,
    inject_x_init: bool = false,
    adapter_kind: LoopAdapterKind = .none,
    adapter_rank: u32 = 0,
    adapter_packed_rank: u32 = 0,
    enforce_zero_alpha_r0: bool = true,
    early_exit_gate: EarlyExitGateKind = .none,
    early_exit_threshold: f32 = 0.5,

    pub fn isRecurrent(self: LoopSchedule) bool {
        return self.mode != .single_pass and self.max_loops > 1;
    }

    pub fn validate(self: LoopSchedule, num_layers: u32) GraphConfig.ConfigError!void {
        if (self.max_loops == 0) return error.InvalidConfig;
        if (num_layers == 0) return error.InvalidConfig;
        const end_l = if (self.end_layer == 0) num_layers else self.end_layer;
        if (self.start_layer >= end_l or end_l > num_layers) return error.InvalidConfig;
        if (self.adapter_kind == .sme2_step_lora) {
            if (self.adapter_rank == 0 or self.adapter_rank > sme.MAX_STEP_LORA_RANK) return error.InvalidConfig;
            if (self.adapter_packed_rank == 0 or (self.adapter_packed_rank % sme.GROUP_SIZE) != 0) return error.InvalidConfig;
            if (self.adapter_packed_rank < self.adapter_rank) return error.InvalidConfig;
            if (!self.enforce_zero_alpha_r0) return error.InvalidConfig;
        }
    }
};

/// Composable model graph operator axes (`W9`, ADR-0015).
pub const OperatorAxes = struct {
    backbone_kind: BackboneKind = .decoder_only,
    layer_attention_kinds: []AttentionKind = &.{},
    ffn_kind: FfnKind = .dense_swiglu,
    has_shared_expert: bool = false,
    residual_kind: ResidualKind = .single_stream,
    residual_branches: u32 = 1,
    memory_lookup_kind: MemoryLookupKind = .none,
    loop_schedule: LoopSchedule = .{},

    pub fn hasGdn(self: OperatorAxes) bool {
        for (self.layer_attention_kinds) |k| if (k == .gated_deltanet) return true;
        return false;
    }

    pub fn hasQsa(self: OperatorAxes) bool {
        for (self.layer_attention_kinds) |k| if (k == .qsa_sparse) return true;
        return false;
    }

    pub fn hasGqa(self: OperatorAxes) bool {
        for (self.layer_attention_kinds) |k| if (k == .gqa_full) return true;
        return false;
    }

    pub fn attentionSummaryLabel(self: OperatorAxes) []const u8 {
        if (self.layer_attention_kinds.len == 0) return "gqa_full";
        const first = self.layer_attention_kinds[0];
        for (self.layer_attention_kinds[1..]) |k| {
            if (k != first) return "hybrid_gdn_qsa";
        }
        return first.label();
    }
};

// ===========================================================================
// Section 2: `GraphConfig` & JSON Parsing (`qwen4_exp`, `LoopLM`, `Step-LoRA`)
// ===========================================================================

pub const GraphConfig = struct {
    axes: OperatorAxes,
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    norm_topk_prob: bool = true,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1_000_000.0,
    max_position_embeddings: u32 = 32768,
    tie_word_embeddings: bool = false,
    bos_token_id: u32 = 151643,
    eos_token_id: u32 = 151645,
    quant_bits: u8 = 3,
    group_size: u16 = 64,

    // Gated DeltaNet (`R6`/`G8`) geometry
    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 0,
    linear_value_head_dim: u32 = 0,
    linear_conv_kernel_dim: u32 = 4,

    // QSA (`G9`/`R5`) geometry + YaRN + interleaved MRoPE
    qsa_block_size: u32 = hybrid.QSA_DEFAULT_BLOCK_TOKENS,
    qsa_top_k_blocks: u32 = 8,
    qsa_window_size: u32 = 0,
    qsa_index_dim: u32 = 64,
    yarn: hybrid.YarnConfig = .{},
    mrope_section: hybrid.MropeSection = hybrid.MropeSection.QWEN38_DEFAULT,
    mrope_interleaved: bool = true,

    // PLE (`C10`) metadata
    ple_num_rows: u64 = 0,
    ple_embed_dim: u32 = 0,
    ple_ngram_order: u8 = 2,
    ple_num_shards: u32 = 0,
    ple_shard_files: [][]u8 = &.{},

    // MTP metadata (optional file name relative to config directory)
    mtp_amf_file: ?[]u8 = null,
    mtp_config_file: ?[]u8 = null,

    pub const ConfigError = error{
        InvalidJson,
        InvalidConfig,
        UnsupportedOperatorAxis,
        OutOfMemory,
    };

    pub fn deinit(self: *GraphConfig, allocator: std.mem.Allocator) void {
        if (self.axes.layer_attention_kinds.len > 0) {
            allocator.free(self.axes.layer_attention_kinds);
            self.axes.layer_attention_kinds = &.{};
        }
        for (self.ple_shard_files) |s| allocator.free(s);
        if (self.ple_shard_files.len > 0) {
            allocator.free(self.ple_shard_files);
            self.ple_shard_files = &.{};
        }
        if (self.mtp_amf_file) |s| {
            allocator.free(s);
            self.mtp_amf_file = null;
        }
        if (self.mtp_config_file) |s| {
            allocator.free(s);
            self.mtp_config_file = null;
        }
    }

    pub fn qDim(self: GraphConfig) u32 {
        return self.num_attention_heads * self.head_dim;
    }

    pub fn kvDim(self: GraphConfig) u32 {
        return self.num_key_value_heads * self.head_dim;
    }

    pub fn gdnQkDim(self: GraphConfig) u32 {
        return self.linear_num_key_heads * self.linear_key_head_dim;
    }

    pub fn gdnVDim(self: GraphConfig) u32 {
        return self.linear_num_value_heads * self.linear_value_head_dim;
    }

    pub fn validate(self: GraphConfig) ConfigError!void {
        if (self.axes.backbone_kind != .decoder_only) return error.UnsupportedOperatorAxis;
        if (self.axes.residual_kind == .mhc_single_pass) return error.UnsupportedOperatorAxis;
        if (self.axes.memory_lookup_kind == .engram) return error.UnsupportedOperatorAxis;
        if (self.vocab_size == 0 or self.hidden_size == 0 or self.num_hidden_layers == 0) return error.InvalidConfig;
        if (self.axes.layer_attention_kinds.len != self.num_hidden_layers) return error.InvalidConfig;
        for (self.axes.layer_attention_kinds) |ak| {
            if (ak == .mla_csa2) return error.UnsupportedOperatorAxis;
        }
        if (self.axes.hasQsa() or self.axes.hasGqa()) {
            if (self.num_attention_heads == 0 or self.num_key_value_heads == 0 or self.head_dim == 0) return error.InvalidConfig;
            if ((self.num_attention_heads % self.num_key_value_heads) != 0) return error.InvalidConfig;
            if ((self.head_dim % 4) != 0 or self.head_dim > hybrid.MAX_QSA_HEAD_DIM) return error.InvalidConfig;
        }
        if (self.axes.hasGdn()) {
            if (self.linear_num_key_heads == 0 or self.linear_num_value_heads == 0 or
                self.linear_key_head_dim == 0 or self.linear_value_head_dim == 0) return error.InvalidConfig;
            if ((self.linear_num_value_heads % self.linear_num_key_heads) != 0) return error.InvalidConfig;
            if (self.linear_key_head_dim > hybrid.MAX_GDN_HEAD_DIM or self.linear_value_head_dim > hybrid.MAX_GDN_HEAD_DIM) {
                return error.InvalidConfig;
            }
            if (self.linear_conv_kernel_dim > 8) return error.InvalidConfig;
        }
        switch (self.axes.ffn_kind) {
            .dense_swiglu => {
                if (self.intermediate_size == 0) return error.InvalidConfig;
            },
            .moe_swiglu => {
                if (self.num_experts == 0 or self.num_experts > moe.MAX_EXPERTS) return error.InvalidConfig;
                if (self.num_experts_per_tok == 0 or self.num_experts_per_tok > moe.MAX_TOP_K or
                    self.num_experts_per_tok > self.num_experts) return error.InvalidConfig;
                if (self.moe_intermediate_size == 0) return error.InvalidConfig;
                if (self.axes.has_shared_expert and self.shared_expert_intermediate_size == 0) return error.InvalidConfig;
            },
        }
        if (self.axes.residual_kind == .gated_4branch and self.axes.residual_branches != hybrid.GR_NUM_BRANCHES) {
            return error.InvalidConfig;
        }
        if (self.axes.memory_lookup_kind == .ple_ngram) {
            if (self.ple_embed_dim != 0 and self.ple_embed_dim != self.hidden_size) return error.InvalidConfig;
        }
        try self.axes.loop_schedule.validate(self.num_hidden_layers);
    }

    /// Lightweight detector telling CLI / server whether a `config.json` requires
    /// the composable `GraphModel` executor (`qwen4_exp`, `LoopLM`, `Step-LoRA`,
    /// hybrid GDN/QSA, 4-branch GR, PLE, or `max_loops > 1`).
    pub fn isGraphConfigJson(json_bytes: []const u8) bool {
        if (std.mem.indexOf(u8, json_bytes, "\"qwen4_exp\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"looplm\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"gated_deltanet\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"qsa_sparse\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"gated_4branch\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"ple_ngram\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"native_recurrent\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"gated_loop_adapter\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"sme2_step_lora\"") != null) return true;
        return false;
    }

    pub fn fromJson(allocator: std.mem.Allocator, json_bytes: []const u8) ConfigError!GraphConfig {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch {
            return error.InvalidJson;
        };
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidJson;
        const root = parsed.value.object;
        const obj = if (root.get("text_config")) |tc|
            if (tc == .object) tc.object else root
        else
            root;

        const num_layers = jsonU32(obj, "num_hidden_layers") orelse return error.InvalidConfig;
        const vocab_size = jsonU32(obj, "vocab_size") orelse return error.InvalidConfig;
        const hidden_size = jsonU32(obj, "hidden_size") orelse return error.InvalidConfig;
        const num_attn_heads = jsonU32(obj, "num_attention_heads") orelse 0;
        const num_kv_heads = jsonU32(obj, "num_key_value_heads") orelse num_attn_heads;
        const head_dim = jsonU32(obj, "head_dim") orelse
            if (num_attn_heads > 0) (hidden_size / num_attn_heads) else 0;

        const model_type = jsonStr(obj, "model_type") orelse jsonStr(root, "model_type") orelse "";
        const is_qwen4 = std.mem.eql(u8, model_type, "qwen4_exp");
        const is_looplm = std.mem.eql(u8, model_type, "looplm") or
            std.mem.eql(u8, model_type, "huginn") or
            std.mem.eql(u8, model_type, "ouro");

        var backbone_kind: BackboneKind = .decoder_only;
        var ffn_kind: FfnKind = if ((jsonU32(obj, "num_experts") orelse 0) > 0) .moe_swiglu else .dense_swiglu;
        var has_shared_expert: bool = (jsonU32(obj, "shared_expert_intermediate_size") orelse 0) > 0;
        var residual_branches: u32 = jsonU32(obj, "residual_branches") orelse (if (is_qwen4) @as(u32, 4) else 1);
        var residual_kind: ResidualKind = if (residual_branches == 4) .gated_4branch else .single_stream;
        var mem_kind: MemoryLookupKind = if (obj.get("ple") != null or root.get("ple") != null) .ple_ngram else .none;

        const layer_attns = try allocator.alloc(AttentionKind, num_layers);
        errdefer allocator.free(layer_attns);
        const full_interval = jsonU32(obj, "full_attention_interval") orelse 4;
        for (0..num_layers) |l| {
            layer_attns[l] = if (is_qwen4)
                (if (((l + 1) % full_interval) == 0) .qsa_sparse else .gated_deltanet)
            else
                .gqa_full;
        }

        // Override from `layer_types` if provided.
        if (obj.get("layer_types") orelse root.get("layer_types")) |lt_val| {
            if (lt_val == .array and lt_val.array.items.len == num_layers) {
                for (lt_val.array.items, 0..) |item, l| {
                    if (item == .string) {
                        layer_attns[l] = AttentionKind.parse(item.string) catch return error.UnsupportedOperatorAxis;
                    }
                }
            }
        }

        // Override from canonical `operator_axes` if present.
        if (obj.get("operator_axes") orelse root.get("operator_axes")) |oa_val| {
            if (oa_val == .object) {
                const oa = oa_val.object;
                if (jsonStr(oa, "backbone_kind")) |s| {
                    backbone_kind = BackboneKind.parse(s) catch return error.UnsupportedOperatorAxis;
                }
                if (jsonStr(oa, "ffn_kind")) |s| {
                    ffn_kind = FfnKind.parse(s) catch return error.UnsupportedOperatorAxis;
                }
                if (jsonBool(oa, "has_shared_expert")) |b| has_shared_expert = b;
                if (jsonStr(oa, "residual_kind")) |s| {
                    residual_kind = ResidualKind.parse(s) catch return error.UnsupportedOperatorAxis;
                }
                if (jsonU32(oa, "residual_branches")) |rb| residual_branches = rb;
                if (jsonStr(oa, "memory_lookup_kind")) |s| {
                    mem_kind = MemoryLookupKind.parse(s) catch return error.UnsupportedOperatorAxis;
                }
                if (oa.get("layer_attention_kinds")) |lak| {
                    if (lak == .array and lak.array.items.len == num_layers) {
                        for (lak.array.items, 0..) |item, l| {
                            if (item == .string) {
                                layer_attns[l] = AttentionKind.parse(item.string) catch return error.UnsupportedOperatorAxis;
                            }
                        }
                    }
                }
            }
        }

        // Parse `loop_schedule` (or infer from `looplm` / `recurrent_depth`).
        var ls: LoopSchedule = .{
            .mode = .single_pass,
            .start_layer = 0,
            .end_layer = num_layers,
            .max_loops = 1,
        };
        if (obj.get("loop_schedule") orelse root.get("loop_schedule")) |ls_val| {
            if (ls_val == .object) {
                const lo = ls_val.object;
                if (jsonStr(lo, "mode")) |s| {
                    ls.mode = LoopMode.parse(s) catch return error.InvalidConfig;
                }
                ls.start_layer = jsonU32(lo, "start_layer") orelse 0;
                ls.end_layer = jsonU32(lo, "end_layer") orelse num_layers;
                ls.max_loops = jsonU32(lo, "max_loops") orelse 1;
                ls.inject_x_init = jsonBool(lo, "inject_x_init") orelse false;
                if (jsonStr(lo, "adapter_kind")) |s| {
                    ls.adapter_kind = LoopAdapterKind.parse(s) catch return error.InvalidConfig;
                }
                ls.adapter_rank = jsonU32(lo, "adapter_rank") orelse 0;
                ls.adapter_packed_rank = jsonU32(lo, "adapter_packed_rank") orelse
                    if (ls.adapter_rank > 0) @intCast(amf.align64(ls.adapter_rank)) else 0;
                ls.enforce_zero_alpha_r0 = jsonBool(lo, "enforce_zero_alpha_r0") orelse true;
                if (jsonStr(lo, "early_exit_gate")) |s| {
                    ls.early_exit_gate = EarlyExitGateKind.parse(s) catch return error.InvalidConfig;
                }
                if (jsonF32(lo, "early_exit_threshold")) |th| ls.early_exit_threshold = th;
            }
        } else if (is_looplm or (jsonU32(obj, "recurrent_depth") orelse 1) > 1) {
            const prelude = jsonU32(obj, "num_prelude_layers") orelse (if (num_layers >= 3) @as(u32, 1) else 0);
            const coda = jsonU32(obj, "num_coda_layers") orelse (if (num_layers >= 3) @as(u32, 1) else 0);
            ls = .{
                .mode = .native_recurrent,
                .start_layer = @min(prelude, num_layers - 1),
                .end_layer = @max(@min(prelude, num_layers - 1) + 1, num_layers -| coda),
                .max_loops = @max(2, jsonU32(obj, "recurrent_depth") orelse 3),
                .inject_x_init = jsonBool(obj, "inject_x_init") orelse true,
                .adapter_kind = .none,
                .early_exit_gate = .none,
            };
        }

        // Parse YaRN + MRoPE `rope_scaling`.
        var yarn: hybrid.YarnConfig = .{};
        var mrope_sec: hybrid.MropeSection = if (head_dim == 64)
            hybrid.MropeSection.QWEN38_DEFAULT
        else if (head_dim > 0)
            hybrid.MropeSection.text1d(head_dim)
        else
            hybrid.MropeSection.QWEN38_DEFAULT;
        var mrope_interleaved: bool = true;
        if (obj.get("rope_scaling")) |rs_val| {
            if (rs_val == .object) {
                const rs = rs_val.object;
                if (jsonF32(rs, "factor")) |f| yarn.factor = f;
                if (jsonF32(rs, "mscale")) |m| yarn.mscale = m;
                if (jsonF32(rs, "beta_fast")) |bf| yarn.beta_fast = bf;
                if (jsonF32(rs, "beta_slow")) |bs| yarn.beta_slow = bs;
                if (jsonU32(rs, "original_max_position_embeddings")) |omp| yarn.original_max_pos = omp;
                if (jsonBool(rs, "mrope_interleaved")) |mi| mrope_interleaved = mi;
                if (rs.get("mrope_section")) |ms_val| {
                    if (ms_val == .array and ms_val.array.items.len == 3) {
                        mrope_sec = .{ .section = .{
                            jsonItemU32(ms_val.array.items[0]) orelse 11,
                            jsonItemU32(ms_val.array.items[1]) orelse 11,
                            jsonItemU32(ms_val.array.items[2]) orelse 10,
                        } };
                    }
                }
            }
        }

        // Parse PLE metadata.
        var ple_num_rows: u64 = 0;
        var ple_embed_dim: u32 = hidden_size;
        var ple_ngram_order: u8 = 2;
        var ple_num_shards: u32 = 0;
        var ple_shard_files: [][]u8 = &.{};
        errdefer {
            for (ple_shard_files) |s| allocator.free(s);
            if (ple_shard_files.len > 0) allocator.free(ple_shard_files);
        }
        if (obj.get("ple") orelse root.get("ple")) |ple_val| {
            if (ple_val == .object) {
                const po = ple_val.object;
                ple_num_rows = jsonU64(po, "num_rows") orelse 0;
                ple_embed_dim = jsonU32(po, "embed_dim") orelse hidden_size;
                ple_ngram_order = @intCast(@min(8, @max(1, jsonU32(po, "ngram_order") orelse 2)));
                ple_num_shards = jsonU32(po, "num_shards") orelse 0;
                if (po.get("shard_files")) |sf_val| {
                    if (sf_val == .array and sf_val.array.items.len > 0) {
                        const files = try allocator.alloc([]u8, sf_val.array.items.len);
                        var built: usize = 0;
                        errdefer {
                            for (0..built) |i| allocator.free(files[i]);
                            allocator.free(files);
                        }
                        for (sf_val.array.items, 0..) |item, i| {
                            if (item != .string) return error.InvalidConfig;
                            files[i] = try allocator.dupe(u8, item.string);
                            built += 1;
                        }
                        ple_shard_files = files;
                        if (ple_num_shards == 0) ple_num_shards = @intCast(files.len);
                    }
                }
            }
        }

        // Parse optional MTP filenames.
        var mtp_amf_file: ?[]u8 = null;
        errdefer if (mtp_amf_file) |s| allocator.free(s);
        var mtp_config_file: ?[]u8 = null;
        errdefer if (mtp_config_file) |s| allocator.free(s);
        if (obj.get("mtp") orelse root.get("mtp")) |mtp_val| {
            if (mtp_val == .object) {
                if (jsonStr(mtp_val.object, "amf_file")) |s| mtp_amf_file = try allocator.dupe(u8, s);
                if (jsonStr(mtp_val.object, "config_file")) |s| mtp_config_file = try allocator.dupe(u8, s);
            }
        }

        // Quantization hints.
        var qbits: u8 = 3;
        var gsize: u16 = 64;
        if (root.get("quantization") orelse obj.get("quantization")) |qv| {
            if (qv == .object) {
                if (jsonU32(qv.object, "bits")) |b| qbits = @intCast(b);
                if (jsonU32(qv.object, "group_size")) |g| gsize = @intCast(g);
            }
        }

        const cfg: GraphConfig = .{
            .axes = .{
                .backbone_kind = backbone_kind,
                .layer_attention_kinds = layer_attns,
                .ffn_kind = ffn_kind,
                .has_shared_expert = has_shared_expert,
                .residual_kind = residual_kind,
                .residual_branches = residual_branches,
                .memory_lookup_kind = mem_kind,
                .loop_schedule = ls,
            },
            .vocab_size = vocab_size,
            .hidden_size = hidden_size,
            .num_hidden_layers = num_layers,
            .num_attention_heads = num_attn_heads,
            .num_key_value_heads = num_kv_heads,
            .head_dim = head_dim,
            .intermediate_size = jsonU32(obj, "intermediate_size") orelse 0,
            .moe_intermediate_size = jsonU32(obj, "moe_intermediate_size") orelse 0,
            .shared_expert_intermediate_size = jsonU32(obj, "shared_expert_intermediate_size") orelse 0,
            .num_experts = jsonU32(obj, "num_experts") orelse 0,
            .num_experts_per_tok = jsonU32(obj, "num_experts_per_tok") orelse 0,
            .norm_topk_prob = jsonBool(obj, "norm_topk_prob") orelse true,
            .rms_norm_eps = jsonF32(obj, "rms_norm_eps") orelse 1e-6,
            .rope_theta = jsonF32(obj, "rope_theta") orelse 1_000_000.0,
            .max_position_embeddings = jsonU32(obj, "max_position_embeddings") orelse 32768,
            .tie_word_embeddings = jsonBool(obj, "tie_word_embeddings") orelse false,
            .bos_token_id = jsonU32(obj, "bos_token_id") orelse jsonU32(root, "bos_token_id") orelse 151643,
            .eos_token_id = jsonU32(obj, "eos_token_id") orelse jsonU32(root, "eos_token_id") orelse 151645,
            .quant_bits = qbits,
            .group_size = gsize,
            .linear_num_key_heads = jsonU32(obj, "linear_num_key_heads") orelse 0,
            .linear_num_value_heads = jsonU32(obj, "linear_num_value_heads") orelse 0,
            .linear_key_head_dim = jsonU32(obj, "linear_key_head_dim") orelse 0,
            .linear_value_head_dim = jsonU32(obj, "linear_value_head_dim") orelse 0,
            .linear_conv_kernel_dim = jsonU32(obj, "linear_conv_kernel_dim") orelse 4,
            .qsa_block_size = jsonU32(obj, "qsa_block_size") orelse hybrid.QSA_DEFAULT_BLOCK_TOKENS,
            .qsa_top_k_blocks = jsonU32(obj, "qsa_top_k_blocks") orelse 8,
            .qsa_window_size = jsonU32(obj, "qsa_window_size") orelse 0,
            .qsa_index_dim = jsonU32(obj, "qsa_index_dim") orelse 64,
            .yarn = yarn,
            .mrope_section = mrope_sec,
            .mrope_interleaved = mrope_interleaved,
            .ple_num_rows = ple_num_rows,
            .ple_embed_dim = ple_embed_dim,
            .ple_ngram_order = ple_ngram_order,
            .ple_num_shards = ple_num_shards,
            .ple_shard_files = ple_shard_files,
            .mtp_amf_file = mtp_amf_file,
            .mtp_config_file = mtp_config_file,
        };
        try cfg.validate();
        return cfg;
    }
};

fn jsonStr(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = obj.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn jsonBool(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const v = obj.get(key) orelse return null;
    return if (v == .bool) v.bool else null;
}

fn jsonU32(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = obj.get(key) orelse return null;
    return jsonItemU32(v);
}

fn jsonU64(obj: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        else => null,
    };
}

fn jsonItemU32(v: std.json.Value) ?u32 {
    return switch (v) {
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        else => null,
    };
}

fn jsonF32(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

// ===========================================================================
// Section 3: Load Options, Errors & ADR-0019 Loop Isolation Telemetry
// ===========================================================================

pub const GraphLoadOptions = struct {
    max_seq_len: u32 = 2048,
    max_tree_nodes: u32 = 64,
    chunk_tokens: u32 = attention.DEFAULT_CHUNK_TOKENS,
    timeout_ms: u64 = 30_000,
    wired_budget_override: ?budget_mod.Budget = null,
    enable_feature_taps: bool = false,
    aux_layer_ids: ?[3]u32 = null,
    /// Optional explicit PLE shard `.amf` paths (`C10`, `F_NOCACHE`).
    ple_paths: []const [:0]const u8 = &.{},
    /// Optional external `Step-LoRA` adapter `.amf` file (`W16`).
    loop_adapter_amf_path: ?[:0]const u8 = null,
    /// Optional runtime override for `LoopSchedule.max_loops` (`--max-loops`).
    max_loops_override: ?u32 = null,
    /// Opt-in approximate token-adaptive early exit (`--loop-early-exit`).
    enable_loop_early_exit: bool = false,
    early_exit_threshold: ?f32 = null,
    /// Cold-expert `ExpertSlotPool` slot count (`0` auto-sizes when cold experts exist).
    cold_slot_count: u32 = 0,
    ple_cache_pages: usize = ple_mod.DEFAULT_CACHE_PAGES,
};

pub const LoopIsolationStats = struct {
    total_loop_steps: u64 = 0,
    early_exit_tokens: u64 = 0,
    /// Isolation #1 (GDN R6/G8): scratch evaluations on `r < R-1` and final commits on `r == R-1`.
    gdn_scratch_evals: u64 = 0,
    gdn_final_commits: u64 = 0,
    /// Isolation #2 (PLE C10): `r = 0` fills vs `r >= 1` zero-NVMe-read scratch reuses.
    ple_r0_fills: u64 = 0,
    ple_r_ge1_reuses: u64 = 0,
    ple_r_ge1_nvme_reads: u64 = 0,
    /// Isolation #3 (QSA G9): `r = 0` block-index dispatches vs `r >= 1` index reuses.
    qsa_index_computes: u64 = 0,
    qsa_index_reuses: u64 = 0,
    /// Isolation #4 (MoE G7/C5): `r >= 1` pinned-only routing masks & cold stalls on `r >= 1`.
    moe_r_ge1_pinned_routes: u64 = 0,
    moe_r_ge1_cold_stalls: u64 = 0,
    /// W16 Step-LoRA: `r = 0` zero-alpha skips vs active SME2 evaluations.
    steplora_zero_alpha_skips: u64 = 0,
    steplora_active_evals: u64 = 0,
};

pub const GraphLoadError = error{
    MissingTensor,
    TensorShapeMismatch,
    UnsupportedDType,
    UnsupportedOperatorAxis,
    InvalidJson,
    InvalidConfig,
    NonZeroLoopZeroAlpha,
    BudgetExceeded,
    TooManyWeightShards,
    Overflow,
    OutOfMemory,
} || amf.ParseError || amf.IoError || metal.Error || sysinfo.Error ||
    page.ShareError || expert.NvmeExpert.InitError ||
    arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError || hybrid.ArenaError || hybrid.HybridKvRing.InitError ||
    moe.WorkspaceError || ple_mod.Error || sme.WiredScratchError ||
    cache_mod.PoolError || cache_mod.CacheError || aio.EcorePoolError;

pub const GraphStepError = error{
    SequenceFull,
    TokenOutOfRange,
    TreeNotEnabled,
    TreeTooLarge,
    InvalidBatchSize,
    InvalidLoopStep,
    Overflow,
} || metal.Error || dequant.EncodeError || moe.EncodeError ||
    hybrid.EncodeError || hybrid.ArenaError || ple_mod.Error ||
    sme.Error || cache_mod.CacheError || accept_mod.VerifyError;

// ===========================================================================
// Section 4: Internal Layer & Runtime Structures
// ===========================================================================

const WeightShard = struct {
    arena: arena_mod.WiredArena,
    buffer: metal.Buffer,
};

const GrLayerWeights = struct {
    read_gate: metal.BufferBinding,
    write_gate: metal.BufferBinding,
    decay_gate: metal.BufferBinding,
    dtype: quant.FloatDType,
};

const GdnLayerWeights = struct {
    gdn_idx: u32,
    q_proj: qwen3.LinearBinding,
    k_proj: qwen3.LinearBinding,
    v_proj: qwen3.LinearBinding,
    g_proj: qwen3.LinearBinding,
    o_proj: qwen3.LinearBinding,
    a_proj: metal.BufferBinding,
    b_proj: metal.BufferBinding,
    ab_dtype: quant.FloatDType,
    a_log: metal.BufferBinding,
    dt_bias: metal.BufferBinding,
    conv1d_w: ?metal.BufferBinding,
    conv_k: u32,
    out_norm: ?qwen3.NormBinding,
    // Per-layer scratch offsets for tree commit (`k_tree`, `v_tree`, `g_tree`, `beta_tree`, raw QKV).
    tree_k_off: u64,
    tree_v_off: u64,
    tree_g_off: u64,
    tree_beta_off: u64,
    tree_q_raw_off: u64,
    tree_k_raw_off: u64,
    tree_v_raw_off: u64,
    conv_state_off: u64,
    conv_state_tmp_off: u64,
};

const QsaLayerWeights = struct {
    qsa_idx: u32,
    q_proj: qwen3.LinearBinding,
    k_proj: qwen3.LinearBinding,
    v_proj: qwen3.LinearBinding,
    o_proj: qwen3.LinearBinding,
    q_norm: ?metal.BufferBinding,
    k_norm: ?metal.BufferBinding,
    qk_norm_dtype: quant.FloatDType,
    kv_ring: hybrid.HybridKvRing,
    index_buf: hybrid.QsaIndexBuffer,
    block_keys_off: u64,
    max_blocks: u32,
    tree_k_off: u64,
    tree_v_off: u64,
};

const GqaLayerWeights = struct {
    gqa_idx: u32,
    q_proj: qwen3.LinearBinding,
    k_proj: qwen3.LinearBinding,
    v_proj: qwen3.LinearBinding,
    o_proj: qwen3.LinearBinding,
    q_norm: ?metal.BufferBinding,
    k_norm: ?metal.BufferBinding,
    qk_norm_dtype: quant.FloatDType,
    kv_tiles_binding: metal.BufferBinding,
    kv_tile_start: usize,
    tree_k_off: u64,
    tree_v_off: u64,
};

const LayerAttention = union(enum) {
    gqa: GqaLayerWeights,
    gdn: GdnLayerWeights,
    qsa: QsaLayerWeights,
};

const GraphLayerWeights = struct {
    shard_idx: u16,
    input_layernorm: qwen3.NormBinding,
    post_attention_layernorm: qwen3.NormBinding,
    gr: ?GrLayerWeights = null,
    attn: LayerAttention,

    // Dense SwiGLU (when `ffn_kind == .dense_swiglu`)
    dense_gate: qwen3.LinearBinding = undefined,
    dense_up: qwen3.LinearBinding = undefined,
    dense_down: qwen3.LinearBinding = undefined,

    // MoE SwiGLU (when `ffn_kind == .moe_swiglu`)
    router_binding: metal.BufferBinding = undefined,
    router_dtype: moe.RouterWeightDType = .bf16,
    router_quant_dtype: ?quant.WeightDType = null,
    router_scale_bias: ?metal.BufferBinding = null,
    expert_weights_buf: metal.Buffer = undefined,
    expert_table: metal.BufferBinding = undefined,
    expert_table_host: []moe.ExpertWeights = &.{},
    expert_weight_dtype: quant.WeightDType = .q3_affine,
    expert_dtypes_binding: ?metal.BufferBinding = null,
    expert_scheme_counts: [moe.NUM_SCHEME_BUCKETS]u32 = @splat(0),
    is_hetero_moe: bool = false,
    has_cold_experts: bool = false,
    cold_expert_count: u32 = 0,
    shared_expert: ?moe.ExpertWeights = null,
    shared_expert_buf: ?metal.Buffer = null,
    shared_expert_dtype: ?quant.WeightDType = null,
    shared_gate_weight: ?metal.BufferBinding = null,
    shared_gate_dtype: moe.RouterWeightDType = .bf16,
};

const PleRuntime = struct {
    store: ple_mod.PleStore,
    scratch: ple_mod.LoopPleScratch,
    f32_tmp: []f32,
};

const ColdRuntime = struct {
    pool: cache_mod.ExpertSlotPool,
    reader: aio.AsyncExpertReader,
    expert_cache: cache_mod.ExpertCache,
    amf_file: ?amf.AmfFile,
    slot_count: u32,
};

const LoopAdapterRuntime = struct {
    external_arena: ?arena_mod.WiredArena = null,
    dispatcher: sme.SmeDispatcher,
    scratch: sme.StepLoraScratch,
    delta_f32: []f32,
    max_steps: u32,
    num_layers: u32,
    /// Flat `[max_steps * num_layers * MAX_ADAPTER_PROJS_PER_LAYER]` table.
    entries: []?sme.SmeStepLoraAdapter,

    fn get(self: *const LoopAdapterRuntime, step: u32, layer: u32, proj: AdapterProjKind) ?sme.SmeStepLoraAdapter {
        if (step >= self.max_steps or layer >= self.num_layers) return null;
        const idx = (@as(usize, step) * self.num_layers + layer) * MAX_ADAPTER_PROJS_PER_LAYER + @backingInt(proj);
        return self.entries[idx];
    }

    fn set(self: *LoopAdapterRuntime, step: u32, layer: u32, proj: AdapterProjKind, adapter: sme.SmeStepLoraAdapter) void {
        const idx = (@as(usize, step) * self.num_layers + layer) * MAX_ADAPTER_PROJS_PER_LAYER + @backingInt(proj);
        self.entries[idx] = adapter;
    }

    fn deinit(self: *LoopAdapterRuntime, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
        allocator.free(self.delta_f32);
        self.scratch.deinit();
        self.dispatcher.deinit();
        if (self.external_arena) |*ea| ea.deinit();
    }
};

// ===========================================================================
// Section 5: `GraphModel` Composable Executor
// ===========================================================================

pub const GraphModel = struct {
    config: GraphConfig,
    effective_max_loops: u32,
    enable_loop_early_exit: bool,
    early_exit_threshold: f32,
    max_seq_len: u32,
    max_tree_nodes: u32,
    chunk_tokens: u32,
    num_chunks: u32,
    seq_len: u32,
    timeout_ms: u64,
    wired_budget_bytes: u64,
    total_wired_bytes: u64,
    enable_feature_taps: bool,
    aux_layer_ids: [3]u32,

    // Hardware handles (public for `DraftHead` & `SpecEngine` interop)
    device: metal.Device,
    library: metal.Library,
    compute: metal.Compute,
    dequant_pipes: dequant.DequantPipelines,
    attn_pipes: attention.AttentionPipelines,
    moe_pipes: moe.MoePipelines,
    hybrid_pipes: hybrid.HybridPipelines,

    // Multi-shard wired weight arenas + scratch arena
    weight_shards: []WeightShard,
    scratch_arena: arena_mod.WiredArena,
    scratch_buf: metal.Buffer,

    // Optional hybrid state arenas
    gdn_arena: ?hybrid.GdnStateArena,
    gr_arena: ?hybrid.GrStreamArena,
    gqa_kv_tiles: []metal.KvTile,

    // Global weights
    embed: qwen3.LinearBinding,
    final_norm: qwen3.NormBinding,
    lm_head: qwen3.LinearBinding,
    early_exit_gate_host: ?[]const u8,
    early_exit_gate_dtype: quant.FloatDType,
    layers: []GraphLayerWeights,

    // Optional subsystems (heap-pinned where internal pointers exist)
    ple_rt: ?*PleRuntime,
    cold_rt: ?*ColdRuntime,
    adapter_rt: ?*LoopAdapterRuntime,

    // Preallocated hot-path scratch host slices & GPU bindings (zero alloc on decode/verify)
    token_history: []u32,
    token_ids_host: []u32,
    positions_host: []u32,
    pos3d_host: []hybrid.Pos3D,
    ancestor_masks_host: []u64,
    out_token_ids_host: []u32,
    logits_host: []const f16,
    topk_host: []moe.TopKCandidate,
    ple_row_indices_host: []u64,
    ple_f16_host: []f16,
    x_host: []f16,
    x_norm_host: []f16,
    q_host: []f16,
    k_host: []f16,
    v_host: []f16,
    attn_out_host: []f16,
    proj_out_host: []f16,
    dense_gate_host: []f16,
    dense_up_host: []f16,
    dense_act_host: []f16,
    accepted_top_host: []f16,
    accepted_aux_host: []f16,

    token_ids_bind: metal.BufferBinding,
    positions_bind: metal.BufferBinding,
    pos3d_bind: metal.BufferBinding,
    ancestor_masks_bind: metal.BufferBinding,
    out_token_id_bind: metal.BufferBinding,
    logits_bind: metal.BufferBinding,
    topk_bind: metal.BufferBinding,
    ple_f16_base_off: u64,
    x_bind: metal.BufferBinding,
    x_init_bind: metal.BufferBinding,
    x_norm_bind: metal.BufferBinding,
    q_bind: metal.BufferBinding,
    k_bind: metal.BufferBinding,
    v_bind: metal.BufferBinding,
    q_conv_bind: metal.BufferBinding,
    k_conv_bind: metal.BufferBinding,
    v_conv_bind: metal.BufferBinding,
    gdn_z_bind: metal.BufferBinding,
    gdn_g_bind: metal.BufferBinding,
    gdn_beta_bind: metal.BufferBinding,
    gr_g_read_bind: metal.BufferBinding,
    gr_g_write_bind: metal.BufferBinding,
    gr_decay_bind: metal.BufferBinding,
    attn_out_bind: metal.BufferBinding,
    proj_out_bind: metal.BufferBinding,
    dense_gate_bind: metal.BufferBinding,
    dense_up_bind: metal.BufferBinding,
    dense_act_bind: metal.BufferBinding,
    gqa_scratch_bind: metal.BufferBinding,
    moe_ws: ?moe.MoeWorkspace,
    layer_sched_hdr_raw: []u8,
    layer_sched_hdr_off: u64,
    accepted_top_bind: metal.BufferBinding,
    accepted_aux_bind: metal.BufferBinding,
    aux_tree_base_off: u64,

    // Accounting for `SpecEngine` & `bench-m6`
    single_expert_bytes: u64,
    total_layers_non_expert_bytes: u64,
    total_layers_static_bytes: u64,
    global_head_bytes: u64,
    embed_row_alg_bytes: u64,
    last_logits_row: usize,
    last_verified_nodes: u32,
    last_verified_tokens: [hybrid.MAX_TREE_NODES]u32,
    last_verified_masks: [hybrid.MAX_TREE_NODES]u64,
    loop_stats: LoopIsolationStats,

    // -----------------------------------------------------------------------
    // Loading & Teardown
    // -----------------------------------------------------------------------

    const AmfSource = union(enum) {
        bytes: struct { parsed: *const amf.Parsed, raw: []const u8 },
        file: struct { file: *amf.AmfFile, path: [:0]const u8 },

        fn entries(self: AmfSource) []const amf.TensorEntry {
            return switch (self) {
                .bytes => |b| b.parsed.entries,
                .file => |f| f.file.parsed.entries,
            };
        }

        fn find(self: AmfSource, name: []const u8) ?*const amf.TensorEntry {
            return switch (self) {
                .bytes => |b| b.parsed.find(name),
                .file => |f| f.file.parsed.find(name),
            };
        }

        fn loadIntoArena(self: AmfSource, entry: *const amf.TensorEntry, arena: *arena_mod.WiredArena) (amf.ParseError || amf.IoError || arena_mod.WiredArena.AllocError)![]align(page.APPLE_PAGE_SIZE) u8 {
            return switch (self) {
                .bytes => |b| b.parsed.loadBlobFromBytes(b.raw, entry, arena),
                .file => |f| f.file.loadBlob(entry, arena),
            };
        }
    };

    pub fn loadFromBytes(
        allocator: std.mem.Allocator,
        amf_bytes: []const u8,
        config: GraphConfig,
        options: GraphLoadOptions,
    ) GraphLoadError!GraphModel {
        var parsed = try amf.Parsed.parseBytes(allocator, amf_bytes);
        defer parsed.deinit(allocator);
        return loadInternal(allocator, .{ .bytes = .{ .parsed = &parsed, .raw = amf_bytes } }, config, options);
    }

    pub fn loadFromFile(
        allocator: std.mem.Allocator,
        amf_path: [:0]const u8,
        config: GraphConfig,
        options: GraphLoadOptions,
    ) GraphLoadError!GraphModel {
        var file = try amf.AmfFile.open(allocator, amf_path);
        defer file.close(allocator);
        return loadInternal(allocator, .{ .file = .{ .file = &file, .path = amf_path } }, config, options);
    }

    pub fn deinit(self: *GraphModel, allocator: std.mem.Allocator) void {
        if (self.adapter_rt) |art| {
            art.deinit(allocator);
            allocator.destroy(art);
            self.adapter_rt = null;
        }
        if (self.cold_rt) |crt| {
            crt.expert_cache.deinit(allocator);
            crt.reader.deinit(allocator);
            crt.pool.deinit();
            if (crt.amf_file) |*f| f.close(allocator);
            allocator.destroy(crt);
            self.cold_rt = null;
        }
        if (self.ple_rt) |prt| {
            allocator.free(prt.f32_tmp);
            prt.scratch.deinit(allocator);
            prt.store.deinit(allocator);
            allocator.destroy(prt);
            self.ple_rt = null;
        }
        for (self.layers) |*layer| {
            switch (layer.attn) {
                .qsa => |*qsa| {
                    qsa.kv_ring.removeResident(self.compute) catch |err| {
                        std.debug.panic("QSA kv_ring.removeResident failed: {}", .{err});
                    };
                    qsa.kv_ring.deinit(allocator);
                },
                else => {},
            }
        }
        if (self.gqa_kv_tiles.len > 0) {
            self.compute.removeTilesResident(self.gqa_kv_tiles) catch |err| {
                std.debug.panic("removeTilesResident failed in GraphModel.deinit: {}", .{err});
            };
            for (self.gqa_kv_tiles) |tile| tile.close();
            allocator.free(self.gqa_kv_tiles);
        }
        self.compute.removeResident(&.{self.scratch_buf}) catch |err| {
            std.debug.panic("removeResident(scratch) failed in GraphModel.deinit: {}", .{err});
        };
        for (self.weight_shards) |ws| {
            self.compute.removeResident(&.{ws.buffer}) catch |err| {
                std.debug.panic("removeResident(weight_shard) failed in GraphModel.deinit: {}", .{err});
            };
        }
        self.scratch_buf.close();
        self.scratch_arena.deinit();
        for (self.weight_shards) |*ws| {
            ws.buffer.close();
            ws.arena.deinit();
        }
        allocator.free(self.weight_shards);
        allocator.free(self.layers);
        allocator.free(self.token_history);
        allocator.free(self.ple_row_indices_host);

        self.hybrid_pipes.close();
        self.moe_pipes.close();
        self.attn_pipes.close();
        self.dequant_pipes.close();
        self.compute.close();
        self.library.close();
        self.device.close();
    }

    pub fn resetKvCache(self: *GraphModel) void {
        self.seq_len = 0;
        self.last_logits_row = 0;
        self.last_verified_nodes = 0;
        if (self.gdn_arena) |*ga| ga.resetAll();
        for (self.layers) |layer| {
            switch (layer.attn) {
                .gdn => |g| {
                    if (g.conv_k > 1) {
                        const conv_dim = 2 * self.config.gdnQkDim() + self.config.gdnVDim();
                        const state_bytes = @as(usize, g.conv_k - 1) * conv_dim * @sizeOf(f16);
                        const base = self.scratch_arena.bytes();
                        @memset(base[@intCast(g.conv_state_off)..][0..state_bytes], 0);
                        @memset(base[@intCast(g.conv_state_tmp_off)..][0..state_bytes], 0);
                    }
                },
                else => {},
            }
        }
        if (self.ple_rt) |prt| {
            prt.scratch.resetStep();
        }
    }

    pub fn resetLoopIsolationStats(self: *GraphModel) void {
        self.loop_stats = .{};
    }

    pub fn loopIsolationStats(self: *const GraphModel) LoopIsolationStats {
        return self.loop_stats;
    }

    pub fn pleStats(self: *const GraphModel) ?ple_mod.Stats {
        const prt = self.ple_rt orelse return null;
        return prt.store.stats();
    }

    pub fn hasColdExperts(self: *const GraphModel) bool {
        return self.cold_rt != null;
    }

    pub fn coldExpertStats(self: *const GraphModel) ?qwen3.ColdExpertStats {
        const crt = self.cold_rt orelse return null;
        var total_cold: u32 = 0;
        for (self.layers) |l| total_cold += l.cold_expert_count;
        return .{
            .total_cold_experts = total_cold,
            .slot_count = crt.slot_count,
            .io_backend = crt.reader.activeBackend(),
            .ane_predictor_active = false,
            .ane_predictor_uses_ne = false,
            .ane_predictor_compile_ms = 0.0,
            .cache_telemetry = crt.expert_cache.telemetry(),
        };
    }

    pub fn totalWiredBytes(self: *const GraphModel) u64 {
        return self.total_wired_bytes;
    }

    pub fn shardCount(self: *const GraphModel) usize {
        return self.weight_shards.len;
    }

    pub fn resetColdTelemetry(self: *GraphModel) void {
        if (self.cold_rt) |crt| {
            crt.expert_cache.resetTelemetry();
        }
    }

    pub fn singleExpertBytes(self: *const GraphModel) u64 {
        return self.single_expert_bytes;
    }

    pub fn totalLayersNonExpertBytes(self: *const GraphModel) u64 {
        return self.total_layers_non_expert_bytes;
    }

    pub fn bytesPerDecodeToken(self: *const GraphModel, seq_len: u32) u64 {
        const loops: u64 = self.effective_max_loops;
        var kv_bytes: u64 = 0;
        for (self.layers) |layer| {
            switch (layer.attn) {
                .gqa => {
                    kv_bytes += 2 * @as(u64, self.config.num_key_value_heads) * seq_len * self.config.head_dim * @sizeOf(f16);
                },
                .qsa => |qsa| {
                    const vis_tok: u64 = if (qsa.kv_ring.window_size > 0)
                        @min(@as(u64, seq_len), qsa.kv_ring.window_size)
                    else
                        @min(@as(u64, seq_len), @as(u64, self.config.qsa_top_k_blocks) * self.config.qsa_block_size);
                    kv_bytes += 2 * @as(u64, self.config.num_key_value_heads) * vis_tok * self.config.head_dim * @sizeOf(f16);
                },
                .gdn => {
                    const state_b = @as(u64, self.config.linear_num_value_heads) *
                        self.config.linear_key_head_dim * self.config.linear_value_head_dim * @sizeOf(f32);
                    kv_bytes += 2 * state_b; // read + write recurrent state
                },
            }
        }
        const layer_bytes: u64 = if (self.config.axes.ffn_kind == .moe_swiglu)
            self.total_layers_non_expert_bytes +
                @as(u64, self.config.num_hidden_layers) * self.config.num_experts_per_tok * self.single_expert_bytes
        else
            self.total_layers_static_bytes;
        const ls = self.config.axes.loop_schedule;
        const core_layers: u64 = if (ls.end_layer > ls.start_layer) (ls.end_layer - ls.start_layer) else self.config.num_hidden_layers;
        const extra_loop_bytes: u64 = if (loops > 1)
            (loops - 1) * (layer_bytes * core_layers / self.config.num_hidden_layers)
        else
            0;
        return layer_bytes + extra_loop_bytes + kv_bytes + self.global_head_bytes;
    }

    pub fn readLogitsF16(self: *const GraphModel) []const f16 {
        return self.logits_host[0..self.config.vocab_size];
    }

    pub fn readLastLogitsF16(self: *const GraphModel) []const f16 {
        const v: usize = self.config.vocab_size;
        return self.logits_host[self.last_logits_row * v .. (self.last_logits_row + 1) * v];
    }

    pub fn readTopHiddenF16(self: *const GraphModel, num_nodes: u32) []const f16 {
        return self.accepted_top_host[0 .. @as(usize, num_nodes) * self.config.hidden_size];
    }

    pub fn readAuxConcatF16(self: *const GraphModel, num_nodes: u32) []const f16 {
        return self.accepted_aux_host[0 .. @as(usize, num_nodes) * 3 * self.config.hidden_size];
    }

    pub fn sampleFromLastLogits(
        self: *const GraphModel,
        temperature: f32,
        rng: std.Random,
        p_res: []f32,
    ) accept_mod.VerifyError!u32 {
        try accept_mod.softmaxRow(f16, self.readLastLogitsF16(), temperature, p_res);
        return accept_mod.sampleCategorical(p_res[0..self.config.vocab_size], rng);
    }

    // -----------------------------------------------------------------------
    // GPU Linear Encoding Helpers (Compatible with `DraftHead` / `Qwen3Model`)
    // -----------------------------------------------------------------------

    pub fn encodeLinearGemv(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        lin: qwen3.LinearBinding,
        x: metal.BufferBinding,
        y: metal.BufferBinding,
        barrier_before: bool,
    ) dequant.EncodeError!void {
        switch (lin.dtype) {
            .q2_affine => try self.dequant_pipes.encodeGemv(frame, .q2, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .q3_affine => try self.dequant_pipes.encodeGemv(frame, .q3, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .q4_affine => try self.dequant_pipes.encodeGemv(frame, .q4, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .f16 => try self.dequant_pipes.encodeDenseGemv(frame, .f16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .bf16 => try self.dequant_pipes.encodeDenseGemv(frame, .bf16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
        }
    }

    pub fn encodeLinearBatched(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        lin: qwen3.LinearBinding,
        x: metal.BufferBinding,
        y: metal.BufferBinding,
        m: u32,
        barrier_before: bool,
    ) dequant.EncodeError!void {
        if (m == 1) {
            return self.encodeLinearGemv(frame, lin, x, y, barrier_before);
        }
        switch (lin.dtype) {
            .q2_affine => try self.dequant_pipes.encodeGemvBatch(frame, .q2, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .q3_affine => try self.dequant_pipes.encodeGemvBatch(frame, .q3, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .q4_affine => try self.dequant_pipes.encodeGemvBatch(frame, .q4, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .f16 => try self.dequant_pipes.encodeDenseGemvBatch(frame, .f16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .bf16 => try self.dequant_pipes.encodeDenseGemvBatch(frame, .bf16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
        }
    }

    // -----------------------------------------------------------------------
    // Public Inference Entrypoints: Prefill, Decode, VerifyTree, CommitTree, DraftShallow
    // -----------------------------------------------------------------------

    /// Batched prompt prefill across `tokens` starting at `start_pos` in chunks
    /// of `<= max_tree_nodes` with full `LoopSchedule` & 4-isolation enforcement.
    pub fn prefillChunk(
        self: *GraphModel,
        tokens: []const u32,
        start_pos: u32,
        compute_logits: bool,
    ) GraphStepError!u32 {
        if (tokens.len == 0) return error.InvalidBatchSize;
        if (start_pos + tokens.len > self.max_seq_len) return error.SequenceFull;
        for (tokens) |tid| {
            if (tid >= self.config.vocab_size) return error.TokenOutOfRange;
        }

        const chunk_cap: usize = @max(@as(usize, 1), @as(usize, self.max_tree_nodes));
        var offset: usize = 0;
        var last_argmax: u32 = 0;
        while (offset < tokens.len) {
            const take: usize = @min(chunk_cap, tokens.len - offset);
            const cur_base: u32 = start_pos + @as(u32, @intCast(offset));
            const is_last_chunk = (offset + take == tokens.len);

            var pos_buf: [hybrid.MAX_TREE_NODES]u32 = undefined;
            var mask_buf: [hybrid.MAX_TREE_NODES]u64 = undefined;
            var accept_nodes: [hybrid.MAX_TREE_NODES]u16 = undefined;
            for (0..take) |i| {
                pos_buf[i] = cur_base + @as(u32, @intCast(i));
                // Causal chain within the chunk: node `i` attends to `0..=i`.
                mask_buf[i] = if (i == 63) std.math.maxInt(u64) else ((@as(u64, 1) << @intCast(i + 1)) - 1);
                accept_nodes[i] = @intCast(i);
            }

            const vr = try self.runForwardInternal(
                tokens[offset .. offset + take],
                pos_buf[0..take],
                mask_buf[0..take],
                cur_base,
                self.effective_max_loops,
                is_last_chunk and compute_logits,
                0,
                null,
                true, // commit state on final loop step
            );
            // Record committed token history for subsequent PLE N-gram lookups.
            for (0..take) |i| {
                self.token_history[cur_base + @as(u32, @intCast(i))] = tokens[offset + i];
            }
            if (self.enable_feature_taps) {
                try self.commitFeatureTaps(accept_nodes[0..take]);
            }
            self.seq_len = cur_base + @as(u32, @intCast(take));
            if (is_last_chunk and compute_logits) {
                last_argmax = vr.argmax_tokens[take - 1];
            }
            offset += take;
        }
        return last_argmax;
    }

    pub fn prefillToken(self: *GraphModel, token_id: u32, pos: u32) GraphStepError!void {
        const single = [_]u32{token_id};
        _ = try self.prefillChunk(&single, pos, false);
    }

    /// Single-step decode at `pos` with full `LoopSchedule` (`R = effective_max_loops`),
    /// enforcing all four `ADR-0019` loop isolations and committing KV/GDN state once at `r = R - 1`.
    pub fn stepDecode(self: *GraphModel, token_id: u32, pos: u32) GraphStepError!u32 {
        if (pos >= self.max_seq_len) return error.SequenceFull;
        if (token_id >= self.config.vocab_size) return error.TokenOutOfRange;

        const tok = [_]u32{token_id};
        const p = [_]u32{pos};
        const mask = [_]u64{1};
        const vr = try self.runForwardInternal(
            &tok,
            &p,
            &mask,
            pos,
            self.effective_max_loops,
            true,
            0,
            null,
            true, // commit state on final pass `r == R - 1`
        );
        self.token_history[pos] = token_id;
        if (self.enable_feature_taps) {
            const acc = [_]u16{0};
            try self.commitFeatureTaps(&acc);
        }
        self.seq_len = pos + 1;
        return vr.argmax_tokens[0];
    }

    /// Evaluates an `M = 1..64` speculative candidate tree under full `Core x R`
    /// without mutating committed KV tiles or committed `GDN` state `S_t`.
    /// Intermediate loop steps `r < R - 1` skip full-vocabulary `lm_head`.
    pub fn verifyTree(
        self: *GraphModel,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
    ) GraphStepError!qwen3.VerifyTreeResult {
        return self.verifyTreeWithLoops(token_ids, positions, ancestor_masks, base_pos, self.effective_max_loops);
    }

    pub fn verifyTreeWithLoops(
        self: *GraphModel,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        num_loops: u32,
    ) GraphStepError!qwen3.VerifyTreeResult {
        if (self.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (token_ids.len == 0 or token_ids.len > self.max_tree_nodes) return error.TreeTooLarge;
        if (positions.len != token_ids.len or ancestor_masks.len != token_ids.len) return error.InvalidBatchSize;
        for (token_ids, positions) |tid, pos| {
            if (tid >= self.config.vocab_size) return error.TokenOutOfRange;
            if (pos >= self.max_seq_len) return error.SequenceFull;
        }
        const vr = try self.runForwardInternal(
            token_ids,
            positions,
            ancestor_masks,
            base_pos,
            num_loops,
            true,
            0,
            null,
            false, // NEVER mutate committed KV / GDN state during tree verification
        );
        self.last_verified_nodes = @intCast(token_ids.len);
        for (token_ids, 0..) |tid, i| {
            self.last_verified_tokens[i] = tid;
            self.last_verified_masks[i] = ancestor_masks[i];
        }
        return vr;
    }

    /// Commits the accepted path `accepted_nodes[0..k]` from the most recent
    /// `verifyTree` pass into committed `KvTile`s, `HybridKvRing` + `block_keys`,
    /// `GDN` committed `S_t` + `conv_state`, feature taps, and PLE `token_history`.
    pub fn commitTree(self: *GraphModel, base_pos: u32, accepted_nodes: []const u16) GraphStepError!void {
        return self.commitTreeKv(base_pos, accepted_nodes);
    }

    pub fn commitTreeKv(self: *GraphModel, base_pos: u32, accepted_nodes: []const u16) GraphStepError!void {
        if (accepted_nodes.len == 0) return;
        if (accepted_nodes.len > self.max_tree_nodes) return error.TreeTooLarge;
        if (base_pos + accepted_nodes.len > self.max_seq_len) return error.SequenceFull;

        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();
        var had_dispatch = false;

        for (self.layers) |*layer| {
            switch (layer.attn) {
                .gdn => |g| {
                    const ga = &(self.gdn_arena.?);
                    try self.hybrid_pipes.encodeCommitTreeState(&frame, .{
                        .k_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_k_off },
                        .v_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_v_off },
                        .g_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_g_off },
                        .beta_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_beta_off },
                        .accepted_nodes = .{ .inline_u16 = accepted_nodes },
                        .accepted_count = @intCast(accepted_nodes.len),
                        .state_in = try ga.committedBinding(self.scratch_buf, g.gdn_idx),
                        .state_out = try ga.committedBinding(self.scratch_buf, g.gdn_idx),
                        .num_k_heads = self.config.linear_num_key_heads,
                        .num_v_heads = self.config.linear_num_value_heads,
                        .head_k_dim = self.config.linear_key_head_dim,
                        .head_v_dim = self.config.linear_value_head_dim,
                    }, had_dispatch);
                    had_dispatch = true;
                    if (g.conv1d_w != null and g.conv_k > 1) {
                        try self.hybrid_pipes.encodeGdnConv1dCommit(&frame, .{
                            .q_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_q_raw_off },
                            .k_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_k_raw_off },
                            .v_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_v_raw_off },
                            .conv_state_in = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                            .conv_state_out = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                            .accepted_nodes = accepted_nodes,
                            .gdn_qk = self.config.gdnQkDim(),
                            .gdn_v = self.config.gdnVDim(),
                            .conv_k = g.conv_k,
                        }, true);
                    }
                },
                .qsa => |*qsa| {
                    try self.encodeQsaCommitAccepted(&frame, qsa, base_pos, accepted_nodes, had_dispatch);
                    had_dispatch = true;
                },
                .gqa => |gqa| {
                    try self.encodeGqaCommitAccepted(&frame, gqa, base_pos, accepted_nodes, had_dispatch);
                    had_dispatch = true;
                },
            }
        }

        if (self.enable_feature_taps) {
            try self.encodeCommitFeatureTaps(&frame, accepted_nodes, had_dispatch);
            had_dispatch = true;
        }

        if (had_dispatch) {
            const sig = try frame.commit();
            try self.compute.wait(sig, self.timeout_ms);
        } else {
            frame.abort();
        }

        // Advance committed token history for PLE N-gram hashes.
        for (accepted_nodes, 0..) |nid, i| {
            if (nid < self.last_verified_nodes) {
                self.token_history[base_pos + @as(u32, @intCast(i))] = self.last_verified_tokens[nid];
            }
        }
        self.seq_len = base_pos + @as(u32, @intCast(accepted_nodes.len));
    }

    /// `D12` `SelfSpecRecurrent` shallow drafting primitive (`0` extra draft-model RAM):
    /// evaluates `token_ids[0..m]` with `loops = 1` (`r = 0` only, without mutating
    /// committed KV/GDN state) and fills `out_candidates[0 .. m * eff_k]` via a single
    /// GPU softmax top-K dispatch.
    pub fn draftShallowBatch(
        self: *GraphModel,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        top_k: u32,
        out_candidates: []moe.TopKCandidate,
    ) GraphStepError!u32 {
        if (token_ids.len == 0 or token_ids.len > self.max_tree_nodes) return error.TreeTooLarge;
        if (positions.len != token_ids.len or ancestor_masks.len != token_ids.len) return error.InvalidBatchSize;
        if (top_k == 0 or top_k > moe.MAX_TOP_K) return error.InvalidBatchSize;
        const eff_k = @min(top_k, self.config.vocab_size);
        if (out_candidates.len < token_ids.len * eff_k) return error.InvalidBatchSize;
        for (token_ids, positions) |tid, pos| {
            if (tid >= self.config.vocab_size) return error.TokenOutOfRange;
            if (pos >= self.max_seq_len) return error.SequenceFull;
        }

        _ = try self.runForwardInternal(
            token_ids,
            positions,
            ancestor_masks,
            base_pos,
            1, // shallow r = 1 / r = 0 pass (0 extra draft-model RAM)
            true,
            eff_k,
            out_candidates[0 .. token_ids.len * eff_k],
            false, // NEVER mutate committed state while drafting
        );
        return eff_k;
    }

    // -----------------------------------------------------------------------
    // Core Forward Execution with All 4 ADR-0019 Loop Isolations
    // -----------------------------------------------------------------------

    fn runForwardInternal(
        self: *GraphModel,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        num_loops: u32,
        compute_logits: bool,
        topk_k: u32,
        topk_out: ?[]moe.TopKCandidate,
        commit_state_on_final: bool,
    ) GraphStepError!qwen3.VerifyTreeResult {
        const m: u32 = @intCast(token_ids.len);
        const h: u32 = self.config.hidden_size;
        const ls = self.config.axes.loop_schedule;
        const R: u32 = @max(@as(u32, 1), num_loops);
        const core_start: u32 = if (R > 1) ls.start_layer else 0;
        const core_end: u32 = if (R > 1)
            (if (ls.end_layer > ls.start_layer) ls.end_layer else self.config.num_hidden_layers)
        else
            self.config.num_hidden_layers;

        // Stage inputs into preallocated wired host slices (0 heap allocs).
        for (0..m) |i| {
            self.token_ids_host[i] = token_ids[i];
            self.positions_host[i] = positions[i];
            self.pos3d_host[i] = hybrid.Pos3D.text(positions[i]);
            self.ancestor_masks_host[i] = ancestor_masks[i];
        }

        // Reset `LoopPleScratch` at the start of every step (`r = 0` will populate).
        if (self.ple_rt) |prt| {
            prt.scratch.resetStep();
        }

        var frame_opt: ?metal.Compute.Frame = null;
        errdefer if (frame_opt) |*f| f.abort();

        // 1. Token embedding lookup `[m, H]`.
        {
            const f = try self.ensureFrame(&frame_opt);
            try self.moe_pipes.encodeEmbedLookup(f, self.embed.dtype, .{
                .weight = self.embed.weight,
                .scale_bias = self.embed.scale_bias,
                .tokens = .{ .token_ids = self.token_ids_bind },
                .out = self.x_bind,
                .m = m,
                .vocab_size = self.config.vocab_size,
                .hidden_size = h,
            }, false);

            // Initialize 4-branch GR stream (`G10`) if active.
            if (self.gr_arena) |*gr| {
                try self.hybrid_pipes.encodeGrInitStream0(f, .{
                    .x0 = self.x_bind,
                    .stream = gr.binding(self.scratch_buf),
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = self.max_tree_nodes,
                }, true);
            }
        }

        // 2. Prelude layers `[0 .. core_start)` (executed once at `r = 0`).
        for (0..core_start) |l| {
            try self.execLayer(
                &frame_opt,
                l,
                token_ids,
                ancestor_masks,
                base_pos,
                m,
                0,
                true, // prelude always commits on single pass when `commit_state_on_final`
                commit_state_on_final,
            );
        }

        // Save `x_init` after Prelude when `inject_x_init` is active (`LoopLM`).
        if (R > 1 and ls.inject_x_init) {
            const f = try self.ensureFrame(&frame_opt);
            if (self.gr_arena) |*gr| {
                try self.hybrid_pipes.encodeGrCollapse(f, .{
                    .stream = gr.binding(self.scratch_buf),
                    .out = self.x_init_bind,
                    .mode = .stream0,
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = self.max_tree_nodes,
                }, true);
            } else {
                try self.moe_pipes.encodeCopyStrided(f, .{
                    .src = self.x_bind,
                    .dst = self.x_init_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
            }
        }

        // 3. Core recurrent block `[core_start .. core_end) x R`.
        //    Enforces all 4 ADR-0019 loop isolations on `r < R - 1` and skips `lm_head` on `r < R - 1`.
        var r: u32 = 0;
        while (r < R) : (r += 1) {
            self.loop_stats.total_loop_steps += 1;

            // Inject saved `x_init` on loop re-entry `r >= 1` (`0.5 * h + 0.5 * x_init`).
            if (r >= 1 and ls.inject_x_init) {
                const f = try self.ensureFrame(&frame_opt);
                if (self.gr_arena) |*gr| {
                    try self.hybrid_pipes.encodeBlendF16(f, .{
                        .dst = gr.binding(self.scratch_buf), // branch 0 starts at offset 0
                        .src = self.x_init_bind,
                        .count = m * h,
                        .a = 0.5,
                        .b = 0.5,
                    }, true);
                } else {
                    try self.hybrid_pipes.encodeBlendF16(f, .{
                        .dst = self.x_bind,
                        .src = self.x_init_bind,
                        .count = m * h,
                        .a = 0.5,
                        .b = 0.5,
                    }, true);
                }
            }

            const is_final_pass = (r == R - 1);
            for (core_start..core_end) |l| {
                try self.execLayer(
                    &frame_opt,
                    l,
                    token_ids,
                    ancestor_masks,
                    base_pos,
                    m,
                    r,
                    is_final_pass,
                    commit_state_on_final and is_final_pass,
                );
            }

            // Opt-in approximate token-adaptive early exit (`--loop-early-exit`) after `r = 0`.
            if (r == 0 and R > 1 and self.enable_loop_early_exit and
                ls.early_exit_gate == .sigmoid_linear and self.early_exit_gate_host != null)
            {
                if (self.gr_arena) |*gr| {
                    const f = try self.ensureFrame(&frame_opt);
                    try self.hybrid_pipes.encodeGrCollapse(f, .{
                        .stream = gr.binding(self.scratch_buf),
                        .out = self.x_bind,
                        .mode = .stream0,
                        .m = m,
                        .hidden_dim = h,
                        .branch_token_stride = self.max_tree_nodes,
                    }, true);
                }
                try self.flushFrame(&frame_opt);
                if (self.shouldEarlyExitCpu(m)) {
                    self.loop_stats.early_exit_tokens += m;
                    // Commit core layers' `r = 0` scratch states if `commit_state_on_final`!
                    if (commit_state_on_final) {
                        try self.commitCoreLayersAfterEarlyExit(&frame_opt, core_start, core_end, base_pos, m);
                    }
                    break;
                }
            }
        }

        // 4. Coda layers `[core_end .. num_hidden_layers)` (executed once after Core).
        for (core_end..self.config.num_hidden_layers) |l| {
            try self.execLayer(
                &frame_opt,
                l,
                token_ids,
                ancestor_masks,
                base_pos,
                m,
                0,
                true,
                commit_state_on_final,
            );
        }

        // Collapse 4-branch GR stream to single-stream `x_bind` (`mean` of 4 branches).
        if (self.gr_arena) |*gr| {
            const f = try self.ensureFrame(&frame_opt);
            try self.hybrid_pipes.encodeGrCollapse(f, .{
                .stream = gr.binding(self.scratch_buf),
                .out = self.x_bind,
                .mode = .mean,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
            }, true);
        }

        // 5. Final norm + `lm_head` + argmax / optional top-K (NEVER run on `r < R - 1`).
        if (compute_logits) {
            const f = try self.ensureFrame(&frame_opt);
            try self.attn_pipes.encodeRmsNorm(f, .{
                .x = self.x_bind,
                .w = self.final_norm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = self.final_norm.dtype,
            }, true);
            try self.encodeLinearBatched(f, self.lm_head, self.x_norm_bind, self.logits_bind, m, true);
            try self.moe_pipes.encodeArgmaxRows(f, .{
                .logits = self.logits_bind,
                .out_token_ids = self.out_token_id_bind,
                .num_rows = m,
                .vocab_size = self.config.vocab_size,
                .logits_dtype = .f16,
            }, true);
            if (topk_k > 0) {
                try self.moe_pipes.encodeTopKLogitsRows(f, .{
                    .logits = self.logits_bind,
                    .out_candidates = self.topk_bind,
                    .num_rows = m,
                    .vocab_size = self.config.vocab_size,
                    .top_k = topk_k,
                    .logits_dtype = .f16,
                }, true);
            }
        }

        try self.flushFrame(&frame_opt);
        if (compute_logits) {
            self.last_logits_row = @as(usize, m) - 1;
            if (topk_k > 0) {
                if (topk_out) |dst| {
                    const count = @as(usize, m) * topk_k;
                    @memcpy(dst[0..count], self.topk_host[0..count]);
                }
            }
        }

        // Gather measured unique active experts across MoE layers.
        var total_unique: u64 = 0;
        if (self.config.axes.ffn_kind == .moe_swiglu) {
            for (0..self.layers.len) |l| {
                const hdr: *const moe.ScheduleHeader = @ptrCast(@alignCast(self.layer_sched_hdr_raw.ptr + l * 64));
                total_unique += hdr.num_unique_experts;
            }
        }
        const avg_unique: f64 = if (self.config.axes.ffn_kind == .moe_swiglu)
            @as(f64, @floatFromInt(total_unique)) / @as(f64, @floatFromInt(self.layers.len))
        else
            0.0;

        const core_layers: u64 = core_end - core_start;
        const base_layer_bytes: u64 = if (self.config.axes.ffn_kind == .moe_swiglu)
            self.total_layers_non_expert_bytes + total_unique * self.single_expert_bytes
        else
            self.total_layers_static_bytes;
        const loop_extra_bytes: u64 = if (R > 1)
            (R - 1) * (base_layer_bytes * core_layers / self.config.num_hidden_layers)
        else
            0;
        const head_bytes: u64 = if (compute_logits)
            @as(u64, m) * self.embed_row_alg_bytes + (self.global_head_bytes - self.embed_row_alg_bytes)
        else
            @as(u64, m) * self.embed_row_alg_bytes;

        return .{
            .num_nodes = m,
            .argmax_tokens = self.out_token_ids_host[0..m],
            .logits_f16 = self.logits_host[0 .. @as(usize, m) * self.config.vocab_size],
            .total_unique_experts = total_unique,
            .avg_unique_experts_per_layer = avg_unique,
            .actual_bytes_moved = base_layer_bytes + loop_extra_bytes + head_bytes,
        };
    }

    fn execLayer(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        token_ids: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        m: u32,
        loop_step: u32,
        is_final_pass: bool,
        commit_kv_and_gdn: bool,
    ) GraphStepError!void {
        const layer = &self.layers[l];
        const h = self.config.hidden_size;

        // --- Isolation #2 (PLE C10): single lookup at `r = 0`, 0 NVMe reads at `r >= 1` ---
        if (self.ple_rt) |prt| {
            try self.preparePleLayer(prt, l, token_ids, ancestor_masks, base_pos, m, loop_step);
        }

        // --- Pre-Attention 4-Branch GR Read Gate (G10) ---
        if (self.gr_arena) |*gr| {
            const gr_w = layer.gr.?;
            const f = try self.ensureFrame(frame_opt);
            try self.hybrid_pipes.encodeGrComputeGates(f, .{
                .stream = gr.binding(self.scratch_buf),
                .w_read = gr_w.read_gate,
                .w_write = gr_w.write_gate,
                .w_decay = gr_w.decay_gate,
                .g_read_out = self.gr_g_read_bind,
                .g_write_out = self.gr_g_write_bind,
                .decay_out = self.gr_decay_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
                .weight_dtype = gr_w.dtype,
            }, true);
            try self.hybrid_pipes.encodeGrReadGate(f, .{
                .stream = gr.binding(self.scratch_buf),
                .g_read = self.gr_g_read_bind,
                .x_in = self.x_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
            }, true);
        }

        // Accumulate PLE contribution into `x_bind` (inside the same Metal CB, 0 mid-pass sync).
        if (self.ple_rt != null) {
            const f = try self.ensureFrame(frame_opt);
            try self.moe_pipes.encodeResidualAdd(f, .{
                .x = self.x_bind,
                .delta = self.pleLayerBinding(l),
                .count = m * h,
            }, true);
        }

        // Input RMSNorm -> `x_norm_bind`.
        {
            const f = try self.ensureFrame(frame_opt);
            try self.attn_pipes.encodeRmsNorm(f, .{
                .x = self.x_bind,
                .w = layer.input_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.input_layernorm.dtype,
            }, true);
        }

        // --- Attention / Sequence Mixer Dispatch ---
        switch (layer.attn) {
            .gdn => |g| {
                try self.execGdnAttention(frame_opt, l, g, m, loop_step, is_final_pass, commit_kv_and_gdn);
            },
            .qsa => |*qsa| {
                try self.execQsaAttention(frame_opt, l, qsa, base_pos, m, loop_step, commit_kv_and_gdn);
            },
            .gqa => |gqa| {
                try self.execGqaAttention(frame_opt, l, gqa, base_pos, m, loop_step, commit_kv_and_gdn);
            },
        }

        // Residual update after attention (`G10` 4-branch write-gate or standard residual add).
        if (self.gr_arena) |*gr| {
            const f = try self.ensureFrame(frame_opt);
            try self.hybrid_pipes.encodeGrWriteGate(f, .{
                .stream = gr.binding(self.scratch_buf),
                .y_delta = self.proj_out_bind,
                .decay = self.gr_decay_bind,
                .g_write = self.gr_g_write_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
            }, true);
            // Read updated stream back into `x_bind` for FFN sub-block.
            try self.hybrid_pipes.encodeGrReadGate(f, .{
                .stream = gr.binding(self.scratch_buf),
                .g_read = self.gr_g_read_bind,
                .x_in = self.x_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
            }, true);
        } else {
            const f = try self.ensureFrame(frame_opt);
            try self.moe_pipes.encodeResidualAdd(f, .{
                .x = self.x_bind,
                .delta = self.proj_out_bind,
                .count = m * h,
            }, true);
        }

        // Post-attention RMSNorm -> `x_norm_bind`.
        {
            const f = try self.ensureFrame(frame_opt);
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

        // --- FFN / MoE Dispatch ---
        switch (self.config.axes.ffn_kind) {
            .dense_swiglu => {
                try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .gate_proj, layer.dense_gate, self.x_norm_bind, self.x_norm_host, self.dense_gate_bind, self.dense_gate_host, m, true);
                try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .up_proj, layer.dense_up, self.x_norm_bind, self.x_norm_host, self.dense_up_bind, self.dense_up_host, m, false);
                {
                    const f = try self.ensureFrame(frame_opt);
                    try self.moe_pipes.encodeSiluMul(f, .{
                        .gate = self.dense_gate_bind,
                        .up = self.dense_up_bind,
                        .out = self.dense_act_bind,
                        .count = m * self.config.intermediate_size,
                    }, true);
                }
                try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .down_proj, layer.dense_down, self.dense_act_bind, self.dense_act_host, self.proj_out_bind, self.proj_out_host, m, true);
                try self.applyFfnResidual(frame_opt, m, h);
            },
            .moe_swiglu => {
                try self.execMoeFfn(frame_opt, l, layer, m, loop_step);
            },
        }

        // Capture EAGLE-3 aux taps on final loop pass if enabled.
        if (self.enable_feature_taps and is_final_pass) {
            for (self.aux_layer_ids, 0..) |tap_l, slot| {
                if (tap_l == @as(u32, @intCast(l))) {
                    const f = try self.ensureFrame(frame_opt);
                    if (self.gr_arena) |*gr| {
                        try self.hybrid_pipes.encodeGrCollapse(f, .{
                            .stream = gr.binding(self.scratch_buf),
                            .out = self.x_bind,
                            .mode = .mean,
                            .m = m,
                            .hidden_dim = h,
                            .branch_token_stride = self.max_tree_nodes,
                        }, true);
                    }
                    try self.moe_pipes.encodeCopyStrided(f, .{
                        .src = self.x_bind,
                        .dst = .{ .buffer = self.scratch_buf, .offset = self.aux_tree_base_off },
                        .num_rows = m,
                        .copy_cols = h,
                        .src_stride = h,
                        .dst_stride = 3 * h,
                        .dst_col_off = @as(u32, @intCast(slot)) * h,
                    }, true);
                }
            }
        }
    }

    fn applyFfnResidual(self: *GraphModel, frame_opt: *?metal.Compute.Frame, m: u32, h: u32) GraphStepError!void {
        const f = try self.ensureFrame(frame_opt);
        if (self.gr_arena) |*gr| {
            try self.hybrid_pipes.encodeGrWriteGate(f, .{
                .stream = gr.binding(self.scratch_buf),
                .y_delta = self.proj_out_bind,
                .decay = self.gr_decay_bind,
                .g_write = self.gr_g_write_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
            }, true);
        } else {
            try self.moe_pipes.encodeResidualAdd(f, .{
                .x = self.x_bind,
                .delta = self.proj_out_bind,
                .count = m * h,
            }, true);
        }
    }

    fn execGdnAttention(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        g: GdnLayerWeights,
        m: u32,
        loop_step: u32,
        is_final_pass: bool,
        commit_state: bool,
    ) GraphStepError!void {
        const ga = &(self.gdn_arena.?);
        const qk_dim = self.config.gdnQkDim();
        const v_dim = self.config.gdnVDim();

        // Linear projections: `q`, `k`, `v`, `z` (output gate), plus `g` & `beta`.
        const q_raw_bind: metal.BufferBinding = if (g.conv1d_w != null)
            .{ .buffer = self.scratch_buf, .offset = g.tree_q_raw_off }
        else
            self.q_bind;
        const k_raw_bind: metal.BufferBinding = if (g.conv1d_w != null)
            .{ .buffer = self.scratch_buf, .offset = g.tree_k_raw_off }
        else
            .{ .buffer = self.scratch_buf, .offset = g.tree_k_off };
        const v_raw_bind: metal.BufferBinding = if (g.conv1d_w != null)
            .{ .buffer = self.scratch_buf, .offset = g.tree_v_raw_off }
        else
            .{ .buffer = self.scratch_buf, .offset = g.tree_v_off };

        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .q_proj, g.q_proj, self.x_norm_bind, self.x_norm_host, q_raw_bind, self.q_host, m, true);
        {
            const f = try self.ensureFrame(frame_opt);
            try self.encodeLinearBatched(f, g.k_proj, self.x_norm_bind, k_raw_bind, m, false);
            try self.encodeLinearBatched(f, g.v_proj, self.x_norm_bind, v_raw_bind, m, false);
            try self.encodeLinearBatched(f, g.g_proj, self.x_norm_bind, self.gdn_z_bind, m, false);
            try self.hybrid_pipes.encodeGdnComputeGates(f, .{
                .x = self.x_norm_bind,
                .w_a = g.a_proj,
                .w_b = g.b_proj,
                .a_log = g.a_log,
                .dt_bias = g.dt_bias,
                .g_out = .{ .buffer = self.scratch_buf, .offset = g.tree_g_off },
                .beta_out = .{ .buffer = self.scratch_buf, .offset = g.tree_beta_off },
                .m = m,
                .hidden_dim = self.config.hidden_size,
                .num_v_heads = self.config.linear_num_value_heads,
                .weight_dtype = g.ab_dtype,
            }, false);

            // Optional depthwise causal Conv1D + SiLU with loop-isolated conv state.
            const q_eff_bind: metal.BufferBinding = if (g.conv1d_w != null) self.q_conv_bind else q_raw_bind;
            const k_eff_bind: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = g.tree_k_off };
            const v_eff_bind: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = g.tree_v_off };
            if (g.conv1d_w) |cw| {
                const is_tree = !commit_state or m > 1;
                const conv_out_off = if (commit_state) g.conv_state_off else g.conv_state_tmp_off;
                try self.hybrid_pipes.encodeGdnConv1dSilu(f, .{
                    .q_in = q_raw_bind,
                    .k_in = k_raw_bind,
                    .v_in = v_raw_bind,
                    .conv_w = cw,
                    .conv_state_in = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                    .conv_state_out = .{ .buffer = self.scratch_buf, .offset = conv_out_off },
                    .ancestor_masks = if (is_tree) self.ancestor_masks_bind else null,
                    .q_out = q_eff_bind,
                    .k_out = k_eff_bind,
                    .v_out = v_eff_bind,
                    .m = m,
                    .gdn_qk = qk_dim,
                    .gdn_v = v_dim,
                    .conv_k = g.conv_k,
                    .is_tree = is_tree,
                    .write_state = commit_state and !is_tree,
                }, true);
                // When committing a multi-token prefill chunk (`m > 1`), advance conv state cleanly.
                if (commit_state and is_tree and g.conv_k > 1) {
                    var chain: [hybrid.MAX_TREE_NODES]u16 = undefined;
                    for (0..m) |i| chain[i] = @intCast(i);
                    try self.hybrid_pipes.encodeGdnConv1dCommit(f, .{
                        .q_raw = q_raw_bind,
                        .k_raw = k_raw_bind,
                        .v_raw = v_raw_bind,
                        .conv_state_in = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                        .conv_state_out = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                        .accepted_nodes = chain[0..m],
                        .gdn_qk = qk_dim,
                        .gdn_v = v_dim,
                        .conv_k = g.conv_k,
                    }, true);
                }
            }

            // --- Isolation #1 (GDN R6/G8): NEVER mutate `S_t` on `r < R - 1` or tree verify ---
            const committed_s = try ga.committedBinding(self.scratch_buf, g.gdn_idx);
            const scratch_s = try ga.scratchBinding(self.scratch_buf, g.gdn_idx);
            if (m == 1) {
                try self.hybrid_pipes.encodeGdnDecode(f, .{
                    .q = q_eff_bind,
                    .k = k_eff_bind,
                    .v = v_eff_bind,
                    .g = .{ .buffer = self.scratch_buf, .offset = g.tree_g_off },
                    .beta = .{ .buffer = self.scratch_buf, .offset = g.tree_beta_off },
                    .z = self.gdn_z_bind,
                    .norm_w = if (g.out_norm) |n| n.binding else null,
                    .norm_dtype = if (g.out_norm) |n| n.dtype else .f16,
                    .state_in = committed_s,
                    .state_out = scratch_s,
                    .out = self.attn_out_bind,
                    .num_k_heads = self.config.linear_num_key_heads,
                    .num_v_heads = self.config.linear_num_value_heads,
                    .head_k_dim = self.config.linear_key_head_dim,
                    .head_v_dim = self.config.linear_value_head_dim,
                    .rms_eps = self.config.rms_norm_eps,
                }, true);
                if (!is_final_pass or !commit_state) {
                    self.loop_stats.gdn_scratch_evals += 1;
                } else {
                    try self.hybrid_pipes.encodeCommitScratchState(f, ga, self.scratch_buf, g.gdn_idx, true);
                    self.loop_stats.gdn_final_commits += 1;
                }
            } else if (commit_state) {
                try self.hybrid_pipes.encodeGdnPrefill(f, .{
                    .q = q_eff_bind,
                    .k = k_eff_bind,
                    .v = v_eff_bind,
                    .g = .{ .buffer = self.scratch_buf, .offset = g.tree_g_off },
                    .beta = .{ .buffer = self.scratch_buf, .offset = g.tree_beta_off },
                    .z = self.gdn_z_bind,
                    .norm_w = if (g.out_norm) |n| n.binding else null,
                    .norm_dtype = if (g.out_norm) |n| n.dtype else .f16,
                    .state_in = committed_s,
                    .state_out = scratch_s,
                    .out = self.attn_out_bind,
                    .seq_len = m,
                    .num_k_heads = self.config.linear_num_key_heads,
                    .num_v_heads = self.config.linear_num_value_heads,
                    .head_k_dim = self.config.linear_key_head_dim,
                    .head_v_dim = self.config.linear_value_head_dim,
                    .rms_eps = self.config.rms_norm_eps,
                }, true);
                try self.hybrid_pipes.encodeCommitScratchState(f, ga, self.scratch_buf, g.gdn_idx, true);
                self.loop_stats.gdn_final_commits += 1;
            } else {
                // Tree verify or intermediate loop pass (`r < R - 1`): committed `S_t` stays untouched!
                try self.hybrid_pipes.encodeGdnTreeVerify(f, .{
                    .q = q_eff_bind,
                    .k = k_eff_bind,
                    .v = v_eff_bind,
                    .g = .{ .buffer = self.scratch_buf, .offset = g.tree_g_off },
                    .beta = .{ .buffer = self.scratch_buf, .offset = g.tree_beta_off },
                    .z = self.gdn_z_bind,
                    .norm_w = if (g.out_norm) |n| n.binding else null,
                    .norm_dtype = if (g.out_norm) |n| n.dtype else .f16,
                    .ancestor_masks = self.ancestor_masks_bind,
                    .state_in = committed_s,
                    .state_out = scratch_s,
                    .scratch_leaf_node = null,
                    .out = self.attn_out_bind,
                    .num_nodes = m,
                    .num_k_heads = self.config.linear_num_key_heads,
                    .num_v_heads = self.config.linear_num_value_heads,
                    .head_k_dim = self.config.linear_key_head_dim,
                    .head_v_dim = self.config.linear_value_head_dim,
                    .rms_eps = self.config.rms_norm_eps,
                }, true);
                self.loop_stats.gdn_scratch_evals += 1;
            }
        }

        // Output projection `o_proj` (with optional Step-LoRA).
        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .o_proj, g.o_proj, self.attn_out_bind, self.attn_out_host, self.proj_out_bind, self.proj_out_host, m, true);
    }

    fn execQsaAttention(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        qsa: *const QsaLayerWeights,
        base_pos: u32,
        m: u32,
        loop_step: u32,
        commit_kv: bool,
    ) GraphStepError!void {
        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .q_proj, qsa.q_proj, self.x_norm_bind, self.x_norm_host, self.q_bind, self.q_host, m, true);
        {
            const f = try self.ensureFrame(frame_opt);
            const tree_k: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = qsa.tree_k_off };
            const tree_v: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = qsa.tree_v_off };

            // Shared r=0 KV across recurrent passes (`LoopLM` KV sharing, ADR-0019):
            // compute `K, V` once at `r = 0` into `tree_k/tree_v` and reuse them on `r >= 1`
            // so batched `verifyTree(M > 1)` matches serial `stepDecode(M = 1)` bit-for-bit
            // and QSA `reuse_block_indices` sees identical KV keys across loops.
            if (loop_step == 0) {
                try self.encodeLinearBatched(f, qsa.k_proj, self.x_norm_bind, self.k_bind, m, false);
                try self.encodeLinearBatched(f, qsa.v_proj, self.x_norm_bind, self.v_bind, m, false);
            }

            // QK-Norm + YaRN + Interleaved 3D MRoPE: at `r = 0` write into `tree_k/tree_v`;
            // at `r >= 1` rotate `Q` in-place and route K/V output to scratch `k_bind/v_bind`
            // so `tree_k/tree_v` preserve `r = 0` K/V untouched.
            try self.hybrid_pipes.encodeQsaTreeRope(f, .{
                .q = self.q_bind,
                .k = self.k_bind,
                .v = self.v_bind,
                .q_norm = qsa.q_norm,
                .k_norm = qsa.k_norm,
                .positions_3d = self.pos3d_bind,
                .k_tree_out = if (loop_step == 0) tree_k else self.k_bind,
                .v_tree_out = if (loop_step == 0) tree_v else self.v_bind,
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .rotary_dim = self.config.head_dim,
                .mrope = self.config.mrope_section,
                .rms_eps = self.config.rms_norm_eps,
                .rope_theta = self.config.rope_theta,
                .yarn = self.config.yarn,
                .norm_dtype = qsa.qk_norm_dtype,
            }, true);

            // --- Isolation #3 (QSA G9): compute block indices at `r = 0`, reuse on `r >= 1` ---
            const reuse_idx = (loop_step >= 1);
            if (base_pos > 0) {
                if (reuse_idx) {
                    self.loop_stats.qsa_index_reuses += 1;
                } else {
                    self.loop_stats.qsa_index_computes += 1;
                }
            }
            try self.hybrid_pipes.encodeQsaAttention(f, .{
                .q = self.q_bind,
                .kv_tiles = if (base_pos > 0) qsa.kv_ring.tileTableBinding(self.scratch_buf) else null,
                .block_keys = .{ .buffer = self.scratch_buf, .offset = qsa.block_keys_off },
                .index_buf = &qsa.index_buf,
                .index_metal_buf = self.scratch_buf,
                .k_tree = tree_k,
                .v_tree = tree_v,
                .ancestor_masks = self.ancestor_masks_bind,
                .out = self.attn_out_bind,
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .kv_len = base_pos,
                .window_size = qsa.kv_ring.window_size,
                .ring_capacity = qsa.kv_ring.ring_capacity,
                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                .block_tokens = self.config.qsa_block_size,
                .top_k_blocks = self.config.qsa_top_k_blocks,
                .reuse_block_indices = reuse_idx,
            }, true);

            if (commit_kv) {
                var chain: [hybrid.MAX_TREE_NODES]u16 = undefined;
                for (0..m) |i| chain[i] = @intCast(i);
                try self.encodeQsaCommitAccepted(f, qsa, base_pos, chain[0..m], true);
            }
        }

        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .o_proj, qsa.o_proj, self.attn_out_bind, self.attn_out_host, self.proj_out_bind, self.proj_out_host, m, true);
    }

    fn execGqaAttention(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        gqa: GqaLayerWeights,
        base_pos: u32,
        m: u32,
        loop_step: u32,
        commit_kv: bool,
    ) GraphStepError!void {
        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .q_proj, gqa.q_proj, self.x_norm_bind, self.x_norm_host, self.q_bind, self.q_host, m, true);
        {
            const f = try self.ensureFrame(frame_opt);
            const tree_k: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = gqa.tree_k_off };
            const tree_v: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = gqa.tree_v_off };

            // Shared r=0 KV across recurrent passes (`LoopLM` KV sharing, ADR-0019):
            // compute `K, V` once at `r = 0` into `tree_k/tree_v` and reuse on `r >= 1`
            // so parallel `prefill` / `verifyTree(M > 1)` matches serial `stepDecode(M = 1)` bit-for-bit.
            if (loop_step == 0) {
                try self.encodeLinearBatched(f, gqa.k_proj, self.x_norm_bind, self.k_bind, m, false);
                try self.encodeLinearBatched(f, gqa.v_proj, self.x_norm_bind, self.v_bind, m, false);
            }

            try self.attn_pipes.encodeTreeQkNormRope(f, .{
                .q = self.q_bind,
                .k = self.k_bind,
                .v = self.v_bind,
                .q_norm = gqa.q_norm,
                .k_norm = gqa.k_norm,
                .positions = self.positions_bind,
                .k_tree_out = if (loop_step == 0) tree_k else self.k_bind,
                .v_tree_out = if (loop_step == 0) tree_v else self.v_bind,
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .rms_eps = self.config.rms_norm_eps,
                .rope_theta = self.config.rope_theta,
                .norm_dtype = gqa.qk_norm_dtype,
            }, true);

            try self.attn_pipes.encodeGqaTreeVerifyAttn(f, .{
                .q = self.q_bind,
                .kv_tiles = if (base_pos > 0) gqa.kv_tiles_binding else null,
                .base_pos = base_pos,
                .k_tree = tree_k,
                .v_tree = tree_v,
                .ancestor_masks = self.ancestor_masks_bind,
                .out = self.attn_out_bind,
                .scratch = self.gqa_scratch_bind,
                .max_splits = attention.MAX_SPLITS,
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .chunk_tokens = self.chunk_tokens,
            }, true);

            if (commit_kv) {
                var chain: [hybrid.MAX_TREE_NODES]u16 = undefined;
                for (0..m) |i| chain[i] = @intCast(i);
                try self.encodeGqaCommitAccepted(f, gqa, base_pos, chain[0..m], true);
            }
        }

        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .o_proj, gqa.o_proj, self.attn_out_bind, self.attn_out_host, self.proj_out_bind, self.proj_out_host, m, true);
    }

    fn execMoeFfn(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        layer: *GraphLayerWeights,
        m: u32,
        loop_step: u32,
    ) GraphStepError!void {
        const h = self.config.hidden_size;
        var layer_ws = self.moe_ws.?;
        layer_ws.schedule_header_off = self.layer_sched_hdr_off + @as(u64, @intCast(l)) * 64;
        layer_ws.schedule_header = @ptrCast(@alignCast(self.layer_sched_hdr_raw.ptr + l * 64));

        const router_src: moe.RouterSource = if (layer.router_quant_dtype) |rqd| .{
            .quant_weights = .{
                .qweight = layer.router_binding,
                .scale_bias = layer.router_scale_bias.?,
                .bits = switch (rqd) {
                    .q2_affine => .q2,
                    .q3_affine => .q3,
                    .q4_affine => .q4,
                    else => return error.InvalidConfig,
                },
            },
        } else .{
            .weights = .{
                .binding = layer.router_binding,
                .dtype = layer.router_dtype,
            },
        };

        // Residual target: for 4-branch GR, write FFN delta into `proj_out_bind` (zero residual)
        // and feed into `ami_gr_write_gate`; for single-stream, fuse `x + ffn` directly into `x_bind`.
        const moe_residual: ?metal.BufferBinding = if (self.gr_arena != null) null else self.x_bind;
        const moe_out: metal.BufferBinding = if (self.gr_arena != null) self.proj_out_bind else self.x_bind;

        // --- Isolation #4 (MoE G7, C5): cold-expert resolution on `r = 0`, pinned-only routing on `r >= 1` ---
        if (layer.has_cold_experts and self.cold_rt != null) {
            const crt = self.cold_rt.?;
            const logits_bind: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = layer_ws.router_logits_off };
            // 1. Run router projection on GPU and read `router_logits`.
            {
                const f = try self.ensureFrame(frame_opt);
                switch (router_src) {
                    .weights => |w| try self.moe_pipes.encodeRouterProj(f, w.dtype, .{
                        .x = self.x_norm_bind,
                        .router_weights = w.binding,
                        .logits = logits_bind,
                        .m = m,
                        .num_experts = self.config.num_experts,
                        .hidden_size = h,
                    }, true),
                    .quant_weights => |qw| try self.moe_pipes.encodeRouterProjQuant(f, qw.bits, .{
                        .x = self.x_norm_bind,
                        .qweight = qw.qweight,
                        .scale_bias = qw.scale_bias,
                        .logits = logits_bind,
                        .m = m,
                        .num_experts = self.config.num_experts,
                        .hidden_size = h,
                    }, true),
                    .logits => {},
                }
            }
            try self.flushFrame(frame_opt);

            const n_exp: usize = self.config.num_experts;
            const before_misses = crt.expert_cache.telemetry().sync_misses;
            if (loop_step >= 1) {
                // ADR-0019 Isolation #4: restrict `r >= 1` routing strictly to already
                // wired/resident `ExpertSlotPool` slots -> 0 cold-expert NVMe stalls!
                for (0..m) |tok| {
                    const row = layer_ws.router_logits[tok * n_exp ..][0..n_exp];
                    for (0..n_exp) |eid| {
                        if (!crt.expert_cache.isResident(@intCast(l), @intCast(eid))) {
                            row[eid] = -std.math.inf(f32);
                        }
                    }
                }
                self.loop_stats.moe_r_ge1_pinned_routes += 1;
            }

            // 2. Run top-k CSR schedule on GPU (`ami_moe_route_topk`), then pin active experts.
            {
                const f = try self.ensureFrame(frame_opt);
                try self.moe_pipes.encodeRouteTopK(f, .f32, .{
                    .logits = logits_bind,
                    .workspace = &layer_ws,
                    .workspace_buf = self.scratch_buf,
                    .m = m,
                    .num_experts = self.config.num_experts,
                    .top_k = self.config.num_experts_per_tok,
                    .norm_topk_prob = self.config.norm_topk_prob,
                }, false);
            }
            try self.flushFrame(frame_opt);

            const num_unique: usize = layer_ws.schedule_header.num_unique_experts;
            for (0..num_unique) |u| {
                const eid: u16 = @intCast(layer_ws.unique_experts[u]);
                const pin = try crt.expert_cache.acquirePinned(@intCast(l), eid);
                layer.expert_table_host[eid] = pin.weights;
            }
            errdefer crt.expert_cache.releasePins();

            if (loop_step >= 1) {
                const after_misses = crt.expert_cache.telemetry().sync_misses;
                self.loop_stats.moe_r_ge1_cold_stalls += (after_misses - before_misses);
            }

            // 3. Execute deduplicated SwiGLU FFN + optional `shared_expert` on GPU using pinned slots!
            const moe_args: moe.MoELayerArgs = .{
                .x = self.x_norm_bind,
                .router = .{ .logits = .{ .binding = logits_bind, .dtype = .f32 } },
                .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                .expert_table = layer.expert_table,
                .workspace = &layer_ws,
                .workspace_buf = self.scratch_buf,
                .residual = moe_residual,
                .out = moe_out,
                .m = m,
                .num_experts = self.config.num_experts,
                .top_k = self.config.num_experts_per_tok,
                .hidden_size = h,
                .intermediate_size = self.config.moe_intermediate_size,
                .norm_topk_prob = self.config.norm_topk_prob,
                .shared_expert = layer.shared_expert,
                .shared_weights_base = if (layer.shared_expert_buf) |sb| .{ .buffer = sb, .offset = 0 } else null,
                .shared_expert_dtype = layer.shared_expert_dtype,
                .shared_intermediate_size = self.config.shared_expert_intermediate_size,
                .shared_gate_weight = layer.shared_gate_weight,
                .shared_gate_dtype = layer.shared_gate_dtype,
            };
            {
                const f = try self.ensureFrame(frame_opt);
                try self.encodeMoEFfnOnlyDispatch(f, layer, moe_args, false);
            }
            try self.flushFrame(frame_opt);
            crt.expert_cache.releasePins();
        } else {
            // Fast all-wired MoE path: single command buffer, 0 CPU-GPU syncs!
            if (loop_step >= 1) {
                self.loop_stats.moe_r_ge1_pinned_routes += 1;
            }
            const f = try self.ensureFrame(frame_opt);
            const moe_args: moe.MoELayerArgs = .{
                .x = self.x_norm_bind,
                .router = router_src,
                .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                .expert_table = layer.expert_table,
                .workspace = &layer_ws,
                .workspace_buf = self.scratch_buf,
                .residual = moe_residual,
                .out = moe_out,
                .m = m,
                .num_experts = self.config.num_experts,
                .top_k = self.config.num_experts_per_tok,
                .hidden_size = h,
                .intermediate_size = self.config.moe_intermediate_size,
                .norm_topk_prob = self.config.norm_topk_prob,
                .shared_expert = layer.shared_expert,
                .shared_weights_base = if (layer.shared_expert_buf) |sb| .{ .buffer = sb, .offset = 0 } else null,
                .shared_expert_dtype = layer.shared_expert_dtype,
                .shared_intermediate_size = self.config.shared_expert_intermediate_size,
                .shared_gate_weight = layer.shared_gate_weight,
                .shared_gate_dtype = layer.shared_gate_dtype,
            };
            try self.encodeMoELayerDispatch(f, layer, moe_args, true);
        }

        if (self.gr_arena != null) {
            try self.applyFfnResidual(frame_opt, m, h);
        }
    }

    fn encodeMoELayerDispatch(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        layer: *const GraphLayerWeights,
        args: moe.MoELayerArgs,
        barrier_before: bool,
    ) moe.EncodeError!void {
        if (layer.is_hetero_moe) {
            return self.moe_pipes.encodeMoELayerHetero(frame, .{
                .base = args,
                .expert_schemes = layer.expert_dtypes_binding.?,
                .scheme_counts = layer.expert_scheme_counts,
            }, barrier_before);
        }
        return switch (layer.expert_weight_dtype) {
            .q2_affine => self.moe_pipes.encodeMoELayer(frame, .q2, args, barrier_before),
            .q3_affine => self.moe_pipes.encodeMoELayer(frame, .q3, args, barrier_before),
            .q4_affine => self.moe_pipes.encodeMoELayer(frame, .q4, args, barrier_before),
            .f16 => self.moe_pipes.encodeMoELayerFloat(frame, .f16, args, barrier_before),
            .bf16 => self.moe_pipes.encodeMoELayerFloat(frame, .bf16, args, barrier_before),
        };
    }

    fn encodeMoEFfnOnlyDispatch(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        layer: *const GraphLayerWeights,
        args: moe.MoELayerArgs,
        barrier_before: bool,
    ) moe.EncodeError!void {
        // Router logits & schedule header are already populated; dispatch directly.
        return self.encodeMoELayerDispatch(frame, layer, args, barrier_before);
    }

    /// Evaluates a base linear projection on GPU plus optional L2-resident SME2
    /// `Step-LoRA` adapter (`W16`, ADR-0019).
    /// When `alpha == 0.0` (`r = 0`), executes strictly on GPU with zero CPU syncs
    /// and zero writes (100% bit-exact identity).
    fn execLinearWithOptionalStepLora(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        loop_step: u32,
        proj: AdapterProjKind,
        lin: qwen3.LinearBinding,
        x_bind: metal.BufferBinding,
        x_host: []const f16,
        y_bind: metal.BufferBinding,
        y_host: []f16,
        m: u32,
        barrier_before: bool,
    ) GraphStepError!void {
        const adapter_opt: ?sme.SmeStepLoraAdapter = if (self.adapter_rt) |art|
            art.get(loop_step, @intCast(l), proj)
        else
            null;

        if (adapter_opt) |adapter| {
            if (adapter.alpha == 0.0) {
                // Mandatory ADR-0019 r = 0 bit-exact identity fast path!
                self.loop_stats.steplora_zero_alpha_skips += 1;
                const f = try self.ensureFrame(frame_opt);
                try self.encodeLinearBatched(f, lin, x_bind, y_bind, m, barrier_before);
                return;
            }

            // Active `r >= 1` Step-LoRA: flush input readiness, launch GPU base linear
            // concurrently with L2-resident P-core SME2 `SmeStepLoraAdapter.evalU16`!
            const art = self.adapter_rt.?;
            try self.flushFrame(frame_opt);

            var base_frame = try self.compute.beginFrame(self.timeout_ms);
            errdefer base_frame.abort();
            try self.encodeLinearBatched(&base_frame, lin, x_bind, y_bind, m, false);
            const sig = try base_frame.commit();

            const mn: usize = @as(usize, m) * adapter.n;
            const mk: usize = @as(usize, m) * adapter.k;
            const delta_slice = art.delta_f32[0..mn];
            @memset(delta_slice, 0.0);
            const x_u16: []const u16 = @ptrCast(x_host[0..mk]);
            try adapter.evalU16(&art.dispatcher, &art.scratch, x_u16, delta_slice, m, null, null);

            try self.compute.wait(sig, self.timeout_ms);
            for (0..mn) |i| {
                const sum: f32 = @as(f32, @floatCast(y_host[i])) + delta_slice[i];
                y_host[i] = @floatCast(sum);
            }
            self.loop_stats.steplora_active_evals += 1;
            return;
        }

        const f = try self.ensureFrame(frame_opt);
        try self.encodeLinearBatched(f, lin, x_bind, y_bind, m, barrier_before);
    }

    /// Populates `ple_f16_host` for `layer l` via `PleStore.lookupForLoopStep` (`C10`, ADR-0019 #2).
    fn preparePleLayer(
        self: *GraphModel,
        prt: *PleRuntime,
        l: usize,
        token_ids: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        m: u32,
        loop_step: u32,
    ) GraphStepError!void {
        const layer_rows = try prt.store.layerTotalRows(@intCast(l));
        const order = self.config.ple_ngram_order;
        const h: usize = self.config.hidden_size;

        // Compute deterministic N-gram row indices per node using ancestor masks + history.
        for (0..m) |node| {
            var ngram_buf: [8]u32 = undefined;
            var collected: usize = 0;
            ngram_buf[0] = token_ids[node];
            collected = 1;

            if (order > 1) {
                var anc_nodes: [8]u32 = undefined;
                var n_anc: usize = 0;
                const mask = ancestor_masks[node];
                if (node > 0) {
                    var cand: isize = @as(isize, @intCast(node)) - 1;
                    while (cand >= 0 and n_anc < order - 1) : (cand -= 1) {
                        if (((mask >> @intCast(cand)) & 1) != 0) {
                            anc_nodes[n_anc] = @intCast(cand);
                            n_anc += 1;
                        }
                    }
                }
                for (0..n_anc) |a| {
                    ngram_buf[collected] = token_ids[anc_nodes[a]];
                    collected += 1;
                }
                var past_back: u32 = 1;
                while (collected < order and base_pos >= past_back) : (past_back += 1) {
                    ngram_buf[collected] = self.token_history[base_pos - past_back];
                    collected += 1;
                }
            }
            // Reverse collected suffix into chronological order `[oldest .. newest]`.
            var chrono: [8]u32 = undefined;
            for (0..collected) |k| {
                chrono[k] = ngram_buf[collected - 1 - k];
            }
            self.ple_row_indices_host[node] = ple_mod.hashPleNgram(chrono[0..collected], order, layer_rows);
        }

        const f32_slice = prt.f32_tmp[0 .. @as(usize, m) * h];
        @memset(f32_slice, 0.0);

        const before_nvme = prt.store.stats().nvme_reads;
        try prt.store.lookupForLoopStep(
            @intCast(l),
            self.ple_row_indices_host[0..m],
            1.0,
            loop_step,
            &prt.scratch,
            f32_slice,
        );
        const after_nvme = prt.store.stats().nvme_reads;
        if (loop_step == 0) {
            self.loop_stats.ple_r0_fills += 1;
        } else {
            self.loop_stats.ple_r_ge1_reuses += 1;
            self.loop_stats.ple_r_ge1_nvme_reads += (after_nvme - before_nvme);
        }

        const layer_stride = @as(usize, self.max_tree_nodes) * h;
        const dst_f16 = self.ple_f16_host[l * layer_stride ..][0 .. @as(usize, m) * h];
        for (0..@as(usize, m) * h) |i| {
            dst_f16[i] = @floatCast(f32_slice[i]);
        }
    }

    fn pleLayerBinding(self: *const GraphModel, l: usize) metal.BufferBinding {
        const layer_bytes = @as(u64, self.max_tree_nodes) * self.config.hidden_size * @sizeOf(f16);
        return .{
            .buffer = self.scratch_buf,
            .offset = self.ple_f16_base_off + @as(u64, @intCast(l)) * layer_bytes,
        };
    }

    fn shouldEarlyExitCpu(self: *const GraphModel, m: u32) bool {
        const w_bytes = self.early_exit_gate_host orelse return false;
        const h: usize = self.config.hidden_size;
        for (0..m) |tok| {
            const x_row = self.x_host[tok * h ..][0..h];
            var dot: f32 = 0.0;
            for (0..h) |d| {
                const wv = readFloatWeightAt(w_bytes, self.early_exit_gate_dtype, d);
                dot += @as(f32, @floatCast(x_row[d])) * wv;
            }
            const g: f32 = 1.0 / (1.0 + @exp(-dot));
            if (g < self.early_exit_threshold) return false;
        }
        return true;
    }

    fn commitCoreLayersAfterEarlyExit(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        core_start: u32,
        core_end: u32,
        base_pos: u32,
        m: u32,
    ) GraphStepError!void {
        var chain: [hybrid.MAX_TREE_NODES]u16 = undefined;
        for (0..m) |i| chain[i] = @intCast(i);
        const f = try self.ensureFrame(frame_opt);
        for (core_start..core_end) |l| {
            switch (self.layers[l].attn) {
                .gdn => |g| {
                    const ga = &(self.gdn_arena.?);
                    try self.hybrid_pipes.encodeCommitScratchState(f, ga, self.scratch_buf, g.gdn_idx, true);
                    self.loop_stats.gdn_final_commits += 1;
                    if (g.conv1d_w != null and g.conv_k > 1) {
                        try self.hybrid_pipes.encodeGdnConv1dCommit(f, .{
                            .q_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_q_raw_off },
                            .k_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_k_raw_off },
                            .v_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_v_raw_off },
                            .conv_state_in = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                            .conv_state_out = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                            .accepted_nodes = chain[0..m],
                            .gdn_qk = self.config.gdnQkDim(),
                            .gdn_v = self.config.gdnVDim(),
                            .conv_k = g.conv_k,
                        }, true);
                    }
                },
                .qsa => |*qsa| {
                    try self.encodeQsaCommitAccepted(f, qsa, base_pos, chain[0..m], true);
                },
                .gqa => |gqa| {
                    try self.encodeGqaCommitAccepted(f, gqa, base_pos, chain[0..m], true);
                },
            }
        }
    }

    fn encodeQsaCommitAccepted(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        qsa: *const QsaLayerWeights,
        base_pos: u32,
        accepted_nodes: []const u16,
        barrier_before: bool,
    ) GraphStepError!void {
        const tree_k: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = qsa.tree_k_off };
        const tree_v: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = qsa.tree_v_off };
        var offset: usize = 0;
        var first = barrier_before;
        while (offset < accepted_nodes.len) {
            const cur_base = base_pos + @as(u32, @intCast(offset));
            const max_span = @min(accepted_nodes.len - offset, @as(usize, qsa.kv_ring.chunk_tokens));
            const take: u32 = @intCast(max_span);
            const span = try qsa.kv_ring.commitTilesForSpan(cur_base, take);
            try self.hybrid_pipes.encodeQsaKvTreeCommit(frame, .{
                .k_tree = tree_k,
                .v_tree = tree_v,
                .accepted_nodes = .{ .inline_u16 = accepted_nodes[offset .. offset + take] },
                .accepted_count = take,
                .k_tile0 = span.k_tile0,
                .v_tile0 = span.v_tile0,
                .k_tile1 = span.k_tile1,
                .v_tile1 = span.v_tile1,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .base_pos = cur_base,
                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                .ring_capacity = qsa.kv_ring.ring_capacity,
            }, first);
            first = true;
            offset += take;
        }

        // Refresh touched micro-block key summaries for `ami_qsa_index_blocks`.
        const new_kv_len: u32 = base_pos + @as(u32, @intCast(accepted_nodes.len));
        const blk_tok = self.config.qsa_block_size;
        const start_block: u32 = if (qsa.kv_ring.window_size > 0) 0 else (base_pos / blk_tok);
        const total_blocks: u32 = if (qsa.kv_ring.window_size > 0)
            (@min(new_kv_len, qsa.kv_ring.window_size) + blk_tok - 1) / blk_tok
        else
            (new_kv_len + blk_tok - 1) / blk_tok;
        const num_blocks: u32 = if (total_blocks > start_block) (total_blocks - start_block) else 1;
        try self.hybrid_pipes.encodeQsaUpdateBlockSummary(frame, .{
            .kv_tiles = qsa.kv_ring.tileTableBinding(self.scratch_buf),
            .block_keys = .{ .buffer = self.scratch_buf, .offset = qsa.block_keys_off },
            .num_kv_heads = self.config.num_key_value_heads,
            .head_dim = self.config.head_dim,
            .kv_len = new_kv_len,
            .window_size = qsa.kv_ring.window_size,
            .ring_capacity = qsa.kv_ring.ring_capacity,
            .chunk_tokens = qsa.kv_ring.chunk_tokens,
            .block_tokens = blk_tok,
            .start_block = start_block,
            .num_blocks = num_blocks,
        }, true);
    }

    fn encodeGqaCommitAccepted(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        gqa: GqaLayerWeights,
        base_pos: u32,
        accepted_nodes: []const u16,
        barrier_before: bool,
    ) GraphStepError!void {
        const tree_k: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = gqa.tree_k_off };
        const tree_v: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = gqa.tree_v_off };
        var offset: usize = 0;
        var first = barrier_before;
        while (offset < accepted_nodes.len) {
            const cur_base = base_pos + @as(u32, @intCast(offset));
            const chunk0: u32 = cur_base / self.chunk_tokens;
            const max_end: u32 = (chunk0 + 2) * self.chunk_tokens;
            const take: u32 = @intCast(@min(accepted_nodes.len - offset, @as(usize, max_end - cur_base)));
            const crosses = ((cur_base + take - 1) / self.chunk_tokens) > chunk0;
            const t0 = self.gqaTilePair(gqa, chunk0);
            const t1 = if (crosses) self.gqaTilePair(gqa, chunk0 + 1) else null;
            try self.attn_pipes.encodeKvTreeCommit(frame, .{
                .k_tree = tree_k,
                .v_tree = tree_v,
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
            }, first);
            first = true;
            offset += take;
        }
    }

    fn gqaTilePair(self: *const GraphModel, gqa: GqaLayerWeights, chunk: u32) struct { k: metal.KvTile, v: metal.KvTile } {
        const idx = gqa.kv_tile_start + @as(usize, chunk) * 2;
        return .{ .k = self.gqa_kv_tiles[idx], .v = self.gqa_kv_tiles[idx + 1] };
    }

    fn commitFeatureTaps(self: *GraphModel, accepted_nodes: []const u16) GraphStepError!void {
        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();
        try self.encodeCommitFeatureTaps(&frame, accepted_nodes, false);
        const sig = try frame.commit();
        try self.compute.wait(sig, self.timeout_ms);
    }

    fn encodeCommitFeatureTaps(
        self: *GraphModel,
        frame: *metal.Compute.Frame,
        accepted_nodes: []const u16,
        barrier_before: bool,
    ) GraphStepError!void {
        const h = self.config.hidden_size;
        const count: u32 = @intCast(accepted_nodes.len);
        for (accepted_nodes, 0..) |nid, i| {
            self.token_ids_host[i] = nid;
        }
        try self.moe_pipes.encodeGatherRows(frame, .{
            .src = self.x_bind,
            .indices = self.token_ids_bind,
            .dst = self.accepted_top_bind,
            .count = count,
            .row_elems = h,
        }, barrier_before);
        try self.moe_pipes.encodeGatherRows(frame, .{
            .src = .{ .buffer = self.scratch_buf, .offset = self.aux_tree_base_off },
            .indices = self.token_ids_bind,
            .dst = self.accepted_aux_bind,
            .count = count,
            .row_elems = 3 * h,
        }, false);
    }

    fn ensureFrame(self: *GraphModel, frame_opt: *?metal.Compute.Frame) metal.Error!*metal.Compute.Frame {
        if (frame_opt.* == null) {
            frame_opt.* = try self.compute.beginFrame(self.timeout_ms);
        }
        return &(frame_opt.*.?);
    }

    fn flushFrame(self: *GraphModel, frame_opt: *?metal.Compute.Frame) metal.Error!void {
        if (frame_opt.*) |*f| {
            const sig = try f.commit();
            frame_opt.* = null;
            try self.compute.wait(sig, self.timeout_ms);
        }
    }

    // -----------------------------------------------------------------------
    // Internal Loader
    // -----------------------------------------------------------------------

    fn loadInternal(
        allocator: std.mem.Allocator,
        source: AmfSource,
        config: GraphConfig,
        options: GraphLoadOptions,
    ) GraphLoadError!GraphModel {
        try config.validate();
        if (options.max_seq_len == 0 or options.chunk_tokens == 0) return error.InvalidConfig;
        if (options.max_tree_nodes == 0 or options.max_tree_nodes > hybrid.MAX_TREE_NODES) return error.InvalidConfig;

        const effective_max_loops: u32 = options.max_loops_override orelse config.axes.loop_schedule.max_loops;
        if (effective_max_loops == 0) return error.InvalidConfig;

        const device = try metal.Device.open();
        errdefer device.close();
        const dev_info = try device.info();
        const library = try metal.Library.openEmbedded(device);
        errdefer library.close();
        const compute = try metal.Compute.init(device);
        errdefer compute.close();
        const dequant_pipes = try dequant.DequantPipelines.init(device, library);
        errdefer dequant_pipes.close();
        const attn_pipes = try attention.AttentionPipelines.init(device, library);
        errdefer attn_pipes.close();
        const moe_pipes = try moe.MoePipelines.init(device, library);
        errdefer moe_pipes.close();
        const hybrid_pipes = try hybrid.HybridPipelines.init(device, library);
        errdefer hybrid_pipes.close();

        const facts = try sysinfo.Facts.read();
        const wired_budget = options.wired_budget_override orelse budget_mod.Budget.compute(.{
            .memory_bytes = facts.memory_bytes,
            .user_wire_limit = facts.user_wire_limit,
            .global_user_wire_limit = facts.global_user_wire_limit,
            .gpu_working_set_bytes = dev_info.recommendedMaxWorkingSetSize(),
            .configured_cap = null,
            .headroom_bytes = budget_mod.DEFAULT_HEADROOM_BYTES,
        });

        // --- 1. Partition wired tensors into <= MAX_SHARD_BYTES shards (skipping cold/sparse) ---
        const entries = source.entries();
        var shard_sizes: std.ArrayListUnmanaged(usize) = .empty;
        defer shard_sizes.deinit(allocator);
        var entry_shard = try allocator.alloc(u16, entries.len);
        defer allocator.free(entry_shard);
        @memset(entry_shard, std.math.maxInt(u16));

        var has_any_cold = false;
        var cold_entry_count: usize = 0;
        var max_cold_triple_bytes: usize = 0;
        var max_cold_gate_bytes: usize = 0;
        var max_cold_up_bytes: usize = 0;
        var max_cold_down_bytes: usize = 0;
        var inline_loop_adapters: usize = 0;

        // First scan: compute total bytes and identify cold experts & inline loop adapters.
        var cur_shard: usize = 0;
        try shard_sizes.append(allocator, 0);
        for (entries, 0..) |*e, idx| {
            const kind = try e.tensorKind();
            if (kind == .cold_expert) {
                has_any_cold = true;
                cold_entry_count += 1;
                const aligned = try page.alignUp(@intCast(e.blob_bytes));
                const nm = e.nameSlice();
                if (std.mem.endsWith(u8, nm, ".gate_proj")) {
                    max_cold_gate_bytes = @max(max_cold_gate_bytes, aligned);
                } else if (std.mem.endsWith(u8, nm, ".up_proj")) {
                    max_cold_up_bytes = @max(max_cold_up_bytes, aligned);
                } else if (std.mem.endsWith(u8, nm, ".down_proj")) {
                    max_cold_down_bytes = @max(max_cold_down_bytes, aligned);
                }
                continue;
            }
            if (kind == .sparse_table) continue;
            if (kind == .loop_adapter) inline_loop_adapters += 1;

            const aligned = try page.alignUp(@intCast(e.blob_bytes));
            if (shard_sizes.items[cur_shard] > 0 and shard_sizes.items[cur_shard] + aligned > MAX_SHARD_BYTES) {
                cur_shard += 1;
                if (cur_shard >= 64) return error.TooManyWeightShards;
                try shard_sizes.append(allocator, 0);
            }
            entry_shard[idx] = @intCast(cur_shard);
            shard_sizes.items[cur_shard] += aligned;
        }
        max_cold_triple_bytes = max_cold_gate_bytes + max_cold_up_bytes + max_cold_down_bytes;

        // Ensure shard 0 has room for at least 1 page even for empty edge cases.
        if (shard_sizes.items[0] == 0) shard_sizes.items[0] = page.APPLE_PAGE_SIZE;

        // If cold experts exist, reserve `ExpertSlotPool` space at the end of shard 0 so
        // hot experts and pinned cold experts share shard 0's `MTLBuffer` zero-copy!
        var cold_slot_count: u32 = 0;
        var cold_pool_spec: cache_mod.PoolSpec = undefined;
        var cold_pool_bytes: usize = 0;
        if (has_any_cold) {
            if (source == .bytes) return error.InvalidConfig; // Cold experts require file fd
            cold_slot_count = if (options.cold_slot_count > 0)
                options.cold_slot_count
            else
                @max(config.num_experts_per_tok * 2, 32);
            cold_pool_spec = .{
                .num_slots = cold_slot_count,
                .gate_blob_bytes = max_cold_gate_bytes,
                .up_blob_bytes = max_cold_up_bytes,
                .down_blob_bytes = max_cold_down_bytes,
            };
            cold_pool_bytes = try cold_pool_spec.totalBytes();
            shard_sizes.items[0] += cold_pool_bytes;
        }

        // Allocate wired weight shards & load every wired tensor once.
        const weight_shards = try allocator.alloc(WeightShard, shard_sizes.items.len);
        var shards_built: usize = 0;
        errdefer {
            for (0..shards_built) |si| {
                compute.removeResident(&.{weight_shards[si].buffer}) catch |err| {
                    std.debug.panic("removeResident failed during GraphModel errdefer: {}", .{err});
                };
                weight_shards[si].buffer.close();
                weight_shards[si].arena.deinit();
            }
            allocator.free(weight_shards);
        }

        var total_wired_bytes: u64 = 0;
        for (shard_sizes.items, 0..) |sz, si| {
            const aligned_sz = try page.alignUp(sz);
            try wired_budget.checkPlan(total_wired_bytes + aligned_sz);
            var w_arena = try arena_mod.WiredArena.init(aligned_sz, wired_budget);
            errdefer w_arena.deinit();
            const w_buf = try metal.Buffer.wrap(device, w_arena.bytes());
            errdefer w_buf.close();
            try compute.makeResident(&.{w_buf});
            weight_shards[si] = .{ .arena = w_arena, .buffer = w_buf };
            shards_built += 1;
            total_wired_bytes += aligned_sz;
        }

        const entry_offsets = try allocator.alloc(u64, entries.len);
        defer allocator.free(entry_offsets);
        const entry_slices = try allocator.alloc([]u8, entries.len);
        defer allocator.free(entry_slices);
        const entry_page_slices = try allocator.alloc([]align(page.APPLE_PAGE_SIZE) u8, entries.len);
        defer allocator.free(entry_page_slices);
        @memset(entry_offsets, 0);
        @memset(entry_slices, &.{});
        @memset(entry_page_slices, &.{});

        for (entries, 0..) |*e, idx| {
            const si = entry_shard[idx];
            if (si == std.math.maxInt(u16)) continue;
            const len: usize = @intCast(e.blob_bytes);
            const raw = try source.loadIntoArena(e, &weight_shards[si].arena);
            entry_offsets[idx] = weight_shards[si].arena.offsetOf(raw.ptr);
            entry_slices[idx] = raw[0..len];
            entry_page_slices[idx] = raw;
        }

        // Build lookup map: entry pointer -> index.
        const entry_base_ptr = @intFromPtr(entries.ptr);
        const entryIndexOf = struct {
            fn call(base: usize, e: *const amf.TensorEntry) usize {
                return (@intFromPtr(e) - base) / @sizeOf(amf.TensorEntry);
            }
        }.call;

        const bindLinear = struct {
            fn call(
                src: AmfSource,
                shards: []const WeightShard,
                e_shard: []const u16,
                e_off: []const u64,
                e_slice: []const []u8,
                base_ptr: usize,
                name: []const u8,
                expected_n: u32,
                expected_k: u32,
            ) GraphLoadError!qwen3.LinearBinding {
                const e = src.find(name) orelse blk: {
                    var alt_buf: [96]u8 = undefined;
                    const alt = std.fmt.bufPrint(&alt_buf, "{s}.weight", .{name}) catch return error.MissingTensor;
                    break :blk src.find(alt) orelse return error.MissingTensor;
                };
                const idx = entryIndexOf(base_ptr, e);
                const si = e_shard[idx];
                if (si == std.math.maxInt(u16)) return error.MissingTensor;
                if (e.shape[0] != expected_n or e.shape[1] != expected_k) return error.TensorShapeMismatch;
                const dt = try e.dataType();
                const wdt = amfToWeightDType(dt) orelse return error.UnsupportedDType;
                const buf = shards[si].buffer;
                const off = e_off[idx];
                const raw = e_slice[idx];
                const sb_bind: ?metal.BufferBinding = if (wdt.isQuantized())
                    .{ .buffer = buf, .offset = off + e.params_offset }
                else
                    null;
                const alg_b: u64 = e.blob_bytes;
                return .{
                    .weight = .{ .buffer = buf, .offset = off },
                    .scale_bias = sb_bind,
                    .n = expected_n,
                    .k = expected_k,
                    .dtype = wdt,
                    .alg_bytes = alg_b,
                    .layout = try e.tensorLayout(),
                    .param_kind = try e.quantParamKind(),
                    .weight_host = raw,
                };
            }
        }.call;

        const bindNorm = struct {
            fn call(
                src: AmfSource,
                shards: []const WeightShard,
                e_shard: []const u16,
                e_off: []const u64,
                e_slice: []const []u8,
                base_ptr: usize,
                name: []const u8,
                expected_len: u32,
            ) GraphLoadError!qwen3.NormBinding {
                const e = src.find(name) orelse blk: {
                    var alt_buf: [96]u8 = undefined;
                    const alt = std.fmt.bufPrint(&alt_buf, "{s}.weight", .{name}) catch return error.MissingTensor;
                    break :blk src.find(alt) orelse return error.MissingTensor;
                };
                const idx = entryIndexOf(base_ptr, e);
                const si = e_shard[idx];
                if (si == std.math.maxInt(u16)) return error.MissingTensor;
                if (e.shape[0] != expected_len) return error.TensorShapeMismatch;
                const dt = try e.dataType();
                const fdt: quant.FloatDType = switch (dt) {
                    .f16 => .f16,
                    .bf16 => .bf16,
                    else => return error.UnsupportedDType,
                };
                return .{
                    .binding = .{ .buffer = shards[si].buffer, .offset = e_off[idx] },
                    .dtype = fdt,
                    .host_bytes = e_slice[idx],
                };
            }
        }.call;

        // --- 2. Global embeddings, final norm, lm_head, optional early-exit gate ---
        const h = config.hidden_size;
        const embed = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, "embed_tokens", config.vocab_size, h);
        const final_norm = try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, "norm.weight", h);
        const lm_head = if (config.tie_word_embeddings)
            embed
        else
            try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, "lm_head", config.vocab_size, h);

        var early_exit_gate_host: ?[]const u8 = null;
        var early_exit_gate_dtype: quant.FloatDType = .bf16;
        if (source.find("loop.early_exit_gate.weight")) |eg_e| {
            const idx = entryIndexOf(entry_base_ptr, eg_e);
            early_exit_gate_host = entry_slices[idx];
            early_exit_gate_dtype = switch (try eg_e.dataType()) {
                .f16 => .f16,
                .bf16 => .bf16,
                else => return error.UnsupportedDType,
            };
        }

        // --- 3. Compute exact scratch arena size ---
        const max_m = options.max_tree_nodes;
        var num_gdn_layers: u32 = 0;
        var num_qsa_layers: u32 = 0;
        var num_gqa_layers: u32 = 0;
        for (config.axes.layer_attention_kinds) |ak| {
            switch (ak) {
                .gated_deltanet => num_gdn_layers += 1,
                .qsa_sparse => num_qsa_layers += 1,
                .gqa_full => num_gqa_layers += 1,
                .mla_csa2 => return error.UnsupportedOperatorAxis,
            }
        }
        const num_chunks: u32 = (options.max_seq_len + options.chunk_tokens - 1) / options.chunk_tokens;
        const qsa_ring_mode: hybrid.HybridKvRingMode = if (config.qsa_window_size > 0 and config.qsa_window_size <= hybrid.QSA_SWA_WINDOW_TOKENS)
            .sliding_window_128
        else
            .full_chunk_512;
        const qsa_num_tiles: u32 = switch (qsa_ring_mode) {
            .sliding_window_128 => 1,
            .full_chunk_512 => (options.max_seq_len + hybrid.QSA_FULL_CHUNK_TOKENS - 1) / hybrid.QSA_FULL_CHUNK_TOKENS,
        };
        const qsa_max_blocks: u32 = switch (qsa_ring_mode) {
            .sliding_window_128 => (hybrid.QSA_SWA_WINDOW_TOKENS + config.qsa_block_size - 1) / config.qsa_block_size,
            .full_chunk_512 => (options.max_seq_len + config.qsa_block_size - 1) / config.qsa_block_size,
        };

        var scratch_bytes: usize = 0;
        const addAligned = struct {
            fn call(total: *usize, n_bytes: usize) void {
                total.* += amf.align64(n_bytes) + 64;
            }
        }.call;

        if (num_gdn_layers > 0) {
            scratch_bytes += try hybrid.GdnStateArena.requiredBytes(.{
                .num_gdn_layers = num_gdn_layers,
                .num_v_heads = config.linear_num_value_heads,
                .head_k_dim = config.linear_key_head_dim,
                .head_v_dim = config.linear_value_head_dim,
            });
            const qk_d = config.gdnQkDim();
            const v_d = config.gdnVDim();
            const conv_d = 2 * qk_d + v_d;
            const conv_hist = if (config.linear_conv_kernel_dim > 1) config.linear_conv_kernel_dim - 1 else 1;
            for (0..num_gdn_layers) |_| {
                addAligned(&scratch_bytes, @as(usize, max_m) * qk_d * 2); // tree_k
                addAligned(&scratch_bytes, @as(usize, max_m) * v_d * 2); // tree_v
                addAligned(&scratch_bytes, @as(usize, max_m) * config.linear_num_value_heads * 4); // tree_g
                addAligned(&scratch_bytes, @as(usize, max_m) * config.linear_num_value_heads * 4); // tree_beta
                addAligned(&scratch_bytes, @as(usize, max_m) * qk_d * 2); // tree_q_raw
                addAligned(&scratch_bytes, @as(usize, max_m) * qk_d * 2); // tree_k_raw
                addAligned(&scratch_bytes, @as(usize, max_m) * v_d * 2); // tree_v_raw
                addAligned(&scratch_bytes, @as(usize, conv_hist) * conv_d * 2); // conv_state
                addAligned(&scratch_bytes, @as(usize, conv_hist) * conv_d * 2); // conv_state_tmp
            }
        }
        if (config.axes.residual_kind == .gated_4branch) {
            scratch_bytes += try hybrid.GrStreamArena.requiredBytes(.{
                .max_tokens = max_m,
                .hidden_dim = h,
            });
            addAligned(&scratch_bytes, @as(usize, max_m) * 4 * 4); // gr_g_read
            addAligned(&scratch_bytes, @as(usize, max_m) * 4 * 4); // gr_g_write
            addAligned(&scratch_bytes, @as(usize, max_m) * 4 * 4); // gr_decay
        }
        if (num_qsa_layers > 0) {
            for (0..num_qsa_layers) |_| {
                scratch_bytes += try hybrid.HybridKvRing.tableRequiredBytes(qsa_num_tiles) + 64;
                scratch_bytes += try hybrid.QsaIndexBuffer.requiredBytes(.{
                    .max_nodes = max_m,
                    .num_kv_heads = config.num_key_value_heads,
                    .top_k_blocks = config.qsa_top_k_blocks,
                });
                addAligned(&scratch_bytes, @as(usize, config.num_key_value_heads) * qsa_max_blocks * config.head_dim * 2); // block_keys
                addAligned(&scratch_bytes, @as(usize, max_m) * config.kvDim() * 2); // tree_k
                addAligned(&scratch_bytes, @as(usize, max_m) * config.kvDim() * 2); // tree_v
            }
        }
        if (num_gqa_layers > 0) {
            for (0..num_gqa_layers) |_| {
                addAligned(&scratch_bytes, @as(usize, num_chunks) * @sizeOf(attention.KvTilePair));
                addAligned(&scratch_bytes, @as(usize, max_m) * config.kvDim() * 2);
                addAligned(&scratch_bytes, @as(usize, max_m) * config.kvDim() * 2);
            }
            addAligned(&scratch_bytes, @as(usize, max_m) * config.num_attention_heads * attention.MAX_SPLITS * (config.head_dim + 2) * 4);
        }

        const max_q_dim: u32 = @max(@max(config.qDim(), config.gdnQkDim()), h);
        const max_kv_dim: u32 = @max(@max(config.kvDim(), config.gdnQkDim()), @max(config.gdnVDim(), h));
        const max_attn_out: u32 = @max(@max(config.qDim(), config.gdnVDim()), h);
        const max_inter: u32 = @max(@max(config.intermediate_size, config.moe_intermediate_size), @max(config.shared_expert_intermediate_size, h));

        addAligned(&scratch_bytes, @as(usize, max_m) * 4); // token_ids
        addAligned(&scratch_bytes, @as(usize, max_m) * 4); // positions
        addAligned(&scratch_bytes, @as(usize, max_m) * @sizeOf(hybrid.Pos3D)); // pos3d
        addAligned(&scratch_bytes, @as(usize, max_m) * 8); // ancestor_masks
        addAligned(&scratch_bytes, @as(usize, max_m) * 4); // out_token_ids
        addAligned(&scratch_bytes, @as(usize, max_m) * config.vocab_size * 2); // logits
        addAligned(&scratch_bytes, @as(usize, max_m) * moe.MAX_TOP_K * @sizeOf(moe.TopKCandidate)); // topk
        addAligned(&scratch_bytes, @as(usize, config.num_hidden_layers) * max_m * h * 2); // ple_f16
        addAligned(&scratch_bytes, @as(usize, max_m) * h * 2); // x
        addAligned(&scratch_bytes, @as(usize, max_m) * h * 2); // x_init
        addAligned(&scratch_bytes, @as(usize, max_m) * h * 2); // x_norm
        addAligned(&scratch_bytes, @as(usize, max_m) * max_q_dim * 2); // q
        addAligned(&scratch_bytes, @as(usize, max_m) * max_kv_dim * 2); // k
        addAligned(&scratch_bytes, @as(usize, max_m) * max_kv_dim * 2); // v
        addAligned(&scratch_bytes, @as(usize, max_m) * max_q_dim * 2); // q_conv
        addAligned(&scratch_bytes, @as(usize, max_m) * max_kv_dim * 2); // k_conv
        addAligned(&scratch_bytes, @as(usize, max_m) * max_kv_dim * 2); // v_conv
        addAligned(&scratch_bytes, @as(usize, max_m) * max_attn_out * 2); // gdn_z
        addAligned(&scratch_bytes, @as(usize, max_m) * @max(@as(u32, 1), config.linear_num_value_heads) * 4); // gdn_g
        addAligned(&scratch_bytes, @as(usize, max_m) * @max(@as(u32, 1), config.linear_num_value_heads) * 4); // gdn_beta
        addAligned(&scratch_bytes, @as(usize, max_m) * max_attn_out * 2); // attn_out
        addAligned(&scratch_bytes, @as(usize, max_m) * h * 2); // proj_out
        addAligned(&scratch_bytes, @as(usize, max_m) * max_inter * 2); // dense_gate
        addAligned(&scratch_bytes, @as(usize, max_m) * max_inter * 2); // dense_up
        addAligned(&scratch_bytes, @as(usize, max_m) * max_inter * 2); // dense_act
        addAligned(&scratch_bytes, @as(usize, max_m) * h * 2); // accepted_top
        addAligned(&scratch_bytes, @as(usize, max_m) * 3 * h * 2); // accepted_aux
        addAligned(&scratch_bytes, @as(usize, max_m) * 3 * h * 2); // aux_tree

        const moe_ws_spec: moe.WorkspaceSpec = .{
            .max_tokens = max_m,
            .num_experts = if (config.axes.ffn_kind == .moe_swiglu) config.num_experts else 1,
            .top_k = if (config.axes.ffn_kind == .moe_swiglu) config.num_experts_per_tok else 1,
            .hidden_size = h,
            .intermediate_size = if (config.axes.ffn_kind == .moe_swiglu) config.moe_intermediate_size else h,
            .shared_intermediate_size = config.shared_expert_intermediate_size,
        };
        if (config.axes.ffn_kind == .moe_swiglu) {
            const ws_bytes = try moe.MoeWorkspace.requiredBytes(moe_ws_spec);
            addAligned(&scratch_bytes, ws_bytes + 4096);
            addAligned(&scratch_bytes, @as(usize, config.num_hidden_layers) * 64); // per-layer sched headers
            for (0..config.num_hidden_layers) |_| {
                addAligned(&scratch_bytes, @as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights));
                addAligned(&scratch_bytes, @as(usize, config.num_experts)); // hetero expert_dtypes
            }
        }

        // Generous page margin for sub-arena page rounding.
        scratch_bytes = try page.alignUp(scratch_bytes + 32 * page.APPLE_PAGE_SIZE);
        try wired_budget.checkPlan(total_wired_bytes + scratch_bytes);

        var scratch_arena = try arena_mod.WiredArena.init(scratch_bytes, wired_budget);
        errdefer scratch_arena.deinit();
        const scratch_buf = try metal.Buffer.wrap(device, scratch_arena.bytes());
        errdefer scratch_buf.close();
        try compute.makeResident(&.{scratch_buf});
        total_wired_bytes += scratch_bytes;

        // Allocate sub-arenas inside `scratch_arena`.
        const gdn_arena: ?hybrid.GdnStateArena = if (num_gdn_layers > 0)
            try hybrid.GdnStateArena.init(&scratch_arena, .{
                .num_gdn_layers = num_gdn_layers,
                .num_v_heads = config.linear_num_value_heads,
                .head_k_dim = config.linear_key_head_dim,
                .head_v_dim = config.linear_value_head_dim,
            })
        else
            null;

        const gr_arena: ?hybrid.GrStreamArena = if (config.axes.residual_kind == .gated_4branch)
            try hybrid.GrStreamArena.init(&scratch_arena, .{
                .max_tokens = max_m,
                .hidden_dim = h,
            })
        else
            null;

        const gr_g_read_raw = try scratch_arena.allocAligned(@as(usize, max_m) * 4 * 4, 64);
        const gr_g_write_raw = try scratch_arena.allocAligned(@as(usize, max_m) * 4 * 4, 64);
        const gr_decay_raw = try scratch_arena.allocAligned(@as(usize, max_m) * 4 * 4, 64);

        const tok_raw = try scratch_arena.allocAligned(@as(usize, max_m) * 4, 64);
        const pos_raw = try scratch_arena.allocAligned(@as(usize, max_m) * 4, 64);
        const pos3d_raw = try scratch_arena.allocAligned(@as(usize, max_m) * @sizeOf(hybrid.Pos3D), 64);
        const msk_raw = try scratch_arena.allocAligned(@as(usize, max_m) * 8, 64);
        const out_tok_raw = try scratch_arena.allocAligned(@as(usize, max_m) * 4, 64);
        const logits_raw = try scratch_arena.allocAligned(@as(usize, max_m) * config.vocab_size * 2, 64);
        const topk_raw = try scratch_arena.allocAligned(@as(usize, max_m) * moe.MAX_TOP_K * @sizeOf(moe.TopKCandidate), 64);
        const ple_f16_raw = try scratch_arena.allocAligned(@as(usize, config.num_hidden_layers) * max_m * h * 2, 64);
        @memset(ple_f16_raw, 0);
        const x_raw = try scratch_arena.allocAligned(@as(usize, max_m) * h * 2, 64);
        const x_init_raw = try scratch_arena.allocAligned(@as(usize, max_m) * h * 2, 64);
        const x_norm_raw = try scratch_arena.allocAligned(@as(usize, max_m) * h * 2, 64);
        const q_raw = try scratch_arena.allocAligned(@as(usize, max_m) * max_q_dim * 2, 64);
        const k_raw = try scratch_arena.allocAligned(@as(usize, max_m) * max_kv_dim * 2, 64);
        const v_raw = try scratch_arena.allocAligned(@as(usize, max_m) * max_kv_dim * 2, 64);
        const q_conv_raw = try scratch_arena.allocAligned(@as(usize, max_m) * max_q_dim * 2, 64);
        const k_conv_raw = try scratch_arena.allocAligned(@as(usize, max_m) * max_kv_dim * 2, 64);
        const v_conv_raw = try scratch_arena.allocAligned(@as(usize, max_m) * max_kv_dim * 2, 64);
        const gdn_z_raw = try scratch_arena.allocAligned(@as(usize, max_m) * max_attn_out * 2, 64);
        const gdn_g_raw = try scratch_arena.allocAligned(@as(usize, max_m) * @max(@as(u32, 1), config.linear_num_value_heads) * 4, 64);
        const gdn_beta_raw = try scratch_arena.allocAligned(@as(usize, max_m) * @max(@as(u32, 1), config.linear_num_value_heads) * 4, 64);
        const attn_out_raw = try scratch_arena.allocAligned(@as(usize, max_m) * max_attn_out * 2, 64);
        const proj_out_raw = try scratch_arena.allocAligned(@as(usize, max_m) * h * 2, 64);
        const dgate_raw = try scratch_arena.allocAligned(@as(usize, max_m) * max_inter * 2, 64);
        const dup_raw = try scratch_arena.allocAligned(@as(usize, max_m) * max_inter * 2, 64);
        const dact_raw = try scratch_arena.allocAligned(@as(usize, max_m) * max_inter * 2, 64);
        const acc_top_raw = try scratch_arena.allocAligned(@as(usize, max_m) * h * 2, 64);
        const acc_aux_raw = try scratch_arena.allocAligned(@as(usize, max_m) * 3 * h * 2, 64);
        const aux_tree_raw = try scratch_arena.allocAligned(@as(usize, max_m) * 3 * h * 2, 64);
        const gqa_scratch_raw = if (num_gqa_layers > 0)
            try scratch_arena.allocAligned(@as(usize, max_m) * config.num_attention_heads * attention.MAX_SPLITS * (config.head_dim + 2) * 4, 64)
        else
            try scratch_arena.allocAligned(64, 64);

        const moe_ws: ?moe.MoeWorkspace = if (config.axes.ffn_kind == .moe_swiglu)
            try moe.MoeWorkspace.init(&scratch_arena, moe_ws_spec)
        else
            null;
        const layer_sched_hdr_raw = try scratch_arena.allocAligned(@as(usize, config.num_hidden_layers) * 64, 64);
        @memset(layer_sched_hdr_raw, 0);

        // --- 4. Initialize optional ColdRuntime (`C3`-`C5`) ---
        var cold_rt: ?*ColdRuntime = null;
        errdefer if (cold_rt) |crt| {
            crt.expert_cache.deinit(allocator);
            crt.reader.deinit(allocator);
            crt.pool.deinit();
            if (crt.amf_file) |*f| f.close(allocator);
            allocator.destroy(crt);
        };
        if (has_any_cold) {
            const crt = try allocator.create(ColdRuntime);
            errdefer allocator.destroy(crt);
            var pool = try cache_mod.ExpertSlotPool.initFromArena(&weight_shards[0].arena, weight_shards[0].buffer, cold_pool_spec);
            errdefer pool.deinit();
            var reader = try aio.AsyncExpertReader.init(allocator, .{ .max_inflight = 16 });
            errdefer reader.deinit(allocator);
            var cold_amf = try amf.AmfFile.open(allocator, source.file.path);
            errdefer cold_amf.close(allocator);
            crt.* = .{
                .pool = pool,
                .reader = reader,
                .expert_cache = undefined,
                .amf_file = cold_amf,
                .slot_count = cold_slot_count,
            };
            crt.expert_cache = try cache_mod.ExpertCache.init(allocator, &crt.pool, &crt.reader, .{
                .num_layers = @intCast(config.num_hidden_layers),
                .num_experts = @intCast(config.num_experts),
            });
            cold_rt = crt;
        }

        // --- 5. Bind per-layer weights, KV tiles/rings, and MoE expert tables ---
        const gqa_kv_tiles = if (num_gqa_layers > 0)
            try allocator.alloc(metal.KvTile, @as(usize, num_gqa_layers) * num_chunks * 2)
        else
            try allocator.alloc(metal.KvTile, 0);
        var gqa_tiles_built: usize = 0;
        errdefer {
            if (gqa_tiles_built > 0) {
                compute.removeTilesResident(gqa_kv_tiles[0..gqa_tiles_built]) catch |err| {
                    std.debug.panic("removeTilesResident failed during GraphModel errdefer: {}", .{err});
                };
                for (0..gqa_tiles_built) |ti| gqa_kv_tiles[ti].close();
            }
            allocator.free(gqa_kv_tiles);
        }

        const layers = try allocator.alloc(GraphLayerWeights, config.num_hidden_layers);
        var layers_built: usize = 0;
        errdefer {
            for (0..layers_built) |li| {
                switch (layers[li].attn) {
                    .qsa => |*qsa| {
                        qsa.kv_ring.removeResident(compute) catch |err| {
                            std.debug.panic("QSA kv_ring.removeResident failed during errdefer: {}", .{err});
                        };
                        qsa.kv_ring.deinit(allocator);
                    },
                    else => {},
                }
            }
            allocator.free(layers);
        }

        var gdn_cursor: u32 = 0;
        var qsa_cursor: u32 = 0;
        var gqa_cursor: u32 = 0;
        var name_buf: [160]u8 = undefined;
        var single_expert_bytes: u64 = 0;
        var total_layers_non_expert_bytes: u64 = 0;
        var total_layers_static_bytes: u64 = 0;

        for (0..config.num_hidden_layers) |l| {
            const in_ln_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable;
            const in_ln = try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, in_ln_nm, h);
            const post_ln_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable;
            const post_ln = try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, post_ln_nm, h);

            var gr_w: ?GrLayerWeights = null;
            if (config.axes.residual_kind == .gated_4branch) {
                const r_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.gr.read_gate.weight", .{l}) catch unreachable;
                const re = source.find(r_nm) orelse return error.MissingTensor;
                const r_idx = entryIndexOf(entry_base_ptr, re);
                const w_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.gr.write_gate.weight", .{l}) catch unreachable;
                const we = source.find(w_nm) orelse return error.MissingTensor;
                const w_idx = entryIndexOf(entry_base_ptr, we);
                const d_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.gr.decay_gate.weight", .{l}) catch unreachable;
                const de = source.find(d_nm) orelse return error.MissingTensor;
                const d_idx = entryIndexOf(entry_base_ptr, de);
                const gdt: quant.FloatDType = if ((try re.dataType()) == .bf16) .bf16 else .f16;
                gr_w = .{
                    .read_gate = .{ .buffer = weight_shards[entry_shard[r_idx]].buffer, .offset = entry_offsets[r_idx] },
                    .write_gate = .{ .buffer = weight_shards[entry_shard[w_idx]].buffer, .offset = entry_offsets[w_idx] },
                    .decay_gate = .{ .buffer = weight_shards[entry_shard[d_idx]].buffer, .offset = entry_offsets[d_idx] },
                    .dtype = gdt,
                };
                total_layers_non_expert_bytes += 3 * 4 * @as(u64, h) * 2;
            }

            var attn_union: LayerAttention = undefined;
            var layer_attn_bytes: u64 = 0;
            switch (config.axes.layer_attention_kinds[l]) {
                .gated_deltanet => {
                    const qk_d = config.gdnQkDim();
                    const v_d = config.gdnVDim();
                    const q_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.q_proj", .{l}) catch unreachable;
                    const q_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, q_nm, qk_d, h);
                    const k_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.k_proj", .{l}) catch unreachable;
                    const k_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, k_nm, qk_d, h);
                    const v_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.v_proj", .{l}) catch unreachable;
                    const v_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, v_nm, v_d, h);
                    const g_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.g_proj", .{l}) catch unreachable;
                    const g_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, g_nm, v_d, h);
                    const o_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.o_proj", .{l}) catch unreachable;
                    const o_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, o_nm, h, v_d);

                    const a_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.a_proj.weight", .{l}) catch unreachable;
                    const ae = source.find(a_nm) orelse return error.MissingTensor;
                    const a_idx = entryIndexOf(entry_base_ptr, ae);
                    const b_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.b_proj.weight", .{l}) catch unreachable;
                    const be = source.find(b_nm) orelse return error.MissingTensor;
                    const b_idx = entryIndexOf(entry_base_ptr, be);
                    const al_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.A_log", .{l}) catch unreachable;
                    const ale = source.find(al_nm) orelse return error.MissingTensor;
                    const al_idx = entryIndexOf(entry_base_ptr, ale);
                    const dt_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.dt_bias", .{l}) catch unreachable;
                    const dte = source.find(dt_nm) orelse return error.MissingTensor;
                    const dt_idx = entryIndexOf(entry_base_ptr, dte);

                    var conv_w: ?metal.BufferBinding = null;
                    var conv_k: u32 = 0;
                    const cw_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.conv1d.weight", .{l}) catch unreachable;
                    if (source.find(cw_nm)) |cwe| {
                        const cw_idx = entryIndexOf(entry_base_ptr, cwe);
                        conv_w = .{ .buffer = weight_shards[entry_shard[cw_idx]].buffer, .offset = entry_offsets[cw_idx] };
                        conv_k = @intCast(cwe.shape[1]);
                    }
                    const norm_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.norm.weight", .{l}) catch unreachable;
                    const out_norm: ?qwen3.NormBinding = if (source.find(norm_nm) != null)
                        try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, norm_nm, config.linear_value_head_dim)
                    else
                        null;

                    const tk_r = try scratch_arena.allocAligned(@as(usize, max_m) * qk_d * 2, 64);
                    const tv_r = try scratch_arena.allocAligned(@as(usize, max_m) * v_d * 2, 64);
                    const tg_r = try scratch_arena.allocAligned(@as(usize, max_m) * config.linear_num_value_heads * 4, 64);
                    const tb_r = try scratch_arena.allocAligned(@as(usize, max_m) * config.linear_num_value_heads * 4, 64);
                    const tqr_r = try scratch_arena.allocAligned(@as(usize, max_m) * qk_d * 2, 64);
                    const tkr_r = try scratch_arena.allocAligned(@as(usize, max_m) * qk_d * 2, 64);
                    const tvr_r = try scratch_arena.allocAligned(@as(usize, max_m) * v_d * 2, 64);
                    const conv_hist = if (conv_k > 1) conv_k - 1 else 1;
                    const conv_d = 2 * qk_d + v_d;
                    const cs_r = try scratch_arena.allocAligned(@as(usize, conv_hist) * conv_d * 2, 64);
                    const cst_r = try scratch_arena.allocAligned(@as(usize, conv_hist) * conv_d * 2, 64);
                    @memset(cs_r, 0);
                    @memset(cst_r, 0);

                    attn_union = .{ .gdn = .{
                        .gdn_idx = gdn_cursor,
                        .q_proj = q_p,
                        .k_proj = k_p,
                        .v_proj = v_p,
                        .g_proj = g_p,
                        .o_proj = o_p,
                        .a_proj = .{ .buffer = weight_shards[entry_shard[a_idx]].buffer, .offset = entry_offsets[a_idx] },
                        .b_proj = .{ .buffer = weight_shards[entry_shard[b_idx]].buffer, .offset = entry_offsets[b_idx] },
                        .ab_dtype = if ((try ae.dataType()) == .bf16) .bf16 else .f16,
                        .a_log = .{ .buffer = weight_shards[entry_shard[al_idx]].buffer, .offset = entry_offsets[al_idx] },
                        .dt_bias = .{ .buffer = weight_shards[entry_shard[dt_idx]].buffer, .offset = entry_offsets[dt_idx] },
                        .conv1d_w = conv_w,
                        .conv_k = conv_k,
                        .out_norm = out_norm,
                        .tree_k_off = scratch_arena.offsetOf(tk_r.ptr),
                        .tree_v_off = scratch_arena.offsetOf(tv_r.ptr),
                        .tree_g_off = scratch_arena.offsetOf(tg_r.ptr),
                        .tree_beta_off = scratch_arena.offsetOf(tb_r.ptr),
                        .tree_q_raw_off = scratch_arena.offsetOf(tqr_r.ptr),
                        .tree_k_raw_off = scratch_arena.offsetOf(tkr_r.ptr),
                        .tree_v_raw_off = scratch_arena.offsetOf(tvr_r.ptr),
                        .conv_state_off = scratch_arena.offsetOf(cs_r.ptr),
                        .conv_state_tmp_off = scratch_arena.offsetOf(cst_r.ptr),
                    } };
                    gdn_cursor += 1;
                    layer_attn_bytes = q_p.alg_bytes + k_p.alg_bytes + v_p.alg_bytes + g_p.alg_bytes + o_p.alg_bytes;
                },
                .qsa_sparse => {
                    const q_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable;
                    const q_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, q_nm, config.qDim(), h);
                    const k_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable;
                    const k_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, k_nm, config.kvDim(), h);
                    const v_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable;
                    const v_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, v_nm, config.kvDim(), h);
                    const o_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable;
                    const o_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, o_nm, h, config.qDim());

                    const qn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable;
                    const qn: ?qwen3.NormBinding = if (source.find(qn_nm) != null)
                        try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, qn_nm, config.head_dim)
                    else
                        null;
                    const kn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable;
                    const kn: ?qwen3.NormBinding = if (source.find(kn_nm) != null)
                        try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, kn_nm, config.head_dim)
                    else
                        null;

                    var kv_ring = try hybrid.HybridKvRing.init(allocator, device, &scratch_arena, .{
                        .mode = qsa_ring_mode,
                        .num_kv_heads = config.num_key_value_heads,
                        .head_dim = config.head_dim,
                        .num_tiles = qsa_num_tiles,
                        .window_size_override = if (config.qsa_window_size > 0 and qsa_ring_mode == .sliding_window_128) config.qsa_window_size else null,
                    });
                    errdefer kv_ring.deinit(allocator);
                    try kv_ring.makeResident(compute);
                    for (kv_ring.k_tiles) |kt| total_wired_bytes += (try kt.info()).allocBytes();
                    for (kv_ring.v_tiles) |vt| total_wired_bytes += (try vt.info()).allocBytes();

                    const idx_buf = try hybrid.QsaIndexBuffer.init(&scratch_arena, .{
                        .max_nodes = max_m,
                        .num_kv_heads = config.num_key_value_heads,
                        .top_k_blocks = config.qsa_top_k_blocks,
                    });
                    const bk_raw = try scratch_arena.allocAligned(@as(usize, config.num_key_value_heads) * qsa_max_blocks * config.head_dim * 2, 64);
                    @memset(bk_raw, 0);
                    const tk_r = try scratch_arena.allocAligned(@as(usize, max_m) * config.kvDim() * 2, 64);
                    const tv_r = try scratch_arena.allocAligned(@as(usize, max_m) * config.kvDim() * 2, 64);

                    attn_union = .{ .qsa = .{
                        .qsa_idx = qsa_cursor,
                        .q_proj = q_p,
                        .k_proj = k_p,
                        .v_proj = v_p,
                        .o_proj = o_p,
                        .q_norm = if (qn) |n| n.binding else null,
                        .k_norm = if (kn) |n| n.binding else null,
                        .qk_norm_dtype = if (qn) |n| n.dtype else .bf16,
                        .kv_ring = kv_ring,
                        .index_buf = idx_buf,
                        .block_keys_off = scratch_arena.offsetOf(bk_raw.ptr),
                        .max_blocks = qsa_max_blocks,
                        .tree_k_off = scratch_arena.offsetOf(tk_r.ptr),
                        .tree_v_off = scratch_arena.offsetOf(tv_r.ptr),
                    } };
                    qsa_cursor += 1;
                    layer_attn_bytes = q_p.alg_bytes + k_p.alg_bytes + v_p.alg_bytes + o_p.alg_bytes;
                },
                .gqa_full => {
                    const q_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable;
                    const q_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, q_nm, config.qDim(), h);
                    const k_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable;
                    const k_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, k_nm, config.kvDim(), h);
                    const v_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable;
                    const v_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, v_nm, config.kvDim(), h);
                    const o_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable;
                    const o_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, o_nm, h, config.qDim());

                    const qn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable;
                    const qn: ?qwen3.NormBinding = if (source.find(qn_nm) != null)
                        try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, qn_nm, config.head_dim)
                    else
                        null;
                    const kn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable;
                    const kn: ?qwen3.NormBinding = if (source.find(kn_nm) != null)
                        try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, kn_nm, config.head_dim)
                    else
                        null;

                    const tile_start = gqa_tiles_built;
                    const tbl_raw = try scratch_arena.allocAligned(@as(usize, num_chunks) * @sizeOf(attention.KvTilePair), 64);
                    const tbl: []attention.KvTilePair = @alignCast(std.mem.bytesAsSlice(attention.KvTilePair, tbl_raw[0 .. @as(usize, num_chunks) * @sizeOf(attention.KvTilePair)]));
                    const rows = config.num_key_value_heads * options.chunk_tokens;
                    for (0..num_chunks) |c| {
                        gqa_kv_tiles[gqa_tiles_built] = try metal.KvTile.init(device, config.head_dim, rows, .{});
                        total_wired_bytes += (try gqa_kv_tiles[gqa_tiles_built].info()).allocBytes();
                        gqa_tiles_built += 1;
                        gqa_kv_tiles[gqa_tiles_built] = try metal.KvTile.init(device, config.head_dim, rows, .{});
                        total_wired_bytes += (try gqa_kv_tiles[gqa_tiles_built].info()).allocBytes();
                        gqa_tiles_built += 1;
                        tbl[c] = attention.KvTilePair.fromTiles(gqa_kv_tiles[gqa_tiles_built - 2], gqa_kv_tiles[gqa_tiles_built - 1], .rgba16f);
                    }
                    try compute.makeTilesResident(gqa_kv_tiles[tile_start..gqa_tiles_built]);
                    const tk_r = try scratch_arena.allocAligned(@as(usize, max_m) * config.kvDim() * 2, 64);
                    const tv_r = try scratch_arena.allocAligned(@as(usize, max_m) * config.kvDim() * 2, 64);

                    attn_union = .{ .gqa = .{
                        .gqa_idx = gqa_cursor,
                        .q_proj = q_p,
                        .k_proj = k_p,
                        .v_proj = v_p,
                        .o_proj = o_p,
                        .q_norm = if (qn) |n| n.binding else null,
                        .k_norm = if (kn) |n| n.binding else null,
                        .qk_norm_dtype = if (qn) |n| n.dtype else .bf16,
                        .kv_tiles_binding = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tbl_raw.ptr) },
                        .kv_tile_start = tile_start,
                        .tree_k_off = scratch_arena.offsetOf(tk_r.ptr),
                        .tree_v_off = scratch_arena.offsetOf(tv_r.ptr),
                    } };
                    gqa_cursor += 1;
                    layer_attn_bytes = q_p.alg_bytes + k_p.alg_bytes + v_p.alg_bytes + o_p.alg_bytes;
                },
                .mla_csa2 => return error.UnsupportedOperatorAxis,
            }

            var lw: GraphLayerWeights = .{
                .shard_idx = 0,
                .input_layernorm = in_ln,
                .post_attention_layernorm = post_ln,
                .gr = gr_w,
                .attn = attn_union,
            };

            switch (config.axes.ffn_kind) {
                .dense_swiglu => {
                    const gp_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable;
                    lw.dense_gate = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, gp_nm, config.intermediate_size, h);
                    const up_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}) catch unreachable;
                    lw.dense_up = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, up_nm, config.intermediate_size, h);
                    const dp_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}) catch unreachable;
                    lw.dense_down = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, dp_nm, h, config.intermediate_size);
                    total_layers_static_bytes += layer_attn_bytes + lw.dense_gate.alg_bytes + lw.dense_up.alg_bytes + lw.dense_down.alg_bytes;
                },
                .moe_swiglu => {
                    // Router (`layers.{l}.mlp.gate.weight` or `layers.{l}.mlp.gate`)
                    const r_nm_w = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate.weight", .{l}) catch unreachable;
                    const re = source.find(r_nm_w) orelse blk: {
                        const r_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate", .{l}) catch unreachable;
                        break :blk source.find(r_nm) orelse return error.MissingTensor;
                    };
                    const r_idx = entryIndexOf(entry_base_ptr, re);
                    const r_si = entry_shard[r_idx];
                    lw.router_binding = .{ .buffer = weight_shards[r_si].buffer, .offset = entry_offsets[r_idx] };
                    const rdt = try re.dataType();
                    if (amfToWeightDType(rdt)) |wdt| {
                        if (wdt.isQuantized()) {
                            lw.router_quant_dtype = wdt;
                            lw.router_scale_bias = .{ .buffer = weight_shards[r_si].buffer, .offset = entry_offsets[r_idx] + re.params_offset };
                        } else {
                            lw.router_dtype = if (wdt == .bf16) .bf16 else .f16;
                        }
                    } else if (rdt == .f32) {
                        lw.router_dtype = .f32;
                    } else return error.UnsupportedDType;

                    // Optional `shared_expert` + `shared_expert_gate` (`G7`).
                    const sg_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_expert.gate_proj", .{l}) catch unreachable;
                    if (source.find(sg_nm) != null) {
                        const sh_inter = if (config.shared_expert_intermediate_size > 0) config.shared_expert_intermediate_size else config.moe_intermediate_size;
                        const sg = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, sg_nm, sh_inter, h);
                        const su_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_expert.up_proj", .{l}) catch unreachable;
                        const su = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, su_nm, sh_inter, h);
                        const sd_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_expert.down_proj", .{l}) catch unreachable;
                        const sd = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, sd_nm, h, sh_inter);
                        lw.shared_expert = .{
                            .gate_qweight_offset = sg.weight.offset,
                            .gate_sb_offset = if (sg.scale_bias) |b| b.offset else 0,
                            .up_qweight_offset = su.weight.offset,
                            .up_sb_offset = if (su.scale_bias) |b| b.offset else 0,
                            .down_qweight_offset = sd.weight.offset,
                            .down_sb_offset = if (sd.scale_bias) |b| b.offset else 0,
                        };
                        lw.shared_expert_buf = sg.weight.buffer;
                        lw.shared_expert_dtype = sg.dtype;
                        layer_attn_bytes += sg.alg_bytes + su.alg_bytes + sd.alg_bytes;

                        const sgate_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_expert_gate.weight", .{l}) catch unreachable;
                        if (source.find(sgate_nm)) |sge| {
                            const sg_idx = entryIndexOf(entry_base_ptr, sge);
                            lw.shared_gate_weight = .{ .buffer = weight_shards[entry_shard[sg_idx]].buffer, .offset = entry_offsets[sg_idx] };
                            lw.shared_gate_dtype = switch (try sge.dataType()) {
                                .bf16 => .bf16,
                                .f16 => .f16,
                                .f32 => .f32,
                                else => return error.UnsupportedDType,
                            };
                        }
                    }

                    // Routed experts table `[num_experts]ExpertWeights` (`E <= 1024`).
                    const tbl_raw = try scratch_arena.allocAligned(@as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights), 64);
                    const tbl: []moe.ExpertWeights = @alignCast(std.mem.bytesAsSlice(moe.ExpertWeights, tbl_raw[0 .. @as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights)]));
                    const dt_raw = try scratch_arena.allocAligned(config.num_experts, 64);
                    lw.expert_table = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tbl_raw.ptr) };
                    lw.expert_table_host = tbl;

                    var first_dt: ?quant.WeightDType = null;
                    var hetero = false;
                    var exp_buf: metal.Buffer = weight_shards[0].buffer;
                    for (0..config.num_experts) |e| {
                        const g_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }) catch unreachable;
                        const ge = source.find(g_nm) orelse return error.MissingTensor;
                        const u_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }) catch unreachable;
                        const ue = source.find(u_nm) orelse return error.MissingTensor;
                        const d_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }) catch unreachable;
                        const de = source.find(d_nm) orelse return error.MissingTensor;

                        const wdt = amfToWeightDType(try ge.dataType()) orelse return error.UnsupportedDType;
                        if (first_dt) |fdt| {
                            if (fdt != wdt) hetero = true;
                        } else {
                            first_dt = wdt;
                        }
                        dt_raw[e] = @backingInt(wdt);
                        if (wdt.isQuantized()) {
                            const bucket: usize = switch (wdt) {
                                .q2_affine => 0,
                                .q3_affine => 1,
                                .q4_affine => 2,
                                else => 0,
                            };
                            lw.expert_scheme_counts[bucket] += 1;
                        }

                        if ((try ge.tensorKind()) == .cold_expert) {
                            lw.has_cold_experts = true;
                            lw.cold_expert_count += 1;
                            const triple = try cache_mod.ColdExpertTriple.fromAmfEntries(cold_rt.?.amf_file.?.fd, 0, ge, ue, de);
                            try cold_rt.?.expert_cache.registerCold(@intCast(l), @intCast(e), triple);
                            tbl[e] = std.mem.zeroes(moe.ExpertWeights);
                            if (single_expert_bytes == 0) single_expert_bytes = triple.totalBlobBytes();
                        } else {
                            const g_idx = entryIndexOf(entry_base_ptr, ge);
                            const u_idx = entryIndexOf(entry_base_ptr, ue);
                            const d_idx = entryIndexOf(entry_base_ptr, de);
                            exp_buf = weight_shards[entry_shard[g_idx]].buffer;
                            const ew = makeExpertWeights(wdt, entry_offsets[g_idx], ge.params_offset, entry_offsets[u_idx], ue.params_offset, entry_offsets[d_idx], de.params_offset);
                            tbl[e] = ew;
                            if (cold_rt) |crt| {
                                try crt.expert_cache.registerHot(@intCast(l), @intCast(e), .{
                                    .gate = try expert.WiredExpert.init(entry_page_slices[g_idx]),
                                    .up = try expert.WiredExpert.init(entry_page_slices[u_idx]),
                                    .down = try expert.WiredExpert.init(entry_page_slices[d_idx]),
                                    .weights = ew,
                                    .metal_buffer = exp_buf,
                                });
                            }
                            if (single_expert_bytes == 0) {
                                single_expert_bytes = ge.blob_bytes + ue.blob_bytes + de.blob_bytes;
                            }
                        }
                    }
                    lw.expert_weights_buf = exp_buf;
                    lw.expert_weight_dtype = first_dt orelse .q3_affine;
                    lw.is_hetero_moe = hetero;
                    if (hetero) {
                        lw.expert_dtypes_binding = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(dt_raw.ptr) };
                    }
                    total_layers_non_expert_bytes += layer_attn_bytes;
                },
            }

            layers[l] = lw;
            layers_built += 1;
        }

        // --- 6. Optional PLE Store (`C10`, `F_NOCACHE`) ---
        var ple_rt: ?*PleRuntime = null;
        errdefer if (ple_rt) |prt| {
            allocator.free(prt.f32_tmp);
            prt.scratch.deinit(allocator);
            prt.store.deinit(allocator);
            allocator.destroy(prt);
        };
        if (options.ple_paths.len > 0) {
            const prt = try allocator.create(PleRuntime);
            errdefer allocator.destroy(prt);
            var store = try ple_mod.PleStore.open(allocator, options.ple_paths, .{
                .cache = .{
                    .cache_pages = options.ple_cache_pages,
                    .budget_override = wired_budget,
                },
            });
            errdefer store.deinit(allocator);
            var ple_scratch = try ple_mod.LoopPleScratch.init(
                allocator,
                @intCast(config.num_hidden_layers),
                @intCast(max_m),
                h,
                wired_budget,
            );
            errdefer ple_scratch.deinit(allocator);
            const f32_tmp = try allocator.alloc(f32, @as(usize, max_m) * h);
            errdefer allocator.free(f32_tmp);
            total_wired_bytes += store.totalWiredBytes() + ple_scratch.totalWiredBytes();
            prt.* = .{ .store = store, .scratch = ple_scratch, .f32_tmp = f32_tmp };
            ple_rt = prt;
        }

        // --- 7. Optional SME2 Step-LoRA Loop Adapters (`W16`, inline or external `.amf`) ---
        var adapter_rt: ?*LoopAdapterRuntime = null;
        errdefer if (adapter_rt) |art| {
            art.deinit(allocator);
            allocator.destroy(art);
        };
        if (inline_loop_adapters > 0 or options.loop_adapter_amf_path != null) {
            adapter_rt = try loadStepLoraAdapters(
                allocator,
                source,
                entry_slices,
                entry_base_ptr,
                options.loop_adapter_amf_path,
                config,
                effective_max_loops,
                max_m,
                wired_budget,
                &total_wired_bytes,
            );
        }

        const token_history = try allocator.alloc(u32, options.max_seq_len + max_m);
        errdefer allocator.free(token_history);
        @memset(token_history, 0);

        const ple_row_indices_host = try allocator.alloc(u64, max_m);
        errdefer allocator.free(ple_row_indices_host);
        @memset(ple_row_indices_host, 0);

        const aux_ids = options.aux_layer_ids orelse qwen3.defaultEagle3AuxLayerIds(config.num_hidden_layers);
        const embed_row_alg_bytes = embed.alg_bytes / config.vocab_size;
        const global_head_bytes = embed_row_alg_bytes + lm_head.alg_bytes;

        return .{
            .config = config,
            .effective_max_loops = effective_max_loops,
            .enable_loop_early_exit = options.enable_loop_early_exit,
            .early_exit_threshold = options.early_exit_threshold orelse config.axes.loop_schedule.early_exit_threshold,
            .max_seq_len = options.max_seq_len,
            .max_tree_nodes = max_m,
            .chunk_tokens = options.chunk_tokens,
            .num_chunks = num_chunks,
            .seq_len = 0,
            .timeout_ms = options.timeout_ms,
            .wired_budget_bytes = wired_budget.limit_bytes,
            .total_wired_bytes = total_wired_bytes,
            .enable_feature_taps = options.enable_feature_taps,
            .aux_layer_ids = aux_ids,
            .device = device,
            .library = library,
            .compute = compute,
            .dequant_pipes = dequant_pipes,
            .attn_pipes = attn_pipes,
            .moe_pipes = moe_pipes,
            .hybrid_pipes = hybrid_pipes,
            .weight_shards = weight_shards,
            .scratch_arena = scratch_arena,
            .scratch_buf = scratch_buf,
            .gdn_arena = gdn_arena,
            .gr_arena = gr_arena,
            .gqa_kv_tiles = gqa_kv_tiles,
            .embed = embed,
            .final_norm = final_norm,
            .lm_head = lm_head,
            .early_exit_gate_host = early_exit_gate_host,
            .early_exit_gate_dtype = early_exit_gate_dtype,
            .layers = layers,
            .ple_rt = ple_rt,
            .cold_rt = cold_rt,
            .adapter_rt = adapter_rt,
            .token_history = token_history,
            .token_ids_host = @alignCast(std.mem.bytesAsSlice(u32, tok_raw[0 .. @as(usize, max_m) * 4])),
            .positions_host = @alignCast(std.mem.bytesAsSlice(u32, pos_raw[0 .. @as(usize, max_m) * 4])),
            .pos3d_host = @alignCast(std.mem.bytesAsSlice(hybrid.Pos3D, pos3d_raw[0 .. @as(usize, max_m) * @sizeOf(hybrid.Pos3D)])),
            .ancestor_masks_host = @alignCast(std.mem.bytesAsSlice(u64, msk_raw[0 .. @as(usize, max_m) * 8])),
            .out_token_ids_host = @alignCast(std.mem.bytesAsSlice(u32, out_tok_raw[0 .. @as(usize, max_m) * 4])),
            .logits_host = @alignCast(std.mem.bytesAsSlice(f16, logits_raw[0 .. @as(usize, max_m) * config.vocab_size * 2])),
            .topk_host = @alignCast(std.mem.bytesAsSlice(moe.TopKCandidate, topk_raw[0 .. @as(usize, max_m) * moe.MAX_TOP_K * @sizeOf(moe.TopKCandidate)])),
            .ple_row_indices_host = ple_row_indices_host,
            .ple_f16_host = @alignCast(std.mem.bytesAsSlice(f16, ple_f16_raw[0 .. @as(usize, config.num_hidden_layers) * max_m * h * 2])),
            .x_host = @alignCast(std.mem.bytesAsSlice(f16, x_raw[0 .. @as(usize, max_m) * h * 2])),
            .x_norm_host = @alignCast(std.mem.bytesAsSlice(f16, x_norm_raw[0 .. @as(usize, max_m) * h * 2])),
            .q_host = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0 .. @as(usize, max_m) * max_q_dim * 2])),
            .k_host = @alignCast(std.mem.bytesAsSlice(f16, k_raw[0 .. @as(usize, max_m) * max_kv_dim * 2])),
            .v_host = @alignCast(std.mem.bytesAsSlice(f16, v_raw[0 .. @as(usize, max_m) * max_kv_dim * 2])),
            .attn_out_host = @alignCast(std.mem.bytesAsSlice(f16, attn_out_raw[0 .. @as(usize, max_m) * max_attn_out * 2])),
            .proj_out_host = @alignCast(std.mem.bytesAsSlice(f16, proj_out_raw[0 .. @as(usize, max_m) * h * 2])),
            .dense_gate_host = @alignCast(std.mem.bytesAsSlice(f16, dgate_raw[0 .. @as(usize, max_m) * max_inter * 2])),
            .dense_up_host = @alignCast(std.mem.bytesAsSlice(f16, dup_raw[0 .. @as(usize, max_m) * max_inter * 2])),
            .dense_act_host = @alignCast(std.mem.bytesAsSlice(f16, dact_raw[0 .. @as(usize, max_m) * max_inter * 2])),
            .accepted_top_host = @alignCast(std.mem.bytesAsSlice(f16, acc_top_raw[0 .. @as(usize, max_m) * h * 2])),
            .accepted_aux_host = @alignCast(std.mem.bytesAsSlice(f16, acc_aux_raw[0 .. @as(usize, max_m) * 3 * h * 2])),
            .token_ids_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tok_raw.ptr) },
            .positions_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(pos_raw.ptr) },
            .pos3d_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(pos3d_raw.ptr) },
            .ancestor_masks_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(msk_raw.ptr) },
            .out_token_id_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(out_tok_raw.ptr) },
            .logits_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(logits_raw.ptr) },
            .topk_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(topk_raw.ptr) },
            .ple_f16_base_off = scratch_arena.offsetOf(ple_f16_raw.ptr),
            .x_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_raw.ptr) },
            .x_init_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_init_raw.ptr) },
            .x_norm_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_norm_raw.ptr) },
            .q_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(q_raw.ptr) },
            .k_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(k_raw.ptr) },
            .v_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(v_raw.ptr) },
            .q_conv_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(q_conv_raw.ptr) },
            .k_conv_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(k_conv_raw.ptr) },
            .v_conv_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(v_conv_raw.ptr) },
            .gdn_z_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gdn_z_raw.ptr) },
            .gdn_g_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gdn_g_raw.ptr) },
            .gdn_beta_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gdn_beta_raw.ptr) },
            .gr_g_read_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gr_g_read_raw.ptr) },
            .gr_g_write_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gr_g_write_raw.ptr) },
            .gr_decay_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gr_decay_raw.ptr) },
            .attn_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(attn_out_raw.ptr) },
            .proj_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(proj_out_raw.ptr) },
            .dense_gate_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(dgate_raw.ptr) },
            .dense_up_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(dup_raw.ptr) },
            .dense_act_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(dact_raw.ptr) },
            .gqa_scratch_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gqa_scratch_raw.ptr) },
            .moe_ws = moe_ws,
            .layer_sched_hdr_raw = layer_sched_hdr_raw,
            .layer_sched_hdr_off = scratch_arena.offsetOf(layer_sched_hdr_raw.ptr),
            .accepted_top_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(acc_top_raw.ptr) },
            .accepted_aux_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(acc_aux_raw.ptr) },
            .aux_tree_base_off = scratch_arena.offsetOf(aux_tree_raw.ptr),
            .single_expert_bytes = single_expert_bytes,
            .total_layers_non_expert_bytes = total_layers_non_expert_bytes,
            .total_layers_static_bytes = total_layers_static_bytes,
            .global_head_bytes = global_head_bytes,
            .embed_row_alg_bytes = embed_row_alg_bytes,
            .last_logits_row = 0,
            .last_verified_nodes = 0,
            .last_verified_tokens = @splat(0),
            .last_verified_masks = @splat(0),
            .loop_stats = .{},
        };
    }
};

// ===========================================================================
// Section 6: Loader Helpers (Step-LoRA Parsing, DType Conversion, Expert Weights)
// ===========================================================================

fn amfToWeightDType(dt: amf.DType) ?quant.WeightDType {
    return switch (dt) {
        .q2_affine => .q2_affine,
        .q3_affine => .q3_affine,
        .q4_affine => .q4_affine,
        .f16 => .f16,
        .bf16 => .bf16,
        else => null,
    };
}

fn readFloatWeightAt(raw: []const u8, dt: quant.FloatDType, idx: usize) f32 {
    const bits = std.mem.readInt(u16, raw[idx * 2 ..][0..2], .little);
    return quant.floatBitsToF32(dt, bits);
}

fn makeExpertWeights(
    wdt: quant.WeightDType,
    gate_off: u64,
    gate_params_off: u64,
    up_off: u64,
    up_params_off: u64,
    down_off: u64,
    down_params_off: u64,
) moe.ExpertWeights {
    if (wdt.isQuantized()) {
        return .{
            .gate_qweight_offset = gate_off,
            .gate_sb_offset = gate_off + gate_params_off,
            .up_qweight_offset = up_off,
            .up_sb_offset = up_off + up_params_off,
            .down_qweight_offset = down_off,
            .down_sb_offset = down_off + down_params_off,
        };
    }
    return .{
        .gate_qweight_offset = gate_off,
        .gate_sb_offset = 0,
        .up_qweight_offset = up_off,
        .up_sb_offset = 0,
        .down_qweight_offset = down_off,
        .down_sb_offset = 0,
    };
}

const ParsedAdapterName = struct {
    step: u32,
    layer: u32,
    proj: AdapterProjKind,
    part: enum { lora_a, lora_b, alpha },
};

fn parseLoopAdapterName(name: []const u8) ?ParsedAdapterName {
    // Expected: `loop.step.{r}.layers.{l}.{proj}.{lora_a|lora_b|alpha}`
    const prefix = "loop.step.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    var rest = name[prefix.len..];
    const dot1 = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const step = std.fmt.parseInt(u32, rest[0..dot1], 10) catch return null;
    rest = rest[dot1 + 1 ..];
    const l_pref = "layers.";
    if (!std.mem.startsWith(u8, rest, l_pref)) return null;
    rest = rest[l_pref.len..];
    const dot2 = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, rest[0..dot2], 10) catch return null;
    rest = rest[dot2 + 1 ..];
    const dot3 = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const proj = AdapterProjKind.parse(rest[0..dot3]) orelse return null;
    const tail = rest[dot3 + 1 ..];
    const part: @TypeOf(@as(ParsedAdapterName, undefined).part) = if (std.mem.eql(u8, tail, "lora_a"))
        .lora_a
    else if (std.mem.eql(u8, tail, "lora_b"))
        .lora_b
    else if (std.mem.eql(u8, tail, "alpha"))
        .alpha
    else
        return null;
    return .{ .step = step, .layer = layer, .proj = proj, .part = part };
}

fn loadStepLoraAdapters(
    allocator: std.mem.Allocator,
    main_source: GraphModel.AmfSource,
    main_entry_slices: []const []u8,
    main_base_ptr: usize,
    external_path: ?[:0]const u8,
    config: GraphConfig,
    effective_max_loops: u32,
    max_m: u32,
    wired_budget: budget_mod.Budget,
    total_wired_bytes: *u64,
) GraphLoadError!*LoopAdapterRuntime {
    var ext_file: ?amf.AmfFile = null;
    defer if (ext_file) |*f| f.close(allocator);
    var ext_arena: ?arena_mod.WiredArena = null;
    errdefer if (ext_arena) |*ea| ea.deinit();
    var ext_slices: [][]u8 = &.{};
    defer if (ext_slices.len > 0) allocator.free(ext_slices);

    if (external_path) |ep| {
        var f = try amf.AmfFile.open(allocator, ep);
        errdefer f.close(allocator);
        var need_bytes: usize = 0;
        for (f.parsed.entries) |*e| {
            need_bytes += try page.alignUp(@intCast(e.blob_bytes));
        }
        const aligned = try page.alignUp(@max(need_bytes, page.APPLE_PAGE_SIZE));
        try wired_budget.checkPlan(total_wired_bytes.* + aligned);
        var ea = try arena_mod.WiredArena.init(aligned, wired_budget);
        errdefer ea.deinit();
        const slices = try allocator.alloc([]u8, f.parsed.entries.len);
        errdefer allocator.free(slices);
        for (f.parsed.entries, 0..) |*e, i| {
            const len: usize = @intCast(e.blob_bytes);
            const raw = try f.loadBlob(e, &ea);
            slices[i] = raw[0..len];
        }
        total_wired_bytes.* += aligned;
        ext_file = f;
        ext_arena = ea;
        ext_slices = slices;
    }

    const use_entries: []const amf.TensorEntry = if (ext_file) |*f| f.parsed.entries else main_source.entries();
    const use_slices: []const []u8 = if (ext_file != null) ext_slices else main_entry_slices;
    const use_base: usize = if (ext_file != null) @intFromPtr(use_entries.ptr) else main_base_ptr;
    const findIn = struct {
        fn call(ext: ?*const amf.AmfFile, main_src: GraphModel.AmfSource, name: []const u8) ?*const amf.TensorEntry {
            if (ext) |f| return f.parsed.find(name);
            return main_src.find(name);
        }
    }.call;

    // Discover max_steps, max_k, max_n, max_rank across all `loop_adapter` tensors.
    var max_steps: u32 = @max(effective_max_loops, config.axes.loop_schedule.max_loops);
    var max_k: usize = @max(config.hidden_size, @max(config.qDim(), config.gdnVDim()));
    var max_n: usize = max_k;
    var max_rank: usize = sme.GROUP_SIZE;

    for (use_entries) |*e| {
        if ((try e.tensorKind()) != .loop_adapter) continue;
        const p = parseLoopAdapterName(e.nameSlice()) orelse continue;
        if (p.step + 1 > max_steps) max_steps = p.step + 1;
        if (p.part == .lora_a) {
            max_rank = @max(max_rank, @as(usize, @intCast(e.shape[0])));
            max_k = @max(max_k, @as(usize, @intCast(e.shape[1])));
        } else if (p.part == .lora_b) {
            max_n = @max(max_n, @as(usize, @intCast(e.shape[0])));
        }
    }

    const total_slots = @as(usize, max_steps) * config.num_hidden_layers * MAX_ADAPTER_PROJS_PER_LAYER;
    const table = try allocator.alloc(?sme.SmeStepLoraAdapter, total_slots);
    errdefer allocator.free(table);
    @memset(table, null);

    var nm_buf: [160]u8 = undefined;
    const ext_ptr: ?*const amf.AmfFile = if (ext_file) |*f| f else null;
    for (use_entries) |*e| {
        if ((try e.tensorKind()) != .loop_adapter) continue;
        const p = parseLoopAdapterName(e.nameSlice()) orelse continue;
        if (p.part != .lora_a) continue;
        if (p.layer >= config.num_hidden_layers) continue;

        const proj_name = @tagName(p.proj);
        const b_nm = std.fmt.bufPrint(&nm_buf, "loop.step.{d}.layers.{d}.{s}.lora_b", .{ p.step, p.layer, proj_name }) catch unreachable;
        const b_e = findIn(ext_ptr, main_source, b_nm) orelse return error.MissingTensor;
        const al_nm = std.fmt.bufPrint(&nm_buf, "loop.step.{d}.layers.{d}.{s}.alpha", .{ p.step, p.layer, proj_name }) catch unreachable;
        const al_e = findIn(ext_ptr, main_source, al_nm) orelse return error.MissingTensor;

        const a_idx = (@intFromPtr(e) - use_base) / @sizeOf(amf.TensorEntry);
        const b_idx = (@intFromPtr(b_e) - use_base) / @sizeOf(amf.TensorEntry);
        const al_idx = (@intFromPtr(al_e) - use_base) / @sizeOf(amf.TensorEntry);
        if (use_slices[al_idx].len < 4) return error.TensorShapeMismatch;
        const alpha: f32 = @bitCast(std.mem.readInt(u32, use_slices[al_idx][0..4], .little));
        if (p.step == 0 and config.axes.loop_schedule.enforce_zero_alpha_r0 and alpha != 0.0) {
            return error.NonZeroLoopZeroAlpha;
        }
        const adapter = try sme.SmeStepLoraAdapter.fromAmfEntries(e, use_slices[a_idx], b_e, use_slices[b_idx], alpha);
        const slot_idx = (@as(usize, p.step) * config.num_hidden_layers + p.layer) * MAX_ADAPTER_PROJS_PER_LAYER + @backingInt(p.proj);
        table[slot_idx] = adapter;
    }

    var scratch = try sme.StepLoraScratch.init(max_m, @min(max_rank, sme.MAX_STEP_LORA_RANK), max_k, wired_budget);
    errdefer scratch.deinit();
    total_wired_bytes.* += scratch.totalWiredBytes();

    const delta_f32 = try allocator.alloc(f32, @as(usize, max_m) * max_n);
    errdefer allocator.free(delta_f32);
    @memset(delta_f32, 0.0);

    var disp = try sme.SmeDispatcher.init(allocator, .{});
    errdefer disp.deinit();

    const art = try allocator.create(LoopAdapterRuntime);
    art.* = .{
        .external_arena = ext_arena,
        .dispatcher = disp,
        .scratch = scratch,
        .delta_f32 = delta_f32,
        .max_steps = max_steps,
        .num_layers = config.num_hidden_layers,
        .entries = table,
    };
    ext_arena = null; // ownership transferred
    return art;
}

// ===========================================================================
// Section 7: Unit Tests — CPU FP32 Parity, Qwen4/512E, LoopLM, Step-LoRA &
//            All Four ADR-0019 Hybrid-Sparse Loop Isolations
// ===========================================================================

const TestTensorBlob = struct {
    name: []const u8,
    kind: amf.TensorKind,
    dtype: amf.DType,
    layout: amf.TensorLayout = .row_major,
    layer: u16 = amf.NO_LAYER,
    expert: u16 = amf.NO_EXPERT,
    shape: [4]u32,
    ndim: u8,
    bytes: []u8,
    data_bytes_override: u64 = 0,
    params_offset: u64 = 0,
    params_bytes: u64 = 0,
};

fn buildTestAmf(allocator: std.mem.Allocator, model_id: []const u8, blobs: []const TestTensorBlob) ![]u8 {
    const dir_offset: u64 = page.APPLE_PAGE_SIZE;
    const raw_dir_bytes: usize = blobs.len * @sizeOf(amf.TensorEntry);
    const dir_bytes: u64 = try page.alignUp(@max(raw_dir_bytes, page.APPLE_PAGE_SIZE));
    var offset: u64 = dir_offset + dir_bytes;

    var entries = try allocator.alloc(amf.TensorEntry, blobs.len);
    defer allocator.free(entries);

    for (blobs, 0..) |b, i| {
        const data_len: u64 = if (b.data_bytes_override > 0) b.data_bytes_override else b.bytes.len;
        const blob_len: u64 = try page.alignUp(@max(b.bytes.len, 1));
        var e = amf.TensorEntry.zeroed(b.name, b.kind, b.dtype);
        e.setLayoutAndParamKind(b.layout, .affine_scale_bias);
        e.quant_bits = switch (b.dtype) {
            .q4_affine => 4,
            .q3_affine => 3,
            .q2_affine => 2,
            else => 0,
        };
        e.group_size = if (e.quant_bits > 0) 64 else 0;
        e.layer = b.layer;
        e.expert = b.expert;
        e.rank = b.ndim;
        e.shape = b.shape;
        e.blob_offset = offset;
        e.blob_bytes = blob_len;
        e.data_bytes = data_len;
        e.params_offset = b.params_offset;
        e.params_bytes = b.params_bytes;
        entries[i] = e;
        offset += blob_len;
    }

    const total: usize = @intCast(offset);
    const out = try allocator.alloc(u8, total);
    @memset(out, 0);

    for (blobs, 0..) |b, i| {
        const bo: usize = @intCast(entries[i].blob_offset);
        const bl: usize = @intCast(entries[i].blob_bytes);
        @memcpy(out[bo .. bo + b.bytes.len], b.bytes);
        entries[i].blob_crc32 = std.hash.Crc32.hash(out[bo .. bo + bl]);
    }

    const dir_slice = std.mem.sliceAsBytes(entries);
    var hdr: amf.Header = std.mem.zeroes(amf.Header);
    hdr.magic = amf.MAGIC;
    hdr.version = amf.FORMAT_VERSION;
    hdr.flags = 0;
    hdr.page_size = page.APPLE_PAGE_SIZE;
    hdr.checksum_algo = amf.CHECKSUM_CRC32_IEEE;
    hdr.tensor_count = @intCast(blobs.len);
    hdr.directory_offset = dir_offset;
    hdr.directory_bytes = dir_bytes;
    hdr.directory_crc32 = std.hash.Crc32.hash(dir_slice);
    const id_len = @min(model_id.len, hdr.model_id.len - 1);
    @memcpy(hdr.model_id[0..id_len], model_id[0..id_len]);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);

    @memcpy(out[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    @memcpy(out[@intCast(dir_offset)..][0..dir_slice.len], dir_slice);
    return out;
}

fn fillDeterministicF16(allocator: std.mem.Allocator, seed: u64, count: usize, amp: f32) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try allocator.alloc(u8, count * 2);
    for (0..count) |i| {
        const v: f32 = (r.float(f32) * 2.0 - 1.0) * amp;
        const bits: u16 = @bitCast(@as(f16, @floatCast(v)));
        std.mem.writeInt(u16, out[i * 2 ..][0..2], bits, .little);
    }
    return out;
}

fn fillDeterministicBf16(allocator: std.mem.Allocator, seed: u64, count: usize, amp: f32) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try allocator.alloc(u8, count * 2);
    for (0..count) |i| {
        const v: f32 = (r.float(f32) * 2.0 - 1.0) * amp;
        const bits = quant.f32ToFloatBits(.bf16, v);
        std.mem.writeInt(u16, out[i * 2 ..][0..2], bits, .little);
    }
    return out;
}

fn fillConstantBf16(allocator: std.mem.Allocator, count: usize, val: f32) ![]u8 {
    const out = try allocator.alloc(u8, count * 2);
    const bits = quant.f32ToFloatBits(.bf16, val);
    for (0..count) |i| {
        std.mem.writeInt(u16, out[i * 2 ..][0..2], bits, .little);
    }
    return out;
}

fn fillDeterministicF32(allocator: std.mem.Allocator, seed: u64, count: usize, lo: f32, hi: f32) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try allocator.alloc(u8, count * 4);
    for (0..count) |i| {
        const v: f32 = lo + r.float(f32) * (hi - lo);
        std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(v), .little);
    }
    return out;
}

fn fillDeterministicStepLoraSme2F16(
    allocator: std.mem.Allocator,
    seed: u64,
    n_rows: usize,
    k_cols: usize,
    active_rows: usize,
    active_cols: usize,
    amp: f32,
) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const row_major = try allocator.alloc(u16, n_rows * k_cols);
    defer allocator.free(row_major);
    @memset(row_major, 0);
    for (0..active_rows) |row| {
        for (0..active_cols) |col| {
            const v: f32 = (r.float(f32) * 2.0 - 1.0) * amp;
            row_major[row * k_cols + col] = @bitCast(@as(f16, @floatCast(v)));
        }
    }
    const packed_len = try sme.packedBlobBytes(.f16, n_rows, k_cols);
    const packed_buf = try allocator.alloc(u8, packed_len);
    try sme.packDensePanel64(.f16, row_major, n_rows, k_cols, packed_buf);
    return packed_buf;
}

/// CPU FP32 reference helpers for exact step-by-step numerical parity checks.
fn cpuRmsNorm(x: []const f32, w_bytes: []const u8, w_dt: quant.FloatDType, eps: f32, out: []f32) void {
    var ss: f32 = 0.0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (0..x.len) |i| {
        out[i] = x[i] * inv * readFloatWeightAt(w_bytes, w_dt, i);
    }
}

fn cpuLinearF16(x: []const f32, w_bytes: []const u8, w_dt: quant.FloatDType, n: usize, k: usize, out: []f32) void {
    for (0..n) |row| {
        var acc: f32 = 0.0;
        for (0..k) |col| {
            acc += x[col] * readFloatWeightAt(w_bytes, w_dt, row * k + col);
        }
        out[row] = acc;
    }
}

fn cpuSilu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

fn cpuSigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn fillDeterministicQ4(
    allocator: std.mem.Allocator,
    seed: u64,
    n: usize,
    k: usize,
    amp: f32,
) !struct { bytes: []u8, data_bytes: u64, params_offset: u64, params_bytes: u64 } {
    const groups_per_row = k / 64;
    const q_bytes = n * (k / 2);
    const p_off: usize = amf.align64(q_bytes);
    const p_bytes = n * groups_per_row * 4;
    const out = try allocator.alloc(u8, p_off + p_bytes);
    @memset(out, 0);
    // Fast deterministic SplitMix64 fill for nibbles so large `>= 50 MiB` bench tensors build in milliseconds.
    var s: u64 = seed ^ 0x9e3779b97f4a7c15;
    const words = std.mem.bytesAsSlice(u64, out[0 .. (q_bytes / 8) * 8]);
    for (words) |*w| {
        s +%= 0x9e3779b97f4a7c15;
        var z = s;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        w.* = z ^ (z >> 31);
    }
    const rem_start = words.len * 8;
    for (rem_start..q_bytes) |i| {
        s +%= 0x9e3779b97f4a7c15;
        out[i] = @truncate(s >> 24);
    }
    const scale_f16: u16 = @bitCast(@as(f16, @floatCast(amp / 7.5)));
    const bias_f16: u16 = @bitCast(@as(f16, @floatCast(-amp)));
    var pi: usize = p_off;
    for (0..n * groups_per_row) |_| {
        std.mem.writeInt(u16, out[pi..][0..2], scale_f16, .little);
        std.mem.writeInt(u16, out[pi + 2 ..][0..2], bias_f16, .little);
        pi += 4;
    }
    return .{
        .bytes = out,
        .data_bytes = @intCast(q_bytes),
        .params_offset = @intCast(p_off),
        .params_bytes = @intCast(p_bytes),
    };
}

pub const SynthQwen4BenchSpec = struct {
    hidden_size: u32 = 64,
    vocab_size: u32 = 128,
    num_attention_heads: u32 = 2,
    num_key_value_heads: u32 = 1,
    head_dim: u32 = 64,
    gdn_kh: u32 = 2,
    gdn_vh: u32 = 2,
    gdn_kd: u32 = 32,
    gdn_vd: u32 = 32,
    moe_inter: u32 = 64,
    shared_inter: u32 = 64,
    num_experts: u32 = 8,
    top_k: u32 = 2,
    cold_experts_per_layer: u32 = 0,
    max_loops: u32 = 1,
    ple_num_shards: u32 = 0,
    ple_rows: u64 = 64,
    use_q4_experts: bool = false,
    tmp_dir: ?[]const u8 = null,
};

pub const SynthQwen4Bundle = struct {
    cfg: GraphConfig,
    amf_bytes: []u8,
    ple_paths: [][:0]u8,
    amf_file_path: ?[:0]u8,

    pub fn deinit(self: *SynthQwen4Bundle, allocator: std.mem.Allocator) void {
        self.cfg.deinit(allocator);
        allocator.free(self.amf_bytes);
        for (self.ple_paths) |p| allocator.free(p);
        allocator.free(self.ple_paths);
        if (self.amf_file_path) |p| allocator.free(p);
    }
};

pub fn buildSynthQwen4Bundle(
    allocator: std.mem.Allocator,
    spec: SynthQwen4BenchSpec,
) !SynthQwen4Bundle {
    const h: u32 = spec.hidden_size;
    const vocab: u32 = spec.vocab_size;
    const n_layers: u32 = 4; // 3:1 GDN:QSA (L0,L1,L2 = GDN, L3 = QSA)
    const n_heads: u32 = spec.num_attention_heads;
    const n_kv: u32 = spec.num_key_value_heads;
    const head_dim: u32 = spec.head_dim;
    const gdn_kh: u32 = spec.gdn_kh;
    const gdn_vh: u32 = spec.gdn_vh;
    const gdn_kd: u32 = spec.gdn_kd;
    const gdn_vd: u32 = spec.gdn_vd;
    const moe_inter: u32 = spec.moe_inter;
    const sh_inter: u32 = spec.shared_inter;
    const ple_rows: u64 = spec.ple_rows;
    const num_experts = spec.num_experts;
    const top_k = spec.top_k;
    const cold_experts_per_layer = spec.cold_experts_per_layer;
    const max_loops = spec.max_loops;
    const ple_num_shards = spec.ple_num_shards;
    const tmp_dir = spec.tmp_dir;

    const layer_attns = try allocator.alloc(AttentionKind, n_layers);
    layer_attns[0] = .gated_deltanet;
    layer_attns[1] = .gated_deltanet;
    layer_attns[2] = .gated_deltanet;
    layer_attns[3] = .qsa_sparse;

    const cfg: GraphConfig = .{
        .axes = .{
            .backbone_kind = .decoder_only,
            .layer_attention_kinds = layer_attns,
            .ffn_kind = .moe_swiglu,
            .has_shared_expert = true,
            .residual_kind = .gated_4branch,
            .residual_branches = 4,
            .memory_lookup_kind = if (ple_num_shards > 0) .ple_ngram else .none,
            .loop_schedule = .{
                .mode = if (max_loops > 1) .native_recurrent else .single_pass,
                .start_layer = if (max_loops > 1) 1 else 0,
                .end_layer = n_layers,
                .max_loops = max_loops,
                .inject_x_init = (max_loops > 1),
            },
        },
        .vocab_size = vocab,
        .hidden_size = h,
        .num_hidden_layers = n_layers,
        .num_attention_heads = n_heads,
        .num_key_value_heads = n_kv,
        .head_dim = head_dim,
        .moe_intermediate_size = moe_inter,
        .shared_expert_intermediate_size = sh_inter,
        .num_experts = num_experts,
        .num_experts_per_tok = top_k,
        .norm_topk_prob = true,
        .linear_num_key_heads = gdn_kh,
        .linear_num_value_heads = gdn_vh,
        .linear_key_head_dim = gdn_kd,
        .linear_value_head_dim = gdn_vd,
        .linear_conv_kernel_dim = 4,
        .qsa_block_size = 32,
        .qsa_top_k_blocks = 4,
        .qsa_window_size = 128,
        .yarn = .{ .factor = 2.0, .original_max_pos = 1024 },
        .mrope_section = hybrid.MropeSection.QWEN38_DEFAULT,
        .ple_num_rows = ple_rows,
        .ple_embed_dim = h,
        .ple_ngram_order = 2,
        .ple_num_shards = ple_num_shards,
    };

    var blobs: std.ArrayListUnmanaged(TestTensorBlob) = .empty;
    var owned_names: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (blobs.items) |b| allocator.free(b.bytes);
        blobs.deinit(allocator);
        for (owned_names.items) |n| allocator.free(n);
        owned_names.deinit(allocator);
    }

    const addBlob = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayListUnmanaged(TestTensorBlob),
            names: *std.ArrayListUnmanaged([]u8),
            nm: []const u8,
            kind: amf.TensorKind,
            dt: amf.DType,
            layer: u16,
            expert_id: u16,
            shape: [4]u32,
            ndim: u8,
            data: []u8,
            p_off: u64,
            p_bytes: u64,
        ) !void {
            const dup = try alloc.dupe(u8, nm);
            try names.append(alloc, dup);
            try list.append(alloc, .{
                .name = dup,
                .kind = kind,
                .dtype = dt,
                .layer = layer,
                .expert = expert_id,
                .shape = shape,
                .ndim = ndim,
                .bytes = data,
                .data_bytes_override = if (p_bytes > 0) @as(u64, shape[0]) * (@as(u64, shape[1]) / 2) else 0,
                .params_offset = p_off,
                .params_bytes = p_bytes,
            });
        }
    }.call;

    try addBlob(allocator, &blobs, &owned_names, "embed_tokens", .embedding, .f16, amf.NO_LAYER, amf.NO_EXPERT, .{ vocab, h, 0, 0 }, 2, try fillDeterministicF16(allocator, 101, vocab * h, 0.2), 0, 0);

    var nbuf: [160]u8 = undefined;
    for (0..n_layers) |l| {
        const sbase: u64 = 1000 + l * 100;
        const ln_in = try std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l});
        try addBlob(allocator, &blobs, &owned_names, ln_in, .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);
        const ln_po = try std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l});
        try addBlob(allocator, &blobs, &owned_names, ln_po, .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);

        const gr_r = try std.fmt.bufPrint(&nbuf, "layers.{d}.gr.read_gate.weight", .{l});
        try addBlob(allocator, &blobs, &owned_names, gr_r, .dense, .bf16, @intCast(l), amf.NO_EXPERT, .{ 4, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 1, 4 * h, 0.1), 0, 0);
        const gr_w = try std.fmt.bufPrint(&nbuf, "layers.{d}.gr.write_gate.weight", .{l});
        try addBlob(allocator, &blobs, &owned_names, gr_w, .dense, .bf16, @intCast(l), amf.NO_EXPERT, .{ 4, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 2, 4 * h, 0.1), 0, 0);
        const gr_d = try std.fmt.bufPrint(&nbuf, "layers.{d}.gr.decay_gate.weight", .{l});
        try addBlob(allocator, &blobs, &owned_names, gr_d, .dense, .bf16, @intCast(l), amf.NO_EXPERT, .{ 4, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 3, 4 * h, 0.1), 0, 0);

        if (layer_attns[l] == .gated_deltanet) {
            const qk = gdn_kh * gdn_kd;
            const vd = gdn_vh * gdn_vd;
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.q_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ qk, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 10, qk * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.k_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ qk, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 11, qk * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.v_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ vd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 12, vd * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.g_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ vd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 13, vd * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.o_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ h, vd, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 14, h * vd, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.a_proj.weight", .{l}), .dense, .bf16, @intCast(l), amf.NO_EXPERT, .{ gdn_vh, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 15, gdn_vh * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.b_proj.weight", .{l}), .dense, .bf16, @intCast(l), amf.NO_EXPERT, .{ gdn_vh, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 16, gdn_vh * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.A_log", .{l}), .dense, .f32, @intCast(l), amf.NO_EXPERT, .{ gdn_vh, 0, 0, 0 }, 1, try fillDeterministicF32(allocator, sbase + 17, gdn_vh, -1.0, 0.0), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.dt_bias", .{l}), .dense, .f32, @intCast(l), amf.NO_EXPERT, .{ gdn_vh, 0, 0, 0 }, 1, try fillDeterministicF32(allocator, sbase + 18, gdn_vh, -0.2, 0.2), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.conv1d.weight", .{l}), .dense, .f32, @intCast(l), amf.NO_EXPERT, .{ 2 * qk + vd, 4, 0, 0 }, 2, try fillDeterministicF32(allocator, sbase + 19, (2 * qk + vd) * 4, -0.25, 0.25), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.norm.weight", .{l}), .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ gdn_vd, 0, 0, 0 }, 1, try fillConstantBf16(allocator, gdn_vd, 1.0), 0, 0);
        } else {
            const qd = n_heads * head_dim;
            const kvd = n_kv * head_dim;
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ qd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 20, qd * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ kvd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 21, kvd * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ kvd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 22, kvd * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ h, qd, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 23, h * qd, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}), .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ head_dim, 0, 0, 0 }, 1, try fillConstantBf16(allocator, head_dim, 1.0), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}), .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ head_dim, 0, 0, 0 }, 1, try fillConstantBf16(allocator, head_dim, 1.0), 0, 0);
        }

        // Router + Shared Expert + E Routed Experts
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate.weight", .{l}), .router, .bf16, @intCast(l), amf.NO_EXPERT, .{ num_experts, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 30, num_experts * h, 0.15), 0, 0);
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.shared_expert.gate_proj", .{l}), .shared_expert, .f16, @intCast(l), amf.NO_EXPERT, .{ sh_inter, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 31, sh_inter * h, 0.06), 0, 0);
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.shared_expert.up_proj", .{l}), .shared_expert, .f16, @intCast(l), amf.NO_EXPERT, .{ sh_inter, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 32, sh_inter * h, 0.06), 0, 0);
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.shared_expert.down_proj", .{l}), .shared_expert, .f16, @intCast(l), amf.NO_EXPERT, .{ h, sh_inter, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 33, h * sh_inter, 0.06), 0, 0);
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.shared_expert_gate.weight", .{l}), .shared_expert, .bf16, @intCast(l), amf.NO_EXPERT, .{ 1, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 34, h, 0.1), 0, 0);

        for (0..num_experts) |e| {
            const is_cold = (cold_experts_per_layer > 0 and e >= num_experts - cold_experts_per_layer);
            const ekind: amf.TensorKind = if (is_cold) .cold_expert else .hot_expert;
            const eseed = sbase * 1000 + e * 7;
            if (spec.use_q4_experts) {
                const gq = try fillDeterministicQ4(allocator, eseed + 1, moe_inter, h, 0.05);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }), ekind, .q4_affine, @intCast(l), @intCast(e), .{ moe_inter, h, 0, 0 }, 2, gq.bytes, gq.params_offset, gq.params_bytes);
                const uq = try fillDeterministicQ4(allocator, eseed + 2, moe_inter, h, 0.05);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }), ekind, .q4_affine, @intCast(l), @intCast(e), .{ moe_inter, h, 0, 0 }, 2, uq.bytes, uq.params_offset, uq.params_bytes);
                const dq = try fillDeterministicQ4(allocator, eseed + 3, h, moe_inter, 0.05);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }), ekind, .q4_affine, @intCast(l), @intCast(e), .{ h, moe_inter, 0, 0 }, 2, dq.bytes, dq.params_offset, dq.params_bytes);
            } else {
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }), ekind, .f16, @intCast(l), @intCast(e), .{ moe_inter, h, 0, 0 }, 2, try fillDeterministicF16(allocator, eseed + 1, moe_inter * h, 0.05), 0, 0);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }), ekind, .f16, @intCast(l), @intCast(e), .{ moe_inter, h, 0, 0 }, 2, try fillDeterministicF16(allocator, eseed + 2, moe_inter * h, 0.05), 0, 0);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }), ekind, .f16, @intCast(l), @intCast(e), .{ h, moe_inter, 0, 0 }, 2, try fillDeterministicF16(allocator, eseed + 3, h * moe_inter, 0.05), 0, 0);
            }
        }
    }

    try addBlob(allocator, &blobs, &owned_names, "norm.weight", .norm, .bf16, amf.NO_LAYER, amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);
    try addBlob(allocator, &blobs, &owned_names, "lm_head", .lm_head, .f16, amf.NO_LAYER, amf.NO_EXPERT, .{ vocab, h, 0, 0 }, 2, try fillDeterministicF16(allocator, 999, vocab * h, 0.12), 0, 0);

    const amf_bytes = try buildTestAmf(allocator, "test-qwen4-exp", blobs.items);

    var amf_file_path: ?[:0]u8 = null;
    if (tmp_dir) |td| {
        const p = try std.fmt.allocPrintSentinel(allocator, "{s}/model.amf", .{td}, 0);
        try writeBytesFile(p, amf_bytes);
        amf_file_path = p;
    }

    var ple_paths = try allocator.alloc([:0]u8, ple_num_shards);
    if (ple_num_shards > 0) {
        const td = tmp_dir.?;
        const rows_per_shard = ple_rows / ple_num_shards;
        for (0..ple_num_shards) |s| {
            var pblobs: std.ArrayListUnmanaged(TestTensorBlob) = .empty;
            var pnames: std.ArrayListUnmanaged([]u8) = .empty;
            defer {
                for (pblobs.items) |b| allocator.free(b.bytes);
                pblobs.deinit(allocator);
                for (pnames.items) |n| allocator.free(n);
                pnames.deinit(allocator);
            }
            for (0..n_layers) |l| {
                const pnm = if (ple_num_shards == 1)
                    try std.fmt.bufPrint(&nbuf, "ple.layers.{d}.weight", .{l})
                else
                    try std.fmt.bufPrint(&nbuf, "ple.layers.{d}.shard.{d}.weight", .{ l, s });
                try addBlob(allocator, &pblobs, &pnames, pnm, .sparse_table, .bf16, @intCast(l), amf.NO_EXPERT, .{ @intCast(rows_per_shard), h, 0, 0 }, 2, try fillDeterministicBf16(allocator, 77700 + l * 17 + s, @intCast(rows_per_shard * h), 0.08), 0, 0);
            }
            const pbytes = try buildTestAmf(allocator, "test-qwen4-ple", pblobs.items);
            defer allocator.free(pbytes);
            const pp = try std.fmt.allocPrintSentinel(allocator, "{s}/ple_{d}.amf", .{ td, s }, 0);
            try writeBytesFile(pp, pbytes);
            ple_paths[s] = pp;
        }
    }

    return .{ .cfg = cfg, .amf_bytes = amf_bytes, .ple_paths = ple_paths, .amf_file_path = amf_file_path };
}

fn buildSynthQwen4TestBundle(
    allocator: std.mem.Allocator,
    num_experts: u32,
    top_k: u32,
    cold_experts_per_layer: u32,
    max_loops: u32,
    ple_num_shards: u32,
    tmp_dir: ?[]const u8,
) !SynthQwen4Bundle {
    return buildSynthQwen4Bundle(allocator, .{
        .num_experts = num_experts,
        .top_k = top_k,
        .cold_experts_per_layer = cold_experts_per_layer,
        .max_loops = max_loops,
        .ple_num_shards = ple_num_shards,
        .tmp_dir = tmp_dir,
    });
}

pub const SynthLoopDenseBenchSpec = struct {
    hidden_size: u32 = 64,
    vocab_size: u32 = 128,
    num_layers: u32 = 4,
    num_attention_heads: u32 = 2,
    num_key_value_heads: u32 = 1,
    head_dim: u32 = 64,
    intermediate_size: u32 = 128,
    start_layer: u32 = 1,
    end_layer: u32 = 3,
    mode: LoopMode = .native_recurrent,
    max_loops: u32 = 3,
    use_q4: bool = false,
    include_early_exit: bool = false,
    include_steplora: bool = false,
};

pub const SynthLoopDenseBundle = struct {
    cfg: GraphConfig,
    amf_bytes: []u8,

    pub fn deinit(self: *SynthLoopDenseBundle, allocator: std.mem.Allocator) void {
        self.cfg.deinit(allocator);
        allocator.free(self.amf_bytes);
    }
};

pub fn buildSynthLoopDenseBundle(
    allocator: std.mem.Allocator,
    spec: SynthLoopDenseBenchSpec,
) !SynthLoopDenseBundle {
    const h: u32 = spec.hidden_size;
    const vocab: u32 = spec.vocab_size;
    const n_layers: u32 = spec.num_layers;
    const n_heads: u32 = spec.num_attention_heads;
    const n_kv: u32 = spec.num_key_value_heads;
    const head_dim: u32 = spec.head_dim;
    const inter: u32 = spec.intermediate_size;
    const start_l: u32 = spec.start_layer;
    const end_l: u32 = spec.end_layer;
    const mode = spec.mode;
    const max_loops = spec.max_loops;
    const include_early_exit = spec.include_early_exit;
    const include_steplora = spec.include_steplora;

    const layer_attns = try allocator.alloc(AttentionKind, n_layers);
    @memset(layer_attns, .gqa_full);

    const cfg: GraphConfig = .{
        .axes = .{
            .backbone_kind = .decoder_only,
            .layer_attention_kinds = layer_attns,
            .ffn_kind = .dense_swiglu,
            .has_shared_expert = false,
            .residual_kind = .single_stream,
            .residual_branches = 1,
            .memory_lookup_kind = .none,
            .loop_schedule = .{
                .mode = mode,
                .start_layer = start_l,
                .end_layer = end_l,
                .max_loops = max_loops,
                .inject_x_init = (mode == .native_recurrent),
                .adapter_kind = if (include_steplora) .sme2_step_lora else .none,
                .adapter_rank = if (include_steplora) 16 else 0,
                .adapter_packed_rank = if (include_steplora) 64 else 0,
                .enforce_zero_alpha_r0 = true,
                .early_exit_gate = if (include_early_exit) .sigmoid_linear else .none,
                .early_exit_threshold = 0.5,
            },
        },
        .vocab_size = vocab,
        .hidden_size = h,
        .num_hidden_layers = n_layers,
        .num_attention_heads = n_heads,
        .num_key_value_heads = n_kv,
        .head_dim = head_dim,
        .intermediate_size = inter,
        .rope_theta = 10_000.0,
    };

    var blobs: std.ArrayListUnmanaged(TestTensorBlob) = .empty;
    var owned_names: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (blobs.items) |b| allocator.free(b.bytes);
        blobs.deinit(allocator);
        for (owned_names.items) |n| allocator.free(n);
        owned_names.deinit(allocator);
    }

    const addB = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayListUnmanaged(TestTensorBlob),
            names: *std.ArrayListUnmanaged([]u8),
            nm: []const u8,
            kind: amf.TensorKind,
            dt: amf.DType,
            layout: amf.TensorLayout,
            layer: u16,
            shape: [4]u32,
            ndim: u8,
            data: []u8,
            p_off: u64,
            p_bytes: u64,
        ) !void {
            const dup = try alloc.dupe(u8, nm);
            try names.append(alloc, dup);
            try list.append(alloc, .{
                .name = dup,
                .kind = kind,
                .dtype = dt,
                .layout = layout,
                .layer = layer,
                .expert = amf.NO_EXPERT,
                .shape = shape,
                .ndim = ndim,
                .bytes = data,
                .data_bytes_override = if (p_bytes > 0) @as(u64, shape[0]) * (@as(u64, shape[1]) / 2) else 0,
                .params_offset = p_off,
                .params_bytes = p_bytes,
            });
        }
    }.call;

    const addLin = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayListUnmanaged(TestTensorBlob),
            names: *std.ArrayListUnmanaged([]u8),
            nm: []const u8,
            kind: amf.TensorKind,
            layer: u16,
            n_out: u32,
            k_in: u32,
            seed: u64,
            amp: f32,
            use_q4: bool,
        ) !void {
            if (use_q4) {
                const q = try fillDeterministicQ4(alloc, seed, n_out, k_in, amp);
                try addB(alloc, list, names, nm, kind, .q4_affine, .row_major, layer, .{ n_out, k_in, 0, 0 }, 2, q.bytes, q.params_offset, q.params_bytes);
            } else {
                try addB(alloc, list, names, nm, kind, .f16, .row_major, layer, .{ n_out, k_in, 0, 0 }, 2, try fillDeterministicF16(alloc, seed, @as(usize, n_out) * k_in, amp), 0, 0);
            }
        }
    }.call;

    try addB(allocator, &blobs, &owned_names, "embed_tokens", .embedding, .f16, .row_major, amf.NO_LAYER, .{ vocab, h, 0, 0 }, 2, try fillDeterministicF16(allocator, 501, vocab * h, 0.25), 0, 0);

    var nbuf: [160]u8 = undefined;
    for (0..n_layers) |l| {
        const sb: u64 = 5000 + l * 50;
        const qd = n_heads * head_dim;
        const kvd = n_kv * head_dim;
        try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l}), .norm, .bf16, .row_major, @intCast(l), .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj", .{l}), .dense, @intCast(l), qd, h, sb + 1, 0.08, spec.use_q4);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj", .{l}), .dense, @intCast(l), kvd, h, sb + 2, 0.08, spec.use_q4);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj", .{l}), .dense, @intCast(l), kvd, h, sb + 3, 0.08, spec.use_q4);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj", .{l}), .dense, @intCast(l), h, qd, sb + 4, 0.08, spec.use_q4);
        try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}), .norm, .bf16, .row_major, @intCast(l), .{ head_dim, 0, 0, 0 }, 1, try fillConstantBf16(allocator, head_dim, 1.0), 0, 0);
        try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}), .norm, .bf16, .row_major, @intCast(l), .{ head_dim, 0, 0, 0 }, 1, try fillConstantBf16(allocator, head_dim, 1.0), 0, 0);
        try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l}), .norm, .bf16, .row_major, @intCast(l), .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj", .{l}), .dense, @intCast(l), inter, h, sb + 5, 0.08, spec.use_q4);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj", .{l}), .dense, @intCast(l), inter, h, sb + 6, 0.08, spec.use_q4);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj", .{l}), .dense, @intCast(l), h, inter, sb + 7, 0.08, spec.use_q4);
    }

    try addB(allocator, &blobs, &owned_names, "norm.weight", .norm, .bf16, .row_major, amf.NO_LAYER, .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);
    try addB(allocator, &blobs, &owned_names, "lm_head", .lm_head, .f16, .row_major, amf.NO_LAYER, .{ vocab, h, 0, 0 }, 2, try fillDeterministicF16(allocator, 599, vocab * h, 0.15), 0, 0);

    if (include_early_exit) {
        try addB(allocator, &blobs, &owned_names, "loop.early_exit_gate.weight", .dense, .bf16, .row_major, amf.NO_LAYER, .{ 1, h, 0, 0 }, 2, try fillConstantBf16(allocator, h, 0.5), 0, 0);
    }

    if (include_steplora) {
        const qd = n_heads * head_dim;
        for (0..max_loops) |step| {
            const alpha_val: f32 = if (step == 0) 0.0 else 0.15 * @as(f32, @floatFromInt(step));
            for (start_l..end_l) |l| {
                const sd: u64 = 9000 + step * 300 + l * 30;
                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.q_proj.lora_a", .{ step, l }), .loop_adapter, .f16, .sme2_panel64, @intCast(l), .{ 64, h, 0, 0 }, 2, try fillDeterministicStepLoraSme2F16(allocator, sd + 1, 64, h, 16, h, 0.08), 0, 0);
                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.q_proj.lora_b", .{ step, l }), .loop_adapter, .f16, .sme2_panel64, @intCast(l), .{ qd, 64, 0, 0 }, 2, try fillDeterministicStepLoraSme2F16(allocator, sd + 2, qd, 64, qd, 16, 0.08), 0, 0);
                const al_q = try allocator.alloc(u8, 4);
                std.mem.writeInt(u32, al_q[0..4], @bitCast(alpha_val), .little);
                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.q_proj.alpha", .{ step, l }), .loop_adapter, .f32, .row_major, @intCast(l), .{ 1, 0, 0, 0 }, 1, al_q, 0, 0);

                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.o_proj.lora_a", .{ step, l }), .loop_adapter, .f16, .sme2_panel64, @intCast(l), .{ 64, qd, 0, 0 }, 2, try fillDeterministicStepLoraSme2F16(allocator, sd + 3, 64, qd, 16, qd, 0.08), 0, 0);
                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.o_proj.lora_b", .{ step, l }), .loop_adapter, .f16, .sme2_panel64, @intCast(l), .{ h, 64, 0, 0 }, 2, try fillDeterministicStepLoraSme2F16(allocator, sd + 4, h, 64, h, 16, 0.08), 0, 0);
                const al_o = try allocator.alloc(u8, 4);
                std.mem.writeInt(u32, al_o[0..4], @bitCast(alpha_val), .little);
                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.o_proj.alpha", .{ step, l }), .loop_adapter, .f32, .row_major, @intCast(l), .{ 1, 0, 0, 0 }, 1, al_o, 0, 0);
            }
        }
    }

    const amf_bytes = try buildTestAmf(allocator, "test-loop-dense", blobs.items);
    return .{ .cfg = cfg, .amf_bytes = amf_bytes };
}

fn buildSynthLoopDenseTestBundle(
    allocator: std.mem.Allocator,
    mode: LoopMode,
    max_loops: u32,
    include_early_exit: bool,
    include_steplora: bool,
) !SynthLoopDenseBundle {
    return buildSynthLoopDenseBundle(allocator, .{
        .mode = mode,
        .max_loops = max_loops,
        .include_early_exit = include_early_exit,
        .include_steplora = include_steplora,
    });
}

fn writeBytesFile(path: [:0]const u8, data: []const u8) !void {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.IoFailed;
    defer _ = std.c.close(fd);
    var written: usize = 0;
    while (written < data.len) {
        const rc = std.c.write(fd, data.ptr + written, data.len - written);
        if (rc <= 0) return error.IoFailed;
        written += @intCast(rc);
    }
}

fn makeTestTempDir(allocator: std.mem.Allocator, tag: []const u8) ![]u8 {
    const base = std.c.getenv("TMPDIR");
    const root_slice: []const u8 = if (base) |p| std.mem.span(p) else "/Volumes/Akari/.tmp/m6-engine";
    const dir_path = try std.fmt.allocPrint(allocator, "{s}/graph_test_{s}_{d}", .{ root_slice, tag, std.c.getpid() });
    errdefer allocator.free(dir_path);
    const dir_z = try std.fmt.allocPrintSentinel(allocator, "{s}", .{dir_path}, 0);
    defer allocator.free(dir_z);
    _ = std.c.mkdir(dir_z.ptr, 0o755);
    return dir_path;
}

fn removeTestFile(path: [:0]const u8) void {
    _ = std.c.unlink(path.ptr);
}

fn removeTestDir(allocator: std.mem.Allocator, path: []const u8) void {
    const z = std.fmt.allocPrintSentinel(allocator, "{s}", .{path}, 0) catch return;
    defer allocator.free(z);
    _ = std.c.rmdir(z.ptr);
}

/// CPU FP32 reference forward pass for dense `gqa_full` + `LoopLM` (`R = 1..3`)
/// verifying exact numerical agreement with `GraphModel`.
fn evalLoopDenseCpuRef(
    allocator: std.mem.Allocator,
    amf_bytes: []const u8,
    cfg: GraphConfig,
    tokens: []const u32,
    num_loops: u32,
) ![]f32 {
    var parsed = try amf.Parsed.parseBytes(allocator, amf_bytes);
    defer parsed.deinit(allocator);
    const blobOf = struct {
        fn call(p: *const amf.Parsed, raw: []const u8, name: []const u8) []const u8 {
            const e = p.find(name).?;
            return raw[@intCast(e.blob_offset)..][0..@intCast(e.blob_bytes)];
        }
    }.call;

    const h: usize = cfg.hidden_size;
    const qd: usize = cfg.qDim();
    const kvd: usize = cfg.kvDim();
    const inter: usize = cfg.intermediate_size;
    const vocab: usize = cfg.vocab_size;
    const n_layers: usize = cfg.num_hidden_layers;
    const max_t = tokens.len;

    const k_cache = try allocator.alloc(f32, n_layers * max_t * kvd);
    defer allocator.free(k_cache);
    const v_cache = try allocator.alloc(f32, n_layers * max_t * kvd);
    defer allocator.free(v_cache);
    @memset(k_cache, 0.0);
    @memset(v_cache, 0.0);

    const x = try allocator.alloc(f32, h);
    defer allocator.free(x);
    const x_init = try allocator.alloc(f32, h);
    defer allocator.free(x_init);
    const xn = try allocator.alloc(f32, h);
    defer allocator.free(xn);
    const q = try allocator.alloc(f32, qd);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, kvd);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, kvd);
    defer allocator.free(v);
    const attn_out = try allocator.alloc(f32, qd);
    defer allocator.free(attn_out);
    const proj = try allocator.alloc(f32, h);
    defer allocator.free(proj);
    const gbuf = try allocator.alloc(f32, inter);
    defer allocator.free(gbuf);
    const ubuf = try allocator.alloc(f32, inter);
    defer allocator.free(ubuf);
    const logits = try allocator.alloc(f32, vocab);

    const ls = cfg.axes.loop_schedule;
    const core_start: usize = if (num_loops > 1) ls.start_layer else 0;
    const core_end: usize = if (num_loops > 1) ls.end_layer else n_layers;

    const emb_raw = blobOf(&parsed, amf_bytes, "embed_tokens");
    var nbuf: [160]u8 = undefined;

    for (tokens, 0..) |tid, pos| {
        for (0..h) |d| x[d] = readFloatWeightAt(emb_raw, .f16, @as(usize, tid) * h + d);

        const runLayer = struct {
            fn call(
                p: *const amf.Parsed,
                raw: []const u8,
                c: GraphConfig,
                l: usize,
                pos_i: usize,
                commit_kv: bool,
                kc: []f32,
                vc: []f32,
                xv: []f32,
                xnv: []f32,
                qv: []f32,
                kv: []f32,
                vv: []f32,
                aov: []f32,
                pv: []f32,
                gv: []f32,
                uv: []f32,
                nb: *[160]u8,
            ) void {
                const hh: usize = c.hidden_size;
                const qdd: usize = c.qDim();
                const kvdd: usize = c.kvDim();
                const hdd: usize = c.head_dim;
                const nhh: usize = c.num_attention_heads;
                const nkvv: usize = c.num_key_value_heads;
                const ii: usize = c.intermediate_size;
                const mt: usize = kc.len / (c.num_hidden_layers * kvdd);

                cpuRmsNorm(xv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable), .bf16, c.rms_norm_eps, xnv);
                cpuLinearF16(xnv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable), .f16, qdd, hh, qv);
                if (commit_kv) {
                    cpuLinearF16(xnv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable), .f16, kvdd, hh, kv);
                    cpuLinearF16(xnv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable), .f16, kvdd, hh, vv);
                }

                const qnw = blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable);
                const knw = blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable);
                const half_hd = hdd / 2;
                for (0..nhh) |hd_i| {
                    const seg = qv[hd_i * hdd ..][0..hdd];
                    var tmp: [256]f32 = undefined;
                    cpuRmsNorm(seg, qnw, .bf16, c.rms_norm_eps, tmp[0..hdd]);
                    for (0..half_hd) |j| {
                        const inv_f = std.math.pow(f32, c.rope_theta, -2.0 * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(hdd)));
                        const ang = @as(f32, @floatFromInt(pos_i)) * inv_f;
                        const cs = @cos(ang);
                        const sn = @sin(ang);
                        const a = tmp[j];
                        const b = tmp[j + half_hd];
                        seg[j] = a * cs - b * sn;
                        seg[j + half_hd] = a * sn + b * cs;
                    }
                }
                if (commit_kv) {
                    for (0..nkvv) |hd_i| {
                        const seg = kv[hd_i * hdd ..][0..hdd];
                        var tmp: [256]f32 = undefined;
                        cpuRmsNorm(seg, knw, .bf16, c.rms_norm_eps, tmp[0..hdd]);
                        for (0..half_hd) |j| {
                            const inv_f = std.math.pow(f32, c.rope_theta, -2.0 * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(hdd)));
                            const ang = @as(f32, @floatFromInt(pos_i)) * inv_f;
                            const cs = @cos(ang);
                            const sn = @sin(ang);
                            const a = tmp[j];
                            const b = tmp[j + half_hd];
                            seg[j] = a * cs - b * sn;
                            seg[j + half_hd] = a * sn + b * cs;
                        }
                    }
                    @memcpy(kc[(l * mt + pos_i) * kvdd ..][0..kvdd], kv);
                    @memcpy(vc[(l * mt + pos_i) * kvdd ..][0..kvdd], vv);
                }

                // Causal GQA attending over committed `0..=pos_i` (using r=0 shared KV at `pos_i`).
                const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hdd)));
                const group = nhh / nkvv;
                for (0..nhh) |qh| {
                    const kvh = qh / group;
                    var max_s: f32 = -std.math.inf(f32);
                    var scores: [64]f32 = undefined;
                    for (0..pos_i + 1) |t| {
                        const k_src = kc[(l * mt + t) * kvdd + kvh * hdd ..][0..hdd];
                        var dot: f32 = 0.0;
                        for (0..hdd) |d| dot += qv[qh * hdd + d] * k_src[d];
                        scores[t] = dot * scale;
                        if (scores[t] > max_s) max_s = scores[t];
                    }
                    var sum_e: f32 = 0.0;
                    for (0..pos_i + 1) |t| {
                        scores[t] = @exp(scores[t] - max_s);
                        sum_e += scores[t];
                    }
                    for (0..hdd) |d| {
                        var acc: f32 = 0.0;
                        for (0..pos_i + 1) |t| {
                            const v_src = vc[(l * mt + t) * kvdd + kvh * hdd ..][0..hdd];
                            acc += (scores[t] / sum_e) * v_src[d];
                        }
                        aov[qh * hdd + d] = acc;
                    }
                }

                cpuLinearF16(aov, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable), .f16, hh, qdd, pv);
                for (0..hh) |d| xv[d] += pv[d];

                cpuRmsNorm(xv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable), .bf16, c.rms_norm_eps, xnv);
                cpuLinearF16(xnv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable), .f16, ii, hh, gv);
                cpuLinearF16(xnv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.mlp.up_proj", .{l}) catch unreachable), .f16, ii, hh, uv);
                for (0..ii) |d| gv[d] = cpuSilu(gv[d]) * uv[d];
                cpuLinearF16(gv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.mlp.down_proj", .{l}) catch unreachable), .f16, hh, ii, pv);
                for (0..hh) |d| xv[d] += pv[d];
            }
        }.call;

        for (0..core_start) |l| runLayer(&parsed, amf_bytes, cfg, l, pos, true, k_cache, v_cache, x, xn, q, k, v, attn_out, proj, gbuf, ubuf, &nbuf);
        if (num_loops > 1 and ls.inject_x_init) @memcpy(x_init, x);
        for (0..num_loops) |r| {
            if (r >= 1 and ls.inject_x_init) {
                for (0..h) |d| x[d] = 0.5 * x[d] + 0.5 * x_init[d];
            }
            const is_r0 = (r == 0);
            for (core_start..core_end) |l| runLayer(&parsed, amf_bytes, cfg, l, pos, is_r0, k_cache, v_cache, x, xn, q, k, v, attn_out, proj, gbuf, ubuf, &nbuf);
        }
        for (core_end..n_layers) |l| runLayer(&parsed, amf_bytes, cfg, l, pos, true, k_cache, v_cache, x, xn, q, k, v, attn_out, proj, gbuf, ubuf, &nbuf);

        if (pos == tokens.len - 1) {
            cpuRmsNorm(x, blobOf(&parsed, amf_bytes, "norm.weight"), .bf16, cfg.rms_norm_eps, xn);
            cpuLinearF16(xn, blobOf(&parsed, amf_bytes, "lm_head"), .f16, vocab, h, logits);
        }
    }
    return logits;
}

test "W9+W10: tiny-qwen4 (3:1 GDN:QSA, 4-branch GR, shared MoE, PLE) prefill, decode & tree parity" {
    const allocator = std.testing.allocator;
    const tmp_dir = try makeTestTempDir(allocator, "tiny_qwen4");
    defer {
        removeTestDir(allocator, tmp_dir);
        allocator.free(tmp_dir);
    }

    var b = try buildSynthQwen4TestBundle(allocator, 16, 4, 0, 1, 1, tmp_dir);
    defer {
        b.cfg.deinit(allocator);
        allocator.free(b.amf_bytes);
        if (b.amf_file_path) |p| {
            removeTestFile(p);
            allocator.free(p);
        }
        for (b.ple_paths) |p| {
            removeTestFile(p);
            allocator.free(p);
        }
        allocator.free(b.ple_paths);
    }

    var ple_const = try allocator.alloc([:0]const u8, b.ple_paths.len);
    defer allocator.free(ple_const);
    for (b.ple_paths, 0..) |p, i| ple_const[i] = p;

    var serial_model = try GraphModel.loadFromBytes(allocator, b.amf_bytes, b.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .ple_paths = ple_const,
    });
    defer serial_model.deinit(allocator);

    var tree_model = try GraphModel.loadFromBytes(allocator, b.amf_bytes, b.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .ple_paths = ple_const,
    });
    defer tree_model.deinit(allocator);

    // Prefill prompt tokens on both models.
    const prompt = [_]u32{ 3, 17, 42, 7 };
    const s_next = try serial_model.prefillChunk(&prompt, 0, true);
    const t_next = try tree_model.prefillChunk(&prompt, 0, true);
    try std.testing.expectEqual(s_next, t_next);

    // Run 3 serial decode steps vs 1 batched verifyTree + commitTree on a 4-node branching tree!
    // Tree layout: 0 (root = s_next at pos 4) -> 1 (pos 5, child of 0) -> 2 (pos 6, child of 1);
    //              3 (pos 5, sibling child of 0)
    const d0 = try serial_model.stepDecode(s_next, 4);
    const d1 = try serial_model.stepDecode(d0, 5);
    const d2 = try serial_model.stepDecode(d1, 6);
    var serial_logits_last: [128]f16 = undefined;
    @memcpy(&serial_logits_last, serial_model.readLastLogitsF16());

    const tree_toks = [_]u32{ s_next, d0, d1, (d0 + 5) % 128 };
    const tree_pos = [_]u32{ 4, 5, 6, 5 };
    const tree_masks = [_]u64{ 0b0001, 0b0011, 0b0111, 0b1001 };
    const vr = try tree_model.verifyTree(&tree_toks, &tree_pos, &tree_masks, 4);
    try std.testing.expectEqual(d0, vr.argmax_tokens[0]);
    try std.testing.expectEqual(d1, vr.argmax_tokens[1]);
    try std.testing.expectEqual(d2, vr.argmax_tokens[2]);

    // Check tight numerical parity on node 2's full logit vector vs serial decode!
    const v_row = vr.logits_f16[2 * 128 .. 3 * 128];
    var max_err: f32 = 0.0;
    for (0..128) |i| {
        const diff = @abs(@as(f32, @floatCast(v_row[i])) - @as(f32, @floatCast(serial_logits_last[i])));
        if (diff > max_err) max_err = diff;
    }
    try std.testing.expect(max_err < 1e-2);

    // Commit accepted path [0, 1, 2] and verify subsequent decode matches serial!
    try tree_model.commitTree(4, &.{ 0, 1, 2 });
    const s_after = try serial_model.stepDecode(d2, 7);
    const t_after = try tree_model.stepDecode(d2, 7);
    try std.testing.expectEqual(s_after, t_after);
}

test "W10+G7+C10+ADR-0019: scale-qwen4-512e (E=512, top-10 + shared, 4-shard PLE, cold experts, R=3) 4 loop isolations" {
    const allocator = std.testing.allocator;
    const tmp_dir = try makeTestTempDir(allocator, "scale_512e_iso");
    defer {
        removeTestDir(allocator, tmp_dir);
        allocator.free(tmp_dir);
    }

    // E=512, top_k=10, 64 cold experts/layer, R=3 recurrent loops, 4-shard PLE!
    var b = try buildSynthQwen4TestBundle(allocator, 512, 10, 64, 3, 4, tmp_dir);
    defer {
        b.cfg.deinit(allocator);
        allocator.free(b.amf_bytes);
        if (b.amf_file_path) |p| {
            removeTestFile(p);
            allocator.free(p);
        }
        for (b.ple_paths) |p| {
            removeTestFile(p);
            allocator.free(p);
        }
        allocator.free(b.ple_paths);
    }

    var ple_const = try allocator.alloc([:0]const u8, b.ple_paths.len);
    defer allocator.free(ple_const);
    for (b.ple_paths, 0..) |p, i| ple_const[i] = p;

    var model = try GraphModel.loadFromFile(allocator, b.amf_file_path.?, b.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .ple_paths = ple_const,
        .cold_slot_count = 32,
    });
    defer model.deinit(allocator);

    try std.testing.expect(model.hasColdExperts());
    const prompt = [_]u32{ 5, 11, 23, 37 };
    const t0 = try model.prefillChunk(&prompt, 0, true);

    // Snapshot committed GDN state `S_t` before tree verify!
    const ga = &(model.gdn_arena.?);
    const s_before = try allocator.dupe(f32, try ga.committedLayerSlice(1));
    defer allocator.free(s_before);

    model.resetLoopIsolationStats();
    // 1. Run `verifyTree` (`M = 4`, `R = 3`): committed GDN `S_t` MUST be byte-exact untouched!
    const toks = [_]u32{ t0, (t0 + 1) % 128, (t0 + 2) % 128, (t0 + 3) % 128 };
    const pos = [_]u32{ 4, 5, 6, 5 };
    const masks = [_]u64{ 0b0001, 0b0011, 0b0111, 0b1001 };
    const vr = try model.verifyTree(&toks, &pos, &masks, 4);
    try std.testing.expectEqual(@as(u32, 4), vr.num_nodes);
    try std.testing.expect(vr.total_unique_experts > 0);

    const s_after_verify = try ga.committedLayerSlice(1);
    try std.testing.expectEqualSlices(f32, s_before, s_after_verify);

    // Check ALL FOUR ADR-0019 loop isolations on `R = 3`:
    const iso_verify = model.loopIsolationStats();
    // #1 GDN: scratch evals on r=0,1,2 (`0` commits during verifyTree)
    try std.testing.expect(iso_verify.gdn_scratch_evals >= 6);
    try std.testing.expectEqual(@as(u64, 0), iso_verify.gdn_final_commits);
    // #2 PLE: `r = 0` fills, `r >= 1` reuses with EXACTLY `0` extra NVMe reads!
    try std.testing.expect(iso_verify.ple_r0_fills > 0);
    try std.testing.expect(iso_verify.ple_r_ge1_reuses > 0);
    try std.testing.expectEqual(@as(u64, 0), iso_verify.ple_r_ge1_nvme_reads);
    // #3 QSA: `r = 0` computes indices once, `r >= 1` reuses indices (`reuse_block_indices = true`)!
    try std.testing.expectEqual(@as(u64, 1), iso_verify.qsa_index_computes);
    try std.testing.expectEqual(@as(u64, 2), iso_verify.qsa_index_reuses);
    // #4 MoE: `r >= 1` pinned-only routes with `0` cold NVMe stalls!
    try std.testing.expect(iso_verify.moe_r_ge1_pinned_routes > 0);
    try std.testing.expectEqual(@as(u64, 0), iso_verify.moe_r_ge1_cold_stalls);

    // 2. Run single-step `stepDecode` (`R = 3`): commits `S_tmp -> S_t` ONCE at `r = R - 1`!
    model.resetLoopIsolationStats();
    _ = try model.stepDecode(t0, 4);
    const iso_dec = model.loopIsolationStats();
    // 2 core GDN layers (L1, L2) x 2 intermediate steps (r=0,1) = 4 scratch evals,
    // plus L0 prelude (1) + L1,L2 final r=2 (2) = 3 final commits!
    try std.testing.expectEqual(@as(u64, 4), iso_dec.gdn_scratch_evals);
    try std.testing.expectEqual(@as(u64, 3), iso_dec.gdn_final_commits);
    try std.testing.expectEqual(@as(u64, 0), iso_dec.ple_r_ge1_nvme_reads);
    try std.testing.expectEqual(@as(u64, 1), iso_dec.qsa_index_computes);
    try std.testing.expectEqual(@as(u64, 2), iso_dec.qsa_index_reuses);
    try std.testing.expectEqual(@as(u64, 0), iso_dec.moe_r_ge1_cold_stalls);
}

test "W9+ADR-0019: native tiny-looplm (R=1..3, inject_x_init) CPU fp32 parity, tree parity & early-exit gate" {
    const allocator = std.testing.allocator;
    var b = try buildSynthLoopDenseTestBundle(allocator, .native_recurrent, 3, true, false);
    defer {
        b.cfg.deinit(allocator);
        allocator.free(b.amf_bytes);
    }

    const prompt = [_]u32{ 4, 19, 63 };
    for ([_]u32{ 1, 2, 3 }) |R| {
        const ref_logits = try evalLoopDenseCpuRef(allocator, b.amf_bytes, b.cfg, &prompt, R);
        defer allocator.free(ref_logits);

        var gpu_model = try GraphModel.loadFromBytes(allocator, b.amf_bytes, b.cfg, .{
            .max_seq_len = 128,
            .max_tree_nodes = 16,
            .max_loops_override = R,
            .enable_loop_early_exit = false,
        });
        defer gpu_model.deinit(allocator);

        var last_tok: u32 = 0;
        for (prompt, 0..) |tid, i| {
            last_tok = try gpu_model.stepDecode(tid, @intCast(i));
        }
        const gpu_logits = gpu_model.readLastLogitsF16();
        var max_err: f32 = 0.0;
        var ref_argmax: u32 = 0;
        var ref_max: f32 = -std.math.inf(f32);
        for (0..b.cfg.vocab_size) |i| {
            if (ref_logits[i] > ref_max) {
                ref_max = ref_logits[i];
                ref_argmax = @intCast(i);
            }
            const diff = @abs(@as(f32, @floatCast(gpu_logits[i])) - ref_logits[i]);
            if (diff > max_err) max_err = diff;
        }
        try std.testing.expect(max_err < 2e-2);
        try std.testing.expectEqual(ref_argmax, last_tok);
    }

    // Verify opt-in `--loop-early-exit` (`sigmoid_linear`) triggers at threshold 0.0 (always >= 0.0).
    var ee_model = try GraphModel.loadFromBytes(allocator, b.amf_bytes, b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .max_loops_override = 3,
        .enable_loop_early_exit = true,
        .early_exit_threshold = 0.0,
    });
    defer ee_model.deinit(allocator);
    _ = try ee_model.stepDecode(4, 0);
    const ee_stats = ee_model.loopIsolationStats();
    try std.testing.expectEqual(@as(u64, 1), ee_stats.total_loop_steps);
    try std.testing.expectEqual(@as(u64, 1), ee_stats.early_exit_tokens);
}

test "W16: tiny-steplora alpha_0 == 0 bit-exact identity at R=1 & active SME2 Step-LoRA at R=2..3" {
    const allocator = std.testing.allocator;
    var base_b = try buildSynthLoopDenseTestBundle(allocator, .single_pass, 1, false, false);
    defer {
        base_b.cfg.deinit(allocator);
        allocator.free(base_b.amf_bytes);
    }
    var lora_b = try buildSynthLoopDenseTestBundle(allocator, .gated_loop_adapter, 3, false, true);
    defer {
        lora_b.cfg.deinit(allocator);
        allocator.free(lora_b.amf_bytes);
    }

    var base_model = try GraphModel.loadFromBytes(allocator, base_b.amf_bytes, base_b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .max_loops_override = 1,
    });
    defer base_model.deinit(allocator);

    var lora_r1 = try GraphModel.loadFromBytes(allocator, lora_b.amf_bytes, lora_b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .max_loops_override = 1,
    });
    defer lora_r1.deinit(allocator);

    const prompt = [_]u32{ 9, 22, 51 };
    for (prompt, 0..) |tid, i| {
        const b_tok = try base_model.stepDecode(tid, @intCast(i));
        const l_tok = try lora_r1.stepDecode(tid, @intCast(i));
        try std.testing.expectEqual(b_tok, l_tok);
    }
    // Bit-exact `f16` logits identity at `R = 1` (`alpha_0 == 0.0`)!
    const base_logits = base_model.readLastLogitsF16();
    const lora_r1_logits = lora_r1.readLastLogitsF16();
    for (0..base_b.cfg.vocab_size) |i| {
        try std.testing.expectEqual(@as(u16, @bitCast(base_logits[i])), @as(u16, @bitCast(lora_r1_logits[i])));
    }
    const iso_r1 = lora_r1.loopIsolationStats();
    try std.testing.expect(iso_r1.steplora_zero_alpha_skips > 0);
    try std.testing.expectEqual(@as(u64, 0), iso_r1.steplora_active_evals);

    // At R = 3, SME2 Step-LoRA runs actively on r = 1, 2 and `verifyTree` matches `stepDecode`!
    var lora_r3_serial = try GraphModel.loadFromBytes(allocator, lora_b.amf_bytes, lora_b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .max_loops_override = 3,
    });
    defer lora_r3_serial.deinit(allocator);

    var lora_r3_tree = try GraphModel.loadFromBytes(allocator, lora_b.amf_bytes, lora_b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .max_loops_override = 3,
    });
    defer lora_r3_tree.deinit(allocator);

    const s0 = try lora_r3_serial.prefillChunk(&prompt, 0, true);
    const t0 = try lora_r3_tree.prefillChunk(&prompt, 0, true);
    try std.testing.expectEqual(s0, t0);

    const iso_r3 = lora_r3_serial.loopIsolationStats();
    try std.testing.expect(iso_r3.steplora_zero_alpha_skips > 0);
    try std.testing.expect(iso_r3.steplora_active_evals > 0);

    const sd1 = try lora_r3_serial.stepDecode(s0, 3);
    const sd2 = try lora_r3_serial.stepDecode(sd1, 4);
    const vr = try lora_r3_tree.verifyTree(&.{ t0, sd1 }, &.{ 3, 4 }, &.{ 0b01, 0b11 }, 3);
    try std.testing.expectEqual(sd1, vr.argmax_tokens[0]);
    try std.testing.expectEqual(sd2, vr.argmax_tokens[1]);
}
