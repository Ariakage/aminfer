//! Milestone M9.4 (`W28`, `Qwen3.8-Flash-Next` / `qwen4_exp`, `DeepSeek-V4.1-Flash-1017` /
//! `deepseek41`, `3L DSpark`, `W20` vision sidecar, `C27`/`24gb-m4` tiering, and `W30` LoRA/DoRA)
//! `aminfer bench-m94` parity, alignment, memory-safety, and throughput suite.
//!
//! Zero hardcoded volume paths; synthetic fixtures live inside temporary directories created
//! via `graph.makeTestTempDir` and removed on return. Staged Hugging Face bundles
//! (`qwen4-real-hf-synth`, `qwen4-tiny-random-bf16`, `deepseek41-real-hf-synth`) are checked
//! whenever `AMINFER_MODELS_DIR` (or `--models-dir`) is configured. Enforces
//! `swapouts_delta == 0` and steady-state `malloc_blocks_delta <= 0` (`MTL_DEBUG_LAYER` /
//! `MTL_SHADER_VALIDATION` aware).

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const arena_mod = @import("../mem/arena.zig");
const vmstat = @import("../platform/vmstat.zig");
const paths = @import("../platform/paths.zig");
const quant = @import("../quant/spec.zig");
const quant_ref = @import("../quant/formats.zig");
const qformats = @import("../gpu/quant_formats.zig");
const amf = @import("../format/amf.zig");
const hybrid = @import("../gpu/hybrid.zig");
const graph = @import("../model/graph.zig");
const lora = @import("../model/lora.zig");
const m82_rt = @import("../model/m82_runtime.zig");
const draft = @import("../spec/draft.zig");
const dspark = @import("../spec/dspark.zig");
const tree_mod = @import("../spec/tree.zig");
const engine_mod = @import("../spec/engine.zig");
const moe = @import("../gpu/moe.zig");
const m81 = @import("m81.zig");

const Io = std.Io;

const MallocStatistics = extern struct {
    blocks_in_use: u32,
    size_in_use: usize,
    max_size_in_use: usize,
    size_allocated: usize,
};

extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *MallocStatistics) void;
extern "c" fn objc_autoreleasePoolPush() ?*anyopaque;
extern "c" fn objc_autoreleasePoolPop(pool: ?*anyopaque) void;

pub const M94_COLD_KL_THRESHOLDS = [_]f32{ 0.0, 0.03, 0.05 };

pub const M94Config = struct {
    warmup_iters: u32 = 4,
    bench_iters: u32 = 16,
    e2e_tokens: u32 = 8,
    spec_tokens: u32 = 8,
    kl_steps: u32 = 6,
    lora_iters: u32 = 4,
    models_dir: ?[]const u8 = null,
};

pub const M94Report = struct {
    // Section 1: W28 Header Gate & Schema Audit
    w28_synth_files_checked: u32 = 0,
    w28_synth_max_name_bytes: u32 = 0,
    w28_synth_long_names: u32 = 0,
    w28_synth_ok: bool = false,
    staged_models_present: bool = false,
    w28_staged_files_checked: u32 = 0,
    w28_staged_tensors_checked: u32 = 0,
    w28_staged_max_name_bytes: u32 = 0,
    w28_staged_long_names: u32 = 0,
    w28_staged_crc_errors: u32 = 0,
    w28_staged_align_errors: u32 = 0,
    w28_ok: bool = false,

    // Section 2: Qwen3.8-Flash-Next (W10 / G10 / C10 / MTP / W20)
    qwen4_hc_cos_sim: f64 = 0.0,
    qwen4_hc_max_err: f64 = 0.0,
    qwen4_ple_cos_sim: f64 = 0.0,
    qwen4_ple_max_err: f64 = 0.0,
    qwen4_gdn_sig_gate_cos_sim: f64 = 0.0,
    qwen4_gdn_sig_gate_max_err: f64 = 0.0,
    qwen4_kernel_parity_ok: bool = false,
    qwen4_synth_four_stream_norm: f64 = 0.0,
    qwen4_synth_ple_r_ge1_nvme_reads: u64 = 0,
    qwen4_synth_ple_r_ge1_reuses: u64 = 0,
    qwen4_synth_wide_tree_ok: bool = false,
    qwen4_staged_checked: bool = false,
    qwen4_staged_ple_r_ge1_nvme_reads: u64 = 0,
    qwen4_staged_ple_r_ge1_reuses: u64 = 0,
    qwen4_staged_decode_tok_s: f64 = 0.0,
    qwen4_staged_mtp_tok_s: f64 = 0.0,
    qwen4_staged_mtp_mean_accept: f64 = 0.0,
    qwen4_staged_mtp_speedup: f64 = 0.0,
    qwen4_staged_mtp_parity_ok: bool = false,
    qwen4_staged_mtp_sme2_cos_sim: f64 = 0.0,
    qwen4_staged_mtp_sme2_evals: u64 = 0,
    qwen4_staged_mtp_512e_pruned_ok: bool = false,
    qwen4_staged_malloc_delta: i64 = 0,
    qwen4_staged_ok: bool = false,
    qwen4_tiny_checked: bool = false,
    qwen4_tiny_omitted_norms_ok: bool = false,
    qwen4_vision_sidecar_ok: bool = false,
    qwen4_vision_tokens: u32 = 0,
    qwen4_section_ok: bool = false,

    // Section 3: DeepSeek-V4.1-Flash-1017 (W11 / G10 / G11 / D10 / 24gb-m4 / W30)
    ds_mxfp8_cos_sim: f64 = 0.0,
    ds_mxfp8_max_err: f64 = 0.0,
    ds_mxfp4_cos_sim: f64 = 0.0,
    ds_mxfp4_max_err: f64 = 0.0,
    ds_mhc_sinkhorn_cos_sim: f64 = 0.0,
    ds_mhc_sinkhorn_max_err: f64 = 0.0,
    ds_compressor_cos_sim: f64 = 0.0,
    ds_compressor_max_err: f64 = 0.0,
    ds_sqrtsoftplus_max_err: f64 = 0.0,
    ds_mla_engram_cos_sim: f64 = 0.0,
    ds_mla_engram_max_err: f64 = 0.0,
    ds_wo_grouped_ok: bool = false,
    ds_kernel_parity_ok: bool = false,
    ds_synth_encoder_decode_dispatches: u64 = 0,
    c27_lfu_baseline_hit_rate: f64 = 0.0,
    c27_lfu_promoted_hit_rate: f64 = 0.0,
    c27_session_promotions: u64 = 0,
    c27_lfu_ok: bool = false,
    ds_cold_kl_thresholds: [3]f32 = M94_COLD_KL_THRESHOLDS,
    ds_cold_kl_mean: [3]f64 = .{ 0.0, 0.0, 0.0 },
    ds_cold_skipped_low_weight: [3]u64 = .{ 0, 0, 0 },
    ds_cold_lossless_ok: bool = false,
    ds_staged_checked: bool = false,
    ds_staged_encoder_prefill_dispatches: u64 = 0,
    ds_staged_encoder_decode_dispatches: u64 = 0,
    ds_staged_decode_tok_s: f64 = 0.0,
    ds_staged_dspark_tok_s: f64 = 0.0,
    ds_staged_dspark_mean_accept: f64 = 0.0,
    ds_staged_dspark_speedup: f64 = 0.0,
    ds_staged_dspark_parity_ok: bool = false,
    ds_staged_dspark_cold_hits: u64 = 0,
    ds_staged_dspark_cold_misses: u64 = 0,
    ds_staged_malloc_delta: i64 = 0,
    ds_staged_ok: bool = false,
    w30_ds_lora_logit_diff: f64 = 0.0,
    w30_ds_unload_max_err: f64 = 0.0,
    w30_ds_lora_malloc_delta: i64 = 0,
    w30_ds_lora_ok: bool = false,
    ds_section_ok: bool = false,

    // Global runtime invariants
    mtl_validation_active: bool = false,
    malloc_blocks_delta: i64 = 0,
    swapouts_delta: u64 = 0,
    pageouts_delta: u64 = 0,
    all_passed: bool = false,

    fn passLabel(ok: bool) []const u8 {
        return if (ok) "PASS" else "FAIL";
    }

    fn checkedPassLabel(checked: bool, ok: bool) []const u8 {
        return if (!checked) "SKIP" else passLabel(ok);
    }

    pub fn write(self: M94Report, out: *Io.Writer) !void {
        try out.print(
            \\bench.m94.w28.synth           files={d} max_name_bytes={d} long_names={d} status={s}
            \\bench.m94.w28.staged          present={s} files={d} tensors={d} max_name_bytes={d} long_names={d} crc_err={d} align_err={d} status={s}
            \\bench.m94.qwen4.kernels       hc_cos={d:.6} (err={d:.5}) ple_cos={d:.6} (err={d:.5}) sig_gate_cos={d:.6} (err={d:.5}) status={s}
            \\bench.m94.qwen4.synth_e2e     4s_norm={d:.4} ple_r_ge1_nvme={d} ple_r_ge1_reuse={d} status={s}
            \\
        , .{
            self.w28_synth_files_checked,
            self.w28_synth_max_name_bytes,
            self.w28_synth_long_names,
            passLabel(self.w28_synth_ok),
            if (self.staged_models_present) "true" else "false",
            self.w28_staged_files_checked,
            self.w28_staged_tensors_checked,
            self.w28_staged_max_name_bytes,
            self.w28_staged_long_names,
            self.w28_staged_crc_errors,
            self.w28_staged_align_errors,
            passLabel(self.w28_ok),
            self.qwen4_hc_cos_sim,
            self.qwen4_hc_max_err,
            self.qwen4_ple_cos_sim,
            self.qwen4_ple_max_err,
            self.qwen4_gdn_sig_gate_cos_sim,
            self.qwen4_gdn_sig_gate_max_err,
            passLabel(self.qwen4_kernel_parity_ok),
            self.qwen4_synth_four_stream_norm,
            self.qwen4_synth_ple_r_ge1_nvme_reads,
            self.qwen4_synth_ple_r_ge1_reuses,
            passLabel(self.qwen4_synth_wide_tree_ok),
        });
        try out.print(
            \\bench.m94.qwen4.staged_mtp    checked={s} dec_tok_s={d:.2} mtp_tok_s={d:.2} accept={d:.2} speedup={d:.2}x sme2_cos={d:.6} sme2_evals={d} 512e_d7={s} malloc_delta={d} status={s}
            \\bench.m94.qwen4.tiny_vision   tiny_checked={s} omitted_norms={s} vision_sidecar={s} vis_tokens={d}
            \\bench.m94.ds.kernels          mxfp8_cos={d:.6} mxfp4_cos={d:.6} mhc_sinkhorn_cos={d:.6} comp_cos={d:.6} sqrtsp_err={d:.6} mla_engram_cos={d:.6} wo_grouped={s} status={s}
            \\
        , .{
            if (self.qwen4_staged_checked) "true" else "false",
            self.qwen4_staged_decode_tok_s,
            self.qwen4_staged_mtp_tok_s,
            self.qwen4_staged_mtp_mean_accept,
            self.qwen4_staged_mtp_speedup,
            self.qwen4_staged_mtp_sme2_cos_sim,
            self.qwen4_staged_mtp_sme2_evals,
            checkedPassLabel(self.qwen4_staged_checked, self.qwen4_staged_mtp_512e_pruned_ok),
            self.qwen4_staged_malloc_delta,
            checkedPassLabel(self.qwen4_staged_checked, self.qwen4_staged_ok),
            if (self.qwen4_tiny_checked) "true" else "false",
            checkedPassLabel(self.qwen4_tiny_checked, self.qwen4_tiny_omitted_norms_ok),
            passLabel(self.qwen4_vision_sidecar_ok),
            self.qwen4_vision_tokens,
            self.ds_mxfp8_cos_sim,
            self.ds_mxfp4_cos_sim,
            self.ds_mhc_sinkhorn_cos_sim,
            self.ds_compressor_cos_sim,
            self.ds_sqrtsoftplus_max_err,
            self.ds_mla_engram_cos_sim,
            passLabel(self.ds_wo_grouped_ok),
            passLabel(self.ds_kernel_parity_ok),
        });
        try out.print(
            \\bench.m94.ds.c27_lfu          base_hit={d:.4} lfu_hit={d:.4} promotions={d} status={s}
            \\bench.m94.ds.cold_kl          thr=[{d:.2},{d:.2},{d:.2}] mean_kl=[{d:.6},{d:.6},{d:.6}] skipped=[{d},{d},{d}] lossless={s}
            \\
        , .{
            self.c27_lfu_baseline_hit_rate,
            self.c27_lfu_promoted_hit_rate,
            self.c27_session_promotions,
            passLabel(self.c27_lfu_ok),
            self.ds_cold_kl_thresholds[0],
            self.ds_cold_kl_thresholds[1],
            self.ds_cold_kl_thresholds[2],
            self.ds_cold_kl_mean[0],
            self.ds_cold_kl_mean[1],
            self.ds_cold_kl_mean[2],
            self.ds_cold_skipped_low_weight[0],
            self.ds_cold_skipped_low_weight[1],
            self.ds_cold_skipped_low_weight[2],
            passLabel(self.ds_cold_lossless_ok),
        });
        try out.print(
            \\bench.m94.ds.staged_dspark    checked={s} enc_pre={d} enc_dec={d} dec_tok_s={d:.2} dspark_tok_s={d:.2} accept={d:.2} speedup={d:.2}x cold_hits={d} cold_misses={d} malloc_delta={d} status={s}
            \\bench.m94.ds.w30_lora         logit_diff={d:.4} unload_err={d:.6} malloc_delta={d} status={s}
            \\bench.m94.summary             mtl_validation={s} malloc_delta={d} swapouts_delta={d} pageouts_delta={d} all_passed={s}
            \\
        , .{
            if (self.ds_staged_checked) "true" else "false",
            self.ds_staged_encoder_prefill_dispatches,
            self.ds_staged_encoder_decode_dispatches,
            self.ds_staged_decode_tok_s,
            self.ds_staged_dspark_tok_s,
            self.ds_staged_dspark_mean_accept,
            self.ds_staged_dspark_speedup,
            self.ds_staged_dspark_cold_hits,
            self.ds_staged_dspark_cold_misses,
            self.ds_staged_malloc_delta,
            checkedPassLabel(self.ds_staged_checked, self.ds_staged_ok),
            self.w30_ds_lora_logit_diff,
            self.w30_ds_unload_max_err,
            self.w30_ds_lora_malloc_delta,
            passLabel(self.w30_ds_lora_ok),
            if (self.mtl_validation_active) "true" else "false",
            self.malloc_blocks_delta,
            self.swapouts_delta,
            self.pageouts_delta,
            if (self.all_passed) "true" else "false",
        });
    }

    pub fn writeJson(self: M94Report, out: *Io.Writer) !void {
        try out.print(
            \\{{"w28_synth_files_checked":{d},"w28_synth_max_name_bytes":{d},"w28_synth_long_names":{d},"w28_synth_ok":{s},"staged_models_present":{s},"w28_staged_files_checked":{d},"w28_staged_tensors_checked":{d},"w28_staged_max_name_bytes":{d},"w28_staged_long_names":{d},"w28_staged_crc_errors":{d},"w28_staged_align_errors":{d},"w28_ok":{s},"qwen4_hc_cos_sim":{d:.6},"qwen4_hc_max_err":{d:.6},"qwen4_ple_cos_sim":{d:.6},"qwen4_ple_max_err":{d:.6},"qwen4_gdn_sig_gate_cos_sim":{d:.6},"qwen4_gdn_sig_gate_max_err":{d:.6},"qwen4_kernel_parity_ok":{s},"qwen4_synth_four_stream_norm":{d:.6},"qwen4_synth_ple_r_ge1_nvme_reads":{d},"qwen4_synth_ple_r_ge1_reuses":{d},"qwen4_synth_wide_tree_ok":{s},
        , .{
            self.w28_synth_files_checked,
            self.w28_synth_max_name_bytes,
            self.w28_synth_long_names,
            if (self.w28_synth_ok) "true" else "false",
            if (self.staged_models_present) "true" else "false",
            self.w28_staged_files_checked,
            self.w28_staged_tensors_checked,
            self.w28_staged_max_name_bytes,
            self.w28_staged_long_names,
            self.w28_staged_crc_errors,
            self.w28_staged_align_errors,
            if (self.w28_ok) "true" else "false",
            self.qwen4_hc_cos_sim,
            self.qwen4_hc_max_err,
            self.qwen4_ple_cos_sim,
            self.qwen4_ple_max_err,
            self.qwen4_gdn_sig_gate_cos_sim,
            self.qwen4_gdn_sig_gate_max_err,
            if (self.qwen4_kernel_parity_ok) "true" else "false",
            self.qwen4_synth_four_stream_norm,
            self.qwen4_synth_ple_r_ge1_nvme_reads,
            self.qwen4_synth_ple_r_ge1_reuses,
            if (self.qwen4_synth_wide_tree_ok) "true" else "false",
        });
        try out.print(
            \\"qwen4_staged_checked":{s},"qwen4_staged_ple_r_ge1_nvme_reads":{d},"qwen4_staged_ple_r_ge1_reuses":{d},"qwen4_staged_decode_tok_s":{d:.2},"qwen4_staged_mtp_tok_s":{d:.2},"qwen4_staged_mtp_mean_accept":{d:.3},"qwen4_staged_mtp_speedup":{d:.3},"qwen4_staged_mtp_parity_ok":{s},"qwen4_staged_mtp_sme2_cos_sim":{d:.6},"qwen4_staged_mtp_sme2_evals":{d},"qwen4_staged_mtp_512e_pruned_ok":{s},"qwen4_staged_malloc_delta":{d},"qwen4_staged_ok":{s},"qwen4_tiny_checked":{s},"qwen4_tiny_omitted_norms_ok":{s},"qwen4_vision_sidecar_ok":{s},"qwen4_vision_tokens":{d},"qwen4_section_ok":{s},
        , .{
            if (self.qwen4_staged_checked) "true" else "false",
            self.qwen4_staged_ple_r_ge1_nvme_reads,
            self.qwen4_staged_ple_r_ge1_reuses,
            self.qwen4_staged_decode_tok_s,
            self.qwen4_staged_mtp_tok_s,
            self.qwen4_staged_mtp_mean_accept,
            self.qwen4_staged_mtp_speedup,
            if (self.qwen4_staged_mtp_parity_ok) "true" else "false",
            self.qwen4_staged_mtp_sme2_cos_sim,
            self.qwen4_staged_mtp_sme2_evals,
            if (self.qwen4_staged_mtp_512e_pruned_ok) "true" else "false",
            self.qwen4_staged_malloc_delta,
            if (self.qwen4_staged_ok) "true" else "false",
            if (self.qwen4_tiny_checked) "true" else "false",
            if (self.qwen4_tiny_omitted_norms_ok) "true" else "false",
            if (self.qwen4_vision_sidecar_ok) "true" else "false",
            self.qwen4_vision_tokens,
            if (self.qwen4_section_ok) "true" else "false",
        });
        try out.print(
            \\"ds_mxfp8_cos_sim":{d:.6},"ds_mxfp8_max_err":{d:.6},"ds_mxfp4_cos_sim":{d:.6},"ds_mxfp4_max_err":{d:.6},"ds_mhc_sinkhorn_cos_sim":{d:.6},"ds_mhc_sinkhorn_max_err":{d:.6},"ds_compressor_cos_sim":{d:.6},"ds_compressor_max_err":{d:.6},"ds_sqrtsoftplus_max_err":{d:.6},"ds_mla_engram_cos_sim":{d:.6},"ds_mla_engram_max_err":{d:.6},"ds_wo_grouped_ok":{s},"ds_kernel_parity_ok":{s},"ds_synth_encoder_decode_dispatches":{d},"c27_lfu_baseline_hit_rate":{d:.6},"c27_lfu_promoted_hit_rate":{d:.6},"c27_session_promotions":{d},"c27_lfu_ok":{s},
        , .{
            self.ds_mxfp8_cos_sim,
            self.ds_mxfp8_max_err,
            self.ds_mxfp4_cos_sim,
            self.ds_mxfp4_max_err,
            self.ds_mhc_sinkhorn_cos_sim,
            self.ds_mhc_sinkhorn_max_err,
            self.ds_compressor_cos_sim,
            self.ds_compressor_max_err,
            self.ds_sqrtsoftplus_max_err,
            self.ds_mla_engram_cos_sim,
            self.ds_mla_engram_max_err,
            if (self.ds_wo_grouped_ok) "true" else "false",
            if (self.ds_kernel_parity_ok) "true" else "false",
            self.ds_synth_encoder_decode_dispatches,
            self.c27_lfu_baseline_hit_rate,
            self.c27_lfu_promoted_hit_rate,
            self.c27_session_promotions,
            if (self.c27_lfu_ok) "true" else "false",
        });
        try out.print(
            \\"ds_cold_kl_thresholds":[{d:.2},{d:.2},{d:.2}],"ds_cold_kl_mean":[{d:.6},{d:.6},{d:.6}],"ds_cold_skipped_low_weight":[{d},{d},{d}],"ds_cold_lossless_ok":{s},
        , .{
            self.ds_cold_kl_thresholds[0],
            self.ds_cold_kl_thresholds[1],
            self.ds_cold_kl_thresholds[2],
            self.ds_cold_kl_mean[0],
            self.ds_cold_kl_mean[1],
            self.ds_cold_kl_mean[2],
            self.ds_cold_skipped_low_weight[0],
            self.ds_cold_skipped_low_weight[1],
            self.ds_cold_skipped_low_weight[2],
            if (self.ds_cold_lossless_ok) "true" else "false",
        });
        try out.print(
            \\"ds_staged_checked":{s},"ds_staged_encoder_prefill_dispatches":{d},"ds_staged_encoder_decode_dispatches":{d},"ds_staged_decode_tok_s":{d:.2},"ds_staged_dspark_tok_s":{d:.2},"ds_staged_dspark_mean_accept":{d:.3},"ds_staged_dspark_speedup":{d:.3},"ds_staged_dspark_parity_ok":{s},"ds_staged_dspark_cold_hits":{d},"ds_staged_dspark_cold_misses":{d},"ds_staged_malloc_delta":{d},"ds_staged_ok":{s},"w30_ds_lora_logit_diff":{d:.6},"w30_ds_unload_max_err":{d:.6},"w30_ds_lora_malloc_delta":{d},"w30_ds_lora_ok":{s},"ds_section_ok":{s},"mtl_validation_active":{s},"malloc_blocks_delta":{d},"swapouts_delta":{d},"pageouts_delta":{d},"all_passed":{s}}}
            \\
        , .{
            if (self.ds_staged_checked) "true" else "false",
            self.ds_staged_encoder_prefill_dispatches,
            self.ds_staged_encoder_decode_dispatches,
            self.ds_staged_decode_tok_s,
            self.ds_staged_dspark_tok_s,
            self.ds_staged_dspark_mean_accept,
            self.ds_staged_dspark_speedup,
            if (self.ds_staged_dspark_parity_ok) "true" else "false",
            self.ds_staged_dspark_cold_hits,
            self.ds_staged_dspark_cold_misses,
            self.ds_staged_malloc_delta,
            if (self.ds_staged_ok) "true" else "false",
            self.w30_ds_lora_logit_diff,
            self.w30_ds_unload_max_err,
            self.w30_ds_lora_malloc_delta,
            if (self.w30_ds_lora_ok) "true" else "false",
            if (self.ds_section_ok) "true" else "false",
            if (self.mtl_validation_active) "true" else "false",
            self.malloc_blocks_delta,
            self.swapouts_delta,
            self.pageouts_delta,
            if (self.all_passed) "true" else "false",
        });
    }
};

// ---------------------------------------------------------------------------
// Generic Numerics, File & Hot-Path Malloc Helpers
// ---------------------------------------------------------------------------

fn envFlagEnabled(name: [:0]const u8) bool {
    const raw = std.c.getenv(name.ptr) orelse return false;
    const val = std.mem.sliceTo(raw, 0);
    return val.len > 0 and !std.mem.eql(u8, val, "0");
}

fn sigmoidF32(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn deterministicF32(seed: u64, idx: usize, amp: f32, bias: f32) f32 {
    var s: u64 = seed +% @as(u64, idx) *% 0x9e3779b97f4a7c15;
    s = (s ^ (s >> 30)) *% 0xbf58476d1ce4e5b9;
    s = (s ^ (s >> 27)) *% 0x94d049bb133111eb;
    s ^= (s >> 31);
    const u: f32 = @as(f32, @floatFromInt(@as(u32, @truncate(s >> 40)))) / 16777216.0;
    return (u * 2.0 - 1.0) * amp + bias;
}

fn fillDeterministicF16(allocator: std.mem.Allocator, seed: u64, count: usize, amp: f32) ![]u8 {
    const out = try allocator.alloc(u8, count * 2);
    for (0..count) |i| {
        const h_val: f16 = @floatCast(deterministicF32(seed, i, amp, 0.0));
        std.mem.writeInt(u16, out[i * 2 ..][0..2], @bitCast(h_val), .little);
    }
    return out;
}

fn fillDeterministicBf16(allocator: std.mem.Allocator, seed: u64, count: usize, amp: f32, bias: f32) ![]u8 {
    const out = try allocator.alloc(u8, count * 2);
    for (0..count) |i| {
        std.mem.writeInt(u16, out[i * 2 ..][0..2], quant.f32ToBf16(deterministicF32(seed, i, amp, bias)), .little);
    }
    return out;
}

fn fillDeterministicF32Bytes(allocator: std.mem.Allocator, seed: u64, count: usize, amp: f32, bias: f32) ![]u8 {
    const out = try allocator.alloc(u8, count * 4);
    for (0..count) |i| {
        std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(deterministicF32(seed, i, amp, bias)), .little);
    }
    return out;
}

const CosineStats = struct { cos: f64, max_err: f64 };

fn cosineF64(a: []const f32, b: []const f32) CosineStats {
    var dot: f64 = 0.0;
    var na: f64 = 0.0;
    var nb: f64 = 0.0;
    var max_e: f64 = 0.0;
    for (a, b) |av, bv| {
        const af: f64 = av;
        const bf: f64 = bv;
        dot += af * bf;
        na += af * af;
        nb += bf * bf;
        max_e = @max(max_e, @abs(af - bf));
    }
    const denom = @sqrt(na) * @sqrt(nb);
    return .{
        .cos = if (denom > 0.0) dot / denom else 0.0,
        .max_err = max_e,
    };
}

fn cosineF16(a: []const f16, b: []const f16) f64 {
    var dot: f64 = 0.0;
    var na: f64 = 0.0;
    var nb: f64 = 0.0;
    for (a, b) |av, bv| {
        const af: f64 = @floatCast(av);
        const bf: f64 = @floatCast(bv);
        dot += af * bf;
        na += af * af;
        nb += bf * bf;
    }
    const denom = @sqrt(na) * @sqrt(nb);
    return if (denom > 0.0) dot / denom else 0.0;
}

fn argmaxF16(logits: []const f16) u32 {
    var best_i: usize = 0;
    var best_v: f32 = -std.math.inf(f32);
    for (logits, 0..) |v, i| {
        const f: f32 = @floatCast(v);
        if (f > best_v) {
            best_v = f;
            best_i = i;
        }
    }
    return @intCast(best_i);
}

fn allFiniteF16(slice: []const f16) bool {
    for (slice) |v| {
        if (!std.math.isFinite(@as(f32, @floatCast(v)))) return false;
    }
    return true;
}

fn fileExistsPosix(path: [:0]const u8) bool {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return false;
    _ = std.c.close(fd);
    return true;
}

fn writeFileBytesPosix(path: [:0]const u8, data: []const u8) !void {
    const fd = std.c.open(path.ptr, .{
        .ACCMODE = .WRONLY,
        .CREAT = true,
        .TRUNC = true,
    }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.FileCreateFailed;
    defer _ = std.c.close(fd);
    var off: usize = 0;
    while (off < data.len) {
        const rc = std.c.write(fd, data.ptr + off, data.len - off);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.WriteFailed;
        }
        if (rc == 0) return error.WriteFailed;
        off += @intCast(rc);
    }
}

fn readFileBytesPosixAlloc(allocator: std.mem.Allocator, path: [:0]const u8, max_bytes: usize) !?[]u8 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var buf: [16384]u8 = undefined;
    while (true) {
        const rc = std.c.read(fd, &buf, buf.len);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.ReadFailed;
        }
        if (rc == 0) break;
        const n: usize = @intCast(rc);
        if (out.items.len + n > max_bytes) return error.FileTooBig;
        try out.appendSlice(allocator, buf[0..n]);
    }
    return try out.toOwnedSlice(allocator);
}

fn resolveModelFileCandidateZ(
    buf: *[std.fs.max_path_bytes]u8,
    models_dir: []const u8,
    subpath: []const u8,
) ?[:0]const u8 {
    const z = paths.joinPathZ(buf, models_dir, subpath) catch return null;
    return if (fileExistsPosix(z)) z else null;
}

fn fillDeterministicVisionFrame(frame: *metal.VisionFrame, r: u8, g: u8, b: u8) !void {
    const info = try frame.info();
    const w = info.width();
    const h = info.height();
    var lock = try frame.lockPlane(0, false);
    errdefer lock.release();
    for (0..h) |y| {
        const row = lock.rowBytes(@intCast(y));
        for (0..w) |x| {
            row[x * 4 + 0] = b;
            row[x * 4 + 1] = g;
            row[x * 4 + 2] = r;
            row[x * 4 + 3] = 255;
        }
    }
    try lock.unlock();
}

fn measureStepEncodeMallocDelta(
    model: *graph.GraphModel,
    cur_tok: *u32,
    pos: *u32,
) !i64 {
    const EncodeCapture = struct {
        m0: MallocStatistics = std.mem.zeroes(MallocStatistics),
        m1: MallocStatistics = std.mem.zeroes(MallocStatistics),
        captured_begin: bool = false,
        captured: bool = false,
        fn onPostBegin(ctx: ?*anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            malloc_zone_statistics(null, &self.m0);
            self.captured_begin = true;
        }
        fn onPreCommit(ctx: ?*anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            malloc_zone_statistics(null, &self.m1);
            self.captured = true;
        }
    };
    for (0..4) |_| {
        const wpool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(wpool);
        cur_tok.* = try model.stepDecode(cur_tok.*, pos.*);
        pos.* += 1;
    }
    var best_encode_delta: i64 = std.math.maxInt(i64);
    var best_steady_delta: i64 = std.math.maxInt(i64);
    var prev_m0: ?u64 = null;
    for (0..32) |_| {
        const enc_pool = objc_autoreleasePoolPush();
        var cap: EncodeCapture = .{};
        model.pre_commit_bench_ctx = &cap;
        model.post_begin_bench_hook = EncodeCapture.onPostBegin;
        model.pre_commit_bench_hook = EncodeCapture.onPreCommit;
        defer {
            model.post_begin_bench_hook = null;
            model.pre_commit_bench_hook = null;
            model.pre_commit_bench_ctx = null;
            objc_autoreleasePoolPop(enc_pool);
        }
        var fallback_m0: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &fallback_m0);
        cur_tok.* = try model.stepDecode(cur_tok.*, pos.*);
        pos.* += 1;
        const m0_ref = if (cap.captured_begin) cap.m0 else fallback_m0;
        if (!cap.captured) malloc_zone_statistics(null, &cap.m1);
        const edelta = @as(i64, cap.m1.blocks_in_use) - @as(i64, m0_ref.blocks_in_use);
        if (edelta < best_encode_delta) best_encode_delta = edelta;
        if (prev_m0) |pm0| {
            const sdelta = @as(i64, @intCast(m0_ref.blocks_in_use)) - @as(i64, @intCast(pm0));
            if (sdelta < best_steady_delta) best_steady_delta = sdelta;
        }
        prev_m0 = m0_ref.blocks_in_use;
        if (best_encode_delta <= 0 or best_steady_delta <= 0) break;
    }
    return @min(best_encode_delta, best_steady_delta);
}

fn benchGreedyDecode(
    model: *graph.GraphModel,
    prompt: []const u32,
    n_tokens: u32,
    out_tokens: []u32,
    measure_malloc: bool,
) !struct { tok_s: f64, malloc_delta: i64 } {
    model.resetKvCache();
    model.resetColdTelemetry();
    var cur = try model.prefillChunk(prompt, 0, true);
    out_tokens[0] = cur;
    var pos: u32 = @intCast(prompt.len);
    var emitted: u32 = 1;
    while (emitted < n_tokens) : (emitted += 1) {
        cur = try model.stepDecode(cur, pos);
        out_tokens[emitted] = cur;
        pos += 1;
    }

    const step_iters = @max(n_tokens, 6);
    var best_tok_s: f64 = 0.0;
    for (0..2) |_| {
        const outer_pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(outer_pool);
        model.resetKvCache();
        var tok = try model.prefillChunk(prompt, 0, true);
        var p: u32 = @intCast(prompt.len);
        const t0 = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        for (0..step_iters) |_| {
            const sp = objc_autoreleasePoolPush();
            defer objc_autoreleasePoolPop(sp);
            tok = try model.stepDecode(tok, p);
            p += 1;
        }
        const t1 = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const elapsed_ns: i96 = @max(@as(i96, 1), t0.durationTo(t1).toNanoseconds());
        const tok_s = @as(f64, @floatFromInt(step_iters)) * 1e9 / @as(f64, @floatFromInt(elapsed_ns));
        best_tok_s = @max(best_tok_s, tok_s);
    }
    const enc_delta: i64 = if (measure_malloc) blk: {
        model.resetKvCache();
        var tok = try model.prefillChunk(prompt, 0, true);
        var p: u32 = @intCast(prompt.len);
        break :blk try measureStepEncodeMallocDelta(model, &tok, &p);
    } else 0;
    return .{ .tok_s = best_tok_s, .malloc_delta = enc_delta };
}

fn evalSpecDecodeRun(
    eng: *engine_mod.SpecEngine,
    prompt: []const u32,
    out_tokens: []u32,
) !struct { produced: usize, tok_s: f64, mean_accept: f64 } {
    eng.reset();
    const t0 = std.Io.Timestamp.now(std.Options.debug_io, .awake);
    out_tokens[0] = try eng.prefill(prompt, null);
    var prod: usize = 1;
    while (prod < out_tokens.len) {
        const rs = try eng.stepRound(null, out_tokens[prod..]);
        if (rs.emitted_count == 0) break;
        prod += rs.emitted_count;
    }
    const t1 = std.Io.Timestamp.now(std.Options.debug_io, .awake);
    const elapsed_ns: i96 = @max(@as(i96, 1), t0.durationTo(t1).toNanoseconds());
    const dec_generated = if (prod > 1) prod - 1 else 1;
    const tok_s = @as(f64, @floatFromInt(dec_generated)) * 1e9 / @as(f64, @floatFromInt(elapsed_ns));
    return .{
        .produced = prod,
        .tok_s = tok_s,
        .mean_accept = eng.stats.meanAcceptanceLength(),
    };
}

// ---------------------------------------------------------------------------
// Synthetic 3L DSpark Fixture Builder (for offline W28 + W24 KL + C27 tests)
// ---------------------------------------------------------------------------

const SynthDspark3LArtifact = struct {
    amf_path: [:0]u8,
    cfg_json: []u8,
    bytes: []u8,

    fn deinit(self: *SynthDspark3LArtifact, allocator: std.mem.Allocator) void {
        allocator.free(self.amf_path);
        allocator.free(self.cfg_json);
        allocator.free(self.bytes);
    }
};

fn buildSynthDspark3LArtifact(
    allocator: std.mem.Allocator,
    tmp_dir: []const u8,
    filename: []const u8,
) !SynthDspark3LArtifact {
    const h: u32 = 64;
    const v: u32 = 48;
    const hc: u32 = 4;
    const mix_hc: u32 = (2 + hc) * hc; // 24
    const n_heads: u32 = 2;
    const hd: u32 = 32;
    const q_lora: u32 = 32;
    const o_groups: u32 = 2;
    const o_lora: u32 = 16;
    const moe_i: u32 = 64;
    const n_exp: u32 = 6;
    const top_k: u32 = 3;
    const mr: u32 = 16;

    const BlobRec = struct {
        entry: amf.TensorEntry,
        bytes: []u8,
    };
    var recs: std.ArrayList(BlobRec) = .empty;
    defer {
        for (recs.items) |rec| allocator.free(rec.bytes);
        recs.deinit(allocator);
    }

    const addRaw = struct {
        fn f(
            a: std.mem.Allocator,
            list: *std.ArrayList(BlobRec),
            name: []const u8,
            kind: amf.TensorKind,
            dt: amf.DType,
            shape: [4]u32,
            rank: u8,
            layer: u16,
            exp: u16,
            bytes: []u8,
        ) !void {
            var e = amf.TensorEntry.zeroed(name, kind, dt);
            e.rank = rank;
            e.shape = shape;
            e.layer = layer;
            e.expert = exp;
            e.setLayoutAndParamKind(.row_major, .affine_scale_bias);
            e.data_bytes = bytes.len;
            try list.append(a, .{ .entry = e, .bytes = bytes });
        }
    }.f;

    var nb: [96]u8 = undefined;
    try addRaw(allocator, &recs, "mtp.0.main_proj", .dense, .bf16, .{ h, h * 3, 0, 0 }, 2, 0, amf.NO_EXPERT, try fillDeterministicBf16(allocator, 94002, @as(usize, h) * h * 3, 0.15, 0.0));
    try addRaw(allocator, &recs, "mtp.0.main_norm.weight", .norm, .f32, .{ h, 0, 0, 0 }, 1, 0, amf.NO_EXPERT, try fillDeterministicF32Bytes(allocator, 94001, h, 0.05, 1.0));

    for (0..3) |li| {
        const l16: u16 = @intCast(li);
        const s0: u64 = 94100 + @as(u64, li) * 500;
        inline for (.{ "hc_attn", "hc_ffn" }, 0..) |tag, ti| {
            const sub_seed: u64 = s0 + ti * 20;
            try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.{s}_fn", .{ li, tag }) catch unreachable, .dense, .bf16, .{ mix_hc, hc * h, 0, 0 }, 2, l16, amf.NO_EXPERT, try fillDeterministicBf16(allocator, sub_seed + 1, @as(usize, mix_hc) * hc * h, 0.05, 0.0));
            try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.{s}_base", .{ li, tag }) catch unreachable, .dense, .f32, .{ mix_hc, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, try fillDeterministicF32Bytes(allocator, sub_seed + 2, mix_hc, 0.05, 0.0));
            try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.{s}_scale", .{ li, tag }) catch unreachable, .dense, .f32, .{ 3, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, try fillDeterministicF32Bytes(allocator, sub_seed + 3, 3, 0.02, 0.1));
        }

        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.attn_norm.weight", .{li}) catch unreachable, .norm, .f32, .{ h, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, try fillDeterministicF32Bytes(allocator, s0 + 30, h, 0.05, 1.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn_norm.weight", .{li}) catch unreachable, .norm, .f32, .{ h, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, try fillDeterministicF32Bytes(allocator, s0 + 31, h, 0.05, 1.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.attn.q_norm.weight", .{li}) catch unreachable, .norm, .f32, .{ q_lora, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, try fillDeterministicF32Bytes(allocator, s0 + 32, q_lora, 0.05, 1.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.attn.kv_norm.weight", .{li}) catch unreachable, .norm, .f32, .{ hd, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, try fillDeterministicF32Bytes(allocator, s0 + 33, hd, 0.05, 1.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.attn.attn_sink", .{li}) catch unreachable, .dense, .f32, .{ n_heads, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, try fillDeterministicF32Bytes(allocator, s0 + 34, n_heads, 0.10, -0.5));

        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.attn.wq_a", .{li}) catch unreachable, .dense, .bf16, .{ q_lora, h, 0, 0 }, 2, l16, amf.NO_EXPERT, try fillDeterministicBf16(allocator, s0 + 40, @as(usize, q_lora) * h, 0.15, 0.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.attn.wq_b", .{li}) catch unreachable, .dense, .bf16, .{ n_heads * hd, q_lora, 0, 0 }, 2, l16, amf.NO_EXPERT, try fillDeterministicBf16(allocator, s0 + 41, @as(usize, n_heads * hd) * q_lora, 0.15, 0.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.attn.wkv", .{li}) catch unreachable, .dense, .bf16, .{ hd, h, 0, 0 }, 2, l16, amf.NO_EXPERT, try fillDeterministicBf16(allocator, s0 + 42, @as(usize, hd) * h, 0.15, 0.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.attn.wo_a", .{li}) catch unreachable, .dense, .bf16, .{ o_groups * o_lora, (n_heads * hd) / o_groups, 0, 0 }, 2, l16, amf.NO_EXPERT, try fillDeterministicBf16(allocator, s0 + 43, @as(usize, o_groups * o_lora) * ((n_heads * hd) / o_groups), 0.15, 0.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.attn.wo_b", .{li}) catch unreachable, .dense, .bf16, .{ h, o_groups * o_lora, 0, 0 }, 2, l16, amf.NO_EXPERT, try fillDeterministicBf16(allocator, s0 + 44, @as(usize, h) * (o_groups * o_lora), 0.15, 0.0));

        // Large-variance router weights produce heavy-tailed `sqrtsoftplus` weights in `[0.005, 0.8]`
        // so `--cold-expert-min-weight` `[0.00, 0.03, 0.05]` skips tail cold experts smoothly.
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.gate.weight", .{li}) catch unreachable, .dense, .bf16, .{ n_exp, h, 0, 0 }, 2, l16, amf.NO_EXPERT, try fillDeterministicBf16(allocator, s0 + 45, @as(usize, n_exp) * h, 1.5, 0.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.gate.bias", .{li}) catch unreachable, .router, .f32, .{ n_exp, 0, 0, 0 }, 1, l16, amf.NO_EXPERT, try fillDeterministicF32Bytes(allocator, s0 + 55, n_exp, 0.05, 0.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.shared_experts.w1", .{li}) catch unreachable, .dense, .bf16, .{ moe_i, h, 0, 0 }, 2, l16, amf.NO_EXPERT, try fillDeterministicBf16(allocator, s0 + 46, @as(usize, moe_i) * h, 0.15, 0.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.shared_experts.w2", .{li}) catch unreachable, .dense, .bf16, .{ h, moe_i, 0, 0 }, 2, l16, amf.NO_EXPERT, try fillDeterministicBf16(allocator, s0 + 47, @as(usize, h) * moe_i, 0.15, 0.0));
        try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.shared_experts.w3", .{li}) catch unreachable, .dense, .bf16, .{ moe_i, h, 0, 0 }, 2, l16, amf.NO_EXPERT, try fillDeterministicBf16(allocator, s0 + 48, @as(usize, moe_i) * h, 0.15, 0.0));

        for (0..n_exp) |ei| {
            const e16: u16 = @intCast(ei);
            const es: u64 = s0 + 100 + @as(u64, ei) * 8;
            try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.experts.{d}.w1.weight", .{ li, ei }) catch unreachable, .cold_expert, .bf16, .{ moe_i, h, 0, 0 }, 2, l16, e16, try fillDeterministicBf16(allocator, es + 1, @as(usize, moe_i) * h, 0.20, 0.0));
            try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.experts.{d}.w2.weight", .{ li, ei }) catch unreachable, .cold_expert, .bf16, .{ h, moe_i, 0, 0 }, 2, l16, e16, try fillDeterministicBf16(allocator, es + 2, @as(usize, h) * moe_i, 0.20, 0.0));
            try addRaw(allocator, &recs, std.fmt.bufPrint(&nb, "mtp.{d}.ffn.experts.{d}.w3.weight", .{ li, ei }) catch unreachable, .cold_expert, .bf16, .{ moe_i, h, 0, 0 }, 2, l16, e16, try fillDeterministicBf16(allocator, es + 3, @as(usize, moe_i) * h, 0.20, 0.0));
        }
    }
    try addRaw(allocator, &recs, "mtp.2.norm.weight", .norm, .f32, .{ h, 0, 0, 0 }, 1, 2, amf.NO_EXPERT, try fillDeterministicF32Bytes(allocator, 94900, h, 0.05, 1.0));
    try addRaw(allocator, &recs, "mtp.2.markov_head.embed.weight", .embedding, .bf16, .{ v, mr, 0, 0 }, 2, 2, amf.NO_EXPERT, try fillDeterministicBf16(allocator, 94901, @as(usize, v) * mr, 0.20, 0.0));
    try addRaw(allocator, &recs, "mtp.2.markov_head.head.weight", .dense, .bf16, .{ v, mr, 0, 0 }, 2, 2, amf.NO_EXPERT, try fillDeterministicBf16(allocator, 94902, @as(usize, v) * mr, 0.20, 0.0));
    try addRaw(allocator, &recs, "mtp.2.confidence_head.proj.weight", .dense, .f32, .{ 1, h + mr, 0, 0 }, 2, 2, amf.NO_EXPERT, try fillDeterministicF32Bytes(allocator, 94903, h + mr, 0.02, 0.0));

    const dir_off: u64 = page.APPLE_PAGE_SIZE;
    const raw_dir_bytes: usize = recs.items.len * @sizeOf(amf.TensorEntry);
    const dir_bytes: u64 = try page.alignUp(@max(raw_dir_bytes, page.APPLE_PAGE_SIZE));
    var cursor: u64 = dir_off + dir_bytes;
    for (recs.items) |*rec| {
        rec.entry.blob_offset = cursor;
        rec.entry.blob_bytes = try page.alignUp(@max(rec.bytes.len, 1));
        cursor += rec.entry.blob_bytes;
    }

    const total_size: usize = @intCast(cursor);
    const img = try allocator.alloc(u8, total_size);
    errdefer allocator.free(img);
    @memset(img, 0);

    for (recs.items, 0..) |*rec, ri| {
        const b_off: usize = @intCast(rec.entry.blob_offset);
        const b_len: usize = @intCast(rec.entry.blob_bytes);
        @memcpy(img[b_off..][0..rec.bytes.len], rec.bytes);
        rec.entry.blob_crc32 = std.hash.Crc32.hash(img[b_off..][0..b_len]);
        const dir_slice = img[@intCast(dir_off + @as(u64, ri) * @sizeOf(amf.TensorEntry))..][0..@sizeOf(amf.TensorEntry)];
        @memcpy(dir_slice, std.mem.asBytes(&rec.entry));
    }

    var hdr: amf.Header = std.mem.zeroes(amf.Header);
    hdr.magic = amf.MAGIC;
    hdr.version = amf.FORMAT_VERSION;
    hdr.flags = 0;
    hdr.page_size = page.APPLE_PAGE_SIZE;
    hdr.checksum_algo = amf.CHECKSUM_CRC32_IEEE;
    hdr.tensor_count = @intCast(recs.items.len);
    hdr.directory_offset = dir_off;
    hdr.directory_bytes = dir_bytes;
    hdr.directory_crc32 = std.hash.Crc32.hash(img[@intCast(dir_off)..][0..raw_dir_bytes]);
    const mid = "m94-synth-dspark-3l";
    @memcpy(hdr.model_id[0..mid.len], mid);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(img[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));

    const exact_path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ tmp_dir, filename }, 0);
    errdefer allocator.free(exact_path);
    try writeFileBytesPosix(exact_path, img);

    const cfg_json = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "draft_format": "dspark_3l_mtp",
        \\  "hidden_size": {d},
        \\  "vocab_size": {d},
        \\  "num_hidden_layers": 3,
        \\  "num_attention_heads": {d},
        \\  "head_dim": {d},
        \\  "q_lora_rank": {d},
        \\  "kv_lora_rank": {d},
        \\  "o_lora_rank": {d},
        \\  "o_groups": {d},
        \\  "moe_intermediate_size": {d},
        \\  "dspark_n_routed_experts": {d},
        \\  "dspark_routed_top_k": {d},
        \\  "dspark_block_size": 5,
        \\  "dspark_markov_rank": {d},
        \\  "dspark_target_layer_ids": [0, 2, 3],
        \\  "routed_scaling_factor": 1.0,
        \\  "moe_scoring_sqrtsoftplus": true
        \\}}
        \\
    , .{ h, v, n_heads, hd, q_lora, hd, o_lora, o_groups, moe_i, n_exp, top_k, mr });

    return .{
        .amf_path = exact_path,
        .cfg_json = cfg_json,
        .bytes = img,
    };
}

// ---------------------------------------------------------------------------
// Section 1: W28 Header Gate & Schema Audit
// ---------------------------------------------------------------------------

fn inspectAmfBytesW28(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    max_name_out: *u32,
    long_names_out: *u32,
) !void {
    var parsed = try amf.Parsed.parseBytes(allocator, bytes);
    defer parsed.deinit(allocator);
    try parsed.verifyAllBlobsFromBytes(bytes);
    if ((parsed.header.flags & ~amf.KNOWN_HEADER_FLAGS) != 0) return error.InvalidHeaderFlags;
    for (parsed.entries) |entry| {
        const nm = entry.nameSlice();
        const nlen: u32 = @intCast(nm.len);
        max_name_out.* = @max(max_name_out.*, nlen);
        if (nlen > 47) long_names_out.* += 1;
        if (entry.blob_offset % page.APPLE_PAGE_SIZE != 0 or entry.blob_bytes % page.APPLE_PAGE_SIZE != 0) {
            return error.MisalignedBlob;
        }
    }
}

fn inspectAmfFileW28(
    allocator: std.mem.Allocator,
    path_z: [:0]const u8,
    report: *M94Report,
) void {
    report.w28_staged_files_checked += 1;
    var af = amf.AmfFile.open(allocator, path_z) catch {
        report.w28_staged_align_errors += 1;
        return;
    };
    defer af.close(allocator);
    af.verifyAllBlobs(allocator) catch {
        report.w28_staged_crc_errors += 1;
    };
    if ((af.parsed.header.flags & ~amf.KNOWN_HEADER_FLAGS) != 0) {
        report.w28_staged_align_errors += 1;
    }
    report.w28_staged_tensors_checked += @intCast(af.parsed.entries.len);
    for (af.parsed.entries) |entry| {
        const nm = entry.nameSlice();
        const nlen: u32 = @intCast(nm.len);
        report.w28_staged_max_name_bytes = @max(report.w28_staged_max_name_bytes, nlen);
        if (nlen > 47) report.w28_staged_long_names += 1;
        if (entry.blob_offset % page.APPLE_PAGE_SIZE != 0 or entry.blob_bytes % page.APPLE_PAGE_SIZE != 0) {
            report.w28_staged_align_errors += 1;
        }
    }
}

fn runW28GateChecks(
    allocator: std.mem.Allocator,
    tmp_dir: []const u8,
    cfg: M94Config,
    report: *M94Report,
) !void {
    // 1. Always verify synthetic W28 gate on synth qwen4, deepseek41, and 3L DSpark artifacts.
    var s_q4 = try graph.buildSynthQwen4Bundle(allocator, .{ .tmp_dir = tmp_dir });
    defer s_q4.deinit(allocator);
    try inspectAmfBytesW28(allocator, s_q4.amf_bytes, &report.w28_synth_max_name_bytes, &report.w28_synth_long_names);

    var s_ds = try graph.buildSynthDeepSeek41Bundle(allocator, .{ .tmp_dir = tmp_dir, .include_dspark = false });
    defer s_ds.deinit(allocator);
    try inspectAmfBytesW28(allocator, s_ds.amf_bytes, &report.w28_synth_max_name_bytes, &report.w28_synth_long_names);

    var s_dspark = try buildSynthDspark3LArtifact(allocator, tmp_dir, "w28_dspark3l.amf");
    defer s_dspark.deinit(allocator);
    try inspectAmfBytesW28(allocator, s_dspark.bytes, &report.w28_synth_max_name_bytes, &report.w28_synth_long_names);

    report.w28_synth_files_checked = 3;
    report.w28_synth_ok = (report.w28_synth_files_checked == 3) and
        (report.w28_synth_max_name_bytes <= 47) and
        (report.w28_synth_long_names == 0);

    // 2. When staged bundles exist under `models_dir`, verify all 13 staged `.amf` files.
    const models_dir = paths.resolveModelsDir(cfg.models_dir) orelse {
        report.staged_models_present = false;
        report.w28_ok = report.w28_synth_ok;
        return;
    };
    const staged_files = [_][]const u8{
        "qwen4-real-hf-synth/model.amf",
        "qwen4-real-hf-synth/mtp.amf",
        "qwen4-real-hf-synth/vision.amf",
        "qwen4-real-hf-synth/ple-shard-00-of-02.amf",
        "qwen4-real-hf-synth/ple-shard-01-of-02.amf",
        "qwen4-tiny-random-bf16/model.amf",
        "qwen4-tiny-random-bf16/vision.amf",
        "qwen4-tiny-random-bf16/ple-shard-00-of-02.amf",
        "qwen4-tiny-random-bf16/ple-shard-01-of-02.amf",
        "deepseek41-real-hf-synth/model.amf",
        "deepseek41-real-hf-synth/dspark.amf",
        "deepseek41-real-hf-synth/engram-shard-00-of-02.amf",
        "deepseek41-real-hf-synth/engram-shard-01-of-02.amf",
    };

    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    var found_any = false;
    for (staged_files) |subpath| {
        if (resolveModelFileCandidateZ(&pbuf, models_dir, subpath)) |full_z| {
            found_any = true;
            inspectAmfFileW28(allocator, full_z, report);
        }
    }
    report.staged_models_present = found_any;
    if (found_any) {
        report.w28_ok = report.w28_synth_ok and
            (report.w28_staged_files_checked > 0) and
            (report.w28_staged_max_name_bytes <= 47) and
            (report.w28_staged_long_names == 0) and
            (report.w28_staged_crc_errors == 0) and
            (report.w28_staged_align_errors == 0);
    } else {
        report.w28_ok = report.w28_synth_ok;
    }
}

// ---------------------------------------------------------------------------
// Section 2: Qwen3.8-Flash-Next (W10 / G10 / C10 / MTP / W20 Vision)
// ---------------------------------------------------------------------------

fn runQwen4KernelParity(
    allocator: std.mem.Allocator,
    device: metal.Device,
    compute: *metal.Compute,
    pipes: hybrid.HybridPipelines,
    budget: budget_mod.Budget,
    report: *M94Report,
) !void {
    const arena_cap = try page.alignUp(1024 * 1024);
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();
    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    try compute.makeResident(&.{buf});
    defer {
        compute.removeResident(&.{buf}) catch |err| std.debug.panic("removeResident failed: {}", .{err});
    }
    const bind = struct {
        fn call(b: metal.Buffer, a: *const arena_mod.WiredArena, ptr: [*]const u8) metal.BufferBinding {
            return .{ .buffer = b, .offset = a.offsetOf(ptr) };
        }
    }.call;

    const m: u32 = 6;
    const hc: u32 = 4;
    const h: u32 = 64;
    const total_hc = hc * h;

    // 1. `hc_count = 4` HC norm + mix_reduce + inject GPU vs CPU-f32 parity
    const h_raw = try arena.allocAligned(m * total_hc * 2, 64);
    const hn_raw = try arena.allocAligned(m * total_hc * 2, 64);
    const nw_raw = try arena.allocAligned(total_hc * 2, 64);
    const wmix_raw = try arena.allocAligned(m * total_hc * 2, 64);
    const inj_in_raw = try arena.allocAligned(m * hc * 2, 64);
    const mixed_raw = try arena.allocAligned(m * h * 2, 64);
    const inj_out_raw = try arena.allocAligned(m * hc * 4, 64);
    const sub_raw = try arena.allocAligned(m * h * 2, 64);

    const h_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, h_raw));
    const nw_s: []u16 = @alignCast(std.mem.bytesAsSlice(u16, nw_raw));
    const wmix_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wmix_raw));
    const inj_in_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, inj_in_raw));
    const mixed_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, mixed_raw));
    const sub_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, sub_raw));

    var h_init: [m * total_hc]f32 = undefined;
    for (0..m * total_hc) |i| {
        const v: f32 = @sin(@as(f32, @floatFromInt(i + 1)) * 0.13) * 0.7;
        h_s[i] = @floatCast(v);
        h_init[i] = @floatCast(h_s[i]);
        wmix_s[i] = @floatCast(@cos(@as(f32, @floatFromInt(i + 3)) * 0.17) * 1.1);
    }
    for (0..total_hc) |i| nw_s[i] = quant.f32ToBf16(@sin(@as(f32, @floatFromInt(i + 7)) * 0.09) * 0.25);
    for (0..m * hc) |i| inj_in_s[i] = @floatCast(@sin(@as(f32, @floatFromInt(i + 11)) * 0.31) * 1.5);
    for (0..m * h) |i| sub_s[i] = @floatCast(@cos(@as(f32, @floatFromInt(i + 5)) * 0.21) * 0.5);

    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeQwen4HcNorm(&frame, .{
            .x = bind(buf, &arena, h_raw.ptr),
            .w = bind(buf, &arena, nw_raw.ptr),
            .y = bind(buf, &arena, hn_raw.ptr),
            .m = m,
            .num_groups = hc,
            .group_size = h,
            .w_dtype = .bf16,
            .unit_offset = true,
            .rms_eps = 1e-6,
        }, false);
        try pipes.encodeQwen4HcMixReduce(&frame, .{
            .h_normed = bind(buf, &arena, hn_raw.ptr),
            .w_mix = bind(buf, &arena, wmix_raw.ptr),
            .inj_raw = bind(buf, &arena, inj_in_raw.ptr),
            .mixed_out = bind(buf, &arena, mixed_raw.ptr),
            .inj_out = bind(buf, &arena, inj_out_raw.ptr),
            .m = m,
            .hc_count = hc,
            .hidden_size = h,
        }, true);
        try pipes.encodeQwen4HcInject(&frame, .{
            .h_streams = bind(buf, &arena, h_raw.ptr),
            .sub_out = bind(buf, &arena, sub_raw.ptr),
            .inj = bind(buf, &arena, inj_out_raw.ptr),
            .m = m,
            .hc_count = hc,
            .hidden_size = h,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    const hc_ref = try allocator.alloc(f32, m * (h + total_hc));
    defer allocator.free(hc_ref);
    const hc_got = try allocator.alloc(f32, m * (h + total_hc));
    defer allocator.free(hc_got);
    for (0..m) |tok| {
        var hn_ref: [total_hc]f32 = undefined;
        for (0..hc) |b| {
            var sq: f32 = 0.0;
            for (0..h) |d| {
                const v = h_init[tok * total_hc + b * h + d];
                sq += v * v;
            }
            const inv_r = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(h)) + 1e-6);
            for (0..h) |d| {
                const wt = 1.0 + quant.bf16ToF32(nw_s[b * h + d]);
                hn_ref[b * h + d] = (h_init[tok * total_hc + b * h + d] * inv_r) * wt;
            }
        }
        for (0..h) |d| {
            var acc: f32 = 0.0;
            for (0..hc) |b| {
                acc += sigmoidF32(@as(f32, @floatCast(wmix_s[tok * total_hc + b * h + d]))) * hn_ref[b * h + d];
            }
            hc_ref[tok * h + d] = acc * 0.25;
            hc_got[tok * h + d] = @floatCast(mixed_s[tok * h + d]);
        }
        for (0..hc) |b| {
            const ref_inj = 2.0 * sigmoidF32(@as(f32, @floatCast(inj_in_s[tok * hc + b])) * 0.25);
            for (0..h) |d| {
                const off = m * h + tok * total_hc + b * h + d;
                hc_ref[off] = h_init[tok * total_hc + b * h + d] + ref_inj * @as(f32, @floatCast(sub_s[tok * h + d]));
                hc_got[off] = @floatCast(h_s[tok * total_hc + b * h + d]);
            }
        }
    }
    const hc_stat = cosineF64(hc_ref, hc_got);
    report.qwen4_hc_cos_sim = hc_stat.cos;
    report.qwen4_hc_max_err = hc_stat.max_err;

    // 2. Dense PLE gate + causal dilated Conv1D (`K=4, d=3`): seq vs tree parity
    const ple_k_raw = try arena.allocAligned(m * total_hc * 2, 64);
    const ple_v_raw = try arena.allocAligned(m * h * 2, 64);
    const ple_gv_raw = try arena.allocAligned(m * total_hc * 2, 64);
    const ple_gvn_raw = try arena.allocAligned(m * total_hc * 2, 64);
    const ple_cw_raw = try arena.allocAligned(total_hc * 4 * 2, 64);
    const ple_cs_raw = try arena.allocAligned(9 * total_hc * 2, 64);
    const h_seq_raw = try arena.allocAligned(m * total_hc * 2, 64);
    const h_tree_raw = try arena.allocAligned(m * total_hc * 2, 64);
    const masks_raw = try arena.allocAligned(m * 8, 64);

    const ple_k_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, ple_k_raw));
    const ple_v_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, ple_v_raw));
    const ple_cw_s: []u16 = @alignCast(std.mem.bytesAsSlice(u16, ple_cw_raw));
    const ple_cs_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, ple_cs_raw));
    const h_seq_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, h_seq_raw));
    const h_tree_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, h_tree_raw));
    const masks_s: []u64 = @alignCast(std.mem.bytesAsSlice(u64, masks_raw));

    for (0..m * total_hc) |i| {
        ple_k_s[i] = @floatCast(@sin(@as(f32, @floatFromInt(i + 2)) * 0.19) * 0.8);
        h_seq_s[i] = h_s[i];
        h_tree_s[i] = h_s[i];
    }
    for (0..m * h) |i| ple_v_s[i] = @floatCast(@cos(@as(f32, @floatFromInt(i + 9)) * 0.11) * 0.6);
    for (0..total_hc * 4) |i| ple_cw_s[i] = quant.f32ToBf16(@sin(@as(f32, @floatFromInt(i + 1)) * 0.07) * 0.3);
    for (0..9 * total_hc) |i| ple_cs_s[i] = @floatCast(@cos(@as(f32, @floatFromInt(i + 4)) * 0.05) * 0.2);
    for (0..m) |i| masks_s[i] = (@as(u64, 1) << @intCast(i + 1)) - 1;

    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeQwen4PleGate(&frame, .{
            .h_streams = bind(buf, &arena, h_seq_raw.ptr),
            .k_raw = bind(buf, &arena, ple_k_raw.ptr),
            .value_in = bind(buf, &arena, ple_v_raw.ptr),
            .norm_key_w = bind(buf, &arena, nw_raw.ptr),
            .norm_query_w = bind(buf, &arena, nw_raw.ptr),
            .norm_conv_w = bind(buf, &arena, nw_raw.ptr),
            .gv_out = bind(buf, &arena, ple_gv_raw.ptr),
            .gv_normed_out = bind(buf, &arena, ple_gvn_raw.ptr),
            .m = m,
            .hc_count = hc,
            .hidden_size = h,
            .norm_dtype = .bf16,
            .rms_eps = 1e-6,
        }, false);
        try pipes.encodeQwen4PleDilatedConvAdd(&frame, .{
            .h_streams = bind(buf, &arena, h_seq_raw.ptr),
            .gv_in = bind(buf, &arena, ple_gv_raw.ptr),
            .gv_normed = bind(buf, &arena, ple_gvn_raw.ptr),
            .conv_w = bind(buf, &arena, ple_cw_raw.ptr),
            .conv_state_in = bind(buf, &arena, ple_cs_raw.ptr),
            .m = m,
            .channels = total_hc,
            .kernel_size = 4,
            .dilation = 3,
            .w_dtype = .bf16,
            .is_tree = false,
        }, true);
        try pipes.encodeQwen4PleDilatedConvAdd(&frame, .{
            .h_streams = bind(buf, &arena, h_tree_raw.ptr),
            .gv_in = bind(buf, &arena, ple_gv_raw.ptr),
            .gv_normed = bind(buf, &arena, ple_gvn_raw.ptr),
            .conv_w = bind(buf, &arena, ple_cw_raw.ptr),
            .conv_state_in = bind(buf, &arena, ple_cs_raw.ptr),
            .ancestor_masks = bind(buf, &arena, masks_raw.ptr),
            .m = m,
            .channels = total_hc,
            .kernel_size = 4,
            .dilation = 3,
            .w_dtype = .bf16,
            .is_tree = true,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    var ple_max_e: f64 = 0.0;
    for (h_seq_s, h_tree_s) |sa, sb| {
        ple_max_e = @max(ple_max_e, @abs(@as(f64, @floatCast(sa)) - @as(f64, @floatCast(sb))));
    }
    report.qwen4_ple_cos_sim = cosineF16(h_seq_s, h_tree_s);
    report.qwen4_ple_max_err = ple_max_e;

    // 3. QSA / GDN elementwise sigmoid gate + MLA latent sigmoid parity check vs CPU f32
    const n_q_heads: u32 = 4;
    const d_kv: u32 = 32;
    const qn_raw = try arena.allocAligned(m * n_q_heads * d_kv * 2, 64);
    const lat_raw = try arena.allocAligned(6 * d_kv * 2, 64);
    const gate_ew_raw = try arena.allocAligned(m * n_q_heads * d_kv * 2, 64);
    const mla_out_raw = try arena.allocAligned(m * n_q_heads * d_kv * 2, 64);

    const qn_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, qn_raw));
    const lat_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, lat_raw));
    const gate_ew_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate_ew_raw));
    const mla_out_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, mla_out_raw));

    for (0..m * n_q_heads * d_kv) |i| {
        qn_s[i] = @floatCast(deterministicF32(94401, i, 0.5, 0.0));
        gate_ew_s[i] = @floatCast(deterministicF32(94402, i, 1.4, 0.0));
    }
    for (0..5 * d_kv) |i| lat_s[i] = @floatCast(deterministicF32(94403, i, 0.6, 0.0));

    var mla_kv = try hybrid.G27MlaKvArena.init(&arena, .{
        .num_layers = 1,
        .max_seq_len = 16,
        .kv_lora_rank = d_kv,
        .qk_rope_head_dim = 0,
    });

    const scale_q: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_kv)));
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG27MlaKvWrite(&frame, .{
            .latent_in = bind(buf, &arena, lat_raw.ptr),
            .rope_in = null,
            .kv_cache = try mla_kv.layerBinding(buf, 0),
            .num_tokens = 5,
            .kv_lora_rank = d_kv,
            .qk_rope_head_dim = 0,
            .start_pos = 0,
        }, false);
        try pipes.encodeG28MlaFullAttn(&frame, .{
            .q_nope = bind(buf, &arena, qn_raw.ptr),
            .q_pe = null,
            .kv_cache = try mla_kv.layerBinding(buf, 0),
            .out = bind(buf, &arena, mla_out_raw.ptr),
            .num_rows = 1,
            .num_q_heads = n_q_heads,
            .kv_lora_rank = d_kv,
            .qk_rope_head_dim = 0,
            .committed_kv_len = 5,
            .causal_base_pos = 4,
            .is_tree = false,
            .scale = scale_q,
        }, true);
        try pipes.encodeG28MlaOutGate(&frame, .{
            .attn_out = bind(buf, &arena, mla_out_raw.ptr),
            .gate = bind(buf, &arena, gate_ew_raw.ptr),
            .num_rows = 1,
            .num_heads = n_q_heads,
            .v_head_dim = d_kv,
            .head_wise = false,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    const sig_ref = try allocator.alloc(f32, n_q_heads * d_kv);
    defer allocator.free(sig_ref);
    const sig_got = try allocator.alloc(f32, n_q_heads * d_kv);
    defer allocator.free(sig_got);
    for (0..n_q_heads) |hd| {
        var scores: [5]f32 = undefined;
        var mx: f32 = -std.math.inf(f32);
        for (0..5) |k| {
            var dv: f32 = 0.0;
            for (0..d_kv) |d| dv += @as(f32, @floatCast(qn_s[hd * d_kv + d])) * @as(f32, @floatCast(lat_s[k * d_kv + d]));
            scores[k] = dv * scale_q;
            mx = @max(mx, scores[k]);
        }
        var den: f32 = 0.0;
        for (0..5) |k| {
            scores[k] = @exp(scores[k] - mx);
            den += scores[k];
        }
        for (0..d_kv) |d| {
            var acc: f32 = 0.0;
            for (0..5) |k| acc += (scores[k] / den) * @as(f32, @floatCast(lat_s[k * d_kv + d]));
            const gi = hd * d_kv + d;
            sig_ref[gi] = acc * sigmoidF32(@as(f32, @floatCast(gate_ew_s[gi])));
            sig_got[gi] = @floatCast(mla_out_s[gi]);
        }
    }
    const sg_stat = cosineF64(sig_ref, sig_got);
    report.qwen4_gdn_sig_gate_cos_sim = sg_stat.cos;
    report.qwen4_gdn_sig_gate_max_err = sg_stat.max_err;

    report.qwen4_kernel_parity_ok = (report.qwen4_hc_cos_sim >= 0.9999) and
        (report.qwen4_ple_cos_sim >= 0.99999) and
        (report.qwen4_ple_max_err <= 1e-4) and
        (report.qwen4_gdn_sig_gate_cos_sim >= 0.9999);
}

fn runQwen4VisionCheck(
    allocator: std.mem.Allocator,
    tmp_dir: []const u8,
    vis_path_opt: ?[:0]const u8,
    model_path_opt: ?[:0]const u8,
    gcfg_opt: ?graph.GraphConfig,
    ple_paths: []const [:0]const u8,
    report: *M94Report,
) !void {
    if (vis_path_opt != null and model_path_opt != null and gcfg_opt != null) {
        var vis_m = try graph.GraphModel.loadFromFile(allocator, model_path_opt.?, gcfg_opt.?, .{
            .max_seq_len = 64,
            .max_tree_nodes = 8,
            .ple_paths = ple_paths,
            .vision_amf_path = vis_path_opt.?,
            .vision_max_patches = 64,
            .startup_canary = false,
        });
        defer vis_m.deinit(allocator);

        // `qwen4-tiny-random-bf16` vision uses patch_size=2, t_patch=2, spatial_merge=2,
        // `num_position_embeddings = 16` -> two 8x8 frames produce 4x4=16 patches -> 4 vision tokens!
        var f0 = try metal.VisionFrame.init(vis_m.device, .bgra8, 8, 8);
        defer f0.close();
        try fillDeterministicVisionFrame(&f0, 220, 40, 20);
        var f1 = try metal.VisionFrame.init(vis_m.device, .bgra8, 8, 8);
        defer f1.close();
        try fillDeterministicVisionFrame(&f1, 20, 40, 220);

        const prefix = [_]u32{ 1, 2 };
        const suffix = [_]u32{3};
        const mm = try vis_m.prefillMultimodal(allocator, &prefix, &suffix, &.{ f0, f1 }, .{
            .anyres_mode = .smart_resize,
            .anyres_min_pixels = 64,
        });
        report.qwen4_vision_tokens = mm.vision_tokens;
        report.qwen4_vision_sidecar_ok = mm.vision_tower_active and
            mm.all_frames_zero_copy and
            (mm.vision_tokens > 0) and
            (mm.first_token < gcfg_opt.?.vocab_size);
        return;
    }

    // Offline synthetic fallback when staged bundles are absent.
    var s_q4 = try graph.buildSynthQwen4Bundle(allocator, .{ .tmp_dir = tmp_dir });
    defer s_q4.deinit(allocator);
    const vis_bytes = try graph.buildSynthVisionTowerAmfBytes(allocator, .{
        .num_layers = 2,
        .vis_hidden = 64,
        .vis_intermediate = 128,
        .lm_hidden = s_q4.cfg.hidden_size,
        .merger_hidden = 128,
        .patch_size = 16,
        .t_patch = 2,
        .spatial_merge = 2,
        .q4_qkv = true,
    });
    defer allocator.free(vis_bytes);
    const synth_vis_path = try std.fmt.allocPrintSentinel(allocator, "{s}/synth_q4_vis.amf", .{tmp_dir}, 0);
    defer allocator.free(synth_vis_path);
    try writeFileBytesPosix(synth_vis_path, vis_bytes);

    var vis_m = try graph.GraphModel.loadFromBytes(allocator, s_q4.amf_bytes, s_q4.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .vision_amf_path = synth_vis_path,
        .vision_max_patches = 64,
        .startup_canary = false,
    });
    defer vis_m.deinit(allocator);
    var f0 = try metal.VisionFrame.init(vis_m.device, .bgra8, 64, 64);
    defer f0.close();
    try fillDeterministicVisionFrame(&f0, 200, 30, 15);
    var f1 = try metal.VisionFrame.init(vis_m.device, .bgra8, 64, 64);
    defer f1.close();
    try fillDeterministicVisionFrame(&f1, 15, 30, 200);
    const prefix = [_]u32{ 1, 2 };
    const suffix = [_]u32{3};
    const mm = try vis_m.prefillMultimodal(allocator, &prefix, &suffix, &.{ f0, f1 }, .{
        .anyres_mode = .smart_resize,
    });
    report.qwen4_vision_tokens = mm.vision_tokens;
    report.qwen4_vision_sidecar_ok = mm.vision_tower_active and
        mm.all_frames_zero_copy and
        (mm.vision_tokens > 0) and
        (mm.first_token < s_q4.cfg.vocab_size);
}

fn runQwen4E2eSynthAndStaged(
    allocator: std.mem.Allocator,
    tmp_dir: []const u8,
    cfg: M94Config,
    report: *M94Report,
) !void {
    // 1. Always verify synthetic `qwen4_exp` `R=2` PLE zero-NVMe-reuse + `wide` vs `tree64`.
    {
        var s_q4 = try graph.buildSynthQwen4Bundle(allocator, .{
            .hidden_size = 64,
            .vocab_size = 128,
            .num_attention_heads = 2,
            .num_key_value_heads = 1,
            .head_dim = 64,
            .moe_inter = 64,
            .shared_inter = 64,
            .num_experts = 8,
            .top_k = 2,
            .max_loops = 2,
            .ple_num_shards = 2,
            .ple_rows = 64,
            .tmp_dir = tmp_dir,
        });
        defer s_q4.deinit(allocator);
        const ple_c: []const [:0]const u8 = s_q4.ple_paths;

        var m_w = try graph.GraphModel.loadFromBytes(allocator, s_q4.amf_bytes, s_q4.cfg, .{
            .max_seq_len = 64,
            .max_tree_nodes = 8,
            .gpu_prefill_mode = .wide,
            .ple_paths = ple_c,
            .startup_canary = false,
        });
        defer m_w.deinit(allocator);
        var m_t = try graph.GraphModel.loadFromBytes(allocator, s_q4.amf_bytes, s_q4.cfg, .{
            .max_seq_len = 64,
            .max_tree_nodes = 8,
            .gpu_prefill_mode = .tree64,
            .ple_paths = ple_c,
            .startup_canary = false,
        });
        defer m_t.deinit(allocator);

        const prompt = [_]u32{ 3, 7, 11, 19, 23 };
        const tw = try m_w.prefillChunk(&prompt, 0, true);
        const tt = try m_t.prefillChunk(&prompt, 0, true);
        const dw = try m_w.stepDecode(tw, @intCast(prompt.len));
        const dt = try m_t.stepDecode(tt, @intCast(prompt.len));
        const st = m_w.loopIsolationStats();
        report.qwen4_synth_ple_r_ge1_nvme_reads = st.ple_r_ge1_nvme_reads;
        report.qwen4_synth_ple_r_ge1_reuses = st.ple_r_ge1_reuses;
        report.qwen4_synth_wide_tree_ok = (tw == tt) and
            (dw == dt) and
            (st.ple_r_ge1_nvme_reads == 0) and
            (st.ple_r_ge1_reuses > 0) and
            allFiniteF16(m_w.readLastLogitsF16()[0..s_q4.cfg.vocab_size]);
    }

    // 2. Staged `qwen4-real-hf-synth` + `qwen4-tiny-random-bf16` checks (if `models_dir` exists).
    const models_dir = paths.resolveModelsDir(cfg.models_dir);
    if (models_dir) |mdir| {
        var m_buf: [std.fs.max_path_bytes]u8 = undefined;
        var s0_buf: [std.fs.max_path_bytes]u8 = undefined;
        var s1_buf: [std.fs.max_path_bytes]u8 = undefined;
        var mtp_buf: [std.fs.max_path_bytes]u8 = undefined;
        var mcfg_buf: [std.fs.max_path_bytes]u8 = undefined;

        if (resolveModelFileCandidateZ(&m_buf, mdir, "qwen4-real-hf-synth/model.amf")) |model_path| {
            const s0 = resolveModelFileCandidateZ(&s0_buf, mdir, "qwen4-real-hf-synth/ple-shard-00-of-02.amf");
            const s1 = resolveModelFileCandidateZ(&s1_buf, mdir, "qwen4-real-hf-synth/ple-shard-01-of-02.amf");
            const mtp_path = resolveModelFileCandidateZ(&mtp_buf, mdir, "qwen4-real-hf-synth/mtp.amf");
            const mtp_cfg_path = resolveModelFileCandidateZ(&mcfg_buf, mdir, "qwen4-real-hf-synth/mtp_config.json");

            var af = try amf.AmfFile.open(allocator, model_path);
            defer af.close(allocator);
            var cfg_meta = try amf.resolveConfigJsonAlloc(allocator, &af, model_path, null);
            defer cfg_meta.deinit(allocator);
            var gcfg = try graph.GraphConfig.fromJson(allocator, cfg_meta.bytes);
            defer gcfg.deinit(allocator);

            var ple_list: [2][:0]const u8 = undefined;
            var ple_len: usize = 0;
            if (s0) |p0| {
                ple_list[ple_len] = p0;
                ple_len += 1;
            }
            if (s1) |p1| {
                ple_list[ple_len] = p1;
                ple_len += 1;
            }
            const ple_slice: []const [:0]const u8 = ple_list[0..ple_len];

            const prompt = [_]u32{ 3, 7, 11, 19, 23, 31 };
            const n_gen: usize = @max(@as(usize, cfg.e2e_tokens), 6);
            const base_toks = try allocator.alloc(u32, n_gen);
            defer allocator.free(base_toks);

            var m_base = try graph.GraphModel.loadFromFile(allocator, model_path, gcfg, .{
                .max_seq_len = 128,
                .max_tree_nodes = 32,
                .gpu_prefill_mode = .wide,
                .ple_paths = ple_slice,
                .startup_canary = false,
            });
            defer m_base.deinit(allocator);

            const dec_res = try benchGreedyDecode(&m_base, &prompt, @intCast(n_gen), base_toks, true);
            report.qwen4_staged_decode_tok_s = dec_res.tok_s;
            report.qwen4_staged_malloc_delta = dec_res.malloc_delta;

            // Check `[4*H]` four-stream hidden tap!
            var hs4_norm: f64 = 0.0;
            if (m_base.lastFourStreamHiddenF32()) |hs4| {
                for (hs4) |v| {
                    if (!std.math.isFinite(v)) hs4_norm = -1.0;
                    hs4_norm += @as(f64, v) * @as(f64, v);
                }
            }
            report.qwen4_synth_four_stream_norm = hs4_norm;

            // Check `R = 2` PLE zero-NVMe-reuse on real-HF staged bundle.
            var m_r2 = try graph.GraphModel.loadFromFile(allocator, model_path, gcfg, .{
                .max_seq_len = 64,
                .max_tree_nodes = 8,
                .gpu_prefill_mode = .wide,
                .ple_paths = ple_slice,
                .max_loops_override = 2,
                .startup_canary = false,
            });
            defer m_r2.deinit(allocator);
            const r2p = try m_r2.prefillChunk(&prompt, 0, true);
            _ = try m_r2.stepDecode(r2p, @intCast(prompt.len));
            report.qwen4_staged_ple_r_ge1_nvme_reads = m_r2.loop_stats.ple_r_ge1_nvme_reads;
            report.qwen4_staged_ple_r_ge1_reuses = m_r2.loop_stats.ple_r_ge1_reuses;

            // Speculative `decode == spec-decode -M mtp` parity + D13 L2 SME2 parity + 512E D7 check.
            var mtp_parity_ok = false;
            var sme2_ok = false;
            var d7_512e_ok = false;

            if (mtp_path != null and mtp_cfg_path != null) {
                const mtp_cfg_bytes = (try readFileBytesPosixAlloc(allocator, mtp_cfg_path.?, 1024 * 1024)) orelse return error.MissingConfig;
                defer allocator.free(mtp_cfg_bytes);
                const mtp_cfg = try draft.DraftConfig.fromJson(allocator, mtp_cfg_bytes);

                var m_spec = try graph.GraphModel.loadFromFile(allocator, model_path, gcfg, .{
                    .max_seq_len = 128,
                    .max_tree_nodes = 32,
                    .gpu_prefill_mode = .wide,
                    .ple_paths = ple_slice,
                    .enable_feature_taps = true,
                    .startup_canary = false,
                });
                defer m_spec.deinit(allocator);

                var dh_sme = try draft.DraftHead.loadFromFileForGraph(allocator, &m_spec, mtp_path.?, mtp_cfg, .{
                    .placement = .cpu_sme2,
                    .max_seq_len = 128,
                    .max_tree_slots = 32,
                });
                defer dh_sme.deinitForGraph(allocator, &m_spec);

                var eng = try engine_mod.SpecEngine.initWithGraph(allocator, &m_spec, &dh_sme, .{
                    .mode = .mtp,
                    .max_tree_nodes = 24,
                    .max_draft_depth = 4,
                    .enable_pruning = true,
                    .temperature = 0.0,
                });
                defer eng.deinit(allocator);

                const spec_toks = try allocator.alloc(u32, n_gen);
                defer allocator.free(spec_toks);
                const sres = try evalSpecDecodeRun(&eng, &prompt, spec_toks);
                report.qwen4_staged_mtp_tok_s = sres.tok_s;
                report.qwen4_staged_mtp_mean_accept = sres.mean_accept;
                report.qwen4_staged_mtp_speedup = if (dec_res.tok_s > 0.0) sres.tok_s / dec_res.tok_s else 0.0;
                report.qwen4_staged_mtp_sme2_evals = dh_sme.sme2_eval_count;

                mtp_parity_ok = (sres.produced == n_gen) and std.mem.eql(u32, base_toks, spec_toks);

                // D13 L2 SME2 vs GPU parity on `qwen4_exp_mtp` `fc_hidden`
                var dh_gpu = try draft.DraftHead.loadFromFileForGraph(allocator, &m_spec, mtp_path.?, mtp_cfg, .{
                    .placement = .gpu,
                    .max_seq_len = 128,
                    .max_tree_slots = 32,
                });
                defer dh_gpu.deinitForGraph(allocator, &m_spec);
                var dh_s2 = try draft.DraftHead.loadFromFileForGraph(allocator, &m_spec, mtp_path.?, mtp_cfg, .{
                    .placement = .cpu_sme2,
                    .max_seq_len = 128,
                    .max_tree_slots = 32,
                });
                defer dh_s2.deinitForGraph(allocator, &m_spec);

                m_spec.resetKvCache();
                _ = try m_spec.prefillChunkWithOptions(&prompt, 0, .{ .compute_logits = true, .preserve_all_top_hidden = true });
                try dh_gpu.syncAcceptedForGraph(&m_spec, &prompt, m_spec.accepted_top_bind, 0, 4);
                try dh_s2.syncAcceptedForGraph(&m_spec, &prompt, m_spec.accepted_top_bind, 0, 4);

                const t_toks = [_]u32{ 17, 29 };
                const t_pars = [_]u32{ 0, 0 };
                const t_pos = [_]u32{ @intCast(prompt.len), @intCast(prompt.len) };
                const t_masks = [_]u64{ 0b01, 0b10 };
                var c_g: [2 * 4]moe.TopKCandidate = undefined;
                var c_s: [2 * 4]moe.TopKCandidate = undefined;
                try dh_gpu.stepTreeBatchForGraph(&m_spec, &t_toks, &t_pars, &t_pos, &t_masks, 0, 4, &c_g);
                try dh_s2.stepTreeBatchForGraph(&m_spec, &t_toks, &t_pars, &t_pos, &t_masks, 0, 4, &c_s);

                const h_d: usize = gcfg.hidden_size;
                var dot: f64 = 0.0;
                var na: f64 = 0.0;
                var nb: f64 = 0.0;
                for (0..3 * h_d) |i| {
                    const a: f64 = @floatCast(dh_gpu.hidden_pool_host[i]);
                    const b: f64 = @floatCast(dh_s2.hidden_pool_host[i]);
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                const cos_sme = if (na > 0.0 and nb > 0.0) dot / @sqrt(na * nb) else 0.0;
                report.qwen4_staged_mtp_sme2_cos_sim = cos_sme;
                sme2_ok = (cos_sme >= 0.999) and (dh_s2.sme2_eval_count >= 2) and (dh_s2.gpu_eh_proj_dispatches == 0);

                // D7 `512E` router pruning + `predictMtpRouterExpertsForTokens` check
                const mtp_l = try eng.predictMtpRouterExpertsForTokens(&dh_gpu, &m_spec, &prompt);
                var r_scratch: [tree_mod.ExpertBitset.MAX_EXPERTS]f32 = undefined;
                const ne512: u32 = 512;
                const rw512 = try allocator.alloc(f32, ne512 * h_d);
                defer allocator.free(rw512);
                @memset(rw512, 0.0);
                const winners = [_]u32{ 3, 41, 77, 128, 205, 256, 319, 400, 488, 511 };
                for (winners, 0..) |e, rank| rw512[@as(usize, e) * h_d + 0] = 10.0 - @as(f32, @floatFromInt(rank)) * 0.2;
                const probe512 = try allocator.alloc(f32, h_d);
                defer allocator.free(probe512);
                @memset(probe512, 0.0);
                probe512[0] = 1.0;
                const bs512 = tree_mod.predictRouterTopK(std.mem.sliceAsBytes(rw512), .f32, probe512, ne512, 10, null, false, &r_scratch);
                var all_w = (bs512.count() == 10);
                for (winners) |e| {
                    if (!bs512.contains(e)) all_w = false;
                }
                d7_512e_ok = (mtp_l == 1) and all_w;
            }

            report.qwen4_staged_checked = true;
            report.qwen4_staged_mtp_parity_ok = mtp_parity_ok;
            report.qwen4_staged_mtp_512e_pruned_ok = d7_512e_ok;
            const malloc_ok = report.mtl_validation_active or (dec_res.malloc_delta <= 0);
            report.qwen4_staged_ok = (hs4_norm > 0.0) and
                (m_base.has_qwen4_hc and m_base.has_qwen4_ple and m_base.has_qsa_real_idx) and
                (report.qwen4_staged_ple_r_ge1_nvme_reads == 0) and
                (report.qwen4_staged_ple_r_ge1_reuses > 0) and
                mtp_parity_ok and sme2_ok and d7_512e_ok and malloc_ok;
        }

        // Check `qwen4-tiny-random-bf16` + `vision.amf` sidecar!
        var tm_buf: [std.fs.max_path_bytes]u8 = undefined;
        var ts0_buf: [std.fs.max_path_bytes]u8 = undefined;
        var ts1_buf: [std.fs.max_path_bytes]u8 = undefined;
        var tvis_buf: [std.fs.max_path_bytes]u8 = undefined;
        if (resolveModelFileCandidateZ(&tm_buf, mdir, "qwen4-tiny-random-bf16/model.amf")) |tiny_path| {
            const ts0 = resolveModelFileCandidateZ(&ts0_buf, mdir, "qwen4-tiny-random-bf16/ple-shard-00-of-02.amf");
            const ts1 = resolveModelFileCandidateZ(&ts1_buf, mdir, "qwen4-tiny-random-bf16/ple-shard-01-of-02.amf");
            const tvis = resolveModelFileCandidateZ(&tvis_buf, mdir, "qwen4-tiny-random-bf16/vision.amf");

            var af_t = try amf.AmfFile.open(allocator, tiny_path);
            defer af_t.close(allocator);
            var cfg_t_meta = try amf.resolveConfigJsonAlloc(allocator, &af_t, tiny_path, null);
            defer cfg_t_meta.deinit(allocator);
            var tcfg = try graph.GraphConfig.fromJson(allocator, cfg_t_meta.bytes);
            defer tcfg.deinit(allocator);

            var tple_list: [2][:0]const u8 = undefined;
            var tple_len: usize = 0;
            if (ts0) |p0| {
                tple_list[tple_len] = p0;
                tple_len += 1;
            }
            if (ts1) |p1| {
                tple_list[tple_len] = p1;
                tple_len += 1;
            }
            const tple_slice: []const [:0]const u8 = tple_list[0..tple_len];

            var mt_w = try graph.GraphModel.loadFromFile(allocator, tiny_path, tcfg, .{
                .max_seq_len = 64,
                .max_tree_nodes = 8,
                .gpu_prefill_mode = .wide,
                .ple_paths = tple_slice,
                .startup_canary = false,
            });
            defer mt_w.deinit(allocator);
            var mt_t = try graph.GraphModel.loadFromFile(allocator, tiny_path, tcfg, .{
                .max_seq_len = 64,
                .max_tree_nodes = 8,
                .gpu_prefill_mode = .tree64,
                .ple_paths = tple_slice,
                .startup_canary = false,
            });
            defer mt_t.deinit(allocator);

            const t_prompt = [_]u32{ 1, 5, 9, 13, 17, 21 };
            const tw = try mt_w.prefillChunk(&t_prompt, 0, true);
            const tt = try mt_t.prefillChunk(&t_prompt, 0, true);
            const dw = try mt_w.stepDecode(tw, @intCast(t_prompt.len));
            const dt = try mt_t.stepDecode(tt, @intCast(t_prompt.len));
            report.qwen4_tiny_checked = true;
            report.qwen4_tiny_omitted_norms_ok = (!mt_w.has_final_norm) and
                mt_w.has_qwen4_hc and (mt_w.qwen4_hc_mixer != null) and
                (tw == tt) and (dw == dt) and
                allFiniteF16(mt_w.readLastLogitsF16()[0..tcfg.vocab_size]);

            try runQwen4VisionCheck(allocator, tmp_dir, tvis, tiny_path, tcfg, tple_slice, report);
        } else {
            try runQwen4VisionCheck(allocator, tmp_dir, null, null, null, &.{}, report);
        }
    } else {
        try runQwen4VisionCheck(allocator, tmp_dir, null, null, null, &.{}, report);
    }

    const staged_ok = if (report.qwen4_staged_checked) report.qwen4_staged_ok else true;
    const tiny_ok = if (report.qwen4_tiny_checked) report.qwen4_tiny_omitted_norms_ok else true;
    report.qwen4_section_ok = report.qwen4_kernel_parity_ok and
        report.qwen4_synth_wide_tree_ok and
        report.qwen4_vision_sidecar_ok and
        staged_ok and
        tiny_ok;
}

// ---------------------------------------------------------------------------
// Section 3: DeepSeek-V4.1-Flash-1017 (W11 / G10 / G11 / D10 / 24gb-m4 / W30)
// ---------------------------------------------------------------------------

fn verifyQuantFormatParity(
    allocator: std.mem.Allocator,
    device: metal.Device,
    lib: metal.Library,
    compute: *metal.Compute,
    dtype: qformats.DType,
    seed: u64,
) !CosineStats {
    const specs = [_]qformats.FormatSpec{.{ .dtype = dtype, .group_size = 32 }};
    const pipes = try qformats.QuantFormatPipelines.initWithOptions(device, lib, .{
        .specs = &specs,
        .include_moe = false,
        .include_act_quant = false,
    });
    defer pipes.close();

    const n: usize = 32;
    const k: usize = 32;
    const w_bytes = try dtype.packedWeightBytes(n, k, 32);
    const p_bytes = try dtype.paramsBytes(n, k, 32);
    const x_bytes = k * @sizeOf(f16);
    const y_bytes = n * @sizeOf(f16);

    const w_off: usize = 0;
    const p_off: usize = (w_bytes + 63) & ~@as(usize, 63);
    const x_off: usize = (p_off + p_bytes + 63) & ~@as(usize, 63);
    const y_off: usize = (x_off + x_bytes + 63) & ~@as(usize, 63);
    const reg_bytes = try page.alignUp(y_off + y_bytes);
    const reg_budget = budget_mod.Budget.compute(.{
        .memory_bytes = reg_bytes + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = reg_bytes,
        .global_user_wire_limit = reg_bytes,
        .gpu_working_set_bytes = reg_bytes,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(reg_bytes, reg_budget);
    defer arena.deinit();
    const raw = try arena.allocAligned(y_off + y_bytes, 64);
    @memset(raw, 0);

    const qweight = raw[w_off..][0..w_bytes];
    const params = raw[p_off..][0..p_bytes];
    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, raw[x_off..][0..x_bytes]));
    const y_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, raw[y_off..][0..y_bytes]));

    // Fill deterministic `[32, 32]` quantized payload with `ue8m0` scales cleanly
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    if (dtype == .mxfp8_g32) {
        for (qweight) |*b| {
            const v = (rng.float(f32) * 2.0 - 1.0) * 6.0;
            b.* = quant_ref.f32ToE4m3(v);
        }
        for (params) |*p_b| p_b.* = @intCast(125 + rng.uintLessThan(u8, 5)); // 2^(-2..+2)
    } else {
        // `gguf_mxfp4`: 17 bytes per 32-elem block -> byte 0 = ue8m0 exponent, bytes 1..16 = 32 E2M1 nibbles.
        for (0..n) |r| {
            const blk = qweight[r * 17 ..][0..17];
            blk[0] = @intCast(125 + (r % 5));
            for (1..17) |bi| blk[bi] = rng.int(u8);
        }
    }
    for (0..k) |i| x_f16[i] = @floatCast(deterministicF32(seed + 19, i, 0.75, 0.0));

    const ref_row = try allocator.alloc(f32, k);
    defer allocator.free(ref_row);
    const cpu_y = try allocator.alloc(f32, n);
    defer allocator.free(cpu_y);
    const gpu_y = try allocator.alloc(f32, n);
    defer allocator.free(gpu_y);

    for (0..n) |r| {
        try qformats.dequantRowCpuRef(dtype, 32, qweight, params, r, n, k, ref_row);
        var acc: f32 = 0.0;
        for (0..k) |c_idx| acc += ref_row[c_idx] * @as(f32, @floatCast(x_f16[c_idx]));
        cpu_y[r] = acc;
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    try compute.makeResident(&.{gpu_buf});
    defer {
        compute.removeResident(&.{gpu_buf}) catch |err| std.debug.panic("removeResident failed: {}", .{err});
    }

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeGemv(&frame, dtype, 32, .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(raw.ptr) + x_off },
        .qweight = .{ .buffer = gpu_buf, .offset = arena.offsetOf(raw.ptr) + w_off },
        .params = if (p_bytes > 0) .{ .buffer = gpu_buf, .offset = arena.offsetOf(raw.ptr) + p_off } else null,
        .y = .{ .buffer = gpu_buf, .offset = arena.offsetOf(raw.ptr) + y_off },
        .n = @intCast(n),
        .k = @intCast(k),
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    for (0..n) |r| gpu_y[r] = @floatCast(y_f16[r]);
    return cosineF64(cpu_y, gpu_y);
}

fn runDeepSeekKernelParity(
    allocator: std.mem.Allocator,
    device: metal.Device,
    lib: metal.Library,
    compute: *metal.Compute,
    pipes: hybrid.HybridPipelines,
    budget: budget_mod.Budget,
    tmp_dir: []const u8,
    report: *M94Report,
) !void {
    // 1. `[32, 32]` `ue8m0` `MXFP8_G32` and `GGUF_MXFP4` GEMV GPU vs CPU-f32 parity
    const mxfp8_stat = try verifyQuantFormatParity(allocator, device, lib, compute, .mxfp8_g32, 94501);
    report.ds_mxfp8_cos_sim = mxfp8_stat.cos;
    report.ds_mxfp8_max_err = mxfp8_stat.max_err;

    const mxfp4_stat = try verifyQuantFormatParity(allocator, device, lib, compute, .gguf_mxfp4, 94502);
    report.ds_mxfp4_cos_sim = mxfp4_stat.cos;
    report.ds_mxfp4_max_err = mxfp4_stat.max_err;

    const arena_cap = try page.alignUp(1024 * 1024);
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();
    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    try compute.makeResident(&.{buf});
    defer {
        compute.removeResident(&.{buf}) catch |err| std.debug.panic("removeResident failed: {}", .{err});
    }
    const bind = struct {
        fn call(b: metal.Buffer, a: *const arena_mod.WiredArena, ptr: [*]const u8) metal.BufferBinding {
            return .{ .buffer = b, .offset = a.offsetOf(ptr) };
        }
    }.call;

    // 2. `20`-step Sinkhorn `mHC` cross-sublayer `pre_mix` handoff parity
    {
        const m_r: u32 = 2;
        const h: u32 = 64;
        const hc: u32 = 4;
        const mix_hc: u32 = (2 + hc) * hc; // 24

        const st_raw = try arena.allocAligned(m_r * hc * h * 2, 64);
        const fn_raw = try arena.allocAligned(mix_hc * hc * h * 4, 64);
        const base_raw = try arena.allocAligned(mix_hc * 4, 64);
        const sc_raw = try arena.allocAligned(3 * 4, 64);
        const nw_raw = try arena.allocAligned(h * 2, 64);
        const br_raw = try arena.allocAligned(m_r * h * 2, 64);
        const coeffs_raw = try arena.allocAligned(m_r * 20 * 4, 64);
        const pm_in_raw = try arena.allocAligned(m_r * hc * 4, 64);
        const pm_out_raw = try arena.allocAligned(m_r * hc * 4, 64);

        const st_s: []u16 = @alignCast(std.mem.bytesAsSlice(u16, st_raw));
        const fn_s: []f32 = @alignCast(std.mem.bytesAsSlice(f32, fn_raw));
        const base_s: []f32 = @alignCast(std.mem.bytesAsSlice(f32, base_raw));
        const sc_s: []f32 = @alignCast(std.mem.bytesAsSlice(f32, sc_raw));
        const nw_s: []u16 = @alignCast(std.mem.bytesAsSlice(u16, nw_raw));
        const br_s: []u16 = @alignCast(std.mem.bytesAsSlice(u16, br_raw));
        const pm_out_s: []f32 = @alignCast(std.mem.bytesAsSlice(f32, pm_out_raw));

        for (0..m_r * hc * h) |i| st_s[i] = quant.f32ToBf16(deterministicF32(94511, i, 0.7, 0.0));
        for (0..mix_hc * hc * h) |i| fn_s[i] = deterministicF32(94512, i, 0.05, 0.0);
        for (0..mix_hc) |i| base_s[i] = deterministicF32(94513, i, 0.2, 0.0);
        sc_s[0] = 0.25;
        sc_s[1] = 0.25;
        sc_s[2] = 0.25;
        for (0..h) |i| nw_s[i] = quant.f32ToBf16(1.0 + deterministicF32(94514, i, 0.1, 0.0));

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeM94MhcInitPreMix(&frame, bind(buf, &arena, pm_in_raw.ptr), m_r, hc, false);
        try pipes.encodeM94MhcPreSinkhornHandoff(&frame, .{
            .streams = bind(buf, &arena, st_raw.ptr),
            .fn_w = bind(buf, &arena, fn_raw.ptr),
            .base_w = bind(buf, &arena, base_raw.ptr),
            .scale_w = bind(buf, &arena, sc_raw.ptr),
            .norm_w = bind(buf, &arena, nw_raw.ptr),
            .branch_out = bind(buf, &arena, br_raw.ptr),
            .coeffs_out = bind(buf, &arena, coeffs_raw.ptr),
            .pre_mix_in = bind(buf, &arena, pm_in_raw.ptr),
            .pre_mix_out = bind(buf, &arena, pm_out_raw.ptr),
            .m = m_r,
            .hidden_size = h,
            .sinkhorn_iters = 20,
            .stream_dtype = .bf16,
            .act_dtype = .bf16,
            .fn_dtype = .f32,
            .norm_dtype = .bf16,
            .rms_eps = 1e-6,
            .hc_eps = 1e-4,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        const mhc_ref = try allocator.alloc(f32, m_r * (h + hc));
        defer allocator.free(mhc_ref);
        const mhc_got = try allocator.alloc(f32, m_r * (h + hc));
        defer allocator.free(mhc_got);
        const total_d = hc * h;
        for (0..m_r) |tok| {
            var sq_tot: f32 = 0.0;
            for (0..total_d) |j| {
                const sv = quant.bf16ToF32(st_s[tok * total_d + j]);
                sq_tot += sv * sv;
            }
            const inv_r = 1.0 / @sqrt(sq_tot / @as(f32, @floatFromInt(total_d)) + 1e-6);
            const s_pre = sc_s[0] * inv_r;
            for (0..hc) |r_i| {
                var acc: f32 = 0.0;
                for (0..total_d) |j| {
                    acc += quant.bf16ToF32(st_s[tok * total_d + j]) * fn_s[r_i * total_d + j];
                }
                mhc_ref[m_r * h + tok * hc + r_i] = sigmoidF32(acc * s_pre + base_s[r_i]) + 1e-4;
                mhc_got[m_r * h + tok * hc + r_i] = pm_out_s[tok * hc + r_i];
            }
            var sq_sub: f32 = 0.0;
            var mix_buf: [h]f32 = undefined;
            for (0..h) |d| {
                // `encodeM94MhcInitPreMix` initializes `pre_mix_in` to `[1, 0, 0, 0]`.
                const mv: f32 = quant.bf16ToF32(st_s[tok * total_d + d]);
                mix_buf[d] = mv;
                sq_sub += mv * mv;
            }
            const inv_sub = 1.0 / @sqrt(sq_sub / @as(f32, @floatFromInt(h)) + 1e-6);
            for (0..h) |d| {
                mhc_ref[tok * h + d] = (mix_buf[d] * inv_sub) * quant.bf16ToF32(nw_s[d]);
                mhc_got[tok * h + d] = quant.bf16ToF32(br_s[tok * h + d]);
            }
        }
        const mhc_stat = cosineF64(mhc_ref, mhc_got);
        report.ds_mhc_sinkhorn_cos_sim = mhc_stat.cos;
        report.ds_mhc_sinkhorn_max_err = mhc_stat.max_err;
    }

    // 3. `compressor` (`compress_ratios = 2`) weighted softmax pooling + RMSNorm parity
    {
        const n_tok: u32 = 4;
        const ratio: u32 = 2;
        const d_dim: u32 = 32;
        const n_out: u32 = n_tok / ratio;
        const kv_in_raw = try arena.allocAligned(n_tok * d_dim * 2, 64);
        const gate_raw = try arena.allocAligned(n_tok * d_dim * 2, 64);
        const nw_raw = try arena.allocAligned(d_dim * 2, 64);
        const kv_st_raw = try arena.allocAligned(ratio * d_dim * 4, 64);
        const sc_st_raw = try arena.allocAligned(ratio * d_dim * 4, 64);
        const out_raw = try arena.allocAligned(n_out * d_dim * 2, 64);

        const kv_in_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kv_in_raw));
        const gate_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gate_raw));
        const nw_s: []u16 = @alignCast(std.mem.bytesAsSlice(u16, nw_raw));
        const out_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));

        for (0..n_tok * d_dim) |i| {
            kv_in_s[i] = @floatCast(deterministicF32(94521, i, 0.7, 0.0));
            gate_s[i] = @floatCast(deterministicF32(94522, i, 0.9, 0.0));
        }
        for (0..d_dim) |i| nw_s[i] = quant.f32ToBf16(1.0 + deterministicF32(94523, i, 0.1, 0.0));

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeM94DsCompressorPool(&frame, .{
            .kv_in = bind(buf, &arena, kv_in_raw.ptr),
            .gate_in = bind(buf, &arena, gate_raw.ptr),
            .norm_w = bind(buf, &arena, nw_raw.ptr),
            .kv_state = bind(buf, &arena, kv_st_raw.ptr),
            .score_state = bind(buf, &arena, sc_st_raw.ptr),
            .latent_out = bind(buf, &arena, out_raw.ptr),
            .num_tokens = n_tok,
            .start_pos = 0,
            .head_dim = d_dim,
            .ratio = ratio,
            .gate_dim = d_dim,
            .norm_dtype = .bf16,
            .rms_eps = 1e-6,
            .write_state = true,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        const comp_ref = try allocator.alloc(f32, n_out * d_dim);
        defer allocator.free(comp_ref);
        const comp_got = try allocator.alloc(f32, n_out * d_dim);
        defer allocator.free(comp_got);
        for (0..n_out) |g| {
            var tmp_p: [d_dim]f32 = undefined;
            var sq_p: f32 = 0.0;
            for (0..d_dim) |d| {
                const s0: f32 = @floatCast(gate_s[(g * ratio + 0) * d_dim + d]);
                const s1: f32 = @floatCast(gate_s[(g * ratio + 1) * d_dim + d]);
                const k0: f32 = @floatCast(kv_in_s[(g * ratio + 0) * d_dim + d]);
                const k1: f32 = @floatCast(kv_in_s[(g * ratio + 1) * d_dim + d]);
                const mx = @max(s0, s1);
                const w0 = @exp(s0 - mx);
                const w1 = @exp(s1 - mx);
                const p_val = (w0 * k0 + w1 * k1) / (w0 + w1);
                tmp_p[d] = p_val;
                sq_p += p_val * p_val;
            }
            const inv_r = 1.0 / @sqrt(sq_p / @as(f32, @floatFromInt(d_dim)) + 1e-6);
            for (0..d_dim) |d| {
                comp_ref[g * d_dim + d] = (tmp_p[d] * inv_r) * quant.bf16ToF32(nw_s[d]);
                comp_got[g * d_dim + d] = @floatCast(out_s[g * d_dim + d]);
            }
        }
        const comp_stat = cosineF64(comp_ref, comp_got);
        report.ds_compressor_cos_sim = comp_stat.cos;
        report.ds_compressor_max_err = comp_stat.max_err;
    }

    // 4. `sqrtsoftplus` + `noaux_tc` router (`score_mode = 1`) parity
    {
        const m_r: u32 = 2;
        const n_exp: u32 = 8;
        const top_k: u32 = 2;
        const scale_r: f32 = 1.5;

        const logits_raw = try arena.allocAligned(m_r * n_exp * 4, 64);
        const bias_raw = try arena.allocAligned(n_exp * 4, 64);
        const sched_raw = try arena.allocAligned(128, 64);
        const uniq_raw = try arena.allocAligned(n_exp * 4, 64);
        const off_raw = try arena.allocAligned((n_exp + 1) * 4, 64);
        const tok_raw = try arena.allocAligned(m_r * top_k * 4, 64);
        const rw_raw = try arena.allocAligned(m_r * top_k * 4, 64);
        const sk_raw = try arena.allocAligned(m_r * top_k * 4, 64);
        const top_e_raw = try arena.allocAligned(m_r * top_k * 4, 64);
        const top_w_raw = try arena.allocAligned(m_r * top_k * 4, 64);

        const logits_s: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
        const bias_s: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bias_raw));
        const top_e_s: []u32 = @alignCast(std.mem.bytesAsSlice(u32, top_e_raw));
        const top_w_s: []f32 = @alignCast(std.mem.bytesAsSlice(f32, top_w_raw));
        for (0..m_r * n_exp) |i| logits_s[i] = @sin(@as(f32, @floatFromInt(i + 1)) * 0.9) * 2.0;
        for (0..n_exp) |e| bias_s[e] = if (e == 5) 2.5 else -0.1 * @as(f32, @floatFromInt(e));

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeG26MoeRouteSigmoid(&frame, .{
            .logits = bind(buf, &arena, logits_raw.ptr),
            .logits_f16 = false,
            .bias = bind(buf, &arena, bias_raw.ptr),
            .bias_dtype = .f32,
            .sched_hdr = bind(buf, &arena, sched_raw.ptr),
            .unique_experts = bind(buf, &arena, uniq_raw.ptr),
            .expert_token_offsets = bind(buf, &arena, off_raw.ptr),
            .token_indices = bind(buf, &arena, tok_raw.ptr),
            .routing_weights = bind(buf, &arena, rw_raw.ptr),
            .slot_k_indices = bind(buf, &arena, sk_raw.ptr),
            .scratch_topk_experts = bind(buf, &arena, top_e_raw.ptr),
            .scratch_topk_weights = bind(buf, &arena, top_w_raw.ptr),
            .m = m_r,
            .num_experts = n_exp,
            .top_k = top_k,
            .norm_topk_prob = true,
            .score_mode = 1,
            .routed_scaling_factor = scale_r,
            .gate_up_groups_x_q = 1,
            .gate_up_groups_x_f = 1,
            .down_groups_x = 1,
            .down_groups_x_q8 = 1,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        var max_w_err: f64 = 0.0;
        var ids_match = true;
        for (0..m_r) |tok| {
            var scores: [n_exp]f32 = undefined;
            for (0..n_exp) |e| scores[e] = @sqrt(@log(1.0 + @exp(logits_s[tok * n_exp + e])));
            var chosen: [top_k]u32 = undefined;
            var picked: [n_exp]bool = @splat(false);
            var wsum: f32 = 0.0;
            for (0..top_k) |k_i| {
                var best_c: f32 = -std.math.inf(f32);
                var best_e: u32 = 0;
                for (0..n_exp) |e| {
                    if (picked[e]) continue;
                    const ch = scores[e] + bias_s[e];
                    if (ch > best_c) {
                        best_c = ch;
                        best_e = @intCast(e);
                    }
                }
                picked[best_e] = true;
                chosen[k_i] = best_e;
                wsum += scores[best_e];
            }
            for (0..top_k) |k_i| {
                if (chosen[k_i] != top_e_s[tok * top_k + k_i]) ids_match = false;
                const exp_w = (scores[chosen[k_i]] / (wsum + 1e-20)) * scale_r;
                max_w_err = @max(max_w_err, @abs(@as(f64, exp_w) - @as(f64, top_w_s[tok * top_k + k_i])));
            }
        }
        report.ds_sqrtsoftplus_max_err = if (ids_match) max_w_err else 1.0;
    }

    // 5. Real-HF complex RoPE + Latent Attn (`attn_sink`) + Engram per-stream `rstd` signed-sqrt gate
    {
        const num_q: u32 = 2;
        const n_heads: u32 = 4;
        const d_dim: u32 = 64;
        const rd: u32 = 16;
        const win: u32 = 16;
        const h_e: u32 = 64;

        const q_raw = try arena.allocAligned(num_q * n_heads * d_dim * 2, 64);
        const kv_c_raw = try arena.allocAligned(win * d_dim * 2, 64);
        const kv_cur_raw = try arena.allocAligned(num_q * d_dim * 2, 64);
        const sink_raw = try arena.allocAligned(n_heads * 4, 64);
        const out_raw = try arena.allocAligned(num_q * n_heads * d_dim * 2, 64);

        const st_raw = try arena.allocAligned(num_q * 4 * h_e * 2, 64);
        const wkv_raw = try arena.allocAligned(num_q * 5 * h_e * 2, 64);
        const qw_raw = try arena.allocAligned(4 * h_e * 2, 64);
        const kw_raw = try arena.allocAligned(4 * h_e * 2, 64);

        const q_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
        const kv_cur_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kv_cur_raw));
        const sink_s: []f32 = @alignCast(std.mem.bytesAsSlice(f32, sink_raw));
        const out_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
        const st_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, st_raw));
        const wkv_s: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wkv_raw));
        const qw_s: []u16 = @alignCast(std.mem.bytesAsSlice(u16, qw_raw));
        const kw_s: []u16 = @alignCast(std.mem.bytesAsSlice(u16, kw_raw));

        @memset(kv_c_raw, 0);
        for (0..num_q * n_heads * d_dim) |i| q_s[i] = @floatCast(@sin(@as(f32, @floatFromInt(i + 1)) * 0.17) * 0.5);
        for (0..num_q * d_dim) |i| kv_cur_s[i] = @floatCast(@cos(@as(f32, @floatFromInt(i + 3)) * 0.13) * 0.5);
        for (0..n_heads) |hd| sink_s[hd] = -0.3 + 0.25 * @as(f32, @floatFromInt(hd));

        var st_init: [num_q * 4 * h_e]f32 = undefined;
        for (0..num_q * 4 * h_e) |i| {
            st_s[i] = @floatCast(@sin(@as(f32, @floatFromInt(i + 3)) * 0.11) * 0.8);
            st_init[i] = @floatCast(st_s[i]);
        }
        for (0..num_q * 5 * h_e) |i| wkv_s[i] = @floatCast(@cos(@as(f32, @floatFromInt(i + 5)) * 0.09) * 0.7);
        for (0..4 * h_e) |i| {
            qw_s[i] = quant.f32ToBf16(1.0 + 0.1 * @sin(@as(f32, @floatFromInt(i))));
            kw_s[i] = quant.f32ToBf16(1.0 + 0.1 * @cos(@as(f32, @floatFromInt(i))));
        }

        const q_init = try allocator.alloc(f32, num_q * n_heads * d_dim);
        defer allocator.free(q_init);
        const k_init = try allocator.alloc(f32, num_q * d_dim);
        defer allocator.free(k_init);
        for (0..num_q * n_heads * d_dim) |i| q_init[i] = @floatCast(q_s[i]);
        for (0..num_q * d_dim) |i| k_init[i] = @floatCast(kv_cur_s[i]);

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try pipes.encodeM94RopeComplex(&frame, .{
            .x = bind(buf, &arena, q_raw.ptr),
            .num_tokens = num_q,
            .num_heads = n_heads,
            .head_stride = d_dim,
            .rot_offset = d_dim - rd,
            .rot_dim = rd,
            .base_pos = 0,
            .inverse = false,
            .rope_theta = 10_000.0,
        }, false);
        try pipes.encodeM94RopeComplex(&frame, .{
            .x = bind(buf, &arena, kv_cur_raw.ptr),
            .num_tokens = num_q,
            .num_heads = 1,
            .head_stride = d_dim,
            .rot_offset = d_dim - rd,
            .rot_dim = rd,
            .base_pos = 0,
            .inverse = false,
            .rope_theta = 10_000.0,
        }, false);
        try pipes.encodeM94DsLatentAttn(&frame, .{
            .q = bind(buf, &arena, q_raw.ptr),
            .win_kv_cache = bind(buf, &arena, kv_c_raw.ptr),
            .win_kv_cur = bind(buf, &arena, kv_cur_raw.ptr),
            .attn_sink = bind(buf, &arena, sink_raw.ptr),
            .out = bind(buf, &arena, out_raw.ptr),
            .num_q = num_q,
            .n_heads = n_heads,
            .q_head_dim = d_dim,
            .kv_dim = d_dim,
            .v_head_dim = d_dim,
            .nope_dim = d_dim - rd,
            .rope_dim = rd,
            .window_size = win,
            .start_pos = 0,
            .apply_inv_rope = true,
            .rope_theta = 10_000.0,
        }, true);
        try pipes.encodeM94DsEngramGateAdd(&frame, .{
            .streams = bind(buf, &arena, st_raw.ptr),
            .wkv_out = bind(buf, &arena, wkv_raw.ptr),
            .emb_in = bind(buf, &arena, wkv_raw.ptr),
            .q_weight = bind(buf, &arena, qw_raw.ptr),
            .k_weight = bind(buf, &arena, kw_raw.ptr),
            .m = num_q,
            .hidden_size = h_e,
            .wkv_out_dim = 5 * h_e,
            .emb_dim = h_e,
            .k_dim = h_e,
            .stream_dtype = .f16,
            .qk_dtype = .bf16,
            .rms_eps = 1e-20,
        }, false);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        // Compare Engram output (stored in `st_s`) + latent attn output (`out_s`) vs CPU fp32 reference.
        const eng_ref = try allocator.alloc(f32, num_q * 4 * h_e + num_q * n_heads * d_dim);
        defer allocator.free(eng_ref);
        const eng_got = try allocator.alloc(f32, num_q * 4 * h_e + num_q * n_heads * d_dim);
        defer allocator.free(eng_got);
        for (0..num_q) |tok| {
            for (0..4) |s_i| {
                var sq_h: f32 = 0.0;
                var sq_k: f32 = 0.0;
                var dot_p: f32 = 0.0;
                for (0..h_e) |d| {
                    const hv = st_init[(tok * 4 + s_i) * h_e + d];
                    const kv: f32 = @floatCast(wkv_s[tok * 5 * h_e + s_i * h_e + d]);
                    const qw = quant.bf16ToF32(qw_s[s_i * h_e + d]);
                    const kw = quant.bf16ToF32(kw_s[s_i * h_e + d]);
                    sq_h += hv * hv;
                    sq_k += kv * kv;
                    dot_p += hv * qw * kw * kv;
                }
                const rstd = (1.0 / @sqrt(sq_h / @as(f32, @floatFromInt(h_e)) + 1e-20)) *
                    (1.0 / @sqrt(sq_k / @as(f32, @floatFromInt(h_e)) + 1e-20));
                const dv = dot_p * rstd / @sqrt(@as(f32, @floatFromInt(h_e)));
                const sgn: f32 = if (dv < 0.0) -1.0 else 1.0;
                const gate = sigmoidF32(sgn * @sqrt(@max(@abs(dv), 1e-6)));
                for (0..h_e) |d| {
                    const idx = (tok * 4 + s_i) * h_e + d;
                    eng_ref[idx] = st_init[idx] + gate * @as(f32, @floatCast(wkv_s[tok * 5 * h_e + 4 * h_e + d]));
                    eng_got[idx] = @floatCast(st_s[idx]);
                }
            }
        }
        // Complex RoPE + Latent Attention + invRoPE CPU ref for `num_q = 2` tokens
        const ro: usize = d_dim - rd;
        const scale_la: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_dim)));
        var q_rot: [num_q * n_heads * d_dim]f32 = undefined;
        var k_rot: [num_q * d_dim]f32 = undefined;
        @memcpy(&q_rot, q_init);
        @memcpy(&k_rot, k_init);
        for (0..num_q) |ti| {
            for (0..n_heads) |hd| {
                for (0..rd / 2) |j| {
                    const inv_f = @exp(-@log(@as(f32, 10_000.0)) * (@as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(rd))));
                    const ang = @as(f32, @floatFromInt(ti)) * inv_f;
                    const c = @cos(ang);
                    const s = @sin(ang);
                    const base = (ti * n_heads + hd) * d_dim + ro + 2 * j;
                    const x0 = q_rot[base];
                    const x1 = q_rot[base + 1];
                    q_rot[base] = x0 * c - x1 * s;
                    q_rot[base + 1] = x1 * c + x0 * s;
                }
            }
            for (0..rd / 2) |j| {
                const inv_f = @exp(-@log(@as(f32, 10_000.0)) * (@as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(rd))));
                const ang = @as(f32, @floatFromInt(ti)) * inv_f;
                const c = @cos(ang);
                const s = @sin(ang);
                const base = ti * d_dim + ro + 2 * j;
                const x0 = k_rot[base];
                const x1 = k_rot[base + 1];
                k_rot[base] = x0 * c - x1 * s;
                k_rot[base + 1] = x1 * c + x0 * s;
            }
        }
        const mla_off: usize = num_q * 4 * h_e;
        for (0..num_q) |qi| {
            for (0..n_heads) |hd| {
                var scores: [num_q]f32 = undefined;
                var mx: f32 = sink_s[hd];
                for (0..qi + 1) |ki| {
                    var dp: f32 = 0.0;
                    for (0..d_dim) |d| dp += q_rot[(qi * n_heads + hd) * d_dim + d] * k_rot[ki * d_dim + d];
                    scores[ki] = dp * scale_la;
                    mx = @max(mx, scores[ki]);
                }
                var den: f32 = @exp(sink_s[hd] - mx);
                for (0..qi + 1) |ki| {
                    scores[ki] = @exp(scores[ki] - mx);
                    den += scores[ki];
                }
                var ref_o: [d_dim]f32 = @splat(0.0);
                for (0..qi + 1) |ki| {
                    const w = scores[ki] / den;
                    for (0..d_dim) |d| ref_o[d] += w * k_rot[ki * d_dim + d];
                }
                for (0..rd / 2) |j| {
                    const inv_f = @exp(-@log(@as(f32, 10_000.0)) * (@as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(rd))));
                    const ang = @as(f32, @floatFromInt(qi)) * inv_f;
                    const c = @cos(ang);
                    const s = -@sin(ang);
                    const x0 = ref_o[ro + 2 * j];
                    const x1 = ref_o[ro + 2 * j + 1];
                    ref_o[ro + 2 * j] = x0 * c - x1 * s;
                    ref_o[ro + 2 * j + 1] = x1 * c + x0 * s;
                }
                for (0..d_dim) |d| {
                    const gidx = (qi * n_heads + hd) * d_dim + d;
                    eng_ref[mla_off + gidx] = ref_o[d];
                    eng_got[mla_off + gidx] = @floatCast(out_s[gidx]);
                }
            }
        }
        const mla_stat = cosineF64(eng_ref, eng_got);
        report.ds_mla_engram_cos_sim = mla_stat.cos;
        report.ds_mla_engram_max_err = mla_stat.max_err;
    }

    // 6. Synthetic DeepSeek-V4.1 `24gb-m4` CED (`encoder_layer_dispatches_decode == 0`) & grouped `wo_a`/`wo_b` check
    {
        var s_ds = try graph.buildSynthDeepSeek41Bundle(allocator, .{ .tmp_dir = tmp_dir, .include_dspark = false });
        defer s_ds.deinit(allocator);
        const eng_c: []const [:0]const u8 = s_ds.engram_paths;
        var m_ced = try graph.GraphModel.loadFromBytes(allocator, s_ds.amf_bytes, s_ds.cfg, .{
            .max_seq_len = 64,
            .max_tree_nodes = 8,
            .ple_paths = eng_c,
            .startup_canary = false,
        });
        defer m_ced.deinit(allocator);
        const prompt = [_]u32{ 2, 6, 10, 14 };
        const t0 = try m_ced.prefillChunk(&prompt, 0, true);
        _ = try m_ced.stepDecode(t0, @intCast(prompt.len));
        report.ds_synth_encoder_decode_dispatches = m_ced.m7Stats().encoder_layer_dispatches_decode;
        report.ds_wo_grouped_ok = (m_ced.m7Stats().encoder_layer_dispatches_prefill > 0) and
            (m_ced.m7Stats().encoder_layer_dispatches_decode == 0) and
            allFiniteF16(m_ced.readLastLogitsF16()[0..s_ds.cfg.vocab_size]);
    }

    report.ds_kernel_parity_ok = (report.ds_mxfp8_cos_sim >= 0.999) and
        (report.ds_mxfp4_cos_sim >= 0.999) and
        (report.ds_mhc_sinkhorn_cos_sim >= 0.999) and
        (report.ds_compressor_cos_sim >= 0.999) and
        (report.ds_sqrtsoftplus_max_err <= 1e-4) and
        (report.ds_mla_engram_cos_sim >= 0.999) and
        report.ds_wo_grouped_ok;
}

fn runDeepSeekColdTierAndKl(
    allocator: std.mem.Allocator,
    tmp_dir: []const u8,
    device: metal.Device,
    budget: budget_mod.Budget,
    report: *M94Report,
) !void {
    // 1. `C27` session LFU promotion hit-rate check over cold expert trace
    {
        const num_slots: u32 = 8;
        const num_experts: u32 = 16;
        const rows: u32 = 32;
        const cols: u32 = 64;
        const nbytes: usize = rows * cols * 2;
        var blobs: [num_experts * 2]graph.TestTensorBlob = undefined;
        var names: [num_experts * 2][]u8 = undefined;
        var nb_buf: [64]u8 = undefined;
        var bi: usize = 0;
        errdefer {
            for (0..bi) |j| {
                allocator.free(blobs[j].bytes);
                allocator.free(names[j]);
            }
        }
        for (0..num_experts) |e| {
            const n1 = try allocator.dupe(u8, try std.fmt.bufPrint(&nb_buf, "layers.0.mlp.experts.{d}.gate_up_proj", .{e}));
            const b1 = try allocator.alloc(u8, nbytes);
            @memset(b1, @truncate((e * 17 + 5) & 0xFF));
            blobs[bi] = .{ .name = n1, .kind = .cold_expert, .dtype = .f16, .layer = 0, .expert = @intCast(e), .shape = .{ rows, cols, 0, 0 }, .ndim = 2, .bytes = b1 };
            names[bi] = n1;
            bi += 1;

            const n2 = try allocator.dupe(u8, try std.fmt.bufPrint(&nb_buf, "layers.0.mlp.experts.{d}.down_proj", .{e}));
            const b2 = try allocator.alloc(u8, nbytes);
            @memset(b2, @truncate((e * 29 + 11) & 0xFF));
            blobs[bi] = .{ .name = n2, .kind = .cold_expert, .dtype = .f16, .layer = 0, .expert = @intCast(e), .shape = .{ cols, rows, 0, 0 }, .ndim = 2, .bytes = b2 };
            names[bi] = n2;
            bi += 1;
        }
        const pack_amf = try graph.buildTestAmf(allocator, "m94-c27-lfu", &blobs);
        defer {
            allocator.free(pack_amf);
            for (0..bi) |j| {
                allocator.free(blobs[j].bytes);
                allocator.free(names[j]);
            }
        }
        const pack_path = try std.fmt.allocPrintSentinel(allocator, "{s}/c27_lfu_pack.amf", .{tmp_dir}, 0);
        defer allocator.free(pack_path);
        try writeFileBytesPosix(pack_path, pack_amf);

        var af = try amf.AmfFile.open(allocator, pack_path);
        defer af.close(allocator);
        var metas: [num_experts]m82_rt.M82ColdExpertMeta = undefined;
        for (0..num_experts) |e| {
            const w1_e = af.parsed.find(try std.fmt.bufPrint(&nb_buf, "layers.0.mlp.experts.{d}.gate_up_proj", .{e})) orelse return error.MissingTensor;
            const w2_e = af.parsed.find(try std.fmt.bufPrint(&nb_buf, "layers.0.mlp.experts.{d}.down_proj", .{e})) orelse return error.MissingTensor;
            metas[e] = .{
                .layer = 0,
                .expert = @intCast(e),
                .w1_offset = w1_e.blob_offset,
                .w1_bytes = w1_e.blob_bytes,
                .w1_params_offset = w1_e.params_offset,
                .w1_amf_dtype = .f16,
                .w1_group_size = 0,
                .w1_n = rows,
                .w1_k = cols,
                .w2_offset = w2_e.blob_offset,
                .w2_bytes = w2_e.blob_bytes,
                .w2_params_offset = w2_e.params_offset,
                .w2_amf_dtype = .f16,
                .w2_group_size = 0,
                .w2_n = cols,
                .w2_k = rows,
            };
        }

        var compute = try metal.Compute.initWithFrames(device, 2);
        defer compute.close();
        const runTrace = struct {
            fn call(alloc: std.mem.Allocator, dev: metal.Device, comp: metal.Compute, path: [:0]const u8, bgt: budget_mod.Budget, all_metas: []const m82_rt.M82ColdExpertMeta, two_tier: bool) !m82_rt.M82Stats {
                var cache = if (two_tier)
                    try m82_rt.M82ColdExpertCache.initWithTiering(alloc, dev, comp, path, num_slots, try page.alignUp(nbytes), 0, try page.alignUp(nbytes), bgt, 1, num_experts, 4)
                else blk: {
                    var bc = try m82_rt.M82ColdExpertCache.init(alloc, dev, comp, path, num_slots, try page.alignUp(nbytes), 0, try page.alignUp(nbytes), bgt);
                    bc.working_set_cap = 0;
                    break :blk bc;
                };
                defer cache.deinit(alloc, comp);
                var st: m82_rt.M82Stats = .{};
                for (0..2) |_| {
                    for (0..4) |e| _ = try cache.pinExpert(all_metas[e], &st);
                }
                var cur: usize = 4;
                for (0..4) |_| {
                    for (0..8) |_| {
                        _ = try cache.pinExpert(all_metas[cur], &st);
                        cur = if (cur + 1 >= all_metas.len) 4 else cur + 1;
                    }
                    for (0..4) |e| _ = try cache.pinExpert(all_metas[e], &st);
                }
                return st;
            }
        }.call;

        const base_s = try runTrace(allocator, device, compute, pack_path, budget, &metas, false);
        const lfu_s = try runTrace(allocator, device, compute, pack_path, budget, &metas, true);
        const base_tot = base_s.cold_expert_fetches + base_s.cold_expert_cache_hits;
        const lfu_tot = lfu_s.cold_expert_fetches + lfu_s.cold_expert_cache_hits;
        report.c27_lfu_baseline_hit_rate = if (base_tot > 0) @as(f64, @floatFromInt(base_s.cold_expert_cache_hits)) / @as(f64, @floatFromInt(base_tot)) else 0.0;
        report.c27_lfu_promoted_hit_rate = if (lfu_tot > 0) @as(f64, @floatFromInt(lfu_s.cold_expert_cache_hits)) / @as(f64, @floatFromInt(lfu_tot)) else 0.0;
        report.c27_session_promotions = lfu_s.session_promotions;
        report.c27_lfu_ok = (report.c27_lfu_promoted_hit_rate > report.c27_lfu_baseline_hit_rate) and (lfu_s.session_promotions > 0);
    }

    // 2. `--cold-expert-min-weight` `[0.00, 0.03, 0.05]` W24 KL check on cold-tiered 3L DSpark
    {
        var art = try buildSynthDspark3LArtifact(allocator, tmp_dir, "kl_dspark3l.amf");
        defer art.deinit(allocator);

        const h: usize = 64;
        var taps_buf: [3 * h]f32 = undefined;
        for (0..3 * h) |i| taps_buf[i] = deterministicF32(94601, i, 0.35, 0.0);
        const taps = [_][]const f32{ taps_buf[0..h], taps_buf[h .. 2 * h], taps_buf[2 * h .. 3 * h] };

        const EmbedHook = struct {
            fn read(ctx: ?*anyopaque, tok: u32, dst: []f32) dspark.Error!void {
                _ = ctx;
                for (dst, 0..) |*v, d| v.* = @sin(@as(f32, @floatFromInt(tok * 7 + d + 1)) * 0.05) * 0.2;
            }
        };
        const LmHeadHook = struct {
            fn eval(ctx: ?*anyopaque, sh: []const f32, bw: u8, hs: u32, vs: u32, out: []f32) dspark.Error!void {
                _ = ctx;
                for (0..bw) |s| {
                    const row_h = sh[s * hs ..][0..hs];
                    const dst = out[s * vs ..][0..vs];
                    for (dst, 0..) |*lo, v| lo.* = row_h[v % row_h.len] * 0.5;
                }
            }
        };

        const cfg_3l = try dspark.DSpark3LConfig.fromJson(allocator, art.cfg_json);
        var rt_hot = try dspark.DSpark3LRuntime.loadFromFile(allocator, art.amf_path, cfg_3l, .{
            .embed_hook = .{ .read_fn = EmbedHook.read },
            .lm_head_hook = .{ .callback = .{ .eval_fn = LmHeadHook.eval } },
            .lm_head_mode = .markov_low_rank,
            .force_cold_experts = false,
            .verify_crcs = true,
        });
        defer rt_hot.deinit(allocator);

        var p_hot = dspark.DSparkProposer.init3L(rt_hot);
        try p_hot.seed3LWindow(&taps, 0);
        p_hot.set3LTargetTaps(&taps, 1);
        var tr_hot: dspark.CandidateTree = undefined;
        _ = try p_hot.propose(7, taps[0], taps[2], null, &tr_hot);

        const v_sz: usize = 48;
        const hot_logits = try allocator.dupe(f32, rt_hot.workspace.slot_logits[0..v_sz]);
        defer allocator.free(hot_logits);

        var lossless_ok = false;
        var monotonic_skip = true;
        for (M94_COLD_KL_THRESHOLDS, 0..) |thr, idx| {
            var rt_cold = try dspark.DSpark3LRuntime.loadFromFile(allocator, art.amf_path, cfg_3l, .{
                .embed_hook = .{ .read_fn = EmbedHook.read },
                .lm_head_hook = .{ .callback = .{ .eval_fn = LmHeadHook.eval } },
                .lm_head_mode = .markov_low_rank,
                .force_cold_experts = true,
                .cold_slot_pool_slots = 4,
                .cold_expert_min_weight = thr,
                .verify_crcs = false,
            });
            defer rt_cold.deinit(allocator);
            var p_cold = dspark.DSparkProposer.init3L(rt_cold);
            try p_cold.seed3LWindow(&taps, 0);
            p_cold.set3LTargetTaps(&taps, 1);
            var tr_cold: dspark.CandidateTree = undefined;
            _ = try p_cold.propose(7, taps[0], taps[2], null, &tr_cold);

            const kl = m81.computeLogitsKlAndTop1F32(hot_logits, rt_cold.workspace.slot_logits[0..v_sz]);
            const cs = rt_cold.coldStats();
            report.ds_cold_kl_mean[idx] = kl.kl_divergence;
            report.ds_cold_skipped_low_weight[idx] = cs.skipped_low_weight;
            if (idx == 0) {
                lossless_ok = kl.all_finite and kl.top1_match and (kl.kl_divergence <= 1e-4) and (cs.skipped_low_weight == 0);
            } else {
                if (cs.skipped_low_weight < report.ds_cold_skipped_low_weight[idx - 1]) monotonic_skip = false;
            }
        }
        report.ds_cold_lossless_ok = lossless_ok and monotonic_skip and (report.ds_cold_skipped_low_weight[2] > 0);
    }
}

fn runW30DeepSeekLoraCheck(
    allocator: std.mem.Allocator,
    model: *graph.GraphModel,
    budget: budget_mod.Budget,
    cfg: M94Config,
    report: *M94Report,
) !void {
    const h: usize = model.config.hidden_size;
    const n_layers: usize = model.config.num_hidden_layers;
    const dec_layer: u32 = model.config.num_encoder_layers; // first decoder layer (active on every decode step)
    const q_lora_r: usize = model.config.q_lora_rank;
    const kv_a_dim: usize = model.config.mlaKvADim();

    var reg = try lora.LoraRegistry.init(.{
        .pool_bytes = 8 * 1024 * 1024,
        .allocator = allocator,
        .wired_budget = budget,
        .base_shape = model.baseModelShape(),
    });
    defer reg.deinit(allocator);
    try model.attachLoraRegistry(&reg);
    defer model.attachLoraRegistry(null) catch |err| {
        std.debug.panic("GraphModel.attachLoraRegistry(null) failed in defer: {}", .{err});
    };

    const prompt = [_]u32{ 2, 6, 10, 14 };
    model.setActiveLoraSlot(lora.NO_ADAPTER);
    model.resetKvCache();
    const base_t0 = try model.prefillChunk(&prompt, 0, true);
    const base_logits = try allocator.dupe(f16, model.readLastLogitsF16()[0..model.config.vocab_size]);
    defer allocator.free(base_logits);

    const r_l: usize = 4;
    const a_qa = try allocator.alloc(f32, r_l * h);
    defer allocator.free(a_qa);
    const b_qa = try allocator.alloc(f32, q_lora_r * r_l);
    defer allocator.free(b_qa);
    const a_kva = try allocator.alloc(f32, r_l * h);
    defer allocator.free(a_kva);
    const b_kva = try allocator.alloc(f32, kv_a_dim * r_l);
    defer allocator.free(b_kva);
    const m_kva = try allocator.alloc(f32, kv_a_dim);
    defer allocator.free(m_kva);

    for (a_qa, 0..) |*v, i| v.* = deterministicF32(94701, i, 0.25, 0.0);
    for (b_qa, 0..) |*v, i| v.* = deterministicF32(94702, i, 0.25, 0.0);
    for (a_kva, 0..) |*v, i| v.* = deterministicF32(94703, i, 0.25, 0.0);
    for (b_kva, 0..) |*v, i| v.* = deterministicF32(94704, i, 0.25, 0.0);
    @memset(m_kva, 1.15);

    const target_l = if (dec_layer < n_layers) dec_layer else 0;
    const specs = [_]lora.SyntheticLoraSpec{
        .{ .layer = @intCast(target_l), .target = .q_a_proj, .n = @intCast(q_lora_r), .k = @intCast(h), .rank = r_l, .layout = .row_major, .dtype = .f16, .a_f32 = a_qa, .b_f32 = b_qa, .custom_suffix = "self_attn.q_a_proj" },
        .{ .layer = @intCast(target_l), .target = .kv_a_proj, .n = @intCast(kv_a_dim), .k = @intCast(h), .rank = r_l, .layout = .row_major, .dtype = .f16, .a_f32 = a_kva, .b_f32 = b_kva, .dora_mag_f32 = m_kva, .custom_suffix = "self_attn.kv_a_proj_with_mqa" },
    };
    const amlora_bytes = try lora.buildSyntheticAmloraBytes(allocator, "ds41-w30", r_l, 16.0, &specs);
    defer allocator.free(amlora_bytes);

    const slot = try reg.loadAdapterFromBytes(allocator, "ds41-w30", amlora_bytes);
    try model.calibrateLoraDora(allocator);

    model.setActiveLoraSlot(slot);
    model.resetKvCache();
    var cur = try model.prefillChunk(&prompt, 0, true);
    var active_diff: f64 = 0.0;
    for (base_logits, model.readLastLogitsF16()[0..model.config.vocab_size]) |bl, al| {
        active_diff = @max(active_diff, @abs(@as(f64, @floatCast(al)) - @as(f64, @floatCast(bl))));
    }

    var pos: u32 = @intCast(prompt.len);
    const iters = @max(cfg.lora_iters, 4);
    for (0..iters) |_| {
        const sp = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(sp);
        cur = try model.stepDecode(cur, pos);
        pos += 1;
    }
    const malloc_delta = try measureStepEncodeMallocDelta(model, &cur, &pos);

    model.setActiveLoraSlot(lora.NO_ADAPTER);
    try reg.unloadAdapter(allocator, "ds41-w30");
    model.resetKvCache();
    const restored_t0 = try model.prefillChunk(&prompt, 0, true);
    var unload_err: f64 = 0.0;
    for (base_logits, model.readLastLogitsF16()[0..model.config.vocab_size]) |bl, rl| {
        unload_err = @max(unload_err, @abs(@as(f64, @floatCast(rl)) - @as(f64, @floatCast(bl))));
    }

    report.w30_ds_lora_logit_diff = active_diff;
    report.w30_ds_unload_max_err = unload_err;
    report.w30_ds_lora_malloc_delta = malloc_delta;
    const malloc_ok = report.mtl_validation_active or (malloc_delta <= 0);
    report.w30_ds_lora_ok = (active_diff > 1e-4) and (unload_err == 0.0) and (base_t0 == restored_t0) and malloc_ok;
}

fn runDeepSeekE2eAndW30(
    allocator: std.mem.Allocator,
    tmp_dir: []const u8,
    budget: budget_mod.Budget,
    cfg: M94Config,
    report: *M94Report,
) !void {
    const models_dir = paths.resolveModelsDir(cfg.models_dir);
    if (models_dir) |mdir| {
        var m_buf: [std.fs.max_path_bytes]u8 = undefined;
        var s0_buf: [std.fs.max_path_bytes]u8 = undefined;
        var s1_buf: [std.fs.max_path_bytes]u8 = undefined;
        var dsp_buf: [std.fs.max_path_bytes]u8 = undefined;

        if (resolveModelFileCandidateZ(&m_buf, mdir, "deepseek41-real-hf-synth/model.amf")) |model_path| {
            const s0 = resolveModelFileCandidateZ(&s0_buf, mdir, "deepseek41-real-hf-synth/engram-shard-00-of-02.amf");
            const s1 = resolveModelFileCandidateZ(&s1_buf, mdir, "deepseek41-real-hf-synth/engram-shard-01-of-02.amf");
            const dspark_path = resolveModelFileCandidateZ(&dsp_buf, mdir, "deepseek41-real-hf-synth/dspark.amf");

            var af = try amf.AmfFile.open(allocator, model_path);
            defer af.close(allocator);
            var cfg_meta = try amf.resolveConfigJsonAlloc(allocator, &af, model_path, null);
            defer cfg_meta.deinit(allocator);
            var gcfg = try graph.GraphConfig.fromJson(allocator, cfg_meta.bytes);
            defer gcfg.deinit(allocator);

            var eng_list: [2][:0]const u8 = undefined;
            var eng_len: usize = 0;
            if (s0) |p0| {
                eng_list[eng_len] = p0;
                eng_len += 1;
            }
            if (s1) |p1| {
                eng_list[eng_len] = p1;
                eng_len += 1;
            }
            const eng_slice: []const [:0]const u8 = eng_list[0..eng_len];

            const prompt = [_]u32{ 5, 11, 23, 7, 19, 31, 5, 11, 23, 7 };
            const n_gen: usize = @max(@as(usize, cfg.e2e_tokens), 6);
            const base_toks = try allocator.alloc(u32, n_gen);
            defer allocator.free(base_toks);

            var m_base = try graph.GraphModel.loadFromFile(allocator, model_path, gcfg, .{
                .max_seq_len = 128,
                .max_tree_nodes = 32,
                .ple_paths = eng_slice,
                .cold_slot_count = 16,
                .startup_canary = false,
            });
            defer m_base.deinit(allocator);

            const dec_res = try benchGreedyDecode(&m_base, &prompt, @intCast(n_gen), base_toks, true);
            report.ds_staged_decode_tok_s = dec_res.tok_s;
            report.ds_staged_malloc_delta = dec_res.malloc_delta;

            // W30 LoRA/DoRA compatibility check on staged `deepseek41-real-hf-synth`
            try runW30DeepSeekLoraCheck(allocator, &m_base, budget, cfg, report);

            // Real-HF `3L DSpark` E2E greedy `decode == spec-decode -M dspark` parity
            var dspark_parity_ok = false;
            var enc_pre: u64 = 0;
            var enc_dec: u64 = 0;
            if (dspark_path) |dp| {
                const aux_ids: ?[3]u32 = if (gcfg.dspark_target_layer_ids.len >= 3)
                    .{ gcfg.dspark_target_layer_ids[0], gcfg.dspark_target_layer_ids[1], gcfg.dspark_target_layer_ids[2] }
                else
                    null;
                var m_spec = try graph.GraphModel.loadFromFile(allocator, model_path, gcfg, .{
                    .max_seq_len = 128,
                    .max_tree_nodes = 32,
                    .ple_paths = eng_slice,
                    .cold_slot_count = 16,
                    .enable_feature_taps = true,
                    .aux_layer_ids = aux_ids,
                    .startup_canary = false,
                });
                defer m_spec.deinit(allocator);

                var dspark_loaded = try engine_mod.LoadedDSparkProposer.loadFromFileWithOptions(
                    allocator,
                    &m_spec,
                    dp,
                    null,
                    .{
                        .force_cold_experts = true,
                        .cold_slot_pool_slots = 12,
                        .cold_expert_min_weight = 0.0,
                        .lm_head_mode = .markov_low_rank,
                        .verify_crcs = true,
                    },
                    null,
                );
                defer dspark_loaded.deinit(allocator);

                var eng = try engine_mod.SpecEngine.initWithDSpark(allocator, &m_spec, &dspark_loaded.proposer, .{
                    .mode = .dspark,
                    .max_tree_nodes = 32,
                    .max_draft_depth = 6,
                    .enable_pruning = true,
                    .temperature = 0.0,
                });
                defer eng.deinit(allocator);

                const spec_toks = try allocator.alloc(u32, n_gen);
                defer allocator.free(spec_toks);
                const sres = try evalSpecDecodeRun(&eng, &prompt, spec_toks);
                report.ds_staged_dspark_tok_s = sres.tok_s;
                report.ds_staged_dspark_mean_accept = sres.mean_accept;
                report.ds_staged_dspark_speedup = if (dec_res.tok_s > 0.0) sres.tok_s / dec_res.tok_s else 0.0;
                if (dspark_loaded.real_hf_3l) |rt3l| {
                    const cs = rt3l.coldStats();
                    report.ds_staged_dspark_cold_hits = cs.cache_hits;
                    report.ds_staged_dspark_cold_misses = cs.cache_misses;
                }
                const m7s = m_spec.m7Stats();
                enc_pre = m7s.encoder_layer_dispatches_prefill;
                enc_dec = m7s.encoder_layer_dispatches_decode;
                dspark_parity_ok = (sres.produced == n_gen) and
                    std.mem.eql(u32, base_toks, spec_toks) and
                    (enc_pre > 0) and (enc_dec == 0);
            }

            report.ds_staged_checked = true;
            report.ds_staged_encoder_prefill_dispatches = enc_pre;
            report.ds_staged_encoder_decode_dispatches = enc_dec;
            report.ds_staged_dspark_parity_ok = dspark_parity_ok;
            const malloc_ok = report.mtl_validation_active or (dec_res.malloc_delta <= 0);
            report.ds_staged_ok = dspark_parity_ok and malloc_ok;
            report.ds_section_ok = report.ds_kernel_parity_ok and
                report.c27_lfu_ok and
                report.ds_cold_lossless_ok and
                report.w30_ds_lora_ok and
                report.ds_staged_ok;
            return;
        }
    }

    // Offline synthetic W30 LoRA/DoRA check on `deepseek41` when staged bundles are absent.
    var s_ds = try graph.buildSynthDeepSeek41Bundle(allocator, .{ .tmp_dir = tmp_dir, .include_dspark = false });
    defer s_ds.deinit(allocator);
    const eng_c: []const [:0]const u8 = s_ds.engram_paths;
    var m_ds = try graph.GraphModel.loadFromBytes(allocator, s_ds.amf_bytes, s_ds.cfg, .{
        .max_seq_len = 64,
        .max_tree_nodes = 8,
        .ple_paths = eng_c,
        .startup_canary = false,
    });
    defer m_ds.deinit(allocator);
    try runW30DeepSeekLoraCheck(allocator, &m_ds, budget, cfg, report);

    report.ds_section_ok = report.ds_kernel_parity_ok and
        report.c27_lfu_ok and
        report.ds_cold_lossless_ok and
        report.w30_ds_lora_ok;
}

// ---------------------------------------------------------------------------
// Public Entry Points
// ---------------------------------------------------------------------------

pub fn runM94WithConfig(allocator: std.mem.Allocator, cfg: M94Config) !M94Report {
    var report: M94Report = .{
        .mtl_validation_active = envFlagEnabled("MTL_DEBUG_LAYER") or envFlagEnabled("MTL_SHADER_VALIDATION"),
    };

    const snap_start = try vmstat.Snapshot.capture();
    const tmp_dir = try graph.makeTestTempDir(allocator, "m94-bench");
    defer {
        graph.removeTestDir(allocator, tmp_dir);
        allocator.free(tmp_dir);
    }

    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    const pipes = try hybrid.HybridPipelines.init(device, lib);
    defer pipes.close();

    const bud_bytes = 64 * 1024 * 1024;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = bud_bytes + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = bud_bytes,
        .global_user_wire_limit = bud_bytes,
        .gpu_working_set_bytes = bud_bytes,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });

    try runW28GateChecks(allocator, tmp_dir, cfg, &report);
    try runQwen4KernelParity(allocator, device, &compute, pipes, budget, &report);
    try runQwen4E2eSynthAndStaged(allocator, tmp_dir, cfg, &report);
    try runDeepSeekKernelParity(allocator, device, lib, &compute, pipes, budget, tmp_dir, &report);
    try runDeepSeekColdTierAndKl(allocator, tmp_dir, device, budget, &report);
    try runDeepSeekE2eAndW30(allocator, tmp_dir, budget, cfg, &report);

    const snap_end = try vmstat.Snapshot.capture();
    report.swapouts_delta = vmstat.Snapshot.swapoutDelta(snap_start, snap_end);
    report.pageouts_delta = vmstat.Snapshot.pageoutDelta(snap_start, snap_end);
    report.malloc_blocks_delta = @max(
        @max(report.qwen4_staged_malloc_delta, report.ds_staged_malloc_delta),
        report.w30_ds_lora_malloc_delta,
    );

    const mem_ok = (report.swapouts_delta == 0) and
        (report.mtl_validation_active or (report.malloc_blocks_delta <= 0));
    report.all_passed = report.w28_ok and
        report.qwen4_section_ok and
        report.ds_section_ok and
        mem_ok;

    return report;
}

pub fn runM94(allocator: std.mem.Allocator) !M94Report {
    return runM94WithConfig(allocator, .{});
}

test "M9.4 W28, Qwen3.8-Flash-Next, DeepSeek-V4.1, 3L DSpark & W30 parity gate passes with zero swapouts" {
    const allocator = std.testing.allocator;
    const report = try runM94WithConfig(allocator, .{
        .warmup_iters = 2,
        .bench_iters = 4,
        .e2e_tokens = 6,
        .spec_tokens = 6,
        .kl_steps = 4,
        .lora_iters = 4,
    });
    try std.testing.expect(report.w28_synth_ok);
    try std.testing.expect(report.w28_ok);
    try std.testing.expect(report.qwen4_kernel_parity_ok);
    try std.testing.expect(report.qwen4_synth_wide_tree_ok);
    try std.testing.expect(report.qwen4_vision_sidecar_ok);
    try std.testing.expect(report.qwen4_section_ok);
    try std.testing.expect(report.ds_kernel_parity_ok);
    try std.testing.expect(report.c27_lfu_ok);
    try std.testing.expect(report.ds_cold_lossless_ok);
    try std.testing.expect(report.w30_ds_lora_ok);
    if (!report.ds_section_ok) {
        std.debug.print(
            "M9.4 ds_section_ok failed: staged_checked={} dspark_parity_ok={} staged_ok={} ds_malloc_delta={} enc_pre={} enc_dec={}\n",
            .{
                report.ds_staged_checked,
                report.ds_staged_dspark_parity_ok,
                report.ds_staged_ok,
                report.ds_staged_malloc_delta,
                report.ds_staged_encoder_prefill_dispatches,
                report.ds_staged_encoder_decode_dispatches,
            },
        );
    }
    try std.testing.expect(report.ds_section_ok);
    try std.testing.expectEqual(@as(u64, 0), report.swapouts_delta);
    if (report.staged_models_present) {
        try std.testing.expect(report.w28_staged_files_checked > 0);
        try std.testing.expectEqual(@as(u32, 0), report.w28_staged_long_names);
        try std.testing.expectEqual(@as(u32, 0), report.w28_staged_crc_errors);
        try std.testing.expect(report.qwen4_staged_ok);
        try std.testing.expect(report.ds_staged_ok);
    }
    try std.testing.expect(report.all_passed);
}
