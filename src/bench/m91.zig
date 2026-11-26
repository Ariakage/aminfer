//! Milestone M9.1 (`W25`, `G26`, `D15`) `aminfer bench-m91` benchmark and
//! end-to-end verification suite for the `MiMo-V2.6` and `qwen3_5` families.
//!
//! Zero hardcoded volume paths; all synthetic checkpoints, draft heads, vision
//! sidecars, and LoRA/DoRA adapters are created inside a temporary directory and
//! cleaned up on return. Enforces `swapouts_delta == 0` and steady-state
//! `malloc_blocks_delta <= 0`.

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const arena_mod = @import("../mem/arena.zig");
const vmstat = @import("../platform/vmstat.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const hybrid = @import("../gpu/hybrid.zig");
const vision_mod = @import("../gpu/vision.zig");
const graph = @import("../model/graph.zig");
const lora = @import("../model/lora.zig");
const draft = @import("../spec/draft.zig");
const dflare_dfly = @import("../spec/dflare_dfly.zig");
const tree_mod = @import("../spec/tree.zig");
const engine_mod = @import("../spec/engine.zig");

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

pub const M91Config = struct {
    swa_warmup_iters: u32 = 8,
    swa_bench_iters: u32 = 32,
    e2e_tokens: u32 = 16,
    spec_tokens: u32 = 24,
    lora_iters: u32 = 12,
};

pub const M91Report = struct {
    // G26 KV footprint savings (`Flash-RL` & `Pro-RL`)
    flash_full_kv_32k_mib: f64 = 0.0,
    flash_swa_kv_32k_mib: f64 = 0.0,
    flash_kv_reduction_32k: f64 = 0.0,
    flash_full_kv_128k_mib: f64 = 0.0,
    flash_swa_kv_128k_mib: f64 = 0.0,
    flash_kv_reduction_128k: f64 = 0.0,
    pro_full_kv_32k_mib: f64 = 0.0,
    pro_swa_kv_32k_mib: f64 = 0.0,
    pro_kv_reduction_32k: f64 = 0.0,
    pro_full_kv_128k_mib: f64 = 0.0,
    pro_swa_kv_128k_mib: f64 = 0.0,
    pro_kv_reduction_128k: f64 = 0.0,

    // G26 asymmetric `192/128` `128+sink` SWA Metal 4 kernel parity & latency
    swa_prefill_cos_sim: f64 = 0.0,
    swa_decode_cos_sim: f64 = 0.0,
    swa_tree_cos_sim: f64 = 0.0,
    swa_max_abs_err: f64 = 0.0,
    swa_decode_mean_us: f64 = 0.0,
    swa_malloc_blocks_delta: i64 = 0,
    swa_parity_ok: bool = false,

    // W25+G26 `mimo_v2` E2E hot vs `C3-C5` NVMe cold-expert parity
    mimo_hot_tok_s: f64 = 0.0,
    mimo_cold_tok_s: f64 = 0.0,
    mimo_cold_fetches: u64 = 0,
    mimo_malloc_blocks_delta: i64 = 0,
    mimo_commit_malloc_blocks_delta: i64 = 0,
    mimo_parity_ok: bool = false,

    // W25+G26 `qwen3_5` `(1+w)` RMSNorm + GDN + GQA + `W20` vision parity
    qwen35_tok_s: f64 = 0.0,
    qwen35_vision_tokens: u32 = 0,
    qwen35_red_blue_logit_diff: f64 = 0.0,
    qwen35_unit_offset_logit_diff: f64 = 0.0,
    qwen35_malloc_blocks_delta: i64 = 0,
    qwen35_commit_malloc_blocks_delta: i64 = 0,
    qwen35_parity_ok: bool = false,

    // D15 chained `3L` MTP + `DFlash` (`B=8` & `B=16`) + `D7` pruning on `mimo_v2`
    mtp3_tok_s: f64 = 0.0,
    mtp3_mean_accept_len: f64 = 0.0,
    mtp3_d7_prune_reduction: f64 = 0.0,
    mtp3_parity_ok: bool = false,

    dflash_b8_tok_s: f64 = 0.0,
    dflash_b8_mean_accept_len: f64 = 0.0,
    dflash_b8_d7_prune_reduction: f64 = 0.0,
    dflash_b8_rounds: u64 = 0,
    dflash_b8_parity_ok: bool = false,

    dflash_b16_tok_s: f64 = 0.0,
    dflash_b16_mean_accept_len: f64 = 0.0,
    dflash_b16_d7_prune_reduction: f64 = 0.0,
    dflash_b16_rounds: u64 = 0,
    dflash_b16_parity_ok: bool = false,

    // W30 LoRA/DoRA compatibility on both `mimo_v2` and `qwen3_5`
    mimo_lora_logit_diff: f64 = 0.0,
    mimo_unload_max_err: f64 = 0.0,
    mimo_lora_malloc_delta: i64 = 0,
    qwen35_lora_logit_diff: f64 = 0.0,
    qwen35_unload_max_err: f64 = 0.0,
    qwen35_lora_malloc_delta: i64 = 0,
    lora_dora_ok: bool = false,

    // Global invariants
    mtl_validation_active: bool = false,
    malloc_blocks_delta: i64 = 0,
    swapouts_delta: u64 = 0,
    all_passed: bool = false,

    pub fn write(self: *const M91Report, out: *Io.Writer) !void {
        try out.print(
            \\bench.m91.g26.flash_rl.kv_32k_full_mib         {d:.2}
            \\bench.m91.g26.flash_rl.kv_32k_swa_mib          {d:.2}
            \\bench.m91.g26.flash_rl.kv_reduction_32k        {d:.2}x
            \\bench.m91.g26.flash_rl.kv_128k_full_mib        {d:.2}
            \\bench.m91.g26.flash_rl.kv_128k_swa_mib         {d:.2}
            \\bench.m91.g26.flash_rl.kv_reduction_128k       {d:.2}x
            \\bench.m91.g26.pro_rl.kv_32k_full_mib           {d:.2}
            \\bench.m91.g26.pro_rl.kv_32k_swa_mib            {d:.2}
            \\bench.m91.g26.pro_rl.kv_reduction_32k          {d:.2}x
            \\bench.m91.g26.pro_rl.kv_128k_full_mib          {d:.2}
            \\bench.m91.g26.pro_rl.kv_128k_swa_mib           {d:.2}
            \\bench.m91.g26.pro_rl.kv_reduction_128k         {d:.2}x
            \\bench.m91.g26.swa_192_128.prefill_cos_sim      {d:.6}
            \\bench.m91.g26.swa_192_128.decode_cos_sim       {d:.6}
            \\bench.m91.g26.swa_192_128.tree_cos_sim         {d:.6}
            \\bench.m91.g26.swa_192_128.max_abs_err          {d:.6}
            \\bench.m91.g26.swa_192_128.decode_mean_us       {d:.2}
            \\bench.m91.g26.swa_192_128.malloc_delta         {d}
            \\bench.m91.g26.swa_192_128.passed               {s}
            \\bench.m91.w25.mimo_v2.hot_tok_s                {d:.2}
            \\bench.m91.w25.mimo_v2.cold_tok_s               {d:.2}
            \\bench.m91.w25.mimo_v2.cold_fetches             {d}
            \\bench.m91.w25.mimo_v2.malloc_delta             {d}
            \\bench.m91.w25.mimo_v2.commit_malloc_delta      {d}
            \\bench.m91.w25.mimo_v2.parity_passed            {s}
            \\
        , .{
            self.flash_full_kv_32k_mib,
            self.flash_swa_kv_32k_mib,
            self.flash_kv_reduction_32k,
            self.flash_full_kv_128k_mib,
            self.flash_swa_kv_128k_mib,
            self.flash_kv_reduction_128k,
            self.pro_full_kv_32k_mib,
            self.pro_swa_kv_32k_mib,
            self.pro_kv_reduction_32k,
            self.pro_full_kv_128k_mib,
            self.pro_swa_kv_128k_mib,
            self.pro_kv_reduction_128k,
            self.swa_prefill_cos_sim,
            self.swa_decode_cos_sim,
            self.swa_tree_cos_sim,
            self.swa_max_abs_err,
            self.swa_decode_mean_us,
            self.swa_malloc_blocks_delta,
            if (self.swa_parity_ok) "true" else "false",
            self.mimo_hot_tok_s,
            self.mimo_cold_tok_s,
            self.mimo_cold_fetches,
            self.mimo_malloc_blocks_delta,
            self.mimo_commit_malloc_blocks_delta,
            if (self.mimo_parity_ok) "true" else "false",
        });
        try out.print(
            \\bench.m91.w25.qwen3_5.tok_s                    {d:.2}
            \\bench.m91.w25.qwen3_5.vision_tokens            {d}
            \\bench.m91.w25.qwen3_5.red_blue_logit_diff      {d:.6}
            \\bench.m91.w25.qwen3_5.unit_offset_logit_diff   {d:.6}
            \\bench.m91.w25.qwen3_5.malloc_delta             {d}
            \\bench.m91.w25.qwen3_5.commit_malloc_delta      {d}
            \\bench.m91.w25.qwen3_5.parity_passed            {s}
            \\bench.m91.d15.mtp3.tok_s                       {d:.2}
            \\bench.m91.d15.mtp3.mean_accept_len             {d:.3}
            \\bench.m91.d15.mtp3.d7_prune_reduction          {d:.4}
            \\bench.m91.d15.mtp3.parity_passed               {s}
            \\bench.m91.d15.dflash_b8.tok_s                  {d:.2}
            \\bench.m91.d15.dflash_b8.mean_accept_len        {d:.3}
            \\bench.m91.d15.dflash_b8.d7_prune_reduction     {d:.4}
            \\bench.m91.d15.dflash_b8.rounds                 {d}
            \\bench.m91.d15.dflash_b8.parity_passed          {s}
            \\bench.m91.d15.dflash_b16.tok_s                 {d:.2}
            \\bench.m91.d15.dflash_b16.mean_accept_len       {d:.3}
            \\bench.m91.d15.dflash_b16.d7_prune_reduction    {d:.4}
            \\bench.m91.d15.dflash_b16.rounds                {d}
            \\bench.m91.d15.dflash_b16.parity_passed         {s}
            \\bench.m91.w30.mimo_v2.lora_logit_diff          {d:.6}
            \\bench.m91.w30.mimo_v2.unload_max_err           {d:.6}
            \\bench.m91.w30.mimo_v2.malloc_delta             {d}
            \\bench.m91.w30.qwen3_5.lora_logit_diff          {d:.6}
            \\bench.m91.w30.qwen3_5.unload_max_err           {d:.6}
            \\bench.m91.w30.qwen3_5.malloc_delta             {d}
            \\bench.m91.w30.lora_dora.passed                 {s}
            \\bench.m91.mtl_validation_active                {s}
            \\bench.m91.malloc_blocks_delta                  {d}
            \\bench.m91.swapouts_delta                       {d}
            \\bench.m91.all_passed                           {s}
            \\
        , .{
            self.qwen35_tok_s,
            self.qwen35_vision_tokens,
            self.qwen35_red_blue_logit_diff,
            self.qwen35_unit_offset_logit_diff,
            self.qwen35_malloc_blocks_delta,
            self.qwen35_commit_malloc_blocks_delta,
            if (self.qwen35_parity_ok) "true" else "false",
            self.mtp3_tok_s,
            self.mtp3_mean_accept_len,
            self.mtp3_d7_prune_reduction,
            if (self.mtp3_parity_ok) "true" else "false",
            self.dflash_b8_tok_s,
            self.dflash_b8_mean_accept_len,
            self.dflash_b8_d7_prune_reduction,
            self.dflash_b8_rounds,
            if (self.dflash_b8_parity_ok) "true" else "false",
            self.dflash_b16_tok_s,
            self.dflash_b16_mean_accept_len,
            self.dflash_b16_d7_prune_reduction,
            self.dflash_b16_rounds,
            if (self.dflash_b16_parity_ok) "true" else "false",
            self.mimo_lora_logit_diff,
            self.mimo_unload_max_err,
            self.mimo_lora_malloc_delta,
            self.qwen35_lora_logit_diff,
            self.qwen35_unload_max_err,
            self.qwen35_lora_malloc_delta,
            if (self.lora_dora_ok) "true" else "false",
            if (self.mtl_validation_active) "true" else "false",
            self.malloc_blocks_delta,
            self.swapouts_delta,
            if (self.all_passed) "true" else "false",
        });
    }

    pub fn writeJson(self: *const M91Report, out: *Io.Writer) !void {
        try out.print(
            \\{{"flash_kv_reduction_32k":{d:.4},"flash_kv_reduction_128k":{d:.4},"pro_kv_reduction_32k":{d:.4},"pro_kv_reduction_128k":{d:.4},"swa_prefill_cos_sim":{d:.6},"swa_decode_cos_sim":{d:.6},"swa_tree_cos_sim":{d:.6},"swa_max_abs_err":{d:.6},"swa_decode_mean_us":{d:.2},"swa_malloc_delta":{d},"swa_parity_ok":{s},"mimo_hot_tok_s":{d:.2},"mimo_cold_tok_s":{d:.2},"mimo_cold_fetches":{d},"mimo_malloc_delta":{d},"mimo_commit_malloc_delta":{d},"mimo_parity_ok":{s},"qwen35_tok_s":{d:.2},"qwen35_vision_tokens":{d},"qwen35_red_blue_logit_diff":{d:.6},"qwen35_unit_offset_logit_diff":{d:.6},"qwen35_malloc_delta":{d},"qwen35_commit_malloc_delta":{d},"qwen35_parity_ok":{s},
        , .{
            self.flash_kv_reduction_32k,
            self.flash_kv_reduction_128k,
            self.pro_kv_reduction_32k,
            self.pro_kv_reduction_128k,
            self.swa_prefill_cos_sim,
            self.swa_decode_cos_sim,
            self.swa_tree_cos_sim,
            self.swa_max_abs_err,
            self.swa_decode_mean_us,
            self.swa_malloc_blocks_delta,
            if (self.swa_parity_ok) "true" else "false",
            self.mimo_hot_tok_s,
            self.mimo_cold_tok_s,
            self.mimo_cold_fetches,
            self.mimo_malloc_blocks_delta,
            self.mimo_commit_malloc_blocks_delta,
            if (self.mimo_parity_ok) "true" else "false",
            self.qwen35_tok_s,
            self.qwen35_vision_tokens,
            self.qwen35_red_blue_logit_diff,
            self.qwen35_unit_offset_logit_diff,
            self.qwen35_malloc_blocks_delta,
            self.qwen35_commit_malloc_blocks_delta,
            if (self.qwen35_parity_ok) "true" else "false",
        });
        try out.print(
            \\"mtp3_tok_s":{d:.2},"mtp3_mean_accept_len":{d:.3},"mtp3_d7_prune_reduction":{d:.4},"mtp3_parity_ok":{s},"dflash_b8_tok_s":{d:.2},"dflash_b8_mean_accept_len":{d:.3},"dflash_b8_d7_prune_reduction":{d:.4},"dflash_b8_parity_ok":{s},"dflash_b16_tok_s":{d:.2},"dflash_b16_mean_accept_len":{d:.3},"dflash_b16_d7_prune_reduction":{d:.4},"dflash_b16_parity_ok":{s},"mimo_lora_logit_diff":{d:.6},"mimo_unload_max_err":{d:.6},"qwen35_lora_logit_diff":{d:.6},"qwen35_unload_max_err":{d:.6},"lora_dora_ok":{s},"mtl_validation_active":{s},"malloc_blocks_delta":{d},"swapouts_delta":{d},"all_passed":{s}}}
            \\
        , .{
            self.mtp3_tok_s,
            self.mtp3_mean_accept_len,
            self.mtp3_d7_prune_reduction,
            if (self.mtp3_parity_ok) "true" else "false",
            self.dflash_b8_tok_s,
            self.dflash_b8_mean_accept_len,
            self.dflash_b8_d7_prune_reduction,
            if (self.dflash_b8_parity_ok) "true" else "false",
            self.dflash_b16_tok_s,
            self.dflash_b16_mean_accept_len,
            self.dflash_b16_d7_prune_reduction,
            if (self.dflash_b16_parity_ok) "true" else "false",
            self.mimo_lora_logit_diff,
            self.mimo_unload_max_err,
            self.qwen35_lora_logit_diff,
            self.qwen35_unload_max_err,
            if (self.lora_dora_ok) "true" else "false",
            if (self.mtl_validation_active) "true" else "false",
            self.malloc_blocks_delta,
            self.swapouts_delta,
            if (self.all_passed) "true" else "false",
        });
    }
};

fn deterministicF32(seed: u64, idx: usize, amp: f32, bias: f32) f32 {
    var x = seed ^ (@as(u64, idx) *% 0x9e3779b97f4a7c15);
    x ^= x >> 30;
    x *%= 0xbf58476d1ce4e5b9;
    x ^= x >> 27;
    x *%= 0x94d049bb133111eb;
    x ^= x >> 31;
    const u = @as(f32, @floatFromInt(x & 0xFFFF)) / 32767.5 - 1.0;
    return bias + u * amp;
}

fn fillDeterministicF16(allocator: std.mem.Allocator, seed: u64, count: usize, amp: f32) ![]u8 {
    const out = try allocator.alloc(u8, count * 2);
    for (0..count) |i| {
        const h: f16 = @floatCast(deterministicF32(seed, i, amp, 0.0));
        std.mem.writeInt(u16, out[i * 2 ..][0..2], @bitCast(h), .little);
    }
    return out;
}

fn fillDeterministicBf16(allocator: std.mem.Allocator, seed: u64, count: usize, amp: f32, bias: f32) ![]u8 {
    const out = try allocator.alloc(u8, count * 2);
    for (0..count) |i| {
        const b = quant.f32ToFloatBits(.bf16, deterministicF32(seed, i, amp, bias));
        std.mem.writeInt(u16, out[i * 2 ..][0..2], b, .little);
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

fn fillDeterministicMxfp4(allocator: std.mem.Allocator, seed: u64, rows: usize, cols: usize) ![]u8 {
    if ((cols % 32) != 0) return error.InvalidShape;
    const blocks_per_row = cols / 32;
    const total_blocks = rows * blocks_per_row;
    const out = try allocator.alloc(u8, total_blocks * 17);
    for (0..total_blocks) |bi| {
        const blk = out[bi * 17 ..][0..17];
        blk[0] = 124; // UE8M0 scale = 2^(124-127) * 0.5 = 0.0625 -> bounded activations
        for (0..16) |pi| {
            var x = seed ^ (@as(u64, bi * 16 + pi) *% 0x9e3779b97f4a7c15);
            x ^= x >> 30;
            x *%= 0xbf58476d1ce4e5b9;
            x ^= x >> 27;
            blk[1 + pi] = @truncate(x);
        }
    }
    return out;
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

fn cosineF64(a: []const f32, b: []const f32) struct { cos: f64, max_err: f64 } {
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

fn cpuRmsNormRopeKv(
    q_in: []const f16,
    k_in: []const f16,
    v_in: []const f16,
    q_norm_bf16: []const u16,
    k_norm_bf16: []const u16,
    num_tokens: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    rotary_dim: u32,
    start_pos: u32,
    rope_theta: f32,
    v_scale: f32,
    rms_eps: f32,
    q_out: []f32,
    k_out: []f32,
    v_out: []f32,
) void {
    const q_stride: usize = @as(usize, num_q_heads) * k_head_dim;
    const k_stride: usize = @as(usize, num_kv_heads) * k_head_dim;
    const v_stride: usize = @as(usize, num_kv_heads) * v_head_dim;
    const half_rope: usize = rotary_dim / 2;

    for (0..num_tokens) |t| {
        const pos_f: f32 = @floatFromInt(start_pos + @as(u32, @intCast(t)));
        // Q heads
        for (0..num_q_heads) |h| {
            const base = t * q_stride + h * k_head_dim;
            var ss: f32 = 0.0;
            for (0..k_head_dim) |d| {
                const x: f32 = @floatCast(q_in[base + d]);
                ss += x * x;
            }
            const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(k_head_dim)) + rms_eps);
            for (0..k_head_dim) |d| {
                const w = quant.floatBitsToF32(.bf16, q_norm_bf16[d]);
                q_out[base + d] = @as(f32, @floatCast(q_in[base + d])) * inv * w;
            }
            for (0..half_rope) |i| {
                const freq = 1.0 / std.math.pow(f32, rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rotary_dim)));
                const angle = pos_f * freq;
                const c = @cos(angle);
                const s = @sin(angle);
                const x0 = q_out[base + i];
                const x1 = q_out[base + half_rope + i];
                q_out[base + i] = x0 * c - x1 * s;
                q_out[base + half_rope + i] = x1 * c + x0 * s;
            }
        }
        // K + V heads
        for (0..num_kv_heads) |h| {
            const kb = t * k_stride + h * k_head_dim;
            var ss: f32 = 0.0;
            for (0..k_head_dim) |d| {
                const x: f32 = @floatCast(k_in[kb + d]);
                ss += x * x;
            }
            const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(k_head_dim)) + rms_eps);
            for (0..k_head_dim) |d| {
                const w = quant.floatBitsToF32(.bf16, k_norm_bf16[d]);
                k_out[kb + d] = @as(f32, @floatCast(k_in[kb + d])) * inv * w;
            }
            for (0..half_rope) |i| {
                const freq = 1.0 / std.math.pow(f32, rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rotary_dim)));
                const angle = pos_f * freq;
                const c = @cos(angle);
                const s = @sin(angle);
                const x0 = k_out[kb + i];
                const x1 = k_out[kb + half_rope + i];
                k_out[kb + i] = x0 * c - x1 * s;
                k_out[kb + half_rope + i] = x1 * c + x0 * s;
            }
            const vb = t * v_stride + h * v_head_dim;
            for (0..v_head_dim) |d| {
                v_out[vb + d] = @as(f32, @floatCast(v_in[vb + d])) * v_scale;
            }
        }
    }
}

fn cpuSwaAttnSinkRef(
    q: []const f32,
    k_hist: []const f32,
    v_hist: []const f32,
    sink_bf16: []const u16,
    q_pos: u32,
    kv_len_total: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    window_size: u32,
    extra_k: ?[]const f32,
    extra_v: ?[]const f32,
    ancestor_mask: u64,
    extra_positions: ?[]const u32,
    out: []f32,
) void {
    const gqa_ratio = num_q_heads / num_kv_heads;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(k_head_dim)));
    const k_stride: usize = @as(usize, num_kv_heads) * k_head_dim;
    const v_stride: usize = @as(usize, num_kv_heads) * v_head_dim;

    for (0..num_q_heads) |qh| {
        const kvh = qh / gqa_ratio;
        const qb = qh * k_head_dim;
        const ob = qh * v_head_dim;
        const win_start: u32 = if (q_pos + 1 > window_size) (q_pos + 1 - window_size) else 0;

        var m: f32 = quant.floatBitsToF32(.bf16, sink_bf16[qh]);
        var l: f32 = 1.0;
        @memset(out[ob .. ob + v_head_dim], 0.0);

        // Ring history `[0..kv_len_total)`
        for (win_start..kv_len_total) |kp_usize| {
            if (kp_usize > q_pos) break;
            const kb = kp_usize * k_stride + kvh * k_head_dim;
            const vb = kp_usize * v_stride + kvh * v_head_dim;
            var dot: f32 = 0.0;
            for (0..k_head_dim) |d| dot += q[qb + d] * k_hist[kb + d];
            const s = dot * scale;
            const m_new = @max(m, s);
            const alpha = @exp(m - m_new);
            const beta = @exp(s - m_new);
            for (0..v_head_dim) |d| {
                out[ob + d] = out[ob + d] * alpha + beta * v_hist[vb + d];
            }
            l = l * alpha + beta;
            m = m_new;
        }

        // Optional speculative tree nodes
        if (extra_k) |ek| {
            const ev = extra_v.?;
            const pos_slice = extra_positions.?;
            var mask = ancestor_mask;
            while (mask != 0) {
                const ni: u32 = @ctz(mask);
                mask &= mask - 1;
                const kp = pos_slice[ni];
                if (kp < win_start or kp > q_pos) continue;
                const kb = @as(usize, ni) * k_stride + kvh * k_head_dim;
                const vb = @as(usize, ni) * v_stride + kvh * v_head_dim;
                var dot: f32 = 0.0;
                for (0..k_head_dim) |d| dot += q[qb + d] * ek[kb + d];
                const s = dot * scale;
                const m_new = @max(m, s);
                const alpha = @exp(m - m_new);
                const beta = @exp(s - m_new);
                for (0..v_head_dim) |d| {
                    out[ob + d] = out[ob + d] * alpha + beta * ev[vb + d];
                }
                l = l * alpha + beta;
                m = m_new;
            }
        }

        const inv_l = 1.0 / l;
        for (0..v_head_dim) |d| out[ob + d] *= inv_l;
    }
}

fn writeFileBytesPosix(path: [:0]const u8, bytes: []const u8) !void {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.WriteFailed;
    defer _ = std.c.close(fd);
    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + off, bytes.len - off);
        if (n < 0) {
            if (std.c.errno(n) == .INTR) continue;
            return error.WriteFailed;
        }
        off += @intCast(n);
    }
}

fn runG26KvAndSwaBench(
    allocator: std.mem.Allocator,
    device: metal.Device,
    library: metal.Library,
    budget: budget_mod.Budget,
    cfg: M91Config,
    report: *M91Report,
) !void {
    // 1. Exact KV cache footprint savings for MiMo-V2.6 Flash-RL (39/48 SWA) and Pro-RL (60/70 SWA).
    const mib: f64 = 1024.0 * 1024.0;
    const bytes_per_kv_head_tok: f64 = (192.0 + 128.0) * 2.0; // FP16 RGBA16Float K(192)+V(128)
    inline for (.{ 32768.0, 131072.0 }, 0..) |ctx, idx| {
        const flash_full = 48.0 * 8.0 * ctx * bytes_per_kv_head_tok / mib;
        const flash_swa = (39.0 * 8.0 * 128.0 + 9.0 * 4.0 * ctx) * bytes_per_kv_head_tok / mib;
        const pro_full = 70.0 * 8.0 * ctx * bytes_per_kv_head_tok / mib;
        const pro_swa = (60.0 * 8.0 * 128.0 + 10.0 * 8.0 * ctx) * bytes_per_kv_head_tok / mib;
        if (idx == 0) {
            report.flash_full_kv_32k_mib = flash_full;
            report.flash_swa_kv_32k_mib = flash_swa;
            report.flash_kv_reduction_32k = flash_full / flash_swa;
            report.pro_full_kv_32k_mib = pro_full;
            report.pro_swa_kv_32k_mib = pro_swa;
            report.pro_kv_reduction_32k = pro_full / pro_swa;
        } else {
            report.flash_full_kv_128k_mib = flash_full;
            report.flash_swa_kv_128k_mib = flash_swa;
            report.flash_kv_reduction_128k = flash_full / flash_swa;
            report.pro_full_kv_128k_mib = pro_full;
            report.pro_swa_kv_128k_mib = pro_swa;
            report.pro_kv_reduction_128k = pro_full / pro_swa;
        }
    }

    // 2. Asymmetric 192/128 128+sink HybridKvRing SWA GPU parity (`prefill` wrap + `decode` + `M=12 verifyTree`).
    const pipes = try hybrid.HybridPipelines.init(device, library);
    defer pipes.close();

    const num_qh: u32 = 8;
    const num_kvh: u32 = 2;
    const k_hd: u32 = 192;
    const v_hd: u32 = 128;
    const rotary_dim: u32 = 64;
    const win_sz: u32 = 128;
    const prefill_toks: u32 = 140; // Wraps circular 128-token ring!
    const tree_m: u32 = 12;
    const v_scale: f32 = 0.707;
    const rms_eps: f32 = 1e-6;
    const rope_theta: f32 = 10_000.0;

    var arena = try arena_mod.WiredArena.init(try page.alignUp(8 * 1024 * 1024), budget);
    defer arena.deinit();

    var ring = try hybrid.HybridKvRing.init(allocator, device, &arena, .{
        .mode = .sliding_window_128,
        .num_tiles = 1,
        .num_kv_heads = num_kvh,
        .head_dim = k_hd,
        .v_head_dim = v_hd,
    });
    defer ring.deinit(allocator);

    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{buf});
    try ring.makeResident(compute);
    defer ring.removeResident(compute) catch |err| {
        std.debug.panic("removeResident failed: {}", .{err});
    };

    const q_dim: usize = @as(usize, num_qh) * k_hd;
    const k_dim: usize = @as(usize, num_kvh) * k_hd;
    const v_dim: usize = @as(usize, num_kvh) * v_hd;
    const o_dim: usize = @as(usize, num_qh) * v_hd;

    const q_raw_sl = try arena.allocAligned(prefill_toks * q_dim * 2, 64);
    const k_raw_sl = try arena.allocAligned(prefill_toks * k_dim * 2, 64);
    const v_raw_sl = try arena.allocAligned(prefill_toks * v_dim * 2, 64);
    const k_ext_sl = try arena.allocAligned(prefill_toks * k_dim * 2, 64);
    const v_ext_sl = try arena.allocAligned(prefill_toks * v_dim * 2, 64);
    const qn_sl = try arena.allocAligned(k_hd * 2, 64);
    const kn_sl = try arena.allocAligned(k_hd * 2, 64);
    const sink_sl = try arena.allocAligned(num_qh * 2, 64);
    const out_sl = try arena.allocAligned(prefill_toks * o_dim * 2, 64);
    const anc_sl = try arena.allocAligned(tree_m * @sizeOf(u64), 64);
    const pos_sl = try arena.allocAligned(tree_m * @sizeOf(u32), 64);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw_sl[0 .. prefill_toks * q_dim * 2]));
    const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw_sl[0 .. prefill_toks * k_dim * 2]));
    const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw_sl[0 .. prefill_toks * v_dim * 2]));
    const qn_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, qn_sl[0 .. k_hd * 2]));
    const kn_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, kn_sl[0 .. k_hd * 2]));
    const sink_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, sink_sl[0 .. num_qh * 2]));
    const out_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, out_sl[0 .. prefill_toks * o_dim * 2]));
    const anc_u64: []u64 = @alignCast(std.mem.bytesAsSlice(u64, anc_sl[0 .. tree_m * 8]));
    const pos_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, pos_sl[0 .. tree_m * 4]));

    for (q_f16, 0..) |*dst, idx| dst.* = @floatCast(deterministicF32(101, idx, 0.35, 0.0));
    for (k_f16, 0..) |*dst, idx| dst.* = @floatCast(deterministicF32(202, idx, 0.35, 0.0));
    for (v_f16, 0..) |*dst, idx| dst.* = @floatCast(deterministicF32(303, idx, 0.35, 0.0));
    for (qn_u16, 0..) |*dst, idx| dst.* = quant.f32ToFloatBits(.bf16, deterministicF32(404, idx, 0.15, 1.0));
    for (kn_u16, 0..) |*dst, idx| dst.* = quant.f32ToFloatBits(.bf16, deterministicF32(505, idx, 0.15, 1.0));
    for (sink_u16, 0..) |*dst, idx| dst.* = quant.f32ToFloatBits(.bf16, deterministicF32(606, idx, 1.2, 0.5));

    const bind = struct {
        fn of(b: metal.Buffer, a: *const arena_mod.WiredArena, ptr: anytype) metal.BufferBinding {
            return .{ .buffer = b, .offset = a.offsetOf(@ptrCast(ptr)) };
        }
    }.of;

    // CPU reference buffers for prefill (140 tokens) + 1 decode step + M=12 verifyTree
    const total_hist_toks = prefill_toks + 1 + tree_m;
    const cpu_q = try allocator.alloc(f32, total_hist_toks * q_dim);
    defer allocator.free(cpu_q);
    const cpu_k = try allocator.alloc(f32, total_hist_toks * k_dim);
    defer allocator.free(cpu_k);
    const cpu_v = try allocator.alloc(f32, total_hist_toks * v_dim);
    defer allocator.free(cpu_v);
    const gpu_out_f32 = try allocator.alloc(f32, prefill_toks * o_dim);
    defer allocator.free(gpu_out_f32);
    const cpu_out_f32 = try allocator.alloc(f32, prefill_toks * o_dim);
    defer allocator.free(cpu_out_f32);

    cpuRmsNormRopeKv(
        q_f16,
        k_f16,
        v_f16,
        qn_u16,
        kn_u16,
        prefill_toks,
        num_qh,
        num_kvh,
        k_hd,
        v_hd,
        rotary_dim,
        0,
        rope_theta,
        v_scale,
        rms_eps,
        cpu_q[0 .. prefill_toks * q_dim],
        cpu_k[0 .. prefill_toks * k_dim],
        cpu_v[0 .. prefill_toks * v_dim],
    );

    const tile_bind = bind(buf, &arena, ring.tile_table.ptr);
    const q_bind = bind(buf, &arena, q_raw_sl.ptr);
    const k_bind = bind(buf, &arena, k_raw_sl.ptr);
    const v_bind = bind(buf, &arena, v_raw_sl.ptr);
    const k_ext_bind = bind(buf, &arena, k_ext_sl.ptr);
    const v_ext_bind = bind(buf, &arena, v_ext_sl.ptr);
    const qn_bind = bind(buf, &arena, qn_sl.ptr);
    const kn_bind = bind(buf, &arena, kn_sl.ptr);
    const sink_bind = bind(buf, &arena, sink_sl.ptr);
    const out_bind = bind(buf, &arena, out_sl.ptr);
    const anc_bind = bind(buf, &arena, anc_sl.ptr);
    const pos_bind = bind(buf, &arena, pos_sl.ptr);

    // 2A. Prefill 140 tokens (exercising circular wrap on 128-token ring + sink)
    {
        var frame = try compute.beginFrame(10_000);
        errdefer frame.abort();
        try pipes.encodeG26QkNormRopeKv(&frame, .{
            .q = q_bind,
            .k = k_bind,
            .v = v_bind,
            .q_norm = qn_bind,
            .k_norm = kn_bind,
            .norm_dtype = .bf16,
            .k_out = k_ext_bind,
            .v_out = v_ext_bind,
            .num_tokens = prefill_toks,
            .num_q_heads = num_qh,
            .num_kv_heads = num_kvh,
            .k_head_dim = k_hd,
            .v_head_dim = v_hd,
            .rotary_dim = rotary_dim,
            .start_pos = 0,
            .chunk_tokens = win_sz,
            .ring_capacity = win_sz,
            .write_to_tiles = false,
            .mrope = hybrid.MropeSection.text1d(rotary_dim),
            .rms_eps = rms_eps,
            .rope_theta = rope_theta,
            .v_scale = v_scale,
        }, false);
        try pipes.encodeG26SwaAttn(&frame, .{
            .q = q_bind,
            .kv_tiles = tile_bind,
            .k_extra = k_ext_bind,
            .v_extra = v_ext_bind,
            .sink_bias = sink_bind,
            .sink_dtype = .bf16,
            .out = out_bind,
            .num_q = prefill_toks,
            .q_start_pos = 0,
            .chunk_start_pos = 0,
            .kv_len = prefill_toks,
            .num_q_heads = num_qh,
            .num_kv_heads = num_kvh,
            .k_head_dim = k_hd,
            .v_head_dim = v_hd,
            .window_size = win_sz,
            .ring_capacity = win_sz,
            .chunk_tokens = win_sz,
            .is_prefill = true,
        }, true);
        try pipes.encodeG26KvCommit(&frame, .{
            .k_src = k_ext_bind,
            .v_src = v_ext_bind,
            .kv_tiles = tile_bind,
            .num_kv_heads = num_kvh,
            .k_head_dim = k_hd,
            .v_head_dim = v_hd,
            .start_pos = 0,
            .num_tokens = prefill_toks,
            .chunk_tokens = win_sz,
            .ring_capacity = win_sz,
        }, true);
        const sig_pf = try frame.commit();
        try compute.wait(sig_pf, 10_000);
    }

    for (0..prefill_toks) |t| {
        cpuSwaAttnSinkRef(
            cpu_q[t * q_dim ..][0..q_dim],
            cpu_k[0 .. (t + 1) * k_dim],
            cpu_v[0 .. (t + 1) * v_dim],
            sink_u16,
            @intCast(t),
            @intCast(t + 1),
            num_qh,
            num_kvh,
            k_hd,
            v_hd,
            win_sz,
            null,
            null,
            0,
            null,
            cpu_out_f32[t * o_dim ..][0..o_dim],
        );
    }
    for (0..prefill_toks * o_dim) |i| gpu_out_f32[i] = @floatCast(out_f16[i]);
    const pf_cmp = cosineF64(cpu_out_f32, gpu_out_f32);
    report.swa_prefill_cos_sim = pf_cmp.cos;
    report.swa_max_abs_err = pf_cmp.max_err;

    // 2B. M = 1 Decode parity + zero-malloc steady-state microbenchmark at pos = 140
    for (q_f16[0..q_dim], 0..) |*dst, idx| dst.* = @floatCast(deterministicF32(707, idx, 0.35, 0.0));
    for (k_f16[0..k_dim], 0..) |*dst, idx| dst.* = @floatCast(deterministicF32(808, idx, 0.35, 0.0));
    for (v_f16[0..v_dim], 0..) |*dst, idx| dst.* = @floatCast(deterministicF32(909, idx, 0.35, 0.0));

    cpuRmsNormRopeKv(
        q_f16[0..q_dim],
        k_f16[0..k_dim],
        v_f16[0..v_dim],
        qn_u16,
        kn_u16,
        1,
        num_qh,
        num_kvh,
        k_hd,
        v_hd,
        rotary_dim,
        prefill_toks,
        rope_theta,
        v_scale,
        rms_eps,
        cpu_q[prefill_toks * q_dim ..][0..q_dim],
        cpu_k[prefill_toks * k_dim ..][0..k_dim],
        cpu_v[prefill_toks * v_dim ..][0..v_dim],
    );

    anc_u64[0] = 1;
    pos_u32[0] = prefill_toks;
    const encodeSwaStep = struct {
        fn call(
            frame: *metal.Compute.Frame,
            p: hybrid.HybridPipelines,
            qb: metal.BufferBinding,
            kb: metal.BufferBinding,
            vb: metal.BufferBinding,
            qnb: metal.BufferBinding,
            knb: metal.BufferBinding,
            tb: metal.BufferBinding,
            sb: metal.BufferBinding,
            ob: metal.BufferBinding,
            anc_b: metal.BufferBinding,
            pos_b: metal.BufferBinding,
            pos: u32,
            k_ext_b: metal.BufferBinding,
            v_ext_b: metal.BufferBinding,
            commit_into_ring: bool,
        ) !void {
            try p.encodeG26QkNormRopeKv(frame, .{
                .q = qb,
                .k = kb,
                .v = vb,
                .q_norm = qnb,
                .k_norm = knb,
                .norm_dtype = .bf16,
                .positions_1d = pos_b,
                .k_out = k_ext_b,
                .v_out = v_ext_b,
                .num_tokens = 1,
                .num_q_heads = num_qh,
                .num_kv_heads = num_kvh,
                .k_head_dim = k_hd,
                .v_head_dim = v_hd,
                .rotary_dim = rotary_dim,
                .start_pos = pos,
                .chunk_tokens = win_sz,
                .ring_capacity = win_sz,
                .write_to_tiles = false,
                .mrope = hybrid.MropeSection.text1d(rotary_dim),
                .rms_eps = rms_eps,
                .rope_theta = rope_theta,
                .v_scale = v_scale,
            }, false);
            try p.encodeG26SwaAttn(frame, .{
                .q = qb,
                .kv_tiles = tb,
                .k_extra = k_ext_b,
                .v_extra = v_ext_b,
                .ancestor_masks = anc_b,
                .node_positions = pos_b,
                .sink_bias = sb,
                .sink_dtype = .bf16,
                .out = ob,
                .num_q = 1,
                .q_start_pos = pos,
                .chunk_start_pos = 0,
                .kv_len = pos,
                .num_q_heads = num_qh,
                .num_kv_heads = num_kvh,
                .k_head_dim = k_hd,
                .v_head_dim = v_hd,
                .window_size = win_sz,
                .ring_capacity = win_sz,
                .chunk_tokens = win_sz,
                .is_prefill = false,
            }, true);
            if (commit_into_ring) {
                const acc0 = [_]u16{0};
                try p.encodeG26KvCommit(frame, .{
                    .k_src = k_ext_b,
                    .v_src = v_ext_b,
                    .kv_tiles = tb,
                    .num_kv_heads = num_kvh,
                    .k_head_dim = k_hd,
                    .v_head_dim = v_hd,
                    .start_pos = pos,
                    .accepted_nodes = &acc0,
                    .chunk_tokens = win_sz,
                    .ring_capacity = win_sz,
                }, true);
            }
        }
    }.call;

    const runSwaStep = struct {
        fn call(
            comp: *metal.Compute,
            p: hybrid.HybridPipelines,
            qb: metal.BufferBinding,
            kb: metal.BufferBinding,
            vb: metal.BufferBinding,
            qnb: metal.BufferBinding,
            knb: metal.BufferBinding,
            tb: metal.BufferBinding,
            sb: metal.BufferBinding,
            ob: metal.BufferBinding,
            anc_b: metal.BufferBinding,
            pos_b: metal.BufferBinding,
            pos: u32,
            k_ext_b: metal.BufferBinding,
            v_ext_b: metal.BufferBinding,
            commit_into_ring: bool,
        ) !void {
            var frame = try comp.beginFrame(10_000);
            errdefer frame.abort();
            try encodeSwaStep(&frame, p, qb, kb, vb, qnb, knb, tb, sb, ob, anc_b, pos_b, pos, k_ext_b, v_ext_b, commit_into_ring);
            const sig = try frame.commit();
            try comp.wait(sig, 10_000);
        }
    }.call;

    // Parity eval first (commits pos=140 so window becomes [13..140])
    try runSwaStep(&compute, pipes, q_bind, k_bind, v_bind, qn_bind, kn_bind, tile_bind, sink_bind, out_bind, anc_bind, pos_bind, prefill_toks, k_ext_bind, v_ext_bind, true);
    cpuSwaAttnSinkRef(
        cpu_q[prefill_toks * q_dim ..][0..q_dim],
        cpu_k[0 .. (prefill_toks + 1) * k_dim],
        cpu_v[0 .. (prefill_toks + 1) * v_dim],
        sink_u16,
        prefill_toks,
        prefill_toks + 1,
        num_qh,
        num_kvh,
        k_hd,
        v_hd,
        win_sz,
        null,
        null,
        0,
        null,
        cpu_out_f32[0..o_dim],
    );
    for (0..o_dim) |i| gpu_out_f32[i] = @floatCast(out_f16[i]);
    const dec_cmp = cosineF64(cpu_out_f32[0..o_dim], gpu_out_f32[0..o_dim]);
    report.swa_decode_cos_sim = dec_cmp.cos;
    report.swa_max_abs_err = @max(report.swa_max_abs_err, dec_cmp.max_err);

    // 2C. M = 12 branching speculative `verifyTree` parity (`kv_tiles` + `k_extra` tree nodes)
    const tree_start = prefill_toks + 1;
    // Branching tree: 0 -> 1 -> 2 -> 3; 0 -> 4 -> 5; 2 -> 6 -> 7 -> 8 -> 9; 4 -> 10 -> 11
    const parents = [12]u32{ 0, 0, 1, 2, 0, 4, 2, 6, 7, 8, 4, 10 };
    for (0..tree_m) |ni| {
        if (ni == 0) {
            anc_u64[ni] = 1;
            pos_u32[ni] = tree_start;
        } else {
            const p = parents[ni];
            anc_u64[ni] = anc_u64[p] | (@as(u64, 1) << @intCast(ni));
            pos_u32[ni] = pos_u32[p] + 1;
        }
    }
    for (0..tree_m) |ni| {
        const qb_i = ni * q_dim;
        const kb_i = ni * k_dim;
        const vb_i = ni * v_dim;
        for (q_f16[qb_i..][0..q_dim], 0..) |*dst, idx| dst.* = @floatCast(deterministicF32(1100 + ni, idx, 0.3, 0.0));
        for (k_f16[kb_i..][0..k_dim], 0..) |*dst, idx| dst.* = @floatCast(deterministicF32(1200 + ni, idx, 0.3, 0.0));
        for (v_f16[vb_i..][0..v_dim], 0..) |*dst, idx| dst.* = @floatCast(deterministicF32(1300 + ni, idx, 0.3, 0.0));
        cpuRmsNormRopeKv(
            q_f16[qb_i..][0..q_dim],
            k_f16[kb_i..][0..k_dim],
            v_f16[vb_i..][0..v_dim],
            qn_u16,
            kn_u16,
            1,
            num_qh,
            num_kvh,
            k_hd,
            v_hd,
            rotary_dim,
            pos_u32[ni],
            rope_theta,
            v_scale,
            rms_eps,
            cpu_q[(tree_start + ni) * q_dim ..][0..q_dim],
            cpu_k[(tree_start + ni) * k_dim ..][0..k_dim],
            cpu_v[(tree_start + ni) * v_dim ..][0..v_dim],
        );
    }
    {
        var frame = try compute.beginFrame(10_000);
        errdefer frame.abort();
        try pipes.encodeG26QkNormRopeKv(&frame, .{
            .q = q_bind,
            .k = k_bind,
            .v = v_bind,
            .q_norm = qn_bind,
            .k_norm = kn_bind,
            .norm_dtype = .bf16,
            .k_out = k_ext_bind,
            .v_out = v_ext_bind,
            .positions_1d = pos_bind,
            .num_tokens = tree_m,
            .num_q_heads = num_qh,
            .num_kv_heads = num_kvh,
            .k_head_dim = k_hd,
            .v_head_dim = v_hd,
            .rotary_dim = rotary_dim,
            .start_pos = tree_start,
            .chunk_tokens = win_sz,
            .ring_capacity = win_sz,
            .mrope = hybrid.MropeSection.text1d(rotary_dim),
            .rms_eps = rms_eps,
            .rope_theta = rope_theta,
            .v_scale = v_scale,
        }, false);
        try pipes.encodeG26SwaAttn(&frame, .{
            .q = q_bind,
            .kv_tiles = tile_bind,
            .k_extra = k_ext_bind,
            .v_extra = v_ext_bind,
            .ancestor_masks = anc_bind,
            .node_positions = pos_bind,
            .sink_bias = sink_bind,
            .sink_dtype = .bf16,
            .out = out_bind,
            .num_q = tree_m,
            .q_start_pos = tree_start,
            .chunk_start_pos = 0,
            .kv_len = tree_start,
            .num_q_heads = num_qh,
            .num_kv_heads = num_kvh,
            .k_head_dim = k_hd,
            .v_head_dim = v_hd,
            .window_size = win_sz,
            .ring_capacity = win_sz,
            .chunk_tokens = win_sz,
            .is_prefill = false,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 10_000);
    }

    for (0..tree_m) |ni| {
        cpuSwaAttnSinkRef(
            cpu_q[(tree_start + ni) * q_dim ..][0..q_dim],
            cpu_k[0 .. tree_start * k_dim],
            cpu_v[0 .. tree_start * v_dim],
            sink_u16,
            pos_u32[ni],
            tree_start,
            num_qh,
            num_kvh,
            k_hd,
            v_hd,
            win_sz,
            cpu_k[tree_start * k_dim ..][0 .. tree_m * k_dim],
            cpu_v[tree_start * v_dim ..][0 .. tree_m * v_dim],
            anc_u64[ni],
            pos_u32[0..tree_m],
            cpu_out_f32[ni * o_dim ..][0..o_dim],
        );
    }
    for (0..tree_m * o_dim) |i| gpu_out_f32[i] = @floatCast(out_f16[i]);
    const tree_cmp = cosineF64(cpu_out_f32[0 .. tree_m * o_dim], gpu_out_f32[0 .. tree_m * o_dim]);
    report.swa_tree_cos_sim = tree_cmp.cos;
    report.swa_max_abs_err = @max(report.swa_max_abs_err, tree_cmp.max_err);

    // Steady-state M = 1 SWA kernel latency + zero-allocation encode check (isolated from
    // Apple Metal 4 driver async queue-commit completion nodes, matching M9.7).
    anc_u64[0] = 1;
    pos_u32[0] = prefill_toks + 1;
    for (0..@max(cfg.swa_warmup_iters, 4)) |_| {
        try runSwaStep(&compute, pipes, q_bind, k_bind, v_bind, qn_bind, kn_bind, tile_bind, sink_bind, out_bind, anc_bind, pos_bind, prefill_toks + 1, k_ext_bind, v_ext_bind, false);
    }
    var best_swa_delta: i64 = std.math.maxInt(i64);
    for (0..3) |_| {
        const enc_pool = objc_autoreleasePoolPush();
        var frame = try compute.beginFrame(10_000);
        errdefer frame.abort();
        var em0: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &em0);
        try encodeSwaStep(&frame, pipes, q_bind, k_bind, v_bind, qn_bind, kn_bind, tile_bind, sink_bind, out_bind, anc_bind, pos_bind, prefill_toks + 1, k_ext_bind, v_ext_bind, false);
        var em1: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &em1);
        const sig_enc = try frame.commit();
        try compute.wait(sig_enc, 10_000);
        objc_autoreleasePoolPop(enc_pool);
        const edelta: i64 = @as(i64, @intCast(em1.blocks_in_use)) - @as(i64, @intCast(em0.blocks_in_use));
        if (edelta < best_swa_delta) best_swa_delta = edelta;
        if (best_swa_delta <= 0) break;
    }

    var best_swa_us: f64 = 0.0;
    for (0..3) |pass_idx| {
        const pool = objc_autoreleasePoolPush();
        const t0 = try vmstat.nowNs();
        for (0..cfg.swa_bench_iters) |_| {
            const sp = objc_autoreleasePoolPush();
            defer objc_autoreleasePoolPop(sp);
            try runSwaStep(&compute, pipes, q_bind, k_bind, v_bind, qn_bind, kn_bind, tile_bind, sink_bind, out_bind, anc_bind, pos_bind, prefill_toks + 1, k_ext_bind, v_ext_bind, false);
        }
        const t1 = try vmstat.nowNs();
        objc_autoreleasePoolPop(pool);
        const pass_us = @as(f64, @floatFromInt(t1 -| t0)) / (@as(f64, @floatFromInt(cfg.swa_bench_iters)) * 1000.0);
        if (pass_idx == 0 or pass_us < best_swa_us) best_swa_us = pass_us;
    }

    report.swa_decode_mean_us = best_swa_us;
    report.swa_malloc_blocks_delta = best_swa_delta;
    const swa_malloc_ok = report.mtl_validation_active or (report.swa_malloc_blocks_delta <= 0);
    report.swa_parity_ok = (report.swa_prefill_cos_sim >= 0.9999) and
        (report.swa_decode_cos_sim >= 0.9999) and
        (report.swa_tree_cos_sim >= 0.9999) and
        swa_malloc_ok;
}

const SynthMimoBundleSpec = struct {
    num_layers: u32 = 5,
    hidden_size: u32 = 128,
    intermediate_size: u32 = 256,
    moe_intermediate_size: u32 = 128,
    vocab_size: u32 = 64,
    num_experts: u32 = 16,
    cold_experts_per_layer: u32 = 0,
    num_experts_per_tok: u32 = 2,
    use_mxfp4: bool = true,
};

fn buildSynthMimoAmfBytes(allocator: std.mem.Allocator, spec: SynthMimoBundleSpec, seed: u64) !struct {
    amf_bytes: []u8,
    config_json: []u8,
} {
    var blobs: std.ArrayListUnmanaged(graph.TestTensorBlob) = .empty;
    var names: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (blobs.items) |b| allocator.free(b.bytes);
        blobs.deinit(allocator);
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }

    const addBlob = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayListUnmanaged(graph.TestTensorBlob),
            nms: *std.ArrayListUnmanaged([]u8),
            name: []const u8,
            kind: amf.TensorKind,
            dtype: amf.DType,
            layer: u16,
            expert: u16,
            shape: [4]u32,
            ndim: u8,
            bytes: []u8,
        ) !void {
            const dup = try alloc.dupe(u8, name);
            try nms.append(alloc, dup);
            try list.append(alloc, .{
                .name = dup,
                .kind = kind,
                .dtype = dtype,
                .layer = layer,
                .expert = expert,
                .shape = shape,
                .ndim = ndim,
                .bytes = bytes,
            });
        }
    }.call;

    const h: usize = spec.hidden_size;
    const v: usize = spec.vocab_size;
    const inter: usize = spec.intermediate_size;
    const moe_inter: usize = spec.moe_intermediate_size;
    const num_e: usize = spec.num_experts;

    try addBlob(allocator, &blobs, &names, "embed_tokens.weight", .embedding, .f16, amf.NO_LAYER, amf.NO_EXPERT, .{ @intCast(v), @intCast(h), 0, 0 }, 2, try fillDeterministicF16(allocator, seed + 1, v * h, 0.12));
    try addBlob(allocator, &blobs, &names, "norm.weight", .norm, .bf16, amf.NO_LAYER, amf.NO_EXPERT, .{ @intCast(h), 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, seed + 2, h, 0.05, 1.0));
    try addBlob(allocator, &blobs, &names, "lm_head.weight", .lm_head, .f16, amf.NO_LAYER, amf.NO_EXPERT, .{ @intCast(v), @intCast(h), 0, 0 }, 2, try fillDeterministicF16(allocator, seed + 3, v * h, 0.10));

    // Alternating pattern ending with full_attention (like Flash/Pro), 192/128 heads: 2 Q heads, 1 KV head.
    var pattern_buf: [128]u8 = undefined;
    var pat_pos: usize = 0;
    for (0..spec.num_layers) |l| {
        const is_full = (l + 1 == spec.num_layers) or ((l % 4) == 3);
        const entry = if (is_full) "\"full_attention\"" else "\"sliding_attention\"";
        const part = try std.fmt.bufPrint(pattern_buf[pat_pos..], "{s}{s}", .{ if (l == 0) "" else ",", entry });
        pat_pos += part.len;

        var nb: [96]u8 = undefined;
        const sbase = seed + @as(u64, l + 1) * 1000;
        const nq: u32 = 2;
        const nkv: u32 = 1;
        const khd: u32 = 192;
        const vhd: u32 = 128;
        const q_dim: usize = @as(usize, nq) * khd;
        const k_dim: usize = @as(usize, nkv) * khd;
        const v_dim: usize = @as(usize, nkv) * vhd;
        const o_in: usize = @as(usize, nq) * vhd;

        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ @intCast(h), 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sbase + 1, h, 0.04, 1.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ @intCast(h), 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sbase + 2, h, 0.04, 1.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ @intCast(q_dim), @intCast(h), 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 10, q_dim * h, 0.06));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ @intCast(k_dim), @intCast(h), 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 11, k_dim * h, 0.06));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ @intCast(v_dim), @intCast(h), 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 12, v_dim * h, 0.06));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ @intCast(h), @intCast(o_in), 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 13, h * o_in, 0.05));
        if (!is_full) {
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.sink_bias", .{l}) catch unreachable, .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ nq, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sbase + 14, nq, 0.8, 0.2));
        }

        if (l == 0) {
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ @intCast(inter), @intCast(h), 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 20, inter * h, 0.05));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.up_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ @intCast(inter), @intCast(h), 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 21, inter * h, 0.05));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.down_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ @intCast(h), @intCast(inter), 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 22, h * inter, 0.05));
        } else {
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.gate.weight", .{l}) catch unreachable, .router, .f32, @intCast(l), amf.NO_EXPERT, .{ @intCast(num_e), @intCast(h), 0, 0 }, 2, try fillDeterministicF32Bytes(allocator, sbase + 30, num_e * h, 0.15, 0.0));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.gate.e_score_correction_bias", .{l}) catch unreachable, .norm, .f32, @intCast(l), amf.NO_EXPERT, .{ @intCast(num_e), 0, 0, 0 }, 1, try fillDeterministicF32Bytes(allocator, sbase + 31, num_e, 0.25, 0.0));
            const exp_dt: amf.DType = if (spec.use_mxfp4) .gguf_mxfp4 else .f16;
            for (0..num_e) |e| {
                const is_cold = (spec.cold_experts_per_layer > 0 and e >= num_e - spec.cold_experts_per_layer);
                const ekind: amf.TensorKind = if (is_cold) .cold_expert else .hot_expert;
                const es = sbase + 100 + @as(u64, e) * 16;
                const gb = if (spec.use_mxfp4) try fillDeterministicMxfp4(allocator, es + 1, moe_inter, h) else try fillDeterministicF16(allocator, es + 1, moe_inter * h, 0.05);
                const ub = if (spec.use_mxfp4) try fillDeterministicMxfp4(allocator, es + 2, moe_inter, h) else try fillDeterministicF16(allocator, es + 2, moe_inter * h, 0.05);
                const db = if (spec.use_mxfp4) try fillDeterministicMxfp4(allocator, es + 3, h, moe_inter) else try fillDeterministicF16(allocator, es + 3, h * moe_inter, 0.05);
                try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }) catch unreachable, ekind, exp_dt, @intCast(l), @intCast(e), .{ @intCast(moe_inter), @intCast(h), 0, 0 }, 2, gb);
                try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }) catch unreachable, ekind, exp_dt, @intCast(l), @intCast(e), .{ @intCast(moe_inter), @intCast(h), 0, 0 }, 2, ub);
                try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }) catch unreachable, ekind, exp_dt, @intCast(l), @intCast(e), .{ @intCast(h), @intCast(moe_inter), 0, 0 }, 2, db);
            }
        }
    }

    const config_json = try std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "architectures": ["MiMoV2ForCausalLM"],
        \\  "model_type": "mimo_v2_flash",
        \\  "hidden_size": {d},
        \\  "intermediate_size": {d},
        \\  "moe_intermediate_size": {d},
        \\  "num_hidden_layers": {d},
        \\  "num_attention_heads": 2,
        \\  "num_key_value_heads": 1,
        \\  "head_dim": 192,
        \\  "v_head_dim": 128,
        \\  "swa_num_attention_heads": 2,
        \\  "swa_num_key_value_heads": 1,
        \\  "swa_head_dim": 192,
        \\  "swa_v_head_dim": 128,
        \\  "partial_rotary_factor": 0.3333333333333333,
        \\  "sliding_window_size": 128,
        \\  "attention_value_scale": 0.707,
        \\  "add_swa_attention_sink_bias": true,
        \\  "n_routed_experts": {d},
        \\  "num_experts_per_tok": {d},
        \\  "scoring_func": "sigmoid",
        \\  "topk_method": "noaux_tc",
        \\  "routed_scaling_factor": 1.0,
        \\  "n_group": 1,
        \\  "topk_group": 1,
        \\  "vocab_size": {d},
        \\  "rms_norm_eps": 1e-6,
        \\  "rope_theta": 10000.0,
        \\  "hybrid_layer_pattern": [{s}]
        \\}}
    ,
        .{
            spec.hidden_size,
            spec.intermediate_size,
            spec.moe_intermediate_size,
            spec.num_layers,
            spec.num_experts,
            spec.num_experts_per_tok,
            spec.vocab_size,
            pattern_buf[0..pat_pos],
        },
    );
    errdefer allocator.free(config_json);
    try addBlob(allocator, &blobs, &names, "__meta.config_json", .metadata, .u32, amf.NO_LAYER, amf.NO_EXPERT, .{ @intCast(config_json.len), 0, 0, 0 }, 1, try allocator.dupe(u8, config_json));

    const amf_bytes = try graph.buildTestAmf(allocator, "mimo-v2-m91-synth", blobs.items);
    return .{ .amf_bytes = amf_bytes, .config_json = config_json };
}

fn buildSynthQwen35AmfBytes(allocator: std.mem.Allocator, seed: u64, rmsnorm_unit_offset: bool) !struct {
    amf_bytes: []u8,
    config_json: []u8,
} {
    var blobs: std.ArrayListUnmanaged(graph.TestTensorBlob) = .empty;
    var names: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (blobs.items) |b| allocator.free(b.bytes);
        blobs.deinit(allocator);
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }

    const addBlob = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayListUnmanaged(graph.TestTensorBlob),
            nms: *std.ArrayListUnmanaged([]u8),
            name: []const u8,
            kind: amf.TensorKind,
            dtype: amf.DType,
            layer: u16,
            shape: [4]u32,
            ndim: u8,
            bytes: []u8,
        ) !void {
            const dup = try alloc.dupe(u8, name);
            try nms.append(alloc, dup);
            try list.append(alloc, .{
                .name = dup,
                .kind = kind,
                .dtype = dtype,
                .layer = layer,
                .shape = shape,
                .ndim = ndim,
                .bytes = bytes,
            });
        }
    }.call;

    const h: usize = 64;
    const inter: usize = 128;
    const v: usize = 64;
    const n_layers: usize = 4;
    const gdn_kh: usize = 2;
    const gdn_vh: usize = 4;
    const gdn_khd: usize = 32;
    const gdn_vhd: usize = 32;
    const gdn_qkd: usize = gdn_kh * gdn_khd; // 64
    const gdn_vd: usize = gdn_vh * gdn_vhd; // 128
    const conv_d: usize = 2 * gdn_qkd + gdn_vd;
    const conv_k: usize = 4;

    try addBlob(allocator, &blobs, &names, "embed_tokens.weight", .embedding, .f16, amf.NO_LAYER, .{ v, h, 0, 0 }, 2, try fillDeterministicF16(allocator, seed + 1, v * h, 0.14));
    try addBlob(allocator, &blobs, &names, "norm.weight", .norm, .bf16, amf.NO_LAYER, .{ h, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, seed + 2, h, 0.08, 0.1));
    try addBlob(allocator, &blobs, &names, "lm_head.weight", .lm_head, .f16, amf.NO_LAYER, .{ v, h, 0, 0 }, 2, try fillDeterministicF16(allocator, seed + 3, v * h, 0.12));

    for (0..n_layers) |l| {
        var nb: [96]u8 = undefined;
        const sb = seed + @as(u64, l + 1) * 500;
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, .norm, .bf16, @intCast(l), .{ h, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sb + 1, h, 0.08, 0.1));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, .norm, .bf16, @intCast(l), .{ h, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sb + 2, h, 0.08, 0.1));

        if (l < 3) {
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.linear_attn.q_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ gdn_qkd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 10, gdn_qkd * h, 0.07));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.linear_attn.k_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ gdn_qkd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 11, gdn_qkd * h, 0.07));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.linear_attn.v_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ gdn_vd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 12, gdn_vd * h, 0.07));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.linear_attn.g_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ gdn_vd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 13, gdn_vd * h, 0.07));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.linear_attn.o_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ h, gdn_vd, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 14, h * gdn_vd, 0.06));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.linear_attn.a_proj.weight", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ gdn_vh, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 15, gdn_vh * h, 0.07));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.linear_attn.b_proj.weight", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ gdn_vh, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 16, gdn_vh * h, 0.07));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.linear_attn.A_log", .{l}) catch unreachable, .norm, .f32, @intCast(l), .{ gdn_vh, 0, 0, 0 }, 1, try fillDeterministicF32Bytes(allocator, sb + 17, gdn_vh, 0.4, -1.0));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.linear_attn.dt_bias", .{l}) catch unreachable, .norm, .f32, @intCast(l), .{ gdn_vh, 0, 0, 0 }, 1, try fillDeterministicF32Bytes(allocator, sb + 18, gdn_vh, 0.2, 0.0));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.linear_attn.conv1d.weight", .{l}) catch unreachable, .norm, .f32, @intCast(l), .{ conv_d, conv_k, 0, 0 }, 2, try fillDeterministicF32Bytes(allocator, sb + 19, conv_d * conv_k, 0.15, 0.0));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.linear_attn.norm.weight", .{l}) catch unreachable, .norm, .bf16, @intCast(l), .{ gdn_vhd, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sb + 20, gdn_vhd, 0.08, 1.0));
        } else {
            const hd: usize = 64;
            const nq: usize = 2;
            const nkv: usize = 1;
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ nq * hd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 30, nq * hd * h, 0.07));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.q_gate_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ nq * hd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 31, nq * hd * h, 0.07));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ nkv * hd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 32, nkv * hd * h, 0.07));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ nkv * hd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 33, nkv * hd * h, 0.07));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ h, nq * hd, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 34, h * nq * hd, 0.06));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable, .norm, .bf16, @intCast(l), .{ hd, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sb + 35, hd, 0.08, 0.1));
            try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable, .norm, .bf16, @intCast(l), .{ hd, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sb + 36, hd, 0.08, 0.1));
        }

        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ inter, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 40, inter * h, 0.06));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.up_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ inter, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 41, inter * h, 0.06));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.down_proj", .{l}) catch unreachable, .dense, .f16, @intCast(l), .{ h, inter, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 42, h * inter, 0.06));
    }

    const config_json = try std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "architectures": ["Qwen3_5ForConditionalGeneration"],
        \\  "model_type": "qwen3_5_text",
        \\  "hidden_size": {d},
        \\  "intermediate_size": {d},
        \\  "num_hidden_layers": {d},
        \\  "num_attention_heads": 2,
        \\  "num_key_value_heads": 1,
        \\  "head_dim": 64,
        \\  "partial_rotary_factor": 0.25,
        \\  "mrope_section": [4, 2, 2],
        \\  "attn_output_gate": true,
        \\  "rmsnorm_unit_offset": {s},
        \\  "linear_num_key_heads": {d},
        \\  "linear_num_value_heads": {d},
        \\  "linear_key_head_dim": {d},
        \\  "linear_value_head_dim": {d},
        \\  "linear_conv_kernel_dim": {d},
        \\  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        \\  "vocab_size": {d},
        \\  "rms_norm_eps": 1e-6,
        \\  "rope_theta": 10000000.0
        \\}}
    ,
        .{
            h,
            inter,
            n_layers,
            if (rmsnorm_unit_offset) "true" else "false",
            gdn_kh,
            gdn_vh,
            gdn_khd,
            gdn_vhd,
            conv_k,
            v,
        },
    );
    errdefer allocator.free(config_json);
    try addBlob(allocator, &blobs, &names, "__meta.config_json", .metadata, .u32, amf.NO_LAYER, .{ @intCast(config_json.len), 0, 0, 0 }, 1, try allocator.dupe(u8, config_json));

    const amf_bytes = try graph.buildTestAmf(allocator, "qwen35-m91-synth", blobs.items);
    return .{ .amf_bytes = amf_bytes, .config_json = config_json };
}

fn buildSynthDFlash5LAmfBytes(
    allocator: std.mem.Allocator,
    block_size: u8,
    taps: u8,
    h: u32,
    inter: u32,
    vocab: u32,
    mask_token_id: u32,
    seed: u64,
) ![]u8 {
    var blobs: std.ArrayListUnmanaged(graph.TestTensorBlob) = .empty;
    var names: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (blobs.items) |b| allocator.free(b.bytes);
        blobs.deinit(allocator);
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }

    const addBlob = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayListUnmanaged(graph.TestTensorBlob),
            nms: *std.ArrayListUnmanaged([]u8),
            name: []const u8,
            kind: amf.TensorKind,
            dtype: amf.DType,
            layer: u16,
            shape: [4]u32,
            ndim: u8,
            bytes: []u8,
        ) !void {
            const dup = try alloc.dupe(u8, name);
            try nms.append(alloc, dup);
            try list.append(alloc, .{
                .name = dup,
                .kind = kind,
                .dtype = dtype,
                .layer = layer,
                .shape = shape,
                .ndim = ndim,
                .bytes = bytes,
            });
        }
    }.call;

    const num_layers: u32 = 5;
    const nq: u32 = 2;
    const nkv: u32 = 1;
    const hd: u32 = 128;
    const q_dim: usize = @as(usize, nq) * hd;
    const kv_dim: usize = @as(usize, nkv) * hd;
    const hs: usize = h;
    const inter_u: usize = inter;

    try addBlob(allocator, &blobs, &names, "fc.weight", .dense, .bf16, amf.NO_LAYER, .{ h, @as(u32, taps) * h, 0, 0 }, 2, try fillDeterministicBf16(allocator, seed + 1, hs * taps * hs, 0.08, 0.0));
    try addBlob(allocator, &blobs, &names, "hidden_norm.weight", .norm, .bf16, amf.NO_LAYER, .{ h, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, seed + 2, hs, 0.04, 1.0));
    try addBlob(allocator, &blobs, &names, "norm.weight", .norm, .bf16, amf.NO_LAYER, .{ h, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, seed + 3, hs, 0.04, 1.0));

    for (0..num_layers) |l| {
        var nb: [96]u8 = undefined;
        const sb = seed + @as(u64, l + 1) * 700;
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, .norm, .bf16, @intCast(l), .{ h, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sb + 1, hs, 0.04, 1.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, .norm, .bf16, @intCast(l), .{ h, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sb + 2, hs, 0.04, 1.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable, .norm, .bf16, @intCast(l), .{ hd, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sb + 3, hd, 0.04, 1.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable, .norm, .bf16, @intCast(l), .{ hd, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sb + 4, hd, 0.04, 1.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.sink_bias", .{l}) catch unreachable, .norm, .bf16, @intCast(l), .{ nq, 0, 0, 0 }, 1, try fillDeterministicBf16(allocator, sb + 5, nq, 0.7, 0.2));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.q_proj.weight", .{l}) catch unreachable, .dense, .bf16, @intCast(l), .{ @intCast(q_dim), h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sb + 10, q_dim * hs, 0.07, 0.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.k_proj.weight", .{l}) catch unreachable, .dense, .bf16, @intCast(l), .{ @intCast(kv_dim), h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sb + 11, kv_dim * hs, 0.07, 0.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.v_proj.weight", .{l}) catch unreachable, .dense, .bf16, @intCast(l), .{ @intCast(kv_dim), h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sb + 12, kv_dim * hs, 0.07, 0.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.o_proj.weight", .{l}) catch unreachable, .dense, .bf16, @intCast(l), .{ h, @intCast(q_dim), 0, 0 }, 2, try fillDeterministicBf16(allocator, sb + 13, hs * q_dim, 0.06, 0.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.gate_proj.weight", .{l}) catch unreachable, .dense, .bf16, @intCast(l), .{ inter, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sb + 20, inter_u * hs, 0.06, 0.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.up_proj.weight", .{l}) catch unreachable, .dense, .bf16, @intCast(l), .{ inter, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sb + 21, inter_u * hs, 0.06, 0.0));
        try addBlob(allocator, &blobs, &names, std.fmt.bufPrint(&nb, "layers.{d}.mlp.down_proj.weight", .{l}) catch unreachable, .dense, .bf16, @intCast(l), .{ h, inter, 0, 0 }, 2, try fillDeterministicBf16(allocator, sb + 22, hs * inter_u, 0.06, 0.0));
    }

    const dflash_json = try std.fmt.allocPrint(
        allocator,
        \\{{
        \\  "architectures": ["DFlashDraftModel"],
        \\  "hidden_size": {d},
        \\  "intermediate_size": {d},
        \\  "num_hidden_layers": {d},
        \\  "num_attention_heads": {d},
        \\  "num_key_value_heads": {d},
        \\  "head_dim": {d},
        \\  "partial_rotary_factor": 0.5,
        \\  "rope_theta": 10000.0,
        \\  "sliding_window": 1024,
        \\  "attention_value_scale": 0.612,
        \\  "vocab_size": {d},
        \\  "rms_norm_eps": 1e-6,
        \\  "block_size": {d},
        \\  "dflash_config": {{
        \\    "mask_token_id": {d},
        \\    "target_layer_ids": [0, 1, 2, 3, 4]
        \\  }}
        \\}}
    ,
        .{ h, inter, num_layers, nq, nkv, hd, vocab, block_size, mask_token_id },
    );
    defer allocator.free(dflash_json);
    try addBlob(allocator, &blobs, &names, "__meta.config_json", .metadata, .u32, amf.NO_LAYER, .{ @intCast(dflash_json.len), 0, 0, 0 }, 1, try allocator.dupe(u8, dflash_json));

    return graph.buildTestAmf(allocator, "dflash-m91-synth", blobs.items);
}

/// Isolate steady-state CPU + Metal 4 command encoding malloc delta across up
/// to 3 `stepDecode` passes, excluding Metal 4 `[queue commit:]` driver nodes.
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
            const self: *@This() = @ptrCast(@alignCast(ctx orelse return));
            malloc_zone_statistics(null, &self.m0);
            self.captured_begin = true;
        }
        fn onPreCommit(ctx: ?*anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx orelse return));
            malloc_zone_statistics(null, &self.m1);
            self.captured = true;
        }
    };
    var best_encode_delta: i64 = std.math.maxInt(i64);
    for (0..3) |_| {
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
        if (best_encode_delta <= 0) break;
    }
    return best_encode_delta;
}

fn benchGreedyDecode(
    model: *graph.GraphModel,
    prompt: []const u32,
    n_tokens: u32,
    out_tokens: []u32,
    measure_malloc: bool,
) !struct { tok_s: f64, malloc_delta: i64, commit_malloc_delta: i64 } {
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

    var best_tok_s: f64 = 0.0;
    var best_commit_delta: i64 = 0;
    const passes: usize = if (measure_malloc) 3 else 1;
    const step_iters: u32 = @max(n_tokens, 6);
    for (0..passes) |pass_idx| {
        model.resetKvCache();
        cur = try model.prefillChunk(prompt, 0, true);
        pos = @intCast(prompt.len);
        // Warm up steady-state Metal 4 frames (frames_in_flight = 2) and PSO caches.
        for (0..4) |_| {
            cur = try model.stepDecode(cur, pos);
            pos += 1;
        }
        const pool = objc_autoreleasePoolPush();
        var m0: MallocStatistics = std.mem.zeroes(MallocStatistics);
        if (measure_malloc) malloc_zone_statistics(null, &m0);
        const t0 = try vmstat.nowNs();
        for (0..step_iters) |_| {
            const sp = objc_autoreleasePoolPush();
            defer objc_autoreleasePoolPop(sp);
            cur = try model.stepDecode(cur, pos);
            pos += 1;
        }
        const t1 = try vmstat.nowNs();
        objc_autoreleasePoolPop(pool);
        var m1: MallocStatistics = std.mem.zeroes(MallocStatistics);
        if (measure_malloc) malloc_zone_statistics(null, &m1);

        const elapsed_ns = @max(@as(u64, 1), t1 -| t0);
        const tok_s = @as(f64, @floatFromInt(step_iters)) * 1e9 / @as(f64, @floatFromInt(elapsed_ns));
        const cdelta: i64 = if (measure_malloc) @as(i64, m1.blocks_in_use) - @as(i64, m0.blocks_in_use) else 0;
        if (pass_idx == 0) {
            best_tok_s = tok_s;
            best_commit_delta = cdelta;
        } else {
            best_tok_s = @max(best_tok_s, tok_s);
            best_commit_delta = @min(best_commit_delta, cdelta);
        }
    }

    const best_encode_delta: i64 = if (measure_malloc)
        try measureStepEncodeMallocDelta(model, &cur, &pos)
    else
        0;

    return .{
        .tok_s = best_tok_s,
        .malloc_delta = best_encode_delta,
        .commit_malloc_delta = best_commit_delta,
    };
}

fn runMimoV2ParityBench(allocator: std.mem.Allocator, tmp_dir: []const u8, cfg: M91Config, report: *M91Report) !void {
    const hot_spec: SynthMimoBundleSpec = .{ .cold_experts_per_layer = 0 };
    const hot_bundle = try buildSynthMimoAmfBytes(allocator, hot_spec, 4100);
    defer allocator.free(hot_bundle.amf_bytes);
    defer allocator.free(hot_bundle.config_json);

    const cold_spec: SynthMimoBundleSpec = .{ .cold_experts_per_layer = 8 };
    const cold_bundle = try buildSynthMimoAmfBytes(allocator, cold_spec, 4100);
    defer allocator.free(cold_bundle.amf_bytes);
    defer allocator.free(cold_bundle.config_json);

    const cold_path = try std.fmt.allocPrintSentinel(allocator, "{s}/mimo_v2_cold.amf", .{tmp_dir}, 0);
    defer {
        graph.removeTestFile(cold_path);
        allocator.free(cold_path);
    }
    try writeFileBytesPosix(cold_path, cold_bundle.amf_bytes);

    var hot_gcfg = try graph.GraphConfig.fromJson(allocator, hot_bundle.config_json);
    defer hot_gcfg.deinit(allocator);
    var cold_gcfg = try graph.GraphConfig.fromJson(allocator, cold_bundle.config_json);
    defer cold_gcfg.deinit(allocator);

    var hot_model = try graph.GraphModel.loadFromBytes(allocator, hot_bundle.amf_bytes, hot_gcfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .startup_canary = false,
    });
    defer hot_model.deinit(allocator);

    var cold_model = try graph.GraphModel.loadFromFile(allocator, cold_path, cold_gcfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .cold_slot_count = 16,
        .startup_canary = false,
    });
    defer cold_model.deinit(allocator);

    const prompt = [_]u32{ 1, 7, 13, 19, 25 };
    const n_toks = @max(cfg.e2e_tokens, 4);
    const hot_toks = try allocator.alloc(u32, n_toks);
    defer allocator.free(hot_toks);
    const cold_toks = try allocator.alloc(u32, n_toks);
    defer allocator.free(cold_toks);

    const hot_res = try benchGreedyDecode(&hot_model, &prompt, n_toks, hot_toks, true);
    const cold_res = try benchGreedyDecode(&cold_model, &prompt, n_toks, cold_toks, false);

    const cs = cold_model.coldExpertStats() orelse return error.MissingColdExpertRuntime;
    report.mimo_hot_tok_s = hot_res.tok_s;
    report.mimo_cold_tok_s = cold_res.tok_s;
    report.mimo_cold_fetches = cs.cache_telemetry.sync_misses + cs.cache_telemetry.prefetch_hits + cs.cache_telemetry.cache_hits;
    report.mimo_malloc_blocks_delta = hot_res.malloc_delta;
    report.mimo_commit_malloc_blocks_delta = hot_res.commit_malloc_delta;

    const exact_match = std.mem.eql(u32, hot_toks, cold_toks);
    const malloc_ok = report.mtl_validation_active or (hot_res.malloc_delta <= 0);
    report.mimo_parity_ok = exact_match and (report.mimo_cold_fetches > 0) and (cs.cache_telemetry.io_errors == 0) and malloc_ok;
}

fn runQwen35ParityBench(allocator: std.mem.Allocator, tmp_dir: []const u8, cfg: M91Config, report: *M91Report) !void {
    const uo_bundle = try buildSynthQwen35AmfBytes(allocator, 6100, true);
    defer allocator.free(uo_bundle.amf_bytes);
    defer allocator.free(uo_bundle.config_json);

    const std_bundle = try buildSynthQwen35AmfBytes(allocator, 6100, false);
    defer allocator.free(std_bundle.amf_bytes);
    defer allocator.free(std_bundle.config_json);

    const vision_bytes = try graph.buildSynthVisionTowerAmfBytes(allocator, .{
        .num_layers = 2,
        .vis_hidden = 64,
        .vis_intermediate = 128,
        .lm_hidden = 64,
        .merger_hidden = 128,
        .patch_size = 16,
        .t_patch = 2,
        .spatial_merge = 2,
        .q4_qkv = true,
        .swiglu = false,
    });
    defer allocator.free(vision_bytes);

    const vis_path = try std.fmt.allocPrintSentinel(allocator, "{s}/qwen35_vision.amf", .{tmp_dir}, 0);
    defer {
        graph.removeTestFile(vis_path);
        allocator.free(vis_path);
    }
    try writeFileBytesPosix(vis_path, vision_bytes);

    var uo_cfg = try graph.GraphConfig.fromJson(allocator, uo_bundle.config_json);
    defer uo_cfg.deinit(allocator);
    var std_cfg = try graph.GraphConfig.fromJson(allocator, std_bundle.config_json);
    defer std_cfg.deinit(allocator);
    std_cfg.rmsnorm_unit_offset = false;

    var uo_model = try graph.GraphModel.loadFromBytes(allocator, uo_bundle.amf_bytes, uo_cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .vision_amf_path = vis_path,
        .startup_canary = false,
    });
    defer uo_model.deinit(allocator);

    var std_model = try graph.GraphModel.loadFromBytes(allocator, std_bundle.amf_bytes, std_cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 8,
        .startup_canary = false,
    });
    defer std_model.deinit(allocator);

    // Control check: verify `(1+w)` RMSNorm (`rmsnorm_unit_offset = true`) produces distinct logits from `w`.
    const prompt = [_]u32{ 2, 5, 11, 17, 23 };
    _ = try uo_model.prefillChunk(&prompt, 0, true);
    _ = try std_model.prefillChunk(&prompt, 0, true);
    var uo_diff: f64 = 0.0;
    for (uo_model.readLastLogitsF16(), std_model.readLastLogitsF16()) |a, b| {
        const af: f64 = @floatCast(a);
        const bf: f64 = @floatCast(b);
        uo_diff = @max(uo_diff, @abs(af - bf));
    }
    report.qwen35_unit_offset_logit_diff = uo_diff;

    // Text decode speed & zero-malloc gate
    const n_toks = @max(cfg.e2e_tokens, 4);
    const out_toks = try allocator.alloc(u32, n_toks);
    defer allocator.free(out_toks);
    const dec_res = try benchGreedyDecode(&uo_model, &prompt, n_toks, out_toks, true);
    report.qwen35_tok_s = dec_res.tok_s;
    report.qwen35_malloc_blocks_delta = dec_res.malloc_delta;
    report.qwen35_commit_malloc_blocks_delta = dec_res.commit_malloc_delta;

    // Vision sidecar multimodal verification (`SIGLIP_DEFAULT` mean=0.5, std=0.5, BGRA8 zero-copy IOSurface)
    var red_frame = try metal.VisionFrame.init(uo_model.device, .bgra8, 64, 64);
    defer red_frame.close();
    try fillDeterministicVisionFrame(&red_frame, 240, 20, 20);
    var blue_frame = try metal.VisionFrame.init(uo_model.device, .bgra8, 64, 64);
    defer blue_frame.close();
    try fillDeterministicVisionFrame(&blue_frame, 20, 20, 240);

    const mm_prefix = [_]u32{ 1, 2 };
    const mm_suffix = [_]u32{ 3, 4 };
    const mm_opts: graph.GraphModel.MultimodalPrefillOptions = .{
        .t_patch = 2,
        .norm = null,
    };
    uo_model.resetKvCache();
    const red_res = try uo_model.prefillMultimodal(allocator, &mm_prefix, &mm_suffix, &[_]metal.VisionFrame{red_frame}, mm_opts);
    const red_logits = try allocator.dupe(f16, uo_model.readLastLogitsF16());
    defer allocator.free(red_logits);

    uo_model.resetKvCache();
    const blue_res = try uo_model.prefillMultimodal(allocator, &mm_prefix, &mm_suffix, &[_]metal.VisionFrame{blue_frame}, mm_opts);
    var rb_diff: f64 = 0.0;
    for (red_logits, uo_model.readLastLogitsF16()) |rl, bl| {
        const rf: f64 = @floatCast(rl);
        const bf: f64 = @floatCast(bl);
        rb_diff = @max(rb_diff, @abs(rf - bf));
    }
    report.qwen35_vision_tokens = red_res.vision_tokens;
    report.qwen35_red_blue_logit_diff = rb_diff;

    const malloc_ok = report.mtl_validation_active or (dec_res.malloc_delta <= 0);
    report.qwen35_parity_ok = uo_model.hasVisionTower() and
        (red_res.vision_tokens > 0) and
        (blue_res.vision_tokens == red_res.vision_tokens) and
        (rb_diff > 1e-2) and
        (uo_diff > 1e-2) and
        malloc_ok;
}

fn runD15MimoSpecAndD7Bench(allocator: std.mem.Allocator, tmp_dir: []const u8, cfg: M91Config, report: *M91Report) !void {
    const mimo_spec: SynthMimoBundleSpec = .{
        .num_layers = 5,
        .hidden_size = 128,
        .intermediate_size = 256,
        .moe_intermediate_size = 128,
        .vocab_size = 64,
        .num_experts = 16,
        .cold_experts_per_layer = 8,
        .num_experts_per_tok = 2,
        .use_mxfp4 = true,
    };
    const bundle = try buildSynthMimoAmfBytes(allocator, mimo_spec, 8200);
    defer allocator.free(bundle.amf_bytes);
    defer allocator.free(bundle.config_json);

    const tgt_path = try std.fmt.allocPrintSentinel(allocator, "{s}/mimo_v2_d15_tgt.amf", .{tmp_dir}, 0);
    defer {
        graph.removeTestFile(tgt_path);
        allocator.free(tgt_path);
    }
    try writeFileBytesPosix(tgt_path, bundle.amf_bytes);

    var gcfg = try graph.GraphConfig.fromJson(allocator, bundle.config_json);
    defer gcfg.deinit(allocator);
    const tap_layers = [_]u16{ 0, 1, 2, 3, 4 };
    var model = try graph.GraphModel.loadFromFile(allocator, tgt_path, gcfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .cold_slot_count = 16,
        .enable_feature_taps = true,
        .aux_layer_ids = .{ 1, 2, 4 },
        .enable_dflare_taps = true,
        .dflare_target_layers = &tap_layers,
        .startup_canary = false,
    });
    defer model.deinit(allocator);

    const prompt = [_]u32{ 3, 9, 15, 21, 27, 33 };
    const n_toks = @max(cfg.spec_tokens, 8);
    const base_toks = try allocator.alloc(u32, n_toks);
    defer allocator.free(base_toks);
    _ = try benchGreedyDecode(&model, &prompt, n_toks, base_toks, false);

    // Aggressive D7 cost model reflecting cold NVMe expert fetch penalty so cold branches get pruned cleanly.
    const d7_cost: tree_mod.PruneCostModel = tree_mod.PruneCostModel.uniformBeta(
        60.0,
        2.0,
        150_000.0,
        6_000_000.0,
        40.0,
        4,
        16,
    );

    // 1) Chained 3L native MTP over `mimo_v2` `GraphModel`
    {
        const mtp_cfg: draft.DraftConfig = .{
            .format = .mtp,
            .vocab_size = 64,
            .draft_vocab_size = 64,
            .hidden_size = 128,
            .target_hidden_size = 128,
            .num_hidden_layers = 3,
            .num_attention_heads = 2,
            .num_key_value_heads = 1,
            .head_dim = 192,
            .v_head_dim = 128,
            .rope_dim = 64,
            .sliding_window = 128,
            .attention_value_scale = 0.707,
            .add_swa_attention_sink_bias = true,
            .mtp_chained = true,
            .intermediate_size = 256,
            .rms_norm_eps = 1e-6,
            .rope_theta = 10_000.0,
            .share_embed_tokens = true,
            .share_lm_head = true,
        };
        const mtp_bytes = try draft.buildSyntheticDraftAmfImage(allocator, mtp_cfg, 8300);
        defer allocator.free(mtp_bytes);

        var dh = try draft.DraftHead.loadFromBytesForGraph(allocator, &model, mtp_bytes, mtp_cfg, .{
            .max_seq_len = 256,
            .max_tree_slots = 16,
            .placement = .gpu,
        });
        defer dh.deinitForGraph(allocator, &model);

        var se = try engine_mod.SpecEngine.initWithGraph(allocator, &model, &dh, .{
            .mode = .mtp,
            .max_tree_nodes = 16,
            .max_draft_depth = 3,
            .branch_width = 2,
            .top_k_per_node = 3,
            .l1_max_draft = 4,
            .temperature = 0.0,
            .enable_pruning = true,
            .cost_model = d7_cost,
        });
        defer se.deinit(allocator);

        const spec_toks = try allocator.alloc(u32, n_toks);
        defer allocator.free(spec_toks);
        model.resetKvCache();
        spec_toks[0] = try se.prefill(&prompt, null);
        var produced: usize = 1;
        const t0 = try vmstat.nowNs();
        while (produced < n_toks) {
            const rs = try se.stepRound(null, spec_toks[produced..n_toks]);
            if (rs.emitted_count == 0) break;
            produced += rs.emitted_count;
        }
        const t1 = try vmstat.nowNs();
        const elapsed_ns = @max(@as(u64, 1), t1 -| t0);
        report.mtp3_tok_s = @as(f64, @floatFromInt(produced -| 1)) * 1e9 / @as(f64, @floatFromInt(elapsed_ns));
        report.mtp3_mean_accept_len = se.stats.meanAcceptanceLength();
        report.mtp3_d7_prune_reduction = if (se.stats.drafted_nodes > 0)
            @as(f64, @floatFromInt(se.stats.drafted_nodes -| se.stats.verified_nodes)) /
                @as(f64, @floatFromInt(se.stats.drafted_nodes))
        else
            0.0;
        report.mtp3_parity_ok = (produced == n_toks) and std.mem.eql(u32, base_toks, spec_toks);
    }

    // 2) 5L DFlash direct block-diffusion (both B=8 and B=16) over `mimo_v2` `GraphModel`
    inline for (.{ @as(u8, 8), @as(u8, 16) }, 0..) |blk_sz, bi| {
        const df_bytes = try buildSynthDFlash5LAmfBytes(allocator, blk_sz, 5, 128, 256, 64, 31, 8400 + @as(u64, blk_sz));
        defer allocator.free(df_bytes);

        const df_path = try std.fmt.allocPrintSentinel(allocator, "{s}/mimo_dflash_b{d}.amf", .{ tmp_dir, blk_sz }, 0);
        defer {
            graph.removeTestFile(df_path);
            allocator.free(df_path);
        }
        try writeFileBytesPosix(df_path, df_bytes);

        var loaded = try engine_mod.LoadedDFlareProposer.loadFromFile(allocator, &model, df_path, null, .{
            .variant = .dflash_standalone,
            .num_layers = 5,
            .block_size = blk_sz,
            .top_k_per_pos = 2,
            .max_tree_nodes = 16,
            .hidden_size = 128,
            .intermediate_size = 256,
            .num_attention_heads = 2,
            .num_key_value_heads = 1,
            .head_dim = 128,
            .rope_dim = 64,
            .rope_theta = 10_000.0,
            .attention_value_scale = 0.612,
            .sliding_window = 1024,
            .vocab_size = 64,
            .markov_rank = 0,
            .num_target_taps = 5,
            .target_layer_ids = .{ 0, 1, 2, 3, 4, 0, 0, 0 },
            .mask_token_id = 31,
            .refinement_iters = 1,
            .fork_top2_threshold = 0.99,
            .rms_eps = 1e-6,
        }, null);
        defer loaded.deinit(allocator);

        var se = try engine_mod.SpecEngine.initWithDFlare(allocator, &model, loaded.proposer(), .{
            .mode = .dflare,
            .max_tree_nodes = 16,
            .max_draft_depth = blk_sz - 1,
            .l1_max_draft = 0,
            .temperature = 0.0,
            .enable_pruning = true,
            .cost_model = d7_cost,
        });
        defer se.deinit(allocator);

        const spec_toks = try allocator.alloc(u32, n_toks);
        defer allocator.free(spec_toks);
        model.resetKvCache();
        spec_toks[0] = try se.prefill(&prompt, null);
        var produced: usize = 1;
        const t0 = try vmstat.nowNs();
        while (produced < n_toks) {
            const rs = try se.stepRound(null, spec_toks[produced..n_toks]);
            if (rs.emitted_count == 0) break;
            produced += rs.emitted_count;
        }
        const t1 = try vmstat.nowNs();
        const elapsed_ns = @max(@as(u64, 1), t1 -| t0);
        const tok_s = @as(f64, @floatFromInt(produced -| 1)) * 1e9 / @as(f64, @floatFromInt(elapsed_ns));
        const accept_len = se.stats.meanAcceptanceLength();
        const prune_red = if (se.stats.drafted_nodes > 0)
            @as(f64, @floatFromInt(se.stats.drafted_nodes -| se.stats.verified_nodes)) /
                @as(f64, @floatFromInt(se.stats.drafted_nodes))
        else
            0.0;
        const parity_ok = (produced == n_toks) and
            std.mem.eql(u32, base_toks, spec_toks) and
            (se.stats.dflare_rounds > 0);

        if (bi == 0) {
            report.dflash_b8_tok_s = tok_s;
            report.dflash_b8_mean_accept_len = accept_len;
            report.dflash_b8_d7_prune_reduction = prune_red;
            report.dflash_b8_rounds = se.stats.dflare_rounds;
            report.dflash_b8_parity_ok = parity_ok;
        } else {
            report.dflash_b16_tok_s = tok_s;
            report.dflash_b16_mean_accept_len = accept_len;
            report.dflash_b16_d7_prune_reduction = prune_red;
            report.dflash_b16_rounds = se.stats.dflare_rounds;
            report.dflash_b16_parity_ok = parity_ok;
        }
    }
}

fn verifyGraphLoraDoraCycle(
    allocator: std.mem.Allocator,
    model: *graph.GraphModel,
    budget: budget_mod.Budget,
    num_layers: u32,
    h: u32,
    q_n: u32,
    gate_n: u32,
    prompt: []const u32,
    iters: u32,
    mtl_validation: bool,
) !struct { diff: f64, unload_err: f64, malloc_delta: i64, ok: bool } {
    const rank: u32 = 8;
    const a_q = try allocator.alloc(f32, @as(usize, rank) * h);
    defer allocator.free(a_q);
    const b_q = try allocator.alloc(f32, @as(usize, q_n) * rank);
    defer allocator.free(b_q);
    const mag_q = try allocator.alloc(f32, q_n);
    defer allocator.free(mag_q);
    const a_g = try allocator.alloc(f32, @as(usize, rank) * h);
    defer allocator.free(a_g);
    const b_g = try allocator.alloc(f32, @as(usize, gate_n) * rank);
    defer allocator.free(b_g);

    for (a_q, 0..) |*dst, idx| dst.* = deterministicF32(9101, idx, 0.25, 0.0);
    for (b_q, 0..) |*dst, idx| dst.* = deterministicF32(9102, idx, 0.25, 0.0);
    for (mag_q, 0..) |*dst, idx| dst.* = deterministicF32(9103, idx, 0.25, 1.25);
    for (a_g, 0..) |*dst, idx| dst.* = deterministicF32(9104, idx, 0.25, 0.0);
    for (b_g, 0..) |*dst, idx| dst.* = deterministicF32(9105, idx, 0.25, 0.0);

    const amlora_bytes = try lora.buildSyntheticAmloraBytes(allocator, "m91-dora", rank, 16.0, &.{
        .{
            .layer = 0,
            .target = .q_proj,
            .n = q_n,
            .k = h,
            .rank = rank,
            .dtype = .f16,
            .a_f32 = a_q,
            .b_f32 = b_q,
            .dora_mag_f32 = mag_q,
        },
        .{
            .layer = 0,
            .target = .gate_proj,
            .n = gate_n,
            .k = h,
            .rank = rank,
            .dtype = .f16,
            .a_f32 = a_g,
            .b_f32 = b_g,
        },
    });
    defer allocator.free(amlora_bytes);

    var reg = try lora.LoraRegistry.init(.{
        .pool_bytes = 2 * 1024 * 1024,
        .wired_budget = budget,
        .base_shape = .{
            .num_layers = num_layers,
            .hidden_size = h,
            .q_dim = q_n,
            .kv_dim = q_n / 2,
            .intermediate_size = gate_n,
            .flexible_dims = true,
        },
    });
    defer reg.deinit(allocator);

    try model.attachLoraRegistry(&reg);
    defer model.attachLoraRegistry(null) catch unreachable;

    // Base logits before mounting adapter
    model.setActiveLoraSlot(lora.NO_ADAPTER);
    model.resetKvCache();
    const base_t0 = try model.prefillChunk(prompt, 0, true);
    const base_logits = try allocator.dupe(f16, model.readLastLogitsF16());
    defer allocator.free(base_logits);

    // Hot-load + DoRA calibrate
    const slot = try reg.loadAdapterFromBytes(allocator, "m91-dora", amlora_bytes);
    try model.calibrateLoraDora(allocator);

    model.setActiveLoraSlot(slot);
    model.resetKvCache();
    _ = try model.prefillChunk(prompt, 0, true);
    var active_diff: f64 = 0.0;
    for (base_logits, model.readLastLogitsF16()) |bl, al| {
        const bf: f64 = @floatCast(bl);
        const af: f64 = @floatCast(al);
        active_diff = @max(active_diff, @abs(af - bf));
    }

    // Warm up active LoRA M=1 decode, then measure encode malloc delta
    var cur = try model.stepDecode(base_t0, @intCast(prompt.len));
    var pos: u32 = @intCast(prompt.len + 1);
    for (0..iters) |_| {
        const sp = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(sp);
        cur = try model.stepDecode(cur, pos);
        pos += 1;
    }
    const malloc_delta = try measureStepEncodeMallocDelta(model, &cur, &pos);

    // Unload adapter and verify bit-exact restore to base logits
    model.setActiveLoraSlot(lora.NO_ADAPTER);
    try reg.unloadAdapter(allocator, "m91-dora");
    model.resetKvCache();
    const restored_t0 = try model.prefillChunk(prompt, 0, true);
    var unload_err: f64 = 0.0;
    for (base_logits, model.readLastLogitsF16()) |bl, rl| {
        const bf: f64 = @floatCast(bl);
        const rf: f64 = @floatCast(rl);
        unload_err = @max(unload_err, @abs(rf - bf));
    }

    const malloc_ok = mtl_validation or (malloc_delta <= 0);
    return .{
        .diff = active_diff,
        .unload_err = unload_err,
        .malloc_delta = malloc_delta,
        .ok = (active_diff > 1e-3) and (unload_err == 0.0) and (base_t0 == restored_t0) and malloc_ok,
    };
}

fn runW30LoraDoraCheck(allocator: std.mem.Allocator, budget: budget_mod.Budget, cfg: M91Config, report: *M91Report) !void {
    const prompt = [_]u32{ 2, 6, 10, 14 };
    const iters = @max(cfg.lora_iters, 4);

    const mimo_b = try buildSynthMimoAmfBytes(allocator, .{ .num_layers = 4 }, 9200);
    defer allocator.free(mimo_b.amf_bytes);
    defer allocator.free(mimo_b.config_json);
    var mimo_cfg = try graph.GraphConfig.fromJson(allocator, mimo_b.config_json);
    defer mimo_cfg.deinit(allocator);
    var mimo_model = try graph.GraphModel.loadFromBytes(allocator, mimo_b.amf_bytes, mimo_cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 8,
        .startup_canary = false,
    });
    defer mimo_model.deinit(allocator);

    const mimo_res = try verifyGraphLoraDoraCycle(
        allocator,
        &mimo_model,
        budget,
        4,
        128,
        2 * 192,
        256,
        &prompt,
        iters,
        report.mtl_validation_active,
    );
    report.mimo_lora_logit_diff = mimo_res.diff;
    report.mimo_unload_max_err = mimo_res.unload_err;
    report.mimo_lora_malloc_delta = mimo_res.malloc_delta;

    const q35_b = try buildSynthQwen35AmfBytes(allocator, 9300, true);
    defer allocator.free(q35_b.amf_bytes);
    defer allocator.free(q35_b.config_json);
    var q35_cfg = try graph.GraphConfig.fromJson(allocator, q35_b.config_json);
    defer q35_cfg.deinit(allocator);
    var q35_model = try graph.GraphModel.loadFromBytes(allocator, q35_b.amf_bytes, q35_cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 8,
        .startup_canary = false,
    });
    defer q35_model.deinit(allocator);

    const q35_res = try verifyGraphLoraDoraCycle(
        allocator,
        &q35_model,
        budget,
        4,
        64,
        2 * 32, // GDN L0 q_proj n = 64
        128,
        &prompt,
        iters,
        report.mtl_validation_active,
    );
    report.qwen35_lora_logit_diff = q35_res.diff;
    report.qwen35_unload_max_err = q35_res.unload_err;
    report.qwen35_lora_malloc_delta = q35_res.malloc_delta;

    report.lora_dora_ok = mimo_res.ok and q35_res.ok;
}

pub fn runM91WithConfig(allocator: std.mem.Allocator, cfg: M91Config) !M91Report {
    var report: M91Report = .{};
    report.mtl_validation_active = (std.c.getenv("MTL_DEBUG_LAYER") != null) or
        (std.c.getenv("MTL_SHADER_VALIDATION") != null);

    const snap_start = try vmstat.Snapshot.capture();
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const facts = try @import("../platform/sysinfo.zig").Facts.read();
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = facts.memory_bytes,
        .user_wire_limit = facts.user_wire_limit,
        .global_user_wire_limit = facts.global_user_wire_limit,
        .gpu_working_set_bytes = facts.memory_bytes,
    });

    const tmp_dir = try graph.makeTestTempDir(allocator, "m91_bench");
    defer {
        graph.removeTestDir(allocator, tmp_dir);
        allocator.free(tmp_dir);
    }

    try runG26KvAndSwaBench(allocator, device, library, budget, cfg, &report);
    try runMimoV2ParityBench(allocator, tmp_dir, cfg, &report);
    try runQwen35ParityBench(allocator, tmp_dir, cfg, &report);
    try runD15MimoSpecAndD7Bench(allocator, tmp_dir, cfg, &report);
    try runW30LoraDoraCheck(allocator, budget, cfg, &report);

    const snap_end = try vmstat.Snapshot.capture();
    report.swapouts_delta = vmstat.Snapshot.swapoutDelta(snap_start, snap_end);
    report.malloc_blocks_delta = @max(
        @max(report.swa_malloc_blocks_delta, report.mimo_malloc_blocks_delta),
        @max(report.qwen35_malloc_blocks_delta, @max(report.mimo_lora_malloc_delta, report.qwen35_lora_malloc_delta)),
    );

    const kv_reduction_ok = (report.flash_kv_reduction_32k > 5.0) and
        (report.flash_kv_reduction_128k > 5.0) and
        (report.pro_kv_reduction_32k > 5.0) and
        (report.pro_kv_reduction_128k > 5.0);

    report.all_passed = kv_reduction_ok and
        report.swa_parity_ok and
        report.mimo_parity_ok and
        report.qwen35_parity_ok and
        report.mtp3_parity_ok and
        report.dflash_b8_parity_ok and
        report.dflash_b16_parity_ok and
        report.lora_dora_ok and
        (report.swapouts_delta == 0);

    return report;
}

test "M9.1 MiMo-V2.6 & qwen3_5 end-to-end bench parity, zero swapouts, zero hot-path mallocs" {
    const allocator = std.testing.allocator;
    const rep = try runM91WithConfig(allocator, .{
        .swa_warmup_iters = 2,
        .swa_bench_iters = 4,
        .e2e_tokens = 8,
        .spec_tokens = 12,
        .lora_iters = 4,
    });
    try std.testing.expect(rep.all_passed);
    try std.testing.expectEqual(@as(u64, 0), rep.swapouts_delta);
}
